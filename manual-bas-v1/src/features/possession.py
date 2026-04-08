import math
from typing import Optional

from ..config import Config
from ..models import BBox, Frame, FramePossession, PlayerDetection, Vector2D


def _euclidean(a: Vector2D, b: Vector2D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _dist_ball_to_bbox(ball_center: Vector2D, bbox: BBox) -> float:
    """Distance from ball centre to the nearest point on the player bounding box.

    Returns 0 when the ball overlaps the box.  Used specifically for goalkeepers
    where the ball can be caught at hand/chest level — far above the feet.
    """
    bx, by = ball_center
    x1, y1, x2, y2 = bbox
    cx = max(x1, min(bx, x2))
    cy = max(y1, min(by, y2))
    return math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)


def _feet(bbox: BBox) -> Vector2D:
    """Bottom-centre of a bounding box — the player's feet position."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


class PossessionTracker:
    """
    Produces a per-frame possession signal: which team has the ball, or "loose".

    Possession is assigned to the closest eligible player whose distance from
    the ball centre to the player's bounding box boundary is within
    (possession_proximity_factor × player bbox_height).

    Smoothing policy:
    - Possession change (A→B or loose→A) requires min_frames consecutive frames
      to commit, preventing flicker from noisy detections.
    - Ball departure (A→loose) is detected immediately — no smoothing — so that
      PASS event timestamps are accurate.

    Ball absence policy:
    - Short absence (≤ ball_absent_hold_frames): hold last committed state.
    - Long absence (> ball_absent_reset_frames): reset to "loose".
    """

    def __init__(self, config: Config):
        self.config = config

        # Committed possession
        self._team: str = "loose"
        self._player_track_id: Optional[int] = None
        self._player_bbox: Optional[BBox] = None
        self._player_role: Optional[str] = None

        # Candidate (pending confirmation)
        self._candidate_team: Optional[str] = None
        self._candidate_player_track_id: Optional[int] = None
        self._candidate_player_bbox: Optional[BBox] = None
        self._candidate_player_role: Optional[str] = None
        self._candidate_count: int = 0

        self._absent_count: int = 0

        # Consecutive frames the committed team's player has been absent from the
        # ball (ball present but not nearest eligible player from committed team).
        # Used to detect stale possession: if too large, candidate accumulation
        # from a different team returns "loose" rather than the old committed team,
        # preventing the detector from re-entering POSSESSED on stale state.
        self._stale_count: int = 0

    def update(self, frame: Frame) -> FramePossession:
        """Process one frame and return the possession state for that frame."""
        if frame.ball is None:
            return self._handle_absent(frame)

        self._absent_count = 0
        player = self._nearest_eligible_player(frame)

        if player is None:
            # Ball is not near any eligible player — immediate "loose".
            # Increment stale count: committed team is no longer near the ball.
            self._stale_count += 1
            self._reset_candidate()
            return FramePossession(
                team="loose",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
            )

        return self._apply_smoothing(player, frame)

    # --- Internal ---

    def _nearest_eligible_player(self, frame: Frame) -> Optional[PlayerDetection]:
        """Return the closest eligible player within the possession threshold, or None.

        Distance metric is role-dependent:
        - Regular players: ball centre to player's feet (bottom of bbox).
          A dribbled or passed ball sits at foot level; feet-proximity is stable
          and avoids false positives from balls that are near a player's body
          but not actually under their control.
        - Goalkeepers: ball centre to nearest point on the bbox boundary.
          Goalkeepers catch, punch, and distribute with their hands/chest; the
          ball can be well above the feet and still be in possession.
        """
        assert frame.ball is not None
        ball_center = frame.ball.center
        best: Optional[PlayerDetection] = None
        best_dist = float("inf")

        for player in frame.players:
            if not self._is_eligible(player):
                continue
            if player.role == "goalkeeper":
                dist = _dist_ball_to_bbox(ball_center, player.bbox)
            else:
                dist = _euclidean(ball_center, _feet(player.bbox))
            threshold = self.config.possession_proximity_factor * player.bbox_height
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best = player

        return best

    def _is_eligible(self, player: PlayerDetection) -> bool:
        if player.team == "unknown":
            return False
        if player.detection_confidence < self.config.min_detection_confidence:
            return False
        if (
            player.team_confidence is not None
            and player.team_confidence < self.config.min_team_confidence
        ):
            return False
        return True

    def _apply_smoothing(self, player: PlayerDetection, frame: Frame) -> FramePossession:
        team = player.team

        if team == self._team:
            # Possession continues with same team — update player info, reset candidate
            # and stale counter.
            self._player_track_id = player.track_id
            self._player_bbox = player.bbox
            self._player_role = player.role
            self._stale_count = 0
            self._reset_candidate()
            return self._committed_state(frame)

        # Different team from current committed — accumulate candidate frames.
        # Also increment stale counter because the committed team's player is not
        # currently nearest.
        self._stale_count += 1

        if team == self._candidate_team:
            self._candidate_count += 1
            self._candidate_player_track_id = player.track_id
            self._candidate_player_bbox = player.bbox
            self._candidate_player_role = player.role
        else:
            self._candidate_team = team
            self._candidate_player_track_id = player.track_id
            self._candidate_player_bbox = player.bbox
            self._candidate_player_role = player.role
            self._candidate_count = 1

        if self._candidate_count >= self.config.possession_min_frames:
            # Commit to new team; reset stale counter for the fresh possession.
            self._team = self._candidate_team  # type: ignore[assignment]
            self._player_track_id = self._candidate_player_track_id
            self._player_bbox = self._candidate_player_bbox
            self._player_role = self._candidate_player_role
            self._stale_count = 0
            self._reset_candidate()

        # Until committed, report old team — UNLESS possession is stale, in which
        # case return "loose" so the detector cannot reuse the old team's state.
        if self._stale_count > self.config.possession_stale_frames:
            return FramePossession(
                team="loose",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
            )
        # Stale: different team is near ball but not yet committed — return old
        # committed team with is_stale=True so IDLE→POSSESSED won't fire.
        return FramePossession(
            team=self._team,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            player_track_id=self._player_track_id,
            player_bbox=self._player_bbox,
            player_role=self._player_role,
            is_stale=True,
        )

    def _handle_absent(self, frame: Frame) -> FramePossession:
        self._absent_count += 1

        if self._absent_count >= self.config.ball_absent_reset_frames:
            self._team = "loose"
            self._player_track_id = None
            self._player_bbox = None
            self._player_role = None
            self._reset_candidate()

        # Hold state during short absence; report "loose" after hold window expires
        if self._absent_count <= self.config.ball_absent_hold_frames:
            return FramePossession(
                team=self._team,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                ball_detected=False,
                player_track_id=self._player_track_id,
                player_bbox=self._player_bbox,
                player_role=self._player_role,
            )

        return FramePossession(
            team="loose",
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            ball_detected=False,
        )

    def _committed_state(self, frame: Frame) -> FramePossession:
        return FramePossession(
            team=self._team,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            player_track_id=self._player_track_id,
            player_bbox=self._player_bbox,
            player_role=self._player_role,
        )

    def _reset_candidate(self) -> None:
        self._candidate_team = None
        self._candidate_player_track_id = None
        self._candidate_player_bbox = None
        self._candidate_player_role = None
        self._candidate_count = 0
