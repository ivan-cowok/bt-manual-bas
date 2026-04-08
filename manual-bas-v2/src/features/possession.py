import math
from typing import List, Optional, Tuple

from ..config import Config
from ..models import BBox, Frame, FramePossession, PlayerDetection, Vector2D


def _euclidean(a: Vector2D, b: Vector2D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _dist_ball_to_bbox(ball_center: Vector2D, bbox: BBox) -> float:
    """Distance from ball centre to the nearest point on the bounding box.

    Returns 0 when the ball overlaps the box.  Used for goalkeepers where
    the ball can be caught at hand/chest level — well above the feet.
    """
    bx, by = ball_center
    x1, y1, x2, y2 = bbox
    cx = max(x1, min(bx, x2))
    cy = max(y1, min(by, y2))
    return math.sqrt((bx - cx) ** 2 + (by - cy) ** 2)


class PossessionTracker:
    """
    Produces a per-frame possession signal: which team has the ball, or "loose".

    V2 changes vs V1:
    - top-2 candidate tracking: detects CONTESTED when the two nearest eligible
      players are from different teams and both within the possession threshold.
    - is_contested flag on FramePossession feeds the state machine so it can
      enter CONTESTED instead of forcing a noisy team change.
    - Smoothing, stale detection, and ball-absence logic are unchanged from V1.

    Smoothing policy:
        Possession change (A→B or loose→A) requires possession_min_frames
        consecutive frames.  Ball departure (A→loose) is immediate so that
        PASS event timestamps are accurate.

    Stale policy:
        If the committed team's player has been absent from the ball for
        possession_stale_frames consecutive frames, the tracker returns
        is_stale=True.  The detector uses this to block stale interception
        events in IN_FLIGHT.
    """

    def __init__(self, config: Config):
        self.config = config

        # Committed state
        self._team: str = "loose"
        self._player_track_id: Optional[int] = None
        self._player_bbox: Optional[BBox] = None
        self._player_role: Optional[str] = None

        # Candidate being accumulated (different team from committed)
        self._candidate_team: Optional[str] = None
        self._candidate_track_id: Optional[int] = None
        self._candidate_bbox: Optional[BBox] = None
        self._candidate_role: Optional[str] = None
        self._candidate_count: int = 0

        self._absent_count: int = 0
        self._stale_count: int = 0

    def update(self, frame: Frame) -> FramePossession:
        if frame.ball is None:
            return self._handle_absent(frame)

        self._absent_count = 0
        top2 = self._top2_eligible_players(frame)

        if not top2:
            self._stale_count += 1
            self._reset_candidate()
            return FramePossession(
                team="loose",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
            )

        return self._apply_smoothing(top2, frame)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _top2_eligible_players(
        self, frame: Frame
    ) -> List[Tuple[PlayerDetection, float]]:
        """Return up to 2 eligible players closest to the ball, with distances.

        Each entry is (player, distance).  Distance metric is role-dependent:
        - Outfield players: ball centre → player feet (bottom-centre of bbox)
        - Goalkeepers:      ball centre → nearest point on bbox boundary
        """
        assert frame.ball is not None
        ball_center = frame.ball.center
        candidates: List[Tuple[float, PlayerDetection]] = []

        for player in frame.players:
            if not self._is_eligible(player):
                continue
            if player.role == "goalkeeper":
                dist = _dist_ball_to_bbox(ball_center, player.bbox)
            else:
                dist = _euclidean(ball_center, player.feet)
            threshold = self.config.possession_proximity_factor * player.bbox_height
            # Only keep players within threshold for top-2 consideration
            if dist <= threshold:
                candidates.append((dist, player))

        candidates.sort(key=lambda x: x[0])
        return [(p, d) for d, p in candidates[:2]]

    def _is_eligible(self, player: PlayerDetection) -> bool:
        # Eligibility is geometry + team only; bbox detection_confidence is ignored.
        if player.team == "unknown":
            return False
        if (
            player.team_confidence is not None
            and player.team_confidence < self.config.min_team_confidence
        ):
            return False
        return True

    def _apply_smoothing(
        self,
        top2: List[Tuple[PlayerDetection, float]],
        frame: Frame,
    ) -> FramePossession:
        primary, _ = top2[0]
        team = primary.team

        # Detect CONTESTED: top-2 from different teams, both within threshold
        is_contested = (
            len(top2) == 2
            and top2[1][0].team != team
            and top2[1][0].team in ("left", "right")
        )

        second_team: Optional[str] = top2[1][0].team if len(top2) == 2 else None
        second_track_id: Optional[int] = top2[1][0].track_id if len(top2) == 2 else None

        if team == self._team:
            # Possession continues — update player info, reset accumulators
            self._player_track_id = primary.track_id
            self._player_bbox = primary.bbox
            self._player_role = primary.role
            self._stale_count = 0
            self._reset_candidate()
            return self._committed_state(frame, is_contested, second_team, second_track_id)

        # Different team from committed — accumulate candidate frames
        self._stale_count += 1

        if team == self._candidate_team:
            self._candidate_count += 1
            self._candidate_track_id = primary.track_id
            self._candidate_bbox = primary.bbox
            self._candidate_role = primary.role
        else:
            self._candidate_team = team
            self._candidate_track_id = primary.track_id
            self._candidate_bbox = primary.bbox
            self._candidate_role = primary.role
            self._candidate_count = 1

        if self._candidate_count >= self.config.possession_min_frames:
            # Commit to new team
            self._team = self._candidate_team  # type: ignore[assignment]
            self._player_track_id = self._candidate_track_id
            self._player_bbox = self._candidate_bbox
            self._player_role = self._candidate_role
            self._stale_count = 0
            self._reset_candidate()
            return self._committed_state(frame, is_contested, second_team, second_track_id)

        # Not yet committed — return stale or loose depending on staleness
        if self._stale_count > self.config.possession_stale_frames:
            return FramePossession(
                team="loose",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                is_contested=is_contested,
                second_team=second_team,
                second_player_track_id=second_track_id,
            )

        return FramePossession(
            team=self._team,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            player_track_id=self._player_track_id,
            player_bbox=self._player_bbox,
            player_role=self._player_role,
            is_stale=True,
            is_contested=is_contested,
            second_team=second_team,
            second_player_track_id=second_track_id,
        )

    def _handle_absent(self, frame: Frame) -> FramePossession:
        self._absent_count += 1

        if self._absent_count >= self.config.ball_absent_reset_frames:
            self._team = "loose"
            self._player_track_id = None
            self._player_bbox = None
            self._player_role = None
            self._reset_candidate()

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

    def _committed_state(
        self,
        frame: Frame,
        is_contested: bool,
        second_team: Optional[str],
        second_track_id: Optional[int],
    ) -> FramePossession:
        return FramePossession(
            team=self._team,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            player_track_id=self._player_track_id,
            player_bbox=self._player_bbox,
            player_role=self._player_role,
            is_contested=is_contested,
            second_team=second_team,
            second_player_track_id=second_track_id,
        )

    def _reset_candidate(self) -> None:
        self._candidate_team = None
        self._candidate_track_id = None
        self._candidate_bbox = None
        self._candidate_role = None
        self._candidate_count = 0
