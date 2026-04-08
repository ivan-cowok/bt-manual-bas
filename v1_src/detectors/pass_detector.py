import math
from enum import Enum, auto
from typing import FrozenSet, List, Optional

_KNOWN_TEAMS: FrozenSet[str] = frozenset({"left", "right"})

from ..config import Config
from ..models import BBox, Event, Frame, FramePossession, Vector2D


class _State(Enum):
    IDLE = auto()
    POSSESSED = auto()
    IN_FLIGHT = auto()


class PassInterceptionDetector:
    """
    State machine that detects pass, pass_received, and interception events.

    States:
        IDLE       — no possession established yet
        POSSESSED  — a team has confirmed possession of the ball
        IN_FLIGHT  — ball has left the possessor with significant velocity

    Transitions:
        IDLE      → POSSESSED  : a team establishes possession
        POSSESSED → IN_FLIGHT  : ball goes "loose" with speed ≥ velocity_threshold
        IN_FLIGHT → POSSESSED  : ball arrives at same team  → fires PASS + PASS_RECEIVED
        IN_FLIGHT → POSSESSED  : ball arrives at other team → fires INTERCEPTION
        IN_FLIGHT → IDLE       : flight exceeds max_flight_frames (window abandoned)
        POSSESSED → POSSESSED  : direct possession change without flight (e.g. tackle)
                                  transitions silently, no event fired

    Dribble guard:
        If both the initiating and receiving player have known track_ids (not -1),
        the track_id comparison is definitive: same ID = dribble, different ID = pass.
        If either track_id is -1 (untrackable), fall back to the flight duration
        heuristic (dribble_max_flight_frames).
    """

    def __init__(self, config: Config):
        self.config = config
        self._state = _State.IDLE

        # Possession context
        self._poss_team: Optional[str] = None
        self._poss_player_id: Optional[int] = None
        self._poss_player_bbox: Optional[BBox] = None
        self._poss_player_role: Optional[str] = None  # "player" or "goalkeeper"

        # Flight context
        self._flight_start_frame: Optional[int] = None
        self._flight_start_timestamp_ms: Optional[int] = None
        self._flight_frames: int = 0
        self._flight_peak_speed: float = 0.0
        # Rolling max speed while in POSSESSED — initialises _flight_peak_speed so
        # that a kick detected while the ball is still technically "in possession"
        # (ball moves fast but hasn't cleared the feet threshold yet) is not lost.
        self._pre_flight_peak_speed: float = 0.0
        # Frames the ball has been present-but-loose while in POSSESSED state.
        # When this exceeds pass_held_loose_max_frames, possession is stale and
        # the state is reset to IDLE so a fast-moving loose ball is not
        # misattributed to the last team that touched it.
        self._loose_in_possessed: int = 0

    def update(
        self,
        frame: Frame,
        possession: FramePossession,
        velocity: Optional[Vector2D],
    ) -> List[Event]:
        """Process one frame. Returns any events detected in this frame."""
        speed = _speed(velocity)

        if self._state == _State.IDLE:
            return self._on_idle(possession)
        if self._state == _State.POSSESSED:
            return self._on_possessed(frame, possession, speed)
        if self._state == _State.IN_FLIGHT:
            return self._on_in_flight(frame, possession, speed)
        return []

    # --- State handlers ---

    def _on_idle(self, possession: FramePossession) -> List[Event]:
        # Don't enter POSSESSED when the ball is absent — we need the ball to
        # actually be visible near a player before committing to a possession.
        # Don't enter POSSESSED on stale signals — those are case-3 accumulation
        # frames where a different team's player is near the ball but the tracker
        # returns the old committed team.  Entering POSSESSED from a stale signal
        # would attribute a subsequent fast ball movement to the wrong team.
        if (
            possession.ball_detected
            and possession.team in _KNOWN_TEAMS
            and not possession.is_stale
        ):
            self._enter_possessed(possession)
        return []

    def _on_possessed(
        self, frame: Frame, possession: FramePossession, speed: float
    ) -> List[Event]:
        # Track rolling peak speed — decays each frame so stale kicks don't persist
        self._pre_flight_peak_speed = max(speed, self._pre_flight_peak_speed * 0.85)

        if possession.team == self._poss_team:
            # Active possession — reset loose counter and keep tracking player
            self._loose_in_possessed = 0
            self._poss_player_id = possession.player_track_id
            self._poss_player_bbox = possession.player_bbox
            return []

        if possession.team == "loose" and speed >= self.config.ball_velocity_threshold:
            # Ball was kicked — enter flight; timestamp PASS at this moment.
            # But first check for stale possession: if the ball has been loose for
            # too long, the committed team did not kick it.  Reset instead of
            # starting a false flight attributed to a team that last touched the
            # ball many frames ago.
            stale = self._loose_in_possessed
            self._loose_in_possessed = 0
            if stale > self.config.pass_held_loose_max_frames:
                self._reset()
            else:
                self._enter_flight(frame)
            return []

        if possession.team == "loose":
            # Ball is present but no player is near it.  Count how long this
            # has persisted.  If the ball has been loose too long, the stored
            # possession team is stale — reset so we don't later fire a false
            # pass attributed to whoever last touched the ball.
            self._loose_in_possessed += 1
            if self._loose_in_possessed > self.config.pass_held_loose_max_frames:
                self._reset()
            return []

        if possession.team in _KNOWN_TEAMS:
            # Direct possession change.  If the ball was moving fast, treat it as a
            # kicked pass and enter IN_FLIGHT so we can find the real receiver.
            # Slow/static changes (tackle, challenge) transition silently.
            self._loose_in_possessed = 0
            if (
                possession.team != self._poss_team
                and speed >= self.config.ball_velocity_threshold
            ):
                self._enter_flight(frame)
                return []
            self._enter_possessed(possession)

        return []

    def _on_in_flight(
        self, frame: Frame, possession: FramePossession, speed: float
    ) -> List[Event]:
        self._flight_frames += 1
        self._flight_peak_speed = max(self._flight_peak_speed, speed)

        max_frames = (
            self.config.gk_pass_max_flight_frames
            if self._poss_player_role == "goalkeeper"
            else self.config.pass_max_flight_frames
        )
        if self._flight_frames > max_frames:
            self._reset()
            return []

        # Ball was absent this frame — hold flight state, never attempt reception
        if not possession.ball_detected:
            return []

        if possession.team not in _KNOWN_TEAMS:
            return []

        is_interception = possession.team != self._poss_team

        # Stale possession — block events when the tracker hasn't committed to a
        # new DIFFERENT-team player.  Stale interception signals are tracker noise:
        # the ball arced close to an opposing player but hasn't truly been received.
        # Same-team and same-player stale signals are handled normally below (dribble
        # detection, pass_received timing, etc.) since those don't create false events.
        if possession.is_stale and is_interception:
            return []

        if self._flight_frames < self.config.pass_min_flight_frames:
            # Too brief — could be a ball touch or detection noise; keep waiting
            return []

        if self._is_same_player_dribble(possession):
            # Dribble: ball returned to the same player
            self._enter_possessed(possession)
            return []

        # The ball never reached a kick-like speed during this flight window.
        # Treat as a rolling loose ball / recovery — silently hand possession over.
        if self._flight_peak_speed < self.config.pass_min_peak_flight_speed:
            self._enter_possessed(possession)
            return []

        if is_interception:
            # Flight too long to be a directed interception.  Two sub-cases:
            # (a) Ball still fast at the candidate receiver — it's still arcing through
            #     the air and just passed close to this player.  Keep the flight open
            #     and wait for the real receiver.
            # (b) Ball has slowed — the player genuinely picked up a loose ball.
            #     Adopt the new possessor and stop the flight.
            if self._flight_frames > self.config.interception_max_flight_frames:
                if speed > self.config.ball_velocity_threshold:
                    return []  # still in the air — ignore this player, keep flight open
                self._enter_possessed(possession)
                return []

        events = self._build_events(frame, possession)
        self._enter_possessed(possession)
        return events

    # --- Event builders ---

    def _build_events(self, frame: Frame, possession: FramePossession) -> List[Event]:
        assert self._poss_team is not None
        assert self._flight_start_frame is not None
        assert self._flight_start_timestamp_ms is not None

        # Shot detection: receiver is a goalkeeper from the opposing team AND the
        # ball was moving fast enough to be a genuine shot (not a slow back-pass
        # to one's own goalkeeper).  Speed is the primary discriminator; role is
        # the confirmation.
        if (
            possession.player_role == "goalkeeper"
            and possession.team != self._poss_team
            and self._flight_peak_speed >= self.config.shot_min_peak_flight_speed
        ):
            return [
                Event(
                    event_type="shot",
                    frame_id=self._flight_start_frame,
                    timestamp_ms=self._flight_start_timestamp_ms,
                    team=self._poss_team,
                    confidence=0.80,
                ),
                Event(
                    event_type="save",
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    team=possession.team,
                    confidence=0.80,
                ),
            ]

        if possession.team == self._poss_team:
            return [
                Event(
                    event_type="pass",
                    frame_id=self._flight_start_frame,
                    timestamp_ms=self._flight_start_timestamp_ms,
                    team=self._poss_team,
                    confidence=0.90,
                ),
                Event(
                    event_type="pass_received",
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    team=possession.team,
                    confidence=0.90,
                ),
            ]
        else:
            return [
                Event(
                    event_type="interception",
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    team=possession.team,
                    confidence=0.85,
                )
            ]

    # --- Helpers ---

    def _is_same_player_dribble(self, possession: FramePossession) -> bool:
        """
        Returns True if the receiving player is the same as the one who last had
        possession — meaning this is a dribble, not a pass.

        Priority rules:
        1. If both track_ids are known (not None / not -1): the track_id match is
           the definitive answer.  Same ID = dribble always, different ID = pass
           always.  Flight duration is irrelevant — we trust the tracker.
        2. If either track_id is unknown (-1 or None): fall back to the flight
           duration heuristic.  Short flight = likely dribble, long flight = likely pass.
        """
        initiator_id = self._poss_player_id
        receiver_id = possession.player_track_id

        both_known = (
            initiator_id is not None and initiator_id != -1
            and receiver_id is not None and receiver_id != -1
        )

        if both_known:
            return initiator_id == receiver_id

        # Unknown track_id on either side — fall back to flight duration heuristic
        return self._flight_frames <= self.config.dribble_max_flight_frames

    def _enter_possessed(self, possession: FramePossession) -> None:
        self._state = _State.POSSESSED
        self._poss_team = possession.team
        self._poss_player_id = possession.player_track_id
        self._poss_player_bbox = possession.player_bbox
        self._poss_player_role = possession.player_role
        self._flight_start_frame = None
        self._flight_start_timestamp_ms = None
        self._flight_frames = 0
        self._flight_peak_speed = 0.0
        self._pre_flight_peak_speed = 0.0
        self._loose_in_possessed = 0

    def _enter_flight(self, frame: Frame) -> None:
        self._state = _State.IN_FLIGHT
        self._flight_start_frame = frame.frame_id
        self._flight_start_timestamp_ms = frame.timestamp_ms
        self._flight_frames = 0
        # Carry pre-flight peak so kicks detected while ball is still at player's
        # feet (fast but within possession threshold) are not lost.
        self._flight_peak_speed = self._pre_flight_peak_speed

    def _reset(self) -> None:
        self._state = _State.IDLE
        self._poss_team = None
        self._poss_player_id = None
        self._poss_player_bbox = None
        self._poss_player_role = None
        self._flight_start_frame = None
        self._flight_start_timestamp_ms = None
        self._flight_frames = 0
        self._flight_peak_speed = 0.0
        self._pre_flight_peak_speed = 0.0
        self._loose_in_possessed = 0


def _speed(velocity: Optional[Vector2D]) -> float:
    if velocity is None:
        return 0.0
    return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
