from collections import deque
from typing import Deque, FrozenSet, List, Optional

from ..config import Config
from ..features.velocity import BallVelocityCalculator
from ..models import BallState, Event, Frame, FramePossession, Vector2D

_KNOWN_TEAMS: FrozenSet[str] = frozenset({"left", "right"})


class PassDetector:
    """
    State machine that detects pass, pass_received, interception, and recovery.

    States (BallState):
        DEAD       — startup, or ball stationary with no possessor for N frames.
                     All flight/possession context is cleared.  Kicker unknown.
        POSSESSED  — one team has committed, uncontested possession.
        CONTESTED  — players from both teams within threshold simultaneously.
                     All event firing is suppressed until one team clearly wins.
        IN_FLIGHT  — ball moving above speed threshold; no current possessor.
                     Carries kicker_team / kicker_known from prior state.

    Events fired:
        pass           — at flight start frame (POSSESSED→IN_FLIGHT, kicker known)
        pass_received  — at receiver commit frame (same team as kicker)
        interception   — at receiver commit frame (diff team, short flight, kicker known)
        recovery       — at receiver commit frame (long loose / kicker unknown)

    Internal suppressors (emittable=False):
        shot           — ball to opposing GK at high speed; suppresses pass_received

    Look-back:
        The Pipeline passes a rolling window (deque) of recent FramePossession
        objects.  In CONTESTED→IN_FLIGHT transitions where kicker is unclear,
        the window is scanned backward to recover the last clear possessor.
    """

    def __init__(self, config: Config):
        self.config = config
        self._state: BallState = BallState.DEAD

        # --- POSSESSED context ---
        self._poss_team: Optional[str] = None
        self._poss_player_id: Optional[int] = None
        self._poss_player_role: Optional[str] = None
        self._loose_frames: int = 0          # ball loose while in POSSESSED
        self._pre_flight_peak_speed: float = 0.0  # rolling max speed in POSSESSED
        self._poss_frame_count: int = 0      # same-team frames since possession committed
        self._poss_from_dead: bool = False   # True if possession was established from DEAD
        self._poss_from_recovery: bool = False  # True if possession came from a recovery event

        # --- CONTESTED context ---
        self._pre_contest_team: Optional[str] = None
        self._pre_contest_player_id: Optional[int] = None
        self._pre_contest_role: Optional[str] = None
        self._contest_frames: int = 0

        # --- IN_FLIGHT context ---
        self._kicker_team: Optional[str] = None
        self._kicker_player_id: Optional[int] = None
        self._kicker_role: Optional[str] = None
        self._kicker_known: bool = False
        self._flight_start_frame: Optional[int] = None
        self._flight_start_timestamp_ms: Optional[int] = None
        self._flight_frames: int = 0
        self._peak_speed: float = 0.0

        # --- DEAD context ---
        self._dead_frames: int = 0           # frames of stationary + no possessor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        frame: Frame,
        possession: FramePossession,
        velocity: Optional[Vector2D],
        lookback: Deque[FramePossession],
    ) -> List[Event]:
        speed = BallVelocityCalculator.speed(velocity)

        if self._state == BallState.DEAD:
            return self._on_dead(frame, possession, speed)
        if self._state == BallState.POSSESSED:
            return self._on_possessed(frame, possession, speed)
        if self._state == BallState.CONTESTED:
            return self._on_contested(frame, possession, speed, lookback)
        if self._state == BallState.IN_FLIGHT:
            return self._on_in_flight(frame, possession, speed)
        return []

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _on_dead(
        self, frame: Frame, possession: FramePossession, speed: float
    ) -> List[Event]:
        if not possession.ball_detected:
            return []

        if possession.team in _KNOWN_TEAMS and not possession.is_stale:
            # Fresh possession from dead — mark as from_dead so the detector
            # requires a minimum number of active possession frames before
            # attributing a kick to this team (prevents glitch-triggered FPs).
            self._enter_possessed(possession, from_dead=True)
            return []

        if possession.team == "loose" and speed >= self.config.ball_velocity_threshold:
            # Ball kicked with no prior possessor committed (restart, throw-in etc.)
            # Enter flight with kicker unknown — only RECOVERY can result
            self._enter_flight(frame, kicker_known=False)
            return []

        # Still dead
        return []

    def _on_possessed(
        self, frame: Frame, possession: FramePossession, speed: float
    ) -> List[Event]:
        self._pre_flight_peak_speed = max(
            speed, self._pre_flight_peak_speed * 0.85
        )

        if not possession.ball_detected:
            return []

        # Contested — two teams near ball simultaneously
        if possession.is_contested:
            self._enter_contested(possession)
            return []

        if possession.team == self._poss_team:
            # Active possession — update and reset loose counter
            self._loose_frames = 0
            self._poss_player_id = possession.player_track_id
            self._poss_player_role = possession.player_role
            self._poss_frame_count += 1
            return []

        if possession.team == "loose":
            if speed >= self.config.ball_velocity_threshold:
                # Ball kicked — enter flight, kicker is the possessed team.
                # Exception: if this possession was established from a RECOVERY event
                # and the team has had very few active frames (< min), the ball may
                # have been picked up via a tracker glitch (e.g. ball teleporting near
                # a player after a save/scramble).  Treat as unknown kicker (→ recovery,
                # not emitted) to suppress false positive interceptions.
                stale = self._loose_frames
                self._loose_frames = 0
                post_recovery_glitch = (
                    self._poss_from_recovery
                    and self._poss_frame_count
                    < self.config.min_post_recovery_possession_frames
                )
                if stale > self.config.pass_held_loose_max_frames:
                    # Ball has been drifting loose too long — kicker attribution
                    # is unreliable; treat as a dead-ball restart
                    self._enter_dead()
                else:
                    self._enter_flight(frame, kicker_known=not post_recovery_glitch)
                return []

            self._loose_frames += 1
            if self._loose_frames > self.config.pass_held_loose_max_frames:
                self._enter_dead()
            return []

        if possession.team in _KNOWN_TEAMS:
            # Direct team change — if fast, treat as a kick and enter flight
            # (the kicker was the previously possessed team).
            # If slow and contested, should have been caught above; if slow and
            # uncontested it's a silent adoption (tackle/challenge resolved).
            self._loose_frames = 0
            if (
                possession.team != self._poss_team
                and speed >= self.config.ball_velocity_threshold
            ):
                self._enter_flight(frame, kicker_known=True)
                return []
            self._enter_possessed(possession)

        return []

    def _on_contested(
        self,
        frame: Frame,
        possession: FramePossession,
        speed: float,
        lookback: Deque[FramePossession],
    ) -> List[Event]:
        self._contest_frames += 1

        if not possession.ball_detected:
            return []

        # Contest resolved: one team is now sole possessor
        if possession.team in _KNOWN_TEAMS and not possession.is_contested:
            # Silent resolution — no event regardless of which team won
            self._enter_possessed(possession)
            return []

        # Ball kicked out of contest at speed
        if possession.team == "loose" and speed >= self.config.ball_velocity_threshold:
            # Try to recover kicker from look-back window
            kicker_team, kicker_id, kicker_role = self._recover_kicker(lookback)
            if kicker_team is not None:
                # Override possessed context with recovered kicker before entering flight
                self._poss_team = kicker_team
                self._poss_player_id = kicker_id
                self._poss_player_role = kicker_role
                self._enter_flight(frame, kicker_known=True)
            else:
                self._enter_flight(frame, kicker_known=False)
            return []

        # Ball loose but slow during contest — treat as dead if stalemate
        if self._contest_frames > self.config.contested_max_frames:
            self._enter_dead()
            return []

        # Still contested — stay and wait
        return []

    def _on_in_flight(
        self, frame: Frame, possession: FramePossession, speed: float
    ) -> List[Event]:
        self._flight_frames += 1
        self._peak_speed = max(self._peak_speed, speed)

        max_frames = (
            self.config.gk_pass_max_flight_frames
            if self._kicker_role == "goalkeeper"
            else self.config.pass_max_flight_frames
        )
        if self._flight_frames > max_frames:
            self._enter_dead()
            return []

        if not possession.ball_detected:
            return []

        if possession.team not in _KNOWN_TEAMS:
            return []

        # Contested arrival — ball reached both teams simultaneously; suppress event,
        # enter CONTESTED and let it resolve before attributing the reception.
        if possession.is_contested:
            self._enter_contested(possession)
            return []

        if self._flight_frames < self.config.pass_min_flight_frames:
            return []

        # Stale different-team signal — ball passed near an opposing player but
        # they haven't truly committed.  Keep the flight open.
        if possession.is_stale and possession.team != self._kicker_team:
            return []

        # Dribble guard
        if self._is_dribble(possession):
            self._enter_possessed(possession)
            return []

        # Shot suppressor (internal): ball arrives at opposing GK at high speed.
        # Do not emit pass_received / interception — this is a shot, not a pass.
        if (
            possession.player_role == "goalkeeper"
            and possession.team != self._kicker_team
            and self._peak_speed >= self.config.shot_min_peak_flight_speed
        ):
            self._enter_possessed(possession)
            return []  # no emittable event

        events = self._classify_reception(frame, possession)
        # If the reception was a recovery (unknown kicker), flag the new possession
        # so that a very-brief subsequent kick is also treated as unknown.
        is_recovery = any(e.event_type == "recovery" for e in events)
        self._enter_possessed(possession, from_recovery=is_recovery)
        return events

    # ------------------------------------------------------------------
    # Event classification
    # ------------------------------------------------------------------

    def _classify_reception(
        self, frame: Frame, possession: FramePossession
    ) -> List[Event]:
        assert self._flight_start_frame is not None
        assert self._flight_start_timestamp_ms is not None

        same_team = possession.team == self._kicker_team

        if not self._kicker_known:
            # Kicker was unknown (came from DEAD or unresolved CONTESTED)
            # → always a recovery for whichever team picks it up
            return [self._make_event("recovery", frame, possession.team, 0.80)]

        if same_team:
            if self._flight_frames > self.config.pass_max_flight_frames:
                # Exceeded normal pass window for outfield — treat as recovery
                return [self._make_event("recovery", frame, possession.team, 0.75)]
            return [
                Event(
                    event_type="pass",
                    frame_id=self._flight_start_frame,
                    timestamp_ms=self._flight_start_timestamp_ms,
                    team=self._kicker_team,  # type: ignore[arg-type]
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
            # Different team received
            if self._flight_frames <= self.config.interception_max_flight_frames:
                # Short flight from known kicker → interception
                return [self._make_event("interception", frame, possession.team, 0.85)]
            else:
                # Long loose ball picked up by opposing team → recovery
                return [self._make_event("recovery", frame, possession.team, 0.75)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recover_kicker(
        self, lookback: Deque[FramePossession]
    ) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Scan look-back window backward for the last clear, non-contested possessor."""
        for fp in reversed(lookback):
            if (
                fp.team in _KNOWN_TEAMS
                and not fp.is_stale
                and not fp.is_contested
            ):
                return fp.team, fp.player_track_id, fp.player_role
        return None, None, None

    def _is_dribble(self, possession: FramePossession) -> bool:
        """True if receiver is the same player as the kicker — dribble, not pass."""
        initiator_id = self._poss_player_id
        receiver_id = possession.player_track_id

        both_known = (
            initiator_id is not None and initiator_id != -1
            and receiver_id is not None and receiver_id != -1
        )
        if both_known:
            return initiator_id == receiver_id

        # Unknown IDs — fall back to flight duration heuristic
        return self._flight_frames <= self.config.dribble_max_flight_frames

    def _make_event(
        self, event_type: str, frame: Frame, team: str, confidence: float
    ) -> Event:
        return Event(
            event_type=event_type,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            team=team,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _enter_dead(self) -> None:
        self._state = BallState.DEAD
        self._poss_team = None
        self._poss_player_id = None
        self._poss_player_role = None
        self._loose_frames = 0
        self._pre_flight_peak_speed = 0.0
        self._poss_frame_count = 0
        self._poss_from_dead = False
        self._poss_from_recovery = False
        self._pre_contest_team = None
        self._pre_contest_player_id = None
        self._pre_contest_role = None
        self._contest_frames = 0
        self._kicker_team = None
        self._kicker_player_id = None
        self._kicker_role = None
        self._kicker_known = False
        self._flight_start_frame = None
        self._flight_start_timestamp_ms = None
        self._flight_frames = 0
        self._peak_speed = 0.0
        self._dead_frames = 0

    def _enter_possessed(
        self, possession: FramePossession, from_dead: bool = False, from_recovery: bool = False
    ) -> None:
        self._state = BallState.POSSESSED
        self._poss_team = possession.team
        self._poss_player_id = possession.player_track_id
        self._poss_player_role = possession.player_role
        self._loose_frames = 0
        self._pre_flight_peak_speed = 0.0
        self._poss_frame_count = 0
        self._poss_from_dead = from_dead
        self._poss_from_recovery = from_recovery
        self._pre_contest_team = None
        self._pre_contest_player_id = None
        self._pre_contest_role = None
        self._contest_frames = 0
        self._kicker_team = None
        self._kicker_player_id = None
        self._kicker_role = None
        self._kicker_known = False
        self._flight_start_frame = None
        self._flight_start_timestamp_ms = None
        self._flight_frames = 0
        self._peak_speed = 0.0

    def _enter_contested(self, possession: FramePossession) -> None:
        self._state = BallState.CONTESTED
        # Save current possessor as pre-contest context
        self._pre_contest_team = self._poss_team
        self._pre_contest_player_id = self._poss_player_id
        self._pre_contest_role = self._poss_player_role
        self._contest_frames = 0

    def _enter_flight(self, frame: Frame, kicker_known: bool) -> None:
        self._state = BallState.IN_FLIGHT
        self._kicker_known = kicker_known
        if kicker_known:
            self._kicker_team = self._poss_team
            self._kicker_player_id = self._poss_player_id
            self._kicker_role = self._poss_player_role
        else:
            self._kicker_team = None
            self._kicker_player_id = None
            self._kicker_role = None
        self._flight_start_frame = frame.frame_id
        self._flight_start_timestamp_ms = frame.timestamp_ms
        self._flight_frames = 0
        # Carry pre-flight peak so kicks detected at player's feet are not lost
        self._peak_speed = self._pre_flight_peak_speed
        self._pre_flight_peak_speed = 0.0
        self._loose_frames = 0
