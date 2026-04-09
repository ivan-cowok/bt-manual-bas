from collections import deque
import math
from typing import List, Optional

from .config import Config
from .detectors.pass_detector import PassDetector
from .features.possession import PossessionTracker
from .features.velocity import BallVelocityCalculator
from .models import Event, Frame, FramePossession, Vector2D
from .postprocessing import (
    remove_pass_received_before_interception,
    shift_short_pass_frames,
    temporal_nms,
)


class Pipeline:
    """
    Event detection pipeline.

    Per-frame (single pass):
        BallVelocityCalculator  — transform-compensated velocity vector (vx, vy)
        PossessionTracker       — per-frame possession signal with top-2 and
                                  CONTESTED detection
        Glitch check (inline)   — if ball position jumped a huge raw-pixel
                                  distance from the previous frame (backward)
                                  AND the ball disappears the very next frame
                                  (forward, safe to look up since all frames
                                  are available), mark FramePossession.ball_glitch
                                  = True so the PassDetector treats it as absent.
        PassDetector            — state machine: DEAD / POSSESSED / CONTESTED /
                                  IN_FLIGHT → fires pass, pass_received,
                                  interception, recovery

    Look-back buffer:
        A rolling deque of the last `lookback_window_frames` FramePossession
        objects is maintained here and passed to PassDetector on each frame.
        The detector uses it to recover kicker identity after a CONTESTED
        sequence without keeping its own history.

    Post-processing:
        Temporal NMS per event type to suppress boundary-effect duplicates.
    """

    def __init__(self, config: Config):
        self.config = config
        self._velocity_calc = BallVelocityCalculator(
            config.transform_max_chain_frames,
            config.max_velocity_gap,
        )
        self._possession_tracker = PossessionTracker(config)
        self._pass_detector = PassDetector(config)

    def run(self, frames: List[Frame]) -> List[Event]:
        raw_events: List[Event] = []
        lookback: deque[FramePossession] = deque(
            maxlen=self.config.lookback_window_frames
        )
        glitch_threshold = self.config.ball_glitch_speed_threshold
        prev_ball_center: Optional[tuple] = None

        for i, frame in enumerate(frames):
            velocity = self._velocity_calc.update(frame)
            possession = self._possession_tracker.update(frame)

            # ----------------------------------------------------------
            # Inline glitch check: ball jumped huge distance (backward)
            # AND disappears the very next frame (forward look-ahead).
            # Raw pixel distance is used — camera pans are far smaller
            # than the 100 px default threshold.
            # ----------------------------------------------------------
            if possession.ball_detected and frame.ball is not None:
                next_no_ball = (
                    i + 1 >= len(frames) or frames[i + 1].ball is None
                )
                if next_no_ball and prev_ball_center is not None:
                    curr = frame.ball.center
                    dist = math.hypot(
                        curr[0] - prev_ball_center[0],
                        curr[1] - prev_ball_center[1],
                    )
                    if dist > glitch_threshold:
                        possession.ball_glitch = True
                prev_ball_center = frame.ball.center
            else:
                prev_ball_center = None  # reset on ball absence

            events = self._pass_detector.update(frame, possession, velocity, lookback)
            raw_events.extend(events)
            lookback.append(possession)

        events = temporal_nms(raw_events, self.config)
        if self.config.is_production:
            events = shift_short_pass_frames(events, self.config)
            events = remove_pass_received_before_interception(events, self.config)
        return events
