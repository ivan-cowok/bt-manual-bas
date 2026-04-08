from collections import deque
from typing import List

from .config import Config
from .detectors.pass_detector import PassDetector
from .features.possession import PossessionTracker
from .features.velocity import BallVelocityCalculator
from .models import Event, Frame, FramePossession
from .postprocessing import temporal_nms


class Pipeline:
    """
    Two-pass event detection pipeline.

    Pass 1 (frame by frame):
        BallVelocityCalculator  — transform-compensated velocity vector (vx, vy)
        PossessionTracker       — per-frame possession signal with top-2 and
                                  CONTESTED detection

    Pass 2 (possession + velocity → events):
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

        for frame in frames:
            velocity = self._velocity_calc.update(frame)
            possession = self._possession_tracker.update(frame)
            events = self._pass_detector.update(frame, possession, velocity, lookback)
            raw_events.extend(events)
            lookback.append(possession)

        return temporal_nms(raw_events, self.config)
