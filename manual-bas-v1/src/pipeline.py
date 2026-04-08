from typing import List

from .config import Config
from .detectors.pass_detector import PassInterceptionDetector
from .features.possession import PossessionTracker
from .features.velocity import BallVelocityCalculator
from .models import Event, Frame
from .postprocessing import temporal_nms


class Pipeline:
    """
    Two-pass event detection pipeline.

    Pass 1 (frame by frame):
        - BallVelocityCalculator  → velocity vector with transform compensation
        - PossessionTracker       → per-frame team possession signal

    Pass 2 (driven by possession + velocity):
        - PassInterceptionDetector → state machine that fires pass / pass_received /
                                     interception events

    Post-processing:
        - Temporal NMS to suppress duplicate detections of the same event.
    """

    def __init__(self, config: Config):
        self.config = config
        self._velocity_calc = BallVelocityCalculator(config.transform_max_chain_frames)
        self._possession_tracker = PossessionTracker(config)
        self._pass_detector = PassInterceptionDetector(config)

    def run(self, frames: List[Frame]) -> List[Event]:
        """Process all frames and return the deduplicated event list."""
        raw_events: List[Event] = []

        for frame in frames:
            velocity = self._velocity_calc.update(frame)
            possession = self._possession_tracker.update(frame)
            events = self._pass_detector.update(frame, possession, velocity)
            raw_events.extend(events)

        return temporal_nms(raw_events, self.config)
