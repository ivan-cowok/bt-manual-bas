"""Trace the actual detector state for video 6 around the two problem areas."""
import sys
sys.path.insert(0, '.')

from src.parser import parse_clip
from src.config import Config
from src.features.possession import PossessionTracker
from src.features.velocity import BallVelocityCalculator
from src.detectors.pass_detector import PassInterceptionDetector

config = Config()
metadata, frames = parse_clip('data/6/output.json')

tracker = PossessionTracker(config)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
detector = PassInterceptionDetector(config)

RANGES = [(88, 118), (295, 365)]

def in_range(fn):
    return any(lo <= fn <= hi for lo, hi in RANGES)

current_range = None

for frame in frames:
    fn = frame.frame_id
    poss = tracker.update(frame)
    vel = vel_calc.update(frame)
    import math
    speed = math.sqrt(vel[0]**2 + vel[1]**2) if vel else 0.0
    events = detector.update(frame, poss, vel)

    if not in_range(fn):
        continue

    # Detect range change for header
    new_range = next((r for r in RANGES if r[0] <= fn <= r[1]), None)
    if new_range != current_range:
        current_range = new_range
        print(f"\n{'='*65}")
        print(f"RANGE f{new_range[0]}-{new_range[1]}")
        print(f"{'='*65}")

    ball_str = f"({frame.ball.bbox[0]+(frame.ball.bbox[2]-frame.ball.bbox[0])//2},{frame.ball.bbox[1]+(frame.ball.bbox[3]-frame.ball.bbox[1])//2})" if frame.ball else "ABSENT"
    poss_str = f"team={poss.team} tid={poss.player_track_id} stale={poss.is_stale}"
    ev_str = ' '.join(f">>> {e.event_type.upper()}({e.team})" for e in events)
    print(f"f{fn:4d} ball={ball_str:16s} spd={speed:5.1f}  poss=[{poss_str}]  {ev_str}")
