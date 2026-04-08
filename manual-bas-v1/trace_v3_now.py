import sys
sys.path.insert(0, '.')
from src.parser import parse_clip
from src.config import Config
from src.features.possession import PossessionTracker
from src.features.velocity import BallVelocityCalculator
from src.detectors.pass_detector import PassInterceptionDetector
import math

config = Config()
metadata, frames = parse_clip('data/3/output.json')
tracker = PossessionTracker(config)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
detector = PassInterceptionDetector(config)

for frame in frames:
    fn = frame.frame_id
    poss = tracker.update(frame)
    vel = vel_calc.update(frame)
    speed = math.sqrt(vel[0]**2 + vel[1]**2) if vel else 0.0
    events = detector.update(frame, poss, vel)
    if fn < 83 or fn > 175:
        continue
    ball_str = f"({frame.ball.bbox[0]+(frame.ball.bbox[2]-frame.ball.bbox[0])//2},{frame.ball.bbox[1]+(frame.ball.bbox[3]-frame.ball.bbox[1])//2})" if frame.ball else "ABSENT"
    ev_str = ' '.join(f">>>{e.event_type}({e.team})" for e in events)
    print(f"f{fn:4d} ball={ball_str:16s} spd={speed:5.1f}  poss=[{poss.team} tid={poss.player_track_id} stale={poss.is_stale}]  {ev_str}")
