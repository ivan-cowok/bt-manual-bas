"""Debug script: trace pass detector state frame by frame for a given range."""
import json, sys
sys.path.insert(0, '.')

from src.config import Config
from src.parser import parse_clip
from src.features.velocity import BallVelocityCalculator
from src.features.possession import PossessionTracker
from src.detectors.pass_detector import PassInterceptionDetector

with open('output.json') as f:
    data = json.load(f)

config = Config()
metadata, frames = parse_clip(data)

vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
poss_tracker = PossessionTracker(config)
pass_det = PassInterceptionDetector(config)

import math
START, END = 310, 415

print(f"{'frm':>4}  {'poss_team':>10}  {'p_tid':>5}  {'speed':>6}  {'det_state':>10}  events")
print('-' * 75)

for frame in frames:
    fn = frame.frame_id
    velocity = vel_calc.update(frame)
    possession = poss_tracker.update(frame)
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
    events = pass_det.update(frame, possession, velocity)

    if fn >= START and fn <= END:
        evts = [e.event_type for e in events]
        flight_info = f' flight={pass_det._flight_frames}' if pass_det._state.name == 'IN_FLIGHT' else ''
        print(f'{fn:>4}  {possession.team:>10}  {str(possession.player_track_id):>5}  {speed:>6.1f}  {pass_det._state.name:>10}  {evts}{flight_info}')
