"""Diagnostic trace for new video: show GT events, detected events, and pipeline state."""
import json, math, sys
sys.path.insert(0, '.')
from src.config import Config
from src.parser import parse_clip
from src.features.velocity import BallVelocityCalculator
from src.features.possession import PossessionTracker
from src.detectors.pass_detector import PassInterceptionDetector

with open('output.json') as f: data = json.load(f)
with open('ground_truth.json') as f: gt_raw = json.load(f)

fps = data.get('fps', 25.0)
config = Config()
_, frames = parse_clip(data)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
poss_tracker = PossessionTracker(config)
pass_det = PassInterceptionDetector(config)

# Index GT by approximate frame
gt_by_frame = {}
print("=== GROUND TRUTH ===")
for e in gt_raw:
    f_approx = int(e['chunk_time_ms'] * fps / 1000)
    gt_by_frame[f_approx] = e['type']
    print(f"  {e['type']:<20} {e['chunk_time_ms']:>7}ms  ~frame {f_approx}")

print()
print(f"{'frm':>4}  {'poss':>9}  {'tid':>4}  {'spd':>6}  {'state':>9}  {'fl':>3}  events / GT")
print('-' * 72)

for frame in frames:
    fn = frame.frame_id
    velocity = vel_calc.update(frame)
    possession = poss_tracker.update(frame)
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
    events = pass_det.update(frame, possession, velocity)
    evts = [e.event_type for e in events]
    fl = str(pass_det._flight_frames) if pass_det._state.name == 'IN_FLIGHT' else '-'

    # nearby GT (within ±10 frames)
    nearby_gt = [f'{gt_by_frame[gf]}@{gf}' for gf in gt_by_frame if abs(gf - fn) <= 10]

    if evts or nearby_gt:
        marker = ' <GT: ' + ', '.join(nearby_gt) + '>' if nearby_gt else ''
        print(f"{fn:>4}  {possession.team:>9}  {str(possession.player_track_id):>4}  {speed:>6.1f}  "
              f"{pass_det._state.name:>9}  {fl:>3}  {evts}{marker}")
