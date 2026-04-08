import json, math, sys
sys.path.insert(0, '.')
from src.config import Config
from src.parser import parse_clip
from src.features.velocity import BallVelocityCalculator
from src.features.possession import PossessionTracker
from src.detectors.pass_detector import PassInterceptionDetector

with open('output.json') as f:
    data = json.load(f)
config = Config()
_, frames = parse_clip(data)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
poss_tracker = PossessionTracker(config)
pass_det = PassInterceptionDetector(config)

print(f'frm  poss_team  p_tid   spd   state       fl   peak  events')
print('-'*74)
for frame in frames:
    fn = frame.frame_id
    velocity = vel_calc.update(frame)
    possession = poss_tracker.update(frame)
    speed = math.sqrt(velocity[0]**2+velocity[1]**2) if velocity else 0.0
    events = pass_det.update(frame, possession, velocity)
    if (614 <= fn <= 650) or (682 <= fn <= 720):
        fl = str(pass_det._flight_frames) if pass_det._state.name == 'IN_FLIGHT' else '-'
        pk = f'{pass_det._flight_peak_speed:.1f}' if pass_det._state.name == 'IN_FLIGHT' else '-'
        ev = [e.event_type for e in events]
        sep = ' <<' if ev else ''
        print(f'{fn:>3}  {possession.team:>9}  {str(possession.player_track_id):>5}  {speed:>5.1f}  {pass_det._state.name:>9}  {fl:>3}  {pk:>5}  {ev}{sep}')
