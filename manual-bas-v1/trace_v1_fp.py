import sys
sys.path.insert(0, '.')
from src.parser import parse_clip
from src.config import Config
from src.features.possession import PossessionTracker
from src.features.velocity import BallVelocityCalculator
from src.detectors.pass_detector import PassInterceptionDetector, _State
import math

config = Config()
metadata, frames = parse_clip('data/1/output.json')
tracker = PossessionTracker(config)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
detector = PassInterceptionDetector(config)

for frame in frames:
    fn = frame.frame_id
    poss = tracker.update(frame)
    vel = vel_calc.update(frame)
    speed = math.sqrt(vel[0]**2 + vel[1]**2) if vel else 0.0
    state_before = detector._state
    poss_team_before = detector._poss_team
    poss_id_before = detector._poss_player_id
    events = detector.update(frame, poss, vel)
    if fn < 160 or fn > 215:
        continue
    ball_str = f"({frame.ball.bbox[0]+(frame.ball.bbox[2]-frame.ball.bbox[0])//2},{frame.ball.bbox[1]+(frame.ball.bbox[3]-frame.ball.bbox[1])//2})" if frame.ball else "ABSENT"
    ev_str = ' '.join(f">>>{e.event_type}({e.team})" for e in events)
    st = state_before.name[:4]
    print(f"f{fn:4d} [{st} {poss_team_before}/{poss_id_before}] ball={ball_str:16s} spd={speed:5.1f}  poss=[{poss.team} tid={poss.player_track_id} stale={poss.is_stale}]  {ev_str}")
