import json, math, sys
sys.path.insert(0, '.')
from src.config import Config
from src.parser import parse_clip
from src.features.possession import PossessionTracker
from src.features.velocity import BallVelocityCalculator
from src.detectors.pass_detector import PassInterceptionDetector

config = Config()
metadata, frames = parse_clip('data/5/output.json')
tracker = PossessionTracker(config)
vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
detector = PassInterceptionDetector(config)

frames_by_id = {f.frame_id: f for f in frames}

with open('trace_v5_out.txt', 'w') as out:
    out.write(f"Total frames: {len(frames)}\n")
    out.write(f"pass_max_flight_frames: {config.pass_max_flight_frames}\n")
    out.write(f"pass_held_loose_max_frames: {config.pass_held_loose_max_frames}\n\n")

    for f in frames:
        vel = vel_calc.update(f)
        poss = tracker.update(f)
        events = detector.update(f, poss, vel)
        spd = math.sqrt(vel[0]**2+vel[1]**2) if vel else 0.0
        ball_str = f"({f.ball.bbox[0]},{f.ball.bbox[1]})" if f.ball else "ABSENT"
        det_state = detector._state.name
        loose_ct = detector._loose_in_possessed
        poss_team = detector._poss_team or '-'

        if f.frame_id >= 1 or events:
            line = (f"f{f.frame_id:4d} det={det_state:<10} poss_team={poss_team:<6} "
                    f"tracker={poss.team:<6} stale={str(poss.is_stale):<5} "
                    f"loose_ct={loose_ct} spd={spd:5.1f} ball={ball_str}")
            if events:
                line += f"  *** {[e.event_type for e in events]}"
            out.write(line + '\n')

print("Done. Written to trace_v5_out.txt")
