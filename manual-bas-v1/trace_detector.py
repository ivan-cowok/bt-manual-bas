"""Trace pass_detector state frame-by-frame for video 3, frames 350-510."""
import sys
sys.path.insert(0, ".")

from src.config import Config
from src.parser import parse_clip
from src.features.possession import PossessionTracker
from src.features.velocity import BallVelocityCalculator
from src.detectors.pass_detector import PassInterceptionDetector, _State


def run():
    _, frames = parse_clip("data/4/output.json")
    config = Config()
    poss_tracker = PossessionTracker(config)
    vel_calc = BallVelocityCalculator(config.transform_max_chain_frames)
    detector = PassInterceptionDetector(config)

    START, END = 380, 470

    for frame in frames:
        vel = vel_calc.update(frame)
        speed = (vel[0]**2 + vel[1]**2)**0.5 if vel else 0.0
        possession = poss_tracker.update(frame)
        events = detector.update(frame, possession, vel)

        fn = frame.frame_id
        if fn < START:
            continue
        if fn > END:
            break

        # Detector internals
        state_name = detector._state.name
        poss_team = detector._poss_team or '-'
        loose_ct = detector._loose_in_possessed
        stale_tr = poss_tracker._stale_count

        ball_str = f"({frame.ball.center[0]:.0f},{frame.ball.center[1]:.0f}) spd={speed:.1f}" if frame.ball else "ABSENT"
        ev_str = " ".join(f"[{e.event_type}@{e.team}]" for e in events) if events else ""

        print(f"f{fn:4d} {state_name:<12} poss={poss_team:<6} loose_ct={loose_ct:2d} "
              f"stale={stale_tr:2d} is_stale={possession.is_stale} | poss.team={possession.team:<6} {ball_str} {ev_str}")


if __name__ == "__main__":
    run()
