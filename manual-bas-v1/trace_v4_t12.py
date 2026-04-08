"""Trace tid=12 possession from 437 to 510, find when/why the detector would reset."""
import json, math

with open('data/4/output.json') as f:
    data = json.load(f)

frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fw = data['video_resolution'][0]
PROX = 0.5

def feet_dist(bc, bbox):
    fx = (bbox[0]+bbox[2])/2.0; fy = bbox[3]
    return math.sqrt((bc[0]-fx)**2 + (bc[1]-fy)**2)

prev_bc = None; prev_fn = None
for fn in range(437, 510):
    fd = frames_raw.get(fn)
    if not fd:
        print(f"f{fn:4d}: MISSING"); prev_bc=None; prev_fn=None; continue

    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball'), None)

    if ball:
        b = ball['bbox']; bc = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
        spd = 0.0
        if prev_bc and prev_fn:
            spd = math.sqrt((bc[0]-prev_bc[0])**2 + (bc[1]-prev_bc[1])**2) / (fn-prev_fn)
        prev_bc = bc; prev_fn = fn
    else:
        bc = None; prev_bc = None; prev_fn = None

    t12 = next((o for o in objs if o.get('track_id')==12), None)

    if bc and t12:
        d12 = feet_dist(bc, t12['bbox'])
        h12 = t12['bbox'][3] - t12['bbox'][1]
        thresh = PROX * h12
        status = "POSS" if d12 < thresh else "loose"
        print(f"f{fn:4d} tid12 d={d12:5.1f} thresh={thresh:.0f} --> {status:<6}  ball=({bc[0]:.0f},{bc[1]:.0f}) spd={spd:.1f}")
    elif bc and not t12:
        print(f"f{fn:4d} tid12=ABSENT                        ball=({bc[0]:.0f},{bc[1]:.0f}) spd={spd:.1f}")
    else:
        print(f"f{fn:4d} ball=ABSENT  tid12={'present' if t12 else 'ABSENT'}")
