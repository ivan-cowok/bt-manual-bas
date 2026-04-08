"""Trace possession + ball speed for video 4, frames 490-580, focusing on tid=12 pass."""
import json, math

with open('data/4/output.json') as f:
    data = json.load(f)

frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fw = data['video_resolution'][0]
PROX = 0.5

def feet_dist(bc, bbox):
    fx = (bbox[0]+bbox[2])/2.0; fy = bbox[3]
    return math.sqrt((bc[0]-fx)**2 + (bc[1]-fy)**2)

def bbox_dist(bc, bbox):
    x1,y1,x2,y2 = bbox
    cx = max(x1, min(bc[0], x2)); cy = max(y1, min(bc[1], y2))
    return math.sqrt((bc[0]-cx)**2 + (bc[1]-cy)**2)

prev_bc = None; prev_fn = None
for fn in range(490, 580):
    fd = frames_raw.get(fn)
    if not fd:
        print(f"f{fn:4d}: FRAME MISSING"); prev_bc = None; prev_fn = None; continue

    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball'), None)

    if ball:
        b = ball['bbox']; bc = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
        spd = 0.0
        if prev_bc and prev_fn:
            g = fn - prev_fn
            spd = math.sqrt((bc[0]-prev_bc[0])**2 + (bc[1]-prev_bc[1])**2) / g
        prev_bc = bc; prev_fn = fn
    else:
        bc = None; prev_bc = None; prev_fn = None

    # Find who is nearest ball
    nearest = None; best_d = 9999
    poss_list = []
    for p in objs:
        if p['role'] not in ('player','goalkeeper'): continue
        if p['detection_confidence'] < 0.70: continue
        if p['role'] == 'goalkeeper':
            cx = (p['bbox'][0]+p['bbox'][2])/2.0
            team = 'left' if cx < fw/2 else 'right'
        else:
            team = p.get('team_name','')
        if team not in ('left','right'): continue
        if not bc: continue
        d = bbox_dist(bc, p['bbox']) if p['role']=='goalkeeper' else feet_dist(bc, p['bbox'])
        h = p['bbox'][3] - p['bbox'][1]
        thresh = PROX * h
        tid = p['track_id']
        if d < thresh:
            poss_list.append(f"tid={tid}({team}) d={d:.0f}<{thresh:.0f}")
        if d < best_d:
            best_d = d; nearest = (tid, team, d, thresh)

    ball_str = f"({bc[0]:.0f},{bc[1]:.0f}) spd={spd:.1f}" if bc else "ABSENT"

    # Highlight tid=12 and tid=51 specifically
    t12 = next((o for o in objs if o.get('track_id')==12), None)
    t51 = next((o for o in objs if o.get('track_id')==51), None)
    extras = []
    if t12 and bc:
        d12 = feet_dist(bc, t12['bbox']); h12 = t12['bbox'][3]-t12['bbox'][1]
        extras.append(f"tid12 d={d12:.0f}/thresh={h12*PROX:.0f}")
    if t51 and bc:
        d51 = feet_dist(bc, t51['bbox']); h51 = t51['bbox'][3]-t51['bbox'][1]
        extras.append(f"tid51 d={d51:.0f}/thresh={h51*PROX:.0f}")

    poss_str = ', '.join(poss_list) if poss_list else 'LOOSE'
    extra_str = '  [' + ', '.join(extras) + ']' if extras else ''
    print(f"f{fn:4d} {poss_str:<45} ball={ball_str}{extra_str}")
