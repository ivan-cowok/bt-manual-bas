import json, math
with open('data/3/output.json') as f:
    data = json.load(f)
frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fw = data['video_resolution'][0]
PROX = 0.35

def feet_dist(bc, bbox):
    bx, by = bc
    fx = (bbox[0]+bbox[2])/2; fy = bbox[3]
    return math.sqrt((bx-fx)**2+(by-fy)**2)

def bbox_dist(bc, bbox):
    bx, by = bc; x1,y1,x2,y2 = bbox
    cx = max(x1, min(bx, x2)); cy = max(y1, min(by, y2))
    return math.sqrt((bx-cx)**2+(by-cy)**2)

prev_bc = None; prev_fn = None
for fn in range(415, 497):
    fd = frames_raw.get(fn)
    if not fd:
        continue
    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role'] == 'ball' and o['track_id'] != -1), None)
    if ball:
        b = ball['bbox']; bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        spd = ''
        if prev_bc and prev_fn:
            g = fn-prev_fn; dx = bc[0]-prev_bc[0]; dy = bc[1]-prev_bc[1]
            spd = f' spd={math.sqrt(dx*dx+dy*dy)/g:.1f}'
        prev_bc = bc; prev_fn = fn
    else:
        bc = None; prev_bc = None; prev_fn = None
    poss = []
    for p in objs:
        if p['role'] not in ('player', 'goalkeeper'):
            continue
        if p['role'] == 'goalkeeper':
            cx = (p['bbox'][0]+p['bbox'][2])/2
            team = 'left' if cx < fw/2 else 'right'
        else:
            team = p.get('team_name', '')
        if team not in ('left', 'right'):
            continue
        if p['detection_confidence'] < 0.70:
            continue
        if bc:
            d = bbox_dist(bc, p['bbox']) if p['role'] == 'goalkeeper' else feet_dist(bc, p['bbox'])
            h = p['bbox'][3]-p['bbox'][1]; thresh = PROX*h
            if d <= thresh:
                poss.append(f"{p['role'][:2]}:{p['track_id']}:{team} d={d:.0f}")
    ball_str = f"({bc[0]:.0f},{bc[1]:.0f}){spd}" if bc else 'ABSENT'
    tag = 'POSS:' + ','.join(poss) if poss else 'loose'
    print(f'f{fn:4d} {tag:<45} ball={ball_str}')
