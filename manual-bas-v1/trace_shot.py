import json, math, sys

video = sys.argv[1] if len(sys.argv) > 1 else 'data/3/output.json'
with open(video) as f:
    data = json.load(f)

frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fps = data['fps']
fw = data['video_resolution'][0]

def bbox_dist(bc, bbox):
    bx, by = bc
    x1, y1, x2, y2 = bbox
    cx = max(x1, min(bx, x2)); cy = max(y1, min(by, y2))
    return math.sqrt((bx-cx)**2+(by-cy)**2)

def feet_dist(bc, bbox):
    bx, by = bc
    fx = (bbox[0]+bbox[2])/2; fy = bbox[3]
    return math.sqrt((bx-fx)**2+(by-fy)**2)

PROX = 0.35

print("\n=== FRAMES 480-520: Shot area ===")
prev_bc = None
prev_fn = None
for fn in range(480, 521):
    fd = frames_raw.get(fn)
    if not fd: continue
    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball' and o['track_id'] != -1), None)
    gks   = [o for o in objs if o['role']=='goalkeeper']
    players = [o for o in objs if o['role']=='player']

    if ball:
        b = ball['bbox']
        bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        spd_str = ""
        if prev_bc is not None and prev_fn is not None:
            gap = fn - prev_fn
            dx = bc[0]-prev_bc[0]; dy = bc[1]-prev_bc[1]
            spd = math.sqrt(dx*dx+dy*dy)/gap
            spd_str = f" spd={spd:.1f}"
        bstr = f"ball tid={ball['track_id']} conf={ball['detection_confidence']:.2f} ({bc[0]:.0f},{bc[1]:.0f}){spd_str}"
        prev_bc = bc; prev_fn = fn
    else:
        bstr = "ball=ABSENT"
        bc = None

    # All players with possession check
    poss_players = []
    nearby = []
    for p in gks + players:
        team_raw = p.get('team_name', '') if p['role']=='player' else ''
        if p['role'] == 'goalkeeper':
            cx = (p['bbox'][0]+p['bbox'][2])/2
            team = 'left' if cx < fw/2 else 'right'
        else:
            team = team_raw if team_raw in ('left','right') else 'unknown'
        if team == 'unknown': continue
        if p['detection_confidence'] < 0.70: continue
        if bc:
            d = bbox_dist(bc, p['bbox']) if p['role']=='goalkeeper' else feet_dist(bc, p['bbox'])
            h = p['bbox'][3]-p['bbox'][1]
            thresh = PROX * h
            role_short = 'GK' if p['role']=='goalkeeper' else 'PL'
            if d <= thresh:
                poss_players.append(f"{role_short} tid={p['track_id']:3d} team={team} dist={d:.1f}<{thresh:.0f}")
            elif d < thresh * 3:
                nearby.append(f"{role_short} tid={p['track_id']:3d} team={team} dist={d:.1f}/need<{thresh:.0f}")

    if poss_players:
        print(f"  f{fn:4d} POSS: {', '.join(poss_players)} | {bstr}")
    elif nearby:
        print(f"  f{fn:4d} LOOSE near: {', '.join(nearby[:2])} | {bstr}")
    else:
        print(f"  f{fn:4d} LOOSE | {bstr}")
