import json, math, sys

video = sys.argv[1] if len(sys.argv) > 1 else 'data/3/output.json'
with open(video) as f:
    data = json.load(f)

frames = {fd['frame_number']: fd for fd in data['frames']}
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

def trace_range(start, end, label):
    print(f"\n=== {label} ===")
    prev_ball = None
    for fn in range(start, end + 1):
        fd = frames.get(fn)
        if not fd:
            continue
        objs = fd.get('objects', [])
        ball = next((o for o in objs if o['role'] == 'ball'), None)
        gks  = [o for o in objs if o['role'] == 'goalkeeper']
        players = [o for o in objs if o['role'] == 'player']

        if ball:
            b = ball['bbox']
            bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
            speed_str = ""
            if prev_ball is not None:
                dx = bc[0]-prev_ball[0]; dy = bc[1]-prev_ball[1]
                spd = math.sqrt(dx*dx+dy*dy)
                speed_str = f" spd={spd:.1f}"
            bstr = f"ball tid={ball['track_id']} conf={ball['detection_confidence']:.2f} ({bc[0]:.0f},{bc[1]:.0f}){speed_str}"
            prev_ball = bc
        else:
            bstr = "ball=ABSENT"
            bc = None
            prev_ball = None

        # Show which player is NEAREST (eligible)
        best_p = None; best_d = float('inf')
        all_p = []
        for p in players + gks:
            team = p.get('team_name', '') if p['role']=='player' else ('left' if (p['bbox'][0]+p['bbox'][2])/2 < fw/2 else 'right')
            if team not in ('left','right'): continue
            if p['detection_confidence'] < 0.70: continue
            if bc:
                d = bbox_dist(bc, p['bbox']) if p['role']=='goalkeeper' else feet_dist(bc, p['bbox'])
                h = p['bbox'][3]-p['bbox'][1]
                thresh = PROX * h
                in_p = d <= thresh
                all_p.append((d, p['track_id'], p['role'], team, h, thresh, in_p))
                if in_p and d < best_d:
                    best_d = d; best_p = (p['track_id'], p['role'], team, h, thresh, d)

        if best_p:
            tid, role, team, h, thresh, d = best_p
            print(f"  f{fn:4d} POSS tid={tid:3d} {role[:3]} team={team} h={h:.0f} dist={d:.1f}/<{thresh:.0f} | {bstr}")
        else:
            if all_p:
                # show nearest even if not in possession
                all_p.sort()
                d, tid, role, team, h, thresh, _ = all_p[0]
                print(f"  f{fn:4d} LOOSE nearest tid={tid:3d} {role[:3]} team={team} dist={d:.1f}/need<{thresh:.0f} | {bstr}")
            else:
                print(f"  f{fn:4d} LOOSE no_players | {bstr}")

trace_range(75, 100, "FRAMES 75-100: GK distribution")
trace_range(305, 415, "FRAMES 305-415: 4th pass area")
