"""Debug video 6 issues:
1. f92-112: pass tid=4->tid=10 (right) detected as interception at f108
2. f305-358: pass tid=18->tid=10 (right) missed (tid=9 left nearby)
"""
import json, math, collections, statistics

with open('data/6/output.json') as f:
    data = json.load(f)

fps = data.get('fps', 25)
fw = data['video_resolution'][0]
frames_raw = {fd['frame_number']: fd for fd in data['frames']}

# ---- GK team assignment (mirror parser logic) ----
gk_x_samples = {}
gk_neighbor_votes = {}
for fd in data['frames']:
    objs = fd.get('objects', [])
    gks = [o for o in objs if o['role'] == 'goalkeeper' and o['track_id'] != -1]
    field_players = [o for o in objs if o['role'] == 'player' and o.get('team_name') in ('left','right')]
    for obj in gks:
        tid = obj['track_id']
        cx = (obj['bbox'][0]+obj['bbox'][2])/2.0
        cy = (obj['bbox'][1]+obj['bbox'][3])/2.0
        gk_x_samples.setdefault(tid, []).append(cx)
        if field_players:
            dists = sorted((math.sqrt((cx-(p['bbox'][0]+p['bbox'][2])/2)**2+(cy-(p['bbox'][1]+p['bbox'][3])/2)**2), p['team_name']) for p in field_players)
            gk_neighbor_votes.setdefault(tid, collections.Counter()).update(t for _,t in dists[:3])

gk_team = {}
for tid, xs in gk_x_samples.items():
    votes = gk_neighbor_votes.get(tid)
    if votes:
        gk_team[tid] = votes.most_common(1)[0][0]
    else:
        gk_team[tid] = 'left' if statistics.median(xs) < fw/2 else 'right'

PROX = 0.5

def get_players(fn):
    fd = frames_raw.get(fn)
    if not fd: return None, []
    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball' and o['track_id']!=-1), None)
    players = []
    for o in objs:
        if o['role'] == 'ball': continue
        if o['role'] == 'referee': continue
        team = o.get('team_name', '')
        if o['role'] == 'goalkeeper':
            team = gk_team.get(o['track_id'], 'left' if (o['bbox'][0]+o['bbox'][2])/2 < fw/2 else 'right')
        if team not in ('left', 'right'): continue
        players.append({'tid': o['track_id'], 'team': team, 'role': o['role'], 'bbox': o['bbox']})
    return ball, players

def ball_center(ball):
    if ball is None: return None
    return ((ball['bbox'][0]+ball['bbox'][2])/2.0, (ball['bbox'][1]+ball['bbox'][3])/2.0)

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def player_feet(bbox):
    return ((bbox[0]+bbox[2])/2.0, bbox[3])

def prox_thresh(bbox):
    return (bbox[3]-bbox[1]) * PROX

def summarize_frame(fn, tids_of_interest):
    ball, players = get_players(fn)
    bc = ball_center(ball)
    if bc is None:
        print(f"  f{fn}: ball=ABSENT")
        return
    parts = [f"ball=({bc[0]:.0f},{bc[1]:.0f})"]
    for p in players:
        if p['tid'] not in tids_of_interest: continue
        feet = player_feet(p['bbox'])
        d = dist(bc, feet)
        thresh = prox_thresh(p['bbox'])
        status = 'POSS' if d <= thresh else f'near{thresh:.0f}' if d <= thresh*2 else f'd={d:.0f}'
        parts.append(f"tid={p['tid']}({p['team']}) {status}")
    print(f"  f{fn}: {' | '.join(parts)}")

# ---- Issue 1: f90-115, tid=4, tid=10, tid=? (interception) ----
print("="*60)
print("ISSUE 1: f90-115 — pass tid=4->tid=10(right) vs interception f108")
print("="*60)
for fn in range(90, 116):
    ball, players = get_players(fn)
    bc = ball_center(ball)
    if bc is None:
        print(f"  f{fn}: ball=ABSENT")
        continue
    poss_list = []
    near_list = []
    for p in players:
        feet = player_feet(p['bbox'])
        d = dist(bc, feet)
        thresh = prox_thresh(p['bbox'])
        if d <= thresh:
            poss_list.append(f"tid={p['tid']}({p['team']},POSS,d={d:.0f}<{thresh:.0f})")
        elif d <= thresh*2:
            near_list.append(f"tid={p['tid']}({p['team']},near,d={d:.0f}<{thresh*2:.0f})")
    items = poss_list + near_list
    line = ' | '.join(items) if items else 'LOOSE'
    print(f"  f{fn}: ball=({bc[0]:.0f},{bc[1]:.0f})  {line}")

# ---- Issue 2: f300-360, tid=18, tid=10, tid=9 ----
print()
print("="*60)
print("ISSUE 2: f300-360 — pass tid=18->tid=10(right) missed, tid=9(left) nearby")
print("="*60)
for fn in range(300, 365):
    ball, players = get_players(fn)
    bc = ball_center(ball)
    if bc is None:
        print(f"  f{fn}: ball=ABSENT")
        continue
    poss_list = []
    near_list = []
    for p in players:
        feet = player_feet(p['bbox'])
        d = dist(bc, feet)
        thresh = prox_thresh(p['bbox'])
        if d <= thresh:
            poss_list.append(f"tid={p['tid']}({p['team']},POSS,d={d:.0f}<{thresh:.0f})")
        elif d <= thresh*2:
            near_list.append(f"tid={p['tid']}({p['team']},near,d={d:.0f}<{thresh*2:.0f})")
    items = poss_list + near_list
    line = ' | '.join(items) if items else 'LOOSE'
    print(f"  f{fn}: ball=({bc[0]:.0f},{bc[1]:.0f})  {line}")
