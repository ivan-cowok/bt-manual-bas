"""Compare position-based vs neighbor-vote GK team assignment across all videos."""
import json, math, statistics, collections

for v in [1, 2, 3, 4, 5]:
    try:
        with open(f'data/{v}/output.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        continue
    fw = data['video_resolution'][0]

    # Position-based (current logic)
    gk_xs: dict = {}
    for fd in data['frames']:
        for o in fd.get('objects', []):
            if o['role'] == 'goalkeeper' and o['track_id'] != -1:
                cx = (o['bbox'][0] + o['bbox'][2]) / 2.0
                gk_xs.setdefault(o['track_id'], []).append(cx)
    pos_based = {tid: ('left' if statistics.median(xs) < fw/2 else 'right')
                 for tid, xs in gk_xs.items()}

    # Neighbor-vote (new logic)
    gnt: dict = {}
    for fd in data['frames']:
        objs = fd.get('objects', [])
        gks = [o for o in objs if o['role'] == 'goalkeeper' and o['track_id'] != -1]
        players = [o for o in objs if o['role'] == 'player' and o.get('team_name') in ('left', 'right')]
        if not gks or not players:
            continue
        for gk in gks:
            gx = (gk['bbox'][0] + gk['bbox'][2]) / 2.0
            gy = (gk['bbox'][1] + gk['bbox'][3]) / 2.0
            dists = sorted([
                (math.sqrt((gx - (p['bbox'][0]+p['bbox'][2])/2)**2 + (gy - (p['bbox'][1]+p['bbox'][3])/2)**2),
                 p.get('team_name'))
                for p in players
            ])
            tid = gk['track_id']
            gnt.setdefault(tid, collections.Counter()).update(t for _, t in dists[:3])
    neighbor = {tid: c.most_common(1)[0][0] for tid, c in gnt.items()}

    print(f"\n=== Video {v} ===")
    all_tids = set(pos_based) | set(neighbor)
    for tid in sorted(all_tids):
        p = pos_based.get(tid, '?')
        n = neighbor.get(tid, '?')
        counter = gnt.get(tid, {})
        flag = " *** DIFFERENT ***" if p != n else ""
        print(f"  GK tid={tid}: pos={p}  neighbor={n}  votes={dict(counter)}{flag}")
