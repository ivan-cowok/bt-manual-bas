import json
import math

with open('output.json') as f:
    data = json.load(f)

frames = {f['frame_number']: f for f in data['frames']}

PROX = 0.5

print('=== BALL + NEAREST PLAYER: frames 47-130 ===')
print(f"{'frm':>4}  {'ball_cx':>7}  {'ball_tid':>8}  | {'near_tid':>8}  {'team':>6}  {'dist':>6}  {'thr':>6}  poss")
print('-' * 75)

prev_bx = None
for fn in range(47, 131):
    if fn not in frames:
        continue
    frame = frames[fn]
    objs = frame['objects']

    balls = [o for o in objs if o['role'] == 'ball']
    players = [o for o in objs if o['role'] in ('player', 'goalkeeper')]

    if not balls:
        print(f'{fn:>4}  NO BALL')
        prev_bx = None
        continue

    b = balls[0]
    bx = (b['bbox'][0] + b['bbox'][2]) / 2
    by = (b['bbox'][1] + b['bbox'][3]) / 2
    ball_tid = b['track_id']

    # ball velocity magnitude
    vel = abs(bx - prev_bx) if prev_bx is not None else 0.0
    prev_bx = bx

    best_p, best_d, best_thr = None, float('inf'), 0
    for p in players:
        px = (p['bbox'][0] + p['bbox'][2]) / 2
        py = (p['bbox'][1] + p['bbox'][3]) / 2
        ph = p['bbox'][3] - p['bbox'][1]
        d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
        thr = PROX * ph
        if d < best_d:
            best_d = d
            best_p = p
            best_thr = thr

    in_poss = best_d <= best_thr if best_p else False
    team = best_p.get('team_name', '?') if best_p else '?'
    tid = best_p['track_id'] if best_p else -1

    flag = ' <-- LOOSE' if not in_poss and vel > 5 else ''
    print(f'{fn:>4}  {bx:>7.1f}  {ball_tid:>8}  | {tid:>8}  {team:>6}  {best_d:>6.1f}  {best_thr:>6.1f}  {str(in_poss):>5}  vel={vel:>5.1f}{flag}')
