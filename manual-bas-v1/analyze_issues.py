import json
import math

with open('output.json') as f:
    data = json.load(f)

frames = {f['frame_number']: f for f in data['frames']}
PROX = 0.5

def analyze_range(start, end, label):
    print(f'\n=== {label}: frames {start}-{end} ===')
    print(f"{'frm':>4}  {'ball_cx':>7}  {'btid':>4}  | {'near_tid':>8}  {'team':>6}  {'dist':>6}  {'thr':>6}  {'poss':>5}  {'vel':>6}")
    print('-' * 78)
    prev_bx, prev_by = None, None
    for fn in range(start, end + 1):
        if fn not in frames:
            print(f'{fn:>4}  MISSING FRAME')
            continue
        frame = frames[fn]
        balls = [o for o in frame['objects'] if o['role'] == 'ball']
        players = [o for o in frame['objects'] if o['role'] in ('player', 'goalkeeper')]
        if not balls:
            print(f'{fn:>4}  NO BALL')
            prev_bx = prev_by = None
            continue
        b = balls[0]
        bx = (b['bbox'][0] + b['bbox'][2]) / 2
        by = (b['bbox'][1] + b['bbox'][3]) / 2
        ball_tid = b['track_id']
        vx = bx - prev_bx if prev_bx is not None else 0.0
        vy = by - prev_by if prev_by is not None else 0.0
        speed = math.sqrt(vx**2 + vy**2)
        prev_bx, prev_by = bx, by
        best_p, best_d, best_thr = None, float('inf'), 0
        for p in players:
            feet_x = (p['bbox'][0] + p['bbox'][2]) / 2
            feet_y = p['bbox'][3]
            ph = p['bbox'][3] - p['bbox'][1]
            d = math.sqrt((bx - feet_x)**2 + (by - feet_y)**2)
            thr = PROX * ph
            if d < best_d:
                best_d = d
                best_p = p
                best_thr = thr
        poss = best_d <= best_thr if best_p else False
        team = best_p.get('team_name', '?') if best_p else '?'
        tid = best_p['track_id'] if best_p else -1
        flag = ' <<KICK?' if speed >= 5.0 and not poss else ''
        print(f'{fn:>4}  {bx:>7.1f}  {ball_tid:>4}  | {tid:>8}  {team:>6}  {best_d:>6.1f}  {best_thr:>6.1f}  {str(poss):>5}  {speed:>6.1f}{flag}')

analyze_range(182, 240, 'Issue 1: missed pass 220 + received 235')
analyze_range(368, 415, 'Issue 2: pass detected at 404 instead of 372')
