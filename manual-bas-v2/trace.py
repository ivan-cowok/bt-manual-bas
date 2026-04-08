"""
Trace possession + velocity + detector state for a frame range.
Usage: python trace.py <output.json> <start_frame> <end_frame>
"""
import json
import math
import sys

input_path = sys.argv[1]
START = int(sys.argv[2])
END   = int(sys.argv[3])

with open(input_path, encoding="utf-8") as f:
    data = json.load(f)

PROX = 0.5

frames = {fd["frame_number"]: fd for fd in data["frames"]}

print(f"{'frm':>4}  {'ball_xy':>15}  {'spd':>5}  "
      f"{'p1_tid':>6} {'p1_team':>5} {'p1_dst':>6} {'p1_thr':>6} {'p1_in':>5}  "
      f"{'p2_tid':>6} {'p2_team':>5} {'p2_dst':>6} {'p2_thr':>6} {'p2_in':>5}  "
      f"{'contested':>9}")
print("-" * 110)

prev_bx, prev_by = None, None

for fn in range(START, END + 1):
    if fn not in frames:
        print(f"{fn:>4}  MISSING")
        continue

    fd = frames[fn]
    balls   = [o for o in fd["objects"] if o["role"] == "ball" and o["track_id"] != -1]
    players = [o for o in fd["objects"] if o["role"] in ("player", "goalkeeper")]

    if not balls:
        print(f"{fn:>4}  NO BALL")
        prev_bx = prev_by = None
        continue

    b  = balls[0]
    bx = (b["bbox"][0] + b["bbox"][2]) / 2
    by = (b["bbox"][1] + b["bbox"][3]) / 2

    vx = bx - prev_bx if prev_bx is not None else 0.0
    vy = by - prev_by if prev_by is not None else 0.0
    spd = math.sqrt(vx**2 + vy**2)
    prev_bx, prev_by = bx, by

    # Compute distance from ball to each eligible player
    candidates = []
    for p in players:
        team = p.get("team_name", "")
        if team not in ("left", "right"):
            continue
        h = p["bbox"][3] - p["bbox"][1]
        fx = (p["bbox"][0] + p["bbox"][2]) / 2
        fy = p["bbox"][3]
        dist = math.sqrt((bx - fx)**2 + (by - fy)**2)
        thr  = PROX * h
        candidates.append((dist, p["track_id"], team, dist, thr, dist <= thr))

    candidates.sort(key=lambda x: x[0])

    def fmt_cand(c):
        return f"{c[1]:>6} {c[2]:>5} {c[3]:>6.1f} {c[4]:>6.1f} {str(c[5]):>5}"

    p1 = fmt_cand(candidates[0]) if len(candidates) > 0 else " " * 34
    p2 = fmt_cand(candidates[1]) if len(candidates) > 1 else " " * 34

    contested = ""
    if len(candidates) >= 2:
        if candidates[0][5] and candidates[1][5] and candidates[0][2] != candidates[1][2]:
            contested = "CONTESTED"

    print(f"{fn:>4}  ({bx:>6.1f},{by:>6.1f})  {spd:>5.1f}  {p1}  {p2}  {contested}")
