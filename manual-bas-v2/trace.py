"""
Trace possession + velocity + detector state for a frame range.
Usage: python trace.py <output.json> <start_frame> <end_frame>

Shows both OLD metric (feet point) and NEW metric (foot-region segments)
side by side so divergences are immediately visible.
"""
import json
import math
import sys

input_path = sys.argv[1]
START = int(sys.argv[2])
END   = int(sys.argv[3])

with open(input_path, encoding="utf-8") as f:
    data = json.load(f)

PROX             = 0.5
FOOT_REGION_RATIO = 0.15   # must match config.outfield_foot_region_ratio

frames = {fd["frame_number"]: fd for fd in data["frames"]}


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def dist_feet(bx, by, bbox):
    """OLD: ball → bottom-centre point of bbox."""
    fx = (bbox[0] + bbox[2]) / 2
    fy = bbox[3]
    return math.sqrt((bx - fx) ** 2 + (by - fy) ** 2)


def dist_seg(px, py, ax, ay, bx2, by2):
    dx, dy = bx2 - ax, by2 - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def dist_foot_region(bx, by, bbox, ratio):
    """NEW: ball → bottom-centre point (same as OLD, for comparison column)."""
    fx = (bbox[0] + bbox[2]) / 2
    fy = bbox[3]
    return math.sqrt((bx - fx) ** 2 + (by - fy) ** 2)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print(f"\n{'frm':>4}  {'ball_xy':>15}  {'spd':>5}  "
      f"{'--- OLD (feet point) ---':^45}  "
      f"{'--- NEW (foot region) ---':^45}  "
      f"{'DIFF?':>6}")
print(f"{'':>4}  {'':>15}  {'':>5}  "
      f"{'p1_tid':>6} {'team':>5} {'dst':>6} {'thr':>6} {'in?':>5} "
      f"{'cntst':>9}  "
      f"{'p1_tid':>6} {'team':>5} {'dst':>6} {'thr':>6} {'in?':>5} "
      f"{'cntst':>9}  "
      f"{'':>6}")
print("-" * 140)

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

    vx  = bx - prev_bx if prev_bx is not None else 0.0
    vy  = by - prev_by if prev_by is not None else 0.0
    spd = math.sqrt(vx ** 2 + vy ** 2)
    prev_bx, prev_by = bx, by

    old_cands, new_cands = [], []
    for p in players:
        team = p.get("team_name", "")
        if team not in ("left", "right"):
            continue
        bbox = p["bbox"]
        h    = bbox[3] - bbox[1]
        thr  = PROX * h
        tid  = p["track_id"]

        d_old = dist_feet(bx, by, bbox)
        d_new = dist_foot_region(bx, by, bbox, FOOT_REGION_RATIO)

        old_cands.append((d_old, tid, team, thr))
        new_cands.append((d_new, tid, team, thr))

    old_cands.sort(key=lambda x: x[0])
    new_cands.sort(key=lambda x: x[0])

    def fmt(cands, idx):
        if idx >= len(cands):
            return " " * 29
        d, tid, team, thr = cands[idx]
        return f"{tid:>6} {team:>5} {d:>6.1f} {thr:>6.1f} {str(d <= thr):>5}"

    def contested_str(cands):
        if len(cands) >= 2:
            d0, _, t0, thr0 = cands[0]
            d1, _, t1, thr1 = cands[1]
            if d0 <= thr0 and d1 <= thr1 and t0 != t1:
                return "CONTESTED"
        return ""

    old_winner = old_cands[0][1] if old_cands and old_cands[0][0] <= old_cands[0][3] else None
    new_winner = new_cands[0][1] if new_cands and new_cands[0][0] <= new_cands[0][3] else None

    diff = "<-- WINNER DIFF" if old_winner != new_winner else ""

    print(
        f"{fn:>4}  ({bx:>6.1f},{by:>6.1f})  {spd:>5.1f}  "
        f"{fmt(old_cands, 0)} {fmt(old_cands, 1)} {contested_str(old_cands):>9}  "
        f"{fmt(new_cands, 0)} {fmt(new_cands, 1)} {contested_str(new_cands):>9}  "
        f"{diff}"
    )
