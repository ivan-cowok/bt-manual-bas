"""Score a result.json against ground_truth.json using competition formula.

Usage:
    python score.py [result.json] [ground_truth.json]
    Defaults to result.json / ground_truth.json in the current directory.
"""
import json
import sys

result_path = sys.argv[1] if len(sys.argv) > 1 else "result.json"
gt_path     = sys.argv[2] if len(sys.argv) > 2 else "ground_truth.json"

with open(result_path) as f:
    result = json.load(f)
with open(gt_path) as f:
    gt = json.load(f)

fps = result.get("metadata", {}).get("fps", 25.0)

# event_type → (weight, tolerance_seconds)
ACTIONS = {
    "pass":             (1.0, 1.0),
    "pass_received":    (1.4, 1.0),
    "recovery":         (1.5, 1.5),
    "tackle":           (2.5, 1.5),
    "interception":     (2.8, 2.0),
    "ball_out_of_play": (2.9, 2.0),
    "clearance":        (3.1, 2.0),
    "take_on":          (3.2, 2.0),
    "substitution":     (4.2, 2.0),
    "block":            (4.2, 2.0),
    "arial_duel":       (4.3, 2.0),
    "shot":             (4.7, 2.0),
    "save":             (7.3, 2.0),
    "foul":             (7.7, 2.5),
    "goal":             (10.9, 3.0),
}
MIN_SCORE = 0.0

gt_events = [
    (e["type"], e["chunk_time_ms"], round(e["chunk_time_ms"] * fps / 1000))
    for e in gt
    if e["type"] in ACTIONS
]

if "predictions" in result:
    our_events = [
        (p["action"], round(p["frame"] * 1000 / fps), p["frame"])
        for p in result["predictions"]
        if p["action"] in ACTIONS
    ]
else:
    our_events = [
        (e["event_type"], e["timestamp_ms"], e["frame_id"])
        for e in result["events"]
        if e["event_type"] in ACTIONS
    ]

matched_our, matched_gt = set(), set()
rows = []

for i, (gt_type, gt_ms, gt_frame) in enumerate(gt_events):
    weight, tol_s = ACTIONS[gt_type]
    tol_ms = tol_s * 1000
    best_j, best_diff = None, float("inf")
    for j, (our_type, our_ms, our_frame) in enumerate(our_events):
        if our_type == gt_type and j not in matched_our:
            diff = abs(our_ms - gt_ms)
            if diff <= tol_ms and diff < best_diff:
                best_j, best_diff = j, diff
    if best_j is not None:
        matched_our.add(best_j)
        matched_gt.add(i)
        decay = 1 - (best_diff / tol_ms) * (1 - MIN_SCORE)
        pts = weight * decay
        _, our_ms_m, our_frame_m = our_events[best_j]
        rows.append(("MATCH", gt_type, gt_frame, gt_ms, our_frame_m, our_ms_m, best_diff, pts))
    else:
        rows.append(("MISS", gt_type, gt_frame, gt_ms, None, None, None, 0))

for j, (our_type, our_ms, our_frame) in enumerate(our_events):
    if j not in matched_our:
        weight, _ = ACTIONS[our_type]
        rows.append(("FP", our_type, None, None, our_frame, our_ms, None, -weight))

total_gt_weight = sum(ACTIONS[t][0] for t, _, _ in gt_events)
matched_score   = sum(r[7] for r in rows if r[0] == "MATCH")
fp_penalty      = sum(-r[7] for r in rows if r[0] == "FP")
raw_score       = (matched_score - fp_penalty) / total_gt_weight if total_gt_weight else 0.0
final_score     = max(0.0, min(1.0, raw_score))

print(f"{'Type':<12} {'Event':<16} {'GT_frm':>6}  {'GT_ms':>7}  {'Our_frm':>7}  {'Our_ms':>7}  {'Diff':>5}  {'Pts':>6}")
print("-" * 82)
for kind, evtype, gt_frame, gt_ms, our_frame, our_ms, diff, pts in rows:
    gf  = f"{gt_frame:>6}"  if gt_frame  is not None else "     -"
    gms = f"{gt_ms:>7}"     if gt_ms     is not None else "      -"
    of  = f"{our_frame:>7}" if our_frame is not None else "      -"
    oms = f"{our_ms:>7}"    if our_ms    is not None else "      -"
    dstr = f"{diff:>5}"     if diff      is not None else "    -"
    print(f"{kind:<12} {evtype:<16} {gf}  {gms}  {of}  {oms}  {dstr}  {pts:>6.3f}")

print(f"\nTotal GT weight : {total_gt_weight:.1f}")
print(f"Matched score   : {matched_score:.3f}")
print(f"FP penalty      : {fp_penalty:.3f}")
print(f"Raw score       : {raw_score:.4f}")
print(f"Final score     : {final_score:.4f}")
print(f"Summary         : Matched={len(matched_gt)}/{len(gt_events)}  FP={len(our_events) - len(matched_our)}")
