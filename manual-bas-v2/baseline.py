"""Compare result.json against a hand-crafted baseline.json (no ground truth needed).

baseline.json uses the same format as result.json:
    {"predictions": [{"frame": N, "action": "event_type"}, ...]}

Usage:
    python baseline.py <video_number>
    python baseline.py <video_number> [result_path] [baseline_path]

Output: MATCH / MISS / FP table + summary counts (no scoring).
"""
import json
import os
import sys

# ---------------------------------------------------------------------------
# Tolerance window (seconds) per event type — used for loose frame matching.
# Baseline is "minimum expected", so we use generous windows.
# ---------------------------------------------------------------------------
TOLERANCES = {
    "pass":             1.0,
    "pass_received":    1.0,
    "recovery":         1.5,
    "tackle":           1.5,
    "interception":     2.0,
    "ball_out_of_play": 2.0,
    "clearance":        2.0,
    "take_on":          2.0,
    "shot":             2.0,
    "save":             2.0,
    "foul":             2.5,
    "goal":             3.0,
}
DEFAULT_TOLERANCE = 1.5   # fallback for unknown types

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python baseline.py <video_number> [result_path] [baseline_path]")
    sys.exit(1)

video_num = sys.argv[1]
data_dir  = os.path.join("data", video_num)

result_path   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, "result.json")
baseline_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(data_dir, "baseline.json")

# ---------------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------------
with open(result_path) as f:
    result = json.load(f)

with open(baseline_path) as f:
    baseline = json.load(f)

fps = result.get("metadata", {}).get("fps", 25.0)

def load_predictions(data: dict) -> list:
    """Return list of (action, frame) from a predictions dict."""
    if "predictions" in data:
        return [(p["action"], p["frame"]) for p in data["predictions"]]
    # fallback: internal event format
    return [(e["event_type"], e["frame_id"]) for e in data.get("events", [])]

result_events   = load_predictions(result)    # our detections
baseline_events = load_predictions(baseline)  # hand-crafted expected minimum

# ---------------------------------------------------------------------------
# Matching: for each baseline event find the closest result event of the
# same type within the tolerance window.
# ---------------------------------------------------------------------------
matched_result, matched_baseline = set(), set()
rows = []

for i, (b_action, b_frame) in enumerate(baseline_events):
    tol_frames = TOLERANCES.get(b_action, DEFAULT_TOLERANCE) * fps
    best_j, best_diff = None, float("inf")
    for j, (r_action, r_frame) in enumerate(result_events):
        if r_action == b_action and j not in matched_result:
            diff = abs(r_frame - b_frame)
            if diff <= tol_frames and diff < best_diff:
                best_j, best_diff = j, diff
    if best_j is not None:
        matched_result.add(best_j)
        matched_baseline.add(i)
        _, r_frame_m = result_events[best_j]
        rows.append(("MATCH", b_action, b_frame, r_frame_m, int(best_diff)))
    else:
        rows.append(("MISS", b_action, b_frame, None, None))

# False positives: result events not matched to any baseline event
for j, (r_action, r_frame) in enumerate(result_events):
    if j not in matched_result:
        rows.append(("FP", r_action, None, r_frame, None))

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
print(f"\nBaseline comparison — video {video_num}")
print(f"  Result  : {result_path}")
print(f"  Baseline: {baseline_path}\n")

COL = f"{'Status':<8} {'Action':<16} {'Base_frm':>8}  {'Our_frm':>8}  {'Frame_diff':>10}"
print(COL)
print("-" * len(COL))

# Sort: MATCHes first (by baseline frame), then MISSes, then FPs
def sort_key(row):
    status = row[0]
    order = {"MATCH": 0, "MISS": 1, "FP": 2}
    frame = row[2] if row[2] is not None else (row[3] if row[3] is not None else 9999)
    return (order[status], frame)

for row in sorted(rows, key=sort_key):
    status, action, b_frame, r_frame, diff = row
    bf   = f"{b_frame:>8}" if b_frame is not None else "       -"
    rf   = f"{r_frame:>8}" if r_frame is not None else "       -"
    df   = f"{diff:>10}" if diff    is not None else "         -"
    print(f"{status:<8} {action:<16} {bf}  {rf}  {df}")

n_match  = sum(1 for r in rows if r[0] == "MATCH")
n_miss   = sum(1 for r in rows if r[0] == "MISS")
n_fp     = sum(1 for r in rows if r[0] == "FP")
total_b  = len(baseline_events)

print(f"\nSummary  : Matched={n_match}/{total_b}  Missing={n_miss}  FP={n_fp}")
