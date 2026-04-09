"""Evaluate all videos under data/ — summary only.

For each video directory that contains output.json:
  1. Generate result.json via main.py
  2. If ground_truth.json exists → compute score inline
  3. If baseline.json exists      → compute match/FP counts inline
  4. Otherwise                    → listed under "No evaluation"

For full details on a specific video use score.py or baseline.py directly.

Usage:
    python evaluate.py
"""
import json
import math
import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PYTHON   = sys.executable

# event_type → (weight, tolerance_seconds) — mirrors score.py
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
DEFAULT_TOLERANCE = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def video_dirs() -> list[str]:
    dirs = []
    for name in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "output.json")):
            dirs.append(name)
    def sort_key(n):
        try:
            return (0, int(n))
        except ValueError:
            return (1, n)
    return sorted(dirs, key=sort_key)


def load_predictions(data: dict, fps: float) -> list[tuple[str, float, int]]:
    """Return list of (action, timestamp_ms, frame)."""
    if "predictions" in data:
        return [
            (p["action"], round(p["frame"] * 1000 / fps), p["frame"])
            for p in data["predictions"]
            if p["action"] in ACTIONS
        ]
    return [
        (e["event_type"], e["timestamp_ms"], e["frame_id"])
        for e in data.get("events", [])
        if e["event_type"] in ACTIONS
    ]


def compute_score(result: dict, gt_list: list) -> tuple[float, int, int, int]:
    """Return (final_score, n_matched, n_gt, n_fp)."""
    fps = result.get("metadata", {}).get("fps", 25.0)
    our_events = load_predictions(result, fps)
    gt_events = [
        (e["type"], e["chunk_time_ms"], round(e["chunk_time_ms"] * fps / 1000))
        for e in gt_list
        if e["type"] in ACTIONS
    ]

    matched_our, matched_gt = set(), set()
    matched_score = 0.0
    fp_penalty    = 0.0

    for i, (gt_type, gt_ms, _) in enumerate(gt_events):
        weight, tol_s = ACTIONS[gt_type]
        tol_ms = tol_s * 1000
        best_j, best_diff = None, float("inf")
        for j, (our_type, our_ms, _) in enumerate(our_events):
            if our_type == gt_type and j not in matched_our:
                diff = abs(our_ms - gt_ms)
                if diff <= tol_ms and diff < best_diff:
                    best_j, best_diff = j, diff
        if best_j is not None:
            matched_our.add(best_j)
            matched_gt.add(i)
            decay = 1 - (best_diff / tol_ms)
            matched_score += weight * decay

    for j, (our_type, _, _) in enumerate(our_events):
        if j not in matched_our:
            fp_penalty += ACTIONS[our_type][0]

    total_gt_weight = sum(ACTIONS[t][0] for t, _, _ in gt_events)
    raw = (matched_score - fp_penalty) / total_gt_weight if total_gt_weight else 0.0
    final = max(0.0, min(1.0, raw))
    return final, len(matched_gt), len(gt_events), len(our_events) - len(matched_our)


def compute_baseline(result: dict, bl: dict) -> tuple[int, int, int]:
    """Return (n_matched, n_baseline, n_fp)."""
    fps = result.get("metadata", {}).get("fps", 25.0)

    def preds(data):
        if "predictions" in data:
            return [(p["action"], p["frame"]) for p in data["predictions"]]
        return [(e["event_type"], e["frame_id"]) for e in data.get("events", [])]

    r_events = preds(result)
    b_events = preds(bl)

    matched_r = set()
    matched_b = set()
    for i, (b_action, b_frame) in enumerate(b_events):
        tol = ACTIONS.get(b_action, (0, DEFAULT_TOLERANCE))[1] * fps
        best_j, best_diff = None, float("inf")
        for j, (r_action, r_frame) in enumerate(r_events):
            if r_action == b_action and j not in matched_r:
                diff = abs(r_frame - b_frame)
                if diff <= tol and diff < best_diff:
                    best_j, best_diff = j, diff
        if best_j is not None:
            matched_r.add(best_j)
            matched_b.add(i)

    fp = len(r_events) - len(matched_r)
    return len(matched_b), len(b_events), fp


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

videos = video_dirs()

gt_scores      = {}   # vid → (score, matched, total, fp)
baseline_stats = {}   # vid → (matched, total, fp)
no_eval        = []

for vid in videos:
    data_path   = os.path.join(DATA_DIR, vid)
    output_path = os.path.join(data_path, "output.json")
    result_path = os.path.join(data_path, "result.json")
    gt_path     = os.path.join(data_path, "ground_truth.json")
    bl_path     = os.path.join(data_path, "baseline.json")

    # Generate result.json in dev mode (no production post-processing)
    subprocess.run(
        [PYTHON, "main.py", output_path, result_path, "--dev"],
        capture_output=True, cwd=os.path.dirname(__file__)
    )

    with open(result_path) as f:
        result = json.load(f)

    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        score, matched, total, fp = compute_score(result, gt)
        gt_scores[vid] = (score, matched, total, fp)

    elif os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        matched, total, fp = compute_baseline(result, bl)
        baseline_stats[vid] = (matched, total, fp)

    else:
        no_eval.append(vid)

# ---------------------------------------------------------------------------
# Print summary only
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

if gt_scores:
    print("\nGround-truth videos:")
    for vid in sorted(gt_scores, key=lambda n: (0, int(n)) if n.isdigit() else (1, n)):
        score, matched, total, fp = gt_scores[vid]
        bar = "#" * int(score * 30)
        print(f"  Video {vid:>3}:  {score:.4f}  [{bar:<30}]  Matched={matched}/{total}  FP={fp}")

if baseline_stats:
    print("\nBaseline videos:")
    for vid in sorted(baseline_stats, key=lambda n: (0, int(n)) if n.isdigit() else (1, n)):
        matched, total, fp = baseline_stats[vid]
        print(f"  Video {vid:>3}:  Matched={matched}/{total}  FP={fp}")

if no_eval:
    print(f"\nNo evaluation: {', '.join(no_eval)}")
