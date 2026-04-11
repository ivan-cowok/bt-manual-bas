"""
GMC (Global Motion Compensation) pre-computation for the soccer analysis pipeline.

Pre-computes all warp matrices before tracking starts, so that the tracker
does only a cheap matrix lookup per frame (no optical flow during tracking).

Two modes
---------
sequential (default)
    Runs frame-by-frame using the existing GMC object.
    Detection bounding boxes are passed so player regions are excluded from
    feature detection — uses the same improved masking logic in gmc.py.

parallel
    Two-phase multiprocessing.  Applies the SAME detection masking as the
    sequential mode so both modes produce identical results:
      Phase 1: extract features from every frame independently (parallel)
               → detection mask applied to goodFeaturesToTrack (PLACE 1)
      Phase 2: compute homographies for consecutive pairs (parallel)
               → prev keypoints falling inside current-frame detections
                 are filtered before optical flow (PLACE 2)

    Frames are pre-converted to downscaled grayscale in the MAIN PROCESS
    before dispatching to workers.  This avoids two Windows-specific issues:
      (a) BufferError: memoryview has 1 exported buffer — original frames may
          have active memoryview references from earlier pipeline slicing;
          pickle cannot serialize such arrays.
      (b) Excessive IPC pipe traffic — BGR 1920×1080 is ~6 MB/frame vs
          ~0.5 MB/frame after gray + downscale-2 conversion.

Output
------
np.ndarray of shape (N, 2, 3), dtype=float32
    matrices[0]  = identity  (first frame has no previous frame)
    matrices[i]  = 2×3 affine warp from frame i-1 → frame i
"""

import time
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import Dict, List, Optional

from tqdm import tqdm

from tracker.gmc import GMC


# ---------------------------------------------------------------------------
# Helper: build xyxy box array from detection dicts for one frame
# ---------------------------------------------------------------------------

def _build_boxes(dets: List[dict]) -> Optional[np.ndarray]:
    """Convert list of detection dicts (bbox=[x,y,w,h]) to (N,4) xyxy float32."""
    if not dets:
        return None
    return np.array(
        [[d['bbox'][0], d['bbox'][1],
          d['bbox'][0] + d['bbox'][2],
          d['bbox'][1] + d['bbox'][3]] for d in dets],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Sequential (detection-aware)
# ---------------------------------------------------------------------------

def precompute_gmc_sequential(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    frame_detections: Dict[int, List[dict]],
    method: str = 'sparseOptFlow',
    downscale: int = 2,
) -> np.ndarray:
    """
    Compute GMC warp matrices sequentially.

    Detection boxes are forwarded to the GMC object so that player/ball
    regions are excluded from corner / feature detection (masking fix in
    applySparseOptFlow).  Both PLACE 1 (new keypoint mask) and PLACE 2
    (prev-keypoint filter) are applied.

    Returns ndarray (N, 2, 3) float32.
    """
    gmc = GMC(method=method, downscale=downscale)
    matrices: List[np.ndarray] = []
    t0 = time.time()

    for frame_num, frame in tqdm(
        zip(frame_numbers, frames),
        total=len(frames),
        desc="[GMC] Sequential",
    ):
        boxes = _build_boxes(frame_detections.get(frame_num, []))
        H = gmc.apply(frame, boxes)
        matrices.append(H.copy())

    elapsed = time.time() - t0
    print(f"[GMC] Sequential done: {elapsed:.2f}s  "
          f"({elapsed / len(frames) * 1000:.1f} ms/frame)")
    return np.array(matrices, dtype=np.float32)


# ---------------------------------------------------------------------------
# Parallel workers  (must be module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _extract_features_worker(args):
    """
    Phase-1 worker: extract features from a pre-downscaled grayscale frame.

    args = (gray, method, downscale, det_boxes)
        gray:      already downscaled grayscale frame (H×W uint8 numpy array).
                   Prepared in the main process — no resize / cvtColor needed here.
        downscale: used only to scale det_box coordinates into gray-frame space.
        det_boxes: (N,4) xyxy float32 in original-frame coordinates, or None.

    Applies the same two-step masking as applySparseOptFlow PLACE 1:
      - Border exclusion (2% margin on all sides)
      - Detection-box exclusion (player/ball/GK/referee regions)

    Returns a lightweight feature dict — does NOT include the gray frame
    (the main process already holds gray_frames[] and will pass them
    directly to Phase-2 workers).
    """
    gray, method, downscale, det_boxes = args
    if gray is None:
        return None
    try:
        h, w = gray.shape[:2]

        if method == 'sparseOptFlow':
            # ── PLACE 1: build background-only mask ──────────────────────
            mask = np.ones((h, w), dtype=np.uint8) * 255
            # border exclusion
            mask[:int(0.02 * h), :]  = 0
            mask[int(0.98 * h):, :]  = 0
            mask[:, :int(0.02 * w)]  = 0
            mask[:, int(0.98 * w):]  = 0
            # detection-box exclusion
            if det_boxes is not None and len(det_boxes) > 0:
                for det in det_boxes:
                    tlbr = (det[:4] / downscale).astype(np.int_)
                    x1 = int(np.clip(tlbr[0], 0, w - 1))
                    y1 = int(np.clip(tlbr[1], 0, h - 1))
                    x2 = int(np.clip(tlbr[2], 0, w - 1))
                    y2 = int(np.clip(tlbr[3], 0, h - 1))
                    mask[y1:y2, x1:x2] = 0

            features = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )
            return {'features': features, 'method': method}

        elif method == 'orb':
            # PLACE 1: same border + detection-box mask as applyFeaures (sequential ORB)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(0.02 * h): int(0.98 * h), int(0.02 * w): int(0.98 * w)] = 255
            if det_boxes is not None and len(det_boxes) > 0:
                for det in det_boxes:
                    tlbr = (det[:4] / downscale).astype(np.int_)
                    y1 = int(np.clip(tlbr[1], 0, h - 1))
                    y2 = int(np.clip(tlbr[3], 0, h - 1))
                    x1 = int(np.clip(tlbr[0], 0, w - 1))
                    x2 = int(np.clip(tlbr[2], 0, w - 1))
                    mask[y1:y2, x1:x2] = 0
            orb = cv2.ORB_create(nfeatures=1000)
            kps, descs = orb.detectAndCompute(gray, mask)
            if kps:
                kp_data = np.array(
                    [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response)
                     for kp in kps],
                    dtype=np.float32,
                )
            else:
                kp_data = np.array([], dtype=np.float32)
            return {'keypoints': kp_data, 'descriptors': descs, 'method': method}

    except Exception:
        pass
    return None


def _compute_homography_worker(args):
    """
    Phase-2 worker: compute 2×3 affine warp between two consecutive frames.

    args = (feat_prev, feat_curr, gray_prev, gray_curr, method, downscale,
            det_boxes_curr)
        feat_prev/curr:  feature dicts returned by Phase 1 (keypoints only,
                         NO gray frame inside).
        gray_prev/curr:  downscaled grayscale frames passed from the main
                         process's gray_frames[] list.
        det_boxes_curr:  (N,4) xyxy float32 for frame i+1 in original coords.

    For sparseOptFlow, applies PLACE 2 masking before optical flow:
    any keypoint from frame i that now falls inside a detection box of
    frame i+1 is removed (a player may have walked over it).
    """
    feat_prev, feat_curr, gray_prev, gray_curr, method, downscale, det_boxes_curr = args
    identity = np.eye(2, 3, dtype=np.float32)

    if feat_prev is None or feat_curr is None:
        return identity

    try:
        if method == 'sparseOptFlow':
            prev_pts = feat_prev.get('features')
            if prev_pts is None or len(prev_pts) == 0:
                return identity

            # ── PLACE 2: filter prev keypoints covered by current detections ─
            if det_boxes_curr is not None and len(det_boxes_curr) > 0:
                pts = prev_pts.reshape(-1, 2)
                valid = np.ones(len(pts), dtype=bool)
                for det in det_boxes_curr:
                    tlbr = det[:4] / downscale   # scale to downscaled coords
                    in_box = (
                        (pts[:, 0] >= tlbr[0]) & (pts[:, 0] <= tlbr[2]) &
                        (pts[:, 1] >= tlbr[1]) & (pts[:, 1] <= tlbr[3])
                    )
                    valid &= ~in_box
                if valid.sum() > 4:
                    prev_pts = prev_pts[valid]

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, prev_pts, None
            )
            if curr_pts is None or status is None:
                return identity

            ok = status.flatten() == 1
            prev_pts = prev_pts[ok]
            curr_pts = curr_pts[ok]
            if len(prev_pts) < 4:
                return identity

            H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts,
                                               method=cv2.RANSAC)

        elif method == 'orb':
            prev_kp = feat_prev.get('keypoints', np.array([]))
            curr_kp = feat_curr.get('keypoints', np.array([]))
            prev_d  = feat_prev.get('descriptors')
            curr_d  = feat_curr.get('descriptors')

            if (prev_d is None or curr_d is None
                    or len(prev_kp) == 0 or len(curr_kp) == 0):
                return identity

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_matches = matcher.knnMatch(prev_d, curr_d, k=2)
            good = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < 4:
                return identity

            pp = np.float32([prev_kp[m.queryIdx][:2] for m in good])
            cp = np.float32([curr_kp[m.trainIdx][:2] for m in good])
            H, _ = cv2.estimateAffinePartial2D(pp, cp, method=cv2.RANSAC)

        else:
            return identity

        return H.astype(np.float32) if H is not None else identity

    except Exception:
        return identity


# ---------------------------------------------------------------------------
# Parallel (same detection masking as sequential)
# ---------------------------------------------------------------------------

def precompute_gmc_parallel(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    frame_detections: Dict[int, List[dict]],
    method: str = 'sparseOptFlow',
    downscale: int = 2,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Compute GMC warp matrices using 2-phase parallel multiprocessing.

    Applies the SAME detection masking as the sequential mode:
      Phase 1 (parallel): goodFeaturesToTrack with border + detection mask.
      Phase 2 (parallel): prev-keypoint filter by current-frame detections,
                          then optical flow + RANSAC.

    Returns ndarray (N, 2, 3) float32 — identical to sequential results.
    """
    if num_workers is None:
        num_workers = cpu_count()

    n = len(frames)
    t0 = time.time()

    # Build per-frame detection box arrays (original coords, xyxy float32)
    det_boxes_list: List[Optional[np.ndarray]] = [
        _build_boxes(frame_detections.get(fn, [])) for fn in frame_numbers
    ]

    # ── Pre-convert frames to downscaled grayscale in the MAIN PROCESS ───────
    #
    # WHY: The original frames from the pipeline may hold active memoryview
    # buffers (from numpy slicing / crop extraction in Stages 1-3).  Python's
    # pickle raises "BufferError: memoryview has 1 exported buffer" when it
    # tries to serialize such arrays across the IPC pipe, which crashes the
    # worker pool (Windows exit code 0xC0000005).
    #
    # By converting here we:
    #  (a) produce fresh, self-owned arrays with no external buffer references,
    #  (b) reduce IPC payload from ~6 MB/frame (BGR 1920×1080) to ~0.5 MB/frame
    #      (gray 960×540 for downscale=2) — a 12× reduction.
    #
    # Phase-1 workers receive the gray frame directly (no resize/cvtColor needed).
    # Phase-2 workers receive the same gray frames from this list (no round-trip
    # through Phase-1 results).
    print("[GMC] Pre-converting frames to grayscale ...", end=" ", flush=True)
    gray_frames: List[np.ndarray] = []
    for f in frames:
        h0, w0 = f.shape[:2]
        small = cv2.resize(f, (w0 // downscale, h0 // downscale)) if downscale > 1 else f
        # .copy() ensures the array owns its buffer (no active memoryview)
        gray_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).copy())
    print(f"done  ({len(gray_frames)} frames, "
          f"{gray_frames[0].nbytes // 1024} KB/frame)")

    # ── Phase 1: feature extraction with detection masking ────────────────
    extract_args = [
        (gray_frames[i], method, downscale, det_boxes_list[i])
        for i in range(n)
    ]
    with ThreadPool(num_workers) as pool:
        features = list(tqdm(
            pool.imap(_extract_features_worker, extract_args),
            total=n,
            desc=f"[GMC] Phase 1  ({num_workers} threads)",
        ))
    t1 = time.time()
    print(f"[GMC] Phase 1 done: {t1 - t0:.2f}s  "
          f"({(t1 - t0) / n * 1000:.1f} ms/frame)")

    # ── Phase 2: homographies with prev-keypoint filtering ────────────────
    # Gray frames are passed from the main-process list (not from Phase-1
    # results) — avoids a second pipe round-trip for each gray frame.
    homo_args = [
        (features[i], features[i + 1],
         gray_frames[i], gray_frames[i + 1],
         method, downscale, det_boxes_list[i + 1])
        for i in range(n - 1)
    ]
    with ThreadPool(num_workers) as pool:
        pair_matrices = list(tqdm(
            pool.imap(_compute_homography_worker, homo_args),
            total=len(homo_args),
            desc=f"[GMC] Phase 2  ({num_workers} threads)",
        ))
    t2 = time.time()
    print(f"[GMC] Phase 2 done: {t2 - t1:.2f}s  "
          f"({(t2 - t1) / len(pair_matrices) * 1000:.1f} ms/pair)")

    # First frame → identity (no previous frame)
    all_matrices = [np.eye(2, 3, dtype=np.float32)] + pair_matrices
    total = time.time() - t0
    print(f"[GMC] Parallel total: {total:.2f}s  "
          f"({total / n * 1000:.1f} ms/frame)  "
          f"threads={num_workers}")
    return np.array(all_matrices, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def precompute_gmc(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    frame_detections: Dict[int, List[dict]],
    method: str = 'sparseOptFlow',
    downscale: int = 2,
    parallel: bool = False,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Pre-compute GMC warp matrices for all frames.

    Both sequential and parallel modes apply identical detection masking,
    so results are equivalent regardless of which mode is chosen.

    Args:
        frames:            List of original BGR frames (from Stage 1).
        frame_numbers:     Corresponding frame indices.
        frame_detections:  Stage-1 detection dicts keyed by frame number.
        method:            GMC method: 'sparseOptFlow' | 'orb' | 'none'.
        downscale:         Frame downscale factor for GMC (default 2).
        parallel:          Use parallel processing (default False).
        num_workers:       Worker count for parallel mode (default: cpu_count).

    Returns:
        np.ndarray of shape (N, 2, 3), dtype float32.
        Inject into DualTrackerFromDetections with inject_gmc_matrices().
    """
    if method == 'none':
        n = len(frames)
        return np.tile(np.eye(2, 3, dtype=np.float32), (n, 1, 1))

    if parallel:
        return precompute_gmc_parallel(
            frames, frame_numbers, frame_detections, method, downscale, num_workers
        )
    else:
        return precompute_gmc_sequential(
            frames, frame_numbers, frame_detections, method, downscale
        )
