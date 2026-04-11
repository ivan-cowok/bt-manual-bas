"""
Dual Soccer Tracker — Ball + People

Cleaned from track_from_detections_dual.py (soccergame project):
  - Removed CLI / file-loading helpers (unneeded here).
  - Removed utils_calib dependency (calib_module kept as Optional[object]).
  - Uses tracker/ package already copied into this project.

Classes exported:
  DualTrackerFromDetections  – main entry point
  DistanceBasedBallTracker   – inner ball tracker (BoTSORT subclass)
  ClassSpecificPeopleTracker – inner people tracker (BoTSORT subclass)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from tracker.bot_sort import BoTSORT, STrack
from tracker.basetrack import BaseTrack, TrackState
from tracker import matching

# ---------------------------------------------------------------------------
# ReID EMA constants (used by ClassSpecificPeopleTracker)
# ---------------------------------------------------------------------------
_REID_EMA_ALPHA     = 0.1   # weight of new observation (10% new, 90% old)
_REID_UPDATE_THRESH = 0.75  # minimum detection score to update the EMA


class Args:
    """Arguments container for BoT-SORT."""

    def __init__(self, **kwargs):
        self.track_high_thresh  = kwargs.get('track_high_thresh',  0.7)
        self.track_low_thresh   = kwargs.get('track_low_thresh',   0.1)
        self.new_track_thresh   = kwargs.get('new_track_thresh',   0.6)
        self.track_buffer       = kwargs.get('track_buffer',       30)
        self.match_thresh       = kwargs.get('match_thresh',       0.8)
        self.proximity_thresh   = kwargs.get('proximity_thresh',   0.5)
        self.appearance_thresh  = kwargs.get('appearance_thresh',  0.25)
        self.cmc_method         = kwargs.get('cmc_method',         'sparseOptFlow')
        self.mot20              = kwargs.get('mot20',              False)
        self.name               = kwargs.get('name',               'soccer')
        self.ablation           = kwargs.get('ablation',           False)
        self.with_reid          = kwargs.get('with_reid',          False)
        self.fast_reid_config   = kwargs.get('fast_reid_config',   None)
        self.fast_reid_weights  = kwargs.get('fast_reid_weights',  None)
        self.device             = kwargs.get('device',             'cpu')


# ---------------------------------------------------------------------------
# Ball tracker
# ---------------------------------------------------------------------------

class DistanceBasedBallTracker(BoTSORT):
    """Ball tracker: distance-based matching, BallKalmanFilter."""

    def __init__(
        self,
        args,
        frame_rate: int = 30,
        max_ball_speed_pixels: int = 250,
        calib_module: Optional[object] = None,
        resolution_scale: float = 1.0,
        enable_debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(args, frame_rate=frame_rate, calib_module=calib_module)
        self.max_ball_speed_pixels = int(max_ball_speed_pixels * resolution_scale)
        self.resolution_scale = resolution_scale
        self._max_ball_speed_base = max_ball_speed_pixels
        self.enable_debug = enable_debug
        if not hasattr(self, 'use_cmc'):
            self.use_cmc = (args.cmc_method != 'none')

        from tracker.ball_kalman_filter import BallKalmanFilter
        self.kalman_filter = BallKalmanFilter()

    def update(self, output_results, img=None, warp_matrix=None):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if len(output_results):
            scores  = output_results[:, 4]
            bboxes  = output_results[:, :4]
            classes = output_results[:, -1]
            lowest_inds = scores > self.args.track_low_thresh
            bboxes, scores, classes = bboxes[lowest_inds], scores[lowest_inds], classes[lowest_inds]
            remain_inds = scores > self.args.track_high_thresh
            dets, scores_keep, classes_keep = bboxes[remain_inds], scores[remain_inds], classes[remain_inds]
        else:
            bboxes = scores = classes = dets = scores_keep = classes_keep = []

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=cls)
                          for tlbr, s, cls in zip(dets, scores_keep, classes_keep)]
            for det in detections:
                det.enable_debug = self.enable_debug
        else:
            detections = []

        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (tracked_stracks if track.is_activated else unconfirmed).append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        if self.use_cmc and img is not None:
            warp = warp_matrix if warp_matrix is not None else self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # Primary matching (distance-based)
        if strack_pool and detections:
            dists = self.distance_cost_matrix(strack_pool, detections, self.max_ball_speed_pixels)
        else:
            dists = np.empty((len(strack_pool), len(detections)))
        matches, u_track, u_detection = self.matching_with_threshold(dists, self.max_ball_speed_pixels)

        for itracked, idet in matches:
            track, det = strack_pool[itracked], detections[idet]
            track.last_observed_tlbr = track.tlbr.copy()
            track.det_tlbr = det.tlbr.copy()
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)

        # Second association (low-score)
        if len(scores):
            inds_low   = scores > self.args.track_low_thresh
            inds_high  = scores < self.args.track_high_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second      = bboxes[inds_second]
            scores_second    = scores[inds_second]
            classes_second   = classes[inds_second]
        else:
            dets_second = scores_second = classes_second = []

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=cls)
                                  for tlbr, s, cls in zip(dets_second, scores_second, classes_second)]
            for det in detections_second:
                det.enable_debug = self.enable_debug
        else:
            detections_second = []

        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked and detections_second:
            dists2 = self.distance_cost_matrix(r_tracked, detections_second, self.max_ball_speed_pixels)
        else:
            dists2 = np.empty((len(r_tracked), len(detections_second)))
        matches2, u_track2, _ = self.matching_with_threshold(dists2, self.max_ball_speed_pixels)

        for itracked, idet in matches2:
            track, det = r_tracked[itracked], detections_second[idet]
            track.last_observed_tlbr = track.tlbr.copy()
            track.det_tlbr = det.tlbr.copy()
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)

        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost(); lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        if unconfirmed and detections:
            du = self.distance_cost_matrix(unconfirmed, detections, self.max_ball_speed_pixels)
        else:
            du = np.empty((len(unconfirmed), len(detections)))
        mu, u_unc, u_det3 = self.matching_with_threshold(du, self.max_ball_speed_pixels * 1.2)

        for itracked, idet in mu:
            track = unconfirmed[itracked]
            track.last_observed_tlbr = track.tlbr.copy()
            track.det_tlbr = detections[idet].tlbr.copy()
            track.update(detections[idet], self.frame_id); activated_stracks.append(track)
        for it in u_unc:
            unconfirmed[it].mark_removed(); removed_stracks.append(unconfirmed[it])

        # Extended ball recovery
        detections = [detections[i] for i in u_det3]
        extended_lost = [t for t in self.lost_stracks
                         if 10 < self.frame_id - t.end_frame <= self.max_time_lost]
        if extended_lost and detections:
            max_lost_frames = max(self.frame_id - t.end_frame for t in extended_lost)
            large_radius = (np.inf if max_lost_frames > 150
                            else self.max_ball_speed_pixels * 5 if max_lost_frames > 50
                            else self.max_ball_speed_pixels * 3)
            de = self.distance_cost_matrix(extended_lost, detections, large_radius)
            ext_matches, _, u_det4 = self.matching_with_threshold(de, large_radius)
            for itracked, idet in ext_matches:
                track, det = extended_lost[itracked], detections[idet]
                track.last_observed_tlbr = det.tlbr.copy()
                track.det_tlbr = det.tlbr.copy()
                track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            detections = [detections[i] for i in u_det4]

        # Allow only ONE ball track
        existing = [t for t in self.tracked_stracks if t.is_activated] + list(refind_stracks)
        lost_ball = list(self.lost_stracks)
        for track in detections:
            if track.score < self.args.new_track_thresh:
                continue
            if existing or lost_ball:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.last_observed_tlbr = track.tlbr.copy()
            track.det_tlbr = track.tlbr.copy()
            activated_stracks.append(track)
            existing.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed(); removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks
                if t.is_activated and t.state == TrackState.Tracked]

    def distance_cost_matrix(self, tracks, detections, max_distance):
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)))
        predicted = np.array([self._get_center(t.tlbr) for t in tracks])
        last_known = np.array([
            self._get_center(t.last_observed_tlbr) if hasattr(t, 'last_observed_tlbr')
            else self._get_center(t.tlbr)
            for t in tracks
        ])
        det_centers = np.array([self._get_center(d.tlbr) for d in detections])
        dp = predicted[:, None, :] - det_centers[None, :, :]
        dl = last_known[:, None, :] - det_centers[None, :, :]
        dists = np.minimum(
            np.sqrt(np.sum(dp**2, axis=2)),
            np.sqrt(np.sum(dl**2, axis=2))
        )
        if not np.isinf(max_distance):
            dists[dists > max_distance] = max_distance + 1e9
        return dists

    @staticmethod
    def _get_center(tlbr):
        x1, y1, x2, y2 = tlbr
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @staticmethod
    def matching_with_threshold(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, ua, ub = [], [], []
        matched_r, matched_c = set(), set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= thresh:
                matches.append((i, j)); matched_r.add(i); matched_c.add(j)
            else:
                ua.append(i); ub.append(j)
        ua += [i for i in range(cost_matrix.shape[0]) if i not in matched_r and i not in ua]
        ub += [j for j in range(cost_matrix.shape[1]) if j not in matched_c and j not in ub]
        return matches, ua, ub

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists, res = {}, []
        for t in tlista:
            exists[t.track_id] = 1; res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id):
                exists[t.track_id] = 1; res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        d = {t.track_id: t for t in tlista}
        for t in tlistb:
            d.pop(t.track_id, None)
        return list(d.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = np.zeros((len(stracksa), len(stracksb)))
        for i, ta in enumerate(stracksa):
            for j, tb in enumerate(stracksb):
                pdist[i, j] = matching.iou_distance([ta.tlbr], [tb.tlbr])[0, 0]
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            if stracksa[p].frame_id - stracksa[p].start_frame > stracksb[q].frame_id - stracksb[q].start_frame:
                dupb.append(q)
            else:
                dupa.append(p)
        return ([t for i, t in enumerate(stracksa) if i not in dupa],
                [t for j, t in enumerate(stracksb) if j not in dupb])


# ---------------------------------------------------------------------------
# People tracker
# ---------------------------------------------------------------------------

class ClassSpecificPeopleTracker(BoTSORT):
    """People tracker with class-specific matching and cross-class penalties."""

    def __init__(
        self,
        args,
        frame_rate: int = 30,
        cross_class_penalty: float = 0.3,
        max_goalkeepers: int = 2,
        max_referees: int = 3,
        calib_module: Optional[object] = None,
        resolution_scale: float = 1.0,
        enable_debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(args, frame_rate=frame_rate, calib_module=calib_module)
        self.cross_class_penalty = cross_class_penalty
        self.max_goalkeepers = max_goalkeepers
        self.max_referees    = max_referees
        self.resolution_scale = resolution_scale
        self.enable_debug    = enable_debug
        self.gk_min_distance      = int(600 * resolution_scale)
        self.gk_priority_distance = int(300 * resolution_scale)
        self.ref_spatial_max      = int(300 * resolution_scale)
        self.player_spatial_max   = int(200 * resolution_scale)
        if not hasattr(self, 'use_cmc'):
            self.use_cmc = (args.cmc_method != 'none')

        from tracker.people_kalman_filter import PeopleKalmanFilter
        self.kalman_filter = PeopleKalmanFilter()

    def update(self, output_results, img=None, warp_matrix=None, reid_embeddings=None):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if len(output_results):
            scores  = output_results[:, 4]
            bboxes  = output_results[:, :4]
            classes = output_results[:, -1]

            # Capture index mapping BEFORE variable reassignment so we can map
            # reid_embeddings (parallel to output_results rows) through both filters.
            lowest  = scores > self.args.track_low_thresh
            lowest_pos = np.where(lowest)[0]          # M original-row indices

            bboxes, scores, classes = bboxes[lowest], scores[lowest], classes[lowest]
            remain  = scores > self.args.track_high_thresh
            remain_pos_in_low = np.where(remain)[0]   # within M-sized arrays

            dets, scores_keep, classes_keep = bboxes[remain], scores[remain], classes[remain]

            # High-score detection embeddings (K items, parallel to dets/scores_keep)
            if reid_embeddings is not None and len(reid_embeddings) > 0:
                high_embs = [reid_embeddings[lowest_pos[p]] for p in remain_pos_in_low]
            else:
                high_embs = None
        else:
            bboxes = scores = classes = dets = scores_keep = classes_keep = []
            lowest_pos = np.array([], dtype=int)
            high_embs  = None

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=cls)
                          for tlbr, s, cls in zip(dets, scores_keep, classes_keep)]
            for det in detections:
                det.enable_debug = self.enable_debug
                det.reid_emb = None                   # default; overwritten below
            if high_embs is not None:
                for det, emb in zip(detections, high_embs):
                    det.reid_emb = emb
        else:
            detections = []

        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (tracked_stracks if track.is_activated else unconfirmed).append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        if self.use_cmc and img is not None:
            warp = warp_matrix if warp_matrix is not None else self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # ── helper: spatial distance limit per class ──────────────────────────
        def _spatial_max(tc):
            if tc == 1:   return self.gk_priority_distance   # GK
            elif tc == 3: return self.ref_spatial_max         # Referee
            else:         return self.player_spatial_max      # Player (class 2 or unknown)

        def _apply_spatial(cost_mat, tracks_list, dets_list):
            """Zero-out (set 1e9) pairs that exceed the track's per-class distance limit."""
            for i, trk in enumerate(tracks_list):
                tc   = int(trk.class_id) if hasattr(trk, 'class_id') and trk.class_id is not None else 2
                tc_c = self._get_center(trk.tlbr)
                maxd = _spatial_max(tc)
                for j, det in enumerate(dets_list):
                    dist = np.sqrt(np.sum((tc_c - self._get_center(det.tlbr)) ** 2))
                    if dist > maxd:
                        cost_mat[i, j] = 1e9
            return cost_mat

        # ── Stage 1: high-confidence detections ──────────────────────────────
        # ReID: 40% weight, hard gate 0.60.  EMA updated after confirmed match.
        if strack_pool and detections:
            iou_dist    = matching.iou_distance(strack_pool, detections)
            reid_dist   = self._compute_reid_distance(strack_pool, detections)
            cost_matrix = self._blend_costs(iou_dist, reid_dist, 0.60, 0.40, gate=0.60)
            cost_matrix = self.add_cross_class_penalty(strack_pool, detections, cost_matrix, 0.5)
            cost_matrix = _apply_spatial(cost_matrix, strack_pool, detections)
            matches, u_track, u_detection = self.matching_with_hungarian(
                cost_matrix, self.args.proximity_thresh)
        else:
            matches, u_track, u_detection = [], list(range(len(strack_pool))), list(range(len(detections)))

        for itracked, idet in matches:
            track, det = strack_pool[itracked], detections[idet]
            track.det_tlbr = det.tlbr.copy()
            track.detection_class_id = det.class_id
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            # EMA update: only for high-quality crops
            if det.score >= _REID_UPDATE_THRESH:
                track.update_reid_emb(det.reid_emb, alpha=_REID_EMA_ALPHA)

        # ── Stage 2: low-score detections ────────────────────────────────────
        # ReID: 25% weight (IoU dominates), no hard gate, NO EMA update.
        if len(scores):
            inds_second = np.logical_and(scores > self.args.track_low_thresh,
                                         scores < self.args.track_high_thresh)
            second_pos_in_low = np.where(inds_second)[0]
            dets_second    = bboxes[inds_second]
            scores_second  = scores[inds_second]
            classes_second = classes[inds_second]
            if reid_embeddings is not None and len(lowest_pos) > 0:
                second_embs = [reid_embeddings[lowest_pos[p]] for p in second_pos_in_low]
            else:
                second_embs = None
        else:
            dets_second = scores_second = classes_second = []
            second_embs = None

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=cls)
                                  for tlbr, s, cls in zip(dets_second, scores_second, classes_second)]
            for det in detections_second:
                det.enable_debug = self.enable_debug
                det.reid_emb = None
            if second_embs is not None:
                for det, emb in zip(detections_second, second_embs):
                    det.reid_emb = emb
        else:
            detections_second = []

        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked and detections_second:
            iou_d2 = matching.iou_distance(r_tracked, detections_second)
            reid_d2 = self._compute_reid_distance(r_tracked, detections_second)
            cm2    = self._blend_costs(iou_d2, reid_d2, 0.75, 0.25, gate=None)
            cm2    = self.add_cross_class_penalty(r_tracked, detections_second, cm2, 0.3)
            cm2    = _apply_spatial(cm2, r_tracked, detections_second)
        else:
            cm2 = np.empty((len(r_tracked), len(detections_second)))

        matches2, u_track2, _ = self.matching_with_hungarian(cm2, self.args.proximity_thresh)

        for itracked, idet in matches2:
            track, det = r_tracked[itracked], detections_second[idet]
            track.det_tlbr = det.tlbr.copy()
            track.detection_class_id = det.class_id
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False); refind_stracks.append(track)
            # Stage 2: low-quality crops — NO EMA update

        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost(); lost_stracks.append(track)

        # ── Stage 3: Unconfirmed tracks ───────────────────────────────────────
        # Pure IoU only — unconfirmed tracks have only one embedding observation
        # (too unreliable to gate on).  No EMA update.
        detections = [detections[i] for i in u_detection]
        if unconfirmed and detections:
            iu = matching.iou_distance(unconfirmed, detections)
            cu = self.add_cross_class_penalty(unconfirmed, detections, iu, 0.4)
        else:
            cu = np.empty((len(unconfirmed), len(detections)))

        mu3, u_unc, u_det3 = self.matching_with_hungarian(cu, self.args.proximity_thresh,
                                                           allow_cross_class=False)
        for itracked, idet in mu3:
            unconfirmed[itracked].det_tlbr = detections[idet].tlbr.copy()
            unconfirmed[itracked].detection_class_id = detections[idet].class_id
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unc:
            unconfirmed[it].mark_removed(); removed_stracks.append(unconfirmed[it])

        # ── Stage 4: Recent-loss recovery ─────────────────────────────────────
        # ReID: 50% weight, relaxed gate 0.70, EMA updated on match.
        detections = [detections[i] for i in u_det3]
        recently_lost = (
            [t for t in self.lost_stracks
             if self.frame_id - t.end_frame <= 10
             and hasattr(t, 'class_id') and t.class_id in [1, 3]] +
            [t for t in self.lost_stracks
             if self.frame_id - t.end_frame <= 2
             and (not hasattr(t, 'class_id') or t.class_id == 2)]
        )
        if recently_lost and detections:
            ir     = matching.iou_distance(recently_lost, detections)
            reid_r = self._compute_reid_distance(recently_lost, detections)
            cr     = self._blend_costs(ir, reid_r, 0.50, 0.50, gate=0.70)
            cr     = self.add_cross_class_penalty(recently_lost, detections, cr, 0.35)
            cr     = _apply_spatial(cr, recently_lost, detections)
            rec_matches, _, u_det_r = self.matching_with_hungarian(cr, 0.5)
            for itracked, idet in rec_matches:
                track, det = recently_lost[itracked], detections[idet]
                track.det_tlbr = det.tlbr.copy()
                track.detection_class_id = det.class_id
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                if det.score >= _REID_UPDATE_THRESH:
                    track.update_reid_emb(det.reid_emb, alpha=_REID_EMA_ALPHA)
            detections = [detections[i] for i in u_det_r]

        # ── Stage 5: Extended recovery (GK ≤240 frames, REF ≤140 frames) ─────
        # ReID dominant: 80% weight, very relaxed gate 0.85, EMA updated.
        ext_gk  = [t for t in self.lost_stracks
                   if 10 < self.frame_id - t.end_frame <= 240
                   and hasattr(t, 'class_id') and t.class_id == 1]
        ext_ref = [t for t in self.lost_stracks
                   if 10 < self.frame_id - t.end_frame <= 140
                   and hasattr(t, 'class_id') and t.class_id == 3]
        ext_imp = ext_gk + ext_ref
        if ext_imp and detections:
            ie     = matching.iou_distance(ext_imp, detections)
            reid_e = self._compute_reid_distance(ext_imp, detections)
            ce     = self._blend_costs(ie, reid_e, 0.20, 0.80, gate=0.85)
            ce     = self.add_cross_class_penalty(ext_imp, detections, ce, 0.25)
            tc_arr = np.array([self._get_center(t.tlbr) for t in ext_imp])
            dc_arr = np.array([self._get_center(d.tlbr) for d in detections])
            sdist  = np.sqrt(np.sum((tc_arr[:, None, :] - dc_arr[None, :, :]) ** 2, axis=2))
            for i, track in enumerate(ext_imp):
                tc   = int(track.class_id) if hasattr(track, 'class_id') and track.class_id is not None else 2
                maxd = _spatial_max(tc)
                ce[i, sdist[i] > maxd] = 1e9
            ext_matches, _, u_det_e = self.matching_with_hungarian(ce, 0.8)
            for itracked, idet in ext_matches:
                track, det = ext_imp[itracked], detections[idet]
                track.det_tlbr = det.tlbr.copy()
                track.detection_class_id = det.class_id
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                if det.score >= _REID_UPDATE_THRESH:
                    track.update_reid_emb(det.reid_emb, alpha=_REID_EMA_ALPHA)
            detections = [detections[i] for i in u_det_e]

        # ── Init new tracks (respecting class limits) ─────────────────────────
        ex_gk   = [t for t in self.tracked_stracks if hasattr(t, 'class_id') and t.class_id == 1 and t.is_activated]
        ex_ref  = [t for t in self.tracked_stracks if hasattr(t, 'class_id') and t.class_id == 3 and t.is_activated]
        lost_gk = [t for t in self.lost_stracks if hasattr(t, 'class_id') and t.class_id == 1]
        lost_r  = [t for t in self.lost_stracks if hasattr(t, 'class_id') and t.class_id == 3]

        for track in detections:
            tc = track.class_id if hasattr(track, 'class_id') else 2
            if track.score < self.args.new_track_thresh:
                continue
            if tc == 1 and (len(ex_gk) + len(lost_gk)) >= self.max_goalkeepers:
                continue
            if tc == 3 and (len(ex_ref) + len(lost_r)) >= self.max_referees:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.detection_class_id = track.class_id
            track.det_tlbr = track.tlbr.copy()
            activated_stracks.append(track)
            if tc == 1: ex_gk.append(track)
            elif tc == 3: ex_ref.append(track)

        # ── Remove aged-out lost tracks (class-specific persistence) ──────────
        for track in self.lost_stracks:
            tc = track.class_id if hasattr(track, 'class_id') and track.class_id is not None else 2
            max_lt = (self.max_time_lost * 10 if tc == 1 else
                      self.max_time_lost * 6  if tc == 3 else
                      self.max_time_lost * 4)
            if self.frame_id - track.end_frame > max_lt:
                track.mark_removed(); removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks
                if t.is_activated and t.state == TrackState.Tracked]

    @staticmethod
    def _get_center(tlbr):
        x1, y1, x2, y2 = tlbr
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def add_cross_class_penalty(self, tracks, detections, iou_dist, penalty_value=0.3):
        # Asymmetric penalty matrix [track_class][det_class] (classes 0..3)
        pm = [
            [0.0, 0.0,  0.0,  0.0],   # class 0 (unused)
            [0.0, 0.0,  0.4,  0.6],   # GK track
            [0.0, 0.8,  0.0,  0.9],   # Player track
            [0.0, 0.6,  0.35, 0.0],   # REF track
        ]
        cost = iou_dist.copy()
        for i, track in enumerate(tracks):
            tc = int(track.class_id) if hasattr(track, 'class_id') and track.class_id is not None else 2
            for j, det in enumerate(detections):
                dc = int(det.class_id) if hasattr(det, 'class_id') and det.class_id is not None else 2
                if tc != dc:
                    ds = det.score if hasattr(det, 'score') else 0.5
                    base = pm[tc][dc] if 0 <= tc < 4 and 0 <= dc < 4 else 0.5
                    factor = 1.0 if ds > 0.8 else 0.7 if ds > 0.5 else 0.5
                    cost[i, j] += base * factor
        return cost

    @staticmethod
    def _compute_reid_distance(tracks, detections) -> np.ndarray:
        """
        Cosine distance matrix between track EMA embeddings and detection embeddings.

        Shape: (n_tracks, n_dets), values in [0, 2].
        When either side has no embedding the cell is 0.0 (neutral — no penalty).
        """
        n_t, n_d = len(tracks), len(detections)
        if n_t == 0 or n_d == 0:
            return np.zeros((n_t, n_d), dtype=np.float32)
        cost = np.zeros((n_t, n_d), dtype=np.float32)
        for i, trk in enumerate(tracks):
            t_emb = getattr(trk, 'reid_emb', None)
            if t_emb is None:
                continue                             # row stays 0.0 (neutral)
            for j, det in enumerate(detections):
                d_emb = getattr(det, 'reid_emb', None)
                if d_emb is None:
                    continue                         # cell stays 0.0 (neutral)
                sim = float(np.dot(t_emb, d_emb))
                cost[i, j] = 1.0 - float(np.clip(sim, -1.0, 1.0))
        return cost

    @staticmethod
    def _blend_costs(iou_dist: np.ndarray, reid_dist: np.ndarray,
                     iou_w: float, reid_w: float,
                     gate: Optional[float] = None) -> np.ndarray:
        """
        Blend IoU and ReID distance matrices.

        Hard gate: if reid_dist > gate, the pair is blocked (cost set to 1e9)
        regardless of IoU.  Pass gate=None to disable.
        """
        cost = iou_w * iou_dist + reid_w * reid_dist
        if gate is not None:
            cost[reid_dist > gate] = 1e9
        return cost

    @staticmethod
    def matching_with_hungarian(cost_matrix, thresh, allow_cross_class=True):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        max_cost = thresh + 0.5 if allow_cross_class else thresh
        cm = cost_matrix.copy()
        cm[cm > max_cost] = 1e9
        row_ind, col_ind = linear_sum_assignment(cm)
        matches, ua, ub = [], [], []
        matched_r, matched_c = set(), set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= max_cost:
                matches.append((i, j)); matched_r.add(i); matched_c.add(j)
            else:
                ua.append(i); ub.append(j)
        ua += [i for i in range(cost_matrix.shape[0]) if i not in matched_r and i not in ua]
        ub += [j for j in range(cost_matrix.shape[1]) if j not in matched_c and j not in ub]
        return matches, ua, ub

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists, res = {}, []
        for t in tlista: exists[t.track_id] = 1; res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id): exists[t.track_id] = 1; res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        d = {t.track_id: t for t in tlista}
        for t in tlistb: d.pop(t.track_id, None)
        return list(d.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = np.zeros((len(stracksa), len(stracksb)))
        for i, ta in enumerate(stracksa):
            for j, tb in enumerate(stracksb):
                pdist[i, j] = matching.iou_distance([ta.tlbr], [tb.tlbr])[0, 0]
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            if stracksa[p].frame_id - stracksa[p].start_frame > stracksb[q].frame_id - stracksb[q].start_frame:
                dupb.append(q)
            else:
                dupa.append(p)
        return ([t for i, t in enumerate(stracksa) if i not in dupa],
                [t for j, t in enumerate(stracksb) if j not in dupb])


# ---------------------------------------------------------------------------
# Main dual tracker
# ---------------------------------------------------------------------------

class DualTrackerFromDetections:
    """
    Dual tracker: distance-based ball + class-specific people.

    Usage::

        tracker = DualTrackerFromDetections(frame_rate=25)
        for frame, det_dict in frames_and_dets:
            _, ball_tracks, people_tracks = tracker.process_frame_with_detections(
                frame, det_dict
            )

    ``det_dict`` keys: ``boxes`` (N,4 xyxy float32), ``confidences`` (N,),
    ``class_ids`` (N,)  where class 0=ball, 1=GK, 2=player, 3=referee.
    """

    def __init__(
        self,
        ball_track_buffer: int = 250,
        ball_new_track_thresh: float = 0.05,
        ball_track_high_thresh: float = 0.05,
        ball_max_speed_pixels: int = 250,
        people_track_buffer: int = 30,
        people_match_thresh: float = 0.8,
        people_new_track_thresh: float = 0.6,
        people_track_high_thresh: float = 0.7,
        cross_class_penalty: float = 0.3,
        max_goalkeepers: int = 2,
        max_referees: int = 3,
        frame_rate: int = 25,
        use_cmc: bool = True,
        cmc_method: str = 'sparseOptFlow',
        use_reid: bool = False,
        fast_reid_config: Optional[str] = None,
        fast_reid_weights: Optional[str] = None,
        device: str = 'cpu',
        calib_module: Optional[object] = None,
        input_width: Optional[int] = None,
        input_height: Optional[int] = None,
        reference_width: int = 1920,
        verbose: bool = False,
    ):
        self.input_width  = input_width
        self.input_height = input_height
        self.reference_width = reference_width
        self.resolution_scale = (input_width / reference_width
                                 if input_width is not None else 1.0)
        self._resolution_detected = input_width is not None
        self.verbose = verbose
        self.calib_module = calib_module

        ball_args = Args(
            track_high_thresh=ball_track_high_thresh,
            track_low_thresh=0.01,
            new_track_thresh=ball_new_track_thresh,
            track_buffer=ball_track_buffer,
            match_thresh=0.3,
            proximity_thresh=0.95,
            cmc_method=cmc_method if use_cmc else 'none',
            name='ball',
        )
        people_args = Args(
            track_high_thresh=people_track_high_thresh,
            track_low_thresh=0.1,
            new_track_thresh=people_new_track_thresh,
            track_buffer=people_track_buffer,
            match_thresh=people_match_thresh,
            proximity_thresh=0.5,
            cmc_method=cmc_method if use_cmc else 'none',
            name='people',
            with_reid=use_reid,
            fast_reid_config=fast_reid_config,
            fast_reid_weights=fast_reid_weights,
            device=device,
        )

        self.ball_tracker = DistanceBasedBallTracker(
            ball_args,
            frame_rate=frame_rate,
            max_ball_speed_pixels=ball_max_speed_pixels,
            calib_module=calib_module,
            resolution_scale=self.resolution_scale,
        )
        self.people_tracker = ClassSpecificPeopleTracker(
            people_args,
            frame_rate=frame_rate,
            cross_class_penalty=cross_class_penalty,
            max_goalkeepers=max_goalkeepers,
            max_referees=max_referees,
            calib_module=calib_module,
            resolution_scale=self.resolution_scale,
        )

        if use_cmc:
            from tracker.gmc import GMC
            self.shared_gmc = GMC(method=cmc_method, downscale=2)
        else:
            self.shared_gmc = None

        self.frame_count = 0
        self._precomputed_gmc: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pre-computed GMC injection
    # ------------------------------------------------------------------

    def inject_gmc_matrices(self, matrices: np.ndarray) -> None:
        """
        Replace live GMC with pre-computed warp matrices.

        After injection, process_frame_with_detections() reads the warp matrix
        directly from self._precomputed_gmc[frame_count] — no optical flow or
        ORB is run, and the frame is never passed to any GMC object.

        The live shared_gmc is set to None so the old code path is unreachable.
        Sub-tracker .gmc attributes are replaced with GMCFromMatrices instances
        as a fallback safety net (only reached if warp_matrix is somehow None).

        Args:
            matrices: ndarray (N, 2, 3) float32 as returned by
                      tracking.gmc_precompute.precompute_gmc().
                      matrices[0] should be identity (first frame).
        """
        from tracker.gmc_from_matrices import GMCFromMatrices
        self._precomputed_gmc = matrices   # direct array — indexed by frame_count
        self.shared_gmc       = None       # live GMC disabled; pre-computed takes over
        # Sub-tracker fallbacks (only used if warp_matrix is None for some reason)
        self.ball_tracker.gmc   = GMCFromMatrices(matrices, method='sparseOptFlow')
        self.people_tracker.gmc = GMCFromMatrices(matrices, method='sparseOptFlow')

    def process_frame_with_detections(
        self,
        frame: np.ndarray,
        detections: dict,
        calib_module: Optional[object] = None,
        reid_embeddings: Optional[List] = None,
    ) -> Tuple[np.ndarray, List, List]:
        """
        Process one frame with pre-computed detections.

        Args:
            frame:           Original BGR frame (H, W, 3).
            detections:      Dict with 'boxes' (N,4 xyxy), 'confidences' (N,),
                             'class_ids' (N,).
            calib_module:    Optional calibration (ignored if None).
            reid_embeddings: Optional list of length N, parallel to the detection
                             arrays.  Each entry is a unit-vector embedding
                             (np.ndarray) or None for detections without a crop
                             (e.g. the ball).  Forwarded to the people tracker.

        Returns:
            (frame, ball_tracks, people_tracks)
        """
        # Auto-detect resolution from first frame
        if not self._resolution_detected and frame is not None:
            h, w = frame.shape[:2]
            self.input_width  = w
            self.input_height = h
            self.resolution_scale = w / self.reference_width
            self._resolution_detected = True
            self.ball_tracker.resolution_scale = self.resolution_scale
            self.ball_tracker.max_ball_speed_pixels = int(
                self.ball_tracker._max_ball_speed_base * self.resolution_scale)
            self.people_tracker.resolution_scale      = self.resolution_scale
            self.people_tracker.gk_min_distance       = int(600 * self.resolution_scale)
            self.people_tracker.gk_priority_distance  = int(300 * self.resolution_scale)
            self.people_tracker.ref_spatial_max        = int(300 * self.resolution_scale)
            self.people_tracker.player_spatial_max     = int(200 * self.resolution_scale)

        boxes      = detections.get('boxes',       np.zeros((0, 4), np.float32))
        confidences = detections.get('confidences', np.zeros(0, np.float32))
        class_ids  = detections.get('class_ids',   np.zeros(0, np.int32))

        if len(boxes) > 0:
            ball_mask   = class_ids == 0
            people_mask = (class_ids >= 1) & (class_ids <= 3)

            def _build(mask):
                if not np.any(mask):
                    return np.zeros((0, 6), np.float32)
                r = np.zeros((mask.sum(), 6), np.float32)
                r[:, :4] = boxes[mask]
                r[:, 4]  = confidences[mask]
                r[:, 5]  = class_ids[mask]
                return r

            ball_results   = _build(ball_mask)
            people_results = _build(people_mask)

            # Split ReID embeddings: only people detections go to the people tracker.
            # Ball detections have no ReID embedding (None).
            if reid_embeddings is not None:
                people_reid_embs = [reid_embeddings[i]
                                    for i in range(len(class_ids))
                                    if people_mask[i]]
            else:
                people_reid_embs = None
        else:
            ball_results     = np.zeros((0, 6), np.float32)
            people_results   = np.zeros((0, 6), np.float32)
            people_reid_embs = None

        # Get warp matrix: direct index lookup (pre-computed) or live computation.
        # Pre-computed path never touches the frame — pure array indexing.
        warp_matrix = None
        if self._precomputed_gmc is not None:
            if self.frame_count < len(self._precomputed_gmc):
                warp_matrix = self._precomputed_gmc[self.frame_count]
        elif self.shared_gmc is not None:
            all_dets = (np.vstack([ball_results, people_results])
                        if len(ball_results) > 0 and len(people_results) > 0
                        else ball_results if len(ball_results) > 0
                        else people_results if len(people_results) > 0
                        else np.zeros((0, 6), np.float32))
            warp_matrix = self.shared_gmc.apply(frame, all_dets)

        ball_tracks   = self.ball_tracker.update(ball_results,   frame, warp_matrix=warp_matrix)
        people_tracks = self.people_tracker.update(people_results, frame,
                                                   warp_matrix=warp_matrix,
                                                   reid_embeddings=people_reid_embs)

        # Enforce at most one ball track
        if len(ball_tracks) > 1:
            ball_tracks = [max(ball_tracks, key=lambda t: (t.score, t.tracklet_len))]

        self.frame_count += 1
        return frame, ball_tracks, people_tracks
