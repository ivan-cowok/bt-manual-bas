import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import Optional

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

try:
    # Try importing from local fast_reid directory first
    from fast_reid.fast_reid_interfece import FastReIDInterface
    REID_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try importing from installed fastreid package
        import fastreid
        # If fastreid is installed, we can create the interface
        # But we still need the wrapper, so mark as unavailable
        REID_AVAILABLE = False
        FastReIDInterface = None
    except ImportError:
        REID_AVAILABLE = False
        FastReIDInterface = None


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50, class_id=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.class_id = class_id  # Store class ID for multi-class tracking

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        # EMA ReID embedding (our pre-computed Stage-2 OSNet features).
        # Separate from smooth_feat (which is for FastReID online inference).
        self.reid_emb: Optional[np.ndarray] = None

        # Detection class history for dynamic class determination
        self.detection_class_history = deque(maxlen=30)  # Last 20 detections (~0.8s @ 25fps)
        if class_id is not None:
            self.detection_class_history.append(class_id)  # Initialize with first detection

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_reid_emb(self, emb: Optional[np.ndarray], alpha: float = 0.1) -> None:
        """
        Update EMA ReID embedding with a new observation.

        Uses Stage-2 pre-computed OSNet embeddings (separate from smooth_feat
        which is for the FastReID online path).

        Args:
            emb:   New embedding vector (will be L2-normalised internally).
                   Pass None to skip silently.
            alpha: Weight of the new observation (default 0.1 → 10% new, 90% old).
                   The EMA is re-normalised after every update to stay on the unit
                   sphere so that cosine distance remains valid.
        """
        if emb is None:
            return
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return
        emb_n = emb / norm                          # ensure unit vector
        if self.reid_emb is None:
            self.reid_emb = emb_n.copy()            # first observation: init directly
        else:
            self.reid_emb = (1.0 - alpha) * self.reid_emb + alpha * emb_n
            n = np.linalg.norm(self.reid_emb)
            if n > 1e-6:
                self.reid_emb /= n                  # re-normalise to unit sphere

    def update_class_from_history(self, min_consensus=0.7, min_history=5, enable_debug=False):
        """
        Update track class based on detection class history.
        
        This makes tracks self-correcting: if detector consistently reports
        a different class than the track's current class, we update it.
        
        Example: Track initially detected as "Referee" but next 15/20 detections
                 are "Player" -> Track class changes to "Player"
        
        Args:
            min_consensus: Minimum ratio to change class (0.7 = 70% agreement)
            min_history: Minimum history length before considering class change
            enable_debug: Print debug messages when class changes
        
        Returns:
            bool: True if class was changed, False otherwise
        """
        if len(self.detection_class_history) < min_history:
            return False  # Need minimum history
        
        # Count each class
        class_counts = {}
        for cls in self.detection_class_history:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Find majority class
        majority_class = max(class_counts, key=class_counts.get)
        majority_ratio = class_counts[majority_class] / len(self.detection_class_history)
        
        # Update if strong consensus and different from current class
        if majority_ratio >= min_consensus and majority_class != self.class_id:
            old_class = self.class_id
            self.class_id = majority_class
            if enable_debug:
                print(f"  [CLASS CHANGE] Track ID:{self.track_id} {old_class}->{majority_class} "
                      f"({class_counts[majority_class]}/{len(self.detection_class_history)} = {majority_ratio:.1%})")
            return True  # Class changed!
        
        return False  # No change

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
        # Add detection class to history and update track class
        if hasattr(new_track, 'class_id') and new_track.class_id is not None:
            self.detection_class_history.append(new_track.class_id)
            # Update track class based on detection history (self-correcting)
            enable_debug = getattr(self, 'enable_debug', False)
            self.update_class_from_history(min_consensus=0.7, min_history=5, enable_debug=enable_debug)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        
        # Add detection class to history
        if hasattr(new_track, 'class_id') and new_track.class_id is not None:
            self.detection_class_history.append(new_track.class_id)
            # Update track class based on detection history (self-correcting)
            enable_debug = getattr(self, 'enable_debug', False)
            self.update_class_from_history(min_consensus=0.7, min_history=5, enable_debug=enable_debug)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=25, calib_module=None):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            if not REID_AVAILABLE:
                raise ImportError(
                    "REID requested but fast_reid is not available.\n"
                    "Please install FastReID:\n"
                    "  1. Clone: git clone https://github.com/JDAI-CV/fast-reid.git\n"
                    "  2. Install: cd fast-reid && pip install -e .\n"
                    "  3. Or see INSTALL_REID.md for detailed instructions.\n"
                    "Alternatively, run without REID by removing --with-reid flag."
                )
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        # GMC with optional camera calibration support
        if hasattr(args, 'cmc_method') and args.cmc_method in ['calib', 'calibration']:
            self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation], calib_module=calib_module)
        else:
            self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        
        # Store calib_module for updates
        self.calib_module = calib_module

    def update_calib(self, calib_module):
        """Update camera calibration module for GMC."""
        self.calib_module = calib_module
        if hasattr(self.gmc, 'update_calib'):
            self.gmc.update_calib(calib_module)

    def update(self, output_results, img, warp_matrix=None):
        """
        Update tracker with detections.
        
        Args:
            output_results: Detection results
            img: Current frame
            warp_matrix: Optional pre-computed GMC warp matrix. If provided, GMC calculation is skipped.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Update GMC with current calibration if available (only if not using pre-computed warp)
        if warp_matrix is None and self.calib_module is not None and hasattr(self.gmc, 'update_calib'):
            self.gmc.update_calib(self.calib_module)

        if len(output_results):
            # Handle different input formats
            if output_results.shape[1] == 5:
                # Format: [x1, y1, x2, y2, score]
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = np.zeros(len(output_results), dtype=int)  # No class info
            elif output_results.shape[1] == 6:
                # Format: [x1, y1, x2, y2, score, class] (our format)
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, 5].astype(int)
            else:
                # Format: [x1, y1, x2, y2, score1, score2, class] (original BoT-SORT format)
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]
                classes = output_results[:, 6].astype(int) if output_results.shape[1] > 6 else np.zeros(len(output_results), dtype=int)

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds].astype(int)

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            # classes_keep is already extracted above
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, class_id=int(cls)) for
                              (tlbr, s, f, cls) in zip(dets, scores_keep, features_keep, classes_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=int(cls)) for
                              (tlbr, s, cls) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        if warp_matrix is not None:
            # Use pre-computed warp matrix (GMC already calculated once for this frame)
            warp = warp_matrix
        else:
            # Calculate GMC for this frame
            warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            # Extract class IDs from original classes array (already filtered by lowest_inds)
            classes_second = classes[inds_second].astype(int) if len(classes) > 0 else np.zeros(len(dets_second), dtype=int)
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id=int(cls)) for
                                 (tlbr, s, cls) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

