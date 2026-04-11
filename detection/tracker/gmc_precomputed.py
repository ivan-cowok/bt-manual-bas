"""
GMC (Global Motion Compensation) with Pre-computed Features

This version uses pre-extracted features instead of computing them on-the-fly,
significantly speeding up tracking.
"""

import cv2
import numpy as np
import copy


class GMCPrecomputed:
    """GMC using pre-computed features for fast tracking."""
    
    def __init__(self, precomputed_features, method='sparseOptFlow', downscale=2):
        """
        Initialize GMC with pre-computed features.
        
        Args:
            precomputed_features: Dict loaded from precompute_features_calibration.py
            method: GMC method used during precomputation
            downscale: Downscale factor used during precomputation
        """
        self.method = method
        self.downscale = downscale
        self.precomputed = precomputed_features
        self.num_frames = precomputed_features.get('num_frames', 0)
        
        # State tracking
        self.current_frame_idx = 0
        self.prev_features = None
        self.initializedFirstFrame = False
        
        # Initialize matcher based on method
        if self.method == 'orb':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == 'sift':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Don't print during initialization to avoid cluttering output
        # print(f"✓ GMC initialized with pre-computed features ({self.num_frames} frames)")
    
    def _load_frame_features(self, frame_idx: int):
        """Load pre-computed features for a specific frame."""
        prefix = f"gmc_{frame_idx:06d}_"
        
        features = {
            'frame_idx': frame_idx,
            'method': self.method
        }
        
        # Load based on method
        if self.method in ['orb', 'sift']:
            # Load keypoints
            kp_key = f"{prefix}keypoints"
            if kp_key in self.precomputed:
                kp_array = self.precomputed[kp_key]
                # Reconstruct cv2.KeyPoint objects
                keypoints = []
                for kp_data in kp_array:
                    kp = cv2.KeyPoint(
                        x=float(kp_data[0]),
                        y=float(kp_data[1]),
                        size=float(kp_data[2]),
                        angle=float(kp_data[3]),
                        response=float(kp_data[4]),
                        octave=int(kp_data[5]),
                        class_id=int(kp_data[6])
                    )
                    keypoints.append(kp)
                features['keypoints'] = keypoints
            else:
                features['keypoints'] = []
            
            # Load descriptors
            desc_key = f"{prefix}descriptors"
            if desc_key in self.precomputed:
                features['descriptors'] = self.precomputed[desc_key]
            else:
                features['descriptors'] = None
        
        elif self.method == 'sparseOptFlow':
            # Load corners
            corners_key = f"{prefix}corners"
            if corners_key in self.precomputed:
                features['corners'] = self.precomputed[corners_key]
            else:
                features['corners'] = None
            
            # Load gray frame for optical flow
            gray_key = f"{prefix}gray"
            if gray_key in self.precomputed:
                features['gray_frame'] = self.precomputed[gray_key]
            else:
                features['gray_frame'] = None
        
        # Load frame shape
        shape_key = f"{prefix}shape"
        if shape_key in self.precomputed:
            features['frame_shape'] = tuple(self.precomputed[shape_key])
        
        return features
    
    def apply(self, raw_frame, detections=None):
        """
        Apply GMC using pre-computed features.
        
        Args:
            raw_frame: Current frame (only used for shape, not for feature extraction)
            detections: Detections (not used with pre-computed features)
            
        Returns:
            2x3 affine transformation matrix
        """
        if self.current_frame_idx >= self.num_frames:
            print(f"⚠ Warning: Frame {self.current_frame_idx} exceeds pre-computed frames ({self.num_frames})")
            return np.eye(2, 3)
        
        # Load current frame features
        curr_features = self._load_frame_features(self.current_frame_idx)
        self.current_frame_idx += 1
        
        # Initialize on first frame
        if not self.initializedFirstFrame:
            self.prev_features = curr_features
            self.initializedFirstFrame = True
            return np.eye(2, 3)
        
        # Compute transformation based on method
        if self.method in ['orb', 'sift']:
            H = self._compute_transform_features(curr_features)
        elif self.method == 'sparseOptFlow':
            H = self._compute_transform_optflow(curr_features)
        else:
            H = np.eye(2, 3)
        
        # Update previous features
        self.prev_features = curr_features
        
        return H
    
    def _compute_transform_features(self, curr_features):
        """Compute transformation using feature matching (ORB/SIFT)."""
        H = np.eye(2, 3)
        
        prev_keypoints = self.prev_features.get('keypoints', [])
        prev_descriptors = self.prev_features.get('descriptors')
        curr_keypoints = curr_features.get('keypoints', [])
        curr_descriptors = curr_features.get('descriptors')
        
        if (not prev_keypoints or not curr_keypoints or 
            prev_descriptors is None or curr_descriptors is None or
            len(prev_descriptors) == 0 or len(curr_descriptors) == 0):
            return H
        
        # Match descriptors
        try:
            knnMatches = self.matcher.knnMatch(prev_descriptors, curr_descriptors, 2)
        except:
            return H
        
        if not knnMatches or len(knnMatches) < 2:
            return H
        
        # Filter matches
        matches = []
        spatialDistances = []
        
        frame_shape = curr_features.get('frame_shape', (480, 640))
        height, width = frame_shape
        maxSpatialDistance = 0.25 * np.array([width, height])
        
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prev_point = np.array(prev_keypoints[m.queryIdx].pt)
                curr_point = np.array(curr_keypoints[m.trainIdx].pt)
                
                spatialDistance = np.linalg.norm(prev_point - curr_point)
                
                if spatialDistance < maxSpatialDistance.max():
                    matches.append(m)
                    spatialDistances.append(spatialDistance)
        
        if len(matches) < 10:
            return H
        
        # Extract matched points
        prevPoints = np.array([prev_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
        currPoints = np.array([curr_keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        # Estimate affine transformation
        try:
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, method=cv2.RANSAC)
            if H is None:
                H = np.eye(2, 3)
        except:
            H = np.eye(2, 3)
        
        return H
    
    def _compute_transform_optflow(self, curr_features):
        """Compute transformation using optical flow."""
        H = np.eye(2, 3)
        
        prev_corners = self.prev_features.get('corners')
        prev_gray = self.prev_features.get('gray_frame')
        curr_gray = curr_features.get('gray_frame')
        
        if prev_corners is None or prev_gray is None or curr_gray is None:
            return H
        
        if len(prev_corners) == 0:
            return H
        
        # Optical flow
        try:
            curr_corners, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_corners, None
            )
        except:
            return H
        
        if curr_corners is None or status is None:
            return H
        
        # Filter valid matches
        prev_points = prev_corners[status == 1]
        curr_points = curr_corners[status == 1]
        
        if len(prev_points) < 10:
            return H
        
        # Estimate affine transformation
        try:
            H, inliers = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
            if H is None:
                H = np.eye(2, 3)
        except:
            H = np.eye(2, 3)
        
        return H
    
    def reset(self):
        """Reset GMC state."""
        self.current_frame_idx = 0
        self.prev_features = None
        self.initializedFirstFrame = False

