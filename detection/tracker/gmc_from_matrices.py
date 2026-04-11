"""
GMC using pre-computed transformation matrices.

This loads exact transformation matrices computed by the original GMC,
ensuring 100% identical tracking results with significant speedup.
"""

import numpy as np


class GMCFromMatrices:
    """GMC that uses pre-computed transformation matrices."""
    
    def __init__(self, matrices, method='sparseOptFlow'):
        """
        Initialize GMC with pre-computed matrices.
        
        Args:
            matrices: numpy array of shape (num_frames, 2, 3) containing transformation matrices
            method: GMC method used (for compatibility)
        """
        self.method = method
        self.matrices = matrices
        self.num_frames = len(matrices)
        self.current_frame_idx = 0
        self.initializedFirstFrame = False
        
        # For compatibility with original GMC interface
        self.downscale = 2
        self.prev_rotation = None
        self.prev_position = None
        self.prev_calibration = None
        self.calib_module = None
    
    def apply(self, raw_frame, detections=None):
        """
        Get pre-computed transformation matrix for current frame.
        
        Args:
            raw_frame: Current frame (not used, only for interface compatibility)
            detections: Detections (not used)
            
        Returns:
            2x3 affine transformation matrix (pre-computed)
        """
        if self.current_frame_idx >= self.num_frames:
            print(f"[WARNING] GMCFromMatrices: frame {self.current_frame_idx} "
                  f"exceeds pre-computed frames ({self.num_frames}), returning identity")
            return np.eye(2, 3, dtype=np.float32)
        
        # Get pre-computed matrix
        H = self.matrices[self.current_frame_idx].copy()
        self.current_frame_idx += 1
        
        # Mark as initialized after first frame
        if not self.initializedFirstFrame:
            self.initializedFirstFrame = True
        
        return H
    
    def update_calib(self, calib_module):
        """Update calibration module (for interface compatibility)."""
        self.calib_module = calib_module
    
    def reset(self):
        """Reset GMC state."""
        self.current_frame_idx = 0
        self.initializedFirstFrame = False

