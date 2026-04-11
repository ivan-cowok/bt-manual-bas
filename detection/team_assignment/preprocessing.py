"""
Team Crop Preprocessing Module

Two preprocessing strategies:

TeamCropPreprocessor (original / autoencoder):
  1. Resize crop to 100×100
  2. Extract center 50% (scale=0.50)
  3. Resize to 64×64
  4. Normalize to [0, 1], NHWC format

ReIDCropPreprocessor (OSNet ReID):
  1. Resize crop directly to 128×256 (W×H)
  2. BGR → RGB
  3. Apply ImageNet mean/std normalization
  4. Transpose HWC → CHW  →  NCHW batch format
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# ImageNet normalization constants (matches extract_feature_vector_onnx.py)
_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# OSNet input size: height=256, width=128
_REID_HEIGHT = 256
_REID_WIDTH  = 128


class TeamCropPreprocessor:
    """
    Preprocess player crops for team classification.
    Matches project-RG preprocessing exactly.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Final output size (default: 64×64)
        """
        self.target_size = target_size
        self.intermediate_size = (100, 100)
        self.center_scale = 0.50
    
    def preprocess_single_crop(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Preprocess single player crop.
        
        Args:
            frame: Full frame image [H, W, 3] BGR
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Preprocessed crop [64, 64, 3] or None if invalid
        """
        x1, y1, x2, y2 = bbox
        
        # Clip to frame boundaries
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        # Check validity
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop from frame
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size == 0:
            return None
        
        try:
            # Step 1: Resize to 100×100 (matching main_onnx.py line 87)
            img = cv2.resize(crop_img, self.intermediate_size)
            
            # Step 2: Extract center 50% (matching main_onnx.py lines 89-95)
            center_x = img.shape[1] / 2
            center_y = img.shape[0] / 2
            width_scaled = img.shape[1] * self.center_scale
            height_scaled = img.shape[0] * self.center_scale
            
            left_x = center_x - width_scaled / 2
            right_x = center_x + width_scaled / 2
            top_y = center_y - height_scaled / 2
            bottom_y = center_y + height_scaled / 2
            
            img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
            
            # Step 3: Resize to 64×64 (matching main_onnx.py line 98)
            img_final = cv2.resize(img_cropped, self.target_size)
            
            return img_final
        
        except Exception as e:
            return None
    
    def preprocess_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Preprocess batch of crops from single frame.
        
        Args:
            frame: Full frame image [H, W, 3]
            bboxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        
        Returns:
            crops: List of valid preprocessed crops
            valid_indices: Indices of valid crops (for mapping back)
        """
        crops = []
        valid_indices = []
        
        for idx, bbox in enumerate(bboxes):
            crop = self.preprocess_single_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(idx)
        
        return crops, valid_indices
    
    def normalize_for_inference(
        self,
        crops: List[np.ndarray]
    ) -> np.ndarray:
        """
        Normalize crops for ONNX inference.
        
        Args:
            crops: List of crops [64, 64, 3] BGR uint8
        
        Returns:
            Normalized batch [B, 64, 64, 3] float32 (NHWC format)
        """
        if not crops:
            return np.array([], dtype=np.float32).reshape(0, 64, 64, 3)
        
        # Stack to batch
        batch = np.stack(crops, axis=0).astype(np.float32)  # [B, 64, 64, 3]
        
        # Normalize to [0, 1]
        batch = batch / 255.0
        
        # Keep in NHWC format (model expects [B, 64, 64, 3])
        # No transpose needed!
        
        return batch


# =============================================================================
# ReID Preprocessor (OSNet / torchreid)
# =============================================================================

class ReIDCropPreprocessor:
    """
    Preprocess player crops for OSNet ReID-based team classification.

    Pipeline (mirrors extract_feature_vector_onnx.py):
      1. Crop player from frame
      2. Resize to 128×256 (W×H)
      3. BGR → RGB
      4. Divide by 255, apply ImageNet mean/std
      5. Transpose HWC → CHW
    """

    def __init__(
        self,
        height: int = _REID_HEIGHT,
        width: int = _REID_WIDTH,
    ):
        """
        Args:
            height: Model input height (default: 256)
            width:  Model input width  (default: 128)
        """
        self.height = height
        self.width  = width

    def preprocess_single_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Preprocess one player crop.

        Args:
            frame: Full frame [H, W, 3] BGR uint8
            bbox:  Bounding box [x1, y1, x2, y2]

        Returns:
            Preprocessed crop [3, H, W] float32 (CHW, ImageNet-normalized),
            or None if the crop is invalid.
        """
        x1, y1, x2, y2 = bbox

        h_img, w_img = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            # Resize directly to model input size (no center-crop needed for ReID)
            resized = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # BGR → RGB
            rgb = resized[:, :, ::-1].astype(np.float32)

            # Normalize to [0, 1] then apply ImageNet stats
            rgb /= 255.0
            rgb = (rgb - _REID_MEAN) / _REID_STD  # HWC

            # HWC → CHW
            chw = rgb.transpose(2, 0, 1)  # [3, H, W]
            return chw

        except Exception:
            return None

    def preprocess_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Preprocess a batch of crops from one frame.

        Args:
            frame:  Full frame [H, W, 3] BGR
            bboxes: List of [x1, y1, x2, y2] boxes

        Returns:
            crops:         List of valid [3, H, W] float32 arrays
            valid_indices: Indices of bboxes that produced valid crops
        """
        crops: List[np.ndarray] = []
        valid_indices: List[int] = []

        for idx, bbox in enumerate(bboxes):
            crop = self.preprocess_single_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(idx)

        return crops, valid_indices

    def normalize_for_inference(
        self,
        crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Stack pre-processed CHW crops into an NCHW batch.

        Args:
            crops: List of [3, H, W] float32 arrays (already normalized)

        Returns:
            Batch [B, 3, H, W] float32 ready for OSNet ONNX inference
        """
        if not crops:
            return np.array([], dtype=np.float32).reshape(0, 3, self.height, self.width)

        return np.stack(crops, axis=0)  # [B, 3, H, W]
