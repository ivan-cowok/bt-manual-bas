"""
Keypoint Detector Module

Detects field line keypoints using dual ONNX models:
- keypoint.onnx: Detects point keypoints (e.g., corners, penalty spots)
- line.onnx: Detects line endpoints (e.g., sidelines, goal lines)

Uses ONNX Runtime for standalone inference (no PyTorch/TensorFlow).
"""

import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Dict, Tuple
from scipy.ndimage import maximum_filter
from keypoint_mapping_32 import convert_57_to_32, KEYPOINT_57_TO_32
from utils.complete_keypoints import complete_keypoints

class KeypointDetector:
    """
    Detects soccer field keypoints using dual ONNX models.
    """
    
    def __init__(
        self,
        keypoint_model_path: str,
        line_model_path: str,
        providers: List[str],
        kp_threshold: float = 0.05,
        line_threshold: float = 0.05,
        use_32_keypoints: bool = True
    ):
        """
        Initialize keypoint detector.
        
        Args:
            keypoint_model_path: Path to keypoint ONNX model
            line_model_path: Path to line ONNX model
            providers: ONNX Runtime providers
            kp_threshold: Keypoint confidence threshold
            line_threshold: Line confidence threshold
            use_32_keypoints: Use 32-keypoint system (vs 57-keypoint)
        """
        self.keypoint_model_path = keypoint_model_path
        self.line_model_path = line_model_path
        self.providers = providers
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.use_32_keypoints = use_32_keypoints
        self.model_size = (960, 540)  # Width x Height
        
        self._init_sessions()
    
    def _init_sessions(self):
        """Initialize ONNX sessions for both models."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load keypoint model
        self.session_kp = ort.InferenceSession(
            self.keypoint_model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.input_name_kp = self.session_kp.get_inputs()[0].name
        
        # Load line model
        self.session_line = ort.InferenceSession(
            self.line_model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.input_name_line = self.session_line.get_inputs()[0].name
        
        print(f"[KeypointDetector] Loaded: {self.keypoint_model_path}")
        print(f"[KeypointDetector] Loaded: {self.line_model_path}")
        print(f"[KeypointDetector] Model input size: {self.model_size}")
        print(f"[KeypointDetector] Providers: {self.session_kp.get_providers()}")
    
    def _preprocess_batch(self, frames_batch: np.ndarray) -> np.ndarray:
        """
        Preprocess batch of frames for keypoint detection.
        
        Args:
            frames_batch: [B, H, W, 3] BGR uint8
        
        Returns:
            Preprocessed batch: [B, 3, 540, 960] RGB float32
        """
        batch_size = frames_batch.shape[0]
        model_w, model_h = self.model_size
        
        # Preallocate output
        processed = np.zeros((batch_size, 3, model_h, model_w), dtype=np.float32)
        
        for i in range(batch_size):
            frame = frames_batch[i]
            
            # Resize to model input size
            resized = cv2.resize(frame, (model_w, model_h))
            
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Transpose to CHW and store
            processed[i] = np.transpose(normalized, (2, 0, 1))
        
        return processed
    
    def _get_keypoints_from_heatmap_batch(
        self,
        heatmap: np.ndarray,
        scale: int = 2,
        max_keypoints: int = 1,
        min_distance: int = 1,
        return_scores: bool = True
    ) -> np.ndarray:
        """
        Extract keypoints from heatmap batch using max pooling.
        
        Args:
            heatmap: [B, C, H, W] heatmap batch
            scale: Upscaling factor
            max_keypoints: Max keypoints per channel
            min_distance: Minimum distance between keypoints
            return_scores: Include confidence scores
        
        Returns:
            Array of keypoint coordinates per batch
        """
        batch_size, n_channels, height, width = heatmap.shape
        
        kernel = min_distance * 2 + 1
        pad = min_distance
        
        # Pad heatmap
        padded_heatmap = np.pad(
            heatmap,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode='constant',
            constant_values=0.0
        )
        
        # Max pooling
        max_pooled_heatmap = np.zeros_like(padded_heatmap)
        for b in range(batch_size):
            for c in range(n_channels):
                max_pooled_heatmap[b, c] = maximum_filter(
                    padded_heatmap[b, c],
                    size=(kernel, kernel),
                    mode='constant'
                )
        
        max_pooled_heatmap = max_pooled_heatmap[:, :, pad:-pad, pad:-pad]
        
        # Find local maxima
        local_maxima = (max_pooled_heatmap == heatmap).astype(np.float32)
        heatmap = heatmap * local_maxima
        
        # Extract top-k keypoints per channel
        filtered_indices = []
        for batch_idx in range(batch_size):
            batch_indices = []
            for channel_idx in range(n_channels):
                flat_heatmap = heatmap[batch_idx, channel_idx].flatten()
                top_k_indices = np.argsort(flat_heatmap)[-max_keypoints:][::-1]
                top_k_scores = flat_heatmap[top_k_indices]
                
                locs = []
                for idx, score in zip(top_k_indices, top_k_scores):
                    y = idx // width
                    x = idx % width
                    loc = [x * scale, y * scale]
                    if return_scores:
                        loc.append(float(score))
                    locs.append(loc)
                batch_indices.append(locs)
            filtered_indices.append(batch_indices)
        
        return np.array(filtered_indices, dtype=object)
    
    def _get_lines_from_heatmap_batch(
        self,
        heatmap: np.ndarray,
        scale: int = 2,
        max_keypoints: int = 2,
        min_distance: int = 1,
        return_scores: bool = True
    ) -> np.ndarray:
        """
        Extract line endpoints from heatmap batch.
        Similar to keypoint extraction but allows 2 points per channel.
        """
        batch_size, n_channels, height, width = heatmap.shape
        kernel = min_distance * 2 + 1
        pad = int((kernel - 1) / 2)
        
        # Max pooling
        max_pooled_heatmap = np.zeros_like(heatmap)
        for b in range(batch_size):
            for c in range(n_channels):
                max_pooled_heatmap[b, c] = maximum_filter(
                    heatmap[b, c],
                    size=(kernel, kernel),
                    mode='constant'
                )
        
        local_maxima = (max_pooled_heatmap == heatmap).astype(np.float32)
        heatmap = heatmap * local_maxima
        
        filtered_indices = []
        for batch_idx in range(batch_size):
            batch_indices = []
            for channel_idx in range(n_channels):
                flat_heatmap = heatmap[batch_idx, channel_idx].flatten()
                top_k_indices = np.argsort(flat_heatmap)[-max_keypoints:][::-1]
                top_k_scores = flat_heatmap[top_k_indices]
                
                locs = []
                for idx, score in zip(top_k_indices, top_k_scores):
                    y = idx // width
                    x = idx % width
                    loc = [x * scale, y * scale]
                    if return_scores:
                        loc.append(float(score))
                    locs.append(loc)
                batch_indices.append(locs)
            filtered_indices.append(batch_indices)
        
        return np.array(filtered_indices, dtype=object)
    
    def _coords_to_dict(
        self,
        kp_coords: np.ndarray,
        line_coords: np.ndarray,
        original_width: int,
        original_height: int
    ) -> Dict:
        """
        Convert coordinate arrays to dictionary format.
        
        Args:
            kp_coords: Keypoint coordinates
            line_coords: Line coordinates
            original_width: Original frame width
            original_height: Original frame height
        
        Returns:
            Dictionary with keypoint data scaled to original resolution (32-keypoint system)
        """
        model_w, model_h = self.model_size
        scale_x = original_width / model_w
        scale_y = original_height / model_h
        
        keypoints_57 = {}
        
        # Process keypoints (single points)
        for channel_idx, coords in enumerate(kp_coords):
            if len(coords) == 0:
                continue
            
            if len(coords) > 0 and len(coords[0]) >= 3:
                x, y, score = coords[0][:3]
                if score > self.kp_threshold:
                    # Scale to original resolution
                    keypoints_57[channel_idx + 1] = {
                        'x': float(x * scale_x),
                        'y': float(y * scale_y),
                        'confidence': float(score)
                    }
        
        # Process lines (two endpoints)
        line_channel_offset = len(kp_coords)
        for channel_idx, coords in enumerate(line_coords):
            if len(coords) < 2:
                continue
            
            if len(coords[0]) >= 3 and len(coords[1]) >= 3:
                x1, y1, score1 = coords[0][:3]
                x2, y2, score2 = coords[1][:3]
                
                if score1 > self.line_threshold and score2 > self.line_threshold:
                    # Scale to original resolution
                    keypoints_57[line_channel_offset + channel_idx + 1] = {
                        'x1': float(x1 * scale_x),
                        'y1': float(y1 * scale_y),
                        'confidence1': float(score1),
                        'x2': float(x2 * scale_x),
                        'y2': float(y2 * scale_y),
                        'confidence2': float(score2)
                    }
        
        # Convert from 57-keypoint system to 32-keypoint system
        if self.use_32_keypoints:
            keypoints_32 = convert_57_to_32(keypoints_57)
            
            # IMPORTANT: Preserve lines that don't map to 32-keypoint system
            # Lines are stored with IDs > len(kp_coords), which aren't in the mapping
            # So we need to add them back after conversion
            for orig_id, kp_data in keypoints_57.items():
                # If it's a line (has x1, y1, x2, y2) and not in 32-keypoint mapping
                if 'x1' in kp_data and 'x2' in kp_data:
                    if orig_id not in KEYPOINT_57_TO_32:
                        # This is a line that wasn't mapped - preserve it with original ID
                        keypoints_32[orig_id] = kp_data.copy()
            
            return keypoints_32
        else:
            return keypoints_57
    
    def detect_batch(
        self,
        frames_batch: np.ndarray,
        original_width: int,
        original_height: int
    ) -> List[List[float]]:
        """
        Detect keypoints on a batch of frames.
        
        Args:
            frames_batch: [B, H, W, 3] BGR uint8 frames
            original_width: Original frame width for scaling
            original_height: Original frame height for scaling
        
        Returns:
            List of keypoint lists (32 keypoints as [x1, y1, x2, y2, ..., x32, y32])
        """
        batch_size = frames_batch.shape[0]
        
        # Preprocess
        frames_preprocessed = self._preprocess_batch(frames_batch)
        
        # Run inference on both models
        heatmaps = self.session_kp.run(
            None,
            {self.input_name_kp: frames_preprocessed}
        )[0]

        heatmaps_l = self.session_line.run(
            None,
            {self.input_name_line: frames_preprocessed}
        )[0]

        # Extract coordinates (exclude last channel which is background)
        kp_coords = self._get_keypoints_from_heatmap_batch(
            heatmaps[:, :-1, :, :]
        )
        line_coords = self._get_lines_from_heatmap_batch(
            heatmaps_l[:, :-1, :, :],
            max_keypoints=2
        )
        
        # Convert to output format
        results = []
        for i in range(batch_size):
            # Get keypoint dict for this frame
            kp_dict = self._coords_to_dict(
                kp_coords[i],
                line_coords[i],
                original_width,
                original_height
            )
            
            # Separate keypoints and lines
            lines_dict = {}
            kp_dict_only = {}
            
            for kp_id, kp_data in kp_dict.items():
                if 'x1' in kp_data and 'x2' in kp_data:
                    # This is a line
                    lines_dict[kp_id] = kp_data.copy()
                else:
                    # This is a keypoint
                    kp_dict_only[kp_id] = kp_data.copy()
            
            # Complete missing keypoints using line intersections
            if len(lines_dict) >= 2:
                kp_dict_complete, lines_dict = complete_keypoints(
                    kp_dict_only,
                    lines_dict,
                    w=original_width,
                    h=original_height,
                    normalize=False,
                    use_line_mapping=True
                )
                
                # Merge completed keypoints back into main dict
                # (lines_dict is already updated)
                for kp_id, kp_data in kp_dict_complete.items():
                    if kp_id not in kp_dict or kp_id <= 32:
                        # Only update if it's a new keypoint or within 1-32 range
                        kp_dict[kp_id] = kp_data
            
            # Convert to flat list of 32 keypoints (64 values)
            keypoint_list = self._dict_to_flat_list(kp_dict)
            results.append(keypoint_list)
        
        return results
    
    def _dict_to_flat_list(self, kp_dict: Dict) -> List[float]:
        """
        Convert keypoint dictionary to flat list for JSON output.
        Output format: [x1, y1, x2, y2, ..., x32, y32]
        
        Creates a fixed-size array with 32 keypoints (64 values).
        Keypoint IDs 1-32 map to specific positions in the array.
        Missing keypoints are filled with 0.0.
        """
        # Initialize array with zeros (32 keypoints × 2 coordinates = 64 values)
        coords = [0.0] * 64
        
        for kp_id, kp in kp_dict.items():
            # Only process keypoints 1-32
            if kp_id < 1 or kp_id > 32:
                continue
            
            # Calculate position in flat array (0-indexed)
            idx = (kp_id - 1) * 2
            
            # Single point keypoint
            if 'x' in kp and 'y' in kp:
                coords[idx] = float(kp['x'])
                coords[idx + 1] = float(kp['y'])
            
            # Line keypoint (use first endpoint)
            elif 'x1' in kp and 'y1' in kp:
                coords[idx] = float(kp['x1'])
                coords[idx + 1] = float(kp['y1'])
        
        return coords

