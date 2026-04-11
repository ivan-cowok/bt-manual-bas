#!/usr/bin/env python3
"""
Object Detector for Soccer Analysis Pipeline

STANDALONE VERSION - No PyTorch/TensorFlow required!
Uses only: onnxruntime, opencv-python, numpy

Adapted from detect_batch_images_gpu_accelerated.py
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple


def nms_class_agnostic(boxes, scores, class_ids, iou_threshold=0.9):
    """Apply Class-Agnostic Non-Maximum Suppression."""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while len(sorted_indices) > 0:
        best_idx = sorted_indices[0]
        keep_indices.append(best_idx)
        
        if len(sorted_indices) == 1:
            break
        
        best_box = boxes[best_idx]
        other_indices = sorted_indices[1:]
        other_boxes = boxes[other_indices]
        
        # Calculate IOU
        x1 = np.maximum(best_box[0], other_boxes[:, 0])
        y1 = np.maximum(best_box[1], other_boxes[:, 1])
        x2 = np.minimum(best_box[2], other_boxes[:, 2])
        y2 = np.minimum(best_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        union = best_area + other_areas - intersection
        iou = intersection / (union + 1e-6)
        
        keep_mask = iou < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    keep_indices = sorted(keep_indices)
    return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]


class SoccerHeuristicFilter:
    """Heuristic filter for soccer video detections."""
    
    def __init__(
        self,
        min_box_area_ratio: float = 0.00002,
        max_box_area_ratio: float = 0.05,
        max_balls: int = 1,
        max_goalkeepers: int = 2,
        max_players: int = 20,
        max_referees: int = 3,
        high_threshold: float = 0.6
    ):
        self.min_box_area_ratio = min_box_area_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.max_balls = max_balls
        self.max_goalkeepers = max_goalkeepers
        self.max_players = max_players
        self.max_referees = max_referees
        self.high_threshold = high_threshold
        
        # Class-specific constraints
        self.class_constraints = {
            0: {'min_aspect_ratio': 0.25, 'max_aspect_ratio': 5.0, 
                'max_size_ratio': 0.003, 'min_size_ratio': 0.000003},
            1: {'min_aspect_ratio': 0.2, 'max_aspect_ratio': 7.0,
                'max_size_ratio': 0.05, 'min_size_ratio': 0.00005},
            2: {'min_aspect_ratio': 0.2, 'max_aspect_ratio': 7.0,
                'max_size_ratio': 0.05, 'min_size_ratio': 0.000005},
            3: {'min_aspect_ratio': 0.2, 'max_aspect_ratio': 7.0,
                'max_size_ratio': 0.05, 'min_size_ratio': 0.0005}
        }
    
    def filter(self, boxes, confidences, class_ids, img_shape):
        """Apply heuristic filtering."""
        if len(boxes) == 0:
            return boxes, confidences, class_ids
        
        img_width, img_height = img_shape
        img_area = img_height * img_width
        keep_mask = np.ones(len(boxes), dtype=bool)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cls_id = int(class_ids[i])
            
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            aspect_ratio = box_height / box_width if box_width > 0 else 0
            area_ratio = box_area / img_area
            
            # Size constraints
            if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
                keep_mask[i] = False
                continue
            
            # Class-specific constraints
            if cls_id in self.class_constraints:
                constraints = self.class_constraints[cls_id]
                if (aspect_ratio < constraints['min_aspect_ratio'] or
                    aspect_ratio > constraints['max_aspect_ratio'] or
                    area_ratio > constraints['max_size_ratio'] or
                    area_ratio < constraints['min_size_ratio']):
                    keep_mask[i] = False
                    continue
            
            # Invalid box check
            if cls_id != 0 and (box_width < 5 or box_height < 5):
                keep_mask[i] = False
                continue
        
        # Apply mask
        filtered_boxes = boxes[keep_mask]
        filtered_confidences = confidences[keep_mask]
        filtered_class_ids = class_ids[keep_mask]
        
        # Class count limits
        class_max_counts = {
            0: self.max_balls,
            1: self.max_goalkeepers,
            2: self.max_players,
            3: self.max_referees
        }
        
        all_kept_indices = []
        for class_id in range(4):
            class_mask = filtered_class_ids == class_id
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            class_confidences = filtered_confidences[class_indices]
            sorted_indices = np.argsort(class_confidences)[::-1]
            sorted_class_indices = class_indices[sorted_indices]
            
            if class_id in class_max_counts:
                max_count = class_max_counts[class_id]
                sorted_class_indices = sorted_class_indices[:max_count]
            
            all_kept_indices.extend(sorted_class_indices.tolist())
        
        all_kept_indices = np.array(all_kept_indices, dtype=np.int32)
        return (filtered_boxes[all_kept_indices], 
                filtered_confidences[all_kept_indices], 
                filtered_class_ids[all_kept_indices])


class ObjectDetector:
    """
    STANDALONE Object detector for soccer analysis pipeline.
    Uses ONLY: onnxruntime, opencv-python, numpy (NO PyTorch/TensorFlow!)
    
    Detects players, goalkeepers, referees, and balls.
    """
    
    CLASS_NAMES = {
        0: 'ball',
        1: 'goalkeeper',
        2: 'player',
        3: 'referee'
    }
    
    def __init__(
        self,
        model_path: str,
        providers: List[str],
        conf_threshold: float = 0.4,
        imgsz: int = 640
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to ONNX model
            providers: ONNX Runtime providers
            conf_threshold: Confidence threshold
            imgsz: Input image size
        """
        self.model_path = model_path
        self.providers = providers
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        # Initialize heuristic filter
        self.heuristic_filter = SoccerHeuristicFilter()
        
        # Initialize ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"[ObjectDetector] Model: {model_path}")
        print(f"[ObjectDetector] Input size: {imgsz}×{imgsz}")
        print(f"[ObjectDetector] Confidence threshold: {conf_threshold}")
        print(f"[ObjectDetector] Providers: {self.session.get_providers()}")
    
    def detect_batch(
        self, 
        frames_batch: np.ndarray,
        original_size: Tuple[int, int] = (1080, 1920)
    ) -> List[List[Dict]]:
        """
        Detect objects in a batch of frames.
        
        Args:
            frames_batch: [B, 640, 640, 3] BGR uint8 frames (already resized)
            original_size: Original video resolution (height, width) for bbox scaling
        
        Returns:
            List of detections for each frame.
            Each detection: {'class_id': int, 'confidence': float, 'bbox': [x, y, w, h]}
        """
        batch_size = len(frames_batch)
        
        # Preprocess using ONLY NumPy and OpenCV (no PyTorch!)
        input_tensor = self._preprocess_numpy(frames_batch)
        
        # Original size for scaling (dynamic based on input video)
        orig_size = np.array([list(original_size)] * batch_size, dtype=np.int64)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                'images': input_tensor,
                'orig_target_sizes': orig_size
            }
        )
        
        # Parse outputs
        labels, boxes, scores = outputs
        
        # Post-process each frame
        results = []
        for i in range(batch_size):
            # Filter by confidence
            ind = scores[i] > self.conf_threshold
            frame_labels = labels[i][ind]
            frame_boxes = boxes[i][ind]
            frame_scores = scores[i][ind]
            
            if len(frame_boxes) == 0:
                results.append([])
                continue
            
            # Apply heuristic filtering (use dynamic original resolution)
            boxes_filtered, scores_filtered, labels_filtered = self.heuristic_filter.filter(
                frame_boxes, frame_scores, frame_labels, (original_size[1], original_size[0])
            )
            
            # Apply class-agnostic NMS
            boxes_nms, scores_nms, labels_nms = nms_class_agnostic(
                boxes_filtered, scores_filtered, labels_filtered, iou_threshold=0.9
            )
            
            # Convert to pipeline format
            frame_detections = []
            for box, score, label in zip(boxes_nms, scores_nms, labels_nms):
                x1, y1, x2, y2 = box
                detection = {
                    'class_id': int(label),
                    'confidence': float(score),
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # [x, y, w, h]
                }
                frame_detections.append(detection)
            
            results.append(frame_detections)
        
        return results
    
    def _preprocess_numpy(self, frames_batch: np.ndarray) -> np.ndarray:
        """
        Preprocess using ONLY NumPy and OpenCV (NO PyTorch!).
        
        Args:
            frames_batch: [B, 640, 640, 3] BGR uint8
        
        Returns:
            input_tensor: [B, 3, 640, 640] RGB float32, normalized 0-1
        """
        # Frames are already 640×640 (pre-resized in pipeline)
        # Just need to:
        # 1. Convert BGR → RGB
        # 2. Normalize to 0-1
        # 3. Convert HWC → CHW
        
        # Convert BGR to RGB
        frames_rgb = frames_batch[..., ::-1].copy()
        
        # Normalize to 0-1
        frames_float = frames_rgb.astype(np.float32) / 255.0
        
        # HWC → CHW (NHWC → NCHW)
        frames_chw = np.transpose(frames_float, (0, 3, 1, 2))
        
        return frames_chw
