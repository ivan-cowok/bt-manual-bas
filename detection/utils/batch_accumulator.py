"""
Batch Accumulator

Accumulates individual frames into batches for efficient GPU processing.
"""

import numpy as np
from typing import List, Tuple, Optional


class BatchAccumulator:
    """
    Accumulate frames into batches.
    Processes batch when full or on flush.
    """
    
    def __init__(self, batch_size: int = 4):
        """
        Initialize accumulator.
        
        Args:
            batch_size: Number of frames per batch
        """
        self.batch_size = batch_size
        self.reset()
    
    def reset(self):
        """Clear current batch"""
        self.frames = []
        self.frame_numbers = []
    
    def add_frame(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        Add ONE frame to accumulator.
        
        Args:
            frame: Single frame [H, W, 3]
            frame_number: Frame index
        
        Returns:
            True if batch is now full (ready to process)
        """
        self.frames.append(frame)
        self.frame_numbers.append(frame_number)
        
        return len(self.frames) >= self.batch_size
    
    def is_full(self) -> bool:
        """Check if batch is ready"""
        return len(self.frames) >= self.batch_size
    
    def is_empty(self) -> bool:
        """Check if accumulator is empty"""
        return len(self.frames) == 0
    
    def get_batch(self) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Get accumulated batch and reset.
        
        Returns:
            frames_batch: [B, H, W, 3] stacked frames (or None if empty)
            frame_numbers: List of frame indices
        """
        if self.is_empty():
            return None, []
        
        frames_batch = np.stack(self.frames, axis=0)
        frame_numbers = self.frame_numbers.copy()
        
        self.reset()
        
        return frames_batch, frame_numbers
    
    def __len__(self):
        """Current batch size"""
        return len(self.frames)

