"""
Frame Iterator

Unified interface for video files and image folders.
Yields ONE frame at a time for memory efficiency.
"""

import cv2
from pathlib import Path
from typing import Tuple, Dict


class FrameIterator:
    """
    Unified frame iterator for videos and image folders.
    Loads frames ONE at a time for memory efficiency.
    """
    
    def __init__(self, source_path: str):
        """
        Initialize frame iterator.
        
        Args:
            source_path: Path to video file OR image folder
        """
        self.source_path = Path(source_path)
        self.is_video = self._is_video_file()
        self.frame_count = 0
        
        if self.is_video:
            self._init_video()
        else:
            self._init_image_folder()
    
    def _is_video_file(self) -> bool:
        """Check if source is video file"""
        if self.source_path.is_file():
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            return self.source_path.suffix.lower() in video_exts
        return False
    
    def _init_video(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.source_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def _init_image_folder(self):
        """Initialize image folder iterator"""
        if not self.source_path.is_dir():
            raise ValueError(f"Source must be video or directory: {self.source_path}")
        
        # Collect image files
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.image_files = []
        
        for ext in image_exts:
            self.image_files.extend(self.source_path.glob(f'*{ext}'))
            self.image_files.extend(self.source_path.glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(set(self.image_files), key=lambda x: x.name)
        
        if not self.image_files:
            raise ValueError(f"No images found in: {self.source_path}")
        
        self.total_frames = len(self.image_files)
        self.current_idx = 0
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(self.image_files[0]))
        self.height, self.width = first_frame.shape[:2]
        self.fps = 25  # Default FPS for image sequences
    
    def __iter__(self):
        """Make iterator iterable"""
        return self
    
    def __next__(self) -> Tuple[cv2.Mat, int]:
        """
        Get next frame (ONE at a time).
        
        Returns:
            frame: BGR numpy array [H, W, 3]
            frame_number: Frame index
        """
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            frame_number = self.frame_count
        else:
            if self.current_idx >= len(self.image_files):
                raise StopIteration
            
            image_path = self.image_files[self.current_idx]
            frame = cv2.imread(str(image_path))
            
            if frame is None:
                raise ValueError(f"Failed to read: {image_path}")
            
            frame_number = self.current_idx
            self.current_idx += 1
        
        self.frame_count += 1
        return frame, frame_number
    
    def __len__(self):
        """Total number of frames"""
        return self.total_frames
    
    def get_properties(self) -> Dict:
        """Get video/image sequence properties"""
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'source_type': 'video' if self.is_video else 'images'
        }
    
    def close(self):
        """Release resources"""
        if self.is_video and hasattr(self, 'cap'):
            self.cap.release()

