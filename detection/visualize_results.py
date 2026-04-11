#!/usr/bin/env python3
"""
Soccer Game Analysis - Result Visualization

Visualizes detection results on video with:
- Color-coded bounding boxes (color indicates class and team)
- Field line keypoints
- Info overlay with counts
- Legend for color reference
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from utils.frame_iterator import FrameIterator


class SoccerResultVisualizer:
    """
    Visualize soccer game analysis results on video.
    Uses color-coded bounding boxes without text labels for clean visualization.
    """
    
    # Color scheme (BGR format for OpenCV)
    COLORS = {
        'team_a': (0, 255, 0),      # Green for Team A
        'team_b': (255, 0, 0),      # Blue for Team B
        'goalkeeper': (0, 165, 255), # Orange for goalkeepers
        'referee': (0, 255, 255),    # Yellow for referees
        'ball': (0, 0, 255),         # Red for ball
        'unknown': (128, 128, 128),  # Gray for unknown
        'keypoint': (255, 0, 255)    # Magenta for keypoints
    }
    
    def __init__(
        self,
        video_path: str,
        json_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        fps: Optional[int] = None,
        overlay_scale: float = 0.6,
    ):
        """
        Initialize visualizer.
        
        Args:
            video_path:     Path to input video
            json_path:      Path to analysis results JSON
            output_path:    Path to save output video (optional)
            display:        Whether to display in real-time
            fps:            Output video FPS (default: same as input)
            overlay_scale:  Scale multiplier for all on-screen text / overlays.
                            All overlays are placed at the bottom of the frame.
                            0.5 = small, 1.0 = larger, 0.0 = hide all overlays.
        """
        self.video_path = video_path
        self.json_path = json_path
        self.output_path = output_path
        self.display = display
        self.fps = fps
        self.overlay_scale = overlay_scale
        
        # Load results
        self.results = self._load_results()
        
        # Create frame lookup for fast access
        self.frame_lookup = {
            frame['frame_number']: frame
            for frame in self.results['frames']
        }
        
        print(f"Loaded results: {len(self.results['frames'])} frames")
    
    def _load_results(self) -> Dict:
        """Load JSON results."""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def visualize(self):
        """Run visualization."""
        # Open video or image folder using FrameIterator
        frame_iterator = FrameIterator(self.video_path)
        props = frame_iterator.get_properties()
        
        width = props['width']
        height = props['height']
        input_fps = props['fps']
        total_frames = props['total_frames']
        
        output_fps = self.fps if self.fps else input_fps
        
        print(f"Source: {width}×{height}, {input_fps:.2f} fps, {total_frames} frames")
        
        # Create video writer if saving
        writer = None
        frames_dir = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                output_fps,
                (width, height)
            )
            print(f"Saving to: {self.output_path}")

            # Automatically save annotated frames alongside the video.
            # e.g.  output/out.mp4  →  output/frames/
            frames_dir = os.path.join(os.path.dirname(self.output_path), 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Saving frames to: {frames_dir}/")

        try:
            for frame, frame_number in frame_iterator:
                # Draw annotations
                annotated = self._draw_frame(frame, frame_number)
                
                # Display
                if self.display:
                    try:
                        cv2.imshow('Soccer Analysis Results', annotated)
                        
                        # Press 'q' to quit, 'p' to pause
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quit by user")
                            break
                        elif key == ord('p'):
                            print("Paused (press any key to continue)")
                            cv2.waitKey(0)
                    except cv2.error as e:
                        print(f"⚠ Display not available: {e}")
                        print("⚠ Continuing without display (saving only)")
                        self.display = False  # Disable display for remaining frames
                
                # Write to output video
                if writer:
                    writer.write(annotated)

                # Save individual annotated frame as PNG (named by frame number)
                if frames_dir is not None:
                    cv2.imwrite(os.path.join(frames_dir, f"{frame_number}.png"), annotated)
                
                # Progress
                if (frame_number + 1) % 30 == 0:
                    progress = ((frame_number + 1) / total_frames) * 100
                    print(f"Progress: {frame_number + 1}/{total_frames} ({progress:.1f}%)")
        
        finally:
            if writer:
                writer.release()
            if self.display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # Ignore if GUI not available
        
        print(f"✓ Visualization complete: {frame_number + 1} frames processed")
    
    def _draw_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Draw annotations on a frame.
        
        Args:
            frame: Input frame
            frame_number: Frame index
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Get results for this frame
        frame_data = self.frame_lookup.get(frame_number)
        
        if not frame_data:
            # No results for this frame
            self._draw_info(annotated, frame_number, 0, 0, 0)
            return annotated
        
        # Draw keypoints (field lines)
        if frame_data.get('keypoints'):
            self._draw_keypoints(annotated, frame_data['keypoints'])
        
        # Draw objects (detections)
        objects = frame_data.get('objects', [])
        
        # Count by team
        team_a_count = sum(1 for obj in objects if obj.get('team_id') == 0)
        team_b_count = sum(1 for obj in objects if obj.get('team_id') == 1)
        
        for obj in objects:
            self._draw_object(annotated, obj)
        
        # Draw info overlay (skip if overlay_scale is zero)
        if self.overlay_scale > 0:
            self._draw_info(annotated, frame_number, len(objects), team_a_count, team_b_count)
        
        return annotated
    
    def _draw_object(self, frame: np.ndarray, obj: Dict):
        """
        Draw a single object: colored bbox + track-ID label above it.

        Label format (reference: soccergame draw_tracked_objects):
          Players:     "A#12"  /  "B#7"   (team colour background)
          Goalkeeper:  "GK#3"  (orange)
          Referee:     "RF#5"  (yellow)
          Ball:        "BA#1"  (red)
        When track_id == -1 (untracked), the label shows only the prefix.
        """
        x1, y1, x2, y2 = map(int, obj['bbox'])
        role     = obj.get('role', 'unknown')
        team_name = obj.get('team_name', '')
        track_id = obj.get('track_id', -1)

        # Color + short label prefix — derived from role string
        if role == 'player':
            if team_name == 'left':
                color, prefix = self.COLORS['team_a'],    'A'
            elif team_name == 'right':
                color, prefix = self.COLORS['team_b'],    'B'
            else:
                color, prefix = self.COLORS['unknown'],   'P'
        elif role == 'goalkeeper':
            color, prefix = self.COLORS['goalkeeper'], 'GK'
        elif role == 'referee':
            color, prefix = self.COLORS['referee'],    'RF'
        elif role == 'ball':
            color, prefix = self.COLORS['ball'],       'BA'
        else:
            color, prefix = self.COLORS['unknown'],    '?'

        # Bounding box (thicker for ball)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if role == 'ball' else 2)

        # Track-ID label (skip when overlay is fully hidden)
        if self.overlay_scale > 0:
            det_id = obj.get('detection_id', None)
            if track_id >= 0:
                label = (f"{prefix}#{track_id}({det_id})"
                         if det_id is not None else f"{prefix}#{track_id}")
            else:
                label = (f"{prefix}({det_id})"
                         if det_id is not None else prefix)
            self._draw_track_label(frame, label, x1, y1, color)

    def _draw_track_label(
        self,
        frame: np.ndarray,
        label: str,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
    ):
        """
        Draw a small filled label above a bounding box.

        Placement mirrors soccergame's draw_tracked_objects:
          - Filled rectangle in bbox colour
          - Black text on top for contrast
          - Falls back inside the box if bbox is at the top of the frame
        """
        s = self.overlay_scale
        h, w = frame.shape[:2]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.50, 0.80 * s)
        thickness  = max(1, int(2 * s))

        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Horizontal: align with bbox left edge, clamped so it stays in frame
        lx = int(np.clip(x1, 0, w - lw - 4))

        # Vertical: above bbox if there is room, otherwise just inside the top edge
        if y1 - lh - 5 >= 0:
            text_y  = y1 - 3
            rect_y1 = y1 - lh - 5
            rect_y2 = y1
        else:
            text_y  = y1 + lh + 2
            rect_y1 = y1
            rect_y2 = y1 + lh + 5

        # Filled background + black text
        cv2.rectangle(frame, (lx - 1, rect_y1), (lx + lw + 2, rect_y2), color, -1)
        cv2.putText(frame, label, (lx, text_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: List[float]):
        """
        Draw field line keypoints as circles only.
        
        Args:
            frame: Frame to draw on
            keypoints: List of [x1, y1, x2, y2, ..., x32, y32]
        """
        if not keypoints or len(keypoints) < 2:
            return
        
        # Draw keypoints as circles with white border (like track_soccer_complete.py)
        inner_color = self.COLORS['keypoint']  # Magenta fill
        border_color = (255, 255, 255)  # White border
        
        for i in range(0, len(keypoints), 2):
            if i + 1 < len(keypoints):
                x, y = int(keypoints[i]), int(keypoints[i + 1])
                
                # Skip if coordinates are zero (no detection)
                if x == 0 and y == 0:
                    continue
                
                # Draw keypoint with border (more visible)
                r_inner = max(3, int(5 * self.overlay_scale))
                r_outer = r_inner + 2
                cv2.circle(frame, (x, y), r_inner, inner_color, -1)
                cv2.circle(frame, (x, y), r_outer, border_color, 1)
    
    def _draw_score(
        self,
        frame: np.ndarray,
        team_a_count: int,
        team_b_count: int
    ):
        """Draw a compact score pill at the bottom centre."""
        h, w = frame.shape[:2]
        s = self.overlay_scale

        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = 0.45 * s
        font_thickness = max(1, int(1 * s))

        ta  = f"A:{team_a_count}"
        sep = "  "
        tb  = f"B:{team_b_count}"
        full = ta + sep + tb

        (fw, fh), baseline = cv2.getTextSize(full, font, font_scale, font_thickness)
        (ta_w, _), _       = cv2.getTextSize(ta,   font, font_scale, font_thickness)
        (sep_w, _), _      = cv2.getTextSize(sep,  font, font_scale, font_thickness)

        pad = max(3, int(5 * s))
        x = (w - fw) // 2
        y = h - pad - baseline

        # Background pill
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x - pad, y - fh - pad),
                      (x + fw + pad, y + baseline + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, ta,  (x,                y), font, font_scale, self.COLORS['team_a'], font_thickness)
        cv2.putText(frame, sep, (x + ta_w,         y), font, font_scale, (200, 200, 200),        font_thickness)
        cv2.putText(frame, tb,  (x + ta_w + sep_w, y), font, font_scale, self.COLORS['team_b'], font_thickness)

    def _draw_info(
        self,
        frame: np.ndarray,
        frame_number: int,
        total_objects: int,
        team_a_count: int,
        team_b_count: int
    ):
        """Draw a single-line info strip at the bottom-left + score pill at bottom-centre."""
        h, w = frame.shape[:2]
        s    = self.overlay_scale

        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = 0.40 * s
        font_thickness = max(1, int(1 * s))

        # Single-line: "F:473  19obj"
        info = f"F:{frame_number}  {total_objects}obj"
        (iw, ih), baseline = cv2.getTextSize(info, font, font_scale, font_thickness)

        pad = max(3, int(4 * s))
        x, y = 6, h - pad - baseline

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x - pad, y - ih - pad),
                      (x + iw + pad, y + baseline + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, info, (x, y), font, font_scale, (220, 220, 220), font_thickness)

        # Score pill at bottom centre
        self._draw_score(frame, team_a_count, team_b_count)

        # Legend at bottom-right
        self._draw_legend(frame)
    
    def _draw_legend(self, frame: np.ndarray):
        """Draw compact color legend at the bottom-right corner."""
        h, w  = frame.shape[:2]
        s     = self.overlay_scale

        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = 0.35 * s
        font_thickness = max(1, int(1 * s))

        legend_items = [
            ("A",  self.COLORS['team_a']),
            ("B",  self.COLORS['team_b']),
            ("GK", self.COLORS['goalkeeper']),
            ("RF", self.COLORS['referee']),
            ("BA", self.COLORS['ball']),
        ]

        box_sz  = max(6, int(10 * s))
        gap     = max(2, int(3 * s))
        item_w  = box_sz + gap + max(14, int(22 * s))  # box + space + text
        total_w = item_w * len(legend_items) + gap * (len(legend_items) - 1)
        pad     = max(3, int(4 * s))

        x0 = w - total_w - pad * 2 - 6
        y0 = h - box_sz - pad * 2 - 6

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x0 - pad, y0 - pad),
                      (w - 6, h - 6),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (label, color) in enumerate(legend_items):
            x = x0 + i * (item_w + gap)
            # Color swatch
            cv2.rectangle(frame,
                          (x, y0),
                          (x + box_sz, y0 + box_sz),
                          color, -1)
            # Short label
            cv2.putText(frame, label,
                        (x + box_sz + 2, y0 + box_sz),
                        font, font_scale, (220, 220, 220), font_thickness)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize soccer game analysis results'
    )
    parser.add_argument(
        '--video',
        default='D:/Data/9.mp4',
        help='Input video file'
    )
    parser.add_argument(
        '--json',
        default='D:/work/44/soccergame_turbo/output/output.json',
        help='Analysis results JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        default='D:/work/44/soccergame_turbo/output/out.mp4',
        help='Output video file (optional)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display video (only save)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        help='Output video FPS (default: same as input)'
    )
    parser.add_argument(
        '--overlay-scale',
        type=float,
        default=0.6,
        help='Scale for on-screen overlays, placed at the bottom of the frame. '
             '0.6=default, 1.0=larger, 0.0=hide all overlays.',
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return 1
    
    print("="*70)
    print("SOCCER ANALYSIS VISUALIZATION")
    print("="*70)
    print()
    print(f"Video: {args.video}")
    print(f"Results: {args.json}")
    if args.output:
        print(f"Output: {args.output}")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print()
    
    # Create visualizer
    visualizer = SoccerResultVisualizer(
        video_path=args.video,
        json_path=args.json,
        output_path=args.output,
        display=not args.no_display,
        fps=args.fps,
        overlay_scale=args.overlay_scale,
    )
    
    # Run visualization
    try:
        visualizer.visualize()
        print()
        print("="*70)
        print("✓ VISUALIZATION COMPLETE")
        print("="*70)
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

