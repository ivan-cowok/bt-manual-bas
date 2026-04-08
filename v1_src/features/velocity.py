import math
from typing import List, Optional

from ..models import Frame, Vector2D


def apply_affine(point: Vector2D, transform: List[List[float]]) -> Vector2D:
    """Apply a 2×3 affine transform matrix to a 2D point."""
    x, y = point
    m = transform
    return (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )


class BallVelocityCalculator:
    """
    Computes frame-to-frame ball velocity in pixels/frame.

    Uses the frame's affine transform (previous → current) to compensate for
    any residual camera micro-motion before computing the delta.

    Resets when:
    - Ball is absent (undetected)
    - Frame gap exceeds max_chain_frames (avoids accumulated transform error)
    """

    def __init__(self, max_chain_frames: int = 25):
        self.max_chain_frames = max_chain_frames
        self._prev_center: Optional[Vector2D] = None
        self._prev_frame_id: Optional[int] = None

    def update(self, frame: Frame) -> Optional[Vector2D]:
        """
        Returns velocity vector (vx, vy) in pixels/frame, or None if unavailable.
        Must be called once per frame in order.

        Ball absence handling: the last known ball position is retained across
        absent frames so that the kick velocity is preserved when the ball
        disappears momentarily after being struck (e.g. goalkeeper distribution).
        The gap is bounded by max_chain_frames to avoid large accumulated
        transform errors.
        """
        if frame.ball is None:
            # Do NOT reset — keep the last known position so that velocity can
            # be computed correctly across brief absences (e.g. post-kick frames).
            return None

        current_center = frame.ball.center

        if self._prev_center is None:
            self._prev_center = current_center
            self._prev_frame_id = frame.frame_id
            return None

        gap = frame.frame_id - self._prev_frame_id
        if gap <= 0 or gap > self.max_chain_frames:
            self._prev_center = current_center
            self._prev_frame_id = frame.frame_id
            return None

        # Project previous ball center into current frame's coordinate space
        prev_compensated = apply_affine(self._prev_center, frame.transform)
        vx = (current_center[0] - prev_compensated[0]) / gap
        vy = (current_center[1] - prev_compensated[1]) / gap

        self._prev_center = current_center
        self._prev_frame_id = frame.frame_id
        return (vx, vy)

    def _reset(self) -> None:
        self._prev_center = None
        self._prev_frame_id = None

    @staticmethod
    def speed(velocity: Optional[Vector2D]) -> float:
        """Compute scalar speed magnitude from a velocity vector."""
        if velocity is None:
            return 0.0
        return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
