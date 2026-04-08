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
    Computes frame-to-frame ball velocity vector (vx, vy) in pixels/frame.

    Camera motion compensation:
        Each frame's affine transform (previous → current) is applied to the
        previous ball position before computing the delta, removing residual
        camera micro-motion from the velocity signal.

    Ball track_id continuity:
        If the ball's track_id changes between frames, the tracker re-acquired
        the ball (or confused it with another object).  The position jump is
        meaningless as velocity, so the chain is reset on an ID change.

    Resets when:
        - Ball track_id changes (tracker re-acquisition)
        - Ball absent longer than max_chain_frames
        - Frame gap exceeds max_chain_frames (limits accumulated transform error)
    """

    # Maximum plausible ball speed in pixels/frame at 25 fps.
    # Real football max ~35 m/s; at typical camera scale ≈ 40 px/m → ~56 px/frame.
    # Values above this threshold almost certainly indicate a tracker glitch
    # (ball ID reuse, mis-detected object), not genuine ball motion.
    MAX_PLAUSIBLE_SPEED: float = 55.0

    def __init__(self, max_chain_frames: int = 25):
        self.max_chain_frames = max_chain_frames
        self._prev_center: Optional[Vector2D] = None
        self._prev_frame_id: Optional[int] = None
        self._prev_track_id: Optional[int] = None

    def update(self, frame: Frame) -> Optional[Vector2D]:
        """
        Returns velocity vector (vx, vy) in pixels/frame, or None if unavailable.
        Must be called once per frame in order.
        """
        if frame.ball is None:
            # Hold last known position across brief absences so post-kick velocity
            # is preserved (ball momentarily hidden after being struck).
            # Do NOT reset _prev_track_id here — if the same ball ID reappears
            # we resume the chain; a different ID on return will reset below.
            return None

        current_center = frame.ball.center
        current_track_id = frame.ball.track_id

        # Track_id changed → tracker re-acquired or mis-identified the ball.
        # The position jump is not real motion; reset the chain.
        if self._prev_track_id is not None and current_track_id != self._prev_track_id:
            self._prev_center = current_center
            self._prev_frame_id = frame.frame_id
            self._prev_track_id = current_track_id
            return None

        if self._prev_center is None:
            self._prev_center = current_center
            self._prev_frame_id = frame.frame_id
            self._prev_track_id = current_track_id
            return None

        gap = frame.frame_id - self._prev_frame_id  # type: ignore[operator]
        if gap <= 0 or gap > self.max_chain_frames:
            self._prev_center = current_center
            self._prev_frame_id = frame.frame_id
            self._prev_track_id = current_track_id
            return None

        # Project previous ball center into current frame's coordinate space
        prev_compensated = apply_affine(self._prev_center, frame.transform)
        vx = (current_center[0] - prev_compensated[0]) / gap
        vy = (current_center[1] - prev_compensated[1]) / gap

        self._prev_center = current_center
        self._prev_frame_id = frame.frame_id
        self._prev_track_id = current_track_id
        return (vx, vy)

    @staticmethod
    def speed(velocity: Optional[Vector2D]) -> float:
        """Scalar speed magnitude from a velocity vector."""
        if velocity is None:
            return 0.0
        return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
