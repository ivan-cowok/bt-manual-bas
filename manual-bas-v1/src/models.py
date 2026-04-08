from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Vector2D = Tuple[float, float]


@dataclass
class BallDetection:
    track_id: int
    bbox: BBox
    detection_confidence: float

    @property
    def center(self) -> Vector2D:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


@dataclass
class PlayerDetection:
    track_id: int
    team: str  # "A", "B", or "unknown"
    role: str  # "player" or "goalkeeper"
    bbox: BBox
    detection_confidence: float
    team_confidence: Optional[float] = None

    @property
    def center(self) -> Vector2D:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def feet(self) -> Vector2D:
        """Bottom-centre of bounding box — the physical location of the player's feet."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3],
        )

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Frame:
    frame_id: int
    timestamp_ms: int
    transform: List[List[float]]  # 2×3 affine matrix: previous frame → current frame
    ball: Optional[BallDetection] = None
    players: List[PlayerDetection] = field(default_factory=list)


@dataclass
class FramePossession:
    """Per-frame possession signal produced by PossessionTracker."""
    team: str  # "left", "right", or "loose"
    frame_id: int
    timestamp_ms: int
    ball_detected: bool = True   # False when ball was absent this frame
    player_track_id: Optional[int] = None
    player_bbox: Optional[BBox] = None
    player_role: Optional[str] = None  # "player" or "goalkeeper"
    # True when the team is returned as a stale committed state (the committed
    # team's player was NOT nearest this frame; a different team's player was
    # accumulating but not yet committed).  Consumers that only act on fresh
    # possession (e.g. IDLE → POSSESSED transition) should ignore stale signals.
    is_stale: bool = False


@dataclass
class Event:
    event_type: str  # "pass", "pass_received", "interception"
    frame_id: int
    timestamp_ms: int
    team: str
    confidence: float
