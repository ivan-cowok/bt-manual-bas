from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]   # x1, y1, x2, y2
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
    team: str            # "left", "right", or "unknown"
    role: str            # "player" or "goalkeeper"
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
    transform: List[List[float]]   # 2×3 affine: previous frame → current frame
    ball: Optional[BallDetection] = None
    players: List[PlayerDetection] = field(default_factory=list)


class BallState(Enum):
    POSSESSED  = auto()   # one team has uncontested, committed possession
    CONTESTED  = auto()   # players from both teams within threshold simultaneously
    IN_FLIGHT  = auto()   # ball moving above speed threshold, no current possessor
    DEAD       = auto()   # ball stationary + no possessor for N frames


@dataclass
class FramePossession:
    """Per-frame possession signal produced by PossessionTracker."""
    team: str             # "left", "right", or "loose"
    frame_id: int
    timestamp_ms: int
    ball_detected: bool = True

    # Primary (closest) candidate
    player_track_id: Optional[int] = None
    player_bbox: Optional[BBox] = None
    player_role: Optional[str] = None

    # True when committed team's player is no longer nearest (stale carry-over)
    is_stale: bool = False

    # Second candidate — populated when top-2 players are from different teams
    second_team: Optional[str] = None
    second_player_track_id: Optional[int] = None

    # True when top-2 players are from different teams AND both within threshold
    is_contested: bool = False


@dataclass
class Event:
    event_type: str    # "pass", "pass_received", "interception", "recovery"
    frame_id: int
    timestamp_ms: int
    team: str
    confidence: float
    emittable: bool = True   # False = internal suppressor only (e.g. shot)
