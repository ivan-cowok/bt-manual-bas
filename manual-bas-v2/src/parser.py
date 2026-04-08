import collections
import json
import math
import statistics
from typing import Dict, List, Tuple, Union

from .models import BallDetection, Frame, PlayerDetection


def parse_clip(source: Union[str, dict]) -> Tuple[Dict, List[Frame]]:
    """
    Parse a clip JSON file (or dict) into metadata and a list of Frame objects.

    Format expectations:
    - Top-level: challenge_id, fps, video_resolution, frames
    - Per frame: frame_number, transform, objects (mixed roles)
    - Object roles: "ball", "player", "goalkeeper", "referee"
    - team_name: "left", "right", or "" (empty = unknown / GK)

    GK team inference:
        Model outputs team_name="" for goalkeepers.  We infer team via a
        neighbor vote: for each GK track_id, count which team's field players
        appear most often among the 3 nearest players across all frames.
        Falls back to median-x position if no field players are ever nearby.
        If the model starts providing team for GKs, that value takes precedence.
    """
    if isinstance(source, str):
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    fps: float = data.get("fps", 25)
    resolution = data.get("video_resolution", [1920, 1080])
    frame_width: float = resolution[0]

    # --- First pass: infer GK team via neighbor vote ---
    gk_x_samples: Dict[int, List[float]] = {}
    gk_neighbor_votes: Dict[int, collections.Counter] = {}

    for fd in data["frames"]:
        objs = fd.get("objects", [])
        gks = [o for o in objs if o["role"] == "goalkeeper" and o["track_id"] != -1]
        field_players = [
            o for o in objs
            if o["role"] == "player" and o.get("team_name") in ("left", "right")
        ]

        for obj in gks:
            tid = obj["track_id"]

            # If the model already supplies a valid team for this GK, skip inference
            if obj.get("team_name") in ("left", "right"):
                continue

            bbox = obj["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            gk_x_samples.setdefault(tid, []).append(cx)

            if field_players:
                dists = sorted(
                    (
                        math.sqrt(
                            (cx - (p["bbox"][0] + p["bbox"][2]) / 2.0) ** 2
                            + (cy - (p["bbox"][1] + p["bbox"][3]) / 2.0) ** 2
                        ),
                        p["team_name"],
                    )
                    for p in field_players
                )
                nearest_teams = [t for _, t in dists[:3]]
                gk_neighbor_votes.setdefault(tid, collections.Counter()).update(nearest_teams)

    gk_inferred_team: Dict[int, str] = {}
    for tid, x_samples in gk_x_samples.items():
        votes = gk_neighbor_votes.get(tid)
        if votes and sum(votes.values()) > 0:
            gk_inferred_team[tid] = votes.most_common(1)[0][0]
        else:
            gk_inferred_team[tid] = (
                "left" if statistics.median(x_samples) < frame_width / 2 else "right"
            )

    # --- Second pass: build Frame objects ---
    frames: List[Frame] = []

    for fd in data["frames"]:
        frame_number: int = fd["frame_number"]
        timestamp_ms: int = int(frame_number * 1000 / fps)
        transform: List[List[float]] = fd["transform"]

        ball: BallDetection | None = None
        players: List[PlayerDetection] = []

        for obj in fd.get("objects", []):
            role: str = obj["role"]
            bbox = tuple(obj["bbox"])
            track_id: int = obj["track_id"]
            det_conf: float = obj["detection_confidence"]

            if role == "ball":
                # track_id == -1: tracker lost ball, unconfirmed detection.
                # Positional jumps from untracked balls poison velocity; treat as absent.
                if track_id == -1:
                    continue
                ball = BallDetection(
                    track_id=track_id,
                    bbox=bbox,
                    detection_confidence=det_conf,
                )

            elif role == "goalkeeper":
                raw_team = obj.get("team_name", "")
                if raw_team in ("left", "right"):
                    # Model provided team — trust it directly
                    team = raw_team
                elif track_id != -1 and track_id in gk_inferred_team:
                    team = gk_inferred_team[track_id]
                else:
                    # Last-resort: per-frame x-position
                    center_x = (bbox[0] + bbox[2]) / 2
                    team = "left" if center_x < frame_width / 2 else "right"

                players.append(
                    PlayerDetection(
                        track_id=track_id,
                        team=team,
                        role="goalkeeper",
                        bbox=bbox,
                        detection_confidence=det_conf,
                        team_confidence=None,
                    )
                )

            elif role == "player":
                raw_team = obj.get("team_name", "")
                team = raw_team if raw_team in ("left", "right") else "unknown"
                team_conf = obj.get("team_confidence") if team != "unknown" else None
                players.append(
                    PlayerDetection(
                        track_id=track_id,
                        team=team,
                        role="player",
                        bbox=bbox,
                        detection_confidence=det_conf,
                        team_confidence=team_conf,
                    )
                )

            # role == "referee" → ignored

        frames.append(
            Frame(
                frame_id=frame_number,
                timestamp_ms=timestamp_ms,
                transform=transform,
                ball=ball,
                players=players,
            )
        )

    metadata = {k: v for k, v in data.items() if k != "frames"}
    metadata["video_id"] = metadata.pop("challenge_id", metadata.get("video_id", "unknown"))
    return metadata, frames
