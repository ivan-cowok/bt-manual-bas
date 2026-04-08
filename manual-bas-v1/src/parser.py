import collections
import json
import math
import statistics
from typing import Dict, List, Tuple, Union

from .models import BallDetection, Frame, PlayerDetection


def parse_clip(source: Union[str, dict]) -> Tuple[Dict, List[Frame]]:
    """
    Parse a clip JSON file (or dict) into metadata and a list of Frame objects.

    Handles the real model output format:
    - Top-level: challenge_id, fps, video_resolution, frames
    - Per frame: frame_number, transform, objects (mixed roles)
    - Object roles: "ball", "player", "goalkeeper", "referee"
    - team_name: "left", "right", or "" (empty = unknown)
    - Goalkeeper team inferred via neighbor vote: for each GK, count which team's
      field players appear most often among the 3 nearest players across all frames.
      A GK is surrounded by their own team's defenders, making this more reliable
      than position alone (which fails after halftime side-swaps).
      Falls back to median-x position if no field players are ever nearby.

    Returns:
        metadata: top-level fields excluding "frames"
        frames:   ordered list of Frame objects
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
    # For each GK track_id, collect which team's field players are most often
    # among the 3 closest players. A GK is flanked by their own defenders, so
    # the majority neighbor team reliably identifies the GK's team even when
    # teams have swapped sides (second half) and position-based inference fails.
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

    gk_team: Dict[int, str] = {}
    for tid, x_samples in gk_x_samples.items():
        votes = gk_neighbor_votes.get(tid)
        if votes and sum(votes.values()) > 0:
            gk_team[tid] = votes.most_common(1)[0][0]
        else:
            # Fallback: position-based (no field players were ever nearby)
            gk_team[tid] = "left" if statistics.median(x_samples) < frame_width / 2 else "right"

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
                # track_id == -1 means the tracker lost the ball and produced an
                # unconfirmed detection.  These tend to appear at spurious positions
                # (large positional jumps, wrong side of the pitch) and inject noise
                # into velocity and possession calculations.  Treat them as absent.
                if track_id == -1:
                    continue
                ball = BallDetection(
                    track_id=track_id,
                    bbox=bbox,
                    detection_confidence=det_conf,
                )

            elif role == "goalkeeper":
                if track_id != -1 and track_id in gk_team:
                    inferred_team = gk_team[track_id]
                else:
                    # Fallback for untracked detections: use per-frame x-position
                    center_x = (bbox[0] + bbox[2]) / 2
                    inferred_team = "left" if center_x < frame_width / 2 else "right"
                players.append(
                    PlayerDetection(
                        track_id=track_id,
                        team=inferred_team,
                        role="goalkeeper",
                        bbox=bbox,
                        detection_confidence=det_conf,
                        team_confidence=None,  # spatially inferred, no model score
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

    # Normalise metadata keys
    metadata = {k: v for k, v in data.items() if k != "frames"}
    metadata["video_id"] = metadata.pop("challenge_id", metadata.get("video_id", "unknown"))
    return metadata, frames
