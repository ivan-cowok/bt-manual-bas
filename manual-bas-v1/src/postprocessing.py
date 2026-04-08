from typing import List

from .config import Config
from .models import Event


def temporal_nms(events: List[Event], config: Config) -> List[Event]:
    """
    Suppress duplicate events of the same type within a sliding frame window.

    For each event type independently, only the first occurrence within any
    nms_window_frames window is kept. This prevents multiple detections of the
    same physical event caused by frame-level boundary effects.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.frame_id)
    result: List[Event] = []
    last_kept: dict[str, int] = {}  # event_type → last kept frame_id

    for event in sorted_events:
        last_frame = last_kept.get(event.event_type, -(config.nms_window_frames + 1))
        if event.frame_id - last_frame >= config.nms_window_frames:
            result.append(event)
            last_kept[event.event_type] = event.frame_id

    return result
