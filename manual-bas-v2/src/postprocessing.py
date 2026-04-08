from typing import List

from .config import Config
from .models import Event

# NMS window per event type — separate windows prevent pass/pass_received pairs
# from suppressing each other while still deduplicating same-type boundary noise.
_NMS_WINDOWS = {
    "pass":           "nms_window_pass",
    "pass_received":  "nms_window_pass_received",
    "interception":   "nms_window_interception",
    "recovery":       "nms_window_recovery",
}


def temporal_nms(events: List[Event], config: Config) -> List[Event]:
    """
    Suppress duplicate events of the same type within a per-type frame window.

    Only emittable events participate; internal suppressors are dropped here.
    For each event type, only the first occurrence within any NMS window is kept.
    """
    emittable = [e for e in events if e.emittable]
    if not emittable:
        return []

    sorted_events = sorted(emittable, key=lambda e: e.frame_id)
    result: List[Event] = []
    last_kept: dict[str, int] = {}

    for event in sorted_events:
        window_attr = _NMS_WINDOWS.get(event.event_type, "nms_window_pass")
        window = getattr(config, window_attr, 15)
        last_frame = last_kept.get(event.event_type, -(window + 1))
        if event.frame_id - last_frame >= window:
            result.append(event)
            last_kept[event.event_type] = event.frame_id

    return result
