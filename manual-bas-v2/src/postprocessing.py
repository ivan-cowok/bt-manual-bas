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


def shift_short_pass_frames(events: List[Event], config: Config) -> List[Event]:
    """Shift pass events backward by config.short_pass_frame_backshift frames when
    the next event (any type) is within config.short_pass_max_gap_frames.

    Quick exchanges have a consistent ~2-frame detection latency because the
    state machine needs a few in-flight frames before firing the pass event.
    For short passes this latency is a noticeable fraction of the flight time,
    so shifting back improves alignment with ground-truth annotations.
    """
    for i, event in enumerate(events):
        if event.event_type != "pass":
            continue
        if i + 1 >= len(events):
            continue
        gap = events[i + 1].frame_id - event.frame_id
        if gap <= config.short_pass_max_gap_frames:
            event.frame_id = max(0, event.frame_id - config.short_pass_frame_backshift)
    return events


def relabel_consecutive_interceptions(events: List[Event]) -> List[Event]:
    """Relabel the second of two back-to-back interceptions as a recovery.

    Two consecutive interception events (nothing else in between) almost never
    represent two genuine steals-from-a-pass.  The more common reality is that
    the ball was loose after the first interception and a player picked it up —
    which is a recovery.  Relabeling reduces the false-positive interception
    penalty (2.8) to the lower recovery penalty (1.5) and may gain recovery
    match points when ground-truth has it labelled that way.
    """
    for i in range(len(events) - 1):
        if events[i].event_type == "interception" and events[i + 1].event_type == "interception":
            events[i + 1].event_type = "recovery"
    return events


def remove_pass_received_before_interception(events: List[Event], config: Config) -> List[Event]:
    """Remove a pass_received that is immediately followed by an interception of the
    opposing team within config.pass_received_before_interception_max_gap frames.

    This targets the near-miss false positive: a player briefly approached the ball
    (triggering an eager pass_received) but the true contact was the opponent's
    interception a few frames later.  We require different teams to avoid removing
    a legitimate receive that is followed by a same-team interception/recovery.
    """
    to_remove: set[int] = set()
    for i, event in enumerate(events):
        if event.event_type != "pass_received":
            continue
        if i + 1 >= len(events):
            continue
        nxt = events[i + 1]
        if nxt.event_type != "interception":
            continue
        gap = nxt.frame_id - event.frame_id
        if 0 < gap <= config.pass_received_before_interception_max_gap and event.team != nxt.team:
            to_remove.add(i)
    return [e for i, e in enumerate(events) if i not in to_remove]
