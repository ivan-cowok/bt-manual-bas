import json
import sys

from src.config import Config
from src.parser import parse_clip
from src.pipeline import Pipeline
from src.postprocessing import relabel_consecutive_interceptions


def run_inference(source: str | dict, production: bool = False) -> dict:
    config = Config(is_production=production)
    metadata, frames = parse_clip(source)

    video_id = metadata.get("video_id", "unknown")
    mode_tag = "PROD" if production else "DEV"
    print(f"[{mode_tag}] Processing '{video_id}' — {len(frames)} frames at {metadata.get('fps', '?')} fps")

    EMITTED_TYPES = {"pass", "pass_received", "interception"}

    pipeline = Pipeline(config)
    all_events = pipeline.run(frames)
    emitted = [e for e in all_events if e.event_type in EMITTED_TYPES]

    if production:
        relabel_consecutive_interceptions(emitted)

    print(f"Detected {len(emitted)} events ({len(all_events) - len(emitted)} internal-only suppressed)")
    for e in emitted:
        print(
            f"  [{e.event_type:>15}]  frame={e.frame_id:>6}  "
            f"t={e.timestamp_ms:>8}ms  team={e.team}  conf={e.confidence:.2f}"
        )

    return {
        "predictions": [
            {"frame": e.frame_id, "action": e.event_type}
            for e in emitted
        ],
    }


def main(input_path: str, output_path: str | None = None, production: bool = False) -> None:
    output = run_inference(input_path, production=production)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {output_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_json> [output_json] [--prod]")
        sys.exit(1)

    args = sys.argv[1:]
    prod_mode = "--prod" in args
    positional = [a for a in args if not a.startswith("--")]

    main(
        positional[0],
        positional[1] if len(positional) > 1 else None,
        production=prod_mode,
    )
