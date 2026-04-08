import json
import sys

from src.config import Config
from src.parser import parse_clip
from src.pipeline import Pipeline


def main(input_path: str, output_path: str | None = None) -> None:
    config = Config()
    metadata, frames = parse_clip(input_path)

    video_id = metadata.get("video_id", "unknown")
    print(f"Processing '{video_id}' — {len(frames)} frames at {metadata.get('fps', '?')} fps")

    pipeline = Pipeline(config)
    events = pipeline.run(frames)

    print(f"Detected {len(events)} events")  
    
    EMITTED_TYPES = {"pass", "pass_received"}

    output = {
        "predictions": [
            {
                "frame": e.frame_id,
                "action": e.event_type,
            }
            for e in events
            if e.event_type in EMITTED_TYPES
        ],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {output_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_json> [output_json]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
