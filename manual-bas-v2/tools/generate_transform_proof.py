import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply chained affine transforms from output.json to frames."
    )
    parser.add_argument("--data-dir", required=True, help="Path like data/6")
    parser.add_argument("--start-frame", type=int, required=True, help="Start frame")
    parser.add_argument("--end-frame", type=int, required=True, help="End frame")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frames_dir = data_dir / "frames"
    output_json = data_dir / "output.json"

    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    transform_by_frame = {
        int(frame["frame_number"]): frame.get("transform") for frame in data["frames"]
    }

    start_frame = args.start_frame
    end_frame = args.end_frame
    if end_frame <= start_frame:
        raise SystemExit("end-frame must be greater than start-frame")

    start_img_path = frames_dir / f"{start_frame}.png"
    generated_end_path = frames_dir / f"{end_frame}_temp_from_{start_frame}.png"

    image = cv2.imread(str(start_img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read start frame image: {start_img_path}")

    height, width = image.shape[:2]
    current = image.copy()

    applied_count = 0
    for frame_no in range(start_frame + 1, end_frame + 1):
        transform = transform_by_frame.get(frame_no)
        if transform is None:
            raise SystemExit(f"Missing transform for frame {frame_no}")

        matrix = np.array(transform, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise SystemExit(
                f"Invalid transform shape for frame {frame_no}: {matrix.shape}"
            )

        # transform maps (frame_no - 1) -> frame_no
        current = cv2.warpAffine(
            current,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        applied_count += 1
        print(
            f"Applied transform {applied_count}: frame {frame_no - 1} -> {frame_no}, "
            f"tx={matrix[0, 2]:.4f}, ty={matrix[1, 2]:.4f}"
        )

    if not cv2.imwrite(str(generated_end_path), current):
        raise SystemExit(f"Failed to write generated image: {generated_end_path}")

    print(f"Applied transforms total: {applied_count}")
    print(f"Generated final image: {generated_end_path}")


if __name__ == "__main__":
    main()
