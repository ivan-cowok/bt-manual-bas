import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _matrix_to_str(matrix: np.ndarray) -> str:
    return (
        f"[[{matrix[0, 0]: .8f}, {matrix[0, 1]: .8f}, {matrix[0, 2]: .8f}], "
        f"[{matrix[1, 0]: .8f}, {matrix[1, 1]: .8f}, {matrix[1, 2]: .8f}]]"
    )


def _apply_to_point(matrix_2x3: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (
        float(matrix_2x3[0, 0] * x + matrix_2x3[0, 1] * y + matrix_2x3[0, 2]),
        float(matrix_2x3[1, 0] * x + matrix_2x3[1, 1] * y + matrix_2x3[1, 2]),
    )


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
    cumulative = np.eye(3, dtype=np.float64)

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

        matrix_3x3 = np.array(
            [
                [float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])],
                [float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cumulative = matrix_3x3 @ cumulative

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

        x0, y0 = _apply_to_point(matrix, 0.0, 0.0)
        x1, y1 = _apply_to_point(matrix, float(width - 1), 0.0)
        x2, y2 = _apply_to_point(matrix, 0.0, float(height - 1))
        x3, y3 = _apply_to_point(matrix, float(width - 1), float(height - 1))

        cumulative_2x3 = cumulative[:2, :]
        print(
            f"Applied transform {applied_count}: frame {frame_no - 1} -> {frame_no}\n"
            f"  matrix(2x3): {_matrix_to_str(matrix)}\n"
            f"  mapped corners: "
            f"(0,0)->({x0:.3f},{y0:.3f}), "
            f"({width - 1},0)->({x1:.3f},{y1:.3f}), "
            f"(0,{height - 1})->({x2:.3f},{y2:.3f}), "
            f"({width - 1},{height - 1})->({x3:.3f},{y3:.3f})\n"
            f"  cumulative(2x3 from start frame): {_matrix_to_str(cumulative_2x3)}"
        )

    if not cv2.imwrite(str(generated_end_path), current):
        raise SystemExit(f"Failed to write generated image: {generated_end_path}")

    print(f"Applied transforms total: {applied_count}")
    print(f"Generated final image: {generated_end_path}")


if __name__ == "__main__":
    main()
