import sys
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Query

ROOT_DIR = Path(__file__).resolve().parent
DETECTION_DIR = ROOT_DIR / "detection"
POSTPROCESS_DIR = ROOT_DIR / "manual-bas-v2"

if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))
if str(POSTPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_DIR))

from analyze_soccer import SoccerAnalysisPipeline  # noqa: E402
import main as postprocess_main  # noqa: E402
from download_video import delete_video, download_video  # noqa: E402

app = FastAPI(title="Soccer Inference API", version="0.1.0")

_PIPELINE_LOCK = Lock()
_PIPELINE: SoccerAnalysisPipeline | None = None


def _get_pipeline() -> SoccerAnalysisPipeline:
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = SoccerAnalysisPipeline(
                team_model_path=str(DETECTION_DIR / "models" / "team" / "team.onnx"),
                reid_model_path=str(DETECTION_DIR / "models" / "team" / "osnet.onnx"),
                detection_model_path=str(DETECTION_DIR / "models" / "detection" / "objectdetect.onnx"),
                keypoint_model_path=str(DETECTION_DIR / "models" / "keypoint" / "keypoint.onnx"),
                line_model_path=str(DETECTION_DIR / "models" / "keypoint" / "line.onnx"),
                # Keep behavior aligned with analyze_soccer.py CLI defaults.
                use_gpu=True,
                detection_batch_size=8,
                encoder_batch_size=64,
                keypoint_batch_size=1,
                encoder_type="reid",
                outlier_sigma=-1,
                overlay_scale=0.1,
            )
    return _PIPELINE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/infer")
def infer(video_url: str = Query(..., description="Direct downloadable video URL")) -> dict:
    started_at = time.perf_counter()
    print(f"[infer] start video_url={video_url}", flush=True)
    video_path: Path | None = None
    try:
        try:
            print("[infer] downloading video...", flush=True)
            video_path = download_video(video_url)
            print(f"[infer] download complete path={video_path}", flush=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {exc}") from exc

        pipeline = _get_pipeline()
        try:
            print("[infer] detection started", flush=True)
            detection_output = pipeline.process(
                source_path=str(video_path),
                output_json=None,
                visualize=False,
                display=False,
                # Keep behavior aligned with analyze_soccer.py CLI defaults.
                skip_keypoints=True,
                skip_tracking=False,
                track_cmc=True,
                track_cmc_method="sparseOptFlow",
                parallel_gmc=True,
            )
            print("[infer] detection finished", flush=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Detection pipeline failed: {exc}") from exc

        try:
            print("[infer] post-processing started", flush=True)
            result_output = postprocess_main.run_inference(detection_output, production=True)
            print("[infer] post-processing finished", flush=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Post-processing failed: {exc}") from exc
    finally:
        if video_path is not None:
            delete_video(video_path)
            print(f"[infer] deleted temp video path={video_path}", flush=True)

    elapsed_sec = round(time.perf_counter() - started_at, 3)
    print(f"[infer] done elapsed={elapsed_sec}s", flush=True)
    return {
        "result": result_output,
        "meta": {
            "video_url": video_url,
            "frames_processed": len(detection_output.get("frames", [])),
            "elapsed_seconds": elapsed_sec,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
