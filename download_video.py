import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import cv2

_stdlib_preped = False


def _prepend_stdlib_to_path() -> None:
    global _stdlib_preped
    if _stdlib_preped:
        return
    p = sysconfig.get_path("stdlib")
    if p:
        sys.path.insert(0, p)
    _stdlib_preped = True

_WGET_LINUX_UA = "Mozilla/5.0 (X11; Linux x86_64)"
_SCOREDATA_REFERER = "https://scoredata.me"


def _request_headers(url: str) -> dict[str, str]:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    scoredata = host == "scoredata.me" or host.endswith(".scoredata.me")
    if scoredata:
        h = {
            "User-Agent": _WGET_LINUX_UA,
            "Referer": _SCOREDATA_REFERER,
        }
    else:
        h = {
            "User-Agent": _WGET_LINUX_UA,
            "Accept": "*/*",
        }
        if p.scheme and p.netloc:
            h["Referer"] = f"{p.scheme}://{p.netloc}/"
    c = os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_COOKIE", "").strip()
    if c:
        h["Cookie"] = c
    return h


def _download_system_curl(url: str, headers: dict[str, str], path: Path) -> None:
    exe = shutil.which("curl")
    if not exe:
        raise RuntimeError("curl executable not found on PATH")
    max_time = os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_MAX_TIME_S", "600")
    cmd = [
        exe,
        "-sSL",
        "--fail",
        "--connect-timeout",
        "30",
        "--max-time",
        max_time,
        "-o",
        str(path),
    ]
    if os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_IPV4", "").strip() in ("1", "true", "yes"):
        cmd.append("-4")
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    cmd.append(url)
    subprocess.run(cmd, check=True)


def _download_system_wget(url: str, headers: dict[str, str], path: Path) -> None:
    exe = shutil.which("wget")
    if not exe:
        raise RuntimeError("wget executable not found on PATH")
    max_time = os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_MAX_TIME_S", "600")
    cmd = [
        exe,
        "-q",
        "-O",
        str(path),
        f"--timeout={max_time}",
        "--tries=3",
    ]
    if os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_IPV4", "").strip() in ("1", "true", "yes"):
        cmd.append("-4")
    for k, v in headers.items():
        cmd.append(f"--header={k}: {v}")
    cmd.append(url)
    subprocess.run(cmd, check=True)


def _download_curl_cffi(url: str, headers: dict[str, str], path: Path) -> None:
    _prepend_stdlib_to_path()
    from curl_cffi.requests import Session

    timeout = float(os.environ.get("SCOREVISION_VIDEO_DOWNLOAD_TIMEOUT_S", "120"))
    with Session() as session:
        with session.stream(
            "GET",
            url,
            headers=headers,
            impersonate="chrome131",
            timeout=timeout,
            allow_redirects=True,
        ) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content():
                    f.write(chunk)


def download_video(url: str) -> Path:
    headers = _request_headers(url)
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()
    path = Path(temp_file.name)
    last_err: BaseException | None = None
    steps = []
    if shutil.which("wget"):
        steps.append(_download_system_wget)
    steps.append(_download_curl_cffi)
    if shutil.which("curl"):
        steps.append(_download_system_curl)
    for fn in steps:
        try:
            fn(url, headers, path)
            return path
        except BaseException as e:
            last_err = e
            try:
                path.unlink()
            except OSError:
                pass
    assert last_err is not None
    raise last_err


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def delete_video(video_path: Path) -> None:
    try:
        video_path.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    _test_url = "https://scoredata.me/chunks/398d80cb07d74fcd8de4cd198751e1.mp4"
    path = download_video(_test_url)
    n = path.stat().st_size
    print(f"saved {n} bytes -> {path}")
    delete_video(path)
