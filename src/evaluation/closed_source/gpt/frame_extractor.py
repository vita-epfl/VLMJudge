"""Sample 1 frame/second from a clip and return base64 data URLs.

Clips are ~5 seconds at ~24 fps → 5 JPEG frames per clip. Frames are cached on
disk so re-runs (and the batch-submit → retrieve split) don't redo the work.

Prompt-cache hit on OpenAI's side requires the image bytes — and therefore
tokens — to be byte-identical across requests for the same video. We achieve
that by serialising the PNG→JPEG encoding deterministically (quality=90, no
extra metadata) and caching the resulting file on disk.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_DEFAULT_CACHE_DIR = _HERE / ".frame_cache"
JPEG_QUALITY = 90
DEFAULT_FPS = 1
DEFAULT_NUM_FRAMES = 5  # clips are 5s long


@dataclass(frozen=True)
class Frame:
    index: int      # 0-based frame index in the sampled sequence
    path: Path      # on-disk JPEG
    data_url: str   # `data:image/jpeg;base64,...` ready for OpenAI vision


def _cache_dir() -> Path:
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_CACHE_DIR


def _cache_key(video: Path) -> str:
    stat = video.stat()
    h = hashlib.sha1(
        f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()
    ).hexdigest()[:16]
    return h


def _encode_data_url(jpeg_path: Path) -> str:
    b = jpeg_path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


def extract_frames(
    video_path: Path | str,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
    force: bool = False,
) -> list[Frame]:
    """Return `num_frames` evenly-spaced JPEGs (cached), at 1 per `1/fps` seconds.

    We pick frames by timestamp (0s, 1s, 2s, ...) rather than by frame index, so
    videos with different native fps still give us the same logical sampling.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(video)

    key = _cache_key(video)
    out_dir = _cache_dir() / key
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"frame_{i:02d}.jpg" for i in range(num_frames)]

    if not force and all(p.is_file() for p in expected):
        return [Frame(i, p, _encode_data_url(p)) for i, p in enumerate(expected)]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Timestamps in seconds: 0, 1, 2, ..., num_frames-1. Clip to last available.
    targets = [min(int(round(i * native_fps / fps)), max(0, total - 1))
               for i in range(num_frames)]

    try:
        for i, frame_idx in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} of {video} "
                    f"(native_fps={native_fps}, total={total})"
                )
            ok = cv2.imwrite(
                str(expected[i]),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
            )
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {expected[i]}")
    finally:
        cap.release()

    return [Frame(i, p, _encode_data_url(p)) for i, p in enumerate(expected)]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: frame_extractor.py <video_path>")
        sys.exit(1)
    frames = extract_frames(sys.argv[1])
    for f in frames:
        print(f"{f.index}: {f.path}  ({f.path.stat().st_size} bytes)")
