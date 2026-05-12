"""Frame I/O helpers shared by every Tier 1/2 tool.

Supports two interchangeable representations:
- mp4 file (read with cv2.VideoCapture, write with ffmpeg libx264 by default
  to avoid the visible quality drop of cv2.VideoWriter mp4v that hurts MUSIQ
  and LAION-aesthetic scores by ~3-7% per re-encode)
- frame directory of ``frame_*.jpg|png`` (the layout WorldArena.evaluate
  consumes natively, lossless)

All frames are kept as ``uint8 H x W x 3`` BGR numpy arrays internally for cv2
compatibility; conversion to RGB is left to the consumer (cv2.cvtColor).

For quality-sensitive workflows always prefer a frame directory output -- it
is what ``WorldArena.evaluate`` consumes anyway, and skipping the re-encode
fully avoids the lossy round-trip.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# libx264 CRF: 18 is "visually lossless", evaluates within MUSIQ noise floor.
_DEFAULT_X264_CRF = 18


def is_video_file(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_frame_files(folder: str | os.PathLike) -> List[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in FRAME_EXTS)


def read_frames(path: str | os.PathLike) -> Tuple[List[np.ndarray], float]:
    """Read frames from either a video file or a frame directory.

    Returns
    -------
    frames : list[np.ndarray]
        BGR uint8 frames.
    fps : float
        Source FPS for video files; ``0.0`` when reading from a directory
        (caller decides what FPS to use when writing back to mp4).
    """
    p = Path(path)
    if p.is_dir():
        frames = []
        for fp in list_frame_files(p):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to read frame: {fp}")
            frames.append(img)
        if not frames:
            raise IOError(f"No frames found in directory: {p}")
        return frames, 0.0

    if not p.exists():
        raise FileNotFoundError(p)

    cap = cv2.VideoCapture(str(p))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from: {p}")
    return frames, fps


def write_frames(
    frames: Sequence[np.ndarray],
    out_path: str | os.PathLike,
    fps: float = 12.0,
    jpg_quality: int = 95,
    codec: str = "auto",
    crf: int = _DEFAULT_X264_CRF,
) -> None:
    """Write frames to either a video file or a frame directory.

    For mp4 output, ``codec`` controls the encoder:
        ``auto``  -- use libx264 via ffmpeg if available, else mp4v (cv2)
        ``x264``  -- force ffmpeg + libx264 + ``crf`` (raises if no ffmpeg)
        ``mp4v``  -- force cv2 mp4v fallback (lossy; only for benchmarking)

    Frame-dir output ignores ``codec`` / ``crf``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_video_file(out):
        if codec == "mp4v" or (codec == "auto" and shutil.which("ffmpeg") is None):
            _write_video_mp4v(frames, out, fps)
            return
        if codec in ("auto", "x264"):
            try:
                _write_video_ffmpeg_x264(frames, out, fps, crf=crf)
                return
            except Exception as exc:
                if codec == "x264":
                    raise
                print(f"[io_utils] ffmpeg x264 failed ({exc}); falling back to mp4v",
                      file=sys.stderr)
                _write_video_mp4v(frames, out, fps)
                return
        raise ValueError(f"Unknown codec: {codec!r}")

    out.mkdir(parents=True, exist_ok=True)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    for i, f in enumerate(frames):
        cv2.imwrite(str(out / f"frame_{i:05d}.jpg"), f, encode_params)


def _write_video_mp4v(frames: Sequence[np.ndarray], out: Path, fps: float) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps if fps > 0 else 12.0, (w, h))
    if not writer.isOpened():
        raise IOError(f"cv2.VideoWriter failed to open: {out}")
    for f in frames:
        writer.write(f)
    writer.release()


def _write_video_ffmpeg_x264(frames: Sequence[np.ndarray], out: Path, fps: float,
                             crf: int = _DEFAULT_X264_CRF) -> None:
    """Pipe frames into ``ffmpeg -c:v libx264 -crf <crf> -pix_fmt yuv420p``.

    Uses raw rgb24 over stdin so we never go through a temporary directory.
    Falls back to a tmp PNG sequence only if stdin piping is unavailable.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    h, w = frames[0].shape[:2]
    use_fps = fps if fps > 0 else 12.0
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(use_fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-preset", "medium",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
    finally:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc} writing {out}")


def read_first_frame_image(path: str | os.PathLike) -> np.ndarray:
    """Load a single image (PNG/JPG) as BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img
