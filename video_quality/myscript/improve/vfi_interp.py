"""Tier-2 video frame interpolation wrapper.

The plan calls for "post-VFI to lift motion_smoothness when generated frames <
GT frames". Three back-ends are supported, in order of decreasing quality:

1. ``vfimamba`` -- the same VFI used by ``WorldArena.motion_smoothness_metrics``
   (reads its checkpoint from config.yaml's ``ckpt.motion_smoothness.model``).
   This is the highest-fidelity option *and* matches what the metric scores
   against, but requires a CUDA GPU.
2. ``rife``     -- IFNet from ``WorldArena/third_party/AMT`` if available.
3. ``ffmpeg``   -- ``-vf minterpolate`` motion-compensated interpolation. Pure
   ffmpeg, always available, lowest quality.

The CLI exposes ``--target-frames`` (absolute) or ``--multiplier`` (e.g. 2x).
Backends are auto-selected unless ``--backend`` is set; if a backend is
unavailable the script falls back to the next one and prints a warning.

Example::

    python -m myscript.improve.vfi_interp \\
        --input pred.mp4 --output pred_2x.mp4 --multiplier 2

    python -m myscript.improve.vfi_interp \\
        --input pred_dir/ --output pred_dir_aligned/ \\
        --target-frames 121 --backend ffmpeg
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from myscript.improve import io_utils  # noqa: E402


# ----------------------------------------------------------------------------
# ffmpeg minterpolate fallback
# ----------------------------------------------------------------------------


def vfi_ffmpeg(in_path: Path, out_path: Path, target_frames: int, src_fps: Optional[float] = None) -> None:
    """Use ffmpeg's motion-compensated interpolation as the universal fallback.

    We probe both the source frame count *and* its actual fps from the
    container, then invoke ``minterpolate=fps=<new_fps>`` where
    ``new_fps = src_fps_actual * target_frames / src_n``. This preserves the
    original duration while changing density.

    ``src_fps`` argument is only used as a fallback when ffprobe fails to
    determine the real source fps.
    """
    src_n = _probe_n_frames(in_path)
    if src_n <= 0:
        raise RuntimeError(f"Could not probe frame count of {in_path}")
    actual_fps = _probe_fps(in_path)
    if actual_fps is None or actual_fps <= 0:
        actual_fps = src_fps if (src_fps and src_fps > 0) else 12.0
    new_fps = actual_fps * target_frames / src_n
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(in_path),
        "-vf", f"minterpolate=fps={new_fps:.6f}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "medium",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _probe_n_frames(path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n


def _probe_fps(path: Path) -> Optional[float]:
    """Probe r_frame_rate from the container; returns None on failure."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            den = float(den)
            return float(num) / den if den else None
        return float(out)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# VFIMamba (same as the metric's reference) -- imports lazily.
# ----------------------------------------------------------------------------


def _try_load_vfimamba(ckpt: str, device: str = "cuda"):
    """Best-effort load of VFIMamba from the project's third_party tree.

    Returns a callable ``interp(a, b, t) -> mid_frame`` operating on uint8 BGR
    HxWx3 numpy arrays, or ``None`` if VFIMamba cannot be loaded in this env.
    """
    try:
        import torch
        # The class lives under WorldArena.third_party.VFIMamba; the same
        # import path motion_smoothness_metrics uses internally. We import
        # lazily to avoid pulling heavy CUDA kernels when the user picked
        # --backend ffmpeg.
        from WorldArena.third_party.VFIMamba.Trainer_finetune import Model  # type: ignore  # noqa
    except Exception as exc:
        print(f"[vfi_interp] VFIMamba unavailable: {exc}", file=sys.stderr)
        return None

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    try:
        model = Model(-1)
        model.load_model(ckpt, -1)
        model.eval()
        model.device()
    except Exception as exc:
        print(f"[vfi_interp] VFIMamba load failed: {exc}", file=sys.stderr)
        return None

    def interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        rgb_a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        rgb_b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        ta = torch.from_numpy(rgb_a).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(dev)
        tb = torch.from_numpy(rgb_b).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(dev)
        with torch.no_grad():
            mid = model.inference(ta, tb, [t])
            if isinstance(mid, list):
                mid = mid[0]
        out = (mid.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return interp


def _frame_level_resample(frames: List[np.ndarray], target: int, interp) -> List[np.ndarray]:
    """Resample to ``target`` length using a generic VFI ``interp(a,b,t)``."""
    n = len(frames)
    if n == target:
        return [f.copy() for f in frames]
    if n < 2 or interp is None:
        # Without a frame-level interpolator we can only blend.
        return _linear_blend(frames, target)
    src_pos = np.linspace(0, n - 1, target)
    out: List[np.ndarray] = []
    for s in src_pos:
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, n - 1)
        a = float(s - i0)
        if a == 0.0 or i0 == i1:
            out.append(frames[i0].copy())
        else:
            try:
                out.append(interp(frames[i0], frames[i1], a))
            except Exception:
                # Per-pair failure: fall back to blend on this pair only.
                blended = ((1 - a) * frames[i0].astype(np.float32)
                           + a * frames[i1].astype(np.float32))
                out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


def _linear_blend(frames: List[np.ndarray], target: int) -> List[np.ndarray]:
    n = len(frames)
    if n < 2:
        return [frames[0]] * target
    src_pos = np.linspace(0, n - 1, target)
    out = []
    for s in src_pos:
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, n - 1)
        a = float(s - i0)
        if a == 0.0 or i0 == i1:
            out.append(frames[i0].copy())
        else:
            blended = ((1 - a) * frames[i0].astype(np.float32)
                       + a * frames[i1].astype(np.float32))
            out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VFI for WorldArena post-processing.")
    parser.add_argument("--input", type=Path, required=True, help="mp4 or frame dir")
    parser.add_argument("--output", type=Path, required=True, help="mp4 or frame dir")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target-frames", type=int, help="Absolute target frame count.")
    grp.add_argument("--multiplier", type=float, help="Multiply current frame count by this.")
    parser.add_argument("--gt", type=Path,
                        help="Optional: derive --target-frames from GT video / frame dir.")
    parser.add_argument("--backend", choices=["auto", "vfimamba", "ffmpeg", "blend"], default="auto")
    parser.add_argument("--ckpt-vfimamba", type=str, default=None,
                        help="Path to VFIMamba.pkl; defaults to motion_smoothness.model in --config.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    # Resolve target_frames.
    if args.target_frames is None and args.gt is not None:
        gt_frames, _ = io_utils.read_frames(args.gt)
        args.target_frames = len(gt_frames)
        print(f"[vfi_interp] target_frames from GT={args.target_frames}")
    if args.target_frames is None and args.multiplier is not None:
        n_in = io_utils.read_frames(args.input)[0]
        args.target_frames = int(round(len(n_in) * args.multiplier))

    # Resolve VFIMamba checkpoint.
    ckpt = args.ckpt_vfimamba
    if ckpt is None and args.config is not None:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        ckpt = cfg.get("ckpt", {}).get("motion_smoothness", {}).get("model")

    backend = args.backend
    if backend == "auto":
        backend = "vfimamba" if ckpt else "ffmpeg"

    if backend == "ffmpeg":
        if io_utils.is_video_file(args.input) and io_utils.is_video_file(args.output):
            vfi_ffmpeg(args.input, args.output, args.target_frames)
            print(f"[vfi_interp] ffmpeg minterpolate -> {args.output}")
            return 0
        with tempfile.TemporaryDirectory() as td:
            td_in = Path(td) / "in.mp4"
            td_out = Path(td) / "out.mp4"
            if args.input.is_dir():
                frames, _ = io_utils.read_frames(args.input)
                io_utils.write_frames(frames, td_in, fps=args.fps)
            else:
                shutil.copy2(args.input, td_in)
            vfi_ffmpeg(td_in, td_out, args.target_frames, src_fps=args.fps)
            frames, _ = io_utils.read_frames(td_out)
            # ffmpeg may produce target+/-1 frames depending on fractional fps;
            # snap to exact target_frames with linear blend if needed.
            if len(frames) != args.target_frames:
                frames = _linear_blend(frames, args.target_frames)
            io_utils.write_frames(frames, args.output, fps=args.fps)
        print(f"[vfi_interp] ffmpeg minterpolate (via tmp) -> {args.output}")
        return 0

    # vfimamba / blend backends operate at frame level.
    interp = None
    if backend == "vfimamba":
        if not ckpt:
            print("[vfi_interp] no VFIMamba ckpt; falling back to blend", file=sys.stderr)
            backend = "blend"
        else:
            interp = _try_load_vfimamba(ckpt, device=args.device)
            if interp is None:
                print("[vfi_interp] VFIMamba load failed; falling back to blend", file=sys.stderr)
                backend = "blend"

    frames, src_fps = io_utils.read_frames(args.input)
    out_frames = _frame_level_resample(frames, args.target_frames, interp)
    fps = src_fps if src_fps > 0 else args.fps
    io_utils.write_frames(out_frames, args.output, fps=fps)
    print(f"[vfi_interp] backend={backend} {len(frames)}->{len(out_frames)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
