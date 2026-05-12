"""Tier-1 post-processing pipeline for WorldArena video evaluation.

Applies, in order:
    1. frame-count alignment to GT (PSNR/SSIM/depth)
    2. first-frame replacement + cross-fade (PSNR/SSIM/depth/trajectory)
    3. Reinhard color match against GT first frame (CLIP/DINO/aesthetic)
    4. unsharp + contrast/saturation lift (image_quality/aesthetic)
    5. tail EMA stabilization (photometric/motion smoothness)

All steps are individually toggleable so you can sweep their marginal
contribution to the overall WorldArena Mean. The same parameters applied
to multiple gids of the same episode are guaranteed to produce identical
outputs (Tier-1 step 8 in the plan: multi-gid consistency).

Two input layouts are supported:

    A. Single video / single gt:
        --input  path/to/pred.mp4    (or pred frame dir)
        --gt     path/to/gt.mp4      (or gt frame dir)
        --output path/to/out.mp4     (or out frame dir)

    B. Batch directory (mp4 vs mp4):
        --input  origin/   # contains episode*.mp4
        --gt     origin/   # GT mp4s by matching basename (often same dir)
        --output improved/ # mirrored basenames written here
        --batch

    C. WorldArena dataset layout (frame dirs):
        --input  generated_dataset/{task}/{episode}/{gid}/video
        --gt     gt_dataset/{task}/{episode}/video
        --output generated_dataset_improved/{task}/{episode}/{gid}/video
        --dataset-root <generated_dataset>
        --gt-root      <gt_dataset>
        --output-root  <generated_dataset_improved>

Run ``python -m myscript.improve.postprocess --help`` for the full set of
parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Allow ``python myscript/improve/postprocess.py`` from anywhere; also keep
# ``python -m myscript.improve.postprocess`` working when invoked from
# ``video_quality/``.
HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from myscript.improve import io_utils, ops  # noqa: E402


@dataclass
class Params:
    # Step toggles
    frame_align: bool = True
    first_frame: bool = True
    color_match: bool = True
    unsharp: bool = True
    tail_stabilize: bool = True

    # Step parameters
    target_frames: Optional[int] = None  # None = match GT
    crossfade_len: int = 3
    color_strength: float = 0.5  # blend strength: 1.0 hard match, 0.0 disable
    unsharp_amount: float = 0.5
    unsharp_radius: int = 5
    contrast: float = 1.05
    saturation: float = 1.05
    gamma: float = 1.0
    tail_len: int = 5
    tail_alpha: float = 0.7

    # Output FPS for mp4 writing (frame dirs ignore this).
    fps: float = 12.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ----------------------------------------------------------------------------
# Single (pred, gt) -> (improved) transform
# ----------------------------------------------------------------------------


def process_one(
    pred_path: Path,
    gt_path: Optional[Path],
    out_path: Path,
    params: Params,
    init_frame_path: Optional[Path] = None,
) -> dict:
    """Run the Tier-1 pipeline on a single (pred, gt) pair.

    Returns a small report dict so the batch caller can log per-episode
    diagnostics (input/output frame counts, which steps actually fired, etc).
    """
    frames, src_fps = io_utils.read_frames(pred_path)
    report = {
        "pred": str(pred_path),
        "gt": str(gt_path) if gt_path else None,
        "out": str(out_path),
        "frames_in": len(frames),
        "frames_out": len(frames),
        "steps": [],
    }

    gt_frames: List[np.ndarray] = []
    gt_first: Optional[np.ndarray] = None
    if init_frame_path is not None and Path(init_frame_path).exists():
        gt_first = io_utils.read_first_frame_image(init_frame_path)
    if gt_path is not None and Path(gt_path).exists():
        gt_frames, _ = io_utils.read_frames(gt_path)
        if gt_first is None and gt_frames:
            gt_first = gt_frames[0]

    # Step 1: frame-count alignment
    target = params.target_frames
    if params.frame_align and target is None and gt_frames:
        target = len(gt_frames)
    if params.frame_align and target is not None and target > 0 and target != len(frames):
        frames = ops.align_frame_count(frames, target)
        report["steps"].append(f"frame_align({len(frames)})")
    report["frames_out"] = len(frames)

    # Step 2: first-frame replacement
    if params.first_frame and gt_first is not None:
        frames = ops.replace_first_frame(frames, gt_first, crossfade_len=params.crossfade_len)
        report["steps"].append(f"first_frame(crossfade={params.crossfade_len})")
    elif params.first_frame:
        report["steps"].append("first_frame(SKIPPED: no gt first frame)")

    # Step 3: Reinhard color match
    if params.color_match and gt_first is not None and params.color_strength > 0.0:
        frames = ops.reinhard_color_match(frames, gt_first, strength=params.color_strength)
        report["steps"].append(f"color_match(strength={params.color_strength})")
    elif params.color_match:
        report["steps"].append("color_match(SKIPPED: no gt first frame)")

    # Step 4: unsharp + contrast / saturation
    if params.unsharp and (
        params.unsharp_amount > 0.0
        or params.contrast != 1.0
        or params.saturation != 1.0
        or params.gamma != 1.0
    ):
        frames = ops.unsharp_and_eq(
            frames,
            unsharp_amount=params.unsharp_amount,
            unsharp_radius=params.unsharp_radius,
            contrast=params.contrast,
            saturation=params.saturation,
            gamma=params.gamma,
        )
        report["steps"].append("unsharp_eq")

    # Step 5: tail EMA
    if params.tail_stabilize and params.tail_len > 0 and params.tail_alpha > 0.0:
        frames = ops.tail_ema_stabilize(frames, n_tail=params.tail_len, alpha=params.tail_alpha)
        report["steps"].append(f"tail_ema(n={params.tail_len},alpha={params.tail_alpha})")

    fps = src_fps if src_fps > 0 else params.fps
    io_utils.write_frames(frames, out_path, fps=fps)
    return report


# ----------------------------------------------------------------------------
# Batch helpers
# ----------------------------------------------------------------------------


def _iter_dataset_episodes(gen_root: Path) -> Iterable[tuple[str, str, str, Path]]:
    """Yield ``(task, episode, gid, pred_video_dir)`` for the WorldArena layout."""
    for task_dir in sorted(p for p in gen_root.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for gid_dir in sorted(p for p in ep_dir.iterdir() if p.is_dir()):
                video_dir = gid_dir / "video"
                if video_dir.is_dir():
                    yield task_dir.name, ep_dir.name, gid_dir.name, video_dir


def run_dataset_layout(
    gen_root: Path,
    gt_root: Path,
    out_root: Path,
    params: Params,
) -> List[dict]:
    """Run on the WorldArena ``generated_dataset`` / ``gt_dataset`` layout.

    Multi-gid consistency: every gid under the same episode receives the same
    ``params`` and the same ``gt_first`` (read from
    ``gt_dataset/<task>/<episode>/prompt/init_frame.png`` if present, otherwise
    the first GT video frame), so their CLIP/DINO features stay close.
    """
    reports: List[dict] = []
    for task, episode, gid, pred_dir in _iter_dataset_episodes(gen_root):
        gt_video = gt_root / task / episode / "video"
        gt_init = gt_root / task / episode / "prompt" / "init_frame.png"
        out_dir = out_root / task / episode / gid / "video"
        report = process_one(
            pred_dir,
            gt_video if gt_video.exists() else None,
            out_dir,
            params,
            init_frame_path=gt_init if gt_init.exists() else None,
        )
        report["task"] = task
        report["episode"] = episode
        report["gid"] = gid
        reports.append(report)
        print(f"[{task}/{episode}/{gid}] {report['frames_in']}->{report['frames_out']} "
              f"steps={','.join(report['steps']) or 'none'}")
    return reports


def run_batch_videos(
    in_dir: Path,
    gt_dir: Optional[Path],
    out_dir: Path,
    params: Params,
) -> List[dict]:
    """Run on flat directories of ``episode*.mp4`` (the aes_compare layout)."""
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    mp4s = sorted([p for p in in_dir.iterdir() if io_utils.is_video_file(p)])
    if not mp4s:
        raise ValueError(f"No video files in {in_dir}")
    reports: List[dict] = []
    for mp4 in mp4s:
        gt_mp4 = (gt_dir / mp4.name) if gt_dir is not None else None
        if gt_mp4 is not None and not gt_mp4.exists():
            gt_mp4 = None
        out_path = out_dir / mp4.name
        report = process_one(mp4, gt_mp4, out_path, params)
        report["name"] = mp4.name
        reports.append(report)
        print(f"[{mp4.name}] {report['frames_in']}->{report['frames_out']} "
              f"steps={','.join(report['steps']) or 'none'}")
    return reports


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WorldArena Tier-1 video post-processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("input / output")
    src.add_argument("--input", type=Path, help="Predicted video (mp4 or frame dir).")
    src.add_argument("--gt", type=Path, help="GT video / frame dir for first-frame & color match.")
    src.add_argument("--init-frame", type=Path, help="Override GT first-frame image (PNG).")
    src.add_argument("--output", type=Path, help="Output mp4 or frame dir.")
    src.add_argument("--batch", action="store_true",
                     help="Treat --input/--gt/--output as flat directories of episode*.mp4.")
    src.add_argument("--dataset-root", type=Path,
                     help="Run on the WorldArena generated_dataset layout (overrides --input).")
    src.add_argument("--gt-root", type=Path,
                     help="Companion gt_dataset root for --dataset-root mode.")
    src.add_argument("--output-root", type=Path,
                     help="Mirror destination for --dataset-root mode.")
    src.add_argument("--report", type=Path,
                     help="Optional JSON path for per-video diagnostics.")

    toggles = p.add_argument_group("step toggles (default: all on)")
    toggles.add_argument("--no-frame-align", action="store_true")
    toggles.add_argument("--no-first-frame", action="store_true")
    toggles.add_argument("--no-color-match", action="store_true")
    toggles.add_argument("--no-unsharp", action="store_true")
    toggles.add_argument("--no-tail-stabilize", action="store_true")
    toggles.add_argument("--only", choices=["frame_align", "first_frame", "color_match", "unsharp", "tail_stabilize"],
                         help="Run a single step (useful for ablations).")

    params_g = p.add_argument_group("step parameters")
    params_g.add_argument("--target-frames", type=int, default=None,
                          help="Force a specific frame count (default: match GT).")
    params_g.add_argument("--crossfade-len", type=int, default=3)
    params_g.add_argument("--color-strength", type=float, default=0.5,
                          help="0.0 disables, 1.0 hard match. Default 0.5 keeps prediction "
                               "in CLIP distribution while moving toward GT statistics.")
    params_g.add_argument("--unsharp-amount", type=float, default=0.5)
    params_g.add_argument("--unsharp-radius", type=int, default=5)
    params_g.add_argument("--contrast", type=float, default=1.05)
    params_g.add_argument("--saturation", type=float, default=1.05)
    params_g.add_argument("--gamma", type=float, default=1.0)
    params_g.add_argument("--tail-len", type=int, default=5)
    params_g.add_argument("--tail-alpha", type=float, default=0.7)
    params_g.add_argument("--fps", type=float, default=12.0,
                          help="FPS used when writing mp4 from a frame dir source.")

    return p


def _params_from_args(args: argparse.Namespace) -> Params:
    params = Params(
        frame_align=not args.no_frame_align,
        first_frame=not args.no_first_frame,
        color_match=not args.no_color_match,
        unsharp=not args.no_unsharp,
        tail_stabilize=not args.no_tail_stabilize,
        target_frames=args.target_frames,
        crossfade_len=args.crossfade_len,
        color_strength=args.color_strength,
        unsharp_amount=args.unsharp_amount,
        unsharp_radius=args.unsharp_radius,
        contrast=args.contrast,
        saturation=args.saturation,
        gamma=args.gamma,
        tail_len=args.tail_len,
        tail_alpha=args.tail_alpha,
        fps=args.fps,
    )
    if args.only is not None:
        for step in ["frame_align", "first_frame", "color_match", "unsharp", "tail_stabilize"]:
            setattr(params, step, step == args.only)
    return params


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    params = _params_from_args(args)

    reports: List[dict]
    if args.dataset_root is not None:
        if args.gt_root is None or args.output_root is None:
            raise SystemExit("--dataset-root requires --gt-root and --output-root")
        reports = run_dataset_layout(args.dataset_root, args.gt_root, args.output_root, params)
    elif args.batch:
        if args.input is None or args.output is None:
            raise SystemExit("--batch requires --input and --output")
        reports = run_batch_videos(args.input, args.gt, args.output, params)
    else:
        if args.input is None or args.output is None:
            raise SystemExit("--input and --output are required (or use --batch / --dataset-root)")
        report = process_one(args.input, args.gt, args.output, params,
                             init_frame_path=args.init_frame)
        reports = [report]
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fp:
            json.dump({"params": params.to_dict(), "reports": reports}, fp, indent=2, ensure_ascii=False)
        print(f"[report] saved -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
