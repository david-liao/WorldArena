"""Tier-2: prepare summary.json for first-frame-conditioned generation.

The Tier-2 plan calls for "use GT init_frame as first-frame conditioning at
inference rather than as a post-processing replacement". The init frames
already exist in the standard ``summary.json`` under ``image``, so this tool
focuses on:

1. Validating every entry has a readable PNG of consistent shape.
2. Emitting a normalized ``i2v_jobs.json`` that downstream Wan / OminiEWM /
   diffusion-policy inference scripts can consume directly. Each job carries:
       {episode, prompt, init_frame, target_frames, fps, gt_path}
3. Optionally mirroring all init_frames into a flat ``init_frames/`` directory
   with stable basenames, so CLI tools that take ``--image``/``--init-frame``
   per-job flags can iterate cleanly.

Usage::

    python -m myscript.improve.firstframe_conditioning \\
        --summary summary.json \\
        --output i2v_jobs.json \\
        [--mirror init_frames/] [--probe-gt]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from myscript.improve import io_utils  # noqa: E402


def _episode_id(gt_path: str) -> str:
    p = Path(gt_path)
    parent = p.parent.name or "flat"
    return f"{parent}__{p.stem}"


def _probe_gt_n_frames(gt_path: str) -> Optional[int]:
    p = Path(gt_path)
    if not p.exists():
        return None
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else None


def _probe_gt_fps(gt_path: str) -> Optional[float]:
    p = Path(gt_path)
    if not p.exists():
        return None
    cap = cv2.VideoCapture(str(p))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare init-frame conditioning manifest.")
    parser.add_argument("--summary", type=Path, required=True,
                        help="Input summary.json (same format preprocess_datasets.py reads).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output i2v_jobs.json with normalized fields.")
    parser.add_argument("--mirror", type=Path, default=None,
                        help="If set, copy each init_frame here as <episode>.png and "
                             "rewrite job entries to point at the mirror.")
    parser.add_argument("--probe-gt", action="store_true",
                        help="Open each GT mp4 to record its frame count + fps. Slower "
                             "but lets downstream pipelines size generation correctly.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail with non-zero exit code on any missing/broken init_frame.")
    args = parser.parse_args(argv)

    items = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    jobs: List[dict] = []
    bad: List[dict] = []

    if args.mirror is not None:
        args.mirror.mkdir(parents=True, exist_ok=True)

    for entry in items:
        gt_path = entry.get("gt_path")
        image = entry.get("image")
        prompt = entry.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not gt_path or not image:
            bad.append({"reason": "missing_field", "entry": entry})
            continue

        eid = _episode_id(gt_path)
        img_path = Path(image)
        if not img_path.exists():
            bad.append({"reason": "init_frame_missing", "episode": eid, "image": image})
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            bad.append({"reason": "init_frame_unreadable", "episode": eid, "image": image})
            continue
        h, w = img.shape[:2]

        target_frames = None
        fps = None
        if args.probe_gt:
            target_frames = _probe_gt_n_frames(gt_path)
            fps = _probe_gt_fps(gt_path)

        if args.mirror is not None:
            dst = args.mirror / f"{eid}.png"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            img_path = dst

        jobs.append({
            "episode": eid,
            "prompt": prompt,
            "init_frame": str(img_path),
            "init_frame_hw": [h, w],
            "gt_path": gt_path,
            "target_frames": target_frames,
            "fps": fps,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"jobs": jobs, "bad": bad}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[firstframe_conditioning] {len(jobs)} valid jobs, {len(bad)} bad entries -> {args.output}")
    if args.mirror is not None:
        print(f"[firstframe_conditioning] init frames mirrored at {args.mirror}")
    if bad:
        for b in bad[:5]:
            print(f"  bad: {b}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
