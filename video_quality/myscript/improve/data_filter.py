"""Tier-3: filter a training-video manifest by aesthetic + motion.

Given a manifest of training videos (one mp4 per row in a CSV / JSONL / plain
list), this script scores every video on:

* ``aesthetic`` -- LAION ViT-L/14 mean over evenly-sampled frames.
* ``motion``    -- mean Farneback flow magnitude std-dev (cheap dynamic proxy).

It then writes a kept / dropped manifest based on user-supplied thresholds:

* ``--min-aesthetic`` (e.g. ``0.55``) drops "ugly" clips whose features confuse
  the aesthetic head; matches the plan's "filter by LAION-aesthetic".
* ``--min-motion`` (e.g. ``0.20``) drops near-static clips that would push the
  generator toward outputs which trigger the low-dynamic gate shared by
  background_consistency / subject_consistency / photometric_smoothness in
  WorldArena.

Both thresholds are independent. Setting either to ``-1`` disables that side.
By default we evaluate on 8 evenly-spaced frames per clip to keep this
script usable on commodity GPUs at training-set scale.

Usage::

    python -m myscript.improve.data_filter \\
        --input train_videos.csv --csv-column path \\
        --output kept.csv --dropped dropped.csv \\
        --min-aesthetic 0.55 --min-motion 0.20 \\
        --config config/config.yaml \\
        --frames-per-clip 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from myscript.improve import io_utils  # noqa: E402


# ----------------------------------------------------------------------------
# Manifest parsing
# ----------------------------------------------------------------------------


def _read_manifest(path: Path, csv_column: Optional[str]) -> List[Path]:
    """Accept .csv / .jsonl / .txt / .json with a flat list of paths."""
    suffix = path.suffix.lower()
    out: List[Path] = []
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            col = csv_column or "path"
            if col not in (reader.fieldnames or []):
                raise ValueError(f"CSV column {col!r} not found in {path}; "
                                 f"available: {reader.fieldnames}")
            for row in reader:
                out.append(Path(row[col]))
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj["path"] if isinstance(obj, dict) else obj
                out.append(Path(p))
    elif suffix == ".json":
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            for entry in data:
                p = entry["path"] if isinstance(entry, dict) else entry
                out.append(Path(p))
        else:
            raise ValueError(f"{path}: top-level JSON must be a list")
    else:  # plain text, one path per line
        out = [Path(line.strip()) for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#")]
    return out


def _write_csv(rows: Iterable[dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def _sample_frames(video_path: Path, n: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = max(1, min(n, total))
    idx = np.linspace(0, total - 1, n).astype(int)
    out: List[np.ndarray] = []
    cur = 0
    target = set(int(i) for i in idx)
    while cur < total:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in target:
            out.append(frame)
            if len(out) == n:
                break
        cur += 1
    cap.release()
    return out


def _motion_score(frames: List[np.ndarray]) -> float:
    """Mean(std(Farneback flow magnitude)) across consecutive pairs."""
    if len(frames) < 2:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    stds = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        stds.append(float(mag.std()))
    return float(np.mean(stds))


class _Aesthetic:
    def __init__(self, clip_ckpt: str, head_ckpt: str, device: str = "cuda"):
        import clip
        import torch
        import torch.nn as nn
        from WorldArena.utils import clip_transform  # noqa
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(clip_ckpt, device=self.device)
        head = nn.Linear(768, 1)
        head.load_state_dict(torch.load(head_ckpt, weights_only=False))
        self.head = head.to(self.device).eval()
        self.tx = clip_transform(224)

    def __call__(self, frames: List[np.ndarray]) -> float:
        torch = self.torch
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        batch = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).contiguous()
        batch = self.tx(batch).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = self.head(feats).squeeze(-1)
        return float(scores.mean()) / 10.0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Filter training videos by aesthetic + motion.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Manifest: csv (with --csv-column), jsonl, json list, or plain txt.")
    parser.add_argument("--csv-column", type=str, default="path",
                        help="Column name when --input is csv.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Kept manifest path (csv).")
    parser.add_argument("--dropped", type=Path, default=None,
                        help="Optional csv listing dropped clips with their scores.")
    parser.add_argument("--min-aesthetic", type=float, default=-1.0,
                        help="Aesthetic threshold in [0,1]; -1 disables filter.")
    parser.add_argument("--min-motion", type=float, default=-1.0,
                        help="Mean Farneback flow std-dev threshold; -1 disables filter.")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--config", type=Path, default=None,
                        help="config.yaml to source CLIP + aesthetic-head paths.")
    parser.add_argument("--ckpt-clip", type=str, default=None)
    parser.add_argument("--ckpt-aes", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="Score at most N clips (0 = all).")
    parser.add_argument("--skip-broken", action="store_true",
                        help="If a clip fails to open, drop it silently instead of erroring.")
    args = parser.parse_args(argv)

    paths = _read_manifest(args.input, args.csv_column)
    if args.limit > 0:
        paths = paths[: args.limit]

    aesthetic = None
    if args.min_aesthetic >= 0.0:
        clip_ckpt = args.ckpt_clip
        head_ckpt = args.ckpt_aes
        if (clip_ckpt is None or head_ckpt is None) and args.config is not None:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()) or {}
            aes = cfg.get("ckpt", {}).get("aesthetic_quality", {})
            clip_ckpt = clip_ckpt or aes.get("clip")
            head_ckpt = head_ckpt or aes.get("aesthetic_head")
        if not (clip_ckpt and head_ckpt):
            raise SystemExit("--min-aesthetic >=0 requires --ckpt-clip + --ckpt-aes (or config).")
        aesthetic = _Aesthetic(clip_ckpt, head_ckpt, device=args.device)

    rows_kept: List[dict] = []
    rows_dropped: List[dict] = []
    for i, p in enumerate(paths):
        try:
            frames = _sample_frames(p, args.frames_per_clip)
        except Exception as exc:
            if args.skip_broken:
                rows_dropped.append({"path": str(p), "reason": f"open_error:{exc}"})
                continue
            raise

        if not frames:
            rows_dropped.append({"path": str(p), "reason": "no_frames"})
            continue

        aes = aesthetic(frames) if aesthetic is not None else None
        motion = _motion_score(frames)

        keep = True
        reason = ""
        if args.min_aesthetic >= 0.0 and aes is not None and aes < args.min_aesthetic:
            keep = False
            reason = f"aesthetic<{args.min_aesthetic:.3f}"
        if keep and args.min_motion >= 0.0 and motion < args.min_motion:
            keep = False
            reason = f"motion<{args.min_motion:.3f}"

        record = {
            "path": str(p),
            "aesthetic": aes if aes is not None else "",
            "motion": motion,
        }
        if keep:
            rows_kept.append(record)
        else:
            rows_dropped.append({**record, "reason": reason})

        if (i + 1) % 50 == 0:
            print(f"[data_filter] {i + 1}/{len(paths)} kept={len(rows_kept)} dropped={len(rows_dropped)}")

    fieldnames = ["path", "aesthetic", "motion"]
    _write_csv(rows_kept, args.output, fieldnames)
    if args.dropped is not None:
        _write_csv(rows_dropped, args.dropped, fieldnames + ["reason"])
    print(f"[data_filter] DONE: kept {len(rows_kept)}/{len(paths)}, dropped {len(rows_dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
