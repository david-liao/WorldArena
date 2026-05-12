"""Tier-2 seed selection: pick the best of N candidate generations per episode.

Score = w_q * MUSIQ + w_a * LAION-aesthetic + w_s * smoothness_proxy

* MUSIQ comes from the same pyiqa checkpoint ``image_quality`` uses, so the
  proxy is exactly the technical-quality metric WorldArena reports.
* LAION-aesthetic is the same CLIP-ViT-L/14 + linear head as
  ``WorldArena.aesthetic_quality.get_aesthetic_model``.
* The smoothness proxy is ``1 / (mean Farneback-flow magnitude std)``: a cheap
  stand-in for ``1 / mean_EPE`` that does NOT require SEA-RAFT, so this script
  stays runnable on a single mid-range GPU without the full evaluator stack.

The script intentionally does NOT compute photometric_smoothness exactly: the
goal is to *select* among seeds, not to publish a score. Evaluation always
goes through ``evaluate.py`` afterwards.

Usage::

    python -m myscript.improve.seed_select \\
        --candidates seeds_root/ \\
        --pattern '{episode}_seed*.mp4' \\
        --output best/ \\
        --config config/config.yaml

Or with an explicit manifest JSON:

    python -m myscript.improve.seed_select \\
        --manifest seeds.json \\
        --output best/ \\
        --config config/config.yaml

manifest format (list of groups):
    [{"episode": "ep_0", "candidates": ["a.mp4", "b.mp4", "c.mp4"]}, ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from myscript.improve import io_utils  # noqa: E402


# ----------------------------------------------------------------------------
# Lightweight scorers
# ----------------------------------------------------------------------------


class _MusiqScorer:
    """Per-frame MUSIQ; mean over frames matches imaging_quality.py."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        import torch
        from pyiqa.archs.musiq_arch import MUSIQ

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = MUSIQ(pretrained_model_path=ckpt).to(self.device).eval()

    def __call__(self, frames: List[np.ndarray]) -> float:
        torch = self.torch
        scores = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with torch.no_grad():
                s = self.model(t)
            scores.append(float(s))
        return float(np.mean(scores)) / 100.0  # match imaging_quality normalization


class _AestheticScorer:
    """LAION aesthetic linear head over CLIP-ViT-L/14 image features."""

    def __init__(self, clip_ckpt: str, head_ckpt: str, device: str = "cuda"):
        import clip
        import torch
        import torch.nn as nn

        from WorldArena.utils import clip_transform  # noqa

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load(clip_ckpt, device=self.device)
        head = nn.Linear(768, 1)
        head.load_state_dict(torch.load(head_ckpt, weights_only=False))
        self.head = head.to(self.device).eval()
        self.tx = clip_transform(224)

    def __call__(self, frames: List[np.ndarray]) -> float:
        torch = self.torch
        # frames are BGR numpy uint8 H W 3; build a CHW tensor batch.
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        batch = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).contiguous()  # N 3 H W
        batch = self.tx(batch).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = self.head(feats).squeeze(-1)
        return float(scores.mean()) / 10.0  # match aesthetic_quality normalization


def smoothness_proxy(frames: List[np.ndarray]) -> float:
    """1 / mean(std-per-frame of Farneback flow magnitude).

    Higher = smoother flow field across the clip. Cheap stand-in for the
    SEA-RAFT EPE that ``photometric_smoothness`` reports.
    """
    if len(frames) < 2:
        return 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    stds = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        stds.append(float(mag.std()))
    mean_std = float(np.mean(stds))
    return 1.0 / (mean_std + 1e-3)


# ----------------------------------------------------------------------------
# Group discovery
# ----------------------------------------------------------------------------


def _discover_groups(root: Path, pattern: str) -> Dict[str, List[Path]]:
    """Group mp4 files under ``root`` using ``pattern``.

    ``pattern`` is a glob-with-placeholders string of the form
    ``"{episode}_seed*.mp4"``. ``{episode}`` is substituted with ``*`` for the
    glob and used for grouping.
    """
    if "{episode}" not in pattern:
        raise ValueError("pattern must contain '{episode}'")
    glob_pat = pattern.replace("{episode}", "*")
    groups: Dict[str, List[Path]] = defaultdict(list)
    # Replace {episode} with a regex capture for grouping.
    import re
    rx = re.compile("^" + re.escape(pattern).replace(r"\{episode\}", r"(?P<episode>.+?)").replace(r"\*", r".*") + "$")
    for p in sorted(root.rglob(glob_pat)):
        m = rx.match(p.name)
        if m:
            groups[m.group("episode")].append(p)
    return dict(groups)


def _load_manifest(path: Path) -> Dict[str, List[Path]]:
    data = json.loads(Path(path).read_text())
    groups: Dict[str, List[Path]] = {}
    for entry in data:
        ep = entry["episode"]
        groups[ep] = [Path(c) for c in entry["candidates"]]
    return groups


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _load_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def _resolve_ckpts(cfg: dict, args: argparse.Namespace) -> dict:
    ckpt = cfg.get("ckpt", {})
    musiq = args.ckpt_musiq or ckpt.get("image_quality", {}).get("musiq")
    aes_clip = args.ckpt_clip or ckpt.get("aesthetic_quality", {}).get("clip")
    aes_head = args.ckpt_aes or ckpt.get("aesthetic_quality", {}).get("aesthetic_head")
    return {"musiq": musiq, "aes_clip": aes_clip, "aes_head": aes_head}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-2 seed selection.")
    parser.add_argument("--candidates", type=Path,
                        help="Directory of candidate mp4s to be discovered with --pattern.")
    parser.add_argument("--pattern", type=str, default="{episode}_seed*.mp4",
                        help="Glob with {episode} placeholder for grouping.")
    parser.add_argument("--manifest", type=Path,
                        help="Optional explicit manifest JSON; supersedes --candidates/--pattern.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination directory for the selected best mp4 per episode.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--config", type=Path, default=None,
                        help="config.yaml to source ckpt paths from.")
    parser.add_argument("--ckpt-musiq", type=str, default=None)
    parser.add_argument("--ckpt-clip", type=str, default=None)
    parser.add_argument("--ckpt-aes", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w-musiq", type=float, default=1.0)
    parser.add_argument("--w-aesthetic", type=float, default=1.0)
    parser.add_argument("--w-smoothness", type=float, default=0.5)
    parser.add_argument("--copy", action="store_true",
                        help="Physically copy the best mp4 into --output (otherwise symlink).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="If >0 and a candidate has more frames, evenly sub-sample to "
                             "speed up scoring. Selection alone, never the final output.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    ckpts = _resolve_ckpts(cfg, args)

    if args.manifest is not None:
        groups = _load_manifest(args.manifest)
    elif args.candidates is not None:
        groups = _discover_groups(args.candidates, args.pattern)
    else:
        raise SystemExit("Either --manifest or --candidates is required.")

    if not groups:
        raise SystemExit("No candidate groups found.")

    # Lazy-init scorers: skip if weight is zero so users can run with just MUSIQ.
    musiq = aesthetic = None
    if args.w_musiq > 0.0:
        if not ckpts["musiq"]:
            raise SystemExit("--w-musiq > 0 requires --ckpt-musiq or a config.yaml with image_quality.musiq")
        musiq = _MusiqScorer(ckpts["musiq"], device=args.device)
    if args.w_aesthetic > 0.0:
        if not (ckpts["aes_clip"] and ckpts["aes_head"]):
            raise SystemExit("--w-aesthetic > 0 requires aesthetic_quality.clip + aesthetic_head")
        aesthetic = _AestheticScorer(ckpts["aes_clip"], ckpts["aes_head"], device=args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    report: List[dict] = []
    for episode, paths in groups.items():
        per_cand: List[dict] = []
        for p in paths:
            frames, _ = io_utils.read_frames(p)
            if args.max_frames > 0 and len(frames) > args.max_frames:
                idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
                frames_eval = [frames[i] for i in idx]
            else:
                frames_eval = frames
            s_musiq = musiq(frames_eval) if musiq is not None else 0.0
            s_aes = aesthetic(frames_eval) if aesthetic is not None else 0.0
            s_smooth = smoothness_proxy(frames_eval) if args.w_smoothness > 0.0 else 0.0
            score = (args.w_musiq * s_musiq
                     + args.w_aesthetic * s_aes
                     + args.w_smoothness * s_smooth)
            per_cand.append({
                "path": str(p),
                "musiq": s_musiq,
                "aesthetic": s_aes,
                "smoothness": s_smooth,
                "score": score,
            })

        per_cand.sort(key=lambda d: d["score"], reverse=True)
        best = per_cand[0]
        out_path = args.output / Path(best["path"]).name
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        if args.copy:
            shutil.copy2(best["path"], out_path)
        else:
            out_path.symlink_to(Path(best["path"]).resolve())
        print(f"[{episode}] {len(per_cand)} cands; best={Path(best['path']).name} score={best['score']:.4f}")
        report.append({"episode": episode, "best": best, "candidates": per_cand})

    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"[report] saved -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
