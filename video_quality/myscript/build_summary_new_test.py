"""Rewrite `gt_path` in a WorldArena summary.json so it points to the real GT
videos shipped in `datasets/new_test_dataset/`.

Input summary (from new_test_dataset) uses placeholder gt_path such as
`/data/fixed_scene_task/a/b/c/episodeK.mp4`. This script:

1. Replaces `gt_path` with
   `<DATASET_ROOT>/gt_video/<task_group>/episodeK.mp4`
2. Optionally filters out items whose real GT video is missing (default: keep).
3. Writes the result to the requested output path.

Usage::

    python myscript/build_summary_new_test.py \
        --src-summary /mydir/code/WorldArena/datasets/new_test_dataset/summary.json \
        --dataset-root /mydir/code/WorldArena/datasets/new_test_dataset \
        --out /mydir/code/WorldArena/video_quality/summary_new_test.json \
        --task-group fixed_scene_task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src-summary",
        type=Path,
        default=Path("/mydir/code/WorldArena/datasets/new_test_dataset/summary.json"),
        help="Source summary.json (from new_test_dataset)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mydir/code/WorldArena/datasets/new_test_dataset"),
        help="Root directory that contains gt_video/<task_group>/*.mp4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mydir/code/WorldArena/video_quality/summary_new_test.json"),
        help="Where to write the rewritten summary.json",
    )
    parser.add_argument(
        "--task-group",
        type=str,
        default="fixed_scene_task",
        help="Single group name under gt_video/ (matches new_test_dataset layout)",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop entries whose GT video does not exist on disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.src_summary.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gt_dir = args.dataset_root / "gt_video" / args.task_group
    if not gt_dir.is_dir():
        raise SystemExit(f"gt_video dir not found: {gt_dir}")

    rewritten, missing = [], []
    for item in data:
        orig_gt_path = item.get("gt_path", "")
        episode_stem = Path(orig_gt_path).stem or ""
        real_gt_path = gt_dir / f"{episode_stem}.mp4"

        if not real_gt_path.exists():
            missing.append(episode_stem)
            if args.drop_missing:
                continue

        new_item = dict(item)
        new_item["gt_path"] = str(real_gt_path)
        rewritten.append(new_item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    print(f">>> wrote {len(rewritten)} items to {args.out}")
    print(f">>> missing GT videos: {len(missing)}")
    if missing[:5]:
        print(f"    sample missing: {missing[:5]}")


if __name__ == "__main__":
    main()
