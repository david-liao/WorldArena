"""Merge per-shard evaluation result JSONs into a single result file.

Each shard JSON has the structure:
  { "metric": [avg, [{"video_path":..., "video_results":..., ...}, ...]], ... }

The merge concatenates the video lists from all shards and recomputes the
average as the mean of per-video scores.

Usage:
  python merge_results.py --output merged.json shard0.json shard1.json ...
"""

import argparse
import json
import re
import sys
from pathlib import Path

_SHARD_RE = re.compile(r'/shard_\d+/')


def _normalize_video_path(vp):
    """Replace shard directory with generated_dataset so downstream tools are shard-agnostic."""
    return _SHARD_RE.sub('/generated_dataset/', vp)


def merge(shard_paths, output_path):
    merged = {}

    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for metric, payload in data.items():
            if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                continue
            video_list = payload[1]
            if not isinstance(video_list, list):
                continue
            for entry in video_list:
                if "video_path" in entry:
                    entry["video_path"] = _normalize_video_path(entry["video_path"])
            merged.setdefault(metric, []).extend(video_list)

    result = {}
    for metric, all_videos in merged.items():
        if not all_videos:
            result[metric] = [0.0, []]
            continue
        total = sum(v.get("video_results", 0.0) for v in all_videos)
        avg = total / len(all_videos)
        result[metric] = [avg, all_videos]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Merged {len(shard_paths)} shards -> {output_path}  ({len(next(iter(merged.values()), []))} videos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge shard evaluation results")
    parser.add_argument("shards", nargs="+", help="Paths to shard result JSONs")
    parser.add_argument("--output", "-o", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    missing = [p for p in args.shards if not Path(p).exists()]
    if missing:
        print(f"ERROR: missing shard files: {missing}", file=sys.stderr)
        sys.exit(1)

    merge(args.shards, args.output)
