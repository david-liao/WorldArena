"""Pilot data sampler for the VLM filtering experiment.

Reads a RoboTwin-style ``*.jsonl`` (each line carrying ``data_type``,
``root``, ``camera_video_info`` and ``lang``), filters ``data_type==clean``
rows, samples ``--num_total`` rows at random with a fixed seed, splits them
into ``--num_chunks`` even chunks, and for each chunk produces:

  - ``<flat_root>/<run_name>/chunk{i}/<safe_name>.mp4``   (symlink to source)
  - ``<summary_root>/<run_name>/chunk{i}.json``           (summary.json fed
    to ``video_quality/VLM_judge.py``)

The script then prints the four shell commands to launch
``run_VLM_judge_multi_gpu.sh`` on each node, one chunk per node.

Sampling is per-line (NOT per-uuid), so the same physical video may appear
under different ``lang`` instructions.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, help="Source jsonl path")
    p.add_argument("--run_name", default="robotwin_0507_clean_pilot",
                   help="Used as subdir name + MODEL_NAME prefix")
    p.add_argument("--num_total", type=int, default=1024,
                   help="Total rows to sample (across all chunks)")
    p.add_argument("--num_chunks", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_type", default="clean",
                   help="Filter rows by data_type before sampling")
    p.add_argument("--video_quality_dir", default=None,
                   help="Path to video_quality/ (default: auto-detect from script location)")
    p.add_argument("--summary_root", default=None,
                   help="Override summary output dir (default: <video_quality>/summary)")
    p.add_argument("--flat_root", default=None,
                   help="Override flat-symlink root (default: <video_quality>/data_pool)")
    p.add_argument("--output_root", default=None,
                   help="Override VLM output root passed to run_VLM_judge_multi_gpu.sh "
                        "(default: <video_quality>/output_VLM)")
    p.add_argument("--config_path", default=None,
                   help="Override config path for VLM judge "
                        "(default: <video_quality>/config/config.yaml)")
    p.add_argument("--max_videos_per_chunk", type=int, default=0,
                   help="Pass through to run_VLM_judge_multi_gpu.sh (0 = all chunk rows)")
    p.add_argument("--ngpus", type=int, default=0,
                   help="NGPUS arg for run_VLM_judge_multi_gpu.sh on each node "
                        "(0 = auto-detect)")
    p.add_argument("--dry_run", action="store_true",
                   help="Plan only, do not write summaries / create symlinks")
    return p.parse_args()


def safe_name(uuid: str, episode_index: int, lang_idx: int) -> str:
    """Turn ``task/robot_config/episodeN`` into a unique flat filename.

    ``lang_idx`` differentiates rows that share the same uuid but have
    different ``lang`` (per-line sampling can pick the same uuid twice).
    """
    base = uuid.replace("/", "__")
    return f"{base}__row{lang_idx}.mp4"


def load_clean_rows(jsonl_path: str, data_type: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {line_no + 1}: JSON decode error: {e}")
                continue
            if obj.get("data_type") != data_type:
                continue
            obj["__line_no__"] = line_no
            rows.append(obj)
    return rows


def resolve_video_path(row: Dict) -> str:
    root = row.get("root", "")
    cam = row.get("camera_video_info", {}).get("head_camera", {})
    rel = cam.get("video", "")
    if not root or not rel:
        return ""
    return os.path.join(root, rel)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    vq_dir = Path(args.video_quality_dir).resolve() if args.video_quality_dir else script_dir.parent.parent
    if not (vq_dir / "VLM_judge.py").exists():
        raise SystemExit(
            f"video_quality dir not found at {vq_dir}; pass --video_quality_dir explicitly")

    summary_root = Path(args.summary_root).resolve() if args.summary_root else vq_dir / "summary"
    flat_root = Path(args.flat_root).resolve() if args.flat_root else vq_dir / "data_pool"
    output_root = Path(args.output_root).resolve() if args.output_root else vq_dir / "output_VLM"
    config_path = Path(args.config_path).resolve() if args.config_path else vq_dir / "config" / "config.yaml"

    print(f">>> video_quality dir : {vq_dir}")
    print(f">>> summary root      : {summary_root}")
    print(f">>> flat symlink root : {flat_root}")
    print(f">>> output root       : {output_root}")
    print(f">>> config path       : {config_path}")
    print()

    rows = load_clean_rows(args.jsonl, args.data_type)
    print(f">>> {len(rows)} rows with data_type={args.data_type!r} found")
    if len(rows) < args.num_total:
        raise SystemExit(
            f"Not enough rows: needed {args.num_total}, only {len(rows)} available")

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.num_total)
    print(f">>> Randomly sampled {len(sampled)} rows (seed={args.seed})")

    chunk_size = args.num_total // args.num_chunks
    if chunk_size * args.num_chunks != args.num_total:
        raise SystemExit(
            f"num_total ({args.num_total}) must be divisible by num_chunks ({args.num_chunks})")
    print(f">>> Each chunk has {chunk_size} rows")

    summary_dir = summary_root / args.run_name
    flat_run_dir = flat_root / args.run_name
    if not args.dry_run:
        summary_dir.mkdir(parents=True, exist_ok=True)
        flat_run_dir.mkdir(parents=True, exist_ok=True)

    missing_videos: List[str] = []
    chunks_meta: List[Dict] = []

    for c in range(args.num_chunks):
        chunk_rows = sampled[c * chunk_size:(c + 1) * chunk_size]
        chunk_dir = flat_run_dir / f"chunk{c}"
        chunk_summary = summary_dir / f"chunk{c}.json"
        if not args.dry_run:
            chunk_dir.mkdir(parents=True, exist_ok=True)

        summary_items: List[Dict] = []
        for idx, row in enumerate(chunk_rows):
            src = resolve_video_path(row)
            if not src or not os.path.exists(src):
                missing_videos.append(src or f"<row line {row.get('__line_no__')}>")
                continue
            fname = safe_name(row["uuid"], row.get("episode_index", -1), idx)
            link = chunk_dir / fname
            if not args.dry_run and not link.exists():
                os.symlink(src, link)
            summary_items.append({
                "gt_path": f"/pool/{fname}",
                "prompt": row.get("lang", ""),
                "uuid": row["uuid"],
                "task_name": row.get("task_name"),
                "robot_config": row.get("robot_config"),
                "data_type": row.get("data_type"),
                "orig_video_path": src,
                "src_line_no": row["__line_no__"],
            })

        if not args.dry_run:
            with open(chunk_summary, "w", encoding="utf-8") as f:
                json.dump(summary_items, f, ensure_ascii=False, indent=2)

        chunks_meta.append({
            "chunk_id": c,
            "size": len(summary_items),
            "summary": str(chunk_summary),
            "video_dir": str(chunk_dir),
            "model_name": f"{args.run_name}_chunk{c}",
        })
        print(f"[chunk{c}] {len(summary_items)} videos -> {chunk_dir}")

    if missing_videos:
        print(f"\n[WARN] {len(missing_videos)} sampled rows have missing videos "
              f"(skipped). First 3:")
        for m in missing_videos[:3]:
            print(f"  - {m}")

    print("\n" + "=" * 80)
    print("Per-node commands (run one chunk per node):")
    print("=" * 80)
    for m in chunks_meta:
        cmd = (
            f"cd {vq_dir} && "
            f"bash run_VLM_judge_multi_gpu.sh "
            f"\"{m['model_name']}\" "
            f"\"{m['video_dir']}\" "
            f"\"{m['summary']}\" "
            f"\"{config_path}\" "
            f"all "
            f"{args.max_videos_per_chunk} "
            f"{args.ngpus}"
        )
        print(f"\n# Node {m['chunk_id']} ({m['size']} videos):")
        print(cmd)

    print("\n" + "=" * 80)
    print("After all 4 nodes finish, merge results with:")
    print("=" * 80)
    merged_path = output_root / f"{args.run_name}_merged.json"
    shard_paths = [
        str(output_root / m["model_name"] / f"{m['model_name']}_summary_val_all_intern.json")
        for m in chunks_meta
    ]
    merge_snippet = (
        "python3 - <<'PY'\n"
        "import json\n"
        f"paths = {shard_paths!r}\n"
        "merged = []\n"
        "for p in paths:\n"
        "    with open(p) as f:\n"
        "        merged.extend(json.load(f))\n"
        f"out = {str(merged_path)!r}\n"
        "with open(out, 'w') as f:\n"
        "    json.dump(merged, f, ensure_ascii=False, indent=2)\n"
        "print(f'Merged {len(merged)} entries -> {out}')\n"
        "PY"
    )
    print(merge_snippet)

    plan_path = summary_dir / "plan.json"
    if not args.dry_run:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_name": args.run_name,
                "jsonl": os.path.abspath(args.jsonl),
                "num_total": args.num_total,
                "num_chunks": args.num_chunks,
                "seed": args.seed,
                "data_type": args.data_type,
                "video_quality_dir": str(vq_dir),
                "summary_root": str(summary_root),
                "flat_root": str(flat_root),
                "output_root": str(output_root),
                "config_path": str(config_path),
                "chunks": chunks_meta,
                "missing_videos": missing_videos,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n>>> Plan saved to {plan_path}")


if __name__ == "__main__":
    main()
