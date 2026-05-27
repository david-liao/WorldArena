"""Prepare full-scale chunked dataset for distributed VLM filtering.

Reads a RoboTwin-style ``*.jsonl``, optionally filters by ``data_type``,
splits into fixed-size chunks (default 1000 rows each), and for each chunk
writes:
  - ``<summary_root>/<run_name>/chunk{NNN}.json``: summary fed to VLM_judge
  - ``<flat_root>/<run_name>/chunk{NNN}/<safe_name>.mp4`` (symlinks; lazy by
    default — the per-node script creates them on demand)

A ``plan.json`` summarizing all chunks is written alongside the summaries.
``run_chunks_on_node.sh`` consumes this plan to dispatch chunks to nodes.

Symlinks are NOT created by default since 100k+ symlinks on shared FS can
take a while and any aborted re-prep would leave orphans. Pass
``--materialize_links_now`` to create them all upfront.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--run_name", required=True,
                   help="Used as subdir name + parent of model_name path")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--data_type", default="",
                   help="Optional data_type filter (e.g. 'clean'); empty = no filter")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle rows before chunking (default off, keep natural order)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--materialize_links_now", action="store_true",
                   help="Create all symlinks now (default: lazy, by node script)")
    p.add_argument("--video_quality_dir", default=None)
    p.add_argument("--summary_root", default=None)
    p.add_argument("--flat_root", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument("--config_path", default=None)
    return p.parse_args()


def safe_name(uuid: str, src_line_no: int) -> str:
    """Globally unique flat filename: <task>__<robot_cfg>__<episode>__line<N>.mp4

    Using src_line_no (which is unique across the whole jsonl) guarantees no
    collisions even when the same uuid recurs with different ``lang`` values.
    """
    return f"{uuid.replace('/', '__')}__line{src_line_no}.mp4"


def resolve_video_path(row: Dict) -> str:
    root = row.get("root", "")
    cam = row.get("camera_video_info", {}).get("head_camera", {}) or {}
    rel = cam.get("video", "")
    if not root or not rel:
        return ""
    return os.path.join(root, rel)


def load_filtered_rows(jsonl_path: str, data_type: str) -> List[Dict]:
    rows: List[Dict] = []
    keep_all = not data_type
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
            if (not keep_all) and obj.get("data_type") != data_type:
                continue
            obj["__line_no__"] = line_no
            rows.append(obj)
    return rows


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    vq_dir = Path(args.video_quality_dir).resolve() if args.video_quality_dir \
        else script_dir.parent.parent
    if not (vq_dir / "VLM_judge.py").exists():
        raise SystemExit(f"video_quality dir not found at {vq_dir}; "
                         "pass --video_quality_dir explicitly")

    summary_root = Path(args.summary_root).resolve() if args.summary_root \
        else vq_dir / "summary"
    flat_root = Path(args.flat_root).resolve() if args.flat_root \
        else vq_dir / "data_pool"
    output_root = Path(args.output_root).resolve() if args.output_root \
        else vq_dir / "output_VLM"
    config_path = Path(args.config_path).resolve() if args.config_path \
        else vq_dir / "config" / "config.yaml"

    print(f">>> video_quality dir : {vq_dir}")
    print(f">>> summary root      : {summary_root}")
    print(f">>> flat symlink root : {flat_root}")
    print(f">>> output root       : {output_root}")
    print(f">>> config path       : {config_path}")
    print(f">>> chunk size        : {args.chunk_size}")
    print(f">>> data_type filter  : {args.data_type or '(none)'}")
    print(f">>> shuffle           : {args.shuffle} (seed={args.seed})")
    print(f">>> materialize links : {args.materialize_links_now}")
    print()

    rows = load_filtered_rows(args.jsonl, args.data_type)
    print(f">>> Loaded {len(rows)} rows after data_type filter")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        print(f">>> Shuffled with seed={args.seed}")

    total = len(rows)
    num_chunks = math.ceil(total / args.chunk_size)
    pad = max(3, len(str(num_chunks - 1)))
    print(f">>> Will produce {num_chunks} chunks "
          f"(last chunk size = {total - (num_chunks - 1) * args.chunk_size})")

    summary_dir = summary_root / args.run_name
    flat_run_dir = flat_root / args.run_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    flat_run_dir.mkdir(parents=True, exist_ok=True)

    chunks_meta: List[Dict] = []
    missing_videos: List[str] = []
    n_links_created = 0

    for c in range(num_chunks):
        chunk_rows = rows[c * args.chunk_size:(c + 1) * args.chunk_size]
        leaf = f"chunk{str(c).zfill(pad)}"
        chunk_dir = flat_run_dir / leaf
        chunk_summary = summary_dir / f"{leaf}.json"

        summary_items: List[Dict] = []
        for row in chunk_rows:
            src = resolve_video_path(row)
            if not src or not os.path.exists(src):
                missing_videos.append(src or f"<row line {row.get('__line_no__')}>")
                continue
            fname = safe_name(row["uuid"], row["__line_no__"])
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

        with open(chunk_summary, "w", encoding="utf-8") as f:
            json.dump(summary_items, f, ensure_ascii=False, indent=2)

        if args.materialize_links_now:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for it in summary_items:
                fname = os.path.basename(it["gt_path"])
                link = chunk_dir / fname
                if not link.exists():
                    try:
                        os.symlink(it["orig_video_path"], link)
                        n_links_created += 1
                    except OSError as e:
                        print(f"[WARN] symlink failed for {fname}: {e}")

        chunks_meta.append({
            "chunk_id": c,
            "size": len(summary_items),
            "summary": str(chunk_summary),
            "video_dir": str(chunk_dir),
            "model_name": f"{args.run_name}/{leaf}",
            "leaf_name": leaf,
        })
        if (c + 1) % 10 == 0 or c == num_chunks - 1:
            print(f"  prepared {c + 1}/{num_chunks} chunks "
                  f"(last: {leaf}, {len(summary_items)} items)")

    if missing_videos:
        print(f"\n[WARN] {len(missing_videos)} rows have missing videos (skipped).")
        for m in missing_videos[:3]:
            print(f"  - {m}")

    plan = {
        "run_name": args.run_name,
        "jsonl": os.path.abspath(args.jsonl),
        "data_type": args.data_type or None,
        "total_rows": total,
        "chunk_size": args.chunk_size,
        "num_chunks": num_chunks,
        "shuffle": args.shuffle,
        "seed": args.seed if args.shuffle else None,
        "video_quality_dir": str(vq_dir),
        "summary_root": str(summary_root),
        "flat_root": str(flat_root),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "links_materialized": args.materialize_links_now,
        "chunks": chunks_meta,
        "missing_videos_count": len(missing_videos),
        "missing_videos_examples": missing_videos[:5],
    }
    plan_path = summary_dir / "plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(f"\n>>> Plan saved to {plan_path}")
    if args.materialize_links_now:
        print(f">>> Created {n_links_created} symlinks across all chunks")
    else:
        print(">>> Symlinks deferred — run_chunks_on_node.sh will create them lazily")

    print("\n" + "=" * 80)
    print("Next: dispatch with run_chunks_on_node.sh")
    print("=" * 80)
    print(f"# On each node (i in 0..NUM_NODES-1):")
    print(f"bash {script_dir}/run_chunks_on_node.sh \\")
    print(f"  --plan {plan_path} \\")
    print(f"  --node_id <NODE_ID> --num_nodes <NUM_NODES>")
    print()
    print("# Or pin specific chunks on a node:")
    print(f"bash {script_dir}/run_chunks_on_node.sh \\")
    print(f"  --plan {plan_path} \\")
    print(f"  --chunks \"0-9,15,42-50\"")


if __name__ == "__main__":
    main()
