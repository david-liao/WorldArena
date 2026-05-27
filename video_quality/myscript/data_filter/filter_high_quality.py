"""Filter VLM-scored samples into a high-quality training manifest.

Inputs (typically auto-resolved from a plan.json produced by
``prepare_robotwin_chunks.py``):
  - merged VLM result JSON (list of ``{"video", "metrics", "error", ...}``)
  - per-chunk summary JSON files (carry ``orig_video_path``, ``uuid``,
    ``task_name``, ``src_line_no`` and ``prompt`` for each video filename)

Selection:
  Each sample is kept iff every dimension's normalized score is strictly
  greater than its threshold. Defaults match the ``> {0.7746, 0.9728, 0.8612}``
  rule, which on a 1-5 integer scale collapses to:
      Interaction_Quality   >= 4
      Perspectivity         == 5
      Instruction_Following == 5

Outputs:
  - JSONL training manifest: one row per kept sample, with
    ``video_path / instruction / uuid / task_name / src_line_no / vlm_scores``.
  - Stats JSON: pass counts per dimension + AND, score histograms, per-task
    keep / total counts, and threshold echo. Same content also pretty-printed
    to stdout.

Usage:
  python3 filter_high_quality.py \
      --plan video_quality/summary/robotwin_0507_clean_pilot/plan.json
  # or fully explicit:
  python3 filter_high_quality.py \
      --merged path/to/merged.json \
      --summary_glob 'video_quality/summary/<run>/chunk*.json' \
      --out_jsonl video_quality/filtered/<run>_high.jsonl
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DIMS: Tuple[str, ...] = ("Interaction_Quality", "Perspectivity", "Instruction_Following")
DEFAULT_THRESHOLDS = (0.7746, 0.9728, 0.8612)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default=None,
                   help="Path to plan.json from prepare_robotwin_chunks.py "
                        "(auto-derives --merged / --summary_glob / --out_jsonl)")
    p.add_argument("--merged", default=None,
                   help="Override merged VLM result JSON path "
                        "(default: <output_root>/<run_name>/merged.json from plan)")
    p.add_argument("--summary_glob", default=None,
                   help="Glob for per-chunk summary JSON files "
                        "(default: <summary_root>/<run_name>/chunk*.json from plan)")
    p.add_argument("--out_jsonl", default=None,
                   help="Output JSONL manifest "
                        "(default: <video_quality>/filtered/<run_name>_high.jsonl from plan)")
    p.add_argument("--out_stats", default=None,
                   help="Output stats JSON (default: alongside --out_jsonl as <name>_stats.json)")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                   help="Comma-separated normalized thresholds for "
                        f"{DIMS}. Default: {DEFAULT_THRESHOLDS}")
    p.add_argument("--strict", action="store_true", default=True,
                   help="Use strict > comparison (default true). Use --geq for >=.")
    p.add_argument("--geq", dest="strict", action="store_false",
                   help="Use >= comparison instead of strict >")
    return p.parse_args()


def load_plan(plan_path: str) -> Dict:
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """Resolve (merged_path, summary_glob, out_jsonl, out_stats)."""
    merged = args.merged
    summary_glob = args.summary_glob
    out_jsonl = args.out_jsonl

    if args.plan:
        plan = load_plan(args.plan)
        run_name = plan["run_name"]
        out_root = plan["output_root"]
        sum_root = plan["summary_root"]
        vq_dir = plan["video_quality_dir"]
        merged = merged or os.path.join(out_root, run_name, "merged.json")
        summary_glob = summary_glob or os.path.join(sum_root, run_name, "chunk*.json")
        out_jsonl = out_jsonl or os.path.join(vq_dir, "filtered", f"{run_name}_high.jsonl")

    if not merged or not summary_glob or not out_jsonl:
        raise SystemExit("--merged, --summary_glob, --out_jsonl required when --plan absent")

    out_stats = args.out_stats
    if not out_stats:
        out_stats = os.path.splitext(out_jsonl)[0] + "_stats.json"
    return merged, summary_glob, out_jsonl, out_stats


def load_summary_index(summary_glob: str) -> Dict[str, Dict]:
    """Map basename(gt_path) -> summary item, across all matching summary files."""
    index: Dict[str, Dict] = {}
    files = sorted(glob.glob(summary_glob))
    if not files:
        raise SystemExit(f"No summary files matched: {summary_glob}")
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for it in json.load(f):
                key = os.path.basename(it.get("gt_path", ""))
                if key:
                    index[key] = it
    return index


def get_normalized(item: Dict, dim: str) -> Optional[float]:
    val = item.get("metrics", {}).get(dim, {}) or {}
    s = val.get("score_normalized")
    if s is None:
        s = val.get("score")
        if isinstance(s, (int, float)):
            s = s / 5.0
    return s if isinstance(s, (int, float)) else None


def main() -> None:
    args = parse_args()
    merged_path, summary_glob, out_jsonl, out_stats = resolve_paths(args)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    if len(thresholds) != len(DIMS):
        raise SystemExit(f"--thresholds must have {len(DIMS)} comma-separated values")
    thr_map = dict(zip(DIMS, thresholds))

    print(f">>> merged       : {merged_path}")
    print(f">>> summary glob : {summary_glob}")
    print(f">>> out jsonl    : {out_jsonl}")
    print(f">>> out stats    : {out_stats}")
    print(f">>> thresholds   : "
          + ", ".join(f"{d}{'>' if args.strict else '>='}{thr_map[d]}" for d in DIMS))

    summary_idx = load_summary_index(summary_glob)
    with open(merged_path, "r", encoding="utf-8") as f:
        merged: List[Dict] = json.load(f)

    n_total = len(merged)
    n_err = sum(1 for r in merged if r.get("error"))
    n_eval = n_total - n_err

    pass_each: Dict[str, int] = {d: 0 for d in DIMS}
    histograms: Dict[str, Counter] = {d: Counter() for d in DIMS}
    per_task_total: Counter = Counter()
    per_task_keep: Counter = Counter()
    per_robot_keep: Counter = Counter()
    missing_summary: List[str] = []

    cmp = (lambda x, y: x > y) if args.strict else (lambda x, y: x >= y)

    Path(os.path.dirname(out_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    n_kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for r in merged:
            if r.get("error"):
                continue
            scores = {d: get_normalized(r, d) for d in DIMS}
            valid = all(isinstance(scores[d], (int, float)) for d in DIMS)
            if not valid:
                continue

            for d in DIMS:
                histograms[d][int(round(scores[d] * 5))] += 1
                if cmp(scores[d], thr_map[d]):
                    pass_each[d] += 1

            video_name = r.get("video", "")
            meta = summary_idx.get(video_name, {})
            task = meta.get("task_name") or "<unknown>"
            per_task_total[task] += 1

            if not all(cmp(scores[d], thr_map[d]) for d in DIMS):
                continue

            if not meta:
                missing_summary.append(video_name)

            row = {
                "video_path": meta.get("orig_video_path"),
                "instruction": meta.get("prompt"),
                "uuid": meta.get("uuid"),
                "task_name": meta.get("task_name"),
                "robot_config": meta.get("robot_config"),
                "data_type": meta.get("data_type"),
                "src_line_no": meta.get("src_line_no"),
                "vlm_scores": {d: scores[d] for d in DIMS},
                "vlm_score_avg": round(sum(scores.values()) / len(scores), 4),
                "video": video_name,
                "chunk_id": r.get("chunk_id"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
            per_task_keep[task] += 1
            if meta.get("robot_config"):
                per_robot_keep[meta["robot_config"]] += 1

    stats = {
        "merged_path": merged_path,
        "summary_glob": summary_glob,
        "out_jsonl": out_jsonl,
        "thresholds": thr_map,
        "comparison": ">" if args.strict else ">=",
        "totals": {
            "merged": n_total,
            "errors": n_err,
            "evaluated": n_eval,
            "kept_AND": n_kept,
            "keep_rate_pct": round(100 * n_kept / n_eval, 2) if n_eval else 0.0,
        },
        "pass_per_dim": {d: pass_each[d] for d in DIMS},
        "histograms_raw_score": {d: dict(sorted(histograms[d].items())) for d in DIMS},
        "per_task": [
            {"task": t, "kept": per_task_keep.get(t, 0), "total": per_task_total[t]}
            for t in sorted(per_task_total, key=lambda x: -per_task_keep.get(x, 0))
        ],
        "per_robot_keep": dict(per_robot_keep),
        "missing_summary_count": len(missing_summary),
        "missing_summary_examples": missing_summary[:5],
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Pretty print
    print(f"\n=== summary ===")
    print(f"  merged total     : {n_total}")
    print(f"  errors           : {n_err}")
    print(f"  pass per dim     : "
          + " | ".join(f"{d}={pass_each[d]}" for d in DIMS))
    print(f"  pass AND all     : {n_kept}  "
          f"({100 * n_kept / n_eval:.2f}%)")
    if missing_summary:
        print(f"  WARN: {len(missing_summary)} kept rows had no summary metadata "
              f"(orig_video_path/uuid will be null)")
    print(f"\n=== top-10 tasks by kept count ===")
    print(f"  {'task':<28} kept / total   keep%")
    for row in stats["per_task"][:10]:
        rate = (100 * row["kept"] / row["total"]) if row["total"] else 0
        print(f"  {row['task']:<28} {row['kept']:4d} / {row['total']:<5d}  {rate:5.1f}%")

    print(f"\nWrote {n_kept} rows -> {out_jsonl}")
    print(f"Stats              -> {out_stats}")


if __name__ == "__main__":
    main()
