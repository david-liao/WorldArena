"""Extract original jsonl rows that pass the VLM filter.

Reads:
  - ``--filtered`` (output of ``filter_high_quality.py``): each line carries
    ``src_line_no`` pointing back to the source jsonl.
  - ``--source_jsonl``: the original training pool (e.g. robotwin_0507_230k.jsonl).

Writes:
  - ``--out_jsonl``: a new jsonl, schema-identical to the source by default,
    with optional ``vlm_scores`` / ``vlm_score_avg`` fields appended for
    traceability.

Usage:
  python3 extract_filtered_jsonl.py \
      --filtered    video_quality/filtered/robotwin_0507_clean_pilot_high.jsonl \
      --source_jsonl RoboTwin2.0/robotwin_0507_230k.jsonl \
      --out_jsonl   video_quality/filtered/robotwin_0507_clean_pilot_high_source.jsonl

Or auto-derive source from a plan.json:

  python3 extract_filtered_jsonl.py \
      --plan video_quality/summary/robotwin_0507_clean_pilot/plan.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default=None,
                   help="plan.json from prepare_robotwin_chunks.py "
                        "(auto-fills --filtered / --source_jsonl / --out_jsonl)")
    p.add_argument("--filtered", default=None,
                   help="JSONL produced by filter_high_quality.py")
    p.add_argument("--source_jsonl", default=None,
                   help="Original jsonl to extract from")
    p.add_argument("--out_jsonl", default=None,
                   help="Output jsonl. Default: <filtered>_source.jsonl")
    p.add_argument("--annotate_scores", action="store_true", default=True,
                   help="Append vlm_scores / vlm_score_avg / vlm_video to each row "
                        "(default true; use --no_annotate_scores to disable)")
    p.add_argument("--no_annotate_scores", dest="annotate_scores",
                   action="store_false")
    p.add_argument("--sort_by", choices=["source", "filtered"], default="source",
                   help="'source' = ascending src_line_no (matches original order); "
                        "'filtered' = preserve filtered jsonl order")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    filtered = args.filtered
    source = args.source_jsonl
    out = args.out_jsonl

    if args.plan:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)
        run_name = plan["run_name"]
        vq_dir = plan["video_quality_dir"]
        filtered = filtered or os.path.join(vq_dir, "filtered", f"{run_name}_high.jsonl")
        source = source or plan["jsonl"]
        if not out:
            out = os.path.splitext(filtered)[0] + "_source.jsonl"

    if not filtered or not source:
        raise SystemExit("Need --filtered and --source_jsonl (or --plan)")
    if not out:
        out = os.path.splitext(filtered)[0] + "_source.jsonl"

    return filtered, source, out


def load_filtered_index(path: str) -> Dict[int, Dict]:
    """Map src_line_no -> filtered row (the score sidecar)."""
    idx: Dict[int, Dict] = {}
    dups: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            ln = row.get("src_line_no")
            if not isinstance(ln, int):
                continue
            if ln in idx:
                dups.append(ln)
                continue
            idx[ln] = row
    if dups:
        print(f"[WARN] {len(dups)} duplicate src_line_no in filtered jsonl "
              f"(kept first occurrence). Examples: {dups[:3]}")
    return idx


def main() -> None:
    args = parse_args()
    filtered_path, source_path, out_path = resolve_paths(args)
    print(f">>> filtered     : {filtered_path}")
    print(f">>> source jsonl : {source_path}")
    print(f">>> out jsonl    : {out_path}")
    print(f">>> annotate     : {args.annotate_scores}")
    print(f">>> sort_by      : {args.sort_by}")

    want = load_filtered_index(filtered_path)
    print(f">>> need {len(want)} rows from source")

    extracted: Dict[int, Dict] = {}
    with open(source_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f):
            if line_no not in want:
                continue
            obj = json.loads(raw)
            if args.annotate_scores:
                meta = want[line_no]
                obj["vlm_scores"] = meta.get("vlm_scores")
                obj["vlm_score_avg"] = meta.get("vlm_score_avg")
                obj["vlm_video"] = meta.get("video")
            extracted[line_no] = obj
            if len(extracted) == len(want):
                break

    missing = sorted(set(want) - set(extracted))
    if missing:
        print(f"[WARN] {len(missing)} src_line_no not found in source "
              f"(likely beyond EOF). First 5: {missing[:5]}")

    if args.sort_by == "source":
        ordered = [extracted[k] for k in sorted(extracted)]
    else:
        with open(filtered_path, "r", encoding="utf-8") as f:
            order = [json.loads(line)["src_line_no"] for line in f if line.strip()]
        ordered = [extracted[k] for k in order if k in extracted]

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for row in ordered:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(ordered)} rows -> {out_path}")
    src_size = os.path.getsize(source_path)
    out_size = os.path.getsize(out_path)
    print(f"Source size: {src_size / 1024 / 1024:.1f} MB  "
          f"Out size: {out_size / 1024 / 1024:.2f} MB  "
          f"(ratio {out_size / src_size:.4%})")


if __name__ == "__main__":
    main()
