"""Aggregate per-episode video-quality metrics by RoboTwin 2.0 task.

Takes an aggregated_results.csv (one row per video/episode, columns = metrics)
and joins it against the episode -> task mapping produced by
`classify_test_dataset_tasks.py`, then prints / writes the per-task mean of
every numeric metric.

Typical usage:
    python aggregate_by_task.py \
        video_quality/csv_results/wan/aggregated_results.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = (
    REPO_ROOT
    / "video_quality"
    / "myscript"
    / "task_classification_output"
    / "episode_task_mapping.csv"
)

_EPISODE_RE = re.compile(r"(episode\d+)", re.IGNORECASE)


def _video_id_to_episode(video_id: str) -> str | None:
    """Extract ``episodeK`` from a Video_ID string like ``fixed_scene_task_episode123``."""
    if not isinstance(video_id, str):
        return None
    m = _EPISODE_RE.search(video_id)
    return m.group(1).lower() if m else None


def load_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["episode_id", "predicted_task"])
    df["episode_id"] = df["episode_id"].str.lower()
    return df


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Video_ID" not in df.columns:
        sys.exit(f"[error] {path} has no 'Video_ID' column")
    df["episode_id"] = df["Video_ID"].map(_video_id_to_episode)
    missing = df["episode_id"].isna().sum()
    if missing:
        print(f"[warn] {missing} rows have no episodeK token in Video_ID", file=sys.stderr)
    return df


def _to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal Markdown table writer (avoids the optional ``tabulate`` dep)."""

    def fmt(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def aggregate(results: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(mapping, on="episode_id", how="left")
    unmapped = merged["predicted_task"].isna().sum()
    if unmapped:
        print(
            f"[warn] {unmapped} episodes in results have no task mapping; dropped from task stats",
            file=sys.stderr,
        )
    merged = merged.dropna(subset=["predicted_task"])

    numeric_cols = [
        c
        for c in merged.columns
        if c not in {"Model_Name", "Video_ID", "episode_id", "predicted_task"}
        and pd.api.types.is_numeric_dtype(merged[c])
    ]

    grouped = merged.groupby("predicted_task", sort=True)

    means = grouped[numeric_cols].mean(numeric_only=True).round(6)
    counts = grouped.size().rename("n_episodes")

    out = pd.concat([counts, means], axis=1)

    valid_counts = grouped[numeric_cols].count()
    valid_counts.columns = [f"{c}__n" for c in valid_counts.columns]

    overall_mean = merged[numeric_cols].mean(numeric_only=True).round(6)
    overall_row = pd.DataFrame([[len(merged), *overall_mean.tolist()]],
                               index=pd.Index(["__ALL__"], name="predicted_task"),
                               columns=out.columns)
    out = pd.concat([out, overall_row])

    return out, valid_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, help="aggregated_results.csv from a model run")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help=f"episode -> task mapping CSV (default: {DEFAULT_MAPPING})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results_csv parent>/per_task_means.csv)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown output path (pretty-printed table).",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        nargs="?",
        const=Path("__default__"),
        default=None,
        help=(
            "Also write a TSV file for easy paste into Excel. "
            "Pass a path to customize; omit the value to use <results_csv parent>/per_task_means.tsv."
        ),
    )
    parser.add_argument(
        "--sort",
        default="predicted_task",
        help="Column to sort the output table by (default: predicted_task).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending when sorting by metric, ascending by task name).",
    )
    args = parser.parse_args()

    if not args.results_csv.exists():
        sys.exit(f"[error] results csv not found: {args.results_csv}")
    if not args.mapping.exists():
        sys.exit(f"[error] mapping csv not found: {args.mapping}")

    mapping = load_mapping(args.mapping)
    results = load_results(args.results_csv)

    table, _valid_counts = aggregate(results, mapping)

    if args.sort in table.columns:
        ascending = args.ascending or args.sort == "predicted_task"
        all_row = table.loc[["__ALL__"]]
        body = table.drop(index="__ALL__")
        if args.sort == "predicted_task":
            body = body.sort_index(ascending=ascending)
        else:
            body = body.sort_values(args.sort, ascending=ascending)
        table = pd.concat([body, all_row])

    output_csv = args.output or (args.results_csv.parent / "per_task_means.csv")
    table.to_csv(output_csv)
    print(f"[ok] wrote {output_csv}")

    if args.tsv is not None:
        tsv_path = args.tsv
        if tsv_path == Path("__default__"):
            tsv_path = args.results_csv.parent / "per_task_means.tsv"
        table.to_csv(tsv_path, sep="\t", float_format="%.6f")
        print(f"[ok] wrote {tsv_path}")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda v: f"{v:.4f}",
    ):
        print()
        print(table)

    if args.markdown:
        md = _to_markdown(table.reset_index(), floatfmt=".4f")
        args.markdown.write_text(md + "\n", encoding="utf-8")
        print(f"[ok] wrote {args.markdown}")


if __name__ == "__main__":
    main()
