"""Compute per-metric averages from aggregated_results.csv.

Output a single Tab-separated line (values * 100, 2-decimal precision) in the
order shown in the summary sheet, so it can be pasted directly into Excel.

Usage:
    python summarize_csv.py path/to/aggregated_results.csv
"""

import argparse
import csv
import os
import sys
from statistics import mean

# Compact 12-column layout (default, matches the original Excel summary template)
ORDERED_COLUMNS = [
    "Image Quality",
    "Aesthetic Quality",
    "Dynamic Degree",
    "Flow Score",
    "Motion Smoothness",
    "Subject Consistency",
    "Background Consistency",
    "Photometric Consistency",
    "Interaction Quality",
    "Perspectivity",
    "Instruction Following",
    "Action Following",
]

# Full 16-column layout grouped by category (matches the two-row Excel header
# shown in the shared screenshot). Order is: Visual Quality, Motion Quality,
# Content Consistency, Physics Adherence, 3D Accuracy, Controllability.
CATEGORY_GROUPS = [
    ("Visual Quality",       ["Image Quality", "Aesthetic Quality", "JEPA Similarity"]),
    ("Motion Quality",       ["Dynamic Degree", "Flow Score", "Motion Smoothness"]),
    ("Content Consistency",  ["Subject Consistency", "Background Consistency", "Photometric Consistency"]),
    ("Physics Adherence",    ["Interaction Quality", "Trajectory Accuracy"]),
    ("3D Accuracy",          ["Depth Accuracy", "Perspectivity"]),
    ("Controllability",      ["Instruction Following", "Semantic Alignment", "Action Following"]),
]
FULL_ORDERED_COLUMNS = [col for _, cols in CATEGORY_GROUPS for col in cols]


def compute_averages(csv_path, columns):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"ERROR: empty CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    averages = {}
    for col in columns:
        if col not in reader.fieldnames:
            averages[col] = None
            continue
        values = []
        for row in rows:
            v = row.get(col, "").strip()
            if not v:
                continue
            try:
                values.append(float(v))
            except ValueError:
                continue
        averages[col] = mean(values) if values else None
    return averages


def build_category_header(groups):
    """Build a TSV header row where each category name appears once in its
    first column and the continuation columns are empty, matching the merged-
    cell look in the screenshot."""
    cells = []
    for name, cols in groups:
        cells.append(name)
        cells.extend([""] * (len(cols) - 1))
    return cells


def format_cell(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize aggregated_results.csv into a TSV line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="Path to aggregated_results.csv")
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Output all 16 metrics grouped by category (Visual Quality / Motion Quality / "
            "Content Consistency / Physics Adherence / 3D Accuracy / Controllability), "
            "with a two-row header matching the Excel summary template. "
            "Implies --with-header."
        ),
    )
    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Also print a TSV header line (ignored when --full is used; full mode always writes headers).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Write TSV to this path. "
            "If omitted, defaults to <csv_dir>/summary.tsv (simple mode) or "
            "<csv_dir>/summary_full.tsv (--full mode), next to the input CSV. "
            "Pass '-' to skip writing and only print to stdout."
        ),
    )
    args = parser.parse_args()

    if args.full:
        columns = FULL_ORDERED_COLUMNS
        default_basename = "summary_full.tsv"
    else:
        columns = ORDERED_COLUMNS
        default_basename = "summary.tsv"

    averages = compute_averages(args.csv_path, columns)

    lines = []
    if args.full:
        lines.append("\t".join(build_category_header(CATEGORY_GROUPS)))
        lines.append("\t".join(columns))
    elif args.with_header:
        lines.append("\t".join(columns))
    lines.append("\t".join(format_cell(averages[c]) for c in columns))
    content = "\n".join(lines) + "\n"

    # Always print to stdout so pipelines / eyeballing still work.
    sys.stdout.write(content)

    # Decide where to save (default: sibling of the input CSV).
    if args.output == "-":
        output_path = None
    elif args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.csv_path)), default_basename)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote TSV to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
