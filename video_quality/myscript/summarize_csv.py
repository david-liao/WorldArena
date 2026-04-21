"""Compute per-metric averages from aggregated_results.csv.

Output a single Tab-separated line (values * 100, 2-decimal precision) in the
order shown in the summary sheet, so it can be pasted directly into Excel.

Usage:
    python summarize_csv.py path/to/aggregated_results.csv
"""

import argparse
import csv
import sys
from statistics import mean

# Output column order (matches the Excel summary template)
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


def compute_averages(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"ERROR: empty CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    averages = {}
    for col in ORDERED_COLUMNS:
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


def format_cell(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize aggregated_results.csv into a TSV line")
    parser.add_argument("csv_path", help="Path to aggregated_results.csv")
    parser.add_argument("--with-header", action="store_true", help="Also print a TSV header line")
    parser.add_argument("-o", "--output", help="Write TSV to file instead of stdout (recommended for Excel)")
    args = parser.parse_args()

    averages = compute_averages(args.csv_path)

    lines = []
    if args.with_header:
        lines.append("\t".join(ORDERED_COLUMNS))
    lines.append("\t".join(format_cell(averages[c]) for c in ORDERED_COLUMNS))
    content = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote TSV to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(content)


if __name__ == "__main__":
    main()
