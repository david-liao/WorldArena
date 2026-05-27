"""从 sam3_stride_robustness 产出的 stats.csv 只画两张子图:
   1) SAM3 detection rate vs stride
   3) NDTW vs stride=1 baseline

用法:
    python plot_det_and_ndtw.py \
        --stats sam3_stride_test/robotwin_8videos/stats.csv \
        --out   sam3_stride_test/robotwin_8videos/det_and_ndtw.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NDTW_BOUND = 40.85  # WorldArena._EMPIRICAL_BOUNDS["trajectory_accuracy"]


def load_rows(stats_csv: Path) -> list[dict]:
    rows = []
    with open(stats_csv) as f:
        for r in csv.DictReader(f):
            r["stride"] = int(r["stride"])
            for k in ("det_rate_left", "det_rate_right", "det_rate_any",
                     "ndtw_vs_baseline"):
                try:
                    r[k] = float(r[k]) if r[k] != "" else float("nan")
                except ValueError:
                    r[k] = float("nan")
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="stats.csv 路径")
    ap.add_argument("--out",   required=True, help="输出 png 路径")
    ap.add_argument("--title_suffix", default="", help="标题后缀, 可空")
    args = ap.parse_args()

    rows = load_rows(Path(args.stats))
    n_vid = len(set(r["video"] for r in rows))

    agg = defaultdict(lambda: {"left": [], "right": [], "any": [], "ndtw": []})
    for r in rows:
        s = r["stride"]
        agg[s]["left"].append(r["det_rate_left"])
        agg[s]["right"].append(r["det_rate_right"])
        agg[s]["any"].append(r["det_rate_any"])
        v = r["ndtw_vs_baseline"]
        if np.isfinite(v) and v >= 0:
            agg[s]["ndtw"].append(v)
    strides = sorted(agg.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(strides, [np.mean(agg[s]["left"])  for s in strides], "o-", label="left  hand")
    ax.plot(strides, [np.mean(agg[s]["right"]) for s in strides], "s-", label="right hand")
    ax.plot(strides, [np.mean(agg[s]["any"])   for s in strides], "^--",
            label="any (left|right)", color="gray")
    ax.set_xlabel("stride")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log", base=2)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    ax.set_title(f"(1) SAM3 detection rate vs stride  ({n_vid} videos){args.title_suffix}")

    ax = axes[1]
    ndtw_strides = [s for s in strides if s > 1 and agg[s]["ndtw"]]
    ndtw_means = [np.mean(agg[s]["ndtw"]) for s in ndtw_strides]
    ndtw_medians = [np.median(agg[s]["ndtw"]) for s in ndtw_strides]
    ax.plot(ndtw_strides, ndtw_means,   "o-",  color="C2", label="mean")
    ax.plot(ndtw_strides, ndtw_medians, "s--", color="C0", label="median")
    ax.axhline(NDTW_BOUND, color="red", ls=":", alpha=0.7,
               label=f"trajectory_accuracy upper bound = {NDTW_BOUND}")
    ax.set_xlabel("stride")
    ax.set_ylabel("NDTW vs stride=1 baseline  (higher = closer)")
    ax.set_xscale("log", base=2)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    ax.set_title(f"(2) Trajectory similarity vs stride{args.title_suffix}")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f">>> Saved: {out_path}")


if __name__ == "__main__":
    main()
