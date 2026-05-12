"""Helper: print a per-episode delta between two seed_select reports.

Used by ``run_compare.sh`` to summarize the smoothness-proxy delta between an
``origin/`` directory and the post-processed output. Works on any pair of
``seed_select.py`` ``--report`` JSONs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--postproc", type=Path, required=True)
    parser.add_argument("--key", type=str, default="smoothness",
                        help="Which scorer key to diff (smoothness | musiq | aesthetic | score).")
    args = parser.parse_args()

    o = json.loads(Path(args.origin).read_text())
    p = json.loads(Path(args.postproc).read_text())
    om = {e["episode"]: e["best"][args.key] for e in o}
    pm = {e["episode"]: e["best"][args.key] for e in p}

    print(f"{'episode':<14} {'origin':>10} {'postproc':>10} {'delta':>10}")
    print("-" * 48)
    deltas = []
    for k in sorted(om):
        d = pm.get(k, 0.0) - om[k]
        deltas.append(d)
        print(f"{k:<14} {om[k]:>10.4f} {pm.get(k, 0.0):>10.4f} {d:>+10.4f}")
    print("-" * 48)
    print(f"{'mean':<14} {statistics.mean(om.values()):>10.4f} "
          f"{statistics.mean(pm.values()):>10.4f} "
          f"{statistics.mean(deltas):>+10.4f}")
    wins = sum(1 for d in deltas if d > 0)
    print(f"wins (postproc better on key={args.key}): {wins}/{len(deltas)}")


if __name__ == "__main__":
    main()
