"""Progress monitor for chunked VLM filtering runs.

Reads a ``plan.json`` produced by ``prepare_full_chunks.py`` and reports:
  - per-chunk state (done / running / partial / missing / stale)
  - per-shard wall time from the literal ``Total time : Xs (Ymin)`` line in
    each ``shard_*.log`` (this is the only reliable timing source on NFS;
    file ctime/mtime are unreliable because of metadata caching and log
    truncation on rerun)
  - real per-chunk wall time = max over the 8 parallel shards
  - per-node breakdown + corrected ETA

Usage:
  python3 check_progress.py --plan summary/<run>/plan.json
  python3 check_progress.py --plan ... --num_nodes 12   # default 12
"""

import argparse
import glob
import json
import os
import re
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


TOTAL_RE = re.compile(r"Total time\s*:\s*([\d.]+)s\s*\(([\d.]+)\s*min\)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--num_nodes", type=int, default=12,
                   help="Used to reconstruct round-robin assignment for per-node view")
    p.add_argument("--running_threshold_sec", type=int, default=600,
                   help="Shards whose newest log mtime is within this window are 'running'; "
                        "older = stale. Default 10 min.")
    return p.parse_args()


def parse_shard_total(log_path: str) -> Optional[float]:
    """Return seconds reported by 'Total time : Xs (Ymin)' if present."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    m = TOTAL_RE.search(tail)
    return float(m.group(1)) if m else None


def shard_progress(log_path: str) -> Optional[Tuple[int, int]]:
    """Return (current, total) from latest tqdm 'XX/YY [' line."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    m = re.findall(r"(\d+)/(\d+) \[", tail)
    if not m:
        return None
    return int(m[-1][0]), int(m[-1][1])


def main() -> None:
    args = parse_args()
    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)
    out_root = plan["output_root"]
    num_chunks = plan["num_chunks"]
    nn = args.num_nodes

    now = time.time()
    states: List[Dict] = []
    shard_secs_done: List[float] = []

    for c in plan["chunks"]:
        cid = c["chunk_id"]
        leaf = c["leaf_name"]
        size = c["size"]
        cdir = os.path.join(out_root, c["model_name"])
        out_file = os.path.join(cdir, f"{leaf}_summary_val_all_intern.json")

        st: Dict = {"cid": cid, "leaf": leaf, "size": size, "node": cid % nn,
                    "model_name": c["model_name"], "kind": "missing",
                    "shard_secs": [], "log_files": []}
        if not os.path.isdir(cdir):
            states.append(st)
            continue

        log_files = sorted(glob.glob(os.path.join(cdir, "shard_*.log")))
        st["log_files"] = log_files
        finished_shards = 0
        for lp in log_files:
            secs = parse_shard_total(lp)
            if secs is not None:
                st["shard_secs"].append(secs)
                shard_secs_done.append(secs)
                finished_shards += 1
        st["finished_shards"] = finished_shards

        if os.path.isfile(out_file):
            try:
                n = len(json.load(open(out_file, "r", encoding="utf-8")))
                st["kind"] = "done" if n == size else f"partial({n}/{size})"
            except Exception:
                st["kind"] = "broken_out"
        else:
            if log_files:
                latest_mtime = max(os.path.getmtime(p) for p in log_files)
                age = now - latest_mtime
                st["kind"] = "running" if age < args.running_threshold_sec \
                    else f"stale({age / 60:.0f}m)"
            else:
                st["kind"] = "starting"
        states.append(st)

    done_walls = [max(s["shard_secs"]) for s in states
                  if s["kind"] == "done" and s["shard_secs"]]
    mean_wall = statistics.mean(done_walls) if done_walls else 5.5 * 3600

    n_done = sum(1 for s in states if s["kind"] == "done")
    n_running = sum(1 for s in states if s["kind"] == "running")
    n_missing = sum(1 for s in states if s["kind"] in ("missing", "starting"))
    n_stale = sum(1 for s in states if s["kind"].startswith("stale"))
    n_partial = sum(1 for s in states if s["kind"].startswith("partial"))

    print(f"=== overall ({plan['run_name']}) ===")
    print(f"  total   : {num_chunks}")
    print(f"  done    : {n_done}  ({n_done / num_chunks * 100:.1f}%)")
    print(f"  running : {n_running}")
    print(f"  missing : {n_missing}")
    print(f"  stale   : {n_stale}")
    print(f"  partial : {n_partial}")

    if done_walls:
        print(f"\n=== per-chunk wall time (max over parallel shards) ===")
        print(f"  range    : [{min(done_walls) / 60:.0f} – "
              f"{max(done_walls) / 60:.0f}] min")
        print(f"  median   : {statistics.median(done_walls) / 60:.1f} min")
        print(f"  mean     : {mean_wall / 60:.1f} min")
        print(f"  per video: {mean_wall / 1000:.1f} s "
              f"(1000 videos / chunk, 8-way parallel)")

    print(f"\n=== running chunks ===")
    running_remaining: Dict[int, float] = {}
    for s in states:
        if s["kind"] != "running":
            continue
        progress_fracs: List[float] = []
        for lp in s["log_files"]:
            sp = shard_progress(lp)
            if sp and sp[1] > 0:
                progress_fracs.append(sp[0] / sp[1])
        if progress_fracs:
            slowest = min(progress_fracs)
            avg_frac = sum(progress_fracs) / len(progress_fracs)
            elapsed_est = mean_wall * slowest
            remaining = max(0.0, mean_wall - elapsed_est)
            running_remaining[s["cid"]] = remaining
            print(f"  node {s['node']:>2d}  {s['leaf']}: slowest {slowest * 100:.0f}% "
                  f"(avg {avg_frac * 100:.0f}%), ~{elapsed_est / 3600:.1f}h elapsed, "
                  f"~{remaining / 3600:.1f}h to go")
        else:
            running_remaining[s["cid"]] = mean_wall
            print(f"  node {s['node']:>2d}  {s['leaf']}: no progress info")

    if n_stale:
        print(f"\n=== STALE chunks (no log update >{args.running_threshold_sec // 60} min) ===")
        for s in states:
            if s["kind"].startswith("stale"):
                print(f"  node {s['node']:>2d}  {s['leaf']}  state={s['kind']}")

    if n_partial:
        print(f"\n=== PARTIAL chunks (need rerun with --force) ===")
        for s in states:
            if s["kind"].startswith("partial"):
                print(f"  node {s['node']:>2d}  {s['leaf']}  {s['kind']}")

    print(f"\n=== per-node ETA (using mean {mean_wall / 60:.0f} min / chunk) ===")
    print(f"  node  done running missing  rem_chunks  est_done_at")
    max_eta_h = 0.0
    for nid in range(nn):
        chunks = [s for s in states if s["node"] == nid]
        nd = sum(1 for s in chunks if s["kind"] == "done")
        nr = sum(1 for s in chunks if s["kind"] == "running")
        nm = sum(1 for s in chunks if s["kind"] in ("missing", "starting"))
        rem_sec = nm * mean_wall
        for s in chunks:
            if s["kind"] == "running":
                rem_sec += running_remaining.get(s["cid"], mean_wall)
        eta_h = rem_sec / 3600
        max_eta_h = max(max_eta_h, eta_h)
        eta_dt = (datetime.now() + timedelta(seconds=rem_sec)).strftime("%m-%d %H:%M")
        print(f"  {nid:>4d}  {nd:>4d}  {nr:>6d}  {nm:>6d}  {nm + nr:>6d}      "
              f"{eta_dt}  (~{eta_h:.1f}h)")

    print(f"\n=== Wall-clock to completion ===")
    finish_dt = (datetime.now() + timedelta(hours=max_eta_h))
    print(f"  slowest node finishes ~{finish_dt.strftime('%m-%d %H:%M')}  "
          f"(~{max_eta_h:.1f}h, ~{max_eta_h / 24:.1f} days)")


if __name__ == "__main__":
    main()
