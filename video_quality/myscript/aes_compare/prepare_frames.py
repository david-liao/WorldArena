"""Extract frames from each mp4 into the 4-layer dir layout that
WorldArena.build_full_info_json expects:

    <out_root>/<testset>/generated_dataset/task_0/<ep_stem>/gid_0/video/00001.jpg

Run:
    python prepare_frames.py --src <SRC> --out <OUT_ROOT> [--fps 0] [--workers 16]

--fps 0 keeps every frame (no -vf). Otherwise resamples to the given fps.
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SUBSETS = ["clean_testset", "light_testset"]


def extract_one(args):
    mp4_path, out_dir, fps = args
    os.makedirs(out_dir, exist_ok=True)
    # Idempotent: if frames already exist (>=1), skip
    if any(fn.endswith(".jpg") for fn in os.listdir(out_dir)):
        return mp4_path, "skip"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(mp4_path),
    ]
    if fps and fps > 0:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-q:v", "2", os.path.join(out_dir, "%05d.jpg")]
    try:
        subprocess.run(cmd, check=True)
        return mp4_path, "ok"
    except subprocess.CalledProcessError as exc:
        return mp4_path, f"fail: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="root dir containing clean_testset/ and light_testset/")
    parser.add_argument("--out", required=True,
                        help="output root for the 4-layer dataset trees")
    parser.add_argument("--fps", type=int, default=0,
                        help="resample fps (0 = keep all frames)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS,
                        help="subset directory names under --src")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    jobs = []
    for subset in args.subsets:
        src_dir = src_root / subset
        if not src_dir.is_dir():
            print(f"[warn] missing {src_dir}", file=sys.stderr)
            continue
        mp4s = sorted(src_dir.glob("*.mp4"))
        print(f"[{subset}] {len(mp4s)} mp4 files")
        for mp4 in mp4s:
            stem = mp4.stem
            out_dir = out_root / subset / "generated_dataset" / "task_0" / f"ep_{stem}" / "gid_0" / "video"
            jobs.append((str(mp4), str(out_dir), args.fps))

    print(f"total jobs: {len(jobs)}; workers: {args.workers}; fps: {args.fps or 'native'}")
    fail = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (mp4, status) in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            if status not in ("ok", "skip"):
                fail += 1
                print(f"[{i}/{len(jobs)}] {status}  <-  {mp4}")
            elif i % 50 == 0:
                print(f"[{i}/{len(jobs)}] progress")
    print(f"done. failures: {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
