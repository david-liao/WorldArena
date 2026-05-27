"""评估 SAM3 检测对抽帧间隔 (stride) 的鲁棒性.

对同一段视频按多个 stride 抽帧, 测:
  - detection rate    最终 traj 非 [-1,-1] 帧的比例
  - mean valid run    连续成功检测段的平均长度 (轨迹被切碎程度)

输出:
  <out>/
    <video_name>/
      frames/                                   原始抽帧 (stride=1)
      stride{k}/
        video/                                  按 stride 选出的帧 (软链)
        gripper_detection/video.mp4             SAM3 标注视频
        traj/traj.npy                           轨迹 (经兜底/插值/平滑后处理)
    stats.csv                                   所有视频所有 stride 的统计
    det_rate_vs_stride.png                      折线图

注意:
  - process_video_with_tracking 内部硬编码 prompts=["robot arm", "end effector", "gripper"].
    若评估的目标非机械臂, 需要先去 processing/detection_tracking.py 改 prompts.
  - 大 stride 下 traj.npy 的帧数会变少 (~ 原帧数/stride).
    "detection rate" 是在 stride 后的帧子集上算的, 不是原始时间轴.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# 让本脚本能 import processing/detection_tracking 和 WorldArena/trajectory_accuracy
HERE = Path(__file__).resolve()
PROJ = HERE.parents[3]
sys.path.insert(0, str(PROJ / "processing"))
sys.path.insert(0, str(PROJ))

from detection_tracking import GripperDetector, process_video_with_tracking  # noqa: E402
from WorldArena.trajectory_accuracy import (  # noqa: E402
    traj_interpo_fill,
    select_farthest_traj_index,
    NDTW,
)


def extract_frames(mp4_path: Path, out_dir: Path, limit: int = 0) -> int:
    """抽帧到 out_dir/frame_NNNNN.jpg, 返回实际写出帧数."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return len(existing)
    cap = cv2.VideoCapture(str(mp4_path))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"frame_{n:05d}.jpg"), frame)
        n += 1
        if limit and n >= limit:
            break
    cap.release()
    return n


def make_stride_dir(frames_dir: Path, stride: int, target_dir: Path) -> int:
    """按 stride 从 frames_dir 软链子集到 target_dir, 返回选出的帧数."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in list(target_dir.glob("*.jpg")):
        f.unlink()
    frame_files = sorted(frames_dir.glob("*.jpg"))
    selected = frame_files[::stride]
    for i, src in enumerate(selected):
        dst = target_dir / f"frame_{i:05d}.jpg"
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    return len(selected)


def _mean_consecutive_run(mask: np.ndarray) -> float:
    runs = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def compute_stats(traj_path: Path) -> dict:
    """traj.npy shape: (N, 2, 2)  -- N 帧, [left/right], (x_norm, y_norm)."""
    traj = np.load(traj_path)
    left_valid = ~np.all(traj[:, 0] == -1, axis=1)
    right_valid = ~np.all(traj[:, 1] == -1, axis=1)
    return {
        "n_frames": int(traj.shape[0]),
        "det_rate_left":  float(left_valid.mean()),
        "det_rate_right": float(right_valid.mean()),
        "det_rate_any":   float((left_valid | right_valid).mean()),
        "mean_run_left":  _mean_consecutive_run(left_valid),
        "mean_run_right": _mean_consecutive_run(right_valid),
    }


def compute_ndtw_vs_baseline(traj_k_path: Path, traj_1_path: Path, stride: int) -> float:
    """对比 stride=k 的轨迹 vs stride=1 按 k 子采样后的 baseline.

    思路:
      - 把 stride=1 的完整轨迹 traj_1 按 [::stride] 子采样, 得到与 traj_k 相同时间分辨率的 "真值"
      - 调用 trajectory_accuracy.NDTW 计算两条轨迹的相似度
    Returns:
        NDTW score (越高越像), 失败/无意义时返回 0.0
        stride=1 时返回 -1.0 表示 "不适用 (self-compare)"
    """
    if stride <= 1:
        return -1.0  # sentinel
    traj_k = np.load(traj_k_path).astype("float32")
    traj_1 = np.load(traj_1_path).astype("float32")
    baseline = traj_1[::stride].astype("float32")

    n = min(len(traj_k), len(baseline))
    if n < 2:
        return 0.0
    traj_k = traj_k[:n].copy()
    baseline = baseline[:n].copy()

    traj_k_filled, invalid_k = traj_interpo_fill(traj_k)
    baseline_filled, invalid_b = traj_interpo_fill(baseline)

    max_idx = select_farthest_traj_index(baseline_filled, invalid_b)
    try:
        ds = NDTW(traj_k_filled, baseline_filled, invalid_k, invalid_b, max_idx)
        if not np.isfinite(ds):
            return 0.0
        return float(ds)
    except ZeroDivisionError:
        return 0.0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--videos", nargs="+", required=True, help="一个或多个 mp4 路径")
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 5, 8, 13, 21],
                    help="测试的 stride 列表 (默认 1 2 3 5 8 13 21)")
    ap.add_argument("--output_dir", required=True, help="输出根目录")
    ap.add_argument("--config_path", default=str(PROJ / "config" / "config.yaml"),
                    help="主 config.yaml, 用于读 ckpt.sam3_model_ckpt")
    ap.add_argument("--limit_frames", type=int, default=0,
                    help="每段视频最多用前 N 帧 (0 = 全部)")
    ap.add_argument("--force", action="store_true",
                    help="强制重跑 (忽略 frames/ 和 traj.npy 已有结果)")
    args = ap.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.config_path))
    sam3_path = cfg["ckpt"]["sam3_model_ckpt"]
    print(f">>> Loading SAM3 from {sam3_path}")
    detector = GripperDetector(model_path=sam3_path)

    rows = []
    # 记录每个视频 stride=1 的 traj 路径, 后续 stride 用作 baseline
    baseline_traj: dict[str, Path] = {}
    for vid in args.videos:
        vid_path = Path(vid).resolve()
        if not vid_path.exists():
            print(f"!! skip (not found): {vid_path}")
            continue
        vid_name = vid_path.stem
        vid_out = out_root / vid_name
        frames_dir = vid_out / "frames"

        if args.force or not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
            print(f">>> [{vid_name}] 抽帧"
                  + (f" (limit={args.limit_frames})" if args.limit_frames else ""))
            n_extracted = extract_frames(vid_path, frames_dir, args.limit_frames)
            print(f"    抽出 {n_extracted} 帧")
        else:
            n_extracted = len(list(frames_dir.glob("*.jpg")))
            print(f">>> [{vid_name}] 复用已抽帧 ({n_extracted})")

        for stride in args.strides:
            if stride < 1:
                continue
            stride_dir = vid_out / f"stride{stride}"
            video_dir = stride_dir / "video"
            traj_path = stride_dir / "traj" / "traj.npy"

            if not args.force and traj_path.exists():
                print(f"    stride={stride}: 复用已有 traj")
            else:
                n_used = make_stride_dir(frames_dir, stride, video_dir)
                if n_used < 2:
                    print(f"    stride={stride}: 帧数<2 跳过")
                    continue
                print(f"    stride={stride}: 用 {n_used} 帧, 跑 SAM3 ...")
                ok = process_video_with_tracking(
                    input_path=str(video_dir),
                    output_path=str(stride_dir),
                    detector=detector,
                    data_type="gt",                    # 直接写到 stride_dir/gripper_detection 与 stride_dir/traj
                    force_reprocess=args.force,
                )
                if not ok or not traj_path.exists():
                    print(f"      stride={stride} 失败, 跳过")
                    continue

            if stride == 1:
                baseline_traj[vid_name] = traj_path

            stats = compute_stats(traj_path)
            stats.update({"video": vid_name, "stride": stride})

            # 计算 NDTW vs baseline (stride=1 子采样)
            if vid_name in baseline_traj:
                stats["ndtw_vs_baseline"] = compute_ndtw_vs_baseline(
                    traj_path, baseline_traj[vid_name], stride
                )
            else:
                stats["ndtw_vs_baseline"] = float("nan")
            rows.append(stats)
            ndtw_str = (f"{stats['ndtw_vs_baseline']:.2f}"
                        if stride > 1 else "  -  ")
            print(f"      stride={stride}: det_any={stats['det_rate_any']:.3f}  "
                  f"run_L={stats['mean_run_left']:.1f}  "
                  f"run_R={stats['mean_run_right']:.1f}  "
                  f"ndtw_vs_base={ndtw_str}")

    if not rows:
        sys.exit("!! 没有得到任何统计, 检查输入路径与 SAM3 ckpt.")

    # === 写 CSV ===
    stats_csv = out_root / "stats.csv"
    metric_cols = ["n_frames", "det_rate_left", "det_rate_right", "det_rate_any",
                   "mean_run_left", "mean_run_right", "ndtw_vs_baseline"]
    with open(stats_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "stride"] + metric_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f">>> Stats: {stats_csv}")

    # === 出图 ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        agg = defaultdict(lambda: {"left": [], "right": [], "any": [],
                                   "run_left": [], "run_right": [], "ndtw": []})
        for r in rows:
            agg[r["stride"]]["left"].append(r["det_rate_left"])
            agg[r["stride"]]["right"].append(r["det_rate_right"])
            agg[r["stride"]]["any"].append(r["det_rate_any"])
            agg[r["stride"]]["run_left"].append(r["mean_run_left"])
            agg[r["stride"]]["run_right"].append(r["mean_run_right"])
            v = r.get("ndtw_vs_baseline", float("nan"))
            if isinstance(v, float) and np.isfinite(v) and v >= 0:
                agg[r["stride"]]["ndtw"].append(v)
        strides = sorted(agg.keys())
        n_vid = len(set(r["video"] for r in rows))

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        ax = axes[0]
        ax.plot(strides, [np.mean(agg[s]["left"])  for s in strides], "o-", label="left  hand")
        ax.plot(strides, [np.mean(agg[s]["right"]) for s in strides], "s-", label="right hand")
        ax.plot(strides, [np.mean(agg[s]["any"])   for s in strides], "^--", label="any (left|right)", color="gray")
        ax.set_xlabel("stride")
        ax.set_ylabel("detection rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log", base=2)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        ax.set_title(f"SAM3 detection rate vs stride  ({n_vid} videos)")

        ax = axes[1]
        ax.plot(strides, [np.mean(agg[s]["run_left"])  for s in strides], "o-", label="left  hand")
        ax.plot(strides, [np.mean(agg[s]["run_right"]) for s in strides], "s-", label="right hand")
        ax.set_xlabel("stride")
        ax.set_ylabel("mean consecutive valid-run (frames)")
        ax.set_xscale("log", base=2)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        ax.set_title(f"Trajectory fragmentation vs stride")

        ax = axes[2]
        ndtw_strides = [s for s in strides if s > 1 and agg[s]["ndtw"]]
        ndtw_means = [np.mean(agg[s]["ndtw"]) for s in ndtw_strides]
        ax.plot(ndtw_strides, ndtw_means, "o-", color="C2",
                label="NDTW vs stride=1 baseline")
        # 也画 EMPIRICAL_BOUND 上限作参考线 (用于 trajectory_accuracy normalize 的 40.85)
        ax.axhline(40.85, color="red", ls=":", alpha=0.7,
                   label="trajectory_accuracy NDTW upper bound = 40.85")
        ax.set_xlabel("stride")
        ax.set_ylabel("NDTW (higher = closer to baseline)")
        ax.set_xscale("log", base=2)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        ax.set_title(f"Trajectory similarity vs stride")

        fig.tight_layout()
        png_path = out_root / "det_rate_vs_stride.png"
        fig.savefig(png_path, dpi=120)
        print(f">>> Plot:  {png_path}")
    except Exception as e:
        print(f"!! plot skipped: {e}")

    # === 保存 selection ===
    (out_root / "summary.json").write_text(json.dumps({
        "videos": args.videos,
        "strides": args.strides,
        "n_records": len(rows),
        "limit_frames": args.limit_frames,
    }, indent=2, ensure_ascii=False))
    print(">>> Done.")


if __name__ == "__main__":
    main()
