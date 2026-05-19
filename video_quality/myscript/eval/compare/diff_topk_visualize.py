"""挑选 trajectory_accuracy 差值最大 / 最小的 episode, 将其 SAM3 轨迹视频做并排对比.

数据目录自动从 config/config_<MODEL>.yaml 读取 (data.gt_path / val_base);
若找不到则回退到旧约定 video_quality/compare_data/.

输入:
    output/<MODEL>/generated_results.json
    <DATA_BASE>/gt_dataset/<TASK>/<EP>/gripper_detection/video.mp4               (GT 轨迹视频)
    <DATA_BASE>/generated_dataset/<TASK>/<EP>/1/gripper_detection/video.mp4      (Pred 轨迹视频)

输出 (默认写入 <DATA_BASE>/diff_visualize/<MODEL>/):
    {low,high}_rank<NN>_<ep>_norm<score>.mp4   每个 episode 一个左右拼接视频
    overview_{low,high}.mp4                    上下堆叠总览
    selection.json                              挑选记录

依赖: ffmpeg / ffprobe.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _episode_from_path(video_path: str):
    """从 .../<task>/<ep>/1/video 提取 (task, ep)."""
    parts = Path(video_path).parts
    if "generated_dataset" not in parts:
        return None, None
    i = parts.index("generated_dataset")
    return parts[i + 1], parts[i + 2]


def _resolve_data_base(base: Path, model_name: str, override: Optional[str] = None) -> Path:
    """决定 SAM3 轨迹视频所在的 data_base.

    优先级:
        1. --data_base 显式覆盖
        2. config/config_<MODEL>.yaml -> data.gt_path 的父目录
        3. 默认 <base>/compare_data (兼容旧实验)
    """
    if override:
        return Path(override).resolve()

    if yaml is not None:
        config_path = base / "config" / f"config_{model_name}.yaml"
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text()) or {}
            gt_path = (cfg.get("data") or {}).get("gt_path")
            if gt_path:
                return Path(gt_path).resolve().parent

    return base / "compare_data"


def _build_pair(gt_mp4: Path, pred_mp4: Path, label_gt: str, label_pred: str, out_mp4: Path):
    """左右拼接两段视频, 各自顶部加文字."""
    filt = (
        f"[0:v]drawtext=fontfile={FONT}:text='{label_gt}':"
        f"fontcolor=white:fontsize=22:box=1:boxcolor=black@0.55:boxborderw=6:x=10:y=10[g];"
        f"[1:v]drawtext=fontfile={FONT}:text='{label_pred}':"
        f"fontcolor=white:fontsize=22:box=1:boxcolor=black@0.55:boxborderw=6:x=10:y=10[p];"
        f"[g][p]hstack=inputs=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(gt_mp4), "-i", str(pred_mp4),
        "-filter_complex", filt,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _stack_overview(pair_mp4s, out_mp4: Path):
    """把若干并排视频上下堆叠成一个总览. 全部按最短长度对齐."""
    inputs = []
    for p in pair_mp4s:
        inputs.extend(["-i", str(p)])
    n = len(pair_mp4s)
    in_streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{in_streams}vstack=inputs={n}[out]"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filt,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True,
                    help="实验名 (与 eval_compare_any.sh 传入的 NAME 一致)")
    ap.add_argument("--base_dir", default=".",
                    help="video_quality 根目录")
    ap.add_argument("--data_base", default=None,
                    help="SAM3 轨迹视频根目录 (含 gt_dataset/generated_dataset). "
                         "不指定则自动从 config/config_<MODEL>.yaml 读, 再回退到 compare_data/")
    ap.add_argument("--top_k", type=int, default=5,
                    help="挑选最低分的样本数")
    ap.add_argument("--also_high", action="store_true",
                    help="同时挑选最高分样本作为对照组")
    ap.add_argument("--include_zero", action="store_true",
                    help="包含 NDTW=0 (检测完全失败) 的样本; 默认排除")
    ap.add_argument("--label_gt", default="GT",
                    help="对比视频上 GT 一侧的文字标签 (默认 GT)")
    ap.add_argument("--label_pred", default="Pred",
                    help="对比视频上 Pred 一侧的文字标签 (默认 Pred)")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    result_json = base / "output" / args.model_name / "generated_results.json"
    if not result_json.exists():
        sys.exit(f"ERROR: result json not found: {result_json}")

    data_base = _resolve_data_base(base, args.model_name, args.data_base)
    if not data_base.exists():
        sys.exit(f"ERROR: data_base 不存在: {data_base}\n"
                 f"       请检查 config/config_{args.model_name}.yaml 中的 data.gt_path 或显式传 --data_base")
    print(f">>> Using data_base: {data_base}")

    data = json.loads(result_json.read_text())
    items = data["trajectory_accuracy"][1]
    items_with_meta = []
    for it in items:
        task, ep = _episode_from_path(it["video_path"])
        if not task:
            continue
        if not args.include_zero and float(it["video_results"]) == 0.0:
            continue
        items_with_meta.append({
            "task": task,
            "ep": ep,
            "ndtw": float(it["video_results"]),
            "norm": float(it["video_results_normalized"]),
            "video_path": it["video_path"],
        })

    items_sorted = sorted(items_with_meta, key=lambda x: x["norm"])
    low = items_sorted[:args.top_k]
    high = items_sorted[-args.top_k:][::-1]  # 从高到低

    out_dir = data_base / "diff_visualize" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    excl_note = "包含" if args.include_zero else "排除"
    print(f">>> Output dir: {out_dir}")
    print(f">>> 总样本 ({excl_note} NDTW=0): {len(items_with_meta)} / {len(items)}")

    def _process(group, tag):
        pair_paths = []
        for rank, it in enumerate(group, start=1):
            ep = it["ep"]
            gt_mp4   = data_base / "gt_dataset"        / it["task"] / ep / "gripper_detection" / "video.mp4"
            pred_mp4 = data_base / "generated_dataset" / it["task"] / ep / "1" / "gripper_detection" / "video.mp4"
            if not gt_mp4.exists() or not pred_mp4.exists():
                print(f"  [skip] {ep}: video missing  ({gt_mp4 if not gt_mp4.exists() else pred_mp4})")
                continue
            out_mp4 = out_dir / f"{tag}_rank{rank:02d}_{ep}_norm{it['norm']:.3f}.mp4"
            label_gt = f"{args.label_gt}  {ep}  norm={it['norm']:.3f}  ndtw={it['ndtw']:.2f}"
            label_pred = f"{args.label_pred}  {ep}"
            print(f"  [{tag} #{rank}] {ep}  norm={it['norm']:.3f}  ndtw={it['ndtw']:.2f}")
            _build_pair(gt_mp4, pred_mp4, label_gt, label_pred, out_mp4)
            pair_paths.append(out_mp4)

        if pair_paths:
            overview = out_dir / f"overview_{tag}.mp4"
            print(f">>> Stacking {len(pair_paths)} pairs -> {overview}")
            _stack_overview(pair_paths, overview)
            return overview, pair_paths
        return None, []

    print(f"\n=== LOW {args.top_k} (轨迹差异最大) ===")
    low_overview, low_pairs = _process(low, "low")

    high_overview = None
    high_pairs = []
    if args.also_high:
        print(f"\n=== HIGH {args.top_k} (轨迹最相似) ===")
        high_overview, high_pairs = _process(high, "high")

    selection = {
        "model_name": args.model_name,
        "low":  [{k: v for k, v in it.items() if k != "video_path"} for it in low],
        "high": [{k: v for k, v in it.items() if k != "video_path"} for it in high] if args.also_high else [],
        "low_overview":  str(low_overview)  if low_overview  else None,
        "high_overview": str(high_overview) if high_overview else None,
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    print(f"\n>>> Selection saved to {out_dir / 'selection.json'}")
    print(">>> Done.")


if __name__ == "__main__":
    main()
