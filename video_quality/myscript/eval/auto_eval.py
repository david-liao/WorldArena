#!/usr/bin/env python3
"""自动化评测调度器（Auto Eval Scheduler）

功能:
    1. 持续扫描 ``--scan-path`` 下的所有 ``test_40`` 目录，当目录里
       视频数（默认 *.mp4）达到 ``--target-count``（默认 1000）即认为
       推理已完成，将其加入待测队列。
    2. 顺序执行 ``eval_full.sh`` 风格的评测流水线（标准指标 +
       VLM judge + JEPA + 聚合 + summarize）。
    3. 每次评测完成后，将 ``summary_full.tsv`` 中 16 项指标作为一行
       追加到指定 TSV 中，最后一列为该行有效值的平均。
    4. 已测试路径写入 state JSON，避免重复测试；失败路径下次会重试。

约定:
    - 必须在 ``video_quality/`` 目录所在主机运行（脚本会自动 cd 过去）。
    - 默认参数与 ``myscript/eval/origin_test/eval_full.sh`` 保持一致：
        * ``--summary-json   ./summary.json``
        * ``--config-path    ./config/config.yaml``
        * ``--metrics``      包含 image / aesthetic / background / subject /
                             dynamic_degree / flow / photometric / motion 八项
        * 默认运行 VLM judge，但 *不* 运行 JEPA（用 ``--run-jepa`` 启用）。
    - ``--max-videos N`` 用于调试，会作为 LIMIT/MAX_VIDEOS/MAX_SAMPLES 透传
      给三个评测子脚本。

用法（典型）:
    python myscript/eval/auto_eval.py \\
        --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/<run_dir> \\
        --results-dir /mnt/jackzou/WorldArena/results

按 Ctrl-C 退出。当前运行中的评测会让其完成，再退出循环。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


# 与 myscript/summarize_csv.py 的 FULL_ORDERED_COLUMNS 完全一致
METRIC_COLUMNS: List[str] = [
    "Image Quality",
    "Aesthetic Quality",
    "JEPA Similarity",
    "Dynamic Degree",
    "Flow Score",
    "Motion Smoothness",
    "Subject Consistency",
    "Background Consistency",
    "Photometric Consistency",
    "Interaction Quality",
    "Trajectory Accuracy",
    "Depth Accuracy",
    "Perspectivity",
    "Instruction Following",
    "Semantic Alignment",
    "Action Following",
]

# 默认值参考 myscript/eval/origin_test/eval_full.sh：
#   - SUMMARY_JSON=./summary.json
#   - CONFIG_PATH=./config/config.yaml
#   - 指标不含 trajectory_accuracy / semantic_alignment / depth_accuracy
#   - 默认不跑 JEPA（origin_test 流程没有该步骤）
DEFAULT_METRICS = (
    "image_quality,aesthetic_quality,background_consistency,subject_consistency,"
    "dynamic_degree,flow_score,photometric_smoothness,motion_smoothness"
)

DEFAULT_VIDEO_QUALITY_DIR = Path("/mydir/code/WorldArena/video_quality")
DEFAULT_GT_VIDEO_DIR = Path(
    "/mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task"
)
DEFAULT_SUMMARY_JSON = "./summary.json"
DEFAULT_CONFIG_PATH = "./config/config.yaml"

TSV_HEADER: List[str] = (
    ["timestamp", "model_name", "video_dir"] + METRIC_COLUMNS + ["Mean"]
)


# ------------------------------ 工具函数 -----------------------------------


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[auto_eval {ts}] {msg}", flush=True)


def count_videos(directory: Path, exts: Tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    cnt = 0
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in exts:
            cnt += 1
    return cnt


def find_test_40_dirs(scan_path: Path, target_dir_name: str) -> List[Path]:
    """查找 ``scan_path`` 下所有名为 ``target_dir_name`` 的目录。"""
    if not scan_path.is_dir():
        return []
    found: List[Path] = []
    for root, dirs, _ in os.walk(scan_path, followlinks=False):
        if target_dir_name in dirs:
            found.append(Path(root) / target_dir_name)
            # 不再向下挖（test_40 内部不会再有同名目录）
            dirs[:] = [d for d in dirs if d != target_dir_name]
    return sorted(found)


def derive_model_name(scan_path: Path, test_dir: Path, prefix: str = "") -> str:
    """从扫描根 + test_40 路径推导唯一 MODEL_NAME。"""
    rel = test_dir.relative_to(scan_path)
    scan_name = scan_path.name
    parts = list(rel.parts[:-1])  # 去掉末尾的 test_40 本身
    step_part = next((p for p in parts if p.startswith("step-")), None)
    if step_part:
        suffix = step_part
    elif parts:
        suffix = "_".join(parts)
    else:
        suffix = "root"
    base = f"{scan_name}__{suffix}" if scan_name else suffix
    return f"{prefix}{base}" if prefix else base


# ------------------------------ State 持久化 --------------------------------


class State:
    def __init__(self, path: Path):
        self.path = path
        self.completed: Dict[str, Dict[str, Any]] = {}
        self.failed: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.completed = data.get("completed", {}) or {}
            self.failed = data.get("failed", {}) or {}
        except Exception as e:
            log(f"WARN: 无法解析 state file {self.path}: {e}; 将从空状态开始。")
            self.completed = {}
            self.failed = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                {"completed": self.completed, "failed": self.failed},
                f,
                ensure_ascii=False,
                indent=2,
            )
        tmp.replace(self.path)

    def is_completed(self, video_dir: str) -> bool:
        return video_dir in self.completed

    def mark_completed(self, video_dir: str, info: Dict[str, Any]) -> None:
        self.completed[video_dir] = info
        self.failed.pop(video_dir, None)
        self.save()

    def mark_failed(self, video_dir: str, info: Dict[str, Any]) -> None:
        prev = self.failed.get(video_dir, {"attempts": 0})
        info["attempts"] = int(prev.get("attempts", 0)) + 1
        self.failed[video_dir] = info
        self.save()


# ------------------------------ TSV 输出 -----------------------------------


def ensure_tsv_header(tsv_path: Path) -> None:
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(TSV_HEADER) + "\n")


def parse_summary_full_tsv(summary_tsv: Path) -> Dict[str, Optional[float]]:
    """解析 ``summarize_csv.py --full --with-header`` 的输出。"""
    with summary_tsv.open("r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"summary_full.tsv 行数异常: {summary_tsv}")
    headers = lines[1].split("\t")
    values = lines[2].split("\t")
    if len(headers) != len(values):
        # 末尾可能少了空 cell；右侧 padding
        if len(values) < len(headers):
            values = values + [""] * (len(headers) - len(values))
        else:
            values = values[: len(headers)]
    out: Dict[str, Optional[float]] = {}
    for h, v in zip(headers, values):
        v = v.strip()
        if not v:
            out[h] = None
        else:
            try:
                out[h] = float(v)
            except ValueError:
                out[h] = None
    return out


def append_tsv_row(
    tsv_path: Path,
    timestamp: str,
    model_name: str,
    video_dir: str,
    metrics: Dict[str, Optional[float]],
) -> Tuple[List[str], Optional[float]]:
    ensure_tsv_header(tsv_path)
    cells: List[str] = [timestamp, model_name, video_dir]
    valid: List[float] = []
    for col in METRIC_COLUMNS:
        v = metrics.get(col)
        if v is None:
            cells.append("")
        else:
            cells.append(f"{v:.2f}")
            valid.append(v)
    avg: Optional[float] = mean(valid) if valid else None
    cells.append(f"{avg:.2f}" if avg is not None else "")
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(cells) + "\n")
    return cells, avg


# ------------------------------ 评测调用 -----------------------------------


def run_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    description: str,
    dry_run: bool = False,
) -> int:
    log(f"[{description}] cwd={cwd}")
    log(f"[{description}] CMD: {' '.join(shlex.quote(c) for c in cmd)}")
    log(f"[{description}] LOG: {log_path}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n========== {description} @ {datetime.now().isoformat()} ==========\n")
        logf.write(f"CWD: {cwd}\n")
        logf.write(f"CMD: {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def evaluate_one(
    video_dir: Path,
    model_name: str,
    *,
    video_quality_dir: Path,
    summary_json: str,
    config_path: str,
    metrics: str,
    gt_video_dir: Path,
    skip_vlm: bool,
    run_jepa: bool,
    max_videos: int,
    log_dir: Path,
    dry_run: bool,
) -> Tuple[bool, Optional[Path], str]:
    """对单个 ``test_40`` 目录跑完整评测。

    返回 (success, summary_full_tsv_path, message)。
    """
    log_path = log_dir / f"{model_name}.log"

    # 1. 标准指标 —— run_evaluation_multi_gpu.sh 的第 6 个位置参数是 LIMIT，
    # 默认 0 表示全量；为了能在第 7 位传 NGPUS 而不影响 LIMIT，仅在用户显式
    # 指定 max_videos 时才追加。
    eval_cmd = [
        "bash",
        "run_evaluation_multi_gpu.sh",
        model_name,
        str(video_dir),
        summary_json,
        config_path,
        metrics,
    ]
    if max_videos > 0:
        eval_cmd.append(str(max_videos))
    rc = run_cmd(
        eval_cmd,
        cwd=video_quality_dir,
        log_path=log_path,
        description="run_evaluation_multi_gpu",
        dry_run=dry_run,
    )
    if rc != 0:
        return False, None, f"run_evaluation_multi_gpu.sh 退出码 {rc}"

    # 2. VLM judge —— 第 6 个位置参数是 MAX_VIDEOS（0 = 全量）
    if not skip_vlm:
        vlm_cmd = [
            "bash",
            "run_VLM_judge_multi_gpu.sh",
            model_name,
            str(video_dir),
            summary_json,
            config_path,
            "all",
        ]
        if max_videos > 0:
            vlm_cmd.append(str(max_videos))
        rc = run_cmd(
            vlm_cmd,
            cwd=video_quality_dir,
            log_path=log_path,
            description="run_VLM_judge_multi_gpu",
            dry_run=dry_run,
        )
        if rc != 0:
            return False, None, f"run_VLM_judge_multi_gpu.sh 退出码 {rc}"
    else:
        log("[skip] VLM judge")

    # 3. JEPA —— 第 4 个位置参数是 MAX_SAMPLES（0 = 全量）。默认关闭，
    # 与 myscript/eval/origin_test/eval_full.sh 一致；用 --run-jepa 显式开启。
    if run_jepa:
        jepa_cmd = [
            "bash",
            "run_evaluation_JEPA.sh",
            model_name,
            str(video_dir),
            str(gt_video_dir),
        ]
        if max_videos > 0:
            jepa_cmd.append(str(max_videos))
        rc = run_cmd(
            jepa_cmd,
            cwd=video_quality_dir,
            log_path=log_path,
            description="run_evaluation_JEPA",
            dry_run=dry_run,
        )
        if rc != 0:
            return False, None, f"run_evaluation_JEPA.sh 退出码 {rc}"
    else:
        log("[skip] JEPA (使用 --run-jepa 启用)")

    # 4. 聚合
    rc = run_cmd(
        [
            "python",
            "csv_results/aggregate_results.py",
            "--model_name",
            model_name,
            "--base_dir",
            ".",
        ],
        cwd=video_quality_dir,
        log_path=log_path,
        description="aggregate_results",
        dry_run=dry_run,
    )
    if rc != 0:
        return False, None, f"aggregate_results.py 退出码 {rc}"

    # 5. summarize_csv（生成 csv_results/<model>/summary_full.tsv）
    aggregated_csv = video_quality_dir / "csv_results" / model_name / "aggregated_results.csv"
    rc = run_cmd(
        [
            "python",
            "myscript/summarize_csv.py",
            str(aggregated_csv),
            "--with-header",
            "--full",
        ],
        cwd=video_quality_dir,
        log_path=log_path,
        description="summarize_csv",
        dry_run=dry_run,
    )
    if rc != 0:
        return False, None, f"summarize_csv.py 退出码 {rc}"

    summary_full = aggregated_csv.parent / "summary_full.tsv"
    if dry_run:
        return True, summary_full, "dry-run"
    if not summary_full.exists():
        return False, None, f"summary_full.tsv 未生成: {summary_full}"
    return True, summary_full, "ok"


# ------------------------------ 主循环 -----------------------------------


_INTERRUPTED = False


def _on_sigint(signum, frame):  # noqa: ANN001
    global _INTERRUPTED
    _INTERRUPTED = True
    log("收到中断信号，将在当前评测结束后退出主循环…")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-path",
        required=True,
        type=Path,
        help="需要持续扫描的根目录（如某次推理 run 的输出目录）",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="结果 TSV 与 state 保存目录（如 /mnt/jackzou/WorldArena/results）",
    )
    parser.add_argument(
        "--target-dir-name",
        default="test_40",
        help="待匹配的子目录名（默认 test_40）",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="判定推理完成的视频数（默认 1000）",
    )
    parser.add_argument(
        "--video-exts",
        default=".mp4",
        help="逗号分隔的视频扩展名（默认 .mp4）",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="两轮扫描之间的秒数（默认 120）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只扫描一次并把当前可测项跑完，结束后退出（不进入循环）",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="本次最多评测多少个新条目（0 表示不限制）",
    )
    parser.add_argument(
        "--video-quality-dir",
        type=Path,
        default=DEFAULT_VIDEO_QUALITY_DIR,
        help="video_quality 目录（默认 %(default)s）",
    )
    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help="评测使用的 summary json（相对 video-quality-dir，默认 %(default)s）",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="评测使用的 config（相对 video-quality-dir，默认 %(default)s）",
    )
    parser.add_argument(
        "--gt-video-dir",
        type=Path,
        default=DEFAULT_GT_VIDEO_DIR,
        help="JEPA 使用的 GT 视频目录（默认 %(default)s）",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help="标准指标列表，逗号分隔（默认与 origin_test/eval_full.sh 一致）",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="跳过 VLM judge（origin_test/eval_full.sh 默认包含 VLM）",
    )
    parser.add_argument(
        "--run-jepa",
        action="store_true",
        help="启用 JEPA 评测（默认关闭，与 origin_test/eval_full.sh 一致）",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        metavar="N",
        help=(
            "调试用：限制每个评测脚本最多处理前 N 条数据"
            "（透传给 run_evaluation_multi_gpu.sh 的 LIMIT、"
            "run_VLM_judge_multi_gpu.sh 的 MAX_VIDEOS 以及 "
            "run_evaluation_JEPA.sh 的 MAX_SAMPLES）。0 = 全量（默认）"
        ),
    )
    parser.add_argument(
        "--model-prefix",
        default="",
        help="生成的 MODEL_NAME 前缀，便于区分多组实验",
    )
    parser.add_argument(
        "--tsv-name",
        default="",
        help="结果 TSV 文件名（默认按扫描根名生成）",
    )
    parser.add_argument(
        "--state-name",
        default="",
        help="state JSON 文件名（默认按扫描根名生成）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不实际执行评测命令，仅打印调度过程（用于自检）",
    )

    args = parser.parse_args()

    scan_path: Path = args.scan_path.resolve()
    if not scan_path.is_dir():
        log(f"ERROR: 扫描路径不存在或不是目录: {scan_path}")
        return 2

    video_quality_dir: Path = args.video_quality_dir.resolve()
    if not video_quality_dir.is_dir():
        log(f"ERROR: video_quality_dir 不是目录: {video_quality_dir}")
        return 2

    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    base_tag = scan_path.name or "auto_eval"
    tsv_path = results_dir / (args.tsv_name or f"{base_tag}.tsv")
    state_path = results_dir / (args.state_name or f".{base_tag}.state.json")
    log_dir = results_dir / "logs" / base_tag

    state = State(state_path)
    ensure_tsv_header(tsv_path)

    exts = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in args.video_exts.split(",")
        if e.strip()
    )

    log(f"扫描路径   : {scan_path}")
    log(f"结果 TSV   : {tsv_path}")
    log(f"state 文件 : {state_path}")
    log(f"日志目录   : {log_dir}")
    log(f"目标数     : count({'/'.join(exts) or 'mp4'}) >= {args.target_count}")
    log(f"video_quality 目录: {video_quality_dir}")
    log(f"summary_json     : {args.summary_json}")
    log(f"config_path      : {args.config_path}")
    log(f"gt_video_dir     : {args.gt_video_dir}")
    log(f"metrics          : {args.metrics}")
    log(f"skip_vlm         : {args.skip_vlm}")
    log(f"run_jepa         : {args.run_jepa}")
    log(f"max_videos       : {args.max_videos if args.max_videos > 0 else 'all'}")
    log(f"interval         : {args.interval}s, once={args.once}, max_runs={args.max_runs}")
    if args.dry_run:
        log("[DRY-RUN] 评测命令不会真正执行")

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    runs_done = 0
    iteration = 0

    while not _INTERRUPTED:
        iteration += 1
        log(f"=== 扫描轮次 #{iteration} ===")
        candidates = find_test_40_dirs(scan_path, args.target_dir_name)
        log(f"找到 {len(candidates)} 个 {args.target_dir_name} 目录")

        # 过滤：未完成 + 视频数达标
        ready: List[Path] = []
        for d in candidates:
            key = str(d)
            if state.is_completed(key):
                continue
            n = count_videos(d, exts)
            status = "ready" if n >= args.target_count else "waiting"
            log(f"  - [{status}] {d} ({n} videos)")
            if n >= args.target_count:
                ready.append(d)

        if not ready:
            log("当前无可执行任务。")
        else:
            log(f"待执行任务 {len(ready)} 个，开始顺序处理。")

        for test_dir in ready:
            if _INTERRUPTED:
                break
            if args.max_runs and runs_done >= args.max_runs:
                log(f"已达到 --max-runs={args.max_runs}，停止本轮评测。")
                break

            video_dir_str = str(test_dir)
            model_name = derive_model_name(scan_path, test_dir, args.model_prefix)
            log(f"--> 开始评测  model_name={model_name}")
            log(f"             video_dir ={video_dir_str}")
            t0 = time.time()
            ok, summary_tsv, msg = evaluate_one(
                video_dir=test_dir,
                model_name=model_name,
                video_quality_dir=video_quality_dir,
                summary_json=args.summary_json,
                config_path=args.config_path,
                metrics=args.metrics,
                gt_video_dir=args.gt_video_dir,
                skip_vlm=args.skip_vlm,
                run_jepa=args.run_jepa,
                max_videos=args.max_videos,
                log_dir=log_dir,
                dry_run=args.dry_run,
            )
            elapsed = time.time() - t0

            if not ok:
                log(f"<-- 评测失败 ({elapsed:.1f}s): {msg}")
                state.mark_failed(
                    video_dir_str,
                    {
                        "model_name": model_name,
                        "reason": msg,
                        "last_attempt_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            if args.dry_run:
                log(f"<-- [DRY-RUN] 流水线检查通过 ({elapsed:.1f}s)，跳过 TSV 追加")
                runs_done += 1
                continue

            try:
                metrics_map = parse_summary_full_tsv(summary_tsv)
            except Exception as e:  # noqa: BLE001
                log(f"<-- 解析 summary_full.tsv 失败: {e}")
                state.mark_failed(
                    video_dir_str,
                    {
                        "model_name": model_name,
                        "reason": f"parse summary_full.tsv: {e}",
                        "last_attempt_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row, avg = append_tsv_row(
                tsv_path, ts, model_name, video_dir_str, metrics_map
            )
            log(
                f"<-- 评测完成 ({elapsed:.1f}s)  Mean={avg if avg is None else f'{avg:.2f}'}  -> {tsv_path}"
            )

            state.mark_completed(
                video_dir_str,
                {
                    "model_name": model_name,
                    "completed_at": ts,
                    "elapsed_sec": round(elapsed, 1),
                    "metrics": {
                        k: (None if v is None else round(v, 4))
                        for k, v in metrics_map.items()
                    },
                    "mean": None if avg is None else round(avg, 4),
                    "tsv_row": row,
                },
            )
            runs_done += 1

        if args.once:
            log("--once 已设置，退出。")
            break
        if args.max_runs and runs_done >= args.max_runs:
            log(f"--max-runs={args.max_runs} 已达成，退出。")
            break

        log(f"等待 {args.interval}s 后再次扫描…（Ctrl-C 退出）")
        # 分片 sleep 以快速响应 SIGINT
        for _ in range(args.interval):
            if _INTERRUPTED:
                break
            time.sleep(1)

    log("auto_eval 主循环结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
