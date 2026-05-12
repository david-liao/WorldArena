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

多节点并发（共享存储）:
    - ``--instance-id auto`` 默认使用 hostname 作为节点 ID（不同节点务必不同）。
    - state.json 增加 ``running`` 段并用 ``fcntl.flock`` 在锁内 read-modify-write，
      用 ``try_claim`` 原子认领任务；评测期间后台心跳线程每 ``--heartbeat-interval``
      秒刷新一次，崩溃节点的任务会在 ``--heartbeat-timeout`` 秒后被其他节点重抢。
    - 默认会基于 ``--config-path`` 派生 ``config/.auto_eval_config_<instance>.yaml``，
      并把 ``data.gt_path`` / ``data.val_base`` 切到 ``data_<instance>/`` 下，
      使每节点的预处理目录互不冲突。
    - 旧的 ``state.json`` 中的 ``completed`` 项继续被识别，已完成的评测不会再重做。
    - GT 跨节点共享：``--shared-gt-mirror <dir>`` 启用 mirror，节点会用 rsync
      同步最贵的 SAM3 产物（``traj/`` 与 ``gripper_detection/``）。首个节点完成
      首次评测后写入 ``ready.sentinel``；后续节点启动时直接 pull，省去 SAM3 重算。

用法（典型，单节点）:
    python myscript/eval/auto_eval.py \\
        --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/<run_dir> \\
        --results-dir /mnt/jackzou/WorldArena/results

用法（多节点，共享存储）:
    # 在每个节点上分别启动；instance-id auto 会取 hostname
    python myscript/eval/auto_eval.py \\
        --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/<run_dir> \\
        --results-dir /mnt/jackzou/WorldArena/results \\
        --shared-gt-mirror /mnt/jackzou/WorldArena/results/_shared_gt

按 Ctrl-C 退出。当前运行中的评测会让其完成，再退出循环。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import fcntl
import fnmatch
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple


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

# 多节点并发相关常量
DEFAULT_HEARTBEAT_INTERVAL = 30   # 心跳更新间隔（秒）
DEFAULT_HEARTBEAT_TIMEOUT = 600   # 心跳超时阈值（秒）；超过则该 running 项视为僵尸可重抢
DEFAULT_GT_SYNC_TIMEOUT = 1800    # GT mirror rsync 锁等待时长（秒）

TSV_HEADER: List[str] = (
    ["timestamp", "model_name", "video_dir", "instance_id"] + METRIC_COLUMNS + ["Mean"]
)


# ------------------------------ 工具函数 -----------------------------------


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[auto_eval {ts}] {msg}", flush=True)


def get_default_instance_id() -> str:
    """默认 instance_id = hostname；带特殊字符的会被规范化。"""
    raw = socket.gethostname() or "node"
    safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in raw)
    return safe or "node"


@contextlib.contextmanager
def file_lock(lock_path: Path, exclusive: bool = True, timeout: float = -1.0):
    """fcntl.flock 上下文管理器（POSIX）。

    Args:
        lock_path: lock 哨兵文件路径（不存在会被创建）。
        exclusive: True = LOCK_EX，False = LOCK_SH。
        timeout: < 0 表示阻塞到拿到锁；> 0 表示在该秒数内反复 LOCK_NB 尝试，超时抛 TimeoutError。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        if timeout < 0:
            fcntl.flock(fd, op)
        else:
            deadline = time.time() + timeout
            while True:
                try:
                    fcntl.flock(fd, op | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    if time.time() >= deadline:
                        raise TimeoutError(
                            f"acquire lock timed out after {timeout}s: {lock_path}"
                        )
                    time.sleep(1.0)
        try:
            yield fd
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


def count_videos(directory: Path, exts: Tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    cnt = 0
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in exts:
            cnt += 1
    return cnt


def _has_glob(pattern: str) -> bool:
    """是否包含 fnmatch 通配符（``*``、``?``、``[seq]``）。"""
    return any(c in pattern for c in "*?[")


def find_test_40_dirs(scan_path: Path, target_dir_name: str) -> List[Path]:
    """查找 ``scan_path`` 下所有目录名匹配 ``target_dir_name`` 的目录。

    ``target_dir_name`` 支持 shell-style glob 通配符（``*``、``?``、``[seq]``,
    见 ``fnmatch``）：

    - 字面量（如 ``test_40``）：精确匹配；
    - 含通配符（如 ``*_frames``、``test_[0-9][0-9]``）：按 ``fnmatch.fnmatchcase``
      匹配目录的 *basename*（不会跨层级）。

    匹配到的目录不会再向下递归查找（避免在已认定的叶子目录里重复扫出同名/同模式
    子目录）。
    """
    if not scan_path.is_dir():
        return []
    use_glob = _has_glob(target_dir_name)
    found: List[Path] = []
    for root, dirs, _ in os.walk(scan_path, followlinks=False):
        if use_glob:
            matched = [d for d in dirs if fnmatch.fnmatchcase(d, target_dir_name)]
            if matched:
                root_path = Path(root)
                for m in matched:
                    found.append(root_path / m)
                # 不再下挖匹配项（与字面量分支语义一致）
                matched_set = set(matched)
                dirs[:] = [d for d in dirs if d not in matched_set]
        else:
            if target_dir_name in dirs:
                found.append(Path(root) / target_dir_name)
                # 不再向下挖（test_40 内部不会再有同名目录）
                dirs[:] = [d for d in dirs if d != target_dir_name]
    return sorted(found)


def derive_model_name(
    scan_path: Path,
    test_dir: Path,
    prefix: str = "",
    *,
    include_leaf: bool = False,
) -> str:
    """从扫描根 + 目标目录路径推导唯一 MODEL_NAME。

    Args:
        include_leaf: 当 ``test_dir`` 的最末段不固定（如使用通配符匹配多个目录，
            ``*_frames`` 会得到 ``cam0_frames`` / ``cam1_frames`` …）时应置为
            True，把最末段也并入 suffix，避免不同目录映射到同一 model_name。
    """
    rel = test_dir.relative_to(scan_path)
    scan_name = scan_path.name
    parts = list(rel.parts[:-1])  # 去掉末尾的目标目录本身
    leaf = rel.parts[-1] if rel.parts else ""
    step_part = next((p for p in parts if p.startswith("step-")), None)
    if step_part:
        suffix = step_part
    elif parts:
        suffix = "_".join(parts)
    else:
        suffix = "" if (include_leaf and leaf) else "root"
    if include_leaf and leaf:
        suffix = f"{suffix}_{leaf}" if suffix else leaf
    base = f"{scan_name}__{suffix}" if scan_name else suffix
    return f"{prefix}{base}" if prefix else base


# ------------------------------ State 持久化 --------------------------------


class State:
    """多节点共享 state.json 的安全访问层。

    state 结构（向后兼容旧版的 ``completed`` / ``failed`` 段，自动补 ``running``）::

        {
          "completed": { "<video_dir>": {... metrics, mean ...} },
          "failed":    { "<video_dir>": {"reason": "...", "attempts": N} },
          "running":   { "<video_dir>": {
                "instance_id": "...",
                "model_name":  "...",
                "pid":         12345,
                "started_at":  "ISO timestamp",
                "heartbeat_at_ts": <unix seconds>,
                "heartbeat_at":    "ISO timestamp",
            } }
        }

    所有 read/modify/write 都在 ``fcntl.flock`` 排他锁内完成，确保多节点安全。
    """

    def __init__(self, path: Path, instance_id: str, heartbeat_timeout: int):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.instance_id = instance_id
        self.heartbeat_timeout = heartbeat_timeout

    def _read_unlocked(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {"completed": {}, "failed": {}, "running": {}}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            log(f"WARN: 无法解析 state file {self.path}: {e}; 视作空状态")
            return {"completed": {}, "failed": {}, "running": {}}
        # 兼容旧版（缺少 running 段）
        data.setdefault("completed", {})
        data.setdefault("failed", {})
        data.setdefault("running", {})
        return data

    def _write_unlocked(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    @contextlib.contextmanager
    def _locked_data(self):
        with file_lock(self.lock_path, exclusive=True):
            data = self._read_unlocked()
            yield data
            self._write_unlocked(data)

    def _is_alive_running(self, info: Dict[str, Any], now_ts: float) -> bool:
        hb_ts = float(info.get("heartbeat_at_ts", 0) or 0)
        return (now_ts - hb_ts) < self.heartbeat_timeout

    def blocking_set(self) -> Set[str]:
        """返回当前不能被本节点认领的 video_dir 集合（已完成 + 仍活跃的 running）。"""
        with self._locked_data() as data:
            now = time.time()
            blocked: Set[str] = set(data["completed"].keys())
            for vd, info in data["running"].items():
                if self._is_alive_running(info, now):
                    blocked.add(vd)
            return blocked

    def try_claim(self, video_dir: str, model_name: str) -> bool:
        """原子地认领某个 ``video_dir``。

        - 已 completed -> False（永远不再做）
        - running 且心跳新鲜 -> False（让别的节点继续做）
        - running 但心跳过期 -> 视为僵尸，本节点抢占并覆盖
        - 不在 running -> 写入 running，返回 True
        """
        with self._locked_data() as data:
            if video_dir in data["completed"]:
                return False

            now = time.time()
            now_iso = datetime.now().isoformat(timespec="seconds")
            existing = data["running"].get(video_dir)
            if existing is not None and self._is_alive_running(existing, now):
                if existing.get("instance_id") != self.instance_id:
                    return False
                # 是本节点先前的认领（应当已被 mark_completed/mark_failed 清理过；
                # 走到这里说明上一次评测异常退出且 heartbeat 还没超时，直接复用）

            data["running"][video_dir] = {
                "instance_id": self.instance_id,
                "model_name": model_name,
                "pid": os.getpid(),
                "started_at": now_iso,
                "heartbeat_at_ts": now,
                "heartbeat_at": now_iso,
            }
            return True

    def heartbeat(self, video_dir: str) -> None:
        with self._locked_data() as data:
            info = data["running"].get(video_dir)
            if info is None or info.get("instance_id") != self.instance_id:
                return
            info["heartbeat_at_ts"] = time.time()
            info["heartbeat_at"] = datetime.now().isoformat(timespec="seconds")

    def mark_completed(self, video_dir: str, info: Dict[str, Any]) -> None:
        with self._locked_data() as data:
            data["completed"][video_dir] = info
            data["running"].pop(video_dir, None)
            data["failed"].pop(video_dir, None)

    def mark_failed(self, video_dir: str, info: Dict[str, Any]) -> None:
        with self._locked_data() as data:
            prev = data["failed"].get(video_dir, {"attempts": 0})
            info["attempts"] = int(prev.get("attempts", 0)) + 1
            data["failed"][video_dir] = info
            data["running"].pop(video_dir, None)


class HeartbeatThread(threading.Thread):
    """后台线程：定期更新 state.running[<video_dir>].heartbeat_at_ts。"""

    def __init__(self, state: State, video_dir: str, interval: int):
        super().__init__(daemon=True)
        self.state = state
        self.video_dir = video_dir
        self.interval = max(1, int(interval))
        # 注意：不能命名为 _stop —— 会覆盖 threading.Thread._stop() 方法，
        # 导致 join() 内部调用 _stop 时报 'Event is not callable'。
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.state.heartbeat(self.video_dir)
            except Exception as e:  # noqa: BLE001
                log(f"WARN: heartbeat 失败: {e}")

    def stop(self) -> None:
        self._stop_event.set()


# ------------------------- 节点专属 DATA_DIR / config -------------------------


def setup_per_node_config(
    base_config_relpath: str,
    instance_id: str,
    video_quality_dir: Path,
) -> Tuple[str, Path]:
    """从 base config 派生节点专属 config，把 ``data.gt_path`` / ``data.val_base``
    指到 ``data_<instance>/``，避免多节点共享预处理数据互踩。

    Returns:
        (新 config 相对 ``video_quality_dir`` 的路径, 节点 DATA_DIR 绝对路径)
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "需要 PyYAML 才能生成节点专属 config（请激活 WorldArena conda 环境）"
        ) from e

    base_path = (video_quality_dir / base_config_relpath).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base config 不存在: {base_path}")

    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)

    node_data_dir = video_quality_dir / f"data_{instance_id}"
    node_data_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("data", {})
    cfg["data"]["gt_path"] = str(node_data_dir / "gt_dataset")
    cfg["data"]["val_base"] = str(node_data_dir / "generated_dataset")

    af_dir = video_quality_dir / f"data_action_following_{instance_id}"
    cfg.setdefault("data_action_following", {})
    cfg["data_action_following"]["gt_path"] = str(af_dir / "gt_dataset")
    cfg["data_action_following"]["val_base"] = str(af_dir / "generated_dataset")

    out_relative = f"config/.auto_eval_config_{instance_id}.yaml"
    out_path = video_quality_dir / out_relative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_relative, node_data_dir


# ------------------------------ GT 跨节点共享 -------------------------------


def _rsync(src_dir: Path, dst_dir: Path, *, ignore_existing: bool) -> Tuple[int, str]:
    """rsync 包装；返回 (returncode, message)。"""
    if shutil.which("rsync") is None:
        return 127, "rsync 未安装"
    src_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a"]
    if ignore_existing:
        cmd.append("--ignore-existing")
    cmd.extend([str(src_dir).rstrip("/") + "/", str(dst_dir).rstrip("/") + "/"])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def pull_gt_mirror(
    mirror_dir: Path,
    node_data_dir: Path,
    *,
    timeout: float = DEFAULT_GT_SYNC_TIMEOUT,
) -> bool:
    """从共享 GT mirror 拉取 SAM3 缓存（traj/ + gripper_detection/）到节点 GT。

    若 mirror 还没有 ``ready.sentinel`` 则跳过；用共享锁允许多节点并行 pull。
    """
    sentinel = mirror_dir / "ready.sentinel"
    if not sentinel.exists():
        log(f"[gt-mirror] mirror 尚未就绪（无 ready.sentinel），跳过 pull: {mirror_dir}")
        return False
    lock_path = mirror_dir / ".sync.lock"
    try:
        with file_lock(lock_path, exclusive=False, timeout=timeout):
            if not sentinel.exists():
                return False
            src = mirror_dir / "gt_dataset"
            dst = node_data_dir / "gt_dataset"
            if not src.is_dir():
                return False
            rc, msg = _rsync(src, dst, ignore_existing=True)
            if rc != 0:
                log(f"[gt-mirror] pull rsync 退出码 {rc}: {msg}")
                return False
            log(f"[gt-mirror] pull 完成: {src} -> {dst}")
            return True
    except TimeoutError as e:
        log(f"[gt-mirror] pull 等锁超时: {e}")
        return False


def push_gt_mirror(
    mirror_dir: Path,
    node_data_dir: Path,
    *,
    timeout: float = DEFAULT_GT_SYNC_TIMEOUT,
) -> bool:
    """评测完成后，把节点 GT 缓存（含 traj/、gripper_detection/）push 回共享 mirror。

    使用 ``--ignore-existing``：mirror 已有的就不覆盖（避免和并发 pull/push 冲突），
    成功后写 ``ready.sentinel`` 让后续节点可以使用。
    """
    src = node_data_dir / "gt_dataset"
    if not src.is_dir():
        return False
    lock_path = mirror_dir / ".sync.lock"
    try:
        with file_lock(lock_path, exclusive=True, timeout=timeout):
            dst = mirror_dir / "gt_dataset"
            rc, msg = _rsync(src, dst, ignore_existing=True)
            if rc != 0:
                log(f"[gt-mirror] push rsync 退出码 {rc}: {msg}")
                return False
            (mirror_dir / "ready.sentinel").touch()
            log(f"[gt-mirror] push 完成: {src} -> {dst}")
            return True
    except TimeoutError as e:
        log(f"[gt-mirror] push 等锁超时: {e}")
        return False


# ------------------------------ TSV 输出 -----------------------------------


def ensure_tsv_header(tsv_path: Path) -> None:
    """无表头则写一次表头；用文件锁防止多节点并发情况下重复写表头。"""
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = tsv_path.with_suffix(tsv_path.suffix + ".header.lock")
    with file_lock(lock_path, exclusive=True):
        if tsv_path.exists() and tsv_path.stat().st_size > 0:
            return
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
    instance_id: str,
    metrics: Dict[str, Optional[float]],
) -> Tuple[List[str], Optional[float]]:
    """追加一行 TSV。POSIX 上 ``open('a')`` 使用 ``O_APPEND``，单行追加在
    < PIPE_BUF（通常 4 KB）时是原子的；多节点并发 append 不会互相截断。
    """
    ensure_tsv_header(tsv_path)
    cells: List[str] = [timestamp, model_name, video_dir, instance_id]
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
        help=(
            "待匹配的子目录名（默认 ``test_40``）。支持 shell-style 通配符 "
            "（``*``、``?``、``[seq]``，见 fnmatch），按目录 basename 匹配。"
            "示例: ``test_40`` 精确匹配；``*_frames`` 匹配所有以 ``_frames`` "
            "结尾的目录；``test_[0-9][0-9]`` 匹配 ``test_00``..``test_99``。"
            "使用通配符时，model_name 会自动包含末段目录名以避免冲突。"
        ),
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
    # ----- 多节点并发 -----
    parser.add_argument(
        "--instance-id",
        default="auto",
        help=(
            "节点标识，默认 ``auto`` = hostname。多节点共享存储时必须保证不同节点"
            "的 instance_id 互不相同（避免 data_<id>/ 与节点 config 冲突）"
        ),
    )
    parser.add_argument(
        "--no-per-node-config",
        action="store_true",
        help=(
            "禁用 per-node 派生 config（仅当确实只在单节点跑时才使用）。默认会基于"
            " --config-path 生成 config/.auto_eval_config_<instance>.yaml，并把"
            " data.gt_path / data.val_base 改到 data_<instance>/ 下"
        ),
    )
    parser.add_argument(
        "--shared-gt-mirror",
        type=Path,
        default=None,
        help=(
            "共享 GT 缓存镜像目录（如 /mnt/jackzou/WorldArena/results/_shared_gt）。"
            "节点首次启动时会从该目录 rsync 拉取 SAM3 缓存（traj/ + "
            "gripper_detection/）；每次评测完成后再 push 回去（--ignore-existing）。"
            "未指定则不共享，节点之间各自独立预处理 GT。"
        ),
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=DEFAULT_HEARTBEAT_INTERVAL,
        help="评测过程中心跳更新间隔（秒，默认 %(default)s）",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=int,
        default=DEFAULT_HEARTBEAT_TIMEOUT,
        help="心跳超时阈值（秒，默认 %(default)s），超过则 running 项视为僵尸可重抢",
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

    instance_id = (
        get_default_instance_id() if args.instance_id == "auto" else args.instance_id
    )

    state = State(
        state_path,
        instance_id=instance_id,
        heartbeat_timeout=max(60, args.heartbeat_timeout),
    )
    ensure_tsv_header(tsv_path)

    # 派生节点专属 config（默认开启），把 data.* 路径切到 data_<instance>/
    effective_config_relpath = args.config_path
    node_data_dir: Optional[Path] = None
    if not args.no_per_node_config and not args.dry_run:
        try:
            effective_config_relpath, node_data_dir = setup_per_node_config(
                args.config_path,
                instance_id,
                video_quality_dir,
            )
            log(f"已生成节点专属 config: {video_quality_dir / effective_config_relpath}")
            log(f"节点 DATA_DIR    : {node_data_dir}")
        except Exception as e:  # noqa: BLE001
            log(f"WARN: 生成节点专属 config 失败，继续使用原 config: {e}")
    elif args.no_per_node_config:
        log("⚠️  --no-per-node-config 已设置；多节点同时跑时存在 data/ 写争用风险")

    # GT mirror 同步（启动阶段先 pull 一次）
    shared_gt_mirror: Optional[Path] = None
    if args.shared_gt_mirror is not None and not args.dry_run:
        mirror_dir: Path = args.shared_gt_mirror.resolve()
        mirror_dir.mkdir(parents=True, exist_ok=True)
        shared_gt_mirror = mirror_dir
        if node_data_dir is not None:
            log(f"[gt-mirror] 启动时尝试从 {mirror_dir} 拉 GT SAM3 缓存…")
            pull_gt_mirror(mirror_dir, node_data_dir)
        else:
            log("[gt-mirror] 未启用 per-node config，跳过 GT mirror（避免污染共享 data/）")

    exts = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in args.video_exts.split(",")
        if e.strip()
    )

    log(f"扫描路径   : {scan_path}")
    log(f"结果 TSV   : {tsv_path}")
    log(f"state 文件 : {state_path}")
    log(f"日志目录   : {log_dir}")
    log(f"instance_id: {instance_id}")
    log(f"目标数     : count({'/'.join(exts) or 'mp4'}) >= {args.target_count}")
    log(f"video_quality 目录: {video_quality_dir}")
    log(f"summary_json     : {args.summary_json}")
    log(f"effective config : {effective_config_relpath}")
    log(f"node DATA_DIR    : {node_data_dir if node_data_dir else '(unchanged)'}")
    log(f"shared_gt_mirror : {shared_gt_mirror if shared_gt_mirror else '(disabled)'}")
    log(f"gt_video_dir     : {args.gt_video_dir}")
    log(f"metrics          : {args.metrics}")
    log(f"skip_vlm         : {args.skip_vlm}")
    log(f"run_jepa         : {args.run_jepa}")
    log(f"max_videos       : {args.max_videos if args.max_videos > 0 else 'all'}")
    log(f"heartbeat        : every {args.heartbeat_interval}s, timeout {args.heartbeat_timeout}s")
    log(f"interval         : {args.interval}s, once={args.once}, max_runs={args.max_runs}")
    if args.dry_run:
        log("[DRY-RUN] 评测命令不会真正执行")

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    runs_done = 0
    iteration = 0
    target_uses_glob = _has_glob(args.target_dir_name)
    if target_uses_glob:
        log(f"target_dir_name   : {args.target_dir_name} (glob; 启用 leaf-aware model_name)")
    else:
        log(f"target_dir_name   : {args.target_dir_name}")

    while not _INTERRUPTED:
        iteration += 1
        log(f"=== 扫描轮次 #{iteration} ===")
        candidates = find_test_40_dirs(scan_path, args.target_dir_name)
        log(f"找到 {len(candidates)} 个 {args.target_dir_name} 目录")

        # 一次性读取阻塞集合（已 completed + 别的节点活跃 running）
        blocked = state.blocking_set()

        ready: List[Path] = []
        for d in candidates:
            key = str(d)
            n = count_videos(d, exts)
            if key in blocked:
                # 区分 completed / running 给用户看
                tag = "done" if n >= args.target_count else "done(<target)"
                log(f"  - [{tag}/skip] {d} ({n} videos)")
                continue
            status = "ready" if n >= args.target_count else "waiting"
            log(f"  - [{status}] {d} ({n} videos)")
            if n >= args.target_count:
                ready.append(d)

        if not ready:
            log("当前无可执行任务。")
        else:
            log(f"候选任务 {len(ready)} 个，开始尝试认领并顺序处理。")

        for test_dir in ready:
            if _INTERRUPTED:
                break
            if args.max_runs and runs_done >= args.max_runs:
                log(f"已达到 --max-runs={args.max_runs}，停止本轮评测。")
                break

            video_dir_str = str(test_dir)
            model_name = derive_model_name(
                scan_path,
                test_dir,
                args.model_prefix,
                include_leaf=target_uses_glob,
            )

            # 原子认领：失败则被别的节点抢先
            if not state.try_claim(video_dir_str, model_name):
                log(f"  [skip-claimed] 已被其他节点认领或已完成: {video_dir_str}")
                continue

            log(f"--> 开始评测  model_name={model_name}")
            log(f"             video_dir ={video_dir_str}")
            log(f"             instance  ={instance_id}")
            heartbeat = HeartbeatThread(state, video_dir_str, args.heartbeat_interval)
            heartbeat.start()
            t0 = time.time()
            try:
                ok, summary_tsv, msg = evaluate_one(
                    video_dir=test_dir,
                    model_name=model_name,
                    video_quality_dir=video_quality_dir,
                    summary_json=args.summary_json,
                    config_path=effective_config_relpath,
                    metrics=args.metrics,
                    gt_video_dir=args.gt_video_dir,
                    skip_vlm=args.skip_vlm,
                    run_jepa=args.run_jepa,
                    max_videos=args.max_videos,
                    log_dir=log_dir,
                    dry_run=args.dry_run,
                )
            finally:
                heartbeat.stop()
                heartbeat.join(timeout=args.heartbeat_interval + 5)
            elapsed = time.time() - t0

            if not ok:
                log(f"<-- 评测失败 ({elapsed:.1f}s): {msg}")
                state.mark_failed(
                    video_dir_str,
                    {
                        "instance_id": instance_id,
                        "model_name": model_name,
                        "reason": msg,
                        "last_attempt_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            if args.dry_run:
                log(f"<-- [DRY-RUN] 流水线检查通过 ({elapsed:.1f}s)，跳过 TSV 追加")
                # dry-run 仍然需要释放 running，避免别的节点等心跳超时
                state.mark_completed(
                    video_dir_str,
                    {
                        "instance_id": instance_id,
                        "model_name": model_name,
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_sec": round(elapsed, 1),
                        "dry_run": True,
                    },
                )
                runs_done += 1
                continue

            assert summary_tsv is not None, "ok==True 时 summary_tsv 不应为 None"
            try:
                metrics_map = parse_summary_full_tsv(summary_tsv)
            except Exception as e:  # noqa: BLE001
                log(f"<-- 解析 summary_full.tsv 失败: {e}")
                state.mark_failed(
                    video_dir_str,
                    {
                        "instance_id": instance_id,
                        "model_name": model_name,
                        "reason": f"parse summary_full.tsv: {e}",
                        "last_attempt_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row, avg = append_tsv_row(
                tsv_path, ts, model_name, video_dir_str, instance_id, metrics_map
            )
            log(
                f"<-- 评测完成 ({elapsed:.1f}s)  Mean={avg if avg is None else f'{avg:.2f}'}  -> {tsv_path}"
            )

            state.mark_completed(
                video_dir_str,
                {
                    "instance_id": instance_id,
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

            # 评测成功后把节点 GT SAM3 缓存 push 回 mirror（如启用）
            if shared_gt_mirror is not None and node_data_dir is not None:
                push_gt_mirror(shared_gt_mirror, node_data_dir)

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
