"""
batch_generate.py — 批量调用 Wan2.2-TI2V-5B (I2V) 为 WorldArena 测试集生成视频

用法示例（单 GPU）:
    python batch_generate.py \
        --summary_json  ../video_quality/summary.json \
        --output_dir    /path/to/wan_test \
        --checkpoint_dir ./models/Wan2.2-TI2V-5B \
        --pt_dir        ./models/wan_video/wan_video.pt \
        --size 640*480 --frame_num 121 --seed 42

多 GPU 并行（按 episode 分段）:
    CUDA_VISIBLE_DEVICES=0 python batch_generate.py --start_idx 0   --end_idx 250  ... &
    CUDA_VISIBLE_DEVICES=1 python batch_generate.py --start_idx 250 --end_idx 500  ... &
    CUDA_VISIBLE_DEVICES=2 python batch_generate.py --start_idx 500 --end_idx 750  ... &
    CUDA_VISIBLE_DEVICES=3 python batch_generate.py --start_idx 750 --end_idx 1000 ... &

也可以直接从 dataset 目录读取首帧和指令（不使用 summary.json）:
    python batch_generate.py \
        --dataset_dir /path/to/test_dataset \
        --output_dir  /path/to/wan_test \
        --checkpoint_dir ./models/Wan2.2-TI2V-5B
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import torch
torch.backends.cudnn.enabled = False
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.textimage2video import WanTI2V
from wan.utils.utils import safe_save_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2-TI2V-5B batch video generation")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--summary_json", type=str,
                     help="video_quality/summary.json 路径（已整理好的首帧+prompt）")
    src.add_argument("--dataset_dir", type=str,
                     help="test_dataset 根路径（内含 first_frame/ 和 instructions/）")

    p.add_argument("--instruction_subdir", type=str, default="instructions",
                   choices=["instructions", "instructions_1", "instructions_2"],
                   help="使用哪组指令（仅 --dataset_dir 模式有效）")
    p.add_argument("--instruction_field", type=str, default="instruction",
                   help="JSON 中的指令字段名")

    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录，视频保存为 episode{N}.mp4")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Wan2.2-TI2V-5B 模型目录")
    p.add_argument("--pt_dir", type=str, default=None,
                   help="微调权重路径（如 wan_video.pt），不传则用原始权重")

    p.add_argument("--size", type=str, default="640*480",
                   help="输出分辨率 宽*高，支持: 640*480, 640*736, 704*1280, 1280*704")
    p.add_argument("--frame_num", type=int, default=121,
                   help="生成帧数（必须为 4n+1）")
    p.add_argument("--fps", type=int, default=24,
                   help="输出视频帧率")
    p.add_argument("--sampling_steps", type=int, default=50,
                   help="扩散采样步数")
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="无分类器引导尺度")
    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"],
                   help="采样求解器")
    p.add_argument("--shift", type=float, default=None,
                   help="噪声调度偏移（默认: 480p→3.0, 其他→5.0）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（-1 为随机）")

    p.add_argument("--start_idx", type=int, default=0,
                   help="起始索引（含），基于 0")
    p.add_argument("--end_idx", type=int, default=None,
                   help="结束索引（不含），默认到末尾")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--t5_cpu", action="store_true",
                   help="将 T5 放在 CPU 上以节省显存")
    p.add_argument("--offload_model", action="store_true", default=True,
                   help="推理后卸载模型到 CPU（省显存，稍慢）")
    p.add_argument("--no_offload", action="store_true",
                   help="禁用模型卸载（需要更多显存，但更快）")

    return p.parse_args()


def load_episodes_from_summary(summary_json):
    """从 summary.json 加载 episode 列表"""
    with open(summary_json) as f:
        data = json.load(f)
    episodes = []
    for item in data:
        img_path = item["image"]
        prompt = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
        m = re.search(r"episode(\d+)", img_path)
        ep_num = int(m.group(1)) if m else len(episodes) + 1
        episodes.append({
            "ep_num": ep_num,
            "img_path": img_path,
            "prompt": prompt,
        })
    return episodes


def load_episodes_from_dataset(dataset_dir, instruction_subdir, instruction_field):
    """从 dataset 目录结构加载 episode 列表"""
    ff_dir = os.path.join(dataset_dir, "first_frame", "fixed_scene_task")
    inst_dir = os.path.join(dataset_dir, instruction_subdir, "fixed_scene_task")
    episodes = []

    for fname in sorted(os.listdir(ff_dir)):
        m = re.match(r"episode(\d+)\.(png|jpg)", fname)
        if not m:
            continue
        ep_num = int(m.group(1))
        img_path = os.path.join(ff_dir, fname)
        json_path = os.path.join(inst_dir, f"episode{ep_num}.json")
        if not os.path.exists(json_path):
            log.warning("指令文件不存在: %s，跳过", json_path)
            continue
        with open(json_path) as f:
            prompt = json.load(f).get(instruction_field, "")
        episodes.append({
            "ep_num": ep_num,
            "img_path": img_path,
            "prompt": prompt,
        })

    episodes.sort(key=lambda x: x["ep_num"])
    return episodes


def main():
    args = parse_args()

    if args.no_offload:
        args.offload_model = False

    if args.shift is None:
        args.shift = 3.0 if "480" in args.size else 5.0

    max_area = MAX_AREA_CONFIGS.get(args.size)
    if max_area is None:
        w, h = [int(x) for x in args.size.split("*")]
        max_area = w * h
        log.warning("非标准分辨率 %s，使用 max_area=%d", args.size, max_area)

    # ---- 加载 episode 列表 ----
    if args.summary_json:
        episodes = load_episodes_from_summary(args.summary_json)
        log.info("从 summary.json 加载了 %d 个 episode", len(episodes))
    else:
        episodes = load_episodes_from_dataset(
            args.dataset_dir, args.instruction_subdir, args.instruction_field)
        log.info("从 dataset 目录加载了 %d 个 episode", len(episodes))

    end_idx = args.end_idx if args.end_idx is not None else len(episodes)
    episodes = episodes[args.start_idx:end_idx]
    log.info("本次生成范围: 索引 [%d, %d)，共 %d 个", args.start_idx, end_idx, len(episodes))

    if not episodes:
        log.warning("没有需要生成的 episode，退出")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 跳过已生成的 ----
    existing = set(os.listdir(args.output_dir))
    todo = [ep for ep in episodes if f"episode{ep['ep_num']}.mp4" not in existing]
    log.info("已存在 %d 个，待生成 %d 个", len(episodes) - len(todo), len(todo))

    if not todo:
        log.info("全部已生成，无需操作")
        return

    # ---- 加载模型 ----
    config = WAN_CONFIGS["ti2v-5B"]
    log.info("加载 WanTI2V (checkpoint=%s, pt=%s)", args.checkpoint_dir, args.pt_dir)
    model = WanTI2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        pt_dir=args.pt_dir,
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
    )
    log.info("模型加载完成")

    # ---- 逐个生成 ----
    total = len(todo)
    for i, ep in enumerate(todo):
        ep_num = ep["ep_num"]
        output_path = os.path.join(args.output_dir, f"episode{ep_num}.mp4")

        log.info("[%d/%d] episode%d — %s", i + 1, total, ep_num, ep["prompt"][:80])
        t0 = time.time()

        img = Image.open(ep["img_path"]).convert("RGB")

        video_tensor = model.generate(
            input_prompt=ep["prompt"],
            img=img,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=args.shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=args.offload_model,
        )

        if video_tensor is not None:
            # video_tensor: (C, N, H, W)  值域 [-1, 1]
            safe_save_video(
                video_tensor.unsqueeze(0),
                save_file=output_path,
                fps=args.fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            elapsed = time.time() - t0
            log.info("  已保存 %s  (%.1fs)", output_path, elapsed)
        else:
            log.warning("  episode%d 生成返回 None，跳过", ep_num)

        torch.cuda.empty_cache()

    log.info("全部完成! 输出目录: %s", args.output_dir)


if __name__ == "__main__":
    main()
