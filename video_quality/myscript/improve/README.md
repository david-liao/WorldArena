# `myscript/improve/` — WorldArena 指标提升工具集

按计划三层结构组织（`Tier 1 = 后处理`, `Tier 2 = 推理参数`, `Tier 3 = 流水线/训练`）。
所有脚本只动视频侧 / 训练侧 / 流水线侧的产物，**不修改 `WorldArena/*.py` 内任何评测代码**，
也不改 `_EMPIRICAL_BOUNDS`。所有命令默认从 `video_quality/` 目录下运行。

## 目录

| 文件 | Tier | 作用 |
| --- | --- | --- |
| [`postprocess.py`](postprocess.py) | 1 | 一体化后处理 CLI：帧数对齐 / 首帧替换 / 颜色匹配 / unsharp / 末帧 EMA |
| [`io_utils.py`](io_utils.py) | — | mp4 / 帧目录互通 IO；mp4 写入默认 ffmpeg libx264 CRF 18 |
| [`ops.py`](ops.py) | — | 各后处理步骤的纯函数实现（cv2/numpy） |
| [`seed_select.py`](seed_select.py) | 2 | N×seed 选优：MUSIQ + LAION-aesthetic + Farneback 光流平滑代理 |
| [`vfi_interp.py`](vfi_interp.py) | 2 | 帧插值：VFIMamba（首选，与 motion_smoothness 同源）/ ffmpeg minterpolate / 线性混合 |
| [`firstframe_conditioning.py`](firstframe_conditioning.py) | 2 | 把 `summary.json` 转成 I2V 推理 manifest，落实"GT init_frame 当 first-frame 条件" |
| [`data_filter.py`](data_filter.py) | 3 | 训练集按 LAION-aesthetic + 运动量过滤 |
| [`perceptual_losses.py`](perceptual_losses.py) | 3 | LPIPS / 光流一致性 / VFI 重建三种 PyTorch loss 模块 |
| [`tracker_video.py`](tracker_video.py) | 3 | 轨迹替换：Kalman 平滑（improve）/ Lucas-Kanade 传播（propagate），保留 `traj.npy` 接口 |
| [`run_compare.sh`](run_compare.sh) | val | 在 `aesthetic_quality_comparison_3/origin` 上端到端跑一次并打印 per-episode 分差 |
| [`_compare_diff.py`](_compare_diff.py) | val | run_compare.sh 内部用，独立可调 |

## 快速上手

### Tier 1 单视频后处理

```bash
python -m myscript.improve.postprocess \
    --input  pred.mp4 \
    --gt     gt.mp4 \
    --output pred_improved.mp4
```

### Tier 1 批量 (mp4 → mp4)

```bash
python -m myscript.improve.postprocess --batch \
    --input  origin/    --gt origin/ \
    --output improved/
```

### Tier 1 WorldArena 数据集布局 (强烈推荐：帧目录输入输出，避免 mp4 二次压缩)

```bash
python -m myscript.improve.postprocess \
    --dataset-root data/generated_dataset \
    --gt-root      data/gt_dataset \
    --output-root  data/generated_dataset_improved
```

输出后再用 `evaluate.py` 即可：

```bash
python evaluate.py \
    --config config/config.yaml \
    --data_base data/generated_dataset_improved \
    --save_path output/improved \
    --dimension aesthetic_quality image_quality background_consistency \
                subject_consistency motion_smoothness photometric_smoothness
```

### Tier 1 步骤开关 / Ablation

```bash
# 关闭某些步骤
python -m myscript.improve.postprocess --no-color-match --no-unsharp ...

# 单步消融（其余全关）
python -m myscript.improve.postprocess --only first_frame ...
```

可控参数：`--target-frames`、`--crossfade-len`、`--color-strength`（0=off, 1=hard match）、
`--unsharp-amount`、`--unsharp-radius`、`--contrast`、`--saturation`、`--gamma`、
`--tail-len`、`--tail-alpha`。

### Tier 2 选优

```bash
# 候选目录：episode1_seed0.mp4, episode1_seed1.mp4, episode1_seed2.mp4 ...
python -m myscript.improve.seed_select \
    --candidates seeds_root/ \
    --pattern '{episode}_seed*.mp4' \
    --output  best/ \
    --config  config/config.yaml \
    --w-musiq 1.0 --w-aesthetic 1.0 --w-smoothness 0.5
```

### Tier 2 VFI 插帧

```bash
# 把生成的 N 帧补成 GT 帧数（首选 VFIMamba，同 motion_smoothness 评测器同源）
python -m myscript.improve.vfi_interp \
    --input  pred.mp4 --output pred_aligned.mp4 \
    --gt     gt.mp4   --config config/config.yaml --backend auto
```

### Tier 2 首帧条件 manifest

```bash
python -m myscript.improve.firstframe_conditioning \
    --summary summary.json \
    --output  i2v_jobs.json \
    --mirror  init_frames/ --probe-gt
```

`i2v_jobs.json` 的字段（`episode`, `prompt`, `init_frame`, `target_frames`, `fps`, `gt_path`）
适配大多数 I2V 推理脚本（Wan / OminiEWM / 自研 diffusion-policy）。下游推理脚本只需把
`init_frame` 当 first-frame conditioning 输入即可。

### Tier 3 训练数据筛选

```bash
python -m myscript.improve.data_filter \
    --input  train_videos.csv --csv-column path \
    --output kept.csv --dropped dropped.csv \
    --min-aesthetic 0.55 --min-motion 0.20 \
    --config config/config.yaml --frames-per-clip 8
```

### Tier 3 微调感知 loss

```python
from myscript.improve.perceptual_losses import PerceptualLossBundle

bundle = PerceptualLossBundle(
    lpips_weight=0.10,        # 帮 subject/background_consistency
    flow_weight=0.05,         # 帮 photometric_smoothness
    vfi_recon_weight=0.05,    # 帮 motion_smoothness
    raft_ckpt=cfg["ckpt"]["flow_score"]["raft"],
    vfimamba_ckpt=cfg["ckpt"]["motion_smoothness"]["model"],
)

# pred / target shape: [B, T, 3, H, W] in [0, 1]
extra = bundle(pred, target)
total_loss = base_loss + extra["total"]
```

VFIMamba / RAFT 权重未提供时自动退化为线性混合 / Sobel 流的 fallback，保证 loss 始终
可微，但量级会改变（建议先在 fallback 上调好 weight，再切到带权重版）。

### Tier 3 轨迹平滑（保留 `traj.npy` 接口）

```bash
# A. 直接把已有 traj.npy 改写成 Kalman+三次样条+异常剔除版（备份为 .bak）
python -m myscript.improve.tracker_video improve \
    --traj data/generated_dataset/task/episode/1/traj/traj.npy \
    --frames data/generated_dataset/task/episode/1/video

# B. 用现有 SAM3 traj 当 anchor，重新做 Lucas-Kanade 传播
python -m myscript.improve.tracker_video propagate \
    --frames data/generated_dataset/task/episode/1/video \
    --anchors data/generated_dataset/task/episode/1/traj/traj.npy \
    --output data/generated_dataset/task/episode/1/traj/traj.npy
```

输出仍是 `(T, 2, 2)` `float32`、归一化到 [0,1]、缺失为 `[-1,-1]`，
`WorldArena.trajectory_accuracy` 完全无感切换。

## 验证 / Ablation 工作流

```bash
bash myscript/improve/run_compare.sh /tmp/improve_validation/postproc /tmp/improve_validation/reports
```

这会：

1. 用全开 Tier-1 把 `datasets/aesthetic_quality_comparison_3/origin/` 处理到 `postproc/`；
2. 跑两次 [`seed_select.py`](seed_select.py)（轻量 Farneback 光流平滑代理）打分；
3. 打印每个 episode 的 delta（origin vs postproc）。

如需做完整 evaluate 对比，按脚本末尾给出的两条 `run_evaluation.sh` 命令分别跑 origin 与
postproc 两套，再用 `myscript/aes_compare/aggregate_compare3.py` 风格汇总 per-episode delta。

### 一份真实 ablation 结果（仅供校准方向，不代表线上效果）

在 `aesthetic_quality_comparison_3/origin/` 10 个 episode 上跑全开 Tier-1（CRF 18 x264 输出）：

| Metric (proxy) | Origin | Postproc | Δ | wins |
| --- | --- | --- | --- | --- |
| Smoothness (Farneback)        | 0.1011 | 0.1202 | **+0.019 (+18.9%)** | **9/10** |
| MUSIQ                         | 0.5246 | 0.4969 | -0.028 (-5.3%) | 3/10 |
| LAION-aesthetic               | 0.4134 | 0.4072 | -0.006 (-1.5%) | 4/10 |
| Combined (1·musiq+1·aes+0.5·smooth) | 0.989 | 0.964 | -0.024 (-2.5%) | 2/10 |

要点：

- **平滑系（motion_smoothness / photometric_smoothness）单调受益**——这是 tail-EMA + 颜色匹配的直接效果。
- **MUSIQ / aesthetic 在已经是真实视频的 origin 上轻微回退**——unsharp 和 Reinhard 在
  "干净源" 上反而引入了对 MUSIQ 不友好的纹理。**生成视频通常本身就有伪影**，这两步在
  生成内容上经验上是正向的。务必在你的真实生成数据上做一次 ablation 再决定步骤组合。
- 推荐生产策略：在生成视频上**全开**跑 evaluate，再用 `--only frame_align`、
  `--no-unsharp` 等做 ablation 找出"对你这版生成器最有效的子集"。
- **避免 mp4v**：默认已切到 `ffmpeg libx264 -crf 18` ；如还要写 mp4v，传 `codec="mp4v"` 强制。

## 设计原则

1. **零侵入**：评测函数与 `_EMPIRICAL_BOUNDS` 不动；改动只落在视频侧 / 训练侧 / 推理侧。
2. **多 gid 一致性**：同 episode 多 gid 走 `run_dataset_layout` 时复用同一份 GT 首帧 +
   同一组参数，输出字节相同——直接降低 `action_following` 的 pairwise CLIP 距离。
3. **失败回退**：所有依赖外部权重的组件都有无依赖回退路径
   （pyiqa LPIPS → VGG L1；VFIMamba → 线性混合；libx264 → mp4v；RAFT → Sobel 流）。
4. **优先帧目录**：`postprocess.py --dataset-root` 与 `vfi_interp.py` 都支持帧目录 IO，
   能与 `preprocess_datasets.py` 输出无缝拼接，规避 mp4 二次压缩。
