cd /mydir/code/WorldArena/video_quality

# ============================================================
# 准备符号链接目录：将 sample_NNNNNN_xxx.mp4 映射为 episodeK.mp4
# 用法: bash myscript/make_symlinks.sh <源视频目录> <链接输出目录> [后缀，默认 _concat]
# 只需运行一次，后续评测直接用链接目录即可
# ============================================================

# --- VACE 模型示例 ---
VACE_SRC=/mnt/world_foundational_model/jackzou/OminiEWM/infer_output/0411_vace_action_robotwin_worldarena_action_161_finetune_step902500/step-902500_infer/videos
VACE_LINKS=/mydir/code/WorldArena/video_quality/data/vace_links

bash myscript/make_symlinks.sh "$VACE_SRC" "$VACE_LINKS" _concat

# 单 GPU 评测（用链接目录作为 GEN_VIDEO_DIR）
bash run_evaluation.sh \
  vace \
  "$VACE_LINKS" \
  ./summary.json \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness" \
  10

# 多 GPU 评测
bash run_evaluation_multi_gpu.sh vace \
  "$VACE_LINKS" \
  ./summary.json \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness" \
  10

# 汇总结果
python csv_results/aggregate_results.py --model_name vace --base_dir .


