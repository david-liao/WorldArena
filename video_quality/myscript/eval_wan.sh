cd /mydir/code/WorldArena/video_quality

# MODEL_NAME=wan
# VIDEO_DIR=/mydir/code/WorldArena/embodied_task/output/wan_test
# # 原占位 summary（gt_path 是假路径，不能跑 GT 相关指标）
# SUMMARY_JSON=./summary.json
# CONFIG_PATH=./config/config.yaml


MODEL_NAME=wan_newtest
VIDEO_DIR=/mydir/code/WorldArena/embodied_task/output/wan_newtest
# 新测试集的 summary.json，含真实 GT 视频路径（由 myscript/build_summary_new_test.py 生成）
# 如果还没生成，先执行：python myscript/build_summary_new_test.py
SUMMARY_JSON=./summary_new_test.json
CONFIG_PATH=./config/config_newtest.yaml
LIMIT=1


########################################################
# 标准评测（8 个无需 GT 的指标）
########################################################

# --- WAN 模型示例 ---
# bash run_evaluation.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH \
#   "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness" \
#   10


########################################################
# GT 相关指标（4 个，需要真实 GT 视频 → 必须用 summary_new_test.json）
#   - trajectory_accuracy : SAM3 检测 GT 轨迹
#   - semantic_alignment  : Qwen2.5-VL caption + CLIPScore
#   - depth_accuracy      : Depth-Anything 对齐
#   - psnr / ssim         : 像素级对齐（evaluate.py 内部会合并成 psnr_ssim）
#
# 建议先用 LIMIT=10 小样本冒烟，全量 1000 条跑 trajectory_accuracy 比较耗时
########################################################

# bash run_evaluation.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH \
#   "trajectory_accuracy,semantic_alignment,depth_accuracy" \
#   1



########################################################
# 全指标评测 11个指标
########################################################
bash run_evaluation.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,trajectory_accuracy,semantic_alignment,depth_accuracy" \
  $LIMIT


########################################################
# VLM 评测
########################################################

bash run_VLM_judge.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH all $LIMIT



########################################################
# action following 评测（使用 preprocess_datasets_diversity.py）
########################################################

bash run_action_following.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH $LIMIT

########################################################
# JEPA 相似度（直接吃 GT 目录，不走 summary.json）
# 必须在 video_quality/ 目录下执行
#   参数1：MODEL_NAME，结果保存在 output_JEDi/<MODEL_NAME>/
#   参数2：生成视频目录（mp4 扁平平铺）
#   参数3：GT 视频目录（默认 new_test_dataset/gt_video/fixed_scene_task）
#   参数4：MAX_SAMPLES，0=全量，>0 只跑前 N 对（调试用）
# 两边通过文件名 stem 对齐（episode{N}.mp4）
########################################################

# 调试：只跑前 $LIMIT 对
bash run_evaluation_JEPA.sh $MODEL_NAME $VIDEO_DIR /mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task $LIMIT

# 全量
# bash run_evaluation_JEPA.sh $MODEL_NAME $VIDEO_DIR /mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task

########################################################
# 结果聚合
########################################################

python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full
