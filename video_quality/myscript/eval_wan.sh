cd /mydir/code/WorldArena/video_quality

########################################################
# 标准评测
########################################################

# --- WAN 模型示例 ---
bash run_evaluation.sh \
  wan \
  /mydir/code/WorldArena/embodied_task/output/wan_test \
  ./summary.json \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness" \
  10


# 多 GPU 版本
bash run_evaluation_multi_gpu.sh wan \
  /mydir/code/WorldArena/embodied_task/output/wan_test \
  ./summary.json \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness" \
  10

bash run_evaluation_multi_gpu.sh wan \
  /mydir/code/WorldArena/embodied_task/output/wan_test \
  ./summary.json \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness"


########################################################
# VLM 评测
########################################################

bash run_VLM_judge.sh wan /mydir/code/WorldArena/embodied_task/output/wan_test ./summary.json all "" 2

# 多 GPU 版本，使用多个 GPU 进行评测，自动检测 GPU 数量
bash run_VLM_judge_multi_gpu.sh wan /mydir/code/WorldArena/embodied_task/output/wan_test ./summary.json all "" 2


########################################################
# action following 评测
########################################################

bash run_action_following.sh wan /mydir/code/WorldArena/embodied_task/output/wan_test ./summary.json 40

########################################################
# 结果聚合
########################################################

python csv_results/aggregate_results.py --model_name wan --base_dir .
python myscript/summarize_csv.py csv_results/wan/aggregated_results.csv --with-header