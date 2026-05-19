#!/bin/bash
# 用途: 对比两个生成视频集合的 trajectory_accuracy
#   - 视频 A (作为 GT):    /mnt/jackzou/WorldArena_Submission/Pelican-Unify_eval/Pelican-Unify_test
#   - 视频 B (作为 pred):  /mnt/jackzou/WorldArena_Submission/Pelican-Unified_eval/Pelican-Unified_test
#
# 配套文件:
#   - summary/summary_compare.json    (列出视频 A 的 mp4 作为 gt_path)
#   - config/config_compare.yaml      (data 路径仍指向旧的 compare_data/)
#
# 重跑提示:
#   - 默认会复用上一次 SAM3 跑出的 GT 轨迹缓存; 如果更换了对比目录, 必须设
#     FORCE_GT_CLEAN=1 + FORCE_PREP=1 重新预处理.
#
set -euo pipefail
cd /mydir/code/WorldArena/video_quality

MODEL_NAME=compare_pelican_unify_vs_unified
# 视频 B 所在目录: preprocess_datasets.py 会按 episode{K}.mp4 文件名去这里找
VIDEO_DIR=/mnt/jackzou/WorldArena_Submission/Pelican-Unified_eval/Pelican-Unified_test
SUMMARY_JSON=./summary/summary_compare.json
CONFIG_PATH=./config/config_compare.yaml

# 只跑轨迹相似度
METRICS="trajectory_accuracy"

echo ">>> Compare:"
echo "      A (GT)  = $(python -c "import json;d=json.load(open('$SUMMARY_JSON'));print(d[0]['gt_path'].rsplit('/',1)[0])")"
echo "      B (pred)= $VIDEO_DIR"
echo "      metric  = $METRICS"

bash run_evaluation_multi_gpu.sh \
    "$MODEL_NAME" "$VIDEO_DIR" "$SUMMARY_JSON" "$CONFIG_PATH" "$METRICS"

echo ">>> Aggregating..."
python csv_results/aggregate_results.py --model_name "$MODEL_NAME" --base_dir .
python myscript/summarize_csv.py csv_results/"$MODEL_NAME"/aggregated_results.csv --with-header --full

echo "Done. Results:"
echo "  - output/$MODEL_NAME/generated_results.json"
echo "  - csv_results/$MODEL_NAME/aggregated_results.csv"
