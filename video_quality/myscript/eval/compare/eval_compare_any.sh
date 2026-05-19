#!/bin/bash
# ============================================================================
# 通用 trajectory_accuracy 对比 launcher (任意两个视频目录)
# ============================================================================
#
# 用法:
#   bash myscript/eval/compare/eval_compare_any.sh \
#       <NAME> <DIR_A> <DIR_B> [LIMIT] [NGPUS] [VISUALIZE]
#
#   NAME       实验名 (字母/数字/下划线). 会作为 model_name / 数据/结果目录隔离标识.
#   DIR_A      作为 GT 的视频目录 (.mp4 文件).
#   DIR_B      作为 Pred 的视频目录, 文件名需与 A 一致.
#   LIMIT      仅评测前 N 对 (默认 0 = 全量).
#   NGPUS      GPU 数  (默认 0 = 自动检测).
#   VISUALIZE  1 = 跑完自动生成 Top-5 高/低分对比视频, 0 = 否 (默认 0).
#
# 例:
#   bash myscript/eval/compare/eval_compare_any.sh \
#       pelican_unify_vs_unified \
#       /mnt/.../Pelican-Unify_test \
#       /mnt/.../Pelican-Unified_test \
#       0 0 1
#
# 注意:
#   - A 和 B 共享文件名 (episodeK.mp4 形式). 若不一致, build_compare_assets.py 会
#     提示 missing 数量并仅评测匹配部分.
#   - 不同 NAME 互不污染: 数据落 data_trj_acc_compare/<NAME>/, 结果落 output/<NAME>/.
#   - 同一 NAME 重跑会清空 generated_dataset 触发完整重跑 (SAM3 较慢, 谨慎).
# ============================================================================
set -euo pipefail
cd /mydir/code/WorldArena/video_quality

if [ $# -lt 3 ]; then
    sed -n '4,28p' "$0"
    exit 1
fi

NAME=$1
DIR_A=$2
DIR_B=$3
LIMIT=${4:-0}
NGPUS=${5:-0}
VISUALIZE=${6:-0}

echo "============================================================"
echo ">>> Compare run: $NAME"
echo "      A (GT)   = $DIR_A"
echo "      B (Pred) = $DIR_B"
echo "      LIMIT    = $LIMIT       NGPUS = $NGPUS       VISUALIZE = $VISUALIZE"
echo "============================================================"

# 1. 生成 summary + config
python myscript/eval/compare/build_compare_assets.py \
    --name "$NAME" --dir_a "$DIR_A" --dir_b "$DIR_B"

SUMMARY="./summary/summary_${NAME}.json"
CONFIG="./config/config_${NAME}.yaml"

# 2. 跑标准评测流水线 (仅 trajectory_accuracy)
bash run_evaluation_multi_gpu.sh \
    "$NAME" "$DIR_B" "$SUMMARY" "$CONFIG" "trajectory_accuracy" "$LIMIT" "$NGPUS"

# 3. 汇总
python csv_results/aggregate_results.py --model_name "$NAME" --base_dir .
python myscript/summarize_csv.py "csv_results/${NAME}/aggregated_results.csv" --with-header --full

# 4. 可选: 自动生成 Top-K 对比视频
if [ "$VISUALIZE" = "1" ]; then
    python myscript/eval/compare/diff_topk_visualize.py \
        --model_name "$NAME" --top_k 5 --also_high
fi

echo ""
echo "============================================================"
echo ">>> Done: $NAME"
echo "      result json     : output/${NAME}/generated_results.json"
echo "      aggregated csv  : csv_results/${NAME}/aggregated_results.csv"
echo "      summary tsv     : csv_results/${NAME}/summary_full.tsv"
if [ "$VISUALIZE" = "1" ]; then
    echo "      visualize dir   : data_trj_acc_compare/${NAME}/diff_visualize/${NAME}/"
fi
echo "============================================================"
