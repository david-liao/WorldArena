#!/bin/bash
set -euo pipefail
# Usage:
#   run_evaluation_JEPA.sh <MODEL_NAME> <GEN_VIDEO_DIR> [REAL_VIDEO_DIR] [MAX_SAMPLES]
#   - MODEL_NAME       结果会保存到 output_JEDi/<MODEL_NAME>/ 下
#   - GEN_VIDEO_DIR    生成视频目录（扁平平铺的 mp4）
#   - REAL_VIDEO_DIR   GT 视频目录，默认指向 new_test_dataset 的 GT
#   - MAX_SAMPLES      默认 0（全量），>0 时只处理前 N 对，方便调试

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
REAL_VIDEO_DIR=${3:-/mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task}
MAX_SAMPLES=${4:-0}

if [[ -z "$MODEL_NAME" || -z "$GEN_VIDEO_DIR" ]]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> [REAL_VIDEO_DIR] [MAX_SAMPLES]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="$SCRIPT_DIR/output_JEDi"

cd "$SCRIPT_DIR/JEDi"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mydir/envs/WorldArena_JEPA
export PATH="/mydir/envs/WorldArena_JEPA/bin:$PATH"

python batch.py \
    --model_name "$MODEL_NAME" \
    --real_dir "$REAL_VIDEO_DIR" \
    --gen_dir "$GEN_VIDEO_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --output_root "$OUTPUT_ROOT"
