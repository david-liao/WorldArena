#!/bin/bash
set -euo pipefail

# Usage: run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [LIMIT] [CONFIG_PATH]
# METRIC_LIST example: "image_quality,photometric_smoothness,action_following"
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all). Useful for debugging.

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
LIMIT=${5:-0}
CONFIG_PATH=${6:-"./config/config.yaml"}
if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [LIMIT] [CONFIG_PATH]"
    exit 1
fi

if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    echo ">>> DEBUG MODE: limiting to first $LIMIT videos"
fi

fmt_elapsed() {
    local secs=$1
    printf "%dm%02ds" $((secs / 60)) $((secs % 60))
}

TOTAL_START=$SECONDS

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mydir/envs/WorldArena
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Parse metrics
CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
echo ">>> Input metrics: $RAW_METRICS"
echo ">>> Formatted for evaluate.py: ${METRIC_ARRAY[*]}"

DATA_DIR="./data"
CONFIG_DIR="./config"
OUTPUT_DIR="./output/$MODEL_NAME"
OUTPUT_DIR_ACTION="./output_action_following/$MODEL_NAME"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"
echo ">>> Results will be saved to: $OUTPUT_DIR"

# Split metrics
EVAL_METRICS=()
RUN_ACTION=false
RUN_TRACKING=false
for metric in "${METRIC_ARRAY[@]}"; do
    if [ "$metric" == "action_following" ]; then
        RUN_ACTION=true
    else
        EVAL_METRICS+=("$metric")
    fi
    if [ "$metric" == "trajectory_accuracy" ]; then
        RUN_TRACKING=true
    fi
done

# Standard metrics
if [ ${#EVAL_METRICS[@]} -gt 0 ]; then
    echo ">>> Cleaning previous preprocessed data..."
    rm -rf "$DATA_DIR/gt_dataset" "$DATA_DIR/generated_dataset"

    STEP_START=$SECONDS
    echo ">>> Running Preprocessing for standard metrics..."
    LIMIT_FLAG=""
    if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
        LIMIT_FLAG="--limit $LIMIT"
    fi
    python preprocess_datasets.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR" $LIMIT_FLAG
    echo ">>> [Preprocessing] $(fmt_elapsed $((SECONDS - STEP_START)))"

    STEP_START=$SECONDS
    echo ">>> Running Resize..."
    python ./processing/video_resize.py --config_path "$CONFIG_PATH"
    echo ">>> [Resize] $(fmt_elapsed $((SECONDS - STEP_START)))"

    if [ "$RUN_TRACKING" = true ]; then
        STEP_START=$SECONDS
        echo ">>> Running Detection & Tracking (for trajectory_accuracy)..."
        python ./processing/detection_tracking.py --config_path "$CONFIG_PATH" --detect_gt
        echo ">>> [Detection & Tracking] $(fmt_elapsed $((SECONDS - STEP_START)))"
    else
        echo ">>> Skipping Detection & Tracking (trajectory_accuracy not requested)"
    fi

    STEP_START=$SECONDS
    echo ">>> Starting Standard Evaluation: ${EVAL_METRICS[*]}"
    python evaluate.py --dimension ${EVAL_METRICS[@]} --config "$CONFIG_PATH" --overwrite --save_path "$OUTPUT_DIR"
    echo ">>> [Evaluation] $(fmt_elapsed $((SECONDS - STEP_START)))"
fi

echo ""
echo ">>> ✅ All evaluations finished — total $(fmt_elapsed $((SECONDS - TOTAL_START)))"


