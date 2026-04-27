#!/bin/bash
set -euo pipefail

# Usage: run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [LIMIT] [CONFIG_PATH]
# METRIC_LIST example: "image_quality,photometric_smoothness,action_following"
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all). Useful for debugging.
# Env:
#   FORCE_GT_CLEAN=1  Force re-extracting GT frames and re-computing GT trajectories.
#                     By default the GT directory is preserved across runs so that
#                     trajectory_accuracy can reuse cached <gt_path>/.../traj/traj.npy
#                     produced by processing/detection_tracking.py (SAM3 is expensive).
#                     Set to 1 whenever the GT dataset or its videos have changed.

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"./config/config.yaml"}
RAW_METRICS=${5:-}
LIMIT=${6:-0}
if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [LIMIT]"
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

# Derive every path from config.yaml (single source of truth).
# This lets two concurrent runs live in the same code directory as long as
# their config files point to different data / save paths.
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config file not found: $CONFIG_PATH"
    exit 1
fi

eval "$(python - "$CONFIG_PATH" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f) or {}
d  = c.get('data', {}) or {}
da = c.get('data_action_following', {}) or {}
out = {
    'GT_PATH':          d.get('gt_path',  ''),
    'VAL_BASE':         d.get('val_base', ''),
    'GT_PATH_ACTION':   da.get('gt_path',  ''),
    'VAL_BASE_ACTION':  da.get('val_base', ''),
    'SAVE_PATH':        c.get('save_path', ''),
    'SAVE_PATH_ACTION': c.get('save_path_action_following', ''),
}
for k, v in out.items():
    print(f'{k}={shlex.quote(str(v))}')
PY
)"

# ---- validate standard data paths ---------------------------------------
if [ -z "$GT_PATH" ] || [ -z "$VAL_BASE" ]; then
    echo "ERROR: data.gt_path and data.val_base must be set in $CONFIG_PATH"
    exit 1
fi
GT_PARENT=$(dirname "$GT_PATH")
VAL_PARENT=$(dirname "$VAL_BASE")
if [ "$GT_PARENT" != "$VAL_PARENT" ]; then
    echo "ERROR: data.gt_path and data.val_base must share the same parent directory."
    echo "  gt_path  = $GT_PATH   (parent: $GT_PARENT)"
    echo "  val_base = $VAL_BASE (parent: $VAL_PARENT)"
    echo "  preprocess_datasets.py writes to <parent>/gt_dataset and <parent>/generated_dataset,"
    echo "  so both paths must sit under the same parent."
    exit 1
fi
if [ "$(basename "$GT_PATH")" != "gt_dataset" ] || [ "$(basename "$VAL_BASE")" != "generated_dataset" ]; then
    echo "ERROR: expected data.gt_path to end with 'gt_dataset' and data.val_base to end with 'generated_dataset'."
    echo "  gt_path  = $GT_PATH"
    echo "  val_base = $VAL_BASE"
    exit 1
fi
DATA_DIR="$GT_PARENT"

# ---- validate action_following data paths (optional; only if provided) --
DATA_DIR_ACTION=""
if [ -n "$GT_PATH_ACTION" ] || [ -n "$VAL_BASE_ACTION" ]; then
    if [ -z "$GT_PATH_ACTION" ] || [ -z "$VAL_BASE_ACTION" ]; then
        echo "ERROR: data_action_following.gt_path and data_action_following.val_base must both be set (or both omitted)."
        exit 1
    fi
    GT_PARENT_ACTION=$(dirname "$GT_PATH_ACTION")
    VAL_PARENT_ACTION=$(dirname "$VAL_BASE_ACTION")
    if [ "$GT_PARENT_ACTION" != "$VAL_PARENT_ACTION" ]; then
        echo "ERROR: data_action_following.gt_path and data_action_following.val_base must share the same parent directory."
        echo "  gt_path  = $GT_PATH_ACTION   (parent: $GT_PARENT_ACTION)"
        echo "  val_base = $VAL_BASE_ACTION (parent: $VAL_PARENT_ACTION)"
        exit 1
    fi
    if [ "$(basename "$GT_PATH_ACTION")" != "gt_dataset" ] || [ "$(basename "$VAL_BASE_ACTION")" != "generated_dataset" ]; then
        echo "ERROR: expected data_action_following.gt_path to end with 'gt_dataset' and data_action_following.val_base to end with 'generated_dataset'."
        exit 1
    fi
    DATA_DIR_ACTION="$GT_PARENT_ACTION"
fi

# ---- validate save paths ------------------------------------------------
if [ -z "$SAVE_PATH" ]; then
    echo "ERROR: save_path must be set in $CONFIG_PATH"
    exit 1
fi
# save_path_action_following falls back to save_path when not provided
if [ -z "$SAVE_PATH_ACTION" ]; then
    SAVE_PATH_ACTION="$SAVE_PATH"
fi

CONFIG_DIR="./config"
OUTPUT_DIR="${SAVE_PATH%/}/$MODEL_NAME"
OUTPUT_DIR_ACTION="${SAVE_PATH_ACTION%/}/$MODEL_NAME"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"
if [ -n "$DATA_DIR_ACTION" ]; then
    mkdir -p "$DATA_DIR_ACTION"
fi

echo ">>> Data dir (from config):        $DATA_DIR"
if [ -n "$DATA_DIR_ACTION" ]; then
    echo ">>> Data dir action (from config): $DATA_DIR_ACTION"
fi
echo ">>> Results will be saved to:        $OUTPUT_DIR"
echo ">>> Action results will be saved to: $OUTPUT_DIR_ACTION"

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
    echo "    - $VAL_BASE"
    rm -rf "$VAL_BASE"
    if [ "${FORCE_GT_CLEAN:-0}" = "1" ]; then
        echo "    - $GT_PATH (FORCE_GT_CLEAN=1)"
        rm -rf "$GT_PATH"
    else
        echo "    - (preserving $GT_PATH to reuse GT trajectory cache; set FORCE_GT_CLEAN=1 to invalidate)"
    fi

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


