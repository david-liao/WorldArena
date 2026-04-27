#!/bin/bash
set -euo pipefail

# Usage: run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [LIMIT] [CONFIG_PATH]
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all). Useful for debugging.

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"./config/config.yaml"}
LIMIT=${5:-0}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ]; then
  echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [LIMIT] [CONFIG_PATH]"
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
# export PATH="your absolute path:$PATH"

# Derive all paths from config.yaml (single source of truth).
# Two concurrent runs can share this code directory as long as their config
# files point to different data_action_following / save_path_action_following.
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config file not found: $CONFIG_PATH"
  exit 1
fi

eval "$(python - "$CONFIG_PATH" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f) or {}
da = c.get('data_action_following', {}) or {}
out = {
    'GT_PATH_ACTION':   da.get('gt_path',  ''),
    'VAL_BASE_ACTION':  da.get('val_base', ''),
    'SAVE_PATH_ACTION': c.get('save_path_action_following', '') or c.get('save_path', ''),
}
for k, v in out.items():
    print(f'{k}={shlex.quote(str(v))}')
PY
)"

if [ -z "$GT_PATH_ACTION" ] || [ -z "$VAL_BASE_ACTION" ]; then
  echo "ERROR: data_action_following.gt_path and data_action_following.val_base must be set in $CONFIG_PATH"
  exit 1
fi
GT_PARENT_ACTION=$(dirname "$GT_PATH_ACTION")
VAL_PARENT_ACTION=$(dirname "$VAL_BASE_ACTION")
if [ "$GT_PARENT_ACTION" != "$VAL_PARENT_ACTION" ]; then
  echo "ERROR: data_action_following.gt_path and data_action_following.val_base must share the same parent directory."
  echo "  gt_path  = $GT_PATH_ACTION   (parent: $GT_PARENT_ACTION)"
  echo "  val_base = $VAL_BASE_ACTION (parent: $VAL_PARENT_ACTION)"
  echo "  preprocess_datasets_diversity.py writes to <parent>/gt_dataset and <parent>/generated_dataset,"
  echo "  so both paths must sit under the same parent."
  exit 1
fi
if [ "$(basename "$GT_PATH_ACTION")" != "gt_dataset" ] || [ "$(basename "$VAL_BASE_ACTION")" != "generated_dataset" ]; then
  echo "ERROR: expected data_action_following.gt_path to end with 'gt_dataset' and data_action_following.val_base to end with 'generated_dataset'."
  exit 1
fi
if [ -z "$SAVE_PATH_ACTION" ]; then
  echo "ERROR: save_path_action_following (or save_path as fallback) must be set in $CONFIG_PATH"
  exit 1
fi

DATA_DIR="$GT_PARENT_ACTION"
CONFIG_DIR="./config"
OUTPUT_DIR_ACTION="${SAVE_PATH_ACTION%/}"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR_ACTION"

echo ">>> Data dir (from config):    $DATA_DIR"
echo ">>> Results will be saved to:  $OUTPUT_DIR_ACTION/$MODEL_NAME"

LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LIMIT_FLAG="--limit $LIMIT"
fi

echo ">>> Cleaning previous preprocessed data..."
rm -rf "$DATA_DIR/gt_dataset" "$DATA_DIR/generated_dataset"

STEP_START=$SECONDS
echo ">>> Running action_following preprocessing..."
python preprocess_datasets_diversity.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR" $LIMIT_FLAG
echo ">>> [Preprocessing] $(fmt_elapsed $((SECONDS - STEP_START)))"

STEP_START=$SECONDS
echo ">>> Running action_following evaluation..."
python evaluate.py --dimension "action_following" --config "$CONFIG_PATH" --overwrite --save_path "$OUTPUT_DIR_ACTION/$MODEL_NAME" || echo ">>> [WARNING] evaluate.py (action_following) returned non-zero code"
echo ">>> [Evaluation] $(fmt_elapsed $((SECONDS - STEP_START)))"

echo ""
echo ">>> ✅ Action Following Script Finished — total $(fmt_elapsed $((SECONDS - TOTAL_START)))"
