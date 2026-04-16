#!/bin/bash
set -euo pipefail

# Usage: run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [LIMIT] [CONFIG_PATH]
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all). Useful for debugging.

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
LIMIT=${4:-0}
CONFIG_PATH=${5:-"./config/config.yaml"}

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

DATA_DIR="./data_action_following"
CONFIG_DIR="./config"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR_ACTION"

LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LIMIT_FLAG="--limit $LIMIT"
fi

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
