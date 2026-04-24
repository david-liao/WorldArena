#!/bin/bash
set -euo pipefail

MODEL_NAME=${1:?model name required}
VIDEO_DIR=${2:?video dir required}
SUMMARY_JSON=${3:?summary json required}
# Optional: pass a custom config path; defaults to the repo config
CONFIG_PATH=${4:-}
METRICS=${5:-all}
# Optional: limit to first N videos for debugging (0 = no limit)
MAX_VIDEOS=${6:-0}

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$ROOT_DIR/VLM_judge.py"
OUTPUT_ROOT="$ROOT_DIR/output_VLM"
TMP_ROOT="$ROOT_DIR/tmp_VLM"
DEFAULT_CONFIG="$ROOT_DIR/config/config.yaml"

# Use provided config path if set, otherwise default
CONFIG_ARG=${CONFIG_PATH:-$DEFAULT_CONFIG}

export PYTORCH_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mydir/envs/WorldArena_VLM
# export PATH="your absolute path:$PATH"

MAX_VIDEOS_ARG=()
if [[ "$MAX_VIDEOS" -gt 0 ]]; then
  MAX_VIDEOS_ARG=(--max_videos "$MAX_VIDEOS")
fi
python3 "$PY" \
  --model_name "$MODEL_NAME" \
  --video_dir "$VIDEO_DIR" \
  --summary_json "$SUMMARY_JSON" \
  --metrics "$METRICS" \
  --num_frames 16 \
  --output_root "$OUTPUT_ROOT" \
  --tmp_root "$TMP_ROOT" \
  --config_path "$CONFIG_ARG" \
  "${MAX_VIDEOS_ARG[@]}"
