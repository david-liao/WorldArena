#!/bin/bash
set -euo pipefail

# Multi-GPU VLM judge evaluation via shard-and-merge.
#
# Usage: run_VLM_judge_multi_gpu.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> [METRICS] [CONFIG_PATH] [MAX_VIDEOS] [NGPUS]
#
# Strategy:
#   1. Auto-detect GPU count (or use provided NGPUS)
#   2. Launch one VLM_judge.py per GPU with --shard_id/--num_shards
#   3. Wait for all, then merge shard results into final JSON

MODEL_NAME=${1:?model name required}
VIDEO_DIR=${2:?video dir required}
SUMMARY_JSON=${3:?summary json required}
CONFIG_PATH=${4:-}
METRICS=${5:-all}
MAX_VIDEOS=${6:-0}
NGPUS=${7:-0}

# Allow MODEL_NAME to be a subpath (e.g. "run_name/chunk0") so multiple chunks
# can share a single parent directory under output_VLM/. File names always use
# the leaf segment to avoid nested duplication. Plain names (no slash) behave
# identically to before since basename(x) == x.
LEAF_NAME=$(basename "$MODEL_NAME")

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$ROOT_DIR/VLM_judge.py"
OUTPUT_ROOT="$ROOT_DIR/output_VLM"
TMP_ROOT="$ROOT_DIR/tmp_VLM"
DEFAULT_CONFIG="$ROOT_DIR/config/config.yaml"
CONFIG_ARG=${CONFIG_PATH:-$DEFAULT_CONFIG}

export PYTORCH_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mydir/envs/WorldArena_VLM

if [ "$NGPUS" -le 0 ] 2>/dev/null || [ -z "$NGPUS" ]; then
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NGPUS" -le 0 ] 2>/dev/null; then
        NGPUS=1
    fi
fi

fmt_elapsed() {
    local secs=$1
    printf "%dm%02ds" $((secs / 60)) $((secs % 60))
}

MAX_VIDEOS_ARG=()
if [[ "$MAX_VIDEOS" -gt 0 ]]; then
    MAX_VIDEOS_ARG=(--max_videos "$MAX_VIDEOS")
    echo ">>> DEBUG MODE: limiting to first $MAX_VIDEOS videos"
fi

echo ">>> Model: $MODEL_NAME"
echo ">>> GPUs:  $NGPUS"
echo ">>> Video: $VIDEO_DIR"

TOTAL_START=$SECONDS

if [ "$NGPUS" -le 1 ]; then
    echo ">>> Single GPU mode"
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
else
    echo ">>> Launching $NGPUS parallel shards..."
    LOG_DIR="$OUTPUT_ROOT/$MODEL_NAME"
    mkdir -p "$LOG_DIR"

    PIDS=()
    SHARD_FILES=()
    for (( i=0; i<NGPUS; i++ )); do
        shard_log="$LOG_DIR/shard_${i}.log"
        SHARD_FILES+=("$LOG_DIR/${LEAF_NAME}_summary_val_all_intern_shard${i}.json")

        CUDA_VISIBLE_DEVICES=$i python3 "$PY" \
            --model_name "$MODEL_NAME" \
            --video_dir "$VIDEO_DIR" \
            --summary_json "$SUMMARY_JSON" \
            --metrics "$METRICS" \
            --num_frames 16 \
            --output_root "$OUTPUT_ROOT" \
            --tmp_root "$TMP_ROOT" \
            --config_path "$CONFIG_ARG" \
            --shard_id "$i" \
            --num_shards "$NGPUS" \
            "${MAX_VIDEOS_ARG[@]}" \
            > "$shard_log" 2>&1 &
        PIDS+=($!)
        echo ">>>   GPU $i: PID $! -> $shard_log"
    done

    FAILED=0
    for (( i=0; i<NGPUS; i++ )); do
        if ! wait "${PIDS[$i]}"; then
            echo ">>> ERROR: GPU $i (PID ${PIDS[$i]}) failed. Log: $LOG_DIR/shard_${i}.log"
            FAILED=$((FAILED + 1))
        else
            echo ">>>   GPU $i: done"
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo ">>> $FAILED/$NGPUS shards failed. Check logs above."
        exit 1
    fi

    # Merge shard results
    MERGED="$LOG_DIR/${LEAF_NAME}_summary_val_all_intern.json"
    python3 -c "
import json, sys
merged = []
for path in sys.argv[1:]:
    with open(path) as f:
        merged.extend(json.load(f))
merged.sort(key=lambda x: x['video'])
with open('$MERGED', 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f'Merged {len(merged)} results -> $MERGED')
" "${SHARD_FILES[@]}"

    # Clean up shard files
    for f in "${SHARD_FILES[@]}"; do
        rm -f "$f"
    done
fi

echo ""
echo ">>> All done ($NGPUS GPUs) — total $(fmt_elapsed $((SECONDS - TOTAL_START)))"
