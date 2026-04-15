#!/bin/bash
set -euo pipefail

# Multi-GPU evaluation via shard-and-merge.
#
# Usage: run_evaluation_multi_gpu.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [LIMIT] [CONFIG_PATH] [NGPUS]
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all).
# NGPUS: number of GPUs (0 or omit = auto-detect).
#
# Strategy:
#   1. Preprocess / Resize / Tracking — single process (shared data)
#   2. Shard episodes into per-GPU symlink trees
#   3. Launch one evaluate.py per GPU in parallel (CUDA_VISIBLE_DEVICES)
#   4. Wait for all, then merge shard results into final JSON

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
LIMIT=${5:-0}
CONFIG_PATH=${6:-"./config/config.yaml"}
NGPUS=${7:-0}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [LIMIT] [CONFIG_PATH] [NGPUS]"
    exit 1
fi

# Auto-detect GPU count
if [ "$NGPUS" -le 0 ] 2>/dev/null || [ -z "$NGPUS" ]; then
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NGPUS" -le 0 ] 2>/dev/null; then
        NGPUS=1
    fi
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
echo ">>> GPUs: $NGPUS"

DATA_DIR="./data"
GEN_DATASET_DIR="$DATA_DIR/generated_dataset"
OUTPUT_DIR="./output/$MODEL_NAME"
OUTPUT_DIR_ACTION="./output_action_following/$MODEL_NAME"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"
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
    # === Phase 1: Preprocessing (single process) ===
    echo ">>> Cleaning previous preprocessed data..."
    rm -rf "$DATA_DIR/gt_dataset" "$GEN_DATASET_DIR" "$DATA_DIR"/shard_*

    STEP_START=$SECONDS
    echo ">>> Running Preprocessing..."
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

    # === Phase 2: Create shard symlink trees ===
    STEP_START=$SECONDS

    # Collect all task/episode pairs
    EPISODES=()
    for task_dir in "$GEN_DATASET_DIR"/*/; do
        task_name=$(basename "$task_dir")
        for ep_dir in "$task_dir"*/; do
            ep_name=$(basename "$ep_dir")
            # Skip non-directory entries
            [ -d "$ep_dir" ] || continue
            EPISODES+=("$task_name/$ep_name")
        done
    done

    NUM_EPISODES=${#EPISODES[@]}
    echo ">>> Found $NUM_EPISODES episodes, distributing across $NGPUS GPUs"

    # Round-robin distribute episodes to shards via symlinks
    for (( i=0; i<NGPUS; i++ )); do
        rm -rf "$DATA_DIR/shard_$i"
    done

    for (( idx=0; idx<NUM_EPISODES; idx++ )); do
        shard_id=$((idx % NGPUS))
        task_ep="${EPISODES[$idx]}"
        task_name="${task_ep%%/*}"
        ep_name="${task_ep##*/}"

        shard_task_dir="$DATA_DIR/shard_$shard_id/$task_name"
        mkdir -p "$shard_task_dir"

        # Absolute symlink to the actual episode directory
        src="$(cd "$GEN_DATASET_DIR/$task_name/$ep_name" && pwd)"
        ln -s "$src" "$shard_task_dir/$ep_name"
    done

    # Print shard distribution
    for (( i=0; i<NGPUS; i++ )); do
        count=$(find "$DATA_DIR/shard_$i" -mindepth 2 -maxdepth 2 -type l 2>/dev/null | wc -l)
        echo ">>>   GPU $i: $count episodes"
    done

    # === Phase 3: Launch parallel evaluation ===
    echo ">>> Starting parallel evaluation on $NGPUS GPUs: ${EVAL_METRICS[*]}"
    PIDS=()
    SHARD_RESULTS=()
    for (( i=0; i<NGPUS; i++ )); do
        shard_data="$(cd "$DATA_DIR/shard_$i" && pwd)"
        shard_output="$OUTPUT_DIR/shard_$i"
        mkdir -p "$shard_output"
        SHARD_RESULTS+=("$shard_output/shard_${i}_results.json")

        CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500 + i)) python evaluate.py \
            --dimension ${EVAL_METRICS[@]} \
            --config "$CONFIG_PATH" \
            --overwrite \
            --save_path "$shard_output" \
            --data_base "$shard_data" \
            > "$shard_output/eval.log" 2>&1 &
        PIDS+=($!)
        echo ">>>   GPU $i: PID $! -> $shard_output"
    done

    # Wait for all processes and track failures
    FAILED=0
    for (( i=0; i<NGPUS; i++ )); do
        if ! wait "${PIDS[$i]}"; then
            echo ">>> ERROR: GPU $i (PID ${PIDS[$i]}) failed. Log: $OUTPUT_DIR/shard_$i/eval.log"
            FAILED=$((FAILED + 1))
        else
            echo ">>>   GPU $i: done"
        fi
    done
    echo ">>> [Parallel Evaluation] $(fmt_elapsed $((SECONDS - STEP_START)))"

    if [ "$FAILED" -gt 0 ]; then
        echo ">>> $FAILED/$NGPUS shards failed. Check logs above."
        exit 1
    fi

    # === Phase 4: Merge results ===
    echo ">>> Merging shard results..."
    python merge_results.py -o "$OUTPUT_DIR/generated_results.json" "${SHARD_RESULTS[@]}"

    # Clean up shard symlink trees (keep shard results for debugging)
    for (( i=0; i<NGPUS; i++ )); do
        rm -rf "$DATA_DIR/shard_$i"
    done
fi

echo ""
echo ">>> All evaluations finished ($NGPUS GPUs) — total $(fmt_elapsed $((SECONDS - TOTAL_START)))"
