#!/bin/bash
set -euo pipefail

# Multi-GPU evaluation via shard-and-merge.
#
# Usage: run_evaluation_multi_gpu.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <CONFIG_PATH> <METRIC_LIST> [LIMIT] [NGPUS]
# METRIC_LIST example: "image_quality,photometric_smoothness,trajectory_accuracy"
# LIMIT: only preprocess/evaluate the first N videos (0 or omit = all). Useful for debugging.
# NGPUS: number of GPUs (0 or omit = auto-detect).
#
# Strategy:
#   1. Preprocess / Resize / Tracking — single process (shared data)
#      (Auto-skipped when $DATA_DIR/shard_0..N-1 already exist, unless
#       SKIP_PREP=0 is passed. Pass SKIP_PREP=1 to force skip even if sharding
#       looks incomplete; FORCE_PREP=1 to force re-preprocess.)
#   2. Shard episodes into per-GPU symlink trees (skipped together with Phase 1)
#   3. Launch one evaluate.py per GPU in parallel (CUDA_VISIBLE_DEVICES)
#   4. Wait for all, then merge shard results into final JSON
#
# Env:
#   FORCE_GT_CLEAN=1  Force re-extracting GT frames and re-computing GT trajectories.
#                     By default the GT directory is preserved across runs so that
#                     trajectory_accuracy can reuse cached <gt_path>/.../traj/traj.npy
#                     produced by processing/detection_tracking.py (SAM3 is expensive).
#                     Set to 1 whenever the GT dataset or its videos have changed.
#   SKIP_PREP=1       Force skip preprocessing (use existing data/shard_*)
#   FORCE_PREP=1      Force re-run preprocessing even if data/shard_* already exists
#   OVERWRITE=0       Do not pass --overwrite to evaluate.py so it can resume
#                     already-computed dimensions from shard_N_results.json
#                     (default 1 preserves legacy behavior of re-running everything).

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"./config/config.yaml"}
RAW_METRICS=${5:-}
LIMIT=${6:-0}
NGPUS=${7:-0}

SKIP_PREP=${SKIP_PREP:-auto}
FORCE_PREP=${FORCE_PREP:-0}
OVERWRITE=${OVERWRITE:-1}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <CONFIG_PATH> <METRIC_LIST> [LIMIT] [NGPUS]"
    exit 1
fi

# Auto-detect GPU count
if ! [[ "$NGPUS" =~ ^[0-9]+$ ]] || [ "$NGPUS" -le 0 ]; then
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
GEN_DATASET_DIR="$VAL_BASE"

# ---- validate save paths ------------------------------------------------
if [ -z "$SAVE_PATH" ]; then
    echo "ERROR: save_path must be set in $CONFIG_PATH"
    exit 1
fi
if [ -z "$SAVE_PATH_ACTION" ]; then
    SAVE_PATH_ACTION="$SAVE_PATH"
fi

CONFIG_DIR="./config"
OUTPUT_DIR="${SAVE_PATH%/}/$MODEL_NAME"
OUTPUT_DIR_ACTION="${SAVE_PATH_ACTION%/}/$MODEL_NAME"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"

echo ">>> Data dir (from config):   $DATA_DIR"
echo ">>> Generated dataset:        $GEN_DATASET_DIR"
echo ">>> GT dataset:               $GT_PATH"
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
    # === Decide whether preprocessing + sharding can be skipped ===
    PREP_NEEDED=1
    if [ "$FORCE_PREP" = "1" ]; then
        PREP_NEEDED=1
        echo ">>> FORCE_PREP=1 — re-running preprocessing"
    elif [ "$SKIP_PREP" = "1" ]; then
        PREP_NEEDED=0
        echo ">>> SKIP_PREP=1 — reusing existing $DATA_DIR/shard_* as-is"
    else
        # auto-detect: all $NGPUS shards present AND generated_dataset + gt_dataset exist
        ALL_SHARDS_OK=1
        for (( i=0; i<NGPUS; i++ )); do
            if [ ! -d "$DATA_DIR/shard_$i" ]; then
                ALL_SHARDS_OK=0
                break
            fi
        done
        if [ "$ALL_SHARDS_OK" = "1" ] && [ -d "$GEN_DATASET_DIR" ] && [ -d "$GT_PATH" ]; then
            PREP_NEEDED=0
            echo ">>> Found existing $GEN_DATASET_DIR + $DATA_DIR/shard_0..$((NGPUS-1)); skipping Preprocessing/Resize/Shard."
            echo ">>> (Pass FORCE_PREP=1 to regenerate them.)"
        fi
    fi

    if [ "$PREP_NEEDED" = "1" ]; then
        # === Phase 1: Preprocessing (single process) ===
        echo ">>> Cleaning previous preprocessed data..."
        echo "    - $VAL_BASE"
        rm -rf "$VAL_BASE" "$DATA_DIR"/shard_*
        if [ "${FORCE_GT_CLEAN:-0}" = "1" ]; then
            echo "    - $GT_PATH (FORCE_GT_CLEAN=1)"
            rm -rf "$GT_PATH"
        else
            echo "    - (preserving $GT_PATH to reuse GT trajectory cache; set FORCE_GT_CLEAN=1 to invalidate)"
        fi

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
            echo ">>> Running Detection & Tracking on $NGPUS GPUs (for trajectory_accuracy)..."
            # SAM3 gripper detection is the heaviest preprocessing step.
            # We shard (GT + generated) work items across GPUs: each process
            # loads its own SAM3 copy, sees a single visible device, and writes
            # trajectories to disjoint episode/gid subdirectories, so no IPC
            # or locking is required.
            TRACK_LOG_DIR="$DATA_DIR/tracking_logs"
            mkdir -p "$TRACK_LOG_DIR"
            TRACK_PIDS=()
            for (( i=0; i<NGPUS; i++ )); do
                CUDA_VISIBLE_DEVICES=$i python ./processing/detection_tracking.py \
                    --config_path "$CONFIG_PATH" \
                    --detect_gt \
                    --shard_id "$i" \
                    --num_shards "$NGPUS" \
                    > "$TRACK_LOG_DIR/shard_$i.log" 2>&1 &
                TRACK_PIDS+=($!)
                echo ">>>   GPU $i: PID $! -> $TRACK_LOG_DIR/shard_$i.log"
            done

            TRACK_FAILED=0
            for (( i=0; i<NGPUS; i++ )); do
                if ! wait "${TRACK_PIDS[$i]}"; then
                    echo ">>> ERROR: tracking shard $i (PID ${TRACK_PIDS[$i]}) failed. Log: $TRACK_LOG_DIR/shard_$i.log"
                    TRACK_FAILED=$((TRACK_FAILED + 1))
                else
                    echo ">>>   GPU $i: tracking done"
                fi
            done
            if [ "$TRACK_FAILED" -gt 0 ]; then
                echo ">>> $TRACK_FAILED/$NGPUS tracking shards failed. Check logs in $TRACK_LOG_DIR."
                exit 1
            fi
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
        echo ">>> [Shard] $(fmt_elapsed $((SECONDS - STEP_START)))"
    fi

    # Print shard distribution
    for (( i=0; i<NGPUS; i++ )); do
        count=$(find "$DATA_DIR/shard_$i" -mindepth 2 -maxdepth 2 -type l 2>/dev/null | wc -l)
        echo ">>>   GPU $i: $count episodes"
    done

    # === Phase 3: Launch parallel evaluation ===
    STEP_START=$SECONDS
    echo ">>> Starting parallel evaluation on $NGPUS GPUs: ${EVAL_METRICS[*]}"
    PIDS=()
    SHARD_RESULTS=()
    for (( i=0; i<NGPUS; i++ )); do
        shard_data="$(cd "$DATA_DIR/shard_$i" && pwd)"
        shard_output="$OUTPUT_DIR/shard_$i"
        mkdir -p "$shard_output"
        SHARD_RESULTS+=("$shard_output/shard_${i}_results.json")

        OVERWRITE_FLAG=""
        if [ "$OVERWRITE" = "1" ]; then
            OVERWRITE_FLAG="--overwrite"
        fi
        CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500 + i)) python evaluate.py \
            --dimension ${EVAL_METRICS[@]} \
            --config "$CONFIG_PATH" \
            $OVERWRITE_FLAG \
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
