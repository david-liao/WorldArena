#!/bin/bash
set -e

TOTAL=${1:-1000}
NUM_GPUS=8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
cd "$BASE_DIR"

NVIDIA_LIBS=$(python -c "import nvidia; import os; base=os.path.dirname(nvidia.__file__); dirs=[os.path.join(base,d,'lib') for d in os.listdir(base) if os.path.isdir(os.path.join(base,d,'lib'))]; print(':'.join(dirs))")
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$TORCH_LIB"

SCRIPT="$SCRIPT_DIR/batch_generate.py"
DATASET_DIR="/mydir/code/WorldArena/datasets/WorldArena_Robotwin2.0/test_dataset"
CKPT_DIR="./models/Wan2.2-TI2V-5B"
PT_DIR="./models/worldarena_weights/models/wan_video/epoch.pt"
COMMON="--checkpoint_dir $CKPT_DIR --pt_dir $PT_DIR --size 640*480 --frame_num 121 --seed 42"

PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "[INFO] Total: $TOTAL videos, $NUM_GPUS GPUs, ~$PER_GPU per GPU"

# ============ Instruction 1 ============
echo "[INFO] Generating $TOTAL videos with instructions_1 ..."
OUT1="./output/wan_test_1"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    [ $END -gt $TOTAL ] && END=$TOTAL
    [ $START -ge $TOTAL ] && continue
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --instruction_subdir instructions_1 \
        --output_dir "$OUT1" \
        $COMMON --start_idx $START --end_idx $END &
done
wait
echo "[INFO] instructions_1 done! Output: $OUT1"

# ============ Instruction 2 ============
echo "[INFO] Generating $TOTAL videos with instructions_2 ..."
OUT2="./output/wan_test_2"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    [ $END -gt $TOTAL ] && END=$TOTAL
    [ $START -ge $TOTAL ] && continue
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --instruction_subdir instructions_2 \
        --output_dir "$OUT2" \
        $COMMON --start_idx $START --end_idx $END &
done
wait
echo "[INFO] instructions_2 done! Output: $OUT2"

echo "[INFO] All finished!"
echo "  instructions_1 -> $OUT1"
echo "  instructions_2 -> $OUT2"
