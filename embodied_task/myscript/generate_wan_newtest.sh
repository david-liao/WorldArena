#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
cd "$BASE_DIR"

NVIDIA_LIBS=$(python -c "import nvidia; import os; base=os.path.dirname(nvidia.__file__); dirs=[os.path.join(base,d,'lib') for d in os.listdir(base) if os.path.isdir(os.path.join(base,d,'lib'))]; print(':'.join(dirs))")
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$TORCH_LIB"

SCRIPT="$SCRIPT_DIR/batch_generate.py"
COMMON="--summary_json /mydir/code/WorldArena/datasets/new_test_dataset/summary.json --output_dir ./output/wan_newtest --checkpoint_dir ./models/Wan2.2-TI2V-5B --pt_dir ./models/worldarena_weights/models/wan_video/epoch.pt --size 640*480 --frame_num 121 --seed 42"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" $COMMON --start_idx 0    --end_idx 125  &
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" $COMMON --start_idx 125  --end_idx 250  &
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" $COMMON --start_idx 250  --end_idx 375  &
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" $COMMON --start_idx 375  --end_idx 500  &
CUDA_VISIBLE_DEVICES=4 python "$SCRIPT" $COMMON --start_idx 500  --end_idx 625  &
CUDA_VISIBLE_DEVICES=5 python "$SCRIPT" $COMMON --start_idx 625  --end_idx 750  &
CUDA_VISIBLE_DEVICES=6 python "$SCRIPT" $COMMON --start_idx 750  --end_idx 875  &
CUDA_VISIBLE_DEVICES=7 python "$SCRIPT" $COMMON --start_idx 875  --end_idx 1000 &

wait
echo "All 8 GPUs finished!"
