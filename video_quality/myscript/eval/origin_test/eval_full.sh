cd /mydir/code/WorldArena/video_quality

MODEL_NAME=OminiEWM-AC.sr_x4_step-5w
VIDEO_DIR=/mnt/jackzou/AngryBird_eval/example_test
SUMMARY_JSON=./summary.json
CONFIG_PATH=./config/config.yaml

echo "Warming up torch hub cache (resnet34 for SEA-RAFT)..."
RESNET34_CKPT="$HOME/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"
if [ -s "$RESNET34_CKPT" ]; then
    echo ">>> resnet34 weight already cached at $RESNET34_CKPT, skip warm-up."
else
    http_proxy=http://192.168.32.28:18000 https_proxy=http://192.168.32.28:18000 \
        /mydir/envs/WorldArena/bin/python -c "from torchvision.models import resnet34, ResNet34_Weights; resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)"
fi

echo "Running evaluation..."
bash run_evaluation_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness"

echo "Running VLM judge..."
bash run_VLM_judge_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH all

echo "Aggregating results..."
python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full

echo "Done!"
