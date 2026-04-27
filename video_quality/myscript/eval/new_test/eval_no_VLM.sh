# /mnt/xukai/code/OminiEWM/infer_output/step-645000-ema_infer/videos
# MODEL_NAME=
# MODEL_NAME=worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions
# VIDEO_DIR=/mnt/xukai/code/ckp/OminiEWM/infer_output/20260416/worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions/step-655000/step-655000-ema_infer/videos

MODEL_NAME=jack_0425_1
VIDEO_DIR=/mnt/jackzou/ckp/OminiEWM/infer_output/20260425/worldarena_robotwin2_0424_vace_action_robotwin_frame_121_640p_test_dataset_instructions_test_dataset_x4_finetune_200000/step-825000/step-825000_infer/videos
SUMMARY_JSON=./summary_new_test.json
CONFIG_PATH=./config/config_newtest.yaml

echo "Running evaluation..."
bash run_evaluation_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,trajectory_accuracy,semantic_alignment,depth_accuracy"

echo "Running action following..."
bash run_action_following.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH

echo "Running JEPA..."
bash run_evaluation_JEPA.sh $MODEL_NAME $VIDEO_DIR /mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task

echo "Aggregating results..."
python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full

echo "Done!"
