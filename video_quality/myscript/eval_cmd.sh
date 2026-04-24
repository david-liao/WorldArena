# /mnt/xukai/code/OminiEWM/infer_output/step-645000-ema_infer/videos
# MODEL_NAME=
# MODEL_NAME=worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions
# VIDEO_DIR=/mnt/xukai/code/ckp/OminiEWM/infer_output/20260416/worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions/step-655000/step-655000-ema_infer/videos

MODEL_NAME=OminiEWM_0422_newtest
VIDEO_DIR=/mnt/jackzou/ckp/OminiEWM/infer_output/20260422/worldarena_robotwin2_0419_vace_action_robotwin_frame_121_640p_test_dataset_instructions/step-792500/step-792500_infer/videos
SUMMARY_JSON=./summary_new_test.json

echo "Running evaluation..."
bash run_evaluation_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON \
  "image_quality,aesthetic_quality,background_consistency,subject_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,semantic_alignment,depth_accuracy"

echo "Running VLM judge..."
bash run_VLM_judge_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON

bash run_evaluation_JEPA.sh $MODEL_NAME $VIDEO_DIR /mydir/code/WorldArena/datasets/new_test_dataset/gt_video/fixed_scene_task

echo "Aggregating results..."
python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .

echo "Summarizing results..."
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header -o csv_results/$MODEL_NAME/summary.tsv

echo "Done!"
