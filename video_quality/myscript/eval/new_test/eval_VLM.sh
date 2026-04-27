# /mnt/xukai/code/OminiEWM/infer_output/step-645000-ema_infer/videos
# MODEL_NAME=
# MODEL_NAME=worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions
# VIDEO_DIR=/mnt/xukai/code/ckp/OminiEWM/infer_output/20260416/worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260416_480p_ema_test_dataset_instructions/step-655000/step-655000-ema_infer/videos

MODEL_NAME=jack_0425
VIDEO_DIR=/mnt/jackzou/ckp/OminiEWM/infer_output/20260425/worldarena_robotwin2_0424_vace_action_robotwin_frame_121_640p_test_dataset_instructions_test_dataset_x4_finetune_200000/step-825000/step-825000_infer/videos
SUMMARY_JSON=./summary_new_test.json
CONFIG_PATH=./config/config_newtest.yaml

echo "Running VLM judge..."
bash run_VLM_judge_multi_gpu.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH

echo "Aggregating results..."
python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full

echo "Done!"
