cd /mydir/code/WorldArena/video_quality

MODEL_NAME=jack_0509_2_ac
VIDEO_DIR=/mnt/jackzou/Eval_0509_2/model_test
SUMMARY_JSON=./summary.json
CONFIG_PATH=./config/config.yaml


########################################################
# action following 评测（使用 preprocess_datasets_diversity.py）
########################################################

echo "Running action following..."
bash run_action_following.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH


########################################################
# 结果聚合
########################################################

echo "Aggregating results..."
python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full

echo "Done!"