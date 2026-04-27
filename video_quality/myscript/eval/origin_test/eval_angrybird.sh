cd /mydir/code/WorldArena/video_quality


MODEL_NAME=OminiEWM-AC.sr_x4_step-5w
VIDEO_DIR=/mnt/jackzou/AngryBird_eval/example_test
# 新测试集的 summary.json，含真实 GT 视频路径（由 myscript/build_summary_new_test.py 生成）
# 如果还没生成，先执行：python myscript/build_summary_new_test.py
SUMMARY_JSON=./summary.json
CONFIG_PATH=./config/config.yaml
LIMIT=0


########################################################
# action following 评测（使用 preprocess_datasets_diversity.py）
########################################################

bash run_action_following.sh $MODEL_NAME $VIDEO_DIR $SUMMARY_JSON $CONFIG_PATH $LIMIT


########################################################
# 结果聚合
########################################################

python csv_results/aggregate_results.py --model_name $MODEL_NAME --base_dir .
python myscript/summarize_csv.py csv_results/$MODEL_NAME/aggregated_results.csv --with-header --full
