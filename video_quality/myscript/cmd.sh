cd /mydir/code/WorldArena/video_quality
python myscript/aggregate_by_task.py csv_results/OminiEWM-AC（step-3w）/qwen_757500.csv --tsv

# 自动评估
bash myscript/eval/auto_eval.sh \
    /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/worldarena_robotwin2_0427_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    /mnt/jackzou/WorldArena/results \
    --once --max-runs 1 --max-videos 2