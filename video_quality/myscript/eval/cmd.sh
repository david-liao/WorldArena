cd /mydir/code/WorldArena/video_quality
python myscript/aggregate_by_task.py csv_results/OminiEWM-AC（step-3w）/qwen_757500.csv --tsv

# obsolete:自动评估(单节点) (不要用这个，用下面的多节点共享存储)
bash myscript/eval/auto_eval.sh \
    /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/worldarena_robotwin2_0427_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    /mnt/jackzou/WorldArena/results \
    --once --max-runs 1 --max-videos 2

# 自动评估(多节点共享存储)
python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/worldarena_robotwin2_0427_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    --results-dir /mnt/jackzou/WorldArena/results

# 自动评估(多节点共享存储)，目标目录支持通配符
python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260428/worldarena_robotwin2_0427_vace_action_robotwin_frame_121_640p_test_dataset_instructions_x4_finetune_155000/step-955000/step-955000_infer/test_multi_depth \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '*_frames'

python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260510/worldarena_robotwin2_0509_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260511/worldarena_robotwin2_0509_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260511/worldarena_robotwin2_0510_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260512/worldarena_robotwin2_0510_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260512/worldarena_robotwin2_0510_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000.step-950000 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

# 0513
python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/ckp/OminiEWM/infer_output/20260513 \
    --results-dir /mnt/jackzou/WorldArena/results \
    --target-dir-name '40_frames'

# Pelican-Unify submission
python myscript/eval/auto_eval.py \
    --scan-path /mnt/jackzou/WorldArena_Submission/Pelican-Unify_eval \
    --results-dir myscript/eval/results \
    --target-dir-name 'Pelican-Unify_test'
