cd /mydir/code/WorldArena/video_quality
# root dir
# /mnt/jackzou/ckp/OminiEWM/infer_output/20260510/worldarena_robotwin2_0509_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000

GT_DIR=/mnt/jackzou/ckp/OminiEWM/infer_output/20260510/worldarena_robotwin2_0509_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000/step-985000/step-985000_infer/videos
AngryBird_DIR=/mnt/jackzou/WorldArena_Submission/AngryBird_eval/example_test
PelicanUnify_DIR=/mnt/jackzou/WorldArena_Submission/Pelican-Unify_eval/Pelican-Unify_test
PelicanUnified_DIR=/mnt/jackzou/WorldArena_Submission/Pelican-Unified_eval/Pelican-Unified_test

VIDEO_DIR=/mnt/jackzou/ckp/OminiEWM/infer_output/20260522/worldarena_robotwin2_0521_vace_action_robotwin_frame_121_640p_test_dataset_instructions_1_x4_finetune_155000/step-992500/step-992500_infer/40_frames

#bash myscript/eval/compare/eval_compare_any.sh GT_vs_AngryBird $GT_DIR $AngryBird_DIR
#bash myscript/eval/compare/eval_compare_any.sh GT_vs_PelicanUnify $GT_DIR $PelicanUnify_DIR
bash myscript/eval/compare/eval_compare_any.sh GT_vs_PelicanUnified $GT_DIR $PelicanUnified_DIR

bash myscript/eval/compare/eval_compare_any.sh AngryBird_vs_Video $AngryBird_DIR $VIDEO_DIR

# visualize
python myscript/eval/compare/diff_topk_visualize.py --model_name GT_vs_AngryBird
python myscript/eval/compare/diff_topk_visualize.py --model_name GT_vs_PelicanUnify
python myscript/eval/compare/diff_topk_visualize.py --model_name GT_vs_PelicanUnified