cd /work/ai_projects/WorldArena_fix2/video_quality && source /work/ai_projects/venv/WorldArena_VLM/bin/activate && bash run_VLM_judge.sh QWEN_v2_partB1 "/work/ai_projects/5.0/20260409/worldarena_robotwin2_vlm_qwen35_worldarena_ft_20260407_ema_test_dataset_instructions/step-760000/step-760000-ema_infer/videos" "/work/ai_projects/WorldArena_fix2/video_quality/json_score/summary_qwen_v2_step760000_partB1_300_300to599.json"

bash run_evaluation.sh \
  T5_v2_rest700 \
  /work/ai_projects/WorldArena/video_quality/mapped_t5v2_step602500_videos \
  /work/ai_projects/WorldArena/video_quality/summary_t5_v2_remaining_700_for_eval_match.json \
  "image_quality,subject_consistency,aesthetic_quality,background_consistency,dynamic_degree,flow_score,motion_smoothness,photometric_smoothness"