#!/usr/bin/env bash
# Tier 1 validation harness: post-process datasets/aesthetic_quality_comparison_3/origin/
# with myscript/improve/postprocess.py, then run a quick aesthetic + image_quality
# spot-check that does NOT require a full WorldArena.evaluate run.
#
# Usage:
#   bash myscript/improve/run_compare.sh [postproc_out_dir] [report_dir]
#
# Defaults:
#   postproc_out_dir = /tmp/improve_validation/postproc
#   report_dir        = /tmp/improve_validation/reports
#
# Then for a full WorldArena.evaluate run on the new directory:
#   bash run_evaluation.sh test_postproc <postproc_out_dir> summary.json \
#        config/.tmp.yaml aesthetic_quality,image_quality 10
set -euo pipefail

ROOT="/mydir/code/WorldArena/video_quality"
ORIGIN="/mydir/code/WorldArena/datasets/aesthetic_quality_comparison_3/origin"
OUT_POSTPROC="${1:-/tmp/improve_validation/postproc}"
OUT_REPORT="${2:-/tmp/improve_validation/reports}"
PYTHON="${PYTHON:-/mydir/envs/WorldArena/bin/python}"

mkdir -p "$OUT_POSTPROC" "$OUT_REPORT"
cd "$ROOT"

echo "=== [Tier 1] Post-processing origin/ -> $OUT_POSTPROC ==="
$PYTHON -m myscript.improve.postprocess \
    --batch \
    --input "$ORIGIN" \
    --gt    "$ORIGIN" \
    --output "$OUT_POSTPROC" \
    --report "$OUT_REPORT/postproc_report.json" \
    --crossfade-len 3 \
    --color-strength 0.5 \
    --unsharp-amount 0.5 \
    --tail-len 5 --tail-alpha 0.7

echo
echo "=== Quick smoke comparison (aesthetic + smoothness proxies only) ==="
$PYTHON -m myscript.improve.seed_select \
    --candidates "$ORIGIN" \
    --pattern '{episode}.mp4' \
    --output "$OUT_REPORT/_origin_score" \
    --report "$OUT_REPORT/origin_scores.json" \
    --w-musiq 0 --w-aesthetic 0 --w-smoothness 1.0
$PYTHON -m myscript.improve.seed_select \
    --candidates "$OUT_POSTPROC" \
    --pattern '{episode}.mp4' \
    --output "$OUT_REPORT/_postproc_score" \
    --report "$OUT_REPORT/postproc_scores.json" \
    --w-musiq 0 --w-aesthetic 0 --w-smoothness 1.0

echo
echo "=== Per-episode delta ==="
$PYTHON "$ROOT/myscript/improve/_compare_diff.py" \
    --origin "$OUT_REPORT/origin_scores.json" \
    --postproc "$OUT_REPORT/postproc_scores.json"
echo
echo "Files written:"
ls -la "$OUT_POSTPROC"
echo
echo "Reports:"
ls -la "$OUT_REPORT"
echo
echo "Next: for full WorldArena.evaluate, run for each directory:"
echo "  bash run_evaluation.sh origin   $ORIGIN summary.json config/config.yaml aesthetic_quality,image_quality,background_consistency,subject_consistency,motion_smoothness,photometric_smoothness 10"
echo "  bash run_evaluation.sh postproc $OUT_POSTPROC summary.json config/config.yaml aesthetic_quality,image_quality,background_consistency,subject_consistency,motion_smoothness,photometric_smoothness 10"
