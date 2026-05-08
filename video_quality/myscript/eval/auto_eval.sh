#!/bin/bash
# 便捷启动脚本：调用 myscript/eval/auto_eval.py
#
# 用法（典型）:
#   bash myscript/eval/auto_eval.sh \
#       /mnt/jackzou/ckp/OminiEWM/infer_output/20260507/<run_dir> \
#       /mnt/jackzou/WorldArena/results
#
# 也可以通过环境变量传入：
#   SCAN_PATH=...  RESULTS_DIR=...  bash myscript/eval/auto_eval.sh
#
# 其余参数可通过 EXTRA_ARGS 透传给 python 脚本：
#   EXTRA_ARGS="--once --skip-vlm --skip-jepa" bash myscript/eval/auto_eval.sh ...
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VIDEO_QUALITY_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

SCAN_PATH=${1:-${SCAN_PATH:-}}
RESULTS_DIR=${2:-${RESULTS_DIR:-}}

if [ -z "${SCAN_PATH}" ] || [ -z "${RESULTS_DIR}" ]; then
    echo "Usage: $0 <SCAN_PATH> <RESULTS_DIR> [extra args...]" >&2
    echo "  SCAN_PATH    需要持续扫描的根目录" >&2
    echo "  RESULTS_DIR  结果 TSV / state / log 目录" >&2
    exit 1
fi
shift $(( $# >= 2 ? 2 : $# )) || true

EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$VIDEO_QUALITY_DIR"

# 优先使用项目主 conda 环境，便于 import yaml 等依赖；若激活失败也不致命，
# 让 python 直接接管，主要的子脚本会各自再激活自己的环境。
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate /mydir/envs/WorldArena 2>/dev/null || true
fi

exec python "$SCRIPT_DIR/auto_eval.py" \
    --scan-path "$SCAN_PATH" \
    --results-dir "$RESULTS_DIR" \
    --video-quality-dir "$VIDEO_QUALITY_DIR" \
    ${EXTRA_ARGS} \
    "$@"
