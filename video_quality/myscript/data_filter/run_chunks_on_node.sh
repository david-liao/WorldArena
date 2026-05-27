#!/bin/bash
# ============================================================================
# Per-node dispatcher for full-scale VLM filtering.
#
# Reads a plan.json produced by prepare_full_chunks.py, computes which chunks
# this node should run (round-robin by default, or via explicit --chunks),
# materializes symlinks lazily, and serially invokes
# run_VLM_judge_multi_gpu.sh on each chunk. Re-running skips chunks whose
# output JSON already has the expected number of entries (resume-friendly).
#
# Usage:
#   bash run_chunks_on_node.sh \
#       --plan /path/to/summary/<run>/plan.json \
#       --node_id 0 --num_nodes 12 \
#       [--chunks "0-9,15,42-50"] \
#       [--ngpus 0]                  # 0 = auto-detect on this node
#       [--metrics all] \
#       [--max_videos 0] \
#       [--force]                    # re-run even if output exists
#       [--dry_run]
# ============================================================================
set -euo pipefail

PLAN=""
NODE_ID=""
NUM_NODES=""
EXPLICIT_CHUNKS=""
NGPUS=0
METRICS="all"
MAX_VIDEOS=0
FORCE=0
DRY_RUN=0

usage() {
    sed -n '4,21p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)        PLAN=$2;            shift 2 ;;
        --node_id)     NODE_ID=$2;         shift 2 ;;
        --num_nodes)   NUM_NODES=$2;       shift 2 ;;
        --chunks)      EXPLICIT_CHUNKS=$2; shift 2 ;;
        --ngpus)       NGPUS=$2;           shift 2 ;;
        --metrics)     METRICS=$2;         shift 2 ;;
        --max_videos)  MAX_VIDEOS=$2;      shift 2 ;;
        --force)       FORCE=1;            shift   ;;
        --dry_run)     DRY_RUN=1;          shift   ;;
        -h|--help)     usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

[[ -z "$PLAN" ]] && { echo "ERROR: --plan required"; exit 1; }
[[ ! -f "$PLAN" ]] && { echo "ERROR: plan not found: $PLAN"; exit 1; }

if [[ -z "$EXPLICIT_CHUNKS" ]]; then
    [[ -z "$NODE_ID" ]] && { echo "ERROR: --node_id required (or pass --chunks)"; exit 1; }
    [[ -z "$NUM_NODES" ]] && { echo "ERROR: --num_nodes required (or pass --chunks)"; exit 1; }
fi

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------- Resolve plan + chunk selection via Python (single source of truth)
SELECTION=$(python3 - "$PLAN" "${NODE_ID:-}" "${NUM_NODES:-}" "$EXPLICIT_CHUNKS" <<'PY'
import json, sys

plan_path, node_id, num_nodes, explicit = sys.argv[1:5]
with open(plan_path) as f:
    plan = json.load(f)

num_chunks = plan["num_chunks"]
chunks = plan["chunks"]

def parse_explicit(spec, n):
    out = set()
    for tok in (s.strip() for s in spec.split(",") if s.strip()):
        if "-" in tok:
            a, b = tok.split("-", 1)
            for k in range(int(a), int(b) + 1):
                if 0 <= k < n:
                    out.add(k)
        else:
            k = int(tok)
            if 0 <= k < n:
                out.add(k)
    return sorted(out)

if explicit:
    selected = parse_explicit(explicit, num_chunks)
else:
    nid, nn = int(node_id), int(num_nodes)
    selected = [i for i in range(num_chunks) if i % nn == nid]

# Emit one TSV-ish line per selected chunk: chunk_id<TAB>summary<TAB>video_dir<TAB>model_name<TAB>leaf<TAB>size
print(f"#TOTAL\t{len(selected)}\t{plan['video_quality_dir']}\t{plan['output_root']}\t{plan['config_path']}\t{plan['chunk_size']}")
for c in chunks:
    if c["chunk_id"] in selected:
        print("\t".join([
            str(c["chunk_id"]),
            c["summary"],
            c["video_dir"],
            c["model_name"],
            c["leaf_name"],
            str(c["size"]),
        ]))
PY
)

HEADER=$(echo "$SELECTION" | head -n 1)
ROWS=$(echo "$SELECTION" | tail -n +2)
TOTAL=$(echo "$HEADER" | awk -F'\t' '{print $2}')
VQ_DIR=$(echo "$HEADER" | awk -F'\t' '{print $3}')
OUTPUT_ROOT=$(echo "$HEADER" | awk -F'\t' '{print $4}')
CONFIG_PATH=$(echo "$HEADER" | awk -F'\t' '{print $5}')
CHUNK_SIZE=$(echo "$HEADER" | awk -F'\t' '{print $6}')

echo "============================================================"
echo ">>> plan         : $PLAN"
if [[ -n "$EXPLICIT_CHUNKS" ]]; then
    echo ">>> chunks       : $EXPLICIT_CHUNKS  (=> $TOTAL chunks)"
else
    echo ">>> node         : $NODE_ID / $NUM_NODES   (round-robin => $TOTAL chunks)"
fi
echo ">>> ngpus        : $NGPUS (0=auto)"
echo ">>> metrics      : $METRICS"
echo ">>> force        : $FORCE   dry_run=$DRY_RUN"
echo ">>> video_quality: $VQ_DIR"
echo ">>> output_root  : $OUTPUT_ROOT"
echo ">>> chunk_size   : $CHUNK_SIZE"
echo "============================================================"

if [[ "$TOTAL" -eq 0 ]]; then
    echo ">>> No chunks assigned to this node, exiting."
    exit 0
fi

ensure_links() {
    local summary=$1
    local video_dir=$2
    python3 - "$summary" "$video_dir" <<'PY'
import json, os, sys
summary_path, video_dir = sys.argv[1:3]
os.makedirs(video_dir, exist_ok=True)
with open(summary_path) as f:
    items = json.load(f)
n_new = 0
n_missing = 0
for it in items:
    src = it.get("orig_video_path")
    if not src or not os.path.exists(src):
        n_missing += 1
        continue
    fname = os.path.basename(it["gt_path"])
    dst = os.path.join(video_dir, fname)
    if not os.path.lexists(dst):
        try:
            os.symlink(src, dst)
            n_new += 1
        except OSError as e:
            print(f"[WARN] symlink failed for {fname}: {e}", file=sys.stderr)
print(f"  links: created={n_new}, total_in_summary={len(items)}, missing_src={n_missing}")
PY
}

is_done() {
    local out_file=$1
    local expected=$2
    [[ -f "$out_file" ]] || return 1
    local n
    n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$out_file" 2>/dev/null || echo 0)
    [[ "$n" -eq "$expected" ]]
}

DONE=0
SKIPPED=0
FAILED=0
INDEX=0
START_ALL=$SECONDS

while IFS=$'\t' read -r CID SUMMARY VIDEO_DIR MODEL_NAME LEAF SIZE; do
    [[ -z "$CID" ]] && continue
    INDEX=$((INDEX + 1))
    OUT_FILE="$OUTPUT_ROOT/$MODEL_NAME/${LEAF}_summary_val_all_intern.json"

    echo ""
    echo "------------------------------------------------------------"
    echo ">>> [$INDEX/$TOTAL] chunk_id=$CID  leaf=$LEAF  size=$SIZE"
    echo ">>>   summary  : $SUMMARY"
    echo ">>>   video_dir: $VIDEO_DIR"
    echo ">>>   model    : $MODEL_NAME"
    echo ">>>   out_file : $OUT_FILE"

    if [[ "$FORCE" -ne 1 ]] && is_done "$OUT_FILE" "$SIZE"; then
        echo ">>>   STATUS   : already done ($SIZE entries) — skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo ">>>   DRY_RUN  : would build links + run VLM judge"
        continue
    fi

    echo ">>>   building/refreshing symlinks..."
    ensure_links "$SUMMARY" "$VIDEO_DIR"

    T_CHUNK=$SECONDS
    if bash "$VQ_DIR/run_VLM_judge_multi_gpu.sh" \
         "$MODEL_NAME" "$VIDEO_DIR" "$SUMMARY" "$CONFIG_PATH" \
         "$METRICS" "$MAX_VIDEOS" "$NGPUS"; then
        ELAPSED=$((SECONDS - T_CHUNK))
        echo ">>>   chunk $LEAF done in ${ELAPSED}s"
        DONE=$((DONE + 1))
    else
        ELAPSED=$((SECONDS - T_CHUNK))
        echo "[ERROR] chunk $LEAF failed after ${ELAPSED}s"
        FAILED=$((FAILED + 1))
    fi
done <<< "$ROWS"

TOTAL_ELAPSED=$((SECONDS - START_ALL))
echo ""
echo "============================================================"
echo ">>> Summary on this node:"
echo "      done    = $DONE"
echo "      skipped = $SKIPPED (already complete)"
echo "      failed  = $FAILED"
echo "      total elapsed = ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m)"
echo "============================================================"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
