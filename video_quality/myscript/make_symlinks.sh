#!/usr/bin/env bash
# Create a directory of symlinks mapping sample_NNNNNN{suffix}.mp4 -> episodeK.mp4
# so that preprocess_datasets.py can find them by the expected naming convention.
#
# Usage: bash make_symlinks.sh <source_video_dir> <link_output_dir> [suffix]
#   source_video_dir : directory containing sample_000000{suffix}.mp4 ...
#   link_output_dir  : directory where episode1.mp4, episode2.mp4 ... will be created
#   suffix           : filename suffix before .mp4 (default: _concat)
#
# The mapping is positional: sample_000000 -> episode1, sample_000001 -> episode2, ...

set -euo pipefail

SRC_DIR="${1:?Usage: $0 <source_video_dir> <link_output_dir> [suffix]}"
LINK_DIR="${2:?Usage: $0 <source_video_dir> <link_output_dir> [suffix]}"
SUFFIX="${3:-_concat}"

SRC_DIR=$(realpath "$SRC_DIR")

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source directory does not exist: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$LINK_DIR"

count=0
for f in $(ls "$SRC_DIR"/sample_*"${SUFFIX}".mp4 2>/dev/null | sort); do
    idx=$(basename "$f" | grep -oP '\d{6}')
    ep=$((10#$idx + 1))
    ln -sf "$f" "$LINK_DIR/episode${ep}.mp4"
    count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
    echo "WARNING: no files matching sample_*${SUFFIX}.mp4 found in $SRC_DIR" >&2
    exit 1
fi

echo "Created $count symlinks in $LINK_DIR (episode1.mp4 .. episode${count}.mp4)"
