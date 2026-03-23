#!/bin/bash
# Generate figures for SDRBench results only.
# Usage:
#   bash benchmarks/sdrbench/plot.sh
#   CHUNK_MB=16 bash benchmarks/sdrbench/plot.sh
#   SDR_DIR=benchmarks/sdrbench/results/nyx_balanced bash benchmarks/sdrbench/plot.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPU_DIR"

CHUNK_MB=${CHUNK_MB:-4}
SDR_DIR=${SDR_DIR:-benchmarks/sdrbench/results/eval_chunk${CHUNK_MB}mb}

echo "=== SDRBench Plot Generation ==="
echo "  Data: $SDR_DIR"
echo ""

SDR_DIR="$SDR_DIR" python3 benchmarks/plots/generate_dataset_figures.py --all 2>&1 | grep -v "gray_scott\|vpic"
