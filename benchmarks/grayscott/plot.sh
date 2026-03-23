#!/bin/bash
# Generate figures for Gray-Scott results only.
# Usage:
#   bash benchmarks/grayscott/plot.sh
#   GS_L=256 GS_TS=50 bash benchmarks/grayscott/plot.sh
#   GS_DIR=benchmarks/grayscott/results/my_run bash benchmarks/grayscott/plot.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPU_DIR"

GS_L=${GS_L:-400}
GS_CHUNK=${GS_CHUNK:-4}
GS_TS=${GS_TS:-100}
GS_DIR=${GS_DIR:-benchmarks/grayscott/results/eval_L${GS_L}_chunk${GS_CHUNK}mb_ts${GS_TS}}

echo "=== Gray-Scott Plot Generation ==="
echo "  Data: $GS_DIR"
echo ""

GS_DIR="$GS_DIR" python3 benchmarks/plots/generate_dataset_figures.py --dataset gray_scott
