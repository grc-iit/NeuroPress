#!/bin/bash
# Generate figures for VPIC results only.
# Usage:
#   bash benchmarks/vpic-kokkos/plot.sh
#   VPIC_NX=256 VPIC_TS=100 bash benchmarks/vpic-kokkos/plot.sh
#   VPIC_DIR=benchmarks/vpic-kokkos/results/eval_NX64_chunk4mb_ts50 bash benchmarks/vpic-kokkos/plot.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPU_DIR"

VPIC_NX=${VPIC_NX:-156}
VPIC_CHUNK=${VPIC_CHUNK:-4}
VPIC_TS=${VPIC_TS:-100}
VPIC_DIR=${VPIC_DIR:-benchmarks/vpic-kokkos/results/eval_NX${VPIC_NX}_chunk${VPIC_CHUNK}mb_ts${VPIC_TS}}

echo "=== VPIC Plot Generation ==="
echo "  Data: $VPIC_DIR"
echo ""

VPIC_DIR="$VPIC_DIR" python3 benchmarks/plots/generate_dataset_figures.py --dataset vpic
