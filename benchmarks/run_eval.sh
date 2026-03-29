#!/bin/bash
# ============================================================
# Full evaluation: lossy + lossless VPIC benchmarks
#
# Runs two VPIC benchmarks back-to-back:
#   1. Lossy  (error_bound=0.01, min_psnr=50)
#   2. Lossless
#
# All 12 phases, all 3 policies, 128 MB/timestep, 4 MB chunks, 25 timesteps.
# Results go to separate directories (lossy_001 vs lossless suffix).
#
# Usage:
#   bash benchmarks/run_eval.sh
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common settings
export BENCHMARKS=vpic
export DATA_MB=256
export CHUNK_MB=4
export TIMESTEPS=25
export POLICIES="balanced,ratio,speed"
export VERIFY=0

echo "============================================================"
echo "  Run 1/2: VPIC Lossy (error_bound=0.01, min_psnr=50)"
echo "============================================================"
export VPIC_ERROR_BOUND=0.01
export VPIC_MIN_PSNR=50
export VPIC_EVAL_SUFFIX="_lossy_001"
bash "$SCRIPT_DIR/benchmark.sh"

echo ""
echo "============================================================"
echo "  Run 2/2: VPIC Lossless"
echo "============================================================"
unset VPIC_ERROR_BOUND
unset VPIC_MIN_PSNR
export VPIC_EVAL_SUFFIX="_lossless"
bash "$SCRIPT_DIR/benchmark.sh"

echo ""
echo "============================================================"
echo "  All evaluation runs complete."
echo "============================================================"
