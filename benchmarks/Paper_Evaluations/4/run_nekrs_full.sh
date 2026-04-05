#!/bin/bash
# ============================================================
# Full nekRS Paper Evaluation
#
# Mirrors run_vpic_full.sh for nekRS spectral-element CFD:
# 1. Full benchmark: all 12 phases, 3 policies, fp32, ~456 MB/checkpoint
#
# Key design:
# - TGV (Taylor-Green Vortex) on GPU via OCCA
# - checkpointInterval=10 steps between dumps → data evolves
# - Each NN phase × policy is a separate nekRS run → clean weight state
# - Lossless with bitwise verification
# - Zero nekRS source changes (UDF-level integration)
#
# Prerequisites:
# - nekRS built and installed ($NEKRS_HOME)
# - GPUCompress TGV case prepared (patches/tgv.udf, patches/tgv.par, patches/udf.cmake)
# - OCCA kernels cached (first run of TGV takes ~5-10 min for JIT)
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/run_nekrs_full.sh
# ============================================================
set -e

cd /home/cc/GPUCompress

CASE_DIR="${NEKRS_CASE_DIR:-$HOME/tgv_gpucompress}"

# Verify case exists
if [ ! -f "$CASE_DIR/tgv.udf" ]; then
    echo "ERROR: nekRS case not found at $CASE_DIR"
    echo "  Set up case:"
    echo "    cp -a \$NEKRS_HOME/examples/tgv $CASE_DIR"
    echo "    cp benchmarks/nekrs/patches/* $CASE_DIR/"
    echo "    # Edit $CASE_DIR/udf.cmake with correct paths"
    exit 1
fi

echo "============================================================"
echo "Full nekRS Paper Evaluation"
echo "============================================================"

# ── Step 1: Full benchmark (fp32, all phases, all policies) ──
echo ""
echo ">>> Step 1: Full Benchmark (TGV fp32, ~456 MB/checkpoint, 2 ranks)"
echo "    All 12 phases × 3 policies = 18 runs"
echo "    NOTE: First run may take ~5 min for OCCA JIT compilation"

NEKRS_HOME="${NEKRS_HOME:-$HOME/.local/nekrs}" \
CASE_DIR="$CASE_DIR" \
NP=2 \
USE_FP32=1 \
NUM_STEPS=20 \
CHECKPOINT_INT=10 \
POLICIES="balanced,ratio,speed" \
VERIFY=1 \
RESULTS_DIR="benchmarks/nekrs/results/eval_paper_tgv_fp32_all_phases" \
bash benchmarks/nekrs/run_nekrs_benchmark.sh

echo ""
echo "============================================================"
echo "Full nekRS Paper Evaluation Complete"
echo ""
echo "Results:"
echo "  Full benchmark: benchmarks/nekrs/results/eval_paper_tgv_fp32_all_phases/"
echo "============================================================"
