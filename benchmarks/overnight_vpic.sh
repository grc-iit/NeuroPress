#!/bin/bash
# ============================================================
# Overnight VPIC Benchmark Suite
#
# Runs three VPIC configurations sequentially, 5 repeated runs each,
# for SC-quality experimental error bars.
#
# Config 1: NX=320 (~139 MB), 16 MB chunks, 50 timesteps, lossless, 5 runs
# Config 2: NX=320 (~139 MB),  4 MB chunks, 50 timesteps, lossless, 5 runs
# Config 3: NX=320 (~139 MB), 16 MB chunks, 50 timesteps, lossy,    3 runs
# Config 4: NX=320 (~139 MB),  4 MB chunks, 50 timesteps, lossy,    3 runs
#
# Estimated time: ~3-5 hours total on A100
#   Each config: 50 timesteps × 12 phases × 3-5 runs
#
# Usage:
#   nohup bash benchmarks/overnight_vpic.sh > overnight_vpic.log 2>&1 &
#
# Monitor progress:
#   tail -f overnight_vpic.log
#
# Results:
#   benchmarks/repeated_results/vpic_256mb_chunk16mb_ts50_lossless/
#   benchmarks/repeated_results/vpic_256mb_chunk4mb_ts50_lossless/
#   benchmarks/repeated_results/vpic_256mb_chunk16mb_ts50_lossy/
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source scripts/setup_env.sh

TOTAL_START=$(date +%s)

echo "============================================================"
echo "  Overnight VPIC Benchmark Suite"
echo "  Started: $(date)"
echo "============================================================"
echo ""
echo "  Config 1: NX=320 (~139 MB), 16 MB chunks, lossless, 5 runs"
echo "  Config 2: NX=320 (~139 MB),  4 MB chunks, lossless, 5 runs"
echo "  Config 3: NX=320 (~139 MB), 16 MB chunks, lossy,    3 runs"
echo "  Config 4: NX=320 (~139 MB),  4 MB chunks, lossy,    3 runs"
echo ""

# ── Common parameters ──
COMMON_ARGS=(
    BENCHMARKS=vpic
    VPIC_NX=320
    TIMESTEPS=50
    VPIC_WARMUP_STEPS=500
    VPIC_SIM_INTERVAL=190
    VPIC_MI_ME=25
    VPIC_WPE_WCE=3
    VPIC_TI_TE=1
    POLICIES="balanced,ratio"
    VERIFY=0
)

# ============================================================
# Config 1: NX=320, 16 MB chunks, lossless, 5 runs
# ============================================================
echo "============================================================"
echo "  Config 1/4: NX=320, 16 MB chunks, lossless, 5 runs"
echo "  Started: $(date)"
echo "============================================================"
echo ""

C1_START=$(date +%s)

bash benchmarks/run_repeated.sh --runs 5 -- \
    "${COMMON_ARGS[@]}" CHUNK_MB=16 VERIFY=1

C1_END=$(date +%s)
C1_MIN=$(( (C1_END - C1_START) / 60 ))
echo "  Config 1 done in ${C1_MIN} minutes"

# Move results to a named directory
C1_DIR=$(ls -td benchmarks/repeated_results/vpic_* 2>/dev/null | head -1)
if [ -n "$C1_DIR" ]; then
    mkdir -p benchmarks/repeated_results
    mv "$C1_DIR" benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossless
    echo "  → benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossless/"
fi
echo ""

# ============================================================
# Config 2: NX=320, 4 MB chunks, lossless, 5 runs
# ============================================================
echo "============================================================"
echo "  Config 2/4: NX=320, 4 MB chunks, lossless, 5 runs"
echo "  Started: $(date)"
echo "============================================================"
echo ""

C2_START=$(date +%s)

bash benchmarks/run_repeated.sh --runs 5 -- \
    "${COMMON_ARGS[@]}" CHUNK_MB=4 VERIFY=1

C2_END=$(date +%s)
C2_MIN=$(( (C2_END - C2_START) / 60 ))
echo "  Config 2 done in ${C2_MIN} minutes"

C2_DIR=$(ls -td benchmarks/repeated_results/vpic_* 2>/dev/null | head -1)
if [ -n "$C2_DIR" ]; then
    mv "$C2_DIR" benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossless
    echo "  → benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossless/"
fi
echo ""

# ============================================================
# Config 3: NX=320, 16 MB chunks, lossy (eb=0.01, min_psnr=50), 3 runs
# ============================================================
echo "============================================================"
echo "  Config 3/4: NX=320, 16 MB chunks, lossy (eb=0.01), 3 runs"
echo "  Started: $(date)"
echo "============================================================"
echo ""

C3_START=$(date +%s)

bash benchmarks/run_repeated.sh --runs 3 -- \
    "${COMMON_ARGS[@]}" CHUNK_MB=16 \
    VPIC_ERROR_BOUND=0.01 VPIC_MIN_PSNR=50 \
    VPIC_EVAL_SUFFIX="_lossy"

C3_END=$(date +%s)
C3_MIN=$(( (C3_END - C3_START) / 60 ))
echo "  Config 3 done in ${C3_MIN} minutes"

C3_DIR=$(ls -td benchmarks/repeated_results/vpic_* 2>/dev/null | head -1)
if [ -n "$C3_DIR" ]; then
    mv "$C3_DIR" benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossy
    echo "  → benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossy/"
fi
echo ""

# ============================================================
# Config 4: NX=320, 4 MB chunks, lossy (eb=0.01, min_psnr=50), 3 runs
# ============================================================
echo "============================================================"
echo "  Config 4/4: NX=320, 4 MB chunks, lossy (eb=0.01), 3 runs"
echo "  Started: $(date)"
echo "============================================================"
echo ""

C4_START=$(date +%s)

bash benchmarks/run_repeated.sh --runs 3 -- \
    "${COMMON_ARGS[@]}" CHUNK_MB=4 \
    VPIC_ERROR_BOUND=0.01 VPIC_MIN_PSNR=50 \
    VPIC_EVAL_SUFFIX="_lossy"

C4_END=$(date +%s)
C4_MIN=$(( (C4_END - C4_START) / 60 ))
echo "  Config 4 done in ${C4_MIN} minutes"

C4_DIR=$(ls -td benchmarks/repeated_results/vpic_* 2>/dev/null | head -1)
if [ -n "$C4_DIR" ]; then
    mv "$C4_DIR" benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossy
    echo "  → benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossy/"
fi
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))
TOTAL_HRS=$(( TOTAL_MIN / 60 ))
REMAINDER_MIN=$(( TOTAL_MIN % 60 ))

echo "============================================================"
echo "  Overnight VPIC Benchmark Complete"
echo "  Finished: $(date)"
echo "  Total time: ${TOTAL_MIN} minutes (${TOTAL_HRS}h ${REMAINDER_MIN}m)"
echo "============================================================"
echo ""
echo "  Results:"
echo "    1. benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossless/"
echo "    2. benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossless/"
echo "    3. benchmarks/repeated_results/vpic_NX320_chunk16mb_ts50_lossy/"
echo "    4. benchmarks/repeated_results/vpic_NX320_chunk4mb_ts50_lossy/"
echo ""
echo "  Each contains:"
echo "    run_1/ ... run_5/                    ← per-run aggregate CSVs"
echo "    mean_std_benchmark_vpic_deck.csv     ← mean ± experimental std (N=5)"
echo ""
echo "  Quick comparison:"
echo "    for cfg in vpic_NX320_chunk16mb_ts50_lossless vpic_NX320_chunk4mb_ts50_lossless vpic_NX320_chunk16mb_ts50_lossy vpic_NX320_chunk4mb_ts50_lossy; do"
echo "      echo \"=== \$cfg ===\""
echo "      head -1 benchmarks/repeated_results/\$cfg/mean_std_benchmark_vpic_deck.csv"
echo "      tail -3 benchmarks/repeated_results/\$cfg/mean_std_benchmark_vpic_deck.csv"
echo "      echo"
echo "    done"
echo "============================================================"
