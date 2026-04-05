#!/bin/bash
# ============================================================
# VPIC RL Adaptiveness Evaluation
#
# Demonstrates NN convergence on VPIC simulation data using
# optimal thresholds. Each timestep generates fresh plasma
# physics data — the NN must adapt online.
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/4.2.1_eval_vpic_adaptiveness.sh
#
# Environment overrides:
#   VPIC_NX           100       Grid size
#   CHUNK_MB          4         Chunk size in MB
#   TIMESTEPS         50        Number of timesteps
#   VPIC_ERROR_BOUND  0.01      Error bound (0 for lossless)
#   SGD_LR            0.2       SGD learning rate
#   SGD_MAPE          0.30      X1: MAPE threshold
#   EXPLORE_THRESH    0.50      X2: Exploration threshold
#   POLICY            balanced  Cost model policy
#   DRY_RUN           0         Print commands without running
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Parameters ──
VPIC_BIN="$GPU_DIR/vpic_benchmark_deck.Linux"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_NX=${VPIC_NX:-100}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-50}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SIM_INTERVAL=${SIM_INTERVAL:-190}
VPIC_ERROR_BOUND=${VPIC_ERROR_BOUND:-0.01}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.30}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.50}
POLICY=${POLICY:-balanced}
DRY_RUN=${DRY_RUN:-0}

# Policy weights
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "ERROR: unknown policy '$POLICY' (use balanced, ratio, speed)"; exit 1 ;;
esac

# Error bound tag
if [ "$VPIC_ERROR_BOUND" = "0" ] || [ "$VPIC_ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${VPIC_ERROR_BOUND}"
fi
RESULTS_DIR="$SCRIPT_DIR/results/vpic_adaptiveness_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# ── Validate ──
if [ ! -f "$VPIC_BIN" ]; then
    echo "ERROR: VPIC binary not found at $VPIC_BIN"
    echo "  Build with: bash benchmarks/vpic-kokkos/build_vpic_pm.sh"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found at $WEIGHTS"
    exit 1
fi

echo "============================================================"
echo "VPIC RL Adaptiveness Evaluation"
echo "  VPIC NX:     $VPIC_NX"
echo "  Timesteps:   $TIMESTEPS"
echo "  Chunk MB:    $CHUNK_MB"
echo "  Error bound: $VPIC_ERROR_BOUND"
echo "  Policy:      $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  SGD LR:      $SGD_LR"
echo "  SGD MAPE:    $SGD_MAPE"
echo "  Explore:     $EXPLORE_THRESH"
echo "  Output:      $RESULTS_DIR"
echo "============================================================"

# Skip if results already exist
if [ -f "$RESULTS_DIR/benchmark_vpic_deck.csv" ]; then
    echo ""
    echo "Already done, skipping. (rm -rf $RESULTS_DIR to re-run)"
    exit 0
fi

mkdir -p "$RESULTS_DIR"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN: would run VPIC with above parameters"
    exit 0
fi

GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
VPIC_TIMESTEPS=$TIMESTEPS \
VPIC_WARMUP_STEPS=$WARMUP_STEPS \
VPIC_SIM_INTERVAL=$SIM_INTERVAL \
VPIC_POLICIES=$POLICY \
VPIC_NX=$VPIC_NX \
VPIC_CHUNK_MB=$CHUNK_MB \
VPIC_ERROR_BOUND=$VPIC_ERROR_BOUND \
VPIC_LR=$SGD_LR \
VPIC_MAPE_THRESHOLD=$SGD_MAPE \
VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
VPIC_VERIFY=0 \
VPIC_RESULTS_DIR="$RESULTS_DIR" \
"$VPIC_BIN" 2>&1 | tee "$RESULTS_DIR/vpic.log" | tail -10

if [ ! -f "$RESULTS_DIR/benchmark_vpic_deck.csv" ]; then
    echo "ERROR: VPIC run failed, see $RESULTS_DIR/vpic.log"
    exit 1
fi

# Generate plots
python3 "$GPU_DIR/benchmarks/visualize.py" \
    --vpic-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR" 2>&1 | grep -E "Saved:|Done"

echo ""
echo "============================================================"
echo "VPIC adaptiveness evaluation complete."
echo "Results: $RESULTS_DIR"
echo "============================================================"
