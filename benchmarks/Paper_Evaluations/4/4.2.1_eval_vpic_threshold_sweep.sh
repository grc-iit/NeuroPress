#!/bin/bash
# ============================================================
# VPIC Exploration Threshold Sweep
#
# Measures impact of X1 (SGD MAPE threshold) and X2 (exploration
# threshold) on compression quality using VPIC simulation.
#
# VPIC generates fresh data each timestep, making it ideal for
# threshold sensitivity analysis — enough timesteps for SGD to
# converge and exploration patterns to stabilize.
#
# X1 values:      [5, 10, 20, 30, 50, 100, 1000]%
# X2-X1 delta:    [5, 10, 20, 30, 50, 100, 1000]%
# Total runs: 7 x 7 = 49
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/4.2.1_eval_vpic_threshold_sweep.sh
#
# Environment overrides:
#   VPIC_NX         64          Grid size (NX^3 cells)
#   CHUNK_MB        4           Chunk size in MB
#   TIMESTEPS       50          Number of VPIC timesteps
#   VPIC_ERROR_BOUND 0.01      Lossy error bound (0 for lossless)
#   EXPLORE_K       4           Exploration alternatives
#   SGD_LR          0.2         SGD learning rate
#   POLICY          balanced    Cost model policy
#   DRY_RUN         0           Print commands without running
# ============================================================
set -eo pipefail

command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Parameters ──
VPIC_BIN="$GPU_DIR/vpic_benchmark_deck.Linux"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_NX=${VPIC_NX:-100}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-50}
WARMUP_STEPS=${WARMUP_STEPS:-10}
VPIC_ERROR_BOUND=${VPIC_ERROR_BOUND:-0.01}
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
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
SWEEP_DIR="$SCRIPT_DIR/results/vpic_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

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

# ── Threshold values (as fractions) ──
X1_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)
DELTA_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)

TOTAL=$(( ${#X1_VALUES[@]} * ${#DELTA_VALUES[@]} ))
RUN=0

echo "============================================================"
echo "VPIC Exploration Threshold Sweep: $TOTAL runs"
echo "  VPIC NX:     $VPIC_NX"
echo "  Timesteps:   $TIMESTEPS"
echo "  Chunk MB:    $CHUNK_MB"
echo "  Error bound: $VPIC_ERROR_BOUND"
echo "  Policy:      $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  SGD LR:      $SGD_LR"
echo "  Explore K:   $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"

# ── Sweep ──
for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        RUN=$((RUN + 1))
        x2=$(echo "$x1 + $delta" | bc -l)

        OUT_DIR="$SWEEP_DIR/x1_${x1}_delta_${delta}"

        # Skip if results already exist
        if [ -f "$OUT_DIR/benchmark_vpic_deck.csv" ]; then
            echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2) — already done, skipping."
            continue
        fi

        echo ""
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
        mkdir -p "$OUT_DIR"

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  DRY_RUN: VPIC_NX=$VPIC_NX TIMESTEPS=$TIMESTEPS ..."
            continue
        fi

        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        VPIC_TIMESTEPS=$TIMESTEPS \
        VPIC_WARMUP_STEPS=$WARMUP_STEPS \
        VPIC_NO_RANKING=1 \
        VPIC_POLICIES=$POLICY \
        VPIC_NX=$VPIC_NX \
        VPIC_CHUNK_MB=$CHUNK_MB \
        VPIC_ERROR_BOUND=$VPIC_ERROR_BOUND \
        VPIC_LR=$SGD_LR \
        VPIC_MAPE_THRESHOLD=$x1 \
        VPIC_EXPLORE_THRESH=$x2 \
        VPIC_EXPLORE_K=$EXPLORE_K \
        VPIC_VERIFY=0 \
        VPIC_PHASE=nn-rl+exp50 \
        VPIC_RESULTS_DIR="$OUT_DIR" \
        "$VPIC_BIN" 2>&1 | tee "$OUT_DIR/vpic.log" | tail -5

        if [ ! -f "$OUT_DIR/benchmark_vpic_deck.csv" ]; then
            echo "  ERROR: VPIC run failed, see $OUT_DIR/vpic.log"
            continue
        fi
        echo "  Done. CSV: $OUT_DIR/benchmark_vpic_deck.csv"
    done
done

echo ""
echo "============================================================"
echo "VPIC threshold sweep complete."
echo "Results: $SWEEP_DIR"
echo ""
echo "Generate plots:"
echo "  python3 benchmarks/Paper_Evaluations/4/4.2.1_plot_threshold_sweep.py $SWEEP_DIR"
echo "============================================================"
