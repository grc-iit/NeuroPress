#!/bin/bash
# ============================================================
# Exploration Threshold Sweep
#
# Measures impact of X1 (SGD MAPE threshold) and X2 (exploration
# threshold) on compression quality using the hurricane dataset.
#
# X1 values:      [5, 10, 20, 30, 50, 100, 1000]%
# X2-X1 delta:    [5, 10, 20, 30, 50, 100, 1000]%
# Total runs: 7 x 7 = 49
#
# Usage:
#   bash benchmarks/eval_exploration_threshold.sh
#
# Environment overrides:
#   CHUNK_MB        16         Chunk size in MB
#   ERROR_BOUND     0.01      Lossy error bound
#   EXPLORE_K       4         Exploration alternatives
#   SGD_LR          0.2       SGD learning rate
#   DRY_RUN         0         Print commands without running
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Fixed parameters ──
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
BIN="$GPU_DIR/build/generic_benchmark"
DATA_DIR="$GPU_DIR/data/sdrbench/hurricane_isabel/100x500x500"
DIMS="100,500,500"
EXT=".bin.f32"
CHUNK_MB=${CHUNK_MB:-4}
ERROR_BOUND=${ERROR_BOUND:-0.01}
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

# Error bound tag: 0 → "lossless", 0.01 → "eb0.01"
if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${ERROR_BOUND}"
fi
SWEEP_DIR="$PARENT_DIR/results/threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# ── Validate ──
if [ ! -f "$BIN" ]; then
    echo "ERROR: benchmark binary not found at $BIN"
    echo "  Build with: cd build && cmake .. && make -j generic_benchmark"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found at $WEIGHTS"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: hurricane data not found at $DATA_DIR"
    exit 1
fi

# ── Threshold values (as fractions) ──
X1_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)
DELTA_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)

TOTAL=$(( ${#X1_VALUES[@]} * ${#DELTA_VALUES[@]} ))
RUN=0

echo "============================================================"
echo "Exploration Threshold Sweep: $TOTAL runs"
echo "  Dataset:     hurricane_isabel (100x500x500)"
echo "  Error bound: $ERROR_BOUND"
echo "  Chunk MB:    $CHUNK_MB"
echo "  SGD LR:      $SGD_LR"
echo "  Explore K:   $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"

# ── Run no-comp baseline once ──
BASELINE_DIR="$SWEEP_DIR/baseline"
if [ ! -f "$BASELINE_DIR/benchmark_hurricane_isabel.csv" ]; then
    echo ""
    echo "[baseline] Running no-comp phase..."
    mkdir -p "$BASELINE_DIR"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  DRY_RUN: $BIN $WEIGHTS --data-dir $DATA_DIR --dims $DIMS --ext $EXT --chunk-mb $CHUNK_MB --error-bound $ERROR_BOUND --phase no-comp --out-dir $BASELINE_DIR --name hurricane_isabel --no-verify"
    else
        "$BIN" "$WEIGHTS" \
            --data-dir "$DATA_DIR" --dims "$DIMS" --ext "$EXT" \
            --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
            --phase no-comp \
            --out-dir "$BASELINE_DIR" --name hurricane_isabel \
            --no-verify \
            2>&1 | tail -5
    fi
else
    echo "[baseline] Already exists, skipping."
fi

# ── Sweep ──
for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        RUN=$((RUN + 1))
        x2=$(echo "$x1 + $delta" | bc -l)

        OUT_DIR="$SWEEP_DIR/x1_${x1}_delta_${delta}"

        # Skip if results already exist
        if [ -f "$OUT_DIR/benchmark_hurricane_isabel.csv" ]; then
            echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2) — already done, skipping."
            continue
        fi

        echo ""
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
        mkdir -p "$OUT_DIR"

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  DRY_RUN: $BIN $WEIGHTS --data-dir $DATA_DIR --dims $DIMS --ext $EXT --chunk-mb $CHUNK_MB --error-bound $ERROR_BOUND --phase nn-rl+exp50 --mape $x1 --explore-thresh $x2 --lr $SGD_LR --explore-k $EXPLORE_K --w0 $W0 --w1 $W1 --w2 $W2 --out-dir $OUT_DIR --name hurricane_isabel --no-verify"
            continue
        fi

        NO_RANKING=1 "$BIN" "$WEIGHTS" \
            --data-dir "$DATA_DIR" --dims "$DIMS" --ext "$EXT" \
            --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
            --phase nn-rl+exp50 \
            --mape "$x1" --explore-thresh "$x2" \
            --lr "$SGD_LR" --explore-k "$EXPLORE_K" \
            --w0 $W0 --w1 $W1 --w2 $W2 \
            --out-dir "$OUT_DIR" --name hurricane_isabel \
            --no-verify \
            2>&1 | tail -3

        echo "  Done. CSV: $OUT_DIR/benchmark_hurricane_isabel.csv"
    done
done

echo ""
echo "============================================================"
echo "Sweep complete: $TOTAL runs"
echo "Results: $SWEEP_DIR"
echo ""
echo "Generate plots:"
echo "  python3 benchmarks/plots/plot_threshold_sweep.py $SWEEP_DIR"
echo "============================================================"
