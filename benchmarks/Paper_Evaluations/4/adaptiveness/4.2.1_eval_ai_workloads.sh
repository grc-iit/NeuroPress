#!/bin/bash
# ============================================================
# 4.2.1 RL Adaptiveness on AI Training Workloads
#
# Step 1: Export ViT-Base training checkpoints as raw .f32 files
#         (weights, adam_m, adam_v, gradients at each epoch)
# Step 2: Run NN-RL adaptiveness benchmark on the exported data
#
# Each epoch produces 4 tensor types, processed sequentially
# as "fields" — the NN sees how tensor statistics evolve
# across training epochs.
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/adaptiveness/4.2.1_eval_ai_workloads.sh
#
# Environment overrides:
#   AI_MODEL          vit_b_16    Model: vit_b_16 (86M), resnet18 (11M)
#   AI_EPOCHS         20          Training epochs
#   CHUNK_MB          16          Chunk size in MB
#   ERROR_BOUND       0.01        Lossy error bound
#   SGD_LR            0.2         SGD learning rate
#   SGD_MAPE          0.30        X1: MAPE threshold
#   EXPLORE_THRESH    0.50        X2: Exploration threshold
#   EXPLORE_K         4           Exploration alternatives
#   SKIP_EXPORT       0           Skip export step (reuse existing data)
#   DRY_RUN           0           Print commands without running
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parameters ──
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
BIN="$GPU_DIR/build/generic_benchmark"
AI_MODEL=${AI_MODEL:-vit_b_16}
AI_EPOCHS=${AI_EPOCHS:-20}
CHUNK_MB=${CHUNK_MB:-4}
ERROR_BOUND=${ERROR_BOUND:-0.01}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.30}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.50}
EXPLORE_K=${EXPLORE_K:-4}
POLICY=${POLICY:-balanced}
SKIP_EXPORT=${SKIP_EXPORT:-0}
DRY_RUN=${DRY_RUN:-0}

# Policy weights
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "ERROR: unknown policy '$POLICY' (use balanced, ratio, speed)"; exit 1 ;;
esac

# Generate all epoch indices: 1,2,3,...,N
ALL_EPOCHS=$(seq -s, 1 "$AI_EPOCHS")

# Data and results directories
DATA_DIR="$GPU_DIR/data/ai_training/${AI_MODEL}_checkpoints"
RESULTS_BASE="$PARENT_DIR/results/ai_workloads_${AI_MODEL}_${POLICY}_lr${SGD_LR}"

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

echo "============================================================"
echo "4.2.1 RL Adaptiveness on AI Training Workloads"
echo "  Model:           $AI_MODEL"
echo "  Epochs:          $AI_EPOCHS"
echo "  Chunk MB:        $CHUNK_MB"
echo "  Error bound:     $ERROR_BOUND"
echo "  SGD LR:          $SGD_LR"
echo "  SGD MAPE (X1):   $SGD_MAPE"
echo "  Explore (X2):    $EXPLORE_THRESH"
echo "  Data:            $DATA_DIR"
echo "  Output:          $RESULTS_BASE"
echo "============================================================"

# ════════════════════════════════════════════════════════════
# Step 1: Export training checkpoints
# ════════════════════════════════════════════════════════════

if [ "$SKIP_EXPORT" -eq 1 ]; then
    echo ""
    echo "[Step 1] Skipping export (SKIP_EXPORT=1)"
else
    N_EXISTING=$(find "$DATA_DIR" -name "*.f32" 2>/dev/null | wc -l)
    EXPECTED=$((AI_EPOCHS * 4))

    if [ "$N_EXISTING" -ge "$EXPECTED" ]; then
        echo ""
        echo "[Step 1] Checkpoints already exported ($N_EXISTING files). Skipping."
        echo "  (rm -rf $DATA_DIR to re-export)"
    else
        echo ""
        echo "[Step 1] Exporting $AI_MODEL checkpoints ($AI_EPOCHS epochs)..."

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  DRY_RUN: python3 scripts/train_and_export_checkpoints.py --model $AI_MODEL --epochs $AI_EPOCHS --checkpoint-epochs $ALL_EPOCHS --outdir $DATA_DIR"
        else
            python3 "$GPU_DIR/scripts/train_and_export_checkpoints.py" \
                --model "$AI_MODEL" \
                --epochs "$AI_EPOCHS" \
                --checkpoint-epochs "$ALL_EPOCHS" \
                --outdir "$DATA_DIR" \
                --data-root "$GPU_DIR/data"

            N_FILES=$(find "$DATA_DIR" -name "*.f32" | wc -l)
            echo "  Exported $N_FILES files to $DATA_DIR"
        fi
    fi
fi

# Verify data exists
if [ ! -d "$DATA_DIR" ] || [ "$(find "$DATA_DIR" -name "*.f32" 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "ERROR: No .f32 files found in $DATA_DIR"
    echo "  Run without SKIP_EXPORT=1 to generate data"
    exit 1
fi

# Detect dims from the first .f32 file
FIRST_F32=$(find "$DATA_DIR" -name "*.f32" | sort | head -1)
FILE_BYTES=$(stat --format="%s" "$FIRST_F32")
N_FLOATS=$((FILE_BYTES / 4))
# Use the dims file if the export script created one, otherwise compute 2D factorization
DIMS_FILE="$DATA_DIR/dims.txt"
if [ -f "$DIMS_FILE" ]; then
    DIMS=$(cat "$DIMS_FILE")
else
    DIMS=$(python3 -c "
import math
n = $N_FLOATS
s = int(math.isqrt(n))
while s > 1 and n % s != 0:
    s -= 1
print(f'{s},{n // s}')
")
fi
N_FILES=$(find "$DATA_DIR" -name "*.f32" | wc -l)

echo ""
echo "  Data ready: $N_FILES files, dims=$DIMS, $(echo "scale=1; $FILE_BYTES / 1048576" | bc) MB each"

# ════════════════════════════════════════════════════════════
# Step 2: Run NN-RL adaptiveness benchmark
# ════════════════════════════════════════════════════════════

DS_NAME="${AI_MODEL}"
OUT_DIR="$RESULTS_BASE/$DS_NAME"

if [ -f "$OUT_DIR/benchmark_${DS_NAME}.csv" ]; then
    echo ""
    echo "[Step 2] Already done, skipping. (rm $OUT_DIR to re-run)"
else
    echo ""
    echo "[Step 2] Running nn-rl+exp50 (balanced) on $AI_MODEL checkpoints..."
    mkdir -p "$OUT_DIR"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  DRY_RUN: $BIN $WEIGHTS --data-dir $DATA_DIR --dims $DIMS --ext .f32 --chunk-mb $CHUNK_MB --error-bound $ERROR_BOUND --phase nn-rl+exp50 --mape $SGD_MAPE --explore-thresh $EXPLORE_THRESH --lr $SGD_LR --explore-k $EXPLORE_K --w0 $W0 --w1 $W1 --w2 $W2 --out-dir $OUT_DIR --name $DS_NAME --no-verify"
    else
        "$BIN" "$WEIGHTS" \
            --data-dir "$DATA_DIR" --dims "$DIMS" --ext ".f32" \
            --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
            --phase nn-rl+exp50 \
            --mape "$SGD_MAPE" --explore-thresh "$EXPLORE_THRESH" \
            --lr "$SGD_LR" --explore-k "$EXPLORE_K" \
            --w0 "$W0" --w1 "$W1" --w2 "$W2" \
            --out-dir "$OUT_DIR" --name "$DS_NAME" \
            --no-verify \
            2>&1 | tee "$OUT_DIR/benchmark.log" | tail -5

        echo "  Done. CSV: $OUT_DIR/benchmark_${DS_NAME}.csv"
    fi
fi

echo ""
echo "============================================================"
echo "AI workload evaluation complete."
echo "Results: $RESULTS_BASE"
echo ""
echo "Generate plots:"
echo "  python3 benchmarks/Paper_Evaluations/4/adaptiveness/4.2.1_plot_rl_adaptiveness.py $RESULTS_BASE"
echo "============================================================"
