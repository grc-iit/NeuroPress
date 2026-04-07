#!/bin/bash
# ============================================================
# 4.2.1 RL Adaptiveness on Unseen Workloads
#
# Demonstrates NN convergence across diverse workloads using
# optimal thresholds from the threshold sweep. Balanced policy.
#
# Datasets: nyx, hurricane_isabel, cesm_atm
# Phase: nn-rl+exp50 (RL + exploration)
# Ranking profiler: ON (for regret computation)
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/adaptiveness/4.2.1_eval_rl_adaptiveness.sh
#
# Environment overrides:
#   CHUNK_MB          16        Chunk size in MB
#   ERROR_BOUND       0.01      Lossy error bound
#   SGD_LR            0.2       SGD learning rate
#   SGD_MAPE          0.30      X1: MAPE threshold (optimal from sweep)
#   EXPLORE_THRESH    0.50      X2: Exploration threshold (optimal from sweep)
#   EXPLORE_K         4         Exploration alternatives
#   DATASETS          nyx,hurricane_isabel,cesm_atm
#   DRY_RUN           0         Print commands without running
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Fixed parameters ──
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
BIN="$GPU_DIR/build/generic_benchmark"
DATA_BASE="$GPU_DIR/data/sdrbench"
CHUNK_MB=${CHUNK_MB:-4}
ERROR_BOUND=${ERROR_BOUND:-0.01}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.30}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.50}
EXPLORE_K=${EXPLORE_K:-4}
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
RESULTS_BASE="$PARENT_DIR/results/rl_adaptiveness_${POLICY}_${EB_TAG}_lr${SGD_LR}"
DATASETS=${DATASETS:-"nyx,hurricane_isabel,cesm_atm,cesm_atm_26ts"}

# ── Dataset configs ──
declare -A DS_SUBDIR DS_DIMS DS_EXT
DS_SUBDIR[nyx]="nyx/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DIMS[nyx]="512,512,512"
DS_EXT[nyx]=".f32"
DS_SUBDIR[hurricane_isabel]="hurricane_isabel/100x500x500"
DS_DIMS[hurricane_isabel]="100,500,500"
DS_EXT[hurricane_isabel]=".bin.f32"
DS_SUBDIR[cesm_atm]="cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600"
DS_DIMS[cesm_atm]="1800,3600"
DS_EXT[cesm_atm]=".dat"
DS_SUBDIR[cesm_atm_26ts]="cesm_atm_26ts/SDRBENCH-CESM-ATM-26x1800x3600"
DS_DIMS[cesm_atm_26ts]="26,1800,3600"
DS_EXT[cesm_atm_26ts]=".f32"

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
echo "4.2.1 RL Adaptiveness on Unseen Workloads"
echo "  Datasets:        $DATASETS"
echo "  Error bound:     $ERROR_BOUND"
echo "  Policy:          $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  Chunk MB:        $CHUNK_MB"
echo "  SGD MAPE (X1):   $SGD_MAPE"
echo "  Explore (X2):    $EXPLORE_THRESH"
echo "  SGD LR:          $SGD_LR"
echo "  Explore K:       $EXPLORE_K"
echo "  Output:          $RESULTS_BASE"
echo "============================================================"

IFS=',' read -ra DS_LIST <<< "$DATASETS"

for ds in "${DS_LIST[@]}"; do
    DATA_DIR="$DATA_BASE/${DS_SUBDIR[$ds]}"
    DIMS="${DS_DIMS[$ds]}"
    EXT="${DS_EXT[$ds]}"
    OUT_DIR="$RESULTS_BASE/$ds"

    if [ -z "$DIMS" ]; then
        echo "ERROR: unknown dataset '$ds'"
        continue
    fi
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: data directory not found: $DATA_DIR"
        continue
    fi

    # Skip if results already exist
    if [ -f "$OUT_DIR/benchmark_${ds}.csv" ]; then
        echo ""
        echo "[$ds] Already done, skipping. (rm $OUT_DIR to re-run)"
        continue
    fi

    echo ""
    echo "[$ds] Running nn-rl+exp50 (balanced)..."
    echo "  Data: $DATA_DIR"
    echo "  Dims: $DIMS  Ext: $EXT"
    mkdir -p "$OUT_DIR"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  DRY_RUN: $BIN $WEIGHTS --data-dir $DATA_DIR --dims $DIMS --ext $EXT --chunk-mb $CHUNK_MB --error-bound $ERROR_BOUND --phase nn-rl+exp50 --mape $SGD_MAPE --explore-thresh $EXPLORE_THRESH --lr $SGD_LR --explore-k $EXPLORE_K --w0 $W0 --w1 $W1 --w2 $W2 --out-dir $OUT_DIR --name $ds --no-verify"
        continue
    fi

    "$BIN" "$WEIGHTS" \
        --data-dir "$DATA_DIR" --dims "$DIMS" --ext "$EXT" \
        --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
        --phase nn-rl+exp50 \
        --mape "$SGD_MAPE" --explore-thresh "$EXPLORE_THRESH" \
        --lr "$SGD_LR" --explore-k "$EXPLORE_K" \
        --w0 "$W0" --w1 "$W1" --w2 "$W2" \
        --out-dir "$OUT_DIR" --name "$ds" \
        --no-verify \
        2>&1 | tee "$OUT_DIR/benchmark.log" | tail -5

    echo "  Done. CSV: $OUT_DIR/benchmark_${ds}.csv"
done

echo ""
echo "============================================================"
echo "All datasets complete."
echo "Results: $RESULTS_BASE"
echo ""
echo "Generate plots:"
echo "  python3 benchmarks/Paper_Evaluations/4/adaptiveness/4.2.1_plot_rl_adaptiveness.py $RESULTS_BASE"
echo "============================================================"
