#!/bin/bash
# ============================================================
# SDRBench Benchmark Evaluation Suite
# Runs all SDRBench datasets through the 8-phase benchmark
# with multiple cost model configurations.
#
# Usage:
#   bash benchmarks/sdrbench/run_sdrbench_eval.sh
#
# Output structure:
#   benchmarks/sdrbench/results/eval_chunk4mb/
#     balanced_w1-1-1/
#       benchmark_hurricane_isabel.csv
#       benchmark_nyx.csv
#       benchmark_cesm_atm.csv
#       ...
#     ratio_only_w0-0-1/
#       ...
#     speed_only_w1-1-0/
#       ...
# ============================================================
set +e

# ── Configuration ──
# Override with env vars: RUNS=1 bash benchmarks/sdrbench/run_sdrbench_eval.sh
CHUNK_MB=${CHUNK_MB:-4}
RUNS=${RUNS:-5}
DEBUG_NN=${DEBUG_NN:-0}

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
BIN="$GPU_DIR/build/generic_benchmark"
DATA_DIR="$GPU_DIR/data/sdrbench"

EVAL_NAME="eval_chunk${CHUNK_MB}mb"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"
mkdir -p "$EVAL_DIR"

# ── Verify binary exists ──
if [ ! -f "$BIN" ]; then
    echo "ERROR: generic_benchmark not found: $BIN"
    echo "Build it: cmake --build build --target generic_benchmark -j\$(nproc)"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

# ── Dataset configurations ──
# Format: "name data_subdir dims ext"
DATASETS=(
    "hurricane_isabel   hurricane_isabel/100x500x500                          100,500,500   .bin.f32"
    "nyx                nyx/SDRBENCH-EXASKY-NYX-512x512x512                  512,512,512   .f32"
    "cesm_atm           cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600        1800,3600     .dat"
)

# ── Cost model configurations ──
# Format: "label w0 w1 w2"
CONFIGS=(
    "balanced_w1-1-1       1.0 1.0 1.0"
    "ratio_only_w0-0-1     0.0 0.0 1.0"
    "speed_only_w1-1-0     1.0 1.0 0.0"
)

N_DATASETS=${#DATASETS[@]}
N_CONFIGS=${#CONFIGS[@]}
TOTAL=$((N_DATASETS * N_CONFIGS))

echo "============================================================"
echo "  SDRBench Benchmark Evaluation Suite"
echo "============================================================"
echo "  Chunk size : ${CHUNK_MB} MB"
echo "  Runs       : ${RUNS}"
echo "  Datasets   : ${N_DATASETS}"
echo "  Configs    : ${N_CONFIGS}"
echo "  Total runs : ${TOTAL}"
echo "  Output     : ${EVAL_DIR}"
echo "============================================================"
echo ""

RUN_NUM=0

for cfg_line in "${CONFIGS[@]}"; do
    read -r LABEL W0 W1 W2 <<< "$cfg_line"

    CFG_DIR="$EVAL_DIR/$LABEL"
    mkdir -p "$CFG_DIR"

    echo "============================================================"
    echo "  Config: $LABEL  (w0=$W0  w1=$W1  w2=$W2)"
    echo "============================================================"

    for ds_line in "${DATASETS[@]}"; do
        read -r NAME SUBDIR DIMS EXT <<< "$ds_line"
        RUN_NUM=$((RUN_NUM + 1))

        FULL_DATA_DIR="$DATA_DIR/$SUBDIR"
        if [ ! -d "$FULL_DATA_DIR" ]; then
            echo "  [$RUN_NUM/$TOTAL] SKIP $NAME: $FULL_DATA_DIR not found"
            echo ""
            continue
        fi

        LOG_FILE="$CFG_DIR/${NAME}_benchmark.log"

        echo "  [$RUN_NUM/$TOTAL] $NAME ($LABEL)"
        echo "    Data: $FULL_DATA_DIR"
        echo "    Dims: $DIMS  Ext: $EXT"
        echo "    Started: $(date '+%Y-%m-%d %H:%M:%S')"

        RUN_START=$(date +%s)
        GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
        "$BIN" "$WEIGHTS" \
            --data-dir "$FULL_DATA_DIR" \
            --dims "$DIMS" \
            --ext "$EXT" \
            --chunk-mb $CHUNK_MB \
            --runs $RUNS \
            --w0 $W0 --w1 $W1 --w2 $W2 \
            --out-dir "$CFG_DIR" \
            > "$LOG_FILE" 2>&1 &
        PID=$!

        while kill -0 $PID 2>/dev/null; do
            sleep 10
            ELAPSED=$(( $(date +%s) - RUN_START ))
            printf "\r    Running... %dm %ds" $((ELAPSED/60)) $((ELAPSED%60))
        done
        printf "\r                              \r"
        wait $PID 2>/dev/null || true

        ELAPSED=$(( $(date +%s) - RUN_START ))

        if grep -q "Benchmark PASSED" "$LOG_FILE" 2>/dev/null; then
            echo "    DONE (${ELAPSED}s)"
        else
            echo "    FAILED (${ELAPSED}s) — see $LOG_FILE"
        fi
        echo ""
    done
done

echo "============================================================"
echo "  All runs complete. Results in: $EVAL_DIR"
echo "============================================================"
echo ""
echo "Result directories:"
ls -d "$EVAL_DIR"/*/ 2>/dev/null
echo ""
echo "To visualize a specific config:"
echo "  python3 benchmarks/visualize.py --sdrbench-dir $EVAL_DIR/balanced_w1-1-1"
