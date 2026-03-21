#!/bin/bash
# ============================================================
# Gray-Scott Benchmark Evaluation Suite
# Runs multiple cost model configurations and organizes results.
#
# Usage:
#   bash benchmarks/grayscott/run_gs_eval.sh
#
# Output structure:
#   benchmarks/grayscott/results/eval_L400_chunk4mb_ts100/
#     balanced_w1-1-1/
#       gs_benchmark.log
#       benchmark_grayscott_vol.csv
#       benchmark_grayscott_vol_chunks.csv
#       benchmark_grayscott_timesteps.csv
#       benchmark_grayscott_timestep_chunks.csv
#     ratio_only_w0-0-1/
#       ...
#     speed_only_w1-1-0/
#       ...
# ============================================================
set +e  # Don't exit on error

# ── Configuration ──
L=400                   # Grid size: L^3 * 4 bytes (~244 MB for L=400)
CHUNK_MB=4              # Chunk size in MB
TIMESTEPS=100           # Number of multi-timestep writes
RUNS=1                  # Single-shot repetitions
DEBUG_NN=1              # 1=print NN rankings, 0=quiet

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
GS_BIN="$GPU_DIR/build/grayscott_benchmark"

# ── Eval directory ──
EVAL_NAME="eval_L${L}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"
mkdir -p "$EVAL_DIR"

# ── Cost model configurations ──
# Format: "label w0 w1 w2"
CONFIGS=(
    "balanced_w1-1-1       1.0 1.0 1.0"
    "ratio_only_w0-0-1     0.0 0.0 1.0"
    "speed_only_w1-1-0     1.0 1.0 0.0"
)

# ── Verify binary exists ──
if [ ! -f "$GS_BIN" ]; then
    echo "ERROR: Gray-Scott binary not found: $GS_BIN"
    echo "Build it first: cmake --build build --target grayscott_benchmark -j\$(nproc)"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

DATASET_MB=$(( L * L * L * 4 / 1024 / 1024 ))
N_CHUNKS=$(( DATASET_MB / CHUNK_MB ))

echo "============================================================"
echo "  Gray-Scott Benchmark Evaluation Suite"
echo "============================================================"
echo "  Grid       : ${L}^3  (~${DATASET_MB} MB)"
echo "  Chunks     : ${CHUNK_MB} MB  (~${N_CHUNKS} chunks)"
echo "  Timesteps  : ${TIMESTEPS}"
echo "  Runs       : ${RUNS}"
echo "  Configs    : ${#CONFIGS[@]}"
echo "  Output     : ${EVAL_DIR}"
echo "============================================================"
echo ""

TOTAL=${#CONFIGS[@]}
CFG_NUM=0

for cfg_line in "${CONFIGS[@]}"; do
    read -r LABEL W0 W1 W2 <<< "$cfg_line"
    CFG_NUM=$((CFG_NUM + 1))

    RUN_DIR="$EVAL_DIR/$LABEL"
    mkdir -p "$RUN_DIR"

    LOG_FILE="$RUN_DIR/gs_benchmark.log"

    echo "============================================================"
    echo "  [$CFG_NUM/$TOTAL] $LABEL  (w0=$W0  w1=$W1  w2=$W2)"
    echo "  Output: $RUN_DIR"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Run benchmark in background for progress tracking
    RUN_START=$(date +%s)
    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
    "$GS_BIN" "$WEIGHTS" \
        --L $L --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS --runs $RUNS \
        --w0 $W0 --w1 $W1 --w2 $W2 \
        --out-dir "$RUN_DIR" \
        > "$LOG_FILE" 2>&1 &
    PID=$!

    # Simple progress: show elapsed time
    while kill -0 $PID 2>/dev/null; do
        sleep 10
        ELAPSED=$(( $(date +%s) - RUN_START ))
        printf "\r  Running... %dm %ds" $((ELAPSED/60)) $((ELAPSED%60))
    done
    printf "\r                          \r"
    wait $PID 2>/dev/null || true
    RC=$?

    ELAPSED=$(( $(date +%s) - RUN_START ))

    # Report status
    if grep -q "Benchmark PASSED" "$LOG_FILE" 2>/dev/null; then
        WRITE=$(grep "║  nn " "$LOG_FILE" 2>/dev/null | head -1 | awk -F'║' '{print $4}' | xargs)
        RATIO=$(grep "║  nn " "$LOG_FILE" 2>/dev/null | head -1 | awk -F'║' '{print $7}' | xargs)
        echo "  DONE  write=${WRITE:-N/A} MiB/s  ratio=${RATIO:-N/A}  (${ELAPSED}s)"
    else
        echo "  FAILED (exit code $RC, ${ELAPSED}s)"
    fi
    echo ""
done

echo "============================================================"
echo "  All runs complete. Results in: $EVAL_DIR"
echo "============================================================"
echo ""
echo "To visualize a specific config:"
echo "  python3 benchmarks/visualize.py --gs-dir $EVAL_DIR/balanced_w1-1-1"
echo ""
ls -la "$EVAL_DIR"/*/gs_benchmark.log 2>/dev/null
