#!/bin/bash
# ============================================================
# VPIC Benchmark Evaluation Suite
# Runs multiple cost model configurations and organizes results.
#
# Usage:
#   bash benchmarks/vpic-kokkos/run_vpic_eval.sh
#
# Output structure:
#   benchmarks/vpic-kokkos/results/eval_NX200_chunk8mb_ts100/
#     balanced_w1-1-1/
#       vpic_benchmark.log
#       benchmark_vpic_deck.csv
#       benchmark_vpic_timesteps.csv
#       benchmark_vpic_timestep_chunks.csv
#       benchmark_vpic_chunks.csv
#     ratio_only_w0-0-1/
#       ...
#     speed_only_w1-1-0/
#       ...
# ============================================================
set +e  # Don't exit on error — VPIC cleanup may segfault after benchmark completes

# ── Configuration ──
NX=156                  # Grid size: (NX+2)^3 * 16 * 4 bytes (~253 MB for NX=156)
CHUNK_MB=4              # Chunk size in MB
TIMESTEPS=100           # Number of multi-timestep writes
RUNS=1                  # Single-shot repetitions
WARMUP_STEPS=100        # VPIC physics warmup steps
DEBUG_NN=1              # 1=print NN rankings, 0=quiet

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
MPIRUN="/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/11.8/hpcx/hpcx-2.14/ompi/bin/mpirun"
VPIC_BIN="$SCRIPT_DIR/vpic_benchmark_deck.Linux"

# ── Eval directory ──
EVAL_NAME="eval_NX${NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
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
if [ ! -f "$VPIC_BIN" ]; then
    echo "ERROR: VPIC binary not found: $VPIC_BIN"
    echo "Build it first: cd benchmarks/vpic-kokkos && bash build_vpic_benchmark.sh"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

echo "============================================================"
echo "  VPIC Benchmark Evaluation Suite"
echo "============================================================"
echo "  Grid       : ${NX}^3  (~$(( (NX+2)**3 * 16 * 4 / 1024 / 1024 )) MB)"
echo "  Chunks     : ${CHUNK_MB} MB"
echo "  Timesteps  : ${TIMESTEPS}"
echo "  Runs       : ${RUNS}"
echo "  Configs    : ${#CONFIGS[@]}"
echo "  Output     : ${EVAL_DIR}"
echo "============================================================"
echo ""

TOTAL=${#CONFIGS[@]}
CFG_NUM=0
STARTED=$(date +%s)

for cfg_line in "${CONFIGS[@]}"; do
    # Parse config
    read -r LABEL W0 W1 W2 <<< "$cfg_line"
    CFG_NUM=$((CFG_NUM + 1))

    RUN_DIR="$EVAL_DIR/$LABEL"
    mkdir -p "$RUN_DIR"

    LOG_FILE="$RUN_DIR/vpic_benchmark.log"

    echo "============================================================"
    echo "  [$CFG_NUM/$TOTAL] $LABEL  (w0=$W0  w1=$W1  w2=$W2)"
    echo "  Output: $RUN_DIR"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Run benchmark in background so we can show progress
    RUN_START=$(date +%s)
    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    VPIC_NX=$NX \
    VPIC_CHUNK_MB=$CHUNK_MB \
    VPIC_TIMESTEPS=$TIMESTEPS \
    VPIC_RUNS=$RUNS \
    VPIC_WARMUP_STEPS=$WARMUP_STEPS \
    VPIC_W0=$W0 \
    VPIC_W1=$W1 \
    VPIC_W2=$W2 \
    "$MPIRUN" -np 1 "$VPIC_BIN" > "$LOG_FILE" 2>&1 &
    PID=$!

    # Simple progress: just show elapsed time
    while kill -0 $PID 2>/dev/null; do
        sleep 10
        ELAPSED=$(( $(date +%s) - RUN_START ))
        printf "\r  Running... %dm %ds" $((ELAPSED/60)) $((ELAPSED%60))
    done
    printf "\r                          \r"
    wait $PID 2>/dev/null || true
    RC=$?

    # Move result CSVs into the run directory
    SRC_DIR="$GPU_DIR/benchmarks/vpic-kokkos/results"
    for csv in benchmark_vpic_deck.csv benchmark_vpic_timesteps.csv \
               benchmark_vpic_timestep_chunks.csv benchmark_vpic_chunks.csv; do
        if [ -f "$SRC_DIR/$csv" ]; then
            cp "$SRC_DIR/$csv" "$RUN_DIR/$csv"
        fi
    done

    # Report status
    ELAPSED=$(( $(date +%s) - STARTED ))
    if [ $RC -eq 0 ] || grep -q "Multi-Timestep complete" "$LOG_FILE" 2>/dev/null; then
        WRITE=$(grep "║  nn " "$LOG_FILE" 2>/dev/null | head -1 | awk -F'║' '{print $3}' | xargs)
        RATIO=$(grep "║  nn " "$LOG_FILE" 2>/dev/null | head -1 | awk -F'║' '{print $5}' | xargs)
        echo "  DONE  write=${WRITE:-N/A} MiB/s  ratio=${RATIO:-N/A}  (${ELAPSED}s)"
    else
        echo "  FAILED (exit code $RC, ${ELAPSED}s)"
    fi
    echo ""
    STARTED=$(date +%s)  # reset timer for next config
done

echo "============================================================"
echo "  All runs complete. Results in: $EVAL_DIR"
echo "============================================================"
echo ""
echo "To visualize:"
echo "  python3 benchmarks/visualize.py"
echo ""
ls -la "$EVAL_DIR"/*/vpic_benchmark.log 2>/dev/null
