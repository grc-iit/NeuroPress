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
#       benchmark_vpic_deck_timesteps.csv
#       benchmark_vpic_deck_timestep_chunks.csv
#       benchmark_vpic_deck_chunks.csv
#     ratio_only_w0-0-1/
#       ...
#     speed_only_w1-1-0/
#       ...
# ============================================================
set +e  # Don't exit on error — VPIC cleanup may segfault after benchmark completes

# ── Configuration ──
# Override with env vars: NX=64 TIMESTEPS=5 bash benchmarks/vpic-kokkos/run_vpic_eval.sh
#
# Policy selection (run only specific policies):
#   POLICIES="balanced"              — balanced only
#   POLICIES="balanced,ratio"        — balanced + ratio_only
#   POLICIES="balanced,ratio,speed"  — all 3 (default)
#   POLICIES="ratio"                 — ratio_only only
#
NX=${NX:-156}                  # Grid size: (NX+2)^3 * 16 * 4 bytes (~253 MB for NX=156)
CHUNK_MB=${CHUNK_MB:-4}        # Chunk size in MB
TIMESTEPS=${TIMESTEPS:-100}    # Number of multi-timestep writes
RUNS=${RUNS:-3}                # Single-shot repetitions (3 for error bars)
WARMUP_STEPS=${WARMUP_STEPS:-100}  # VPIC physics warmup steps
DEBUG_NN=${DEBUG_NN:-0}        # 1=print NN debug per-chunk, 0=quiet (default off)
POLICIES=${POLICIES:-balanced,ratio,speed}  # Comma-separated: balanced, ratio, speed

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_BIN="$SCRIPT_DIR/vpic_benchmark_deck.Linux"
VPIC_DECK="$SCRIPT_DIR/vpic_benchmark_deck.cxx"

# LD_LIBRARY_PATH: our HDF5 must come first to avoid conflict with system HDF5
VPIC_LD_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib:/opt/cray/libfabric/1.22.0/lib64:/opt/cray/pe/lib64"

# ── Eval directory ──
EVAL_NAME="eval_NX${NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"
mkdir -p "$EVAL_DIR"

# ── Cost model configurations ──
# Format: "label w0 w1 w2"
ALL_CONFIGS=(
    "balanced_w1-1-1       1.0 1.0 1.0"
    "ratio_only_w0-0-1     0.0 0.0 1.0"
    "speed_only_w1-1-0     1.0 1.0 0.0"
)

# Filter configs based on POLICIES env var
CONFIGS=()
for cfg_line in "${ALL_CONFIGS[@]}"; do
    label=$(echo "$cfg_line" | awk '{print $1}')
    if [[ "$POLICIES" == *"balanced"* ]] && [[ "$label" == *"balanced"* ]]; then
        CONFIGS+=("$cfg_line")
    elif [[ "$POLICIES" == *"ratio"* ]] && [[ "$label" == *"ratio"* ]]; then
        CONFIGS+=("$cfg_line")
    elif [[ "$POLICIES" == *"speed"* ]] && [[ "$label" == *"speed"* ]]; then
        CONFIGS+=("$cfg_line")
    fi
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: No policies matched POLICIES='$POLICIES'"
    echo "  Valid: balanced, ratio, speed (comma-separated)"
    exit 1
fi

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
    LD_LIBRARY_PATH="$VPIC_LD_PATH" \
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
    VPIC_RESULTS_DIR="$RUN_DIR" \
    "$VPIC_BIN" "$VPIC_DECK" > "$LOG_FILE" 2>&1 &
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
