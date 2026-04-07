#!/bin/bash
# ============================================================
# WarpX Threshold Sweep: SGD MAPE (X1) x Exploration delta (X2-X1)
#
# Two-phase design (mirrors 4.2.1_eval_exploration_threshold.sh):
#   Phase 1 — Run warpx.3d once to dump raw E/B/J/rho fields as .f32
#             (cached: re-runs reuse the dump unless WARPX_FORCE_DUMP=1)
#   Phase 2 — Sweep generic_benchmark over a 4x4 (X1, delta) grid on the
#             flattened field dump, recording per-cell MAPE / I/O bandwidth
#             into benchmark_warpx_field_*.csv files
#
# Output layout (under PARENT_DIR/results/warpx_threshold_sweep_<...>):
#   raw_fields/diag*/*.f32       (cached field dump)
#   flat_fields/                  (symlink-flattened pool, 1 file per .f32)
#   baseline/                     (no-comp reference)
#   x1_<x1>_delta_<delta>/        (per-grid-cell sweep result)
#     benchmark_warpx_field.csv
#
# Pair with 4.2.1_plot_threshold_sweep.py to render the heatmaps used in
# the paper's "Effect of online learning thresholds" figure
# (panels a/b = prediction quality, c/d = I/O bandwidth).
#
# Environment overrides:
#   CHUNK_MB             4         Chunk size MB (generic_benchmark)
#   ERROR_BOUND          0.01      Lossy error bound
#   EXPLORE_K            4         Exploration alternatives K
#   SGD_LR               0.2       SGD learning rate
#   POLICY               balanced  Cost policy (balanced|ratio|speed)
#   WARPX_BIN            (auto)    Path to warpx.3d (default: $HOME/sims/warpx/build-gpucompress/bin/warpx.3d)
#   WARPX_INPUTS         (auto)    LWFA inputs file
#   WARPX_MAX_STEP       200       Simulation steps
#   WARPX_DIAG_INT       10        Diagnostics interval
#   WARPX_NCELL          "32 32 256"
#   WARPX_FORCE_DUMP     0         If 1, re-run Phase 1 even if cached
#   DRY_RUN              0         Print commands without running
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Fixed parameters ──
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"
BIN="$GPU_DIR/build/generic_benchmark"
WARPX_BIN="${WARPX_BIN:-$HOME/sims/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-$HOME/sims/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"

CHUNK_MB=${CHUNK_MB:-4}
ERROR_BOUND=${ERROR_BOUND:-0.01}
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
WARPX_MAX_STEP=${WARPX_MAX_STEP:-200}
WARPX_DIAG_INT=${WARPX_DIAG_INT:-10}
WARPX_NCELL="${WARPX_NCELL:-32 32 256}"
WARPX_FORCE_DUMP=${WARPX_FORCE_DUMP:-0}
DRY_RUN=${DRY_RUN:-0}

# Policy weights
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "ERROR: unknown policy '$POLICY' (use balanced, ratio, speed)"; exit 1 ;;
esac

# Error bound tag
if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${ERROR_BOUND}"
fi

SWEEP_DIR="$PARENT_DIR/results/warpx_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"
RAW_DIR="$SWEEP_DIR/raw_fields"
FLAT_DIR="$SWEEP_DIR/flat_fields"

# ── Validate ──
[ -f "$BIN" ]        || { echo "ERROR: generic_benchmark not found at $BIN"; exit 1; }
[ -f "$WEIGHTS" ]    || { echo "ERROR: NN weights not found at $WEIGHTS"; exit 1; }
[ -x "$WARPX_BIN" ]  || { echo "ERROR: warpx.3d not found at $WARPX_BIN"; exit 1; }
[ -f "$WARPX_INPUTS" ] || { echo "ERROR: WarpX input deck not found at $WARPX_INPUTS"; exit 1; }

mkdir -p "$SWEEP_DIR"

export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "WarpX Threshold Sweep"
echo "  Policy:      $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  Grid:        $WARPX_NCELL ($WARPX_MAX_STEP steps, diag every $WARPX_DIAG_INT)"
echo "  Chunk:       ${CHUNK_MB} MB"
echo "  Error bound: $ERROR_BOUND ($EB_TAG)"
echo "  SGD LR:      $SGD_LR    Explore K: $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"
echo ""

# ============================================================
# Phase 1: Dump raw E/B/J/rho fields once (cached)
# ============================================================
echo ">>> Phase 1: Dump raw WarpX fields (cached if present)"

NEED_DUMP=1
if [ "$WARPX_FORCE_DUMP" != "1" ] && [ -d "$RAW_DIR" ] && \
   [ "$(find "$RAW_DIR" -name '*.f32' -type f | wc -l)" -gt 0 ]; then
    NEED_DUMP=0
    echo "  Reusing cached dump in $RAW_DIR"
fi

if [ "$NEED_DUMP" = "1" ]; then
    rm -rf "$RAW_DIR" "$FLAT_DIR"
    mkdir -p "$RAW_DIR"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: would run warpx.3d to dump fields into $RAW_DIR"
    else
        echo "  Running warpx.3d (max_step=$WARPX_MAX_STEP, diag_int=$WARPX_DIAG_INT)"
        export WARPX_DUMP_FIELDS=1
        export WARPX_DUMP_DIR="$RAW_DIR"
        ( cd "$SWEEP_DIR" && "$WARPX_BIN" "$WARPX_INPUTS" \
            warpx.max_step="$WARPX_MAX_STEP" \
            amr.n_cell="$WARPX_NCELL" \
            diagnostics.diags_names=diag1 \
            diag1.intervals="$WARPX_DIAG_INT" \
            diag1.diag_type=Full \
            diag1.format=gpucompress \
            gpucompress.weights_path="$WEIGHTS" \
            gpucompress.algorithm=auto \
            gpucompress.policy=$POLICY \
            gpucompress.error_bound="$ERROR_BOUND" \
            gpucompress.chunk_bytes=$((CHUNK_MB * 1024 * 1024)) \
            > "$SWEEP_DIR/warpx_dump.log" 2>&1 ) || \
            echo "  WARNING: WarpX exited non-zero (teardown segfault is harmless if dump completed)"
        unset WARPX_DUMP_FIELDS WARPX_DUMP_DIR
    fi
fi

N_F32=$(find "$RAW_DIR" -name '*.f32' -type f 2>/dev/null | wc -l)
[ "$N_F32" -gt 0 ] || { echo "ERROR: no .f32 files dumped under $RAW_DIR"; exit 1; }
echo "  Field files: $N_F32"

# Flatten symlinks so generic_benchmark sees a single data dir
mkdir -p "$FLAT_DIR"
for ts_dir in "$RAW_DIR"/diag*; do
    [ -d "$ts_dir" ] || continue
    ts_name=$(basename "$ts_dir")
    for f in "$ts_dir"/*.f32; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$FLAT_DIR/${ts_name}_$(basename "$f")"
    done
done

# Determine per-file dimensions from first .f32 (1D flattened)
FIRST=$(find "$FLAT_DIR" -name '*.f32' -type f | head -1)
N_FLOATS=$(( $(stat -c%s "$FIRST") / 4 ))
DIMS="${N_FLOATS},1"
echo "  Per-file: $N_FLOATS floats ($((N_FLOATS * 4 / 1024 / 1024)) MB), dims=$DIMS"
echo ""

# ============================================================
# Phase 2: Baseline (no-comp) reference
# ============================================================
BASELINE_DIR="$SWEEP_DIR/baseline"
if [ ! -f "$BASELINE_DIR/benchmark_warpx_field.csv" ]; then
    echo ">>> Phase 2a: no-comp baseline"
    mkdir -p "$BASELINE_DIR"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: baseline no-comp"
    else
        "$BIN" "$WEIGHTS" \
            --data-dir "$FLAT_DIR" --dims "$DIMS" --ext .f32 \
            --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
            --phase no-comp \
            --out-dir "$BASELINE_DIR" --name warpx_field \
            --no-verify 2>&1 | tail -5
    fi
    echo ""
fi

# ============================================================
# Phase 2b: (X1, delta) sweep
# ============================================================
echo ">>> Phase 2b: (X1, X2-X1) threshold sweep"
X1_VALUES=(0.05 0.10 0.20 0.30)
DELTA_VALUES=(0.05 0.10 0.20 0.30)
TOTAL=$(( ${#X1_VALUES[@]} * ${#DELTA_VALUES[@]} ))
RUN=0

for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        RUN=$((RUN + 1))
        x2=$(echo "$x1 + $delta" | bc -l)
        OUT_DIR="$SWEEP_DIR/x1_${x1}_delta_${delta}"

        if [ -f "$OUT_DIR/benchmark_warpx_field.csv" ]; then
            echo "[$RUN/$TOTAL] x1=$x1 delta=$delta — already done, skipping."
            continue
        fi

        echo "[$RUN/$TOTAL] x1=$x1 delta=$delta (X2=$x2)"
        mkdir -p "$OUT_DIR"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  DRY_RUN: nn-rl+exp50 mape=$x1 explore-thresh=$x2"
            continue
        fi

        NO_RANKING=1 "$BIN" "$WEIGHTS" \
            --data-dir "$FLAT_DIR" --dims "$DIMS" --ext .f32 \
            --chunk-mb "$CHUNK_MB" --error-bound "$ERROR_BOUND" \
            --phase nn-rl+exp50 \
            --mape "$x1" --explore-thresh "$x2" \
            --lr "$SGD_LR" --explore-k "$EXPLORE_K" \
            --w0 $W0 --w1 $W1 --w2 $W2 \
            --out-dir "$OUT_DIR" --name warpx_field \
            --no-verify \
            2>&1 | tail -3
    done
done

echo ""
echo "============================================================"
echo "WarpX threshold sweep complete: $TOTAL configurations"
echo "Results: $SWEEP_DIR/"
echo ""
echo "Generate heatmaps:"
echo "  python3 $SCRIPT_DIR/4.2.1_plot_threshold_sweep.py $SWEEP_DIR"
echo "============================================================"
