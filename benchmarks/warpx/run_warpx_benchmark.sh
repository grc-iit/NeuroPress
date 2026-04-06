#!/bin/bash
# ============================================================
# WarpX VPIC-Compatible Benchmark
#
# Two-phase approach:
#   Phase 1: Run WarpX LWFA simulation with WARPX_DUMP_FIELDS=1
#            to dump raw .f32 field files per component per FAB
#   Phase 2: Run generic_benchmark per policy for full 12-phase
#            sweep with VPIC-compatible CSV output
#
# Data evolution: Laser wakefield acceleration (LWFA) creates
# evolving electromagnetic fields — laser pulse generates plasma
# wake, accelerating electrons. E/B-fields develop from smooth
# sinusoidal to complex wake structure over ~200 steps.
#
# Usage:
#   bash benchmarks/warpx/run_warpx_benchmark.sh
#
# Environment variables:
#   WARPX_BIN        Path to warpx binary
#   WARPX_INPUTS     Path to LWFA inputs file
#   WARPX_MAX_STEP   Simulation steps [200]
#   WARPX_DIAG_INT   Diagnostic interval [10]
#   WARPX_NCELL      Grid cells "nx ny nz" [32 32 256]
#   CHUNK_MB         HDF5 chunk size [4]
#   POLICIES         NN policies [balanced,ratio]
#   ERROR_BOUND      Lossy error bound [0.01]
#   RESULTS_DIR      Output directory [auto]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──
WARPX_BIN="${WARPX_BIN:-$HOME/src/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
WARPX_MAX_STEP=${WARPX_MAX_STEP:-200}
WARPX_DIAG_INT=${WARPX_DIAG_INT:-10}
WARPX_NCELL="${WARPX_NCELL:-32 32 256}"
CHUNK_MB=${CHUNK_MB:-4}
POLICIES=${POLICIES:-"balanced,ratio"}
ERROR_BOUND=${ERROR_BOUND:-0.01}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
SGD_LR="${SGD_LR:-0.2}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"

# Derived
N_TIMESTEPS=$(( WARPX_MAX_STEP / WARPX_DIAG_INT ))
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/warpx_eval_ms${WARPX_MAX_STEP}_diag${WARPX_DIAG_INT}_chunk${CHUNK_MB}mb}"

export LD_LIBRARY_PATH="$GPUC_DIR/build:$GPUC_DIR/examples:/tmp/hdf5-install/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "WarpX VPIC-Compatible Benchmark (LWFA)"
echo "============================================================"
echo "  Binary:        $WARPX_BIN"
echo "  Inputs:        $WARPX_INPUTS"
echo "  Generic BM:    $GENERIC_BIN"
echo "  Grid:          $WARPX_NCELL"
echo "  Max step:      $WARPX_MAX_STEP"
echo "  Diag interval: $WARPX_DIAG_INT (~$N_TIMESTEPS timesteps)"
echo "  Chunk:         ${CHUNK_MB} MB"
echo "  Error bound:   $ERROR_BOUND"
echo "  Policies:      $POLICIES"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Check binaries ──
if [ ! -x "$WARPX_BIN" ]; then
    echo "ERROR: WarpX binary not found: $WARPX_BIN"
    echo "  Set WARPX_BIN to your warpx.3d binary"
    exit 1
fi
if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    exit 1
fi

# ============================================================
# Phase 1: Run WarpX LWFA with raw field dumping
# ============================================================
echo ">>> Phase 1: Run WarpX LWFA simulation"
echo ""

RAW_DIR="$RESULTS_DIR/raw_fields"
mkdir -p "$RAW_DIR"

echo "  Running WarpX: $WARPX_MAX_STEP steps, diagnostics every $WARPX_DIAG_INT"
export WARPX_DUMP_FIELDS=1
export WARPX_DUMP_DIR="$RAW_DIR"

cd "$RESULTS_DIR"
"$WARPX_BIN" "$WARPX_INPUTS" \
    warpx.max_step="$WARPX_MAX_STEP" \
    amr.n_cell="$WARPX_NCELL" \
    diagnostics.diags_names=diag1 \
    diag1.intervals="$WARPX_DIAG_INT" \
    diag1.diag_type=Full \
    diag1.format=gpucompress \
    gpucompress.weights_path="$WEIGHTS" \
    gpucompress.algorithm=auto \
    gpucompress.policy=ratio \
    gpucompress.error_bound="$ERROR_BOUND" \
    gpucompress.chunk_bytes=$((CHUNK_MB * 1024 * 1024)) \
    > warpx_dump.log 2>&1 || {
        echo "  WARNING: WarpX exited with error (check warpx_dump.log)"
    }
cd "$GPUC_DIR"

# Count dumped directories
N_DIRS=$(ls -d "$RAW_DIR"/diag* 2>/dev/null | wc -l)
echo "  Dumped: $N_DIRS timestep directories"

if [ "$N_DIRS" -eq 0 ]; then
    echo "ERROR: No field directories dumped. Check warpx_dump.log"
    exit 1
fi

# Determine field size from first .f32 file
FIRST_FILE=$(find "$RAW_DIR" -name "*.f32" -type f | head -1)
if [ -z "$FIRST_FILE" ]; then
    echo "ERROR: No .f32 files found in $RAW_DIR"
    exit 1
fi
FIRST_SIZE=$(stat -c%s "$FIRST_FILE")
N_FLOATS=$((FIRST_SIZE / 4))
DIMS="${N_FLOATS},1"
echo "  Per-component: $N_FLOATS floats ($((FIRST_SIZE / 1024 / 1024)) MB), dims=$DIMS"

# Flatten all .f32 files into a single directory
FLAT_DIR="$RESULTS_DIR/flat_fields"
mkdir -p "$FLAT_DIR"
for ts_dir in "$RAW_DIR"/diag*; do
    ts_name=$(basename "$ts_dir")
    for f in "$ts_dir"/*.f32; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$FLAT_DIR/${ts_name}_$(basename "$f")"
    done
done
N_FLAT=$(ls "$FLAT_DIR"/*.f32 2>/dev/null | wc -l)
echo "  Flattened: $N_FLAT total field files"
echo ""

# ============================================================
# Phase 2: Run generic_benchmark per policy
# ============================================================
echo ">>> Phase 2: Run multi-phase benchmark sweep"
echo ""

declare -A POL_W0 POL_W1 POL_W2
POL_W0[balanced]=1.0;  POL_W1[balanced]=1.0;  POL_W2[balanced]=1.0
POL_W0[ratio]=0.0;     POL_W1[ratio]=0.0;     POL_W2[ratio]=1.0
POL_W0[speed]=1.0;     POL_W1[speed]=1.0;     POL_W2[speed]=0.0

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"

for policy in "${POL_ARRAY[@]}"; do
    w0=${POL_W0[$policy]}
    w1=${POL_W1[$policy]}
    w2=${POL_W2[$policy]}

    OUT_DIR="$RESULTS_DIR/${policy}"
    mkdir -p "$OUT_DIR"

    echo "  ── Policy: $policy (w0=$w0, w1=$w1, w2=$w2) ──"
    echo "    Running generic_benchmark on $N_FLAT field files..."

    "$GENERIC_BIN" "$WEIGHTS" \
        --data-dir "$FLAT_DIR" \
        --dims "$DIMS" \
        --ext .f32 \
        --chunk-mb "$CHUNK_MB" \
        --out-dir "$OUT_DIR" \
        --name "warpx_lwfa" \
        --w0 "$w0" --w1 "$w1" --w2 "$w2" \
        --lr "$SGD_LR" \
        --mape "$SGD_MAPE" \
        --explore-k "$EXPLORE_K" \
        --explore-thresh "$EXPLORE_THRESH" \
        --error-bound "$ERROR_BOUND" \
        > "$OUT_DIR/benchmark.log" 2>&1 || {
            echo "    WARNING: benchmark exited with error"
        }

    if [ -f "$OUT_DIR/benchmark_warpx_lwfa.csv" ]; then
        echo "    Summary:"
        tail -n +2 "$OUT_DIR/benchmark_warpx_lwfa.csv" | \
            awk -F, '{ printf "      %-20s ratio=%-6s write=%-8s\n", $3, $11, $5 }'
    fi
    echo ""
done

echo "============================================================"
echo "WarpX VPIC-Compatible Benchmark Complete"
echo "  Results:   $RESULTS_DIR"
echo "  Per-policy: ${POL_ARRAY[*]}"
echo "============================================================"
