#!/bin/bash
# ============================================================
# bench_tests/warpx.sh — Deploy WarpX laser wakefield simulation
#
# WarpX simulates laser wakefield acceleration (LWFA): a high-
# intensity laser pulse propagates through a plasma and drives
# a co-moving wake that accelerates electrons.  EM fields evolve
# from smooth sinusoidal (laser front) to complex nonlinear wake
# structures over ~200 steps — ideal for adaptive compression.
#
# Each diagnostic dump writes 6 field components (Ex Ey Ez Bx By Bz)
# across all FABs as float32.  Total data per dump depends on grid
# resolution.
#
# Workflow:
#   Phase 1 — Run WarpX with WARPX_DUMP_FIELDS=1 to write raw .f32
#              files (one per component per diag* timestep dir).
#   Phase 2 — Flatten those files and run generic_benchmark with
#              the requested HDF5 mode.
#
# HDF5 MODES (HDF5_MODE):
#   default  — No compression.  generic_benchmark runs no-comp.
#              Measures raw HDF5 I/O bandwidth.
#   vol      — GPUCompress HDF5 VOL.  generic_benchmark runs the
#              requested PHASE and POLICY.
#
# PARAMETERS THAT AFFECT I/O VOLUME:
#   WARPX_NCELL      Grid cells "nx ny nz".  Data per dump =
#                    nx*ny*nz * 6 fields * 4 bytes.
#                    "16 16 128"  →  ~0.8 MB  (smoke test)
#                    "32 32 256"  →  ~6 MB    (small, default)
#                    "64 64 512"  →  ~48 MB   (medium)
#                    "128 128 1024" → ~384 MB (large)
#   WARPX_MAX_STEP   Total simulation steps.
#   WARPX_DIAG_INT   Steps between diagnostic dumps.
#                    n_dumps = MAX_STEP / DIAG_INT.
#   CHUNK_MB         HDF5 chunk size for generic_benchmark.
#
# PARAMETERS THAT AFFECT DATA VARIETY (compressibility):
#   WARPX_NCELL      Finer grids resolve small-scale wake structures
#                    (electron spikes, wave breaking) that are harder
#                    to compress.
#   WARPX_DIAG_INT   More steps between dumps → later-stage data
#                    with more nonlinear wake structure.
#   WARPX_MAX_STEP   Early snapshots (smooth laser) compress much
#                    better than late-stage (turbulent wake) ones.
#   ERROR_BOUND      Lossy error bound (0.0 = lossless).  Typical
#                    LWFA post-processing tolerance: 1e-3 – 1e-2.
#
# USAGE:
#   bash bench_tests/warpx.sh
#
#   # Large run, VOL mode, ratio policy
#   WARPX_NCELL="128 128 1024" WARPX_MAX_STEP=200 \
#       HDF5_MODE=vol POLICY=ratio bash bench_tests/warpx.sh
#
#   # Smoke test, default HDF5
#   WARPX_NCELL="16 16 128" WARPX_MAX_STEP=20 WARPX_DIAG_INT=10 \
#       HDF5_MODE=default bash bench_tests/warpx.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HDF5 mode ──────────────────────────────────────────────
# default : no-comp (uncompressed HDF5 baseline)
# vol     : GPUCompress HDF5 VOL
HDF5_MODE=${HDF5_MODE:-default}

# ── I/O volume parameters ──────────────────────────────────
WARPX_NCELL="${WARPX_NCELL:-32 32 256}"  # Grid cells "nx ny nz"
WARPX_MAX_STEP=${WARPX_MAX_STEP:-30}     # Total simulation steps
WARPX_DIAG_INT=${WARPX_DIAG_INT:-10}    # Steps between diagnostic dumps
CHUNK_MB=${CHUNK_MB:-4}                  # HDF5 chunk size in MB
VERIFY=${VERIFY:-1}                      # 1 = bitwise readback verify

# ── Data variety / physics parameters ──────────────────────
# ERROR_BOUND: 0.0 = lossless.  LWFA: 1e-3 – 1e-2 typical.
ERROR_BOUND=${ERROR_BOUND:-0.0}
# Optional: override AMR blocking parameters (leave empty for defaults)
WARPX_MAX_GRID_SIZE="${WARPX_MAX_GRID_SIZE:-}"
WARPX_BLOCKING_FACTOR="${WARPX_BLOCKING_FACTOR:-}"

# ── VOL-mode compression parameters ────────────────────────
# PHASE (HDF5_MODE=vol only):
#   fixed   : lz4 | snappy | deflate | gdeflate | zstd | ans | cascaded | bitcomp
#   adaptive: nn | nn-rl | nn-rl+exp50
PHASE=${PHASE:-lz4}
# POLICY (nn* phases only):
#   balanced → equal speed + ratio
#   ratio    → maximise compression ratio  (recommended for LWFA)
#   speed    → maximise throughput
POLICY=${POLICY:-ratio}

# ── Paths ───────────────────────────────────────────────────
WARPX_BIN="${WARPX_BIN:-${SIMS_DIR:-$HOME/sims}/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-${SIMS_DIR:-$HOME/sims}/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Derived ─────────────────────────────────────────────────
N_DUMPS=$(( WARPX_MAX_STEP / WARPX_DIAG_INT ))
NCELL_COMPACT=$(echo "$WARPX_NCELL" | tr ' ' 'x')

case "$POLICY" in
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *)        W0=1.0; W1=1.0; W2=1.0 ;;
esac

if [ "$HDF5_MODE" = "default" ]; then
    BENCH_PHASE="no-comp"
else
    BENCH_PHASE="$PHASE"
fi

VERIFY_TAG=""; [ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
EB_TAG=""; [ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && EB_TAG="_lossy${ERROR_BOUND}"
RESULTS_DIR="${RESULTS_DIR:-$GPUC_DIR/benchmarks/warpx/results/bench_${NCELL_COMPACT}_ms${WARPX_MAX_STEP}_${HDF5_MODE}${EB_TAG}${VERIFY_TAG}}"

echo "============================================================"
echo "  WarpX Benchmark Deploy  (LWFA)"
echo "============================================================"
echo "  HDF5 mode   : $HDF5_MODE"
echo "  Grid        : $WARPX_NCELL"
echo "  Max step    : $WARPX_MAX_STEP"
echo "  Diag every  : $WARPX_DIAG_INT steps  (~$N_DUMPS dumps)"
echo "  Chunk       : ${CHUNK_MB} MB"
echo "  Error bound : $ERROR_BOUND"
echo "  Verify      : $VERIFY"
if [ "$HDF5_MODE" = "vol" ]; then
echo "  Phase       : $BENCH_PHASE"
echo "  Policy      : $POLICY  (w0=$W0 w1=$W1 w2=$W2)"
fi
echo ""
echo "  Results: $RESULTS_DIR"
echo "============================================================"
echo ""

if [ ! -x "$WARPX_BIN" ]; then
    echo "ERROR: WarpX binary not found: $WARPX_BIN"
    echo "  Set WARPX_BIN to your warpx.3d binary."
    exit 1
fi
if [ ! -f "$WARPX_INPUTS" ]; then
    echo "ERROR: WarpX inputs file not found: $WARPX_INPUTS"
    echo "  Set WARPX_INPUTS to your LWFA inputs_base_3d file."
    exit 1
fi
if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    echo "  Build: cmake --build $GPUC_DIR/build --target generic_benchmark"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ============================================================
# Phase 1: Run WarpX, dump raw .f32 field files into diag* dirs
# ============================================================
RAW_DIR="$RESULTS_DIR/raw_fields"
mkdir -p "$RAW_DIR"

echo ">>> Phase 1: WarpX LWFA ($WARPX_MAX_STEP steps, dump every $WARPX_DIAG_INT)"

WARPX_DUMP_FIELDS=1 \
WARPX_DUMP_DIR="$RAW_DIR" \
    "$WARPX_BIN" "$WARPX_INPUTS" \
    max_step="$WARPX_MAX_STEP" \
    amr.n_cell="$WARPX_NCELL" \
    ${WARPX_MAX_GRID_SIZE:+amr.max_grid_size="$WARPX_MAX_GRID_SIZE"} \
    ${WARPX_BLOCKING_FACTOR:+amr.blocking_factor="$WARPX_BLOCKING_FACTOR"} \
    diagnostics.diags_names=diag1 \
    diag1.intervals="$WARPX_DIAG_INT" \
    diag1.diag_type=Full \
    diag1.format=gpucompress \
    gpucompress.weights_path="$WEIGHTS" \
    gpucompress.algorithm=auto \
    gpucompress.policy=ratio \
    gpucompress.error_bound="$ERROR_BOUND" \
    gpucompress.chunk_bytes=$(( CHUNK_MB * 1024 * 1024 )) \
    > "$RESULTS_DIR/warpx_sim.log" 2>&1
WX_STATUS=$?
[ $WX_STATUS -ne 0 ] && echo "  WARNING: WarpX exited $WX_STATUS (check $RESULTS_DIR/warpx_sim.log)"

N_DIRS=$(ls -d "$RAW_DIR"/diag* 2>/dev/null | wc -l)
echo "  Dumped: $N_DIRS diag* directories"
if [ "$N_DIRS" -eq 0 ]; then
    echo "ERROR: No diag* directories in $RAW_DIR — check $RESULTS_DIR/warpx_sim.log"
    exit 1
fi

# Locate first .f32 file to determine per-component size
FIRST_FILE=$(find "$RAW_DIR" -name "*.f32" -type f | head -1)
if [ -z "$FIRST_FILE" ]; then
    echo "ERROR: No .f32 files found under $RAW_DIR"
    exit 1
fi
N_FLOATS=$(( $(stat -c%s "$FIRST_FILE") / 4 ))
# WarpX dumps each FAB component as a 1-D blob: use flat (1,N) dims
DIMS="1,${N_FLOATS}"
echo "  Per-component: $N_FLOATS floats, dims=$DIMS"

# Flatten: symlink all per-component .f32 files into a single directory
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
echo "  Flattened: $N_FLAT field files"
if [ "$N_FLAT" -eq 0 ]; then
    echo "ERROR: No .f32 files linked into $FLAT_DIR"
    exit 1
fi

# ============================================================
# Phase 2: Run generic_benchmark (HDF5 default or VOL)
# ============================================================
EB_ARG=""
[ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && EB_ARG="--error-bound $ERROR_BOUND"
VERIFY_ARG=""; [ "$VERIFY" = "0" ] && VERIFY_ARG="--no-verify"
BENCH_DIR="$RESULTS_DIR/${HDF5_MODE}_${BENCH_PHASE}"
mkdir -p "$BENCH_DIR"

echo ""
echo ">>> Phase 2: generic_benchmark — mode=$HDF5_MODE phase=$BENCH_PHASE policy=$POLICY"

GPUCOMPRESS_DETAILED_TIMING=1 \
    "$GENERIC_BIN" "$WEIGHTS" \
    --data-dir "$FLAT_DIR" \
    --dims "$DIMS" \
    --ext .f32 \
    --chunk-mb "$CHUNK_MB" \
    --name "warpx_${NCELL_COMPACT}" \
    $EB_ARG \
    $VERIFY_ARG \
    --phase "$BENCH_PHASE" \
    --w0 $W0 --w1 $W1 --w2 $W2 \
    --lr 0.2 --mape 0.10 \
    --explore-k 4 --explore-thresh 0.20 \
    --out-dir "$BENCH_DIR" \
    > "$BENCH_DIR/warpx_bench.log" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "PASS  Log: $BENCH_DIR/warpx_bench.log"
else
    echo "FAIL (exit $STATUS)  Log: $BENCH_DIR/warpx_bench.log"
    tail -20 "$BENCH_DIR/warpx_bench.log"
    exit 1
fi
