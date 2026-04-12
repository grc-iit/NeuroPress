#!/bin/bash
# ============================================================
# bench_tests/nyx.sh — Deploy Nyx astrophysics AMR simulation
#
# Nyx simulates a Sedov blast wave: a point explosion expanding
# into uniform medium.  Each plot-file dump contains 6 state
# variables (density, 3-momentum, total energy, species) on a
# uniform 3-D grid of float32 values.
#
# Data per snapshot = NYX_NCELL^3 × 6 fields × 4 bytes.
# The blast evolves from a smooth initial state (~369× ratio)
# to a developed shock (~141× ratio), providing a strong
# diversity test for adaptive algorithm selection.
#
# Workflow:
#   Phase 1 — Run Nyx with NYX_DUMP_FIELDS=1 to write raw .f32
#              files (one per component per plt* timestep dir).
#   Phase 2 — Flatten those files into a single directory and
#              run generic_benchmark with the requested HDF5 mode.
#
# HDF5 MODES (HDF5_MODE):
#   default  — No compression.  generic_benchmark runs the no-comp
#              phase.  Measures raw HDF5 I/O bandwidth.
#   vol      — GPUCompress HDF5 VOL.  generic_benchmark runs the
#              requested PHASE and POLICY.
#
# PARAMETERS THAT AFFECT I/O VOLUME:
#   NYX_NCELL        Grid cells per dimension.  Data per dump:
#                    NCELL=32   →  ~0.8 MB  (smoke test)
#                    NCELL=64   →  ~6 MB    (small, default)
#                    NCELL=128  →  ~48 MB   (medium)
#                    NCELL=256  →  ~384 MB  (large)
#   NYX_MAX_STEP     Total simulation steps.  More steps means
#                    more developed and diverse snapshots.
#   NYX_PLOT_INT     Steps between dumps.  n_dumps = MAX_STEP /
#                    PLOT_INT.  Higher intervals → data evolves
#                    more between snapshots.
#   CHUNK_MB         HDF5 chunk size for generic_benchmark.
#
# PARAMETERS THAT AFFECT DATA VARIETY (compressibility):
#   NYX_NCELL        Larger grids resolve the shock front more
#                    finely; fully-resolved shock is much harder
#                    to compress than a smeared one.
#   NYX_PLOT_INT     More steps per interval → sharper gradients
#                    and lower compressibility at later stages.
#   NYX_MAX_STEP     How far the blast evolves.  Early snapshots
#                    (smooth) are far more compressible than late
#                    stage (turbulent) ones.
#   ERROR_BOUND      Lossy error bound (0.0 = lossless).  Non-zero
#                    values trade accuracy for ratio.
#
# USAGE:
#   bash bench_tests/nyx.sh
#
#   # Large run, VOL mode, ratio policy
#   NYX_NCELL=256 NYX_MAX_STEP=100 HDF5_MODE=vol POLICY=ratio \
#       bash bench_tests/nyx.sh
#
#   # Smoke test, default HDF5
#   NYX_NCELL=32 NYX_MAX_STEP=20 NYX_PLOT_INT=10 HDF5_MODE=default \
#       bash bench_tests/nyx.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HDF5 mode ──────────────────────────────────────────────
# default : no-comp (uncompressed HDF5 baseline)
# vol     : GPUCompress HDF5 VOL
HDF5_MODE=${HDF5_MODE:-default}

# ── I/O volume parameters ──────────────────────────────────
NYX_NCELL=${NYX_NCELL:-64}          # Grid cells per dim; data = NCELL^3*6*4 B
NYX_MAX_STEP=${NYX_MAX_STEP:-30}    # Total simulation steps
NYX_PLOT_INT=${NYX_PLOT_INT:-10}    # Steps between plot-file dumps
CHUNK_MB=${CHUNK_MB:-4}             # HDF5 chunk size in MB (generic_benchmark)
VERIFY=${VERIFY:-1}                 # 1 = bitwise readback verify

# ── Data variety / physics parameters ──────────────────────
# ERROR_BOUND: 0.0 = lossless.  Typical: 1e-4 (high quality) – 1e-2 (acceptable).
ERROR_BOUND=${ERROR_BOUND:-0.0}

# ── VOL-mode compression parameters ────────────────────────
# PHASE (HDF5_MODE=vol only):
#   fixed   : lz4 | snappy | deflate | gdeflate | zstd | ans | cascaded | bitcomp
#   adaptive: nn | nn-rl | nn-rl+exp50
PHASE=${PHASE:-lz4}
# POLICY (nn* phases only):
#   balanced → equal speed + ratio
#   ratio    → maximise compression ratio
#   speed    → maximise throughput
POLICY=${POLICY:-balanced}

# ── Paths ───────────────────────────────────────────────────
NYX_BIN_A="${SIMS_DIR:-$HOME/sims}/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests"
NYX_BIN_B="${SIMS_DIR:-$HOME/sims}/Nyx/build-gpucompress/Exec/MiniSB/nyx_MiniSB"
NYX_BIN="${NYX_BIN:-}"
[ -z "$NYX_BIN" ] && { [ -x "$NYX_BIN_A" ] && NYX_BIN="$NYX_BIN_A" || NYX_BIN="$NYX_BIN_B"; }
GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Derived ─────────────────────────────────────────────────
N_DUMPS=$(( NYX_MAX_STEP / NYX_PLOT_INT ))
DATA_MB=$(python3 -c "print(f'{$NYX_NCELL**3 * 6 * 4 / 1048576:.1f}')")
MAX_GRID=$(( NYX_NCELL > 128 ? 128 : NYX_NCELL ))

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
RESULTS_DIR="${RESULTS_DIR:-$GPUC_DIR/benchmarks/nyx/results/bench_n${NYX_NCELL}_ms${NYX_MAX_STEP}_${HDF5_MODE}${EB_TAG}${VERIFY_TAG}}"

echo "============================================================"
echo "  Nyx Benchmark Deploy  (Sedov Blast Wave)"
echo "============================================================"
echo "  HDF5 mode   : $HDF5_MODE"
echo "  Grid        : ${NYX_NCELL}^3"
echo "  Data/dump   : ~${DATA_MB} MB (6 state vars: rho, 3-mom, E, species)"
echo "  Dumps       : $N_DUMPS  (every $NYX_PLOT_INT steps, max=$NYX_MAX_STEP)"
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

if [ ! -x "$NYX_BIN" ]; then
    echo "ERROR: Nyx binary not found."
    echo "  Tried: $NYX_BIN_A"
    echo "         $NYX_BIN_B"
    echo "  Set NYX_BIN or build Nyx with GPUCompress support."
    exit 1
fi
if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    echo "  Build: cmake --build $GPUC_DIR/build --target generic_benchmark"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ============================================================
# Phase 1: Run Nyx, dump raw .f32 field files into plt* dirs
# ============================================================
RAW_DIR="$RESULTS_DIR/raw_fields"
mkdir -p "$RAW_DIR"

INPUT_FILE="$RESULTS_DIR/inputs.sedov"
cat > "$INPUT_FILE" << EOF
# Sedov blast wave — based on Nyx/Exec/HydroTests/inputs.regtest.sedov
amr.n_cell         = $NYX_NCELL $NYX_NCELL $NYX_NCELL
amr.max_level      = 0
amr.max_grid_size  = $MAX_GRID
amr.ref_ratio      = 2 2 2 2
amr.regrid_int     = 2
amr.blocking_factor = 4
amr.plot_int       = $NYX_PLOT_INT
amr.check_int      = 0

max_step       = $NYX_MAX_STEP
stop_time      = -1

geometry.coord_sys   = 0
geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 1.0 1.0 1.0
geometry.is_periodic = 0 0 0

# Outflow BCs on all faces (2 = outflow)
nyx.lo_bc          = 2 2 2
nyx.hi_bc          = 2 2 2

# Hydro
nyx.do_hydro       = 1
nyx.do_grav        = 0
nyx.do_santa_barbara = 0
nyx.ppm_type       = 0
nyx.init_shrink    = 0.01
nyx.cfl            = 0.5
nyx.dt_cutoff      = 5.e-20
nyx.change_max     = 1.1

# Comoving (required by Nyx, set for non-cosmological use)
nyx.comoving_OmM   = 1.0
nyx.comoving_OmB   = 1.0
nyx.comoving_h     = 0.0
nyx.initial_z      = 0.0

# Species
nyx.h_species      = 0.76
nyx.he_species     = 0.24

# Problem setup: Sedov blast (prob_type=33)
prob.prob_type     = 33
prob.r_init        = 0.01
prob.p_ambient     = 1.e-5
prob.dens_ambient  = 1.0
prob.exp_energy    = 1.0
prob.nsub          = 10

# GPUCompress integration
nyx.use_gpucompress       = 1
nyx.gpucompress_weights   = $WEIGHTS
nyx.gpucompress_algorithm = auto
nyx.gpucompress_policy    = ratio
nyx.gpucompress_verify    = 0
nyx.gpucompress_chunk_mb  = $CHUNK_MB

# Prevent AMReX from pre-allocating nearly all GPU memory at startup
amrex.the_arena_init_size = 0
amrex.the_async_arena_init_size = 0
EOF

echo ">>> Phase 1: Nyx Sedov blast ($NYX_MAX_STEP steps, dump every $NYX_PLOT_INT)"

NYX_DUMP_FIELDS=1 \
NYX_DUMP_DIR="$RAW_DIR" \
    "$NYX_BIN" "$INPUT_FILE" \
    > "$RESULTS_DIR/nyx_sim.log" 2>&1
NYX_STATUS=$?
[ $NYX_STATUS -ne 0 ] && echo "  WARNING: Nyx exited $NYX_STATUS (check $RESULTS_DIR/nyx_sim.log)"

N_DIRS=$(ls -d "$RAW_DIR"/plt* 2>/dev/null | wc -l)
echo "  Dumped: $N_DIRS plt* directories"
if [ "$N_DIRS" -eq 0 ]; then
    echo "ERROR: No plt* directories in $RAW_DIR — check $RESULTS_DIR/nyx_sim.log"
    exit 1
fi

# Locate the first .f32 file to determine per-component size
FIRST_FILE=$(find "$RAW_DIR" -name "*.f32" -type f | head -1)
if [ -z "$FIRST_FILE" ]; then
    echo "ERROR: No .f32 files found under $RAW_DIR"
    exit 1
fi
N_FLOATS=$(( $(stat -c%s "$FIRST_FILE") / 4 ))
# Nyx dumps each component as a 1-D blob: use flat (N,1) dims
DIMS="${N_FLOATS},1"
echo "  Per-component: $N_FLOATS floats, dims=$DIMS"

# Flatten: symlink all per-component .f32 files into a single directory
# so generic_benchmark treats each file as one "timestep" chunk.
FLAT_DIR="$RESULTS_DIR/flat_fields"
mkdir -p "$FLAT_DIR"
for ts_dir in "$RAW_DIR"/plt*; do
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
    --name "nyx_n${NYX_NCELL}" \
    $EB_ARG \
    $VERIFY_ARG \
    --phase "$BENCH_PHASE" \
    --w0 $W0 --w1 $W1 --w2 $W2 \
    --lr 0.2 --mape 0.10 \
    --explore-k 4 --explore-thresh 0.20 \
    --out-dir "$BENCH_DIR" \
    > "$BENCH_DIR/nyx_bench.log" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "PASS  Log: $BENCH_DIR/nyx_bench.log"
else
    echo "FAIL (exit $STATUS)  Log: $BENCH_DIR/nyx_bench.log"
    tail -20 "$BENCH_DIR/nyx_bench.log"
    exit 1
fi
