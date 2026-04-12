#!/bin/bash
# ============================================================
# Nyx VPIC-Compatible Benchmark
#
# Two-phase approach:
#   Phase 1: Run Nyx Sedov blast with NYX_DUMP_FIELDS=1 to dump
#            raw .f32 field files per component per FArrayBox
#   Phase 2: Run generic_benchmark per component group and policy
#
# Data evolution: Sedov blast wave creates expanding shock front.
# Compression ratio drops from ~369x (initial) to ~141x (late).
# This evolving mix of compressible/incompressible regions is
# ideal for NN algorithm selection benchmarking.
#
# Usage:
#   bash benchmarks/nyx/run_nyx_benchmark.sh
#
# Environment variables:
#   NYX_BIN          Path to Nyx binary (HydroTests or MiniSB)
#   NYX_NCELL        Grid size per dimension [128]
#   NYX_MAX_STEP     Simulation steps [200]
#   NYX_PLOT_INT     Steps between dumps [10]
#   CHUNK_MB         HDF5 chunk size [4]
#   POLICIES         NN policies [balanced,ratio,speed]
#   ERROR_BOUND      Lossy error bound [0.0 = lossless]
#   RESULTS_DIR      Output directory [auto]
#   NO_RANKING       1 to skip ranking profiler (recommended for 256^3+)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──
NYX_BIN_A="$HOME/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests"
NYX_BIN_B="$HOME/Nyx/build-gpucompress/Exec/MiniSB/nyx_MiniSB"
NYX_BIN="${NYX_BIN:-}"
[ -z "$NYX_BIN" ] && { [ -x "$NYX_BIN_A" ] && NYX_BIN="$NYX_BIN_A" || NYX_BIN="$NYX_BIN_B"; }

GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
NYX_NCELL=${NYX_NCELL:-128}
NYX_MAX_STEP=${NYX_MAX_STEP:-200}
NYX_PLOT_INT=${NYX_PLOT_INT:-10}
CHUNK_MB=${CHUNK_MB:-4}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
ERROR_BOUND=${ERROR_BOUND:-0.0}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
SGD_LR="${SGD_LR:-0.2}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"
NO_RANKING="${NO_RANKING:-1}"  # Default OFF for NYX (200 chunks = slow profiler)

# Derived
N_TIMESTEPS=$(( NYX_MAX_STEP / NYX_PLOT_INT ))
NCELLS_TOTAL=$(python3 -c "print($NYX_NCELL**3)" 2>/dev/null || echo "unknown")
# 6 state vars (density, 3-momentum, energy, species), float32
DATA_MB=$(python3 -c "print(f'{$NYX_NCELL**3 * 6 * 4 / 1048576:.1f}')" 2>/dev/null || echo "unknown")
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/vpic_eval_n${NYX_NCELL}_ms${NYX_MAX_STEP}_chunk${CHUNK_MB}mb}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "Nyx VPIC-Compatible Benchmark (Sedov Blast Wave)"
echo "============================================================"
echo "  Binary:        $NYX_BIN"
echo "  Generic BM:    $GENERIC_BIN"
echo "  Grid:          ${NYX_NCELL}^3 ($NCELLS_TOTAL cells)"
echo "  Data/step:     ~$DATA_MB MB (6 state vars, fp32)"
echo "  Max step:      $NYX_MAX_STEP"
echo "  Plot interval: $NYX_PLOT_INT (~$N_TIMESTEPS timesteps)"
echo "  Chunk:         ${CHUNK_MB} MB"
echo "  Policies:      $POLICIES"
echo "  NO_RANKING:    $NO_RANKING"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Check binaries ──
if [ ! -x "$NYX_BIN" ]; then
    echo "ERROR: Nyx binary not found: $NYX_BIN"
    echo "  Set NYX_BIN to point to nyx_HydroTests or nyx_MiniSB"
    exit 1
fi
if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    exit 1
fi

# ============================================================
# Phase 1: Run Nyx Sedov blast with raw field dumping
# ============================================================
echo ">>> Phase 1: Run Nyx Sedov blast simulation"
echo ""

RAW_DIR="$RESULTS_DIR/raw_fields"
mkdir -p "$RAW_DIR"

# Generate Nyx inputs file for Sedov blast wave
INPUT_FILE="$RESULTS_DIR/inputs.sedov"
cat > "$INPUT_FILE" << EOF
# Sedov blast wave — based on Nyx/Exec/HydroTests/inputs.regtest.sedov
amr.n_cell         = $NYX_NCELL $NYX_NCELL $NYX_NCELL
amr.max_level      = 0
amr.max_grid_size  = $(( NYX_NCELL > 128 ? 128 : NYX_NCELL ))
amr.ref_ratio      = 2 2 2 2
amr.regrid_int     = 2
amr.blocking_factor = 4
amr.plot_int       = $NYX_PLOT_INT
amr.check_int      = 0

max_step           = $NYX_MAX_STEP
stop_time          = -1

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

echo "  Running Nyx Sedov: $NYX_MAX_STEP steps, plotting every $NYX_PLOT_INT"
export NYX_DUMP_FIELDS=1
export NYX_DUMP_DIR="$RAW_DIR"

cd "$RESULTS_DIR"
"$NYX_BIN" "$INPUT_FILE" > nyx_dump.log 2>&1 || {
    echo "  WARNING: Nyx exited with error (check nyx_dump.log)"
}
cd "$GPUC_DIR"

# Count dumped field directories
N_DIRS=$(ls -d "$RAW_DIR"/plt* 2>/dev/null | wc -l)
echo "  Dumped: $N_DIRS timestep directories"

if [ "$N_DIRS" -eq 0 ]; then
    echo "ERROR: No field directories dumped. Check nyx_dump.log"
    exit 1
fi

# Determine field dimensions from first .f32 file
FIRST_FILE=$(find "$RAW_DIR" -name "*.f32" -type f | head -1)
if [ -z "$FIRST_FILE" ]; then
    echo "ERROR: No .f32 files found in $RAW_DIR"
    exit 1
fi
FIRST_SIZE=$(stat -c%s "$FIRST_FILE")
N_FLOATS=$((FIRST_SIZE / 4))
DIMS="${N_FLOATS},1"
echo "  Per-component: $N_FLOATS floats ($((FIRST_SIZE / 1024 / 1024)) MB), dims=$DIMS"

# Flatten all .f32 files into a single directory for generic_benchmark
# (each file is one "field" — SGD learns across them)
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

    NO_RANKING=$NO_RANKING \
    "$GENERIC_BIN" "$WEIGHTS" \
        --data-dir "$FLAT_DIR" \
        --dims "$DIMS" \
        --ext .f32 \
        --chunk-mb "$CHUNK_MB" \
        --out-dir "$OUT_DIR" \
        --name "nyx_sedov" \
        --w0 "$w0" --w1 "$w1" --w2 "$w2" \
        --lr "$SGD_LR" \
        --mape "$SGD_MAPE" \
        --explore-k "$EXPLORE_K" \
        --explore-thresh "$EXPLORE_THRESH" \
        ${ERROR_BOUND:+--error-bound "$ERROR_BOUND"} \
        > "$OUT_DIR/benchmark.log" 2>&1 || {
            echo "    WARNING: benchmark exited with error"
        }

    if [ -f "$OUT_DIR/benchmark_nyx_sedov.csv" ]; then
        echo "    Summary:"
        tail -n +2 "$OUT_DIR/benchmark_nyx_sedov.csv" | \
            awk -F, '{ printf "      %-20s ratio=%-6s write=%-8s\n", $3, $11, $5 }'
    fi
    echo ""
done

echo "============================================================"
echo "Nyx VPIC-Compatible Benchmark Complete"
echo "  Results:   $RESULTS_DIR"
echo "  Per-policy: ${POL_ARRAY[*]}"
echo "============================================================"
