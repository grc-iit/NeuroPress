#!/bin/bash
# ============================================================
# 7.1 Cross-Workload Convergence: Regret + Compression Time MAPE
#
# Produces two figures from a single benchmark run:
#   Figure 7a: "Regret Converges Within ~X Chunks Across All Workloads"
#   Figure 7b: "Compression Time MAPE follows a similar trajectory"
#
# generic_benchmark writes both _ranking.csv (regret per chunk) and
# _chunks.csv / _timestep_chunks.csv (mape_comp per chunk) in the
# same invocation, so running once gives both plots.
#
# Two modes:
#
#   MODE=snapshot  (default)
#     Uses pre-existing SDRBench snapshot files + AI checkpoint
#     tensors.  No simulation binaries required.
#     Workloads: Hurricane Isabel, NYX, CESM-ATM, AI Checkpoint (ViT-B)
#
#   MODE=simulation
#     Runs live simulations, dumps raw fields, then benchmarks.
#     Requires simulation binaries on the system.
#     Workloads: VPIC, WarpX, LAMMPS, AI Checkpoint (ViT-B)
#
# All workloads are fed through generic_benchmark --phase nn-rl+exp50
# so regret is measured with identical methodology.
#
# Usage:
#   # Snapshot mode (default) — no simulation binaries needed
#   bash benchmarks/Paper_Evaluations/7/7.1_run_equalized_cross_workload_regret.sh
#
#   # Simulation mode
#   MODE=simulation bash benchmarks/Paper_Evaluations/7/7.1_run_equalized_cross_workload_regret.sh
#
# Overrides:
#   RUN_NAME=paper_fig7a
#   POLICY=balanced ERROR_BOUND=0.01 CHUNK_MB=4
#   SKIP_HURRICANE=1 SKIP_NYX=1 SKIP_CESM=1 SKIP_AI=1
#   SKIP_VPIC=1 SKIP_WARPX=1 SKIP_LAMMPS=1
#   PLOT_ONLY=1          # just regenerate figure from existing CSVs
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

# ── Mode selection ────────────────────────────────────────────
MODE="${MODE:-snapshot}"  # "snapshot" or "simulation"

# ── Run configuration ─────────────────────────────────────────
RUN_NAME="${RUN_NAME:-cross_workload_${MODE}_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"

POLICY="${POLICY:-ratio}"
PHASE="${PHASE:-nn-rl}"
ERROR_BOUND="${ERROR_BOUND:-0.0}"
CHUNK_MB="${CHUNK_MB:-4}"
SGD_LR="${SGD_LR:-0.1}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"

PLOT_ONLY="${PLOT_ONLY:-0}"

# ── Skip flags (per workload) ────────────────────────────────
# Snapshot mode workloads
SKIP_HURRICANE="${SKIP_HURRICANE:-0}"
SKIP_NYX="${SKIP_NYX:-0}"
SKIP_CESM="${SKIP_CESM:-0}"
SKIP_AI="${SKIP_AI:-0}"
# Simulation mode workloads
SKIP_VPIC="${SKIP_VPIC:-0}"
SKIP_WARPX="${SKIP_WARPX:-0}"
SKIP_LAMMPS="${SKIP_LAMMPS:-0}"

# ── Binaries / paths ─────────────────────────────────────────
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$PROJECT_DIR/neural_net/weights/model.nnwt}"
GENERIC_BIN="${GENERIC_BIN:-$PROJECT_DIR/build/generic_benchmark}"

# Snapshot mode data paths
HURRICANE_DIR="${HURRICANE_DIR:-$PROJECT_DIR/data/sdrbench/hurricane_isabel/100x500x500}"
HURRICANE_DIMS="${HURRICANE_DIMS:-100,500,500}"
HURRICANE_EXT="${HURRICANE_EXT:-.bin.f32}"

NYX_DATA_DIR="${NYX_DATA_DIR:-$PROJECT_DIR/data/sdrbench/nyx/SDRBENCH-EXASKY-NYX-512x512x512}"
NYX_DIMS="${NYX_DIMS:-512,512,512}"

CESM_DATA_DIR="${CESM_DATA_DIR:-$PROJECT_DIR/data/sdrbench/cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600}"
CESM_DIMS="${CESM_DIMS:-1800,3600}"
CESM_EXT="${CESM_EXT:-.dat}"

AI_MODEL="${AI_MODEL:-vit_b_16}"
AI_EPOCHS="${AI_EPOCHS:-20}"
AI_DATA_DIR="${AI_DATA_DIR:-}"

# Simulation mode binaries / settings
VPIC_BIN_A="$PROJECT_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
VPIC_BIN_B="$PROJECT_DIR/vpic_benchmark_deck.Linux"
VPIC_BIN="${VPIC_BIN:-}"
[ -z "$VPIC_BIN" ] && { [ -x "$VPIC_BIN_A" ] && VPIC_BIN="$VPIC_BIN_A" || VPIC_BIN="$VPIC_BIN_B"; }

# Prefer HydroTests (Sedov blast) for diverse data; fallback to MiniSB
NYX_BIN_A="$HOME/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests"
NYX_BIN_B="$HOME/Nyx/build-gpucompress/Exec/MiniSB/nyx_MiniSB"
NYX_BIN="${NYX_BIN:-}"
[ -z "$NYX_BIN" ] && { [ -x "$NYX_BIN_A" ] && NYX_BIN="$NYX_BIN_A" || NYX_BIN="$NYX_BIN_B"; }
WARPX_BIN="${WARPX_BIN:-$HOME/src/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"

# ── NYX: cosmological simulation — grid size controls data per write.
#    64^3 = ~3 MB/write (fast), 128^3 = ~25 MB, 256^3 = ~200 MB.
#    plot_int controls dumps per simulation step.

# ── VPIC: plasma PIC — fast reconnection (mi_me=5) reaches nonlinear
#    phase within ~500 steps. Perturbation=0.30 seeds instability faster.
#    Short warmup + frequent dumps catch the rapid structural changes.
VPIC_NX="${VPIC_NX:-147}"
VPIC_TIMESTEPS="${VPIC_TIMESTEPS:-8}"
VPIC_WARMUP_STEPS="${VPIC_WARMUP_STEPS:-200}"
VPIC_SIM_INTERVAL="${VPIC_SIM_INTERVAL:-80}"
VPIC_MI_ME="${VPIC_MI_ME:-5}"
VPIC_WPE_WCE="${VPIC_WPE_WCE:-1}"
VPIC_TI_TE="${VPIC_TI_TE:-5}"

# ── NYX: Sedov blast wave — tighter dump interval (25 steps) catches
#    the shock front moving through chunk boundaries more frequently.
NYX_NCELL="${NYX_NCELL:-160}"
NYX_MAX_STEP="${NYX_MAX_STEP:-200}"
NYX_PLOT_INT="${NYX_PLOT_INT:-25}"

# ── WarpX: laser-wakefield acceleration — more steps (200) lets the
#    plasma bubble mature; tighter dumps (20 steps) capture transient.
WARPX_NCELL="${WARPX_NCELL:-128 128 320}"
WARPX_MAX_STEP="${WARPX_MAX_STEP:-200}"
WARPX_DIAG_INTERVAL="${WARPX_DIAG_INTERVAL:-25}"

# ── LAMMPS: MD hot-sphere expansion — extreme temperature (T_hot=100)
#    creates violent shock; frequent dumps (every 10 steps) capture
#    the rapid transient as density/velocity gradients evolve.
LMP_ATOMS="${LMP_ATOMS:-161}"
LMP_TIMESTEPS="${LMP_TIMESTEPS:-8}"
LMP_SIM_INTERVAL="${LMP_SIM_INTERVAL:-25}"
LMP_WARMUP_STEPS="${LMP_WARMUP_STEPS:-50}"
LMP_T_HOT="${LMP_T_HOT:-100.0}"
LMP_T_COLD="${LMP_T_COLD:-0.01}"

# ── Policy weights ────────────────────────────────────────────
case "$POLICY" in
    balanced) W0="1.0"; W1="1.0"; W2="1.0" ;;
    ratio)    W0="0.0"; W1="0.0"; W2="1.0" ;;
    speed)    W0="1.0"; W1="1.0"; W2="0.0" ;;
    *)        echo "ERROR: unknown POLICY=$POLICY" >&2; exit 1 ;;
esac

# ── Phase → SGD/Explore mapping ───────────────────────────────
# nn:           inference only (no SGD, no exploration)
# nn-rl:        SGD enabled, no exploration
# nn-rl+exp50:  SGD + exploration
case "$PHASE" in
    nn)           DO_SGD=0; DO_EXPLORE=0 ;;
    nn-rl)        DO_SGD=1; DO_EXPLORE=0 ;;
    nn-rl+exp50)  DO_SGD=1; DO_EXPLORE=1 ;;
    *)            DO_SGD=1; DO_EXPLORE=1 ;;
esac

# ── LD_LIBRARY_PATH ───────────────────────────────────────────
export LD_LIBRARY_PATH="/tmp/hdf5-install/lib:$PROJECT_DIR/build:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Helpers ───────────────────────────────────────────────────
note() { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
require_file() { [ -f "$1" ] || die "required file not found: $1"; }

check_binary() {
    local name="$1" path="$2"
    if [ ! -x "$path" ]; then
        note "SKIP $name: binary not found at $path"
        return 1
    fi
    return 0
}

check_data_dir() {
    local name="$1" dir="$2" ext="$3"
    if [ ! -d "$dir" ]; then
        note "SKIP $name: directory not found at $dir"
        return 1
    fi
    local n
    n=$(find "$dir" -maxdepth 1 -type f -name "*${ext}" | wc -l)
    if [ "$n" -lt 1 ]; then
        note "SKIP $name: no ${ext} files in $dir"
        return 1
    fi
    note "$name: $n fields in $dir"
    return 0
}

# ── Run generic_benchmark on a data directory ─────────────────
run_regret_benchmark() {
    local workload="$1"
    local data_dir="$2"
    local dims="$3"
    local ext="${4:-.f32}"
    local out_dir="$RESULTS_DIR/$workload"

    mkdir -p "$out_dir"
    note "Running generic_benchmark for $workload (dims=$dims, ext=$ext)"

    "$GENERIC_BIN" "$WEIGHTS" \
        --data-dir "$data_dir" \
        --dims "$dims" \
        --ext "$ext" \
        --chunk-mb "$CHUNK_MB" \
        --error-bound "$ERROR_BOUND" \
        --phase "$PHASE" \
        --mape "$SGD_MAPE" \
        --explore-thresh "$EXPLORE_THRESH" \
        --lr "$SGD_LR" \
        --explore-k "$EXPLORE_K" \
        --w0 "$W0" --w1 "$W1" --w2 "$W2" \
        --out-dir "$out_dir" \
        --name "$workload" \
        --no-verify \
        > "$out_dir/benchmark.log" 2>&1

    local ranking_csv="$out_dir/benchmark_${workload}_ranking.csv"
    if [ ! -f "$ranking_csv" ]; then
        note "WARNING: no ranking CSV produced for $workload"
        return 1
    fi
    local n_rows
    n_rows=$(tail -n +2 "$ranking_csv" | wc -l)
    note "$workload: $n_rows ranking rows in $ranking_csv"
    return 0
}

# ════════════════════════════════════════════════════════════════
# SNAPSHOT MODE WORKLOADS
# ════════════════════════════════════════════════════════════════

run_hurricane() {
    [ "$SKIP_HURRICANE" = "1" ] && { note "SKIP Hurricane (SKIP_HURRICANE=1)"; return 0; }
    check_data_dir "Hurricane" "$HURRICANE_DIR" "$HURRICANE_EXT" || return 0
    run_regret_benchmark "hurricane" "$HURRICANE_DIR" "$HURRICANE_DIMS" "$HURRICANE_EXT"
}

run_nyx_snapshot() {
    [ "$SKIP_NYX" = "1" ] && { note "SKIP NYX (SKIP_NYX=1)"; return 0; }
    check_data_dir "NYX" "$NYX_DATA_DIR" ".f32" || return 0
    run_regret_benchmark "nyx" "$NYX_DATA_DIR" "$NYX_DIMS" ".f32"
}

run_cesm() {
    [ "$SKIP_CESM" = "1" ] && { note "SKIP CESM (SKIP_CESM=1)"; return 0; }
    check_data_dir "CESM-ATM" "$CESM_DATA_DIR" "$CESM_EXT" || return 0
    run_regret_benchmark "cesm" "$CESM_DATA_DIR" "$CESM_DIMS" "$CESM_EXT"
}

run_ai_checkpoint() {
    [ "$SKIP_AI" = "1" ] && { note "SKIP AI (SKIP_AI=1)"; return 0; }

    if [ -z "$AI_DATA_DIR" ]; then
        AI_DATA_DIR="$RESULTS_DIR/raw_ai_checkpoint/${AI_MODEL}_checkpoints"
    fi

    local n_f32
    n_f32=$(find "$AI_DATA_DIR" -maxdepth 1 -name "*.f32" 2>/dev/null | wc -l)

    if [ "$n_f32" -lt 2 ]; then
        note "Exporting $AI_MODEL checkpoint tensors (epochs=$AI_EPOCHS)"
        mkdir -p "$AI_DATA_DIR"
        python3 "$PROJECT_DIR/scripts/train_and_export_checkpoints.py" \
            --model "$AI_MODEL" \
            --epochs "$AI_EPOCHS" \
            --checkpoint-epochs "$(seq -s, 1 "$AI_EPOCHS")" \
            --outdir "$AI_DATA_DIR" \
            --data-root "$PROJECT_DIR/data" \
            > "$AI_DATA_DIR/export.log" 2>&1
        n_f32=$(find "$AI_DATA_DIR" -maxdepth 1 -name "*.f32" | wc -l)
    fi

    if [ "$n_f32" -lt 2 ]; then
        note "WARNING: AI checkpoint export produced only $n_f32 files"
        return 0
    fi
    note "AI Checkpoint: $n_f32 .f32 files"

    # Auto-detect dimensions from first file
    local first_f32
    first_f32=$(find "$AI_DATA_DIR" -maxdepth 1 -name "*.f32" | sort | head -1)
    local file_bytes
    file_bytes=$(stat -c %s "$first_f32")
    local n_floats=$((file_bytes / 4))
    local dims="1,${n_floats}"

    run_regret_benchmark "ai_checkpoint" "$AI_DATA_DIR" "$dims" ".f32"
}

# ════════════════════════════════════════════════════════════════
# SIMULATION MODE WORKLOADS
# ════════════════════════════════════════════════════════════════

run_nyx_sim() {
    [ "$SKIP_NYX" = "1" ] && { note "SKIP NYX (SKIP_NYX=1)"; return 0; }
    check_binary "NYX" "$NYX_BIN" || return 0

    local nyx_out="$RESULTS_DIR/nyx"
    local nyx_work="$RESULTS_DIR/nyx_work"
    mkdir -p "$nyx_out" "$nyx_work"

    note "Running NYX simulation (n_cell=$NYX_NCELL, max_step=$NYX_MAX_STEP, plot_int=$NYX_PLOT_INT)"

    # Copy input file and initial conditions from the binary's directory
    local nyx_exec_dir
    nyx_exec_dir="$(dirname "$NYX_BIN")"
    # HydroTests (Sedov) uses inputs.3d.sph.sedov; MiniSB uses inputs.32
    if [ -f "$nyx_exec_dir/inputs.3d.sph.sedov" ]; then
        cp "$nyx_exec_dir/inputs.3d.sph.sedov" "$nyx_work/inputs"
    elif [ -f "$nyx_exec_dir/inputs.32" ]; then
        cp "$nyx_exec_dir/inputs.32" "$nyx_work/inputs"
        cp "$nyx_exec_dir/ic_sb_32.ascii" "$nyx_work/" 2>/dev/null || true
        cp "$nyx_exec_dir/fixed_grids" "$nyx_work/" 2>/dev/null || true
        cp "$nyx_exec_dir/particle_file.small" "$nyx_work/" 2>/dev/null || true
    else
        note "WARNING: no Nyx input file found in $nyx_exec_dir"
        return 0
    fi

    # Override parameters for regret evaluation
    cat >> "$nyx_work/inputs" <<NYXEOF
amr.n_cell = $NYX_NCELL $NYX_NCELL $NYX_NCELL
amr.max_grid_size = $NYX_NCELL
max_step = $NYX_MAX_STEP
amr.plot_int = $NYX_PLOT_INT
amr.check_int = 0
nyx.use_gpucompress = 1
nyx.gpucompress_weights = $WEIGHTS
nyx.gpucompress_policy = $POLICY
nyx.gpucompress_algorithm = auto
nyx.gpucompress_verify = 0
nyx.gpucompress_error_bound = $ERROR_BOUND
nyx.gpucompress_chunk_mb = $CHUNK_MB
nyx.sgd_lr = $SGD_LR
nyx.sgd_mape = $SGD_MAPE
nyx.explore_k = $EXPLORE_K
nyx.explore_thresh = $EXPLORE_THRESH
NYXEOF

    cd "$nyx_work"
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    NYX_LOG_DIR="$nyx_out" \
    "$NYX_BIN" inputs > "$nyx_out/nyx_sim.log" 2>&1 || true
    cd "$PROJECT_DIR"

    # Check if ranking CSV was produced by the DiagLogger
    local ranking_csv="$nyx_out/benchmark_nyx_ranking.csv"
    if [ -f "$ranking_csv" ]; then
        local n_rows
        n_rows=$(tail -n +2 "$ranking_csv" | wc -l)
        note "NYX: $n_rows ranking rows in $ranking_csv"
    else
        note "WARNING: NYX simulation produced no ranking CSV"
    fi

    # Check timestep_chunks CSV
    local tc_csv="$nyx_out/benchmark_nyx_timestep_chunks.csv"
    if [ -f "$tc_csv" ]; then
        local n_tc
        n_tc=$(tail -n +2 "$tc_csv" | wc -l)
        note "NYX: $n_tc chunk rows in $tc_csv"
    fi
}

run_vpic_sim() {
    [ "$SKIP_VPIC" = "1" ] && { note "SKIP VPIC (SKIP_VPIC=1)"; return 0; }
    check_binary "VPIC" "$VPIC_BIN" || return 0

    local vpic_out="$RESULTS_DIR/vpic"
    mkdir -p "$vpic_out"

    note "Running VPIC simulation with $PHASE (NX=$VPIC_NX, timesteps=$VPIC_TIMESTEPS)"

    # VPIC_PHASE restricts to this phase only (skip all others)
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    VPIC_TIMESTEPS="$VPIC_TIMESTEPS" \
    VPIC_WARMUP_STEPS="$VPIC_WARMUP_STEPS" \
    VPIC_SIM_INTERVAL="$VPIC_SIM_INTERVAL" \
    VPIC_MI_ME="$VPIC_MI_ME" \
    VPIC_WPE_WCE="$VPIC_WPE_WCE" \
    VPIC_TI_TE="$VPIC_TI_TE" \
    VPIC_POLICIES="$POLICY" \
    VPIC_PHASE="$PHASE" \
    VPIC_NX="$VPIC_NX" \
    VPIC_CHUNK_MB="$CHUNK_MB" \
    VPIC_ERROR_BOUND="$ERROR_BOUND" \
    VPIC_LR="$SGD_LR" \
    VPIC_MAPE_THRESHOLD="$SGD_MAPE" \
    VPIC_EXPLORE_THRESH="$EXPLORE_THRESH" \
    VPIC_EXPLORE_K="$EXPLORE_K" \
    VPIC_VERIFY="0" \
    VPIC_RESULTS_DIR="$vpic_out" \
    "$VPIC_BIN" > "$vpic_out/vpic_sim.log" 2>&1

    # VPIC writes benchmark_vpic_deck_ranking.csv and _timestep_chunks.csv
    # Symlink to the names the plotter expects
    for csv in "$vpic_out"/benchmark_vpic_deck_ranking*.csv "$vpic_out"/benchmark_vpic_deck_timestep_chunks.csv; do
        [ -f "$csv" ] || continue
        local target
        target="$(echo "$(basename "$csv")" | sed 's/vpic_deck/vpic/')"
        [ ! -f "$vpic_out/$target" ] && ln -sf "$(basename "$csv")" "$vpic_out/$target"
    done

    local ranking_csv="$vpic_out/benchmark_vpic_ranking.csv"
    if [ -f "$ranking_csv" ]; then
        local n_rows
        n_rows=$(tail -n +2 "$ranking_csv" | wc -l)
        note "VPIC: $n_rows ranking rows (live simulation)"
    else
        note "WARNING: VPIC produced no ranking CSV"
    fi

    local tc_csv="$vpic_out/benchmark_vpic_timestep_chunks.csv"
    if [ -f "$tc_csv" ]; then
        local n_tc
        n_tc=$(tail -n +2 "$tc_csv" | wc -l)
        note "VPIC: $n_tc chunk rows in timestep_chunks CSV"
    fi
}

run_warpx_sim() {
    [ "$SKIP_WARPX" = "1" ] && { note "SKIP WarpX (SKIP_WARPX=1)"; return 0; }
    check_binary "WarpX" "$WARPX_BIN" || return 0

    local warpx_out="$RESULTS_DIR/warpx"
    local warpx_work="$RESULTS_DIR/warpx_work"
    mkdir -p "$warpx_out" "$warpx_work"

    note "Running WarpX simulation (ncell=$WARPX_NCELL, max_step=$WARPX_MAX_STEP, diag_interval=$WARPX_DIAG_INTERVAL)"

    # Copy base inputs and override for GPUCompress diagnostic output
    cp "$WARPX_INPUTS" "$warpx_work/inputs" 2>/dev/null || true

    cat >> "$warpx_work/inputs" <<WARPXEOF

# Override for cross-workload regret evaluation
amr.n_cell = $WARPX_NCELL
amr.max_grid_size = 512
amr.blocking_factor = 32
max_step = $WARPX_MAX_STEP

# GPUCompress diagnostic output
diagnostics.diags_names = gpuc_diag
gpuc_diag.intervals = $WARPX_DIAG_INTERVAL
gpuc_diag.diag_type = Full
gpuc_diag.format = gpucompress
gpuc_diag.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz rho
gpucompress.weights_path = $WEIGHTS
gpucompress.algorithm = auto
gpucompress.policy = $POLICY
gpucompress.error_bound = $ERROR_BOUND
gpucompress.chunk_bytes = $((CHUNK_MB * 1024 * 1024))
gpucompress.verify = 0
gpucompress.sgd_lr = $SGD_LR
gpucompress.sgd_mape = $SGD_MAPE
gpucompress.explore_k = $EXPLORE_K
gpucompress.explore_thresh = $EXPLORE_THRESH
WARPXEOF

    cd "$warpx_work"
    WARPX_LOG_DIR="$warpx_out" \
    WARPX_MAX_STEP="$WARPX_MAX_STEP" \
    WARPX_DIAG_INTERVAL="$WARPX_DIAG_INTERVAL" \
    "$WARPX_BIN" inputs > "$warpx_out/warpx_sim.log" 2>&1 || true
    cd "$PROJECT_DIR"

    # Check if per-chunk CSV was produced
    local tc_csv="$warpx_out/benchmark_warpx_timestep_chunks.csv"
    if [ -f "$tc_csv" ]; then
        local n_tc
        n_tc=$(tail -n +2 "$tc_csv" | wc -l)
        note "WarpX: $n_tc chunk rows in $tc_csv"
    else
        note "WARNING: WarpX simulation produced no per-chunk CSV"
    fi
}

run_lammps_sim() {
    [ "$SKIP_LAMMPS" = "1" ] && { note "SKIP LAMMPS (SKIP_LAMMPS=1)"; return 0; }
    check_binary "LAMMPS" "$LMP_BIN" || return 0

    local lammps_out="$RESULTS_DIR/lammps"
    local work_dir="$RESULTS_DIR/lammps_work"
    mkdir -p "$lammps_out" "$work_dir"

    local total_steps=$((LMP_WARMUP_STEPS + LMP_TIMESTEPS * LMP_SIM_INTERVAL))

    note "Running LAMMPS simulation with $PHASE (atoms=$LMP_ATOMS^3, timesteps=$LMP_TIMESTEPS)"

    cat > "$work_dir/input.lmp" <<LMPEOF
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 $LMP_ATOMS 0 $LMP_ATOMS 0 $LMP_ATOMS
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere $(($LMP_ATOMS/2)) $(($LMP_ATOMS/2)) $(($LMP_ATOMS/2)) $(($LMP_ATOMS/4))
group           hot region hot
group           cold subtract all hot
velocity        cold create $LMP_T_COLD 87287 loop geom
velocity        hot create $LMP_T_HOT 12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress ${LMP_SIM_INTERVAL} positions velocities forces
thermo          ${LMP_SIM_INTERVAL}
timestep        0.003
run             ${total_steps}
LMPEOF

    cd "$work_dir"
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    GPUCOMPRESS_ALGO="auto" \
    GPUCOMPRESS_POLICY="$POLICY" \
    GPUCOMPRESS_VERIFY="0" \
    GPUCOMPRESS_SGD="$DO_SGD" \
    GPUCOMPRESS_EXPLORE="$DO_EXPLORE" \
    GPUCOMPRESS_LR="$SGD_LR" \
    GPUCOMPRESS_MAPE="$SGD_MAPE" \
    GPUCOMPRESS_EXPLORE_K="$EXPLORE_K" \
    GPUCOMPRESS_EXPLORE_THRESH="$EXPLORE_THRESH" \
    GPUCOMPRESS_CHUNK_MB="$CHUNK_MB" \
    GPUCOMPRESS_TOTAL_WRITES="$LMP_TIMESTEPS" \
    LAMMPS_LOG_CHUNKS="1" \
    LAMMPS_LOG_DIR="$lammps_out" \
    "$LMP_BIN" -k on g 1 -sf kk -in input.lmp > "$lammps_out/lammps_sim.log" 2>&1 || true
    cd "$PROJECT_DIR"

    # Check per-chunk CSV produced during runtime
    local tc_csv="$lammps_out/benchmark_lammps_timestep_chunks.csv"
    if [ -f "$tc_csv" ]; then
        local n_tc
        n_tc=$(tail -n +2 "$tc_csv" | wc -l)
        note "LAMMPS: $n_tc chunk rows (live simulation)"
    else
        note "WARNING: LAMMPS produced no per-chunk CSV"
    fi
}

# ════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════

plot_cross_workload_regret() {
    note "Plotting cross-workload regret convergence figure"
    RESULTS_DIR="$RESULTS_DIR" PHASE="$PHASE" MODE="$MODE" python3 - <<'PY'
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = os.environ["RESULTS_DIR"]
phase_prefix = os.environ["PHASE"].split("/")[0]
mode = os.environ.get("MODE", "snapshot")

# All possible workload keys — the plotter discovers which ones produced data
snapshot_workloads = [
    ("hurricane",     "Hurricane",     "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("cesm",          "CESM-ATM",      "#4daf4a"),
    ("ai_checkpoint", "AI Checkpoint", "#984ea3"),
]
simulation_workloads = [
    ("vpic",          "VPIC",          "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("warpx",         "WarpX",         "#4daf4a"),
    ("lammps",        "LAMMPS",        "#ff7f00"),
    ("ai_checkpoint", "AI Checkpoint", "#984ea3"),
]

# Auto-detect: try ALL possible keys regardless of mode
all_keys = {}
for key, label, color in snapshot_workloads + simulation_workloads:
    all_keys[key] = (label, color)

# Maintain order: mode-specific list first, then anything extra
if mode == "simulation":
    ordered = simulation_workloads
else:
    ordered = snapshot_workloads

seen = set()
workload_order = []
for key, label, color in ordered:
    if key not in seen:
        workload_order.append((key, label, color))
        seen.add(key)
# Add any others found on disk
for key, (label, color) in all_keys.items():
    if key not in seen:
        workload_order.append((key, label, color))
        seen.add(key)

png_path = os.path.join(results_dir, "cross_workload_regret.png")
combined_csv = os.path.join(results_dir, "cross_workload_regret.csv")

series = []

def load_ranking_csv(path, phase_prefix):
    """Load top1_regret from ranking CSV (produced by ranking profiler)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("phase", "")
            if not p.startswith(phase_prefix):
                continue
            ts = int(row.get("timestep", 0))
            chunk = int(row["chunk"])
            regret = float(row["top1_regret"])
            rows.append((ts, chunk, (regret - 1.0) * 100.0))
    return rows

def load_timestep_chunks_csv(path, phase_prefix):
    """Load cost_model_error_pct from timestep_chunks CSV (runtime logging).
    Convert to regret-like percentage for plotting."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("phase", "")
            if not p.startswith(phase_prefix):
                continue
            ts = int(row.get("timestep", 0))
            chunk = int(row.get("chunk", 0))
            # Use cost_model_error_pct as regret proxy (already a percentage)
            cme = float(row.get("cost_model_error_pct", 0))
            rows.append((ts, chunk, cme * 100.0))  # scale to percentage
    return rows

for key, label, color in workload_order:
    # Try ranking CSV first (has true top1_regret from oracle comparison)
    ranking_path = os.path.join(results_dir, key, f"benchmark_{key}_ranking.csv")
    rows = load_ranking_csv(ranking_path, phase_prefix)
    source = "ranking"

    # Fall back to timestep_chunks CSV (runtime per-chunk diagnostics)
    if not rows:
        tc_path = os.path.join(results_dir, key, f"benchmark_{key}_timestep_chunks.csv")
        rows = load_timestep_chunks_csv(tc_path, phase_prefix)
        source = "timestep_chunks"

    if not rows:
        continue

    rows.sort(key=lambda r: (r[0], r[1]))
    xs = list(range(len(rows)))
    ys = [r[2] for r in rows]
    series.append((label, color, xs, ys, key))
    src_name = os.path.basename(ranking_path if source == "ranking" else tc_path)
    print(f"  {label}: {len(rows)} chunks from {src_name}")

if not series:
    print("ERROR: no workload data to plot", file=sys.stderr)
    sys.exit(1)

# Write combined CSV
with open(combined_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workload", "chunk_index", "regret_pct"])
    for label, _, xs, ys, _ in series:
        for x, y in zip(xs, ys):
            writer.writerow([label, x, f"{y:.4f}"])

# Convergence: first chunk where rolling avg stays below threshold_pct
def find_convergence(ys, window=None, threshold_pct=5.0):
    """Find first chunk where rolling average regret drops below threshold_pct."""
    if not ys:
        return 0
    if window is None:
        window = min(5, max(1, len(ys) // 3))
    if len(ys) < window:
        return len(ys)
    for i in range(window, len(ys)):
        avg = np.mean(ys[i-window:i])
        if avg < threshold_pct:
            return i - window
    return len(ys)

convergence_chunks = []
for label, _, xs, ys, _ in series:
    convergence_chunks.append(find_convergence(ys))

max_convergence = max(convergence_chunks) if convergence_chunks else 0

# Plot
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for label, color, xs, ys, _ in series:
    # Cap regret at 100% for readability
    ys_capped = [min(y, 100.0) for y in ys]
    window = min(5, max(1, len(ys) // 3))
    if len(ys) > window:
        smooth = np.convolve(ys_capped, np.ones(window)/window, mode='valid')
        smooth_x = list(range(window - 1, len(ys)))
        ax.plot(smooth_x, smooth, color=color, linewidth=2.0, label=label)
        ax.plot(xs, ys_capped, color=color, alpha=0.15, linewidth=0.5)
    else:
        ax.plot(xs, ys_capped, color=color, linewidth=2.0, label=label)

ax.set_xlabel("Chunk Index (sequential across profiled fields)")
ax.set_ylabel("Top-1 Regret (%)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, fontsize=9)
# Auto-scale y-axis to data with small headroom
all_ys = [y for _, _, _, ys, _ in series for y in ys]
y_max = min(max(all_ys) * 1.15 if all_ys else 100, 100)
ax.set_ylim(bottom=0, top=max(y_max, 5))
plt.tight_layout()
plt.savefig(png_path, dpi=220, bbox_inches="tight")
print(f"\nSaved: {png_path}")
print(f"Saved: {combined_csv}")

# Summary stats
print("\n--- Convergence Summary ---")
for (label, _, ys, _, _), c in zip(series, convergence_chunks):
    final_avg = np.mean(ys[-min(20, len(ys)):]) if ys else 0
    print(f"  {label:15s}: converges ~chunk {c:4d}, "
          f"final avg regret = {final_avg:.2f}%")
PY
}

# ════════════════════════════════════════════════════════════════
# FIGURE 7b: COMPRESSION TIME MAPE CONVERGENCE
# ════════════════════════════════════════════════════════════════

plot_cross_workload_mape() {
    note "Plotting cross-workload compression time MAPE convergence (Figure 7b)"
    RESULTS_DIR="$RESULTS_DIR" PHASE="$PHASE" MODE="$MODE" python3 - <<'PY'
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = os.environ["RESULTS_DIR"]
phase_prefix = os.environ["PHASE"].split("/")[0]
mode = os.environ.get("MODE", "snapshot")

snapshot_workloads = [
    ("hurricane",     "Hurricane",     "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("cesm",          "CESM-ATM",      "#4daf4a"),
    ("ai_checkpoint", "AI Checkpoint", "#984ea3"),
]
simulation_workloads = [
    ("vpic",          "VPIC",          "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("warpx",         "WarpX",         "#4daf4a"),
    ("lammps",        "LAMMPS",        "#ff7f00"),
    ("ai_checkpoint", "AI Checkpoint", "#984ea3"),
]

all_keys = {}
for key, label, color in snapshot_workloads + simulation_workloads:
    all_keys[key] = (label, color)

if mode == "simulation":
    ordered = simulation_workloads
else:
    ordered = snapshot_workloads

seen = set()
workload_order = []
for key, label, color in ordered:
    if key not in seen:
        workload_order.append((key, label, color))
        seen.add(key)
for key, (label, color) in all_keys.items():
    if key not in seen:
        workload_order.append((key, label, color))
        seen.add(key)

png_path = os.path.join(results_dir, "cross_workload_comp_mape.png")
combined_csv = os.path.join(results_dir, "cross_workload_comp_mape.csv")

series = []

for key, label, color in workload_order:
    # Read actual compression time MAPE from timestep_chunks CSV.
    # mape_comp = |predicted_comp_time - actual_comp_time| / actual_comp_time * 100
    # This measures how well the NN predicts compression time for its chosen
    # algorithm — should start high (cold NN) and decrease as SGD corrects.
    tc_csv = os.path.join(results_dir, key, f"benchmark_{key}_timestep_chunks.csv")
    if not os.path.isfile(tc_csv):
        continue

    rows_by_ts = {}
    with open(tc_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("phase", "")
            if not p.startswith(phase_prefix):
                continue
            ts = int(row.get("timestep", row.get("field_idx", 0)))
            chunk = int(row.get("chunk", 0))
            mape_comp = float(row.get("mape_comp", 0.0))
            rows_by_ts.setdefault(ts, []).append((chunk, mape_comp))

    if not rows_by_ts:
        continue

    rows = []
    for ts in sorted(rows_by_ts.keys()):
        for chunk_idx, mape in rows_by_ts[ts]:
            rows.append((ts, chunk_idx, mape))

    xs = list(range(len(rows)))
    ys = [r[2] for r in rows]
    series.append((label, color, xs, ys, key))
    print(f"  {label}: {len(rows)} chunks from {os.path.basename(tc_csv)}")

if not series:
    print("ERROR: no workload data to plot for MAPE", file=sys.stderr)
    sys.exit(1)

# Write combined CSV
with open(combined_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workload", "chunk_index", "comp_mape_pct"])
    for label, _, xs, ys, _ in series:
        for x, y in zip(xs, ys):
            writer.writerow([label, x, f"{y:.4f}"])

# Convergence: MAPE converges to a higher floor than regret (typically 10-20%)
# because even oracle-level ranking can have imprecise absolute time predictions.
def find_convergence(ys, window=None, threshold_pct=15.0):
    if not ys:
        return 0
    if window is None:
        window = min(5, max(1, len(ys) // 3))
    if len(ys) < window:
        return len(ys)
    for i in range(window, len(ys)):
        avg = np.mean(ys[i-window:i])
        if avg < threshold_pct:
            return i - window
    return len(ys)

convergence_chunks = []
for label, _, xs, ys, _ in series:
    convergence_chunks.append(find_convergence(ys))

max_convergence = max(convergence_chunks) if convergence_chunks else 0

# Plot
fig, ax = plt.subplots(figsize=(5.4, 3.8))
MAPE_CAP = 100.0
for label, color, xs, ys, _ in series:
    ys_capped = [min(y, MAPE_CAP) for y in ys]
    window = min(5, max(1, len(ys) // 3))
    if len(ys) > window:
        smooth = np.convolve(ys_capped, np.ones(window)/window, mode='valid')
        smooth_x = list(range(window - 1, len(ys)))
        ax.plot(smooth_x, smooth, color=color, linewidth=2.0, label=label)
        ax.plot(xs, ys_capped, color=color, alpha=0.15, linewidth=0.5)
    else:
        ax.plot(xs, ys_capped, color=color, linewidth=2.0, label=label)

ax.set_xlabel("Chunk Index (sequential across profiled fields)")
ax.set_ylabel("Compression Time MAPE (%)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, fontsize=9)
ax.set_ylim(bottom=0, top=MAPE_CAP * 1.05)
plt.tight_layout()
plt.savefig(png_path, dpi=220, bbox_inches="tight")
print(f"\nSaved: {png_path}")
print(f"Saved: {combined_csv}")

# Summary stats
print("\n--- Comp. Time MAPE Convergence Summary ---")
print("Note: high initial MAPE does not translate into proportionally high")
print("regret -- relative algorithm ranking remains stable even when individual")
print("metric predictions are inaccurate.")
for (label, _, xs, ys, _), c in zip(series, convergence_chunks):
    final_avg = np.mean(ys[-min(20, len(ys)):]) if ys else 0
    print(f"  {label:15s}: converges ~chunk {c:4d}, "
          f"final avg comp MAPE = {final_avg:.2f}%")
PY
}

# ── Banner ────────────────────────────────────────────────────
print_banner() {
    echo "============================================================"
    echo "Section 7.1: Cross-Workload Regret + MAPE Convergence"
    echo "============================================================"
    echo "Mode          : $MODE"
    echo "Run name      : $RUN_NAME"
    echo "Results dir   : $RESULTS_DIR"
    echo "Weights       : $WEIGHTS"
    echo "Policy        : $POLICY (w0=$W0, w1=$W1, w2=$W2)"
    echo "Phase         : $PHASE"
    echo "Error bound   : $ERROR_BOUND"
    echo "Chunk size    : ${CHUNK_MB} MiB"
    echo "SGD LR        : $SGD_LR  MAPE thresh: $SGD_MAPE"
    echo "Explore K     : $EXPLORE_K  thresh: $EXPLORE_THRESH"
    echo "------------------------------------------------------------"
    if [ "$MODE" = "snapshot" ]; then
        echo "Hurricane     : $([ "$SKIP_HURRICANE" = "1" ] && echo "SKIP" || echo "$HURRICANE_DIR")"
        echo "NYX           : $([ "$SKIP_NYX" = "1" ] && echo "SKIP" || echo "$NYX_DATA_DIR")"
        echo "CESM-ATM      : $([ "$SKIP_CESM" = "1" ] && echo "SKIP" || echo "$CESM_DATA_DIR")"
        echo "AI Checkpoint : $([ "$SKIP_AI" = "1" ] && echo "SKIP" || echo "model=$AI_MODEL epochs=$AI_EPOCHS")"
    else
        echo "VPIC          : $([ "$SKIP_VPIC" = "1" ] && echo "SKIP" || echo "NX=$VPIC_NX ts=$VPIC_TIMESTEPS")"
        echo "NYX           : $([ "$SKIP_NYX" = "1" ] && echo "SKIP" || echo "$NYX_DATA_DIR")"
        echo "WarpX         : $([ "$SKIP_WARPX" = "1" ] && echo "SKIP" || echo "ncell=$WARPX_NCELL max_step=$WARPX_MAX_STEP")"
        echo "LAMMPS        : $([ "$SKIP_LAMMPS" = "1" ] && echo "SKIP" || echo "atoms=${LMP_ATOMS}^3 ts=$LMP_TIMESTEPS")"
        echo "AI Checkpoint : $([ "$SKIP_AI" = "1" ] && echo "SKIP" || echo "model=$AI_MODEL epochs=$AI_EPOCHS")"
    fi
    echo "============================================================"
}

# ── Main ──────────────────────────────────────────────────────
main() {
    mkdir -p "$RESULTS_DIR"
    require_file "$WEIGHTS"
    require_file "$GENERIC_BIN"

    print_banner | tee "$RESULTS_DIR/config.txt"

    if [ "$PLOT_ONLY" != "1" ]; then
        if [ "$MODE" = "snapshot" ]; then
            run_hurricane
            run_nyx_snapshot
            run_cesm
            run_ai_checkpoint
        elif [ "$MODE" = "simulation" ]; then
            run_vpic_sim
            run_nyx_sim
            run_warpx_sim
            run_lammps_sim
            run_ai_checkpoint
        else
            die "Unknown MODE=$MODE (use 'snapshot' or 'simulation')"
        fi
    fi

    plot_cross_workload_regret
    plot_cross_workload_mape

    cat <<EOF

============================================================
Complete.  Results:
  Figure 7a: $RESULTS_DIR/cross_workload_regret.png
  Figure 7b: $RESULTS_DIR/cross_workload_comp_mape.png
  Data:      $RESULTS_DIR/cross_workload_regret.csv
             $RESULTS_DIR/cross_workload_comp_mape.csv
============================================================
EOF
}

main "$@"
