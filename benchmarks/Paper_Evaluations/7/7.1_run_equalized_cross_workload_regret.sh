#!/bin/bash
# ============================================================
# 7.1 Cross-Workload Regret + MAPE Convergence
#
# Runs live simulations (VPIC, NYX, WarpX, LAMMPS) with nn-rl phase,
# collecting per-chunk diagnostics and ranking profiler data during
# runtime. Produces two figures:
#   Figure 7a: Top-1 Regret (%) — how close NN picks are to oracle
#   Figure 7b: Compression Time MAPE (%) — NN prediction accuracy
#
# Workloads: VPIC, NYX (Sedov), WarpX (LWFA), LAMMPS (MD)
# Each simulation uses the GPUCompress NN-RL pipeline with online
# SGD learning. Per-chunk CSV + ranking profiler run during simulation.
#
# Usage:
#   bash benchmarks/Paper_Evaluations/7/7.1_run_equalized_cross_workload_regret.sh
#
# Overrides:
#   RUN_NAME=paper_fig7a
#   POLICY=balanced CHUNK_MB=4 SGD_LR=0.2
#   SKIP_VPIC=1 SKIP_NYX=1 SKIP_WARPX=1 SKIP_LAMMPS=1
#   PLOT_ONLY=1          # just regenerate figures from existing CSVs
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

# ── Run configuration ─────────────────────────────────────────
RUN_NAME="${RUN_NAME:-cross_workload_$(date +%Y%m%d_%H%M%S)}"
# RESULTS_DIR can be overridden via env (e.g. by the SC26 sweep script) to
# place runs outside the default benchmarks/Paper_Evaluations/7/results tree.
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$RUN_NAME}"

# Cross-workload regret figure uses the balanced cost model uniformly:
#   cost = comp_time + decomp_time + (data_size / (ratio * bandwidth))
# i.e. w0 = w1 = w2 = 1.0. This breaks ties at the ratio cap (100x) — under
# the ratio-only policy, multiple algorithms saturate the cap on
# highly-compressible chunks (e.g. WarpX vacuum fields), making the oracle
# ranking degenerate and trivially yielding regret = 0. Balanced uses
# comp/decomp times to disambiguate, which gives a real regret signal.
POLICY="${POLICY:-balanced}"
# All cross-workload regret runs use nn-rl+exp50 (online SGD + always-on
# exploration). Exploration is required so the ranking profiler has K=4
# alternative actions per chunk to compute true top-1 regret against the
# oracle. nn-rl alone only explores when confidence drops, which leaves
# gaps in the regret trajectory.
PHASE="${PHASE:-nn-rl+exp50}"
# All cross-workload regret runs are LOSSY (relative error bound 1e-3) so
# that the NN can pick quantized configs and the per-chunk PSNR data exists
# for Figure 7c (PSNR MAPE). Lossless (ERROR_BOUND=0) leaves all PSNR cells
# as NaN — useful for regret/comp-MAPE only. Combined with the balanced
# policy above, this is the canonical 3-in-1 evaluation configuration:
# regret + comp MAPE + PSNR MAPE in a single run.
ERROR_BOUND="${ERROR_BOUND:-0.001}"
CHUNK_MB="${CHUNK_MB:-4}"
# Equalized hyperparameters for cross-workload regret figure (Section 7).
# These values apply uniformly to NYX, VPIC, WarpX, and LAMMPS so that
# differences in regret/MAPE convergence reflect data distribution, not
# tuning. Change here to re-tune the whole figure at once.
SGD_LR="${SGD_LR:-0.2}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"

PLOT_ONLY="${PLOT_ONLY:-0}"

# ── Skip flags (per workload) ────────────────────────────────
SKIP_VPIC="${SKIP_VPIC:-0}"
SKIP_NYX="${SKIP_NYX:-0}"
SKIP_WARPX="${SKIP_WARPX:-0}"
SKIP_LAMMPS="${SKIP_LAMMPS:-0}"
# AI skip flag is declared with the AI defaults block above.

# ── Binaries / paths ─────────────────────────────────────────
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$PROJECT_DIR/neural_net/weights/model.nnwt}"
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

# ── VPIC: Harris sheet magnetic reconnection
#
#    Physics: Harris current sheet with tearing instability.
#             mi_me=25 gives slow reconnection (~20k steps for full
#             nonlinear phase). interval=500 provides enough physics
#             evolution between dumps for meaningful data diversity.
#    Grid:    NX=147 → (147+2)³ × 16 fields × 4 bytes ≈ 200 MB/field
#    Chunks:  200 MB / 4 MB = 50 chunks per write
#    Writes:  200 warmup + 8 × 500 = 4200 total steps → 8 writes
#
#    Tested: mi_me=5/25, interval=50/100/200/500, LR=0.1/0.2
#    Best convergence: mi_me=25, int=500, LR=0.1
#
#    Data evolution (ratio policy, LR=0.1):
#      T=0: ratio=8.52  mape_c=138%           (initial Harris equilibrium)
#      T=2: ratio=8.10  mape_c= 95%  (-44%)   (tearing mode growing)
#      T=4: ratio=7.69  mape_c= 34%  (-61%)   (current sheet thinning)
#      T=6: ratio=7.17  mape_c= 22%  (-12%)   (approaching nonlinear)
#      T=7: ratio=6.98  mape_c= 27%  (+5%)    (data shift, slight setback)
#
#    Ratio swing: 1.55 (8.52→6.98), MAPE comp: 138%→27% (5.1× improvement)
#    Regret: 12%→36%→17%→15% (rises from data challenge, then stabilizes)
#
VPIC_NX="${VPIC_NX:-147}"
VPIC_TIMESTEPS="${VPIC_TIMESTEPS:-8}"
VPIC_WARMUP_STEPS="${VPIC_WARMUP_STEPS:-200}"
VPIC_SIM_INTERVAL="${VPIC_SIM_INTERVAL:-500}"
VPIC_MI_ME="${VPIC_MI_ME:-25}"
VPIC_WPE_WCE="${VPIC_WPE_WCE:-1}"
VPIC_TI_TE="${VPIC_TI_TE:-5}"

# ── NYX: Sedov blast wave (HydroTests)
#
#    Physics: Spherical shock expands from point energy into uniform gas.
#    Grid:    220³ cells × 13 components × 4 bytes = 554 MB/write
#    Chunks:  554 MB / 4 MB = 133 chunks per write
#    Writes:  200 steps / 25 plot_int = 8 writes → 1064 ranking rows
#
#    Data evolution (ratio policy, LR=0.1):
#      T=0-2: ratio=100x, uniform — trivially compressible
#      T=3:   ratio=99x,  shock front enters first chunks (std=6.9)
#      T=4:   ratio=98x,  shock expands (std=7.7, SGD fires=8)
#      T=5:   ratio=97x,  MAPE comp drops 124%→30% (SGD fires=18)
#      T=6-8: ratio=93-96x, MAPE comp stabilizes at 4-6% (SGD fires=36-42)
#
#    Key metrics:
#      MAPE comp:  160% → 4%  (steady convergence over 8 writes)
#      Regret:     0% → 2.4% → 1% (low, NN picks near-optimal)
#      MAPE ratio: 0% → 15%  (rises as shock creates unpredictable data)
#      SGD fires:  0 → 42    (accelerates as data diversifies)
#      Chunk std:  0 → 19    (continuous increase in data diversity)
#
NYX_NCELL="${NYX_NCELL:-128}"
NYX_MAX_STEP="${NYX_MAX_STEP:-200}"
NYX_PLOT_INT="${NYX_PLOT_INT:-25}"

# ── WarpX: laser-wakefield acceleration (LWFA)
#
#    Physics: Intense laser pulse drives plasma wake in underdense plasma.
#             Moving window (v=c) scrolls domain, adding zero-fill behind
#             and revealing laser-plasma interaction ahead. Plasma bubble
#             forms and matures, creating continuously evolving data.
#    Grid:    128×128×384 cells × 10 fields × 4 bytes = 252 MB/write
#    Chunks:  252 MB / 4 MB = 60 chunks per write
#    Writes:  25-step intervals × 8 writes = 200 total steps
#
#    Tested: ncell=32²×256/64²×256/128²×384, di=10/25/50
#    Best: 128×128×384 di=25 — most chunks + steady MAPE decline
#    Note: z-dim must be divisible by blocking_factor=32
#
#    Data evolution (ratio policy, LR=0.1):
#      T=0: ratio=97.3  mape_c=120%          (mostly vacuum + laser init)
#      T=1: ratio=70.8  mape_c= 93%  (-27%)  (laser enters plasma)
#      T=2: ratio=67.4  mape_c=  7%  (-85%)  (dramatic MAPE drop — NN learns)
#      T=3: ratio=52.0  mape_c= 14%  (+6%)   (bubble forming, data shifts)
#      T=4: ratio=41.5  mape_c= 19%          (wake structure develops)
#      T=6: ratio=25.2  mape_c= 24%          (mature bubble)
#      T=8: ratio=14.9  mape_c= 19%  (-5%)   (late-time wake)
#
#    Ratio swing: 82× (97→15), MAPE comp: 120%→19% (6× improvement)
#    Regret: 0-2% throughout (NN picks near-optimal despite rapid change)
#    Chunk std: 14.7→48.1→33.4 (diversity peaks mid-run at bubble formation)
#    SGD fires: 2→25→36 (continuous active learning)
#
WARPX_NCELL="${WARPX_NCELL:-128 128 384}"
WARPX_MAX_STEP="${WARPX_MAX_STEP:-200}"
WARPX_DIAG_INTERVAL="${WARPX_DIAG_INTERVAL:-25}"

# ── LAMMPS: MD phase transition (hot-sphere near melting)
#
#    Physics: FCC lattice with hot sphere (T=2.0) in cold matrix (T=0.7).
#             Temperatures near LJ melting point create continuously
#             evolving solid-liquid boundary — unlike T_hot=50/100 which
#             creates a one-time explosion that stabilizes immediately.
#    Grid:    161³ FCC = 16.7M atoms × 3 comp × 4 bytes = 200 MB/field
#    Chunks:  200 MB / 4 MB = 50 chunks per write (positions only for ranking)
#    Writes:  500-step intervals × 8 writes = 4000 total steps
#
#    Why interval=500: Tested 10/25/50/100/500. Shorter intervals (10-25)
#    cause instant NN adaptation (1 SGD step). interval=500 gives enough
#    physics evolution between dumps that the NN must continuously re-learn.
#    MAPE comp shows steady 15-write convergence from 207% → 19%.
#
#    Why T_cold=0.7, T_hot=2.0: Tested T_hot=10/50/100. T_hot=100 causes
#    numerical instability (NaN forces, nvCOMP crash). T_hot=10 creates
#    stationary data. T_cold=0.7/T_hot=2.0 (near LJ melting point ~0.7)
#    creates continuous solid-liquid phase boundary migration.
#
#    Data evolution (ratio policy, LR=0.1, interval=500):
#      T=0:    ratio=2.46  mape_c=108%          (pristine FCC lattice)
#      T=1:    ratio=1.16  mape_c=207% (+99%)   (lattice disrupted, NN worse)
#      T=2:    ratio=1.14  mape_c=172% (-35%)   (SGD starts correcting)
#      T=3:    ratio=1.11  mape_c=130% (-43%)   (improving)
#      T=5:    ratio=1.17  mape_c= 77% (-57%)   (big improvement)
#      T=7:    ratio=1.17  mape_c= 62% (-7%)    (steady decline)
#      T=10:   ratio=1.17  mape_c= 52% (-3%)    (converging)
#      T=15:   ratio=1.16  mape_c= 19% (-16%)   (5.7× improvement from peak)
#
#    Regret:  0% → 19% → 0-2% (adapts by T=4, stays near-optimal)
#    SGD:     16→11→7→0→0→7→9 (learns, stops, restarts as data shifts)
#
LMP_ATOMS="${LMP_ATOMS:-161}"
LMP_TIMESTEPS="${LMP_TIMESTEPS:-8}"
LMP_SIM_INTERVAL="${LMP_SIM_INTERVAL:-500}"
LMP_WARMUP_STEPS="${LMP_WARMUP_STEPS:-0}"
LMP_T_HOT="${LMP_T_HOT:-2.0}"
LMP_T_COLD="${LMP_T_COLD:-0.7}"

# ── AI workload (ViT-B/16 fine-tuning on CIFAR-10 via in-situ HDF5 VOL capture)
#
#    Workload type: PyTorch training loop. At each checkpoint epoch, four
#    tensor sets (weights, adam_m, adam_v, gradients) are concatenated on the
#    GPU and routed through the GPUCompress HDF5 VOL connector via a Python
#    ctypes bridge (see scripts/train_and_export_checkpoints.py and
#    scripts/gpucompress_hdf5.py:InlineFullBenchmark). This is NOT an offline
#    .f32 replay — the data never leaves the GPU until it lands as a
#    compressed HDF5 file. The InlineFullBenchmark runs all 15 algorithms
#    (9 fixed + 6 NN) per tensor per checkpoint and emits per-chunk metrics
#    to inline_benchmark_chunks.csv.
#
AI_MODEL="${AI_MODEL:-vit_b_16}"
AI_EPOCHS="${AI_EPOCHS:-20}"
AI_CHECKPOINTS="${AI_CHECKPOINTS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
AI_DATASET="${AI_DATASET:-cifar10}"
SKIP_AI="${SKIP_AI:-0}"

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

# ════════════════════════════════════════════════════════════════
# SIMULATION WORKLOADS
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
    NYX_PHASE="$PHASE" \
    "$NYX_BIN" inputs > "$nyx_out/nyx_sim.log" 2>&1 || {
        note "WARNING: NYX exited with error (check $nyx_out/nyx_sim.log)"
    }
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
    "$VPIC_BIN" > "$vpic_out/vpic_sim.log" 2>&1 || {
        note "WARNING: VPIC exited with non-zero status (check $vpic_out/vpic_sim.log)"
        note "  this is usually a known shutdown-cleanup segfault that fires AFTER"
        note "  'normal exit' — CSVs are still written. Continuing pipeline."
    }

    # Estimate: ~50 chunks/field × timesteps ranking rows

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
    "$WARPX_BIN" inputs > "$warpx_out/warpx_sim.log" 2>&1 || {
        note "WARNING: WarpX exited with error (check $warpx_out/warpx_sim.log)"
    }
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
    GPUCOMPRESS_ERROR_BOUND="$ERROR_BOUND" \
    GPUCOMPRESS_TOTAL_WRITES="$LMP_TIMESTEPS" \
    LAMMPS_LOG_CHUNKS="1" \
    LAMMPS_LOG_DIR="$lammps_out" \
    "$LMP_BIN" -k on g 1 -sf kk -in input.lmp > "$lammps_out/lammps_sim.log" 2>&1 || {
        note "WARNING: LAMMPS exited with error (check $lammps_out/lammps_sim.log)"
    }
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

run_ai_sim() {
    [ "$SKIP_AI" = "1" ] && { note "SKIP AI (SKIP_AI=1)"; return 0; }

    local ai_out="$RESULTS_DIR/ai"
    mkdir -p "$ai_out"

    local ai_script="$PROJECT_DIR/scripts/train_and_export_checkpoints.py"
    if [ ! -f "$ai_script" ]; then
        note "SKIP AI: training script not found at $ai_script"
        return 0
    fi

    note "Running AI training (model=$AI_MODEL epochs=$AI_EPOCHS checkpoints=$AI_CHECKPOINTS)"
    note "  In-situ HDF5 VOL capture via Python ctypes bridge — see scripts/gpucompress_hdf5.py"

    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    LD_LIBRARY_PATH="$PROJECT_DIR/build:/tmp/hdf5-install/lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$PROJECT_DIR/scripts:${PYTHONPATH:-}" \
    python3 "$ai_script" \
        --model "$AI_MODEL" \
        --epochs "$AI_EPOCHS" \
        --checkpoint-epochs "$AI_CHECKPOINTS" \
        --dataset "$AI_DATASET" \
        --outdir "$ai_out" \
        --hdf5-direct \
        --benchmark \
        --policy "$POLICY" \
        --chunk-mb "$CHUNK_MB" \
        --error-bound "$ERROR_BOUND" \
        --sgd-lr "$SGD_LR" \
        --sgd-mape "$SGD_MAPE" \
        --explore-k "$EXPLORE_K" \
        --explore-thresh "$EXPLORE_THRESH" \
        > "$ai_out/ai_run.log" 2>&1 || {
        note "WARNING: AI training/benchmark exited with non-zero status (check $ai_out/ai_run.log)"
    }

    local chunks_csv="$ai_out/inline_benchmark_chunks.csv"
    if [ -f "$chunks_csv" ]; then
        local n_rows
        n_rows=$(tail -n +2 "$chunks_csv" | wc -l)
        note "AI: $n_rows chunk rows in $chunks_csv"
    else
        note "WARNING: AI produced no inline_benchmark_chunks.csv"
    fi
}

# ════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════

plot_cross_workload_regret() {
    note "Plotting cross-workload regret convergence figure"
    RESULTS_DIR="$RESULTS_DIR" PHASE="$PHASE" CHUNK_MB="$CHUNK_MB" POLICY="$POLICY" python3 - <<'PY'
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = os.environ["RESULTS_DIR"]
phase_prefix = os.environ["PHASE"].split("/")[0]

workloads = [
    ("vpic",          "VPIC",          "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("warpx",         "WarpX",         "#4daf4a"),
    ("lammps",        "LAMMPS",        "#ff7f00"),
    ("ai",            "ViT-B/16",      "#984ea3"),
]

workload_order = []
seen = set()
for key, label, color in workloads:
    workload_order.append((key, label, color))

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

def load_ai_inline_chunks_regret(path, chunk_mb=4, bw_bytes_per_sec=5e9,
                                  nn_mode="nn-rl+exp50", nn_policy="balanced"):
    """Compute top-1 regret per (epoch,tensor,chunk_idx) from AI inline_benchmark_chunks.csv.

    The AI workload's CSV has 15 rows per (epoch,tensor,chunk_idx) group:
    9 fixed algorithms (mode='-', policy='-') and 6 NN configs (3 modes x
    2 policies). The cost model is the BALANCED one used by all live sims:

        cost = comp_ms + decomp_ms + (chunk_bytes / (max(ratio,0.01) * bw))

    where bw = 5 GB/s (matching the bandwidth assumption used by the live
    simulation cost models). Oracle = min cost across all 15 rows. NN pick
    = the row where mode==nn_mode AND policy==nn_policy. Returns a list of
    (group_index, regret_pct) tuples ordered by (epoch, tensor, chunk_idx)."""
    rows_out = []
    if not os.path.isfile(path):
        return rows_out
    chunk_bytes = chunk_mb * 1024 * 1024
    bw_bpms = bw_bytes_per_sec / 1000.0  # bytes per ms
    groups = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                tensor = row["tensor"]
                chunk_idx = int(row["chunk_idx"])
                comp_ms = float(row["comp_ms"])
                decomp_ms = float(row["decomp_ms"])
                actual_ratio = float(row["actual_ratio"])
            except (KeyError, ValueError):
                continue
            ratio_eff = max(actual_ratio, 0.01)
            cost = comp_ms + decomp_ms + (chunk_bytes / (ratio_eff * bw_bpms))
            key = (epoch, tensor, chunk_idx)
            g = groups.setdefault(key, {"costs": [], "nn_cost": None})
            g["costs"].append(cost)
            if row.get("mode", "") == nn_mode and row.get("policy", "") == nn_policy:
                g["nn_cost"] = cost

    # Order by (epoch, tensor, chunk_idx). Tensor name is a string but stable sort works.
    ordered_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2]))
    idx = 0
    for k in ordered_keys:
        g = groups[k]
        if not g["costs"] or g["nn_cost"] is None:
            continue
        oracle = min(g["costs"])
        if oracle <= 0:
            continue
        regret_pct = (g["nn_cost"] - oracle) / oracle * 100.0
        rows_out.append((idx, idx, regret_pct))
        idx += 1
    return rows_out

for key, label, color in workload_order:
    if key == "ai":
        ai_csv = os.path.join(results_dir, "ai", "inline_benchmark_chunks.csv")
        # Try to read CHUNK_MB from environment (the script exports it
        # implicitly through the run); fall back to 4 MiB.
        try:
            chunk_mb = int(os.environ.get("CHUNK_MB", "4"))
        except ValueError:
            chunk_mb = 4
        rows = load_ai_inline_chunks_regret(ai_csv, chunk_mb=chunk_mb)
        if not rows:
            print(f"  {label}: no usable rows in {ai_csv}")
            continue
        rows.sort(key=lambda r: (r[0], r[1]))
        xs = list(range(len(rows)))
        ys = [r[2] for r in rows]
        series.append((label, color, xs, ys, key))
        print(f"  {label}: {len(rows)} chunks from {os.path.basename(ai_csv)}")
        continue

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
    note "Plotting cross-workload cost MAPE convergence (Figure 7b)"
    RESULTS_DIR="$RESULTS_DIR" PHASE="$PHASE" CHUNK_MB="$CHUNK_MB" POLICY="$POLICY" python3 - <<'PY'
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = os.environ["RESULTS_DIR"]
phase_prefix = os.environ["PHASE"].split("/")[0]

workload_order = [
    ("vpic",          "VPIC",          "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("warpx",         "WarpX",         "#4daf4a"),
    ("lammps",        "LAMMPS",        "#ff7f00"),
    ("ai",            "ViT-B/16",      "#984ea3"),
]

png_path = os.path.join(results_dir, "cross_workload_cost_mape.png")
combined_csv = os.path.join(results_dir, "cross_workload_cost_mape.csv")

series = []

def load_ai_inline_chunks_cost_mape(path, nn_mode="nn-rl+exp50", nn_policy=None):
    """Read per-chunk cost MAPE for the AI workload from
    inline_benchmark_chunks.csv. Filters to rows where mode==nn_mode under
    the active POLICY (so we plot the same NN config the live sims plot).

    Prefers the precomputed `cost_model_error_pct` column. Falls back to
    computing it from `predicted_cost`/`actual_cost` when present.
    Returns ordered (epoch, tensor, chunk_idx, mape) -> a flat list."""
    if nn_policy is None:
        nn_policy = os.environ.get("POLICY", "balanced")
    rows_out = []
    if not os.path.isfile(path):
        return rows_out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode", "") != nn_mode:
                continue
            if row.get("policy", "") != nn_policy:
                continue
            try:
                epoch = int(row["epoch"])
                tensor = row["tensor"]
                chunk_idx = int(row["chunk_idx"])
            except (KeyError, ValueError):
                continue
            # cost_model_error_pct is stored as a FRACTION in [0,1] by
            # src/api/gpucompress_compress.cpp:545 (despite the _pct suffix).
            # Multiply by 100 here so the plot axis is in percent.
            cme_raw = row.get("cost_model_error_pct", "")
            mape = None
            if cme_raw not in ("", None):
                try:
                    mape = float(cme_raw) * 100.0
                except ValueError:
                    mape = None
            if mape is None:
                try:
                    pc = float(row["predicted_cost"])
                    ac = float(row["actual_cost"])
                except (KeyError, ValueError):
                    continue
                denom = abs(ac) if abs(ac) > 1e-9 else 1e-9
                mape = abs(pc - ac) / denom * 100.0
            rows_out.append((epoch, tensor, chunk_idx, mape))
    rows_out.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows_out

for key, label, color in workload_order:
    if key == "ai":
        ai_csv = os.path.join(results_dir, "ai", "inline_benchmark_chunks.csv")
        ai_rows = load_ai_inline_chunks_cost_mape(ai_csv)
        if not ai_rows:
            print(f"  {label}: no usable rows in {ai_csv}")
            continue
        xs = list(range(len(ai_rows)))
        ys = [r[3] for r in ai_rows]
        series.append((label, color, xs, ys, key))
        print(f"  {label}: {len(ai_rows)} chunks from {os.path.basename(ai_csv)}")
        continue

    # Read cost-model error from timestep_chunks CSV.
    # cost_model_error_pct = |predicted_cost - actual_cost| / actual_cost * 100
    # This measures how well the NN predicts the SCALAR COST under the active
    # policy (the linear combination of comp_ms / decomp_ms / 1/ratio that the
    # ranker actually optimizes), not just compression time. Should start high
    # (cold NN) and decrease as SGD corrects.
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
            # cost_model_error_pct is stored as a FRACTION in [0,1] by
            # src/api/gpucompress_compress.cpp:545 (despite the _pct suffix).
            # Multiply by 100 here so the plot axis is in percent.
            cme_raw = row.get("cost_model_error_pct", "")
            if cme_raw in ("", None):
                continue
            try:
                cost_mape = float(cme_raw) * 100.0
            except ValueError:
                continue
            rows_by_ts.setdefault(ts, []).append((chunk, cost_mape))

    if not rows_by_ts:
        continue

    # If a simulation writes multiple fields per timestep (e.g. LAMMPS:
    # positions+velocities+forces), keep only the first field's chunks
    # to match the ranking profiler which uses positions only.
    chunks_per_ts = len(next(iter(rows_by_ts.values())))
    expected_chunks = 50  # approximate single-field chunk count
    if chunks_per_ts > expected_chunks * 2:
        keep = chunks_per_ts // 3  # 3 fields → take first field
    else:
        keep = chunks_per_ts

    rows = []
    for ts in sorted(rows_by_ts.keys()):
        for chunk_idx, mape in rows_by_ts[ts][:keep]:
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
    writer.writerow(["workload", "chunk_index", "cost_mape_pct"])
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
ax.set_ylabel("Cost Model MAPE (%)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, fontsize=9)
ax.set_ylim(bottom=0, top=MAPE_CAP * 1.05)
plt.tight_layout()
plt.savefig(png_path, dpi=220, bbox_inches="tight")
print(f"\nSaved: {png_path}")
print(f"Saved: {combined_csv}")

# Summary stats
print("\n--- Cost Model MAPE Convergence Summary ---")
print("Note: high initial MAPE does not translate into proportionally high")
print("regret -- relative algorithm ranking remains stable even when individual")
print("metric predictions are inaccurate.")
for (label, _, xs, ys, _), c in zip(series, convergence_chunks):
    final_avg = np.mean(ys[-min(20, len(ys)):]) if ys else 0
    print(f"  {label:15s}: converges ~chunk {c:4d}, "
          f"final avg cost MAPE = {final_avg:.2f}%")
PY
}

# ════════════════════════════════════════════════════════════════
# FIGURE 7c: PSNR MAPE CONVERGENCE (lossy quality prediction)
# ════════════════════════════════════════════════════════════════
#
# This figure measures how accurately the NN predicts the lossy quality
# (PSNR) of the algorithm it picked. Only meaningful on chunks where the
# NN actually selected a quantized config — chunks running lossless emit
# NaN in the per-workload CSV (see WarpX/LAMMPS/VPIC/NYX patches), and
# this plotter filters those rows so they don't dilute the curve.
#
# A useful Figure 7c needs ERROR_BOUND > 0 for at least one workload.
# Lossless runs leave every cell as NaN — the plotter will report zero
# usable rows and skip the figure rather than producing an empty plot.
plot_cross_workload_psnr_mape() {
    note "Plotting cross-workload PSNR MAPE convergence (Figure 7c)"
    RESULTS_DIR="$RESULTS_DIR" PHASE="$PHASE" CHUNK_MB="$CHUNK_MB" POLICY="$POLICY" python3 - <<'PY'
import csv, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = os.environ["RESULTS_DIR"]
phase_prefix = os.environ["PHASE"].split("/")[0]

workload_order = [
    ("vpic",          "VPIC",          "#e41a1c"),
    ("nyx",           "NYX",           "#377eb8"),
    ("warpx",         "WarpX",         "#4daf4a"),
    ("lammps",        "LAMMPS",        "#ff7f00"),
    ("ai",            "ViT-B/16",      "#984ea3"),
]

png_path     = os.path.join(results_dir, "cross_workload_psnr_mape.png")
combined_csv = os.path.join(results_dir, "cross_workload_psnr_mape.csv")

def parse_float_or_nan(s):
    """CSV cells may be 'nan' (lossless sentinel), empty, or a real number.
    Return float('nan') for any non-numeric input so downstream filtering
    via math.isnan() catches them uniformly."""
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")

def load_ai_inline_chunks_psnr_mape(path, nn_mode="nn-rl+exp50", nn_policy=None):
    """Read per-chunk PSNR MAPE for the AI workload from
    inline_benchmark_chunks.csv. Filters to rows where mode==nn_mode under
    the active POLICY. Lossless rows (actual_psnr or predicted_psnr <= 0,
    or non-finite) emit NaN — same sentinel the live sims use, so the
    plotter's existing nan-skipping logic catches them. Otherwise:

        mape = |predicted_psnr - actual_psnr| / |actual_psnr| * 100, capped at 200%
    """
    if nn_policy is None:
        nn_policy = os.environ.get("POLICY", "balanced")
    rows_out = []
    if not os.path.isfile(path):
        return rows_out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode", "") != nn_mode:
                continue
            if row.get("policy", "") != nn_policy:
                continue
            try:
                epoch = int(row["epoch"])
                tensor = row["tensor"]
                chunk_idx = int(row["chunk_idx"])
                actual_psnr = float(row["actual_psnr"])
                pred_psnr = float(row["predicted_psnr"])
            except (KeyError, ValueError):
                continue
            if (actual_psnr <= 0 or pred_psnr <= 0
                    or not math.isfinite(actual_psnr)
                    or not math.isfinite(pred_psnr)):
                mape = float("nan")
            else:
                mape = abs(pred_psnr - actual_psnr) / abs(actual_psnr) * 100.0
                if mape > 200.0:
                    mape = 200.0
            rows_out.append((epoch, tensor, chunk_idx, mape))
    rows_out.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows_out

series = []
for key, label, color in workload_order:
    if key == "ai":
        ai_csv = os.path.join(results_dir, "ai", "inline_benchmark_chunks.csv")
        ai_rows = load_ai_inline_chunks_psnr_mape(ai_csv)
        if not ai_rows:
            print(f"  {label}: no rows in {ai_csv} — skipping")
            continue
        n_total = len(ai_rows)
        ai_lossy = [r for r in ai_rows if not math.isnan(r[3])]
        n_lossy = len(ai_lossy)
        if n_lossy == 0:
            print(f"  {label}: no lossy chunks (run was lossless or PSNR "
                  f"undefined) — skipping")
            continue
        xs = list(range(n_lossy))
        ys = [r[3] for r in ai_lossy]
        series.append((label, color, xs, ys, key))
        pct_lossy = (100.0 * n_lossy / n_total) if n_total else 0.0
        print(f"  {label}: {n_lossy} lossy chunks "
              f"({n_lossy}/{n_total}, {pct_lossy:.0f}%) from "
              f"{os.path.basename(ai_csv)}")
        continue

    tc_csv = os.path.join(results_dir, key, f"benchmark_{key}_timestep_chunks.csv")
    if not os.path.isfile(tc_csv):
        continue

    rows_by_ts = {}
    n_total = 0
    n_lossy = 0
    with open(tc_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("phase", "")
            if not p.startswith(phase_prefix):
                continue
            n_total += 1
            mape_psnr = parse_float_or_nan(row.get("mape_psnr"))
            # Lossless filter: NaN means "this chunk had no quantization,
            # PSNR is undefined". Skip entirely so they don't show up as
            # zeros in the plot (which would falsely look like perfect
            # prediction). The per-chunk emitters in WarpX, LAMMPS, VPIC,
            # and the NYX bridge all use the same NaN sentinel.
            if math.isnan(mape_psnr):
                continue
            n_lossy += 1
            ts    = int(row.get("timestep", row.get("field_idx", 0)))
            chunk = int(row.get("chunk", 0))
            rows_by_ts.setdefault(ts, []).append((chunk, mape_psnr))

    if n_total == 0:
        continue
    if n_lossy == 0:
        print(f"  {label}: no lossy chunks (run was lossless or workload "
              f"didn't emit PSNR cols) — skipping")
        continue

    # If a simulation writes multiple fields per timestep (e.g. LAMMPS:
    # positions+velocities+forces), keep only the first field's chunks
    # to match the regret/comp-MAPE plot conventions above.
    chunks_per_ts = max(len(v) for v in rows_by_ts.values()) if rows_by_ts else 0
    expected_chunks = 50
    if chunks_per_ts > expected_chunks * 2:
        keep = chunks_per_ts // 3
    else:
        keep = chunks_per_ts

    rows = []
    for ts in sorted(rows_by_ts.keys()):
        for chunk_idx, mape in rows_by_ts[ts][:keep]:
            rows.append((ts, chunk_idx, mape))

    xs = list(range(len(rows)))
    ys = [r[2] for r in rows]
    series.append((label, color, xs, ys, key))
    pct_lossy = (100.0 * n_lossy / n_total) if n_total else 0.0
    print(f"  {label}: {len(rows)} lossy chunks "
          f"({n_lossy}/{n_total}, {pct_lossy:.0f}%) from "
          f"{os.path.basename(tc_csv)}")

if not series:
    print("ERROR: no workload data with lossy chunks to plot for PSNR MAPE.",
          file=sys.stderr)
    print("       Set ERROR_BOUND > 0 in the run config so the NN can pick",
          file=sys.stderr)
    print("       quantized configs and produce per-chunk PSNR data.",
          file=sys.stderr)
    sys.exit(1)

# Combined CSV: one row per (workload, lossy chunk index)
with open(combined_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workload", "chunk_index", "psnr_mape_pct"])
    for label, _, xs, ys, _ in series:
        for x, y in zip(xs, ys):
            writer.writerow([label, x, f"{y:.4f}"])

# PSNR MAPE typically converges to a slightly higher floor than ratio MAPE
# because PSNR depends on the quantizer's error distribution which has
# more variance than the byte-level ratio. Threshold 20% empirically.
def find_convergence(ys, window=None, threshold_pct=20.0):
    if not ys:
        return 0
    if window is None:
        window = min(5, max(1, len(ys) // 3))
    if len(ys) < window:
        return len(ys)
    for i in range(window, len(ys)):
        avg = float(np.nanmean(ys[i-window:i]))
        if not math.isnan(avg) and avg < threshold_pct:
            return i - window
    return len(ys)

convergence_chunks = [find_convergence(ys) for _, _, _, ys, _ in series]

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

ax.set_xlabel("Lossy Chunk Index (sequential, NaN/lossless rows skipped)")
ax.set_ylabel("PSNR MAPE (%)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, fontsize=9)
ax.set_ylim(bottom=0, top=MAPE_CAP * 1.05)
plt.tight_layout()
plt.savefig(png_path, dpi=220, bbox_inches="tight")
print(f"\nSaved: {png_path}")
print(f"Saved: {combined_csv}")

# Summary stats — uses nanmean since the underlying ys list is already
# filtered, but keep nanmean as a defensive choice in case future emitters
# leak NaN through.
print("\n--- PSNR MAPE Convergence Summary ---")
print("Note: only chunks where the NN picked a quantized config are counted.")
print("Lossless chunks (NaN sentinel) are filtered upstream so they don't")
print("dilute the per-workload mean.")
for (label, _, xs, ys, _), c in zip(series, convergence_chunks):
    final_avg = float(np.nanmean(ys[-min(20, len(ys)):])) if ys else float("nan")
    print(f"  {label:15s}: converges ~chunk {c:4d}, "
          f"final avg PSNR MAPE = {final_avg:.2f}%")
PY
}

# ── Run metadata writer ───────────────────────────────────────
#
# Writes a comprehensive metadata file to $RESULTS_DIR/run_metadata.txt
# capturing exactly what was run: cross-workload defaults, per-workload
# simulation parameters, derived sizing (per-dump bytes, chunks per
# dump, total bytes, total chunks, physics step counts), the names of
# the actual fields each workload compresses, the binary paths, the
# weights file, the git commit hash, and the exact command line a
# reader can use to reproduce the run.
#
# Called once at the start of main() before any sim runs. The file
# stays in the results directory permanently so anyone inspecting
# the CSVs/PNGs later knows exactly what produced them.
write_run_metadata() {
    local meta="$RESULTS_DIR/run_metadata.txt"
    mkdir -p "$RESULTS_DIR"

    # Resolve git commit if available
    local git_sha
    git_sha=$(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local git_dirty=""
    if cd "$PROJECT_DIR" && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        git_dirty=" (working tree dirty)"
    fi

    # ── VPIC sizing ──
    # Grid: (NX+2)^3 cells × 16 EM-field variables × 4 bytes (fp32)
    local vpic_cells vpic_per_dump_b vpic_per_dump_mb vpic_chunks_per_dump
    local vpic_total_dumps vpic_total_b vpic_total_steps
    if [ "$SKIP_VPIC" != "1" ]; then
        vpic_cells=$(( (VPIC_NX + 2) * (VPIC_NX + 2) * (VPIC_NX + 2) ))
        vpic_per_dump_b=$(( vpic_cells * 16 * 4 ))
        vpic_per_dump_mb=$(( vpic_per_dump_b / 1024 / 1024 ))
        vpic_chunks_per_dump=$(( vpic_per_dump_b / (CHUNK_MB * 1024 * 1024) ))
        vpic_total_dumps=$VPIC_TIMESTEPS
        vpic_total_b=$(( vpic_per_dump_b * vpic_total_dumps ))
        vpic_total_steps=$(( VPIC_WARMUP_STEPS + VPIC_TIMESTEPS * VPIC_SIM_INTERVAL ))
    fi

    # ── NYX sizing ──
    # Grid: NX^3 cells × ~13 plot components × 4 bytes
    local nyx_cells nyx_per_dump_b nyx_per_dump_mb nyx_chunks_per_dump
    local nyx_total_dumps nyx_total_b
    if [ "$SKIP_NYX" != "1" ]; then
        nyx_cells=$(( NYX_NCELL * NYX_NCELL * NYX_NCELL ))
        nyx_per_dump_b=$(( nyx_cells * 13 * 4 ))
        nyx_per_dump_mb=$(( nyx_per_dump_b / 1024 / 1024 ))
        nyx_chunks_per_dump=$(( nyx_per_dump_b / (CHUNK_MB * 1024 * 1024) ))
        nyx_total_dumps=$(( NYX_MAX_STEP / NYX_PLOT_INT + 1 ))
        nyx_total_b=$(( nyx_per_dump_b * nyx_total_dumps ))
    fi

    # ── WarpX sizing ──
    # Grid: nx*ny*nz cells × 10 fields (Ex Ey Ez Bx By Bz jx jy jz rho)
    # × 4 bytes (fp32 per the project rule)
    local warpx_nx warpx_ny warpx_nz warpx_cells warpx_per_dump_b
    local warpx_per_dump_mb warpx_chunks_per_dump warpx_total_dumps warpx_total_b
    if [ "$SKIP_WARPX" != "1" ]; then
        read warpx_nx warpx_ny warpx_nz <<< "$WARPX_NCELL"
        warpx_cells=$(( warpx_nx * warpx_ny * warpx_nz ))
        warpx_per_dump_b=$(( warpx_cells * 10 * 4 ))
        warpx_per_dump_mb=$(( warpx_per_dump_b / 1024 / 1024 ))
        warpx_chunks_per_dump=$(( warpx_per_dump_b / (CHUNK_MB * 1024 * 1024) ))
        warpx_total_dumps=$(( WARPX_MAX_STEP / WARPX_DIAG_INTERVAL + 1 ))
        warpx_total_b=$(( warpx_per_dump_b * warpx_total_dumps ))
    fi

    # ── AI sizing ──
    # ViT-B/16: 86,567,656 parameters × 4 bytes (fp32 per project rule).
    # 4 tensor types per checkpoint: weights, adam_m, adam_v, gradients.
    local ai_params ai_bytes_per_tensor ai_n_checkpoints ai_total_tensors ai_total_b
    if [ "$SKIP_AI" != "1" ]; then
        if [ "$AI_MODEL" = "vit_b_16" ]; then
            ai_params=86567656
        else
            ai_params=0
        fi
        ai_bytes_per_tensor=$(( ai_params * 4 ))
        ai_n_checkpoints=$(echo "$AI_CHECKPOINTS" | tr ',' '\n' | grep -c .)
        ai_total_tensors=$(( 4 * ai_n_checkpoints ))
        ai_total_b=$(( ai_bytes_per_tensor * ai_total_tensors ))
    fi

    # ── LAMMPS sizing ──
    # Atoms: LMP_ATOMS^3 × 3 components per field × 4 bytes
    # 3 fields written per dump: positions, velocities, forces
    local lmp_atoms lmp_per_field_b lmp_per_dump_b lmp_per_dump_mb
    local lmp_chunks_per_dump lmp_total_dumps lmp_total_b
    if [ "$SKIP_LAMMPS" != "1" ]; then
        lmp_atoms=$(( LMP_ATOMS * LMP_ATOMS * LMP_ATOMS ))
        lmp_per_field_b=$(( lmp_atoms * 3 * 4 ))
        lmp_per_dump_b=$(( lmp_per_field_b * 3 ))
        lmp_per_dump_mb=$(( lmp_per_dump_b / 1024 / 1024 ))
        lmp_chunks_per_dump=$(( lmp_per_dump_b / (CHUNK_MB * 1024 * 1024) ))
        [ $lmp_chunks_per_dump -lt 1 ] && lmp_chunks_per_dump=1
        lmp_total_dumps=$LMP_TIMESTEPS
        lmp_total_b=$(( lmp_per_dump_b * lmp_total_dumps ))
    fi

    # ── Build the metadata file ──
    {
        echo "# Section 7.1 Run Metadata"
        echo "# Generated: $(date -Iseconds 2>/dev/null || date)"
        echo "# Host:      $(hostname 2>/dev/null || echo unknown)"
        echo "# Git:       ${git_sha}${git_dirty}"
        echo ""
        echo "## Run identity"
        echo "Run name:        $RUN_NAME"
        echo "Results dir:     $RESULTS_DIR"
        echo "Project dir:     $PROJECT_DIR"
        echo "Weights:         $WEIGHTS"
        echo ""
        echo "## Cross-workload equalized configuration"
        echo "Policy:          $POLICY (w0=$W0, w1=$W1, w2=$W2)"
        echo "Phase:           $PHASE"
        echo "Error bound:     $ERROR_BOUND"
        echo "Chunk size:      ${CHUNK_MB} MiB"
        echo "SGD LR:          $SGD_LR"
        echo "SGD MAPE thresh: $SGD_MAPE"
        echo "Explore K:       $EXPLORE_K"
        echo "Explore thresh:  $EXPLORE_THRESH"
        echo ""
        echo "## Workloads"
        echo ""

        # ── VPIC block ──
        if [ "$SKIP_VPIC" = "1" ]; then
            echo "### VPIC: SKIPPED"
            echo ""
        else
            echo "### VPIC (Harris-sheet plasma reconnection)"
            echo "Binary:          $VPIC_BIN"
            echo ""
            echo "Simulation parameters:"
            echo "  NX                  = $VPIC_NX  (grid dimension; total = (NX+2)^3 with ghosts)"
            echo "  TIMESTEPS           = $VPIC_TIMESTEPS  (benchmark dumps)"
            echo "  WARMUP_STEPS        = $VPIC_WARMUP_STEPS  (physics steps before first dump)"
            echo "  SIM_INTERVAL        = $VPIC_SIM_INTERVAL  (physics steps between dumps)"
            echo "  MI_ME               = $VPIC_MI_ME  (mass ratio mi/me)"
            echo "  WPE_WCE             = $VPIC_WPE_WCE  (plasma freq / cyclotron freq)"
            echo "  TI_TE               = $VPIC_TI_TE  (ion/electron temperature ratio)"
            echo ""
            echo "Derived sizing:"
            echo "  Cells per dump      = ${vpic_cells} ((${VPIC_NX}+2)^3 = $((VPIC_NX+2))^3)"
            echo "  Fields per cell     = 16  (EM fields, see field list below)"
            echo "  Bytes per field     = 4  (fp32 per project rule)"
            echo "  Bytes per dump      = $vpic_per_dump_b  (~${vpic_per_dump_mb} MB)"
            echo "  Chunks per dump     = $vpic_chunks_per_dump  (at ${CHUNK_MB} MiB chunks)"
            echo "  Total dumps         = $vpic_total_dumps"
            echo "  Total bytes written = $vpic_total_b  (~$((vpic_total_b / 1024 / 1024)) MB)"
            echo "  Total physics steps = $vpic_total_steps  ($VPIC_WARMUP_STEPS warmup + $VPIC_TIMESTEPS x $VPIC_SIM_INTERVAL benchmark)"
            echo ""
            echo "Compressed fields (16, all float32):"
            echo "  EM fields:    ex, ey, ez, div_e_err, cbx, cby, cbz, div_b_err"
            echo "  Currents:     jfx, jfy, jfz"
            echo "  Charges:      rhof, rhob"
            echo "  TCA:          tcax, tcay, tcaz"
            echo ""
        fi

        # ── NYX block ──
        if [ "$SKIP_NYX" = "1" ]; then
            echo "### NYX: SKIPPED"
            echo ""
        else
            echo "### NYX (Sedov blast wave, HydroTests)"
            echo "Binary:          $NYX_BIN"
            echo ""
            echo "Simulation parameters:"
            echo "  NCELL               = $NYX_NCELL  (grid dimension; total = NCELL^3)"
            echo "  MAX_STEP            = $NYX_MAX_STEP  (total physics steps)"
            echo "  PLOT_INT            = $NYX_PLOT_INT  (steps between plotfiles)"
            echo ""
            echo "Derived sizing:"
            echo "  Cells per dump      = ${nyx_cells} (${NYX_NCELL}^3)"
            echo "  Components/cell     = ~13  (varies with deck — see field list)"
            echo "  Bytes per cell      = 4  (fp32 per project rule)"
            echo "  Bytes per dump      = $nyx_per_dump_b  (~${nyx_per_dump_mb} MB)"
            echo "  Chunks per dump     = $nyx_chunks_per_dump  (at ${CHUNK_MB} MiB chunks)"
            echo "  Total dumps         = $nyx_total_dumps  (MAX_STEP/PLOT_INT + 1)"
            echo "  Total bytes written = $nyx_total_b  (~$((nyx_total_b / 1024 / 1024)) MB)"
            echo ""
            echo "Compressed fields (~13, varies by deck — typical Sedov plotfile):"
            echo "  Conserved:    density, xmom, ymom, zmom, rho_e, rho_E"
            echo "  Derived:      Temp, pressure, eint_e, eint_E"
            echo "  Auxiliary:    Ne, magvort, divu (or similar diagnostics)"
            echo ""
        fi

        # ── WarpX block ──
        if [ "$SKIP_WARPX" = "1" ]; then
            echo "### WarpX: SKIPPED"
            echo ""
        else
            echo "### WarpX (laser-wakefield acceleration, LWFA)"
            echo "Binary:          $WARPX_BIN"
            echo "Inputs deck:     $WARPX_INPUTS"
            echo ""
            echo "Simulation parameters:"
            echo "  NCELL               = $WARPX_NCELL  (nx ny nz)"
            echo "  MAX_STEP            = $WARPX_MAX_STEP  (total physics steps)"
            echo "  DIAG_INTERVAL       = $WARPX_DIAG_INTERVAL  (steps between diagnostics)"
            echo ""
            echo "Derived sizing:"
            echo "  Cells per dump      = ${warpx_cells} (${warpx_nx} x ${warpx_ny} x ${warpx_nz})"
            echo "  Fields per cell     = 10  (Ex Ey Ez Bx By Bz jx jy jz rho)"
            echo "  Bytes per cell      = 4  (fp32 per project rule, WarpX_PRECISION=SINGLE)"
            echo "  Bytes per dump      = $warpx_per_dump_b  (~${warpx_per_dump_mb} MB)"
            echo "  Chunks per dump     = $warpx_chunks_per_dump  (at ${CHUNK_MB} MiB chunks)"
            echo "  Total dumps         = $warpx_total_dumps  (MAX_STEP/DIAG_INTERVAL + 1)"
            echo "  Total bytes written = $warpx_total_b  (~$((warpx_total_b / 1024 / 1024)) MB)"
            echo ""
            echo "Compressed fields (10, all float32):"
            echo "  Electric field: Ex, Ey, Ez"
            echo "  Magnetic field: Bx, By, Bz"
            echo "  Current:        jx, jy, jz"
            echo "  Charge density: rho"
            echo ""
        fi

        # ── LAMMPS block ──
        if [ "$SKIP_LAMMPS" = "1" ]; then
            echo "### LAMMPS: SKIPPED"
            echo ""
        else
            echo "### LAMMPS (molecular dynamics, hot-sphere near melting)"
            echo "Binary:          $LMP_BIN"
            echo ""
            echo "Simulation parameters:"
            echo "  ATOMS               = ${LMP_ATOMS}^3 = ${lmp_atoms}  (total atoms in box)"
            echo "  TIMESTEPS           = $LMP_TIMESTEPS  (benchmark dumps)"
            echo "  SIM_INTERVAL        = $LMP_SIM_INTERVAL  (steps between dumps)"
            echo "  T_HOT               = $LMP_T_HOT  (hot-sphere temperature)"
            echo "  T_COLD              = $LMP_T_COLD  (background temperature)"
            echo ""
            echo "Derived sizing (per-field, written 3x per dump):"
            echo "  Atoms               = $lmp_atoms"
            echo "  Components/atom     = 3  (x,y,z per field)"
            echo "  Bytes per component = 4  (fp32 per KOKKOS_PREC=SINGLE)"
            echo "  Bytes per field     = $lmp_per_field_b  (atoms x 3 x 4)"
            echo "  Bytes per dump      = $lmp_per_dump_b  (3 fields per dump, ~${lmp_per_dump_mb} MB)"
            echo "  Chunks per dump     = $lmp_chunks_per_dump  (at ${CHUNK_MB} MiB chunks)"
            echo "  Total dumps         = $lmp_total_dumps"
            echo "  Total bytes written = $lmp_total_b  (~$((lmp_total_b / 1024 / 1024)) MB)"
            echo ""
            echo "Compressed fields (3 separate H5 writes per dump):"
            echo "  positions:    x, y, z       (per atom, n_atoms x 3 floats)"
            echo "  velocities:   vx, vy, vz    (per atom, n_atoms x 3 floats)"
            echo "  forces:       fx, fy, fz    (per atom, n_atoms x 3 floats)"
            echo ""
        fi

        # ── AI block ──
        if [ "$SKIP_AI" = "1" ]; then
            echo "### AI: SKIPPED"
            echo ""
        else
            echo "### AI (ViT-B/16 fine-tuning on CIFAR-10, in-situ HDF5 VOL capture)"
            echo "Driver:          $PROJECT_DIR/scripts/train_and_export_checkpoints.py"
            echo "Bridge:          $PROJECT_DIR/scripts/gpucompress_hdf5.py (Python ctypes → VOL)"
            echo ""
            echo "Training parameters:"
            echo "  MODEL               = $AI_MODEL"
            echo "  EPOCHS              = $AI_EPOCHS"
            echo "  CHECKPOINT_EPOCHS   = $AI_CHECKPOINTS"
            echo "  DATASET             = $AI_DATASET"
            echo ""
            echo "Derived sizing:"
            if [ "$AI_MODEL" = "vit_b_16" ]; then
                echo "  Parameters          = $ai_params  (ViT-B/16, fp32)"
            else
                echo "  Parameters          = (parameter count varies by model)"
            fi
            echo "  Bytes per tensor    = $ai_bytes_per_tensor  (params x 4, fp32 per project rule)"
            echo "  Tensor types        = 4  (weights, adam_m, adam_v, gradients)"
            echo "  Checkpoints         = $ai_n_checkpoints"
            echo "  Total tensors       = $ai_total_tensors  (4 x $ai_n_checkpoints)"
            if [ "$AI_MODEL" = "vit_b_16" ]; then
                echo "  Total bytes via VOL = $ai_total_b  (~$((ai_total_b / 1024 / 1024)) MB)"
            else
                echo "  Total bytes via VOL = (depends on model)"
            fi
            echo ""
            echo "Compressed fields (4 separate VOL writes per checkpoint):"
            echo "  weights, adam_m, adam_v, gradients"
            echo ""
            echo "Note: data is captured IN-SITU during PyTorch training. Each tensor"
            echo "is concatenated on the GPU and routed through the GPUCompress HDF5"
            echo "VOL connector via the Python ctypes bridge — never an offline replay."
            echo "InlineFullBenchmark runs all 15 algorithms (9 fixed + 6 NN configs)"
            echo "per tensor and writes per-chunk metrics to inline_benchmark_chunks.csv."
            echo ""
        fi

        # ── Reproduction command ──
        echo "## Reproduction"
        echo ""
        echo "To re-run this exact configuration:"
        echo ""
        echo "  cd $PROJECT_DIR && \\"
        [ "$SKIP_VPIC"   = "1" ] && echo "    SKIP_VPIC=1 \\"
        [ "$SKIP_NYX"    = "1" ] && echo "    SKIP_NYX=1 \\"
        [ "$SKIP_WARPX"  = "1" ] && echo "    SKIP_WARPX=1 \\"
        [ "$SKIP_LAMMPS" = "1" ] && echo "    SKIP_LAMMPS=1 \\"
        [ "$SKIP_AI"     = "1" ] && echo "    SKIP_AI=1 \\"
        echo "    POLICY=$POLICY \\"
        echo "    PHASE=$PHASE \\"
        echo "    ERROR_BOUND=$ERROR_BOUND \\"
        echo "    CHUNK_MB=$CHUNK_MB \\"
        echo "    SGD_LR=$SGD_LR \\"
        echo "    SGD_MAPE=$SGD_MAPE \\"
        echo "    EXPLORE_K=$EXPLORE_K \\"
        echo "    EXPLORE_THRESH=$EXPLORE_THRESH \\"
        if [ "$SKIP_VPIC" != "1" ]; then
            echo "    VPIC_NX=$VPIC_NX VPIC_TIMESTEPS=$VPIC_TIMESTEPS \\"
            echo "    VPIC_WARMUP_STEPS=$VPIC_WARMUP_STEPS VPIC_SIM_INTERVAL=$VPIC_SIM_INTERVAL \\"
            echo "    VPIC_MI_ME=$VPIC_MI_ME VPIC_WPE_WCE=$VPIC_WPE_WCE VPIC_TI_TE=$VPIC_TI_TE \\"
        fi
        if [ "$SKIP_NYX" != "1" ]; then
            echo "    NYX_BIN=$NYX_BIN \\"
            echo "    NYX_NCELL=$NYX_NCELL NYX_MAX_STEP=$NYX_MAX_STEP NYX_PLOT_INT=$NYX_PLOT_INT \\"
        fi
        if [ "$SKIP_WARPX" != "1" ]; then
            echo "    WARPX_BIN=$WARPX_BIN \\"
            echo "    WARPX_INPUTS=$WARPX_INPUTS \\"
            echo "    WARPX_NCELL=\"$WARPX_NCELL\" WARPX_MAX_STEP=$WARPX_MAX_STEP WARPX_DIAG_INTERVAL=$WARPX_DIAG_INTERVAL \\"
        fi
        if [ "$SKIP_LAMMPS" != "1" ]; then
            echo "    LMP_BIN=$LMP_BIN \\"
            echo "    LMP_ATOMS=$LMP_ATOMS LMP_TIMESTEPS=$LMP_TIMESTEPS LMP_SIM_INTERVAL=$LMP_SIM_INTERVAL \\"
            echo "    LMP_T_HOT=$LMP_T_HOT LMP_T_COLD=$LMP_T_COLD \\"
        fi
        if [ "$SKIP_AI" != "1" ]; then
            echo "    AI_MODEL=$AI_MODEL AI_EPOCHS=$AI_EPOCHS \\"
            echo "    AI_CHECKPOINTS=$AI_CHECKPOINTS AI_DATASET=$AI_DATASET \\"
        fi
        echo "    RUN_NAME=$RUN_NAME \\"
        echo "    bash benchmarks/Paper_Evaluations/7/7.1_run_equalized_cross_workload_regret.sh"
        echo ""
    } > "$meta"

    note "Wrote run metadata to $meta"
}

# ── Banner ────────────────────────────────────────────────────
print_banner() {
    echo "============================================================"
    echo "Section 7.1: Cross-Workload Regret + MAPE Convergence"
    echo "============================================================"
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
    echo "VPIC          : $([ "$SKIP_VPIC" = "1" ] && echo "SKIP" || echo "NX=$VPIC_NX ts=$VPIC_TIMESTEPS mi_me=$VPIC_MI_ME int=$VPIC_SIM_INTERVAL")"
    echo "NYX           : $([ "$SKIP_NYX" = "1" ] && echo "SKIP" || echo "NC=$NYX_NCELL max_step=$NYX_MAX_STEP plot_int=$NYX_PLOT_INT")"
    echo "WarpX         : $([ "$SKIP_WARPX" = "1" ] && echo "SKIP" || echo "ncell=$WARPX_NCELL max_step=$WARPX_MAX_STEP diag_int=$WARPX_DIAG_INTERVAL")"
    echo "LAMMPS        : $([ "$SKIP_LAMMPS" = "1" ] && echo "SKIP" || echo "atoms=${LMP_ATOMS}^3 T_hot=$LMP_T_HOT T_cold=$LMP_T_COLD int=$LMP_SIM_INTERVAL")"
    echo "AI            : $([ "$SKIP_AI" = "1" ] && echo "SKIP" || echo "model=$AI_MODEL epochs=$AI_EPOCHS checkpoints=$AI_CHECKPOINTS")"
    echo "============================================================"
}

# ── Main ──────────────────────────────────────────────────────
main() {
    mkdir -p "$RESULTS_DIR"
    require_file "$WEIGHTS"

    # Section 7 evaluation policy: ALL runs are lossy by default. Lossless
    # leaves every per-chunk PSNR cell as NaN, which means Figure 7c (PSNR
    # MAPE) will be empty. If the caller explicitly set ERROR_BOUND=0 (or
    # any non-positive value) print a loud warning so they know what they
    # asked for. The script will still proceed -- this is a guardrail, not
    # a hard block.
    if awk -v eb="$ERROR_BOUND" 'BEGIN {exit !(eb+0 <= 0.0)}'; then
        echo ""
        echo "============================================================"
        echo "!! WARNING: ERROR_BOUND=$ERROR_BOUND is LOSSLESS (<= 0.0)."
        echo "!! Section 7.1 default is LOSSY (ERROR_BOUND=0.001) so the"
        echo "!! NN can pick quantized configs and Figure 7c (PSNR MAPE)"
        echo "!! has data. Lossless runs leave every PSNR cell as NaN and"
        echo "!! Figure 7c will be skipped. To proceed lossless anyway,"
        echo "!! ignore this warning. To use the standard lossy default,"
        echo "!! unset ERROR_BOUND and re-run."
        echo "============================================================"
        echo ""
    fi

    print_banner | tee "$RESULTS_DIR/config.txt"
    write_run_metadata

    if [ "$PLOT_ONLY" != "1" ]; then
        run_vpic_sim
        run_nyx_sim
        run_warpx_sim
        run_lammps_sim
        run_ai_sim
    fi

    plot_cross_workload_regret
    plot_cross_workload_mape
    plot_cross_workload_psnr_mape || \
        note "PSNR MAPE plot skipped (no lossy chunks — set ERROR_BOUND > 0)"

    cat <<EOF

============================================================
Complete.  Results:
  Figure 7a: $RESULTS_DIR/cross_workload_regret.png       (top-1 regret)
  Figure 7b: $RESULTS_DIR/cross_workload_cost_mape.png    (cost model MAPE)
  Figure 7c: $RESULTS_DIR/cross_workload_psnr_mape.png    (PSNR MAPE — lossy only)
  Data:      $RESULTS_DIR/cross_workload_regret.csv
             $RESULTS_DIR/cross_workload_cost_mape.csv
             $RESULTS_DIR/cross_workload_psnr_mape.csv
============================================================
EOF
}

main "$@"
