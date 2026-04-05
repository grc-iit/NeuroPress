#!/bin/bash
# ============================================================
# LAMMPS GPUCompress Benchmark
#
# Runs all 12 compression phases (no-comp, 8 fixed, 3 NN) with
# weight isolation between NN phases. Each phase is a separate
# LAMMPS run to guarantee clean NN state.
#
# Follows the same pattern as VPIC benchmark:
# - Fixed phases: policy-independent, run once each
# - NN phases: run once per policy (speed/balanced/ratio)
# - Simulation evolves between dumps (sim_interval steps between writes)
# - Results in VPIC-compatible CSV format
#
# Usage:
#   bash benchmarks/lammps/run_lammps_benchmark.sh
#
# Environment variables:
#   LMP_BIN          Path to lmp binary [$HOME/lammps/build/lmp]
#   LMP_ATOMS        Box size per dimension [80 → 2M atoms]
#   CHUNK_MB         HDF5 chunk size [4]
#   TIMESTEPS        Benchmark write cycles [10]
#   SIM_INTERVAL     Physics steps between dumps [50]
#   WARMUP_STEPS     Physics steps before first dump [100]
#   POLICIES         NN policies [balanced,ratio,speed]
#   VERIFY           Lossless verification [1]
#   RESULTS_DIR      Output directory [auto]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
LMP_ATOMS=${LMP_ATOMS:-80}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-10}
SIM_INTERVAL=${SIM_INTERVAL:-50}
WARMUP_STEPS=${WARMUP_STEPS:-100}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
VERIFY=${VERIFY:-1}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

# Derived
TOTAL_STEPS=$((WARMUP_STEPS + TIMESTEPS * SIM_INTERVAL))
NATOMS_APPROX=$(python3 -c "print(4 * $LMP_ATOMS**3)" 2>/dev/null || echo "unknown")
DATA_MB=$(python3 -c "print(f'{4 * $LMP_ATOMS**3 * 36 / 1048576:.1f}')" 2>/dev/null || echo "unknown")
VERIFY_TAG=""
[ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/eval_box${LMP_ATOMS}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}${VERIFY_TAG}}"

export LD_LIBRARY_PATH="$GPUC_DIR/build:$GPUC_DIR/examples:/tmp/hdf5-install/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Fixed phases (no NN, policy-independent) ──
FIXED_PHASES="no-comp lz4 snappy deflate gdeflate zstd ans cascaded bitcomp"

# ── NN phases (policy-dependent, separate weight state per run) ──
NN_PHASES="nn nn-rl nn-rl+exp50"

echo "============================================================"
echo "LAMMPS GPUCompress Benchmark"
echo "============================================================"
echo "  Binary:        $LMP_BIN"
echo "  Box:           ${LMP_ATOMS}^3 (~$NATOMS_APPROX atoms)"
echo "  Data/dump:     ~$DATA_MB MB (3 fields, fp32)"
echo "  Warmup:        $WARMUP_STEPS steps"
echo "  Timesteps:     $TIMESTEPS (every $SIM_INTERVAL sim steps)"
echo "  Total steps:   $TOTAL_STEPS"
echo "  Chunk:         ${CHUNK_MB} MB"
echo "  Verify:        $VERIFY"
echo "  Policies:      $POLICIES"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Generate input script ──
# Hot sphere expanding into cold lattice for data diversity
generate_input() {
    local dump_freq=$1
    local total_run=$2
    cat << EOF
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
velocity        cold create 0.01 87287 loop geom
velocity        hot create 10.0 12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress ${dump_freq} positions velocities forces
thermo          ${dump_freq}
timestep        0.003
run             ${total_run}
EOF
}

# ── CSV header ──
CSV="$RESULTS_DIR/benchmark_lammps_timesteps.csv"
echo "rank,phase,timestep,write_ms,ratio,orig_mb,comp_mb,verify" > "$CSV"

# ── Run a single phase ──
run_phase() {
    local phase=$1
    local algo=$2
    local policy=$3
    local phase_label=$4  # "lz4" or "nn/balanced"

    local workdir="$RESULTS_DIR/work_${phase_label//\//_}"
    mkdir -p "$workdir"
    cd "$workdir"
    rm -rf gpuc_step_*

    # Generate input with warmup + benchmark dumps
    generate_input "$SIM_INTERVAL" "$TOTAL_STEPS" > input.lmp

    # Set GPUCompress environment
    export GPUCOMPRESS_ALGO="$algo"
    export GPUCOMPRESS_VERIFY="$VERIFY"
    export GPUCOMPRESS_WEIGHTS="$WEIGHTS"

    # Set NN policy weights
    case "$policy" in
        speed)    export GPUCOMPRESS_POLICY="speed" ;;
        ratio)    export GPUCOMPRESS_POLICY="ratio" ;;
        balanced) export GPUCOMPRESS_POLICY="balanced" ;;
        *)        export GPUCOMPRESS_POLICY="ratio" ;;
    esac

    # Handle no-comp: use lz4 but we'll measure uncompressed size
    if [ "$phase" = "no-comp" ]; then
        export GPUCOMPRESS_ALGO="lz4"
    fi

    # Run LAMMPS
    local start_ms=$(date +%s%N)
    "$LMP_BIN" -k on g 1 -sf kk -in input.lmp > lammps.log 2>&1 || true
    local end_ms=$(date +%s%N)
    local total_ms=$(( (end_ms - start_ms) / 1000000 ))

    # Parse results from output directories
    local natoms=$(grep "Created.*atoms$" lammps.log | head -1 | awk '{print $2}')
    [ -z "$natoms" ] && natoms=0
    local orig_bytes=$((natoms * 36))
    local orig_mb=$(echo "scale=1; $orig_bytes/1048576" | bc 2>/dev/null)

    local ts=0
    for d in $(ls -d gpuc_step_* 2>/dev/null | sort); do
        # Skip warmup dumps (first WARMUP_STEPS / SIM_INTERVAL dumps)
        local step_num=$(echo "$d" | sed 's/gpuc_step_0*//')
        [ -z "$step_num" ] && step_num=0
        if [ "$step_num" -lt "$WARMUP_STEPS" ]; then
            continue
        fi

        local comp_bytes=$(du -sb "$d" | awk '{print $1}')
        local comp_mb=$(echo "scale=2; $comp_bytes/1048576" | bc 2>/dev/null)
        local ratio=$(echo "scale=2; $orig_bytes/$comp_bytes" | bc 2>/dev/null)

        if [ "$phase" = "no-comp" ]; then
            ratio="1.00"
            comp_mb="$orig_mb"
        fi

        local write_ms=$(echo "scale=1; $total_ms / ($TIMESTEPS + 1)" | bc 2>/dev/null)  # approximate per-dump

        local verify_ok=1
        if grep -q "VERIFY FAILED" lammps.log 2>/dev/null; then
            verify_ok=0
        fi

        echo "0,$phase_label,$ts,$write_ms,$ratio,$orig_mb,$comp_mb,$verify_ok" >> "$CSV"
        ts=$((ts + 1))
    done

    # Report
    local last_dir=$(ls -d gpuc_step_* 2>/dev/null | sort | tail -1)
    if [ -n "$last_dir" ]; then
        local last_comp=$(du -sb "$last_dir" | awk '{print $1}')
        local last_ratio=$(echo "scale=2; $orig_bytes/$last_comp" | bc 2>/dev/null)
        printf "  %-20s ratio=%-6s (%d dumps, %d ms)\n" "$phase_label" "${last_ratio}x" "$ts" "$total_ms"
    else
        printf "  %-20s NO OUTPUT\n" "$phase_label"
    fi

    cd "$RESULTS_DIR"
}

# ============================================================
# Phase 1: Fixed algorithms (policy-independent)
# ============================================================
echo ">>> Phase 1: Fixed algorithms"
echo ""

for phase in $FIXED_PHASES; do
    case "$phase" in
        no-comp)  algo="lz4" ;;
        *)        algo="$phase" ;;
    esac
    run_phase "$phase" "$algo" "ratio" "$phase"
done

echo ""

# ============================================================
# Phase 2: NN phases (per policy, clean weight state each run)
# ============================================================
echo ">>> Phase 2: NN phases (per policy)"
echo ""

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"
for policy in "${POL_ARRAY[@]}"; do
    echo "  --- Policy: $policy ---"
    for nn_phase in $NN_PHASES; do
        case "$nn_phase" in
            nn)            algo="auto" ;;
            nn-rl)         algo="auto" ;;
            nn-rl+exp50)   algo="auto" ;;
        esac
        phase_label="${nn_phase}/${policy}"
        run_phase "$nn_phase" "$algo" "$policy" "$phase_label"
    done
    echo ""
done

# ============================================================
# Split results into per-policy directories
# ============================================================
echo ">>> Splitting results by policy..."

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"
for pol in "${POL_ARRAY[@]}"; do
    case "$pol" in
        balanced) label="balanced_w1-1-1" ;;
        ratio)    label="ratio_only_w0-0-1" ;;
        speed)    label="speed_only_w1-1-0" ;;
        *)        label="$pol" ;;
    esac
    POL_DIR="$RESULTS_DIR/$label"
    mkdir -p "$POL_DIR"

    # Header + fixed rows (no "/" in phase) + this policy's NN rows (strip suffix)
    head -1 "$CSV" > "$POL_DIR/benchmark_lammps_timesteps.csv"
    awk -F',' -v pol="/$pol" -v OFS=',' '
        NR==1 { next }
        $2 !~ /\// { print }
        index($2, pol) > 0 { sub(pol, "", $2); print }
    ' "$CSV" >> "$POL_DIR/benchmark_lammps_timesteps.csv"

    echo "  $label: $(wc -l < "$POL_DIR/benchmark_lammps_timesteps.csv") rows"
done

echo ""
echo "============================================================"
echo "LAMMPS Benchmark Complete"
echo "  Results: $RESULTS_DIR"
echo "  CSV:     $CSV"
echo "============================================================"
