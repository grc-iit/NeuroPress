#!/bin/bash
# ============================================================
# nekRS GPUCompress Benchmark
#
# Runs all 12 compression phases (no-comp, 8 fixed, 3 NN) with
# weight isolation between NN phases. Each phase is a separate
# nekRS run to guarantee clean NN state.
#
# Follows the same pattern as VPIC benchmark:
# - Fixed phases: policy-independent, run once each
# - NN phases: run once per policy (speed/balanced/ratio)
# - Simulation evolves over checkpointInterval steps between dumps
# - Results in VPIC-compatible CSV format
#
# Usage:
#   bash benchmarks/nekrs/run_nekrs_benchmark.sh
#
# Prerequisites:
#   - nekRS installed ($NEKRS_HOME set)
#   - GPUCompress-enabled TGV case (patches applied)
#   - OCCA kernels cached (run TGV once first)
#
# Environment variables:
#   NEKRS_HOME       nekRS installation [$HOME/.local/nekrs]
#   CASE_DIR         Path to GPUCompress TGV case
#   NP               MPI ranks [2]
#   NUM_STEPS        Total simulation steps [20]
#   CHECKPOINT_INT   Steps between checkpoints [10]
#   POLICIES         NN policies [balanced,ratio,speed]
#   VERIFY           Lossless verification [1]
#   RESULTS_DIR      Output directory [auto]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──
NEKRS_HOME="${NEKRS_HOME:-$HOME/.local/nekrs}"
CASE_DIR="${CASE_DIR:-$HOME/tgv_gpucompress}"
NP=${NP:-2}
NUM_STEPS=${NUM_STEPS:-20}
CHECKPOINT_INT=${CHECKPOINT_INT:-10}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
VERIFY=${VERIFY:-1}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
USE_FP32=${USE_FP32:-1}

# Select binary
if [ "$USE_FP32" = "1" ]; then
    NEKRS_BIN="$NEKRS_HOME/bin/nekrs-fp32"
else
    NEKRS_BIN="$NEKRS_HOME/bin/nekrs"
fi

VERIFY_TAG=""
[ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
FP_TAG="fp32"
[ "$USE_FP32" = "0" ] && FP_TAG="fp64"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/eval_tgv_${FP_TAG}_np${NP}_ts${NUM_STEPS}${VERIFY_TAG}}"

export PATH="$NEKRS_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GPUC_DIR/build:$GPUC_DIR/examples:/tmp/hdf5-install/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Phases ──
FIXED_PHASES="no-comp lz4 snappy deflate gdeflate zstd ans cascaded bitcomp"
NN_PHASES="nn nn-rl nn-rl+exp50"

echo "============================================================"
echo "nekRS GPUCompress Benchmark"
echo "============================================================"
echo "  Binary:        $NEKRS_BIN"
echo "  Case:          $CASE_DIR"
echo "  MPI ranks:     $NP"
echo "  Precision:     $FP_TAG"
echo "  Steps:         $NUM_STEPS (checkpoint every $CHECKPOINT_INT)"
echo "  Verify:        $VERIFY"
echo "  Policies:      $POLICIES"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Verify case directory
if [ ! -f "$CASE_DIR/tgv.udf" ] || [ ! -f "$CASE_DIR/tgv.par" ]; then
    echo "ERROR: Case directory $CASE_DIR missing tgv.udf or tgv.par"
    echo "  Copy patches: cp $SCRIPT_DIR/patches/* $CASE_DIR/"
    exit 1
fi

# ── Update par file for this benchmark ──
update_par() {
    local par="$CASE_DIR/tgv.par"
    sed -i "s/^numSteps = .*/numSteps = $NUM_STEPS/" "$par" 2>/dev/null || true
    sed -i "s/^checkpointInterval = .*/checkpointInterval = $CHECKPOINT_INT/" "$par" 2>/dev/null || true
}
update_par

# ── CSV header ──
CSV="$RESULTS_DIR/benchmark_nekrs_timesteps.csv"
echo "rank,phase,timestep,write_ms,ratio,orig_mb,comp_mb,verify" > "$CSV"

# ── Run a single phase ──
run_phase() {
    local phase=$1
    local algo=$2
    local policy=$3
    local phase_label=$4

    cd "$CASE_DIR"
    rm -rf gpuc_step_* tgv0.f*

    # Set GPUCompress environment
    export GPUCOMPRESS_ALGO="$algo"
    export GPUCOMPRESS_VERIFY="$VERIFY"
    export GPUCOMPRESS_WEIGHTS="$WEIGHTS"

    case "$policy" in
        speed)    export GPUCOMPRESS_POLICY="speed" ;;
        ratio)    export GPUCOMPRESS_POLICY="ratio" ;;
        balanced) export GPUCOMPRESS_POLICY="balanced" ;;
        *)        export GPUCOMPRESS_POLICY="ratio" ;;
    esac

    if [ "$phase" = "no-comp" ]; then
        export GPUCOMPRESS_ALGO="lz4"
    fi

    # Run nekRS
    local start_ms=$(date +%s%N)
    mpirun --oversubscribe -np "$NP" "$NEKRS_BIN" \
        --setup tgv.par --backend CUDA --device-id 0 \
        > "$RESULTS_DIR/nekrs_${phase_label//\//_}.log" 2>&1 || true
    local end_ms=$(date +%s%N)
    local total_ms=$(( (end_ms - start_ms) / 1000000 ))

    # Parse results
    local ts=0
    for d in $(ls -d gpuc_step_* 2>/dev/null | sort); do
        local comp_bytes=$(du -sb "$d" | awk '{print $1}')
        local comp_mb=$(echo "scale=2; $comp_bytes/1048576" | bc 2>/dev/null)

        # Estimate original size from GPUCompress log
        local orig_mb=$(grep "wrote.*fields.*MB" "$RESULTS_DIR/nekrs_${phase_label//\//_}.log" | head -1 | grep -o '[0-9.]* MB' | head -1 | awk '{print $1}')
        [ -z "$orig_mb" ] && orig_mb="0"
        local orig_bytes=$(echo "$orig_mb * 1048576 * $NP" | bc 2>/dev/null | cut -d. -f1)
        [ -z "$orig_bytes" ] && orig_bytes=1

        local ratio=$(echo "scale=2; $orig_bytes/$comp_bytes" | bc 2>/dev/null)
        if [ "$phase" = "no-comp" ]; then
            ratio="1.00"
            comp_mb=$(echo "scale=1; $orig_bytes/1048576" | bc 2>/dev/null)
        fi

        local write_ms=$(echo "scale=1; $total_ms / ($(ls -d gpuc_step_* | wc -l))" | bc 2>/dev/null)

        local verify_ok=1
        if grep -q "VERIFY FAILED" "$RESULTS_DIR/nekrs_${phase_label//\//_}.log" 2>/dev/null; then
            verify_ok=0
        fi

        echo "0,$phase_label,$ts,$write_ms,$ratio,$orig_mb,$comp_mb,$verify_ok" >> "$CSV"
        ts=$((ts + 1))
    done

    # Report
    local last_dir=$(ls -d gpuc_step_* 2>/dev/null | sort | tail -1)
    if [ -n "$last_dir" ]; then
        local last_comp=$(du -sb "$last_dir" | awk '{print $1}')
        local last_ratio=$(echo "scale=2; ${orig_bytes:-1}/$last_comp" | bc 2>/dev/null)
        printf "  %-20s ratio=%-6s (%d dumps, %d ms)\n" "$phase_label" "${last_ratio}x" "$ts" "$total_ms"
    else
        printf "  %-20s NO OUTPUT\n" "$phase_label"
    fi

    cd "$RESULTS_DIR"
}

# ============================================================
# Phase 1: Fixed algorithms
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
# Phase 2: NN phases (per policy, clean weight state)
# ============================================================
echo ">>> Phase 2: NN phases (per policy)"
echo ""

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"
for policy in "${POL_ARRAY[@]}"; do
    echo "  --- Policy: $policy ---"
    for nn_phase in $NN_PHASES; do
        run_phase "$nn_phase" "auto" "$policy" "${nn_phase}/${policy}"
    done
    echo ""
done

# ============================================================
# Split results by policy
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

    head -1 "$CSV" > "$POL_DIR/benchmark_nekrs_timesteps.csv"
    awk -F',' -v pol="/$pol" -v OFS=',' '
        NR==1 { next }
        $2 !~ /\// { print }
        index($2, pol) > 0 { sub(pol, "", $2); print }
    ' "$CSV" >> "$POL_DIR/benchmark_nekrs_timesteps.csv"

    echo "  $label: $(wc -l < "$POL_DIR/benchmark_nekrs_timesteps.csv") rows"
done

echo ""
echo "============================================================"
echo "nekRS Benchmark Complete"
echo "  Results: $RESULTS_DIR"
echo "  CSV:     $CSV"
echo "============================================================"
