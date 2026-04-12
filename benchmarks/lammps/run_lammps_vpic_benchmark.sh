#!/bin/bash
# ============================================================
# LAMMPS VPIC-Compatible Benchmark
#
# Two-phase approach for apple-to-apple compression comparison:
#   Phase 1: Run LAMMPS once with LAMMPS_DUMP_FIELDS=1 to dump
#            raw .f32 field files (positions, velocities, forces)
#   Phase 2: Run generic_benchmark on each policy's fields to
#            sweep all 12 compression phases with weight isolation
#
# This produces VPIC-compatible CSV output with 52 columns per
# timestep, aggregate summary, per-chunk diagnostics, and
# ranking profiler data.
#
# Data evolution: hot sphere expansion (velocity hot=10.0 vs
# cold=0.01) creates evolving density front across timesteps.
#
# Usage:
#   bash benchmarks/lammps/run_lammps_vpic_benchmark.sh
#
# Environment variables:
#   LMP_BIN          Path to lmp binary [$HOME/lammps/build/lmp]
#   LMP_ATOMS        Box size per dimension [80 -> ~2M atoms]
#   CHUNK_MB         HDF5 chunk size [4]
#   TIMESTEPS        Benchmark write cycles [10]
#   SIM_INTERVAL     Physics steps between dumps [50]
#   WARMUP_STEPS     Physics steps before first dump [100]
#   POLICIES         NN policies [balanced,ratio,speed]
#   ERROR_BOUND      Lossy error bound [0.0 = lossless]
#   RESULTS_DIR      Output directory [auto]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ──
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
LMP_ATOMS=${LMP_ATOMS:-80}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-10}
SIM_INTERVAL=${SIM_INTERVAL:-50}
WARMUP_STEPS=${WARMUP_STEPS:-100}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
ERROR_BOUND=${ERROR_BOUND:-0.0}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
SGD_LR="${SGD_LR:-0.2}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"

# Derived
TOTAL_STEPS=$((WARMUP_STEPS + TIMESTEPS * SIM_INTERVAL))
NATOMS_APPROX=$(python3 -c "print(4 * $LMP_ATOMS**3)" 2>/dev/null || echo "unknown")
DATA_MB=$(python3 -c "print(f'{4 * $LMP_ATOMS**3 * 12 / 1048576:.1f}')" 2>/dev/null || echo "unknown")
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/lammps_eval_box${LMP_ATOMS}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "LAMMPS VPIC-Compatible Benchmark"
echo "============================================================"
echo "  Binary:        $LMP_BIN"
echo "  Generic BM:    $GENERIC_BIN"
echo "  Box:           ${LMP_ATOMS}^3 (~$NATOMS_APPROX atoms)"
echo "  Data/field:    ~$DATA_MB MB per field (fp32)"
echo "  Warmup:        $WARMUP_STEPS steps"
echo "  Timesteps:     $TIMESTEPS (every $SIM_INTERVAL sim steps)"
echo "  Total steps:   $TOTAL_STEPS"
echo "  Chunk:         ${CHUNK_MB} MB"
echo "  Policies:      $POLICIES"
echo "  Error bound:   $ERROR_BOUND"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Check binaries ──
if [ ! -x "$LMP_BIN" ]; then
    echo "ERROR: LAMMPS binary not found: $LMP_BIN"
    exit 1
fi
if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    echo "  Build with: cmake --build build --target generic_benchmark"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

# ============================================================
# Phase 1: Run LAMMPS once with raw field dumping
# ============================================================
echo ">>> Phase 1: Run LAMMPS simulation with raw field dumping"
echo ""

RAW_DIR="$RESULTS_DIR/raw_fields"
mkdir -p "$RAW_DIR"

# Generate LAMMPS input script
# Hot sphere expanding into cold lattice for data diversity
INPUT_FILE="$RESULTS_DIR/input.lmp"
cat > "$INPUT_FILE" << EOF
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
fix             gpuc all gpucompress ${SIM_INTERVAL} positions velocities forces
thermo          ${SIM_INTERVAL}
timestep        0.003
run             ${TOTAL_STEPS}
EOF

# Run LAMMPS with field dumping enabled, minimal compression (just need raw dumps)
echo "  Running LAMMPS: $TOTAL_STEPS steps, dumping fields every $SIM_INTERVAL steps"
export LAMMPS_DUMP_FIELDS=1
export LAMMPS_DUMP_DIR="$RAW_DIR"
export GPUCOMPRESS_ALGO="lz4"
export GPUCOMPRESS_VERIFY=0
export GPUCOMPRESS_WEIGHTS="$WEIGHTS"
export GPUCOMPRESS_POLICY="ratio"
export GPUCOMPRESS_SGD=0
export GPUCOMPRESS_EXPLORE=0

cd "$RESULTS_DIR"
"$LMP_BIN" -k on g 1 -sf kk -in "$INPUT_FILE" > lammps_dump.log 2>&1 || {
    echo "  WARNING: LAMMPS exited with error (check lammps_dump.log)"
}
cd "$GPUC_DIR"

# Count dumped fields
N_POS=$(ls "$RAW_DIR"/positions_step*.f32 2>/dev/null | wc -l)
N_VEL=$(ls "$RAW_DIR"/velocities_step*.f32 2>/dev/null | wc -l)
N_FOR=$(ls "$RAW_DIR"/forces_step*.f32 2>/dev/null | wc -l)
echo "  Dumped: $N_POS position files, $N_VEL velocity files, $N_FOR force files"

if [ "$N_POS" -eq 0 ]; then
    echo "ERROR: No field files dumped. Check lammps_dump.log"
    exit 1
fi

# Determine field dimensions from file size
# LAMMPS dumps nlocal*3 floats per field = nlocal*12 bytes
FIRST_FILE=$(ls "$RAW_DIR"/positions_step*.f32 2>/dev/null | head -1)
FIRST_SIZE=$(stat -c%s "$FIRST_FILE")
N_FLOATS=$((FIRST_SIZE / 4))
# generic_benchmark requires >=2D dims; use N,1 for 1D LAMMPS data
DIMS="${N_FLOATS},1"
echo "  Field size: $N_FLOATS floats ($((FIRST_SIZE / 1024 / 1024)) MB), dims=$DIMS"

# Filter to post-warmup fields only
# Warmup steps produce dumps we don't want to benchmark
echo "  Filtering to post-warmup fields (step >= $WARMUP_STEPS)..."
BENCH_DIR="$RESULTS_DIR/bench_fields"
for field_type in positions velocities forces; do
    FIELD_DIR="$BENCH_DIR/$field_type"
    mkdir -p "$FIELD_DIR"
    for f in "$RAW_DIR"/${field_type}_step*.f32; do
        [ -f "$f" ] || continue
        step=$(echo "$(basename "$f")" | sed "s/${field_type}_step0*//" | sed 's/\.f32//')
        [ -z "$step" ] && step=0
        if [ "$step" -ge "$WARMUP_STEPS" ]; then
            ln -sf "$f" "$FIELD_DIR/$(basename "$f")"
        fi
    done
    N_BENCH=$(ls "$FIELD_DIR"/*.f32 2>/dev/null | wc -l)
    echo "    $field_type: $N_BENCH benchmark files"
done

echo ""

# ============================================================
# Phase 2: Run generic_benchmark per field type and policy
# ============================================================
echo ">>> Phase 2: Run multi-phase benchmark sweep"
echo ""

# ── Policy weights ──
declare -A POL_W0 POL_W1 POL_W2
POL_W0[balanced]=1.0;  POL_W1[balanced]=1.0;  POL_W2[balanced]=1.0
POL_W0[ratio]=0.0;     POL_W1[ratio]=0.0;     POL_W2[ratio]=1.0
POL_W0[speed]=1.0;     POL_W1[speed]=1.0;     POL_W2[speed]=0.0

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"

for policy in "${POL_ARRAY[@]}"; do
    w0=${POL_W0[$policy]}
    w1=${POL_W1[$policy]}
    w2=${POL_W2[$policy]}

    echo "  ── Policy: $policy (w0=$w0, w1=$w1, w2=$w2) ──"

    for field_type in positions velocities forces; do
        FIELD_DIR="$BENCH_DIR/$field_type"
        OUT_DIR="$RESULTS_DIR/${policy}/${field_type}"
        mkdir -p "$OUT_DIR"

        N_FILES=$(ls "$FIELD_DIR"/*.f32 2>/dev/null | wc -l)
        [ "$N_FILES" -eq 0 ] && continue

        echo "    Running generic_benchmark on $field_type ($N_FILES files)..."
        "$GENERIC_BIN" "$WEIGHTS" \
            --data-dir "$FIELD_DIR" \
            --dims "$DIMS" \
            --ext .f32 \
            --chunk-mb "$CHUNK_MB" \
            --out-dir "$OUT_DIR" \
            --name "lammps_${field_type}" \
            --w0 "$w0" --w1 "$w1" --w2 "$w2" \
            --lr "$SGD_LR" \
            --mape "$SGD_MAPE" \
            --explore-k "$EXPLORE_K" \
            --explore-thresh "$EXPLORE_THRESH" \
            ${ERROR_BOUND:+--error-bound "$ERROR_BOUND"} \
            > "$OUT_DIR/benchmark.log" 2>&1 || {
                echo "      WARNING: benchmark exited with error"
            }

        # Report summary
        if [ -f "$OUT_DIR/benchmark_lammps_${field_type}.csv" ]; then
            echo "      Summary:"
            tail -n +2 "$OUT_DIR/benchmark_lammps_${field_type}.csv" | \
                awk -F, '{ printf "        %-20s ratio=%-6s write=%-8s\n", $3, $11, $5 }'
        fi
    done
    echo ""
done

# ============================================================
# Combine results across field types per policy
# ============================================================
echo ">>> Combining results across field types..."

for policy in "${POL_ARRAY[@]}"; do
    COMBINED="$RESULTS_DIR/${policy}/benchmark_lammps_combined.csv"
    HEADER_DONE=0
    for field_type in positions velocities forces; do
        CSV="$RESULTS_DIR/${policy}/${field_type}/benchmark_lammps_${field_type}.csv"
        [ -f "$CSV" ] || continue
        if [ "$HEADER_DONE" -eq 0 ]; then
            head -1 "$CSV" > "$COMBINED"
            HEADER_DONE=1
        fi
        tail -n +2 "$CSV" >> "$COMBINED"
    done
    if [ -f "$COMBINED" ]; then
        N_ROWS=$(($(wc -l < "$COMBINED") - 1))
        echo "  $policy: $N_ROWS rows in $COMBINED"
    fi
done

echo ""

# ============================================================
# Phase 3: Generate figures per policy per field type
# ============================================================
echo ">>> Phase 3: Generating figures"

VISUALIZER="$GPUC_DIR/benchmarks/visualize.py"
if [ ! -f "$VISUALIZER" ]; then
    echo "WARNING: visualize.py not found at $VISUALIZER — skipping figures"
else
    for policy in "${POL_ARRAY[@]}"; do
        for field_type in positions velocities forces; do
            FIELD_DIR="$RESULTS_DIR/${policy}/${field_type}"
            [ -d "$FIELD_DIR" ] || continue
            FIG_DIR="$FIELD_DIR/figures"

            echo "  ── $policy / $field_type ──"

            # Symlink CSVs to vpic_deck naming so visualize.py can find them
            for src in "$FIELD_DIR"/benchmark_lammps_${field_type}*.csv; do
                [ -f "$src" ] || continue
                dst=$(basename "$src" | sed "s/benchmark_lammps_${field_type}/benchmark_vpic_deck/")
                ln -sf "$src" "$FIELD_DIR/$dst"
            done

            python3 "$VISUALIZER" \
                --vpic-dir "$FIELD_DIR" \
                --output-dir "$FIG_DIR" \
                2>&1 | grep -E "Saved:|Done" || true
            N_FIGS=$(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l)
            echo "    $N_FIGS figures in $FIG_DIR/"
        done
    done
fi

echo ""
echo "============================================================"
echo "LAMMPS VPIC-Compatible Benchmark Complete"
echo "  Results:   $RESULTS_DIR"
echo "  Per-policy: ${POL_ARRAY[*]}"
echo "============================================================"
