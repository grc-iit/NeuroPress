#!/bin/bash
# ============================================================
# bench_tests/lammps.sh — Deploy LAMMPS molecular dynamics
#
# Simulates an FCC Lennard-Jones lattice with a hot sphere
# expanding into a cold lattice.  At each dump interval LAMMPS
# writes three field arrays (positions, velocities, forces) as
# float32.  Data per dump = 4 * ATOMS^3 atoms × 3 xyz × 3 fields
# × 4 bytes = 4 * ATOMS^3 × 36 bytes.
#
# HDF5 MODES (HDF5_MODE):
#   default  — No compression.  The GPUCompress fix is still
#              active but with GPUCOMPRESS_ALGO=lz4 while the
#              result is reported as uncompressed (ratio=1).
#              This matches how run_lammps_benchmark.sh handles
#              the no-comp baseline.  Use this to measure raw
#              I/O bandwidth through the same code path.
#   vol      — GPUCompress HDF5 VOL.  GPU compresses each chunk
#              with the algorithm specified by PHASE.
#
# PARAMETERS THAT AFFECT I/O VOLUME:
#   LMP_ATOMS        Box edge in FCC unit cells (4 atoms/cell).
#                    Total atoms = 4 * ATOMS^3.
#                    ATOMS=20  →  ~3 MB/dump   (smoke test)
#                    ATOMS=40  →  ~27 MB/dump  (small, default)
#                    ATOMS=80  →  ~216 MB/dump (medium)
#                    ATOMS=120 →  ~730 MB/dump (large)
#   TIMESTEPS        Number of field dumps.  Total I/O =
#                    data_per_dump × TIMESTEPS.
#   SIM_INTERVAL     MD steps between dumps.  More steps allow
#                    the shock front to travel further, increasing
#                    structural diversity across dumps.
#   CHUNK_MB         HDF5 chunk size in MB.
#
# PARAMETERS THAT AFFECT DATA VARIETY (compressibility):
#   T_HOT            Temperature of the hot sphere (LJ units).
#                    Higher → sharper velocity gradients at the
#                    hot/cold interface, less compressible.
#   T_COLD           Cold lattice temperature.  Lower → more
#                    uniform cold region, more compressible.
#   HOT_RADIUS_FRAC  Hot sphere radius as a fraction of the box
#                    half-width.  Larger → more atoms in the
#                    turbulent region, lower overall ratio.
#   SIM_INTERVAL     More MD steps → shock travels further,
#                    increasing diversity across dumps.
#
# USAGE:
#   bash bench_tests/lammps.sh
#
#   # Large run, VOL mode, ratio policy
#   LMP_ATOMS=120 TIMESTEPS=20 HDF5_MODE=vol POLICY=ratio \
#       bash bench_tests/lammps.sh
#
#   # Smoke test, default HDF5
#   LMP_ATOMS=20 TIMESTEPS=3 HDF5_MODE=default \
#       bash bench_tests/lammps.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HDF5 mode ──────────────────────────────────────────────
# default : no-comp baseline (algo=lz4, ratio reported as 1.0)
# vol     : GPUCompress HDF5 VOL
HDF5_MODE=${HDF5_MODE:-default}

# ── I/O volume parameters ──────────────────────────────────
LMP_ATOMS=${LMP_ATOMS:-40}          # Box edge in unit cells; atoms=4*ATOMS^3
TIMESTEPS=${TIMESTEPS:-3}           # Number of field dumps
SIM_INTERVAL=${SIM_INTERVAL:-50}    # MD steps between dumps
CHUNK_MB=${CHUNK_MB:-4}             # HDF5 chunk size in MB
VERIFY=${VERIFY:-1}                 # 1 = bitwise readback verify

# ── Data variety / physics parameters ──────────────────────
T_HOT=${T_HOT:-10.0}               # Hot-sphere temperature (LJ units)
T_COLD=${T_COLD:-0.01}             # Cold-lattice temperature (LJ units)
# HOT_RADIUS_FRAC: radius = int(ATOMS/2 * frac)
HOT_RADIUS_FRAC=${HOT_RADIUS_FRAC:-0.25}
WARMUP_STEPS=${WARMUP_STEPS:-20}    # MD steps before first dump

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
ERROR_BOUND=${ERROR_BOUND:-0.0}     # Lossy error bound (0.0 = lossless)

# ── Paths ───────────────────────────────────────────────────
LMP_BIN="${LMP_BIN:-${SIMS_DIR:-$HOME/sims}/lammps/build/lmp}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Derived ─────────────────────────────────────────────────
NATOMS=$(python3 -c "print(4 * $LMP_ATOMS**3)")
DATA_MB=$(python3 -c "print(f'{4 * $LMP_ATOMS**3 * 36 / 1048576:.1f}')")
TOTAL_STEPS=$(( WARMUP_STEPS + TIMESTEPS * SIM_INTERVAL ))
HOT_RADIUS=$(python3 -c "r=int($LMP_ATOMS/2 * $HOT_RADIUS_FRAC); print(max(1, r))")

# Map PHASE to GPUCOMPRESS_ALGO string.
# no-comp / default mode: algo=lz4 (same as run_lammps_benchmark.sh no-comp branch).
if [ "$HDF5_MODE" = "default" ]; then
    GPUC_ALGO="lz4"
    PHASE_LABEL="no-comp"
else
    GPUC_ALGO="$PHASE"
    PHASE_LABEL="$PHASE"
fi

case "$POLICY" in
    ratio)    GPUC_POLICY="ratio" ;;
    speed)    GPUC_POLICY="speed" ;;
    *)        GPUC_POLICY="balanced" ;;
esac

VERIFY_TAG=""; [ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
RESULTS_DIR="${RESULTS_DIR:-$GPUC_DIR/benchmarks/lammps/results/bench_box${LMP_ATOMS}_ts${TIMESTEPS}_${HDF5_MODE}${VERIFY_TAG}}"

echo "============================================================"
echo "  LAMMPS Benchmark Deploy"
echo "============================================================"
echo "  HDF5 mode   : $HDF5_MODE"
echo "  Box         : ${LMP_ATOMS}^3 cells (~$NATOMS atoms)"
echo "  Data/dump   : ~${DATA_MB} MB (positions + velocities + forces)"
echo "  Timesteps   : $TIMESTEPS  (every $SIM_INTERVAL MD steps)"
echo "  Total steps : $TOTAL_STEPS  (warmup: $WARMUP_STEPS)"
echo "  Chunk       : ${CHUNK_MB} MB"
echo "  Verify      : $VERIFY"
echo ""
echo "  Physics:  T_hot=$T_HOT  T_cold=$T_COLD  hot_radius=$HOT_RADIUS"
if [ "$HDF5_MODE" = "vol" ]; then
echo ""
echo "  Phase/algo  : $PHASE_LABEL ($GPUC_ALGO)"
echo "  Policy      : $POLICY"
echo "  Error bound : $ERROR_BOUND"
fi
echo ""
echo "  Results: $RESULTS_DIR"
echo "============================================================"
echo ""

if [ ! -x "$LMP_BIN" ]; then
    echo "ERROR: LAMMPS binary not found: $LMP_BIN"
    echo "  Build with: bash $GPUC_DIR/benchmarks/lammps/build_lammps.sh"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
WORKDIR="$RESULTS_DIR/work_${PHASE_LABEL}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Generate LAMMPS input script
cat > input.lmp << EOF
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 $LMP_ATOMS 0 $LMP_ATOMS 0 $LMP_ATOMS
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere $(( LMP_ATOMS/2 )) $(( LMP_ATOMS/2 )) $(( LMP_ATOMS/2 )) ${HOT_RADIUS}
group           hot region hot
group           cold subtract all hot
velocity        cold create ${T_COLD} 87287 loop geom
velocity        hot  create ${T_HOT}  12345 loop geom

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

CSV="$RESULTS_DIR/benchmark_lammps_timesteps.csv"
echo "rank,phase,timestep,write_ms,ratio,orig_mb,comp_mb,verify" > "$CSV"

GPUCOMPRESS_ALGO="$GPUC_ALGO" \
GPUCOMPRESS_VERIFY="$VERIFY" \
GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
GPUCOMPRESS_POLICY="$GPUC_POLICY" \
GPUCOMPRESS_ERROR_BOUND="$ERROR_BOUND" \
GPUCOMPRESS_CHUNK_MB="$CHUNK_MB" \
GPUCOMPRESS_DETAILED_TIMING=1 \
    "$LMP_BIN" -k on g 1 -sf kk -in input.lmp \
    > lammps.log 2>&1
STATUS=$?

# Parse results from gpuc_step_* output directories
NATOMS_ACTUAL=$(grep "Created.*atoms$" lammps.log | head -1 | awk '{print $2}')
[ -z "$NATOMS_ACTUAL" ] && NATOMS_ACTUAL=$(( 4 * LMP_ATOMS**3 ))
ORIG_BYTES=$(( NATOMS_ACTUAL * 36 ))
ORIG_MB=$(python3 -c "print(f'{$ORIG_BYTES/1048576:.1f}')")

TOTAL_MS=0
[ -f lammps.log ] && TOTAL_MS=$(( $(date +%s%N) ))  # approximate; LAMMPS logs timing

ts=0
for d in $(ls -d gpuc_step_* 2>/dev/null | sort); do
    STEP_NUM=$(echo "$d" | sed 's/gpuc_step_0*//')
    [ -z "$STEP_NUM" ] && STEP_NUM=0
    [ "$STEP_NUM" -lt "$WARMUP_STEPS" ] && continue

    COMP_BYTES=$(du -sb "$d" | awk '{print $1}')
    COMP_MB=$(python3 -c "print(f'{$COMP_BYTES/1048576:.2f}')")

    if [ "$HDF5_MODE" = "default" ]; then
        RATIO="1.00"
        COMP_MB="$ORIG_MB"
    else
        RATIO=$(python3 -c "print(f'{$ORIG_BYTES/$COMP_BYTES:.2f}')" 2>/dev/null || echo "0")
    fi

    VERIFY_OK=1
    grep -q "VERIFY FAILED" lammps.log 2>/dev/null && VERIFY_OK=0

    echo "0,$PHASE_LABEL,$ts,0,$RATIO,$ORIG_MB,$COMP_MB,$VERIFY_OK" >> "$CSV"
    ts=$(( ts + 1 ))
done

cd "$GPUC_DIR"

if [ $STATUS -eq 0 ]; then
    echo "PASS  ($ts dumps recorded)  Log: $WORKDIR/lammps.log"
else
    echo "FAIL (exit $STATUS)  Log: $WORKDIR/lammps.log"
    tail -20 "$WORKDIR/lammps.log"
    exit 1
fi
