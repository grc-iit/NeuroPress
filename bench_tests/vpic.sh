#!/bin/bash
# ============================================================
# bench_tests/vpic.sh — Deploy VPIC plasma PIC simulation
#
# VPIC simulates magnetic reconnection in a 3-D plasma.
# Each timestep dumps 16 electromagnetic/current-density field
# components, each of size (NX+2)^3 float32 values.
#
# HDF5 MODES (HDF5_MODE):
#   default  — Plain HDF5, no compression.  The no-comp phase
#              writes uncompressed data through the VOL pipeline.
#              Use this to measure baseline raw I/O bandwidth.
#   vol      — GPUCompress HDF5 VOL.  GPU compresses each chunk
#              before it lands on disk.  Algorithm selected by
#              PHASE and POLICY below.
#
# PARAMETERS THAT AFFECT I/O VOLUME:
#   VPIC_NX          Grid dimension.  Data per snapshot =
#                    (NX+2)^3 × 16 fields × 4 bytes.
#                    NX=32   →  ~2 MB    (smoke test)
#                    NX=64   →  ~18 MB   (small, default)
#                    NX=128  →  ~141 MB  (medium)
#                    NX=200  →  ~533 MB  (SC default)
#                    NX=256  →  ~1.1 GB  (large)
#   VPIC_TIMESTEPS   Snapshots written.  Total I/O =
#                    NX volume × TIMESTEPS.
#   VPIC_SIM_INT     Physics steps between snapshots.  More steps
#                    let the plasma evolve further between writes,
#                    increasing diversity across snapshots.
#   CHUNK_MB         HDF5 chunk size.  Each chunk is compressed
#                    independently.  Smaller → more GPU parallelism;
#                    larger → better ratio on smooth regions.
#
# PARAMETERS THAT AFFECT DATA VARIETY (compressibility):
#   VPIC_MI_ME       Ion/electron mass ratio.  Lower (≥1) speeds
#                    reconnection; higher (25) produces richer fields.
#   VPIC_WPE_WCE     Plasma/cyclotron freq ratio.  Higher weakens
#                    the magnetic field and increases turbulence.
#   VPIC_TI_TE       Ion/electron temperature ratio.  Higher adds
#                    free energy, strengthening instabilities.
#   VPIC_PERTURBATION Tearing-mode seed fraction of B0.  0 = no
#                    reconnection; 0.1 = typical; 0.3 = strong.
#   VPIC_GUIDE_FIELD Out-of-plane guide field.  0 = 2-D symmetric;
#                    0.2–0.5 = full 3-D asymmetric structure.
#   VPIC_NPPC        Particles per cell.  Higher → smoother, more
#                    uniform and compressible statistics.
#   VPIC_WARMUP      Physics steps before the first snapshot.
#                    More warmup → more evolved starting state.
#
# USAGE:
#   bash bench_tests/vpic.sh
#
#   # Large run, VOL mode, ratio policy
#   VPIC_NX=200 VPIC_TIMESTEPS=50 HDF5_MODE=vol POLICY=ratio \
#       bash bench_tests/vpic.sh
#
#   # Smoke test, default HDF5
#   VPIC_NX=32 VPIC_TIMESTEPS=3 VPIC_WARMUP=50 HDF5_MODE=default \
#       bash bench_tests/vpic.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HDF5 mode ──────────────────────────────────────────────
# default : no-comp (uncompressed HDF5 baseline)
# vol     : GPUCompress HDF5 VOL
HDF5_MODE=${HDF5_MODE:-default}

# ── I/O volume parameters ──────────────────────────────────
VPIC_NX=${VPIC_NX:-64}              # Grid dim; data = (NX+2)^3*64 B/snapshot
VPIC_TIMESTEPS=${VPIC_TIMESTEPS:-3} # Snapshots to write
VPIC_SIM_INT=${VPIC_SIM_INT:-190}   # Physics steps between snapshots
CHUNK_MB=${CHUNK_MB:-4}             # HDF5 chunk size in MB
VERIFY=${VERIFY:-1}                 # 1 = bitwise readback verify

# ── Data variety / physics parameters ──────────────────────
VPIC_MI_ME=${VPIC_MI_ME:-25}        # Ion/electron mass ratio (1–400)
VPIC_WPE_WCE=${VPIC_WPE_WCE:-3}    # Plasma/cyclotron freq ratio (0.5–5)
VPIC_TI_TE=${VPIC_TI_TE:-1}        # Ion/electron temperature ratio (0.1–10)
VPIC_PERTURBATION=${VPIC_PERTURBATION:-0.1} # Tearing-mode seed (0–0.5)
VPIC_GUIDE_FIELD=${VPIC_GUIDE_FIELD:-0.0}   # Guide field (0–0.5)
VPIC_NPPC=${VPIC_NPPC:-2}           # Particles per cell
VPIC_WARMUP=${VPIC_WARMUP:-100}     # Warmup steps before first snapshot

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
VPIC_BIN="${VPIC_BIN:-$GPUC_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
VPIC_LD_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Derived ─────────────────────────────────────────────────
VPIC_DATA=$(( (VPIC_NX+2) * (VPIC_NX+2) * (VPIC_NX+2) * 64 / 1024 / 1024 ))
TOTAL_IO=$(( VPIC_DATA * VPIC_TIMESTEPS ))

case "$POLICY" in
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *)        W0=1.0; W1=1.0; W2=1.0 ;;
esac

# In default mode run no-comp only; in vol mode run the requested phase.
if [ "$HDF5_MODE" = "default" ]; then
    INCLUDE_PHASES="no-comp"
else
    INCLUDE_PHASES="$PHASE"
fi

# Build the VPIC_EXCLUDE list (all phases not in INCLUDE_PHASES)
ALL_PHASES="no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"
EXCL=""
IFS=',' read -ra _ALL <<< "$ALL_PHASES"
for p in "${_ALL[@]}"; do
    if ! echo ",$INCLUDE_PHASES," | grep -q ",$p,"; then
        EXCL="${EXCL:+$EXCL,}$p"
    fi
done

VERIFY_TAG=""; [ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
RESULTS_DIR="${RESULTS_DIR:-$GPUC_DIR/benchmarks/vpic-kokkos/results/bench_NX${VPIC_NX}_ts${VPIC_TIMESTEPS}_${HDF5_MODE}${VERIFY_TAG}}"

echo "============================================================"
echo "  VPIC Benchmark Deploy"
echo "============================================================"
echo "  HDF5 mode   : $HDF5_MODE"
echo "  NX          : $VPIC_NX  (~${VPIC_DATA} MB/snapshot, ~${TOTAL_IO} MB total)"
echo "  Timesteps   : $VPIC_TIMESTEPS  (sim interval: $VPIC_SIM_INT steps)"
echo "  Warmup      : $VPIC_WARMUP steps"
echo "  Chunk       : ${CHUNK_MB} MB"
echo "  Verify      : $VERIFY"
if [ "$HDF5_MODE" = "vol" ]; then
echo "  Phase       : $PHASE"
echo "  Policy      : $POLICY  (w0=$W0 w1=$W1 w2=$W2)"
echo "  Error bound : $ERROR_BOUND"
fi
echo ""
echo "  Physics:  MI_ME=$VPIC_MI_ME  WPE_WCE=$VPIC_WPE_WCE  TI_TE=$VPIC_TI_TE"
echo "            pert=$VPIC_PERTURBATION  guide=$VPIC_GUIDE_FIELD  nppc=$VPIC_NPPC"
echo ""
echo "  Results: $RESULTS_DIR"
echo "============================================================"
echo ""

if [ ! -f "$VPIC_BIN" ]; then
    echo "ERROR: VPIC binary not found: $VPIC_BIN"
    echo "  Build with: bash $GPUC_DIR/benchmarks/vpic-kokkos/build_vpic_pm.sh"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

LD_LIBRARY_PATH="$VPIC_LD_PATH" \
GPUCOMPRESS_DETAILED_TIMING=1 \
GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
VPIC_NX=$VPIC_NX \
VPIC_NPPC=$VPIC_NPPC \
VPIC_MI_ME=$VPIC_MI_ME \
VPIC_WPE_WCE=$VPIC_WPE_WCE \
VPIC_TI_TE=$VPIC_TI_TE \
VPIC_PERTURBATION=$VPIC_PERTURBATION \
VPIC_GUIDE_FIELD=$VPIC_GUIDE_FIELD \
VPIC_ERROR_BOUND=$ERROR_BOUND \
VPIC_WARMUP_STEPS=$VPIC_WARMUP \
VPIC_TIMESTEPS=$VPIC_TIMESTEPS \
VPIC_SIM_INTERVAL=$VPIC_SIM_INT \
VPIC_CHUNK_MB=$CHUNK_MB \
VPIC_VERIFY=$VERIFY \
VPIC_EXCLUDE="$EXCL" \
VPIC_RESULTS_DIR="$RESULTS_DIR" \
VPIC_POLICIES="$POLICY" \
VPIC_LR=0.2 \
VPIC_MAPE_THRESHOLD=0.10 \
VPIC_EXPLORE_K=4 \
VPIC_EXPLORE_THRESH=0.20 \
    "$VPIC_BIN" \
    > "$RESULTS_DIR/vpic_bench.log" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ] || grep -q "normal exit" "$RESULTS_DIR/vpic_bench.log" 2>/dev/null; then
    echo "PASS  Log: $RESULTS_DIR/vpic_bench.log"
else
    echo "FAIL  Log: $RESULTS_DIR/vpic_bench.log"
    tail -20 "$RESULTS_DIR/vpic_bench.log"
    exit 1
fi
