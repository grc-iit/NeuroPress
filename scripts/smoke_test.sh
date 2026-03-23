#!/bin/bash
# ============================================================
# Smoke Test: Run all benchmark drivers with all phases on small configs.
#
# Runs Gray-Scott, SDRBench (Hurricane, Nyx, CESM-ATM), and VPIC.
#
# Usage:
#   bash scripts/smoke_test.sh
#
# Output:
#   benchmarks/grayscott/results/smoke_L128/
#   benchmarks/sdrbench/results/smoke/
#   benchmarks/vpic-kokkos/results/
#   benchmarks/results/smoke_plots/
# ============================================================
set +e

GPU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GPU_DIR"

GS_BIN="$GPU_DIR/build/grayscott_benchmark"
SDR_BIN="$GPU_DIR/build/generic_benchmark"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_BIN="$GPU_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
VPIC_DECK="$GPU_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx"
VPIC_LD_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib:/opt/cray/libfabric/1.22.0/lib64"

# ── Create output dirs ──
mkdir -p benchmarks/grayscott/results/smoke_L128
mkdir -p benchmarks/sdrbench/results/smoke
mkdir -p benchmarks/vpic-kokkos/results
mkdir -p benchmarks/results/smoke_plots

PASS=0
FAIL=0
SKIP=0

run_test() {
    local NAME="$1"
    local LOG="$2"
    shift 2
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $NAME"
    echo "════════════════════════════════════════════════════════════"
    if eval "$@" > "$LOG" 2>&1; then
        if grep -q "PASSED" "$LOG" 2>/dev/null; then
            echo "  ✓ $NAME — PASSED"
            PASS=$((PASS + 1))
        else
            echo "  ✓ $NAME — completed (check $LOG)"
            PASS=$((PASS + 1))
        fi
    else
        echo "  ✗ $NAME — FAILED (check $LOG)"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Gray-Scott: all phases ──
run_test "Gray-Scott (all phases, L=128, ts=5)" \
    "benchmarks/grayscott/results/smoke_L128/gs_smoke.log" \
    "$GS_BIN" "$WEIGHTS" \
        --L 128 --steps 100 --chunk-mb 4 --timesteps 5 \
        --out-dir benchmarks/grayscott/results/smoke_L128

# ── 2. SDRBench: Hurricane Isabel ──
HURRICANE_DIR="$GPU_DIR/data/sdrbench/hurricane_isabel/100x500x500"
if [ -d "$HURRICANE_DIR" ]; then
    run_test "SDRBench Hurricane Isabel (all phases)" \
        "benchmarks/sdrbench/results/smoke/hurricane_smoke.log" \
        "$SDR_BIN" "$WEIGHTS" \
            --data-dir "$HURRICANE_DIR" \
            --dims 100,500,500 --ext .bin.f32 --runs 1 \
            --out-dir benchmarks/sdrbench/results/smoke
else
    echo "  SKIP Hurricane Isabel — data not found"
    SKIP=$((SKIP + 1))
fi

# ── 3. SDRBench: Nyx ──
NYX_DIR="$GPU_DIR/data/sdrbench/nyx/SDRBENCH-EXASKY-NYX-512x512x512"
if [ -d "$NYX_DIR" ]; then
    run_test "SDRBench Nyx (all phases)" \
        "benchmarks/sdrbench/results/smoke/nyx_smoke.log" \
        "$SDR_BIN" "$WEIGHTS" \
            --data-dir "$NYX_DIR" \
            --dims 512,512,512 --ext .f32 --runs 1 \
            --out-dir benchmarks/sdrbench/results/smoke
else
    echo "  SKIP Nyx — data not found"
    SKIP=$((SKIP + 1))
fi

# ── 4. SDRBench: CESM-ATM ──
CESM_DIR="$GPU_DIR/data/sdrbench/cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600"
if [ -d "$CESM_DIR" ]; then
    run_test "SDRBench CESM-ATM (all phases)" \
        "benchmarks/sdrbench/results/smoke/cesm_smoke.log" \
        "$SDR_BIN" "$WEIGHTS" \
            --data-dir "$CESM_DIR" \
            --dims 1800,3600 --ext .dat --runs 1 \
            --out-dir benchmarks/sdrbench/results/smoke
else
    echo "  SKIP CESM-ATM — data not found"
    SKIP=$((SKIP + 1))
fi

# ── 5. VPIC ──
if [ -f "$VPIC_BIN" ] && [ -f /tmp/hdf5-install/lib/libhdf5.so ]; then
    # Only restore JIT wrapper if the binary is not already a valid ELF executable
    # (build_vpic_benchmark.sh produces a pre-linked binary that should not be overwritten)
    if ! file "$VPIC_BIN" | grep -q "ELF"; then
        VPIC_WRAPPER="${VPIC_DIR:-$HOME/vpic-kokkos}/build-compress/bin/vpic"
        if [ -f "$VPIC_WRAPPER" ]; then
            cp "$VPIC_WRAPPER" "$VPIC_BIN"
        fi
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  VPIC (all phases, NX=64)"
    echo "════════════════════════════════════════════════════════════"
    VPIC_LOG="benchmarks/vpic-kokkos/results/smoke_vpic.log"
    cd "$GPU_DIR/benchmarks/vpic-kokkos"
    LD_LIBRARY_PATH="$VPIC_LD_PATH" \
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    VPIC_NX=64 VPIC_CHUNK_MB=4 VPIC_RUNS=1 \
    ./vpic_benchmark_deck.Linux vpic_benchmark_deck.cxx \
        > "$GPU_DIR/$VPIC_LOG" 2>&1
    if [ $? -eq 0 ] || grep -q "normal exit" "$GPU_DIR/$VPIC_LOG" 2>/dev/null; then
        echo "  ✓ VPIC — PASSED"
        PASS=$((PASS + 1))
    else
        echo "  ✗ VPIC — FAILED (check $VPIC_LOG)"
        FAIL=$((FAIL + 1))
    fi
    cd "$GPU_DIR"
else
    echo "  SKIP VPIC — binary or HDF5 not found"
    echo "  Run: bash scripts/rebuild_vpic.sh"
    SKIP=$((SKIP + 1))
fi

# ── 6. Generate plots ──
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Generating plots"
echo "════════════════════════════════════════════════════════════"
python3 benchmarks/visualize.py \
    --gs-dir benchmarks/grayscott/results/smoke_L128 \
    --vpic-dir benchmarks/vpic-kokkos/results \
    --sdrbench-dir benchmarks/sdrbench/results/smoke \
    2>&1 || echo "  WARNING: visualizer had errors"

# ── Summary ──
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Smoke Test Summary"
echo "════════════════════════════════════════════════════════════"
echo "  PASSED:  $PASS"
echo "  FAILED:  $FAIL"
echo "  SKIPPED: $SKIP"
echo ""
echo "  Logs:"
echo "    benchmarks/grayscott/results/smoke_L128/*.log"
echo "    benchmarks/sdrbench/results/smoke/*.log"
echo "    benchmarks/vpic-kokkos/results/smoke_vpic.log"
echo ""
echo "  Plots:"
echo "    benchmarks/results/smoke_plots/"
echo "════════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
