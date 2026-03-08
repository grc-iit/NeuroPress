#!/bin/bash
# ============================================================
# Build and run the simulation-based VPIC benchmark
#
# Prerequisites:
#   - VPIC-Kokkos source at ~/vpic-kokkos
#   - GPUCompress built at ~/GPUCompress/build
#   - HDF5 2.x at /tmp/hdf5-install
#   - nvcomp at /tmp/lib
#
# Usage:
#   ./benchmarks/vpic/run_sim_benchmark.sh [--build-only] [--run-only]
#
# Runtime environment variables (optional):
#   GPUCOMPRESS_SGD_LR          SGD learning rate       (default: 0.5)
#   GPUCOMPRESS_SGD_MAPE        MAPE threshold          (default: 0.25)
#   GPUCOMPRESS_DIAG_INTERVAL   Diagnostic interval     (default: 34)
#   GPUCOMPRESS_NUM_STEPS       Total simulation steps  (default: 1000)
#   GPUCOMPRESS_CHUNK_MB        Chunk size in MB        (default: 8)
# ============================================================
set -euo pipefail

GPUCOMPRESS_DIR="${GPUCOMPRESS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VPIC_DIR="${VPIC_DIR:-$HOME/vpic-kokkos}"
HDF5_PREFIX="${HDF5_PREFIX:-/tmp/hdf5-install}"
NVCOMP_LIB="${NVCOMP_LIB:-/tmp/lib}"
BUILD_DIR="${VPIC_DIR}/build-sim-benchmark"
DECK_SRC="${GPUCOMPRESS_DIR}/benchmarks/vpic/benchmark_vpic_sim.cxx"
WEIGHTS="${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt"

BUILD_ONLY=0
RUN_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=1 ;;
        --run-only)   RUN_ONLY=1 ;;
    esac
done

# ── Build ──────────────────────────────────────────────────────
if [ "$RUN_ONLY" -eq 0 ]; then
    echo "=== Building simulation benchmark ==="
    echo "  VPIC source : $VPIC_DIR"
    echo "  Deck        : $DECK_SRC"
    echo "  Build dir   : $BUILD_DIR"
    echo ""

    cmake -S "$VPIC_DIR" -B "$BUILD_DIR" \
        -DENABLE_KOKKOS_CUDA=ON \
        -DBUILD_INTERNAL_KOKKOS=ON \
        -DENABLE_KOKKOS_OPENMP=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DKokkos_ARCH_AMPERE80=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DUSER_DECK="$DECK_SRC" \
        -DUSER_DECKS="$DECK_SRC" \
        -DCMAKE_CXX_FLAGS="-I${GPUCOMPRESS_DIR}/include -I${GPUCOMPRESS_DIR}/examples -I${HDF5_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${GPUCOMPRESS_DIR}/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L${HDF5_PREFIX}/lib -lhdf5 -L${NVCOMP_LIB} -lnvcomp"

    cmake --build "$BUILD_DIR" -j"$(nproc)"
    echo ""
    echo "Build complete: $BUILD_DIR/benchmark_vpic_sim"
fi

# ── Run ────────────────────────────────────────────────────────
if [ "$BUILD_ONLY" -eq 0 ]; then
    BINARY="$BUILD_DIR/benchmark_vpic_sim"
    if [ ! -f "$BINARY" ]; then
        echo "ERROR: $BINARY not found. Run with --build-only first."
        exit 1
    fi

    echo "=== Running simulation benchmark ==="
    export LD_LIBRARY_PATH="${GPUCOMPRESS_DIR}/build:${NVCOMP_LIB}:${HDF5_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export GPUCOMPRESS_WEIGHTS="$WEIGHTS"

    # Print runtime config
    echo "  SGD LR       : ${GPUCOMPRESS_SGD_LR:-0.5 (default)}"
    echo "  SGD MAPE     : ${GPUCOMPRESS_SGD_MAPE:-0.25 (default)}"
    echo "  Diag interval: ${GPUCOMPRESS_DIAG_INTERVAL:-34 (default)}"
    echo "  Num steps    : ${GPUCOMPRESS_NUM_STEPS:-1000 (default)}"
    echo "  Chunk MB     : ${GPUCOMPRESS_CHUNK_MB:-8 (default)}"
    echo ""

    cd "$GPUCOMPRESS_DIR"
    mpirun -np 1 "$BINARY"

    echo ""
    echo "=== Results ==="
    echo "  Timestep CSV : benchmarks/vpic/sim_timestep_metrics.csv"
    echo "  Chunk CSV    : benchmarks/vpic/sim_chunk_metrics.csv"
    echo ""
    echo "To visualize:"
    echo "  python3 benchmarks/vpic/visualize_vpic_sim.py"
fi
