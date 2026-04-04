#!/bin/bash
# ============================================================
# Rebuild VPIC benchmark deck for GPUCompress
#
# Must be run on the GPU node after install_dependencies.sh.
# Restores the JIT wrapper script and recompiles the deck
# against the current HDF5/nvcomp/GPUCompress libraries.
#
# Usage:
#   bash scripts/rebuild_vpic.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VPIC_BUILD="/u/imuradli/vpic-kokkos/build-compress"
VPIC_BENCH="$GPU_DIR/benchmarks/vpic-kokkos"
VPIC_WRAPPER="$VPIC_BUILD/bin/vpic"
VPIC_BIN="$VPIC_BENCH/vpic_benchmark_deck.Linux"
VPIC_DECK="$VPIC_BENCH/vpic_benchmark_deck.cxx"

# ── Verify dependencies ──
echo "Checking dependencies..."

if [ ! -f /u/imuradli/GPUCompress/.deps/hdf5/lib/libhdf5.so ]; then
    echo "ERROR: HDF5 not found at /u/imuradli/GPUCompress/.deps/hdf5/lib/"
    echo "Run: bash scripts/install_dependencies.sh"
    exit 1
fi

if [ ! -d "$GPU_DIR/.deps/lib" ] || [ ! -f "$GPU_DIR/.deps/include/nvcomp.hpp" ]; then
    echo "ERROR: nvcomp not found at $GPU_DIR/.deps/lib/"
    echo "Run: bash scripts/install_dependencies.sh"
    exit 1
fi

if [ ! -f "$GPU_DIR/build/libgpucompress.so" ]; then
    echo "ERROR: libgpucompress.so not found. Rebuilding..."
    cd "$GPU_DIR/build" && cmake .. && cmake --build . -j$(nproc)
fi

if [ ! -f "$VPIC_WRAPPER" ]; then
    echo "ERROR: VPIC JIT wrapper not found at $VPIC_WRAPPER"
    echo "The VPIC-Kokkos build directory may be missing."
    echo "Rebuild VPIC-Kokkos first (see cmake/VpicIntegration.cmake)"
    exit 1
fi

# ── Restore JIT wrapper (overwritten by previous JIT compilation) ──
echo "Restoring JIT wrapper script..."
cp "$VPIC_WRAPPER" "$VPIC_BIN"

# ── Symlink bridge header if missing ──
if [ ! -f "$VPIC_BENCH/vpic_kokkos_bridge.hpp" ]; then
    ln -sf "$GPU_DIR/examples/vpic_kokkos_bridge.hpp" "$VPIC_BENCH/vpic_kokkos_bridge.hpp"
fi

# ── JIT compile the deck ──
echo "JIT compiling VPIC benchmark deck..."
cd "$VPIC_BENCH"
LD_LIBRARY_PATH=/u/imuradli/GPUCompress/.deps/hdf5/lib:$GPU_DIR/build:$GPU_DIR/.deps/lib:/opt/cray/libfabric/1.22.0/lib64 \
"$VPIC_BIN" "$VPIC_DECK" > /dev/null 2>&1

# ── Verify the compiled binary links correctly ──
echo "Verifying library links..."
MISSING=$(LD_LIBRARY_PATH=/u/imuradli/GPUCompress/.deps/hdf5/lib:$GPU_DIR/build:$GPU_DIR/.deps/lib:/opt/cray/libfabric/1.22.0/lib64 \
    ldd "$VPIC_BIN" 2>/dev/null | grep "not found" || true)

if [ -n "$MISSING" ]; then
    echo "WARNING: Some libraries not found:"
    echo "$MISSING"
    echo ""
    echo "VPIC may segfault at runtime. Check LD_LIBRARY_PATH."
else
    echo "All libraries resolved."
fi

echo ""
echo "============================================================"
echo "  VPIC benchmark deck rebuilt successfully"
echo "============================================================"
echo "  Binary: $VPIC_BIN"
echo ""
echo "  Run smoke test:"
echo "    cd $VPIC_BENCH"
echo "    LD_LIBRARY_PATH=/u/imuradli/GPUCompress/.deps/hdf5/lib:$GPU_DIR/build:$GPU_DIR/.deps/lib:/opt/cray/libfabric/1.22.0/lib64 \\"
echo "    GPUCOMPRESS_WEIGHTS=\"../../neural_net/weights/model.nnwt\" \\"
echo "    VPIC_NX=64 VPIC_CHUNK_MB=4 VPIC_RUNS=1 \\"
echo "    ./vpic_benchmark_deck.Linux"
echo "============================================================"
