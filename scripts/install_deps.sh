#!/usr/bin/env bash
# scripts/install_deps.sh — stage HDF5-parallel and nvcomp under .deps/
# for a bare-metal (non-Apptainer) NeuroPress build.
#
# Mirrors the dependency-install steps in docker/Dockerfile so reviewers
# following the AD appendix on a bare Linux host get the same layout the
# Apptainer image uses.
#
# Outputs (all under $REPO_ROOT/.deps/):
#   include/          nvcomp + cuFile headers
#   lib/              libnvcomp.so, libnvcomp_cpu.so, CMake config
#   hdf5/{bin,lib,include,share}  HDF5 2.0.0 parallel install tree
#
# Usage:
#   ./scripts/install_deps.sh             # fetch + build + install
#   ./scripts/install_deps.sh --clean     # wipe existing .deps/ first
#
# After this script succeeds, build NeuroPress with:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CUDA_ARCHITECTURES=80 \
#         -DHDF5_ROOT=$PWD/.deps/hdf5 \
#         -DNVCOMP_PREFIX=$PWD/.deps
#   cmake --build build -j$(nproc)
#
# Requires: curl, tar, cmake 3.24+, a C/C++ compiler with CUDA 12.x and
# MPI (OpenMPI 4.x or MPICH 4.x).
#
# Cray systems (e.g. Delta login nodes with PrgEnv-gnu + cray-mpich): the
# Cray compiler wrappers put MPI headers under $MPICH_DIR/include but do
# not expose them to bare gcc/nvcc. Export CPATH before building:
#   export CPATH=$MPICH_DIR/include:$CPATH
# or set -DMPI_C_INCLUDE_DIRS=$MPICH_DIR/include when running cmake.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default layout: $REPO_ROOT/.deps/  (persists across shells and reboots)
# --tmp flag   : /tmp/                (matches docker/Dockerfile; ephemeral)
LAYOUT="deps"
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean)  CLEAN=1 ;;
        --tmp)    LAYOUT="tmp" ;;
        --deps)   LAYOUT="deps" ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--deps|--tmp] [--clean]
  --deps  (default) stage dependencies under \$REPO_ROOT/.deps/
  --tmp   stage dependencies under /tmp/ (matches docker/Dockerfile)
  --clean remove the existing install location before staging
EOF
            exit 0 ;;
    esac
done

if [[ "$LAYOUT" == "tmp" ]]; then
    DEPS_INCLUDE=/tmp/include
    DEPS_LIB=/tmp/lib
    HDF5_PREFIX=/tmp/hdf5-install
else
    DEPS_INCLUDE="$REPO_ROOT/.deps/include"
    DEPS_LIB="$REPO_ROOT/.deps/lib"
    HDF5_PREFIX="$REPO_ROOT/.deps/hdf5"
fi

NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz"
HDF5_URL="https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz"

if [[ $CLEAN -eq 1 ]]; then
    echo "==> Cleaning existing install at layout=$LAYOUT"
    if [[ "$LAYOUT" == "tmp" ]]; then
        rm -rf /tmp/hdf5-install
        # Only remove nvcomp files from /tmp, not the whole /tmp
        rm -rf /tmp/include/nvcomp* /tmp/lib/libnvcomp*
    else
        rm -rf "$REPO_ROOT/.deps"
    fi
fi

echo "==> Layout: $LAYOUT  (nvcomp -> $DEPS_INCLUDE + $DEPS_LIB, HDF5 -> $HDF5_PREFIX)"
mkdir -p "$DEPS_INCLUDE" "$DEPS_LIB"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "==> [1/2] nvcomp 5.1 (pre-built binary archive)"
curl -sSL -o "$TMP/nvcomp.tar.xz" "$NVCOMP_URL"
tar -xf "$TMP/nvcomp.tar.xz" -C "$TMP"
cp -r "$TMP"/nvcomp-linux-x86_64-*/include/* "$DEPS_INCLUDE/"
cp -r "$TMP"/nvcomp-linux-x86_64-*/lib/*     "$DEPS_LIB/"

echo "==> [2/2] HDF5 2.0.0 (parallel, no tools/examples/tests/CPP/Fortran/Java)"
curl -sSL -o "$TMP/hdf5.tar.gz" "$HDF5_URL"
tar xzf "$TMP/hdf5.tar.gz" -C "$TMP"
mkdir "$TMP/hdf5-build"
(
    cd "$TMP/hdf5-build"
    cmake "$TMP/hdf5-2.0.0" \
        -DCMAKE_INSTALL_PREFIX="$HDF5_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_ENABLE_PARALLEL=ON \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_BUILD_HL_LIB=ON \
        >/dev/null
    make -j"$(nproc)" >/dev/null
    make install       >/dev/null
)

echo
echo "==> Dependencies installed (layout=$LAYOUT)"
echo "    HDF5:   $HDF5_PREFIX/lib/libhdf5.so -> $(readlink -f "$HDF5_PREFIX/lib/libhdf5.so" 2>/dev/null || echo '(missing)')"
echo "    nvcomp: $DEPS_LIB/libnvcomp.so       -> $(readlink -f "$DEPS_LIB/libnvcomp.so" 2>/dev/null || echo '(missing)')"
echo
echo "Next: cmake -B build  (auto-detects either layout; explicit -DHDF5_ROOT / -DNVCOMP_PREFIX still wins)"
