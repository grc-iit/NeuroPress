#!/bin/bash
set -e

VPIC_DIR="${VPIC_DIR:-$HOME/vpic-kokkos}"
GPU_DIR="${GPU_DIR:-$HOME/GPUCompress}"
VPIC_BUILD="${VPIC_BUILD:-${VPIC_DIR}/build-compress}"
HDF5_PREFIX="${HDF5_PREFIX:-/tmp/hdf5-install}"
NVCOMP_LIB="${NVCOMP_LIB:-/tmp/lib}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export NVCC_WRAPPER_DEFAULT_COMPILER=${CXX:-g++}

DECK_PATH="${GPU_DIR}/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx"

# Compile GPU comparison kernel separately (avoids VPIC deck preprocessor)
echo "Compiling gpu_compare.cu ..."
nvcc -c "${SCRIPT_DIR}/gpu_compare.cu" -o /tmp/gpu_compare.o -arch=sm_80

# Compile and link the deck
echo "Compiling vpic_benchmark_deck ..."
"${VPIC_DIR}/kokkos/bin/nvcc_wrapper" \
  -I"${GPU_DIR}/include" \
  -I"${GPU_DIR}/examples" \
  -I"${HDF5_PREFIX}/include" \
  -I. \
  -I"${VPIC_DIR}/src" \
  -I"${VPIC_BUILD}/kokkos" \
  -I"${VPIC_BUILD}/kokkos/core/src" \
  -I"${VPIC_DIR}/kokkos/core/src" \
  -I"${VPIC_DIR}/kokkos/tpls/desul/include" \
  -I"${VPIC_BUILD}/kokkos/containers/src" \
  -I"${VPIC_DIR}/kokkos/containers/src" \
  -std=c++17 \
  -DINPUT_DECK="\"${DECK_PATH}\"" \
  -DGPU_DIR="\"${GPU_DIR}\"" \
  -DUSE_KOKKOS -DENABLE_KOKKOS_CUDA \
  -DUSE_LEGACY_PARTICLE_ARRAY -DVPIC_ENABLE_AUTO_TUNING \
  "${VPIC_DIR}/deck/main.cc" \
  "${VPIC_DIR}/deck/wrapper.cc" \
  /tmp/gpu_compare.o \
  -o vpic_benchmark_deck.Linux \
  -Wl,-rpath,"${VPIC_BUILD}" \
  -L"${VPIC_BUILD}" -lvpic \
  -lpthread -ldl \
  -Wl,-rpath,"${VPIC_BUILD}/kokkos/core/src" \
  -L"${VPIC_BUILD}/kokkos/core/src" \
  -L"${VPIC_BUILD}/kokkos/containers/src" \
  -lkokkoscore -lkokkoscontainers \
  -Wl,-rpath,"${GPU_DIR}/build" \
  -L"${GPU_DIR}/build" -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
  -Wl,-rpath,"${HDF5_PREFIX}/lib" \
  -L"${HDF5_PREFIX}/lib" -lhdf5 \
  -Wl,-rpath,"${NVCOMP_LIB}" \
  -L"${NVCOMP_LIB}" -lnvcomp \
  -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lgomp \
  -x cu -expt-extended-lambda -Wext-lambda-captures-this -arch=sm_80

echo "Built: vpic_benchmark_deck.Linux"
