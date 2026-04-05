#!/bin/bash
set -e

VPIC_DIR="${VPIC_DIR:-$HOME/vpic-kokkos}"
GPU_DIR="${GPU_DIR:-$HOME/GPUCompress}"
VPIC_BUILD="${VPIC_BUILD:-${VPIC_DIR}/build-compress}"
HDF5_PREFIX="${HDF5_PREFIX:-/tmp/hdf5-install}"
NVCOMP_LIB="${NVCOMP_LIB:-/tmp/lib}"
CUDA_ARCH="${CUDA_ARCH:-80}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export NVCC_WRAPPER_DEFAULT_COMPILER=${CXX:-g++}

# Auto-detect MPI library path
if [ -n "$CRAY_MPICH_DIR" ]; then
    MPI_LIB_FLAGS="-L${CRAY_MPICH_DIR}/lib -lmpi"
    MPI_INC_FLAGS="-I${CRAY_MPICH_DIR}/include"
    echo "Using Cray MPICH: $CRAY_MPICH_DIR"
elif command -v mpicc &>/dev/null; then
    MPI_LIB_FLAGS="$(mpicc --showme:link 2>/dev/null || pkg-config --libs mpi 2>/dev/null || echo '-lmpi')"
    MPI_INC_FLAGS="$(mpicc --showme:compile 2>/dev/null || pkg-config --cflags mpi 2>/dev/null || echo '')"
    echo "Using system MPI: $(which mpicc)"
else
    MPI_LIB_FLAGS="-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi"
    MPI_INC_FLAGS=""
    echo "WARNING: No MPI detected, falling back to default OpenMPI path"
fi

DECK_PATH="${GPU_DIR}/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx"

# Compile ranking profiler (CUDA) as a separate object
echo "Compiling vpic_ranking_profiler.cu ..."
"${VPIC_DIR}/kokkos/bin/nvcc_wrapper" \
  -I"${GPU_DIR}/include" \
  -I"${GPU_DIR}/benchmarks" \
  -std=c++17 \
  -c "${SCRIPT_DIR}/vpic_ranking_profiler.cu" \
  -o "${SCRIPT_DIR}/vpic_ranking_profiler.o" \
  -x cu -expt-extended-lambda -arch=sm_${CUDA_ARCH}

# Compile PSNR kernel (CUDA) as a separate object
echo "Compiling vpic_psnr.cu ..."
"${VPIC_DIR}/kokkos/bin/nvcc_wrapper" \
  -I"${GPU_DIR}/include" \
  -std=c++17 \
  -c "${SCRIPT_DIR}/vpic_psnr.cu" \
  -o "${SCRIPT_DIR}/vpic_psnr.o" \
  -x cu -expt-extended-lambda -arch=sm_${CUDA_ARCH}

# Compile and link the deck
echo "Compiling vpic_benchmark_deck ..."
"${VPIC_DIR}/kokkos/bin/nvcc_wrapper" \
  -I"${GPU_DIR}/include" \
  -I"${GPU_DIR}/examples" \
  -I"${HDF5_PREFIX}/include" \
  $MPI_INC_FLAGS \
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
  "${SCRIPT_DIR}/vpic_ranking_profiler.o" \
  "${SCRIPT_DIR}/vpic_psnr.o" \
  -o "${SCRIPT_DIR}/vpic_benchmark_deck.Linux" \
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
  $MPI_LIB_FLAGS -lgomp \
  -x cu -expt-extended-lambda -Wext-lambda-captures-this -arch=sm_${CUDA_ARCH} -Xcompiler -fopenmp

echo "Built: vpic_benchmark_deck.Linux"
