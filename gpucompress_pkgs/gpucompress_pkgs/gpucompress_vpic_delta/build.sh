#!/bin/bash
# gpucompress_vpic_delta build script — clones vpic-kokkos, compiles
# libvpic + Kokkos via the upstream cmake, then links the GPUCompress
# VPIC benchmark deck (vpic_benchmark_deck.cxx + vpic_ranking_profiler.cu
# + vpic_psnr.cu + deck/main.cc + deck/wrapper.cc) against HDF5 + nvcomp
# + GPUCompress from the gpucompress_base image.
#
# Must run AFTER gpucompress_base in the pipeline so that:
#   /opt/hdf5-install/                 ← HDF5 (libhdf5, headers)
#   /opt/nvcomp/                       ← nvcomp redist
#   /opt/GPUCompress/                  ← GPUCompress lib, headers, benchmark
#                                        deck sources, NN weights
#
# Placeholders: ##CUDA_ARCH##
#
# Mirrors docker/vpic/Dockerfile but remaps /tmp/hdf5-install → /opt/hdf5-install
# and /tmp/lib → /opt/nvcomp/lib because apptainer bind-mounts the host /tmp
# over the SIF's /tmp at runtime, hiding anything built there.
set -e
export DEBIAN_FRONTEND=noninteractive

# Use the system-default g++ (GCC 13 on Ubuntu 24.04) — mirrors
# docker/vpic/Dockerfile: ENV NVCC_WRAPPER_DEFAULT_COMPILER=g++.
# The NVCC 12.8 ICE (lexical.c:22310 find_allocated_name_reference) is a
# NVCC 12.8 parser bug, not a GCC 13 issue.
export NVCC_WRAPPER_DEFAULT_COMPILER=g++

# ── Clone VPIC-Kokkos (with bundled Kokkos submodule) ───────────────────
git clone --recursive https://github.com/lanl/vpic-kokkos.git /opt/sims/vpic-kokkos

# ── Build VPIC core library + Kokkos (CUDA, Ampere sm_##CUDA_ARCH##) ────
cd /opt/sims/vpic-kokkos
cmake -S . -B build-compress \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_KOKKOS_CUDA=ON \
    -DBUILD_INTERNAL_KOKKOS=ON \
    -DKokkos_ARCH_AMPERE##CUDA_ARCH##=ON \
    -DCMAKE_CXX_COMPILER="$(pwd)/kokkos/bin/nvcc_wrapper"
cmake --build build-compress -j"${BUILD_JOBS:-8}"

# ── Compile CUDA object: Kendall tau ranking profiler ──────────────────
/opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/benchmarks \
    -std=c++17 \
    -c /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_ranking_profiler.cu \
    -o /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_ranking_profiler.o \
    -x cu -expt-extended-lambda -arch=sm_##CUDA_ARCH##

# ── Compile CUDA object: PSNR kernel ───────────────────────────────────
/opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -std=c++17 \
    -c /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_psnr.cu \
    -o /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_psnr.o \
    -x cu -expt-extended-lambda -arch=sm_##CUDA_ARCH##

# ── Link the benchmark deck binary ─────────────────────────────────────
# -DINPUT_DECK points nvcc_wrapper at the GPUCompress benchmark deck cxx;
# -DGPU_DIR lets the deck locate weights/results relative to the install.
# Include path order: GPUCompress headers, vpic_kokkos_bridge, hdf5 headers,
# MPI headers (via mpicc --showme:compile), then the VPIC + Kokkos trees.
/opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/examples \
    -I/opt/GPUCompress/benchmarks \
    -I/opt/hdf5-install/include \
    $(mpicc --showme:compile 2>/dev/null || echo '') \
    -I/opt/sims/vpic-kokkos/src \
    -I/opt/sims/vpic-kokkos/build-compress/kokkos \
    -I/opt/sims/vpic-kokkos/build-compress/kokkos/core/src \
    -I/opt/sims/vpic-kokkos/kokkos/core/src \
    -I/opt/sims/vpic-kokkos/kokkos/tpls/desul/include \
    -I/opt/sims/vpic-kokkos/build-compress/kokkos/containers/src \
    -I/opt/sims/vpic-kokkos/kokkos/containers/src \
    -I/opt/sims/vpic-kokkos/kokkos/tpls/mdspan/include \
    -I/opt/sims/vpic-kokkos/kokkos/algorithms/src \
    -I/opt/sims/vpic-kokkos/build-compress/kokkos/algorithms/src \
    -std=c++20 \
    -DINPUT_DECK=\"/opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx\" \
    -DGPU_DIR=\"/opt/GPUCompress\" \
    -DUSE_KOKKOS -DENABLE_KOKKOS_CUDA \
    -DUSE_LEGACY_PARTICLE_ARRAY -DVPIC_ENABLE_AUTO_TUNING \
    /opt/sims/vpic-kokkos/deck/main.cc \
    /opt/sims/vpic-kokkos/deck/wrapper.cc \
    /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_ranking_profiler.o \
    /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_psnr.o \
    -o /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux \
    -Wl,-rpath,/opt/sims/vpic-kokkos/build-compress \
    -L/opt/sims/vpic-kokkos/build-compress -lvpic \
    -lpthread -ldl \
    -Wl,-rpath,/opt/sims/vpic-kokkos/build-compress/kokkos/core/src \
    -L/opt/sims/vpic-kokkos/build-compress/kokkos/core/src \
    -L/opt/sims/vpic-kokkos/build-compress/kokkos/containers/src \
    -lkokkoscore -lkokkoscontainers \
    -Wl,-rpath,/opt/GPUCompress/build \
    -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -Wl,-rpath,/opt/hdf5-install/lib \
    -L/opt/hdf5-install/lib -lhdf5 \
    -Wl,-rpath,/opt/nvcomp/lib \
    -L/opt/nvcomp/lib -lnvcomp \
    $(mpicc --showme:link 2>/dev/null || echo '-lmpi') -lgomp -lcuda \
    -x cu -expt-extended-lambda -Wext-lambda-captures-this \
    -arch=sm_##CUDA_ARCH## -Xcompiler -fopenmp

# ── Convenience symlink (mirrors Nyx /usr/bin/nyx_HydroTests pattern) ──
ln -sf /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux \
       /usr/bin/vpic_benchmark_deck.Linux

# Sanity check so a broken build fails the image build, not the first run.
test -x /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux
