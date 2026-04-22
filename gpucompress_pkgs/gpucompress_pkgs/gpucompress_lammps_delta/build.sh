#!/bin/bash
# gpucompress_lammps_delta build script — builds the two GPUCompress bridge
# shared libraries, clones LAMMPS-develop, applies the GPUCompress Kokkos fix,
# and compiles the `lmp` binary against HDF5 + nvcomp + GPUCompress from the
# gpucompress_base image. Must run AFTER gpucompress_base in the pipeline.
#
# Mirrors docker/lammps/Dockerfile but remaps
#   /tmp/hdf5-install  -> /opt/hdf5-install
#   /tmp/lib           -> /opt/nvcomp/lib
# so apptainer bind-mounts of the host /tmp cannot shadow the libraries at
# runtime (same rationale as the /tmp -> /opt migration in gpucompress_base).
#
# Placeholders: ##CUDA_ARCH## ##KOKKOS_ARCH##
set -e
export DEBIAN_FRONTEND=noninteractive

# NVCC 12.8 ICE workaround — LAMMPS + Kokkos + lammps_ranking_profiler.cu are
# template-heavy; --expt-relaxed-constexpr sidesteps the parser bug in
# find_allocated_name_reference (same rationale as Nyx / WarpX build.sh).

# ── Bridge lib 1: liblammps_gpucompress_udf.so ──────────────────────────────
# Wraps the GPUCompress VOL + filter for in-situ HDF5 writes from the fix.
g++ -shared -fPIC \
    -o /opt/GPUCompress/examples/liblammps_gpucompress_udf.so \
    /opt/GPUCompress/examples/lammps_gpucompress_udf.cpp \
    -I/opt/GPUCompress/include \
    -I/opt/hdf5-install/include \
    -I/usr/local/cuda/include \
    $(mpicc --showme:compile 2>/dev/null || echo '') \
    -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -L/opt/hdf5-install/lib -lhdf5 \
    -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/opt/GPUCompress/build \
    -Wl,-rpath,/opt/hdf5-install/lib \
 && cp /opt/GPUCompress/examples/liblammps_gpucompress_udf.so /usr/local/lib/ \
 && ldconfig

# ── Bridge lib 2: liblammps_ranking_profiler.so (Kendall-tau) ───────────────
nvcc -O3 -std=c++17 \
    -ccbin "$(which g++)" \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -gencode "arch=compute_##CUDA_ARCH##,code=sm_##CUDA_ARCH##" \
    -Xcompiler -fPIC -shared \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/benchmarks \
    -L/opt/GPUCompress/build -lgpucompress \
    -o /usr/local/lib/liblammps_ranking_profiler.so \
    /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu \
 && ldconfig

# ── Clone LAMMPS develop ─────────────────────────────────────────────────────
git clone --branch develop --depth 1 \
    https://github.com/lammps/lammps.git /opt/sims/lammps

# ── Apply GPUCompress patch + overlay source files ────────────────────────────
cd /opt/sims/lammps
git apply /opt/GPUCompress/benchmarks/lammps/patches/lammps-gpucompress.patch || true
cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.h   src/KOKKOS/
cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp src/KOKKOS/
cp /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu src/KOKKOS/
cp /opt/GPUCompress/benchmarks/lammps/patches/KOKKOS.cmake               cmake/Modules/Packages/KOKKOS.cmake

# ── Build LAMMPS with Kokkos CUDA ─────────────────────────────────────────────
# HDF5_ROOT must be set so CMake's FindHDF5 locates /opt/hdf5-install.
export HDF5_ROOT=/opt/hdf5-install

mkdir -p build && cd build
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_##KOKKOS_ARCH##=ON \
    -DBUILD_MPI=ON \
    -DCMAKE_C_COMPILER="$(which gcc)" \
    -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)" \
    "-DCMAKE_CXX_FLAGS=-Wno-deprecated -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/hdf5-install/include" \
    "-DCMAKE_CUDA_FLAGS=--diag-suppress=1444 --expt-relaxed-constexpr --expt-extended-lambda -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/hdf5-install/include" \
    "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/lib -llammps_gpucompress_udf -llammps_ranking_profiler -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/opt/hdf5-install/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/opt/GPUCompress/build -Wl,-rpath,/opt/hdf5-install/lib -Wl,-rpath,/usr/local/lib"
make -j"${BUILD_JOBS:-8}"

# ── Convenience symlink ──────────────────────────────────────────────────────
ln -sf /opt/sims/lammps/build/lmp /usr/bin/lmp
ln -sf /opt/sims/lammps/build/lmp /usr/local/bin/lmp

# Sanity check so a broken build fails the image build, not the first run.
test -x /opt/sims/lammps/build/lmp
