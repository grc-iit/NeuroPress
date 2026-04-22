#!/bin/bash
# gpucompress_warpx_delta build script — clones BLAST-WarpX, applies GPUCompress
# patches, builds warpx.3d against HDF5 + nvcomp + GPUCompress from the
# gpucompress_base image. Must run AFTER gpucompress_base in the pipeline.
#
# Mirrors docker/warpx/Dockerfile (project-canonical recipe) but remaps
#   /tmp/hdf5-install -> /opt/hdf5-install
#   /tmp/lib          -> /opt/nvcomp/lib
# so runtime apptainer bind-mounts of the host /tmp cannot shadow the libs.
#
# Placeholders: ##CUDA_ARCH##
set -e
export DEBIAN_FRONTEND=noninteractive

# ── Clone BLAST-WarpX ──────────────────────────────────────────────────
git clone https://github.com/BLAST-WarpX/warpx.git /opt/sims/warpx

# ── Apply GPUCompress patch + overlay canonical sources ────────────────
cd /opt/sims/warpx
git apply /opt/GPUCompress/benchmarks/warpx/patches/warpx-gpucompress.patch || true
cp /opt/GPUCompress/benchmarks/warpx/patches/FlushFormatGPUCompress.H \
    Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.H
cp /opt/GPUCompress/benchmarks/warpx/patches/FlushFormatGPUCompress.cpp \
    Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.cpp
cp /opt/GPUCompress/benchmarks/warpx/patches/Diagnostics.cpp \
    Source/Diagnostics/Diagnostics.cpp
cp /opt/GPUCompress/benchmarks/warpx/patches/FullDiagnostics.cpp \
    Source/Diagnostics/FullDiagnostics.cpp
cp /opt/GPUCompress/benchmarks/warpx/patches/GPUCompress.cmake \
    cmake/dependencies/GPUCompress.cmake

# ── Configure + build warpx.3d (3D CUDA, single precision, GPUCompress) ─
mkdir -p build-gpucompress && cd build-gpucompress
CC=$(which gcc) CXX=$(which g++) CUDACXX=$(which nvcc) CUDAHOSTCXX=$(which g++) \
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_MPI=ON \
    -DWarpX_DIMS=3 \
    -DWarpX_PRECISION=SINGLE \
    -DWarpX_PARTICLE_PRECISION=SINGLE \
    -DWarpX_GPUCOMPRESS=ON \
    -DGPUCOMPRESS_PREFIX="/opt/GPUCompress/build" \
    "-DCMAKE_PREFIX_PATH=/opt/hdf5-install" \
    -DAMReX_CUDA_ARCH=##CUDA_ARCH## \
    "-DCMAKE_CXX_FLAGS=-mcmodel=large -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/GPUCompress/benchmarks -I/opt/hdf5-install/include" \
    "-DCMAKE_CUDA_FLAGS=-Xcompiler -mcmodel=large -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/GPUCompress/benchmarks -I/opt/hdf5-install/include --diag-suppress=222 --diag-suppress=221" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-relax -L/opt/hdf5-install/lib -L/opt/nvcomp/lib -L/opt/GPUCompress/build -Wl,--no-as-needed -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -lhdf5 -lnvcomp -Wl,--as-needed -L/usr/local/cuda/lib64 -lcudart"

cmake --build . -j"${BUILD_JOBS:-4}"

# Stable symlink so callers don't have to glob for the feature-tagged binary.
# WarpX names the binary warpx.3d.MPI.CUDA.SP.PSP.OPMD.EB.QED etc. depending
# on flags — pkg.py expects a fixed path at /opt/sims/warpx/build-gpucompress/bin/warpx.3d.
# Glob `warpx.3d.*` (dot-star) not `warpx.3d*` — the latter matches the symlink
# itself once WarpX's CMake creates one, producing a self-referential link.
built_bin=$(ls /opt/sims/warpx/build-gpucompress/bin/warpx.3d.* 2>/dev/null | head -1)
if [ -n "$built_bin" ] && [ "$built_bin" != "/opt/sims/warpx/build-gpucompress/bin/warpx.3d" ]; then
    ln -sf "$built_bin" /opt/sims/warpx/build-gpucompress/bin/warpx.3d
    ln -sf "$built_bin" /usr/local/bin/warpx.3d
fi
