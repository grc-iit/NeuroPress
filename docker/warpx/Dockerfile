FROM gpucompress-base

ARG CUDA_ARCH=80

# Clone WarpX
RUN git clone https://github.com/BLAST-WarpX/warpx.git /opt/sims/warpx

# Apply GPUCompress patch and overlay source files
RUN cd /opt/sims/warpx \
    && git apply /opt/GPUCompress/benchmarks/warpx/patches/warpx-gpucompress.patch || true \
    && cp /opt/GPUCompress/benchmarks/warpx/patches/FlushFormatGPUCompress.H \
       Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.H \
    && cp /opt/GPUCompress/benchmarks/warpx/patches/FlushFormatGPUCompress.cpp \
       Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.cpp \
    && cp /opt/GPUCompress/benchmarks/warpx/patches/Diagnostics.cpp \
       Source/Diagnostics/Diagnostics.cpp \
    && cp /opt/GPUCompress/benchmarks/warpx/patches/FullDiagnostics.cpp \
       Source/Diagnostics/FullDiagnostics.cpp \
    && cp /opt/GPUCompress/benchmarks/warpx/patches/GPUCompress.cmake \
       cmake/dependencies/GPUCompress.cmake

# Build WarpX (3D CUDA, single precision — project rule: all data fp32)
RUN cd /opt/sims/warpx \
    && mkdir -p build-gpucompress && cd build-gpucompress \
    && CC=$(which gcc) CXX=$(which g++) CUDACXX=$(which nvcc) CUDAHOSTCXX=$(which g++) \
       cmake -S .. -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DWarpX_COMPUTE=CUDA \
        -DWarpX_MPI=ON \
        -DWarpX_DIMS=3 \
        -DWarpX_PRECISION=SINGLE \
        -DWarpX_PARTICLE_PRECISION=SINGLE \
        -DWarpX_GPUCOMPRESS=ON \
        -DGPUCOMPRESS_PREFIX="/opt/GPUCompress/build" \
        "-DCMAKE_PREFIX_PATH=/tmp/hdf5-install" \
        -DAMReX_CUDA_ARCH=${CUDA_ARCH} \
        "-DCMAKE_CXX_FLAGS=-mcmodel=large -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/GPUCompress/benchmarks -I/tmp/hdf5-install/include" \
        "-DCMAKE_CUDA_FLAGS=-Xcompiler -mcmodel=large -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/opt/GPUCompress/benchmarks -I/tmp/hdf5-install/include --diag-suppress=222 --diag-suppress=221" \
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-relax -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/tmp/hdf5-install/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart" \
    && cmake --build . -j$(nproc)
