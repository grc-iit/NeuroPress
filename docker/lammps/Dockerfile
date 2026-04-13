FROM gpucompress-base

ARG CUDA_ARCH=80

# Build the LAMMPS bridge library (links against GPUCompress)
RUN g++ -shared -fPIC \
    -o /opt/GPUCompress/examples/liblammps_gpucompress_udf.so \
    /opt/GPUCompress/examples/lammps_gpucompress_udf.cpp \
    -I/opt/GPUCompress/include \
    -I/tmp/hdf5-install/include \
    -I/usr/local/cuda/include \
    -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -L/tmp/hdf5-install/lib -lhdf5 \
    -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/opt/GPUCompress/build \
    -Wl,-rpath,/tmp/hdf5-install/lib \
    && cp /opt/GPUCompress/examples/liblammps_gpucompress_udf.so /usr/local/lib/ \
    && ldconfig

# Clone LAMMPS develop branch (requires Kokkos 5.0+ / C++20 for GPUCompress patch)
RUN git clone --branch develop --depth 1 \
    https://github.com/lammps/lammps.git /opt/sims/lammps

# Apply GPUCompress patch and overlay source files
RUN cd /opt/sims/lammps \
    && git apply /opt/GPUCompress/benchmarks/lammps/patches/lammps-gpucompress.patch || true \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.h   src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/KOKKOS.cmake               cmake/Modules/Packages/KOKKOS.cmake

# Build the Kendall-tau ranking profiler shared library
RUN nvcc -O3 -std=c++17 \
    -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
    -Xcompiler -fPIC -shared \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/benchmarks \
    -L/opt/GPUCompress/build -lgpucompress \
    -o /usr/local/lib/liblammps_ranking_profiler.so \
    /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu \
    && ldconfig

# Build LAMMPS with Kokkos CUDA
ENV HDF5_ROOT=/tmp/hdf5-install
RUN cd /opt/sims/lammps \
    && mkdir -p build && cd build \
    && cmake ../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DPKG_KOKKOS=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_AMPERE${CUDA_ARCH}=ON \
        -DBUILD_MPI=ON \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/tmp/hdf5-install/include" \
        -DCMAKE_CUDA_FLAGS="--diag-suppress=1444 -I/opt/GPUCompress/include -I/opt/GPUCompress/examples -I/tmp/hdf5-install/include" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/lib -llammps_gpucompress_udf -llammps_ranking_profiler -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/tmp/hdf5-install/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/opt/GPUCompress/build -Wl,-rpath,/tmp/hdf5-install/lib -Wl,-rpath,/usr/local/lib" \
    && make -j$(nproc)
