FROM gpucompress-base

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

# Clone LAMMPS
RUN git clone https://github.com/lammps/lammps.git /opt/sims/lammps

# Apply GPUCompress patch and overlay source files
RUN cd /opt/sims/lammps \
    && git apply /opt/GPUCompress/benchmarks/lammps/patches/lammps-gpucompress.patch || true \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.h   src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu src/KOKKOS/ \
    && cp /opt/GPUCompress/benchmarks/lammps/patches/KOKKOS.cmake               cmake/Modules/Packages/KOKKOS.cmake

# Build the Kendall-tau ranking profiler shared library
RUN nvcc -O3 -std=c++17 \
    -gencode "arch=compute_80,code=sm_80" \
    -Xcompiler -fPIC -shared \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/benchmarks \
    -L/opt/GPUCompress/build -lgpucompress \
    -o /usr/local/lib/liblammps_ranking_profiler.so \
    /opt/GPUCompress/benchmarks/lammps/patches/lammps_ranking_profiler.cu \
    && ldconfig

# Build LAMMPS with Kokkos CUDA
RUN cd /opt/sims/lammps \
    && mkdir -p build && cd build \
    && cmake ../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DPKG_KOKKOS=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DBUILD_MPI=ON \
    && make -j$(nproc)
