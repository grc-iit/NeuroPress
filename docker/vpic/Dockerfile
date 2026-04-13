FROM gpucompress-base

ARG CUDA_ARCH=80

# Clone VPIC-Kokkos with Kokkos submodule
RUN git clone --recursive https://github.com/lanl/vpic-kokkos.git /opt/sims/vpic-kokkos

# Build VPIC core library with Kokkos CUDA
RUN cd /opt/sims/vpic-kokkos \
    && cmake -S . -B build-compress \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_KOKKOS_CUDA=ON \
        -DBUILD_INTERNAL_KOKKOS=ON \
        -DKokkos_ARCH_AMPERE${CUDA_ARCH}=ON \
        -DCMAKE_CXX_COMPILER="$(pwd)/kokkos/bin/nvcc_wrapper" \
    && cmake --build build-compress -j$(nproc)

# Compile ranking profiler (CUDA object)
ENV NVCC_WRAPPER_DEFAULT_COMPILER=g++
RUN /opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/benchmarks \
    -std=c++17 \
    -c /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_ranking_profiler.cu \
    -o /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_ranking_profiler.o \
    -x cu -expt-extended-lambda -arch=sm_${CUDA_ARCH}

# Compile PSNR kernel (CUDA object)
RUN /opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -std=c++17 \
    -c /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_psnr.cu \
    -o /opt/GPUCompress/benchmarks/vpic-kokkos/vpic_psnr.o \
    -x cu -expt-extended-lambda -arch=sm_${CUDA_ARCH}

# Compile and link the benchmark deck binary
RUN /opt/sims/vpic-kokkos/kokkos/bin/nvcc_wrapper \
    -I/opt/GPUCompress/include \
    -I/opt/GPUCompress/examples \
    -I/tmp/hdf5-install/include \
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
    -Wl,-rpath,/tmp/hdf5-install/lib \
    -L/tmp/hdf5-install/lib -lhdf5 \
    -Wl,-rpath,/tmp/lib \
    -L/tmp/lib -lnvcomp \
    $(mpicc --showme:link 2>/dev/null || echo '-lmpi') -lgomp -lcuda \
    -x cu -expt-extended-lambda -Wext-lambda-captures-this -arch=sm_${CUDA_ARCH} -Xcompiler -fopenmp
