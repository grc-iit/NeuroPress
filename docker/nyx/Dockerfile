FROM gpucompress-base

# Clone Nyx with AMReX submodule
RUN git clone --recursive https://github.com/AMReX-Astro/Nyx.git /opt/sims/Nyx \
    && cd /opt/sims/Nyx/subprojects/amrex && git checkout development

# Apply GPUCompress patch and overlay source files
RUN cd /opt/sims/Nyx \
    && git apply /opt/GPUCompress/benchmarks/nyx/patches/nyx-gpucompress.patch || true \
    && cp /opt/GPUCompress/benchmarks/nyx/patches/Nyx.H          Source/Driver/Nyx.H \
    && cp /opt/GPUCompress/benchmarks/nyx/patches/Nyx.cpp        Source/Driver/Nyx.cpp \
    && cp /opt/GPUCompress/benchmarks/nyx/patches/Nyx_output.cpp Source/IO/Nyx_output.cpp

# Build Nyx (Sedov blast wave, single precision, CUDA)
RUN cd /opt/sims/Nyx \
    && cmake -S . -B build-gpucompress \
        -DCMAKE_BUILD_TYPE=Release \
        -DNyx_MPI=YES \
        -DNyx_OMP=NO \
        -DNyx_HYDRO=YES \
        -DNyx_HEATCOOL=NO \
        -DNyx_GPU_BACKEND=CUDA \
        -DAMReX_CUDA_ARCH=Ampere \
        -DAMReX_PRECISION=SINGLE \
        -DAMReX_PARTICLES_PRECISION=SINGLE \
        -DCMAKE_C_COMPILER="$(which gcc)" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DCMAKE_CUDA_HOST_COMPILER="$(which g++)" \
        "-DCMAKE_CXX_FLAGS=-DAMREX_USE_GPUCOMPRESS -I/tmp/hdf5-install/include" \
        "-DCMAKE_CUDA_FLAGS=-DAMREX_USE_GPUCOMPRESS -I/tmp/hdf5-install/include" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L/tmp/hdf5-install/lib -L/tmp/lib -L/opt/GPUCompress/build -Wl,--no-as-needed -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -lhdf5 -lnvcomp -Wl,--as-needed" \
        "-DCMAKE_SHARED_LINKER_FLAGS=-L/tmp/hdf5-install/lib -lhdf5" \
    && cmake --build build-gpucompress --target nyx_HydroTests -j$(nproc)
