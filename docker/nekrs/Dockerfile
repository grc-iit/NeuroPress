FROM gpucompress-base

# Install nekRS build dependency (gfortran needed by Nek5000 backend)
RUN apt-get update && apt-get install -y --no-install-recommends gfortran \
    && rm -rf /var/lib/apt/lists/*

# Build the nekRS bridge library (links against GPUCompress)
RUN g++ -shared -fPIC \
    -o /opt/GPUCompress/examples/libnekrs_gpucompress_udf.so \
    /opt/GPUCompress/examples/nekrs_gpucompress_udf.cpp \
    -I/opt/GPUCompress/include \
    -I/tmp/hdf5-install/include \
    -I/usr/local/cuda/include \
    -L/opt/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -L/tmp/hdf5-install/lib -lhdf5 \
    -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/opt/GPUCompress/build \
    -Wl,-rpath,/tmp/hdf5-install/lib \
    && cp /opt/GPUCompress/examples/libnekrs_gpucompress_udf.so /usr/local/lib/ \
    && ldconfig

# Clone nekRS
RUN git clone https://github.com/Nek5000/nekRS.git /opt/sims/nekRS

# Build and install nekRS (no source patches needed)
RUN cd /opt/sims/nekRS \
    && mkdir -p build && cd build \
    && CC=mpicc CXX=mpic++ FC=mpif77 cmake -S .. -B . \
        -DCMAKE_INSTALL_PREFIX=/opt/nekrs \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build . --target install -j$(nproc)

# Set up TGV benchmark case directory
RUN mkdir -p /opt/nekrs/cases/tgv \
    && cp /opt/GPUCompress/benchmarks/nekrs/patches/tgv.udf /opt/nekrs/cases/tgv/ \
    && cp /opt/GPUCompress/benchmarks/nekrs/patches/tgv.par /opt/nekrs/cases/tgv/ \
    && cp /opt/GPUCompress/benchmarks/nekrs/patches/udf.cmake /opt/nekrs/cases/tgv/

ENV NEKRS_HOME=/opt/nekrs
