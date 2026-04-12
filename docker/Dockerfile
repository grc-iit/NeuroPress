# Base image
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04                                             

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \                  
    build-essential cmake curl wget git python3 python3-pip \                         
    openmpi-bin libopenmpi-dev \                                                      
    && rm -rf /var/lib/apt/lists/* 

                                                                                       
RUN mkdir -p /tmp/include /tmp/lib \
    && curl -L -o /tmp/nvcomp.tar.xz \                                               
    ...      

# nvCOMP
RUN curl -L -o /tmp/nvcomp.tar.xz \
    "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz" \
    && cd /tmp && tar -xf nvcomp.tar.xz \
    && cp -r nvcomp-linux-x86_64-*/include/* /tmp/include/ \
    && cp -r nvcomp-linux-x86_64-*/lib/* /tmp/lib/ \
    && rm -rf /tmp/nvcomp.tar.xz /tmp/nvcomp-linux-x86_64-*

# HDF5 2.0.0
RUN curl -L -o /tmp/hdf5.tar.gz \
    "https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz" \
    && cd /tmp && tar xzf hdf5.tar.gz \
    && mkdir hdf5-build && cd hdf5-build \
    && cmake /tmp/hdf5-2.0.0 \
        -DCMAKE_INSTALL_PREFIX=/tmp/hdf5-install \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_BUILD_HL_LIB=ON \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/hdf5.tar.gz /tmp/hdf5-2.0.0 /tmp/hdf5-build


# ENV VARIABLE
ENV LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib
ENV NVCOMP_INCLUDE_DIR=/tmp/include
ENV NVCOMP_LIB_DIR=/tmp/lib
ENV HDF5_VOL_DIR=/tmp/hdf5-install

# Copy source + build NeuroPress

COPY . /opt/GPUCompress
WORKDIR /opt/GPUCompress

RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 \
    && make  -j$(nproc)

ENV LD_LIBRARY_PATH=/opt/GPUCompress/build:${LD_LIBRARY_PATH}
ENV HDF5_PLUGIN_PATH=/opt/GPUCompress/build
ENV GPUCOMPRESS_WEIGHTS=/opt/GPUCompress/neural_net/weights/model.nnwt

CMD ["/bin/bash"]