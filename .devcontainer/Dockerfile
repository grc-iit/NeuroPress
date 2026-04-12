FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System packages + CMake 3.28+ from Kitware (HDF5 2.0.0 requires >= 3.26)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    gpg \
    wget \
    g++ \
    build-essential \
    curl \
    python3 \
    python3-pip \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    git \
    sudo \
    && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor - \
        > /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

# ── nvcomp 5.1.0 ──────────────────────────────────────────────────────────────
ENV NVCOMP_INSTALL_DIR=/opt/nvcomp
RUN set -e && \
    ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  NVCOMP_ARCH="x86_64" ;; \
        aarch64) NVCOMP_ARCH="sbsa" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    NVCOMP_VER="5.1.0.21" && \
    NVCOMP_ARCHIVE="nvcomp-linux-${NVCOMP_ARCH}-${NVCOMP_VER}_cuda12-archive" && \
    curl -L -o /tmp/nvcomp.tar.xz \
        "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-${NVCOMP_ARCH}/${NVCOMP_ARCHIVE}.tar.xz" && \
    mkdir -p ${NVCOMP_INSTALL_DIR}/include ${NVCOMP_INSTALL_DIR}/lib && \
    cd /tmp && tar -xf nvcomp.tar.xz && \
    cp -r /tmp/${NVCOMP_ARCHIVE}/include/* ${NVCOMP_INSTALL_DIR}/include/ && \
    cp -r /tmp/${NVCOMP_ARCHIVE}/lib/*     ${NVCOMP_INSTALL_DIR}/lib/ && \
    rm -rf /tmp/nvcomp*

# ── HDF5 2.0.0 ────────────────────────────────────────────────────────────────
ENV HDF5_INSTALL_DIR=/opt/hdf5
RUN set -e && \
    curl -L -o /tmp/hdf5.tar.gz \
        "https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz" && \
    cd /tmp && tar xzf hdf5.tar.gz && \
    mkdir /tmp/hdf5-build && cd /tmp/hdf5-build && \
    cmake /tmp/hdf5-2.0.0 \
        -DCMAKE_INSTALL_PREFIX=${HDF5_INSTALL_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_BUILD_HL_LIB=ON && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/hdf5*

# ── Node.js + Claude Code ─────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# ── Non-root user (matches host user: iowarp uid=1000 gid=1000) ───────────────
RUN groupadd -g 1000 iowarp \
    && useradd -m -u 1000 -g 1000 -s /bin/bash iowarp \
    && echo 'iowarp ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER iowarp

# ── Shell / home ───────────────────────────────────────────────────────────────
ENV SHELL=/bin/bash
ENV HOME=/home/iowarp

# ── Runtime environment ────────────────────────────────────────────────────────
ENV NVCOMP_INCLUDE_DIR=${NVCOMP_INSTALL_DIR}/include
ENV NVCOMP_LIB_DIR=${NVCOMP_INSTALL_DIR}/lib
ENV HDF5_VOL_DIR=${HDF5_INSTALL_DIR}
ENV LD_LIBRARY_PATH=${NVCOMP_INSTALL_DIR}/lib:${HDF5_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
