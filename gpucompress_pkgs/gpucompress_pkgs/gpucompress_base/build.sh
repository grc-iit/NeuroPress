#!/bin/bash
# gpucompress_base build script — HDF5 + nvcomp + GPUCompress.
# Placeholders: ##HDF5_VERSION## ##NVCOMP_VERSION## ##CUDA_ARCH##
#               ##GPUCOMPRESS_REPO## ##GPUCOMPRESS_REF##
set -e
export DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────────────────
# Use the Ubuntu 24.04 default GCC (13), which matches the Delta system
# (gcc-native/13.2 module). The NVCC 12.8 ICE (find_allocated_name_reference,
# lexical.c:22310) is a NVCC 12.8 parser bug — not a GCC 13 issue. The
# reference docker/Dockerfile uses GCC 13 + CUDA 12.6 without problems.
# Workaround for NVCC 12.8: --expt-relaxed-constexpr in CMAKE_CUDA_FLAGS.
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git cmake build-essential gfortran \
    python3 python3-pip \
    openmpi-bin libopenmpi-dev \
    openssh-server openssh-client \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# SSH setup (required by Jarvis's MPI launcher inside apptainer instances;
# mirrors reference builtin/nyx/build.sh)
mkdir -p /var/run/sshd /root/.ssh \
 && ssh-keygen -A \
 && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
 && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
 && chmod 700 /root/.ssh \
 && chmod 600 /root/.ssh/authorized_keys \
 && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
 && printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" >> /etc/ssh/ssh_config

# ── nvcomp ──────────────────────────────────────────────────────────────
mkdir -p /opt/nvcomp/include /opt/nvcomp/lib
cd /tmp
curl -sL -o nvcomp.tar.xz \
  "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-##NVCOMP_VERSION##-archive.tar.xz"
tar -xf nvcomp.tar.xz
cp -r nvcomp-linux-x86_64-*/include/* /opt/nvcomp/include/
cp -r nvcomp-linux-x86_64-*/lib/*     /opt/nvcomp/lib/
rm -rf nvcomp.tar.xz nvcomp-linux-x86_64-*

# ── HDF5 ────────────────────────────────────────────────────────────────
# Install to /opt/hdf5-install/ — cmake/HDF5Vol.cmake hard-codes that path
# for discovery of the VOL connector. Matches docker/Dockerfile layout.
cd /tmp
curl -sL -o hdf5.tar.gz \
  "https://github.com/HDFGroup/hdf5/releases/download/##HDF5_VERSION##/hdf5-##HDF5_VERSION##.tar.gz"
tar xzf hdf5.tar.gz
mkdir -p hdf5-build && cd hdf5-build
cmake /tmp/hdf5-##HDF5_VERSION## \
    -DCMAKE_INSTALL_PREFIX=/opt/hdf5-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_BUILD_TOOLS=OFF \
    -DHDF5_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DHDF5_BUILD_CPP_LIB=OFF \
    -DHDF5_BUILD_FORTRAN=OFF \
    -DHDF5_BUILD_JAVA=OFF \
    -DHDF5_BUILD_HL_LIB=ON
make -j"${BUILD_JOBS:-8}"
make install
echo '/opt/hdf5-install/lib' > /etc/ld.so.conf.d/hdf5.conf
ldconfig
cd /tmp && rm -rf hdf5.tar.gz hdf5-##HDF5_VERSION## hdf5-build

# ── GPUCompress (CUDA library + VOL + weights) ──────────────────────────
git clone ##GPUCOMPRESS_REPO## /opt/GPUCompress
cd /opt/GPUCompress
git checkout ##GPUCOMPRESS_REF##
git submodule update --init --recursive || true
mkdir -p build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=##CUDA_ARCH## \
    -DCMAKE_PREFIX_PATH="/opt/hdf5-install;/opt/nvcomp" \
    -DHDF5_VOL_PREFIX=/opt/hdf5-install \
    -DNVCOMP_PREFIX=/opt/nvcomp \
    -DCMAKE_CUDA_FLAGS="-I/usr/lib/x86_64-linux-gnu/openmpi/include"
make -j"${BUILD_JOBS:-8}"

# ── Runtime linker hints ────────────────────────────────────────────────
cat > /etc/ld.so.conf.d/gpucompress.conf <<'EOF'
/opt/hdf5-install/lib
/opt/nvcomp/lib
/opt/GPUCompress/build
EOF
ldconfig

# Drop CUDA compat libs — apptainer --nv bind-mounts the host's libcuda.so.1
# (reference: builtin/nyx/build.sh:43-46).
rm -rf /usr/local/cuda/compat 2>/dev/null || true

# Propagate LD_LIBRARY_PATH through sshd sessions (for mpirun over ssh inside
# apptainer instance; reference: builtin/nyx/build.sh:48-53).
cat >> /etc/ssh/sshd_config <<'EOF'
SetEnv LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64:/opt/hdf5-install/lib:/opt/nvcomp/lib:/opt/GPUCompress/build
SetEnv HDF5_PLUGIN_PATH=/opt/GPUCompress/build
SetEnv GPUCOMPRESS_WEIGHTS=/opt/GPUCompress/neural_net/weights/model.nnwt
EOF
