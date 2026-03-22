#!/bin/bash
#
# GPUCompress Dependencies Installation Script
#
# This script installs all required dependencies for the GPUCompress project:
#   1. nvcomp 5.1.0       -> /tmp/include, /tmp/lib
#   2. HDF5 2.0.0         -> /tmp/hdf5-install  (for VOL connector)
#   3. Builds the project -> build/
#   4. SDRBench datasets  -> data/sdrbench/  (for SC benchmarks)
#
# Run with: ./scripts/install_dependencies.sh
#
# Requirements:
#   - Linux x86_64 (Ubuntu 20.04+, RHEL 9+)
#   - NVIDIA GPU with compute capability >= 7.0
#   - CUDA Toolkit >= 12.0 installed
#   - NVIDIA driver >= 525.60.13
#   - cmake >= 3.18, g++ >= 9.0
#

set -e  # Exit on error

# Resolve project root early, before any cd changes the working directory.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_DIR="$(dirname "${_SCRIPT_DIR}")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
NVCOMP_VERSION="5.1.0.21"
NVCOMP_CUDA_VERSION="cuda12"
NVCOMP_INSTALL_DIR="/tmp"
NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-${NVCOMP_VERSION}_${NVCOMP_CUDA_VERSION}-archive.tar.xz"

HDF5_VERSION="2.0.0"
HDF5_INSTALL_DIR="/tmp/hdf5-install"
HDF5_URL="https://github.com/HDFGroup/hdf5/releases/download/${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz"

# SDRBench datasets for SC benchmark evaluation
SDRBENCH_DIR="${_PROJECT_DIR}/data/sdrbench"
SDRBENCH_BASE_URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data"
declare -A SDRBENCH_DATASETS=(
    ["hurricane_isabel"]="Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"
    ["nyx"]="EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"
    ["cesm_atm"]="CESM-ATM/SDRBENCH-CESM-ATM-cleared-1800x3600.tar.gz"
)

# Default CUDA architecture (sm_80 = Ampere A100).
# Override: CUDA_ARCH=90 ./scripts/install_dependencies.sh
CUDA_ARCH="${CUDA_ARCH:-80}"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo_info "Running pre-flight checks..."

# Check if running on Linux x86_64
if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "x86_64" ]]; then
    echo_error "This script only supports Linux x86_64"
    exit 1
fi

# Check for NVIDIA GPU (nvidia-smi may not be available in containers)
if command -v nvidia-smi &> /dev/null; then
    echo_info "Detected GPU:"
    nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || true
else
    echo_warn "nvidia-smi not found. Continuing (may be running in a container)."
fi

# Check for CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo_error "nvcc not found. Please install CUDA Toolkit >= 12.0 first."
    echo_info "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo_info "Detected CUDA version: ${CUDA_VERSION}"

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    echo_error "CUDA >= 12.0 required, found ${CUDA_VERSION}"
    exit 1
fi

# Check for cuFile (GDS support) - search multiple possible locations
CUFILE_LIB=""
for candidate in \
    "/usr/local/cuda/lib64/libcufile.so" \
    "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so" \
    "/opt/nvidia/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/libcufile.so"; do
    if [[ -f "$candidate" ]]; then
        CUFILE_LIB="$(dirname "$candidate")"
        break
    fi
done

# Broader search if not found
if [[ -z "$CUFILE_LIB" ]]; then
    CUFILE_PATH=$(find /opt/nvidia /usr/local/cuda -name "libcufile.so" 2>/dev/null | head -1)
    if [[ -n "$CUFILE_PATH" ]]; then
        CUFILE_LIB="$(dirname "$CUFILE_PATH")"
    fi
fi

if [[ -n "$CUFILE_LIB" ]]; then
    echo_info "cuFile (GPUDirect Storage) found at: ${CUFILE_LIB}"
else
    echo_warn "cuFile not found - GPUDirect Storage may not work"
fi

echo_info "cmake version: $(cmake --version | head -1)"
echo_info "Target CUDA architecture: sm_${CUDA_ARCH}"

# ============================================================================
# Download and install nvcomp
# ============================================================================

echo_info "=== Step 1/4: Installing nvcomp ${NVCOMP_VERSION} ==="

# Create install directories
mkdir -p "${NVCOMP_INSTALL_DIR}/include"
mkdir -p "${NVCOMP_INSTALL_DIR}/lib"

# Download nvcomp
NVCOMP_ARCHIVE="/tmp/nvcomp-${NVCOMP_VERSION}.tar.xz"
if [[ ! -f "${NVCOMP_ARCHIVE}" ]]; then
    echo_info "Downloading nvcomp from NVIDIA..."
    curl -L -o "${NVCOMP_ARCHIVE}" "${NVCOMP_URL}"
else
    echo_info "Using cached nvcomp archive"
fi

# Extract nvcomp
NVCOMP_EXTRACT_DIR="/tmp/nvcomp-linux-x86_64-${NVCOMP_VERSION}_${NVCOMP_CUDA_VERSION}-archive"
if [[ -d "${NVCOMP_EXTRACT_DIR}" ]]; then
    rm -rf "${NVCOMP_EXTRACT_DIR}"
fi

echo_info "Extracting nvcomp..."
cd /tmp
tar -xf "${NVCOMP_ARCHIVE}"

# Copy files to install location
echo_info "Installing nvcomp headers and libraries..."
cp -r "${NVCOMP_EXTRACT_DIR}/include/"* "${NVCOMP_INSTALL_DIR}/include/"
cp -r "${NVCOMP_EXTRACT_DIR}/lib/"* "${NVCOMP_INSTALL_DIR}/lib/"

# Verify installation
if [[ -f "${NVCOMP_INSTALL_DIR}/include/nvcomp.hpp" ]] && \
   [[ -f "${NVCOMP_INSTALL_DIR}/lib/libnvcomp.so" ]]; then
    echo_info "nvcomp installed successfully"
else
    echo_error "nvcomp installation failed"
    exit 1
fi

# ============================================================================
# Download, build, and install HDF5 2.0.0
# ============================================================================

echo_info "=== Step 2/4: Building HDF5 ${HDF5_VERSION} (for VOL connector) ==="

HDF5_ARCHIVE="/tmp/hdf5-${HDF5_VERSION}.tar.gz"
HDF5_SRC_DIR="/tmp/hdf5-${HDF5_VERSION}"
HDF5_BUILD_DIR="/tmp/hdf5-build"

if [[ -f "${HDF5_INSTALL_DIR}/lib/libhdf5.so" ]] && \
   [[ -f "${HDF5_INSTALL_DIR}/include/hdf5.h" ]]; then
    echo_info "HDF5 ${HDF5_VERSION} already installed at ${HDF5_INSTALL_DIR}, skipping build"
else
    # Download
    if [[ ! -f "${HDF5_ARCHIVE}" ]]; then
        echo_info "Downloading HDF5 ${HDF5_VERSION} source..."
        curl -L -o "${HDF5_ARCHIVE}" "${HDF5_URL}"
    else
        echo_info "Using cached HDF5 source archive"
    fi

    # Extract
    if [[ ! -d "${HDF5_SRC_DIR}" ]]; then
        echo_info "Extracting HDF5 source..."
        cd /tmp
        tar xzf "${HDF5_ARCHIVE}"
    fi

    # Build (C library only, minimal config for speed)
    echo_info "Configuring HDF5 build..."
    rm -rf "${HDF5_BUILD_DIR}"
    mkdir -p "${HDF5_BUILD_DIR}"
    cd "${HDF5_BUILD_DIR}"

    cmake "${HDF5_SRC_DIR}" \
        -DCMAKE_INSTALL_PREFIX="${HDF5_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_BUILD_HL_LIB=ON

    echo_info "Building HDF5 (this may take a few minutes)..."
    make -j$(nproc)
    make install

    # Verify
    if [[ -f "${HDF5_INSTALL_DIR}/lib/libhdf5.so" ]] && \
       [[ -f "${HDF5_INSTALL_DIR}/include/hdf5.h" ]]; then
        echo_info "HDF5 ${HDF5_VERSION} installed successfully at ${HDF5_INSTALL_DIR}"
    else
        echo_error "HDF5 installation failed"
        exit 1
    fi
fi

# ============================================================================
# Build project
# ============================================================================

echo_info "=== Step 3/4: Building GPUCompress ==="

cd "${_PROJECT_DIR}"
rm -rf build
mkdir -p build
cd build

# Pass CUDA architecture explicitly to avoid Cray PE overriding to sm_52.
# On Cray PE systems, the CC/cc wrappers can silently override the default
# CUDA architecture, breaking atomicAdd(double*, double) which requires sm_60+.
CMAKE_EXTRA_ARGS=""
if [[ -n "$CUFILE_LIB" ]]; then
    CMAKE_EXTRA_ARGS="-DCMAKE_LIBRARY_PATH=${CUFILE_LIB}"
fi

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    ${CMAKE_EXTRA_ARGS}

make -j$(nproc)

# ============================================================================
# Download SDRBench datasets
# ============================================================================

echo_info "=== Step 4/4: Downloading SDRBench datasets ==="

mkdir -p "${SDRBENCH_DIR}"

for dataset in "${!SDRBENCH_DATASETS[@]}"; do
    DATASET_DIR="${SDRBENCH_DIR}/${dataset}"
    ARCHIVE="${SDRBENCH_DIR}/${dataset}.tar.gz"
    URL="${SDRBENCH_BASE_URL}/${SDRBENCH_DATASETS[$dataset]}"

    if [[ -d "${DATASET_DIR}" ]] && [[ -n "$(ls -A "${DATASET_DIR}" 2>/dev/null)" ]]; then
        echo_info "  ${dataset} already exists, skipping"
        continue
    fi

    echo_info "  Downloading ${dataset}..."
    if curl -L -o "${ARCHIVE}" "${URL}" 2>/dev/null; then
        mkdir -p "${DATASET_DIR}"
        tar xzf "${ARCHIVE}" -C "${DATASET_DIR}"
        rm -f "${ARCHIVE}"
        echo_info "  ${dataset} ... OK"
    else
        echo_warn "  ${dataset} download failed (non-critical, can retry later)"
    fi
done

echo_info "SDRBench datasets installed at ${SDRBENCH_DIR}"

# ============================================================================
# Verify build
# ============================================================================

echo_info "Verifying build..."

FAIL=0
for target in gpu_compress gpu_decompress; do
    if [[ -x "./${target}" ]]; then
        echo_info "  ${target} ... OK"
    else
        echo_error "  ${target} ... MISSING"
        FAIL=1
    fi
done

for lib in libgpucompress.so libH5Zgpucompress.so libH5VLgpucompress.so; do
    if [[ -f "./${lib}" ]] || [[ -L "./${lib}" ]]; then
        echo_info "  ${lib} ... OK"
    else
        echo_warn "  ${lib} ... not built (may be expected if HDF5 not found by cmake)"
    fi
done

if [[ "$FAIL" -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "  GPUCompress Installation Complete"
    echo "============================================================"
    echo ""
    echo "Dependencies installed:"
    echo "  - nvcomp ${NVCOMP_VERSION}       -> /tmp/include, /tmp/lib"
    echo "  - HDF5 ${HDF5_VERSION}           -> ${HDF5_INSTALL_DIR}"
    echo "  - SDRBench datasets    -> ${SDRBENCH_DIR}"
    echo ""
    echo "Built targets:"
    echo "  - build/gpu_compress             (CLI compression)"
    echo "  - build/gpu_decompress           (CLI decompression)"
    echo "  - build/libgpucompress.so        (shared library)"
    echo "  - build/libH5Zgpucompress.so     (HDF5 filter plugin)"
    echo "  - build/libH5VLgpucompress.so    (HDF5 VOL connector)"
    echo ""
    echo "Before running, set up the environment:"
    echo "  source scripts/setup_env.sh"
    echo ""
    echo "Example usage:"
    echo "  ./build/gpu_compress input.bin output.lz4 lz4"
    echo "  ./build/gpu_decompress output.lz4 restored.bin"
    echo ""
else
    echo_error "Build verification failed"
    exit 1
fi
