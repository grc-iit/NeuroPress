#!/bin/bash
# ============================================================
# Install WarpX (AMReX laser-wakefield PIC) + GPUCompress
#
# Clone, patch, and build WarpX 3D with GPUCompress integration.
# Builds in single-precision (SINGLE) to match GPUCompress NN
# training data distribution.
#
# Usage:
#   bash benchmarks/warpx/build_warpx.sh
#
# Environment overrides:
#   GPUC_DIR        /workspaces/GPUCompress
#   SIMS_DIR        $GPUC_DIR/sims
#   CUDA_ARCH       89   (sm_89 = RTX 4070; 80 = A100; 90 = H100)
#   HDF5_ROOT       /opt/hdf5
#   NVCOMP_PREFIX   /opt/nvcomp
#
# Output binary:
#   $SIMS_DIR/warpx/build-gpucompress/bin/warpx.3d
# ============================================================
set -euo pipefail

GPUC_DIR="${GPUC_DIR:-/workspaces/GPUCompress}"
SIMS_DIR="${SIMS_DIR:-${GPUC_DIR}/sims}"
CUDA_ARCH="${CUDA_ARCH:-89}"
HDF5_ROOT="${HDF5_ROOT:-/opt/hdf5}"
NVCOMP_PREFIX="${NVCOMP_PREFIX:-/opt/nvcomp}"

export GPUC_DIR SIMS_DIR CUDA_ARCH HDF5_ROOT NVCOMP_PREFIX

bash "${GPUC_DIR}/deploy/run.sh" setup warpx
