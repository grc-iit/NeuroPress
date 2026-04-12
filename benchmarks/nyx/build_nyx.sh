#!/bin/bash
# ============================================================
# Install Nyx (AMReX cosmological hydro) + GPUCompress
#
# Clone, patch, and build Nyx with GPUCompress integration.
# Builds the HydroTests (Sedov blast wave) target.
#
# Usage:
#   bash benchmarks/nyx/build_nyx.sh
#
# Environment overrides:
#   GPUC_DIR        /workspaces/GPUCompress
#   SIMS_DIR        $GPUC_DIR/sims
#   CUDA_ARCH       89   (sm_89 = RTX 4070; 80 = A100; 90 = H100)
#   HDF5_ROOT       /opt/hdf5
#   NVCOMP_PREFIX   /opt/nvcomp
#
# Output binary:
#   $SIMS_DIR/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests
# ============================================================
set -euo pipefail

GPUC_DIR="${GPUC_DIR:-/workspaces/GPUCompress}"
SIMS_DIR="${SIMS_DIR:-${GPUC_DIR}/sims}"
CUDA_ARCH="${CUDA_ARCH:-89}"
HDF5_ROOT="${HDF5_ROOT:-/opt/hdf5}"
NVCOMP_PREFIX="${NVCOMP_PREFIX:-/opt/nvcomp}"

export GPUC_DIR SIMS_DIR CUDA_ARCH HDF5_ROOT NVCOMP_PREFIX

bash "${GPUC_DIR}/deploy/run.sh" setup nyx
