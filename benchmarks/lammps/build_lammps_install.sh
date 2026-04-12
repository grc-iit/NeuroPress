#!/bin/bash
# ============================================================
# Install LAMMPS + Kokkos + GPUCompress
#
# Clone, patch, and build LAMMPS with GPUCompress integration.
#
# Usage:
#   bash benchmarks/lammps/build_lammps_install.sh
#
# Environment overrides:
#   GPUC_DIR        /workspaces/GPUCompress
#   SIMS_DIR        $GPUC_DIR/sims
#   CUDA_ARCH       89   (sm_89 = RTX 4070; 80 = A100; 90 = H100)
#   HDF5_ROOT       /opt/hdf5
#   NVCOMP_PREFIX   /opt/nvcomp
#   LAMMPS_KOKKOS_ARCH   AMPERE89  (Kokkos arch macro)
# ============================================================
set -euo pipefail

GPUC_DIR="${GPUC_DIR:-/workspaces/GPUCompress}"
SIMS_DIR="${SIMS_DIR:-${GPUC_DIR}/sims}"
CUDA_ARCH="${CUDA_ARCH:-89}"
HDF5_ROOT="${HDF5_ROOT:-/opt/hdf5}"
NVCOMP_PREFIX="${NVCOMP_PREFIX:-/opt/nvcomp}"
LAMMPS_KOKKOS_ARCH="${LAMMPS_KOKKOS_ARCH:-AMPERE89}"

export GPUC_DIR SIMS_DIR CUDA_ARCH HDF5_ROOT NVCOMP_PREFIX LAMMPS_KOKKOS_ARCH

bash "${GPUC_DIR}/deploy/run.sh" setup lammps
