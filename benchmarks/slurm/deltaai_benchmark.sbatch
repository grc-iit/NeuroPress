#!/bin/bash
# ============================================================
# GPUCompress Multi-GPU Benchmark — DeltaAI (GH200)
#
# Submit: sbatch benchmarks/slurm/deltaai_benchmark.sbatch
#
# Customize via environment variables before sbatch:
#   NODES=2 GPUS=4 BENCHMARKS=vpic sbatch benchmarks/slurm/deltaai_benchmark.sbatch
#
# DeltaAI specifics:
#   - ARM aarch64, GH200 (sm_90), CUDA 12.8
#   - 4x GH200 per node, 120GB HBM3 each
#   - Shared filesystem: /work/hdd/bekn/YOUR_USERNAME/
#   - Node-local /tmp: 3.9TB (deleted after job)
#   - No mpirun — use srun for everything
# ============================================================

#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=gpucompress-bench
#SBATCH --output=gpucompress-bench-%j.out
#SBATCH --error=gpucompress-bench-%j.err
#SBATCH --time=02:00:00

# ── Node/GPU configuration (override via env before sbatch) ──
#SBATCH --nodes=${NODES:-1}
#SBATCH --gpus-per-node=${GPUS:-4}
#SBATCH --ntasks-per-node=${GPUS:-4}
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

# ── Resolve configuration ──
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
GPUS_PER_NODE=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

# ── Project paths (shared filesystem) ──
# Adjust GPU_DIR to your clone location on /work/hdd/bekn/
GPU_DIR="${GPU_DIR:-/work/hdd/bekn/${USER}/GPUCompress}"
cd "$GPU_DIR" || { echo "ERROR: GPU_DIR=$GPU_DIR not found"; exit 1; }

# ── Environment setup ──
source scripts/setup_env.sh
export LD_LIBRARY_PATH="$GPU_DIR/build:$LD_LIBRARY_PATH"

echo "============================================================"
echo "  GPUCompress DeltaAI Benchmark"
echo "============================================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Nodes       : $NODES"
echo "  GPUs/node   : $GPUS_PER_NODE"
echo "  Total ranks : $TOTAL_RANKS"
echo "  Partition    : $SLURM_JOB_PARTITION"
echo "  Node list   : $SLURM_JOB_NODELIST"
echo "  GPU_DIR     : $GPU_DIR"
echo "  Architecture: $(uname -m)"
echo "============================================================"
echo ""

# Verify build exists
if [ ! -f "$GPU_DIR/build/libgpucompress.so" ]; then
    echo "ERROR: Build not found. Run install_dependencies.sh first:"
    echo "  CUDA_ARCH=90 bash scripts/install_dependencies.sh"
    exit 1
fi

# ── Run benchmark ──
# srun is auto-detected by mpi_launch() via SLURM_JOB_ID.
# --gpus-per-task=1 gives each rank exactly one GPU.
MPI_NP=$TOTAL_RANKS \
GPUS_PER_NODE=$GPUS_PER_NODE \
BENCHMARKS=${BENCHMARKS:-"vpic"} \
DATA_MB=${DATA_MB:-256} \
CHUNK_MB=${CHUNK_MB:-4} \
TIMESTEPS=${TIMESTEPS:-25} \
POLICIES=${POLICIES:-"balanced"} \
VERIFY=${VERIFY:-0} \
bash benchmarks/benchmark.sh

echo ""
echo "============================================================"
echo "  Job $SLURM_JOB_ID complete."
echo "  Results: benchmarks/*/results/"
echo "============================================================"
