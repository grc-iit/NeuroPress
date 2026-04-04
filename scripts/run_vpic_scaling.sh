#!/bin/bash
# ============================================================
# VPIC Multi-GPU Scaling Benchmark
#
# Usage (interactive, after salloc):
#   bash scripts/run_vpic_scaling.sh
#
# Usage (sbatch):
#   sbatch scripts/run_vpic_scaling.sh
#
# Configure via environment variables:
#   MPI_NP=16 VPIC_NX=320 LOSSY=0.01 bash scripts/run_vpic_scaling.sh
#
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-scaling
#SBATCH --output=vpic-scaling-%j.out
#SBATCH --error=vpic-scaling-%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00

cd /u/$USER/GPUCompress

export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH

# ── Configuration (override via env vars) ──
_MPI_NP=${MPI_NP:-16}
_GPUS_PER_NODE=${GPUS_PER_NODE:-4}
_VPIC_NX=${VPIC_NX:-320}
_CHUNK_MB=${CHUNK_MB:-4}
_TIMESTEPS=${TIMESTEPS:-50}
_VERIFY=${VERIFY:-1}
_POLICIES=${POLICIES:-balanced}
_PHASES=${PHASES:-no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50}
_SGD_LR=${SGD_LR:-0.2}
_LOSSY=${LOSSY:-0}

# Build error bound arg and lossy tag (must match benchmark.sh _LOSSY_TAG logic)
_EB_ARG=""
_MODE="LOSSLESS"
_LOSSY_TAG=""
if [ "$_LOSSY" != "0" ] && [ "$_LOSSY" != "0.0" ]; then
    _EB_ARG="VPIC_ERROR_BOUND=$_LOSSY"
    _MODE="LOSSY (error_bound=$_LOSSY)"
    _LOSSY_TAG="_lossy${_LOSSY}"
fi

# Verify tag (must match benchmark.sh _VERIFY_TAG logic)
_VERIFY_TAG=""
[ "$_VERIFY" = "0" ] && _VERIFY_TAG="_noverify"

# Compute per-rank data size
_CELLS=$(( ($_VPIC_NX + 2) * ($_VPIC_NX + 2) * ($_VPIC_NX + 2) ))
_TOTAL_MB=$(( _CELLS * 64 / 1024 / 1024 ))
_PER_RANK_MB=$(( _TOTAL_MB / _MPI_NP ))

_start=$(date +%s)
echo "============================================================"
echo "  VPIC Scaling Benchmark"
echo "============================================================"
echo "  Job       : ${SLURM_JOB_ID:-interactive}"
echo "  Nodes     : ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "  Ranks     : $_MPI_NP (GPUs/node: $_GPUS_PER_NODE)"
echo "  VPIC_NX   : $_VPIC_NX (~${_PER_RANK_MB} MB/rank, ~${_TOTAL_MB} MB total)"
echo "  Chunk     : ${_CHUNK_MB} MB"
echo "  Timesteps : $_TIMESTEPS"
echo "  Phases    : $_PHASES"
echo "  Policies  : $_POLICIES"
echo "  Mode      : $_MODE"
echo "  Verify    : $_VERIFY"
echo "  SGD_LR    : $_SGD_LR"
echo "  Started   : $(date)"
echo "============================================================"
echo ""

# Back up previous results if they exist, then create fresh directory
_RESULT_DIR="benchmarks/vpic-kokkos/results/eval_NX${_VPIC_NX}_chunk${_CHUNK_MB}mb_ts${_TIMESTEPS}${_VERIFY_TAG}${_LOSSY_TAG}"
if [ -d "$_RESULT_DIR" ]; then
    _BACKUP="${_RESULT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$_RESULT_DIR" "$_BACKUP"
    echo "  Backed up previous results -> $_BACKUP"
fi

# Run benchmark
MPI_NP=$_MPI_NP \
GPUS_PER_NODE=$_GPUS_PER_NODE \
BENCHMARKS=vpic \
VPIC_NX=$_VPIC_NX \
CHUNK_MB=$_CHUNK_MB \
TIMESTEPS=$_TIMESTEPS \
VERIFY=$_VERIFY \
POLICIES=$_POLICIES \
PHASES=$_PHASES \
SGD_LR=$_SGD_LR \
VPIC_ERROR_BOUND=${_LOSSY:-0} \
bash benchmarks/benchmark.sh

_end=$(date +%s)
_elapsed=$((_end - _start))

echo ""
echo "============================================================"
echo "  VPIC Scaling Benchmark Complete"
echo "  Job       : ${SLURM_JOB_ID:-interactive}"
echo "  Wall time : ${_elapsed}s ($((_elapsed/60))m $((_elapsed%60))s)"
echo "  Finished  : $(date)"
echo ""
echo "  Results: $_RESULT_DIR"
ls -lh "$_RESULT_DIR/benchmark_vpic_deck_aggregate_multi_rank.csv" 2>/dev/null
ls -d "$_RESULT_DIR"/*/ 2>/dev/null
echo "============================================================"
