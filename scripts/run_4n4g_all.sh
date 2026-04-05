#!/bin/bash
# ============================================================
# 4n4g Full Benchmark: Lossless + Lossy
# ~139 MB/rank, 50 timesteps, 16 ranks, all 12 phases
# Policies: balanced + ratio
#
# Usage (interactive): bash scripts/run_4n4g_all.sh
# Usage (sbatch):      sbatch scripts/run_4n4g_all.sh
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-4n4g-all
#SBATCH --output=vpic-4n4g-all-%j.out
#SBATCH --error=vpic-4n4g-all-%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00

cd /u/$USER/GPUCompress
_total_start=$(date +%s)

# ── Build hostfile for MPI distribution across all nodes ──
HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=4
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r node; do
    for _g in $(seq 1 $GPUS_PER_NODE); do
        echo "$node"
    done
done > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"
echo "Hostfile: $HOSTFILE"
cat "$HOSTFILE"
echo ""

echo "============================================================"
echo "  4n4g Full Benchmark Suite"
echo "  Run 1: Lossless (balanced + ratio)"
echo "  Run 2: Lossy    (balanced + ratio, error_bound=0.01)"
echo "  Phases: all 12"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Run 1: Lossless ──
echo ">>> [1/2] Lossless benchmark starting..."
_t0=$(date +%s)
MPI_NP=16 GPUS_PER_NODE=4 VPIC_NX=320 CHUNK_MB=32 TIMESTEPS=50 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh
_t1=$(date +%s)
echo ">>> [1/2] Lossless done in $((_t1 - _t0))s"
echo ""

# ── Run 2: Lossy ──
echo ">>> [2/2] Lossy benchmark starting..."
_t0=$(date +%s)
MPI_NP=16 GPUS_PER_NODE=4 VPIC_NX=320 CHUNK_MB=32 TIMESTEPS=50 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0.01 \
bash scripts/run_vpic_scaling.sh
_t1=$(date +%s)
echo ">>> [2/2] Lossy done in $((_t1 - _t0))s"

_total_end=$(date +%s)
_total_elapsed=$((_total_end - _total_start))
echo ""
echo "============================================================"
echo "  4n4g Full Benchmark Complete"
echo "  Total wall time: ${_total_elapsed}s ($((_total_elapsed/60))m $((_total_elapsed%60))s)"
echo "  Finished: $(date)"
echo ""
echo "  Lossless results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50/ 2>/dev/null
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50/balanced_w1-1-1/ 2>/dev/null
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50/ratio_only_w0-0-1/ 2>/dev/null
echo ""
echo "  Lossy results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50_lossy0.01/ 2>/dev/null
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50_lossy0.01/balanced_w1-1-1/ 2>/dev/null
ls -d benchmarks/vpic-kokkos/results/eval_NX320_chunk32mb_ts50_lossy0.01/ratio_only_w0-0-1/ 2>/dev/null
echo "============================================================"

# Cleanup hostfile
rm -f "$HOSTFILE"
