#!/bin/bash
# ============================================================
# 2n2g Quick Test: Lossless + Lossy
# ~128 MB/rank, 100 timesteps, 4 ranks
#
# Usage (interactive): bash scripts/run_2n2g_test.sh
# Usage (sbatch):      sbatch scripts/run_2n2g_test.sh
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-2n2g-test
#SBATCH --output=vpic-2n2g-test-%j.out
#SBATCH --error=vpic-2n2g-test-%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00

_total_start=$(date +%s)

# ── Build hostfile for MPI distribution across all nodes ──
HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=2
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
echo "  2n2g Quick Test"
echo "  Run 1: Lossless"
echo "  Run 2: Lossy (error_bound=0.01)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Run 1: Lossless ──
echo ">>> [1/2] Lossless test starting..."
MPI_NP=4 GPUS_PER_NODE=2 VPIC_NX=200 CHUNK_MB=16 TIMESTEPS=100 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh
echo ""

# ── Run 2: Lossy ──
echo ">>> [2/2] Lossy test starting..."
MPI_NP=4 GPUS_PER_NODE=2 VPIC_NX=200 CHUNK_MB=16 TIMESTEPS=100 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0.01 \
bash scripts/run_vpic_scaling.sh

_total_end=$(date +%s)
_total_elapsed=$((_total_end - _total_start))
echo ""
echo "============================================================"
echo "  2n2g Quick Test Complete"
echo "  Total wall time: ${_total_elapsed}s ($((_total_elapsed/60))m $((_total_elapsed%60))s)"
echo "  Finished: $(date)"
echo ""
echo "  Lossless results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX200_chunk16mb_ts100/ 2>/dev/null
echo "  Lossy results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX200_chunk16mb_ts100_lossy0.01/ 2>/dev/null
echo "============================================================"

# Cleanup hostfile
rm -f "$HOSTFILE"
