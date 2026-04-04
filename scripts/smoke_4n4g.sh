#!/bin/bash
# ============================================================
# 4n4g Smoke Test: Lossless only
# ~64 MB/rank, 7 timesteps, 16 ranks (4 nodes x 4 GPUs), 16 MB chunks
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-4n4g-smoke
#SBATCH --output=vpic-4n4g-smoke-%j.out
#SBATCH --error=vpic-4n4g-smoke-%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00

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
echo "  4n4g Smoke Test (Lossless)"
echo "  NX=160 (~64 MB/rank, ~1 GB total)"
echo "  7 timesteps, 16 MB chunks"
echo "  Ranks: 16 (4 nodes x 4 GPUs)"
echo "  Policies: balanced,ratio"
echo "  Started: $(date)"
echo "============================================================"
echo ""

MPI_NP=16 GPUS_PER_NODE=4 VPIC_NX=160 CHUNK_MB=16 TIMESTEPS=10 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh

_total_end=$(date +%s)
_total_elapsed=$((_total_end - _total_start))
echo ""
echo "============================================================"
echo "  4n4g Smoke Test Complete"
echo "  Wall time: ${_total_elapsed}s ($((_total_elapsed/60))m $((_total_elapsed%60))s)"
echo "  Finished: $(date)"
echo "  Results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts7/ 2>/dev/null
echo "============================================================"

# Cleanup hostfile
rm -f "$HOSTFILE"
