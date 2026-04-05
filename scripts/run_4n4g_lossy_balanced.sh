#!/bin/bash
# 4n4g: Lossy (error_bound=0.01), balanced policy
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-4n4g-ly-bal
#SBATCH --output=vpic-4n4g-ly-bal-%j.out
#SBATCH --error=vpic-4n4g-ly-bal-%j.err
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:15:00

cd /u/$USER/GPUCompress

HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=4
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r node; do
    for _g in $(seq 1 $GPUS_PER_NODE); do echo "$node"; done
done > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"

MPI_NP=16 GPUS_PER_NODE=4 VPIC_NX=320 CHUNK_MB=32 TIMESTEPS=50 \
VERIFY=1 POLICIES=balanced LOSSY=0.01 \
bash scripts/run_vpic_scaling.sh

rm -f "$HOSTFILE"
