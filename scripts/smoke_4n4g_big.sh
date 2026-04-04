#!/bin/bash
# 4n4g bigger smoke test: 20 timesteps, NX=200
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00

cd /u/$USER/GPUCompress

HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=4
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r node; do
    for _g in $(seq 1 $GPUS_PER_NODE); do echo "$node"; done
done > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"

MPI_NP=16 GPUS_PER_NODE=4 VPIC_NX=192 CHUNK_MB=4 TIMESTEPS=20 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh

rm -f "$HOSTFILE"
