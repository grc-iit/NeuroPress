#!/bin/bash
# 2 nodes × 4 GPUs each (8 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_2n4g.sh
NODES=2 GPUS=4 sbatch benchmarks/slurm/deltaai_benchmark.sbatch "$@"
