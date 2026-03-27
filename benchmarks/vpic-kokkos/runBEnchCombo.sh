#!/bin/bash

# First run: CHUNK_MB=4
echo "VPIC 1: Starting first benchmark (CHUNK_MB=4) at $(date)"
BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=4 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "First benchmark finished at $(date)"

# Second run: CHUNK_MB=16
echo "VPIC 2: Starting second benchmark (CHUNK_MB=16) at $(date)"
BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=16 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "Second benchmark finished at $(date)"

# Third run: CHUNK_MB=64
echo "VPIC 3: Starting third benchmark (CHUNK_MB=64) at $(date)"
BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=64 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "Third benchmark finished at $(date)"

echo "Grayscott 1: Starting first benchmark (CHUNK_MB=4) at $(date)"
BENCHMARKS=grayscott DATA_MB=1024 CHUNK_MB=4 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "First benchmark finished at $(date)"

# Second run: CHUNK_MB=16
echo "Grayscott 2: Starting second benchmark (CHUNK_MB=16) at $(date)"
BENCHMARKS=grayscott DATA_MB=1024 CHUNK_MB=16 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "Second benchmark finished at $(date)"

# Third run: CHUNK_MB=64
echo "Grayscott 3: Starting third benchmark (CHUNK_MB=64) at $(date)"
BENCHMARKS=grayscott DATA_MB=1024 CHUNK_MB=64 TIMESTEPS=50 SGD_LR=0.15 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 VERIFY=0 bash benchmarks/run_all.sh
echo "Third benchmark finished at $(date)"