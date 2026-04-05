#!/bin/bash
# ============================================================
# Full LAMMPS Paper Evaluation
#
# Mirrors run_vpic_full.sh for LAMMPS molecular dynamics:
# 1. Full benchmark: all 12 phases, 3 policies, lossless, ~70 MB/dump
# 2. Large-scale data-intensive: 32M atoms, ~1.1 GB/dump
#
# Key design:
# - Hot sphere explosion creates evolving density front
# - sim_interval=50 steps between dumps → data diversity
# - Each NN phase × policy is a separate run → clean weight state
# - Lossless with bitwise verification
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/run_lammps_full.sh
# ============================================================
set -e

cd /home/cc/GPUCompress

echo "============================================================"
echo "Full LAMMPS Paper Evaluation"
echo "============================================================"

# ── Step 1: Standard benchmark (2M atoms, all phases, all policies) ──
echo ""
echo ">>> Step 1: Standard Benchmark (2M atoms, ~70 MB/dump, 10 timesteps)"
echo "    All 12 phases × 3 policies = 18 runs"

LMP_BIN=/home/cc/lammps/build/lmp \
LMP_ATOMS=80 \
CHUNK_MB=4 \
TIMESTEPS=10 \
SIM_INTERVAL=50 \
WARMUP_STEPS=100 \
POLICIES="balanced,ratio,speed" \
VERIFY=1 \
RESULTS_DIR="benchmarks/lammps/results/eval_paper_2M_all_phases" \
bash benchmarks/lammps/run_lammps_benchmark.sh

# ── Step 2: Data-intensive benchmark (32M atoms, fixed algos only) ──
echo ""
echo ">>> Step 2: Data-Intensive (32M atoms, ~1.1 GB/dump, 5 timesteps)"
echo "    Fixed algorithms + auto/ratio only (data-intensive stress test)"

LMP_BIN=/home/cc/lammps/build/lmp \
LMP_ATOMS=200 \
CHUNK_MB=4 \
TIMESTEPS=5 \
SIM_INTERVAL=10 \
WARMUP_STEPS=20 \
POLICIES="ratio" \
VERIFY=1 \
RESULTS_DIR="benchmarks/lammps/results/eval_paper_32M_data_intensive" \
bash benchmarks/lammps/run_lammps_benchmark.sh

echo ""
echo "============================================================"
echo "Full LAMMPS Paper Evaluation Complete"
echo ""
echo "Results:"
echo "  Standard:       benchmarks/lammps/results/eval_paper_2M_all_phases/"
echo "  Data-intensive: benchmarks/lammps/results/eval_paper_32M_data_intensive/"
echo "============================================================"
