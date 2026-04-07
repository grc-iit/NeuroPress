#!/bin/bash
# ============================================================
# Full VPIC Paper Evaluation
#
# 1. Threshold sweep (4x4 grid, 2MB chunks, NX=100)
# 2. Full benchmark (all phases, balanced+ratio, lossy, 32MB chunks, ~1GB dataset, 25 timesteps)
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/run_vpic_full.sh
# ============================================================
set -e

cd /home/cc/GPUCompress

echo "============================================================"
echo "Full VPIC Paper Evaluation"
echo "============================================================"

# ── Step 1: Threshold Sweep ──
echo ""
echo ">>> Step 1: Threshold Sweep (2MB chunks, NX=100)"
rm -rf benchmarks/Paper_Evaluations/4/results/vpic_threshold_sweep_balanced_eb0.01_lr0.2
CHUNK_MB=2 bash benchmarks/Paper_Evaluations/4/threshold_sweep/4.2.1_eval_vpic_threshold_sweep.sh
python3 benchmarks/Paper_Evaluations/4/threshold_sweep/4.2.1_plot_threshold_sweep.py \
    benchmarks/Paper_Evaluations/4/results/vpic_threshold_sweep_balanced_eb0.01_lr0.2

# ── Step 2: Full Benchmark ──
# NX=404 gives (404+2)^3 * 16 * 4 = ~1.07 GB per timestep
# 32MB chunks = ~34 chunks per timestep
echo ""
echo ">>> Step 2: Full Benchmark (all phases, 32MB chunks, ~1GB dataset, 25 timesteps)"

# 2a: Balanced policy, lossy (eb=0.01)
echo ""
echo "  [2a] Balanced, lossy (eb=0.01)"
BENCHMARKS=vpic VPIC_NX=404 CHUNK_MB=32 TIMESTEPS=25 \
    POLICIES=balanced VERIFY=0 VPIC_ERROR_BOUND=0.01 \
    VPIC_WARMUP_STEPS=500 VPIC_SIM_INTERVAL=190 \
    bash benchmarks/benchmark.sh

# 2b: Ratio policy, lossy (eb=0.01)
echo ""
echo "  [2b] Ratio, lossy (eb=0.01)"
BENCHMARKS=vpic VPIC_NX=404 CHUNK_MB=32 TIMESTEPS=25 \
    POLICIES=ratio VERIFY=0 VPIC_ERROR_BOUND=0.01 \
    VPIC_WARMUP_STEPS=500 VPIC_SIM_INTERVAL=190 \
    bash benchmarks/benchmark.sh

echo ""
echo "============================================================"
echo "Full VPIC Paper Evaluation Complete"
echo ""
echo "Results:"
echo "  Sweep:    benchmarks/Paper_Evaluations/4/results/vpic_threshold_sweep_balanced_eb0.01_lr0.2/"
echo "  Balanced: benchmarks/vpic-kokkos/results/eval_NX404_chunk32mb_ts25_noverify_lossy0.01/"
echo "  Ratio:    benchmarks/vpic-kokkos/results/eval_NX404_chunk32mb_ts25_noverify_lossy0.01/"
echo "============================================================"
