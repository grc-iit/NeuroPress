#!/bin/bash
# ============================================================
# AI Workload Benchmarks for SC Paper
#
# Produces all data needed for the AI checkpoint paragraph:
#   - Checkpoint sizes, write times, compression ratios
#   - Per-tensor algorithm selection (weights vs optimizer state)
#   - R², MAPE, MAE for NN prediction quality
#   - PSNR, RMSE, max_abs_err for lossy quality
#   - Tensor characterization per epoch
#   - Algorithm evolution heatmaps
#
# Models:
#   1. ViT-Base  (86M params, 1.3 GB/checkpoint, ~6 min/epoch)
#   2. GPT-2     (124M params, 1.9 GB/checkpoint, ~20 min/epoch)
#
# Configs per model:
#   - 16 MB chunks, lossless (error_bound=0.0)
#   - 16 MB chunks, lossy (error_bound=0.01)
#
# Each config benchmarks 15 algorithms × 4 tensors per checkpoint:
#   9 fixed (no-comp, lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp)
#   6 NN (bal_nn, bal_rl, bal_exp, rat_nn, rat_rl, rat_exp)
#
# Estimated time: ~6 hours total
#   ViT-Base:  ~100 min (60 train + 40 benchmark)
#   GPT-2:     ~260 min (200 train + 60 benchmark)
#
# Usage:
#   nohup bash benchmarks/run_ai_paper.sh > ai_paper.log 2>&1 &
#
# Quick test (ResNet-18, 2 epochs):
#   AI_VIT_MODEL=resnet18 AI_EPOCHS=2 bash benchmarks/run_ai_paper.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:${PROJECT_DIR}/build:${LD_LIBRARY_PATH:-}

# ── Overridable defaults ──
VIT_MODEL=${AI_VIT_MODEL:-vit_b_16}
EPOCHS=${AI_EPOCHS:-10}
ERROR_BOUND=${AI_ERROR_BOUND:-0.01}
CHUNK_MB=${AI_CHUNK_MB:-16}
SGD_LR=${AI_SGD_LR:-0.2}
SGD_MAPE=${AI_SGD_MAPE:-0.10}
EXPLORE_K=${AI_EXPLORE_K:-4}
EXPLORE_THRESH=${AI_EXPLORE_THRESH:-0.20}

RUN_ID=$(date +%Y%m%d_%H%M%S)
CKPT_STR=$(seq -s, 1 "$EPOCHS")
BASE_DIR="${PROJECT_DIR}/data/ai_training"

TOTAL_START=$(date +%s)

echo "============================================================"
echo "  AI Workload Benchmarks for SC Paper"
echo "  Started: $(date)"
echo "  Run ID:  ${RUN_ID}"
echo "============================================================"
echo ""
echo "  ViT model:      ${VIT_MODEL}"
echo "  Epochs:          ${EPOCHS}"
echo "  Chunk size:      ${CHUNK_MB} MB"
echo "  Error bound:     ${ERROR_BOUND} (lossy)"
echo "  SGD LR:          ${SGD_LR}"
echo "  SGD MAPE:        ${SGD_MAPE}"
echo "  Explore K:       ${EXPLORE_K}"
echo "  Explore Thresh:  ${EXPLORE_THRESH}"
echo ""
echo "  Per model: 2 configs × ${EPOCHS} epochs × 4 tensors × 15 algos"
echo "           = $(( 2 * EPOCHS * 4 * 15 )) measurements"
echo ""

# Common args
COMMON_ARGS="--epochs ${EPOCHS} --checkpoint-epochs ${CKPT_STR} \
    --hdf5-direct \
    --sgd-lr ${SGD_LR} --sgd-mape ${SGD_MAPE} \
    --explore-k ${EXPLORE_K} --explore-thresh ${EXPLORE_THRESH}"

# ============================================================
# 1. ViT-Base (or ResNet-18 for quick test)
# ============================================================
if [ "$VIT_MODEL" = "resnet18" ]; then
    VIT_LABEL="resnet18"
else
    VIT_LABEL="vit_b"
fi

VIT_LL="${BASE_DIR}/${VIT_LABEL}_${CHUNK_MB}mb_lossless_${RUN_ID}"
VIT_LY="${BASE_DIR}/${VIT_LABEL}_${CHUNK_MB}mb_lossy_eb${ERROR_BOUND}_${RUN_ID}"
VIT_DEFAULT="${BASE_DIR}/${VIT_LABEL}_default_${RUN_ID}"
VIT_CONFIGS="${CHUNK_MB}:0.0:${VIT_LL},${CHUNK_MB}:${ERROR_BOUND}:${VIT_LY}"

echo "============================================================"
echo "  Model 1/2: ${VIT_MODEL}"
echo "  Lossless → ${VIT_LL}"
echo "  Lossy    → ${VIT_LY}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

V_START=$(date +%s)

python3 "${PROJECT_DIR}/scripts/train_and_export_checkpoints.py" \
    --model "${VIT_MODEL}" \
    ${COMMON_ARGS} \
    --benchmark-configs "${VIT_CONFIGS}" \
    --outdir "${VIT_DEFAULT}"

V_END=$(date +%s)
V_MIN=$(( (V_END - V_START) / 60 ))
echo ""
echo "  ${VIT_MODEL} done in ${V_MIN} minutes"
echo ""

# ============================================================
# 2. GPT-2 Small
# ============================================================
GPT2_LL="${BASE_DIR}/gpt2_${CHUNK_MB}mb_lossless_${RUN_ID}"
GPT2_LY="${BASE_DIR}/gpt2_${CHUNK_MB}mb_lossy_eb${ERROR_BOUND}_${RUN_ID}"
GPT2_DEFAULT="${BASE_DIR}/gpt2_default_${RUN_ID}"
GPT2_CONFIGS="${CHUNK_MB}:0.0:${GPT2_LL},${CHUNK_MB}:${ERROR_BOUND}:${GPT2_LY}"

echo "============================================================"
echo "  Model 2/2: GPT-2 Small"
echo "  Lossless → ${GPT2_LL}"
echo "  Lossy    → ${GPT2_LY}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

G_START=$(date +%s)

python3 "${PROJECT_DIR}/scripts/train_gpt2_checkpoints.py" \
    ${COMMON_ARGS} \
    --benchmark-configs "${GPT2_CONFIGS}" \
    --outdir "${GPT2_DEFAULT}"

G_END=$(date +%s)
G_MIN=$(( (G_END - G_START) / 60 ))
echo ""
echo "  GPT-2 done in ${G_MIN} minutes"
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))
TOTAL_HRS=$(( TOTAL_MIN / 60 ))
REMAINDER_MIN=$(( TOTAL_MIN % 60 ))

echo "============================================================"
echo "  AI Paper Benchmarks Complete"
echo "  Finished: $(date)"
echo "  Total time: ${TOTAL_MIN} minutes (${TOTAL_HRS}h ${REMAINDER_MIN}m)"
echo "  Run ID: ${RUN_ID}"
echo "============================================================"
echo ""
echo "  Results:"
echo "    ${VIT_LL}/"
echo "    ${VIT_LY}/"
echo "    ${GPT2_LL}/"
echo "    ${GPT2_LY}/"
echo ""
echo "  Each contains:"
echo "    inline_benchmark.csv           ← per-epoch aggregate (45 columns)"
echo "    inline_benchmark_chunks.csv    ← per-chunk diagnostics"
echo "    0_tensor_characterization_epoch*.png"
echo "    {weights,adam_m,adam_v,gradients}/{balanced,ratio}/"
echo "      1_summary.png                ← ratio + throughput + Pareto"
echo "      3_algorithm_evolution.png    ← per-chunk NN selection heatmap"
echo "      4_*_predicted_vs_actual.png  ← NN prediction accuracy"
echo "      5a_sgd_convergence.png       ← MAPE over epochs"
echo "      5b_sgd_exploration_firing.png"
echo "      5c_mae_over_time.png         ← MAE over epochs"
echo "      5d_r2_over_time.png          ← R² over epochs"
echo "      6b_pipeline_waterfall.png    ← VOL stage breakdown"
echo "      6c_gpu_breakdown_over_time.png"
echo "      6d_cross_phase_pipeline_overhead.png"
echo ""
echo "  Paper placeholders filled:"
echo "    MODEL NAME:     ${VIT_MODEL} (86M params) + GPT-2 (124M params)"
echo "    Checkpoint size: ~1.3 GB (ViT-B) / ~1.9 GB (GPT-2)"
echo "    Write time:     see no-comp vs bal_nn in inline_benchmark.csv"
echo "    Compression:    see ratio column per tensor type"
echo "    Algorithm routing: see 3_algorithm_evolution.png (lossy)"
echo "    PSNR/RMSE:      see psnr_db/rmse/max_abs_err columns (lossy)"
echo "    R² scores:      see r2_ratio/r2_comp/r2_decomp/r2_psnr columns"
echo "============================================================"
