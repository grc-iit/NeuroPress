#!/bin/bash
# ============================================================
# Run All SDRBench Datasets
#
# Runs NYX, Hurricane Isabel, and CESM-ATM sequentially with
# shared configuration. Uses the same env vars as run_all.sh.
#
# Usage:
#   bash benchmarks/sdrbench/run_all_sdr.sh
#
#   # Custom chunk size + fewer phases
#   CHUNK_MB=64 PHASES="no-comp,lz4,zstd,nn,nn-rl,nn-rl+exp50" \
#     bash benchmarks/sdrbench/run_all_sdr.sh
#
#   # Only NYX and Hurricane
#   SDR_DATASETS="nyx,hurricane_isabel" CHUNK_MB=8 \
#     bash benchmarks/sdrbench/run_all_sdr.sh
#
# Environment Variables:
#   SDR_DATASETS    nyx,hurricane_isabel,cesm_atm   Datasets to run
#   CHUNK_MB        8           Chunk size in MB
#   PHASES          (all 12)    Compression phases
#   POLICIES        balanced,ratio,speed   Cost model policies
#   SGD_LR          0.2         SGD learning rate
#   SGD_MAPE        0.10        MAPE threshold
#   EXPLORE_K       4           Exploration alternatives
#   EXPLORE_THRESH  0.20        Exploration error threshold
#   VERIFY          1           Read-back verification (0=skip)
#   DEBUG_NN        0           NN debug output
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDR_DATASETS=${SDR_DATASETS:-"nyx,hurricane_isabel,cesm_atm"}
CHUNK_MB=${CHUNK_MB:-8}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}
DEBUG_NN=${DEBUG_NN:-0}

IFS=',' read -ra DS_LIST <<< "$SDR_DATASETS"

echo "============================================================"
echo "  SDRBench Suite — All Datasets"
echo "============================================================"
echo "  Datasets  : ${SDR_DATASETS}"
echo "  Chunk size: ${CHUNK_MB} MB"
echo "  Phases    : ${PHASES}"
echo "  Policies  : ${POLICIES}"
echo "  SGD_LR=${SGD_LR}  SGD_MAPE=${SGD_MAPE}"
echo "  EXPLORE_K=${EXPLORE_K}  EXPLORE_THRESH=${EXPLORE_THRESH}"
echo "============================================================"
echo ""

TOTAL=${#DS_LIST[@]}
IDX=0

for ds in "${DS_LIST[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo ">>> [$IDX/$TOTAL] Running SDRBench: $ds ..."
    echo ""

    DATASET="$ds" \
    CHUNK_MB=$CHUNK_MB \
    PHASES=$PHASES \
    POLICIES=$POLICIES \
    SGD_LR=$SGD_LR \
    SGD_MAPE=$SGD_MAPE \
    EXPLORE_K=$EXPLORE_K \
    EXPLORE_THRESH=$EXPLORE_THRESH \
    VERIFY=$VERIFY \
    DEBUG_NN=$DEBUG_NN \
    bash "$SCRIPT_DIR/run_sdr_pm_eval.sh"

    echo ""
    echo ">>> [$IDX/$TOTAL] $ds complete."
    echo ""
done

echo "============================================================"
echo "  All SDRBench datasets complete."
echo "  Results in: $SCRIPT_DIR/results/"
echo "============================================================"
