#!/bin/bash
# ============================================================
# AI Training Checkpoint Workload — Timing Capture
#
# Runs ResNet-18 fine-tuning on CIFAR-10 with checkpoint writes
# going directly to HDF5 via the GPUCompress VOL connector.
# Training is intentionally limited to MAX_BATCHES_PER_EPOCH
# batches/epoch (dumb training) so that I/O is ~30% of runtime.
#
# Model   : resnet18 (~11M params, ~44 MB/tensor/checkpoint)
# Epochs  : 20 (checkpoint at each) × 4 tensor types = 3.5 GB I/O
# Batches : 10/epoch  → ~0.05s GPU compute per batch
# Target  : ~70% compute / ~30% I/O
#
# Results land in output/new_timing/ai/timing.csv
# ============================================================
set -euo pipefail

GPUC_DIR=/workspaces/GPUCompress
AI_DIR=$GPUC_DIR/output/new_timing/ai
mkdir -p "$AI_DIR"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64:$GPUC_DIR/build:$GPUC_DIR/examples${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPUCOMPRESS_WEIGHTS="$GPUC_DIR/neural_net/weights/model.nnwt"
export GPUCOMPRESS_VOL_MODE=trace
export GPUCOMPRESS_TIMING_OUTPUT="$AI_DIR/timing.csv"
export GPUCOMPRESS_TRACE_OUTPUT="$AI_DIR/trace.csv"

# Epochs and batches per epoch — tune MAX_BATCHES to hit ~70/30 compute/IO
EPOCHS=${EPOCHS:-40}
MAX_BATCHES=${MAX_BATCHES:-2}    # 2 batches/epoch; checkpoint every epoch; targets ~70/30 compute/IO
CHUNK_MB=${CHUNK_MB:-4}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== AI Training Checkpoint Benchmark ==="
log "  model=vit_b_16  epochs=$EPOCHS  max_batches/epoch=$MAX_BATCHES"
log "  chunk=${CHUNK_MB}MB  mode=${GPUCOMPRESS_VOL_MODE}"
log "  output: $GPUCOMPRESS_TIMING_OUTPUT"
log ""

CKPT_EPOCHS=$(seq -s, 1 $EPOCHS)

cd "$GPUC_DIR"
python3 scripts/train_and_export_checkpoints.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs $EPOCHS \
    --checkpoint-epochs "$CKPT_EPOCHS" \
    --max-batches-per-epoch $MAX_BATCHES \
    --batch-size 64 \
    --no-validate \
    --hdf5-direct \
    --chunk-mb $CHUNK_MB \
    --outdir "$AI_DIR/checkpoints" \
    --data-root "$GPUC_DIR/data" \
    2>&1 | tee "$AI_DIR/ai_train.log"

log ""
log "=== Done ==="
log "timing.csv:"
cat "$GPUCOMPRESS_TIMING_OUTPUT" 2>/dev/null || echo "  NOT FOUND"
