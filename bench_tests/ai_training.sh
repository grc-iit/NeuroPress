#!/bin/bash
# ============================================================
# bench_tests/ai_training.sh — Deploy AI training checkpoint
#                               compression benchmark
#
# Compresses neural network training checkpoints exported as
# flat float32 tensors.  Weight tensors cover a wide range of
# compressibility: embedding tables and fully-connected weights
# are often highly compressible, while convolutional filters
# and attention heads vary considerably.
#
# Checkpoints must be generated before running this script:
#   python3 scripts/train_and_export_checkpoints.py --model vit_b_16
#   python3 scripts/train_gpt2_checkpoints.py
#
# HDF5 MODES (HDF5_MODE):
#   default  — No compression.  generic_benchmark runs the no-comp
#              phase.  Measures raw HDF5 I/O bandwidth.
#   vol      — GPUCompress HDF5 VOL.  GPU compresses each chunk
#              before it is written to disk.
#
# PARAMETERS THAT AFFECT I/O VOLUME:
#   AI_MODEL         Model whose checkpoints are compressed.
#                    vit_b_16 → ~327 MB per checkpoint file
#                    gpt2     → ~473 MB per checkpoint file
#   AI_CHECKPOINT_DIR Directory holding .f32 or .h5 checkpoint files.
#                    All files in the directory are processed in
#                    sequence (one per benchmark "timestep").
#                    More files = more total I/O.
#   CHUNK_MB         HDF5 chunk size in MB.  Smaller → more GPU
#                    parallelism; larger → better ratio for
#                    algorithms that benefit from wider context.
#
# PARAMETERS THAT AFFECT DATA VARIETY (compressibility):
#   AI_MODEL         Different architectures produce different weight
#                    distributions.  ViT attention weights tend to be
#                    structured; GPT-2 residuals are denser.
#   AI_DATASET       Determines how quickly weights diverge from
#                    initialization across epochs:
#                    cifar10   → fast convergence, later checkpoints
#                                are more structured (higher ratio).
#                    wikitext2 → slower convergence, more diverse
#                                weight distributions across epochs.
#   ERROR_BOUND      Lossy tolerance.  0.0 = lossless.  For
#                    inference deployment: 1e-4 – 1e-3 is often
#                    acceptable with negligible accuracy loss.
#
# USAGE:
#   bash bench_tests/ai_training.sh
#
#   # GPT-2, VOL mode, ratio policy
#   AI_MODEL=gpt2 AI_DATASET=wikitext2 HDF5_MODE=vol POLICY=ratio \
#       bash bench_tests/ai_training.sh
#
#   # ViT lossless default HDF5 baseline
#   AI_MODEL=vit_b_16 HDF5_MODE=default bash bench_tests/ai_training.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HDF5 mode ──────────────────────────────────────────────
# default : no-comp (uncompressed HDF5 baseline)
# vol     : GPUCompress HDF5 VOL
HDF5_MODE=${HDF5_MODE:-default}

# ── I/O volume parameters ──────────────────────────────────
AI_MODEL=${AI_MODEL:-vit_b_16}    # vit_b_16 (~327 MB) or gpt2 (~473 MB)
AI_DATASET=${AI_DATASET:-cifar10} # cifar10 (ViT default) or wikitext2 (GPT-2)
CHUNK_MB=${CHUNK_MB:-4}           # HDF5 chunk size in MB
VERIFY=${VERIFY:-1}               # 1 = bitwise readback verify

# ── Data variety / physics parameters ──────────────────────
# ERROR_BOUND: 0.0 = lossless.  Inference deployment: 1e-4 – 1e-3.
ERROR_BOUND=${ERROR_BOUND:-0.0}

# ── VOL-mode compression parameters ────────────────────────
# PHASE (HDF5_MODE=vol only):
#   fixed   : lz4 | snappy | deflate | gdeflate | zstd | ans | cascaded | bitcomp
#   adaptive: nn | nn-rl | nn-rl+exp50
PHASE=${PHASE:-lz4}
# POLICY (nn* phases only):
#   balanced → equal speed + ratio
#   ratio    → maximise compression ratio
#   speed    → maximise throughput
POLICY=${POLICY:-balanced}

# ── Paths ───────────────────────────────────────────────────
GENERIC_BIN="${GENERIC_BIN:-$GPUC_DIR/build/generic_benchmark}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

# Auto-resolve checkpoint directory
case "$AI_MODEL" in
    vit_b_16) _AI_SHORT="vit_b" ;;
    *)        _AI_SHORT="$AI_MODEL" ;;
esac
AI_DIR_NAME="${_AI_SHORT}_${AI_DATASET}"
AI_CHECKPOINT_DIR="${AI_CHECKPOINT_DIR:-$GPUC_DIR/data/ai_training/${AI_DIR_NAME}}"

export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Derived ─────────────────────────────────────────────────
case "$POLICY" in
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *)        W0=1.0; W1=1.0; W2=1.0 ;;
esac

if [ "$HDF5_MODE" = "default" ]; then
    BENCH_PHASE="no-comp"
else
    BENCH_PHASE="$PHASE"
fi

VERIFY_TAG=""; [ "$VERIFY" = "0" ] && VERIFY_TAG="_noverify"
EB_TAG=""; [ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && EB_TAG="_lossy${ERROR_BOUND}"
RESULTS_DIR="${RESULTS_DIR:-$GPUC_DIR/benchmarks/ai_training/results/bench_${AI_DIR_NAME}_chunk${CHUNK_MB}mb_${HDF5_MODE}${EB_TAG}${VERIFY_TAG}}"

# ── Validate checkpoint directory ───────────────────────────
if [ ! -d "$AI_CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $AI_CHECKPOINT_DIR"
    echo ""
    echo "Generate checkpoints first:"
    case "$AI_MODEL" in
        vit_*)  echo "  python3 scripts/train_and_export_checkpoints.py --model $AI_MODEL" ;;
        gpt2)   echo "  python3 scripts/train_gpt2_checkpoints.py" ;;
        *)      echo "  Place .f32 or .h5 files in $AI_CHECKPOINT_DIR" ;;
    esac
    exit 1
fi

# Auto-detect file format (.f32 preferred over .h5)
_FIRST_F32=$(ls "$AI_CHECKPOINT_DIR"/*.f32 2>/dev/null | head -1)
_FIRST_H5=$(ls  "$AI_CHECKPOINT_DIR"/*.h5  2>/dev/null | head -1)

if [ -n "$_FIRST_F32" ]; then
    AI_EXT=".f32"
    FIRST_FILE="$_FIRST_F32"
    N_FLOATS=$(( $(stat -c%s "$FIRST_FILE") / 4 ))
elif [ -n "$_FIRST_H5" ]; then
    AI_EXT=".h5"
    FIRST_FILE="$_FIRST_H5"
    N_FLOATS=$(python3 -c "
import h5py, sys
try:
    f=h5py.File('$FIRST_FILE','r'); print(f['data'].size); f.close()
except Exception as e:
    print(0, file=sys.stderr); sys.exit(1)
" 2>/dev/null)
    if [ -z "$N_FLOATS" ] || [ "$N_FLOATS" -eq 0 ]; then
        echo "ERROR: Cannot read .h5 file (h5py required): $FIRST_FILE"
        exit 1
    fi
else
    echo "ERROR: No .f32 or .h5 files found in $AI_CHECKPOINT_DIR"
    exit 1
fi

N_FILES=$(ls "$AI_CHECKPOINT_DIR"/*${AI_EXT} | wc -l)
FILE_MB=$(( N_FLOATS * 4 / 1024 / 1024 ))
TOTAL_IO=$(( FILE_MB * N_FILES ))

# Compute 2-D dims: largest factor decomposition (same as benchmark.sh)
DIM0=$(python3 -c "
import math
n=$N_FLOATS; s=int(math.isqrt(n))
while s>1 and n%s!=0: s-=1
print(s)
")
DIM1=$(( N_FLOATS / DIM0 ))
DIMS="${DIM0},${DIM1}"

echo "============================================================"
echo "  AI Training Checkpoint Benchmark Deploy"
echo "============================================================"
echo "  HDF5 mode   : $HDF5_MODE"
echo "  Model       : $AI_MODEL"
echo "  Dataset     : $AI_DATASET"
echo "  Checkpoints : $AI_CHECKPOINT_DIR"
echo "  Files       : $N_FILES × ${FILE_MB} MB (${AI_EXT})  (~${TOTAL_IO} MB total)"
echo "  Dims        : $DIMS"
echo "  Chunk       : ${CHUNK_MB} MB"
echo "  Error bound : $ERROR_BOUND"
echo "  Verify      : $VERIFY"
if [ "$HDF5_MODE" = "vol" ]; then
echo "  Phase       : $BENCH_PHASE"
echo "  Policy      : $POLICY  (w0=$W0 w1=$W1 w2=$W2)"
fi
echo ""
echo "  Results: $RESULTS_DIR"
echo "============================================================"
echo ""

if [ ! -x "$GENERIC_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $GENERIC_BIN"
    echo "  Build: cmake --build $GPUC_DIR/build --target generic_benchmark"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
BENCH_DIR="$RESULTS_DIR/${HDF5_MODE}_${BENCH_PHASE}"
mkdir -p "$BENCH_DIR"

EB_ARG=""
[ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && EB_ARG="--error-bound $ERROR_BOUND"
VERIFY_ARG=""; [ "$VERIFY" = "0" ] && VERIFY_ARG="--no-verify"

echo ">>> Running generic_benchmark — mode=$HDF5_MODE phase=$BENCH_PHASE policy=$POLICY"

GPUCOMPRESS_DETAILED_TIMING=1 \
    "$GENERIC_BIN" "$WEIGHTS" \
    --data-dir "$AI_CHECKPOINT_DIR" \
    --dims "$DIMS" \
    --ext "$AI_EXT" \
    --chunk-mb "$CHUNK_MB" \
    --name "$AI_DIR_NAME" \
    $EB_ARG \
    $VERIFY_ARG \
    --phase "$BENCH_PHASE" \
    --w0 $W0 --w1 $W1 --w2 $W2 \
    --lr 0.2 --mape 0.10 \
    --explore-k 4 --explore-thresh 0.20 \
    --out-dir "$BENCH_DIR" \
    > "$BENCH_DIR/ai_bench.log" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "PASS  Log: $BENCH_DIR/ai_bench.log"
else
    echo "FAIL (exit $STATUS)  Log: $BENCH_DIR/ai_bench.log"
    tail -20 "$BENCH_DIR/ai_bench.log"
    exit 1
fi
