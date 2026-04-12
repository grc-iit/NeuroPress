#!/bin/bash
# ============================================================
# Smoke Test: Run small versions of every workload over default HDF5.
#
# Delegates to benchmarks/benchmark.sh with minimal configs so the
# full HDF5 VOL path (compress → write → read → verify) is exercised
# for each live-simulation and AI-training workload.
#
# Usage:
#   bash scripts/smoke_test.sh [--no-verify]
#
# Workloads:
#   grayscott   — Gray-Scott reaction-diffusion  (L auto from DATA_MB=8)
#   vpic        — VPIC plasma PIC simulation     (NX auto from DATA_MB=8)
#   ai_training — NN checkpoint compression      (skipped if data absent)
#
# Output:
#   benchmarks/grayscott/results/smoke_*/
#   benchmarks/vpic-kokkos/results/smoke_*/
#   benchmarks/ai_training/results/smoke_*/
# ============================================================
set +e

GPU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GPU_DIR"

VERIFY=1
[[ "$1" == "--no-verify" ]] && VERIFY=0

PASS=0
FAIL=0
SKIP=0

# Small config shared across workloads:
#   DATA_MB=8  → GS L≈128 (~8 MB), VPIC NX≈49 (~9 MB)
#   TIMESTEPS=3  keeps runtime short
#   PHASES: one fixed + one NN phase is enough to exercise the HDF5 path
#   POLICIES: balanced only (NN phases run once)
#   CHUNK_MB=4  (default HDF5 chunk)
SMOKE_COMMON="DATA_MB=8 CHUNK_MB=4 TIMESTEPS=3 PHASES=no-comp,lz4,nn POLICIES=balanced VERIFY=$VERIFY"

run_bench() {
    local NAME="$1"
    local LOG="$2"
    shift 2
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $NAME"
    echo "════════════════════════════════════════════════════════════"
    if eval "$@" > "$LOG" 2>&1; then
        echo "  PASS: $NAME"
        echo "  Log: $LOG"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $NAME"
        echo "  Log: $LOG"
        tail -20 "$LOG"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Gray-Scott ──
GS_LOG="benchmarks/grayscott/results/smoke_gs.log"
mkdir -p benchmarks/grayscott/results
run_bench "Gray-Scott (DATA_MB=8, ts=3, phases=no-comp,lz4,nn)" \
    "$GS_LOG" \
    env $SMOKE_COMMON BENCHMARKS=grayscott \
    bash benchmarks/benchmark.sh

# ── 2. VPIC ──
if [ -f "benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux" ] && \
   [ -f /opt/hdf5/lib/libhdf5.so ]; then
    VPIC_LOG="benchmarks/vpic-kokkos/results/smoke_vpic.log"
    mkdir -p benchmarks/vpic-kokkos/results
    run_bench "VPIC (DATA_MB=8, ts=3, phases=no-comp,lz4,nn)" \
        "$VPIC_LOG" \
        env $SMOKE_COMMON BENCHMARKS=vpic \
        VPIC_WARMUP_STEPS=50 VPIC_SIM_INTERVAL=20 \
        bash benchmarks/benchmark.sh
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  VPIC — SKIPPED (binary or HDF5 library not found)"
    echo "  Build with: bash scripts/rebuild_vpic.sh"
    echo "════════════════════════════════════════════════════════════"
    SKIP=$((SKIP + 1))
fi

# ── 3. AI Training ──
# Try vit_b_16 first, then gpt2. Skip if neither data directory exists.
_AI_FOUND=""
for _model in vit_b_16 gpt2; do
    case "$_model" in
        vit_b_16) _short="vit_b"; _dataset="cifar10" ;;
        gpt2)     _short="gpt2";  _dataset="wikitext2" ;;
    esac
    _dir="data/ai_training/${_short}_${_dataset}"
    if [ -d "$_dir" ] && ls "$_dir"/*.f32 "$_dir"/*.h5 2>/dev/null | head -1 | grep -q .; then
        _AI_FOUND="$_model"
        _AI_DATASET="$_dataset"
        _AI_DIR="$_dir"
        break
    fi
done

if [ -n "$_AI_FOUND" ]; then
    AI_LOG="benchmarks/ai_training/results/smoke_ai.log"
    mkdir -p benchmarks/ai_training/results
    run_bench "AI Training (model=$_AI_FOUND, phases=no-comp,lz4,nn)" \
        "$AI_LOG" \
        env BENCHMARKS=ai_training \
        AI_MODEL=$_AI_FOUND AI_DATASET=$_AI_DATASET \
        AI_CHECKPOINT_DIR=$_AI_DIR \
        CHUNK_MB=4 PHASES=no-comp,lz4,nn POLICIES=balanced VERIFY=$VERIFY \
        bash benchmarks/benchmark.sh
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  AI Training — SKIPPED (no checkpoint data found)"
    echo "  Generate with:"
    echo "    python3 scripts/train_and_export_checkpoints.py --model vit_b_16"
    echo "    python3 scripts/train_gpt2_checkpoints.py"
    echo "════════════════════════════════════════════════════════════"
    SKIP=$((SKIP + 1))
fi

# ── Summary ──
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Smoke Test Summary"
echo "════════════════════════════════════════════════════════════"
echo "  PASSED:  $PASS"
echo "  FAILED:  $FAIL"
echo "  SKIPPED: $SKIP"
echo ""
echo "  Logs:"
echo "    benchmarks/grayscott/results/smoke_gs.log"
echo "    benchmarks/vpic-kokkos/results/smoke_vpic.log"
echo "    benchmarks/ai_training/results/smoke_ai.log"
echo "════════════════════════════════════════════════════════════"

[ $FAIL -gt 0 ] && exit 1
exit 0
