#!/usr/bin/env bash
# sweep_lr.sh — Sweep SGD learning rates on the Gray-Scott benchmark
#
# Runs the grayscott_benchmark with different LR values for the nn-rl
# and nn-rl+exp50 phases, plus a baseline nn (inference-only) run.
# Results are saved to benchmarks/grayscott/results_sweep/lr_<value>/.
#
# Usage:
#   ./benchmarks/grayscott/sweep_lr.sh [weights_path]
#
# Environment:
#   GPUCOMPRESS_WEIGHTS  — fallback weights path
#   SWEEP_LRS            — space-separated LR values (default: "0.01 0.05 0.1 0.2 0.4")
#   SWEEP_L              — grid side length (default: 640)
#   SWEEP_STEPS          — sim steps per timestep (default: 10)
#   SWEEP_CHUNK_Z        — chunk Z dimension (default: 20)
#   SWEEP_TIMESTEPS      — multi-timestep count (default: 100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN="$PROJECT_ROOT/build/grayscott_benchmark"

# Weights path: CLI arg > env var
WEIGHTS="${1:-${GPUCOMPRESS_WEIGHTS:-}}"
if [[ -z "$WEIGHTS" ]]; then
    echo "ERROR: No weights path. Usage: $0 <weights.nnwt>" >&2
    echo "       Or set GPUCOMPRESS_WEIGHTS env var." >&2
    exit 1
fi

# Configurable parameters
LRS=(${SWEEP_LRS:-0.05 0.1 0.2 0.25 0.5 0.7 0.9})
L="${SWEEP_L:-640}"
STEPS="${SWEEP_STEPS:-10}"
CHUNK_Z="${SWEEP_CHUNK_Z:-20}"
TIMESTEPS="${SWEEP_TIMESTEPS:-100}"

SWEEP_DIR="$SCRIPT_DIR/results_sweep"
SUMMARY_CSV="$SWEEP_DIR/sweep_summary.csv"

mkdir -p "$SWEEP_DIR"

# Check binary exists
if [[ ! -x "$BIN" ]]; then
    echo "Binary not found at $BIN — building..."
    cmake --build "$PROJECT_ROOT/build" --target grayscott_benchmark -j"$(nproc)"
fi

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  LR Sweep: Per-Output SGD vs Baseline                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  LRs        : ${LRS[*]}"
echo "  Grid       : ${L}^3"
echo "  Steps      : $STEPS"
echo "  Chunk Z    : $CHUNK_Z"
echo "  Timesteps  : $TIMESTEPS"
echo "  Weights    : $WEIGHTS"
echo "  Output     : $SWEEP_DIR"
echo ""

# ── Baseline: nn (inference-only, no SGD) ─────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Baseline: nn (inference-only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BASELINE_DIR="$SWEEP_DIR/baseline_nn"
mkdir -p "$BASELINE_DIR"

"$BIN" "$WEIGHTS" \
    --L "$L" --steps "$STEPS" --chunk-z "$CHUNK_Z" \
    --phase nn \
    2>&1 | tee "$BASELINE_DIR/log.txt"

# Copy results
cp -f "$SCRIPT_DIR/results/"*.csv "$BASELINE_DIR/" 2>/dev/null || true

# ── Sweep LRs ─────────────────────────────────────────────────────

for LR in "${LRS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  LR = $LR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    LR_DIR="$SWEEP_DIR/lr_${LR}"
    mkdir -p "$LR_DIR"

    # Run nn-rl + nn-rl+exp50 with multi-timestep
    "$BIN" "$WEIGHTS" \
        --L "$L" --steps "$STEPS" --chunk-z "$CHUNK_Z" \
        --lr "$LR" \
        --phase nn-rl --phase nn-rl+exp50 \
        --timesteps "$TIMESTEPS" \
        2>&1 | tee "$LR_DIR/log.txt"

    # Copy results
    cp -f "$SCRIPT_DIR/results/"*.csv "$LR_DIR/" 2>/dev/null || true
done

# ── Build summary CSV ─────────────────────────────────────────────
echo ""
echo "Building summary CSV..."

# Header
echo "lr,phase,ratio,write_mibps,mape_ratio_pct,mape_comp_pct,mape_decomp_pct,sgd_fires,explorations" \
    > "$SUMMARY_CSV"

# Baseline nn
if [[ -f "$BASELINE_DIR/benchmark_grayscott_vol.csv" ]]; then
    tail -n +2 "$BASELINE_DIR/benchmark_grayscott_vol.csv" | while IFS=, read -r phase rest; do
        # Extract fields: ratio(col13), write_mibps(col14), mape_ratio(col20), mape_comp(col21), mape_decomp(col22), sgd(col17), expl(col18)
        ratio=$(echo "$rest" | cut -d, -f7)
        write_mibps=$(echo "$rest" | cut -d, -f8)
        sgd=$(echo "$rest" | cut -d, -f11)
        expl=$(echo "$rest" | cut -d, -f12)
        mape_r=$(echo "$rest" | cut -d, -f13)
        mape_c=$(echo "$rest" | cut -d, -f14)
        mape_d=$(echo "$rest" | cut -d, -f15)
        echo "0.0,$phase,$ratio,$write_mibps,$mape_r,$mape_c,$mape_d,$sgd,$expl"
    done >> "$SUMMARY_CSV"
fi

# Per-LR results
for LR in "${LRS[@]}"; do
    LR_DIR="$SWEEP_DIR/lr_${LR}"
    if [[ -f "$LR_DIR/benchmark_grayscott_vol.csv" ]]; then
        tail -n +2 "$LR_DIR/benchmark_grayscott_vol.csv" | while IFS=, read -r phase rest; do
            ratio=$(echo "$rest" | cut -d, -f7)
            write_mibps=$(echo "$rest" | cut -d, -f8)
            sgd=$(echo "$rest" | cut -d, -f11)
            expl=$(echo "$rest" | cut -d, -f12)
            mape_r=$(echo "$rest" | cut -d, -f13)
            mape_c=$(echo "$rest" | cut -d, -f14)
            mape_d=$(echo "$rest" | cut -d, -f15)
            echo "$LR,$phase,$ratio,$write_mibps,$mape_r,$mape_c,$mape_d,$sgd,$expl"
        done >> "$SUMMARY_CSV"
    fi
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Sweep Complete                                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Summary     : $SUMMARY_CSV"
echo "  Per-LR dirs : $SWEEP_DIR/lr_*/"
echo "  Baseline    : $SWEEP_DIR/baseline_nn/"
echo ""

# Print summary table
echo "── Summary Table ──────────────────────────────────────────────"
printf "%-6s  %-14s  %8s  %10s  %8s  %4s  %4s\n" \
    "LR" "Phase" "Ratio" "Write MB/s" "MAPE_R%" "SGD" "Expl"
printf "%-6s  %-14s  %8s  %10s  %8s  %4s  %4s\n" \
    "------" "--------------" "--------" "----------" "--------" "----" "----"

tail -n +2 "$SUMMARY_CSV" | while IFS=, read -r lr phase ratio wmbps mape_r mape_c mape_d sgd expl; do
    printf "%-6s  %-14s  %8s  %10s  %8s  %4s  %4s\n" \
        "$lr" "$phase" "$ratio" "$wmbps" "$mape_r" "$sgd" "$expl"
done

echo ""
echo "To plot convergence for a specific LR:"
echo "  cat $SWEEP_DIR/lr_0.1/benchmark_grayscott_timesteps.csv"
