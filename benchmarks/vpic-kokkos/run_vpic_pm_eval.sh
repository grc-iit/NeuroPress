#!/bin/bash
# ============================================================
# Phase-Major VPIC Benchmark Eval Script
#
# Runs each phase independently across all timesteps, then merges
# per-phase CSVs into a combined result set for plotting.
#
# Usage:
#   bash run_vpic_pm_eval.sh
#
# Environment variables:
#   NX=254          Grid size (default: 156)
#   CHUNK_MB=16     Chunk size in MB (default: 4)
#   TIMESTEPS=100   Number of timesteps (default: 100)
#   PHASES="no-comp,zstd,nn,nn-rl,nn-rl+exp50"  (default: all 12)
#   POLICIES="balanced"  (default: balanced,ratio,speed)
#   DEBUG_NN=0      NN debug output (default: 0)
# ============================================================
set -e
set +e  # Don't exit on VPIC cleanup segfaults

# ── Configuration ──
NX=${NX:-156}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-100}
DEBUG_NN=${DEBUG_NN:-0}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
POLICIES=${POLICIES:-"balanced,ratio,speed"}

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_BIN="$SCRIPT_DIR/vpic_benchmark_deck_pm.Linux"
VPIC_DECK="$SCRIPT_DIR/vpic_benchmark_deck_phase_major.cxx"
VPIC_LD_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib"

# ── Policy configs ──
declare -A POLICY_W0 POLICY_W1 POLICY_W2 POLICY_LABELS
POLICY_W0[balanced]=1.0;  POLICY_W1[balanced]=1.0;  POLICY_W2[balanced]=1.0
POLICY_W0[ratio]=0.0;     POLICY_W1[ratio]=0.0;     POLICY_W2[ratio]=1.0
POLICY_W0[speed]=1.0;     POLICY_W1[speed]=1.0;     POLICY_W2[speed]=0.0
POLICY_LABELS[balanced]="balanced_w1-1-1"
POLICY_LABELS[ratio]="ratio_only_w0-0-1"
POLICY_LABELS[speed]="speed_only_w1-1-0"

# ── Eval directory ──
EVAL_NAME="eval_NX${NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"

# ── Verify binary ──
if [ ! -f "$VPIC_BIN" ]; then
    echo "ERROR: Phase-major binary not found: $VPIC_BIN"
    echo "Build it: bash benchmarks/vpic-kokkos/build_vpic_pm.sh && mv vpic_benchmark_deck_pm.Linux benchmarks/vpic-kokkos/"
    exit 1
fi

echo "============================================================"
echo "  VPIC Phase-Major Benchmark"
echo "============================================================"
echo "  Grid       : ${NX}^3  (~$(( (NX+2)**3 * 16 * 4 / 1024 / 1024 )) MB)"
echo "  Chunks     : ${CHUNK_MB} MB"
echo "  Timesteps  : ${TIMESTEPS}"
echo "  Phases     : ${PHASES}"
echo "  Policies   : ${POLICIES}"
echo "  Output     : ${EVAL_DIR}"
echo "============================================================"
echo ""

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
IFS=',' read -ra POLICY_LIST <<< "$POLICIES"

# ── Separate NN phases (policy-sensitive) from fixed phases (policy-invariant) ──
NN_PHASES=()
FIXED_PHASES=()
for phase in "${PHASE_LIST[@]}"; do
    case "$phase" in
        nn|nn-rl|nn-rl+exp50) NN_PHASES+=("$phase") ;;
        *) FIXED_PHASES+=("$phase") ;;
    esac
done

TOTAL=$(( ${#FIXED_PHASES[@]} + ${#NN_PHASES[@]} * ${#POLICY_LIST[@]} ))
RUN_NUM=0

# ── Run fixed phases once (policy doesn't affect algorithm choice) ──
FIXED_DIR="$EVAL_DIR/fixed_phases"
if [ ${#FIXED_PHASES[@]} -gt 0 ]; then
    mkdir -p "$FIXED_DIR"
    echo "============================================================"
    echo "  Fixed phases (run once — policy-invariant)"
    echo "============================================================"

    for phase in "${FIXED_PHASES[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        PHASE_DIR="$FIXED_DIR/phase_${phase}"
        mkdir -p "$PHASE_DIR"

        echo "  [$RUN_NUM/$TOTAL] $phase → $PHASE_DIR"

        RUN_START=$(date +%s)
        LD_LIBRARY_PATH="$VPIC_LD_PATH" \
        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        GPUCOMPRESS_RANK_W0=1.0 GPUCOMPRESS_RANK_W1=1.0 GPUCOMPRESS_RANK_W2=1.0 \
        VPIC_NX=$NX VPIC_CHUNK_MB=$CHUNK_MB VPIC_TIMESTEPS=$TIMESTEPS \
        VPIC_PHASE="$phase" \
        VPIC_RESULTS_DIR="$PHASE_DIR" \
        "$VPIC_BIN" "$VPIC_DECK" > "$PHASE_DIR/vpic_benchmark.log" 2>&1

        RUN_END=$(date +%s)
        ELAPSED=$((RUN_END - RUN_START))
        echo "    Done (${ELAPSED}s)"
    done
    echo ""
fi

# ── Run NN phases per policy (cost model weights affect algorithm selection) ──
for policy in "${POLICY_LIST[@]}"; do
    LABEL="${POLICY_LABELS[$policy]}"
    W0="${POLICY_W0[$policy]}"
    W1="${POLICY_W1[$policy]}"
    W2="${POLICY_W2[$policy]}"

    POLICY_DIR="$EVAL_DIR/$LABEL"
    mkdir -p "$POLICY_DIR"

    if [ ${#NN_PHASES[@]} -eq 0 ]; then continue; fi

    echo "============================================================"
    echo "  Policy: $LABEL (w0=$W0 w1=$W1 w2=$W2) — NN phases"
    echo "============================================================"

    for phase in "${NN_PHASES[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        PHASE_DIR="$POLICY_DIR/phase_${phase}"
        mkdir -p "$PHASE_DIR"

        echo "  [$RUN_NUM/$TOTAL] $phase → $PHASE_DIR"

        RUN_START=$(date +%s)
        LD_LIBRARY_PATH="$VPIC_LD_PATH" \
        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        GPUCOMPRESS_RANK_W0=$W0 GPUCOMPRESS_RANK_W1=$W1 GPUCOMPRESS_RANK_W2=$W2 \
        VPIC_NX=$NX VPIC_CHUNK_MB=$CHUNK_MB VPIC_TIMESTEPS=$TIMESTEPS \
        VPIC_PHASE="$phase" \
        VPIC_RESULTS_DIR="$PHASE_DIR" \
        "$VPIC_BIN" "$VPIC_DECK" > "$PHASE_DIR/vpic_benchmark.log" 2>&1

        RUN_END=$(date +%s)
        ELAPSED=$((RUN_END - RUN_START))
        echo "    Done (${ELAPSED}s)"
    done

    # ── Merge per-phase CSVs into combined files ──
    MERGE_DIR="$POLICY_DIR/merged_csv"
    mkdir -p "$MERGE_DIR"
    echo ""
    echo "  Merging CSVs for $LABEL → $MERGE_DIR"

    for csv_name in benchmark_vpic_deck_timesteps.csv benchmark_vpic_deck.csv benchmark_vpic_deck_ranking.csv benchmark_vpic_deck_ranking_costs.csv benchmark_vpic_deck_timestep_chunks.csv; do
        MERGED="$MERGE_DIR/$csv_name"
        FIRST=1
        # Merge fixed phases (from fixed_phases/) + NN phases (from policy dir)
        for phase in "${PHASE_LIST[@]}"; do
            case "$phase" in
                nn|nn-rl|nn-rl+exp50) SRC="$POLICY_DIR/phase_${phase}/$csv_name" ;;
                *) SRC="$FIXED_DIR/phase_${phase}/$csv_name" ;;
            esac
            if [ -f "$SRC" ]; then
                if [ $FIRST -eq 1 ]; then
                    cp "$SRC" "$MERGED"
                    FIRST=0
                else
                    tail -n+2 "$SRC" >> "$MERGED"
                fi
            fi
        done
        if [ -f "$MERGED" ]; then
            ROWS=$(( $(wc -l < "$MERGED") - 1 ))
            echo "    $csv_name: $ROWS rows"
            # Symlink to policy dir so visualizer finds them
            ln -sf "merged_csv/$csv_name" "$POLICY_DIR/$csv_name"
        fi
    done

    echo ""
done

echo "============================================================"
echo "  All runs complete. Results in: $EVAL_DIR"
echo "============================================================"
echo ""

# ── Generate plots per policy ──
echo "Generating figures..."
for policy in "${POLICY_LIST[@]}"; do
    LABEL="${POLICY_LABELS[$policy]}"
    POLICY_DIR="$EVAL_DIR/$LABEL"
    FIG_DIR="$POLICY_DIR/figures"
    mkdir -p "$FIG_DIR"

    # Point visualizer at the policy dir (symlinks to merged_csv)
    VPIC_NX=$NX VPIC_CHUNK=$CHUNK_MB VPIC_TS=$TIMESTEPS \
    VPIC_DIR="$POLICY_DIR" \
    python3 "$GPU_DIR/benchmarks/plots/generate_dataset_figures.py" \
        --dataset vpic --policy "$LABEL" 2>&1 | grep -E "Saved|Generated"

    # Move generated figures to the figures/ subdirectory
    PLOT_SRC="$GPU_DIR/benchmarks/results/per_dataset/vpic/$EVAL_NAME/$LABEL"
    if [ -d "$PLOT_SRC" ]; then
        mv "$PLOT_SRC"/*.png "$FIG_DIR/" 2>/dev/null
        echo "  Figures → $FIG_DIR/ ($(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l) plots)"
    fi
done

echo ""
echo "============================================================"
echo "  Results: $EVAL_DIR"
echo "  Structure per policy:"
echo "    phase_<name>/     — raw per-phase CSVs + logs"
echo "    merged_csv/       — combined CSVs (all phases)"
echo "    figures/           — plots (*.png)"
echo "============================================================"
