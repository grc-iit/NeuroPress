#!/bin/bash
# ============================================================
# Phase-Major Gray-Scott Benchmark Eval Script
#
# Runs each phase independently across all timesteps, then merges
# per-phase CSVs into a combined result set for plotting.
#
# Usage:
#   bash run_gs_pm_eval.sh
#
# Environment variables:
#   L=400           Grid size L^3 (default: 400)
#   CHUNK_MB=4      Chunk size in MB (default: 4)
#   TIMESTEPS=100   Number of timesteps (default: 100)
#   STEPS=5000      PDE steps per timestep (default: 5000)
#   PHASES="no-comp,zstd,nn,nn-rl,nn-rl+exp50"  (default: all 12)
#   POLICIES="balanced"  (default: balanced,ratio,speed)
#   DEBUG_NN=0      NN debug output (default: 0)
# ============================================================
set +e

# ── Configuration ──
L=${L:-400}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-100}
STEPS=${STEPS:-5000}
DEBUG_NN=${DEBUG_NN:-0}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
GS_BIN="$GPU_DIR/build/grayscott_benchmark_pm"

# ── Policy configs ──
declare -A POLICY_W0 POLICY_W1 POLICY_W2 POLICY_LABELS
POLICY_W0[balanced]=1.0;  POLICY_W1[balanced]=1.0;  POLICY_W2[balanced]=1.0
POLICY_W0[ratio]=0.0;     POLICY_W1[ratio]=0.0;     POLICY_W2[ratio]=1.0
POLICY_W0[speed]=1.0;     POLICY_W1[speed]=1.0;     POLICY_W2[speed]=0.0
POLICY_LABELS[balanced]="balanced_w1-1-1"
POLICY_LABELS[ratio]="ratio_only_w0-0-1"
POLICY_LABELS[speed]="speed_only_w1-1-0"

# ── Eval directory ──
EVAL_NAME="eval_L${L}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"

# ── Write params.txt so we know how this eval was invoked ──
mkdir -p "$EVAL_DIR"
cat > "$EVAL_DIR/params.txt" <<PARAMS_EOF
# Gray-Scott Phase-Major Benchmark Parameters
# Generated: $(date -Iseconds)
# Command:   $0

L=$L
CHUNK_MB=$CHUNK_MB
TIMESTEPS=$TIMESTEPS
STEPS=$STEPS
PHASES=$PHASES
POLICIES=$POLICIES
SGD_LR=$SGD_LR
SGD_MAPE=$SGD_MAPE
EXPLORE_K=$EXPLORE_K
EXPLORE_THRESH=$EXPLORE_THRESH
VERIFY=$VERIFY
DEBUG_NN=$DEBUG_NN
WEIGHTS=$WEIGHTS
PARAMS_EOF

# ── Verify binary ──
if [ ! -f "$GS_BIN" ]; then
    echo "ERROR: GS phase-major binary not found: $GS_BIN"
    echo "Build it: cmake --build build --target grayscott_benchmark_pm"
    exit 1
fi

echo "============================================================"
echo "  Gray-Scott Phase-Major Benchmark"
echo "============================================================"
echo "  Grid       : ${L}^3  (~$(( L*L*L*4 / 1024 / 1024 )) MB)"
echo "  Chunks     : ${CHUNK_MB} MB"
echo "  Timesteps  : ${TIMESTEPS}"
echo "  PDE steps  : ${STEPS}"
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
        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
        "$GS_BIN" "$WEIGHTS" \
            --L $L --steps $STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
            --phase "$phase" \
            --w0 1.0 --w1 1.0 --w2 1.0 \
            ${VERIFY:+$([ "$VERIFY" = "0" ] && echo "--no-verify")} \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/gs_benchmark.log" 2>&1

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
        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
        "$GS_BIN" "$WEIGHTS" \
            --L $L --steps $STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
            --phase "$phase" \
            --w0 $W0 --w1 $W1 --w2 $W2 \
            --lr $SGD_LR --mape $SGD_MAPE \
            --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
            ${VERIFY:+$([ "$VERIFY" = "0" ] && echo "--no-verify")} \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/gs_benchmark.log" 2>&1

        RUN_END=$(date +%s)
        ELAPSED=$((RUN_END - RUN_START))
        echo "    Done (${ELAPSED}s)"
    done

    # ── Merge per-phase CSVs ──
    MERGE_DIR="$POLICY_DIR/merged_csv"
    mkdir -p "$MERGE_DIR"
    echo ""
    echo "  Merging CSVs for $LABEL → $MERGE_DIR"

    for csv_name in benchmark_grayscott_vol.csv benchmark_grayscott_timesteps.csv benchmark_grayscott_timestep_chunks.csv benchmark_grayscott_vol_chunks.csv benchmark_grayscott_ranking.csv benchmark_grayscott_ranking_costs.csv; do
        MERGED="$MERGE_DIR/$csv_name"
        FIRST=1
        # Merge fixed phases (from fixed_phases/) + NN phases (from policy dir)
        for phase in "${PHASE_LIST[@]}"; do
            # Fixed phases live in fixed_phases/, NN phases in policy dir
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
            ln -sf "merged_csv/$csv_name" "$POLICY_DIR/$csv_name"
        fi
    done

    # ── Generate figures ──
    FIG_DIR="$POLICY_DIR/figures"
    mkdir -p "$FIG_DIR"
    echo ""
    echo "  Generating figures..."
    GS_L=$L GS_CHUNK=$CHUNK_MB GS_TS=$TIMESTEPS \
    GS_DIR="$POLICY_DIR" \
    python3 "$GPU_DIR/benchmarks/plots/generate_dataset_figures.py" \
        --dataset gray_scott --policy "$LABEL" 2>&1 | grep -E "Saved|Generated"

    PLOT_SRC="$GPU_DIR/benchmarks/results/per_dataset/gray_scott/$EVAL_NAME/$LABEL"
    if [ -d "$PLOT_SRC" ]; then
        mv "$PLOT_SRC"/*.png "$FIG_DIR/" 2>/dev/null
        echo "  Figures → $FIG_DIR/ ($(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l) plots)"
    fi

    echo ""
done

echo "============================================================"
echo "  All runs complete. Results in: $EVAL_DIR"
echo "  Structure per policy:"
echo "    phase_<name>/     — raw per-phase CSVs + logs"
echo "    merged_csv/       — combined CSVs (all phases)"
echo "    figures/           — plots (*.png)"
echo "============================================================"
