#!/bin/bash
# ============================================================
# Phase-Major SDRBench Benchmark Eval Script
#
# Runs each phase independently across all field files, then merges
# per-phase CSVs into a combined result set for plotting.
#
# Usage:
#   bash run_sdr_pm_eval.sh
#
# Environment variables:
#   CHUNK_MB=4          Chunk size in MB (default: 4)
#   DATASET=nyx         Dataset name (default: nyx)
#   DATA_DIR=...        Path to field files (auto from DATASET)
#   DIMS=512,512,512    Field dimensions (auto from DATASET)
#   EXT=.f32            File extension (auto from DATASET)
#   PHASES="no-comp,lz4,zstd,nn,nn-rl,nn-rl+exp50"  (default: all 12)
#   POLICIES="balanced"  (default: balanced,ratio,speed)
#   SGD_LR=0.2          SGD learning rate
#   SGD_MAPE=0.10       MAPE threshold for SGD
#   EXPLORE_K=4         Exploration alternatives
#   EXPLORE_THRESH=0.20 Exploration error threshold
#   VERIFY=1            Read-back + bitwise verify (0=skip)
#   DEBUG_NN=0          NN debug output
# ============================================================
set +e

# ── Configuration ──
CHUNK_MB=${CHUNK_MB:-4}
DEBUG_NN=${DEBUG_NN:-0}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}
DATASET=${DATASET:-"nyx"}

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
SDR_BIN="$GPU_DIR/build/generic_benchmark"
SDR_DATA_DIR="$GPU_DIR/data/sdrbench"

# ── Dataset configs (auto-resolve from DATASET name) ──
declare -A DS_SUBDIR DS_DIMS DS_EXT
DS_SUBDIR[hurricane_isabel]="hurricane_isabel/100x500x500"
DS_DIMS[hurricane_isabel]="100,500,500"
DS_EXT[hurricane_isabel]=".bin.f32"
DS_SUBDIR[nyx]="nyx/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DIMS[nyx]="512,512,512"
DS_EXT[nyx]=".f32"
DS_SUBDIR[cesm_atm]="cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600"
DS_DIMS[cesm_atm]="1800,3600"
DS_EXT[cesm_atm]=".dat"

# Use env overrides or auto from DATASET
DATA_DIR=${DATA_DIR:-"$SDR_DATA_DIR/${DS_SUBDIR[$DATASET]}"}
DIMS=${DIMS:-"${DS_DIMS[$DATASET]}"}
EXT=${EXT:-"${DS_EXT[$DATASET]}"}

# ── Policy configs ──
declare -A POLICY_W0 POLICY_W1 POLICY_W2 POLICY_LABELS
POLICY_W0[balanced]=1.0;  POLICY_W1[balanced]=1.0;  POLICY_W2[balanced]=1.0
POLICY_W0[ratio]=0.0;     POLICY_W1[ratio]=0.0;     POLICY_W2[ratio]=1.0
POLICY_W0[speed]=1.0;     POLICY_W1[speed]=1.0;     POLICY_W2[speed]=0.0
POLICY_LABELS[balanced]="balanced_w1-1-1"
POLICY_LABELS[ratio]="ratio_only_w0-0-1"
POLICY_LABELS[speed]="speed_only_w1-1-0"

# ── Eval directory ──
EVAL_NAME="eval_${DATASET}_chunk${CHUNK_MB}mb"
EVAL_DIR="$SCRIPT_DIR/results/$EVAL_NAME"

# ── Write params.txt so we know how this eval was invoked ──
mkdir -p "$EVAL_DIR"
cat > "$EVAL_DIR/params.txt" <<PARAMS_EOF
# SDRBench Phase-Major Benchmark Parameters
# Generated: $(date -Iseconds)
# Command:   $0

DATASET=$DATASET
DATA_DIR=$DATA_DIR
DIMS=$DIMS
EXT=$EXT
CHUNK_MB=$CHUNK_MB
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
if [ ! -f "$SDR_BIN" ]; then
    echo "ERROR: generic_benchmark not found: $SDR_BIN"
    echo "Build it: cmake --build build --target generic_benchmark -j\$(nproc)"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Available datasets: hurricane_isabel, nyx, cesm_atm"
    exit 1
fi

echo "============================================================"
echo "  SDRBench Phase-Major Benchmark"
echo "============================================================"
echo "  Dataset    : ${DATASET}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Dims       : ${DIMS}"
echo "  Ext        : ${EXT}"
echo "  Chunks     : ${CHUNK_MB} MB"
echo "  Phases     : ${PHASES}"
echo "  Policies   : ${POLICIES}"
echo "  Verify     : ${VERIFY}"
echo "  Output     : ${EVAL_DIR}"
echo "============================================================"
echo ""

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
IFS=',' read -ra POLICY_LIST <<< "$POLICIES"

# ── Separate NN phases from fixed phases ──
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

# ── Common args ──
COMMON_ARGS="--data-dir $DATA_DIR --dims $DIMS --ext $EXT --chunk-mb $CHUNK_MB"
VERIFY_ARG=""
[ "$VERIFY" = "0" ] && VERIFY_ARG="--no-verify"

# ── Run fixed phases once ──
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
        "$SDR_BIN" "$WEIGHTS" \
            $COMMON_ARGS \
            --phase "$phase" \
            --w0 1.0 --w1 1.0 --w2 1.0 \
            --name "$DATASET" \
            $VERIFY_ARG \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/sdr_benchmark.log" 2>&1

        RUN_END=$(date +%s)
        ELAPSED=$((RUN_END - RUN_START))
        echo "    Done (${ELAPSED}s)"
    done
    echo ""
fi

# ── Run NN phases per policy ──
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
        "$SDR_BIN" "$WEIGHTS" \
            $COMMON_ARGS \
            --phase "$phase" \
            --w0 $W0 --w1 $W1 --w2 $W2 \
            --lr $SGD_LR --mape $SGD_MAPE \
            --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
            --name "$DATASET" \
            $VERIFY_ARG \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/sdr_benchmark.log" 2>&1

        RUN_END=$(date +%s)
        ELAPSED=$((RUN_END - RUN_START))
        echo "    Done (${ELAPSED}s)"
    done

    # ── Merge per-phase CSVs ──
    MERGE_DIR="$POLICY_DIR/merged_csv"
    mkdir -p "$MERGE_DIR"
    echo ""
    echo "  Merging CSVs for $LABEL → $MERGE_DIR"

    for csv_name in benchmark_${DATASET}.csv benchmark_${DATASET}_timesteps.csv benchmark_${DATASET}_timestep_chunks.csv benchmark_${DATASET}_ranking.csv benchmark_${DATASET}_ranking_costs.csv; do
        MERGED="$MERGE_DIR/$csv_name"
        FIRST=1
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

    SDR_DIR="$EVAL_DIR" SDR_CHUNK="$CHUNK_MB" \
    python3 "$GPU_DIR/benchmarks/plots/generate_dataset_figures.py" \
        --dataset "$DATASET" --policy "$LABEL" 2>&1 | grep -E "Saved|Generated|Error"

    # Move generated figures to the figures/ subdirectory
    PLOT_SRC="$GPU_DIR/benchmarks/results/per_dataset/${DATASET}/${EVAL_NAME}/${LABEL}"
    if [ -d "$PLOT_SRC" ]; then
        mv "$PLOT_SRC"/*.png "$FIG_DIR/" 2>/dev/null
        echo "  Figures → $FIG_DIR/ ($(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l) plots)"
    else
        # Try without eval_name
        PLOT_SRC2="$GPU_DIR/benchmarks/results/per_dataset/${DATASET}/${LABEL}"
        if [ -d "$PLOT_SRC2" ]; then
            mv "$PLOT_SRC2"/*.png "$FIG_DIR/" 2>/dev/null
            echo "  Figures → $FIG_DIR/ ($(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l) plots)"
        fi
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
