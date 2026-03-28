#!/bin/bash
# ============================================================
# Run All Benchmarks (Gray-Scott + VPIC + SDRBench)
#
# Single entry point for all benchmarks with shared configuration.
#
# Examples:
#
#   # Run everything with defaults (512MB, 16MB chunks, 50 timesteps)
#   bash benchmarks/run_all.sh
#
#   # 1GB data, 32MB chunks, all benchmarks
#   DATA_MB=1024 CHUNK_MB=32 bash benchmarks/run_all.sh
#
#   # VPIC only, 1GB, 50 timesteps, balanced policy
#   BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=32 TIMESTEPS=50 \
#     POLICIES=balanced VERIFY=0 bash benchmarks/run_all.sh
#
#   # Gray-Scott only, quick test
#   BENCHMARKS=grayscott DATA_MB=128 CHUNK_MB=16 TIMESTEPS=5 \
#     PHASES="no-comp,lz4,nn,nn-rl" VERIFY=0 bash benchmarks/run_all.sh
#
#   # SDRBench only, NYX dataset
#   BENCHMARKS=sdrbench SDR_DATASETS=nyx CHUNK_MB=8 \
#     POLICIES=balanced VERIFY=0 bash benchmarks/run_all.sh
#
#   # All SDR datasets with specific phases
#   BENCHMARKS=sdrbench CHUNK_MB=64 \
#     PHASES="no-comp,lz4,zstd,nn,nn-rl,nn-rl+exp50" bash benchmarks/run_all.sh
#
#   # VPIC with custom physics (faster reconnection, more data variety)
#   BENCHMARKS=vpic DATA_MB=1024 VPIC_MI_ME=1 VPIC_TI_TE=5 \
#     VPIC_WARMUP_STEPS=500 VPIC_SIM_INTERVAL=20 bash benchmarks/run_all.sh
#
#   # All benchmarks, speed policy only, NN phases only
#   PHASES="nn,nn-rl,nn-rl+exp50" POLICIES=speed \
#     DATA_MB=512 CHUNK_MB=16 TIMESTEPS=50 bash benchmarks/run_all.sh
#
#   # Override grid sizes directly (ignore DATA_MB)
#   GS_L=640 VPIC_NX=254 CHUNK_MB=64 TIMESTEPS=100 bash benchmarks/run_all.sh
#
# ── Environment Variables ──────────────────────────────────
#
#   Variable            Default                 Description
#   ------------------- ----------------------- -----------------------------------
#   BENCHMARKS          grayscott,vpic,sdrbench Which to run (comma-separated)
#   DATA_MB             512                     Target per-snapshot size in MB
#   CHUNK_MB            16                      Chunk size in MB
#   TIMESTEPS           50                      Number of benchmark write cycles
#   POLICIES            balanced,ratio,speed    Cost model policies
#   PHASES              (all 12)                Compression phases to run
#   VERIFY              1                       Bitwise verify (0=skip)
#   DEBUG_NN            0                       NN debug output
#
#   NN hyperparameters:
#   SGD_LR              0.2                     SGD learning rate
#   SGD_MAPE            0.10                    MAPE threshold for SGD
#   EXPLORE_K           4                       Exploration alternatives
#   EXPLORE_THRESH      0.20                    Exploration error threshold
#
#   Gray-Scott specific:
#   GS_L                (auto from DATA_MB)     Grid size (L^3 * 4 bytes)
#   GS_STEPS            500                     PDE sim steps per timestep
#
#   VPIC specific:
#   VPIC_NX             (auto from DATA_MB)     Grid size ((NX+2)^3 * 64 bytes)
#   VPIC_NPPC           2                       Particles per cell
#   VPIC_WARMUP_STEPS   500                     Sim steps before benchmarking
#   VPIC_PERTURBATION   0.1                     Bx tearing mode seed (fraction of b0)
#   VPIC_GUIDE_FIELD    0.0                     By guide field (fraction of b0, 0.2 for 3D)
#   VPIC_SIM_INTERVAL   190                     Sim steps between benchmark writes
#   VPIC_MI_ME          5                       Ion/electron mass ratio (lower=faster reconnection)
#   VPIC_WPE_WCE        1                       Plasma/cyclotron freq ratio (lower=stronger B field)
#   VPIC_TI_TE          5                       Ion/electron temp ratio (higher=more free energy)
#
#   SDRBench specific:
#   SDR_DATASETS        nyx,hurricane_isabel,cesm_atm  Datasets to run
#
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Shared defaults ──
BENCHMARKS=${BENCHMARKS:-"grayscott,vpic,sdrbench"}
DATA_MB=${DATA_MB:-512}
CHUNK_MB=${CHUNK_MB:-16}
TIMESTEPS=${TIMESTEPS:-50}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
DEBUG_NN=${DEBUG_NN:-0}

# ── GS-specific defaults ──
GS_STEPS=${GS_STEPS:-500}

# ── Auto-compute grid sizes from DATA_MB if not explicitly set ──
# Gray-Scott: L^3 * 4 bytes = DATA_MB * 1024^2
# VPIC: (NX+2)^3 * 16 fields * 4 bytes = (NX+2)^3 * 64
if [ -z "$GS_L" ]; then
    # L = cube_root(DATA_MB * 1024 * 1024 / 4)
    GS_L=$(python3 -c "import math; print(int(round(math.pow($DATA_MB * 1024 * 1024 / 4, 1/3))))")
fi
if [ -z "$VPIC_NX" ]; then
    # VPIC writes (NX+2)^3 * 16 fields * 4 bytes = (NX+2)^3 * 64
    # NX = cube_root(DATA_MB * 1024 * 1024 / 64) - 2
    VPIC_NX=$(python3 -c "import math; print(int(round(math.pow($DATA_MB * 1024 * 1024 / 64, 1/3))) - 2)")
fi

GS_DATA=$(( GS_L * GS_L * GS_L * 4 / 1024 / 1024 ))
VPIC_DATA=$(( (VPIC_NX+2) * (VPIC_NX+2) * (VPIC_NX+2) * 64 / 1024 / 1024 ))

echo "============================================================"
echo "  GPUCompress Benchmark Suite"
echo "============================================================"
echo "  Benchmarks  : ${BENCHMARKS}"
echo "  Target data : ${DATA_MB} MB"
echo "  Chunk size  : ${CHUNK_MB} MB"
echo "  Timesteps   : ${TIMESTEPS}"
echo "  Policies    : ${POLICIES}"
echo "  Phases      : ${PHASES}"
echo ""
echo "  NN hyperparameters:"
echo "    SGD_LR=${SGD_LR}  SGD_MAPE=${SGD_MAPE}"
echo "    EXPLORE_K=${EXPLORE_K}  EXPLORE_THRESH=${EXPLORE_THRESH}"
echo ""
echo "  Gray-Scott  : L=${GS_L} (~${GS_DATA} MB), steps=${GS_STEPS}"
echo "  VPIC        : NX=${VPIC_NX} (~${VPIC_DATA} MB), warmup=${VPIC_WARMUP_STEPS:-500}, interval=${VPIC_SIM_INTERVAL:-190}"
echo "  SDRBench    : datasets=${SDR_DATASETS:-nyx,hurricane_isabel,cesm_atm}"
echo "============================================================"
echo ""

IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

for bench in "${BENCH_LIST[@]}"; do
    case "$bench" in
        grayscott)
            echo ""
            echo ">>> Running Gray-Scott benchmark..."
            echo ""

            GS_BIN="$SCRIPT_DIR/../build/grayscott_benchmark_pm"
            GS_WEIGHTS="$SCRIPT_DIR/../neural_net/weights/model.nnwt"
            GS_EVAL_DIR="$SCRIPT_DIR/grayscott/results/eval_L${GS_L}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"

            VERIFY_ARG=""
            [ "$VERIFY" = "0" ] && VERIFY_ARG="--no-verify"

            # Split phases into fixed (policy-independent) and NN (policy-dependent)
            GS_FIXED_ARGS=""
            GS_NN_ARGS=""
            GS_FIXED_LIST=""
            GS_NN_LIST=""
            IFS=',' read -ra _PH <<< "$PHASES"
            for ph in "${_PH[@]}"; do
                case "$ph" in
                    nn|nn-rl|nn-rl+exp50)
                        GS_NN_ARGS="$GS_NN_ARGS --phase $ph"
                        GS_NN_LIST="${GS_NN_LIST:+$GS_NN_LIST,}$ph"
                        ;;
                    *)
                        GS_FIXED_ARGS="$GS_FIXED_ARGS --phase $ph"
                        GS_FIXED_LIST="${GS_FIXED_LIST:+$GS_FIXED_LIST,}$ph"
                        ;;
                esac
            done

            echo "  L=$GS_L (~${GS_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}, steps=${GS_STEPS}"
            echo "  fixed phases: ${GS_FIXED_LIST:-none}"
            echo "  NN phases:    ${GS_NN_LIST:-none}"
            echo ""

            # ── Fixed phases (run once, policy-independent) ──
            if [ -n "$GS_FIXED_ARGS" ]; then
                GS_FIXED_DIR="$GS_EVAL_DIR/fixed_phases"
                mkdir -p "$GS_FIXED_DIR"

                echo "  >>> Fixed phases (run once): $GS_FIXED_LIST"

                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                "$GS_BIN" "$GS_WEIGHTS" \
                    --L $GS_L --steps $GS_STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
                    $GS_FIXED_ARGS \
                    --w0 1.0 --w1 1.0 --w2 1.0 \
                    --lr $SGD_LR --mape $SGD_MAPE \
                    --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                    $VERIFY_ARG \
                    --out-dir "$GS_FIXED_DIR" \
                    > "$GS_FIXED_DIR/gs_benchmark.log" 2>&1

                echo "      Done. Log: $GS_FIXED_DIR/gs_benchmark.log"
                echo ""
            fi

            # ── NN phases (run per policy) ──
            if [ -n "$GS_NN_ARGS" ]; then
                IFS=',' read -ra _POLICIES <<< "$POLICIES"
                POL_IDX=0
                POL_TOTAL=${#_POLICIES[@]}

                for pol in "${_POLICIES[@]}"; do
                    POL_IDX=$((POL_IDX + 1))
                    case "$pol" in
                        balanced) W0=1.0; W1=1.0; W2=1.0; LABEL="balanced_w1-1-1" ;;
                        ratio)    W0=0.0; W1=0.0; W2=1.0; LABEL="ratio_only_w0-0-1" ;;
                        speed)    W0=1.0; W1=1.0; W2=0.0; LABEL="speed_only_w1-1-0" ;;
                        *)        W0=1.0; W1=1.0; W2=1.0; LABEL="$pol" ;;
                    esac

                    GS_NN_DIR="$GS_EVAL_DIR/$LABEL"
                    mkdir -p "$GS_NN_DIR"

                    echo "  >>> NN phases [$LABEL] ($POL_IDX/$POL_TOTAL): $GS_NN_LIST"

                    GPUCOMPRESS_DETAILED_TIMING=1 \
                    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                    "$GS_BIN" "$GS_WEIGHTS" \
                        --L $GS_L --steps $GS_STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
                        $GS_NN_ARGS \
                        --w0 $W0 --w1 $W1 --w2 $W2 \
                        --lr $SGD_LR --mape $SGD_MAPE \
                        --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                        $VERIFY_ARG \
                        --out-dir "$GS_NN_DIR" \
                        > "$GS_NN_DIR/gs_benchmark.log" 2>&1

                    echo "      Done. Log: $GS_NN_DIR/gs_benchmark.log"

                    # Merge fixed + NN CSVs
                    if [ -n "$GS_FIXED_LIST" ] && [ -d "$GS_FIXED_DIR" ]; then
                        MERGE_DIR="$GS_NN_DIR/merged_csv"
                        mkdir -p "$MERGE_DIR"
                        for csv_name in benchmark_grayscott_vol.csv benchmark_grayscott_timesteps.csv benchmark_grayscott_timestep_chunks.csv benchmark_grayscott_ranking.csv benchmark_grayscott_ranking_costs.csv benchmark_grayscott_vol_chunks.csv; do
                            MERGED="$MERGE_DIR/$csv_name"
                            FIRST=1
                            for src in "$GS_FIXED_DIR/$csv_name" "$GS_NN_DIR/$csv_name"; do
                                if [ -f "$src" ]; then
                                    if [ $FIRST -eq 1 ]; then
                                        cp "$src" "$MERGED"
                                        FIRST=0
                                    else
                                        tail -n+2 "$src" >> "$MERGED"
                                    fi
                                fi
                            done
                            [ -f "$MERGED" ] && ln -sf "$MERGE_DIR/$csv_name" "$GS_NN_DIR/$csv_name"
                        done
                    fi

                    GS_DIR="$GS_NN_DIR" \
                    python3 "$SCRIPT_DIR/plots/generate_dataset_figures.py" \
                        --dataset gray_scott --policy "$LABEL" 2>&1 | grep -E "Generated"
                    echo ""
                done
            fi
            ;;
        vpic)
            echo ""
            echo ">>> Running VPIC benchmark (single invocation, GPU weight isolation)..."
            echo ""

            VPIC_BIN="$SCRIPT_DIR/vpic-kokkos/vpic_benchmark_deck.Linux"
            VPIC_DECK="$SCRIPT_DIR/vpic-kokkos/vpic_benchmark_deck.cxx"
            VPIC_WEIGHTS="$SCRIPT_DIR/../neural_net/weights/model.nnwt"
            VPIC_LD_PATH="/tmp/hdf5-install/lib:$SCRIPT_DIR/../build:/tmp/lib"

            # VPIC physics: wpe_wce=1 + Ti_Te=5 gives best compression diversity
            # (2x field range, highest throughput, same runtime as default)
            VPIC_MI_ME=${VPIC_MI_ME:-5}
            VPIC_WPE_WCE=${VPIC_WPE_WCE:-1}
            VPIC_TI_TE=${VPIC_TI_TE:-5}
            VPIC_NPPC=${VPIC_NPPC:-2}
            VPIC_WARMUP=${VPIC_WARMUP_STEPS:-500}
            VPIC_SIM_INT=${VPIC_SIM_INTERVAL:-190}

            # Split phases into fixed (policy-independent) and NN (policy-dependent)
            VPIC_FIXED_PHASES=""
            VPIC_NN_PHASES=""
            IFS=',' read -ra _PH <<< "$PHASES"
            for ph in "${_PH[@]}"; do
                case "$ph" in
                    nn|nn-rl|nn-rl+exp50) VPIC_NN_PHASES="${VPIC_NN_PHASES:+$VPIC_NN_PHASES,}$ph" ;;
                    *) VPIC_FIXED_PHASES="${VPIC_FIXED_PHASES:+$VPIC_FIXED_PHASES,}$ph" ;;
                esac
            done

            VPIC_EVAL_DIR="$SCRIPT_DIR/vpic-kokkos/results/eval_NX${VPIC_NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"

            # Helper: build VPIC_EXCLUDE from a comma-separated list of phases to INCLUDE
            vpic_exclude_from() {
                local INCLUDE="$1"
                local ALL="no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"
                local EXCL=""
                IFS=',' read -ra _A <<< "$ALL"
                for a in "${_A[@]}"; do
                    if ! echo ",$INCLUDE," | grep -q ",$a,"; then
                        EXCL="${EXCL:+$EXCL,}$a"
                    fi
                done
                echo "$EXCL"
            }

            echo "  NX=$VPIC_NX (~${VPIC_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}"
            echo "  warmup=$VPIC_WARMUP, sim_interval=$VPIC_SIM_INT"
            echo "  physics: mi_me=$VPIC_MI_ME wpe_wce=$VPIC_WPE_WCE Ti_Te=$VPIC_TI_TE pert=${VPIC_PERTURBATION:-0.1}"
            echo "  fixed phases: ${VPIC_FIXED_PHASES:-none}"
            echo "  NN phases:    ${VPIC_NN_PHASES:-none}"
            echo ""

            # ── Step 1: Fixed phases (run ONCE, policy-independent) ──
            if [ -n "$VPIC_FIXED_PHASES" ]; then
                FIXED_DIR="$VPIC_EVAL_DIR/fixed_phases"
                mkdir -p "$FIXED_DIR"
                FIXED_EXCLUDE=$(vpic_exclude_from "$VPIC_FIXED_PHASES")

                echo "  >>> Fixed phases (run once): $VPIC_FIXED_PHASES"

                LD_LIBRARY_PATH="$VPIC_LD_PATH" \
                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_WEIGHTS="$VPIC_WEIGHTS" \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                VPIC_NX=$VPIC_NX VPIC_NPPC=$VPIC_NPPC \
                VPIC_MI_ME=$VPIC_MI_ME VPIC_WPE_WCE=$VPIC_WPE_WCE VPIC_TI_TE=$VPIC_TI_TE \
                VPIC_PERTURBATION=${VPIC_PERTURBATION:-0.1} VPIC_GUIDE_FIELD=${VPIC_GUIDE_FIELD:-0.0} \
                VPIC_WARMUP_STEPS=$VPIC_WARMUP VPIC_TIMESTEPS=$TIMESTEPS VPIC_SIM_INTERVAL=$VPIC_SIM_INT \
                VPIC_CHUNK_MB=$CHUNK_MB VPIC_VERIFY=$VERIFY \
                VPIC_EXCLUDE="$FIXED_EXCLUDE" \
                VPIC_RESULTS_DIR="$FIXED_DIR" \
                VPIC_W0=1.0 VPIC_W1=1.0 VPIC_W2=1.0 \
                VPIC_LR=$SGD_LR VPIC_MAPE_THRESHOLD=$SGD_MAPE \
                VPIC_EXPLORE_K=$EXPLORE_K VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
                "$VPIC_BIN" "$VPIC_DECK" \
                > "$FIXED_DIR/vpic_benchmark.log" 2>&1

                echo "      Done. Log: $FIXED_DIR/vpic_benchmark.log"
                echo ""
            fi

            # ── Step 2: NN phases (run per policy) ──
            if [ -n "$VPIC_NN_PHASES" ]; then
                NN_EXCLUDE=$(vpic_exclude_from "$VPIC_NN_PHASES")

                IFS=',' read -ra _POLICIES <<< "$POLICIES"
                POL_IDX=0
                POL_TOTAL=${#_POLICIES[@]}

                for pol in "${_POLICIES[@]}"; do
                    POL_IDX=$((POL_IDX + 1))
                    case "$pol" in
                        balanced) W0=1.0; W1=1.0; W2=1.0; LABEL="balanced_w1-1-1" ;;
                        ratio)    W0=0.0; W1=0.0; W2=1.0; LABEL="ratio_only_w0-0-1" ;;
                        speed)    W0=1.0; W1=1.0; W2=0.0; LABEL="speed_only_w1-1-0" ;;
                        *)        W0=1.0; W1=1.0; W2=1.0; LABEL="$pol" ;;
                    esac

                    NN_DIR="$VPIC_EVAL_DIR/$LABEL"
                    mkdir -p "$NN_DIR"

                    echo "  >>> NN phases [$LABEL] ($POL_IDX/$POL_TOTAL): $VPIC_NN_PHASES"

                    LD_LIBRARY_PATH="$VPIC_LD_PATH" \
                    GPUCOMPRESS_DETAILED_TIMING=1 \
                    GPUCOMPRESS_WEIGHTS="$VPIC_WEIGHTS" \
                    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                    VPIC_NX=$VPIC_NX VPIC_NPPC=$VPIC_NPPC \
                    VPIC_MI_ME=$VPIC_MI_ME VPIC_WPE_WCE=$VPIC_WPE_WCE VPIC_TI_TE=$VPIC_TI_TE \
                    VPIC_PERTURBATION=${VPIC_PERTURBATION:-0.1} VPIC_GUIDE_FIELD=${VPIC_GUIDE_FIELD:-0.0} \
                    VPIC_WARMUP_STEPS=$VPIC_WARMUP VPIC_TIMESTEPS=$TIMESTEPS VPIC_SIM_INTERVAL=$VPIC_SIM_INT \
                    VPIC_CHUNK_MB=$CHUNK_MB VPIC_VERIFY=$VERIFY \
                    VPIC_EXCLUDE="$NN_EXCLUDE" \
                    VPIC_RESULTS_DIR="$NN_DIR" \
                    VPIC_W0=$W0 VPIC_W1=$W1 VPIC_W2=$W2 \
                    VPIC_LR=$SGD_LR VPIC_MAPE_THRESHOLD=$SGD_MAPE \
                    VPIC_EXPLORE_K=$EXPLORE_K VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
                    "$VPIC_BIN" "$VPIC_DECK" \
                    > "$NN_DIR/vpic_benchmark.log" 2>&1

                    echo "      Done. Log: $NN_DIR/vpic_benchmark.log"

                    # Merge fixed + NN CSVs for this policy and generate plots
                    if [ -n "$VPIC_FIXED_PHASES" ] && [ -d "$FIXED_DIR" ]; then
                        MERGE_DIR="$NN_DIR/merged_csv"
                        mkdir -p "$MERGE_DIR"
                        for csv_name in benchmark_vpic_deck_timesteps.csv benchmark_vpic_deck_timestep_chunks.csv benchmark_vpic_deck_ranking.csv benchmark_vpic_deck_ranking_costs.csv; do
                            MERGED="$MERGE_DIR/$csv_name"
                            FIRST=1
                            for src in "$FIXED_DIR/$csv_name" "$NN_DIR/$csv_name"; do
                                if [ -f "$src" ]; then
                                    if [ $FIRST -eq 1 ]; then
                                        cp "$src" "$MERGED"
                                        FIRST=0
                                    else
                                        tail -n+2 "$src" >> "$MERGED"
                                    fi
                                fi
                            done
                            [ -f "$MERGED" ] && ln -sf "$MERGE_DIR/$csv_name" "$NN_DIR/$csv_name"
                        done
                    fi

                    VPIC_DIR="$NN_DIR" \
                    python3 "$SCRIPT_DIR/plots/generate_dataset_figures.py" \
                        --dataset vpic --policy "$LABEL" 2>&1 | grep -E "Generated"
                    echo ""
                done
            fi
            ;;
        sdrbench)
            echo ""
            echo ">>> Running SDRBench benchmark (all datasets)..."
            echo ""
            SDR_DATASETS=${SDR_DATASETS:-"nyx,hurricane_isabel,cesm_atm"} \
            CHUNK_MB=$CHUNK_MB \
            PHASES=$PHASES \
            POLICIES=$POLICIES \
            SGD_LR=$SGD_LR \
            SGD_MAPE=$SGD_MAPE \
            EXPLORE_K=$EXPLORE_K \
            EXPLORE_THRESH=$EXPLORE_THRESH \
            VERIFY=$VERIFY \
            DEBUG_NN=$DEBUG_NN \
            bash "$SCRIPT_DIR/sdrbench/run_all_sdr.sh"
            ;;
        *)
            echo "ERROR: Unknown benchmark '$bench'. Use: grayscott, vpic, sdrbench"
            ;;
    esac
done

echo ""
echo "============================================================"
echo "  All benchmarks complete."
echo "============================================================"
