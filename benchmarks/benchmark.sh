#!/bin/bash
# ============================================================
# GPUCompress Benchmark Suite
#
# Single entry point for all benchmarks: Gray-Scott, VPIC, SDRBench.
# Fixed phases run once; NN phases run per policy with GPU weight isolation.
#
# Usage:
#   bash benchmarks/benchmark.sh [--help]
#
# ============================================================

show_help() {
cat << 'HELP'
GPUCompress Benchmark Suite
===========================

Usage: bash benchmarks/benchmark.sh

All configuration is via environment variables. Defaults shown in [brackets].

GENERAL
  BENCHMARKS          [grayscott,vpic,sdrbench]  Which benchmarks to run
  DATA_MB             [512]                      Per-snapshot data size in MB
  CHUNK_MB            [16]                       HDF5 chunk size in MB
  TIMESTEPS           [50]                       Number of benchmark write cycles
  POLICIES            [balanced,ratio,speed]      Cost model policies (NN phases only)
  PHASES              [all 12]                   Compression phases to run
                        Fixed: no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp
                        NN:    nn,nn-rl,nn-rl+exp50
  VERIFY              [1]                        Bitwise verify (0=skip for speed)
  DEBUG_NN            [0]                        NN debug output

NN HYPERPARAMETERS
  SGD_LR              [0.2]                      SGD learning rate
  SGD_MAPE            [0.10]                     MAPE threshold for SGD updates
  EXPLORE_K           [4]                        Number of exploration alternatives
  EXPLORE_THRESH      [0.20]                     Cost error threshold for exploration

GRAY-SCOTT (GPU reaction-diffusion simulation)
  GS_L                [auto]                     Grid size L (L^3 voxels, 4 bytes each)
  GS_STEPS            [500]                      PDE simulation steps per timestep

VPIC (GPU plasma particle-in-cell simulation)
  VPIC_NX             [auto]                     Grid size NX ((NX+2)^3 * 64 bytes)
  VPIC_NPPC           [2]                        Particles per cell
  VPIC_WARMUP_STEPS   [500]                      Simulation steps before benchmarking
  VPIC_SIM_INTERVAL   [190]                      Simulation steps between benchmark writes
  VPIC_MI_ME          [5]                        Ion/electron mass ratio (lower=faster)
  VPIC_WPE_WCE        [1]                        Plasma/cyclotron freq ratio (lower=stronger B)
  VPIC_TI_TE          [5]                        Ion/electron temp ratio (higher=more energy)
  VPIC_PERTURBATION   [0.1]                      Bx tearing mode seed (0=none, 0.1=10% of B0)
  VPIC_GUIDE_FIELD    [0.0]                      By guide field (0.2-0.5 for 3D structure)
  VPIC_DUMP_FIELDS    [0]                        Dump raw field binary per timestep

SDRBENCH (static scientific datasets)
  SDR_DATASETS        [nyx,hurricane_isabel,cesm_atm]  Datasets to run

EXAMPLES
  # Quick smoke test
  BENCHMARKS=vpic DATA_MB=128 CHUNK_MB=16 TIMESTEPS=3 VERIFY=0 bash benchmarks/benchmark.sh

  # VPIC 1GB, all phases, balanced policy
  BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=32 TIMESTEPS=10 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh

  # Gray-Scott 512MB with selected phases
  BENCHMARKS=grayscott DATA_MB=512 PHASES="no-comp,lz4,zstd,nn,nn-rl" bash benchmarks/benchmark.sh

  # SDRBench NYX only
  BENCHMARKS=sdrbench SDR_DATASETS=nyx CHUNK_MB=8 VERIFY=0 bash benchmarks/benchmark.sh

  # All benchmarks, all policies, 1GB
  DATA_MB=1024 CHUNK_MB=32 TIMESTEPS=50 bash benchmarks/benchmark.sh

  # VPIC with custom physics
  BENCHMARKS=vpic DATA_MB=1024 VPIC_MI_ME=1 VPIC_TI_TE=5 VPIC_WARMUP_STEPS=200 bash benchmarks/benchmark.sh

OUTPUT
  CSVs:   benchmarks/<benchmark>/results/eval_<params>/<policy>/
  Plots:  benchmarks/results/per_dataset/<dataset>/eval_<params>/<policy>/
  Logs:   benchmarks/<benchmark>/results/eval_<params>/<policy>/*_benchmark.log
HELP
exit 0
}

# Show help if --help is passed
[[ "$1" == "--help" || "$1" == "-h" ]] && show_help

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
                    > "$GS_FIXED_DIR/gs_benchmark.log" 2> >(tee -a "$GS_FIXED_DIR/gs_benchmark.log" >&2)

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
                        > "$GS_NN_DIR/gs_benchmark.log" 2> >(tee -a "$GS_NN_DIR/gs_benchmark.log" >&2)

                    echo "      Done. Log: $GS_NN_DIR/gs_benchmark.log"

                    # Merge fixed + NN CSVs.
                    # Build merged file in temp, then overwrite the NN dir copy.
                    if [ -n "$GS_FIXED_LIST" ] && [ -d "$GS_FIXED_DIR" ]; then
                        for csv_name in benchmark_grayscott_vol.csv benchmark_grayscott_timesteps.csv benchmark_grayscott_timestep_chunks.csv benchmark_grayscott_ranking.csv benchmark_grayscott_ranking_costs.csv; do
                            FIXED_SRC="$GS_FIXED_DIR/$csv_name"
                            NN_SRC="$GS_NN_DIR/$csv_name"
                            if [ -f "$FIXED_SRC" ] && [ -f "$NN_SRC" ]; then
                                TMP_MERGED=$(mktemp)
                                cp "$FIXED_SRC" "$TMP_MERGED"
                                tail -n+2 "$NN_SRC" >> "$TMP_MERGED"
                                mv "$TMP_MERGED" "$NN_SRC"
                            elif [ -f "$FIXED_SRC" ] && [ ! -f "$NN_SRC" ]; then
                                cp "$FIXED_SRC" "$NN_SRC"
                            fi
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

            VPIC_EVAL_DIR="$SCRIPT_DIR/vpic-kokkos/results/eval_NX${VPIC_NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}${VPIC_EVAL_SUFFIX:-}"

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

            # Helper: run VPIC binary
            vpic_run() {
                local RESULTS_DIR="$1" INCLUDE_PHASES="$2"
                local EXCL=$(vpic_exclude_from "$INCLUDE_PHASES")
                mkdir -p "$RESULTS_DIR"

                LD_LIBRARY_PATH="$VPIC_LD_PATH" \
                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_WEIGHTS="$VPIC_WEIGHTS" \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                VPIC_NX=$VPIC_NX VPIC_NPPC=$VPIC_NPPC \
                VPIC_MI_ME=$VPIC_MI_ME VPIC_WPE_WCE=$VPIC_WPE_WCE VPIC_TI_TE=$VPIC_TI_TE \
                VPIC_PERTURBATION=${VPIC_PERTURBATION:-0.1} VPIC_GUIDE_FIELD=${VPIC_GUIDE_FIELD:-0.0} \
                VPIC_ERROR_BOUND=${VPIC_ERROR_BOUND:-0.0} \
                VPIC_WARMUP_STEPS=$VPIC_WARMUP VPIC_TIMESTEPS=$TIMESTEPS VPIC_SIM_INTERVAL=$VPIC_SIM_INT \
                VPIC_CHUNK_MB=$CHUNK_MB VPIC_VERIFY=$VERIFY \
                VPIC_EXCLUDE="$EXCL" \
                VPIC_RESULTS_DIR="$RESULTS_DIR" \
                VPIC_POLICIES="$POLICIES" \
                VPIC_LR=$SGD_LR VPIC_MAPE_THRESHOLD=$SGD_MAPE \
                VPIC_EXPLORE_K=$EXPLORE_K VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
                "$VPIC_BIN" "$VPIC_DECK" \
                > "$RESULTS_DIR/vpic_benchmark.log" 2> >(tee -a "$RESULTS_DIR/vpic_benchmark.log" >&2)
            }

            echo "  NX=$VPIC_NX (~${VPIC_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}"
            echo "  warmup=$VPIC_WARMUP, sim_interval=$VPIC_SIM_INT"
            echo "  physics: mi_me=$VPIC_MI_ME wpe_wce=$VPIC_WPE_WCE Ti_Te=$VPIC_TI_TE pert=${VPIC_PERTURBATION:-0.1}"
            echo "  policies: $POLICIES"
            echo ""

            # ── Single invocation: all phases + all policies ──
            # Fixed phases run once. NN phases run once per policy.
            # The VPIC binary handles multi-policy internally via VPIC_POLICIES.
            VPIC_RESULTS="$VPIC_EVAL_DIR"
            mkdir -p "$VPIC_RESULTS"

            echo "  >>> All phases, all policies in single invocation"

            vpic_run "$VPIC_RESULTS" "$PHASES"

            echo "      Done. Log: $VPIC_RESULTS/vpic_benchmark.log"

            # ── Split results into per-policy directories ──
            # Each policy dir gets: fixed phase rows + that policy's NN rows
            IFS=',' read -ra _POLICIES <<< "$POLICIES"
            for pol in "${_POLICIES[@]}"; do
                case "$pol" in
                    balanced) LABEL="balanced_w1-1-1" ;;
                    ratio)    LABEL="ratio_only_w0-0-1" ;;
                    speed)    LABEL="speed_only_w1-1-0" ;;
                    *)        LABEL="$pol" ;;
                esac

                POL_DIR="$VPIC_RESULTS/$LABEL"
                mkdir -p "$POL_DIR"

                # For each CSV: extract fixed rows + this policy's NN rows
                for csv_name in benchmark_vpic_deck_timesteps.csv benchmark_vpic_deck_timestep_chunks.csv benchmark_vpic_deck_ranking.csv benchmark_vpic_deck_ranking_costs.csv; do
                    SRC="$VPIC_RESULTS/$csv_name"
                    [ -f "$SRC" ] || continue
                    DST="$POL_DIR/$csv_name"

                    # Header + fixed rows (no "/" in phase) + this policy's NN rows (strip suffix)
                    head -1 "$SRC" > "$DST"
                    tail -n+2 "$SRC" | awk -F',' -v pol="/$pol" '{
                        phase=$1;
                        if (index(phase, "/") == 0) print;
                        else if (index(phase, pol) > 0) {
                            sub(pol, "", phase);
                            $1 = phase;
                            print;
                        }
                    }' OFS=',' >> "$DST"
                done

                # Regenerate aggregate CSV from the per-policy timestep CSV
                # (the binary's aggregate mixes all policies — we need per-policy)
                TS_CSV="$POL_DIR/benchmark_vpic_deck_timesteps.csv"
                AGG_CSV="$POL_DIR/benchmark_vpic_deck.csv"
                if [ -f "$TS_CSV" ]; then
                    python3 -c "
import csv, sys
rows = list(csv.DictReader(open('$TS_CSV')))
phases = {}
for r in rows:
    p = r['phase']
    if p not in phases:
        phases[p] = {'sum_wr':0,'sum_rd':0,'sum_rat':0,'sum_file_bytes':0,'n':0,'n_chunks':0}
    d = phases[p]
    d['sum_wr'] += float(r.get('write_ms',0))
    d['sum_rd'] += float(r.get('read_ms',0))
    d['sum_rat'] += float(r.get('ratio',0))
    d['sum_file_bytes'] += int(r.get('file_bytes',0))
    d['n_chunks'] = int(r.get('n_chunks',0))
    d['n'] += 1
with open('$AGG_CSV','w') as f:
    f.write('source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,file_mib,orig_mib,ratio,write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks\n')
    for p,d in phases.items():
        n = d['n']
        wr = d['sum_wr']/n if n else 0
        rd = d['sum_rd']/n if n else 0
        rat = d['sum_rat']/n if n else 0
        avg_file_mib = d['sum_file_bytes'] / n / (1024*1024) if n else 0
        orig_mib = avg_file_mib * rat if rat > 0 else avg_file_mib
        wr_mibps = orig_mib / (wr/1000.0) if wr > 0 else 0
        rd_mibps = orig_mib / (rd/1000.0) if rd > 0 else 0
        f.write(f'vpic,{p},{n},{wr:.2f},0.00,{rd:.2f},0.00,{avg_file_mib:.2f},{orig_mib:.2f},{rat:.4f},{wr_mibps:.1f},{rd_mibps:.1f},0,0,0,{d[\"n_chunks\"]}\n')
" 2>/dev/null
                fi

                echo "  [$LABEL] Split CSVs → $POL_DIR/"

                # Generate plots for this policy
                VPIC_DIR="$POL_DIR" \
                python3 "$SCRIPT_DIR/plots/generate_dataset_figures.py" \
                    --dataset vpic --policy "$LABEL" 2>&1 | grep -E "Generated"
            done
            echo ""
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
