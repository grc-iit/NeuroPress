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
                        Options: grayscott, vpic, sdrbench, ai_training
  DATA_MB             [512]                      Per-snapshot data size in MB
  CHUNK_MB            [16]                       HDF5 chunk size in MB
  TIMESTEPS           [50]                       Number of benchmark write cycles
  POLICIES            [balanced,ratio,speed]      Cost model policies (NN phases only)
  PHASES              [all 12]                   Compression phases to run
                        Fixed: no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp
                        NN:    nn,nn-rl,nn-rl+exp50
  VERIFY              [1]                        Bitwise verify (0=skip for speed)
  DEBUG_NN            [0]                        NN debug output

MULTI-GPU / MULTI-NODE
  MPI_NP              [1]                        Total MPI ranks (1 = single-process)
  GPUS_PER_NODE       [1]                        GPUs per node (for CUDA_VISIBLE_DEVICES mapping)
  LAUNCHER            [auto]                     auto (detect SLURM), srun, or mpirun

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

AI_TRAINING (neural network checkpoint compression)
  AI_MODEL            [vit_b_16]                   Model: vit_b_16 (327MB), gpt2 (473MB)
  AI_DATASET          [cifar10]                    Training dataset (cifar10 for ViT, wikitext2 for GPT-2)
  AI_CHECKPOINT_DIR   [auto]                       Path to exported .f32 checkpoint files
                        Generate with: python3 scripts/train_and_export_checkpoints.py

STANDALONE BENCHMARKS
  # ── 1. Gray-Scott (GPU reaction-diffusion simulation) ──
  BENCHMARKS=grayscott DATA_MB=512 CHUNK_MB=4 TIMESTEPS=25 VERIFY=0 bash benchmarks/benchmark.sh

  # ── 2. VPIC (GPU plasma particle-in-cell simulation) ──
  BENCHMARKS=vpic DATA_MB=512 CHUNK_MB=4 TIMESTEPS=25 VERIFY=0 bash benchmarks/benchmark.sh

  # ── 3. SDRBench (static scientific datasets from disk) ──
  # Available: nyx (512MB/field), hurricane_isabel (100MB/field), cesm_atm (25MB/field)
  BENCHMARKS=sdrbench SDR_DATASETS=nyx CHUNK_MB=4 VERIFY=0 bash benchmarks/benchmark.sh
  BENCHMARKS=sdrbench SDR_DATASETS="nyx,hurricane_isabel,cesm_atm" bash benchmarks/benchmark.sh

  # ── 4. AI Training (neural network checkpoint compression) ──
  # Step 1: Generate checkpoint data (one-time)
  python3 scripts/train_and_export_checkpoints.py --model vit_b_16 --epochs 20  # ~25 min, 10 GB
  python3 scripts/train_gpt2_checkpoints.py --epochs 5                          # ~20 min, 7.4 GB
  python3 scripts/train_gpt2_checkpoints.py --epochs 5                          # ~20 min, 7.6 GB
  # Step 2: Run benchmark
  BENCHMARKS=ai_training AI_MODEL=vit_b_16 CHUNK_MB=4 VERIFY=0 bash benchmarks/benchmark.sh
  BENCHMARKS=ai_training AI_MODEL=gpt2 AI_DATASET=wikitext2 CHUNK_MB=4 VERIFY=0 bash benchmarks/benchmark.sh

EXAMPLES
  # Quick smoke test (smallest config, ~2 min)
  BENCHMARKS=grayscott DATA_MB=8 CHUNK_MB=4 TIMESTEPS=3 VERIFY=0 bash benchmarks/benchmark.sh

  # All 4 benchmarks, balanced policy
  BENCHMARKS="grayscott,vpic,sdrbench,ai_training" POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh

  # All 3 policies for comparison
  BENCHMARKS=ai_training POLICIES="balanced,ratio,speed" CHUNK_MB=4 VERIFY=0 bash benchmarks/benchmark.sh

  # Selected phases only
  BENCHMARKS=vpic PHASES="no-comp,lz4,zstd,nn,nn-rl" DATA_MB=1024 bash benchmarks/benchmark.sh

  # Multi-GPU (2 ranks, 2 GPUs per node)
  MPI_NP=2 GPUS_PER_NODE=2 BENCHMARKS=vpic DATA_MB=512 VERIFY=0 bash benchmarks/benchmark.sh

  # VPIC with custom physics
  BENCHMARKS=vpic DATA_MB=1024 VPIC_MI_ME=1 VPIC_TI_TE=5 VPIC_WARMUP_STEPS=200 bash benchmarks/benchmark.sh

OUTPUT
  benchmarks/grayscott/results/eval_<params>/<policy>/      Gray-Scott CSVs + plots
  benchmarks/vpic-kokkos/results/eval_<params>/<policy>/    VPIC CSVs + plots
  benchmarks/sdrbench/results/eval_<dataset>_<params>/      SDRBench CSVs + plots
  benchmarks/ai_training/results/eval_<model>_<params>/     AI Training CSVs + plots
  benchmarks/results/per_dataset/<dataset>/<eval>/<policy>/  Auto-generated plots
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
# Tag for directory naming: _noverify when VERIFY=0, empty when VERIFY=1
_VERIFY_TAG=""
[ "$VERIFY" = "0" ] && _VERIFY_TAG="_noverify"
_LOSSY_TAG=""
if [ "${VPIC_ERROR_BOUND:-0}" != "0" ] && [ "${VPIC_ERROR_BOUND:-0}" != "0.0" ]; then
    _LOSSY_TAG="_lossy${VPIC_ERROR_BOUND}"
fi
POLICIES=${POLICIES:-"balanced,ratio,speed"}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
DEBUG_NN=${DEBUG_NN:-0}

# ── Multi-GPU / Multi-Node (MPI) ──
# MPI_NP: total number of MPI ranks (default 1 = single-process, no mpirun)
# GPUS_PER_NODE: GPUs per node (used for CUDA_VISIBLE_DEVICES mapping)
# LAUNCHER: auto (detect SLURM vs mpirun), srun, or mpirun
#
# Auto-detection: if this script is already running inside an srun step
# (SLURM_STEP_ID is set), we are one of N ranks launched by the user's srun.
# In that case, mpi_launch() runs the binary directly — srun already created
# the MPI ranks. MPI_NP is inferred from SLURM_NTASKS so that the CSV merge
# logic knows how many rank-suffixed files to expect.
if [ -n "$SLURM_STEP_ID" ] && [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
    # Already inside srun with multiple tasks — don't nest srun
    MPI_NP=${MPI_NP:-${SLURM_NTASKS}}
    GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
    LAUNCHER=${LAUNCHER:-none}
else
    MPI_NP=${MPI_NP:-1}
    GPUS_PER_NODE=${GPUS_PER_NODE:-1}
    LAUNCHER=${LAUNCHER:-auto}
fi

# Helper: wrap binary with srun or mpirun for multi-GPU, setting per-rank GPU binding.
# - none: already inside srun — run binary directly (MPI is pre-initialized)
# - srun: SLURM launches ranks with --gpus-per-task=1 (use from salloc or sbatch)
# - mpirun: manual CUDA_VISIBLE_DEVICES mapping per rank (non-SLURM systems)
# - Single-rank (MPI_NP=1): runs binary directly, no launcher.
mpi_launch() {
    if [ "$MPI_NP" -le 1 ]; then
        "$@"
        return
    fi

    # Determine launcher
    local use_launcher="$LAUNCHER"
    if [ "$use_launcher" = "auto" ]; then
        if [ -n "$SLURM_JOB_ID" ]; then
            use_launcher="srun"
        else
            use_launcher="mpirun"
        fi
    fi

    case "$use_launcher" in
        none)
            # Already inside srun — binary runs directly, MPI is pre-initialized
            "$@"
            ;;
        srun)
            # SLURM: use pre-built hostfile from SLURM_HOSTFILE if available
            if [ -n "$SLURM_HOSTFILE" ] && [ -f "$SLURM_HOSTFILE" ]; then
                srun --ntasks="$MPI_NP" \
                     --gpus-per-task=1 \
                     --ntasks-per-node="$GPUS_PER_NODE" \
                     --kill-on-bad-exit=1 \
                     --distribution=arbitrary \
                     "$@"
            else
                srun --ntasks="$MPI_NP" \
                     --gpus-per-task=1 \
                     --kill-on-bad-exit=1 \
                     "$@"
            fi
            ;;
        mpirun)
            # Non-SLURM: manual CUDA_VISIBLE_DEVICES mapping per rank
            mpirun -np "$MPI_NP" \
                bash -c 'export CUDA_VISIBLE_DEVICES=$((${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-0}}}} % '"$GPUS_PER_NODE"')); exec "$@"' \
                _ "$@"
            ;;
        *)
            echo "ERROR: Unknown LAUNCHER=$use_launcher (use: auto, srun, mpirun, none)" >&2
            return 1
            ;;
    esac
}

# ── GS-specific defaults ──
GS_STEPS=${GS_STEPS:-500}

# ── AI Training defaults ──
AI_MODEL=${AI_MODEL:-"vit_b_16"}
AI_DATASET=${AI_DATASET:-"cifar10"}

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

# Determine local MPI rank for rank-0 only output.
# SLURM_PROCID is set by srun; PMI_RANK by Cray PMI; fall back to 0.
MY_RANK=${SLURM_PROCID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}

# Only rank 0 prints the banner and post-processing output
if [ "$MY_RANK" -eq 0 ]; then
echo "============================================================"
echo "  GPUCompress Benchmark Suite"
echo "============================================================"
echo "  Benchmarks  : ${BENCHMARKS}"
echo "  Target data : ${DATA_MB} MB"
echo "  Chunk size  : ${CHUNK_MB} MB"
echo "  Timesteps   : ${TIMESTEPS}"
echo "  Policies    : ${POLICIES}"
echo "  Phases      : ${PHASES}"
if [ "${VPIC_ERROR_BOUND:-0}" != "0" ] && [ "${VPIC_ERROR_BOUND:-0}" != "0.0" ]; then
echo "  Mode        : LOSSY (error_bound=${VPIC_ERROR_BOUND})"
else
echo "  Mode        : LOSSLESS"
fi
if [ "$MPI_NP" -gt 1 ]; then
echo "  MPI ranks   : ${MPI_NP} (GPUs/node: ${GPUS_PER_NODE})"
fi
echo ""
echo "  NN hyperparameters:"
echo "    SGD_LR=${SGD_LR}  SGD_MAPE=${SGD_MAPE}"
echo "    EXPLORE_K=${EXPLORE_K}  EXPLORE_THRESH=${EXPLORE_THRESH}"
echo ""
echo "  Gray-Scott  : L=${GS_L} (~${GS_DATA} MB), steps=${GS_STEPS}"
echo "  VPIC        : NX=${VPIC_NX} (~${VPIC_DATA} MB), warmup=${VPIC_WARMUP_STEPS:-500}, interval=${VPIC_SIM_INTERVAL:-190}"
echo "  SDRBench    : datasets=${SDR_DATASETS:-nyx,hurricane_isabel,cesm_atm}"
echo "  AI Training : model=${AI_MODEL}, dataset=${AI_DATASET}"
echo "============================================================"
echo ""
fi  # end rank-0 banner

# Write params.txt to a given eval directory
write_parameters_txt() {
    local DIR="$1"
    local BENCH_TYPE="$2"
    mkdir -p "$DIR"
    cat > "$DIR/params.txt" <<PARAMS
GPUCompress Benchmark Parameters
================================
Date:               $(date '+%Y-%m-%d %H:%M:%S')
Benchmark:          ${BENCH_TYPE}
Chunk size (MB):    ${CHUNK_MB}
Timesteps:          ${TIMESTEPS}
Phases:             ${PHASES}
Policies:           ${POLICIES}
Verify:             ${VERIFY}
MPI ranks:          ${MPI_NP}
GPUs per node:      ${GPUS_PER_NODE}

NN Hyperparameters:
  SGD_LR:           ${SGD_LR}
  SGD_MAPE:         ${SGD_MAPE}
  EXPLORE_K:        ${EXPLORE_K}
  EXPLORE_THRESH:   ${EXPLORE_THRESH}
PARAMS

    case "$BENCH_TYPE" in
        grayscott)
            cat >> "$DIR/params.txt" <<PARAMS

Gray-Scott:
  L:                ${GS_L}
  Data (MB):        ${GS_DATA}
  Steps:            ${GS_STEPS}
PARAMS
            ;;
        vpic)
            cat >> "$DIR/params.txt" <<PARAMS

VPIC:
  NX:               ${VPIC_NX}
  Data (MB):        ${VPIC_DATA}
  NPPC:             ${VPIC_NPPC:-2}
  MI_ME:            ${VPIC_MI_ME:-5}
  WPE_WCE:          ${VPIC_WPE_WCE:-1}
  TI_TE:            ${VPIC_TI_TE:-5}
  Warmup steps:     ${VPIC_WARMUP_STEPS:-500}
  Sim interval:     ${VPIC_SIM_INTERVAL:-190}
  Perturbation:     ${VPIC_PERTURBATION:-0.1}
  Guide field:      ${VPIC_GUIDE_FIELD:-0.0}
PARAMS
            ;;
        ai_training)
            cat >> "$DIR/params.txt" <<PARAMS

AI Training:
  Model:            ${AI_MODEL}
  Dataset:          ${AI_DATASET}
  Data dir:         ${AI_DATA_DIR}
  Error bound:      ${AI_EB:-0.0}
PARAMS
            ;;
    esac
}

IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

for bench in "${BENCH_LIST[@]}"; do
    case "$bench" in
        grayscott)
            echo ""
            echo ">>> Running Gray-Scott benchmark..."
            echo ""

            GS_BIN="$SCRIPT_DIR/../build/grayscott_benchmark_pm"
            GS_WEIGHTS="$SCRIPT_DIR/../neural_net/weights/model.nnwt"
            GS_EVAL_DIR="$SCRIPT_DIR/grayscott/results/eval_L${GS_L}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}${_VERIFY_TAG}${_LOSSY_TAG}${VPIC_EVAL_SUFFIX:-}"
            write_parameters_txt "$GS_EVAL_DIR" "grayscott"

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

            if [ "$MY_RANK" -eq 0 ]; then
            echo "  L=$GS_L (~${GS_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}, steps=${GS_STEPS}"
            echo "  fixed phases: ${GS_FIXED_LIST:-none}"
            echo "  NN phases:    ${GS_NN_LIST:-none}"
            echo ""
            fi

            # ── Fixed phases (run once, policy-independent) ──
            if [ -n "$GS_FIXED_ARGS" ]; then
                GS_FIXED_DIR="$GS_EVAL_DIR/fixed_phases"
                mkdir -p "$GS_FIXED_DIR"

                if [ "$MY_RANK" -eq 0 ]; then
                echo "  >>> Fixed phases (run once): $GS_FIXED_LIST"
                fi

                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                mpi_launch "$GS_BIN" "$GS_WEIGHTS" \
                    --L $GS_L --steps $GS_STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
                    $GS_FIXED_ARGS \
                    --w0 1.0 --w1 1.0 --w2 1.0 \
                    --lr $SGD_LR --mape $SGD_MAPE \
                    --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                    $VERIFY_ARG \
                    --out-dir "$GS_FIXED_DIR" \
                    > "$GS_FIXED_DIR/gs_benchmark.log" 2> >(tee -a "$GS_FIXED_DIR/gs_benchmark.log" >&2)

                if [ "$MY_RANK" -eq 0 ]; then
                echo "      Done. Log: $GS_FIXED_DIR/gs_benchmark.log"
                echo ""
                fi
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

                    if [ "$MY_RANK" -eq 0 ]; then
                    echo "  >>> NN phases [$LABEL] ($POL_IDX/$POL_TOTAL): $GS_NN_LIST"
                    fi

                    GPUCOMPRESS_DETAILED_TIMING=1 \
                    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                    mpi_launch "$GS_BIN" "$GS_WEIGHTS" \
                        --L $GS_L --steps $GS_STEPS --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
                        $GS_NN_ARGS \
                        --w0 $W0 --w1 $W1 --w2 $W2 \
                        --lr $SGD_LR --mape $SGD_MAPE \
                        --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                        $VERIFY_ARG \
                        --out-dir "$GS_NN_DIR" \
                        > "$GS_NN_DIR/gs_benchmark.log" 2> >(tee -a "$GS_NN_DIR/gs_benchmark.log" >&2)

                    if [ "$MY_RANK" -eq 0 ]; then
                    echo "      Done. Log: $GS_NN_DIR/gs_benchmark.log"

                    # Merge fixed + NN CSVs.
                    # In MPI mode, merge rank-suffixed files into canonical names first,
                    # then merge fixed phases into NN CSVs.
                    if [ -n "$GS_FIXED_LIST" ] && [ -d "$GS_FIXED_DIR" ]; then
                        for csv_base in benchmark_grayscott_vol benchmark_grayscott_timesteps benchmark_grayscott_timestep_chunks benchmark_grayscott_ranking benchmark_grayscott_ranking_costs; do
                            csv_name="${csv_base}.csv"
                            FIXED_SRC="$GS_FIXED_DIR/$csv_name"
                            NN_SRC="$GS_NN_DIR/$csv_name"

                            # MPI mode: merge rank-suffixed files into canonical name
                            if [ "$MPI_NP" -gt 1 ]; then
                                for src_dir in "$GS_FIXED_DIR" "$GS_NN_DIR"; do
                                    canonical="$src_dir/$csv_name"
                                    if [ ! -f "$canonical" ]; then
                                        rank0="$src_dir/${csv_base}_rank0.csv"
                                        if [ -f "$rank0" ]; then
                                            cp "$rank0" "$canonical"
                                            for ri in $(seq 1 $((MPI_NP - 1))); do
                                                rf="$src_dir/${csv_base}_rank${ri}.csv"
                                                [ -f "$rf" ] && tail -n+2 "$rf" >> "$canonical"
                                            done
                                        fi
                                    fi
                                done
                            fi

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

                    timeout 60 bash -c "GS_DIR='$GS_NN_DIR' \
                    python3 '$SCRIPT_DIR/plots/generate_dataset_figures.py' \
                        --dataset gray_scott --policy '$LABEL' 2>&1 | grep -E 'Generated'" \
                    || echo "  [$LABEL] Plot generation skipped (timeout or matplotlib unavailable)"
                    echo ""
                    fi  # end rank-0 GS post-processing
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
            VPIC_LD_PATH="/tmp/hdf5-install/lib:$SCRIPT_DIR/../build:/tmp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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

            VPIC_EVAL_DIR="$SCRIPT_DIR/vpic-kokkos/results/eval_NX${VPIC_NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}${_VERIFY_TAG}${_LOSSY_TAG}${VPIC_EVAL_SUFFIX:-}"
            write_parameters_txt "$VPIC_EVAL_DIR" "vpic"

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

                # Log destination: in LAUNCHER=none mode (interactive srun),
                # each rank runs its own copy of benchmark.sh, so use
                # rank-suffixed logs to avoid file corruption. In srun/mpirun
                # mode, only one benchmark.sh runs, so use a single log.
                local LOG_FILE
                if [ "$LAUNCHER" = "none" ]; then
                    LOG_FILE="$RESULTS_DIR/vpic_benchmark_rank${MY_RANK}.log"
                else
                    LOG_FILE="$RESULTS_DIR/vpic_benchmark.log"
                fi

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
                mpi_launch "$VPIC_BIN" "$VPIC_DECK" \
                > "$LOG_FILE" 2> >(tee -a "$LOG_FILE" >&2)
            }

            if [ "$MY_RANK" -eq 0 ]; then
            echo "  NX=$VPIC_NX (~${VPIC_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}"
            echo "  warmup=$VPIC_WARMUP, sim_interval=$VPIC_SIM_INT"
            echo "  physics: mi_me=$VPIC_MI_ME wpe_wce=$VPIC_WPE_WCE Ti_Te=$VPIC_TI_TE pert=${VPIC_PERTURBATION:-0.1}"
            echo "  policies: $POLICIES"
            echo ""
            echo "  >>> All phases, all policies in single invocation"
            fi

            # ── Single invocation: all phases + all policies ──
            # Fixed phases run once. NN phases run once per policy.
            # The VPIC binary handles multi-policy internally via VPIC_POLICIES.
            VPIC_RESULTS="$VPIC_EVAL_DIR"
            mkdir -p "$VPIC_RESULTS"

            vpic_run "$VPIC_RESULTS" "$PHASES"

            if [ "$MY_RANK" -eq 0 ]; then
            if [ "$LAUNCHER" = "none" ]; then
                echo "      Done. Logs: $VPIC_RESULTS/vpic_benchmark_rank*.log"
            else
                echo "      Done. Log: $VPIC_RESULTS/vpic_benchmark.log"
            fi

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

                # For each CSV: extract fixed rows + this policy's NN rows.
                # With MPI_NP>1, CSV files are rank-suffixed; merge all ranks first.
                for csv_base in benchmark_vpic_deck_timesteps benchmark_vpic_deck_timestep_chunks benchmark_vpic_deck_ranking benchmark_vpic_deck_ranking_costs; do
                    csv_name="${csv_base}.csv"
                    SRC="$VPIC_RESULTS/$csv_name"

                    # If rank-suffixed files exist (MPI mode), merge them (overwrite stale single-rank files)
                    if [ "$MPI_NP" -gt 1 ]; then
                        RANK0="$VPIC_RESULTS/${csv_base}_rank0.csv"
                        if [ -f "$RANK0" ]; then
                            cp "$RANK0" "$SRC"
                            for rank_i in $(seq 1 $((MPI_NP - 1))); do
                                RANKF="$VPIC_RESULTS/${csv_base}_rank${rank_i}.csv"
                                [ -f "$RANKF" ] && tail -n+2 "$RANKF" >> "$SRC"
                            done
                        fi
                    fi

                    [ -f "$SRC" ] || continue
                    DST="$POL_DIR/$csv_name"

                    # Header + fixed rows (no "/" in phase) + this policy's NN rows (strip suffix)
                    # Phase column: $2 for timesteps/chunks (rank,$2,...), $1 for ranking (phase,$2,...)
                    local _phase_col=2
                    case "$csv_base" in
                        *ranking*) _phase_col=1 ;;
                    esac
                    head -1 "$SRC" > "$DST"
                    tail -n+2 "$SRC" | awk -F',' -v pol="/$pol" -v pc="$_phase_col" '{
                        phase=$pc;
                        if (index(phase, "/") == 0) print;
                        else if (index(phase, pol) > 0) {
                            sub(pol, "", phase);
                            $pc = phase;
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
        phases[p] = {'sum_wr':0,'sum_rd':0,'sum_wr_sq':0,'sum_rd_sq':0,
                     'sum_file_bytes':0,'n':0,'n_chunks':0,
                     'sum_sgd':0,'sum_expl':0,'sum_mape_r':0,'sum_mape_c':0,'sum_mape_d':0,
                     'sum_mape_p':0,'sum_mae_r':0,'sum_mae_c':0,'sum_mae_d':0,'sum_mae_p':0,
                     'sum_stats':0,'sum_nn':0,'sum_pre':0,'sum_comp':0,'sum_dec':0,
                     'sum_expl_ms':0,'sum_sgd_ms':0,
                     'sum_r2_ratio':0,'sum_r2_comp':0,'sum_r2_decomp':0,'sum_r2_psnr':0}
    d = phases[p]
    wr_val = float(r.get('write_ms',0))
    rd_val = float(r.get('read_ms',0))
    d['sum_wr'] += wr_val
    d['sum_rd'] += rd_val
    d['sum_wr_sq'] += wr_val * wr_val
    d['sum_rd_sq'] += rd_val * rd_val
    d['sum_file_bytes'] += int(r.get('file_bytes',0))
    d['n_chunks'] = int(r.get('n_chunks',0))
    d['sum_sgd'] += int(r.get('sgd_fires',0))
    d['sum_expl'] += int(r.get('explorations',0))
    d['sum_mape_r'] += float(r.get('mape_ratio',0))
    d['sum_mape_c'] += float(r.get('mape_comp',0))
    d['sum_mape_d'] += float(r.get('mape_decomp',0))
    d['sum_mape_p'] += float(r.get('mape_psnr',0))
    d['sum_mae_r'] += float(r.get('mae_ratio',0))
    d['sum_mae_c'] += float(r.get('mae_comp_ms',0))
    d['sum_mae_d'] += float(r.get('mae_decomp_ms',0))
    d['sum_mae_p'] += float(r.get('mae_psnr_db',0))
    d['sum_stats'] += float(r.get('stats_ms',0))
    d['sum_nn'] += float(r.get('nn_ms',0))
    d['sum_pre'] += float(r.get('preproc_ms',0))
    d['sum_comp'] += float(r.get('comp_ms',0))
    d['sum_dec'] += float(r.get('decomp_ms',0))
    d['sum_expl_ms'] += float(r.get('explore_ms',0))
    d['sum_sgd_ms'] += float(r.get('sgd_ms',0))
    d['sum_r2_ratio'] += float(r.get('r2_ratio',0))
    d['sum_r2_comp'] += float(r.get('r2_comp',0))
    d['sum_r2_decomp'] += float(r.get('r2_decomp',0))
    d['sum_r2_psnr'] += float(r.get('r2_psnr',0))
    d['n'] += 1
# Get constant orig size: prefer orig_mib column if present, else from no-comp file_bytes
first_orig = 0
if 'orig_mib' in rows[0]:
    first_orig = float(rows[0].get('orig_mib',0))
if first_orig <= 0:
    for r in rows:
        if r['phase'] == 'no-comp':
            first_orig = int(r.get('file_bytes',0)) / (1024*1024)
            break
with open('$AGG_CSV','w') as f:
    hdr = ('rank,source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,'
           'file_mib,orig_mib,ratio,write_mibps,read_mibps,mismatches,'
           'sgd_fires,explorations,n_chunks,'
           'nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,'
           'comp_gbps,decomp_gbps,'
           'mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,'
           'mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,'
           'r2_ratio,r2_comp,r2_decomp,r2_psnr')
    f.write(hdr + '\n')
    for p,d in phases.items():
        n = d['n']
        if n == 0: continue
        total_wr = d['sum_wr']
        total_rd = d['sum_rd']
        avg_wr = total_wr / n
        avg_rd = total_rd / n
        import math
        wr_var = (d['sum_wr_sq']/n - avg_wr**2) * n/(n-1) if n > 1 else 0
        rd_var = (d['sum_rd_sq']/n - avg_rd**2) * n/(n-1) if n > 1 else 0
        wr_std = math.sqrt(wr_var) if wr_var > 0 else 0
        rd_std = math.sqrt(rd_var) if rd_var > 0 else 0
        total_file_mib = d['sum_file_bytes'] / (1024*1024)
        avg_file_mib = total_file_mib / n
        orig_mib = first_orig if first_orig > 0 else avg_file_mib
        rat = (n * orig_mib) / total_file_mib if total_file_mib > 0 else 0
        # Throughput: total_data / total_time (not orig / mean_time)
        wr_mibps = (n * orig_mib) / (total_wr / 1000.0) if total_wr > 0 else 0
        rd_mibps = (n * orig_mib) / (total_rd / 1000.0) if total_rd > 0 else 0
        avg_comp = d['sum_comp'] / n
        avg_dec = d['sum_dec'] / n
        orig_bytes = orig_mib * 1024 * 1024
        cgbps = orig_bytes / 1e9 / (avg_comp / 1000.0) if avg_comp > 0 else 0
        dgbps = orig_bytes / 1e9 / (avg_dec / 1000.0) if avg_dec > 0 else 0
        sgd=d['sum_sgd']/n; expl=d['sum_expl']/n; nch=d['n_chunks']
        nn=d['sum_nn']/n; st=d['sum_stats']/n; pre=d['sum_pre']/n
        exms=d['sum_expl_ms']/n; sgms=d['sum_sgd_ms']/n
        mr=min(200,d['sum_mape_r']/n); mc=min(200,d['sum_mape_c']/n)
        md=min(200,d['sum_mape_d']/n); mp=min(200,d['sum_mape_p']/n)
        ar=d['sum_mae_r']/n; ac=d['sum_mae_c']/n
        ad=d['sum_mae_d']/n; ap=d['sum_mae_p']/n
        rr=d['sum_r2_ratio']/n; rc=d['sum_r2_comp']/n
        rd2=d['sum_r2_decomp']/n; rp=d['sum_r2_psnr']/n
        f.write(f'-1,vpic,{p},{n},{avg_wr:.2f},{wr_std:.2f},{avg_rd:.2f},{rd_std:.2f},'
                f'{avg_file_mib:.2f},{orig_mib:.2f},{rat:.4f},{wr_mibps:.1f},{rd_mibps:.1f},0,'
                f'{sgd:.0f},{expl:.0f},{nch},'
                f'{nn:.2f},{st:.2f},{pre:.2f},'
                f'{avg_comp:.2f},{avg_dec:.2f},{exms:.2f},{sgms:.2f},'
                f'{cgbps:.4f},{dgbps:.4f},'
                f'{mr:.2f},{mc:.2f},{md:.2f},{mp:.2f},'
                f'{ar:.4f},{ac:.4f},{ad:.4f},{ap:.4f},'
                f'{rr:.4f},{rc:.4f},{rd2:.4f},{rp:.4f}\n')
" 2>/dev/null
                fi

                echo "  [$LABEL] Split CSVs → $POL_DIR/"

                # Generate plots for this policy (timeout 60s to avoid matplotlib hangs)
                timeout 60 bash -c "VPIC_DIR='$POL_DIR' \
                python3 '$SCRIPT_DIR/plots/generate_dataset_figures.py' \
                    --dataset vpic --policy '$LABEL' 2>&1 | grep -E 'Generated'" \
                || echo "  [$LABEL] Plot generation skipped (timeout or matplotlib unavailable)"
            done

            # ── Multi-rank aggregate throughput summary ──
            if [ "$MPI_NP" -gt 1 ]; then
                MERGED_TS="$VPIC_RESULTS/benchmark_vpic_deck_timesteps.csv"
                AGG_MULTI="$VPIC_RESULTS/benchmark_vpic_deck_aggregate_multi_rank.csv"
                if [ -f "$MERGED_TS" ]; then
                    echo "  Generating multi-rank aggregate throughput ..."
                    python3 -c "
import csv, math

rows = list(csv.DictReader(open('$MERGED_TS')))
if not rows:
    exit()

n_ranks = len(set(r['rank'] for r in rows))

# Group by (phase, timestep): for each group, compute aggregate throughput
# across all ranks running in parallel.
#   aggregate_write_throughput = sum(orig_mib) / (max(write_ms) / 1000)
#   aggregate_read_throughput  = sum(orig_mib) / (max(read_ms) / 1000)
groups = {}
for r in rows:
    key = (r['phase'], r['timestep'])
    if key not in groups:
        groups[key] = []
    groups[key].append(r)

# Aggregate per phase (average across timesteps)
phases = {}
for (phase, ts), ranks in groups.items():
    if phase not in phases:
        phases[phase] = {'n_ts': 0,
                         'sum_agg_wr_mibps': 0, 'sum_agg_rd_mibps': 0,
                         'sum_agg_wr_sq': 0, 'sum_agg_rd_sq': 0,
                         'sum_orig_mib': 0, 'sum_file_mib': 0,
                         'sum_max_wr_ms': 0, 'sum_max_rd_ms': 0,
                         'avg_per_rank_wr': 0, 'avg_per_rank_rd': 0,
                         'n_per_rank': 0}
    d = phases[phase]

    total_orig = sum(float(r.get('orig_mib', 0)) for r in ranks)
    total_file = sum(int(r.get('file_bytes', 0)) for r in ranks) / (1024*1024)
    max_wr = max(float(r.get('write_ms', 0)) for r in ranks)
    max_rd = max(float(r.get('read_ms', 0)) for r in ranks)

    agg_wr = total_orig / (max_wr / 1000.0) if max_wr > 0 else 0
    agg_rd = total_orig / (max_rd / 1000.0) if max_rd > 0 else 0

    d['n_ts'] += 1
    d['sum_agg_wr_mibps'] += agg_wr
    d['sum_agg_rd_mibps'] += agg_rd
    d['sum_agg_wr_sq'] += agg_wr * agg_wr
    d['sum_agg_rd_sq'] += agg_rd * agg_rd
    d['sum_orig_mib'] += total_orig
    d['sum_file_mib'] += total_file
    d['sum_max_wr_ms'] += max_wr
    d['sum_max_rd_ms'] += max_rd

    for r in ranks:
        d['avg_per_rank_wr'] += float(r.get('write_mibps', 0))
        d['avg_per_rank_rd'] += float(r.get('read_mibps', 0))
        d['n_per_rank'] += 1

with open('$AGG_MULTI', 'w') as f:
    f.write('n_ranks,phase,n_timesteps,'
            'avg_agg_write_mibps,avg_agg_read_mibps,'
            'agg_write_mibps_std,agg_read_mibps_std,'
            'avg_per_rank_write_mibps,avg_per_rank_read_mibps,'
            'avg_ratio,total_orig_mib,total_file_mib,'
            'avg_max_write_ms,avg_max_read_ms\n')
    for phase in phases:
        d = phases[phase]
        n = d['n_ts']
        if n == 0:
            continue
        avg_wr = d['sum_agg_wr_mibps'] / n
        avg_rd = d['sum_agg_rd_mibps'] / n
        wr_var = (d['sum_agg_wr_sq']/n - avg_wr**2) * n/(n-1) if n > 1 else 0
        rd_var = (d['sum_agg_rd_sq']/n - avg_rd**2) * n/(n-1) if n > 1 else 0
        wr_std = math.sqrt(wr_var) if wr_var > 0 else 0
        rd_std = math.sqrt(rd_var) if rd_var > 0 else 0
        orig = d['sum_orig_mib'] / n
        fmib = d['sum_file_mib'] / n
        ratio = orig / fmib if fmib > 0 else 0
        pr_wr = d['avg_per_rank_wr'] / d['n_per_rank'] if d['n_per_rank'] > 0 else 0
        pr_rd = d['avg_per_rank_rd'] / d['n_per_rank'] if d['n_per_rank'] > 0 else 0
        avg_mwr = d['sum_max_wr_ms'] / n
        avg_mrd = d['sum_max_rd_ms'] / n
        f.write(f'{n_ranks},{phase},{n},'
                f'{avg_wr:.1f},{avg_rd:.1f},'
                f'{wr_std:.1f},{rd_std:.1f},'
                f'{pr_wr:.1f},{pr_rd:.1f},'
                f'{ratio:.4f},{orig:.2f},{fmib:.2f},'
                f'{avg_mwr:.2f},{avg_mrd:.2f}\n')

print(f'  Aggregate: {n_ranks} ranks, {len(phases)} phases -> $AGG_MULTI')
" 2>/dev/null
                fi
            fi

            echo ""
            fi  # end rank-0 post-processing
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
            ERROR_BOUND=${ERROR_BOUND:-0.0} \
            DEBUG_NN=$DEBUG_NN \
            MPI_NP=$MPI_NP \
            GPUS_PER_NODE=$GPUS_PER_NODE \
            LAUNCHER=$LAUNCHER \
            bash "$SCRIPT_DIR/sdrbench/run_all_sdr.sh"
            ;;
        ai_training)
            if [ "$MY_RANK" -eq 0 ]; then
            echo ""
            echo ">>> Running AI Training Checkpoint benchmark..."
            echo ""
            fi

            AI_BIN="$SCRIPT_DIR/../build/generic_benchmark"
            AI_WEIGHTS="$SCRIPT_DIR/../neural_net/weights/model.nnwt"

            # Resolve checkpoint data directory
            # Naming: vit_b_16 → vit_b_cifar10, gpt2 → gpt2_wikitext2
            case "$AI_MODEL" in
                vit_b_16) _AI_SHORT="vit_b" ;;
                *)        _AI_SHORT="$AI_MODEL" ;;
            esac
            AI_DIR_NAME="${_AI_SHORT}_${AI_DATASET}"
            AI_DATA_DIR="${AI_CHECKPOINT_DIR:-$SCRIPT_DIR/../data/ai_training/${AI_DIR_NAME}}"

            if [ ! -d "$AI_DATA_DIR" ]; then
                if [ "$MY_RANK" -eq 0 ]; then
                echo "ERROR: Checkpoint data not found at $AI_DATA_DIR"
                echo ""
                echo "Generate it first:"
                case "$AI_MODEL" in
                    vit_*) echo "  python3 scripts/train_and_export_checkpoints.py --model $AI_MODEL" ;;
                    gpt2)  echo "  python3 scripts/train_gpt2_checkpoints.py" ;;
                    *)     echo "  Generate .f32 files in data/ai_training/${AI_DIR_NAME}/" ;;
                esac
                echo ""
                fi
                continue
            fi

            # Auto-detect file format (.f32 or .h5) and dims
            _FIRST_FILE=$(ls "$AI_DATA_DIR"/*.f32 2>/dev/null | head -1)
            _AI_EXT=".f32"
            if [ -z "$_FIRST_FILE" ]; then
                _FIRST_FILE=$(ls "$AI_DATA_DIR"/*.h5 2>/dev/null | head -1)
                _AI_EXT=".h5"
            fi
            if [ -z "$_FIRST_FILE" ]; then
                echo "ERROR: No .f32 or .h5 files in $AI_DATA_DIR"
                continue
            fi
            if [ "$_AI_EXT" = ".h5" ]; then
                # For .h5 files, get element count from HDF5 dataset
                _N_FLOATS=$(python3 -c "
import h5py, sys
try:
    f=h5py.File('$_FIRST_FILE','r'); print(f['data'].size); f.close()
except: sys.exit(1)
" 2>/dev/null)
                if [ -z "$_N_FLOATS" ]; then
                    echo "ERROR: Cannot read .h5 file (h5py required): $_FIRST_FILE"
                    continue
                fi
                _FILE_BYTES=$(( _N_FLOATS * 4 ))
            else
                _FILE_BYTES=$(stat --printf="%s" "$_FIRST_FILE")
                _N_FLOATS=$(( _FILE_BYTES / 4 ))
            fi
            _N_FILES=$(ls "$AI_DATA_DIR"/*${_AI_EXT} | wc -l)
            # Use 2D dims: find factor
            _DIM0=$(python3 -c "
import math
n=$_N_FLOATS; s=int(math.isqrt(n))
while s>1 and n%s!=0: s-=1
print(s)
")
            _DIM1=$(( _N_FLOATS / _DIM0 ))
            AI_DIMS="${_DIM0},${_DIM1}"

            _FILE_MB=$(( _FILE_BYTES / 1024 / 1024 ))
            if [ "$MY_RANK" -eq 0 ]; then
            echo "  Model       : ${AI_MODEL}"
            echo "  Dataset     : ${AI_DATASET}"
            echo "  Data dir    : ${AI_DATA_DIR}"
            echo "  Files       : ${_N_FILES} x ${_FILE_MB} MB"
            echo "  Dims        : ${AI_DIMS}"
            echo "  Chunk size  : ${CHUNK_MB} MB"
            echo "  Policies    : ${POLICIES}"
            echo ""
            fi

            # Split phases into fixed and NN
            AI_FIXED_PHASES=""
            AI_NN_PHASES=""
            IFS=',' read -ra _PH <<< "$PHASES"
            for ph in "${_PH[@]}"; do
                case "$ph" in
                    nn|nn-rl|nn-rl+exp50) AI_NN_PHASES="${AI_NN_PHASES:+$AI_NN_PHASES,}$ph" ;;
                    *) AI_FIXED_PHASES="${AI_FIXED_PHASES:+$AI_FIXED_PHASES,}$ph" ;;
                esac
            done

            AI_EVAL_DIR="$SCRIPT_DIR/ai_training/results/eval_${AI_DIR_NAME}_chunk${CHUNK_MB}mb${_VERIFY_TAG}${_LOSSY_TAG}${VPIC_EVAL_SUFFIX:-}"
            VERIFY_ARG=""
            [ "$VERIFY" = "0" ] && VERIFY_ARG="--no-verify"

            # AI_ERROR_BOUND: error bound for lossy compression (default 0.0 = lossless)
            AI_EB=${AI_ERROR_BOUND:-0.0}
            EB_ARG=""
            [ "$AI_EB" != "0.0" ] && [ "$AI_EB" != "0" ] && EB_ARG="--error-bound $AI_EB"
            COMMON_ARGS="--data-dir $AI_DATA_DIR --dims $AI_DIMS --ext $_AI_EXT --chunk-mb $CHUNK_MB --name $AI_DIR_NAME $EB_ARG"
            write_parameters_txt "$AI_EVAL_DIR" "ai_training"

            # ── Fixed phases (run once) ──
            if [ -n "$AI_FIXED_PHASES" ]; then
                AI_FIXED_DIR="$AI_EVAL_DIR/fixed_phases"
                mkdir -p "$AI_FIXED_DIR"

                if [ "$MY_RANK" -eq 0 ]; then
                echo "  >>> Fixed phases: $AI_FIXED_PHASES"
                fi
                PHASE_ARGS=""
                IFS=',' read -ra _FP <<< "$AI_FIXED_PHASES"
                for fp in "${_FP[@]}"; do PHASE_ARGS="$PHASE_ARGS --phase $fp"; done

                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                mpi_launch "$AI_BIN" "$AI_WEIGHTS" \
                    $COMMON_ARGS $PHASE_ARGS \
                    --w0 1.0 --w1 1.0 --w2 1.0 \
                    --lr $SGD_LR --mape $SGD_MAPE \
                    --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                    $VERIFY_ARG \
                    --out-dir "$AI_FIXED_DIR" \
                    > "$AI_FIXED_DIR/ai_benchmark.log" 2> >(tee -a "$AI_FIXED_DIR/ai_benchmark.log" >&2)

                if [ "$MY_RANK" -eq 0 ]; then
                echo "      Done. Log: $AI_FIXED_DIR/ai_benchmark.log"
                echo ""
                fi
            fi

            # ── NN phases (run per policy) ──
            if [ -n "$AI_NN_PHASES" ]; then
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

                    AI_NN_DIR="$AI_EVAL_DIR/$LABEL"
                    mkdir -p "$AI_NN_DIR"

                    PHASE_ARGS=""
                    IFS=',' read -ra _NP <<< "$AI_NN_PHASES"
                    for np in "${_NP[@]}"; do PHASE_ARGS="$PHASE_ARGS --phase $np"; done

                    if [ "$MY_RANK" -eq 0 ]; then
                    echo "  >>> NN phases [$LABEL] ($POL_IDX/$POL_TOTAL): $AI_NN_PHASES"
                    fi

                    GPUCOMPRESS_DETAILED_TIMING=1 \
                    GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                    mpi_launch "$AI_BIN" "$AI_WEIGHTS" \
                        $COMMON_ARGS $PHASE_ARGS \
                        --w0 $W0 --w1 $W1 --w2 $W2 \
                        --lr $SGD_LR --mape $SGD_MAPE \
                        --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                        $VERIFY_ARG \
                        --out-dir "$AI_NN_DIR" \
                        > "$AI_NN_DIR/ai_benchmark.log" 2> >(tee -a "$AI_NN_DIR/ai_benchmark.log" >&2)

                    if [ "$MY_RANK" -eq 0 ]; then
                    echo "      Done. Log: $AI_NN_DIR/ai_benchmark.log"

                    # Merge fixed + NN CSVs
                    if [ -n "$AI_FIXED_PHASES" ] && [ -d "$AI_FIXED_DIR" ]; then
                        for csv_name in benchmark_${AI_DIR_NAME}.csv benchmark_${AI_DIR_NAME}_timesteps.csv benchmark_${AI_DIR_NAME}_timestep_chunks.csv benchmark_${AI_DIR_NAME}_ranking.csv benchmark_${AI_DIR_NAME}_ranking_costs.csv; do
                            FIXED_SRC="$AI_FIXED_DIR/$csv_name"
                            NN_SRC="$AI_NN_DIR/$csv_name"
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

                    # Generate plots (timeout 60s to avoid matplotlib hangs)
                    timeout 60 bash -c "AI_DIR='$AI_EVAL_DIR' AI_CHUNK='$CHUNK_MB' \
                    AI_MODEL='$_AI_SHORT' AI_DATASET='$AI_DATASET' \
                    python3 '$SCRIPT_DIR/plots/generate_dataset_figures.py' \
                        --dataset '$AI_DIR_NAME' --policy '$LABEL' 2>&1 | grep -E 'Generated'" \
                    || echo "  [$LABEL] Plot generation skipped (timeout or matplotlib unavailable)"
                    echo ""
                    fi  # end rank-0 AI post-processing
                done
            fi
            ;;
        *)
            echo "ERROR: Unknown benchmark '$bench'. Use: grayscott, vpic, sdrbench, ai_training"
            ;;
    esac
done

if [ "$MY_RANK" -eq 0 ]; then
echo ""
echo "============================================================"
echo "  All benchmarks complete."
echo "============================================================"
fi
