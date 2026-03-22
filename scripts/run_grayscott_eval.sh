#!/bin/bash
# ============================================================
# Gray-Scott Full Evaluation Suite for SC Paper
#
# Runs 3 experiments at L=512 (512 MB, 128 chunks @ 4 MB):
#
#   A) Chaos convergence  — primary "money plot" (MAPE vs timestep)
#   B) Spots + inject     — algorithm diversity histogram
#   C) 3-mode comparison  — spots / stripes / chaos bar charts
#
# Usage:
#   bash scripts/run_grayscott_eval.sh              # run all
#   bash scripts/run_grayscott_eval.sh --exp A      # run only experiment A
#   bash scripts/run_grayscott_eval.sh --exp B
#   bash scripts/run_grayscott_eval.sh --exp C
#   bash scripts/run_grayscott_eval.sh --L 256      # override grid size
#   bash scripts/run_grayscott_eval.sh --runs 3     # override run count
#   bash scripts/run_grayscott_eval.sh --plot-only   # skip benchmarks, just plot
#
# Output:
#   benchmarks/grayscott/results/chaos_balanced/
#   benchmarks/grayscott/results/spots_inject/
#   benchmarks/grayscott/results/spots_balanced/
#   benchmarks/grayscott/results/stripes_balanced/
# ============================================================
set -e

GPU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GPU_DIR"

GS_BIN="$GPU_DIR/build/grayscott_benchmark"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"
VISUALIZER="$GPU_DIR/benchmarks/visualize.py"
RESULTS_BASE="$GPU_DIR/benchmarks/grayscott/results"

# ── Defaults ──
L="${GS_L:-512}"
CHUNK_MB="${GS_CHUNK_MB:-4}"
RUNS="${GS_RUNS:-5}"
TS_CONVERGE="${GS_TS_CONVERGE:-100}"
TS_COMPARE="${GS_TS_COMPARE:-100}"
RUN_EXP=""    # empty = run all
PLOT_ONLY=0

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)       RUN_EXP="$2"; shift 2 ;;
        --L)         L="$2"; shift 2 ;;
        --chunk-mb)  CHUNK_MB="$2"; shift 2 ;;
        --runs)      RUNS="$2"; shift 2 ;;
        --ts-converge) TS_CONVERGE="$2"; shift 2 ;;
        --ts-compare)  TS_COMPARE="$2"; shift 2 ;;
        --plot-only) PLOT_ONLY=1; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Validate ──
if [[ ! -f "$GS_BIN" ]]; then
    echo "ERROR: grayscott_benchmark not found at $GS_BIN"
    echo "       Run: cmake --build build --target grayscott_benchmark"
    exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: weights not found at $WEIGHTS"
    exit 1
fi

should_run() {
    [[ -z "$RUN_EXP" || "$RUN_EXP" == "$1" ]]
}

DATASET_MIB=$(python3 -c "print(int(($L**3 * 4) / (1024*1024)))")
N_CHUNKS=$(python3 -c "
chunk_bytes = $CHUNK_MB * 1024 * 1024
field_bytes = $L * $L * $L * 4
print(int(field_bytes / chunk_bytes))
")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Gray-Scott Full Evaluation for SC Paper                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Grid: L=$L  (${DATASET_MIB} MiB per field, $N_CHUNKS chunks @ ${CHUNK_MB} MB)"
echo "║  Runs: $RUNS   Convergence TS: $TS_CONVERGE   Compare TS: $TS_COMPARE"
echo "║  Weights: $WEIGHTS"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
SKIP=0

run_experiment() {
    local name="$1"
    local out_dir="$2"
    shift 2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $name"
    echo "  Output:     $out_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "$out_dir"
    if [[ $PLOT_ONLY -eq 1 ]]; then
        echo "  [PLOT-ONLY] Skipping benchmark, plotting existing results..."
        SKIP=$((SKIP + 1))
        return 0
    fi
    if "$GS_BIN" "$WEIGHTS" "$@" --out-dir "$out_dir" 2>&1 | tee "$out_dir/benchmark.log"; then
        echo "  ✓ $name PASSED"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════
# Experiment A: Chaos convergence — the money plot
#   Non-converging simulation, 50 timesteps, 5 runs
#   Shows NN+SGD MAPE improvement over time on evolving data
# ══════════════════════════════════════════════════════════════
if should_run "A"; then
    run_experiment \
        "A: Chaos convergence (F=0.014, k=0.045, ts=$TS_CONVERGE)" \
        "$RESULTS_BASE/chaos_balanced" \
        --L "$L" --steps 100 --chunk-mb "$CHUNK_MB" \
        --F 0.014 --k 0.045 \
        --timesteps "$TS_CONVERGE" --runs "$RUNS"
fi

# ══════════════════════════════════════════════════════════════
# Experiment B: Spots + inject-patterns — algorithm diversity
#   Inject contrasting data into middle chunks to force
#   the NN to select different algorithms per chunk
#   Single run (deterministic injection, seed=42)
# ══════════════════════════════════════════════════════════════
if should_run "B"; then
    run_experiment \
        "B: Spots + inject-patterns (diversity test)" \
        "$RESULTS_BASE/spots_inject" \
        --L "$L" --steps 100 --chunk-mb "$CHUNK_MB" \
        --F 0.04 --k 0.06075 \
        --timesteps "$TS_COMPARE" --runs 1 \
        --inject-patterns
fi

# ══════════════════════════════════════════════════════════════
# Experiment C: 3-mode comparison — spots / stripes / chaos
#   Clean simulation data (no injection)
#   30 timesteps, 5 runs for error bars
# ══════════════════════════════════════════════════════════════
if should_run "C"; then
    # C1: Spots
    run_experiment \
        "C1: Spots (F=0.04, k=0.06075)" \
        "$RESULTS_BASE/spots_balanced" \
        --L "$L" --steps 100 --chunk-mb "$CHUNK_MB" \
        --F 0.04 --k 0.06075 \
        --timesteps "$TS_COMPARE" --runs "$RUNS"

    # C2: Stripes
    run_experiment \
        "C2: Stripes (F=0.035, k=0.065)" \
        "$RESULTS_BASE/stripes_balanced" \
        --L "$L" --steps 100 --chunk-mb "$CHUNK_MB" \
        --F 0.035 --k 0.065 \
        --timesteps "$TS_COMPARE" --runs "$RUNS"

    # C3: Chaos (can reuse Experiment A if already run)
    if [[ -f "$RESULTS_BASE/chaos_balanced/benchmark_grayscott_vol.csv" ]]; then
        echo "  C3: Chaos — reusing Experiment A results"
        SKIP=$((SKIP + 1))
    else
        run_experiment \
            "C3: Chaos (F=0.014, k=0.045)" \
            "$RESULTS_BASE/chaos_balanced" \
            --L "$L" --steps 100 --chunk-mb "$CHUNK_MB" \
            --F 0.014 --k 0.045 \
            --timesteps "$TS_COMPARE" --runs "$RUNS"
    fi
fi

# ══════════════════════════════════════════════════════════════
# Generate plots for each experiment
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating plots..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for dir in chaos_balanced spots_inject spots_balanced stripes_balanced; do
    vol_csv="$RESULTS_BASE/$dir/benchmark_grayscott_vol.csv"
    if [[ -f "$vol_csv" ]]; then
        echo "  Plotting: $dir"
        python3 "$VISUALIZER" \
            --gs-csv "$vol_csv" \
            --gs-dir "$RESULTS_BASE/$dir" \
            --output-dir "$RESULTS_BASE/$dir" 2>&1 | sed 's/^/    /'
    else
        echo "  Skipping $dir (no results found)"
    fi
done

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Summary                                                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Results in: $RESULTS_BASE/"
echo "║    chaos_balanced/    — Experiment A (convergence)"
echo "║    spots_inject/      — Experiment B (diversity)"
echo "║    spots_balanced/    — Experiment C1 (spots comparison)"
echo "║    stripes_balanced/  — Experiment C2 (stripes comparison)"
echo "╚══════════════════════════════════════════════════════════════╝"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
