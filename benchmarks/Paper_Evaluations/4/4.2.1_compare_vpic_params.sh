#!/bin/bash
# ============================================================
# Compare VPIC physics parameters for data diversity
#
# Runs two parameter sets side-by-side to determine which
# produces more diverse, faster-evolving simulation data:
#
# Config A (commit bd3e8a8, "original slow reconnection"):
#   mi_me=25, wpe_wce=3, Ti_Te=1, nppc=2
#   cfl_req=0.99, wpedt_max=0.36, clean_div=0, pert=0.1
#
# Config B (commit 2d0eb2f, "fast reconnection"):
#   mi_me=5, wpe_wce=1, Ti_Te=5, nppc=10
#   cfl_req=0.70, wpedt_max=0.20, clean_div=200, pert=0.05
#
# Each config runs:
#   - balanced policy (w0=1,w1=1,w2=1)
#   - ratio policy (w0=0,w1=0,w2=1)
#   - lossy (eb=0.01)
#   - 20 timesteps, warmup=500, sim_interval=190, NX=100, chunk=4MB
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/4.2.1_compare_vpic_params.sh
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VPIC_BIN="$GPU_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"

# Fixed params
NX=100
CHUNK_MB=4
TIMESTEPS=20
WARMUP=500
SIM_INTERVAL=190
ERROR_BOUND=0.01
LR=0.2
MAPE=0.10
EXPLORE_THRESH=0.20
EXPLORE_K=4

RESULTS_BASE="$SCRIPT_DIR/results/vpic_param_comparison"

echo "============================================================"
echo "VPIC Physics Parameter Comparison"
echo "  NX=$NX  Chunk=${CHUNK_MB}MB  Timesteps=$TIMESTEPS"
echo "  Warmup=$WARMUP  SimInterval=$SIM_INTERVAL  EB=$ERROR_BOUND"
echo "  Output: $RESULTS_BASE"
echo "============================================================"

run_config() {
    local CONFIG_NAME="$1"
    local POLICY="$2"
    local MI_ME="$3"
    local WPE_WCE="$4"
    local TI_TE="$5"
    local NPPC="$6"
    local CFL="$7"
    local WPEDT="$8"
    local PERT="$9"
    local CLEAN_DIV="${10}"

    local OUT_DIR="$RESULTS_BASE/${CONFIG_NAME}_${POLICY}"

    if [ -f "$OUT_DIR/benchmark_vpic_deck.csv" ]; then
        echo "  [$CONFIG_NAME/$POLICY] Already done, skipping."
        return
    fi

    echo ""
    echo "  [$CONFIG_NAME/$POLICY] Running..."
    echo "    Physics: mi_me=$MI_ME wpe_wce=$WPE_WCE Ti_Te=$TI_TE nppc=$NPPC"
    echo "    Solver:  cfl=$CFL wpedt=$WPEDT clean_div=$CLEAN_DIV pert=$PERT"
    mkdir -p "$OUT_DIR"

    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    VPIC_TIMESTEPS=$TIMESTEPS \
    VPIC_WARMUP_STEPS=$WARMUP \
    VPIC_SIM_INTERVAL=$SIM_INTERVAL \
    VPIC_POLICIES=$POLICY \
    VPIC_NX=$NX \
    VPIC_CHUNK_MB=$CHUNK_MB \
    VPIC_ERROR_BOUND=$ERROR_BOUND \
    VPIC_MI_ME=$MI_ME \
    VPIC_WPE_WCE=$WPE_WCE \
    VPIC_TI_TE=$TI_TE \
    VPIC_NPPC=$NPPC \
    VPIC_PERTURBATION=$PERT \
    VPIC_LR=$LR \
    VPIC_MAPE_THRESHOLD=$MAPE \
    VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
    VPIC_EXPLORE_K=$EXPLORE_K \
    VPIC_VERIFY=0 \
    VPIC_RESULTS_DIR="$OUT_DIR" \
    "$VPIC_BIN" 2>&1 | tee "$OUT_DIR/vpic.log" | tail -5

    echo "  [$CONFIG_NAME/$POLICY] Done."
}

# ── Config A: Original slow reconnection (commit bd3e8a8) ──
echo ""
echo ">>> Config A: Original slow reconnection"
run_config "configA_slow" "balanced" 25 3 1 2 0.99 0.36 0.1 0
run_config "configA_slow" "ratio"    25 3 1 2 0.99 0.36 0.1 0

# ── Config B: Fast reconnection (commit 2d0eb2f) ──
echo ""
echo ">>> Config B: Fast reconnection"
run_config "configB_fast" "balanced" 5 1 5 10 0.70 0.20 0.05 200
run_config "configB_fast" "ratio"    5 1 5 10 0.70 0.20 0.05 200

# ── Summary comparison ──
echo ""
echo "============================================================"
echo "Comparison Summary"
echo "============================================================"

python3 -c "
import csv, os, numpy as np

base = '$RESULTS_BASE'
configs = [
    ('configA_slow_balanced', 'A/balanced'),
    ('configA_slow_ratio',    'A/ratio'),
    ('configB_fast_balanced', 'B/balanced'),
    ('configB_fast_ratio',    'B/ratio'),
]

print(f'{\"Config\":>15} {\"Ratio\":>7} {\"WrBW\":>7} {\"MAPE_C\":>7} {\"MAPE_R\":>7} {\"SGD\":>5} {\"Expl\":>5} {\"Algos\":>6}')
print('-' * 70)

for dirname, label in configs:
    agg = os.path.join(base, dirname, 'benchmark_vpic_deck.csv')
    ts = os.path.join(base, dirname, 'benchmark_vpic_deck_timesteps.csv')
    tc = os.path.join(base, dirname, 'benchmark_vpic_deck_timestep_chunks.csv')

    if not os.path.isfile(agg):
        print(f'{label:>15} — not found')
        continue

    with open(agg) as f:
        r = list(csv.DictReader(f))[0]
        ratio = float(r.get('ratio', 0))
        wr = float(r.get('write_mibps', 0))
        mc = float(r.get('mape_comp_pct', 0))
        mr = float(r.get('mape_ratio_pct', 0))

    # Count SGD and exploration from per-timestep
    sgd_total, expl_total = 0, 0
    ratios = []
    if os.path.isfile(ts):
        with open(ts) as f:
            for row in csv.DictReader(f):
                sgd_total += int(row.get('sgd_fires', 0))
                expl_total += int(row.get('explorations', 0))
                ratios.append(float(row.get('ratio', 0)))

    # Count unique algorithms from chunk CSV
    unique_algos = set()
    if os.path.isfile(tc):
        with open(tc) as f:
            for row in csv.DictReader(f):
                unique_algos.add(row.get('action', '?'))

    # Ratio diversity (std across timesteps)
    ratio_std = np.std(ratios) if ratios else 0

    print(f'{label:>15} {ratio:>7.2f} {wr:>7.0f} {mc:>7.1f} {mr:>7.1f} {sgd_total:>5} {expl_total:>5} {len(unique_algos):>6}')

# Show ratio evolution for diversity
print()
print('Ratio evolution (first 5 → last 5 timesteps):')
for dirname, label in configs:
    ts = os.path.join(base, dirname, 'benchmark_vpic_deck_timesteps.csv')
    if not os.path.isfile(ts): continue
    with open(ts) as f:
        rows = list(csv.DictReader(f))
        first = [f'{float(r[\"ratio\"]):.2f}' for r in rows[:5]]
        last = [f'{float(r[\"ratio\"]):.2f}' for r in rows[-5:]]
        print(f'  {label:>15}: {\" \".join(first)} ... {\" \".join(last)}')
"

echo ""
echo "Results: $RESULTS_BASE"
echo "============================================================"
