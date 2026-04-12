#!/bin/bash
# ============================================================
# VPIC-Kokkos Benchmark Wrapper
#
# Runs the self-contained VPIC benchmark deck (Harris sheet
# reconnection) and post-processes results into per-policy
# subdirectories with figures.
#
# The deck runs all 12 phases on live GPU-resident field data
# each timestep with GPU weight isolation between NN phases.
#
# Usage:
#   bash benchmarks/vpic-kokkos/run_vpic_benchmark.sh
#
# Environment variables:
#   VPIC_NX            Grid size per dimension [200]
#   VPIC_CHUNK_MB      HDF5 chunk size in MB [4]
#   VPIC_TIMESTEPS     Benchmark write cycles [20]
#   VPIC_WARMUP_STEPS  Physics warmup steps [500]
#   VPIC_SIM_INTERVAL  Physics steps between writes [190]
#   POLICIES           NN policies [balanced,ratio,speed]
#   ERROR_BOUND        Lossy error bound [0.0 = lossless]
#   SGD_LR             SGD learning rate [0.2]
#   SGD_MAPE           MAPE threshold for SGD firing [0.10]
#   EXPLORE_K          Exploration alternatives [4]
#   EXPLORE_THRESH     Exploration error threshold [0.20]
#   RESULTS_DIR        Output directory [auto]
#   GPUC_DIR           GPUCompress root [auto-detected]
#   NO_RANKING         1 to skip Kendall tau profiler [0]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUC_DIR="${GPUC_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VPIC_DECK="${VPIC_DECK:-$SCRIPT_DIR/vpic_benchmark_deck.Linux}"

# ── Parameters ──
VPIC_NX="${VPIC_NX:-200}"
VPIC_CHUNK_MB="${VPIC_CHUNK_MB:-${CHUNK_MB:-4}}"
VPIC_TIMESTEPS="${VPIC_TIMESTEPS:-20}"
# Paper-grade defaults: 500 warmup ensures fields are well-developed and
# 190 sim_interval ensures consecutive snapshots differ enough for SGD to
# see meaningful adaptation pressure. With slow-reconnection physics
# (mi_me=25, wpe_wce=3) shorter values produce nearly-identical snapshots.
VPIC_WARMUP_STEPS="${VPIC_WARMUP_STEPS:-500}"
VPIC_SIM_INTERVAL="${VPIC_SIM_INTERVAL:-190}"
ERROR_BOUND="${ERROR_BOUND:-0.0}"
SGD_LR="${SGD_LR:-0.2}"
SGD_MAPE="${SGD_MAPE:-0.10}"
EXPLORE_K="${EXPLORE_K:-4}"
EXPLORE_THRESH="${EXPLORE_THRESH:-0.20}"
POLICIES=${POLICIES:-"balanced,ratio,speed"}
NO_RANKING="${NO_RANKING:-0}"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/vpic_eval_NX${VPIC_NX}_chunk${VPIC_CHUNK_MB}mb_ts${VPIC_TIMESTEPS}}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

# ── Environment ──
export LD_LIBRARY_PATH="/opt/hdf5/lib:/opt/nvcomp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPUCOMPRESS_WEIGHTS="$WEIGHTS"
export VPIC_NX VPIC_CHUNK_MB VPIC_TIMESTEPS VPIC_WARMUP_STEPS VPIC_SIM_INTERVAL
export VPIC_POLICIES="$POLICIES"
export VPIC_ERROR_BOUND="$ERROR_BOUND"
export VPIC_LR="$SGD_LR"
export VPIC_MAPE_THRESHOLD="$SGD_MAPE"
export VPIC_EXPLORE_K="$EXPLORE_K"
export VPIC_EXPLORE_THRESH="$EXPLORE_THRESH"
export VPIC_RESULTS_DIR="$RESULTS_DIR"

if [ "$NO_RANKING" = "1" ]; then
    export VPIC_NO_RANKING=1
fi

echo "============================================================"
echo "VPIC-Kokkos GPUCompress Benchmark"
echo "============================================================"
echo "  Deck:          $VPIC_DECK"
echo "  Grid:          ${VPIC_NX}^3"
echo "  Warmup:        $VPIC_WARMUP_STEPS steps"
echo "  Timesteps:     $VPIC_TIMESTEPS"
echo "  Sim interval:  $VPIC_SIM_INTERVAL"
echo "  Chunk:         ${VPIC_CHUNK_MB} MB"
echo "  Error bound:   $ERROR_BOUND"
echo "  Policies:      $POLICIES"
echo "  SGD LR:        $SGD_LR"
echo "  SGD MAPE:      $SGD_MAPE"
echo "  Explore K:     $EXPLORE_K"
echo "  Explore thresh:$EXPLORE_THRESH"
echo "  No ranking:    $NO_RANKING"
echo "  Weights:       $WEIGHTS"
echo "  Results:       $RESULTS_DIR"
echo ""

# ── Verify deck binary ──
if [ ! -x "$VPIC_DECK" ]; then
    echo "ERROR: VPIC deck not found: $VPIC_DECK"
    echo "  Build it: bash benchmarks/vpic-kokkos/build_vpic_pm.sh"
    exit 1
fi

# ── Verify weights ──
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ============================================================
# Phase 1: Run VPIC benchmark deck
# ============================================================
echo ">>> Phase 1: Running VPIC benchmark deck"
echo ""

mpirun --oversubscribe -np 1 "$VPIC_DECK" 2>&1 | tee "$RESULTS_DIR/vpic_benchmark.log"

echo ""
echo ">>> Phase 1 complete"
echo ""

# ============================================================
# Phase 2: Split results per policy
# ============================================================
echo ">>> Phase 2: Splitting results per policy"

TSTEP_CSV="$RESULTS_DIR/benchmark_vpic_deck_timesteps.csv"
CHUNKS_CSV="$RESULTS_DIR/benchmark_vpic_deck_timestep_chunks.csv"
AGG_CSV="$RESULTS_DIR/benchmark_vpic_deck.csv"

if [ ! -f "$TSTEP_CSV" ]; then
    echo "ERROR: Timestep CSV not found: $TSTEP_CSV"
    exit 1
fi

IFS=',' read -ra POL_ARRAY <<< "$POLICIES"

python3 - "$RESULTS_DIR" "${POL_ARRAY[@]}" <<'PYEOF'
import csv, os, sys, math, shutil
from collections import defaultdict

results_dir = sys.argv[1]
policies = sys.argv[2:]

ts_csv = os.path.join(results_dir, 'benchmark_vpic_deck_timesteps.csv')
tc_csv = os.path.join(results_dir, 'benchmark_vpic_deck_timestep_chunks.csv')
agg_csv = os.path.join(results_dir, 'benchmark_vpic_deck.csv')

# Read source CSVs
with open(ts_csv) as f:
    ts_header = next(csv.reader(f))
    ts_rows = list(csv.reader(f))

tc_header, tc_rows = None, []
if os.path.exists(tc_csv):
    with open(tc_csv) as f:
        tc_header = next(csv.reader(f))
        tc_rows = list(csv.reader(f))

agg_header, agg_rows = None, []
if os.path.exists(agg_csv):
    with open(agg_csv) as f:
        agg_header = next(csv.reader(f))
        agg_rows = list(csv.reader(f))

for pol in policies:
    pol_dir = os.path.join(results_dir, pol)
    os.makedirs(pol_dir, exist_ok=True)

    # Split timestep CSV: fixed phases + NN phases for this policy
    with open(os.path.join(pol_dir, 'benchmark_vpic_deck_timesteps.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(ts_header)
        for row in list(ts_rows):  # copy to avoid mutation issues
            r = list(row)
            phase = r[1]
            if '/' not in phase:
                w.writerow(r)
            elif phase.endswith('/' + pol):
                r[1] = phase.rsplit('/', 1)[0]
                w.writerow(r)

    # Split timestep-chunks CSV
    if tc_header:
        with open(os.path.join(pol_dir, 'benchmark_vpic_deck_timestep_chunks.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(tc_header)
            for row in list(tc_rows):
                r = list(row)
                phase = r[1]
                if '/' not in phase:
                    w.writerow(r)
                elif phase.endswith('/' + pol):
                    r[1] = phase.rsplit('/', 1)[0]
                    w.writerow(r)

    # Build per-policy aggregate from timestep data
    with open(os.path.join(pol_dir, 'benchmark_vpic_deck_timesteps.csv')) as f:
        reader = csv.DictReader(f)
        pol_ts = list(reader)

    groups = defaultdict(list)
    for r in pol_ts:
        groups[r['phase']].append(r)

    if agg_header:
        with open(os.path.join(pol_dir, 'benchmark_vpic_deck.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(agg_header)
            for row in agg_rows:
                r = list(row)
                phase = r[2]
                if phase not in groups:
                    continue
                g = groups[phase]
                n = len(g)
                avg_wr = sum(float(x['write_ms']) for x in g) / n
                avg_rd = sum(float(x['read_ms']) for x in g) / n
                wr_var = sum((float(x['write_ms']) - avg_wr)**2 for x in g) / max(n-1, 1)
                rd_var = sum((float(x['read_ms']) - avg_rd)**2 for x in g) / max(n-1, 1)
                total_file = sum(int(x.get('file_bytes', 0)) for x in g)
                orig_mib = float(g[0].get('orig_mib') or total_file / n / (1024*1024))
                avg_file_mib = total_file / n / (1024*1024) if n > 0 else 0
                avg_ratio = (n * orig_mib * 1024*1024) / total_file if total_file > 0 else 1.0
                wr_mbps = orig_mib / (avg_wr / 1000.0) if avg_wr > 0 else 0
                rd_mbps = orig_mib / (avg_rd / 1000.0) if avg_rd > 0 else 0
                r[3] = str(n)
                r[4] = f'{avg_wr:.2f}'
                r[5] = f'{math.sqrt(wr_var):.2f}'
                r[6] = f'{avg_rd:.2f}'
                r[7] = f'{math.sqrt(rd_var):.2f}'
                r[8] = f'{avg_file_mib:.2f}'
                r[10] = f'{avg_ratio:.4f}'
                r[11] = f'{wr_mbps:.1f}'
                r[12] = f'{rd_mbps:.1f}'
                w.writerow(r)

    # Copy ranking CSVs
    for fname in ['benchmark_vpic_deck_ranking.csv', 'benchmark_vpic_deck_ranking_costs.csv']:
        src = os.path.join(results_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pol_dir, fname))

    n_files = len(os.listdir(pol_dir))
    print(f"  {pol}/: {n_files} files")

print("  Split complete")
PYEOF

echo ""

# ============================================================
# Phase 3: Generate figures per policy
# ============================================================
echo ">>> Phase 3: Generating figures"

VISUALIZER="$GPUC_DIR/benchmarks/visualize.py"
if [ ! -f "$VISUALIZER" ]; then
    echo "WARNING: visualize.py not found at $VISUALIZER — skipping figures"
else
    for policy in "${POL_ARRAY[@]}"; do
        POL_DIR="$RESULTS_DIR/$policy"
        FIG_DIR="$POL_DIR/figures"
        echo "  ── $policy ──"
        python3 "$VISUALIZER" \
            --vpic-dir "$POL_DIR" \
            --output-dir "$FIG_DIR" \
            2>&1 | grep -E "Saved:|Done" || true
        N_FIGS=$(ls "$FIG_DIR"/*.png 2>/dev/null | wc -l)
        echo "    $N_FIGS figures in $FIG_DIR/"
    done
fi

echo ""
echo "============================================================"
echo "VPIC Benchmark Complete"
echo "============================================================"
echo "  Results:    $RESULTS_DIR/"
for policy in "${POL_ARRAY[@]}"; do
    echo "    $policy/     CSVs + figures"
done
echo ""
echo "  Raw CSVs:   $RESULTS_DIR/benchmark_vpic_deck_*.csv"
echo "  Log:        $RESULTS_DIR/vpic_benchmark.log"
echo ""
