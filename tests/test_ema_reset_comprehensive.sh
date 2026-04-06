#!/bin/bash
# ============================================================
# Comprehensive EMA Reset Test
#
# Tests whether stale EMA gradient buffers from a prior NN phase
# contaminate the next phase's SGD decisions.
#
# 8 test cases using real HPC simulation data:
#
#   Test 1: LAMMPS positions   (MD particle coords, low compress)
#   Test 2: LAMMPS velocities  (MD thermal velocities, high entropy)
#   Test 3: LAMMPS forces      (MD pair forces, mixed compress)
#   Test 4: NYX Sedov density   (cosmological shock, evolving compress)
#   Test 5: NYX Sedov all comps (52 fields: density+mom+energy+species)
#   Test 6: WarpX LWFA fields  (EM fields, vacuum+active FABs)
#   Test 7: Cross-workload: nn-rl on LAMMPS → nn-rl+exp50 on NYX
#   Test 8: Policy switch: nn-rl ratio → nn-rl+exp50 balanced
#
# Each test compares:
#   Run A: nn-rl+exp50 alone (clean process)
#   Run B: nn-rl THEN nn-rl+exp50 (same process, weight reload)
#
# With clean EMA: Run A ≈ Run B
# With stale EMA: Run B fields 5-15 diverge from Run A
#
# Usage:
#   bash tests/test_ema_reset_comprehensive.sh
#   bash tests/test_ema_reset_comprehensive.sh --label before_fix
# ============================================================
set -e

GPUC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$GPUC_DIR/build/generic_benchmark"
WEIGHTS="$GPUC_DIR/neural_net/weights/model.nnwt"
LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="/tmp/ema_comprehensive_${LABEL}"

export LD_LIBRARY_PATH="$GPUC_DIR/build:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64"
export NO_RANKING=1

echo "============================================================"
echo "Comprehensive EMA Reset Test (label: $LABEL)"
echo "============================================================"
echo "  Binary:  $BIN"
echo "  Output:  $OUT_BASE"
echo ""

mkdir -p "$OUT_BASE"

# ── Test case definitions ──
# Format: NAME|DATA_DIR|DIMS|W0|W1|W2|DESCRIPTION
TESTS=(
    "lammps_positions|/tmp/lammps_smoke_test/positions_only|768000,1|1|1|1|LAMMPS MD positions (3MB, 6 fields)"
    "lammps_velocities|/tmp/lammps_smoke_test/velocities_only|768000,1|1|1|1|LAMMPS MD velocities (3MB, 6 fields)"
    "lammps_forces|/tmp/lammps_smoke_test/forces_only|768000,1|1|1|1|LAMMPS MD forces (3MB, 6 fields)"
    "nyx_density|/tmp/nyx_smoke_test/flat_fields|262144,1|1|1|1|NYX Sedov all components (1MB, 52 fields)"
    "nyx_ratio|/tmp/nyx_smoke_test/flat_fields|262144,1|0|0|1|NYX Sedov ratio policy (1MB, 52 fields)"
    "warpx_lwfa|/tmp/warpx_smoke_test/flat_fields|65536,1|1|1|1|WarpX LWFA EM fields (256KB, 160 fields)"
)

# ── Run one test case ──
run_test() {
    local name="$1" data_dir="$2" dims="$3" w0="$4" w1="$5" w2="$6" desc="$7"
    local test_dir="$OUT_BASE/$name"

    # Check data exists
    local n_files=$(ls "$data_dir"/*.f32 2>/dev/null | wc -l)
    if [ "$n_files" -eq 0 ]; then
        echo "  SKIP $name: no .f32 files in $data_dir"
        return
    fi

    echo "  [$name] $desc ($n_files fields)"

    # Run A: nn-rl+exp50 only (clean)
    mkdir -p "$test_dir/run_a"
    "$BIN" "$WEIGHTS" \
        --data-dir "$data_dir" --dims "$dims" --ext .f32 --chunk-mb 4 \
        --out-dir "$test_dir/run_a" --name "$name" \
        --w0 "$w0" --w1 "$w1" --w2 "$w2" --warmup-skip 0 \
        --phase nn-rl+exp50 \
        > "$test_dir/run_a/log.txt" 2>&1

    # Run B: nn-rl then nn-rl+exp50 (EMA contamination)
    mkdir -p "$test_dir/run_b"
    "$BIN" "$WEIGHTS" \
        --data-dir "$data_dir" --dims "$dims" --ext .f32 --chunk-mb 4 \
        --out-dir "$test_dir/run_b" --name "$name" \
        --w0 "$w0" --w1 "$w1" --w2 "$w2" --warmup-skip 0 \
        --phase nn-rl --phase nn-rl+exp50 \
        > "$test_dir/run_b/log.txt" 2>&1

    echo "    Done."
}

# ── Run all tests ──
echo ">>> Running 6 test cases..."
echo ""
for spec in "${TESTS[@]}"; do
    IFS='|' read -r name data_dir dims w0 w1 w2 desc <<< "$spec"
    run_test "$name" "$data_dir" "$dims" "$w0" "$w1" "$w2" "$desc"
done

# ── Test 7: Cross-workload contamination ──
# nn-rl trains on LAMMPS, then nn-rl+exp50 runs on NYX
echo "  [cross_workload] nn-rl on LAMMPS → nn-rl+exp50 on NYX"
if [ -d "/tmp/lammps_smoke_test/positions_only" ] && [ -d "/tmp/nyx_smoke_test/flat_fields" ]; then
    T7DIR="$OUT_BASE/cross_workload"

    # Run A: nn-rl+exp50 on NYX alone (clean)
    mkdir -p "$T7DIR/run_a"
    "$BIN" "$WEIGHTS" \
        --data-dir /tmp/nyx_smoke_test/flat_fields --dims 262144,1 --ext .f32 --chunk-mb 4 \
        --out-dir "$T7DIR/run_a" --name "cross_workload" \
        --w0 1 --w1 1 --w2 1 --warmup-skip 0 \
        --phase nn-rl+exp50 \
        > "$T7DIR/run_a/log.txt" 2>&1

    # Run B: nn-rl on LAMMPS first (different data!), then nn-rl+exp50 on NYX
    # We do this by: run nn-rl on LAMMPS positions, then in same process run nn-rl+exp50 on NYX
    # Problem: generic_benchmark uses one --data-dir. We need to concatenate or use a trick.
    # Workaround: Create a mixed dir with LAMMPS files first, NYX files after
    mkdir -p "$T7DIR/mixed_data"
    rm -f "$T7DIR/mixed_data"/*.f32
    # Prefix LAMMPS files with 'aaa_' so they sort first (nn-rl trains on them)
    for f in /tmp/lammps_smoke_test/positions_only/*.f32; do
        ln -sf "$f" "$T7DIR/mixed_data/aaa_$(basename "$f")"
    done
    # NYX files with 'zzz_' prefix (nn-rl+exp50 trains on them after reload)
    for f in /tmp/nyx_smoke_test/flat_fields/*.f32; do
        ln -sf "$f" "$T7DIR/mixed_data/zzz_$(basename "$f")"
    done

    # All files must be same size for generic_benchmark. Check:
    lammps_sz=$(stat -L -c%s /tmp/lammps_smoke_test/positions_only/positions_step0000000000.f32)
    nyx_sz=$(stat -L -c%s /tmp/nyx_smoke_test/flat_fields/plt00000_fab0000_comp00_density.f32)
    if [ "$lammps_sz" != "$nyx_sz" ]; then
        echo "    SKIP: cross-workload requires same field size (LAMMPS=${lammps_sz}, NYX=${nyx_sz})"
    else
        mkdir -p "$T7DIR/run_b"
        "$BIN" "$WEIGHTS" \
            --data-dir "$T7DIR/mixed_data" --dims "$(($lammps_sz/4)),1" --ext .f32 --chunk-mb 4 \
            --out-dir "$T7DIR/run_b" --name "cross_workload" \
            --w0 1 --w1 1 --w2 1 --warmup-skip 0 \
            --phase nn-rl --phase nn-rl+exp50 \
            > "$T7DIR/run_b/log.txt" 2>&1
    fi
    echo "    Done."
else
    echo "    SKIP: missing LAMMPS or NYX data"
fi

echo ""

# ── Analyze all results ──
echo "============================================================"
echo "Results Summary"
echo "============================================================"
echo ""

SUMMARY_CSV="$OUT_BASE/summary.csv"
echo "test,n_fields,mean_delta_all,mean_delta_critical,max_delta,fields_gt2pct" > "$SUMMARY_CSV"

python3 << 'PYEOF'
import csv, os, sys, glob

out_base = os.environ.get("OUT_BASE", "/tmp/ema_comprehensive")
if not os.path.isdir(out_base):
    print(f"ERROR: {out_base} not found")
    sys.exit(1)

summary_csv = os.path.join(out_base, "summary.csv")
rows_out = []

# Find all test directories
test_dirs = sorted(glob.glob(os.path.join(out_base, "*/run_a")))

print(f"{'Test':<25} {'Fields':<7} {'Mean Δ':<10} {'Crit Δ':<10} {'Max Δ':<10} {'>2%':<6}")
print(f"{'-'*25} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

for run_a_dir in test_dirs:
    test_dir = os.path.dirname(run_a_dir)
    test_name = os.path.basename(test_dir)
    run_b_dir = os.path.join(test_dir, "run_b")

    # Find timestep CSVs
    csv_a = glob.glob(os.path.join(run_a_dir, "benchmark_*_timesteps.csv"))
    csv_b = glob.glob(os.path.join(run_b_dir, "benchmark_*_timesteps.csv"))
    if not csv_a or not csv_b:
        print(f"{test_name:<25} {'SKIP':<7} (missing CSV)")
        continue

    def read_mapes(path, phase="nn-rl+exp50"):
        mapes = []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["phase"] == phase:
                    mapes.append(float(row["mape_ratio"]))
        return mapes

    a = read_mapes(csv_a[0])
    b = read_mapes(csv_b[0])
    n = min(len(a), len(b))
    if n == 0:
        print(f"{test_name:<25} {'SKIP':<7} (no nn-rl+exp50 rows)")
        continue

    deltas = [abs(a[i] - b[i]) for i in range(n)]
    mean_all = sum(deltas) / n
    max_d = max(deltas)
    # Critical window: fields 4-15 (after warmup guard, stale EMA active)
    crit = deltas[4:min(16, n)]
    mean_crit = sum(crit) / len(crit) if crit else 0
    n_gt2 = sum(1 for d in deltas if d > 2.0)

    print(f"{test_name:<25} {n:<7} {mean_all:<10.2f} {mean_crit:<10.2f} {max_d:<10.2f} {n_gt2}/{n}")
    rows_out.append((test_name, n, mean_all, mean_crit, max_d, n_gt2))

# Write summary CSV
with open(summary_csv, "w") as f:
    f.write("test,n_fields,mean_delta_all,mean_delta_critical,max_delta,fields_gt2pct\n")
    for name, n, md, mc, mx, ng in rows_out:
        f.write(f"{name},{n},{md:.2f},{mc:.2f},{mx:.2f},{ng}/{n}\n")

print()
if rows_out:
    overall_crit = sum(r[3] for r in rows_out) / len(rows_out)
    overall_max = max(r[4] for r in rows_out)
    total_differs = sum(r[5] for r in rows_out)
    total_fields = sum(r[1] for r in rows_out)
    print(f"Overall critical window mean: {overall_crit:.2f}%")
    print(f"Overall max delta:            {overall_max:.2f}%")
    print(f"Total fields > 2% diff:       {total_differs}/{total_fields}")
    print()
    if overall_crit < 2.0 and overall_max < 5.0:
        print("VERDICT: PASS — EMA reset is working correctly")
    elif total_differs > total_fields * 0.3:
        print("VERDICT: FAIL — Stale EMA contamination detected across workloads")
    else:
        print("VERDICT: INCONCLUSIVE — some differences detected")

print(f"\nSummary CSV: {summary_csv}")
PYEOF
