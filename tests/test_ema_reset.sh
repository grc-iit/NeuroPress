#!/bin/bash
# ============================================================
# Test: EMA gradient buffer reset on weight reload
#
# The bug: when generic_benchmark runs nn-rl then nn-rl+exp50
# in the same process, gpucompress_reload_nn() reloads fresh
# weights but does NOT zero the EMA gradient buffer. The nn-rl
# phase's SGD training leaves non-zero EMA which contaminates
# the nn-rl+exp50 phase's first ~15 SGD calls.
#
# Test method:
#   Run A: Only nn-rl+exp50 (clean process, no prior nn-rl)
#   Run B: nn-rl THEN nn-rl+exp50 in same process
#
# With the fix: Run A and Run B's nn-rl+exp50 MAPE should match
# Without fix: Run B's nn-rl+exp50 fields 5-15 differ (stale EMA)
#
# Usage:
#   bash tests/test_ema_reset.sh
# ============================================================
set -e

GPUC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$GPUC_DIR/build/generic_benchmark"
WEIGHTS="$GPUC_DIR/neural_net/weights/model.nnwt"

export LD_LIBRARY_PATH="$GPUC_DIR/build:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64"
export NO_RANKING=1

# Use NYX Sedov fields if available, otherwise LAMMPS
if [ -d "/tmp/nyx_smoke_test/flat_fields" ]; then
    DATA_DIR="/tmp/nyx_smoke_test/flat_fields"
    NAME="nyx"
elif [ -d "/tmp/lammps_smoke_test/positions_only" ]; then
    DATA_DIR="/tmp/lammps_smoke_test/positions_only"
    NAME="lammps"
else
    echo "ERROR: No test data. Run a smoke test first."
    exit 1
fi

FIRST_FILE=$(ls "$DATA_DIR"/*.f32 | head -1)
N_FLOATS=$(( $(stat -L -c%s "$FIRST_FILE") / 4 ))
DIMS="${N_FLOATS},1"
N_FILES=$(ls "$DATA_DIR"/*.f32 | wc -l)
echo "Data: $N_FILES files, ${N_FLOATS} floats each"
echo ""

OUT="/tmp/ema_reset_test"
rm -rf "$OUT"

# ── Run A: nn-rl+exp50 ONLY (clean process, no prior nn-rl) ──
echo ">>> Run A: nn-rl+exp50 only (clean EMA)"
mkdir -p "$OUT/run_a"
"$BIN" "$WEIGHTS" \
    --data-dir "$DATA_DIR" --dims "$DIMS" --ext .f32 --chunk-mb 4 \
    --out-dir "$OUT/run_a" --name "$NAME" \
    --w0 1 --w1 1 --w2 1 --warmup-skip 0 \
    --phase nn-rl+exp50 \
    > "$OUT/run_a/log.txt" 2>&1
echo "  Done."

# ── Run B: nn-rl THEN nn-rl+exp50 (same process, EMA stale) ──
echo ">>> Run B: nn-rl + nn-rl+exp50 (EMA from nn-rl leaks into nn-rl+exp50)"
mkdir -p "$OUT/run_b"
"$BIN" "$WEIGHTS" \
    --data-dir "$DATA_DIR" --dims "$DIMS" --ext .f32 --chunk-mb 4 \
    --out-dir "$OUT/run_b" --name "$NAME" \
    --w0 1 --w1 1 --w2 1 --warmup-skip 0 \
    --phase nn-rl --phase nn-rl+exp50 \
    > "$OUT/run_b/log.txt" 2>&1
echo "  Done."

# ── Compare nn-rl+exp50 MAPE between Run A and Run B ──
echo ""
echo "============================================================"
echo "MAPE comparison: nn-rl+exp50 (clean vs after nn-rl)"
echo "============================================================"

CSV_A="$OUT/run_a/benchmark_${NAME}_timesteps.csv"
CSV_B="$OUT/run_b/benchmark_${NAME}_timesteps.csv"

if [ ! -f "$CSV_A" ] || [ ! -f "$CSV_B" ]; then
    echo "ERROR: Missing timestep CSVs"
    ls -la "$OUT"/run_*/benchmark_*_timesteps.csv 2>/dev/null
    exit 1
fi

python3 -c "
import csv, sys

def read_phase_mapes(path, phase):
    mapes = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['phase'] == phase:
                mapes.append(float(row['mape_ratio']))
    return mapes

# Run A: only has nn-rl+exp50
a = read_phase_mapes('$CSV_A', 'nn-rl+exp50')

# Run B: has both nn-rl and nn-rl+exp50. Extract nn-rl+exp50 only.
b = read_phase_mapes('$CSV_B', 'nn-rl+exp50')

n = min(len(a), len(b))
if n == 0:
    print('ERROR: No nn-rl+exp50 rows found')
    sys.exit(1)

print(f'Fields compared: {n}')
print()
print(f'{\"Field\":<6} {\"Clean(A)\":<12} {\"After-RL(B)\":<12} {\"Delta\":<10} {\"Status\":<10}')
print(f'{\"-----\":<6} {\"--------\":<12} {\"-----------\":<12} {\"-----\":<10} {\"------\":<10}')

deltas = []
for i in range(n):
    d = abs(a[i] - b[i])
    deltas.append(d)
    status = 'DIFFERS' if d > 2.0 else 'ok'
    # Highlight the critical window (fields 5-15 after warmup guard lifts)
    marker = ' *' if 4 <= i <= 15 else ''
    print(f'{i:<6} {a[i]:<12.2f} {b[i]:<12.2f} {d:<10.2f} {status}{marker}')

print()
mean_d = sum(deltas) / n
max_d = max(deltas)
# Focus on the critical window: fields 5-15
window = deltas[4:16] if n > 15 else deltas[4:]
win_mean = sum(window) / len(window) if window else 0
n_differs = sum(1 for d in deltas if d > 2.0)

print(f'=== Summary ===')
print(f'  Mean |delta| (all):       {mean_d:.2f}%')
print(f'  Mean |delta| (fields 5-15): {win_mean:.2f}%  (critical EMA window)')
print(f'  Max |delta|:              {max_d:.2f}%')
print(f'  Fields > 2% diff:         {n_differs}/{n}')
print()
print(f'  * = critical window (fields 4-15): warmup guard lifted, stale EMA active')
print()
if win_mean < 1.0 and max_d < 3.0:
    print('  RESULT: PASS — EMA appears clean (no measurable contamination)')
elif n_differs >= 3 and win_mean > 2.0:
    print('  RESULT: FAIL — Stale EMA detected in critical window')
else:
    print('  RESULT: INCONCLUSIVE — small differences (may be GPU timing noise)')
"
