#!/bin/bash
# ============================================================
# Fix 2 Baseline / After Test
#
# Measures per-chunk vol_stats_malloc_ms and overall Stage 1 timing
# for NN phases to quantify d_stats_copy pre-allocation improvement.
#
# Usage:
#   bash benchmarks/tests/test_fix2_baseline.sh          # baseline
#   # apply fix, rebuild
#   bash benchmarks/tests/test_fix2_baseline.sh after     # after
# ============================================================
set -e

TAG="${1:-baseline}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GS_BIN="$GPU_DIR/build/grayscott_benchmark_pm"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
OUT_DIR="$GPU_DIR/benchmarks/tests/fix2_results/${TAG}"
mkdir -p "$OUT_DIR"

# Test configs: vary chunk count to see scaling
# Config 1: 128MB / 64MB = 2 chunks
# Config 2: 128MB / 16MB = 8 chunks  (need L=320, chunk=16)
# Config 3: 128MB / 8MB  = 16 chunks (need L=320, chunk=8)
declare -a CONFIGS=(
    "L=320 CHUNK_MB=64 LABEL=2chunks"
    "L=320 CHUNK_MB=16 LABEL=8chunks"
    "L=320 CHUNK_MB=8 LABEL=16chunks"
)

echo "============================================================"
echo "  Fix 2 Test: ${TAG}"
echo "  Phases: nn, nn-rl, nn-rl+exp50"
echo "  Timesteps: 3"
echo "  Output: ${OUT_DIR}"
echo "============================================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
    eval "$cfg"
    echo "--- Config: ${LABEL} (L=${L}, chunk=${CHUNK_MB}MB) ---"

    for phase in nn nn-rl nn-rl+exp50; do
        PHASE_DIR="$OUT_DIR/${LABEL}/phase_${phase}"
        mkdir -p "$PHASE_DIR"

        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=0 \
        "$GS_BIN" "$WEIGHTS" \
            --L $L --steps 100 --chunk-mb $CHUNK_MB --timesteps 3 \
            --phase "$phase" \
            --w0 1.0 --w1 1.0 --w2 1.0 \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/log.txt" 2>&1

        echo "  $phase done"
    done
    echo ""
done

# ── Extract key metrics ──
echo "============================================================"
echo "  Results Summary: ${TAG}"
echo "============================================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
    eval "$cfg"
    echo "=== ${LABEL} ==="

    for phase in nn nn-rl nn-rl+exp50; do
        TS_CSV="$OUT_DIR/${LABEL}/phase_${phase}/benchmark_grayscott_timesteps.csv"
        TC_CSV="$OUT_DIR/${LABEL}/phase_${phase}/benchmark_grayscott_timestep_chunks.csv"

        if [ -f "$TS_CSV" ]; then
            echo "  [$phase] Timestep CSV:"
            echo "    phase,ts,write_ms,stats_ms,nn_ms,vol_s1,vol_s2,setup,pipeline"
            tail -n+2 "$TS_CSV" | awk -F',' '{printf "    %s,T%s,%.1f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f\n",$1,$2,$4,$18,$19,$28,$29,$35,$36}'
        fi

        # Extract vol_stats_malloc_ms from chunk-level diagnostics in logs
        LOG="$OUT_DIR/${LABEL}/phase_${phase}/log.txt"
        if [ -f "$LOG" ]; then
            MALLOC_LINES=$(grep -o "stats_malloc=[0-9.]*" "$LOG" 2>/dev/null | head -5)
            if [ -n "$MALLOC_LINES" ]; then
                echo "    vol_stats_malloc_ms samples: $MALLOC_LINES"
            fi
        fi
    done
    echo ""
done

# Save summary CSV
SUMMARY="$OUT_DIR/summary.csv"
echo "tag,config,phase,timestep,write_ms,stats_ms,nn_ms,vol_s1_ms,vol_s2_ms,setup_ms,pipeline_ms" > "$SUMMARY"
for cfg in "${CONFIGS[@]}"; do
    eval "$cfg"
    for phase in nn nn-rl nn-rl+exp50; do
        TS_CSV="$OUT_DIR/${LABEL}/phase_${phase}/benchmark_grayscott_timesteps.csv"
        if [ -f "$TS_CSV" ]; then
            tail -n+2 "$TS_CSV" | awk -F',' -v tag="$TAG" -v cfg="$LABEL" \
                '{printf "%s,%s,%s,%s,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",tag,cfg,$1,$2,$4,$18,$19,$28,$29,$35,$36}' >> "$SUMMARY"
        fi
    done
done
echo "Summary CSV: $SUMMARY"
echo "Done."
