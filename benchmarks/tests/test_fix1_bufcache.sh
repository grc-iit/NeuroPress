#!/bin/bash
# ============================================================
# Fix 1 (Buffer Cache) Before/After Benchmark
#
# Measures vol_setup_ms, h5dclose_ms, write_ms across phases
# and chunk configs to quantify the buffer cache improvement.
#
# Usage:
#   bash benchmarks/tests/test_fix1_bufcache.sh baseline
#   # apply fix, rebuild
#   bash benchmarks/tests/test_fix1_bufcache.sh after
#   # compare
#   bash benchmarks/tests/test_fix1_bufcache.sh compare
# ============================================================
set +e

TAG="${1:-baseline}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GS_BIN="$GPU_DIR/build/grayscott_benchmark_pm"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
RESULTS_DIR="$GPU_DIR/benchmarks/tests/fix1_results"

if [ "$TAG" = "compare" ]; then
    echo "============================================================"
    echo "  Fix 1 Before/After Comparison"
    echo "============================================================"
    echo ""
    B="$RESULTS_DIR/baseline/summary.csv"
    A="$RESULTS_DIR/after/summary.csv"
    if [ ! -f "$B" ] || [ ! -f "$A" ]; then
        echo "ERROR: Need both baseline and after runs first"
        exit 1
    fi
    echo "config,phase,ts,b_write,a_write,delta_write,b_setup,a_setup,delta_setup,b_dclose,a_dclose,delta_dclose,b_pipeline,a_pipeline,delta_pipeline"
    paste -d'|' \
        <(tail -n+2 "$B" | sort -t',' -k1,4) \
        <(tail -n+2 "$A" | sort -t',' -k1,4) | \
    awk -F'[,|]' '{
        printf "%s,%s,T%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            $2,$3,$4,
            $5,$17,$17-$5,
            $6,$18,$18-$6,
            $7,$19,$19-$7,
            $8,$20,$20-$8
    }'
    echo ""
    echo "=== Averages (T>0 steady state only) ==="
    paste -d'|' \
        <(tail -n+2 "$B" | sort -t',' -k1,4) \
        <(tail -n+2 "$A" | sort -t',' -k1,4) | \
    awk -F'[,|]' 'BEGIN{n=0; dw=0; ds=0; dc=0; dp=0}
        $4>0 {
            n++; dw+=$17-$5; ds+=$18-$6; dc+=$19-$7; dp+=$20-$8
        }
        END {
            if(n>0) printf "  write_ms: %+.1f ms/write (avg over %d measurements)\n  setup_ms: %+.1f ms/write\n  dclose_ms: %+.1f ms/write\n  pipeline_ms: %+.1f ms/write\n", dw/n, n, ds/n, dc/n, dp/n
        }'
    exit 0
fi

OUT_DIR="$RESULTS_DIR/$TAG"
mkdir -p "$OUT_DIR"

# Test configs: 2 sizes × 3 phases × 5 timesteps
# 128MB dataset with 64MB chunks (2 chunks) and 16MB chunks (8 chunks)
declare -a CONFIGS=(
    "L=320 CHUNK_MB=64 LABEL=2chunks"
    "L=320 CHUNK_MB=16 LABEL=8chunks"
)
TIMESTEPS=5
PHASES="lz4 nn nn-rl"

echo "============================================================"
echo "  Fix 1 Benchmark: ${TAG}"
echo "  Configs: 128MB/64MB (2 chunks), 128MB/16MB (8 chunks)"
echo "  Phases: ${PHASES}"
echo "  Timesteps: ${TIMESTEPS}"
echo "  Output: ${OUT_DIR}"
echo "============================================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
    eval "$cfg"
    echo "--- Config: ${LABEL} (L=${L}, chunk=${CHUNK_MB}MB) ---"

    for phase in $PHASES; do
        PHASE_DIR="$OUT_DIR/${LABEL}/phase_${phase}"
        mkdir -p "$PHASE_DIR"

        GPUCOMPRESS_DETAILED_TIMING=1 \
        GPUCOMPRESS_DEBUG_NN=0 \
        "$GS_BIN" "$WEIGHTS" \
            --L $L --steps 100 --chunk-mb $CHUNK_MB --timesteps $TIMESTEPS \
            --phase "$phase" \
            --w0 1.0 --w1 1.0 --w2 1.0 \
            --out-dir "$PHASE_DIR" \
            > "$PHASE_DIR/log.txt" 2>&1

        echo "  $phase done"
    done
    echo ""
done

# ── Build summary CSV ──
SUMMARY="$OUT_DIR/summary.csv"
echo "tag,config,phase,timestep,write_ms,setup_ms,dclose_ms,pipeline_ms,vol_s1_ms,vol_s2_ms,vol_s3_ms,stats_ms,nn_ms" > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
    eval "$cfg"
    for phase in $PHASES; do
        TS_CSV="$OUT_DIR/${LABEL}/phase_${phase}/benchmark_grayscott_timesteps.csv"
        if [ -f "$TS_CSV" ]; then
            # Columns: 1=phase,2=ts,4=write_ms,18=stats_ms,19=nn_ms,
            #          28=vol_s1,29=vol_s2,30=vol_s3,
            #          31=h5dwrite,33=h5dclose,
            #          35=setup,36=pipeline
            tail -n+2 "$TS_CSV" | awk -F',' -v tag="$TAG" -v cfg="$LABEL" \
                '{printf "%s,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f\n",
                    tag,cfg,$1,$2,$4,$35,$33,$36,$28,$29,$30,$18,$19}' >> "$SUMMARY"
        fi
    done
done

echo "============================================================"
echo "  Results Summary: ${TAG}"
echo "============================================================"
echo ""
echo "config,phase,ts,write_ms,setup_ms,dclose_ms,pipeline_ms"
tail -n+2 "$SUMMARY" | awk -F',' '{printf "%-9s %-6s T%s  write=%7.1f  setup=%7.1f  dclose=%6.1f  pipeline=%6.1f\n",$2,$3,$4,$5,$6,$7,$8}'
echo ""
echo "Summary CSV: $SUMMARY"
echo "Done."
