#!/bin/bash
# ============================================================
# CPU Zstd Baseline Benchmark
# Compresses SDRBench dataset files using CPU zstd at the same
# chunk size as the GPU benchmark for a fair comparison.
#
# Each file is split into 4 MB chunks, each chunk compressed
# separately, matching the GPU benchmark's chunking behavior.
#
# Usage:
#   bash benchmarks/sdrbench/run_cpu_zstd_baseline.sh
#
# Override defaults:
#   CHUNK_MB=8 ZSTD_LEVEL=3 bash benchmarks/sdrbench/run_cpu_zstd_baseline.sh
#
# Output:
#   benchmarks/sdrbench/results/cpu_zstd_baseline.csv
# ============================================================
set -e

# Parse CLI arguments (override env vars)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunk-mb)   CHUNK_MB="$2";    shift 2 ;;
        --level)      ZSTD_LEVEL="$2";  shift 2 ;;
        --runs)       RUNS="$2";        shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--chunk-mb N] [--level N] [--runs N]"
            echo "  --chunk-mb N   Chunk size in MB (default: 4, must match GPU benchmark)"
            echo "  --level N      Zstd compression level 1-19 (default: 3)"
            echo "  --runs N       Number of repetitions for mean +/- std (default: 5)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CHUNK_MB=${CHUNK_MB:-4}
ZSTD_LEVEL=${ZSTD_LEVEL:-3}
RUNS=${RUNS:-5}
CHUNK_BYTES=$((CHUNK_MB * 1024 * 1024))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$GPU_DIR/data/sdrbench"
OUT_DIR="$SCRIPT_DIR/results"
OUT_CSV="$OUT_DIR/cpu_zstd_baseline.csv"

mkdir -p "$OUT_DIR"

# ── Verify zstd exists ──
if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not found in PATH"
    exit 1
fi

echo "============================================================"
echo "  CPU Zstd Baseline Benchmark"
echo "============================================================"
echo "  Chunk size  : ${CHUNK_MB} MB (matching GPU benchmark)"
echo "  Zstd level  : ${ZSTD_LEVEL}"
echo "  Runs        : ${RUNS}"
echo "  Output      : ${OUT_CSV}"
echo "  zstd version: $(zstd --version 2>&1 | head -1)"
echo "============================================================"
echo ""

# ── Dataset configurations ──
# Format: "name subdir ext"
DATASETS=(
    "hurricane_isabel   hurricane_isabel/100x500x500                      .bin.f32"
    "nyx                nyx/SDRBENCH-EXASKY-NYX-512x512x512              .f32"
    "cesm_atm           cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600    .dat"
)

# ── CSV header ──
echo "dataset,phase,zstd_level,chunk_mb,n_files,n_chunks_total,orig_bytes_total,comp_bytes_total,ratio,comp_mbps,decomp_mbps,comp_mbps_std,decomp_mbps_std,n_runs" > "$OUT_CSV"

TMP_CHUNK=$(mktemp)
TMP_COMP=$(mktemp)
TMP_DECOMP=$(mktemp)
trap "rm -f $TMP_CHUNK $TMP_COMP $TMP_DECOMP" EXIT

for ds_line in "${DATASETS[@]}"; do
    read -r NAME SUBDIR EXT <<< "$ds_line"
    FULL_DIR="$DATA_DIR/$SUBDIR"

    if [ ! -d "$FULL_DIR" ]; then
        echo "SKIP $NAME: $FULL_DIR not found"
        echo ""
        continue
    fi

    # Discover field files
    FILES=()
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(find "$FULL_DIR" -name "*${EXT}" -type f | sort)

    N_FILES=${#FILES[@]}
    if [ $N_FILES -eq 0 ]; then
        echo "SKIP $NAME: no *${EXT} files found"
        continue
    fi

    echo "── $NAME ($N_FILES files, ext=$EXT) ──────────────────────"

    # Arrays to accumulate per-run throughput for std dev calculation
    COMP_MBPS_RUNS=()
    DECOMP_MBPS_RUNS=()

    # These are constant across runs
    TOTAL_ORIG=0
    TOTAL_COMP=0
    N_CHUNKS_TOTAL=0
    FIRST_RUN=1

    for ((run=1; run<=RUNS; run++)); do
        RUN_ORIG=0
        RUN_COMP=0
        RUN_N_CHUNKS=0
        RUN_COMP_MS=0
        RUN_DECOMP_MS=0

        for filepath in "${FILES[@]}"; do
            FILE_SIZE=$(stat -c%s "$filepath")
            N_CHUNKS=$(( (FILE_SIZE + CHUNK_BYTES - 1) / CHUNK_BYTES ))

            for ((c=0; c<N_CHUNKS; c++)); do
                OFFSET=$((c * CHUNK_BYTES))
                REMAINING=$((FILE_SIZE - OFFSET))
                THIS_CHUNK=$((REMAINING < CHUNK_BYTES ? REMAINING : CHUNK_BYTES))

                # Extract chunk (use skip_bytes/count_bytes for proper buffered I/O)
                dd if="$filepath" bs=4096 skip=$OFFSET count=$THIS_CHUNK \
                   iflag=skip_bytes,count_bytes of="$TMP_CHUNK" 2>/dev/null

                # Compress and time
                COMP_START=$(date +%s%N)
                zstd -${ZSTD_LEVEL} -f "$TMP_CHUNK" -o "$TMP_COMP" --no-progress 2>/dev/null
                COMP_END=$(date +%s%N)
                COMP_NS=$((COMP_END - COMP_START))

                COMP_SIZE=$(stat -c%s "$TMP_COMP")

                # Decompress and time
                DECOMP_START=$(date +%s%N)
                zstd -d -f "$TMP_COMP" -o "$TMP_DECOMP" --no-progress 2>/dev/null
                DECOMP_END=$(date +%s%N)
                DECOMP_NS=$((DECOMP_END - DECOMP_START))

                RUN_ORIG=$((RUN_ORIG + THIS_CHUNK))
                RUN_COMP=$((RUN_COMP + COMP_SIZE))
                RUN_N_CHUNKS=$((RUN_N_CHUNKS + 1))
                RUN_COMP_MS=$((RUN_COMP_MS + COMP_NS / 1000000))
                RUN_DECOMP_MS=$((RUN_DECOMP_MS + DECOMP_NS / 1000000))
            done
        done

        # Compute throughput for this run (MiB/s)
        if [ $RUN_COMP_MS -gt 0 ]; then
            COMP_MBPS=$(echo "scale=1; $RUN_ORIG / 1048576 / ($RUN_COMP_MS / 1000)" | bc)
        else
            COMP_MBPS="0.0"
        fi
        if [ $RUN_DECOMP_MS -gt 0 ]; then
            DECOMP_MBPS=$(echo "scale=1; $RUN_ORIG / 1048576 / ($RUN_DECOMP_MS / 1000)" | bc)
        else
            DECOMP_MBPS="0.0"
        fi

        COMP_MBPS_RUNS+=("$COMP_MBPS")
        DECOMP_MBPS_RUNS+=("$DECOMP_MBPS")

        if [ $FIRST_RUN -eq 1 ]; then
            TOTAL_ORIG=$RUN_ORIG
            TOTAL_COMP=$RUN_COMP
            N_CHUNKS_TOTAL=$RUN_N_CHUNKS
            FIRST_RUN=0
        fi

        printf "  Run %d/%d: ratio=%.2fx  comp=%s MiB/s  decomp=%s MiB/s\n" \
            "$run" "$RUNS" \
            "$(echo "scale=2; $RUN_ORIG / $RUN_COMP" | bc)" \
            "$COMP_MBPS" "$DECOMP_MBPS"
    done

    # Compute mean and std dev
    COMP_MEAN=$(printf '%s\n' "${COMP_MBPS_RUNS[@]}" | awk '{s+=$1} END {printf "%.1f", s/NR}')
    DECOMP_MEAN=$(printf '%s\n' "${DECOMP_MBPS_RUNS[@]}" | awk '{s+=$1} END {printf "%.1f", s/NR}')

    if [ $RUNS -gt 1 ]; then
        COMP_STD=$(printf '%s\n' "${COMP_MBPS_RUNS[@]}" | awk -v m="$COMP_MEAN" '{d=$1-m; s+=d*d} END {printf "%.1f", sqrt(s/NR)}')
        DECOMP_STD=$(printf '%s\n' "${DECOMP_MBPS_RUNS[@]}" | awk -v m="$DECOMP_MEAN" '{d=$1-m; s+=d*d} END {printf "%.1f", sqrt(s/NR)}')
    else
        COMP_STD="0.0"
        DECOMP_STD="0.0"
    fi

    RATIO=$(echo "scale=4; $TOTAL_ORIG / $TOTAL_COMP" | bc)

    echo "  ──────────────────────────────────────────"
    printf "  Mean: ratio=%sx  comp=%s +/- %s MiB/s  decomp=%s +/- %s MiB/s\n" \
        "$RATIO" "$COMP_MEAN" "$COMP_STD" "$DECOMP_MEAN" "$DECOMP_STD"
    printf "  Chunks: %d  Files: %d  Orig: %d bytes  Comp: %d bytes\n" \
        "$N_CHUNKS_TOTAL" "$N_FILES" "$TOTAL_ORIG" "$TOTAL_COMP"
    echo ""

    # Write CSV row
    echo "$NAME,cpu-zstd,$ZSTD_LEVEL,$CHUNK_MB,$N_FILES,$N_CHUNKS_TOTAL,$TOTAL_ORIG,$TOTAL_COMP,$RATIO,$COMP_MEAN,$DECOMP_MEAN,$COMP_STD,$DECOMP_STD,$RUNS" >> "$OUT_CSV"
done

echo "============================================================"
echo "  CPU Zstd baseline complete. Results: $OUT_CSV"
echo "============================================================"
cat "$OUT_CSV"
