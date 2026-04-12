#!/bin/bash
# Generate synthetic training data: 2000 chunks x 64 configs = 128K rows.
#
# Usage:
#   bash scripts/gen_synthetic.sh
#   bash scripts/gen_synthetic.sh --output results/my_data.csv --seed 123
#
# Options (all optional):
#   --output <path>    Output CSV  (default: results/synthetic_results.csv)
#   --chunks <N>       Num chunks  (default: 2000 → 128K rows)
#   --seed   <N>       RNG seed    (default: 42)
#   --device <N>       CUDA device (default: 0)
#   --build  <dir>     Build dir   (default: build)

set -euo pipefail

OUTPUT="results/synthetic_results.csv"
CHUNKS=2000
SEED=42
DEVICE=0
BUILD_DIR="build"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --build)  BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BINARY="${BUILD_DIR}/synthetic_benchmark"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY not found. Build the project first."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Generating ${CHUNKS} chunks x 64 configs = $((CHUNKS * 64)) rows"
echo "  Output: $OUTPUT"
echo "  Seed:   $SEED"
echo "  Device: $DEVICE"

"$BINARY" \
    --num-chunks "$CHUNKS" \
    --output     "$OUTPUT" \
    --seed       "$SEED"   \
    --device     "$DEVICE"

echo "Done: $OUTPUT"
