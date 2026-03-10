#!/bin/bash
# test_m9_primary_psnr.sh
#
# M9: Primary exploration sample records psnr=0.0 instead of the actual PSNR.
#     For lossless: should be 120.0 (perfect reconstruction).
#     For lossy: should use quant_result.psnr.
#
# Verifies the source does NOT use hardcoded 0.0 for the PSNR field in the
# primary ExploredResult push_back.

PASS=0
FAIL=0
FILE="src/api/gpucompress_api.cpp"

echo "=== M9: Primary exploration sample records psnr=0.0 ==="
echo

# The bug pattern is: explored_samples.push_back({..., 0.0, 0.0});
# where the last 0.0 is the PSNR. After fix, the last arg should NOT be 0.0.
# The PSNR is the last value before }); on the line after push_back.
# Match lines that end with "0.0, 0.0});" — decomp_time=0.0 is OK but psnr=0.0 is the bug.
COUNT=$(grep -c '0\.0, 0\.0});' "$FILE")

if [ "$COUNT" -eq 0 ]; then
    echo "  PASS: No primary sample push_back with hardcoded psnr=0.0"
    PASS=$((PASS+1))
else
    echo "  FAIL: Found $COUNT primary sample push_back with hardcoded psnr=0.0"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
