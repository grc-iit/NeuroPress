#!/bin/bash
# test_m17_gather_oob.sh
#
# M17: gather_chunk_kernel reads L*L*chunk_z elements even for the last
#      chunk which may have fewer than chunk_z z-slices, causing OOB reads.
#
# Verifies gather_chunk and scatter_chunk accept actual_cz parameter.

PASS=0
FAIL=0
FILE="benchmarks/grayscott/grayscott-benchmark.cu"

echo "=== M17: gather_chunk_kernel OOB read on last chunk ==="
echo

# Check that gather_chunk accepts actual_cz parameter
if grep -q 'gather_chunk.*actual_cz' "$FILE"; then
    echo "  PASS: gather_chunk accepts actual_cz parameter"
    PASS=$((PASS+1))
else
    echo "  FAIL: gather_chunk does not accept actual_cz parameter"
    FAIL=$((FAIL+1))
fi

# Check that scatter_chunk accepts actual_cz parameter
if grep -q 'scatter_chunk.*actual_cz' "$FILE"; then
    echo "  PASS: scatter_chunk accepts actual_cz parameter"
    PASS=$((PASS+1))
else
    echo "  FAIL: scatter_chunk does not accept actual_cz parameter"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
