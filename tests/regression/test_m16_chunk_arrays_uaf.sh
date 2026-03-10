#!/bin/bash
# test_m16_chunk_arrays_uaf.sh
#
# M16: Use-after-free in DeviceChunkArrays. The DeviceChunkArrays destructor
#      frees GPU pointer arrays while the shuffle kernel is still reading them.
#
# Verifies cudaStreamSynchronize is called before returning in both
# byte_shuffle_simple and byte_unshuffle_simple.

PASS=0
FAIL=0
FILE="src/preprocessing/byte_shuffle_kernels.cu"

echo "=== M16: Use-after-free in DeviceChunkArrays ==="
echo

# Extract byte_shuffle_simple function body and check for cudaStreamSynchronize
SHUFFLE_BODY=$(sed -n '/^uint8_t\* byte_shuffle_simple/,/^}/p' "$FILE")
if echo "$SHUFFLE_BODY" | grep -q 'cudaStreamSynchronize'; then
    echo "  PASS: byte_shuffle_simple has cudaStreamSynchronize"
    PASS=$((PASS+1))
else
    echo "  FAIL: byte_shuffle_simple missing cudaStreamSynchronize"
    FAIL=$((FAIL+1))
fi

# Extract byte_unshuffle_simple function body
UNSHUFFLE_BODY=$(sed -n '/^uint8_t\* byte_unshuffle_simple/,/^}/p' "$FILE")
if echo "$UNSHUFFLE_BODY" | grep -q 'cudaStreamSynchronize'; then
    echo "  PASS: byte_unshuffle_simple has cudaStreamSynchronize"
    PASS=$((PASS+1))
else
    echo "  FAIL: byte_unshuffle_simple missing cudaStreamSynchronize"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
