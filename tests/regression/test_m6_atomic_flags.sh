#!/bin/bash
# test_m6_atomic_flags.sh
#
# M6: g_nn_loaded and g_rank_criterion must be std::atomic to avoid
#     data races when read/written from multiple threads.
#
# Verifies the source uses atomic types for these globals.

PASS=0
FAIL=0
FILE="src/nn/nn_gpu.cu"

echo "=== M6: Non-atomic global flags ==="
echo

# Check g_nn_loaded is atomic
if grep -q 'static std::atomic<bool> g_nn_loaded' "$FILE"; then
    echo "  PASS: g_nn_loaded is std::atomic<bool>"
    PASS=$((PASS+1))
else
    echo "  FAIL: g_nn_loaded is NOT std::atomic<bool>"
    FAIL=$((FAIL+1))
fi

# Check g_rank_criterion is atomic
if grep -q 'static std::atomic<int> g_rank_criterion' "$FILE"; then
    echo "  PASS: g_rank_criterion is std::atomic<int>"
    PASS=$((PASS+1))
else
    echo "  FAIL: g_rank_criterion is NOT std::atomic<int>"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
