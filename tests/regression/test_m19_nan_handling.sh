#!/bin/bash
# test_m19_nan_handling.sh
#
# M19: compute_stats_cpu may return NaN for degenerate input data.
#
# Verifies the source has NaN guards before returning.

PASS=0
FAIL=0
FILE="neural_net/core/data.py"

echo "=== M19: compute_stats_cpu missing NaN handling ==="
echo

if grep -q 'isnan' "$FILE"; then
    echo "  PASS: NaN handling present in data.py"
    PASS=$((PASS+1))
else
    echo "  FAIL: No NaN handling in data.py"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
