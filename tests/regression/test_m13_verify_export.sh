#!/bin/bash
# test_m13_verify_export.sh
#
# M13: verify_export passes unnormalized input to model but compares to
#      manual path that normalizes, causing spurious assertion failures.
#
# Verifies the source normalizes test_input before model call.

PASS=0
FAIL=0
FILE="neural_net/export/export_weights.py"

echo "=== M13: verify_export compares incompatible values ==="
echo

# Check that test_input is normalized before model call
if grep -q 'test_input_norm' "$FILE"; then
    echo "  PASS: test_input is normalized before model call"
    PASS=$((PASS+1))
else
    echo "  FAIL: test_input not normalized before model call"
    FAIL=$((FAIL+1))
fi

# Check that expected_output is denormalized
if grep -q 'expected_output.*ys\|expected_output.*ym' "$FILE"; then
    echo "  PASS: expected_output is denormalized"
    PASS=$((PASS+1))
else
    echo "  FAIL: expected_output is not denormalized"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
