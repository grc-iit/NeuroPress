#!/bin/bash
# test_m7_acquire_falloff.sh
#
# M7: acquireCompContext() must have a return statement after its for-loop
#     to avoid falling off the end of a non-void function (UB).
#
# Verifies the source has `return nullptr;` after the for-loop.

PASS=0
FAIL=0
FILE="src/api/gpucompress_api.cpp"

echo "=== M7: acquireCompContext falls off end of non-void function ==="
echo

# Extract the function body and check for return nullptr after the loop
# The function should contain "return nullptr;" after the for-loop closing brace
FUNC_BODY=$(sed -n '/^CompContext\* acquireCompContext/,/^}/p' "$FILE")

if echo "$FUNC_BODY" | grep -q 'return nullptr;'; then
    echo "  PASS: acquireCompContext has 'return nullptr' fallback"
    PASS=$((PASS+1))
else
    echo "  FAIL: acquireCompContext missing 'return nullptr' fallback"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
