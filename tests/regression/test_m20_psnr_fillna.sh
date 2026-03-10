#!/bin/bash
# test_m20_psnr_fillna.sh
#
# M20: cross_validate.py psnr_clamped computation doesn't handle NaN values
#      in psnr_db column (only handles inf via .replace()).
#
# Verifies .fillna(120.0) is chained after .replace().

PASS=0
FAIL=0
FILE="neural_net/training/cross_validate.py"

echo "=== M20: cross_validate.py missing PSNR fillna ==="
echo

if grep -q 'fillna(120.0)' "$FILE"; then
    echo "  PASS: fillna(120.0) present for psnr_clamped"
    PASS=$((PASS+1))
else
    echo "  FAIL: fillna(120.0) missing for psnr_clamped"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
