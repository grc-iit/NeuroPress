#!/bin/bash
# test_m12_sgd_psnr_gradient.sh
#
# M12: SGD skips PSNR gradient for lossless results (actual_psnr=0).
#      The if-else sets s_d3[3]=0 when psnr<=0, so the NN never learns
#      to predict PSNR for lossless configs.
#
# Verifies the source treats psnr<=0 as 120.0 (perfect reconstruction)
# instead of zeroing the gradient.

PASS=0
FAIL=0
FILE="src/nn/nn_gpu.cu"

echo "=== M12: SGD skips PSNR gradient for lossless results ==="
echo

# The bug pattern: "if (sample.actual_psnr > 0.0f)" with an else branch that zeros gradient.
# After fix: psnr<=0 should be treated as 120.0, no else { s_d3[3] = 0.0f; } branch.
if grep -q 'actual_psnr > 0\.0f' "$FILE"; then
    echo "  FAIL: Still conditionally skipping PSNR gradient (actual_psnr > 0.0f check)"
    FAIL=$((FAIL+1))
else
    echo "  PASS: No conditional skip of PSNR gradient"
    PASS=$((PASS+1))
fi

# Check that 120.0 fallback for psnr<=0 is present
if grep -q 'actual_psnr <= 0\.0f.*120' "$FILE" || grep -q 'psnr_val.*120' "$FILE"; then
    echo "  PASS: 120.0 fallback for lossless PSNR present"
    PASS=$((PASS+1))
else
    echo "  FAIL: No 120.0 fallback for lossless PSNR"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
