#!/bin/bash
# test_m15_element_size.sh
#
# M15: launch_byte_shuffle and launch_byte_unshuffle ignore the element_size
#      parameter, always using the <4> specialization.
#
# Verifies that (void)element_size is removed and a switch on element_size exists.

PASS=0
FAIL=0
FILE="src/preprocessing/byte_shuffle_kernels.cu"

echo "=== M15: element_size parameter ignored, always uses <4> ==="
echo

# Check (void)element_size is removed
COUNT=$(grep -c '(void)element_size' "$FILE")
if [ "$COUNT" -eq 0 ]; then
    echo "  PASS: (void)element_size removed"
    PASS=$((PASS+1))
else
    echo "  FAIL: (void)element_size still present ($COUNT occurrences)"
    FAIL=$((FAIL+1))
fi

# Check switch on element_size exists
if grep -q 'switch.*element_size' "$FILE"; then
    echo "  PASS: switch(element_size) dispatch present"
    PASS=$((PASS+1))
else
    echo "  FAIL: no switch(element_size) dispatch"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
