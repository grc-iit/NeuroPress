#!/bin/bash
# test_l2_shuffle_parse.sh
#
# L2: HDF5 filter shuffle size parsing only handles 4, silently ignoring
#     other non-zero values without warning.
#
# Verifies the source warns on unsupported shuffle sizes.

PASS=0
FAIL=0
FILE="src/hdf5/H5Zgpucompress.c"

echo "=== L2: Shuffle size parsing only handles 4 ==="
echo

# Check that unsupported shuffle sizes produce a warning
if grep -q 'shuffle element_size.*not supported\|shuffle.*not supported' "$FILE"; then
    echo "  PASS: Warning emitted for unsupported shuffle sizes"
    PASS=$((PASS+1))
else
    echo "  FAIL: No warning for unsupported shuffle sizes"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
