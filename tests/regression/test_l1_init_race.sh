#!/bin/bash
# test_l1_init_race.sh
#
# L1: ensure_initialized() has a TOCTOU data race — two threads can
#     both see g_gpucompress_initialized==0 and both call gpucompress_init().
#
# Verifies the source uses pthread_once for thread-safe initialization.

PASS=0
FAIL=0
FILE="src/hdf5/H5Zgpucompress.c"

echo "=== L1: Data race in ensure_initialized() ==="
echo

if grep -q 'pthread_once' "$FILE"; then
    echo "  PASS: pthread_once used for initialization"
    PASS=$((PASS+1))
else
    echo "  FAIL: pthread_once not used"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
