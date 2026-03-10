#!/bin/bash
# test_m10_stream_sync.sh
#
# M10: compress_gpu ignores caller's stream_arg, using only ctx->stream.
#      This breaks CUDA stream ordering for callers who expect work to be
#      enqueued on their stream.
#
# Verifies the source no longer has "(void)stream_arg" and instead uses
# bidirectional event sync between caller_stream and ctx->stream.

PASS=0
FAIL=0
FILE="src/api/gpucompress_api.cpp"

echo "=== M10: compress_gpu ignores caller's stream_arg ==="
echo

# Check that (void)stream_arg is removed
if grep -q '(void)stream_arg' "$FILE"; then
    echo "  FAIL: (void)stream_arg still present — stream_arg is being ignored"
    FAIL=$((FAIL+1))
else
    echo "  PASS: (void)stream_arg removed"
    PASS=$((PASS+1))
fi

# Check that caller_stream sync is present in compress_gpu
# Look for cudaStreamWaitEvent usage after acquireCompContext
FUNC_BODY=$(sed -n '/gpucompress_compress_gpu/,/^extern "C"/p' "$FILE")

if echo "$FUNC_BODY" | grep -q 'caller_stream'; then
    echo "  PASS: caller_stream variable used in compress_gpu"
    PASS=$((PASS+1))
else
    echo "  FAIL: no caller_stream handling in compress_gpu"
    FAIL=$((FAIL+1))
fi

if echo "$FUNC_BODY" | grep -q 'cudaStreamWaitEvent'; then
    echo "  PASS: cudaStreamWaitEvent used for stream sync"
    PASS=$((PASS+1))
else
    echo "  FAIL: no cudaStreamWaitEvent for stream sync"
    FAIL=$((FAIL+1))
fi

echo
echo "$PASS pass, $FAIL fail"
exit $FAIL
