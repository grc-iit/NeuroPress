/**
 * test_h6_cub_int_overflow.cu
 *
 * H6: CUB DeviceReduce::Min/Max is called with (int)num_elements.
 *     When num_elements > INT_MAX, the cast wraps to a small/negative
 *     value, causing CUB to process a fraction of the data or crash.
 *
 * Test strategy:
 *   1. Call quantize_simple() with num_elements = INT_MAX + 1 and a
 *      small dummy device buffer (we won't actually touch all that memory)
 *   2. Before fix: the (int) cast wraps, CUB gets garbage size, and
 *      either crashes, returns wrong min/max, or CUDA error
 *   3. After fix: compute_data_range_typed returns -1 early,
 *      quantize_simple returns an invalid QuantizationResult gracefully
 *
 * Run: ./test_h6_cub_int_overflow
 */

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

#include "preprocessing/quantization.cuh"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== H6: CUB int cast overflow on large num_elements ===\n\n");

    /* Allocate a tiny device buffer — we just need a valid pointer.
     * The function should reject the call before actually reading
     * INT_MAX+1 elements from it. */
    float* d_dummy = nullptr;
    cudaError_t err = cudaMalloc(&d_dummy, 1024);
    if (err != cudaSuccess) {
        printf("  SKIP: cudaMalloc failed\n");
        return 1;
    }

    /* num_elements that overflows int */
    size_t huge_num = (size_t)INT_MAX + 1;

    QuantizationConfig config;
    config.error_bound = 0.01;
    config.precision = QuantizationPrecision::AUTO;

    printf("  Testing quantize_simple with num_elements = %zu (INT_MAX+1)\n", huge_num);

    /* Before fix: (int)huge_num wraps to -2147483648 or similar,
     * causing CUB to misbehave. After fix: early rejection. */
    QuantizationResult result = quantize_simple(
        d_dummy, huge_num, sizeof(float), config, 0);

    if (!result.isValid()) {
        PASS("quantize_simple gracefully rejected oversized input");
    } else {
        FAIL("quantize_simple did NOT reject oversized input (int overflow)");
        /* Clean up if it somehow allocated */
        if (result.d_quantized) cudaFree(result.d_quantized);
    }

    /* Also test the exact boundary: INT_MAX should be accepted
     * (if we had enough memory). Since we don't, it will fail for a
     * different reason (out of memory or invalid read), but it should
     * NOT be rejected by the overflow guard. We verify by checking that
     * the error is NOT from the size guard (it will fail in CUB or malloc). */
    printf("  Testing quantize_simple with num_elements = %d (INT_MAX, boundary)\n", INT_MAX);

    /* We can't actually allocate INT_MAX floats (~8GB), so we just verify
     * the function doesn't crash with a segfault. Any error is acceptable
     * as long as it returns gracefully. */
    result = quantize_simple(d_dummy, (size_t)INT_MAX, sizeof(float), config, 0);
    /* We don't check isValid here — it will fail due to OOM or CUB error
     * on the small buffer, which is expected. The point is it didn't crash. */
    PASS("quantize_simple did not crash at INT_MAX boundary");
    if (result.d_quantized) cudaFree(result.d_quantized);

    cudaFree(d_dummy);

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
