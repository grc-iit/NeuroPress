/**
 * test_h9_int8_error_bound.cu
 *
 * H9: Forced INT8 quantization silently violates the error bound when the
 *     data range requires more than 127 quantization levels.
 *
 * Test strategy:
 *   1. Create data with range=1000, error_bound=0.1 → needs 5000 levels
 *   2. Call quantize_simple() with forced INT8 precision
 *   3. Capture stderr and check for a warning about INT8 violation
 *   4. Before fix: no warning. After fix: warning emitted.
 *
 * Run: ./test_h9_int8_error_bound
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <cuda_runtime.h>

#include "preprocessing/quantization.cuh"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== H9: Forced INT8 silently violates error bound ===\n\n");

    /* Create data with range ~1000 on GPU */
    const size_t NUM_FLOATS = 1024;
    const size_t DATA_SIZE = NUM_FLOATS * sizeof(float);
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < NUM_FLOATS; i++)
        h_data[i] = (float)i / (float)NUM_FLOATS * 1000.0f;

    float* d_data = nullptr;
    cudaMalloc(&d_data, DATA_SIZE);
    cudaMemcpy(d_data, h_data, DATA_SIZE, cudaMemcpyHostToDevice);

    /* Force INT8 with error_bound=0.1: needs 5000 levels, INT8 max=127 */
    QuantizationConfig config(QuantizationType::LINEAR, 0.1, NUM_FLOATS, sizeof(float));
    config.precision = QuantizationPrecision::INT8;

    printf("  Data range: [0, 1000), error_bound=0.1\n");
    printf("  Levels needed: %.0f, INT8 max: 127\n", 1000.0 / (2.0 * 0.1));

    /* Capture stderr */
    fflush(stderr);
    int stderr_fd = dup(STDERR_FILENO);
    char tmpfile[] = "/tmp/h9_stderr_XXXXXX";
    int tmp_fd = mkstemp(tmpfile);
    dup2(tmp_fd, STDERR_FILENO);

    QuantizationResult result = quantize_simple(
        d_data, NUM_FLOATS, sizeof(float), config, 0);

    fflush(stderr);
    dup2(stderr_fd, STDERR_FILENO);
    close(stderr_fd);

    /* Read captured stderr */
    lseek(tmp_fd, 0, SEEK_SET);
    char stderr_buf[4096] = {0};
    ssize_t n = read(tmp_fd, stderr_buf, sizeof(stderr_buf) - 1);
    close(tmp_fd);
    unlink(tmpfile);
    if (n > 0) stderr_buf[n] = '\0';

    printf("  quantize_simple result valid: %s\n", result.isValid() ? "yes" : "no");
    printf("  Captured stderr (%zd bytes):\n    %s\n", n, n > 0 ? stderr_buf : "(empty)");

    if (strstr(stderr_buf, "INT8") != NULL ||
        strstr(stderr_buf, "int8") != NULL) {
        PASS("INT8 error bound violation warning emitted");
    } else {
        FAIL("no INT8 error bound violation warning emitted");
    }

    if (result.d_quantized) cudaFree(result.d_quantized);
    cudaFree(d_data);
    free(h_data);

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
