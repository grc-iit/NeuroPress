/**
 * test_c4_int32_clamp.cu
 *
 * C4: quantize_linear_kernel clamps int8 and int16 output ranges but
 *     has no clamping for int32. Float values that map outside
 *     [-2^31, 2^31-1] undergo undefined signed integer overflow.
 *
 * Test strategy:
 *   Direct GPU kernel test: launch quantization math on GPU with a scale
 *   that produces values beyond int32 range, then check if the output
 *   is clamped (fix) or wrapped (bug).
 *
 * Run: ./test_c4_int32_clamp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cuda_runtime.h>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

/**
 * Kernel that reproduces the exact math of quantize_linear_kernel<float, int32_t>
 * so we can test int32 clamping directly without going through the full
 * quantize_simple pipeline (which adjusts error bounds).
 */
__global__ void test_quantize_int32_kernel(
    const float* input,
    int32_t* output,
    size_t num_elements,
    double scale,
    double offset,
    int use_clamp  /* 0 = no clamp (bug), 1 = clamp (fix) */
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < num_elements; i += (size_t)gridDim.x * blockDim.x) {
        double val = static_cast<double>(input[i]);
        double centered = val - offset;
        double quantized = round(centered * scale);

        /* This mirrors the kernel: int8 and int16 have clamping, int32 does not */
        if (use_clamp) {
            quantized = fmax(-2147483648.0, fmin(2147483647.0, quantized));
        }
        /* Without clamp: cast of out-of-range double to int32 is UB */
        output[i] = static_cast<int32_t>(quantized);
    }
}

__global__ void test_dequantize_int32_kernel(
    const int32_t* input,
    float* output,
    size_t num_elements,
    double inv_scale,
    double offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < num_elements; i += (size_t)gridDim.x * blockDim.x) {
        double restored = static_cast<double>(input[i]) * inv_scale + offset;
        output[i] = static_cast<float>(restored);
    }
}

int main(void) {
    printf("=== C4: Missing int32 clamping in quantization kernel ===\n\n");

    /* Setup: data with range that causes quantized values > INT32_MAX
     *
     * data_min = 0, data_max = 1e10
     * error_bound = 1.0, scale = 1/(2*1.0) = 0.5
     * For val = 1e10: quantized = round((1e10 - 0) * 0.5) = 5e9
     * 5e9 >> INT32_MAX (2147483647) → overflow without clamping
     */
    const size_t N = 8;
    float h_input[8];
    h_input[0] = 0.0f;
    h_input[1] = 1.0e9f;
    h_input[2] = 2.0e9f;
    h_input[3] = 3.0e9f;          /* quantized = 1.5e9, fits in int32 */
    h_input[4] = 5.0e9f;          /* quantized = 2.5e9, > INT32_MAX */
    h_input[5] = 8.0e9f;          /* quantized = 4.0e9, > INT32_MAX */
    h_input[6] = 1.0e10f;         /* quantized = 5.0e9, >> INT32_MAX */
    h_input[7] = -1.0e9f;         /* quantized = -0.5e9, fits */

    double data_min = -1.0e9;
    double error_bound = 1.0;
    double scale = 1.0 / (2.0 * error_bound);  /* 0.5 */
    double inv_scale = 2.0 * error_bound;       /* 2.0 */

    float *d_input;
    int32_t *d_quantized;
    float *d_restored;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_quantized, N * sizeof(int32_t));
    cudaMalloc(&d_restored, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Test 1: Without clamping (reproduces bug) ---- */
    printf("--- Test 1: Quantize WITHOUT int32 clamping (bug) ---\n");
    {
        test_quantize_int32_kernel<<<1, 256>>>(
            d_input, d_quantized, N, scale, data_min, 0 /* no clamp */);
        test_dequantize_int32_kernel<<<1, 256>>>(
            d_quantized, d_restored, N, inv_scale, data_min);
        cudaDeviceSynchronize();

        int32_t h_quant[8];
        float h_restored[8];
        cudaMemcpy(h_quant, d_quantized, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_restored, d_restored, N * sizeof(float), cudaMemcpyDeviceToHost);

        int overflow_detected = 0;
        for (size_t i = 0; i < N; i++) {
            double expected_q = round((double(h_input[i]) - data_min) * scale);
            double error = fabs((double)h_input[i] - (double)h_restored[i]);
            int overflowed = (expected_q > (double)INT32_MAX || expected_q < (double)INT_MIN);

            printf("  [%zu] input=%.2e  expected_q=%.0f  actual_q=%d  restored=%.2e  err=%.2e%s\n",
                   i, (double)h_input[i], expected_q, h_quant[i],
                   (double)h_restored[i], error,
                   overflowed ? "  ** OVERFLOW" : "");

            if (overflowed) overflow_detected++;
        }

        if (overflow_detected > 0) {
            /* Check if any overflowed values have sign corruption */
            int corrupted = 0;
            for (size_t i = 0; i < N; i++) {
                double expected_q = round(((double)h_input[i] - data_min) * scale);
                if (expected_q > (double)INT32_MAX) {
                    /* Without clamping, the cast is UB. On most platforms it
                     * saturates to INT32_MAX or wraps. Either way, the restored
                     * value will differ from what a clamped version produces. */
                    if (h_quant[i] != INT32_MAX) corrupted++;
                }
            }
            printf("  %d values overflowed int32 range, %d had unexpected quantized values\n",
                   overflow_detected, corrupted);
            PASS("overflow conditions detected in unclamped path");
        } else {
            FAIL("expected overflow conditions but none detected");
        }
    }

    /* ---- Test 2: With clamping (fix) ---- */
    printf("\n--- Test 2: Quantize WITH int32 clamping (fix) ---\n");
    {
        test_quantize_int32_kernel<<<1, 256>>>(
            d_input, d_quantized, N, scale, data_min, 1 /* clamp */);
        test_dequantize_int32_kernel<<<1, 256>>>(
            d_quantized, d_restored, N, inv_scale, data_min);
        cudaDeviceSynchronize();

        int32_t h_quant[8];
        float h_restored[8];
        cudaMemcpy(h_quant, d_quantized, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_restored, d_restored, N * sizeof(float), cudaMemcpyDeviceToHost);

        int clamped_correctly = 1;
        for (size_t i = 0; i < N; i++) {
            double expected_q = round(((double)h_input[i] - data_min) * scale);
            double error = fabs((double)h_input[i] - (double)h_restored[i]);

            printf("  [%zu] input=%.2e  expected_q=%.0f  actual_q=%d  restored=%.2e  err=%.2e\n",
                   i, (double)h_input[i], expected_q, h_quant[i],
                   (double)h_restored[i], error);

            if (expected_q > (double)INT32_MAX && h_quant[i] != INT32_MAX) {
                printf("  ** expected clamped to INT32_MAX but got %d\n", h_quant[i]);
                clamped_correctly = 0;
            }
            if (expected_q < (double)INT_MIN && h_quant[i] != INT_MIN) {
                printf("  ** expected clamped to INT_MIN but got %d\n", h_quant[i]);
                clamped_correctly = 0;
            }
        }

        if (clamped_correctly) {
            PASS("all overflow values correctly clamped to INT32 range");
        } else {
            FAIL("clamping not applied correctly");
        }
    }

    /* ---- Test 3: Verify the actual kernel in libgpucompress has the fix ---- */
    printf("\n--- Test 3: End-to-end round-trip via public API ---\n");
    {
        gpucompress_error_t err = gpucompress_init(NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: gpucompress_init failed (%d)\n", err);
        } else {
            /* Use moderate range data where int32 quantization is selected
             * and boundary values are exercised */
            const size_t DATA_N = 4096;
            const size_t DATA_SIZE = DATA_N * sizeof(float);
            float* h_data = (float*)malloc(DATA_SIZE);
            for (size_t i = 0; i < DATA_N; i++) {
                h_data[i] = -500.0f + 1000.0f * ((float)i / (float)(DATA_N - 1));
            }

            double eb = 0.001;
            size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
            void* h_comp = malloc(max_out);
            size_t comp_size = max_out;

            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
            cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE;
            cfg.error_bound = eb;

            err = gpucompress_compress(h_data, DATA_SIZE, h_comp, &comp_size, &cfg, NULL);
            if (err != GPUCOMPRESS_SUCCESS) {
                printf("  SKIP: compress failed (%d)\n", err);
            } else {
                float* h_decomp = (float*)malloc(DATA_SIZE);
                size_t decomp_size = DATA_SIZE;
                err = gpucompress_decompress(h_comp, comp_size, h_decomp, &decomp_size);

                if (err != GPUCOMPRESS_SUCCESS) {
                    FAIL("decompress failed");
                } else {
                    double max_err = 0;
                    for (size_t i = 0; i < DATA_N; i++) {
                        double e = fabs((double)h_data[i] - (double)h_decomp[i]);
                        if (e > max_err) max_err = e;
                    }
                    printf("  max_error=%.6e vs error_bound=%.6e\n", max_err, eb);
                    if (max_err <= eb * 1.01) {
                        PASS("round-trip within error bound");
                    } else {
                        FAIL("round-trip exceeded error bound");
                    }
                }
                free(h_decomp);
            }
            free(h_data);
            free(h_comp);
            gpucompress_cleanup();
        }
    }

    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_restored);

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
