/**
 * @file test_quantization_roundtrip.c
 * @brief Visual verification of quantization: prints original vs restored values.
 *
 * Compresses float data with quantization enabled, decompresses, and prints
 * a side-by-side comparison showing original, restored, and error for each value.
 *
 * Usage:
 *   ./build/test_quantization_roundtrip [error_bound]
 *   Default error_bound = 0.01
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gpucompress.h"

/* Small dataset so we can print every value */
#define N_FLOATS 64

static void print_header(const char* test_name, double error_bound) {
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  %s  (error_bound = %.4f)\n", test_name, error_bound);
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  %4s │ %14s │ %14s │ %12s │ %s\n",
           "Idx", "Original", "Restored", "Error", "OK?");
    printf("  ─────┼────────────────┼────────────────┼──────────────┼─────\n");
}

static int run_test(const char* name, const float* data, size_t n,
                    gpucompress_algorithm_t algo, unsigned int preproc,
                    double error_bound) {
    size_t input_bytes = n * sizeof(float);
    size_t max_out = gpucompress_max_compressed_size(input_bytes);
    void* compressed = malloc(max_out);
    float* restored = (float*)malloc(input_bytes);
    if (!compressed || !restored) { free(compressed); free(restored); return -1; }

    /* Compress */
    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = algo;
    cfg.preprocessing = preproc;
    cfg.error_bound = error_bound;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    size_t comp_size = max_out;
    gpucompress_error_t err = gpucompress_compress(data, input_bytes,
                                                    compressed, &comp_size,
                                                    &cfg, &stats);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  COMPRESS FAILED: %s\n", gpucompress_error_string(err));
        free(compressed); free(restored);
        return -1;
    }

    /* Decompress */
    size_t decomp_size = input_bytes;
    err = gpucompress_decompress(compressed, comp_size, restored, &decomp_size);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  DECOMPRESS FAILED: %s\n", gpucompress_error_string(err));
        free(compressed); free(restored);
        return -1;
    }

    /* Print values */
    print_header(name, error_bound);

    int violations = 0;
    double max_err = 0.0;
    double sum_err = 0.0;

    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)data[i] - (double)restored[i]);
        int ok = (e <= error_bound * 1.01);  /* 1% float tolerance */
        if (!ok) violations++;
        if (e > max_err) max_err = e;
        sum_err += e;

        printf("  %4zu │ %14.6f │ %14.6f │ %12.6e │ %s\n",
               i, data[i], restored[i], e, ok ? " ✓" : " ✗");
    }

    printf("  ─────┴────────────────┴────────────────┴──────────────┴─────\n");
    printf("  Ratio: %.2f:1 (%zu → %zu bytes)  Algo: %s\n",
           stats.compression_ratio, input_bytes, comp_size,
           gpucompress_algorithm_name(stats.algorithm_used));
    printf("  Max error: %.6e  Mean error: %.6e\n", max_err, sum_err / n);
    printf("  Violations: %d / %zu\n", violations, n);
    printf("  Result: %s\n", violations == 0 ? "PASSED" : "FAILED");

    free(compressed);
    free(restored);
    return violations == 0 ? 0 : -1;
}

int main(int argc, char** argv) {
    double eb = 0.01;
    if (argc > 1) eb = atof(argv[1]);

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("FATAL: gpucompress_init failed: %s\n", gpucompress_error_string(err));
        return 1;
    }

    printf("=== Quantization Round-Trip Visual Test ===\n");
    printf("Error bound: %.6f\n", eb);

    int failures = 0;

    /* ── Pattern 1: Ramp 0..1 ── */
    {
        float data[N_FLOATS];
        for (int i = 0; i < N_FLOATS; i++)
            data[i] = (float)i / (float)(N_FLOATS - 1);
        failures += (run_test("Ramp [0,1] — LZ4+Quantize",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_LZ4,
                              GPUCOMPRESS_PREPROC_QUANTIZE, eb) != 0);
    }

    /* ── Pattern 2: Sine wave ── */
    {
        float data[N_FLOATS];
        for (int i = 0; i < N_FLOATS; i++)
            data[i] = sinf(2.0f * (float)M_PI * (float)i / (float)N_FLOATS);
        failures += (run_test("Sine — ZSTD+Quantize",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_ZSTD,
                              GPUCOMPRESS_PREPROC_QUANTIZE, eb) != 0);
    }

    /* ── Pattern 3: Large values (tests float precision limits) ── */
    {
        float data[N_FLOATS];
        for (int i = 0; i < N_FLOATS; i++)
            data[i] = 1000.0f + (float)i * 0.1f;
        failures += (run_test("Large [1000,1006] — LZ4+Quantize+Shuffle",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_LZ4,
                              GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4,
                              eb) != 0);
    }

    /* ── Pattern 4: Sparse (mostly zeros) ── */
    {
        float data[N_FLOATS];
        memset(data, 0, sizeof(data));
        data[0] = 1.0f; data[16] = -0.5f; data[32] = 2.5f; data[63] = -1.0f;
        failures += (run_test("Sparse — DEFLATE+Quantize",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_DEFLATE,
                              GPUCOMPRESS_PREPROC_QUANTIZE, eb) != 0);
    }

    /* ── Pattern 5: Constant ── */
    {
        float data[N_FLOATS];
        for (int i = 0; i < N_FLOATS; i++)
            data[i] = 3.14159f;
        failures += (run_test("Constant (pi) — ZSTD+Quantize+Shuffle",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_ZSTD,
                              GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4,
                              eb) != 0);
    }

    /* ── Pattern 6: Negative range ── */
    {
        float data[N_FLOATS];
        for (int i = 0; i < N_FLOATS; i++)
            data[i] = -10.0f + (float)i * 0.3f;
        failures += (run_test("Negative range [-10,9] — ANS+Quantize",
                              data, N_FLOATS, GPUCOMPRESS_ALGO_ANS,
                              GPUCOMPRESS_PREPROC_QUANTIZE, eb) != 0);
    }

    printf("\n═══════════════════════════════════\n");
    printf("  Total: 6 tests, %d failed\n", failures);
    printf("═══════════════════════════════════\n");

    gpucompress_cleanup();
    return failures > 0 ? 1 : 0;
}
