/**
 * @file test_nn_shuffle.cu
 * @brief Test: Does the NN choose shuffle? And can it restore data perfectly?
 *
 * Uses the C API with ALGO_AUTO + NN weights on 32MB float32 data.
 * Reports what the NN chose (algorithm + preprocessing) and verifies round-trip.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

int main() {
    printf("=== NN Shuffle Selection Test (32 MB float32) ===\n\n");

    /* ---- Load 32 MB test file ---- */
    const char* input_path = "/tmp/test_32mb.bin";
    FILE* f = fopen(input_path, "rb");
    if (!f) { printf("Cannot open %s\n", input_path); return 1; }

    fseek(f, 0, SEEK_END);
    size_t input_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("Input: %s (%zu bytes, %.2f MB)\n", input_path, input_size,
           input_size / (1024.0 * 1024.0));

    void* input = malloc(input_size);
    fread(input, 1, input_size, f);
    fclose(f);

    /* ---- Init library with NN weights ---- */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) {
        FILE* f = fopen("neural_net/weights/model.nnwt", "rb");
        if (f) { fclose(f); weights = "neural_net/weights/model.nnwt"; }
        else weights = "../neural_net/weights/model.nnwt";
    }
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("gpucompress_init failed: %s\n", gpucompress_error_string(err));
        free(input);
        return 1;
    }
    printf("NN loaded: %s\n\n", gpucompress_nn_is_loaded() ? "YES" : "NO");

    /* ---- Compress with ALGO_AUTO (NN decides everything) ---- */
    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm = GPUCOMPRESS_ALGO_AUTO;
    /* Do NOT set preprocessing — let NN decide */

    size_t max_out = gpucompress_max_compressed_size(input_size);
    void* compressed = malloc(max_out);
    size_t compressed_size = max_out;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    err = gpucompress_compress(input, input_size, compressed, &compressed_size,
                               &config, &stats);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("Compression failed: %s\n", gpucompress_error_string(err));
        free(input); free(compressed);
        gpucompress_cleanup();
        return 1;
    }

    /* ---- Report what the NN chose ---- */
    printf("========================================\n");
    printf("  NN Selection Results\n");
    printf("========================================\n");
    printf("Algorithm chosen:  %s\n",
           gpucompress_algorithm_name(stats.algorithm_used));

    unsigned int preproc = stats.preprocessing_used;
    int chose_shuffle = (preproc & GPUCOMPRESS_PREPROC_SHUFFLE_4) != 0;
    int chose_quant   = (preproc & GPUCOMPRESS_PREPROC_QUANTIZE) != 0;
    printf("Shuffle chosen:    %s\n", chose_shuffle ? "YES (4-byte)" : "NO");
    printf("Quantization:      %s\n", chose_quant   ? "YES" : "NO");
    printf("Preprocessing:     0x%02X\n", preproc);
    printf("\n");
    printf("Data stats:\n");
    printf("  Entropy:         %.4f bits\n", stats.entropy_bits);
    printf("  MAD:             %.6f\n", stats.mad);
    printf("  2nd derivative:  %.6f\n", stats.second_derivative);
    printf("\n");
    printf("Compression:\n");
    printf("  Original:        %zu bytes (%.2f MB)\n", stats.original_size,
           stats.original_size / (1024.0 * 1024.0));
    printf("  Compressed:      %zu bytes (%.2f MB)\n", stats.compressed_size,
           stats.compressed_size / (1024.0 * 1024.0));
    printf("  Ratio:           %.2fx\n", stats.compression_ratio);
    printf("  NN predicted:    %.2fx\n", stats.predicted_ratio);
    printf("  Prediction err:  %.1f%%\n",
           fabs(stats.predicted_ratio - stats.compression_ratio)
               / stats.compression_ratio * 100.0);
    printf("========================================\n\n");

    /* ---- Decompress and verify ---- */
    size_t restored_size = input_size;
    void* restored = malloc(restored_size);

    err = gpucompress_decompress(compressed, compressed_size,
                                  restored, &restored_size);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("Decompression failed: %s\n", gpucompress_error_string(err));
        free(input); free(compressed); free(restored);
        gpucompress_cleanup();
        return 1;
    }

    printf("Decompressed: %zu bytes\n", restored_size);

    /* Byte-for-byte verification */
    if (restored_size != input_size) {
        printf("FAIL: size mismatch (expected %zu, got %zu)\n",
               input_size, restored_size);
        free(input); free(compressed); free(restored);
        gpucompress_cleanup();
        return 1;
    }

    int mismatch = memcmp(input, restored, input_size);
    if (mismatch != 0) {
        /* Find first differing byte */
        const uint8_t* a = (const uint8_t*)input;
        const uint8_t* b = (const uint8_t*)restored;
        for (size_t i = 0; i < input_size; i++) {
            if (a[i] != b[i]) {
                printf("FAIL: byte mismatch at offset %zu: "
                       "expected 0x%02X got 0x%02X\n", i, a[i], b[i]);
                break;
            }
        }
        free(input); free(compressed); free(restored);
        gpucompress_cleanup();
        return 1;
    }

    printf("\nVERIFICATION: PASS — all %zu bytes match perfectly!\n", input_size);
    printf("  NN chose %s%s%s → compress → decompress%s%s → exact match\n",
           gpucompress_algorithm_name(stats.algorithm_used),
           chose_shuffle ? " + shuffle" : "",
           chose_quant   ? " + quant"   : "",
           chose_shuffle ? " + unshuffle" : "",
           chose_quant   ? " + dequant"   : "");

    free(input);
    free(compressed);
    free(restored);
    gpucompress_cleanup();

    printf("\nDone.\n");
    return 0;
}
