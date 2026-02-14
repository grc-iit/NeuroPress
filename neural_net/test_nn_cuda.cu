/**
 * Quick integration test: load NN weights and run inference on synthetic data.
 * Verifies the full pipeline: weight loading → stats computation → NN inference.
 *
 * Build:
 *   nvcc -o test_nn_cuda test_nn_cuda.cu -L../build -lgpucompress -I../include -Wl,-rpath,../build
 * Run:
 *   ./test_nn_cuda ../neural_net/weights/model.nnwt
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

int main(int argc, char** argv) {
    const char* nn_path = (argc > 1) ? argv[1] : "weights/model.nnwt";

    printf("=== Neural Network CUDA Integration Test ===\n\n");

    // Initialize library with NN weights
    printf("1. Initializing with NN weights: %s\n", nn_path);
    gpucompress_error_t err = gpucompress_init(nn_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "   FAILED: gpucompress_init returned %d: %s\n",
                err, gpucompress_error_string(err));
        return 1;
    }
    printf("   OK: Library initialized\n");

    // Check NN is loaded
    int nn_loaded = gpucompress_nn_is_loaded();
    printf("   NN loaded: %s\n", nn_loaded ? "YES" : "NO");
    if (!nn_loaded) {
        fprintf(stderr, "   FAILED: NN not loaded\n");
        gpucompress_cleanup();
        return 1;
    }

    // Generate synthetic data: a smooth sine wave (should favor algorithms good with smooth data)
    const size_t num_elements = 1024 * 1024;  // 1M floats = 4MB
    const size_t data_size = num_elements * sizeof(float);
    float* h_data = (float*)malloc(data_size);

    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = sinf(static_cast<float>(i) * 0.01f) * 100.0f;
    }

    printf("\n2. Generated synthetic data: %zu floats (%.1f MB)\n",
           num_elements, data_size / (1024.0 * 1024.0));
    printf("   Pattern: sin(i * 0.01) * 100\n");

    // Compress with ALGO_AUTO (should use NN)
    size_t max_output = gpucompress_max_compressed_size(data_size);
    void* output = malloc(max_output);
    size_t output_size = max_output;

    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm = GPUCOMPRESS_ALGO_AUTO;
    config.error_bound = 0.0;  // Lossless

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    printf("\n3. Compressing with ALGO_AUTO (NN-powered)...\n");
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "   FAILED: compress returned %d: %s\n",
                err, gpucompress_error_string(err));
        free(h_data);
        free(output);
        gpucompress_cleanup();
        return 1;
    }

    printf("   OK: Compression succeeded\n");
    printf("   Algorithm selected: %s\n", gpucompress_algorithm_name(stats.algorithm_used));
    printf("   Preprocessing: 0x%02x\n", stats.preprocessing_used);
    printf("   Compression ratio: %.2fx\n", stats.compression_ratio);
    printf("   Entropy: %.4f bits\n", stats.entropy_bits);
    printf("   MAD: %.6f\n", stats.mad);
    printf("   1st Derivative: %.6f\n", stats.first_derivative);
    printf("   Original: %zu bytes → Compressed: %zu bytes\n",
           stats.original_size, stats.compressed_size);

    // Decompress and verify
    printf("\n4. Decompressing and verifying...\n");
    float* h_decompressed = (float*)malloc(data_size);
    size_t decomp_size = data_size;
    err = gpucompress_decompress(output, output_size, h_decompressed, &decomp_size);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "   FAILED: decompress returned %d: %s\n",
                err, gpucompress_error_string(err));
        free(h_data);
        free(output);
        free(h_decompressed);
        gpucompress_cleanup();
        return 1;
    }

    // Verify data integrity
    int mismatches = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        double diff = fabs(static_cast<double>(h_data[i]) - static_cast<double>(h_decompressed[i]));
        if (diff > max_err) max_err = diff;
        if (diff > 1e-6) mismatches++;
    }

    if (mismatches == 0) {
        printf("   OK: Lossless verification PASSED (max error: %.2e)\n", max_err);
    } else {
        printf("   WARNING: %d mismatches (max error: %.2e)\n", mismatches, max_err);
    }

    // Also test with lossy config
    printf("\n5. Testing lossy compression (error_bound=0.01)...\n");
    config.error_bound = 0.01;
    output_size = max_output;
    memset(&stats, 0, sizeof(stats));
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    if (err == GPUCOMPRESS_SUCCESS) {
        printf("   OK: Algorithm: %s, Ratio: %.2fx, Preprocessing: 0x%02x\n",
               gpucompress_algorithm_name(stats.algorithm_used),
               stats.compression_ratio,
               stats.preprocessing_used);
    } else {
        printf("   FAILED: %s\n", gpucompress_error_string(err));
    }

    // Cleanup
    free(h_data);
    free(output);
    free(h_decompressed);
    gpucompress_cleanup();

    printf("\n=== All tests passed ===\n");
    return 0;
}
