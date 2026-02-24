/**
 * @file experiment_128mb.cu
 * @brief Experiment: compress 128MB synthetic data with all algorithms,
 *        with and without neural network selection.
 *
 * Usage: ./experiment_128mb <input.bin> [weights.nnwt]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#include "gpucompress.h"

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

static void print_separator() {
    printf("%-12s %12s %12s %8s %10s %10s\n",
           "Algorithm", "Original", "Compressed", "Ratio", "Comp(ms)", "Decomp(ms)");
    printf("%-12s %12s %12s %8s %10s %10s\n",
           "----------", "----------", "----------", "------", "--------", "--------");
}

static int run_compress_decompress(
    const void* data, size_t data_size,
    gpucompress_algorithm_t algo, const char* algo_name,
    unsigned int preproc, double error_bound)
{
    size_t max_out = gpucompress_max_compressed_size(data_size);
    std::vector<uint8_t> compressed(max_out);
    size_t comp_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = algo;
    cfg.preprocessing = preproc;
    cfg.error_bound = error_bound;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    double t0 = now_ms();
    gpucompress_error_t rc = gpucompress_compress(
        data, data_size, compressed.data(), &comp_size, &cfg, &stats);
    double t1 = now_ms();

    if (rc != GPUCOMPRESS_SUCCESS) {
        printf("%-12s  COMPRESS FAILED: %s\n", algo_name, gpucompress_error_string(rc));
        return -1;
    }

    // Decompress and verify
    std::vector<uint8_t> decompressed(data_size);
    size_t decomp_size = data_size;

    double t2 = now_ms();
    rc = gpucompress_decompress(compressed.data(), comp_size,
                                 decompressed.data(), &decomp_size);
    double t3 = now_ms();

    if (rc != GPUCOMPRESS_SUCCESS) {
        printf("%-12s  DECOMPRESS FAILED: %s\n", algo_name, gpucompress_error_string(rc));
        return -1;
    }

    double ratio = (double)data_size / (double)comp_size;
    double comp_ms = t1 - t0;
    double decomp_ms = t3 - t2;

    // Verify data integrity
    const char* verify = "OK";
    if (error_bound > 0.0) {
        // Lossy: check error bound
        const float* orig = (const float*)data;
        const float* dec = (const float*)decompressed.data();
        size_t n = data_size / sizeof(float);
        double max_err = 0.0;
        for (size_t i = 0; i < n; i++) {
            double e = fabs((double)orig[i] - (double)dec[i]);
            if (e > max_err) max_err = e;
        }
        if (max_err > error_bound * 1.01) verify = "BOUND_EXCEEDED";
    } else {
        if (decomp_size != data_size || memcmp(data, decompressed.data(), data_size) != 0)
            verify = "MISMATCH";
    }

    printf("%-12s %10zuKB %10zuKB %7.2fx %9.1f %9.1f  [%s]",
           algo_name,
           data_size / 1024,
           comp_size / 1024,
           ratio, comp_ms, decomp_ms, verify);

    // Print NN stats if available
    if (stats.predicted_ratio > 0.0) {
        printf("  nn_pred=%.2fx actual=%.2fx", stats.predicted_ratio, ratio);
        if (stats.sgd_fired) printf(" SGD!");
    }
    if (stats.entropy_bits > 0.0) {
        printf("  ent=%.2f", stats.entropy_bits);
    }
    printf("\n");

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.bin> [weights.nnwt]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* weights_path = (argc >= 3) ? argv[2] : nullptr;

    // Read input file
    FILE* f = fopen(input_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t data_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> data(data_size);
    size_t nread = fread(data.data(), 1, data_size, f);
    fclose(f);

    if (nread != data_size) {
        fprintf(stderr, "Short read: %zu / %zu\n", nread, data_size);
        return 1;
    }

    printf("=== GPUCompress 128MB Experiment ===\n");
    printf("Input: %s (%.1f MB, %zu bytes)\n\n", input_path,
           data_size / (1024.0 * 1024.0), data_size);

    // ============================================================
    // Part 1: All algorithms without NN (manual selection)
    // ============================================================
    gpucompress_error_t rc = gpucompress_init(nullptr);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "Init failed: %s\n", gpucompress_error_string(rc));
        return 1;
    }

    // Compute stats first
    double entropy = 0, mad = 0, sd = 0;
    gpucompress_compute_stats(data.data(), data_size, &entropy, &mad, &sd);
    printf("Data stats: entropy=%.4f bits  MAD=%.6f  2nd_deriv=%.6f\n\n", entropy, mad, sd);

    printf("--- Part 1: Manual Algorithm Selection (no NN) ---\n");
    print_separator();

    struct { gpucompress_algorithm_t algo; const char* name; } algos[] = {
        {GPUCOMPRESS_ALGO_LZ4,      "lz4"},
        {GPUCOMPRESS_ALGO_SNAPPY,   "snappy"},
        {GPUCOMPRESS_ALGO_DEFLATE,  "deflate"},
        {GPUCOMPRESS_ALGO_GDEFLATE, "gdeflate"},
        {GPUCOMPRESS_ALGO_ZSTD,     "zstd"},
        {GPUCOMPRESS_ALGO_ANS,      "ans"},
        {GPUCOMPRESS_ALGO_CASCADED, "cascaded"},
        {GPUCOMPRESS_ALGO_BITCOMP,  "bitcomp"},
    };
    int n_algos = sizeof(algos) / sizeof(algos[0]);

    for (int i = 0; i < n_algos; i++) {
        run_compress_decompress(data.data(), data_size,
                                algos[i].algo, algos[i].name,
                                GPUCOMPRESS_PREPROC_NONE, 0.0);
    }

    // Part 1b: With byte shuffle (float32 = 4 byte)
    printf("\n--- Part 1b: With 4-byte Shuffle ---\n");
    print_separator();
    for (int i = 0; i < n_algos; i++) {
        char name[32];
        snprintf(name, sizeof(name), "%s+shuf", algos[i].name);
        run_compress_decompress(data.data(), data_size,
                                algos[i].algo, name,
                                GPUCOMPRESS_PREPROC_SHUFFLE_4, 0.0);
    }

    // Part 1c: With quantization (lossy)
    printf("\n--- Part 1c: With Quantization (error_bound=0.001) ---\n");
    print_separator();
    for (int i = 0; i < 4; i++) {  // Just top 4 algorithms
        char name[32];
        snprintf(name, sizeof(name), "%s+quant", algos[i].name);
        run_compress_decompress(data.data(), data_size,
                                algos[i].algo, name,
                                GPUCOMPRESS_PREPROC_QUANTIZE, 0.001);
    }

    gpucompress_cleanup();

    // ============================================================
    // Part 2: Neural Network auto-selection
    // ============================================================
    if (weights_path) {
        printf("\n\n--- Part 2: Neural Network Auto-Selection ---\n");
        printf("Weights: %s\n\n", weights_path);

        rc = gpucompress_init(weights_path);
        if (rc != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "Init with NN failed: %s\n", gpucompress_error_string(rc));
            return 1;
        }

        if (!gpucompress_nn_is_loaded()) {
            fprintf(stderr, "Warning: NN weights not loaded, ALGO_AUTO will fallback\n");
        } else {
            printf("NN loaded successfully.\n\n");
        }

        print_separator();

        // Run AUTO selection multiple times to see consistency
        for (int run = 0; run < 3; run++) {
            char name[32];
            snprintf(name, sizeof(name), "auto_run%d", run + 1);
            run_compress_decompress(data.data(), data_size,
                                    GPUCOMPRESS_ALGO_AUTO, name,
                                    GPUCOMPRESS_PREPROC_NONE, 0.0);
        }

        // AUTO with shuffle
        printf("\n");
        run_compress_decompress(data.data(), data_size,
                                GPUCOMPRESS_ALGO_AUTO, "auto+shuf",
                                GPUCOMPRESS_PREPROC_SHUFFLE_4, 0.0);

        // AUTO with quantization
        run_compress_decompress(data.data(), data_size,
                                GPUCOMPRESS_ALGO_AUTO, "auto+quant",
                                GPUCOMPRESS_PREPROC_QUANTIZE, 0.001);

        gpucompress_cleanup();
    } else {
        printf("\n\nSkipping Part 2 (no weights file provided).\n");
        printf("Re-run with: %s %s <weights.nnwt>\n", argv[0], input_path);
    }

    printf("\n=== Done ===\n");
    return 0;
}
