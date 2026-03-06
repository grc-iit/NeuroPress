/**
 * @file test_nn_pipeline.cpp
 * @brief End-to-end test: NN inference, active learning, reinforcement, round-trip
 *
 * Generates synthetic float32 data, compresses with ALGO_AUTO (NN),
 * enables active learning + online reinforcement, verifies predictions
 * and round-trip correctness.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "gpucompress.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* algo_name(gpucompress_algorithm_t a) {
    return gpucompress_algorithm_name(a);
}

static const char* preproc_str(unsigned int p) {
    static char buf[64];
    buf[0] = '\0';
    if (p & GPUCOMPRESS_PREPROC_QUANTIZE)  strcat(buf, "quant ");
    if (p & GPUCOMPRESS_PREPROC_SHUFFLE_4) strcat(buf, "shuf4 ");
    if (buf[0] == '\0') strcpy(buf, "none");
    return buf;
}

#define CHECK(expr, msg) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        fprintf(stderr, "  OK: %s\n", msg); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Synthetic data generators
// ---------------------------------------------------------------------------

/** Low-entropy data: smooth sine wave */
static std::vector<float> gen_smooth(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = 0.5f + 0.4f * sinf(static_cast<float>(i) * 0.01f);
    return v;
}

/** Medium-entropy data: noisy ramp */
static std::vector<float> gen_noisy_ramp(size_t n) {
    std::vector<float> v(n);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        float base = static_cast<float>(i) / static_cast<float>(n);
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.3f;
        v[i] = base + noise;
    }
    return v;
}

/** High-entropy data: random uniform */
static std::vector<float> gen_random(size_t n) {
    std::vector<float> v(n);
    srand(123);
    for (size_t i = 0; i < n; i++)
        v[i] = static_cast<float>(rand()) / RAND_MAX;
    return v;
}

// ---------------------------------------------------------------------------
// Test: Basic NN inference with ALGO_AUTO
// ---------------------------------------------------------------------------

static int test_nn_inference(int& failures) {
    fprintf(stderr, "\n=== Test 1: NN Inference with ALGO_AUTO ===\n");

    size_t num_floats = 262144;  // 1 MB
    auto data = gen_noisy_ramp(num_floats);
    size_t data_size = num_floats * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(data_size);
    std::vector<uint8_t> output(max_out);
    size_t out_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
    cfg.error_bound = 0.0;  // lossless

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    gpucompress_error_t rc = gpucompress_compress(
        data.data(), data_size, output.data(), &out_size, &cfg, &stats);

    CHECK(rc == GPUCOMPRESS_SUCCESS, "ALGO_AUTO compress succeeded");
    CHECK(stats.predicted_ratio > 0.0, "NN produced a predicted ratio");
    CHECK(stats.compression_ratio > 0.0, "Actual compression ratio > 0");
    CHECK(stats.entropy_bits > 0.0, "Entropy was computed");
    CHECK(stats.algorithm_used >= GPUCOMPRESS_ALGO_LZ4 &&
          stats.algorithm_used <= GPUCOMPRESS_ALGO_BITCOMP,
          "Valid algorithm selected");

    fprintf(stderr, "  Details:\n");
    fprintf(stderr, "    Algorithm:      %s\n", algo_name(stats.algorithm_used));
    fprintf(stderr, "    Preprocessing:  %s\n", preproc_str(stats.preprocessing_used));
    fprintf(stderr, "    Predicted ratio: %.4f\n", stats.predicted_ratio);
    fprintf(stderr, "    Actual ratio:    %.4f\n", stats.compression_ratio);
    fprintf(stderr, "    Entropy:         %.4f bits\n", stats.entropy_bits);
    fprintf(stderr, "    MAD:             %.6f\n", stats.mad);
    fprintf(stderr, "    2nd derivative:  %.6f\n", stats.second_derivative);
    fprintf(stderr, "    Pred comp time:  %.4f ms\n", stats.predicted_comp_time_ms);

    // Round-trip: decompress and verify
    size_t orig_size = data_size;
    std::vector<uint8_t> decompressed(orig_size);
    size_t decomp_size = orig_size;

    rc = gpucompress_decompress(output.data(), out_size,
                                decompressed.data(), &decomp_size);

    CHECK(rc == GPUCOMPRESS_SUCCESS, "Decompression succeeded");
    CHECK(decomp_size == data_size, "Decompressed size matches original");

    // Lossless: exact match
    bool exact = (memcmp(data.data(), decompressed.data(), data_size) == 0);
    CHECK(exact, "Lossless round-trip: data matches exactly");

    return 0;
}

// ---------------------------------------------------------------------------
// Test: Active learning + reinforcement
// ---------------------------------------------------------------------------

static int test_active_learning_and_reinforce(int& failures) {
    fprintf(stderr, "\n=== Test 2: Active Learning + Reinforcement ===\n");

    // Enable active learning
    gpucompress_error_t rc = gpucompress_enable_active_learning();
    CHECK(rc == GPUCOMPRESS_SUCCESS, "Active learning enabled");
    CHECK(gpucompress_active_learning_enabled() == 1, "Active learning flag is set");

    // Enable reinforcement with low threshold so it triggers easily
    gpucompress_set_reinforcement(1, 1e-4f, 0.01f, 0.0f);

    // Set very low exploration threshold to force exploration
    gpucompress_set_exploration_threshold(0.01);

    // Compress multiple different data patterns
    const char* patterns[] = {"smooth", "noisy_ramp", "random"};
    size_t num_floats = 131072;  // 512 KB
    size_t data_size = num_floats * sizeof(float);

    int total_sgd_fired = 0;

    for (int p = 0; p < 3; p++) {
        std::vector<float> data;
        if (p == 0) data = gen_smooth(num_floats);
        else if (p == 1) data = gen_noisy_ramp(num_floats);
        else data = gen_random(num_floats);

        size_t max_out = gpucompress_max_compressed_size(data_size);
        std::vector<uint8_t> output(max_out);
        size_t out_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = 0.001;  // lossy, enables quantization actions

        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        rc = gpucompress_compress(
            data.data(), data_size, output.data(), &out_size, &cfg, &stats);

        CHECK(rc == GPUCOMPRESS_SUCCESS,
              (std::string("Compress ") + patterns[p] + " succeeded").c_str());

        fprintf(stderr, "  [%s] algo=%s ratio=%.3f predicted=%.3f sgd=%d\n",
                patterns[p], algo_name(stats.algorithm_used),
                stats.compression_ratio, stats.predicted_ratio,
                stats.sgd_fired);

        if (stats.sgd_fired) {
            total_sgd_fired++;
        }

        // Round-trip verify (lossy)
        size_t decomp_size = data_size;
        std::vector<uint8_t> decompressed(data_size);
        gpucompress_error_t drc = gpucompress_decompress(
            output.data(), out_size, decompressed.data(), &decomp_size);
        CHECK(drc == GPUCOMPRESS_SUCCESS,
              (std::string("Decompress ") + patterns[p] + " succeeded").c_str());

        // For lossy: check error bound
        if (stats.preprocessing_used & GPUCOMPRESS_PREPROC_QUANTIZE) {
            const float* orig = data.data();
            const float* dec = reinterpret_cast<const float*>(decompressed.data());
            double max_err = 0.0;
            for (size_t i = 0; i < num_floats; i++) {
                double err = fabs(static_cast<double>(orig[i]) - static_cast<double>(dec[i]));
                if (err > max_err) max_err = err;
            }
            fprintf(stderr, "    Max error: %.8f (bound: 0.001)\n", max_err);
            CHECK(max_err <= 0.002, "Lossy error within 2x bound");
        }
    }

    fprintf(stderr, "\n  SGD fired count: %d / 3 patterns\n", total_sgd_fired);

    // Disable
    gpucompress_disable_active_learning();
    CHECK(gpucompress_active_learning_enabled() == 0, "Active learning disabled");

    return 0;
}

// ---------------------------------------------------------------------------
// Test: Multiple data sizes with NN
// ---------------------------------------------------------------------------

static int test_multiple_sizes(int& failures) {
    fprintf(stderr, "\n=== Test 3: Multiple Data Sizes ===\n");

    size_t sizes[] = {1024, 16384, 131072, 524288};  // 4KB, 64KB, 512KB, 2MB
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        size_t num_floats = sizes[s];
        auto data = gen_noisy_ramp(num_floats);
        size_t data_size = num_floats * sizeof(float);

        size_t max_out = gpucompress_max_compressed_size(data_size);
        std::vector<uint8_t> output(max_out);
        size_t out_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        gpucompress_error_t rc = gpucompress_compress(
            data.data(), data_size, output.data(), &out_size, &cfg, &stats);

        CHECK(rc == GPUCOMPRESS_SUCCESS,
              (std::string("Compress ") + std::to_string(data_size/1024) + "KB succeeded").c_str());
        CHECK(stats.predicted_ratio > 0.0,
              (std::string("NN prediction for ") + std::to_string(data_size/1024) + "KB").c_str());

        fprintf(stderr, "    %6zuKB -> algo=%s ratio=%.3f predicted=%.3f\n",
                data_size / 1024, algo_name(stats.algorithm_used),
                stats.compression_ratio, stats.predicted_ratio);

        // Round-trip
        size_t decomp_size = data_size;
        std::vector<uint8_t> decompressed(data_size);
        rc = gpucompress_decompress(output.data(), out_size,
                                    decompressed.data(), &decomp_size);
        CHECK(rc == GPUCOMPRESS_SUCCESS, "Round-trip decompression");
        CHECK(decomp_size == data_size, "Decompressed size matches");
        CHECK(memcmp(data.data(), decompressed.data(), data_size) == 0,
              "Lossless data integrity");
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Test: Compute stats API
// ---------------------------------------------------------------------------

static int test_compute_stats(int& failures) {
    fprintf(stderr, "\n=== Test 4: Compute Stats API ===\n");

    size_t num_floats = 65536;
    auto data = gen_smooth(num_floats);
    size_t data_size = num_floats * sizeof(float);

    double entropy = 0, mad = 0, deriv = 0;
    gpucompress_error_t rc = gpucompress_compute_stats(
        data.data(), data_size, &entropy, &mad, &deriv);

    CHECK(rc == GPUCOMPRESS_SUCCESS, "compute_stats succeeded");
    CHECK(entropy >= 0.0 && entropy <= 8.0, "Entropy in valid range [0,8]");
    CHECK(mad >= 0.0, "MAD is non-negative");
    CHECK(deriv >= 0.0, "Second derivative is non-negative");

    fprintf(stderr, "    Entropy: %.4f bits, MAD: %.6f, 2nd deriv: %.6f\n",
            entropy, mad, deriv);

    // High-entropy data should have higher entropy
    auto random_data = gen_random(num_floats);
    double ent2 = 0, mad2 = 0, deriv2 = 0;
    rc = gpucompress_compute_stats(
        random_data.data(), data_size, &ent2, &mad2, &deriv2);

    CHECK(rc == GPUCOMPRESS_SUCCESS, "compute_stats random succeeded");
    CHECK(ent2 > entropy, "Random data has higher entropy than smooth");

    fprintf(stderr, "    Random: entropy=%.4f, MAD=%.6f, deriv=%.6f\n",
            ent2, mad2, deriv2);

    return 0;
}

// ---------------------------------------------------------------------------
// Test: NN hot-reload
// ---------------------------------------------------------------------------

static int test_nn_reload(int& failures) {
    fprintf(stderr, "\n=== Test 5: NN Hot-Reload ===\n");

    CHECK(gpucompress_nn_is_loaded() == 1, "NN is loaded before reload");

    // Reload the same weights
    gpucompress_error_t rc = gpucompress_reload_nn("neural_net/weights/model.nnwt");
    CHECK(rc == GPUCOMPRESS_SUCCESS, "Hot-reload succeeded");
    CHECK(gpucompress_nn_is_loaded() == 1, "NN still loaded after reload");

    // Compress should still work after reload
    size_t num_floats = 65536;
    auto data = gen_noisy_ramp(num_floats);
    size_t data_size = num_floats * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(data_size);
    std::vector<uint8_t> output(max_out);
    size_t out_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    rc = gpucompress_compress(
        data.data(), data_size, output.data(), &out_size, &cfg, &stats);

    CHECK(rc == GPUCOMPRESS_SUCCESS, "Compress after reload succeeded");
    CHECK(stats.predicted_ratio > 0.0, "NN prediction after reload works");

    fprintf(stderr, "    Post-reload: algo=%s ratio=%.3f predicted=%.3f\n",
            algo_name(stats.algorithm_used),
            stats.compression_ratio, stats.predicted_ratio);

    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int failures = 0;

    fprintf(stderr, "========================================\n");
    fprintf(stderr, " GPUCompress NN Pipeline End-to-End Test\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "Library version: %s\n", gpucompress_version());

    // Initialize with NN weights
    const char* weights = "neural_net/weights/model.nnwt";
    fprintf(stderr, "\nInitializing with weights: %s\n", weights);

    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }

    CHECK(gpucompress_is_initialized() == 1, "Library initialized");
    CHECK(gpucompress_nn_is_loaded() == 1, "NN weights loaded");

    // Run all tests
    test_nn_inference(failures);
    test_active_learning_and_reinforce(failures);
    test_multiple_sizes(failures);
    test_compute_stats(failures);
    test_nn_reload(failures);

    // Cleanup
    gpucompress_cleanup();

    fprintf(stderr, "\n========================================\n");
    if (failures == 0) {
        fprintf(stderr, " ALL TESTS PASSED\n");
    } else {
        fprintf(stderr, " %d TEST(S) FAILED\n", failures);
    }
    fprintf(stderr, "========================================\n");

    return failures > 0 ? 1 : 0;
}
