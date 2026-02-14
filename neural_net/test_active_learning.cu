/**
 * Integration test for active learning system.
 *
 * Tests:
 *   1. Enable active learning → CSV file created with header
 *   2. Compress with ALGO_AUTO → experience CSV gets row(s)
 *   3. Verify experience_count increments
 *   4. Disable → no more samples collected
 *   5. Hot-reload NN weights
 *   6. Exploration threshold adjustment
 *
 * Build (via CMake target test_active_learning):
 *   cmake --build build --target test_active_learning
 * Run:
 *   ./test_active_learning weights/model.nnwt
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include "gpucompress.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("   PASS: %s\n", msg); \
        g_tests_passed++; \
    } else { \
        printf("   FAIL: %s\n", msg); \
        g_tests_failed++; \
    } \
} while(0)

/** Count lines in a text file (excluding header). */
static int count_csv_data_rows(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int lines = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        lines++;
    }
    fclose(f);
    return lines - 1;  // Subtract header
}

int main(int argc, char** argv) {
    const char* nn_path = (argc > 1) ? argv[1] : "weights/model.nnwt";
    const char* csv_path = "/tmp/test_active_learning.csv";

    printf("=== Active Learning Integration Test ===\n\n");

    // Clean up any leftover test file
    unlink(csv_path);

    // ---- Initialize library ----
    printf("1. Initializing with NN weights: %s\n", nn_path);
    gpucompress_error_t err = gpucompress_init(nn_path);
    CHECK(err == GPUCOMPRESS_SUCCESS, "gpucompress_init succeeds");
    CHECK(gpucompress_nn_is_loaded() == 1, "NN model loaded");

    // ---- Check active learning is off by default ----
    printf("\n2. Verifying active learning defaults\n");
    CHECK(gpucompress_active_learning_enabled() == 0, "Active learning off by default");
    CHECK(gpucompress_experience_count() == 0, "Experience count is 0");

    // ---- Enable active learning ----
    printf("\n3. Enabling active learning\n");
    err = gpucompress_enable_active_learning(csv_path);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Enable active learning succeeds");
    CHECK(gpucompress_active_learning_enabled() == 1, "Active learning is now on");

    // Verify CSV file was created
    int rows = count_csv_data_rows(csv_path);
    CHECK(rows == 0, "CSV created with header, 0 data rows");

    // ---- Generate test data ----
    printf("\n4. Compressing with ALGO_AUTO (active learning enabled)\n");
    const size_t num_elements = 256 * 1024;  // 256K floats = 1MB
    const size_t data_size = num_elements * sizeof(float);
    float* h_data = (float*)malloc(data_size);

    // Smooth sine wave
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = sinf(static_cast<float>(i) * 0.01f) * 100.0f;
    }

    size_t max_output = gpucompress_max_compressed_size(data_size);
    void* output = malloc(max_output);
    size_t output_size = max_output;

    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm = GPUCOMPRESS_ALGO_AUTO;
    config.error_bound = 0.0;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Compression succeeds");
    printf("   Algorithm: %s, Ratio: %.2fx\n",
           gpucompress_algorithm_name(stats.algorithm_used),
           stats.compression_ratio);

    // Check that at least one experience row was written (Level 1)
    size_t count_after_first = gpucompress_experience_count();
    CHECK(count_after_first >= 1, "At least 1 experience sample collected");

    rows = count_csv_data_rows(csv_path);
    CHECK(rows >= 1, "CSV has at least 1 data row");
    printf("   Experience count: %zu, CSV rows: %d\n", count_after_first, rows);

    // ---- Compress again with different data ----
    printf("\n5. Compressing random data (may trigger exploration)\n");
    srand(42);
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    output_size = max_output;
    memset(&stats, 0, sizeof(stats));
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Second compression succeeds");

    size_t count_after_second = gpucompress_experience_count();
    CHECK(count_after_second > count_after_first, "More experience samples collected");
    printf("   Experience count: %zu (was %zu)\n", count_after_second, count_after_first);

    // ---- Test lossy compression ----
    printf("\n6. Compressing with lossy config\n");
    config.error_bound = 0.01;
    output_size = max_output;
    memset(&stats, 0, sizeof(stats));
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Lossy compression succeeds");

    size_t count_after_lossy = gpucompress_experience_count();
    CHECK(count_after_lossy > count_after_second, "Lossy compression adds experience");
    printf("   Experience count: %zu\n", count_after_lossy);

    // ---- Test threshold adjustment ----
    printf("\n7. Testing exploration threshold\n");
    gpucompress_set_exploration_threshold(0.05);  // 5% - very sensitive
    config.error_bound = 0.0;
    output_size = max_output;
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Compression with low threshold succeeds");

    size_t count_after_threshold = gpucompress_experience_count();
    printf("   Experience count: %zu (was %zu)\n", count_after_threshold, count_after_lossy);

    // Reset threshold
    gpucompress_set_exploration_threshold(0.20);

    // ---- Disable active learning ----
    printf("\n8. Disabling active learning\n");
    gpucompress_disable_active_learning();
    CHECK(gpucompress_active_learning_enabled() == 0, "Active learning disabled");

    // Compress again - should NOT add experience
    output_size = max_output;
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Compression works with AL disabled");
    CHECK(gpucompress_experience_count() == 0, "Count reset after disable");

    // ---- Test hot-reload ----
    printf("\n9. Testing hot-reload\n");
    err = gpucompress_reload_nn(nn_path);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Hot-reload succeeds");
    CHECK(gpucompress_nn_is_loaded() == 1, "NN still loaded after reload");

    // Verify compression still works after reload
    output_size = max_output;
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Compression works after hot-reload");

    // ---- Test re-enable after disable ----
    printf("\n10. Re-enabling active learning\n");
    const char* csv_path2 = "/tmp/test_active_learning_2.csv";
    unlink(csv_path2);
    err = gpucompress_enable_active_learning(csv_path2);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Re-enable succeeds with new file");
    CHECK(gpucompress_experience_count() == 0, "Count starts at 0 for new file");

    output_size = max_output;
    err = gpucompress_compress(h_data, data_size, output, &output_size, &config, &stats);
    CHECK(err == GPUCOMPRESS_SUCCESS, "Compression after re-enable succeeds");
    CHECK(gpucompress_experience_count() >= 1, "New samples collected");

    // ---- Cleanup ----
    printf("\n11. Cleanup\n");
    gpucompress_disable_active_learning();
    free(h_data);
    free(output);
    gpucompress_cleanup();

    // Clean up test files
    unlink(csv_path);
    unlink(csv_path2);

    // ---- Summary ----
    printf("\n=== Results: %d passed, %d failed ===\n",
           g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
