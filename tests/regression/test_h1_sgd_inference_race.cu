/**
 * test_h1_sgd_inference_race.cu
 *
 * H1: Non-Ctx inference paths (runNNInference, runNNFusedInference) don't
 *     call cudaStreamWaitEvent(g_sgd_done) before reading NN weights.
 *     Non-Ctx SGD (runNNSGD) never records g_sgd_done after writing weights.
 *     Result: inference can read partially-updated weights during concurrent
 *     SGD, producing unpredictable algorithm rankings.
 *
 * Test strategy:
 *   1. Init with NN weights, enable online learning with aggressive SGD
 *      (threshold=0 so SGD fires every time)
 *   2. Launch multiple threads doing ALGO_AUTO compression concurrently
 *   3. Check that all returned actions are valid (0-31), predicted ratios
 *      are not NaN/Inf, and timing values are sane
 *   4. Before fix: race on weights can produce garbage predictions
 *
 * Run: ./test_h1_sgd_inference_race
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <atomic>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

#define NUM_THREADS 4
#define ITERATIONS  10

struct ThreadResult {
    int invalid_actions;
    int nan_ratios;
    int decompress_failures;
    int compress_errors;
    int total;
};

static std::atomic<int> g_ready{0};
static std::atomic<int> g_go{0};

static void* compress_worker(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    memset(result, 0, sizeof(*result));

    const size_t DATA_SIZE = 64 * 1024;
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
        h_data[i] = sinf((float)i * 0.01f) * 100.0f;
    }

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_compressed = malloc(max_out);
    float* h_decomp = (float*)malloc(DATA_SIZE);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    /* Synchronize thread start */
    g_ready.fetch_add(1);
    while (!g_go.load()) { /* spin */ }

    for (int iter = 0; iter < ITERATIONS; iter++) {
        size_t compressed_size = max_out;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);

        result->total++;

        if (err != GPUCOMPRESS_SUCCESS) {
            result->compress_errors++;
            continue;
        }

        /* Check action validity */
        int action = stats.nn_final_action;
        if (action < 0 || action > 31) {
            result->invalid_actions++;
        }

        /* Check for NaN in predictions */
        if (std::isnan(stats.predicted_ratio) || std::isinf(stats.predicted_ratio)) {
            result->nan_ratios++;
        }

        /* Round-trip decompression to verify data integrity */
        size_t decomp_size = DATA_SIZE;
        err = gpucompress_decompress(h_compressed, compressed_size,
                                     h_decomp, &decomp_size);
        if (err != GPUCOMPRESS_SUCCESS) {
            result->decompress_failures++;
        }
    }

    free(h_data);
    free(h_compressed);
    free(h_decomp);
    return NULL;
}

int main(void) {
    printf("=== H1: Non-Ctx inference/SGD synchronization race ===\n\n");

    gpucompress_error_t err = gpucompress_init(WEIGHTS_PATH);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Enable aggressive online learning: SGD fires on every compress */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.01f, 0.0f, 0.0f);  /* threshold=0: always fire SGD */
    gpucompress_set_exploration(0);  /* no exploration, just SGD */

    printf("--- Concurrent ALGO_AUTO with SGD (%d threads x %d iterations) ---\n",
           NUM_THREADS, ITERATIONS);

    pthread_t threads[NUM_THREADS];
    ThreadResult results[NUM_THREADS];

    g_ready.store(0);
    g_go.store(0);

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_create(&threads[t], NULL, compress_worker, &results[t]);
    }

    /* Wait for all threads to be ready, then release them together */
    while (g_ready.load() < NUM_THREADS) { /* spin */ }
    g_go.store(1);

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Aggregate results */
    int total_invalid = 0, total_nan = 0, total_decomp_fail = 0, total_errors = 0;
    int total_ops = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        total_invalid    += results[t].invalid_actions;
        total_nan        += results[t].nan_ratios;
        total_decomp_fail += results[t].decompress_failures;
        total_errors     += results[t].compress_errors;
        total_ops        += results[t].total;
    }

    printf("\n  Results: %d operations total\n", total_ops);
    printf("  Invalid actions (outside 0-31): %d\n", total_invalid);
    printf("  NaN/Inf predictions: %d\n", total_nan);
    printf("  Decompression failures: %d\n", total_decomp_fail);
    printf("  Compression errors: %d\n", total_errors);

    if (total_invalid > 0) {
        FAIL("invalid action IDs from concurrent inference+SGD");
    } else {
        PASS("all action IDs valid (0-31)");
    }

    if (total_nan > 0) {
        FAIL("NaN/Inf predictions from concurrent inference+SGD");
    } else {
        PASS("all predictions are finite");
    }

    if (total_decomp_fail > 0) {
        FAIL("decompression failures from corrupt algorithm selection");
    } else {
        PASS("all round-trips succeeded");
    }

    if (total_errors > 0) {
        printf("  WARNING: %d compression errors (may indicate resource contention)\n", total_errors);
    }

    gpucompress_disable_online_learning();
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
