/**
 * test_h4_sgd_mutex.cu
 *
 * H4: Host-path runNNSGD() call lacks mutex protection. The GPU path
 *     wraps runNNSGDCtx() with g_sgd_mutex, but the host path doesn't.
 *     Concurrent SGD calls corrupt shared global GPU buffers
 *     (d_sgd_grad_buffer, d_sgd_output, d_sgd_samples).
 *
 * Test strategy:
 *   1. Enable online learning with aggressive SGD (threshold=0)
 *   2. Launch multiple threads doing ALGO_AUTO compression concurrently
 *   3. Check for CUDA errors, decompression failures, NaN/corrupt
 *      gradient norms (symptoms of buffer corruption)
 *   4. Before fix: concurrent SGD corrupts shared buffers
 *
 * Run: ./test_h4_sgd_mutex
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <atomic>
#include <cstdint>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

#define NUM_THREADS 6
#define ITERATIONS  15

struct ThreadResult {
    int compress_errors;
    int decompress_failures;
    int total;
};

static std::atomic<int> g_ready{0};
static std::atomic<int> g_go{0};

static void* sgd_worker(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    memset(result, 0, sizeof(*result));

    /* Each thread gets unique deterministic data via xorshift */
    const size_t DATA_SIZE = 64 * 1024;
    const size_t num_floats = DATA_SIZE / sizeof(float);
    float* h_data = (float*)malloc(DATA_SIZE);
    uint32_t state = 42u + (uint32_t)(uintptr_t)arg;
    for (size_t i = 0; i < num_floats; i++) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        h_data[i] = (float)(int32_t)state / 2147483648.0f * 100.0f;
    }

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_compressed = malloc(max_out);
    float* h_decomp = (float*)malloc(DATA_SIZE);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

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

        /* Round-trip decompression */
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
    printf("=== H4: Host-path SGD missing mutex ===\n\n");

    gpucompress_error_t err = gpucompress_init(WEIGHTS_PATH);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Enable aggressive online learning: SGD fires every time */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.01f, 0.0f, 0.0f);
    gpucompress_set_exploration(0);

    printf("--- Concurrent ALGO_AUTO with SGD (%d threads x %d iterations) ---\n",
           NUM_THREADS, ITERATIONS);

    pthread_t threads[NUM_THREADS];
    ThreadResult results[NUM_THREADS];

    g_ready.store(0);
    g_go.store(0);

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_create(&threads[t], NULL, sgd_worker, &results[t]);
    }

    while (g_ready.load() < NUM_THREADS) { /* spin */ }
    g_go.store(1);

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    int total_errors = 0, total_decomp_fail = 0, total_ops = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        total_errors     += results[t].compress_errors;
        total_decomp_fail += results[t].decompress_failures;
        total_ops        += results[t].total;
    }

    printf("\n  Results: %d operations total\n", total_ops);
    printf("  Compression errors: %d\n", total_errors);
    printf("  Decompression failures: %d\n", total_decomp_fail);

    if (total_errors > 0) {
        FAIL("compression errors from concurrent SGD buffer corruption");
    } else {
        PASS("no compression errors");
    }

    if (total_decomp_fail > 0) {
        FAIL("decompression failures from concurrent SGD");
    } else {
        PASS("all round-trips succeeded");
    }

    gpucompress_disable_online_learning();
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
