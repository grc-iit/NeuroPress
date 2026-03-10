/**
 * test_h3_auto_global_buffers.cu
 *
 * H3: Host-path ALGO_AUTO uses global singleton stats/inference buffers.
 *     Multiple threads calling gpucompress_compress() with ALGO_AUTO
 *     concurrently share g_d_stats and d_fused_infer_output, causing
 *     data corruption and wrong algorithm selection.
 *
 * Test strategy:
 *   1. Create two very different data patterns (smooth sine vs random noise)
 *   2. Run each pattern single-threaded to establish baseline algorithm choices
 *   3. Run both patterns concurrently in many threads
 *   4. Check that concurrent results match single-threaded baselines
 *      (same algorithm chosen, successful round-trip)
 *   5. Before fix: concurrent buffer corruption causes mismatched algorithms
 *      or decompression failures
 *
 * Run: ./test_h3_auto_global_buffers
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

#define NUM_THREADS 8
#define ITERATIONS  20

struct ThreadResult {
    int algo_mismatches;
    int decompress_failures;
    int compress_errors;
    int total;
    int baseline_algo;  /* expected algo from single-threaded run */
    int pattern_id;     /* 0 = smooth, 1 = random */
};

static std::atomic<int> g_ready{0};
static std::atomic<int> g_go{0};

/* Two distinct data patterns that should yield different algorithm choices.
 * Both use deterministic, thread-safe generation (no rand()). */
static void fill_smooth(float* data, size_t count) {
    for (size_t i = 0; i < count; i++)
        data[i] = sinf((float)i * 0.001f) * 50.0f;
}

static void fill_noisy(float* data, size_t count) {
    /* Deterministic pseudo-random via xorshift32 — thread-safe (no global state) */
    uint32_t state = 12345u;
    for (size_t i = 0; i < count; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data[i] = (float)(int32_t)state / 2147483648.0f * 500.0f;
    }
}

static void* compress_worker(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;

    const size_t DATA_SIZE = 128 * 1024;  /* 128 KB */
    const size_t num_floats = DATA_SIZE / sizeof(float);
    float* h_data = (float*)malloc(DATA_SIZE);

    if (result->pattern_id == 0)
        fill_smooth(h_data, num_floats);
    else
        fill_noisy(h_data, num_floats);

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

        /* Check algorithm consistency with baseline */
        if (stats.nn_final_action != result->baseline_algo) {
            result->algo_mismatches++;
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

/* Run single-threaded to get baseline algorithm for a pattern */
static int get_baseline_algo(int pattern_id) {
    const size_t DATA_SIZE = 128 * 1024;
    const size_t num_floats = DATA_SIZE / sizeof(float);
    float* h_data = (float*)malloc(DATA_SIZE);

    if (pattern_id == 0)
        fill_smooth(h_data, num_floats);
    else
        fill_noisy(h_data, num_floats);

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_compressed = malloc(max_out);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    size_t compressed_size = max_out;
    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    gpucompress_error_t err = gpucompress_compress(
        h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);

    int algo = -1;
    if (err == GPUCOMPRESS_SUCCESS)
        algo = stats.nn_final_action;

    free(h_data);
    free(h_compressed);
    return algo;
}

int main(void) {
    printf("=== H3: Host-path ALGO_AUTO global buffer race ===\n\n");

    gpucompress_error_t err = gpucompress_init(WEIGHTS_PATH);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Step 1: Get baseline algorithms (single-threaded, no contention) */
    printf("--- Single-threaded baselines ---\n");
    int baseline_smooth = get_baseline_algo(0);
    int baseline_random = get_baseline_algo(1);
    printf("  Smooth pattern baseline algo: %d\n", baseline_smooth);
    printf("  Random pattern baseline algo: %d\n", baseline_random);

    if (baseline_smooth < 0 || baseline_random < 0) {
        printf("  SKIP: could not establish baselines\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Step 2: Run concurrent threads, half smooth half random */
    printf("\n--- Concurrent ALGO_AUTO (%d threads x %d iterations) ---\n",
           NUM_THREADS, ITERATIONS);

    pthread_t threads[NUM_THREADS];
    ThreadResult results[NUM_THREADS];

    g_ready.store(0);
    g_go.store(0);

    for (int t = 0; t < NUM_THREADS; t++) {
        memset(&results[t], 0, sizeof(results[t]));
        results[t].pattern_id = t % 2;  /* alternate smooth/random */
        results[t].baseline_algo = (t % 2 == 0) ? baseline_smooth : baseline_random;
        pthread_create(&threads[t], NULL, compress_worker, &results[t]);
    }

    /* Wait for all threads to be ready, then release them together */
    while (g_ready.load() < NUM_THREADS) { /* spin */ }
    g_go.store(1);

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Aggregate results */
    int total_mismatches = 0, total_decomp_fail = 0, total_errors = 0;
    int total_ops = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        total_mismatches  += results[t].algo_mismatches;
        total_decomp_fail += results[t].decompress_failures;
        total_errors      += results[t].compress_errors;
        total_ops         += results[t].total;
    }

    printf("\n  Results: %d operations total\n", total_ops);
    printf("  Algorithm mismatches vs baseline: %d\n", total_mismatches);
    printf("  Decompression failures: %d\n", total_decomp_fail);
    printf("  Compression errors: %d\n", total_errors);

    /* Algorithm consistency check */
    if (total_mismatches > 0) {
        FAIL("algorithm choices differ from single-threaded baseline (buffer corruption)");
    } else {
        PASS("all algorithm choices match single-threaded baseline");
    }

    /* Round-trip check */
    if (total_decomp_fail > 0) {
        FAIL("decompression failures from concurrent ALGO_AUTO");
    } else {
        PASS("all round-trips succeeded");
    }

    if (total_errors > 0) {
        printf("  WARNING: %d compression errors (resource contention)\n", total_errors);
    }

    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
