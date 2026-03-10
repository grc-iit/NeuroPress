/**
 * test_c2_timing_race.cu
 *
 * C2: g_t_start / g_t_stop are global CUDA timing events shared by all
 *     threads calling gpucompress_compress(). Concurrent calls race on
 *     cudaEventRecord/cudaEventElapsedTime, producing corrupt timing
 *     values that feed into SGD reinforcement.
 *
 * Test strategy:
 *   1. Run single-threaded compressions to establish baseline timing
 *   2. Run N threads concurrently, each compressing data and recording
 *      actual_comp_time_ms from the stats struct
 *   3. Before fix: race on global events produces negative, zero, or
 *      wildly wrong timings. Check for negative values and large
 *      deviation from baseline.
 *
 * Run: ./test_c2_timing_race
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

#define NUM_THREADS 8
#define ITERATIONS_PER_THREAD 5

struct ThreadResult {
    double timings[ITERATIONS_PER_THREAD];
    int errors;
    int negative_count;
    int zero_count;
};

static const size_t DATA_SIZE = 256 * 1024;

static void* compress_worker(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    result->errors = 0;
    result->negative_count = 0;
    result->zero_count = 0;

    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
        h_data[i] = sinf((float)i * 0.01f) * 100.0f;
    }

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_compressed = malloc(max_out);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    for (int iter = 0; iter < ITERATIONS_PER_THREAD; iter++) {
        size_t compressed_size = max_out;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);

        if (err != GPUCOMPRESS_SUCCESS) {
            result->errors++;
            result->timings[iter] = -999.0;
            continue;
        }

        result->timings[iter] = stats.actual_comp_time_ms;

        if (stats.actual_comp_time_ms < 0.0) {
            result->negative_count++;
        } else if (stats.actual_comp_time_ms == 0.0) {
            result->zero_count++;
        }
    }

    free(h_data);
    free(h_compressed);
    return NULL;
}

int main(void) {
    printf("=== C2: Global CUDA timing events race condition ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Single-threaded baseline ---- */
    printf("--- Test 1: Single-threaded baseline timing ---\n");
    double baseline_sum = 0.0;
    int baseline_count = 0;
    {
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
            h_data[i] = sinf((float)i * 0.01f) * 100.0f;

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

        for (int i = 0; i < 5; i++) {
            size_t compressed_size = max_out;
            gpucompress_stats_t stats;
            memset(&stats, 0, sizeof(stats));
            err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);
            if (err == GPUCOMPRESS_SUCCESS && stats.actual_comp_time_ms > 0.0) {
                baseline_sum += stats.actual_comp_time_ms;
                baseline_count++;
                printf("  baseline[%d] = %.4f ms\n", i, stats.actual_comp_time_ms);
            }
        }
        free(h_data);
        free(h_compressed);
    }

    if (baseline_count == 0) {
        printf("  SKIP: could not establish baseline timing\n");
        gpucompress_cleanup();
        return 1;
    }
    double baseline_avg = baseline_sum / baseline_count;
    printf("  baseline avg = %.4f ms\n\n", baseline_avg);

    /* ---- Test 2: Concurrent compression timing ---- */
    printf("--- Test 2: %d threads x %d iterations concurrent timing ---\n",
           NUM_THREADS, ITERATIONS_PER_THREAD);

    pthread_t threads[NUM_THREADS];
    ThreadResult results[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_create(&threads[t], NULL, compress_worker, &results[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Analyze results */
    int total_negative = 0;
    int total_zero = 0;
    int total_errors = 0;
    int total_corrupt = 0;
    int total_samples = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        total_negative += results[t].negative_count;
        total_zero += results[t].zero_count;
        total_errors += results[t].errors;

        for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
            double val = results[t].timings[i];
            if (val < -900.0) continue; /* compression error, skip */
            total_samples++;

            /* A timing more than 100x baseline or negative is corrupt */
            if (val < 0.0 || val > baseline_avg * 100.0) {
                total_corrupt++;
                printf("  CORRUPT: thread=%d iter=%d timing=%.4f ms (baseline=%.4f)\n",
                       t, i, val, baseline_avg);
            }
        }
    }

    printf("\n  Summary: %d samples, %d negative, %d zero, %d corrupt, %d errors\n",
           total_samples, total_negative, total_zero, total_corrupt, total_errors);

    if (total_negative > 0) {
        printf("  Negative timings detected — events raced\n");
        FAIL("negative timing values from concurrent access");
    } else {
        PASS("no negative timing values");
    }

    if (total_corrupt > 0) {
        printf("  Corrupt timings detected — events raced\n");
        FAIL("corrupt timing values from concurrent access");
    } else {
        PASS("all timings within reasonable range of baseline");
    }

    /* ---- Test 3: Post-concurrent single-thread sanity check ---- */
    printf("\n--- Test 3: Post-concurrent single-thread sanity ---\n");
    {
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
            h_data[i] = sinf((float)i * 0.01f) * 100.0f;

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);
        if (err == GPUCOMPRESS_SUCCESS && stats.actual_comp_time_ms > 0.0) {
            printf("  post-concurrent timing = %.4f ms\n", stats.actual_comp_time_ms);
            PASS("timing still works after concurrent stress");
        } else {
            FAIL("timing broken after concurrent stress");
        }

        free(h_data);
        free(h_compressed);
    }

    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
