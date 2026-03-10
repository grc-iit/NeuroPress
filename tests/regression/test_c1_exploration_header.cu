/**
 * test_c1_exploration_header.cu
 *
 * C1: When exploration finds a better alternative, the winner's
 *     CompressionHeader is built without calling setAlgorithmId().
 *     The algorithm field defaults to 0, causing decompression to
 *     dispatch the wrong nvCOMP decompressor.
 *
 * Test strategy:
 *   1. Init library with NN weights, enable exploration (threshold=0, K=31)
 *   2. Compress multiple data patterns with ALGO_AUTO to maximize chance
 *      that exploration finds a better alternative and replaces the winner
 *   3. When exploration changes the action, verify the header algo ID
 *      matches the final action's algorithm
 *   4. Attempt round-trip decompression on every compressed output
 *
 * Run: ./test_c1_exploration_header
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gpucompress.h"
#include "compression/compression_header.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

/* Generate different data patterns to provoke different NN decisions */
static void fill_pattern(float* data, size_t count, int pattern) {
    switch (pattern) {
        case 0: /* smooth sine */
            for (size_t i = 0; i < count; i++)
                data[i] = sinf((float)i * 0.001f) * 100.0f;
            break;
        case 1: /* high entropy random-ish */
            for (size_t i = 0; i < count; i++)
                data[i] = sinf((float)i * 3.7f) * cosf((float)i * 0.13f) * 50000.0f;
            break;
        case 2: /* near-constant with noise */
            for (size_t i = 0; i < count; i++)
                data[i] = 42.0f + (float)(i % 3) * 0.001f;
            break;
        case 3: /* step function */
            for (size_t i = 0; i < count; i++)
                data[i] = (float)((i / 256) % 7) * 1000.0f;
            break;
        case 4: /* large range sawtooth */
            for (size_t i = 0; i < count; i++)
                data[i] = (float)(i % 1024) * 100.0f - 50000.0f;
            break;
        case 5: /* zeros */
            memset(data, 0, count * sizeof(float));
            break;
        case 6: /* exponential growth */
            for (size_t i = 0; i < count; i++)
                data[i] = expf((float)i / (float)count * 10.0f);
            break;
        case 7: /* alternating sign */
            for (size_t i = 0; i < count; i++)
                data[i] = (i & 1) ? 999.0f : -999.0f;
            break;
    }
}

int main(void) {
    printf("=== C1: Exploration winner header missing setAlgorithmId ===\n\n");

    gpucompress_error_t err = gpucompress_init(WEIGHTS_PATH);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d: %s)\n", err, gpucompress_error_string(err));
        printf("  (need NN weights at %s)\n", WEIGHTS_PATH);
        return 1;
    }

    /* Force exploration on every compress call */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(31);

    const size_t DATA_SIZE = 256 * 1024;
    const size_t NUM_FLOATS = DATA_SIZE / sizeof(float);
    const int NUM_PATTERNS = 8;
    int exploration_changed = 0;

    float* h_data = (float*)malloc(DATA_SIZE);

    for (int p = 0; p < NUM_PATTERNS; p++) {
        fill_pattern(h_data, NUM_FLOATS, p);

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        printf("--- Pattern %d ---\n", p);
        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, &cfg, &stats);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: compression failed (%d)\n", err);
            free(h_compressed);
            continue;
        }

        int triggered = gpucompress_get_last_exploration_triggered();
        int original = stats.nn_original_action;
        int final_act = stats.nn_final_action;

        CompressionHeader hdr;
        memcpy(&hdr, h_compressed, sizeof(CompressionHeader));
        uint8_t header_algo = hdr.getAlgorithmId();
        int expected_algo = (final_act % 8) + 1;

        printf("  triggered=%d original=%d final=%d header_algo=%u expected=%d\n",
               triggered, original, final_act, header_algo, expected_algo);

        if (triggered && final_act != original) {
            /* This is the buggy path — exploration replaced the winner */
            exploration_changed = 1;
            printf("  ** Exploration changed action: %d -> %d **\n", original, final_act);

            if (header_algo == (uint8_t)expected_algo) {
                PASS("exploration winner: header algo ID correct");
            } else {
                printf("  BUG: header_algo=%u expected=%d\n", header_algo, expected_algo);
                FAIL("exploration winner: header algo ID WRONG (C1 bug)");
            }
        }

        /* Round-trip decompression — always test this */
        float* h_decomp = (float*)malloc(DATA_SIZE);
        size_t decomp_size = DATA_SIZE;
        err = gpucompress_decompress(h_compressed, compressed_size, h_decomp, &decomp_size);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  decompress error: %d (%s)\n", err, gpucompress_error_string(err));
            FAIL("round-trip decompression failed");
        } else if (decomp_size != DATA_SIZE) {
            FAIL("decompressed size mismatch");
        } else if (memcmp(h_data, h_decomp, DATA_SIZE) != 0) {
            FAIL("data corruption after round-trip");
        } else {
            PASS("round-trip OK");
        }

        free(h_decomp);
        free(h_compressed);
    }

    if (!exploration_changed) {
        printf("\n  NOTE: Exploration never changed action across %d patterns.\n", NUM_PATTERNS);
        printf("  The C1 bug path was not exercised. Header+roundtrip tests still valid.\n");
    }

    free(h_data);
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
