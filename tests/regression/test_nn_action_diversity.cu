/**
 * test_nn_action_diversity.cu
 *
 * Diagnostic: feeds diverse data patterns through ALGO_AUTO and reports
 * what algorithm + preprocessing the NN selects for each.
 *
 * Checks:
 *   1. Does the NN pick different algorithms for different data?
 *   2. Does the NN ever select quantization (actions 8-15, 24-31)?
 *   3. Does the NN ever select shuffle (actions 16-31)?
 *   4. Are the NN predictions (ratio, comp_time, decomp_time) reasonable?
 *   5. Round-trip correctness for every NN-selected config
 *
 * Run: ./test_nn_action_diversity
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)

static const char* ALGO_NAMES[] = {
    "???", "LZ4", "Snappy", "Deflate", "GDeflate",
    "ZSTD", "ANS", "Cascaded", "Bitcomp"
};

/* Decode action to human-readable string */
static void decode_action(int action, char* buf, size_t buflen) {
    int algo = action % 8;
    int quant = (action / 8) % 2;
    int shuf = (action / 16) % 2;
    const char* algo_name = (algo + 1 <= 8) ? ALGO_NAMES[algo + 1] : "???";
    snprintf(buf, buflen, "action=%2d  algo=%-10s quant=%d  shuffle=%d",
             action, algo_name, quant, shuf);
}

/* ============================================================
 * Data generators — each produces a different statistical profile
 * ============================================================ */

struct DataPattern {
    const char* name;
    void (*generate)(float* data, size_t n);
    double error_bound;  /* 0 = lossless */
};

static void gen_smooth_sine(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = sinf((float)i * 0.001f) * 100.0f;
}

static void gen_random_uniform(float* d, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; i++)
        d[i] = (float)rand() / (float)RAND_MAX * 1000.0f;
}

static void gen_constant(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = 3.14159f;
}

static void gen_spike_train(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (i % 256 == 0) ? 1000.0f : 0.0f;
}

static void gen_linear_ramp(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (float)i * 0.01f;
}

static void gen_high_entropy(float* d, size_t n) {
    /* Pseudo-random with large range */
    unsigned int state = 12345;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        d[i] = (float)(int)state * 1e-5f;
    }
}

static void gen_grayscott_like(float* d, size_t n) {
    /* Simulate GS-like pattern: smooth with occasional sharp gradients */
    for (size_t i = 0; i < n; i++) {
        float x = (float)(i % 512) / 512.0f;
        float y = (float)((i / 512) % 512) / 512.0f;
        float v = sinf(x * 6.28f) * cosf(y * 6.28f);
        /* Add sharp spots */
        if (((i * 7) % 1000) < 10)
            v += 5.0f;
        d[i] = v;
    }
}

static void gen_stepped(float* d, size_t n) {
    /* Piecewise constant — good candidate for byte shuffle */
    for (size_t i = 0; i < n; i++)
        d[i] = (float)((int)(i / 1024)) * 10.0f;
}

static void gen_noisy_sine(float* d, size_t n) {
    unsigned int state = 99999;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        float noise = ((float)(state & 0xFFFF) / 65536.0f - 0.5f) * 0.1f;
        d[i] = sinf((float)i * 0.005f) * 50.0f + noise;
    }
}

static void gen_exponential_decay(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = 1000.0f * expf(-(float)i / (float)n * 10.0f);
}

/* ============================================================
 * Main diagnostic
 * ============================================================ */

int main(void)
{
    printf("=== test_nn_action_diversity ===\n");
    printf("Feeds diverse data patterns through ALGO_AUTO to check NN action diversity.\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %d\n", (int)err);
        return 1;
    }

    /* Load NN weights */
    int nn_loaded = gpucompress_nn_is_loaded();
    if (!nn_loaded) {
        /* Try common paths */
        const char* paths[] = {
            "neural_net/weights/model.nnwt",
            "../neural_net/weights/model.nnwt",
            NULL
        };
        for (int i = 0; paths[i]; i++) {
            if (gpucompress_load_nn(paths[i]) == GPUCOMPRESS_SUCCESS) {
                nn_loaded = 1;
                printf("Loaded NN weights from: %s\n", paths[i]);
                break;
            }
        }
    }
    if (!nn_loaded) {
        fprintf(stderr, "FATAL: Could not load NN weights. "
                "Set GPUCOMPRESS_WEIGHTS or run from repo root.\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Define test patterns */
    DataPattern patterns[] = {
        /* Lossless patterns (error_bound = 0) */
        { "smooth_sine",        gen_smooth_sine,        0.0   },
        { "random_uniform",     gen_random_uniform,     0.0   },
        { "constant",           gen_constant,           0.0   },
        { "spike_train",        gen_spike_train,        0.0   },
        { "linear_ramp",        gen_linear_ramp,        0.0   },
        { "high_entropy",       gen_high_entropy,       0.0   },
        { "grayscott_like",     gen_grayscott_like,     0.0   },
        { "stepped",            gen_stepped,            0.0   },
        { "noisy_sine",         gen_noisy_sine,         0.0   },
        { "exponential_decay",  gen_exponential_decay,  0.0   },
        /* Lossy patterns (error_bound > 0) — enables quantization */
        { "smooth_sine+Q",      gen_smooth_sine,        0.01  },
        { "random_uniform+Q",   gen_random_uniform,     0.01  },
        { "grayscott_like+Q",   gen_grayscott_like,     0.01  },
        { "noisy_sine+Q",       gen_noisy_sine,         0.1   },
        { "exponential_decay+Q",gen_exponential_decay,  1.0   },
        { "high_entropy+Q",     gen_high_entropy,       0.01  },
        { "stepped+Q",          gen_stepped,            0.5   },
        { "spike_train+Q",      gen_spike_train,        0.01  },
    };
    const int N_PATTERNS = sizeof(patterns) / sizeof(patterns[0]);

    const size_t N = 256 * 1024;  /* 256K floats = 1MB */
    const size_t data_bytes = N * sizeof(float);
    size_t max_comp = gpucompress_max_compressed_size(data_bytes);

    /* Tracking: count how many unique actions / algos / preproc combos we see */
    int action_histogram[32] = {0};
    int algo_count[9] = {0};  /* index 1-8 */
    int quant_count = 0;
    int shuffle_count = 0;
    int total_tested = 0;
    int roundtrip_pass = 0;

    printf("%-22s | %-45s | ratio   | pred_r  | pred_ct  | pred_dt  | pred_psnr | RT\n",
           "Pattern", "NN Decision");
    printf("%-22s-+-%-45s-+---------+---------+----------+----------+-----------+----\n",
           "----------------------", "---------------------------------------------");

    for (int p = 0; p < N_PATTERNS; p++) {
        DataPattern& pat = patterns[p];

        /* Generate on host */
        float* h_data = (float*)malloc(data_bytes);
        pat.generate(h_data, N);

        /* Upload to GPU */
        float* d_data = NULL;
        if (cudaMalloc(&d_data, data_bytes) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for pattern %s\n", pat.name);
            free(h_data);
            continue;
        }
        cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);

        uint8_t* d_comp = NULL;
        cudaMalloc(&d_comp, max_comp);

        /* Compress with ALGO_AUTO */
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = pat.error_bound;

        size_t comp_size = max_comp;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        err = gpucompress_compress_gpu(
            d_data, data_bytes, d_comp, &comp_size, &cfg, &stats, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: compress failed (err=%d)", pat.name, (int)err);
            FAIL(msg);
            cudaFree(d_comp);
            cudaFree(d_data);
            free(h_data);
            continue;
        }

        total_tested++;

        /* Decode the action */
        int action = stats.nn_final_action;
        int algo_idx = action % 8;
        int quant = (action / 8) % 2;
        int shuf = (action / 16) % 2;

        char decoded[128];
        decode_action(action, decoded, sizeof(decoded));

        action_histogram[action % 32]++;
        algo_count[algo_idx + 1]++;
        if (quant) quant_count++;
        if (shuf) shuffle_count++;

        /* Round-trip check */
        float* d_dec = NULL;
        cudaMalloc(&d_dec, data_bytes);
        size_t dec_size = data_bytes;
        gpucompress_error_t dec_err = gpucompress_decompress_gpu(
            d_comp, comp_size, d_dec, &dec_size, NULL);

        const char* rt_status = "??";
        if (dec_err == GPUCOMPRESS_SUCCESS) {
            float* h_result = (float*)malloc(data_bytes);
            cudaMemcpy(h_result, d_dec, data_bytes, cudaMemcpyDeviceToHost);

            double max_err = 0.0;
            for (size_t i = 0; i < N; i++) {
                double diff = fabs((double)h_data[i] - (double)h_result[i]);
                if (diff > max_err) max_err = diff;
            }

            if (pat.error_bound > 0.0) {
                rt_status = (max_err <= pat.error_bound) ? "OK" : "!!";
            } else {
                rt_status = (max_err == 0.0) ? "OK" : "!!";
            }
            if (strcmp(rt_status, "OK") == 0) roundtrip_pass++;

            free(h_result);
        } else {
            rt_status = "ER";
        }

        printf("%-22s | %-45s | %5.2fx  | %5.2fx  | %6.2fms | %6.2fms | %7.1fdB  | %s\n",
               pat.name, decoded,
               stats.compression_ratio,
               stats.predicted_ratio,
               stats.predicted_comp_time_ms,
               stats.predicted_decomp_time_ms,
               stats.predicted_psnr_db,
               rt_status);

        cudaFree(d_dec);
        cudaFree(d_comp);
        cudaFree(d_data);
        free(h_data);
    }

    /* ============================================================
     * Summary & checks
     * ============================================================ */
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  NN Action Diversity Summary                     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    /* Count unique actions */
    int unique_actions = 0;
    for (int i = 0; i < 32; i++) {
        if (action_histogram[i] > 0) unique_actions++;
    }

    /* Count unique algorithms */
    int unique_algos = 0;
    for (int i = 1; i <= 8; i++) {
        if (algo_count[i] > 0) unique_algos++;
    }

    printf("  Total patterns tested:    %d\n", total_tested);
    printf("  Round-trip correct:       %d/%d\n", roundtrip_pass, total_tested);
    printf("  Unique actions chosen:    %d / 32\n", unique_actions);
    printf("  Unique algorithms chosen: %d / 8\n", unique_algos);
    printf("  Quantization chosen:      %d / %d patterns\n", quant_count, total_tested);
    printf("  Shuffle chosen:           %d / %d patterns\n", shuffle_count, total_tested);
    printf("\n");

    /* Algorithm breakdown */
    printf("  Algorithm histogram:\n");
    for (int i = 1; i <= 8; i++) {
        if (algo_count[i] > 0)
            printf("    %-10s : %d\n", ALGO_NAMES[i], algo_count[i]);
    }
    printf("\n");

    /* Action histogram */
    printf("  Action histogram (non-zero):\n");
    for (int i = 0; i < 32; i++) {
        if (action_histogram[i] > 0) {
            char decoded[128];
            decode_action(i, decoded, sizeof(decoded));
            printf("    %s  (count=%d)\n", decoded, action_histogram[i]);
        }
    }
    printf("\n");

    /* Checks */
    CHECK(roundtrip_pass == total_tested,
          "all round-trips correct");
    CHECK(unique_algos >= 2,
          "NN picks at least 2 different algorithms");
    CHECK(unique_actions >= 2,
          "NN picks at least 2 different actions");

    /* Diagnostic flags (not failures, just warnings) */
    if (quant_count == 0) {
        printf("  *** WARNING: NN never chose quantization (actions 8-15, 24-31)\n");
        printf("      Possible causes:\n");
        printf("      - Training data did not include lossy configs as winners\n");
        printf("      - NN learned that quantization overhead > benefit at this data size\n");
        printf("      - error_bound too small for quantization to help\n");
        printf("      - Cost weights (w0,w1,w2) penalize quantization's decomp overhead\n");
        FAIL("NN never chose quantization for any lossy pattern");
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "NN chose quantization for %d/%d patterns", quant_count, total_tested);
        PASS(msg);
    }

    if (shuffle_count == 0) {
        printf("  *** WARNING: NN never chose byte shuffle (actions 16-31)\n");
        printf("      Possible causes:\n");
        printf("      - Training data did not include shuffle configs as winners\n");
        printf("      - Shuffle overhead not justified at this data size (1MB)\n");
        printf("      - NN weights not trained on shuffled variants\n");
        FAIL("NN never chose shuffle for any pattern");
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "NN chose shuffle for %d/%d patterns", shuffle_count, total_tested);
        PASS(msg);
    }

    gpucompress_cleanup();

    printf("\n=== test_nn_action_diversity Result: %d passed, %d failed ===\n",
           g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
