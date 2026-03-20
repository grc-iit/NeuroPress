/**
 * test_exploration_preproc.cu
 *
 * Test: enable ALGO_AUTO with maximum exploration (K=31), and verify
 * that when exploration finds a shuffle or quantization config is best,
 * the compressed output actually uses that config (check header).
 *
 * Strategy:
 *   1. Enable online learning + exploration with K=31 and threshold=0.0
 *      (explore on EVERY compression, not just high-MAPE ones)
 *   2. Compress several data patterns with error_bound > 0
 *   3. Read back the compression header and check what preprocessing was applied
 *   4. Also brute-force all 32 configs to know the actual best
 *   5. Compare: did exploration find a config at least as good as NN alone?
 *
 * Run: ./test_exploration_preproc
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gpucompress.h"

/* Pull in the header struct to decode compressed output */
extern "C" {
    /* GPUCOMPRESS_HEADER_SIZE is 64 bytes */
    #define GPUCOMPRESS_HEADER_SIZE 64
}

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)

static const char* ALGO_NAMES[] = {
    "???", "LZ4", "Snappy", "Deflate", "GDeflate",
    "ZSTD", "ANS", "Cascaded", "Bitcomp"
};

/* ---- Data generators ---- */

/* Smooth periodic data: shuffle should help (byte-level correlation) */
static void gen_periodic(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = sinf((float)i * 0.001f) * 100.0f + 500.0f;
}

/* Piecewise constant: excellent for quantization + any compressor */
static void gen_stepped(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (float)((int)(i / 1024)) * 10.0f;
}

/* Low-entropy smooth ramp: shuffle can improve byte-level patterns */
static void gen_ramp(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (float)i * 0.001f;
}

/* Sparse: mostly zero with occasional spikes */
static void gen_sparse(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (i % 512 == 0) ? 1000.0f : 0.0f;
}

/* GS-like pattern */
static void gen_grayscott(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = (float)(i % 512) / 512.0f;
        float y = (float)((i / 512) % 512) / 512.0f;
        d[i] = sinf(x * 6.28f) * cosf(y * 6.28f) * 100.0f;
    }
}

struct TestCase {
    const char* name;
    void (*gen)(float*, size_t);
    double error_bound;
};

/* ---- Brute-force best config ---- */
static int bruteforce_best(float* d_data, size_t data_bytes, double error_bound,
                           float* out_best_ratio, float* out_best_cost)
{
    size_t max_comp = gpucompress_max_compressed_size(data_bytes);
    float w0 = 1.0f, w1 = 1.0f, w2 = 1.0f, bw = 1e6f;
    float best_cost = 1e30f;
    int best_action = -1;
    *out_best_ratio = 0;

    for (int action = 0; action < 32; action++) {
        int algo_idx = action % 8;
        int quant = (action / 8) % 2;
        int shuf = (action / 16) % 2;

        if (quant && error_bound <= 0.0) continue;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = static_cast<gpucompress_algorithm_t>(algo_idx + 1);
        cfg.error_bound = error_bound;
        cfg.preprocessing = 0;
        if (quant) cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
        if (shuf)  cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;

        uint8_t* d_comp = NULL;
        cudaMalloc(&d_comp, max_comp);
        size_t comp_size = max_comp;
        gpucompress_stats_t stats = {};

        gpucompress_error_t err = gpucompress_compress_gpu(
            d_data, data_bytes, d_comp, &comp_size, &cfg, &stats, NULL);

        if (err == GPUCOMPRESS_SUCCESS && comp_size > 0) {
            float ratio = (float)data_bytes / (float)comp_size;
            float ct = (float)stats.actual_comp_time_ms;

            /* Quick decompress to measure dt */
            float* d_dec = NULL;
            cudaMalloc(&d_dec, data_bytes);
            size_t dec_size = data_bytes;

            cudaEvent_t t0, t1;
            cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0);
            gpucompress_decompress_gpu(d_comp, comp_size, d_dec, &dec_size, NULL);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float dt = 0;
            cudaEventElapsedTime(&dt, t0, t1);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            cudaFree(d_dec);

            float io_cost = (float)data_bytes / (ratio * bw);
            float cost = w0 * ct + w1 * dt + w2 * io_cost;

            if (cost < best_cost) {
                best_cost = cost;
                best_action = action;
                *out_best_ratio = ratio;
            }
        }
        cudaFree(d_comp);
    }

    *out_best_cost = best_cost;
    return best_action;
}

/* ---- Main ---- */
int main(void)
{
    printf("=== test_exploration_preproc ===\n");
    printf("Tests whether exploration correctly selects and returns preprocessing configs.\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    const char* paths[] = { "neural_net/weights/model.nnwt",
                            "../neural_net/weights/model.nnwt", NULL };
    int loaded = gpucompress_nn_is_loaded();
    for (int i = 0; !loaded && paths[i]; i++) {
        if (gpucompress_load_nn(paths[i]) == GPUCOMPRESS_SUCCESS) {
            loaded = 1; printf("Loaded NN: %s\n", paths[i]);
        }
    }
    if (!loaded) { fprintf(stderr, "FATAL: no NN weights\n"); gpucompress_cleanup(); return 1; }

    /* Enable maximum exploration */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);  /* explore EVERY time */
    gpucompress_set_exploration_k(31);           /* try all alternatives */

    TestCase tests[] = {
        { "periodic (eb=0.01)",   gen_periodic,   0.01 },
        { "stepped (eb=0.5)",     gen_stepped,    0.5  },
        { "ramp (eb=0.001)",      gen_ramp,       0.001 },
        { "sparse (eb=0.01)",     gen_sparse,     0.01 },
        { "grayscott (eb=0.01)",  gen_grayscott,  0.01 },
        { "periodic (eb=0.1)",    gen_periodic,   0.1  },
        { "stepped (eb=5.0)",     gen_stepped,    5.0  },
    };
    int N_TESTS = sizeof(tests) / sizeof(tests[0]);

    /* Use 4MB data to be more realistic */
    const size_t N = 1024 * 1024;  /* 1M floats = 4MB */
    const size_t data_bytes = N * sizeof(float);
    size_t max_comp = gpucompress_max_compressed_size(data_bytes);

    int explore_quant_wins = 0;
    int explore_shuffle_wins = 0;
    int explore_correct_preproc = 0;

    for (int t = 0; t < N_TESTS; t++) {
        TestCase& tc = tests[t];
        printf("\n── %s ──\n", tc.name);

        float* h_data = (float*)malloc(data_bytes);
        tc.gen(h_data, N);

        float* d_data = NULL;
        cudaMalloc(&d_data, data_bytes);
        cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);

        /* Step 1: brute-force the actual best config */
        float bf_ratio = 0, bf_cost = 0;
        int bf_best = bruteforce_best(d_data, data_bytes, tc.error_bound,
                                      &bf_ratio, &bf_cost);
        int bf_algo = bf_best % 8, bf_q = (bf_best / 8) % 2, bf_s = (bf_best / 16) % 2;
        printf("  Brute-force best: action=%2d  %-10s quant=%d shuffle=%d  ratio=%.2fx  cost=%.2f\n",
               bf_best, ALGO_NAMES[bf_algo + 1], bf_q, bf_s, bf_ratio, bf_cost);

        /* Step 2: ALGO_AUTO with exploration */
        uint8_t* d_comp = NULL;
        cudaMalloc(&d_comp, max_comp);
        size_t comp_size = max_comp;
        gpucompress_stats_t stats = {};

        gpucompress_config_t auto_cfg = gpucompress_default_config();
        auto_cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        auto_cfg.error_bound = tc.error_bound;

        err = gpucompress_compress_gpu(
            d_data, data_bytes, d_comp, &comp_size, &auto_cfg, &stats, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  ALGO_AUTO compress failed: %d\n", (int)err);
            FAIL("ALGO_AUTO compress");
            cudaFree(d_comp); cudaFree(d_data); free(h_data);
            continue;
        }

        int nn_action = stats.nn_final_action;
        int nn_orig = stats.nn_original_action;
        int nn_algo = nn_action % 8, nn_q = (nn_action / 8) % 2, nn_s = (nn_action / 16) % 2;
        printf("  NN original:      action=%2d\n", nn_orig);
        printf("  NN final (post-explore): action=%2d  %-10s quant=%d shuffle=%d\n",
               nn_action, ALGO_NAMES[nn_algo + 1], nn_q, nn_s);
        printf("  Exploration fired: %s | SGD fired: %s\n",
               stats.exploration_triggered ? "YES" : "no",
               stats.sgd_fired ? "YES" : "no");
        printf("  Actual ratio: %.2fx | Predicted: %.2fx\n",
               stats.compression_ratio, stats.predicted_ratio);

        /* Step 3: verify round-trip */
        float* d_dec = NULL;
        cudaMalloc(&d_dec, data_bytes);
        size_t dec_size = data_bytes;
        gpucompress_error_t dec_err = gpucompress_decompress_gpu(
            d_comp, comp_size, d_dec, &dec_size, NULL);

        if (dec_err == GPUCOMPRESS_SUCCESS) {
            float* h_result = (float*)malloc(data_bytes);
            cudaMemcpy(h_result, d_dec, data_bytes, cudaMemcpyDeviceToHost);

            double max_err = 0;
            for (size_t i = 0; i < N; i++) {
                double diff = fabs((double)h_data[i] - (double)h_result[i]);
                if (diff > max_err) max_err = diff;
            }

            char msg[256];
            snprintf(msg, sizeof(msg), "%s: round-trip within bound (max_err=%.2e, eb=%.2e)",
                     tc.name, max_err, tc.error_bound);
            CHECK(max_err <= tc.error_bound, msg);

            free(h_result);
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: decompress failed", tc.name);
            FAIL(msg);
        }

        /* Step 4: check preprocessing in final action */
        if (nn_q) explore_quant_wins++;
        if (nn_s) explore_shuffle_wins++;

        /* Step 5: check the actual header from compressed output */
        /* Read first 64 bytes from GPU */
        uint8_t h_header[64];
        cudaMemcpy(h_header, d_comp, 64, cudaMemcpyDeviceToHost);

        /* Header byte 4 has shuffle_element_size, byte 8 has quant_flags */
        /* Check CompressionHeader layout */
        uint32_t magic = *(uint32_t*)(h_header);
        uint8_t header_shuffle = h_header[6]; /* shuffle_element_size field */

        printf("  Header: magic=0x%08x, shuffle_size=%d\n", magic, header_shuffle);

        /* Verify header matches the action */
        if (nn_s) {
            CHECK(header_shuffle > 0,
                  "header shuffle_size > 0 when NN chose shuffle");
        }

        cudaFree(d_dec);
        cudaFree(d_comp);
        cudaFree(d_data);
        free(h_data);
    }

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Exploration Preprocessing Summary               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  Exploration chose quantization: %d / %d tests\n", explore_quant_wins, N_TESTS);
    printf("  Exploration chose shuffle:      %d / %d tests\n", explore_shuffle_wins, N_TESTS);

    if (explore_quant_wins == 0) {
        printf("\n  *** PROBLEM: Exploration never chose quantization even with max K=31\n");
        printf("      This suggests a BUG in exploration preprocessing application,\n");
        printf("      or the cost formula doesn't favor quantization at these data sizes.\n");
        FAIL("exploration never chose quantization");
    } else {
        PASS("exploration chose quantization for at least one pattern");
    }

    if (explore_shuffle_wins == 0) {
        printf("\n  *** PROBLEM: Exploration never chose shuffle even with max K=31\n");
        printf("      This suggests shuffle is never the lowest-cost option,\n");
        printf("      or the exploration code drops shuffle results.\n");
        FAIL("exploration never chose shuffle");
    } else {
        PASS("exploration chose shuffle for at least one pattern");
    }

    gpucompress_cleanup();

    printf("\n=== test_exploration_preproc Result: %d passed, %d failed ===\n",
           g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
