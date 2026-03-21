/**
 * test_nn_preproc_debug.cu
 *
 * Deep diagnostic: brute-forces all 32 configs at large data sizes (64MB),
 * compares with NN's choice, and inspects compression headers to verify
 * preprocessing was actually applied.
 *
 * Run: ./test_nn_preproc_debug
 */

#include <cuda_runtime.h>
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
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)

static const char* ALGO_NAMES[] = {
    "LZ4", "Snappy", "Deflate", "GDeflate",
    "ZSTD", "ANS", "Cascaded", "Bitcomp"
};

static void decode_action_str(int action, char* buf, size_t len) {
    int algo = action % 8;
    int q = (action / 8) % 2;
    int s = (action / 16) % 2;
    snprintf(buf, len, "%2d:%-8s Q=%d S=%d", action, ALGO_NAMES[algo], q, s);
}

/* ---- Generators ---- */

static void gen_smooth_sine(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = sinf((float)i * 0.001f) * 100.0f;
}

static void gen_random_uniform(float* d, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; i++)
        d[i] = (float)rand() / (float)RAND_MAX * 1000.0f;
}

static void gen_stepped(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (float)((int)(i / 4096)) * 10.0f;
}

static void gen_grayscott(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = (float)(i % 1024) / 1024.0f;
        float y = (float)((i / 1024) % 1024) / 1024.0f;
        d[i] = sinf(x * 6.28f) * cosf(y * 6.28f) * 100.0f;
    }
}

static void gen_high_entropy(float* d, size_t n) {
    unsigned int state = 12345;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        d[i] = (float)(int)state * 1e-5f;
    }
}

static void gen_sparse(float* d, size_t n) {
    for (size_t i = 0; i < n; i++)
        d[i] = (i % 512 == 0) ? 1000.0f : 0.0f;
}

/* Highly structured: repeating byte pattern ideal for shuffle */
static void gen_ieee_structured(float* d, size_t n) {
    /* Values where specific bytes repeat — shuffle should help */
    for (size_t i = 0; i < n; i++) {
        /* Slow-changing exponent, fast-changing mantissa */
        float base = (float)((int)(i / 8192)) * 0.1f;
        d[i] = base + (float)(i % 256) * 1e-6f;
    }
}

/* Turbulence-like: multi-scale with sharp gradients */
static void gen_turbulence(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = (float)i / (float)n;
        d[i] = sinf(x * 6.28f) * 10.0f
             + sinf(x * 62.8f) * 1.0f
             + sinf(x * 628.0f) * 0.1f
             + ((i % 1000 < 5) ? 100.0f : 0.0f);
    }
}

struct DataPattern {
    const char* name;
    void (*gen)(float*, size_t);
    double error_bound;
};

/* ---- Run one config explicitly and measure ---- */
struct ConfigResult {
    float actual_ratio;
    float actual_comp_ms;
    float actual_decomp_ms;
    int succeeded;
};

static ConfigResult run_one_config(float* d_data, size_t data_bytes,
                                   int action, double error_bound,
                                   size_t max_comp)
{
    ConfigResult r = {};
    int algo_idx = action % 8;
    int quant = (action / 8) % 2;
    int shuf = (action / 16) % 2;

    if (quant && error_bound <= 0.0) return r;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = static_cast<gpucompress_algorithm_t>(algo_idx + 1);
    cfg.error_bound = error_bound;
    cfg.preprocessing = 0;
    if (quant) cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
    if (shuf)  cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;

    uint8_t* d_comp = NULL;
    if (cudaMalloc(&d_comp, max_comp) != cudaSuccess) return r;

    size_t comp_size = max_comp;
    gpucompress_stats_t stats = {};

    gpucompress_error_t err = gpucompress_compress_gpu(
        d_data, data_bytes, d_comp, &comp_size, &cfg, &stats, NULL);

    if (err == GPUCOMPRESS_SUCCESS && comp_size > 0) {
        r.actual_ratio = (float)data_bytes / (float)comp_size;
        r.actual_comp_ms = (float)stats.actual_comp_time_ms;
        r.succeeded = 1;

        /* Decompress to measure time */
        float* d_dec = NULL;
        cudaMalloc(&d_dec, data_bytes);
        size_t dec_size = data_bytes;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        gpucompress_decompress_gpu(d_comp, comp_size, d_dec, &dec_size, NULL);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&r.actual_decomp_ms, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_dec);
    }

    cudaFree(d_comp);
    return r;
}

/* ---- Profile all 32 configs ---- */
static void profile_pattern(const char* name, float* d_data, size_t data_bytes,
                            double error_bound)
{
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  %s  |  size=%zuMB  |  eb=%.4f\n", name, data_bytes >> 20, error_bound);
    printf("══════════════════════════════════════════════════════════════════\n");

    size_t max_comp = gpucompress_max_compressed_size(data_bytes);
    float w0 = 1.0f, w1 = 1.0f, w2 = 1.0f, bw = 1e6f;

    /* Brute-force all 32 */
    ConfigResult results[32];
    for (int a = 0; a < 32; a++)
        results[a] = run_one_config(d_data, data_bytes, a, error_bound, max_comp);

    /* Find actual best */
    float best_cost = 1e30f;
    int actual_best = -1;
    for (int a = 0; a < 32; a++) {
        if (!results[a].succeeded) continue;
        ConfigResult& r = results[a];
        float io = (float)data_bytes / (r.actual_ratio * bw);
        float cost = w0 * r.actual_comp_ms + w1 * r.actual_decomp_ms + w2 * io;
        if (cost < best_cost) { best_cost = cost; actual_best = a; }
    }

    /* Now run ALGO_AUTO */
    uint8_t* d_comp = NULL;
    cudaMalloc(&d_comp, max_comp);
    size_t comp_size = max_comp;
    gpucompress_stats_t stats = {};

    gpucompress_config_t auto_cfg = gpucompress_default_config();
    auto_cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
    auto_cfg.error_bound = error_bound;

    gpucompress_error_t err = gpucompress_compress_gpu(
        d_data, data_bytes, d_comp, &comp_size, &auto_cfg, &stats, NULL);

    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  ALGO_AUTO FAILED: err=%d\n", (int)err);
        cudaFree(d_comp);
        FAIL("ALGO_AUTO compress");
        return;
    }

    int nn_action = stats.nn_final_action;
    int nn_orig = stats.nn_original_action;

    /* Read compression header from GPU to verify preprocessing */
    CompressionHeader header;
    cudaMemcpy(&header, d_comp, sizeof(header), cudaMemcpyDeviceToHost);

    int hdr_algo = header.getAlgorithmId();
    int hdr_shuffle = header.shuffle_element_size;
    int hdr_quant = header.hasQuantizationApplied() ? 1 : 0;

    /* Print top 8 by actual cost */
    struct Ranked { int action; float cost; };
    Ranked ranked[32];
    int n_ranked = 0;
    for (int a = 0; a < 32; a++) {
        if (!results[a].succeeded) continue;
        ConfigResult& r = results[a];
        float io = (float)data_bytes / (r.actual_ratio * bw);
        ranked[n_ranked++] = { a, w0 * r.actual_comp_ms + w1 * r.actual_decomp_ms + w2 * io };
    }
    /* Sort by cost */
    for (int i = 0; i < n_ranked - 1; i++)
        for (int j = i + 1; j < n_ranked; j++)
            if (ranked[j].cost < ranked[i].cost) {
                Ranked tmp = ranked[i]; ranked[i] = ranked[j]; ranked[j] = tmp;
            }

    printf("\n  Top 10 by actual cost:\n");
    printf("  %-25s  ratio    ct(ms)   dt(ms)   cost\n", "Config");
    for (int i = 0; i < 10 && i < n_ranked; i++) {
        int a = ranked[i].action;
        ConfigResult& r = results[a];
        char dec[64]; decode_action_str(a, dec, sizeof(dec));
        const char* tag = "";
        if (a == nn_action && a == actual_best) tag = " <<NN+BEST";
        else if (a == nn_action) tag = " <<NN";
        else if (a == actual_best) tag = " <<BEST";
        printf("  %-25s  %6.2fx  %6.2f   %6.2f   %6.2f%s\n",
               dec, r.actual_ratio, r.actual_comp_ms, r.actual_decomp_ms,
               ranked[i].cost, tag);
    }

    /* NN decision */
    char nn_dec[64], best_dec[64];
    decode_action_str(nn_action, nn_dec, sizeof(nn_dec));
    decode_action_str(actual_best, best_dec, sizeof(best_dec));

    printf("\n  NN original action:  %d\n", nn_orig);
    printf("  NN final action:     %s\n", nn_dec);
    printf("  Actual best:         %s\n", best_dec);
    printf("  Predicted ratio:     %.2f\n", stats.predicted_ratio);
    printf("  Actual ratio:        %.2f\n", stats.compression_ratio);
    printf("  Exploration fired:   %s\n", stats.exploration_triggered ? "YES" : "no");
    printf("  SGD fired:           %s\n", stats.sgd_fired ? "YES" : "no");

    /* Header verification */
    printf("\n  === HEADER INSPECTION ===\n");
    printf("  Header algo:         %d (%s)\n", hdr_algo,
           (hdr_algo >= 1 && hdr_algo <= 8) ? ALGO_NAMES[hdr_algo - 1] : "???");
    printf("  Header shuffle:      %d\n", hdr_shuffle);
    printf("  Header quantized:    %d\n", hdr_quant);
    if (hdr_quant) {
        printf("  Header quant_scale:  %.6e\n", header.quant_scale);
        printf("  Header error_bound:  %.6e\n", header.quant_error_bound);
        printf("  Header data_min:     %.6e\n", header.data_min);
        printf("  Header data_max:     %.6e\n", header.data_max);
    }

    /* Cross-check: does header match the action? */
    int nn_algo = nn_action % 8;
    int nn_q = (nn_action / 8) % 2;
    int nn_s = (nn_action / 16) % 2;

    char msg[256];
    snprintf(msg, sizeof(msg), "%s: header algo matches action (hdr=%d, action_algo=%d)",
             name, hdr_algo, nn_algo + 1);
    CHECK(hdr_algo == nn_algo + 1, msg);

    if (nn_q && error_bound > 0.0) {
        snprintf(msg, sizeof(msg), "%s: header shows quantization when action has quant=1", name);
        CHECK(hdr_quant == 1, msg);
    }
    if (nn_s) {
        snprintf(msg, sizeof(msg), "%s: header shows shuffle when action has shuffle=1", name);
        CHECK(hdr_shuffle > 0, msg);
    }

    /* Round-trip verification */
    float* d_dec = NULL;
    cudaMalloc(&d_dec, data_bytes);
    size_t dec_size = data_bytes;
    gpucompress_error_t dec_err = gpucompress_decompress_gpu(
        d_comp, comp_size, d_dec, &dec_size, NULL);

    if (dec_err == GPUCOMPRESS_SUCCESS) {
        size_t N = data_bytes / sizeof(float);
        float* h_orig = (float*)malloc(data_bytes);
        float* h_dec = (float*)malloc(data_bytes);
        cudaMemcpy(h_orig, d_data, data_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dec, d_dec, data_bytes, cudaMemcpyDeviceToHost);

        double max_err = 0;
        for (size_t i = 0; i < N; i++) {
            double diff = fabs((double)h_orig[i] - (double)h_dec[i]);
            if (diff > max_err) max_err = diff;
        }

        if (error_bound > 0.0) {
            snprintf(msg, sizeof(msg), "%s: max_err %.2e <= eb %.2e", name, max_err, error_bound);
            CHECK(max_err <= error_bound, msg);
        } else {
            snprintf(msg, sizeof(msg), "%s: lossless round-trip exact", name);
            CHECK(max_err == 0.0, msg);
        }

        free(h_orig);
        free(h_dec);
    } else {
        snprintf(msg, sizeof(msg), "%s: decompress failed", name);
        FAIL(msg);
    }

    cudaFree(d_dec);
    cudaFree(d_comp);

    /* Quantize / shuffle benefit analysis */
    float best_noq = 1e30f, best_q = 1e30f;
    float best_nos = 1e30f, best_s = 1e30f;
    for (int a = 0; a < 32; a++) {
        if (!results[a].succeeded) continue;
        ConfigResult& r = results[a];
        float io = (float)data_bytes / (r.actual_ratio * bw);
        float cost = w0 * r.actual_comp_ms + w1 * r.actual_decomp_ms + w2 * io;
        if ((a / 8) % 2 == 0) { if (cost < best_noq) best_noq = cost; }
        else { if (cost < best_q) best_q = cost; }
        if ((a / 16) % 2 == 0) { if (cost < best_nos) best_nos = cost; }
        else { if (cost < best_s) best_s = cost; }
    }
    if (error_bound > 0.0) {
        printf("\n  Quant benefit:   no-Q=%.2f  Q=%.2f  (%+.1f%%)\n",
               best_noq, best_q, (best_noq - best_q) / best_noq * 100.0f);
    }
    printf("  Shuffle benefit: no-S=%.2f  S=%.2f  (%+.1f%%)\n",
           best_nos, best_s, (best_nos - best_s) / best_nos * 100.0f);
}

int main(void)
{
    printf("=== test_nn_preproc_debug v2 ===\n");
    printf("64MB data, diverse patterns, header inspection.\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    const char* paths[] = { "neural_net/weights/model.nnwt",
                            "../neural_net/weights/model.nnwt", NULL };
    int loaded = gpucompress_nn_is_loaded();
    for (int i = 0; !loaded && paths[i]; i++) {
        if (gpucompress_load_nn(paths[i]) == GPUCOMPRESS_SUCCESS) {
            loaded = 1; printf("Loaded NN: %s\n", paths[i]);
        }
    }
    if (!loaded) {
        fprintf(stderr, "FATAL: Cannot load NN weights\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Enable online learning + max exploration */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(31);

    /* 64MB data = 16M floats */
    const size_t N = 16 * 1024 * 1024;
    const size_t data_bytes = N * sizeof(float);

    DataPattern patterns[] = {
        { "smooth_sine",       gen_smooth_sine,      0.01   },
        { "random_uniform",    gen_random_uniform,    0.01   },
        { "stepped",           gen_stepped,           0.5    },
        { "grayscott",         gen_grayscott,         0.01   },
        { "high_entropy",      gen_high_entropy,      0.01   },
        { "sparse",            gen_sparse,            0.01   },
        { "ieee_structured",   gen_ieee_structured,   0.001  },
        { "turbulence",        gen_turbulence,        0.1    },
        { "smooth_sine_big_eb",gen_smooth_sine,       1.0    },
        { "stepped_big_eb",    gen_stepped,           5.0    },
        /* Lossless */
        { "smooth_sine_LL",    gen_smooth_sine,       0.0    },
        { "stepped_LL",        gen_stepped,           0.0    },
    };
    int N_PAT = sizeof(patterns) / sizeof(patterns[0]);

    for (int p = 0; p < N_PAT; p++) {
        float* h_data = (float*)malloc(data_bytes);
        patterns[p].gen(h_data, N);

        float* d_data = NULL;
        cudaMalloc(&d_data, data_bytes);
        cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);

        profile_pattern(patterns[p].name, d_data, data_bytes, patterns[p].error_bound);

        cudaFree(d_data);
        free(h_data);
    }

    gpucompress_cleanup();

    printf("\n\n=== test_nn_preproc_debug v2 Result: %d passed, %d failed ===\n",
           g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
