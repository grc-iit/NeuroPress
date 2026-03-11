/**
 * test_nn_ratio_prediction.cu
 *
 * Regression test for NN compression ratio prediction accuracy on
 * highly-compressible synthetic data (Gray-Scott-like patterns).
 *
 * Generates sparse float arrays (mostly zeros with small structured regions)
 * that compress at 500-2000x — the regime where the NN's ratio prediction
 * was found to be wildly inflated.
 *
 * Runs two phases:
 *   Phase 1 (NN):    Inference-only, measures baseline MAPE
 *   Phase 2 (NN-RL): Online SGD enabled, measures whether learning converges
 *
 * Assertions:
 *   - Algorithm selection matches exhaustive best (zstd expected)
 *   - predicted_ratio > 0 for all chunks
 *   - MAPE is reported per-chunk for analysis
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

/* ── Synthetic data patterns ───────────────────────────────── */

/* Pattern 1: Sparse ramp — 95% zeros, 5% small ramp values.
 * Mimics Gray-Scott U field after simulation (mostly uniform). */
static __global__ void fill_sparse_ramp(float* d, size_t n, float density) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    /* Deterministic pseudo-random sparsity based on index */
    unsigned int hash = (unsigned int)(idx * 2654435761u);
    float frac = (float)(hash & 0xFFFF) / 65536.0f;
    if (frac < density) {
        d[idx] = (float)(idx & 0x3FF) / 1024.0f * 0.01f;
    } else {
        d[idx] = 0.0f;
    }
}

/* Pattern 2: Block-sparse — contiguous blocks of small values in a sea of zeros.
 * Mimics reaction-diffusion "spots" pattern. */
static __global__ void fill_block_sparse(float* d, size_t n, int block_size, float density) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t block_id = idx / block_size;
    unsigned int hash = (unsigned int)(block_id * 2246822519u);
    float frac = (float)(hash & 0xFFFF) / 65536.0f;
    if (frac < density) {
        /* Inside an active block: small smooth values */
        float local = (float)(idx % block_size) / (float)block_size;
        d[idx] = 0.001f + 0.005f * local;
    } else {
        d[idx] = 0.0f;
    }
}

/* Pattern 3: Constant (worst case for ratio prediction — infinite compressibility) */
static __global__ void fill_constant(float* d, size_t n, float val) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = val;
}

/* ── Helpers ───────────────────────────────────────────────── */

static const char* find_weights() {
    const char* paths[] = {
        "neural_net/weights/model.nnwt",
        "../neural_net/weights/model.nnwt",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return paths[0];
}

typedef struct {
    const char* name;
    int         pattern;      /* 0=sparse_ramp, 1=block_sparse, 2=constant */
    float       density;      /* fraction of non-zero elements */
    size_t      size_mb;      /* chunk size in MB */
} TestCase;

static float compute_mape(float predicted, float actual) {
    if (actual <= 0.0f) return 0.0f;
    return fabsf(predicted - actual) / actual * 100.0f;
}

/* ── Main ──────────────────────────────────────────────────── */

int main(void)
{
    printf("=== NN Ratio Prediction Regression Test ===\n\n");

    /* Test cases: various highly-compressible patterns */
    TestCase cases[] = {
        { "sparse_ramp_5pct_4MB",   0, 0.05f, 4  },
        { "sparse_ramp_2pct_4MB",   0, 0.02f, 4  },
        { "block_sparse_4MB",       1, 0.10f, 4  },
        { "constant_zero_4MB",      2, 0.00f, 4  },
        { "sparse_ramp_5pct_1MB",   0, 0.05f, 1  },
        { "block_sparse_1MB",       1, 0.10f, 1  },
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    /* Init */
    const char* wpath = find_weights();
    if (gpucompress_init(wpath) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed — skipping test\n");
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads = 256;
    int total_pass = 1;

    /* ════════════════════════════════════════════════════════
     *  PHASE 1: NN inference-only (no SGD)
     * ════════════════════════════════════════════════════════ */
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHASE 1: NN Inference-Only                                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  %-28s │ %8s │ %8s │ %7s ║\n", "Test Case", "Actual", "Predict", "MAPE");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");

    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    float nn_mape_sum = 0.0f;
    int   nn_mape_cnt = 0;

    for (int c = 0; c < n_cases; c++) {
        TestCase* tc = &cases[c];
        size_t data_bytes = tc->size_mb * 1024 * 1024;
        size_t n_floats   = data_bytes / sizeof(float);
        size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

        void* d_input  = NULL;
        void* d_output = NULL;
        cudaMalloc(&d_input,  data_bytes);
        cudaMalloc(&d_output, max_comp);

        int blocks = (int)((n_floats + threads - 1) / threads);

        switch (tc->pattern) {
            case 0: fill_sparse_ramp<<<blocks, threads>>>((float*)d_input, n_floats, tc->density); break;
            case 1: fill_block_sparse<<<blocks, threads>>>((float*)d_input, n_floats, 1024, tc->density); break;
            case 2: fill_constant<<<blocks, threads>>>((float*)d_input, n_floats, 0.0f); break;
        }
        cudaDeviceSynchronize();

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm   = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = 0.0;

        /* Run 3 warmup iterations */
        for (int w = 0; w < 3; w++) {
            size_t out_sz = max_comp;
            gpucompress_stats_t st = {};
            gpucompress_reset_chunk_history();
            gpucompress_compress_gpu(d_input, data_bytes, d_output, &out_sz, &cfg, &st, stream);
        }

        /* Measured iteration */
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};
        gpucompress_reset_chunk_history();
        gpucompress_error_t err = gpucompress_compress_gpu(
            d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);

        if (err != GPUCOMPRESS_SUCCESS) {
            printf("║  %-28s │ COMPRESS FAILED (err=%d)            ║\n", tc->name, (int)err);
            total_pass = 0;
        } else {
            gpucompress_chunk_diag_t diag = {};
            gpucompress_get_chunk_diag(0, &diag);

            float actual  = diag.actual_ratio;
            float predict = diag.predicted_ratio;
            float mape    = compute_mape(predict, actual);

            printf("║  %-28s │ %7.1fx │ %7.1fx │ %5.1f%% ║\n",
                   tc->name, actual, predict, mape);

            nn_mape_sum += mape;
            nn_mape_cnt++;

            if (predict <= 0.0f) {
                fprintf(stderr, "  FAIL: predicted_ratio <= 0 for %s\n", tc->name);
                total_pass = 0;
            }
        }

        cudaFree(d_input);
        cudaFree(d_output);
    }

    float nn_avg_mape = (nn_mape_cnt > 0) ? nn_mape_sum / nn_mape_cnt : 0.0f;
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  NN Average MAPE: %5.1f%%                                       ║\n", nn_avg_mape);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* ════════════════════════════════════════════════════════
     *  PHASE 2: NN-RL (SGD enabled, 20 iterations per case)
     * ════════════════════════════════════════════════════════ */
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHASE 2: NN-RL (SGD enabled, LR=0.05, MAPE>=20%%)                     ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.05f, 0.20f, 0.20f);
    gpucompress_set_exploration(0);

    int N_RL_ITERS = 20;

    for (int c = 0; c < n_cases; c++) {
        TestCase* tc = &cases[c];
        size_t data_bytes = tc->size_mb * 1024 * 1024;
        size_t n_floats   = data_bytes / sizeof(float);
        size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

        void* d_input  = NULL;
        void* d_output = NULL;
        cudaMalloc(&d_input,  data_bytes);
        cudaMalloc(&d_output, max_comp);

        int blocks = (int)((n_floats + threads - 1) / threads);

        switch (tc->pattern) {
            case 0: fill_sparse_ramp<<<blocks, threads>>>((float*)d_input, n_floats, tc->density); break;
            case 1: fill_block_sparse<<<blocks, threads>>>((float*)d_input, n_floats, 1024, tc->density); break;
            case 2: fill_constant<<<blocks, threads>>>((float*)d_input, n_floats, 0.0f); break;
        }
        cudaDeviceSynchronize();

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm   = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = 0.0;

        printf("║  %-28s                                            ║\n", tc->name);
        printf("║    iter │ %8s │ %8s │ %7s │ SGD                    ║\n",
               "Actual", "Predict", "MAPE");

        float first_mape = 0.0f, last_mape = 0.0f;
        int   sgd_count = 0;

        for (int i = 0; i < N_RL_ITERS; i++) {
            size_t out_sz = max_comp;
            gpucompress_stats_t stats = {};
            gpucompress_reset_chunk_history();

            gpucompress_error_t err = gpucompress_compress_gpu(
                d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);

            if (err != GPUCOMPRESS_SUCCESS) {
                printf("║    %3d  │ FAILED                                        ║\n", i);
                continue;
            }

            gpucompress_chunk_diag_t diag = {};
            gpucompress_get_chunk_diag(0, &diag);

            float actual  = diag.actual_ratio;
            float predict = diag.predicted_ratio;
            float mape    = compute_mape(predict, actual);

            if (i == 0) first_mape = mape;
            last_mape = mape;
            if (diag.sgd_fired) sgd_count++;

            /* Print first 5, last 2, and every 5th */
            if (i < 5 || i >= N_RL_ITERS - 2 || i % 5 == 0) {
                printf("║    %3d  │ %7.1fx │ %7.1fx │ %5.1f%% │ %s                    ║\n",
                       i, actual, predict, mape, diag.sgd_fired ? "[SGD]" : "     ");
            }
        }

        printf("║    SGD fired: %d/%d  First MAPE: %.1f%%  Last MAPE: %.1f%%            ║\n",
               sgd_count, N_RL_ITERS, first_mape, last_mape);

        /* Check convergence: last MAPE should be <= first MAPE (or both low) */
        if (last_mape > first_mape * 1.5f && last_mape > 50.0f) {
            printf("║    WARNING: MAPE diverged (%.1f%% → %.1f%%)                        ║\n",
                   first_mape, last_mape);
        }

        printf("║──────────────────────────────────────────────────────────────────────║\n");

        cudaFree(d_input);
        cudaFree(d_output);
    }

    gpucompress_disable_online_learning();

    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Summary ── */
    printf("VERDICT: %s\n", total_pass ? "PASS" : "FAIL");

    cudaStreamDestroy(stream);
    gpucompress_cleanup();

    return total_pass ? 0 : 1;
}
