/**
 * test_sgd_weight_update.cu
 *
 * Verifies that nnSGDKernel actually updates d_nn_weights by checking that
 * repeated training steps cause the NN's predicted_ratio to converge toward
 * the observed actual_ratio.
 *
 * Assertion: over N_ITERS iterations, final |pred-actual| < initial.
 *
 * Setup:
 *   - 4 MB linear ramp data  (OOD for the NN → large initial MAPE → SGD fires every step)
 *   - ALGO_AUTO, lossless, exploration DISABLED (clean signal: one sample per SGD call)
 *   - Reinforcement LR=0.05, MAPE threshold=0.001 (fires whenever MAPE > 0.1%)
 *   - N_ITERS = 30 iterations on the same data
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "gpucompress.h"

#define N_ITERS       30
#define DATA_MB       8
#define REINFORCE_LR  0.05f
#define MAPE_THRESH   0.001f   /* 0.1% — fires on essentially any OOD data */

__global__ static void fill_ramp(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)i / (float)n;
}

int main(int argc, char** argv)
{
    const char* nnwt = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";

    printf("=== SGD Convergence Verification ===\n");
    printf("Data: %d MB ramp, lossless, no exploration\n", DATA_MB);
    printf("SGD:  LR=%.3f, MAPE threshold=%.4f%%\n\n",
           REINFORCE_LR, MAPE_THRESH * 100.0f);

    if (gpucompress_init(nnwt) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    gpucompress_enable_online_learning();
    gpucompress_set_exploration(0);
    gpucompress_set_reinforcement(1, REINFORCE_LR, MAPE_THRESH, MAPE_THRESH);

    size_t data_bytes = (size_t)DATA_MB * 1024 * 1024;
    size_t n_floats   = data_bytes / sizeof(float);

    void* d_input  = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input,  data_bytes);
    cudaMalloc(&d_output, gpucompress_max_compressed_size(data_bytes));
    if (!d_input || !d_output) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }

    int threads = 256, blocks = (int)((n_floats + threads - 1) / threads);
    fill_ramp<<<blocks, threads>>>((float*)d_input, n_floats);
    cudaDeviceSynchronize();

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    printf("%-6s  %-12s  %-12s  %-10s  %-4s\n",
           "Iter", "pred_ratio", "actual_ratio", "MAPE%", "sgd");
    printf("%-6s  %-12s  %-12s  %-10s  %-4s\n",
           "------", "------------", "------------", "----------", "----");

    double pred0 = -1.0, actual0 = -1.0, predN = -1.0, actualN = -1.0;
    int sgd_fires = 0;

    for (int iter = 0; iter < N_ITERS; iter++) {
        size_t out_sz = gpucompress_max_compressed_size(data_bytes);
        gpucompress_stats_t stats = {};
        gpucompress_error_t ce =
            gpucompress_compress_gpu(d_input, data_bytes, d_output, &out_sz,
                                     &cfg, &stats, nullptr);
        if (ce != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "compress failed at iter %d\n", iter);
            break;
        }

        double pred   = stats.predicted_ratio;
        double actual = (out_sz > 0) ? (double)data_bytes / (double)out_sz : 0.0;
        double mape   = (actual > 0.0) ? fabs(pred - actual) / actual * 100.0 : 0.0;

        printf("%-6d  %-12.4f  %-12.4f  %-10.2f  %-4s\n",
               iter, pred, actual, mape, stats.sgd_fired ? "YES" : "no");

        if (stats.sgd_fired) sgd_fires++;
        if (iter == 0)         { pred0 = pred; actual0 = actual; }
        if (iter == N_ITERS-1) { predN = pred; actualN = actual; }
    }

    printf("\n");

    double mape0 = (actual0 > 0.0) ? fabs(pred0 - actual0) / actual0 * 100.0 : -1.0;
    double mapeN = (actualN > 0.0) ? fabs(predN - actualN) / actualN * 100.0 : -1.0;

    printf("=== Convergence Result ===\n");
    printf("  Iter  0: pred=%.4f  actual=%.4f  MAPE=%.2f%%\n", pred0, actual0, mape0);
    printf("  Iter %2d: pred=%.4f  actual=%.4f  MAPE=%.2f%%\n",
           N_ITERS-1, predN, actualN, mapeN);
    printf("  SGD fired: %d / %d iterations\n", sgd_fires, N_ITERS);

    gpucompress_cleanup();
    cudaFree(d_input);
    cudaFree(d_output);

    int pass = (mape0 > 0.0 && mapeN > 0.0 && mapeN < mape0);
    if (pass)
        printf("  PASS — MAPE reduced %.2f%% → %.2f%% (weights updated by SGD)\n", mape0, mapeN);
    else
        printf("  FAIL — MAPE did not decrease (%.2f%% → %.2f%%)\n", mape0, mapeN);

    return pass ? 0 : 1;
}
