/**
 * test_nn_timing_inflation.cu
 *
 * Regression test for NN timing inflation bug.
 *
 * The nn_inference_ms diagnostic field was inflated because it used
 * CPU wall-clock (std::chrono) over a window that included stats kernels,
 * stats D->H copies, mutex acquisition, NN kernel, and stream sync.
 *
 * After the fix, nn_inference_ms uses CUDA events bracketing only the
 * NN kernel + result D->H copy, and a separate stats_ms field captures
 * the stats computation.
 *
 * Test logic:
 *   1. Init gpucompress with NN weights
 *   2. Allocate 64 MB GPU data, fill with deterministic ramp
 *   3. Run 5 warmup iterations via gpucompress_compress_gpu(ALGO_AUTO)
 *   4. Run 20 measured iterations, collect per-chunk diagnostics
 *   5. Assert avg_nn_ms / avg_comp_ms < 0.50
 *      (Before fix this ratio was 0.8-1.5+; after fix, sub-millisecond NN
 *       should be a small fraction of compression time)
 *   6. Assert stats_ms > 0 (new field is populated)
 *   7. Assert nn_inference_ms >= 0 for all chunks
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "gpucompress.h"

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

static __global__ void fill_ramp(float* d, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n)
        d[idx] = (float)(idx & 0xFFFF) / 65536.0f;
}

int main(void)
{
    const int DATA_MB   = 64;
    const int N_WARMUP  = 5;
    const int N_MEASURE = 20;

    printf("=== NN Timing Inflation Regression Test ===\n");
    printf("  DATA_MB=%d  N_WARMUP=%d  N_MEASURE=%d\n\n", DATA_MB, N_WARMUP, N_MEASURE);

    /* Init */
    const char* wpath = find_weights();
    if (gpucompress_init(wpath) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed — skipping test\n");
        return 1;
    }

    /* Allocate */
    size_t data_bytes = (size_t)DATA_MB * 1024 * 1024;
    size_t n_floats   = data_bytes / sizeof(float);
    size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

    void* d_input  = NULL;
    void* d_output = NULL;
    cudaMalloc(&d_input,  data_bytes);
    cudaMalloc(&d_output, max_comp);
    if (!d_input || !d_output) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }

    /* Fill with deterministic ramp pattern */
    int threads = 256;
    int blocks  = (int)((n_floats + threads - 1) / threads);
    fill_ramp<<<blocks, threads>>>((float*)d_input, n_floats);
    cudaDeviceSynchronize();

    /* Config: ALGO_AUTO with lossless */
    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm   = GPUCOMPRESS_ALGO_AUTO;
    cfg.error_bound = 0.0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    printf("[Warmup: %d iterations]\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; i++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};
        gpucompress_reset_chunk_history();
        gpucompress_compress_gpu(d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);
    }
    cudaStreamSynchronize(stream);

    /* Measured iterations */
    printf("[Measuring: %d iterations]\n", N_MEASURE);
    float total_nn_ms   = 0.0f;
    float total_stats_ms = 0.0f;
    float total_comp_ms  = 0.0f;
    int   count          = 0;
    int   failures       = 0;

    for (int i = 0; i < N_MEASURE; i++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};
        gpucompress_reset_chunk_history();

        gpucompress_error_t err = gpucompress_compress_gpu(
            d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);
        if (err != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "  iter[%d] compress failed: err=%d\n", i, (int)err);
            failures++;
            continue;
        }

        gpucompress_chunk_diag_t diag = {};
        if (gpucompress_get_chunk_diag(0, &diag) != 0) {
            fprintf(stderr, "  iter[%d] get_chunk_diag failed\n", i);
            failures++;
            continue;
        }

        /* Sanity: nn_inference_ms must be non-negative */
        if (diag.nn_inference_ms < 0.0f) {
            fprintf(stderr, "  FAIL: iter[%d] nn_inference_ms=%.6f is negative\n",
                    i, diag.nn_inference_ms);
            failures++;
        }

        total_nn_ms   += diag.nn_inference_ms;
        total_stats_ms += diag.stats_ms;
        total_comp_ms  += diag.compression_ms;
        count++;

        if (i < 3 || i == N_MEASURE - 1) {
            printf("  iter[%2d]: nn=%.3f ms  stats=%.3f ms  comp=%.3f ms\n",
                   i, diag.nn_inference_ms, diag.stats_ms, diag.compression_ms);
        }
    }

    if (count == 0) {
        fprintf(stderr, "FAIL: No successful iterations\n");
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        gpucompress_cleanup();
        return 1;
    }

    float avg_nn_ms   = total_nn_ms / count;
    float avg_stats_ms = total_stats_ms / count;
    float avg_comp_ms  = total_comp_ms / count;

    printf("\n=== Results (%d successful iterations) ===\n", count);
    printf("  avg nn_inference_ms : %.3f ms\n", avg_nn_ms);
    printf("  avg stats_ms        : %.3f ms\n", avg_stats_ms);
    printf("  avg compression_ms  : %.3f ms\n", avg_comp_ms);

    float ratio = (avg_comp_ms > 0) ? avg_nn_ms / avg_comp_ms : 999.0f;
    printf("  nn/comp ratio       : %.3f\n", ratio);

    /* Assertions */
    int pass = 1;

    /* Primary assertion: NN should not exceed 50% of compression time */
    if (ratio >= 0.50f) {
        fprintf(stderr, "  FAIL: nn/comp ratio %.3f >= 0.50 — NN timing still inflated\n", ratio);
        pass = 0;
    } else {
        printf("  PASS: nn/comp ratio %.3f < 0.50\n", ratio);
    }

    /* stats_ms should be populated (> 0) */
    if (avg_stats_ms <= 0.0f) {
        fprintf(stderr, "  FAIL: avg stats_ms=%.6f — stats_ms field not populated\n", avg_stats_ms);
        pass = 0;
    } else {
        printf("  PASS: stats_ms > 0 (%.3f ms)\n", avg_stats_ms);
    }

    /* nn_inference_ms > 0 when ALGO_AUTO is used */
    if (avg_nn_ms <= 0.0f) {
        fprintf(stderr, "  FAIL: avg nn_inference_ms=%.6f — should be > 0 for ALGO_AUTO\n", avg_nn_ms);
        pass = 0;
    } else {
        printf("  PASS: nn_inference_ms > 0 (%.3f ms)\n", avg_nn_ms);
    }

    if (failures > 0) {
        fprintf(stderr, "  FAIL: %d iteration failures\n", failures);
        pass = 0;
    }

    printf("\nVERDICT: %s\n", pass ? "PASS" : "FAIL");

    /* Cleanup */
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    gpucompress_cleanup();

    return pass ? 0 : 1;
}
