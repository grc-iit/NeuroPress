/**
 * test_nn_inference_profile.cu
 *
 * Minimal NN inference profiling harness.
 * Isolates ONLY the stats-kernels + NN-inference path for profiling with:
 *
 *   nsys profile ./test_nn_inference_profile
 *   ncu  --set full ./test_nn_inference_profile
 *   nvprof ./test_nn_inference_profile
 *
 * What it does:
 *   1. Allocates GPU data (configurable size via DATA_MB env var)
 *   2. Fills with deterministic pattern
 *   3. Warmup: runs N_WARMUP compress_gpu calls (ALGO_AUTO) to prime caches
 *   4. Profile region: runs N_PROFILE calls bracketed by CUDA events
 *   5. Reports per-call and aggregate timing
 *
 * Three modes (set PROFILE_MODE env var):
 *   "full"      — full compress_gpu (stats + NN + preproc + compression) [default]
 *   "nn_only"   — compress_gpu with stats output to measure NN path
 *   "baseline"  — compress_gpu with explicit algo (no NN) as reference
 *
 * The full mode is the most useful for nsys since you see every kernel
 * and can identify which part of the pipeline is slow.
 *
 * Usage:
 *   # Default: 64MB data, 100 iterations
 *   ./test_nn_inference_profile
 *
 *   # Custom: 16MB, 50 iterations, baseline mode
 *   DATA_MB=16 N_ITERS=50 PROFILE_MODE=baseline ./test_nn_inference_profile
 *
 *   # With nsys (recommended):
 *   nsys profile -o nn_profile --stats=true ./test_nn_inference_profile
 *
 *   # With ncu (single iteration for kernel detail):
 *   N_ITERS=1 ncu --set full -o nn_kernels ./test_nn_inference_profile
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

/* Defaults — override via env vars */
#define DEFAULT_DATA_MB    64
#define DEFAULT_N_WARMUP   5
#define DEFAULT_N_ITERS    100

static int env_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return v ? atoi(v) : fallback;
}

static const char* env_str(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return (v && v[0]) ? v : fallback;
}

/* Fill GPU buffer with a linear ramp (deterministic, OOD for the NN) */
__global__ static void fill_ramp(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)i / (float)n;
}

/* Fill GPU buffer with sine wave (more realistic distribution) */
__global__ static void fill_sine(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = sinf((float)i * 0.001f) * 100.0f;
}

/* Fill GPU buffer with random-ish data (low compressibility) */
__global__ static void fill_noise(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int x = (unsigned int)i * 1103515245u + 12345u;
        x ^= x >> 16;
        d[i] = (float)(x & 0xFFFF) / 65536.0f;
    }
}

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

int main(void)
{
    int data_mb   = env_int("DATA_MB",   DEFAULT_DATA_MB);
    int n_warmup  = env_int("N_WARMUP",  DEFAULT_N_WARMUP);
    int n_iters   = env_int("N_ITERS",   DEFAULT_N_ITERS);
    const char* mode = env_str("PROFILE_MODE", "full");
    const char* pattern = env_str("DATA_PATTERN", "ramp");

    printf("=== NN Inference Profiling Harness ===\n");
    printf("  DATA_MB=%d  N_WARMUP=%d  N_ITERS=%d\n", data_mb, n_warmup, n_iters);
    printf("  PROFILE_MODE=%s  DATA_PATTERN=%s\n\n", mode, pattern);

    /* ── Init ── */
    const char* wpath = find_weights();
    if (gpucompress_init(wpath) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    /* ── Allocate ── */
    size_t data_bytes = (size_t)data_mb * 1024 * 1024;
    size_t n_floats   = data_bytes / sizeof(float);
    size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

    void* d_input  = NULL;
    void* d_output = NULL;
    cudaMalloc(&d_input,  data_bytes);
    cudaMalloc(&d_output, max_comp);
    if (!d_input || !d_output) {
        fprintf(stderr, "cudaMalloc failed (need %zu + %zu bytes)\n", data_bytes, max_comp);
        return 1;
    }

    /* ── Fill data ── */
    int threads = 256;
    int blocks  = (int)((n_floats + threads - 1) / threads);
    if (strcmp(pattern, "sine") == 0)
        fill_sine<<<blocks, threads>>>((float*)d_input, n_floats);
    else if (strcmp(pattern, "noise") == 0)
        fill_noise<<<blocks, threads>>>((float*)d_input, n_floats);
    else
        fill_ramp<<<blocks, threads>>>((float*)d_input, n_floats);
    cudaDeviceSynchronize();

    /* ── Configure ── */
    gpucompress_config_t cfg = gpucompress_default_config();

    int use_nn = 1;
    if (strcmp(mode, "baseline") == 0) {
        /* Bypass NN entirely — use explicit LZ4 */
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        use_nn = 0;
        printf("  Mode: BASELINE (LZ4, no NN)\n");
    } else {
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = 0.0;  /* lossless */
        printf("  Mode: %s (ALGO_AUTO, lossless)\n", mode);
    }
    printf("\n");

    /* ── CUDA events for precise timing ── */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* ── Warmup ── */
    printf("[Warmup: %d iterations]\n", n_warmup);
    for (int i = 0; i < n_warmup; i++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};
        gpucompress_error_t err = gpucompress_compress_gpu(
            d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);
        if (err != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "  warmup[%d] failed: err=%d\n", i, (int)err);
            break;
        }
        printf("  warmup[%d]: ratio=%.2f pred=%.2f action=%d\n",
               i, stats.compression_ratio, stats.predicted_ratio,
               gpucompress_get_last_nn_action());
    }
    cudaStreamSynchronize(stream);
    printf("\n");

    /* ── Profile region ── */
    printf("[Profile: %d iterations]\n", n_iters);
    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
           "Iter", "ratio", "pred", "wall_ms", "event_ms", "action");
    printf("------  ----------  ----------  ----------  ----------  --------\n");

    float total_event_ms = 0.0f;
    float min_event_ms = 1e9f;
    float max_event_ms = 0.0f;

    /* Per-component timing from diag */
    float total_nn_ms      = 0.0f;
    float total_stats_ms   = 0.0f;
    float total_preproc_ms = 0.0f;
    float total_comp_ms    = 0.0f;

    for (int i = 0; i < n_iters; i++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};

        /* Event-based timing: bracket the entire compress_gpu call */
        cudaEventRecord(ev_start, stream);

        gpucompress_error_t err = gpucompress_compress_gpu(
            d_input, data_bytes, d_output, &out_sz, &cfg, &stats, stream);

        cudaEventRecord(ev_stop, stream);
        cudaEventSynchronize(ev_stop);

        float event_ms = 0.0f;
        cudaEventElapsedTime(&event_ms, ev_start, ev_stop);

        if (err != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "  iter[%d] FAILED: err=%d\n", i, (int)err);
            continue;
        }

        /* Get per-chunk diag for component breakdown */
        gpucompress_chunk_diag_t diag = {};
        gpucompress_get_chunk_diag(0, &diag);

        total_event_ms += event_ms;
        if (event_ms < min_event_ms) min_event_ms = event_ms;
        if (event_ms > max_event_ms) max_event_ms = event_ms;

        total_nn_ms      += diag.nn_inference_ms;
        total_stats_ms   += diag.stats_ms;
        total_preproc_ms += diag.preprocessing_ms;
        total_comp_ms    += diag.compression_ms;

        int action = use_nn ? gpucompress_get_last_nn_action() : -1;

        /* Print every 10th iteration + first and last */
        if (i < 3 || i == n_iters - 1 || (i % 10) == 0) {
            printf("%-6d  %-10.2f  %-10.2f  %-10.3f  %-10.3f  %-8d\n",
                   i, stats.compression_ratio, stats.predicted_ratio,
                   event_ms, event_ms, action);
        }

        gpucompress_reset_chunk_history();
    }

    printf("\n");

    /* ── Summary ── */
    float avg_event_ms = total_event_ms / n_iters;
    float avg_nn_ms    = total_nn_ms / n_iters;
    float avg_stats_ms = total_stats_ms / n_iters;
    float avg_prep_ms  = total_preproc_ms / n_iters;
    float avg_comp_ms  = total_comp_ms / n_iters;

    printf("=== Timing Summary (%d iterations, %d MB) ===\n", n_iters, data_mb);
    printf("  Total wall time : %.1f ms\n", total_event_ms);
    printf("  Avg per call    : %.3f ms\n", avg_event_ms);
    printf("  Min per call    : %.3f ms\n", min_event_ms);
    printf("  Max per call    : %.3f ms\n", max_event_ms);
    printf("\n");
    printf("  Component breakdown (avg per call):\n");
    printf("    Stats compute : %.3f ms  (%.1f%%)\n",
           avg_stats_ms, avg_stats_ms / avg_event_ms * 100.0f);
    printf("    NN inference  : %.3f ms  (%.1f%%)\n",
           avg_nn_ms, avg_nn_ms / avg_event_ms * 100.0f);
    printf("    Preprocessing : %.3f ms  (%.1f%%)\n",
           avg_prep_ms, avg_prep_ms / avg_event_ms * 100.0f);
    printf("    Compression   : %.3f ms  (%.1f%%)\n",
           avg_comp_ms, avg_comp_ms / avg_event_ms * 100.0f);
    printf("    Other/overhead: %.3f ms  (%.1f%%)\n",
           avg_event_ms - avg_stats_ms - avg_nn_ms - avg_prep_ms - avg_comp_ms,
           (avg_event_ms - avg_stats_ms - avg_nn_ms - avg_prep_ms - avg_comp_ms) / avg_event_ms * 100.0f);
    printf("\n");

    /* ── Throughput ── */
    float throughput_gbps = (float)data_bytes / (avg_event_ms * 1e6f);
    float nn_only_gbps = (avg_nn_ms > 0) ?
        (float)data_bytes / (avg_nn_ms * 1e6f) : 0.0f;

    printf("  Throughput (full): %.2f GB/s\n", throughput_gbps);
    printf("  Throughput (NN-limited): %.2f GB/s\n", nn_only_gbps);
    printf("\n");

    /* ── Profiler markers hint ── */
    printf("=== Profiling Tips ===\n");
    printf("  nsys profile -o nn_profile --stats=true ./test_nn_inference_profile\n");
    printf("  nsys stats nn_profile.nsys-rep  # kernel summary\n");
    printf("  ncu --set full -o nn_kernels N_ITERS=1 ./test_nn_inference_profile\n");
    printf("\n");
    printf("  Look for:\n");
    printf("    - cudaStreamSynchronize calls (nsys timeline → CUDA API row)\n");
    printf("    - D→H / H→D copies between kernels (nsys → Memory row)\n");
    printf("    - GPU idle gaps between kernel launches\n");
    printf("    - statsPass1/entropy/MAD/finalize kernel durations\n");
    printf("    - nnFusedInferenceKernel duration vs total NN time\n");

    /* ── Cleanup ── */
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    gpucompress_cleanup();

    return 0;
}
