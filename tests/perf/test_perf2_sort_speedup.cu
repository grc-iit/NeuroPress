/**
 * test_perf2_sort_speedup.cu
 *
 * PERF-2: Measures speedup of parallel bitonic sort (all 32 threads) vs
 * serial insertion sort (thread 0 only, 31 threads idle).
 *
 * Both kernels sort 32 float/int pairs in descending order and write the
 * sorted indices to global memory — matching the real workload in
 * nnInferenceKernel / nnFusedInferenceKernel.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 32
#define ITERS 100000

/* ── Insertion sort kernel (old code: thread 0 only) ─────────────────── */
__global__ void kernel_insertion_sort(const float *d_vals_in,
                                      const int   *d_idxs_in,
                                      int         *d_out,
                                      int          iters)
{
    __shared__ float s_vals[N];
    __shared__ int   s_idxs[N];
    int tid = threadIdx.x;

    for (int it = 0; it < iters; it++) {
        s_vals[tid] = d_vals_in[tid];
        s_idxs[tid] = d_idxs_in[tid];
        __syncthreads();

        if (tid == 0) {
            float lv[N]; int li[N];
            for (int i = 0; i < N; i++) { lv[i] = s_vals[i]; li[i] = s_idxs[i]; }
            for (int i = 1; i < N; i++) {
                float kv = lv[i]; int ki = li[i]; int j = i - 1;
                while (j >= 0 && lv[j] < kv) { lv[j+1]=lv[j]; li[j+1]=li[j]; j--; }
                lv[j+1] = kv; li[j+1] = ki;
            }
            for (int i = 0; i < N; i++) d_out[i] = li[i];
        }
        __syncthreads();
    }
}

/* ── Bitonic sort kernel (new code: all 32 threads) ──────────────────── */
__global__ void kernel_bitonic_sort(const float *d_vals_in,
                                    const int   *d_idxs_in,
                                    int         *d_out,
                                    int          iters)
{
    __shared__ float s_vals[N];
    __shared__ int   s_idxs[N];
    int tid = threadIdx.x;

    for (int it = 0; it < iters; it++) {
        s_vals[tid] = d_vals_in[tid];
        s_idxs[tid] = d_idxs_in[tid];
        __syncthreads();

        float my_val = s_vals[tid];
        int   my_idx = s_idxs[tid];

        for (int k = 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j >= 1; j >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFFu, my_val, j);
                int   other_idx = __shfl_xor_sync(0xFFFFFFFFu, my_idx, j);
                int   ixj = tid ^ j;
                if (ixj > tid) {
                    if (((tid & k) == 0 && my_val < other_val) ||
                        ((tid & k) != 0 && my_val > other_val)) {
                        my_val = other_val;
                        my_idx = other_idx;
                    }
                }
            }
        }
        d_out[tid] = my_idx;
    }
}

/* ── Correctness check ───────────────────────────────────────────────── */
static int check_sorted_descending(const float *vals, const int *sorted_idxs)
{
    for (int i = 0; i < N - 1; i++) {
        if (vals[sorted_idxs[i]] < vals[sorted_idxs[i + 1]]) {
            printf("  FAIL at position %d: vals[%d]=%.3f < vals[%d]=%.3f\n",
                   i, sorted_idxs[i], vals[sorted_idxs[i]],
                   sorted_idxs[i+1], vals[sorted_idxs[i+1]]);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    printf("=== PERF-2: bitonic sort speedup vs insertion sort ===\n\n");

    /* Setup: 32 pseudo-random values */
    float h_vals[N]; int h_idxs[N];
    for (int i = 0; i < N; i++) {
        unsigned s = (unsigned)i * 2654435761u;
        h_vals[i] = (float)(s >> 8) / (float)(1 << 24);
        h_idxs[i] = i;
    }

    float *d_vals; int *d_idxs, *d_out;
    cudaMalloc(&d_vals, N * sizeof(float));
    cudaMalloc(&d_idxs, N * sizeof(int));
    cudaMalloc(&d_out,  N * sizeof(int));
    cudaMemcpy(d_vals, h_vals, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idxs, h_idxs, N * sizeof(int),  cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    int h_out[N];
    float ms_insert, ms_bitonic;

    /* Warm up */
    kernel_insertion_sort<<<1, N>>>(d_vals, d_idxs, d_out, 100);
    kernel_bitonic_sort  <<<1, N>>>(d_vals, d_idxs, d_out, 100);
    cudaDeviceSynchronize();

    /* ── Insertion sort timing ── */
    cudaEventRecord(t0);
    kernel_insertion_sort<<<1, N>>>(d_vals, d_idxs, d_out, ITERS);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_insert, t0, t1);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    int ok_insert = check_sorted_descending(h_vals, h_out);

    /* ── Bitonic sort timing ── */
    cudaEventRecord(t0);
    kernel_bitonic_sort<<<1, N>>>(d_vals, d_idxs, d_out, ITERS);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_bitonic, t0, t1);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    int ok_bitonic = check_sorted_descending(h_vals, h_out);

    /* ── Report ── */
    double us_insert  = ms_insert  * 1000.0 / ITERS;
    double us_bitonic = ms_bitonic * 1000.0 / ITERS;
    double speedup    = us_insert / us_bitonic;

    printf("  Insertion sort (thread 0): %.3f us/iter  [%s]\n",
           us_insert,  ok_insert  ? "correct" : "WRONG");
    printf("  Bitonic sort  (32 threads): %.3f us/iter  [%s]\n",
           us_bitonic, ok_bitonic ? "correct" : "WRONG");
    printf("\n  Speedup: %.2fx\n\n", speedup);

    if (speedup >= 1.0)
        printf("VERDICT: bitonic sort is %.2fx faster — PERF-2 fix confirmed.\n", speedup);
    else
        printf("VERDICT: bitonic sort is %.2fx slower (unexpected on this GPU).\n", 1.0/speedup);

    cudaFree(d_vals); cudaFree(d_idxs); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return !(ok_insert && ok_bitonic);
}
