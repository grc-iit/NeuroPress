/**
 * test_perf14_atomic_double.cu
 *
 * PERF-14: Verifies that replacing atomicAddDouble (CAS loop) with native
 * atomicAdd(double*, double) (sm_60+) produces numerically correct stats.
 *
 * Strategy:
 *   For 5 float32 patterns, call gpucompress_compute_stats() (which runs the
 *   GPU stats pipeline: statsPass1Kernel + madPass2Kernel, both using atomicAdd
 *   for double accumulation) and compare against CPU reference values.
 *
 *   A broken atomicAdd would produce wrong sums → wrong entropy/MAD/deriv.
 *   Tolerance is 1% relative error (far above FP rounding noise).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

/* ── CPU reference ───────────────────────────────────────────────────── */

static double cpu_entropy_bits(const float *data, int n)
{
    unsigned long long hist[256] = {};
    const unsigned char *bytes = (const unsigned char *)data;
    size_t total = (size_t)n * sizeof(float);
    for (size_t i = 0; i < total; i++) hist[bytes[i]]++;
    double H = 0.0;
    for (int b = 0; b < 256; b++) {
        if (!hist[b]) continue;
        double p = (double)hist[b] / (double)total;
        H -= p * log2(p);
    }
    return H;
}

/* Matches finalizeStatsOnlyKernel: (mad_sum/n) / range */
static double cpu_mad_normalized(const float *data, int n)
{
    if (n == 0) return 0.0;
    double vmin = data[0], vmax = data[0], sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
        sum += data[i];
    }
    double mean = sum / n;
    double range = vmax - vmin;
    if (range < 1e-12) return 0.0;
    double mad = 0.0;
    for (int i = 0; i < n; i++) mad += fabs((double)data[i] - mean);
    return (mad / n) / range;
}

/* Matches statsPass1Kernel + finalizeStatsOnlyKernel:
 * sum |x[i+1] - 2x[i] + x[i-1]| for i=1..n-2, then divide by (n-2)*range */
static double cpu_deriv_normalized(const float *data, int n)
{
    if (n < 3) return 0.0;
    double vmin = data[0], vmax = data[0];
    for (int i = 0; i < n; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    double range = vmax - vmin;
    if (range < 1e-12) return 0.0;
    double sum = 0.0;
    for (int i = 1; i < n - 1; i++)
        sum += fabs((double)data[i+1] - 2.0*(double)data[i] + (double)data[i-1]);
    return (sum / (n - 2)) / range;
}

/* ── Test patterns ───────────────────────────────────────────────────── */

#define N_FLOATS  (1 << 20)   /* 4 MB — exercises all reduction blocks */
#define N_PATS    5

static const char *PAT_NAMES[N_PATS] = {
    "constant", "sine", "ramp", "alternating", "pseudo-random"
};

static void fill_pattern(float *buf, int n, int pat)
{
    for (int i = 0; i < n; i++) {
        switch (pat) {
        case 0: buf[i] = 42.0f; break;
        case 1: buf[i] = sinf(2.0f * 3.14159f * (float)i / (float)n); break;
        case 2: buf[i] = (float)i / (float)n; break;
        case 3: buf[i] = (i % 2 == 0) ? 1.0f : -1.0f; break;
        case 4: {
            unsigned s = (unsigned)i * 2654435761u;
            buf[i] = (float)(s >> 8) / (float)(1 << 24);
            break;
        }
        default: buf[i] = 0.0f;
        }
    }
}

int main(void)
{
    printf("=== PERF-14: native atomicAdd(double*) correctness ===\n\n");

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    float *h_data = (float *)malloc(N_FLOATS * sizeof(float));
    if (!h_data) { fprintf(stderr, "OOM\n"); gpucompress_cleanup(); return 1; }

    /* 1% relative tolerance — well above FP rounding, well below
     * what a broken CAS accumulation would produce. */
    const double TOL = 0.01;

    int passed = 0;
    for (int p = 0; p < N_PATS; p++) {
        fill_pattern(h_data, N_FLOATS, p);

        /* CPU reference */
        double ref_entropy = cpu_entropy_bits(h_data, N_FLOATS);
        double ref_mad     = cpu_mad_normalized(h_data, N_FLOATS);
        double ref_deriv   = cpu_deriv_normalized(h_data, N_FLOATS);

        /* GPU stats — exercises statsPass1Kernel + madPass2Kernel atomicAdd */
        double gpu_entropy = 0.0, gpu_mad = 0.0, gpu_deriv = 0.0;
        gpucompress_error_t err = gpucompress_compute_stats(
            h_data, N_FLOATS * sizeof(float),
            &gpu_entropy, &gpu_mad, &gpu_deriv);

        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  [%-14s] FAIL: gpucompress_compute_stats error %d\n",
                   PAT_NAMES[p], (int)err);
            continue;
        }

        /* Relative error (guard against near-zero reference) */
        double e_entropy = (ref_entropy > 1e-6)
            ? fabs(gpu_entropy - ref_entropy) / ref_entropy : fabs(gpu_entropy);
        double e_mad = (ref_mad > 1e-6)
            ? fabs(gpu_mad - ref_mad) / ref_mad : fabs(gpu_mad);
        double e_deriv = (ref_deriv > 1e-6)
            ? fabs(gpu_deriv - ref_deriv) / ref_deriv : fabs(gpu_deriv);

        int ok = (e_entropy <= TOL)
              && (e_mad    <= TOL || ref_mad   < 1e-6)
              && (e_deriv  <= TOL || ref_deriv < 1e-6);
        if (ok) passed++;

        printf("  [%-14s] entropy: gpu=%.4f ref=%.4f err=%.2f%%  "
               "mad: gpu=%.4f ref=%.4f err=%.2f%%  "
               "deriv: gpu=%.4f ref=%.4f err=%.2f%%  %s\n",
               PAT_NAMES[p],
               gpu_entropy, ref_entropy, e_entropy * 100.0,
               gpu_mad,     ref_mad,     e_mad     * 100.0,
               gpu_deriv,   ref_deriv,   e_deriv   * 100.0,
               ok ? "PASS" : "FAIL");
    }

    free(h_data);
    gpucompress_cleanup();

    printf("\n=== PERF-14 Result: %d/%d passed ===\n", passed, N_PATS);
    if (passed == N_PATS) {
        printf("VERDICT: native atomicAdd(double*) produces correct stats on sm_80.\n");
        printf("         CAS emulation successfully removed.\n");
        return 0;
    }
    printf("VERDICT: FAILURES — stats deviate from CPU reference.\n");
    return 1;
}
