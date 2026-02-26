/**
 * @file stats_cpu.cpp
 * @brief CPU-side computation of compression stats (entropy, MAD, 2nd derivative)
 *
 * Replaces the GPU stats pipeline for data that starts on the host.
 * Eliminates GPU kernel launches, workspace alloc/free, and blocking sync.
 *
 * Matches the GPU kernels in stats_kernel.cu and entropy_kernel.cu:
 *   - statsPass1Kernel: sum, min, max, second_derivative_sum
 *   - histogramKernel + entropyFromHistogramKernel: byte-level Shannon entropy
 *   - madPass2Kernel: sum(|x[i] - mean|)
 *   - finalizeStatsOnlyKernel: normalize MAD and derivative by range
 */

#include "stats/stats_cpu.h"

#include <cmath>
#include <cstdint>
#include <cfloat>
#include <cstring>

namespace gpucompress {

int computeStatsCPU(const void* data, size_t size_bytes,
                    double* entropy, double* mad_norm, double* deriv_norm) {
    if (data == nullptr || entropy == nullptr || mad_norm == nullptr ||
        deriv_norm == nullptr || size_bytes == 0) {
        return -1;
    }

    if (size_bytes % sizeof(float) != 0) {
        return -1;
    }

    const float* fdata = static_cast<const float*>(data);
    const size_t n = size_bytes / sizeof(float);

    // ================================================================
    // Pass 1: sum, min, max, second_derivative_sum
    // (mirrors statsPass1Kernel)
    // ================================================================
    double sum = 0.0;
    double deriv_sum = 0.0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (size_t i = 0; i < n; i++) {
        float val = fdata[i];
        sum += static_cast<double>(val);
        if (val < vmin) vmin = val;
        if (val > vmax) vmax = val;

        // Second derivative: |data[i+1] - 2*data[i] + data[i-1]| for 0 < i < n-1
        if (i > 0 && i < n - 1) {
            deriv_sum += fabs(static_cast<double>(fdata[i + 1])
                              - 2.0 * static_cast<double>(val)
                              + static_cast<double>(fdata[i - 1]));
        }
    }

    // ================================================================
    // Entropy: byte-level histogram → Shannon entropy
    // (mirrors histogramKernel + entropyFromHistogramKernel)
    // ================================================================
    unsigned int hist[256];
    memset(hist, 0, sizeof(hist));

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size_bytes; i++) {
        hist[bytes[i]]++;
    }

    double H = 0.0;
    for (int b = 0; b < 256; b++) {
        if (hist[b] > 0) {
            double p = static_cast<double>(hist[b]) / static_cast<double>(size_bytes);
            H -= p * log2(p);
        }
    }
    *entropy = H;

    // ================================================================
    // Pass 2: MAD = sum(|x[i] - mean|)
    // (mirrors madPass2Kernel)
    // ================================================================
    double mean = sum / static_cast<double>(n);
    double mad_sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        mad_sum += fabs(static_cast<double>(fdata[i]) - mean);
    }

    // ================================================================
    // Normalize (mirrors finalizeStatsOnlyKernel)
    // ================================================================
    double range = static_cast<double>(vmax) - static_cast<double>(vmin);

    if (range > 0.0 && n > 0) {
        *mad_norm = (mad_sum / static_cast<double>(n)) / range;
    } else {
        *mad_norm = 0.0;
    }

    if (range > 0.0 && n > 2) {
        *deriv_norm = (deriv_sum / static_cast<double>(n - 2)) / range;
    } else {
        *deriv_norm = 0.0;
    }

    return 0;
}

} // namespace gpucompress
