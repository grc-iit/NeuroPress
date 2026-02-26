#ifndef GPUCOMPRESS_STATS_CPU_H
#define GPUCOMPRESS_STATS_CPU_H

#include <cstddef>

namespace gpucompress {

/**
 * Compute entropy, MAD, and second-derivative stats on CPU from host memory.
 *
 * Replicates the GPU stats pipeline (statsPass1 + histogram/entropy +
 * madPass2 + finalize) entirely on the CPU, avoiding GPU kernel launches,
 * workspace allocation, and the blocking D→H sync.
 *
 * @param data           Host pointer to input data (interpreted as float[])
 * @param size_bytes     Size of data in bytes (must be multiple of sizeof(float))
 * @param entropy        [out] Shannon entropy of the byte stream (bits, 0-8)
 * @param mad_norm       [out] Normalized mean absolute deviation
 * @param deriv_norm     [out] Normalized mean second derivative
 * @return 0 on success, -1 on error
 */
int computeStatsCPU(const void* data, size_t size_bytes,
                    double* entropy, double* mad_norm, double* deriv_norm);

} // namespace gpucompress

#endif // GPUCOMPRESS_STATS_CPU_H
