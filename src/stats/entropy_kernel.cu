/**
 * @file entropy_kernel.cu
 * @brief GPU-accelerated Shannon entropy calculation
 *
 * Computes byte-level Shannon entropy using histogram-based approach.
 * Used for RL-based compression algorithm selection.
 *
 * Algorithm:
 *   1. Build histogram of byte values (256 bins) using shared memory atomics
 *   2. Reduce per-block histograms to global histogram
 *   3. Compute entropy: H = -sum(p_i * log2(p_i)) where p_i = count_i / total
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace gpucompress {

/* ============================================================
 * Constants
 * ============================================================ */

/** Number of histogram bins (one per byte value) */
constexpr int NUM_BINS = 256;

/** Block size for histogram kernel */
constexpr int HIST_BLOCK_SIZE = 256;

/** Block size for entropy reduction kernel */
constexpr int ENTROPY_BLOCK_SIZE = 256;

/* ============================================================
 * Histogram Kernel
 * ============================================================ */

/**
 * Build histogram of byte values using shared memory atomics.
 *
 * Each block maintains a local histogram in shared memory,
 * then atomically adds to the global histogram.
 *
 * @param data            Input data buffer
 * @param num_bytes       Number of bytes in data
 * @param global_histogram Output histogram (256 bins)
 */
__global__ void histogramKernel(
    const uint8_t* __restrict__ data,
    size_t num_bytes,
    unsigned int* __restrict__ global_histogram
) {
    /* Per-warp histogram privatization (K1 fix) */
    constexpr int WARPS_PER_BLOCK = HIST_BLOCK_SIZE / 32;
    __shared__ unsigned int s_hist[WARPS_PER_BLOCK][NUM_BINS];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    for (int b = lane_id; b < NUM_BINS; b += 32) {
        s_hist[warp_id][b] = 0;
    }
    __syncthreads();

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = idx; i < num_bytes; i += stride) {
        uint8_t byte_val = data[i];
        atomicAdd(&s_hist[warp_id][byte_val], 1);
    }
    __syncthreads();

    // Merge per-warp histograms to global
    if (tid < NUM_BINS) {
        unsigned int sum = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += s_hist[w][tid];
        }
        if (sum > 0) {
            atomicAdd(&global_histogram[tid], sum);
        }
    }
}

/**
 * Vectorized histogram kernel for improved memory throughput.
 *
 * Processes 4 bytes at a time using uint32_t reads.
 */
/**
 * Vectorized histogram with per-warp privatization (K1 fix).
 *
 * Each warp maintains its own 256-bin histogram to avoid shared memory
 * bank conflicts. With 8 warps/block, uses 8*256 = 2048 shared ints (8 KB).
 * Warp-private histograms are merged to global at the end.
 */
__global__ void histogramKernelVec4(
    const uint8_t* __restrict__ data,
    size_t num_bytes,
    unsigned int* __restrict__ global_histogram
) {
    constexpr int WARPS_PER_BLOCK = HIST_BLOCK_SIZE / 32;
    __shared__ unsigned int s_hist[WARPS_PER_BLOCK][NUM_BINS];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Initialize per-warp histograms (8 passes of 32 threads = 256 bins)
    for (int b = lane_id; b < NUM_BINS; b += 32) {
        s_hist[warp_id][b] = 0;
    }
    __syncthreads();

    // Process 4 bytes at a time — each thread writes to its warp's histogram
    size_t num_words = num_bytes / 4;
    const uint32_t* data32 = reinterpret_cast<const uint32_t*>(data);

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = idx; i < num_words; i += stride) {
        uint32_t word = data32[i];
        atomicAdd(&s_hist[warp_id][(word >>  0) & 0xFF], 1);
        atomicAdd(&s_hist[warp_id][(word >>  8) & 0xFF], 1);
        atomicAdd(&s_hist[warp_id][(word >> 16) & 0xFF], 1);
        atomicAdd(&s_hist[warp_id][(word >> 24) & 0xFF], 1);
    }

    // Handle remaining bytes (only one block to avoid double-counting)
    if (blockIdx.x == 0) {
        size_t remaining_start = num_words * 4;
        for (size_t i = remaining_start + tid; i < num_bytes; i += blockDim.x) {
            atomicAdd(&s_hist[warp_id][data[i]], 1);
        }
    }
    __syncthreads();

    // Merge per-warp histograms to global: each thread handles one bin
    if (tid < NUM_BINS) {
        unsigned int sum = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += s_hist[w][tid];
        }
        if (sum > 0) {
            atomicAdd(&global_histogram[tid], sum);
        }
    }
}

/* ============================================================
 * Entropy Calculation Kernel
 * ============================================================ */

/**
 * Compute Shannon entropy from histogram.
 *
 * Uses warp-level reduction for efficiency.
 * Runs with single block of 256 threads (one per bin).
 *
 * @param histogram   Input histogram (256 bins)
 * @param total_count Total number of bytes
 * @param entropy_out Output entropy value
 */
__global__ void entropyFromHistogramKernel(
    const unsigned int* __restrict__ histogram,
    size_t total_count,
    double* __restrict__ entropy_out
) {
    // Shared memory for block reduction
    __shared__ double s_partial[ENTROPY_BLOCK_SIZE];

    int tid = threadIdx.x;
    double partial_entropy = 0.0;

    // Each thread computes entropy contribution for one or more bins
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int count = histogram[bin];
        if (count > 0) {
            double p = static_cast<double>(count) / static_cast<double>(total_count);
            partial_entropy -= p * log2(p);
        }
    }

    s_partial[tid] = partial_entropy;
    __syncthreads();

    // Block reduction (power of 2 reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        *entropy_out = s_partial[0];
    }
}

/**
 * Launch entropy kernels asynchronously without D->H copy or sync.
 *
 * Fire-and-forget variant for use in the auto-stats GPU pipeline.
 * Writes entropy result directly to a device pointer.
 *
 * @param d_data         Data buffer (GPU memory)
 * @param num_bytes      Size in bytes
 * @param d_histogram    Pre-allocated histogram buffer (256 * sizeof(unsigned int))
 * @param d_entropy_out  Device pointer to write entropy result
 * @param stream         CUDA stream
 */
int launchEntropyKernelsAsync(
    const void* d_data,
    size_t num_bytes,
    unsigned int* d_histogram,
    double* d_entropy_out,
    cudaStream_t stream
) {
    if (d_data == nullptr || num_bytes == 0) {
        return -1;
    }

    // Initialize histogram to zero
    cudaError_t err = cudaMemsetAsync(d_histogram, 0, NUM_BINS * sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        return -1;
    }

    // Calculate grid size
    int num_blocks = static_cast<int>((num_bytes + HIST_BLOCK_SIZE - 1) / HIST_BLOCK_SIZE);
    num_blocks = min(num_blocks, 1024);

    // Launch histogram kernel
    if (num_bytes >= 1024 && (reinterpret_cast<uintptr_t>(d_data) % 4) == 0) {
        histogramKernelVec4<<<num_blocks, HIST_BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint8_t*>(d_data), num_bytes, d_histogram);
    } else {
        histogramKernel<<<num_blocks, HIST_BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint8_t*>(d_data), num_bytes, d_histogram);
    }

    // Launch entropy kernel - writes directly to device pointer
    entropyFromHistogramKernel<<<1, ENTROPY_BLOCK_SIZE, 0, stream>>>(
        d_histogram, num_bytes, d_entropy_out);

    return 0;
}

} // namespace gpucompress
