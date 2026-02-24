/**
 * @file stats_kernel.cu
 * @brief GPU kernels for ALGO_AUTO statistics pipeline
 *
 * Computes MAD, second derivative, and entropy entirely on GPU.
 *
 * Pipeline:
 *   1. statsPass1Kernel:        sum + min + max + second_deriv_sum (1 pass over float data)
 *   2. Entropy kernels:         histogram + entropy               (1 pass over byte data)
 *   3. madPass2Kernel:          sum(|x - mean|)                   (1 pass, needs mean from step 1)
 *   4. finalizeStatsOnlyKernel: normalize MAD and derivative
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>

namespace gpucompress {

/* ============================================================
 * AutoStatsGPU struct (device-side intermediate buffer)
 * ============================================================ */

struct __align__(8) AutoStatsGPU {
    // Pass 1 outputs
    double sum;              // sum of all elements (for mean)
    double abs_diff_sum;     // sum of |x[i+1] - 2*x[i] + x[i-1]|
    float  vmin;             // data min
    float  vmax;             // data max
    size_t num_elements;

    // Pass 2 output
    double mad_sum;          // sum of |x[i] - mean|

    // Entropy pipeline output
    double entropy;

    // Finalize outputs (contiguous for single D->H copy)
    double mad_normalized;
    double deriv_normalized;
    int    state;
    int    action;

    // Error level (input, set from host)
    int    error_level;
};

/* ============================================================
 * Device helper: CAS-based atomicAdd for double
 * ============================================================ */

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* ============================================================
 * Device helper: CAS-based atomicMin/Max for float
 * ============================================================ */

__device__ void atomicMinFloat(float* address, float val) {
    float old = *address;
    while (val < old) {
        unsigned int assumed = __float_as_uint(old);
        unsigned int result = atomicCAS((unsigned int*)address, assumed, __float_as_uint(val));
        if (result == assumed) break;
        old = __uint_as_float(result);
    }
}

__device__ void atomicMaxFloat(float* address, float val) {
    float old = *address;
    while (val > old) {
        unsigned int assumed = __float_as_uint(old);
        unsigned int result = atomicCAS((unsigned int*)address, assumed, __float_as_uint(val));
        if (result == assumed) break;
        old = __uint_as_float(result);
    }
}

/* ============================================================
 * Kernel 1: statsPass1Kernel
 * ============================================================ */

constexpr int STATS_BLOCK_SIZE = 256;
constexpr int STATS_MAX_BLOCKS = 1024;

/**
 * First pass over float data: compute sum, min, max, second_derivative_sum.
 *
 * Uses warp shuffle reduction -> shared memory inter-warp reduction
 * -> atomic global reduction.
 */
__global__ void statsPass1Kernel(
    const float* __restrict__ data,
    size_t num_elements,
    AutoStatsGPU* __restrict__ stats
) {
    // Per-thread accumulators
    double t_sum = 0.0;
    double t_deriv = 0.0;
    float  t_min = FLT_MAX;
    float  t_max = -FLT_MAX;

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    // Grid-stride loop
    for (size_t i = idx; i < num_elements; i += stride) {
        float val = data[i];
        t_sum += static_cast<double>(val);
        t_min = fminf(t_min, val);
        t_max = fmaxf(t_max, val);

        // Second derivative: |data[i+1] - 2*data[i] + data[i-1]| for 0 < i < n-1
        if (i > 0 && i < num_elements - 1) {
            t_deriv += fabs(static_cast<double>(data[i + 1]) - 2.0 * static_cast<double>(val) + static_cast<double>(data[i - 1]));
        }
    }

    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sum   += __shfl_down_sync(mask, t_sum, offset);
        t_deriv += __shfl_down_sync(mask, t_deriv, offset);
        float other_min = __shfl_down_sync(mask, t_min, offset);
        float other_max = __shfl_down_sync(mask, t_max, offset);
        t_min = fminf(t_min, other_min);
        t_max = fmaxf(t_max, other_max);
    }

    // Shared memory for inter-warp reduction
    __shared__ double s_sum[STATS_BLOCK_SIZE / 32];
    __shared__ double s_deriv[STATS_BLOCK_SIZE / 32];
    __shared__ float  s_min[STATS_BLOCK_SIZE / 32];
    __shared__ float  s_max[STATS_BLOCK_SIZE / 32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_sum[warp_id]   = t_sum;
        s_deriv[warp_id] = t_deriv;
        s_min[warp_id]   = t_min;
        s_max[warp_id]   = t_max;
    }
    __syncthreads();

    // First warp reduces shared memory results
    constexpr int num_warps = STATS_BLOCK_SIZE / 32;
    if (warp_id == 0 && lane < num_warps) {
        t_sum   = s_sum[lane];
        t_deriv = s_deriv[lane];
        t_min   = s_min[lane];
        t_max   = s_max[lane];

        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            t_sum   += __shfl_down_sync(mask, t_sum, offset);
            t_deriv += __shfl_down_sync(mask, t_deriv, offset);
            float other_min = __shfl_down_sync(mask, t_min, offset);
            float other_max = __shfl_down_sync(mask, t_max, offset);
            t_min = fminf(t_min, other_min);
            t_max = fmaxf(t_max, other_max);
        }

        if (lane == 0) {
            atomicAddDouble(&stats->sum, t_sum);
            atomicAddDouble(&stats->abs_diff_sum, t_deriv);
            atomicMinFloat(&stats->vmin, t_min);
            atomicMaxFloat(&stats->vmax, t_max);
        }
    }
}

/* ============================================================
 * Kernel 4: madPass2Kernel
 * ============================================================ */

/**
 * Second pass: compute sum(|x[i] - mean|).
 * Reads mean = stats->sum / stats->num_elements.
 */
__global__ void madPass2Kernel(
    const float* __restrict__ data,
    size_t num_elements,
    AutoStatsGPU* __restrict__ stats
) {
    // Compute mean once per block (all threads read same value)
    __shared__ double s_mean;
    if (threadIdx.x == 0) {
        s_mean = stats->sum / static_cast<double>(stats->num_elements);
    }
    __syncthreads();

    double mean = s_mean;
    double t_mad = 0.0;

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = idx; i < num_elements; i += stride) {
        t_mad += fabs(static_cast<double>(data[i]) - mean);
    }

    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_mad += __shfl_down_sync(mask, t_mad, offset);
    }

    // Shared memory inter-warp reduction
    __shared__ double s_mad[STATS_BLOCK_SIZE / 32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_mad[warp_id] = t_mad;
    }
    __syncthreads();

    constexpr int num_warps = STATS_BLOCK_SIZE / 32;
    if (warp_id == 0 && lane < num_warps) {
        t_mad = s_mad[lane];
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            t_mad += __shfl_down_sync(mask, t_mad, offset);
        }
        if (lane == 0) {
            atomicAddDouble(&stats->mad_sum, t_mad);
        }
    }
}

/* ============================================================
 * Kernel 6: finalizeStatsOnlyKernel (for NN pipeline)
 * ============================================================ */

/**
 * Single thread. Normalizes MAD and derivative without Q-Table lookup.
 * Used by the NN pipeline which needs raw normalized values instead of
 * binned states.
 */
__global__ void finalizeStatsOnlyKernel(
    AutoStatsGPU* __restrict__ stats
) {
    if (threadIdx.x != 0) return;

    size_t n = stats->num_elements;
    double range = static_cast<double>(stats->vmax) - static_cast<double>(stats->vmin);

    // Normalize MAD
    double mad_norm = 0.0;
    if (range > 0.0 && n > 0) {
        mad_norm = (stats->mad_sum / static_cast<double>(n)) / range;
    }

    // Normalize second derivative
    double deriv_norm = 0.0;
    if (range > 0.0 && n > 2) {
        deriv_norm = (stats->abs_diff_sum / static_cast<double>(n - 2)) / range;
    }

    stats->mad_normalized = mad_norm;
    stats->deriv_normalized = deriv_norm;
    stats->state = -1;   // Not used in NN mode
    stats->action = -1;  // Will be set by NN inference
}

/* ============================================================
 * External declarations for entropy pipeline
 * ============================================================ */

// From entropy_kernel.cu
int launchEntropyKernelsAsync(
    const void* d_data,
    size_t num_bytes,
    unsigned int* d_histogram,
    double* d_entropy_out,
    cudaStream_t stream
);

// From nn_gpu.cu
bool isNNLoaded();
int runNNInference(
    double entropy,
    double mad_norm,
    double deriv_norm,
    size_t data_size,
    double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio = nullptr,
    float* out_predicted_comp_time = nullptr,
    int* out_top_actions = nullptr
);

/* ============================================================
 * Internal helper: run stats kernels, copy results, free workspace
 * ============================================================ */

/**
 * Shared stats-only kernel sequence used by runStatsOnlyPipeline and
 * runAutoStatsNNPipeline. Allocates workspace, runs all stats kernels
 * with finalizeStatsOnlyKernel, copies entropy/mad/deriv back to host,
 * synchronizes, and frees workspace.
 *
 * Returns 0 on success, -1 on error.
 */
static int runStatsKernels(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
) {
    size_t num_elements = input_size / sizeof(float);
    if (num_elements == 0 || d_input == nullptr) {
        return -1;
    }

    cudaError_t err;

    // Allocate workspace: AutoStatsGPU + 256-uint histogram
    size_t workspace_size = sizeof(AutoStatsGPU) + 256 * sizeof(unsigned int);
    void* d_workspace = nullptr;
    err = cudaMalloc(&d_workspace, workspace_size);
    if (err != cudaSuccess) {
        return -1;
    }

    AutoStatsGPU* d_stats = static_cast<AutoStatsGPU*>(d_workspace);
    unsigned int* d_histogram = reinterpret_cast<unsigned int*>(
        static_cast<uint8_t*>(d_workspace) + sizeof(AutoStatsGPU));

    // Initialize workspace to zero
    err = cudaMemsetAsync(d_workspace, 0, workspace_size, stream);
    if (err != cudaSuccess) {
        cudaFree(d_workspace);
        return -1;
    }

    // Initialize AutoStatsGPU fields
    AutoStatsGPU h_init;
    memset(&h_init, 0, sizeof(h_init));
    h_init.vmin = FLT_MAX;
    h_init.vmax = -FLT_MAX;
    h_init.num_elements = num_elements;
    h_init.error_level = 0;

    err = cudaMemcpyAsync(d_stats, &h_init, sizeof(AutoStatsGPU),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_workspace);
        return -1;
    }

    // Kernel 1: statsPass1 - sum, min, max, derivative_sum
    int num_blocks = static_cast<int>((num_elements + STATS_BLOCK_SIZE - 1) / STATS_BLOCK_SIZE);
    if (num_blocks > STATS_MAX_BLOCKS) num_blocks = STATS_MAX_BLOCKS;

    statsPass1Kernel<<<num_blocks, STATS_BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(d_input), num_elements, d_stats);

    // Kernels 2/3: Entropy pipeline
    launchEntropyKernelsAsync(d_input, input_size, d_histogram,
                              &d_stats->entropy, stream);

    // Kernel 4: madPass2 - sum(|x - mean|)
    madPass2Kernel<<<num_blocks, STATS_BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(d_input), num_elements, d_stats);

    // Kernel 5: finalize stats only (no Q-Table or NN)
    finalizeStatsOnlyKernel<<<1, 1, 0, stream>>>(d_stats);

    // Copy results back to host
    struct StatsResultBlock {
        double entropy;
        double mad_normalized;
        double deriv_normalized;
    };
    StatsResultBlock h_result;

    err = cudaMemcpyAsync(&h_result.entropy, &d_stats->entropy, sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_workspace);
        return -1;
    }

    err = cudaMemcpyAsync(&h_result.mad_normalized, &d_stats->mad_normalized,
                          2 * sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_workspace);
        return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_workspace);
        return -1;
    }

    cudaFree(d_workspace);

    *out_entropy = h_result.entropy;
    *out_mad = h_result.mad_normalized;
    *out_deriv = h_result.deriv_normalized;

    return 0;
}

/* ============================================================
 * Host Pipeline Function (Neural Network mode)
 * ============================================================ */

int runAutoStatsNNPipeline(
    const void* d_input,
    size_t input_size,
    double error_bound,
    cudaStream_t stream,
    int* out_action,
    double* out_entropy,
    double* out_mad,
    double* out_deriv,
    float* out_predicted_ratio,
    float* out_predicted_comp_time,
    int* out_top_actions
) {
    if (!isNNLoaded()) {
        return -1;
    }

    double entropy, mad_norm, deriv_norm;
    int rc = runStatsKernels(d_input, input_size, stream,
                             &entropy, &mad_norm, &deriv_norm);
    if (rc != 0) {
        return -1;
    }

    // Run neural network inference
    int action = runNNInference(
        entropy, mad_norm, deriv_norm,
        input_size, error_bound, stream,
        out_predicted_ratio, out_predicted_comp_time, out_top_actions
    );
    if (action < 0) {
        return -1;
    }

    // Write outputs
    if (out_action) *out_action = action;
    if (out_entropy) *out_entropy = entropy;
    if (out_mad) *out_mad = mad_norm;
    if (out_deriv) *out_deriv = deriv_norm;

    return 0;
}

/* ============================================================
 * Host Pipeline Function (Stats-only mode, no inference)
 * ============================================================ */

int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
) {
    if (out_entropy == nullptr || out_mad == nullptr || out_deriv == nullptr) {
        return -1;
    }

    return runStatsKernels(d_input, input_size, stream,
                           out_entropy, out_mad, out_deriv);
}

} // namespace gpucompress
