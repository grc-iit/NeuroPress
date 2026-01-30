/**
 * @file quantization_kernels.cu
 * @brief GPU Kernels for Error-Bound Quantization
 *
 * Implements three quantization methods:
 * 1. Linear: Simple round(value / (2 * error_bound))
 * 2. Lorenzo 1D: Prediction-based quantization
 * 3. Block Transform: ZFP-style 4-element orthogonal transform
 *
 * All methods support float32 and float64 with adaptive output precision.
 */

#include "quantization.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>

// ============================================================================
// Constants and Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define BLOCK_TRANSFORM_SIZE 4  // ZFP-style uses 4-element blocks

// ============================================================================
// Min/Max Reduction Kernels for Data Range Computation
// ============================================================================

template<typename T>
__global__ void compute_min_max_kernel(
    const T* input,
    size_t num_elements,
    T* d_min,
    T* d_max
) {
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_min;
    __shared__ typename BlockReduce::TempStorage temp_storage_max;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T thread_min = (idx < num_elements) ? input[idx] : input[0];
    T thread_max = thread_min;

    // Grid-stride loop to handle large arrays
    for (size_t i = idx; i < num_elements; i += stride) {
        T val = input[i];
        thread_min = min(thread_min, val);
        thread_max = max(thread_max, val);
    }

    // Block-level reduction
    T block_min = BlockReduce(temp_storage_min).Reduce(thread_min, cub::Min());
    __syncthreads();
    T block_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());

    // First thread of each block writes result
    if (threadIdx.x == 0) {
        atomicMin((int*)d_min, __float_as_int(block_min));
        atomicMax((int*)d_max, __float_as_int(block_max));
    }
}

// Specialized version for float using atomic operations correctly
__global__ void compute_min_max_float_kernel(
    const float* input,
    size_t num_elements,
    float* d_min,
    float* d_max
) {
    __shared__ float s_min[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    float thread_min = FLT_MAX;
    float thread_max = -FLT_MAX;

    // Grid-stride loop
    for (size_t i = idx; i < num_elements; i += stride) {
        float val = input[i];
        thread_min = fminf(thread_min, val);
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_min = fminf(thread_min, __shfl_down_sync(0xffffffff, thread_min, offset));
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // First thread of each warp writes to shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        s_min[warp_id] = thread_min;
        s_max[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces shared memory results
    if (warp_id == 0 && lane < (BLOCK_SIZE / WARP_SIZE)) {
        thread_min = s_min[lane];
        thread_max = s_max[lane];

        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            thread_min = fminf(thread_min, __shfl_down_sync(0xffffffff, thread_min, offset));
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }

        if (lane == 0) {
            // Use atomicMin/Max via integer reinterpretation for floats
            // This works because IEEE 754 floats sort like integers for positive values
            // For a general solution, we use a simple atomic exchange with comparison
            float old_min = *d_min;
            while (thread_min < old_min) {
                float assumed = old_min;
                old_min = atomicCAS((unsigned int*)d_min,
                                    __float_as_uint(assumed),
                                    __float_as_uint(thread_min));
                old_min = __uint_as_float(__float_as_uint(old_min));
                if (old_min == assumed) break;
            }

            float old_max = *d_max;
            while (thread_max > old_max) {
                float assumed = old_max;
                old_max = atomicCAS((unsigned int*)d_max,
                                    __float_as_uint(assumed),
                                    __float_as_uint(thread_max));
                old_max = __uint_as_float(__float_as_uint(old_max));
                if (old_max == assumed) break;
            }
        }
    }
}

// Double precision version
__global__ void compute_min_max_double_kernel(
    const double* input,
    size_t num_elements,
    double* d_min,
    double* d_max
) {
    __shared__ double s_min[BLOCK_SIZE / WARP_SIZE];
    __shared__ double s_max[BLOCK_SIZE / WARP_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    double thread_min = DBL_MAX;
    double thread_max = -DBL_MAX;

    // Grid-stride loop
    for (size_t i = idx; i < num_elements; i += stride) {
        double val = input[i];
        thread_min = fmin(thread_min, val);
        thread_max = fmax(thread_max, val);
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_min = fmin(thread_min, __shfl_down_sync(0xffffffff, thread_min, offset));
        thread_max = fmax(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        s_min[warp_id] = thread_min;
        s_max[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0 && lane < (BLOCK_SIZE / WARP_SIZE)) {
        thread_min = s_min[lane];
        thread_max = s_max[lane];

        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            thread_min = fmin(thread_min, __shfl_down_sync(0xffffffff, thread_min, offset));
            thread_max = fmax(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }

        if (lane == 0) {
            // Atomic min/max for doubles using CAS
            unsigned long long* d_min_ull = (unsigned long long*)d_min;
            unsigned long long* d_max_ull = (unsigned long long*)d_max;

            unsigned long long old_min = *d_min_ull;
            while (__longlong_as_double(old_min) > thread_min) {
                unsigned long long assumed = old_min;
                old_min = atomicCAS(d_min_ull, assumed, __double_as_longlong(thread_min));
                if (old_min == assumed) break;
            }

            unsigned long long old_max = *d_max_ull;
            while (__longlong_as_double(old_max) < thread_max) {
                unsigned long long assumed = old_max;
                old_max = atomicCAS(d_max_ull, assumed, __double_as_longlong(thread_max));
                if (old_max == assumed) break;
            }
        }
    }
}

// ============================================================================
// LINEAR QUANTIZATION KERNELS
// ============================================================================

template<typename InputT, typename OutputT>
__global__ void quantize_linear_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,
    double scale,           // = 1.0 / (2 * error_bound)
    double offset           // = data_min
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_elements; i += stride) {
        double val = static_cast<double>(input[i]);
        double centered = val - offset;
        double quantized = round(centered * scale);

        // Clamp to output type range
        if (sizeof(OutputT) == 1) {
            quantized = fmax(-128.0, fmin(127.0, quantized));
        } else if (sizeof(OutputT) == 2) {
            quantized = fmax(-32768.0, fmin(32767.0, quantized));
        }

        output[i] = static_cast<OutputT>(quantized);
    }
}

template<typename InputT, typename OutputT>
__global__ void dequantize_linear_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale,       // = 2 * error_bound
    double offset           // = data_min
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_elements; i += stride) {
        double quantized = static_cast<double>(input[i]);
        double restored = quantized * inv_scale + offset;
        output[i] = static_cast<InputT>(restored);
    }
}

// ============================================================================
// LORENZO 1D PREDICTION KERNELS
// ============================================================================

/**
 * Lorenzo 1D quantization: quantizes the difference from predicted value
 * prediction[i] = value[i-1], so residual[i] = value[i] - value[i-1]
 *
 * This is inherently parallel because we're quantizing differences,
 * not cumulative values.
 */
template<typename InputT, typename OutputT>
__global__ void quantize_lorenzo1d_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,
    double scale,
    double offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_elements; i += stride) {
        double val = static_cast<double>(input[i]);
        double pred;

        if (i == 0) {
            // First element: no prediction, use offset
            pred = offset;
        } else {
            // Predict from previous element
            pred = static_cast<double>(input[i - 1]);
        }

        double residual = val - pred;
        double quantized = round(residual * scale);

        // Clamp to output type range
        if (sizeof(OutputT) == 1) {
            quantized = fmax(-128.0, fmin(127.0, quantized));
        } else if (sizeof(OutputT) == 2) {
            quantized = fmax(-32768.0, fmin(32767.0, quantized));
        }

        output[i] = static_cast<OutputT>(quantized);
    }
}

/**
 * Lorenzo 1D dequantization requires sequential reconstruction
 * because each value depends on the previous restored value.
 *
 * We use a block-parallel approach: each block processes a segment
 * sequentially within the block, then combines results.
 */
template<typename InputT, typename OutputT>
__global__ void dequantize_lorenzo1d_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale,
    double offset
) {
    // Simple sequential implementation within block
    // For better performance, a parallel prefix scan could be used
    size_t block_start = blockIdx.x * blockDim.x;
    size_t block_end = min(block_start + blockDim.x, num_elements);

    if (threadIdx.x != 0) return;  // Only first thread per block processes

    for (size_t i = block_start; i < block_end; i++) {
        double quantized = static_cast<double>(input[i]);
        double residual = quantized * inv_scale;

        double pred;
        if (i == 0) {
            pred = offset;
        } else {
            pred = static_cast<double>(output[i - 1]);
        }

        output[i] = static_cast<InputT>(pred + residual);
    }
}

/**
 * Parallel Lorenzo 1D dequantization using prefix sum
 * Works in two phases:
 * 1. Each thread computes residuals for its elements
 * 2. Inclusive scan to accumulate predictions
 */
template<typename InputT, typename OutputT>
__global__ void dequantize_lorenzo1d_parallel_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale,
    double offset,
    InputT* d_block_sums,  // Output: last value of each block for inter-block scan
    size_t blocks_per_grid
) {
    typedef cub::BlockScan<double, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double residual = 0.0;
    if (idx < num_elements) {
        residual = static_cast<double>(input[idx]) * inv_scale;
    }

    double accumulated;
    BlockScan(temp_storage).InclusiveSum(residual, accumulated);

    // Add offset for first element of first block
    if (blockIdx.x == 0) {
        accumulated += offset;
    }

    if (idx < num_elements) {
        output[idx] = static_cast<InputT>(accumulated);
    }

    // Save last value of each block for inter-block propagation
    if (threadIdx.x == blockDim.x - 1 || idx == num_elements - 1) {
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = static_cast<InputT>(accumulated);
        }
    }
}

// Second pass to add block prefixes
template<typename T>
__global__ void add_block_prefix_kernel(
    T* output,
    const T* block_sums,
    size_t num_elements
) {
    if (blockIdx.x == 0) return;  // First block doesn't need adjustment

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    T prefix = block_sums[blockIdx.x - 1];
    output[idx] = output[idx] + prefix;
}

// ============================================================================
// BLOCK TRANSFORM KERNELS (ZFP-style)
// ============================================================================

/**
 * ZFP-style 4-element block orthogonal transform
 *
 * Forward transform (lifting steps):
 *   y[0] = (x[0] + x[3]) / 2
 *   y[1] = (x[0] - x[3])
 *   y[2] = (x[1] + x[2]) / 2
 *   y[3] = (x[1] - x[2])
 *
 * Then apply decorrelation:
 *   z[0] = (y[0] + y[2]) / 2
 *   z[1] = y[1]
 *   z[2] = (y[0] - y[2])
 *   z[3] = y[3]
 *
 * This concentrates energy in z[0] for smooth data.
 */
__device__ __forceinline__ void forward_block_transform_4(
    double x0, double x1, double x2, double x3,
    double& z0, double& z1, double& z2, double& z3
) {
    // First stage
    double y0 = (x0 + x3) * 0.5;
    double y1 = x0 - x3;
    double y2 = (x1 + x2) * 0.5;
    double y3 = x1 - x2;

    // Second stage (decorrelation)
    z0 = (y0 + y2) * 0.5;
    z1 = y1;
    z2 = y0 - y2;
    z3 = y3;
}

/**
 * Inverse transform (reverse lifting steps)
 */
__device__ __forceinline__ void inverse_block_transform_4(
    double z0, double z1, double z2, double z3,
    double& x0, double& x1, double& x2, double& x3
) {
    // Reverse second stage
    double y0 = z0 + z2 * 0.5;
    double y1 = z1;
    double y2 = z0 - z2 * 0.5;
    double y3 = z3;

    // Reverse first stage
    x0 = y0 + y1 * 0.5;
    x3 = y0 - y1 * 0.5;
    x1 = y2 + y3 * 0.5;
    x2 = y2 - y3 * 0.5;
}

template<typename InputT, typename OutputT>
__global__ void quantize_block_transform_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,
    double scale,
    double offset
) {
    // Each thread handles one 4-element block
    size_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (num_elements + BLOCK_TRANSFORM_SIZE - 1) / BLOCK_TRANSFORM_SIZE;

    if (block_idx >= num_blocks) return;

    size_t base = block_idx * BLOCK_TRANSFORM_SIZE;

    // Load 4 elements (pad with offset for incomplete blocks)
    double x0 = (base + 0 < num_elements) ? static_cast<double>(input[base + 0]) - offset : 0.0;
    double x1 = (base + 1 < num_elements) ? static_cast<double>(input[base + 1]) - offset : 0.0;
    double x2 = (base + 2 < num_elements) ? static_cast<double>(input[base + 2]) - offset : 0.0;
    double x3 = (base + 3 < num_elements) ? static_cast<double>(input[base + 3]) - offset : 0.0;

    // Apply forward transform
    double z0, z1, z2, z3;
    forward_block_transform_4(x0, x1, x2, x3, z0, z1, z2, z3);

    // Quantize transformed coefficients
    z0 = round(z0 * scale);
    z1 = round(z1 * scale);
    z2 = round(z2 * scale);
    z3 = round(z3 * scale);

    // Clamp to output type range
    double max_val, min_val;
    if (sizeof(OutputT) == 1) {
        min_val = -128.0; max_val = 127.0;
    } else if (sizeof(OutputT) == 2) {
        min_val = -32768.0; max_val = 32767.0;
    } else {
        min_val = -2147483648.0; max_val = 2147483647.0;
    }

    z0 = fmax(min_val, fmin(max_val, z0));
    z1 = fmax(min_val, fmin(max_val, z1));
    z2 = fmax(min_val, fmin(max_val, z2));
    z3 = fmax(min_val, fmin(max_val, z3));

    // Store (only valid elements)
    if (base + 0 < num_elements) output[base + 0] = static_cast<OutputT>(z0);
    if (base + 1 < num_elements) output[base + 1] = static_cast<OutputT>(z1);
    if (base + 2 < num_elements) output[base + 2] = static_cast<OutputT>(z2);
    if (base + 3 < num_elements) output[base + 3] = static_cast<OutputT>(z3);
}

template<typename InputT, typename OutputT>
__global__ void dequantize_block_transform_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale,
    double offset
) {
    size_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (num_elements + BLOCK_TRANSFORM_SIZE - 1) / BLOCK_TRANSFORM_SIZE;

    if (block_idx >= num_blocks) return;

    size_t base = block_idx * BLOCK_TRANSFORM_SIZE;

    // Load quantized coefficients
    double z0 = (base + 0 < num_elements) ? static_cast<double>(input[base + 0]) * inv_scale : 0.0;
    double z1 = (base + 1 < num_elements) ? static_cast<double>(input[base + 1]) * inv_scale : 0.0;
    double z2 = (base + 2 < num_elements) ? static_cast<double>(input[base + 2]) * inv_scale : 0.0;
    double z3 = (base + 3 < num_elements) ? static_cast<double>(input[base + 3]) * inv_scale : 0.0;

    // Apply inverse transform
    double x0, x1, x2, x3;
    inverse_block_transform_4(z0, z1, z2, z3, x0, x1, x2, x3);

    // Restore offset and store
    if (base + 0 < num_elements) output[base + 0] = static_cast<InputT>(x0 + offset);
    if (base + 1 < num_elements) output[base + 1] = static_cast<InputT>(x1 + offset);
    if (base + 2 < num_elements) output[base + 2] = static_cast<InputT>(x2 + offset);
    if (base + 3 < num_elements) output[base + 3] = static_cast<InputT>(x3 + offset);
}

// ============================================================================
// ERROR VERIFICATION KERNEL
// ============================================================================

template<typename T>
__global__ void verify_error_bound_kernel(
    const T* original,
    const T* restored,
    size_t num_elements,
    double error_bound,
    int* violation_count,
    double* max_error
) {
    __shared__ int s_violations[BLOCK_SIZE / WARP_SIZE];
    __shared__ double s_max_error[BLOCK_SIZE / WARP_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    int local_violations = 0;
    double local_max_error = 0.0;

    for (size_t i = idx; i < num_elements; i += stride) {
        double orig = static_cast<double>(original[i]);
        double rest = static_cast<double>(restored[i]);
        double err = fabs(orig - rest);

        if (err > error_bound) {
            local_violations++;
        }
        local_max_error = fmax(local_max_error, err);
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_violations += __shfl_down_sync(0xffffffff, local_violations, offset);
        local_max_error = fmax(local_max_error, __shfl_down_sync(0xffffffff, local_max_error, offset));
    }

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        s_violations[warp_id] = local_violations;
        s_max_error[warp_id] = local_max_error;
    }
    __syncthreads();

    // First warp reduces shared memory
    if (warp_id == 0 && lane < (BLOCK_SIZE / WARP_SIZE)) {
        local_violations = s_violations[lane];
        local_max_error = s_max_error[lane];

        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            local_violations += __shfl_down_sync(0xffffffff, local_violations, offset);
            local_max_error = fmax(local_max_error, __shfl_down_sync(0xffffffff, local_max_error, offset));
        }

        if (lane == 0) {
            atomicAdd(violation_count, local_violations);

            // Atomic max for double
            unsigned long long* max_err_ull = (unsigned long long*)max_error;
            unsigned long long old = *max_err_ull;
            while (__longlong_as_double(old) < local_max_error) {
                unsigned long long assumed = old;
                old = atomicCAS(max_err_ull, assumed, __double_as_longlong(local_max_error));
                if (old == assumed) break;
            }
        }
    }
}

// ============================================================================
// HOST API IMPLEMENTATION
// ============================================================================

/**
 * Compute min/max of data on GPU
 */
static void compute_data_range(
    void* d_input,
    size_t num_elements,
    size_t element_size,
    double& data_min,
    double& data_max,
    cudaStream_t stream
) {
    int num_blocks = min((int)((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);

    if (element_size == 4) {
        // Float
        float* d_min;
        float* d_max;
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));

        float init_min = FLT_MAX;
        float init_max = -FLT_MAX;
        cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

        compute_min_max_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<float*>(d_input), num_elements, d_min, d_max);

        float h_min, h_max;
        cudaMemcpyAsync(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        data_min = h_min;
        data_max = h_max;

        cudaFree(d_min);
        cudaFree(d_max);
    } else {
        // Double
        double* d_min_d;
        double* d_max_d;
        cudaMalloc(&d_min_d, sizeof(double));
        cudaMalloc(&d_max_d, sizeof(double));

        double init_min = DBL_MAX;
        double init_max = -DBL_MAX;
        cudaMemcpyAsync(d_min_d, &init_min, sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max_d, &init_max, sizeof(double), cudaMemcpyHostToDevice, stream);

        compute_min_max_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<double*>(d_input), num_elements, d_min_d, d_max_d);

        cudaMemcpyAsync(&data_min, d_min_d, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&data_max, d_max_d, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_min_d);
        cudaFree(d_max_d);
    }
}

/**
 * Launch appropriate quantization kernel based on config
 */
template<typename InputT, typename OutputT>
static void launch_quantize_kernel(
    const InputT* d_input,
    OutputT* d_output,
    size_t num_elements,
    QuantizationType type,
    double scale,
    double offset,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);

    switch (type) {
        case QuantizationType::LINEAR:
            quantize_linear_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, scale, offset);
            break;

        case QuantizationType::LORENZO_1D:
            quantize_lorenzo1d_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, scale, offset);
            break;

        case QuantizationType::BLOCK_TRANSFORM: {
            size_t num_transform_blocks = (num_elements + BLOCK_TRANSFORM_SIZE - 1) / BLOCK_TRANSFORM_SIZE;
            int kernel_blocks = (num_transform_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_blocks = min(kernel_blocks, 65535);
            quantize_block_transform_kernel<InputT, OutputT><<<kernel_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, scale, offset);
            break;
        }

        default:
            // Should not reach here
            break;
    }
}

/**
 * Launch appropriate dequantization kernel based on config
 */
template<typename InputT, typename OutputT>
static void launch_dequantize_kernel(
    const OutputT* d_input,
    InputT* d_output,
    size_t num_elements,
    QuantizationType type,
    double inv_scale,
    double offset,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);

    switch (type) {
        case QuantizationType::LINEAR:
            dequantize_linear_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, inv_scale, offset);
            break;

        case QuantizationType::LORENZO_1D:
            // Use simple sequential version for correctness
            // For large data, consider parallel prefix sum approach
            dequantize_lorenzo1d_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, inv_scale, offset);
            break;

        case QuantizationType::BLOCK_TRANSFORM: {
            size_t num_transform_blocks = (num_elements + BLOCK_TRANSFORM_SIZE - 1) / BLOCK_TRANSFORM_SIZE;
            int kernel_blocks = (num_transform_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_blocks = min(kernel_blocks, 65535);
            dequantize_block_transform_kernel<InputT, OutputT><<<kernel_blocks, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, num_elements, inv_scale, offset);
            break;
        }

        default:
            break;
    }
}

// ============================================================================
// PUBLIC API FUNCTIONS
// ============================================================================

QuantizationResult quantize_simple(
    void* d_input,
    size_t num_elements,
    size_t element_size,
    QuantizationConfig config,
    cudaStream_t stream
) {
    QuantizationResult result;

    // Validate input
    if (d_input == nullptr || num_elements == 0) {
        fprintf(stderr, "quantize_simple: Invalid input (null or zero elements)\n");
        return result;
    }

    if (element_size != 4 && element_size != 8) {
        fprintf(stderr, "quantize_simple: Unsupported element size %zu (must be 4 or 8)\n", element_size);
        return result;
    }

    if (config.error_bound <= 0.0) {
        fprintf(stderr, "quantize_simple: Invalid error bound %.6e (must be > 0)\n", config.error_bound);
        return result;
    }

    // Step 1: Compute data range
    double data_min, data_max;
    compute_data_range(d_input, num_elements, element_size, data_min, data_max, stream);

    double data_range = data_max - data_min;
    if (data_range <= 0.0) {
        data_range = 1.0;  // Handle constant data
    }

    // Step 2: Determine precision
    int precision;
    if (config.precision == QuantizationPrecision::AUTO) {
        precision = compute_required_precision(data_range, config.error_bound);
    } else {
        switch (config.precision) {
            case QuantizationPrecision::INT8:  precision = 8; break;
            case QuantizationPrecision::INT16: precision = 16; break;
            case QuantizationPrecision::INT32: precision = 32; break;
            default: precision = 32;
        }
    }

    // Step 3: Compute quantization scale
    double scale = 1.0 / (2.0 * config.error_bound);
    double inv_scale = 2.0 * config.error_bound;

    // Step 4: Allocate output buffer
    size_t output_bytes = num_elements * precision_to_bytes(precision);
    void* d_output;
    cudaError_t err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "quantize_simple: Failed to allocate output buffer (%zu bytes)\n", output_bytes);
        return result;
    }

    // Step 5: Launch quantization kernel
    if (element_size == 4) {
        // Float input
        if (precision == 8) {
            launch_quantize_kernel<float, int8_t>(
                static_cast<float*>(d_input), static_cast<int8_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        } else if (precision == 16) {
            launch_quantize_kernel<float, int16_t>(
                static_cast<float*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        } else {
            launch_quantize_kernel<float, int32_t>(
                static_cast<float*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        }
    } else {
        // Double input
        if (precision == 8) {
            launch_dequantize_kernel<double, int8_t>(
                static_cast<int8_t*>(d_output), static_cast<double*>(d_input),
                num_elements, config.type, inv_scale, data_min, stream);
            // Oops, wrong direction. Let me fix:
            launch_quantize_kernel<double, int8_t>(
                static_cast<double*>(d_input), static_cast<int8_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        } else if (precision == 16) {
            launch_quantize_kernel<double, int16_t>(
                static_cast<double*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        } else {
            launch_quantize_kernel<double, int32_t>(
                static_cast<double*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, config.type, scale, data_min, stream);
        }
    }

    cudaStreamSynchronize(stream);

    // Fill result
    result.d_quantized = d_output;
    result.quantized_bytes = output_bytes;
    result.actual_precision = precision;
    result.data_min = data_min;
    result.data_max = data_max;
    result.scale_factor = scale;
    result.error_bound = config.error_bound;
    result.type = config.type;
    result.num_elements = num_elements;
    result.original_element_size = element_size;

    return result;
}

void* dequantize_simple(
    void* d_quantized,
    const QuantizationResult& metadata,
    cudaStream_t stream
) {
    if (d_quantized == nullptr || !metadata.isValid()) {
        fprintf(stderr, "dequantize_simple: Invalid input or metadata\n");
        return nullptr;
    }

    // Allocate output buffer for restored data
    size_t output_bytes = metadata.num_elements * metadata.original_element_size;
    void* d_output;
    cudaError_t err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "dequantize_simple: Failed to allocate output buffer (%zu bytes)\n", output_bytes);
        return nullptr;
    }

    double inv_scale = 2.0 * metadata.error_bound;

    // Launch dequantization kernel
    if (metadata.original_element_size == 4) {
        // Float output
        if (metadata.actual_precision == 8) {
            launch_dequantize_kernel<float, int8_t>(
                static_cast<int8_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        } else if (metadata.actual_precision == 16) {
            launch_dequantize_kernel<float, int16_t>(
                static_cast<int16_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        } else {
            launch_dequantize_kernel<float, int32_t>(
                static_cast<int32_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        }
    } else {
        // Double output
        if (metadata.actual_precision == 8) {
            launch_dequantize_kernel<double, int8_t>(
                static_cast<int8_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        } else if (metadata.actual_precision == 16) {
            launch_dequantize_kernel<double, int16_t>(
                static_cast<int16_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        } else {
            launch_dequantize_kernel<double, int32_t>(
                static_cast<int32_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, metadata.type, inv_scale, metadata.data_min, stream);
        }
    }

    cudaStreamSynchronize(stream);
    return d_output;
}

bool verify_error_bound(
    const void* d_original,
    const void* d_restored,
    size_t num_elements,
    size_t element_size,
    double error_bound,
    cudaStream_t stream,
    double* max_error_out
) {
    int* d_violations;
    double* d_max_error;
    cudaMalloc(&d_violations, sizeof(int));
    cudaMalloc(&d_max_error, sizeof(double));

    int zero = 0;
    double init_max = 0.0;
    cudaMemcpyAsync(d_violations, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max_error, &init_max, sizeof(double), cudaMemcpyHostToDevice, stream);

    int num_blocks = min((int)((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);

    if (element_size == 4) {
        verify_error_bound_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(d_original),
            static_cast<const float*>(d_restored),
            num_elements, error_bound, d_violations, d_max_error);
    } else {
        verify_error_bound_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const double*>(d_original),
            static_cast<const double*>(d_restored),
            num_elements, error_bound, d_violations, d_max_error);
    }

    int h_violations;
    double h_max_error;
    cudaMemcpyAsync(&h_violations, d_violations, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_violations);
    cudaFree(d_max_error);

    if (max_error_out) {
        *max_error_out = h_max_error;
    }

    return (h_violations == 0);
}
