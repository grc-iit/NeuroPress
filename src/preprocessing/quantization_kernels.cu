/**
 * @file quantization_kernels.cu
 * @brief GPU Kernels for Error-Bound Quantization
 *
 * Implements linear quantization: round(value / (2 * error_bound))
 *
 * Supports float32 and float64 with adaptive output precision.
 */

#include "preprocessing/quantization.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>

// ============================================================================
// Constants and Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// Min/Max Reduction Kernels for Data Range Computation
// ============================================================================

// Float min/max using CAS-based atomics (correct for negative values)
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
            // Atomic min for floats using CAS
            // Must use float comparison (not bit comparison) to handle negative values
            float old_min = *d_min;
            while (thread_min < old_min) {
                unsigned int assumed = __float_as_uint(old_min);
                unsigned int result = atomicCAS((unsigned int*)d_min, assumed, __float_as_uint(thread_min));
                if (result == assumed) break;
                old_min = __uint_as_float(result);
            }

            // Atomic max for floats
            float old_max = *d_max;
            while (thread_max > old_max) {
                unsigned int assumed = __float_as_uint(old_max);
                unsigned int result = atomicCAS((unsigned int*)d_max, assumed, __float_as_uint(thread_max));
                if (result == assumed) break;
                old_max = __uint_as_float(result);
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
 * Returns 0 on success, -1 on error.
 */
static int compute_data_range(
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
        float* d_min = nullptr;
        float* d_max = nullptr;
        if (cudaMalloc(&d_min, sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_max, sizeof(float)) != cudaSuccess) {
            cudaFree(d_min);
            cudaFree(d_max);
            return -1;
        }

        // Init min to largest value, max to smallest value
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
        double* d_min_d = nullptr;
        double* d_max_d = nullptr;
        if (cudaMalloc(&d_min_d, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_max_d, sizeof(double)) != cudaSuccess) {
            cudaFree(d_min_d);
            cudaFree(d_max_d);
            return -1;
        }

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
    return 0;
}

/**
 * Launch linear quantization kernel
 */
template<typename InputT, typename OutputT>
static void launch_quantize_kernel(
    const InputT* d_input,
    OutputT* d_output,
    size_t num_elements,
    double scale,
    double offset,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);

    quantize_linear_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, num_elements, scale, offset);
}

/**
 * Launch linear dequantization kernel
 */
template<typename InputT, typename OutputT>
static void launch_dequantize_kernel(
    const OutputT* d_input,
    InputT* d_output,
    size_t num_elements,
    double inv_scale,
    double offset,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);

    dequantize_linear_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, num_elements, inv_scale, offset);
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
    if (compute_data_range(d_input, num_elements, element_size, data_min, data_max, stream) != 0) {
        fprintf(stderr, "quantize_simple: Failed to compute data range\n");
        return result;
    }

    double data_range = data_max - data_min;
    if (data_range <= 0.0) {
        data_range = 1.0;  // Handle constant data
    }

    // Step 2: Compute effective error bound accounting for ALL error sources
    //
    // Total error comes from two sources:
    //   1. Quantization rounding error: max = effective_eb (from scale = 1/(2*effective_eb))
    //   2. Float32 representation error when converting double back to float32:
    //      max ≈ |restored_value| * FLT_EPSILON ≈ max(|data_min|, |data_max|) * 1.19e-7
    //
    // To guarantee total error <= user_error_bound:
    //   effective_eb + float_repr_error <= user_error_bound
    //   effective_eb <= user_error_bound - float_repr_error
    //
    double max_abs_value = fmax(fabs(data_min), fabs(data_max));

    // Use FLT_EPSILON with safety multiplier to account for accumulated rounding
    // FLT_EPSILON ≈ 1.19e-7, use 2x for safety in multi-step operations
    double float_repr_error = max_abs_value * 2.4e-7;

    // Reserve space for float representation error plus safety margin
    double available_for_quant = config.error_bound - float_repr_error;

    // Apply additional 5% safety margin for numeric stability in scale calculations
    double safety_margin = config.error_bound * 0.05;
    available_for_quant -= safety_margin;

    // Compute minimum effective_eb that won't overflow int32 quantized values
    // max_quantized = data_range * scale = data_range / (2 * effective_eb)
    // For int32: data_range / (2 * effective_eb) <= 2^31 - 1
    // So: effective_eb >= data_range / (2^32 - 2)
    double min_eb_for_int32 = data_range / 4.0e9;  // Conservative limit for int32

    double effective_eb;
    if (available_for_quant <= 0) {
        // Error bound is too tight for float32 precision with this data range
        // Use maximum safe precision (limited by int32 quantized value range)
        double min_achievable = fmax(float_repr_error, min_eb_for_int32);

        fprintf(stderr, "Warning: Error bound %.2e is below float32 precision limit for data range [%.2e, %.2e]\n",
                config.error_bound, data_min, data_max);
        fprintf(stderr, "  Float32 representation error: %.2e\n", float_repr_error);
        fprintf(stderr, "  Minimum achievable error: ~%.2e\n", min_achievable);
        fprintf(stderr, "  Using maximum precision quantization (error may exceed bound)\n");

        // Use maximum safe precision
        effective_eb = fmax(min_eb_for_int32, float_repr_error * 0.1);
    } else {
        effective_eb = available_for_quant;
    }

    // Ensure effective_eb doesn't cause overflow
    effective_eb = fmax(effective_eb, min_eb_for_int32);

    // Step 3: Determine precision using effective error bound
    // This ensures the selected precision can hold all quantized values
    int precision;
    if (config.precision == QuantizationPrecision::AUTO) {
        precision = compute_required_precision(data_range, effective_eb);
    } else {
        switch (config.precision) {
            case QuantizationPrecision::INT8:  precision = 8; break;
            case QuantizationPrecision::INT16: precision = 16; break;
            case QuantizationPrecision::INT32: precision = 32; break;
            default: precision = 32;
        }
    }

    // Compute quantization scale from effective error bound
    double scale = 1.0 / (2.0 * effective_eb);

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
                num_elements, scale, data_min, stream);
        } else if (precision == 16) {
            launch_quantize_kernel<float, int16_t>(
                static_cast<float*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, scale, data_min, stream);
        } else {
            launch_quantize_kernel<float, int32_t>(
                static_cast<float*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, scale, data_min, stream);
        }
    } else {
        // Double input
        if (precision == 8) {
            launch_quantize_kernel<double, int8_t>(
                static_cast<double*>(d_input), static_cast<int8_t*>(d_output),
                num_elements, scale, data_min, stream);
        } else if (precision == 16) {
            launch_quantize_kernel<double, int16_t>(
                static_cast<double*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, scale, data_min, stream);
        } else {
            launch_quantize_kernel<double, int32_t>(
                static_cast<double*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, scale, data_min, stream);
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

    // Use the stored scale factor (accounts for safety margin used during quantization)
    double inv_scale = 1.0 / metadata.scale_factor;

    // Launch dequantization kernel
    if (metadata.original_element_size == 4) {
        // Float output
        if (metadata.actual_precision == 8) {
            launch_dequantize_kernel<float, int8_t>(
                static_cast<int8_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
        } else if (metadata.actual_precision == 16) {
            launch_dequantize_kernel<float, int16_t>(
                static_cast<int16_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
        } else {
            launch_dequantize_kernel<float, int32_t>(
                static_cast<int32_t*>(d_quantized), static_cast<float*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
        }
    } else {
        // Double output
        if (metadata.actual_precision == 8) {
            launch_dequantize_kernel<double, int8_t>(
                static_cast<int8_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
        } else if (metadata.actual_precision == 16) {
            launch_dequantize_kernel<double, int16_t>(
                static_cast<int16_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
        } else {
            launch_dequantize_kernel<double, int32_t>(
                static_cast<int32_t*>(d_quantized), static_cast<double*>(d_output),
                metadata.num_elements, inv_scale, metadata.data_min, stream);
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
    int* d_violations = nullptr;
    double* d_max_error = nullptr;
    if (cudaMalloc(&d_violations, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_max_error, sizeof(double)) != cudaSuccess) {
        cudaFree(d_violations);
        cudaFree(d_max_error);
        return false;
    }

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
