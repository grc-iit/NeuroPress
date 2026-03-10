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
#include <climits>
#include <cstdio>

extern bool g_gc_verbose;
#define GC_LOG(fmt, ...) do { if (g_gc_verbose) fprintf(stderr, fmt, ##__VA_ARGS__); } while(0)
#include "xfer_tracker.h"

// ============================================================================
// Constants and Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32  // Used by verify_error_bound_kernel

// Pre-allocated GPU scalars for min/max reduction (avoids per-call malloc/free).
// 8 bytes each — large enough for both float and double paths.
static void* d_range_min = nullptr;
static void* d_range_max = nullptr;

static int ensure_range_bufs() {
    if (d_range_min == nullptr) {
        if (cudaMalloc(&d_range_min, sizeof(double)) != cudaSuccess) return -1;
    }
    if (d_range_max == nullptr) {
        if (cudaMalloc(&d_range_max, sizeof(double)) != cudaSuccess) return -1;
    }
    return 0;
}

void free_range_bufs() {
    if (d_range_min) { cudaFree(d_range_min); d_range_min = nullptr; }
    if (d_range_max) { cudaFree(d_range_max); d_range_max = nullptr; }
}

// ============================================================================
// Min/Max Reduction via CUB DeviceReduce
// ============================================================================

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
        } else if (sizeof(OutputT) == 4) {
            quantized = fmax(-2147483648.0, fmin(2147483647.0, quantized));
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
 * Compute min/max of data on GPU using CUB DeviceReduce.
 *
 * d_buf_min / d_buf_max must each be at least sizeof(double) bytes.
 * CUB handles all the warp/block reduction internally.
 *
 * Returns 0 on success, -1 on error.
 */
template<typename T>
static int compute_data_range_typed(
    T* d_input,
    size_t num_elements,
    double& data_min,
    double& data_max,
    T* d_buf_min,
    T* d_buf_max,
    cudaStream_t stream
) {
    // Guard against int overflow: CUB DeviceReduce takes int count
    if (num_elements > (size_t)INT_MAX) {
        fprintf(stderr, "compute_data_range: num_elements %zu exceeds INT_MAX\n", num_elements);
        return -1;
    }

    // Query temp storage size (same for min and max)
    size_t temp_bytes = 0;
    cudaError_t err = cub::DeviceReduce::Min(nullptr, temp_bytes, d_input, d_buf_min,
                                              (int)num_elements, stream);
    if (err != cudaSuccess) return -1;

    void* d_temp = nullptr;
    err = cudaMalloc(&d_temp, temp_bytes);
    if (err != cudaSuccess) return -1;

    err = cub::DeviceReduce::Min(d_temp, temp_bytes, d_input, d_buf_min,
                                  (int)num_elements, stream);
    if (err != cudaSuccess) { cudaFree(d_temp); return -1; }

    err = cub::DeviceReduce::Max(d_temp, temp_bytes, d_input, d_buf_max,
                                  (int)num_elements, stream);
    if (err != cudaSuccess) { cudaFree(d_temp); return -1; }

    T h_min, h_max;
    XFER_TRACK("quantize data_range: D->H min", sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&h_min, d_buf_min, sizeof(T), cudaMemcpyDeviceToHost, stream);
    XFER_TRACK("quantize data_range: D->H max", sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&h_max, d_buf_max, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_temp);

    data_min = static_cast<double>(h_min);
    data_max = static_cast<double>(h_max);
    return 0;
}

static int compute_data_range(
    void* d_input,
    size_t num_elements,
    size_t element_size,
    double& data_min,
    double& data_max,
    void* d_buf_min,
    void* d_buf_max,
    cudaStream_t stream
) {
    if (element_size == 4) {
        return compute_data_range_typed<float>(
            static_cast<float*>(d_input), num_elements, data_min, data_max,
            static_cast<float*>(d_buf_min), static_cast<float*>(d_buf_max), stream);
    } else {
        return compute_data_range_typed<double>(
            static_cast<double*>(d_input), num_elements, data_min, data_max,
            static_cast<double*>(d_buf_min), static_cast<double*>(d_buf_max), stream);
    }
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

    // Step 1: Compute data range using the legacy shared static buffers.
    // NOTE: Not safe for concurrent calls — use the overload with explicit
    // d_range_min_buf/d_range_max_buf (e.g. from CompContext) for that.
    if (ensure_range_bufs() != 0) {
        fprintf(stderr, "quantize_simple: Failed to allocate range buffers\n");
        return result;
    }
    double data_min, data_max;
    if (compute_data_range(d_input, num_elements, element_size, data_min, data_max,
                           d_range_min, d_range_max, stream) != 0) {
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

    // Warn if forced INT8 cannot represent the data range
    if (precision == 8 && data_range > 0) {
        double max_quant = data_range / (2.0 * config.error_bound);
        if (max_quant > 127.0)
            fprintf(stderr, "gpucompress WARNING: forced INT8 cannot represent data range "
                    "(need %.0f levels, int8 max=127). Error bound may be violated.\n", max_quant);
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

    // Check kernel launch error
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "gpucompress ERROR: quantization kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
        cudaFree(d_output);
        return result;  // returns invalid result
    }

    // Fill result
    result.d_quantized = d_output;
    result.quantized_bytes = output_bytes;
    result.actual_precision = precision;
    result.data_min = data_min;
    result.data_max = data_max;
    result.scale_factor = scale;
    result.error_bound = effective_eb;
    result.type = config.type;
    result.num_elements = num_elements;
    result.original_element_size = element_size;

    return result;
}

/**
 * Thread-safe overload: uses caller-provided pre-allocated device buffers for the
 * min/max reduction instead of the shared static globals.  Pass ctx->d_range_min
 * and ctx->d_range_max from a CompContext slot to avoid concurrent aliasing.
 */
QuantizationResult quantize_simple(
    void* d_input,
    size_t num_elements,
    size_t element_size,
    QuantizationConfig config,
    void* d_range_min_buf,
    void* d_range_max_buf,
    cudaStream_t stream
) {
    QuantizationResult result;

    if (d_input == nullptr || num_elements == 0) {
        fprintf(stderr, "quantize_simple: Invalid input (null or zero elements)\n");
        return result;
    }
    if (element_size != 4 && element_size != 8) {
        fprintf(stderr, "quantize_simple: Unsupported element size %zu\n", element_size);
        return result;
    }
    if (config.error_bound <= 0.0) {
        fprintf(stderr, "quantize_simple: Invalid error bound %.6e\n", config.error_bound);
        return result;
    }
    if (d_range_min_buf == nullptr || d_range_max_buf == nullptr) {
        fprintf(stderr, "quantize_simple: Null range buffers — use per-CompContext buffers\n");
        return result;
    }

    double data_min, data_max;
    if (compute_data_range(d_input, num_elements, element_size, data_min, data_max,
                           d_range_min_buf, d_range_max_buf, stream) != 0) {
        fprintf(stderr, "quantize_simple: Failed to compute data range\n");
        return result;
    }

    double data_range = data_max - data_min;
    if (data_range <= 0.0) data_range = 1.0;

    double max_abs_value   = fmax(fabs(data_min), fabs(data_max));
    double float_repr_error = max_abs_value * 2.4e-7;
    double available_for_quant = config.error_bound - float_repr_error;
    double safety_margin   = config.error_bound * 0.05;
    available_for_quant   -= safety_margin;
    double min_eb_for_int32 = data_range / 4.0e9;

    double effective_eb;
    if (available_for_quant <= 0) {
        double min_achievable = fmax(float_repr_error, min_eb_for_int32);
        (void)min_achievable;
        effective_eb = fmax(min_eb_for_int32, float_repr_error * 0.1);
    } else {
        effective_eb = available_for_quant;
    }
    effective_eb = fmax(effective_eb, min_eb_for_int32);

    int precision;
    if (config.precision == QuantizationPrecision::AUTO) {
        precision = compute_required_precision(data_range, effective_eb);
    } else {
        switch (config.precision) {
            case QuantizationPrecision::INT8:  precision = 8;  break;
            case QuantizationPrecision::INT16: precision = 16; break;
            case QuantizationPrecision::INT32: precision = 32; break;
            default: precision = 32;
        }
    }

    // Warn if forced INT8 cannot represent the data range
    if (precision == 8 && data_range > 0) {
        double max_quant = data_range / (2.0 * config.error_bound);
        if (max_quant > 127.0)
            fprintf(stderr, "gpucompress WARNING: forced INT8 cannot represent data range "
                    "(need %.0f levels, int8 max=127). Error bound may be violated.\n", max_quant);
    }

    double scale = 1.0 / (2.0 * effective_eb);

    size_t output_bytes = num_elements * precision_to_bytes(precision);
    void* d_output;
    if (cudaMalloc(&d_output, output_bytes) != cudaSuccess) {
        fprintf(stderr, "quantize_simple: Failed to allocate output buffer\n");
        return result;
    }

    if (element_size == 4) {
        if (precision == 8)
            launch_quantize_kernel<float, int8_t>(
                static_cast<float*>(d_input), static_cast<int8_t*>(d_output),
                num_elements, scale, data_min, stream);
        else if (precision == 16)
            launch_quantize_kernel<float, int16_t>(
                static_cast<float*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, scale, data_min, stream);
        else
            launch_quantize_kernel<float, int32_t>(
                static_cast<float*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, scale, data_min, stream);
    } else {
        if (precision == 8)
            launch_quantize_kernel<double, int8_t>(
                static_cast<double*>(d_input), static_cast<int8_t*>(d_output),
                num_elements, scale, data_min, stream);
        else if (precision == 16)
            launch_quantize_kernel<double, int16_t>(
                static_cast<double*>(d_input), static_cast<int16_t*>(d_output),
                num_elements, scale, data_min, stream);
        else
            launch_quantize_kernel<double, int32_t>(
                static_cast<double*>(d_input), static_cast<int32_t*>(d_output),
                num_elements, scale, data_min, stream);
    }

    result.d_quantized          = d_output;
    result.quantized_bytes      = output_bytes;
    result.actual_precision     = precision;
    result.data_min             = data_min;
    result.data_max             = data_max;
    result.scale_factor         = scale;
    result.error_bound          = config.error_bound;
    result.type                 = config.type;
    result.num_elements         = num_elements;
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

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "gpucompress ERROR: dequantization kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
        cudaFree(d_output);
        return nullptr;
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
    GC_LOG("[XFER H→D] verify error bound: init violations=0 (%zu B)\n", sizeof(int));
    XFER_TRACK("verify_error_bound: H->D init violations=0", sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_violations, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    GC_LOG("[XFER H→D] verify error bound: init max_error=0 (%zu B)\n", sizeof(double));
    XFER_TRACK("verify_error_bound: H->D init max_error=0", sizeof(double), cudaMemcpyHostToDevice);
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
    GC_LOG("[XFER D→H] verify error bound: violation count (%zu B)\n", sizeof(int));
    XFER_TRACK("verify_error_bound: D->H violation count", sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&h_violations, d_violations, sizeof(int), cudaMemcpyDeviceToHost, stream);
    GC_LOG("[XFER D→H] verify error bound: max_error (%zu B)\n", sizeof(double));
    XFER_TRACK("verify_error_bound: D->H max_error", sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_violations);
    cudaFree(d_max_error);

    if (max_error_out) {
        *max_error_out = h_max_error;
    }

    return (h_violations == 0);
}
