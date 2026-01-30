/**
 * @file test_quantization_suite.cu
 * @brief Comprehensive test suite for GPUCompress quantization preprocessing
 *
 * Tests LINEAR quantization method for:
 * - Correctness (error bound guarantee)
 * - Precision selection (int8/16/32)
 * - Data loss metrics (max error, RMSE, PSNR, correlation)
 * - Compression ratio improvement
 * - Edge cases
 *
 * Usage:
 *   ./test_quantization --unit          # Run unit tests only
 *   ./test_quantization --integration   # Run integration tests only
 *   ./test_quantization --compare       # Run comparison tests
 *   ./test_quantization --files         # Run tests on input data files
 *   ./test_quantization --all           # Run all tests (default)
 *   ./test_quantization --csv <file>    # Output CSV report
 *   ./test_quantization --data-dir <dir> # Set test data directory (default: test_data/)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

// Include GPUCompress headers
#include "../../src/quantization.cuh"
#include "../../src/byte_shuffle.cuh"
#include "../../src/compression_header.h"

// ============================================================================
// Test Framework Macros and Utilities
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "  ASSERT FAILED: %s\n", msg); \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance, msg) \
    do { \
        double _a = (actual); \
        double _e = (expected); \
        double _t = (tolerance); \
        if (fabs(_a - _e) > _t) { \
            fprintf(stderr, "  ASSERT_NEAR FAILED: %s (actual=%.6e, expected=%.6e, tol=%.6e)\n", \
                    msg, _a, _e, _t); \
            return false; \
        } \
    } while(0)

// Block size for CUDA kernels
#define BLOCK_SIZE 256

// ============================================================================
// Atomic Add for Double (for older GPU architectures)
// ============================================================================

__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// ============================================================================
// Test Metrics Structure
// ============================================================================

struct TestMetrics {
    // Sizes
    size_t original_bytes;
    size_t quantized_bytes;
    size_t compressed_bytes;
    size_t decompressed_bytes;

    // Ratios
    double quantization_ratio;  // original / quantized
    double compression_ratio;   // quantized / compressed
    double total_ratio;         // original / compressed

    // Error metrics
    double max_error;
    double mean_error;
    double rmse;
    double psnr_db;
    double correlation;

    // Verification
    bool error_bound_satisfied;
    bool size_matches;
    int precision_bits;

    TestMetrics() :
        original_bytes(0), quantized_bytes(0), compressed_bytes(0), decompressed_bytes(0),
        quantization_ratio(0), compression_ratio(0), total_ratio(0),
        max_error(0), mean_error(0), rmse(0), psnr_db(0), correlation(0),
        error_bound_satisfied(false), size_matches(false), precision_bits(0) {}
};

// ============================================================================
// Data Pattern Generation Kernels (reused from generate_test_data.cu)
// ============================================================================

__global__ void generate_smooth_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float t = (float)idx / (float)num_elements;
    float value = 1000.0f * sinf(t * 2.0f * M_PI * 5.0f);
    value += curand_normal(&state) * 10.0f;  // 1% noise
    data[idx] = value;
}

__global__ void generate_periodic_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float t = (float)idx / 1000.0f;
    float value = 500.0f + 200.0f * sinf(t * 2.0f * M_PI);
    value += 50.0f * sinf(t * 10.0f * M_PI);  // Higher frequency component
    value += curand_normal(&state) * 5.0f;
    data[idx] = value;
}

__global__ void generate_noisy_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float value = curand_normal(&state) * 200.0f + 500.0f;
    data[idx] = value;
}

__global__ void generate_random_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    data[idx] = curand_uniform(&state) * 1000.0f;
}

__global__ void generate_constant_pattern_kernel(
    float* data,
    size_t num_elements,
    float constant_value
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    data[idx] = constant_value;
}

__global__ void generate_gradient_pattern_kernel(
    float* data,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Linear gradient from 0 to 1000
    data[idx] = (float)idx / (float)num_elements * 1000.0f;
}

__global__ void generate_negative_values_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Mix of positive and negative values
    data[idx] = curand_normal(&state) * 500.0f;  // Mean 0, stdev 500
}

__global__ void generate_wide_range_pattern_kernel(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Wide range: [-1e6, 1e6]
    data[idx] = (curand_uniform(&state) - 0.5f) * 2e6f;
}

// Double precision patterns
__global__ void generate_smooth_pattern_double_kernel(
    double* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double t = (double)idx / (double)num_elements;
    double value = 1000.0 * sin(t * 2.0 * M_PI * 5.0);
    value += curand_normal(&state) * 10.0;
    data[idx] = value;
}

// ============================================================================
// Error Metrics Computation Kernel
// ============================================================================

__global__ void compute_error_metrics_kernel(
    const float* original,
    const float* restored,
    size_t num_elements,
    double* d_sum_error,
    double* d_sum_sq_error,
    double* d_sum_orig,
    double* d_sum_rest,
    double* d_sum_orig_sq,
    double* d_sum_rest_sq,
    double* d_sum_orig_rest,
    double* d_max_error,
    float* d_min_val,
    float* d_max_val
) {
    __shared__ double s_sum_error[BLOCK_SIZE];
    __shared__ double s_sum_sq_error[BLOCK_SIZE];
    __shared__ double s_max_error[BLOCK_SIZE];
    __shared__ double s_sum_orig[BLOCK_SIZE];
    __shared__ double s_sum_rest[BLOCK_SIZE];
    __shared__ double s_sum_orig_sq[BLOCK_SIZE];
    __shared__ double s_sum_rest_sq[BLOCK_SIZE];
    __shared__ double s_sum_orig_rest[BLOCK_SIZE];
    __shared__ float s_min_val[BLOCK_SIZE];
    __shared__ float s_max_val[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    double local_sum_error = 0.0;
    double local_sum_sq_error = 0.0;
    double local_max_error = 0.0;
    double local_sum_orig = 0.0;
    double local_sum_rest = 0.0;
    double local_sum_orig_sq = 0.0;
    double local_sum_rest_sq = 0.0;
    double local_sum_orig_rest = 0.0;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (size_t i = idx; i < num_elements; i += stride) {
        float orig = original[i];
        float rest = restored[i];
        double err = fabs((double)orig - (double)rest);

        local_sum_error += err;
        local_sum_sq_error += err * err;
        local_max_error = fmax(local_max_error, err);

        local_sum_orig += orig;
        local_sum_rest += rest;
        local_sum_orig_sq += (double)orig * orig;
        local_sum_rest_sq += (double)rest * rest;
        local_sum_orig_rest += (double)orig * rest;

        local_min = fminf(local_min, orig);
        local_max = fmaxf(local_max, orig);
    }

    s_sum_error[threadIdx.x] = local_sum_error;
    s_sum_sq_error[threadIdx.x] = local_sum_sq_error;
    s_max_error[threadIdx.x] = local_max_error;
    s_sum_orig[threadIdx.x] = local_sum_orig;
    s_sum_rest[threadIdx.x] = local_sum_rest;
    s_sum_orig_sq[threadIdx.x] = local_sum_orig_sq;
    s_sum_rest_sq[threadIdx.x] = local_sum_rest_sq;
    s_sum_orig_rest[threadIdx.x] = local_sum_orig_rest;
    s_min_val[threadIdx.x] = local_min;
    s_max_val[threadIdx.x] = local_max;

    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_error[threadIdx.x] += s_sum_error[threadIdx.x + s];
            s_sum_sq_error[threadIdx.x] += s_sum_sq_error[threadIdx.x + s];
            s_max_error[threadIdx.x] = fmax(s_max_error[threadIdx.x], s_max_error[threadIdx.x + s]);
            s_sum_orig[threadIdx.x] += s_sum_orig[threadIdx.x + s];
            s_sum_rest[threadIdx.x] += s_sum_rest[threadIdx.x + s];
            s_sum_orig_sq[threadIdx.x] += s_sum_orig_sq[threadIdx.x + s];
            s_sum_rest_sq[threadIdx.x] += s_sum_rest_sq[threadIdx.x + s];
            s_sum_orig_rest[threadIdx.x] += s_sum_orig_rest[threadIdx.x + s];
            s_min_val[threadIdx.x] = fminf(s_min_val[threadIdx.x], s_min_val[threadIdx.x + s]);
            s_max_val[threadIdx.x] = fmaxf(s_max_val[threadIdx.x], s_max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(d_sum_error, s_sum_error[0]);
        atomicAddDouble(d_sum_sq_error, s_sum_sq_error[0]);
        atomicAddDouble(d_sum_orig, s_sum_orig[0]);
        atomicAddDouble(d_sum_rest, s_sum_rest[0]);
        atomicAddDouble(d_sum_orig_sq, s_sum_orig_sq[0]);
        atomicAddDouble(d_sum_rest_sq, s_sum_rest_sq[0]);
        atomicAddDouble(d_sum_orig_rest, s_sum_orig_rest[0]);

        // Atomic max for double
        unsigned long long* max_err_ull = (unsigned long long*)d_max_error;
        unsigned long long old = *max_err_ull;
        while (__longlong_as_double(old) < s_max_error[0]) {
            unsigned long long assumed = old;
            old = atomicCAS(max_err_ull, assumed, __double_as_longlong(s_max_error[0]));
            if (old == assumed) break;
        }

        // Atomic min/max for float
        unsigned int* min_as_uint = (unsigned int*)d_min_val;
        unsigned int old_min = *min_as_uint;
        while (__uint_as_float(old_min) > s_min_val[0]) {
            unsigned int assumed = old_min;
            old_min = atomicCAS(min_as_uint, assumed, __float_as_uint(s_min_val[0]));
            if (old_min == assumed) break;
        }

        unsigned int* max_as_uint = (unsigned int*)d_max_val;
        unsigned int old_max = *max_as_uint;
        while (__uint_as_float(old_max) < s_max_val[0]) {
            unsigned int assumed = old_max;
            old_max = atomicCAS(max_as_uint, assumed, __float_as_uint(s_max_val[0]));
            if (old_max == assumed) break;
        }
    }
}

// ============================================================================
// Host-side Error Metrics Computation
// ============================================================================

TestMetrics compute_metrics_gpu(
    const float* d_original,
    const float* d_restored,
    size_t num_elements,
    double error_bound,
    cudaStream_t stream = 0
) {
    TestMetrics metrics;
    metrics.original_bytes = num_elements * sizeof(float);
    metrics.decompressed_bytes = num_elements * sizeof(float);
    metrics.size_matches = true;

    // Allocate reduction buffers
    double *d_sum_error, *d_sum_sq_error, *d_sum_orig, *d_sum_rest;
    double *d_sum_orig_sq, *d_sum_rest_sq, *d_sum_orig_rest, *d_max_error;
    float *d_min_val, *d_max_val;

    cudaMalloc(&d_sum_error, sizeof(double));
    cudaMalloc(&d_sum_sq_error, sizeof(double));
    cudaMalloc(&d_sum_orig, sizeof(double));
    cudaMalloc(&d_sum_rest, sizeof(double));
    cudaMalloc(&d_sum_orig_sq, sizeof(double));
    cudaMalloc(&d_sum_rest_sq, sizeof(double));
    cudaMalloc(&d_sum_orig_rest, sizeof(double));
    cudaMalloc(&d_max_error, sizeof(double));
    cudaMalloc(&d_min_val, sizeof(float));
    cudaMalloc(&d_max_val, sizeof(float));

    // Initialize
    double zero = 0.0;
    float init_min = FLT_MAX;
    float init_max = -FLT_MAX;
    cudaMemcpyAsync(d_sum_error, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_sq_error, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_orig, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_rest, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_orig_sq, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_rest_sq, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_orig_rest, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max_error, &zero, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_min_val, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max_val, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    int num_blocks = std::min((int)((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);
    compute_error_metrics_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_original, d_restored, num_elements,
        d_sum_error, d_sum_sq_error, d_sum_orig, d_sum_rest,
        d_sum_orig_sq, d_sum_rest_sq, d_sum_orig_rest, d_max_error,
        d_min_val, d_max_val
    );

    // Copy back results
    double h_sum_error, h_sum_sq_error, h_sum_orig, h_sum_rest;
    double h_sum_orig_sq, h_sum_rest_sq, h_sum_orig_rest, h_max_error;
    float h_min_val, h_max_val;

    cudaMemcpyAsync(&h_sum_error, d_sum_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_sq_error, d_sum_sq_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_orig, d_sum_orig, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_rest, d_sum_rest, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_orig_sq, d_sum_orig_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_rest_sq, d_sum_rest_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_orig_rest, d_sum_orig_rest, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max_error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_min_val, d_min_val, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Compute final metrics
    double n = (double)num_elements;
    metrics.max_error = h_max_error;
    metrics.mean_error = h_sum_error / n;
    metrics.rmse = sqrt(h_sum_sq_error / n);

    // PSNR
    double data_range = (double)(h_max_val - h_min_val);
    if (data_range > 0 && metrics.rmse > 0) {
        metrics.psnr_db = 20.0 * log10(data_range / metrics.rmse);
    } else if (metrics.rmse == 0) {
        metrics.psnr_db = INFINITY;
    } else {
        metrics.psnr_db = 0.0;
    }

    // Pearson correlation
    double mean_orig = h_sum_orig / n;
    double mean_rest = h_sum_rest / n;
    double numerator = h_sum_orig_rest - n * mean_orig * mean_rest;
    double denom_orig = sqrt(h_sum_orig_sq - n * mean_orig * mean_orig);
    double denom_rest = sqrt(h_sum_rest_sq - n * mean_rest * mean_rest);

    if (denom_orig > 0 && denom_rest > 0) {
        metrics.correlation = numerator / (denom_orig * denom_rest);
    } else {
        metrics.correlation = 1.0;  // Constant data
    }

    // Error bound check
    metrics.error_bound_satisfied = (metrics.max_error <= error_bound);

    // Cleanup
    cudaFree(d_sum_error);
    cudaFree(d_sum_sq_error);
    cudaFree(d_sum_orig);
    cudaFree(d_sum_rest);
    cudaFree(d_sum_orig_sq);
    cudaFree(d_sum_rest_sq);
    cudaFree(d_sum_orig_rest);
    cudaFree(d_max_error);
    cudaFree(d_min_val);
    cudaFree(d_max_val);

    return metrics;
}

// ============================================================================
// Data Generation Helper Functions
// ============================================================================

enum class DataPattern {
    SMOOTH,
    PERIODIC,
    NOISY,
    RANDOM,
    CONSTANT,
    GRADIENT,
    NEGATIVE,
    WIDE_RANGE
};

const char* pattern_name(DataPattern p) {
    switch (p) {
        case DataPattern::SMOOTH: return "smooth";
        case DataPattern::PERIODIC: return "periodic";
        case DataPattern::NOISY: return "noisy";
        case DataPattern::RANDOM: return "random";
        case DataPattern::CONSTANT: return "constant";
        case DataPattern::GRADIENT: return "gradient";
        case DataPattern::NEGATIVE: return "negative";
        case DataPattern::WIDE_RANGE: return "wide_range";
        default: return "unknown";
    }
}

void generate_test_data(
    float* d_data,
    size_t num_elements,
    DataPattern pattern,
    cudaStream_t stream = 0
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned long seed = 12345;

    switch (pattern) {
        case DataPattern::SMOOTH:
            generate_smooth_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
        case DataPattern::PERIODIC:
            generate_periodic_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
        case DataPattern::NOISY:
            generate_noisy_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
        case DataPattern::RANDOM:
            generate_random_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
        case DataPattern::CONSTANT:
            generate_constant_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, 500.0f);
            break;
        case DataPattern::GRADIENT:
            generate_gradient_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements);
            break;
        case DataPattern::NEGATIVE:
            generate_negative_values_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
        case DataPattern::WIDE_RANGE:
            generate_wide_range_pattern_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, num_elements, seed);
            break;
    }
    cudaStreamSynchronize(stream);
}

void generate_test_data_double(
    double* d_data,
    size_t num_elements,
    cudaStream_t stream = 0
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_smooth_pattern_double_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_data, num_elements, 12345);
    cudaStreamSynchronize(stream);
}

// ============================================================================
// File-Based Data Loading
// ============================================================================

// Default test data directory (relative to executable or project root)
std::string g_test_data_dir = "test_data";

/**
 * @brief Load binary float data from file to GPU
 *
 * @param filepath Path to binary file containing float32 data
 * @param d_data Output: device pointer (allocated by this function)
 * @param num_elements Output: number of elements loaded
 * @return true if successful, false on error
 */
bool load_binary_file_to_gpu(
    const std::string& filepath,
    float** d_data,
    size_t* num_elements
) {
    // Open file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "Error: Cannot open file: %s\n", filepath.c_str());
        return false;
    }

    // Get file size
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == 0) {
        fprintf(stderr, "Error: Empty file: %s\n", filepath.c_str());
        return false;
    }

    if (file_size % sizeof(float) != 0) {
        fprintf(stderr, "Error: File size not multiple of float size: %s\n", filepath.c_str());
        return false;
    }

    *num_elements = file_size / sizeof(float);

    // Allocate host buffer
    std::vector<float> h_data(*num_elements);

    // Read file
    file.read(reinterpret_cast<char*>(h_data.data()), file_size);
    if (!file) {
        fprintf(stderr, "Error: Failed to read file: %s\n", filepath.c_str());
        return false;
    }
    file.close();

    // Allocate device memory and copy
    cudaError_t err = cudaMalloc(d_data, file_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to allocate GPU memory for file data\n");
        return false;
    }

    err = cudaMemcpy(*d_data, h_data.data(), file_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to copy file data to GPU\n");
        cudaFree(*d_data);
        return false;
    }

    return true;
}

/**
 * @brief Get list of test data files in directory
 */
std::vector<std::string> get_test_data_files(const std::string& dir) {
    std::vector<std::string> files;

    // Check for known test files
    std::vector<std::string> known_files = {
        "test_float32_smooth.bin",
        "test_float32_periodic.bin",
        "test_float32_noisy.bin"
    };

    for (const auto& name : known_files) {
        std::string path = dir + "/" + name;
        std::ifstream f(path);
        if (f.good()) {
            files.push_back(path);
        }
    }

    return files;
}

/**
 * @brief Extract pattern name from filename
 */
std::string extract_pattern_from_filename(const std::string& filepath) {
    // Extract filename from path
    size_t last_slash = filepath.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos)
        ? filepath.substr(last_slash + 1)
        : filepath;

    // Extract pattern (e.g., "test_float32_smooth.bin" -> "smooth")
    if (filename.find("smooth") != std::string::npos) return "smooth";
    if (filename.find("periodic") != std::string::npos) return "periodic";
    if (filename.find("noisy") != std::string::npos) return "noisy";
    if (filename.find("random") != std::string::npos) return "random";
    if (filename.find("gradient") != std::string::npos) return "gradient";

    return "file";
}

// ============================================================================
// Test Result Tracking
// ============================================================================

struct TestResult {
    std::string test_name;
    std::string method;
    double error_bound;
    std::string pattern;
    std::string algorithm;  // compression algorithm (if integration test)
    TestMetrics metrics;
    bool passed;
    std::string failure_reason;
};

std::vector<TestResult> g_test_results;
int g_tests_passed = 0;
int g_tests_failed = 0;

void print_test_result(const TestResult& result) {
    printf("TEST: %s\n", result.test_name.c_str());
    printf("  Config: %s, eb=%.2e, pattern=%s",
           result.method.c_str(), result.error_bound, result.pattern.c_str());
    if (!result.algorithm.empty()) {
        printf(", algo=%s", result.algorithm.c_str());
    }
    printf(", n=%zu\n", result.metrics.original_bytes / sizeof(float));

    if (result.metrics.quantization_ratio > 0) {
        printf("  Quant: %d-bit precision, %.2fx reduction\n",
               result.metrics.precision_bits, result.metrics.quantization_ratio);
    }
    if (result.metrics.compression_ratio > 0) {
        printf("  Compress: %.2fx, Total: %.2fx\n",
               result.metrics.compression_ratio, result.metrics.total_ratio);
    }

    const char* pass_str = result.metrics.error_bound_satisfied ? "PASS" : "FAIL";
    printf("  Error: max=%.2e [%s], mean=%.2e, RMSE=%.2e, PSNR=%.2fdB, corr=%.6f\n",
           result.metrics.max_error, pass_str,
           result.metrics.mean_error, result.metrics.rmse,
           result.metrics.psnr_db, result.metrics.correlation);

    if (result.metrics.size_matches) {
        printf("  Size: decompressed=%zu [MATCHES original]\n", result.metrics.decompressed_bytes);
    } else {
        printf("  Size: decompressed=%zu [MISMATCH! original=%zu]\n",
               result.metrics.decompressed_bytes, result.metrics.original_bytes);
    }

    if (result.passed) {
        printf("  Result: PASSED\n");
    } else {
        printf("  Result: FAILED (%s)\n", result.failure_reason.c_str());
    }
    printf("\n");
}

void record_test(const TestResult& result) {
    g_test_results.push_back(result);
    if (result.passed) {
        g_tests_passed++;
    } else {
        g_tests_failed++;
    }
    print_test_result(result);
}

// ============================================================================
// Unit Tests: Quantization Correctness
// ============================================================================

bool test_quantization_roundtrip(
    QuantizationType qtype,
    DataPattern pattern,
    double error_bound,
    size_t num_elements = 262144  // 1MB as float32
) {
    const char* method_name = getQuantizationTypeName(qtype);
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "test_%s_roundtrip_%s_eb%.0e",
             method_name, pattern_name(pattern), error_bound);

    TestResult result;
    result.test_name = test_name;
    result.method = method_name;
    result.error_bound = error_bound;
    result.pattern = pattern_name(pattern);
    result.passed = false;

    // Allocate GPU memory
    float* d_original;
    cudaMalloc(&d_original, num_elements * sizeof(float));

    // Generate test data
    generate_test_data(d_original, num_elements, pattern);

    // Configure quantization
    QuantizationConfig config(qtype, error_bound, num_elements, sizeof(float));

    // Quantize
    QuantizationResult quant_result = quantize_simple(d_original, num_elements, sizeof(float), config);
    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        record_test(result);
        cudaFree(d_original);
        return false;
    }

    // Dequantize
    void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        record_test(result);
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        return false;
    }

    // Compute metrics
    result.metrics = compute_metrics_gpu(d_original, (float*)d_restored, num_elements, error_bound);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.quantization_ratio = quant_result.getQuantizationRatio();
    result.metrics.precision_bits = quant_result.actual_precision;

    // Verify error bound
    if (result.metrics.error_bound_satisfied) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "max_error %.6e > error_bound %.6e",
                 result.metrics.max_error, error_bound);
        result.failure_reason = reason;
    }

    // Cleanup
    cudaFree(d_restored);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_original);

    record_test(result);
    return result.passed;
}

bool test_precision_selection(int expected_precision, double error_bound, double data_range_hint) {
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "test_precision_int%d", expected_precision);

    TestResult result;
    result.test_name = test_name;
    result.method = "LINEAR";
    result.error_bound = error_bound;
    result.passed = false;

    // Create data with specific range to force precision selection
    size_t num_elements = 1024;
    float* d_data;
    cudaMalloc(&d_data, num_elements * sizeof(float));

    // Fill with values that span the target range
    std::vector<float> h_data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = (float)i / (float)num_elements * data_range_hint;
    }
    cudaMemcpy(d_data, h_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Quantize
    QuantizationConfig config(QuantizationType::LINEAR, error_bound, num_elements, sizeof(float));
    QuantizationResult quant_result = quantize_simple(d_data, num_elements, sizeof(float), config);

    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        cudaFree(d_data);
        record_test(result);
        return false;
    }

    result.metrics.precision_bits = quant_result.actual_precision;
    result.pattern = "range_test";

    if (quant_result.actual_precision == expected_precision) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "Expected %d-bit precision, got %d-bit",
                 expected_precision, quant_result.actual_precision);
        result.failure_reason = reason;
    }

    cudaFree(quant_result.d_quantized);
    cudaFree(d_data);

    record_test(result);
    return result.passed;
}

bool test_double_support() {
    TestResult result;
    result.test_name = "test_double_support";
    result.method = "LINEAR";
    result.error_bound = 0.001;
    result.pattern = "smooth";
    result.passed = false;

    size_t num_elements = 262144;  // 2MB as double
    double* d_original;
    cudaMalloc(&d_original, num_elements * sizeof(double));

    // Generate double-precision test data
    generate_test_data_double(d_original, num_elements);

    // Configure quantization
    QuantizationConfig config(QuantizationType::LINEAR, 0.001, num_elements, sizeof(double));

    // Quantize
    QuantizationResult quant_result = quantize_simple(d_original, num_elements, sizeof(double), config);
    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Dequantize
    void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Verify error bound (use verify_error_bound function)
    double max_error;
    bool within_bound = verify_error_bound(
        d_original, d_restored, num_elements, sizeof(double),
        0.001, 0, &max_error
    );

    result.metrics.original_bytes = num_elements * sizeof(double);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.decompressed_bytes = num_elements * sizeof(double);
    result.metrics.max_error = max_error;
    result.metrics.error_bound_satisfied = within_bound;
    result.metrics.precision_bits = quant_result.actual_precision;
    result.metrics.quantization_ratio = quant_result.getQuantizationRatio();
    result.metrics.size_matches = true;

    if (within_bound) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "max_error %.6e > error_bound 0.001", max_error);
        result.failure_reason = reason;
    }

    cudaFree(d_restored);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_original);

    record_test(result);
    return result.passed;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

bool test_single_element() {
    TestResult result;
    result.test_name = "test_single_element";
    result.method = "LINEAR";
    result.error_bound = 0.01;
    result.pattern = "single";
    result.passed = false;

    float* d_data;
    cudaMalloc(&d_data, sizeof(float));
    float h_val = 123.456f;
    cudaMemcpy(d_data, &h_val, sizeof(float), cudaMemcpyHostToDevice);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, 1, sizeof(float));
    QuantizationResult quant_result = quantize_simple(d_data, 1, sizeof(float), config);

    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        cudaFree(d_data);
        record_test(result);
        return false;
    }

    void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        cudaFree(quant_result.d_quantized);
        cudaFree(d_data);
        record_test(result);
        return false;
    }

    float h_restored;
    cudaMemcpy(&h_restored, d_restored, sizeof(float), cudaMemcpyDeviceToHost);

    double error = fabs((double)h_val - (double)h_restored);
    result.metrics.original_bytes = sizeof(float);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.decompressed_bytes = sizeof(float);
    result.metrics.max_error = error;
    result.metrics.error_bound_satisfied = (error <= 0.01);
    result.metrics.precision_bits = quant_result.actual_precision;
    result.metrics.size_matches = true;

    if (result.metrics.error_bound_satisfied) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "Error %.6e > bound 0.01", error);
        result.failure_reason = reason;
    }

    cudaFree(d_restored);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_data);

    record_test(result);
    return result.passed;
}

bool test_partial_block() {
    // Test with small number of elements
    TestResult result;
    result.test_name = "test_partial_block";
    result.method = "LINEAR";
    result.error_bound = 0.01;
    result.pattern = "gradient";
    result.passed = false;

    size_t num_elements = 5;
    float* d_data;
    cudaMalloc(&d_data, num_elements * sizeof(float));

    std::vector<float> h_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    cudaMemcpy(d_data, h_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    QuantizationConfig config(QuantizationType::LINEAR, 0.01, num_elements, sizeof(float));
    QuantizationResult quant_result = quantize_simple(d_data, num_elements, sizeof(float), config);

    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        cudaFree(d_data);
        record_test(result);
        return false;
    }

    void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        cudaFree(quant_result.d_quantized);
        cudaFree(d_data);
        record_test(result);
        return false;
    }

    // Verify all elements
    std::vector<float> h_restored(num_elements);
    cudaMemcpy(h_restored.data(), d_restored, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        double err = fabs((double)h_data[i] - (double)h_restored[i]);
        max_err = std::max(max_err, err);
    }

    result.metrics.original_bytes = num_elements * sizeof(float);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.decompressed_bytes = num_elements * sizeof(float);
    result.metrics.max_error = max_err;
    result.metrics.error_bound_satisfied = (max_err <= 0.01);
    result.metrics.precision_bits = quant_result.actual_precision;
    result.metrics.size_matches = true;

    if (result.metrics.error_bound_satisfied) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "max_error %.6e > bound 0.01", max_err);
        result.failure_reason = reason;
    }

    cudaFree(d_restored);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_data);

    record_test(result);
    return result.passed;
}

bool test_constant_data() {
    return test_quantization_roundtrip(QuantizationType::LINEAR, DataPattern::CONSTANT, 0.001, 10000);
}

bool test_negative_values() {
    return test_quantization_roundtrip(QuantizationType::LINEAR, DataPattern::NEGATIVE, 0.01, 100000);
}

bool test_wide_range() {
    return test_quantization_roundtrip(QuantizationType::LINEAR, DataPattern::WIDE_RANGE, 1.0, 100000);
}

bool test_tight_error_bound() {
    return test_quantization_roundtrip(QuantizationType::LINEAR, DataPattern::SMOOTH, 1e-5, 10000);
}

// ============================================================================
// Integration Tests: Quantization + Byte Shuffle
// ============================================================================

bool test_quantization_with_shuffle(
    QuantizationType qtype,
    DataPattern pattern,
    double error_bound,
    size_t num_elements = 262144
) {
    const char* method_name = getQuantizationTypeName(qtype);
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "test_%s_with_shuffle_%s",
             method_name, pattern_name(pattern));

    TestResult result;
    result.test_name = test_name;
    result.method = method_name;
    result.error_bound = error_bound;
    result.pattern = pattern_name(pattern);
    result.algorithm = "shuffle";
    result.passed = false;

    // Allocate GPU memory
    float* d_original;
    cudaMalloc(&d_original, num_elements * sizeof(float));

    // Generate test data
    generate_test_data(d_original, num_elements, pattern);

    // Step 1: Quantize
    QuantizationConfig config(qtype, error_bound, num_elements, sizeof(float));
    QuantizationResult quant_result = quantize_simple(d_original, num_elements, sizeof(float), config);
    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Step 2: Byte shuffle the quantized data
    unsigned int quant_elem_size = precision_to_bytes(quant_result.actual_precision);
    uint8_t* d_shuffled = byte_shuffle_simple(
        quant_result.d_quantized,
        quant_result.quantized_bytes,
        quant_elem_size
    );
    if (d_shuffled == nullptr) {
        result.failure_reason = "Shuffle failed";
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Step 3: Unshuffle
    uint8_t* d_unshuffled = byte_unshuffle_simple(
        d_shuffled,
        quant_result.quantized_bytes,
        quant_elem_size
    );
    if (d_unshuffled == nullptr) {
        result.failure_reason = "Unshuffle failed";
        cudaFree(d_shuffled);
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Step 4: Dequantize (create a copy of quant_result with new pointer)
    QuantizationResult quant_meta = quant_result;
    quant_meta.d_quantized = d_unshuffled;

    void* d_restored = dequantize_simple(d_unshuffled, quant_meta);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        cudaFree(d_unshuffled);
        cudaFree(d_shuffled);
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        record_test(result);
        return false;
    }

    // Compute metrics
    result.metrics = compute_metrics_gpu(d_original, (float*)d_restored, num_elements, error_bound);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.quantization_ratio = quant_result.getQuantizationRatio();
    result.metrics.precision_bits = quant_result.actual_precision;

    // Verify error bound
    if (result.metrics.error_bound_satisfied) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "max_error %.6e > error_bound %.6e",
                 result.metrics.max_error, error_bound);
        result.failure_reason = reason;
    }

    // Cleanup
    cudaFree(d_restored);
    cudaFree(d_unshuffled);
    cudaFree(d_shuffled);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_original);

    record_test(result);
    return result.passed;
}

// ============================================================================
// Comparison Tests: WITH vs WITHOUT Quantization
// ============================================================================

struct ComparisonResult {
    std::string pattern;
    double error_bound;
    std::string quant_method;

    // Without quantization
    size_t baseline_original_bytes;
    size_t baseline_quantized_bytes;  // same as original (no quantization)

    // With quantization
    size_t quant_original_bytes;
    size_t quant_quantized_bytes;
    int quant_precision;
    double quant_ratio;

    double max_error;
    double mean_error;
    double rmse;
    double psnr_db;
};

void run_comparison_test(
    DataPattern pattern,
    double error_bound,
    QuantizationType qtype,
    size_t num_elements,
    std::vector<ComparisonResult>& results
) {
    ComparisonResult comp;
    comp.pattern = pattern_name(pattern);
    comp.error_bound = error_bound;
    comp.quant_method = getQuantizationTypeName(qtype);

    // Allocate and generate data
    float* d_data;
    cudaMalloc(&d_data, num_elements * sizeof(float));
    generate_test_data(d_data, num_elements, pattern);

    // Baseline: no quantization
    comp.baseline_original_bytes = num_elements * sizeof(float);
    comp.baseline_quantized_bytes = num_elements * sizeof(float);  // No reduction

    // With quantization
    QuantizationConfig config(qtype, error_bound, num_elements, sizeof(float));
    QuantizationResult quant_result = quantize_simple(d_data, num_elements, sizeof(float), config);

    if (quant_result.isValid()) {
        comp.quant_original_bytes = num_elements * sizeof(float);
        comp.quant_quantized_bytes = quant_result.quantized_bytes;
        comp.quant_precision = quant_result.actual_precision;
        comp.quant_ratio = quant_result.getQuantizationRatio();

        // Dequantize and measure error
        void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
        if (d_restored) {
            TestMetrics metrics = compute_metrics_gpu(d_data, (float*)d_restored, num_elements, error_bound);
            comp.max_error = metrics.max_error;
            comp.mean_error = metrics.mean_error;
            comp.rmse = metrics.rmse;
            comp.psnr_db = metrics.psnr_db;
            cudaFree(d_restored);
        }
        cudaFree(quant_result.d_quantized);
    }

    cudaFree(d_data);
    results.push_back(comp);
}

void print_comparison_results(const std::vector<ComparisonResult>& results) {
    printf("\n");
    printf("==============================================================================\n");
    printf("                     QUANTIZATION COMPARISON RESULTS\n");
    printf("==============================================================================\n");
    printf("%-10s %-8s %-15s %-6s %-8s %-12s %-12s %-8s\n",
           "Pattern", "EB", "Method", "Bits", "QRatio", "MaxErr", "RMSE", "PSNR(dB)");
    printf("------------------------------------------------------------------------------\n");

    for (const auto& r : results) {
        printf("%-10s %.0e  %-15s %-6d %-8.2f %.6e %.6e %-8.2f\n",
               r.pattern.c_str(),
               r.error_bound,
               r.quant_method.c_str(),
               r.quant_precision,
               r.quant_ratio,
               r.max_error,
               r.rmse,
               r.psnr_db);
    }
    printf("==============================================================================\n\n");
}

// ============================================================================
// CSV Export
// ============================================================================

void export_csv(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        fprintf(stderr, "Error: Cannot open CSV file: %s\n", filename.c_str());
        return;
    }

    // Header
    out << "test_name,method,error_bound,pattern,algorithm,orig_bytes,quant_bytes,comp_bytes,"
        << "total_ratio,max_error,mean_error,rmse,psnr,correlation,pass\n";

    // Data
    for (const auto& r : g_test_results) {
        out << r.test_name << ","
            << r.method << ","
            << r.error_bound << ","
            << r.pattern << ","
            << r.algorithm << ","
            << r.metrics.original_bytes << ","
            << r.metrics.quantized_bytes << ","
            << r.metrics.compressed_bytes << ","
            << r.metrics.total_ratio << ","
            << r.metrics.max_error << ","
            << r.metrics.mean_error << ","
            << r.metrics.rmse << ","
            << r.metrics.psnr_db << ","
            << r.metrics.correlation << ","
            << (r.passed ? "true" : "false") << "\n";
    }

    out.close();
    printf("CSV report written to: %s\n", filename.c_str());
}

// ============================================================================
// Test Runners
// ============================================================================

void run_unit_tests() {
    printf("\n");
    printf("==============================================================================\n");
    printf("                           UNIT TESTS\n");
    printf("==============================================================================\n\n");

    // Test LINEAR quantization with different patterns and error bounds
    std::vector<QuantizationType> methods = {
        QuantizationType::LINEAR
    };

    std::vector<DataPattern> patterns = {
        DataPattern::SMOOTH,
        DataPattern::GRADIENT
    };

    std::vector<double> error_bounds = {0.1, 0.01, 0.001};

    printf("--- Quantization Roundtrip Tests ---\n\n");
    for (auto qtype : methods) {
        for (auto pattern : patterns) {
            for (double eb : error_bounds) {
                test_quantization_roundtrip(qtype, pattern, eb);
            }
        }
    }

    printf("--- Precision Selection Tests ---\n\n");
    // Test precision selection
    // int8: small range relative to error bound (range / 2*eb < 127)
    test_precision_selection(8, 0.1, 10.0);    // 10 / 0.2 = 50 bins -> int8

    // int16: medium range (127 < range / 2*eb < 32767)
    test_precision_selection(16, 0.01, 100.0); // 100 / 0.02 = 5000 bins -> int16

    // int32: large range (range / 2*eb > 32767)
    test_precision_selection(32, 0.0001, 100.0); // 100 / 0.0002 = 500000 bins -> int32

    printf("--- Double Precision Support Test ---\n\n");
    test_double_support();

    printf("--- Edge Case Tests ---\n\n");
    test_single_element();
    test_partial_block();
    test_constant_data();
    test_negative_values();
    test_wide_range();
    test_tight_error_bound();
}

void run_integration_tests() {
    printf("\n");
    printf("==============================================================================\n");
    printf("                        INTEGRATION TESTS\n");
    printf("==============================================================================\n\n");

    printf("--- Quantization + Byte Shuffle Tests ---\n\n");

    std::vector<DataPattern> patterns = {
        DataPattern::SMOOTH,
        DataPattern::PERIODIC
    };

    for (auto pattern : patterns) {
        test_quantization_with_shuffle(QuantizationType::LINEAR, pattern, 0.01);
    }
}

void run_comparison_tests() {
    printf("\n");
    printf("==============================================================================\n");
    printf("                        COMPARISON TESTS\n");
    printf("==============================================================================\n\n");

    std::vector<ComparisonResult> results;

    std::vector<DataPattern> patterns = {
        DataPattern::SMOOTH,
        DataPattern::PERIODIC,
        DataPattern::NOISY,
        DataPattern::RANDOM
    };

    std::vector<double> error_bounds = {0.1, 0.01, 0.001, 0.0001};

    size_t num_elements = 2621440;  // 10MB as float32

    printf("Running comparison tests (10MB data, %zu elements)...\n\n", num_elements);

    for (auto pattern : patterns) {
        for (double eb : error_bounds) {
            run_comparison_test(pattern, eb, QuantizationType::LINEAR, num_elements, results);
        }
    }

    print_comparison_results(results);
}

// ============================================================================
// File-Based Tests
// ============================================================================

/**
 * @brief Test quantization roundtrip on data loaded from file
 */
bool test_file_quantization_roundtrip(
    const std::string& filepath,
    QuantizationType qtype,
    double error_bound
) {
    const char* method_name = getQuantizationTypeName(qtype);
    std::string pattern = extract_pattern_from_filename(filepath);

    char test_name[512];
    snprintf(test_name, sizeof(test_name), "test_file_%s_%s_eb%.0e",
             pattern.c_str(), method_name, error_bound);

    TestResult result;
    result.test_name = test_name;
    result.method = method_name;
    result.error_bound = error_bound;
    result.pattern = pattern;
    result.passed = false;

    // Load data from file
    float* d_original = nullptr;
    size_t num_elements = 0;

    if (!load_binary_file_to_gpu(filepath, &d_original, &num_elements)) {
        result.failure_reason = "Failed to load file";
        record_test(result);
        return false;
    }

    printf("  Loaded %zu elements (%.2f MB) from %s\n",
           num_elements, (num_elements * sizeof(float)) / (1024.0 * 1024.0), filepath.c_str());

    // Configure quantization
    QuantizationConfig config(qtype, error_bound, num_elements, sizeof(float));

    // Quantize
    QuantizationResult quant_result = quantize_simple(d_original, num_elements, sizeof(float), config);
    if (!quant_result.isValid()) {
        result.failure_reason = "Quantization failed";
        record_test(result);
        cudaFree(d_original);
        return false;
    }

    // Dequantize
    void* d_restored = dequantize_simple(quant_result.d_quantized, quant_result);
    if (d_restored == nullptr) {
        result.failure_reason = "Dequantization failed";
        record_test(result);
        cudaFree(quant_result.d_quantized);
        cudaFree(d_original);
        return false;
    }

    // Compute metrics
    result.metrics = compute_metrics_gpu(d_original, (float*)d_restored, num_elements, error_bound);
    result.metrics.quantized_bytes = quant_result.quantized_bytes;
    result.metrics.quantization_ratio = quant_result.getQuantizationRatio();
    result.metrics.precision_bits = quant_result.actual_precision;

    // Verify error bound
    if (result.metrics.error_bound_satisfied) {
        result.passed = true;
    } else {
        char reason[256];
        snprintf(reason, sizeof(reason), "max_error %.6e > error_bound %.6e",
                 result.metrics.max_error, error_bound);
        result.failure_reason = reason;
    }

    // Cleanup
    cudaFree(d_restored);
    cudaFree(quant_result.d_quantized);
    cudaFree(d_original);

    record_test(result);
    return result.passed;
}

/**
 * @brief Run all file-based tests
 */
void run_file_tests() {
    printf("\n");
    printf("==============================================================================\n");
    printf("                         FILE-BASED TESTS\n");
    printf("==============================================================================\n\n");

    // Get list of test data files
    std::vector<std::string> files = get_test_data_files(g_test_data_dir);

    if (files.empty()) {
        printf("No test data files found in: %s\n", g_test_data_dir.c_str());
        printf("Expected files: test_float32_smooth.bin, test_float32_periodic.bin, test_float32_noisy.bin\n");
        printf("\nGenerate test data with:\n");
        printf("  ./build/generate_test_data smooth 2621440 test_data/test_float32_smooth.bin\n");
        return;
    }

    printf("Found %zu test data files in %s:\n", files.size(), g_test_data_dir.c_str());
    for (const auto& f : files) {
        printf("  - %s\n", f.c_str());
    }
    printf("\n");

    // Error bounds to test
    std::vector<double> error_bounds = {0.1, 0.01, 0.001};

    // Run tests on each file
    for (const auto& filepath : files) {
        printf("--- Testing: %s ---\n\n", filepath.c_str());

        for (double eb : error_bounds) {
            test_file_quantization_roundtrip(filepath, QuantizationType::LINEAR, eb);
        }
        printf("\n");
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --unit          Run unit tests only\n");
    printf("  --integration   Run integration tests only\n");
    printf("  --compare       Run comparison tests only\n");
    printf("  --files         Run tests on input data files\n");
    printf("  --all           Run all tests (default)\n");
    printf("  --data-dir <dir> Set test data directory (default: test_data/)\n");
    printf("  --file <path>   Run tests on a single input file\n");
    printf("  --csv <file>    Export results to CSV file\n");
    printf("  --help          Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --all\n", prog);
    printf("  %s --file test_data/test_float32_smooth.bin\n", prog);
    printf("  %s --files --data-dir /path/to/data\n", prog);
    printf("  %s --unit --csv results.csv\n", prog);
}

int main(int argc, char** argv) {
    bool run_unit = false;
    bool run_integration = false;
    bool run_compare = false;
    bool run_files = false;
    std::string csv_file;
    std::string single_file;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--unit") == 0) {
            run_unit = true;
        } else if (strcmp(argv[i], "--integration") == 0) {
            run_integration = true;
        } else if (strcmp(argv[i], "--compare") == 0) {
            run_compare = true;
        } else if (strcmp(argv[i], "--files") == 0) {
            run_files = true;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_unit = run_integration = run_compare = run_files = true;
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            g_test_data_dir = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            single_file = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // If --file is specified, only run file test (skip defaults)
    // Otherwise, default to all tests if none specified
    if (single_file.empty() && !run_unit && !run_integration && !run_compare && !run_files) {
        run_unit = run_integration = run_compare = run_files = true;
    }

    printf("========================================\n");
    printf("  GPUCompress Quantization Test Suite\n");
    printf("========================================\n");

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("\n");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Single file test - if specified, ONLY run this
    if (!single_file.empty()) {
        printf("\n");
        printf("==============================================================================\n");
        printf("                      SINGLE FILE TEST\n");
        printf("==============================================================================\n\n");

        std::vector<double> error_bounds = {0.1, 0.01, 0.001};

        printf("Testing file: %s\n\n", single_file.c_str());

        for (double eb : error_bounds) {
            test_file_quantization_roundtrip(single_file, QuantizationType::LINEAR, eb);
        }
    } else {
        // Run selected test suites
        if (run_unit) {
            run_unit_tests();
        }

        if (run_integration) {
            run_integration_tests();
        }

        if (run_compare) {
            run_comparison_tests();
        }

        if (run_files) {
            run_file_tests();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print summary
    printf("\n");
    printf("========================================\n");
    printf("              TEST SUMMARY\n");
    printf("========================================\n");
    printf("Total tests: %d\n", g_tests_passed + g_tests_failed);
    printf("Passed:      %d\n", g_tests_passed);
    printf("Failed:      %d\n", g_tests_failed);
    printf("Duration:    %ld ms\n", duration.count());
    printf("========================================\n");

    // Export CSV if requested
    if (!csv_file.empty()) {
        export_csv(csv_file);
    }

    return (g_tests_failed > 0) ? 1 : 0;
}
