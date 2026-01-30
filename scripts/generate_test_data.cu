/**
 * @file generate_test_data.cu
 * @brief CUDA-based synthetic data generator for quantization testing
 *
 * Generates test data with different patterns:
 * - Smooth gradients (benefits Lorenzo/block transform)
 * - Random noise (baseline comparison)
 * - Structured grids (benefits block transform)
 * - Mixed patterns (realistic scenario)
 * - Sensor-like readings (periodic + noise)
 *
 * Usage:
 *   nvcc -o generate_test_data generate_test_data.cu
 *   ./generate_test_data <pattern> <num_elements> <output_file>
 *
 * Patterns:
 *   smooth    - Smooth gradient with small noise
 *   random    - Uniform random noise
 *   grid      - Structured 2D grid pattern
 *   mixed     - Combination of patterns
 *   sensor    - Periodic signal with noise (like sensor readings)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define BLOCK_SIZE 256
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Pattern Generation Kernels
// ============================================================================

/**
 * Smooth gradient pattern with small noise
 * Good for testing Lorenzo 1D and block transform
 */
__global__ void generate_smooth_pattern(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Initialize RNG
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Smooth gradient
    float t = (float)idx / (float)num_elements;
    float value = 1000.0f * sinf(t * 2.0f * M_PI * 5.0f);  // 5 cycles

    // Add small noise (1% of range)
    value += curand_normal(&state) * 10.0f;

    data[idx] = value;
}

/**
 * Random uniform noise
 * Baseline for compression ratio comparison
 */
__global__ void generate_random_pattern(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Uniform random in [0, 1000]
    data[idx] = curand_uniform(&state) * 1000.0f;
}

/**
 * Structured 2D grid pattern
 * Good for testing block transform
 */
__global__ void generate_grid_pattern(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Interpret as 2D grid (assume square or close to square)
    size_t grid_size = (size_t)sqrtf((float)num_elements);
    size_t row = idx / grid_size;
    size_t col = idx % grid_size;

    // Structured pattern: combination of row and column effects
    float value = 100.0f * sinf(row * 0.1f) + 100.0f * cosf(col * 0.1f);

    // Add small noise
    curandState state;
    curand_init(seed, idx, 0, &state);
    value += curand_normal(&state) * 5.0f;

    data[idx] = value;
}

/**
 * Mixed pattern combining different characteristics
 * Realistic scenario
 */
__global__ void generate_mixed_pattern(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Different regions have different patterns
    float t = (float)idx / (float)num_elements;
    float value;

    if (t < 0.33f) {
        // Region 1: Smooth gradient
        value = 500.0f * sinf(t * 30.0f * M_PI);
    } else if (t < 0.66f) {
        // Region 2: Structured pattern
        value = 200.0f + 100.0f * sinf(idx * 0.01f) * cosf(idx * 0.02f);
    } else {
        // Region 3: Noisy
        value = curand_normal(&state) * 200.0f + 500.0f;
    }

    data[idx] = value;
}

/**
 * Sensor-like readings: periodic signal with noise
 * Common in scientific/IoT data
 */
__global__ void generate_sensor_pattern(
    float* data,
    size_t num_elements,
    unsigned long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Periodic signal (temperature-like)
    float t = (float)idx / 1000.0f;  // Time in arbitrary units
    float value = 20.0f + 5.0f * sinf(t * 2.0f * M_PI);  // Daily cycle

    // Add drift
    value += 0.001f * t;

    // Add measurement noise
    value += curand_normal(&state) * 0.5f;

    // Occasional spikes (sensor glitches)
    if (curand_uniform(&state) < 0.001f) {
        value += curand_normal(&state) * 10.0f;
    }

    data[idx] = value;
}

// ============================================================================
// Statistics Computation
// ============================================================================

__global__ void compute_stats_kernel(
    const float* data,
    size_t num_elements,
    double* sum,
    double* sum_sq,
    float* min_val,
    float* max_val
) {
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ double s_sum_sq[BLOCK_SIZE];
    __shared__ float s_min[BLOCK_SIZE];
    __shared__ float s_max[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    float local_min = 1e30f;
    float local_max = -1e30f;

    for (size_t i = idx; i < num_elements; i += stride) {
        float val = data[i];
        local_sum += val;
        local_sum_sq += val * val;
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;

    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + s]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, s_sum[0]);
        atomicAdd(sum_sq, s_sum_sq[0]);

        // Atomic min/max for floats
        unsigned int* min_as_uint = (unsigned int*)min_val;
        unsigned int old = *min_as_uint;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(min_as_uint, assumed,
                           __float_as_uint(fminf(__uint_as_float(assumed), s_min[0])));
        } while (assumed != old);

        unsigned int* max_as_uint = (unsigned int*)max_val;
        old = *max_as_uint;
        do {
            assumed = old;
            old = atomicCAS(max_as_uint, assumed,
                           __float_as_uint(fmaxf(__uint_as_float(assumed), s_max[0])));
        } while (assumed != old);
    }
}

void compute_and_print_stats(float* d_data, size_t num_elements) {
    double *d_sum, *d_sum_sq;
    float *d_min, *d_max;

    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));

    double zero = 0.0;
    float init_min = 1e30f;
    float init_max = -1e30f;

    CUDA_CHECK(cudaMemcpy(d_sum, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum_sq, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));

    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = (num_blocks > 1024) ? 1024 : num_blocks;

    compute_stats_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_data, num_elements, d_sum, d_sum_sq, d_min, d_max);

    double h_sum, h_sum_sq;
    float h_min, h_max;

    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

    double mean = h_sum / num_elements;
    double variance = (h_sum_sq / num_elements) - (mean * mean);
    double stddev = sqrt(variance);

    printf("Statistics:\n");
    printf("  Elements: %zu\n", num_elements);
    printf("  Min: %.6f\n", h_min);
    printf("  Max: %.6f\n", h_max);
    printf("  Range: %.6f\n", h_max - h_min);
    printf("  Mean: %.6f\n", mean);
    printf("  Std Dev: %.6f\n", stddev);

    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sum_sq));
    CUDA_CHECK(cudaFree(d_min));
    CUDA_CHECK(cudaFree(d_max));
}

// ============================================================================
// Main Program
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s <pattern> <num_elements> <output_file>\n", prog);
    printf("\n");
    printf("Patterns:\n");
    printf("  smooth  - Smooth gradient with small noise (good for Lorenzo/block)\n");
    printf("  random  - Uniform random noise (baseline)\n");
    printf("  grid    - Structured 2D grid pattern (good for block transform)\n");
    printf("  mixed   - Combination of patterns (realistic)\n");
    printf("  sensor  - Periodic signal with noise (sensor readings)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s smooth 10000000 test_smooth.bin\n", prog);
    printf("  %s sensor 5000000 test_sensor.bin\n", prog);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char* pattern = argv[1];
    size_t num_elements = strtoull(argv[2], nullptr, 10);
    const char* output_file = argv[3];

    printf("========================================\n");
    printf("GPU Synthetic Data Generator\n");
    printf("========================================\n");
    printf("Pattern: %s\n", pattern);
    printf("Elements: %zu (%.2f MB as float32)\n", num_elements,
           (num_elements * sizeof(float)) / (1024.0 * 1024.0));
    printf("Output: %s\n", output_file);
    printf("\n");

    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(float)));

    // Generate data based on pattern
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned long seed = 12345;

    printf("Generating data on GPU...\n");

    if (strcmp(pattern, "smooth") == 0) {
        generate_smooth_pattern<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, seed);
    } else if (strcmp(pattern, "random") == 0) {
        generate_random_pattern<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, seed);
    } else if (strcmp(pattern, "grid") == 0) {
        generate_grid_pattern<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, seed);
    } else if (strcmp(pattern, "mixed") == 0) {
        generate_mixed_pattern<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, seed);
    } else if (strcmp(pattern, "sensor") == 0) {
        generate_sensor_pattern<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, seed);
    } else {
        fprintf(stderr, "Error: Unknown pattern '%s'\n", pattern);
        print_usage(argv[0]);
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Data generated successfully!\n\n");

    // Compute and print statistics
    compute_and_print_stats(d_data, num_elements);

    // Copy to host and write to file
    printf("\nWriting to file...\n");
    float* h_data = new float[num_elements];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    FILE* fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
        delete[] h_data;
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    size_t written = fwrite(h_data, sizeof(float), num_elements, fp);
    fclose(fp);

    if (written != num_elements) {
        fprintf(stderr, "Error: Write failed (expected %zu, wrote %zu)\n", num_elements, written);
        delete[] h_data;
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    printf("Successfully wrote %zu elements to %s\n", written, output_file);

    // Cleanup
    delete[] h_data;
    cudaFree(d_data);

    printf("\n========================================\n");
    printf("SUCCESS!\n");
    printf("========================================\n");

    return EXIT_SUCCESS;
}
