/**
 * @file test_preprocessing.cu
 * @brief Tests for src/preprocessing/ fixes:
 *   1. Byte shuffle/unshuffle round-trip (multiple element sizes + kernel types)
 *   2. Quantization round-trip with error bound verification
 *   3. Quantization with negative data (validates min/max kernel fix)
 *   4. AUTO kernel selection correctness
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) \
    do { printf("  [TEST] %s ... ", name); } while (0)

#define PASS() \
    do { printf("PASS\n"); g_pass++; } while (0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_fail++; } while (0)

#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

/* ============================================================
 * Test 1: Byte shuffle round-trip (4-byte float elements)
 * ============================================================ */
static void test_shuffle_roundtrip_float() {
    TEST("Shuffle round-trip float (4-byte, AUTO kernel, 4M elements)");

    const size_t N = 4 * 1024 * 1024;  // 4M floats = 16 MB
    const size_t bytes = N * sizeof(float);

    // Create test data on host
    float* h_data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i * 1.5f - 500.0f;
    }

    // Copy to device
    void* d_data = nullptr;
    cudaError_t err = cudaMalloc(&d_data, bytes);
    ASSERT(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    ASSERT(err == cudaSuccess, "cudaMemcpy H2D failed");

    // Shuffle
    uint8_t* d_shuffled = byte_shuffle_simple(d_data, bytes, sizeof(float));
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    // Unshuffle
    uint8_t* d_restored = byte_unshuffle_simple(d_shuffled, bytes, sizeof(float));
    ASSERT(d_restored != nullptr, "unshuffle returned null");

    // Sync and copy back
    cudaDeviceSynchronize();
    float* h_restored = (float*)malloc(bytes);
    err = cudaMemcpy(h_restored, d_restored, bytes, cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess, "cudaMemcpy D2H failed");

    // Verify exact match
    bool match = true;
    for (size_t i = 0; i < N; i++) {
        if (h_data[i] != h_restored[i]) {
            match = false;
            break;
        }
    }
    ASSERT(match, "data mismatch after shuffle round-trip");

    free(h_data);
    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 2: Byte shuffle round-trip (8-byte double elements)
 * ============================================================ */
static void test_shuffle_roundtrip_double() {
    TEST("Shuffle round-trip double (8-byte, AUTO kernel, 2M elements)");

    const size_t N = 2 * 1024 * 1024;  // 2M doubles = 16 MB
    const size_t bytes = N * sizeof(double);

    double* h_data = (double*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (double)i * 3.14159 - 1000.0;
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint8_t* d_shuffled = byte_shuffle_simple(d_data, bytes, sizeof(double));
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    uint8_t* d_restored = byte_unshuffle_simple(d_shuffled, bytes, sizeof(double));
    ASSERT(d_restored != nullptr, "unshuffle returned null");

    cudaDeviceSynchronize();
    double* h_restored = (double*)malloc(bytes);
    cudaMemcpy(h_restored, d_restored, bytes, cudaMemcpyDeviceToHost);

    bool match = true;
    for (size_t i = 0; i < N; i++) {
        if (h_data[i] != h_restored[i]) {
            match = false;
            break;
        }
    }
    ASSERT(match, "data mismatch after shuffle round-trip");

    free(h_data);
    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 3: Byte shuffle round-trip (2-byte, SPECIALIZED kernel)
 * ============================================================ */
static void test_shuffle_roundtrip_int16() {
    TEST("Shuffle round-trip int16 (2-byte, SPECIALIZED kernel, 8M elements)");

    const size_t N = 8 * 1024 * 1024;  // 8M int16s = 16 MB
    const size_t bytes = N * sizeof(int16_t);

    int16_t* h_data = (int16_t*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (int16_t)(i % 65536 - 32768);
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint8_t* d_shuffled = byte_shuffle_simple(
        d_data, bytes, 2, 256 * 1024, ShuffleKernelType::SPECIALIZED);
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    uint8_t* d_restored = byte_unshuffle_simple(
        d_shuffled, bytes, 2, 256 * 1024, ShuffleKernelType::SPECIALIZED);
    ASSERT(d_restored != nullptr, "unshuffle returned null");

    cudaDeviceSynchronize();
    int16_t* h_restored = (int16_t*)malloc(bytes);
    cudaMemcpy(h_restored, d_restored, bytes, cudaMemcpyDeviceToHost);

    bool match = (memcmp(h_data, h_restored, bytes) == 0);
    ASSERT(match, "data mismatch after shuffle round-trip");

    free(h_data);
    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 4: Shuffle with non-divisible size (leftover bytes)
 * ============================================================ */
static void test_shuffle_leftover_bytes() {
    TEST("Shuffle round-trip with leftover bytes (4 MB + 3)");

    // 4*1024*1024 + 3 bytes with element_size=4 -> 1M elements + 3 leftover bytes
    const size_t bytes = 4 * 1024 * 1024 + 3;
    uint8_t* h_data = (uint8_t*)malloc(bytes);
    for (size_t i = 0; i < bytes; i++) {
        h_data[i] = (uint8_t)(i & 0xFF);
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint8_t* d_shuffled = byte_shuffle_simple(d_data, bytes, 4);
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    uint8_t* d_restored = byte_unshuffle_simple(d_shuffled, bytes, 4);
    ASSERT(d_restored != nullptr, "unshuffle returned null");

    cudaDeviceSynchronize();
    uint8_t* h_restored = (uint8_t*)malloc(bytes);
    cudaMemcpy(h_restored, d_restored, bytes, cudaMemcpyDeviceToHost);

    bool match = (memcmp(h_data, h_restored, bytes) == 0);
    ASSERT(match, "leftover byte mismatch");

    free(h_data);
    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 5: Quantization round-trip with positive data
 * ============================================================ */
static void test_quantize_roundtrip_positive() {
    TEST("Quantize round-trip (positive floats, eb=0.01, 2M elements)");

    const size_t N = 2 * 1024 * 1024;  // 2M floats = 8 MB
    const double error_bound = 0.01;

    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i * 0.1f;  // 0.0 to 409.5
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    QuantizationConfig config(QuantizationType::LINEAR, error_bound, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization result invalid");
    ASSERT(result.actual_precision > 0, "precision should be > 0");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    // Verify error bound on GPU
    double max_error = 0.0;
    bool within_bound = verify_error_bound(
        d_data, d_restored, N, sizeof(float), error_bound, 0, &max_error);
    ASSERT(within_bound, "error bound violated");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 6: Quantization with negative data
 *   (validates that min/max kernel works for negatives)
 * ============================================================ */
static void test_quantize_negative_data() {
    TEST("Quantize round-trip (negative floats, eb=0.1, 1M elements)");

    const size_t N = 1024 * 1024;  // 1M floats = 4 MB
    const double error_bound = 0.1;

    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        // Range: -1000.0 to ~ -1000.0 + 1M * 0.0001 = -900.0 (all negative)
        h_data[i] = -1000.0f + (float)i * 0.0001f;
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    QuantizationConfig config(QuantizationType::LINEAR, error_bound, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization result invalid");
    ASSERT(result.data_min < 0, "data_min should be negative");
    ASSERT(result.data_max < 0, "data_max should be negative");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool within_bound = verify_error_bound(
        d_data, d_restored, N, sizeof(float), error_bound, 0, &max_error);
    ASSERT(within_bound, "error bound violated for negative data");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 7: Quantization with mixed positive/negative data
 * ============================================================ */
static void test_quantize_mixed_data() {
    TEST("Quantize round-trip (mixed pos/neg, eb=0.001, 4M elements)");

    const size_t N = 4 * 1024 * 1024;  // 4M floats = 16 MB
    const double error_bound = 0.001;

    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = sinf((float)i * 0.01f) * 100.0f;  // -100 to +100
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    QuantizationConfig config(QuantizationType::LINEAR, error_bound, N, sizeof(float));
    QuantizationResult result = quantize_simple(d_data, N, sizeof(float), config);
    ASSERT(result.isValid(), "quantization result invalid");
    ASSERT(result.data_min < 0, "expected negative min");
    ASSERT(result.data_max > 0, "expected positive max");

    void* d_restored = dequantize_simple(result.d_quantized, result);
    ASSERT(d_restored != nullptr, "dequantize returned null");

    double max_error = 0.0;
    bool within_bound = verify_error_bound(
        d_data, d_restored, N, sizeof(float), error_bound, 0, &max_error);
    ASSERT(within_bound, "error bound violated for mixed data");

    free(h_data);
    cudaFree(d_data);
    cudaFree(result.d_quantized);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 8: Shuffle with SHARED_MEMORY kernel (16-byte elements)
 * ============================================================ */
static void test_shuffle_smem_16byte() {
    TEST("Shuffle round-trip (16-byte, SHARED_MEMORY kernel, 1M elements)");

    // Use 16-byte elements (e.g., two doubles packed)
    const size_t elem_size = 16;
    const size_t N_elems = 1024 * 1024;  // 1M elements = 16 MB
    const size_t bytes = N_elems * elem_size;

    uint8_t* h_data = (uint8_t*)malloc(bytes);
    for (size_t i = 0; i < bytes; i++) {
        h_data[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint8_t* d_shuffled = byte_shuffle_simple(
        d_data, bytes, elem_size, 256 * 1024, ShuffleKernelType::SHARED_MEMORY);
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    uint8_t* d_restored = byte_unshuffle_simple(
        d_shuffled, bytes, elem_size, 256 * 1024, ShuffleKernelType::SHARED_MEMORY);
    ASSERT(d_restored != nullptr, "unshuffle returned null");

    cudaDeviceSynchronize();
    uint8_t* h_restored = (uint8_t*)malloc(bytes);
    cudaMemcpy(h_restored, d_restored, bytes, cudaMemcpyDeviceToHost);

    bool match = (memcmp(h_data, h_restored, bytes) == 0);
    ASSERT(match, "data mismatch with SHARED_MEMORY kernel");

    free(h_data);
    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_preprocessing ===\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping GPU tests\n");
        return 1;
    }

    // Byte shuffle tests
    test_shuffle_roundtrip_float();
    test_shuffle_roundtrip_double();
    test_shuffle_roundtrip_int16();
    test_shuffle_leftover_bytes();
    test_shuffle_smem_16byte();

    // Quantization tests
    test_quantize_roundtrip_positive();
    test_quantize_negative_data();
    test_quantize_mixed_data();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
