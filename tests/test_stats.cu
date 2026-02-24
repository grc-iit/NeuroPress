/**
 * @file test_stats.cu
 * @brief Tests for src/stats/ fixes:
 *   1. Entropy correctness for uniform data (all same byte -> entropy 0)
 *   2. Entropy correctness for high-entropy data (random-like -> near 8.0)
 *   3. Entropy with non-4-aligned size (regression test for histogram Bug 1)
 *   4. calculateEntropyGPU returns -1.0 for null input
 *   5. Stats pipeline produces sane output ranges
 *   6. Stats pipeline returns -1 for null input
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

// Declarations from internal.hpp (in gpucompress namespace)
namespace gpucompress {

double calculateEntropyGPU(const void* d_data, size_t num_bytes, cudaStream_t stream);

int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
);

} // namespace gpucompress

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
 * Test 1: Entropy of uniform data (all same byte -> entropy = 0)
 * ============================================================ */
static void test_entropy_uniform() {
    TEST("Entropy of uniform data (all 0xAA -> 0.0, 16 MB)");

    const size_t N = 16 * 1024 * 1024;  // 16 MB
    uint8_t* h_data = (uint8_t*)malloc(N);
    memset(h_data, 0xAA, N);

    void* d_data = nullptr;
    cudaMalloc(&d_data, N);
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    double entropy = gpucompress::calculateEntropyGPU(d_data, N, 0);
    ASSERT(entropy >= 0.0, "entropy should not be negative");
    ASSERT(entropy < 0.01, "uniform data should have ~0 entropy");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 2: Entropy of two-value data (50/50 -> entropy = 1.0)
 * ============================================================ */
static void test_entropy_two_values() {
    TEST("Entropy of two-value data (50/50 -> 1.0, 8 MB)");

    const size_t N = 8 * 1024 * 1024;  // 8 MB
    uint8_t* h_data = (uint8_t*)malloc(N);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (i < N / 2) ? 0x00 : 0xFF;
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N);
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    double entropy = gpucompress::calculateEntropyGPU(d_data, N, 0);
    ASSERT(fabs(entropy - 1.0) < 0.01, "50/50 two-value data should have entropy ~1.0");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 3: Entropy of 256-value data (each byte once -> 8.0)
 * ============================================================ */
static void test_entropy_max() {
    TEST("Entropy of 256-value uniform distribution (-> 8.0, 16 MB)");

    // 256 * 65536 = 16 MB, each byte value appears exactly 65536 times
    const size_t repeats = 65536;
    const size_t N = 256 * repeats;
    uint8_t* h_data = (uint8_t*)malloc(N);
    for (size_t r = 0; r < repeats; r++) {
        for (int v = 0; v < 256; v++) {
            h_data[r * 256 + v] = (uint8_t)v;
        }
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N);
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    double entropy = gpucompress::calculateEntropyGPU(d_data, N, 0);
    ASSERT(fabs(entropy - 8.0) < 0.01, "uniform 256-value distribution should have entropy ~8.0");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 4: Entropy with non-4-aligned size (Bug 1 regression)
 *
 * Before fix: remaining bytes were counted by every block,
 * inflating their histogram counts and producing wrong entropy.
 * ============================================================ */
static void test_entropy_non_aligned_size() {
    TEST("Entropy with non-4-aligned size (Bug 1 regression, 16 MB + 3)");

    // 16*1024*1024 + 3 bytes = non-4-aligned
    const size_t N = 16 * 1024 * 1024 + 3;
    uint8_t* h_data = (uint8_t*)malloc(N);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (uint8_t)(i & 0xFF);
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, N);
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    double entropy = gpucompress::calculateEntropyGPU(d_data, N, 0);
    ASSERT(entropy >= 0.0, "entropy should not be negative");

    // With 16M+3 bytes: near-uniform distribution
    // Before the fix, the 3 remaining bytes would be counted by all blocks,
    // severely skewing entropy downward.
    ASSERT(entropy > 7.9, "near-uniform 16M+3 bytes should have entropy > 7.9");
    ASSERT(entropy <= 8.0, "entropy should not exceed 8.0");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 5: calculateEntropyGPU returns -1.0 for null input
 * ============================================================ */
static void test_entropy_null_input() {
    TEST("calculateEntropyGPU returns -1.0 for null input");

    double entropy = gpucompress::calculateEntropyGPU(nullptr, 1024, 0);
    ASSERT(entropy < 0.0, "null input should return -1.0");
    ASSERT(fabs(entropy - (-1.0)) < 0.001, "should return exactly -1.0");

    PASS();
}

/* ============================================================
 * Test 6: calculateEntropyGPU returns 0.0 for zero-length input
 * ============================================================ */
static void test_entropy_zero_length() {
    TEST("calculateEntropyGPU returns 0.0 for zero-length input");

    void* d_data = nullptr;
    cudaMalloc(&d_data, 16);  // valid pointer, zero length

    double entropy = gpucompress::calculateEntropyGPU(d_data, 0, 0);
    ASSERT(fabs(entropy) < 0.001, "zero-length should return 0.0");

    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 7: Stats pipeline with constant data
 *
 * Constant float data: mean = c, MAD = 0, deriv = 0
 * Byte-level entropy depends on float representation, but
 * MAD_norm and deriv_norm should be 0.
 * ============================================================ */
static void test_stats_pipeline_constant() {
    TEST("Stats pipeline with constant float data (4M elements)");

    const size_t N = 4 * 1024 * 1024;  // 4M floats = 16 MB
    const size_t bytes = N * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = 42.0f;
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d_data, bytes, 0,
                                                &entropy, &mad, &deriv);
    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(entropy >= 0.0 && entropy <= 8.0, "entropy out of range");
    ASSERT(fabs(mad) < 0.001, "constant data should have MAD ~0");
    ASSERT(fabs(deriv) < 0.001, "constant data should have deriv ~0");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 8: Stats pipeline with linear ramp
 *
 * Linear ramp: second derivative is 0 (f''=0 for linear functions).
 * MAD should be > 0. Entropy should be > 0.
 * ============================================================ */
static void test_stats_pipeline_linear() {
    TEST("Stats pipeline with linear ramp data (4M elements)");

    const size_t N = 4 * 1024 * 1024;  // 4M floats = 16 MB
    const size_t bytes = N * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d_data, bytes, 0,
                                                &entropy, &mad, &deriv);
    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(entropy > 0.0, "ramp data should have > 0 entropy");
    ASSERT(mad > 0.0, "ramp data should have MAD > 0");
    // Linear ramp: f(i)=i, so f(i+1)-2f(i)+f(i-1) = (i+1)-2i+(i-1) = 0
    ASSERT(fabs(deriv) < 0.001, "linear data should have ~0 second derivative");

    free(h_data);
    cudaFree(d_data);
    PASS();
}

/* ============================================================
 * Test 9: Stats pipeline returns -1 for null input
 * ============================================================ */
static void test_stats_pipeline_null() {
    TEST("Stats pipeline returns -1 for null input");

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(nullptr, 1024, 0,
                                                &entropy, &mad, &deriv);
    ASSERT(rc == -1, "null input should return -1");

    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_stats ===\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping GPU tests\n");
        return 1;
    }

    // Entropy tests
    test_entropy_uniform();
    test_entropy_two_values();
    test_entropy_max();
    test_entropy_non_aligned_size();
    test_entropy_null_input();
    test_entropy_zero_length();

    // Stats pipeline tests
    test_stats_pipeline_constant();
    test_stats_pipeline_linear();
    test_stats_pipeline_null();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
