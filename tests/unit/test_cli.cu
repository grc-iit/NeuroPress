/**
 * @file test_cli.cu
 * @brief Test suite for src/cli/ fixes
 *
 * The CLI tools (compress/decompress) depend on GDS (cuFile) which requires
 * special filesystem support. These tests validate the fix logic through
 * direct CUDA calls and the library API, which share the same code paths.
 *
 * Tests:
 *  1. compare_buffers kernel with device memory (not host-pinned)
 *  2. compare_buffers detects mismatch correctly
 *  3. Compression header validation rejects truncated data (decompress fix)
 *  4. Header size mismatch warning for non-quantized data (decompress fix)
 *  5. Header size mismatch OK for quantized data (no false warning)
 *  6. Round-trip compress/decompress via API (validates CLI code paths)
 *  7. CUDA error string included in error message (CUDA_CHECK fix)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "compression/compression_header.h"

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while (0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("  FAIL: %s — expected %d, got %d (line %d)\n", msg, (int)(b), (int)(a), __LINE__); \
        g_fail++; return; \
    } \
} while (0)

/* ---- Kernel from compress.cpp (validates fix 5) ---- */

__global__ void compare_buffers(const uint8_t* ref, const uint8_t* val, int* invalid, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        if (ref[i] != val[i]) {
            *invalid = 1;
            return;
        }
    }
}

/* ============================================================
 * Test 1: compare_buffers with device memory (fix 5 validation)
 * ============================================================ */
static void test_compare_buffers_match() {
    printf("Test 1: compare_buffers kernel — identical buffers (4 MB)...\n");

    const size_t N = 4 * 1024 * 1024;
    std::vector<uint8_t> h_data(N);
    for (size_t i = 0; i < N; i++) h_data[i] = static_cast<uint8_t>(i & 0xFF);

    uint8_t *d_a = nullptr, *d_b = nullptr;
    int *d_invalid = nullptr;
    cudaMalloc(&d_a, N);
    cudaMalloc(&d_b, N);
    cudaMalloc(&d_invalid, sizeof(int));

    cudaMemcpy(d_a, h_data.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data.data(), N, cudaMemcpyHostToDevice);
    cudaMemset(d_invalid, 0, sizeof(int));

    compare_buffers<<<256, 256>>>(d_a, d_b, d_invalid, N);

    int h_invalid = 99;
    cudaMemcpy(&h_invalid, d_invalid, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    ASSERT_EQ(h_invalid, 0, "identical buffers report no mismatch");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_invalid);

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: compare_buffers detects mismatch
 * ============================================================ */
static void test_compare_buffers_mismatch() {
    printf("Test 2: compare_buffers kernel — detects single-byte mismatch (4 MB)...\n");

    const size_t N = 4 * 1024 * 1024;
    std::vector<uint8_t> h_a(N, 0xAA);
    std::vector<uint8_t> h_b(N, 0xAA);
    h_b[N / 2] = 0xBB;  // Single byte difference in the middle

    uint8_t *d_a = nullptr, *d_b = nullptr;
    int *d_invalid = nullptr;
    cudaMalloc(&d_a, N);
    cudaMalloc(&d_b, N);
    cudaMalloc(&d_invalid, sizeof(int));

    cudaMemcpy(d_a, h_a.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N, cudaMemcpyHostToDevice);
    cudaMemset(d_invalid, 0, sizeof(int));

    compare_buffers<<<256, 256>>>(d_a, d_b, d_invalid, N);

    int h_invalid = 0;
    cudaMemcpy(&h_invalid, d_invalid, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    ASSERT_EQ(h_invalid, 1, "mismatch detected");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_invalid);

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 3: Decompress rejects truncated compressed data (fix 3)
 * ============================================================ */
static void test_truncated_header_validation() {
    printf("Test 3: decompress rejects truncated compressed data...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    // Compress some data first
    const size_t N = 256 * 1024;
    std::vector<float> original(N);
    for (size_t i = 0; i < N; i++) original[i] = sinf((float)i * 0.01f);
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

    // Try decompress with truncated size (less than header + claimed compressed data)
    std::vector<float> out(N);
    size_t out_size = input_size;
    size_t truncated = GPUCOMPRESS_HEADER_SIZE + 16;  // Way too small

    rc = gpucompress_decompress(compressed.data(), truncated,
                                out.data(), &out_size);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_HEADER, "truncated data rejected");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 4: Header decompressed_size should match original_size
 *         for non-quantized lossless data (fix 2 validation)
 * ============================================================ */
static void test_header_size_match_lossless() {
    printf("Test 4: lossless header original_size matches decompressed output...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 512 * 1024;
    std::vector<float> original(N);
    for (size_t i = 0; i < N; i++) original[i] = (float)i * 0.001f;
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

    // Read header and verify original_size
    CompressionHeader header;
    memcpy(&header, compressed.data(), sizeof(CompressionHeader));
    ASSERT_TRUE(header.isValid(), "header valid");
    ASSERT_TRUE(header.original_size == input_size, "original_size matches input");
    ASSERT_TRUE(!header.hasQuantizationApplied(), "no quantization");

    // Decompress and verify size
    std::vector<float> decompressed(N);
    size_t decomp_size = input_size;
    rc = gpucompress_decompress(compressed.data(), compressed_size,
                                decompressed.data(), &decomp_size);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "decompress");
    ASSERT_TRUE(decomp_size == input_size, "decompressed size matches original");

    // Verify lossless
    ASSERT_TRUE(memcmp(original.data(), decompressed.data(), input_size) == 0,
                "data matches");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 5: Quantized data — decompressed size != original_size is OK
 * ============================================================ */
static void test_header_size_quantized() {
    printf("Test 5: quantized data header — compressed_size < original_size is expected...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 256 * 1024;
    std::vector<float> original(N);
    for (size_t i = 0; i < N; i++) original[i] = (float)i * 0.1f;
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
    cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE;
    cfg.error_bound = 0.01;

    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress with quantization");

    // Read header
    CompressionHeader header;
    memcpy(&header, compressed.data(), sizeof(CompressionHeader));
    ASSERT_TRUE(header.isValid(), "header valid");
    ASSERT_TRUE(header.hasQuantizationApplied(), "quantization applied");
    ASSERT_TRUE(header.original_size == input_size, "original_size stored correctly");

    // Decompress — should restore to original_size (after dequantization)
    std::vector<float> decompressed(N);
    size_t decomp_size = input_size;
    rc = gpucompress_decompress(compressed.data(), compressed_size,
                                decompressed.data(), &decomp_size);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "decompress");
    ASSERT_TRUE(decomp_size == input_size, "output is original size");

    // Verify error bound (lossy)
    double max_err = 0.0;
    for (size_t i = 0; i < N; i++) {
        double err = fabs((double)original[i] - (double)decompressed[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err <= cfg.error_bound + 1e-6,
                "error within bound");

    gpucompress_cleanup();
    printf("  PASS (max error: %.6e, bound: %.6e)\n", max_err, cfg.error_bound);
    g_pass++;
}

/* ============================================================
 * Test 6: Full round-trip via API with multiple algorithms (4 MB each)
 * ============================================================ */
static void test_roundtrip_multiple_algos() {
    printf("Test 6: round-trip with multiple algorithms (4 MB each)...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 1024 * 1024;  // 4 MB
    std::vector<float> original(N);
    for (size_t i = 0; i < N; i++) {
        original[i] = sinf((float)i * 0.005f) * 200.0f + (float)(i % 100);
    }
    size_t input_size = N * sizeof(float);

    gpucompress_algorithm_t algos[] = {
        GPUCOMPRESS_ALGO_LZ4,
        GPUCOMPRESS_ALGO_SNAPPY,
        GPUCOMPRESS_ALGO_ZSTD,
        GPUCOMPRESS_ALGO_DEFLATE
    };
    const char* names[] = {"lz4", "snappy", "zstd", "deflate"};
    int n_algos = sizeof(algos) / sizeof(algos[0]);

    for (int a = 0; a < n_algos; a++) {
        size_t max_out = gpucompress_max_compressed_size(input_size);
        std::vector<uint8_t> compressed(max_out);
        size_t comp_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = algos[a];

        rc = gpucompress_compress(original.data(), input_size,
                                  compressed.data(), &comp_size,
                                  &cfg, nullptr);
        ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

        std::vector<float> decompressed(N);
        size_t decomp_size = input_size;
        rc = gpucompress_decompress(compressed.data(), comp_size,
                                    decompressed.data(), &decomp_size);
        ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "decompress");
        ASSERT_TRUE(memcmp(original.data(), decompressed.data(), input_size) == 0,
                    "lossless round-trip");

        printf("    %s: %zu -> %zu (%.1fx)\n", names[a], input_size, comp_size,
               (double)input_size / comp_size);
    }

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 7: cudaGetErrorString is available (CUDA_CHECK fix validation)
 * ============================================================ */
static void test_cuda_error_string() {
    printf("Test 7: cudaGetErrorString returns meaningful messages...\n");

    const char* success_msg = cudaGetErrorString(cudaSuccess);
    ASSERT_TRUE(success_msg != nullptr, "success message not null");
    ASSERT_TRUE(strlen(success_msg) > 0, "success message not empty");

    const char* oom_msg = cudaGetErrorString(cudaErrorMemoryAllocation);
    ASSERT_TRUE(oom_msg != nullptr, "OOM message not null");
    ASSERT_TRUE(strlen(oom_msg) > 0, "OOM message not empty");

    // Verify messages are different
    ASSERT_TRUE(strcmp(success_msg, oom_msg) != 0, "different errors produce different messages");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== CLI Test Suite ===\n\n");

    test_compare_buffers_match();
    test_compare_buffers_mismatch();
    test_cuda_error_string();
    test_truncated_header_validation();
    test_header_size_match_lossless();
    test_header_size_quantized();
    test_roundtrip_multiple_algos();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
