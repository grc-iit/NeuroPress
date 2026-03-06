/**
 * @file test_api.cu
 * @brief Test suite for src/api/ fixes
 *
 * Tests:
 *  1. Double-cleanup doesn't crash (ref_count underflow guard)
 *  2. Init/cleanup ref-counting works correctly
 *  3. Compress round-trip with LZ4 (4 MB)
 *  4. Compress round-trip with Zstd (4 MB)
 *  5. Decompress rejects truncated input
 *  6. Decompress rejects bad magic
 *  7. Stats compressed_size includes header, ratio is consistent
 *  8. Compress/decompress with NULL params returns correct errors
 *  9. Not-initialized guard returns GPUCOMPRESS_ERROR_NOT_INITIALIZED
 * 10. Algorithm name round-trip (string <-> enum)
 * 11. Error string coverage
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "gpucompress.h"

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

/* ============================================================
 * Helpers
 * ============================================================ */

static void fill_random_floats(float* buf, size_t count) {
    for (size_t i = 0; i < count; i++) {
        buf[i] = static_cast<float>(i) * 0.001f +
                 sinf(static_cast<float>(i) * 0.1f) * 100.0f;
    }
}

/* ============================================================
 * Test 1: Double-cleanup doesn't crash
 * ============================================================ */
static void test_double_cleanup() {
    printf("Test 1: double cleanup (ref_count underflow guard)...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    gpucompress_cleanup();
    ASSERT_EQ(gpucompress_is_initialized(), 0, "not initialized after cleanup");

    // Second cleanup should not crash or double-free
    gpucompress_cleanup();
    gpucompress_cleanup();
    gpucompress_cleanup();

    ASSERT_EQ(gpucompress_is_initialized(), 0, "still not initialized");
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: Init/cleanup ref-counting
 * ============================================================ */
static void test_ref_counting() {
    printf("Test 2: init/cleanup ref counting...\n");

    gpucompress_error_t rc;

    rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init 1");
    ASSERT_EQ(gpucompress_is_initialized(), 1, "initialized after init 1");

    rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init 2 (refcount=2)");
    ASSERT_EQ(gpucompress_is_initialized(), 1, "still initialized");

    gpucompress_cleanup();  // refcount 2 -> 1
    ASSERT_EQ(gpucompress_is_initialized(), 1, "still initialized after first cleanup");

    gpucompress_cleanup();  // refcount 1 -> 0, actual cleanup
    ASSERT_EQ(gpucompress_is_initialized(), 0, "not initialized after second cleanup");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 3: Compress round-trip LZ4 (4 MB)
 * ============================================================ */
static void test_roundtrip_lz4() {
    printf("Test 3: compress/decompress round-trip LZ4 (4 MB)...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 1024 * 1024;  // 1M floats = 4 MB
    std::vector<float> original(N);
    fill_random_floats(original.data(), N);
    size_t input_size = N * sizeof(float);

    // Compress
    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");
    ASSERT_TRUE(compressed_size > 0 && compressed_size <= max_out, "compressed_size sane");

    // Decompress
    std::vector<float> decompressed(N);
    size_t decomp_size = input_size;

    rc = gpucompress_decompress(compressed.data(), compressed_size,
                                decompressed.data(), &decomp_size);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "decompress");
    ASSERT_EQ(decomp_size, input_size, "decompressed size matches");

    // Verify lossless
    ASSERT_TRUE(memcmp(original.data(), decompressed.data(), input_size) == 0,
                "data matches after round-trip");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 4: Compress round-trip Zstd (4 MB)
 * ============================================================ */
static void test_roundtrip_zstd() {
    printf("Test 4: compress/decompress round-trip Zstd (4 MB)...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 1024 * 1024;
    std::vector<float> original(N);
    fill_random_floats(original.data(), N);
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_ZSTD;

    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

    std::vector<float> decompressed(N);
    size_t decomp_size = input_size;

    rc = gpucompress_decompress(compressed.data(), compressed_size,
                                decompressed.data(), &decomp_size);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "decompress");
    ASSERT_TRUE(memcmp(original.data(), decompressed.data(), input_size) == 0,
                "data matches after round-trip");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 5: Decompress rejects truncated input
 * ============================================================ */
static void test_decompress_truncated() {
    printf("Test 5: decompress rejects truncated input...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    // First compress valid data
    const size_t N = 256 * 1024;  // 1 MB
    std::vector<float> original(N);
    fill_random_floats(original.data(), N);
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

    // Try to decompress with truncated size (header + only 10 bytes)
    std::vector<float> out(N);
    size_t out_size = input_size;
    size_t truncated_size = GPUCOMPRESS_HEADER_SIZE + 10;  // Way too small

    rc = gpucompress_decompress(compressed.data(), truncated_size,
                                out.data(), &out_size);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_HEADER, "truncated input rejected");

    // Also test input smaller than header
    out_size = input_size;
    rc = gpucompress_decompress(compressed.data(), 32,
                                out.data(), &out_size);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_HEADER, "sub-header input rejected");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 6: Decompress rejects bad magic
 * ============================================================ */
static void test_decompress_bad_magic() {
    printf("Test 6: decompress rejects bad magic...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    // Create a fake header with wrong magic
    uint8_t fake[128];
    memset(fake, 0, sizeof(fake));
    uint32_t bad_magic = 0xDEADBEEF;
    memcpy(fake, &bad_magic, sizeof(bad_magic));

    std::vector<float> out(1024);
    size_t out_size = 1024 * sizeof(float);

    rc = gpucompress_decompress(fake, sizeof(fake), out.data(), &out_size);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_HEADER, "bad magic rejected");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 7: Stats compressed_size and ratio consistency
 * ============================================================ */
static void test_stats_consistency() {
    printf("Test 7: stats compressed_size and ratio consistency...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    const size_t N = 512 * 1024;  // 2 MB
    std::vector<float> original(N);
    fill_random_floats(original.data(), N);
    size_t input_size = N * sizeof(float);

    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<uint8_t> compressed(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    rc = gpucompress_compress(original.data(), input_size,
                              compressed.data(), &compressed_size,
                              &cfg, &stats);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "compress");

    // compressed_size returned should equal stats.compressed_size
    ASSERT_TRUE(stats.compressed_size == compressed_size,
                "stats.compressed_size == output compressed_size");

    // Ratio should be consistent: original / compressed_size (total)
    double expected_ratio = static_cast<double>(input_size) /
                            static_cast<double>(stats.compressed_size);
    double diff = fabs(stats.compression_ratio - expected_ratio);
    ASSERT_TRUE(diff < 0.001, "ratio consistent with compressed_size");

    // original_size should match input
    ASSERT_TRUE(stats.original_size == input_size, "original_size matches");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 8: NULL parameter checks
 * ============================================================ */
static void test_null_params() {
    printf("Test 8: NULL parameter checks...\n");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_SUCCESS, "init");

    uint8_t buf[64];
    size_t sz = sizeof(buf);

    // Compress with NULL input
    rc = gpucompress_compress(nullptr, 100, buf, &sz, nullptr, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "compress null input");

    // Compress with NULL output
    rc = gpucompress_compress(buf, sizeof(buf), nullptr, &sz, nullptr, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "compress null output");

    // Compress with NULL output_size
    rc = gpucompress_compress(buf, sizeof(buf), buf, nullptr, nullptr, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "compress null output_size");

    // Compress with zero input_size
    rc = gpucompress_compress(buf, 0, buf, &sz, nullptr, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "compress zero input_size");

    // Decompress with NULL input
    rc = gpucompress_decompress(nullptr, 100, buf, &sz);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "decompress null input");

    // Decompress with NULL output
    rc = gpucompress_decompress(buf, sizeof(buf), nullptr, &sz);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "decompress null output");

    // Decompress with NULL output_size
    rc = gpucompress_decompress(buf, sizeof(buf), buf, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_INVALID_INPUT, "decompress null output_size");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 9: Not-initialized guard
 * ============================================================ */
static void test_not_initialized() {
    printf("Test 9: not-initialized guard...\n");

    // Ensure library is not initialized
    ASSERT_EQ(gpucompress_is_initialized(), 0, "not initialized initially");

    uint8_t buf[128];
    size_t sz = sizeof(buf);

    gpucompress_error_t rc;

    rc = gpucompress_compress(buf, sizeof(buf), buf, &sz, nullptr, nullptr);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_NOT_INITIALIZED, "compress before init");

    rc = gpucompress_decompress(buf, sizeof(buf), buf, &sz);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_NOT_INITIALIZED, "decompress before init");

    double entropy;
    rc = gpucompress_calculate_entropy(buf, sizeof(buf), &entropy);
    ASSERT_EQ(rc, GPUCOMPRESS_ERROR_NOT_INITIALIZED, "entropy before init");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 10: Algorithm name round-trip
 * ============================================================ */
static void test_algorithm_names() {
    printf("Test 10: algorithm name round-trip...\n");

    const char* names[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    for (int i = 0; i <= 8; i++) {
        gpucompress_algorithm_t algo = static_cast<gpucompress_algorithm_t>(i);
        const char* name = gpucompress_algorithm_name(algo);
        ASSERT_TRUE(strcmp(name, names[i]) == 0, "name matches");

        gpucompress_algorithm_t parsed = gpucompress_algorithm_from_string(name);
        ASSERT_EQ(parsed, algo, "parsed back to same enum");
    }

    // Unknown name defaults to LZ4
    gpucompress_algorithm_t unknown = gpucompress_algorithm_from_string("unknown_algo");
    ASSERT_EQ(unknown, GPUCOMPRESS_ALGO_LZ4, "unknown defaults to LZ4");

    // Out-of-range enum returns "unknown"
    const char* oor = gpucompress_algorithm_name(static_cast<gpucompress_algorithm_t>(99));
    ASSERT_TRUE(strcmp(oor, "unknown") == 0, "out-of-range returns unknown");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 11: Error string coverage
 * ============================================================ */
static void test_error_strings() {
    printf("Test 11: error string coverage...\n");

    // All defined error codes should return non-null, non-"Unknown" strings
    gpucompress_error_t codes[] = {
        GPUCOMPRESS_SUCCESS,
        GPUCOMPRESS_ERROR_INVALID_INPUT,
        GPUCOMPRESS_ERROR_CUDA_FAILED,
        GPUCOMPRESS_ERROR_COMPRESSION,
        GPUCOMPRESS_ERROR_DECOMPRESSION,
        GPUCOMPRESS_ERROR_OUT_OF_MEMORY,
        GPUCOMPRESS_ERROR_INVALID_HEADER,
        GPUCOMPRESS_ERROR_NOT_INITIALIZED,
        GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL
    };

    for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++) {
        const char* msg = gpucompress_error_string(codes[i]);
        ASSERT_TRUE(msg != nullptr, "error string not null");
        ASSERT_TRUE(strcmp(msg, "Unknown error") != 0, "error string is defined");
    }

    // Out-of-range should return "Unknown error"
    const char* unk = gpucompress_error_string(static_cast<gpucompress_error_t>(-99));
    ASSERT_TRUE(strcmp(unk, "Unknown error") == 0, "out-of-range returns Unknown error");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== API Test Suite ===\n\n");

    // Tests that don't need init first
    test_not_initialized();
    test_algorithm_names();
    test_error_strings();

    // Tests that manage their own init/cleanup
    test_double_cleanup();
    test_ref_counting();
    test_roundtrip_lz4();
    test_roundtrip_zstd();
    test_decompress_truncated();
    test_decompress_bad_magic();
    test_stats_consistency();
    test_null_params();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
