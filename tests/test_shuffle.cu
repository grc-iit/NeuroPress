/**
 * @file test_shuffle.cu
 * @brief Thorough byte shuffle / unshuffle verification for 4-byte data
 *
 * Tests:
 *   1. float32 round-trip (16 MB)
 *   2. int32 round-trip (16 MB)
 *   3. Shuffled layout verification — check bytes land where expected
 *   4. Special values: NaN, Inf, -0, subnormals
 *   5. All-same values (constant data)
 *   6. All-zero data
 *   7. Leftover bytes (size not multiple of 4)
 *   8. Large data (128 MB)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cfloat>

#include "preprocessing/byte_shuffle.cuh"

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) \
    do { printf("  [TEST] %s ... ", name); fflush(stdout); } while (0)
#define PASS() \
    do { printf("PASS\n"); g_pass++; } while (0)
#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_fail++; } while (0)
#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

/* ============================================================
 * Helper: shuffle → unshuffle round-trip and verify bitwise match
 * ============================================================ */
static void verify_roundtrip(const void* h_data, size_t bytes,
                              const char* test_name) {
    TEST(test_name);

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

    /* Bitwise comparison — not float equality, actual byte identity */
    bool match = (memcmp(h_data, h_restored, bytes) == 0);
    if (!match) {
        /* Find first mismatch for debugging */
        const uint8_t* orig = (const uint8_t*)h_data;
        for (size_t i = 0; i < bytes; i++) {
            if (orig[i] != h_restored[i]) {
                char buf[128];
                snprintf(buf, sizeof(buf),
                    "byte mismatch at offset %zu: expected 0x%02X got 0x%02X",
                    i, orig[i], h_restored[i]);
                free(h_restored);
                cudaFree(d_data); cudaFree(d_shuffled); cudaFree(d_restored);
                FAIL(buf);
                return;
            }
        }
    }

    free(h_restored);
    cudaFree(d_data);
    cudaFree(d_shuffled);
    cudaFree(d_restored);
    PASS();
}

/* ============================================================
 * Test 1: float32 round-trip (sequential values)
 * ============================================================ */
static void test_float32_roundtrip() {
    const size_t N = 4 * 1024 * 1024;  /* 16 MB */
    const size_t bytes = N * sizeof(float);
    float* data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++)
        data[i] = (float)i * 1.5f - 3000000.0f;
    verify_roundtrip(data, bytes, "float32 round-trip (16 MB, sequential)");
    free(data);
}

/* ============================================================
 * Test 2: int32 round-trip (full range)
 * ============================================================ */
static void test_int32_roundtrip() {
    const size_t N = 4 * 1024 * 1024;  /* 16 MB */
    const size_t bytes = N * sizeof(int32_t);
    int32_t* data = (int32_t*)malloc(bytes);
    for (size_t i = 0; i < N; i++)
        data[i] = (int32_t)(i * 2654435761u);  /* Knuth multiplicative hash — spreads bits */
    verify_roundtrip(data, bytes, "int32 round-trip (16 MB, hash-spread)");
    free(data);
}

/* ============================================================
 * Test 3: Verify shuffled byte layout is correct
 *
 * For 4-byte elements, shuffle should produce:
 *   byte0 of all elements, byte1 of all, byte2 of all, byte3 of all
 * ============================================================ */
static void test_shuffled_layout() {
    TEST("Shuffled byte layout verification");

    /* 1024 floats = 4 KB — small enough to check every byte */
    const size_t N = 1024;
    const size_t bytes = N * sizeof(float);

    uint8_t* h_data = (uint8_t*)malloc(bytes);
    /* Fill with known pattern: element i has bytes [i*4, i*4+1, i*4+2, i*4+3] */
    for (size_t i = 0; i < bytes; i++)
        h_data[i] = (uint8_t)(i & 0xFF);

    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint8_t* d_shuffled = byte_shuffle_simple(d_data, bytes, 4);
    ASSERT(d_shuffled != nullptr, "shuffle returned null");

    cudaDeviceSynchronize();
    uint8_t* h_shuffled = (uint8_t*)malloc(bytes);
    cudaMemcpy(h_shuffled, d_shuffled, bytes, cudaMemcpyDeviceToHost);

    /*
     * Expected layout after shuffle:
     *   offset [0..N-1]     = byte0 of each element (h_data[0], h_data[4], h_data[8], ...)
     *   offset [N..2N-1]    = byte1 of each element (h_data[1], h_data[5], h_data[9], ...)
     *   offset [2N..3N-1]   = byte2 of each element
     *   offset [3N..4N-1]   = byte3 of each element
     */
    bool layout_ok = true;
    char err_msg[128] = "";
    for (size_t byte_pos = 0; byte_pos < 4 && layout_ok; byte_pos++) {
        for (size_t elem = 0; elem < N && layout_ok; elem++) {
            size_t shuffled_idx = byte_pos * N + elem;
            size_t original_idx = elem * 4 + byte_pos;
            uint8_t expected = h_data[original_idx];
            uint8_t got = h_shuffled[shuffled_idx];
            if (got != expected) {
                snprintf(err_msg, sizeof(err_msg),
                    "at byte_pos=%zu elem=%zu: expected 0x%02X got 0x%02X",
                    byte_pos, elem, expected, got);
                layout_ok = false;
            }
        }
    }

    free(h_data);
    free(h_shuffled);
    cudaFree(d_data);
    cudaFree(d_shuffled);

    if (!layout_ok) { FAIL(err_msg); return; }
    PASS();
}

/* ============================================================
 * Test 4: Special IEEE-754 values
 * ============================================================ */
static void test_special_float_values() {
    const size_t N = 1024 * 1024;  /* 4 MB */
    const size_t bytes = N * sizeof(float);
    float* data = (float*)malloc(bytes);

    for (size_t i = 0; i < N; i++) {
        switch (i % 8) {
            case 0: data[i] = 0.0f; break;
            case 1: data[i] = -0.0f; break;
            case 2: data[i] = INFINITY; break;
            case 3: data[i] = -INFINITY; break;
            case 4: data[i] = NAN; break;
            case 5: data[i] = FLT_MIN; break;          /* smallest normal */
            case 6: { uint32_t bits = 1; memcpy(&data[i], &bits, 4); break; } /* smallest subnormal */
            case 7: data[i] = FLT_MAX; break;
        }
    }

    verify_roundtrip(data, bytes,
        "Special float values (NaN, Inf, -0, subnormal, FLT_MAX) 4 MB");
    free(data);
}

/* ============================================================
 * Test 5: Constant data (all same value)
 * ============================================================ */
static void test_constant_data() {
    const size_t N = 2 * 1024 * 1024;  /* 8 MB */
    const size_t bytes = N * sizeof(float);
    float* data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++)
        data[i] = 42.0f;
    verify_roundtrip(data, bytes, "Constant data (42.0f, 8 MB)");
    free(data);
}

/* ============================================================
 * Test 6: All-zero data
 * ============================================================ */
static void test_zero_data() {
    const size_t N = 2 * 1024 * 1024;  /* 8 MB */
    const size_t bytes = N * sizeof(float);
    float* data = (float*)calloc(N, sizeof(float));
    verify_roundtrip(data, bytes, "All-zero data (8 MB)");
    free(data);
}

/* ============================================================
 * Test 7: Leftover bytes (size not multiple of 4)
 * ============================================================ */
static void test_leftover_bytes() {
    /* 4 MB + 1, +2, +3 bytes — test the leftover handling path */
    for (int extra = 1; extra <= 3; extra++) {
        const size_t bytes = 4 * 1024 * 1024 + extra;
        uint8_t* data = (uint8_t*)malloc(bytes);
        for (size_t i = 0; i < bytes; i++)
            data[i] = (uint8_t)((i * 31 + 17) & 0xFF);

        char name[64];
        snprintf(name, sizeof(name), "Leftover bytes (4 MB + %d)", extra);
        verify_roundtrip(data, bytes, name);
        free(data);
    }
}

/* ============================================================
 * Test 8: Large data (128 MB)
 * ============================================================ */
static void test_large_data() {
    const size_t N = 32 * 1024 * 1024;  /* 128 MB */
    const size_t bytes = N * sizeof(float);
    float* data = (float*)malloc(bytes);
    if (!data) {
        TEST("Large data (128 MB)");
        FAIL("malloc failed for 128 MB");
        return;
    }

    /* Pseudo-random fill */
    uint32_t seed = 12345;
    for (size_t i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        data[i] = (float)(seed >> 8) / 16777216.0f * 2000.0f - 1000.0f;
    }

    verify_roundtrip(data, bytes, "Large data round-trip (128 MB, pseudo-random)");
    free(data);
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_shuffle ===\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices — skipping\n");
        return 0;
    }

    test_float32_roundtrip();
    test_int32_roundtrip();
    test_shuffled_layout();
    test_special_float_values();
    test_constant_data();
    test_zero_data();
    test_leftover_bytes();
    test_large_data();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
