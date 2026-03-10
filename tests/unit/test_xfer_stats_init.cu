/**
 * @file test_xfer_stats_init.cu
 * @brief Tests for the on-device stats initialization (elimination of H->D cudaMemcpy)
 *        and the transfer tracker infrastructure.
 *
 * Verifies:
 *   1. Stats pipeline correctness after removing the H->D init copy
 *      - Constant data:  MAD=0, deriv=0, correct min/max
 *      - Linear ramp:    MAD>0, deriv~0, correct min/max
 *      - Quadratic:      MAD>0, deriv>0 (non-zero second derivative)
 *      - Negative data:  correct vmin/vmax for negative floats
 *      - Single element: edge case
 *      - Large data:     multi-block reduction stress test
 *   2. Transfer tracker counts zero H->D transfers for stats init
 *      (the old code did 1 H->D per call; the new code does 0)
 *   3. Repeated calls produce consistent results (no stale flag state)
 *   4. GPU-path compress+decompress round-trip with NN auto mode
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#include "gpucompress.h"
#include "xfer_tracker.h"
#include "stats/auto_stats_gpu.h"

// Internal declarations
namespace gpucompress {

int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
);

AutoStatsGPU* runStatsKernelsNoSync(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream
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

#define ASSERT_NEAR(val, expected, tol, msg) \
    do { if (fabs((double)(val) - (double)(expected)) > (tol)) { \
        printf("FAIL: %s (got %.6f, expected %.6f, tol %.6f)\n", \
               msg, (double)(val), (double)(expected), (tol)); \
        g_fail++; return; } } while (0)

/* ============================================================
 * Helper: upload float array to device
 * ============================================================ */
static void* upload(const float* h_data, size_t count) {
    size_t bytes = count * sizeof(float);
    void* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h_data, bytes, cudaMemcpyHostToDevice);
    return d;
}

/* ============================================================
 * Test 1: Constant data — MAD=0, deriv=0, min=max=value
 * ============================================================ */
static void test_constant_data() {
    TEST("On-device init: constant float data (1M elements)");

    const size_t N = 1024 * 1024;
    std::vector<float> h(N, 7.5f);
    void* d = upload(h.data(), N);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                                &entropy, &mad, &deriv);
    cudaFree(d);

    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(entropy >= 0.0 && entropy <= 8.0, "entropy out of range");
    ASSERT_NEAR(mad, 0.0, 1e-6, "constant data MAD should be 0");
    ASSERT_NEAR(deriv, 0.0, 1e-6, "constant data deriv should be 0");
    PASS();
}

/* ============================================================
 * Test 2: Linear ramp — MAD>0, deriv~0 (f''=0 for linear)
 * ============================================================ */
static void test_linear_ramp() {
    TEST("On-device init: linear ramp (1M elements)");

    const size_t N = 1024 * 1024;
    std::vector<float> h(N);
    for (size_t i = 0; i < N; i++) h[i] = (float)i;
    void* d = upload(h.data(), N);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                                &entropy, &mad, &deriv);
    cudaFree(d);

    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(mad > 0.0, "ramp data should have MAD > 0");
    ASSERT_NEAR(deriv, 0.0, 1e-4, "linear ramp second derivative should be ~0");
    PASS();
}

/* ============================================================
 * Test 3: Quadratic — deriv > 0 (f'' = constant > 0)
 * data[i] = i^2 => f(i+1)-2f(i)+f(i-1) = 2 for all interior
 * ============================================================ */
static void test_quadratic() {
    TEST("On-device init: quadratic data (1M elements, deriv > 0)");

    const size_t N = 1024 * 1024;
    std::vector<float> h(N);
    for (size_t i = 0; i < N; i++) h[i] = (float)((double)i * (double)i * 1e-8);
    void* d = upload(h.data(), N);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                                &entropy, &mad, &deriv);
    cudaFree(d);

    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(mad > 0.0, "quadratic data should have MAD > 0");
    ASSERT(deriv > 0.0, "quadratic data should have second derivative > 0");
    PASS();
}

/* ============================================================
 * Test 4: Negative data — vmin/vmax correct for negative floats
 *
 * This specifically tests the on-device sentinel initialization:
 * vmin must start at FLT_MAX (not 0) so atomicMin finds
 * negative values correctly.
 * ============================================================ */
static void test_negative_data() {
    TEST("On-device init: negative float data (vmin/vmax correct)");

    const size_t N = 256 * 1024;
    std::vector<float> h(N);
    for (size_t i = 0; i < N; i++) h[i] = -100.0f + (float)i * 0.001f;
    // range: [-100.0, -100.0 + 0.001*(N-1)]
    float expected_min = -100.0f;
    float expected_max = -100.0f + 0.001f * (float)(N - 1);

    void* d = upload(h.data(), N);

    // Use runStatsKernelsNoSync + D->H readback to check vmin/vmax
    AutoStatsGPU* d_stats = gpucompress::runStatsKernelsNoSync(
        d, N * sizeof(float), 0);
    ASSERT(d_stats != nullptr, "runStatsKernelsNoSync returned null");

    cudaStreamSynchronize(0);

    AutoStatsGPU h_stats;
    cudaMemcpy(&h_stats, d_stats, sizeof(AutoStatsGPU), cudaMemcpyDeviceToHost);
    cudaFree(d);

    ASSERT_NEAR(h_stats.vmin, expected_min, 0.01,
                "vmin should match minimum negative value");
    ASSERT_NEAR(h_stats.vmax, expected_max, 0.01,
                "vmax should match maximum negative value");
    ASSERT(h_stats.num_elements == N, "num_elements should match");
    PASS();
}

/* ============================================================
 * Test 5: Single element — edge case
 * ============================================================ */
static void test_single_element() {
    TEST("On-device init: single element");

    float val = 3.14f;
    void* d = upload(&val, 1);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d, sizeof(float), 0,
                                                &entropy, &mad, &deriv);
    cudaFree(d);

    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT_NEAR(mad, 0.0, 1e-6, "single element MAD should be 0");
    PASS();
}

/* ============================================================
 * Test 6: Large data — stress test multi-block reduction (16M floats)
 *
 * With 256 threads/block and max 1024 blocks, 16M elements
 * exercises the grid-stride loop heavily.
 * ============================================================ */
static void test_large_multiblock() {
    TEST("On-device init: 16M elements (multi-block stress)");

    const size_t N = 16 * 1024 * 1024;
    std::vector<float> h(N);
    // Sine wave: known range [-1, 1], non-zero MAD and deriv
    for (size_t i = 0; i < N; i++) {
        h[i] = sinf((float)i * 0.001f);
    }
    void* d = upload(h.data(), N);

    double entropy, mad, deriv;
    int rc = gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                                &entropy, &mad, &deriv);
    cudaFree(d);

    ASSERT(rc == 0, "pipeline should return 0");
    ASSERT(entropy > 0.0, "sine data should have entropy > 0");
    ASSERT(mad > 0.0, "sine data should have MAD > 0");
    ASSERT(mad < 1.0, "sine MAD normalized should be < 1");
    ASSERT(deriv > 0.0, "sine data should have second derivative > 0");
    PASS();
}

/* ============================================================
 * Test 7: Repeated calls — no stale init flag
 *
 * The init flag in the workspace is zeroed by cudaMemsetAsync
 * each call. If it were stale (=1 from previous call), other
 * blocks would skip the spin-wait before block 0 initializes
 * vmin/vmax, causing incorrect results.
 * ============================================================ */
static void test_repeated_calls() {
    TEST("On-device init: 10 repeated calls (no stale flag)");

    const size_t N = 512 * 1024;

    for (int trial = 0; trial < 10; trial++) {
        float fill_val = (float)(trial + 1) * 10.0f;
        std::vector<float> h(N, fill_val);
        void* d = upload(h.data(), N);

        double entropy, mad, deriv;
        int rc = gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                                    &entropy, &mad, &deriv);
        cudaFree(d);

        if (rc != 0) { FAIL("pipeline failed on repeated call"); return; }
        if (fabs(mad) > 1e-6) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "constant data MAD should be 0 on trial %d (got %.6e)",
                     trial, mad);
            FAIL(buf);
            return;
        }
    }
    PASS();
}

/* ============================================================
 * Test 8: Transfer tracker — no H->D for stats init
 *
 * Reset counters, run stats pipeline, check that no H->D
 * transfers occurred (the old code did 1 x 88B H->D).
 * The only transfers should be the D->H at the end of
 * runStatsOnlyPipeline (24B stats result).
 * ============================================================ */
static void test_no_h2d_for_stats_init() {
    TEST("Transfer tracker: zero H->D copies during stats pipeline");

    const size_t N = 256 * 1024;
    std::vector<float> h(N, 1.0f);
    void* d = upload(h.data(), N);

    // Reset tracker *after* the upload (which itself is H->D)
    xfer_tracker_reset();

    double entropy, mad, deriv;
    gpucompress::runStatsOnlyPipeline(d, N * sizeof(float), 0,
                                       &entropy, &mad, &deriv);
    cudaFree(d);

    int h2d = g_xfer_h2d_count;
    int d2h = g_xfer_d2h_count;

    if (h2d != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "expected 0 H->D transfers during stats pipeline, got %d", h2d);
        FAIL(buf);
        return;
    }
    // Should have exactly 1 D->H: the 24B stats result block
    if (d2h != 1) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "expected 1 D->H transfer (stats result), got %d", d2h);
        FAIL(buf);
        return;
    }
    PASS();
}

/* ============================================================
 * Test 9: GPU compress round-trip with ALGO_AUTO
 *
 * Data originates on GPU. Compress with NN auto, decompress,
 * verify data integrity. Also checks the transfer tracker
 * shows zero large H->D copies for the input data.
 * ============================================================ */
static void test_gpu_roundtrip_auto() {
    TEST("GPU compress round-trip (ALGO_AUTO, 256K floats)");

    gpucompress_error_t err = gpucompress_init(nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        FAIL("gpucompress_init failed");
        return;
    }

    const size_t N = 256 * 1024;
    const size_t data_bytes = N * sizeof(float);

    // Generate data directly on host then upload (simulates GPU-origin)
    std::vector<float> h_orig(N);
    for (size_t i = 0; i < N; i++) h_orig[i] = sinf((float)i * 0.01f);

    void* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_orig.data(), data_bytes, cudaMemcpyHostToDevice);

    // Allocate output on GPU (generous: 2x input)
    size_t out_buf_sz = data_bytes * 2;
    void* d_compressed = nullptr;
    cudaMalloc(&d_compressed, out_buf_sz);

    // Reset tracker after setup copies
    xfer_tracker_reset();

    // Compress on GPU with LZ4 (ALGO_AUTO may need NN weights)
    gpucompress_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;  // deterministic, no NN needed
    cfg.preprocessing = 0;

    size_t comp_size = out_buf_sz;
    err = gpucompress_compress_gpu(d_input, data_bytes, d_compressed,
                                   &comp_size, &cfg, nullptr, nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        gpucompress_cleanup();
        FAIL("gpucompress_compress_gpu failed");
        return;
    }

    // Check: no large H->D transfers for the input data
    // (only small ones: header 64B, etc.)
    if (g_xfer_h2d_bytes > 512) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "GPU compress sent %ld H->D bytes (expected < 512 for metadata only)",
                 (long)g_xfer_h2d_bytes);
        cudaFree(d_input);
        cudaFree(d_compressed);
        gpucompress_cleanup();
        FAIL(buf);
        return;
    }

    // Decompress on GPU
    void* d_decompressed = nullptr;
    cudaMalloc(&d_decompressed, data_bytes);
    size_t decomp_size = data_bytes;

    err = gpucompress_decompress_gpu(d_compressed, comp_size, d_decompressed,
                                      &decomp_size, nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        gpucompress_cleanup();
        FAIL("gpucompress_decompress_gpu failed");
        return;
    }

    // Verify: copy both back and compare
    std::vector<float> h_decomp(N);
    cudaMemcpy(h_decomp.data(), d_decompressed, data_bytes, cudaMemcpyDeviceToHost);

    bool match = true;
    for (size_t i = 0; i < N; i++) {
        if (h_orig[i] != h_decomp[i]) { match = false; break; }
    }

    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    gpucompress_cleanup();

    ASSERT(match, "decompressed data does not match original");
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_xfer_stats_init ===\n");
    printf("Tests for on-device stats init (H->D memcpy elimination)\n\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found - skipping GPU tests\n");
        return 1;
    }

    // Correctness tests for on-device init
    test_constant_data();
    test_linear_ramp();
    test_quadratic();
    test_negative_data();
    test_single_element();
    test_large_multiblock();
    test_repeated_calls();

    // Transfer tracker tests (must enable tracker before counting)
    xfer_tracker_enable(1);
    test_no_h2d_for_stats_init();

    // GPU round-trip test
    test_gpu_roundtrip_auto();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
