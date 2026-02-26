/**
 * @file test_cpu_stats.cu
 * @brief Correctness tests for CPU-side stats computation
 *
 * Verifies that computeStatsCPU matches the GPU stats pipeline (runStatsOnlyPipeline)
 * and that NN inference produces identical actions from both paths.
 *
 * Tests:
 *   1. CPU vs GPU stats match on constant data
 *   2. CPU vs GPU stats match on linear ramp data
 *   3. CPU vs GPU stats match on quadratic data (nonzero 2nd derivative)
 *   4. CPU vs GPU stats match on sine wave data
 *   5. CPU vs GPU stats match on random data
 *   6. NN inference produces same action from CPU stats vs GPU stats
 *   7. Full ALGO_AUTO compress round-trip uses CPU stats correctly
 *   8. gpucompress_compute_stats public API uses CPU path
 *   9. Edge case: very small input (16 floats)
 *  10. Edge case: error returns for null/zero/bad inputs
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>

#include "gpucompress.h"
#include "nn/nn_weights.h"
#include "stats/stats_cpu.h"

/* Forward declarations for gpucompress namespace functions */
namespace gpucompress {

int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
);

bool loadNNFromBinary(const char* filepath);
void cleanupNN();
bool isNNLoaded();

int runNNInference(
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound, cudaStream_t stream,
    float* out_predicted_ratio = nullptr,
    float* out_predicted_comp_time = nullptr,
    int* out_top_actions = nullptr
);

} // namespace gpucompress

/* ============================================================
 * Test framework
 * ============================================================ */

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
 * Helper: write synthetic .nnwt with known weights
 * ============================================================ */
static constexpr uint32_t NN_MAGIC = 0x4E4E5754;

static bool write_synthetic_nnwt(const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t version = 2, n_layers = 3;
    uint32_t input_dim = NN_INPUT_DIM, hidden_dim = NN_HIDDEN_DIM, output_dim = NN_OUTPUT_DIM;
    file.write(reinterpret_cast<const char*>(&NN_MAGIC), 4);
    file.write(reinterpret_cast<const char*>(&version), 4);
    file.write(reinterpret_cast<const char*>(&n_layers), 4);
    file.write(reinterpret_cast<const char*>(&input_dim), 4);
    file.write(reinterpret_cast<const char*>(&hidden_dim), 4);
    file.write(reinterpret_cast<const char*>(&output_dim), 4);

    NNWeightsGPU w;
    memset(&w, 0, sizeof(w));

    for (int i = 0; i < NN_INPUT_DIM; i++) {
        w.x_means[i] = 0.0f;
        w.x_stds[i] = 1.0f;
    }
    for (int i = 0; i < NN_OUTPUT_DIM; i++) {
        w.y_means[i] = 0.0f;
        w.y_stds[i] = 1.0f;
    }

    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) w.w1[i] = 0.01f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b1[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w2[i] = 0.001f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b2[i] = 0.1f;
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) w.w3[i] = 0.01f;
    for (int i = 0; i < NN_OUTPUT_DIM; i++) w.b3[i] = 0.0f;

    file.write(reinterpret_cast<const char*>(w.x_means), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.x_stds), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_means), NN_OUTPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_stds), NN_OUTPUT_DIM * sizeof(float));

    file.write(reinterpret_cast<const char*>(w.w1), NN_HIDDEN_DIM * NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b1), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w2), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b2), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w3), NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b3), NN_OUTPUT_DIM * sizeof(float));

    for (int i = 0; i < NN_INPUT_DIM; i++) {
        w.x_mins[i] = -10.0f;
        w.x_maxs[i] = 10.0f;
    }
    file.write(reinterpret_cast<const char*>(w.x_mins), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.x_maxs), NN_INPUT_DIM * sizeof(float));

    return file.good();
}

/* ============================================================
 * Helper: compare CPU and GPU stats for a given host buffer
 *
 * Uploads data to GPU, runs both pipelines, compares within tolerance.
 * Returns true if all stats match.
 * ============================================================ */
struct StatsPair {
    double cpu_entropy, cpu_mad, cpu_deriv;
    double gpu_entropy, gpu_mad, gpu_deriv;
};

static bool compare_cpu_gpu_stats(const float* h_data, size_t num_floats,
                                   double tol, StatsPair* out, char* err_buf) {
    size_t bytes = num_floats * sizeof(float);

    // CPU stats
    int cpu_rc = gpucompress::computeStatsCPU(
        h_data, bytes, &out->cpu_entropy, &out->cpu_mad, &out->cpu_deriv);
    if (cpu_rc != 0) {
        sprintf(err_buf, "computeStatsCPU returned %d", cpu_rc);
        return false;
    }

    // GPU stats
    void* d_data = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_data, bytes);
    if (cuda_err != cudaSuccess) {
        sprintf(err_buf, "cudaMalloc failed: %s", cudaGetErrorString(cuda_err));
        return false;
    }
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int gpu_rc = gpucompress::runStatsOnlyPipeline(
        d_data, bytes, 0, &out->gpu_entropy, &out->gpu_mad, &out->gpu_deriv);
    cudaFree(d_data);

    if (gpu_rc != 0) {
        sprintf(err_buf, "runStatsOnlyPipeline returned %d", gpu_rc);
        return false;
    }

    // Compare
    if (fabs(out->cpu_entropy - out->gpu_entropy) > tol) {
        sprintf(err_buf, "entropy mismatch: cpu=%.10f gpu=%.10f diff=%.2e",
                out->cpu_entropy, out->gpu_entropy,
                fabs(out->cpu_entropy - out->gpu_entropy));
        return false;
    }
    if (fabs(out->cpu_mad - out->gpu_mad) > tol) {
        sprintf(err_buf, "MAD mismatch: cpu=%.10f gpu=%.10f diff=%.2e",
                out->cpu_mad, out->gpu_mad,
                fabs(out->cpu_mad - out->gpu_mad));
        return false;
    }
    if (fabs(out->cpu_deriv - out->gpu_deriv) > tol) {
        sprintf(err_buf, "deriv mismatch: cpu=%.10f gpu=%.10f diff=%.2e",
                out->cpu_deriv, out->gpu_deriv,
                fabs(out->cpu_deriv - out->gpu_deriv));
        return false;
    }

    return true;
}

/* ============================================================
 * Test 1: Constant data — MAD=0, deriv=0
 * ============================================================ */
static void test_constant_data() {
    TEST("CPU vs GPU stats: constant data (1M floats)");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) h_data[i] = 42.0f;

    StatsPair sp;
    char err[256];
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-6, &sp, err);
    free(h_data);

    ASSERT(ok, err);
    ASSERT(fabs(sp.cpu_mad) < 1e-9, "constant data MAD should be 0");
    ASSERT(fabs(sp.cpu_deriv) < 1e-9, "constant data deriv should be 0");

    printf("(entropy=%.4f) ", sp.cpu_entropy);
    PASS();
}

/* ============================================================
 * Test 2: Linear ramp — deriv=0 (f''=0)
 * ============================================================ */
static void test_linear_ramp() {
    TEST("CPU vs GPU stats: linear ramp (1M floats)");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) h_data[i] = (float)i;

    StatsPair sp;
    char err[256];
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-6, &sp, err);
    free(h_data);

    ASSERT(ok, err);
    ASSERT(sp.cpu_mad > 0.0, "ramp data should have MAD > 0");
    ASSERT(fabs(sp.cpu_deriv) < 1e-6, "linear data should have deriv ~0");

    printf("(entropy=%.4f mad=%.6f) ", sp.cpu_entropy, sp.cpu_mad);
    PASS();
}

/* ============================================================
 * Test 3: Quadratic data — nonzero 2nd derivative
 * f(i) = i^2 => f''(i) = 2 for all interior points
 * ============================================================ */
static void test_quadratic_data() {
    TEST("CPU vs GPU stats: quadratic data (1M floats)");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        // Scale to avoid float overflow: (i/1000)^2
        double x = (double)i / 1000.0;
        h_data[i] = (float)(x * x);
    }

    StatsPair sp;
    char err[256];
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-4, &sp, err);
    free(h_data);

    ASSERT(ok, err);
    ASSERT(sp.cpu_deriv > 0.0, "quadratic data should have nonzero 2nd derivative");

    printf("(entropy=%.4f mad=%.6f deriv=%.6f) ",
           sp.cpu_entropy, sp.cpu_mad, sp.cpu_deriv);
    PASS();
}

/* ============================================================
 * Test 4: Sine wave — smooth, periodic
 * ============================================================ */
static void test_sine_wave() {
    TEST("CPU vs GPU stats: sine wave (1M floats)");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)sin(2.0 * M_PI * (double)i / 1000.0);
    }

    StatsPair sp;
    char err[256];
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-4, &sp, err);
    free(h_data);

    ASSERT(ok, err);
    ASSERT(sp.cpu_entropy > 0.0, "sine data should have nonzero entropy");
    ASSERT(sp.cpu_mad > 0.0, "sine data should have nonzero MAD");
    ASSERT(sp.cpu_deriv > 0.0, "sine data should have nonzero 2nd derivative");

    printf("(entropy=%.4f mad=%.6f deriv=%.6f) ",
           sp.cpu_entropy, sp.cpu_mad, sp.cpu_deriv);
    PASS();
}

/* ============================================================
 * Test 5: Pseudo-random data — high entropy
 * ============================================================ */
static void test_random_data() {
    TEST("CPU vs GPU stats: pseudo-random data (1M floats)");

    const size_t N = 1024 * 1024;
    float* h_data = (float*)malloc(N * sizeof(float));
    srand(12345);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)rand() / (float)RAND_MAX * 200.0f - 100.0f;
    }

    StatsPair sp;
    char err[256];
    // Random data may have larger floating-point accumulation differences
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-3, &sp, err);
    free(h_data);

    ASSERT(ok, err);
    ASSERT(sp.cpu_entropy > 5.0, "random data should have high entropy");

    printf("(entropy=%.4f mad=%.6f deriv=%.6f) ",
           sp.cpu_entropy, sp.cpu_mad, sp.cpu_deriv);
    PASS();
}

/* ============================================================
 * Test 6: NN inference produces same action from CPU vs GPU stats
 * ============================================================ */
static void test_nn_same_action() {
    TEST("NN inference: same action from CPU stats vs GPU stats");

    const char* nnwt_path = "/tmp/test_cpu_stats.nnwt";
    ASSERT(write_synthetic_nnwt(nnwt_path), "failed to write .nnwt file");
    ASSERT(gpucompress::loadNNFromBinary(nnwt_path), "failed to load NN weights");

    // Generate test data
    const size_t N = 256 * 1024;  // 256K floats = 1 MB
    float* h_data = (float*)malloc(N * sizeof(float));
    srand(42);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)sin(2.0 * M_PI * (double)i / 500.0) + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    size_t bytes = N * sizeof(float);

    // CPU stats
    double cpu_ent, cpu_mad, cpu_deriv;
    int cpu_rc = gpucompress::computeStatsCPU(h_data, bytes, &cpu_ent, &cpu_mad, &cpu_deriv);
    ASSERT(cpu_rc == 0, "computeStatsCPU should succeed");

    // GPU stats
    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    double gpu_ent, gpu_mad, gpu_deriv;
    int gpu_rc = gpucompress::runStatsOnlyPipeline(
        d_data, bytes, 0, &gpu_ent, &gpu_mad, &gpu_deriv);
    cudaFree(d_data);
    free(h_data);
    ASSERT(gpu_rc == 0, "GPU stats pipeline should succeed");

    // NN inference from CPU stats
    float cpu_ratio = 0.0f, cpu_comp_time = 0.0f;
    int cpu_top[32];
    int cpu_action = gpucompress::runNNInference(
        cpu_ent, cpu_mad, cpu_deriv, bytes, 0.001, 0,
        &cpu_ratio, &cpu_comp_time, cpu_top);
    ASSERT(cpu_action >= 0 && cpu_action < 32, "CPU-stats action should be valid");

    // NN inference from GPU stats
    float gpu_ratio = 0.0f, gpu_comp_time = 0.0f;
    int gpu_top[32];
    int gpu_action = gpucompress::runNNInference(
        gpu_ent, gpu_mad, gpu_deriv, bytes, 0.001, 0,
        &gpu_ratio, &gpu_comp_time, gpu_top);
    ASSERT(gpu_action >= 0 && gpu_action < 32, "GPU-stats action should be valid");

    // Actions must match
    ASSERT(cpu_action == gpu_action,
           "NN action mismatch: CPU and GPU stats should produce identical action");

    printf("(action=%d ratio=%.4f) ", cpu_action, cpu_ratio);

    gpucompress::cleanupNN();
    PASS();
}

/* ============================================================
 * Test 7: Full ALGO_AUTO compress round-trip
 *
 * Exercises the complete path: computeStatsCPU → runNNInference →
 * compress → decompress. Verifies data integrity.
 * ============================================================ */
static void test_algo_auto_roundtrip() {
    TEST("ALGO_AUTO compress/decompress round-trip with CPU stats");

    const char* nnwt_path = "/tmp/test_cpu_stats.nnwt";
    ASSERT(write_synthetic_nnwt(nnwt_path), "failed to write .nnwt");

    gpucompress_error_t rc = gpucompress_init(nnwt_path);
    ASSERT(rc == GPUCOMPRESS_SUCCESS, "gpucompress_init should succeed");
    ASSERT(gpucompress_nn_is_loaded(), "NN should be loaded");

    // Generate input data (sine + noise)
    const size_t N = 256 * 1024;  // 256K floats = 1 MB
    const size_t bytes = N * sizeof(float);
    float* input = (float*)malloc(bytes);
    srand(99);
    for (size_t i = 0; i < N; i++) {
        input[i] = (float)sin(2.0 * M_PI * (double)i / 1000.0)
                   + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    // Compress with ALGO_AUTO
    size_t comp_size = bytes * 2;  // generous buffer
    void* compressed = malloc(comp_size);
    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    gpucompress_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
    cfg.error_bound = 0.0;  // lossless

    rc = gpucompress_compress(input, bytes, compressed, &comp_size, &cfg, &stats);
    ASSERT(rc == GPUCOMPRESS_SUCCESS, "compress should succeed");
    ASSERT(comp_size > 0, "compressed size should be > 0");
    ASSERT(stats.algorithm_used >= GPUCOMPRESS_ALGO_LZ4 &&
           stats.algorithm_used <= GPUCOMPRESS_ALGO_BITCOMP,
           "algorithm_used should be valid (1-8)");

    // Verify stats were populated from CPU path
    ASSERT(stats.entropy_bits >= 0.0 && stats.entropy_bits <= 8.0,
           "stats entropy should be in [0, 8]");

    printf("(algo=%d entropy=%.2f ratio=%.2f) ",
           stats.algorithm_used, stats.entropy_bits, stats.compression_ratio);

    // Decompress and verify round-trip
    size_t decomp_size = bytes;
    float* output = (float*)malloc(decomp_size);

    rc = gpucompress_decompress(compressed, comp_size, output, &decomp_size);
    ASSERT(rc == GPUCOMPRESS_SUCCESS, "decompress should succeed");
    ASSERT(decomp_size == bytes, "decompressed size should match original");

    // Bitwise comparison (lossless mode)
    bool data_match = (memcmp(input, output, bytes) == 0);
    ASSERT(data_match, "decompressed data should match original (lossless)");

    free(input);
    free(compressed);
    free(output);
    gpucompress_cleanup();
    PASS();
}

/* ============================================================
 * Test 8: gpucompress_compute_stats public API uses CPU path
 *
 * The public API should now compute stats without GPU round-trip.
 * Cross-check with direct computeStatsCPU call.
 * ============================================================ */
static void test_public_api_compute_stats() {
    TEST("gpucompress_compute_stats matches computeStatsCPU directly");

    gpucompress_error_t rc = gpucompress_init(nullptr);
    ASSERT(rc == GPUCOMPRESS_SUCCESS, "gpucompress_init should succeed");

    const size_t N = 512 * 1024;
    const size_t bytes = N * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = (float)(i % 1000) * 0.1f;
    }

    // Via public API
    double api_ent, api_mad, api_deriv;
    rc = gpucompress_compute_stats(h_data, bytes, &api_ent, &api_mad, &api_deriv);
    ASSERT(rc == GPUCOMPRESS_SUCCESS, "gpucompress_compute_stats should succeed");

    // Via direct CPU call
    double cpu_ent, cpu_mad, cpu_deriv;
    int cpu_rc = gpucompress::computeStatsCPU(h_data, bytes, &cpu_ent, &cpu_mad, &cpu_deriv);
    ASSERT(cpu_rc == 0, "computeStatsCPU should succeed");

    // Must be bitwise identical (same code path)
    ASSERT(api_ent == cpu_ent, "entropy should be identical");
    ASSERT(api_mad == cpu_mad, "MAD should be identical");
    ASSERT(api_deriv == cpu_deriv, "deriv should be identical");

    printf("(entropy=%.4f mad=%.6f deriv=%.6f) ", api_ent, api_mad, api_deriv);

    free(h_data);
    gpucompress_cleanup();
    PASS();
}

/* ============================================================
 * Test 9: Small input (16 floats — edge case)
 * ============================================================ */
static void test_small_input() {
    TEST("CPU vs GPU stats: small input (16 floats)");

    const size_t N = 16;
    float h_data[16] = {
        1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 8.0f, 4.0f, 2.0f,
        1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.5f
    };

    StatsPair sp;
    char err[256];
    bool ok = compare_cpu_gpu_stats(h_data, N, 1e-6, &sp, err);

    ASSERT(ok, err);
    ASSERT(sp.cpu_mad > 0.0, "small input should have nonzero MAD");
    ASSERT(sp.cpu_deriv > 0.0, "small input should have nonzero 2nd derivative");

    printf("(entropy=%.4f mad=%.6f deriv=%.6f) ",
           sp.cpu_entropy, sp.cpu_mad, sp.cpu_deriv);
    PASS();
}

/* ============================================================
 * Test 10: Error handling — null, zero size, non-float-aligned
 * ============================================================ */
static void test_error_cases() {
    TEST("computeStatsCPU error handling");

    double ent, mad, deriv;

    // Null input
    ASSERT(gpucompress::computeStatsCPU(nullptr, 1024, &ent, &mad, &deriv) == -1,
           "null data should return -1");

    // Zero size
    float dummy = 1.0f;
    ASSERT(gpucompress::computeStatsCPU(&dummy, 0, &ent, &mad, &deriv) == -1,
           "zero size should return -1");

    // Non-float-aligned size (e.g., 5 bytes)
    ASSERT(gpucompress::computeStatsCPU(&dummy, 5, &ent, &mad, &deriv) == -1,
           "non-aligned size should return -1");

    // Null output pointers
    ASSERT(gpucompress::computeStatsCPU(&dummy, 4, nullptr, &mad, &deriv) == -1,
           "null entropy out should return -1");
    ASSERT(gpucompress::computeStatsCPU(&dummy, 4, &ent, nullptr, &deriv) == -1,
           "null mad out should return -1");
    ASSERT(gpucompress::computeStatsCPU(&dummy, 4, &ent, &mad, nullptr) == -1,
           "null deriv out should return -1");

    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_cpu_stats ===\n\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping\n");
        return 1;
    }

    printf("--- CPU vs GPU stats comparison ---\n");
    test_constant_data();
    test_linear_ramp();
    test_quadratic_data();
    test_sine_wave();
    test_random_data();

    printf("\n--- NN inference consistency ---\n");
    test_nn_same_action();

    printf("\n--- Integration tests ---\n");
    test_algo_auto_roundtrip();
    test_public_api_compute_stats();

    printf("\n--- Edge cases ---\n");
    test_small_input();
    test_error_cases();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
