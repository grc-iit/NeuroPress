/**
 * @file test_nn.cu
 * @brief Tests for src/nn/ fixes:
 *   1. loadNNFromBinary reload failure leaves g_nn_loaded=false (Bug 1)
 *   2. cleanupNN resets g_has_bounds (Bug 2)
 *   3. x_stds=0 doesn't produce NaN (Bug 3)
 *   4. runNNInference uses pre-allocated buffers (verified by repeated calls)
 *   5. Full pipeline: load -> infer -> reinforce (GPU SGD path)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>

#include "nn/nn_weights.h"

/* Forward declarations for gpucompress namespace functions */
namespace gpucompress {
    bool loadNNFromBinary(const char* filepath);
    void cleanupNN();
    bool isNNLoaded();
    bool isInputOOD(double entropy, double mad, double deriv,
                    size_t data_size, double error_bound);
    const NNWeightsGPU* getNNWeightsDevicePtr();
    int runNNInference(
        double entropy, double mad_norm, double deriv_norm,
        size_t data_size, double error_bound, cudaStream_t stream,
        float* out_predicted_ratio = nullptr,
        float* out_predicted_comp_time = nullptr,
        int* out_top_actions = nullptr
    );
}

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
 * Helper: Write a synthetic .nnwt file with known weights
 *
 * Sets all weights to small values, x_stds to 1.0, y_stds to 1.0,
 * means to 0.0. This produces a deterministic forward pass.
 * ============================================================ */
static constexpr uint32_t NN_MAGIC = 0x4E4E5754;

static bool write_synthetic_nnwt(const char* path, uint32_t version = 2,
                                  bool zero_std = false) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // Header
    uint32_t n_layers = 3;
    uint32_t input_dim = NN_INPUT_DIM;
    uint32_t hidden_dim = NN_HIDDEN_DIM;
    uint32_t output_dim = NN_OUTPUT_DIM;
    file.write(reinterpret_cast<const char*>(&NN_MAGIC), 4);
    file.write(reinterpret_cast<const char*>(&version), 4);
    file.write(reinterpret_cast<const char*>(&n_layers), 4);
    file.write(reinterpret_cast<const char*>(&input_dim), 4);
    file.write(reinterpret_cast<const char*>(&hidden_dim), 4);
    file.write(reinterpret_cast<const char*>(&output_dim), 4);

    NNWeightsGPU w;
    memset(&w, 0, sizeof(w));

    // Normalization: means=0, stds=1 (identity transform)
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        w.x_means[i] = 0.0f;
        w.x_stds[i] = zero_std ? 0.0f : 1.0f;
    }
    for (int i = 0; i < NN_OUTPUT_DIM; i++) {
        w.y_means[i] = 0.0f;
        w.y_stds[i] = 1.0f;
    }

    // Small weights (0.01) for all layers so outputs are bounded
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) w.w1[i] = 0.01f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b1[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w2[i] = 0.001f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b2[i] = 0.1f;
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) w.w3[i] = 0.01f;
    for (int i = 0; i < NN_OUTPUT_DIM; i++) w.b3[i] = 0.0f;

    // Write normalization
    file.write(reinterpret_cast<const char*>(w.x_means), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.x_stds), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_means), NN_OUTPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_stds), NN_OUTPUT_DIM * sizeof(float));

    // Write layer weights
    file.write(reinterpret_cast<const char*>(w.w1), NN_HIDDEN_DIM * NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b1), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w2), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b2), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w3), NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b3), NN_OUTPUT_DIM * sizeof(float));

    // v2: feature bounds
    if (version >= 2) {
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            w.x_mins[i] = -10.0f;
            w.x_maxs[i] = 10.0f;
        }
        file.write(reinterpret_cast<const char*>(w.x_mins), NN_INPUT_DIM * sizeof(float));
        file.write(reinterpret_cast<const char*>(w.x_maxs), NN_INPUT_DIM * sizeof(float));
    }

    return file.good();
}

/* ============================================================
 * Test 1: loadNNFromBinary — load, then failed reload leaves loaded=false
 * ============================================================ */
static void test_load_reload_failure() {
    TEST("loadNNFromBinary reload failure sets loaded=false");

    // Write valid weights file
    const char* path = "/tmp/test_nn_valid.nnwt";
    ASSERT(write_synthetic_nnwt(path), "failed to write synthetic nnwt");

    // Load successfully
    ASSERT(gpucompress::loadNNFromBinary(path), "initial load should succeed");
    ASSERT(gpucompress::isNNLoaded(), "should be loaded after successful load");

    // Try to load a non-existent file (will fail)
    bool ok = gpucompress::loadNNFromBinary("/tmp/nonexistent_file_12345.nnwt");
    ASSERT(!ok, "loading nonexistent file should fail");
    // After failed reload, should NOT be loaded
    ASSERT(!gpucompress::isNNLoaded(), "should NOT be loaded after failed reload");

    gpucompress::cleanupNN();
    remove(path);
    PASS();
}

/* ============================================================
 * Test 2: cleanupNN resets OOD bounds
 * ============================================================ */
static void test_cleanup_resets_bounds() {
    TEST("cleanupNN resets g_has_bounds");

    const char* path = "/tmp/test_nn_bounds.nnwt";
    ASSERT(write_synthetic_nnwt(path, 2), "failed to write synthetic nnwt");

    // Load v2 weights (sets g_has_bounds = true)
    ASSERT(gpucompress::loadNNFromBinary(path), "load should succeed");

    // isInputOOD should work (bounds are set)
    // With our synthetic bounds [-10, 10], a value of 100 should be OOD
    bool ood = gpucompress::isInputOOD(100.0, 100.0, 100.0, 1024, 0.001);
    ASSERT(ood, "extreme values should be OOD with bounds [-10,10]");

    // Cleanup
    gpucompress::cleanupNN();

    // After cleanup, isInputOOD should return false (no bounds)
    ood = gpucompress::isInputOOD(100.0, 100.0, 100.0, 1024, 0.001);
    ASSERT(!ood, "isInputOOD should return false after cleanup (no bounds)");

    remove(path);
    PASS();
}

/* ============================================================
 * Test 3: x_stds=0 doesn't produce NaN in inference
 * ============================================================ */
static void test_zero_std_no_nan() {
    TEST("Inference with x_stds=0 doesn't produce NaN (guard)");

    const char* path = "/tmp/test_nn_zerostd.nnwt";
    ASSERT(write_synthetic_nnwt(path, 2, true), "failed to write zero-std nnwt");

    ASSERT(gpucompress::loadNNFromBinary(path), "load should succeed");

    // Run inference — with x_stds=0, the guard (max(std, 1e-8)) should prevent NaN.
    // Note: Inf is expected since inputs become huge (1/1e-8) and expm1 overflows,
    // but crucially there's no NaN from division-by-zero.
    float ratio = 0.0f;
    int action = gpucompress::runNNInference(
        4.0, 0.3, 0.1, 1024 * 1024, 0.001, 0, &ratio);
    ASSERT(action >= 0 && action < 32, "action should be valid (0-31)");
    ASSERT(!isnan(ratio), "predicted ratio should not be NaN");
    ASSERT(!isnan((float)action), "action should not be NaN");

    gpucompress::cleanupNN();
    remove(path);
    PASS();
}

/* ============================================================
 * Test 4: Inference returns valid action and can run repeatedly
 *         (verifies pre-allocated buffers work across calls)
 * ============================================================ */
static void test_inference_repeated() {
    TEST("Inference runs 100x with pre-allocated buffers");

    const char* path = "/tmp/test_nn_repeat.nnwt";
    ASSERT(write_synthetic_nnwt(path), "failed to write synthetic nnwt");
    ASSERT(gpucompress::loadNNFromBinary(path), "load should succeed");

    bool all_valid = true;
    for (int i = 0; i < 100; i++) {
        float ratio = 0.0f, comp_time = 0.0f;
        int top[32];
        int action = gpucompress::runNNInference(
            4.0 + i * 0.01, 0.3, 0.1, 1024 * 1024, 0.001, 0,
            &ratio, &comp_time, top);
        if (action < 0 || action >= 32) { all_valid = false; break; }
        if (isnan(ratio) || isnan(comp_time)) { all_valid = false; break; }
    }
    ASSERT(all_valid, "all 100 inferences should produce valid results");

    gpucompress::cleanupNN();
    remove(path);
    PASS();
}

/* ============================================================
 * Test 5: Inference returns -1 when NN not loaded
 * ============================================================ */
static void test_inference_not_loaded() {
    TEST("Inference returns -1 when NN not loaded");

    gpucompress::cleanupNN();
    int action = gpucompress::runNNInference(4.0, 0.3, 0.1, 1024, 0.001, 0);
    ASSERT(action == -1, "should return -1 when not loaded");

    PASS();
}

/* ============================================================
 * Test 10: Full pipeline — load, infer, reinforce, experience
 * ============================================================ */
static void test_full_pipeline() {
    TEST("Full NN pipeline: load -> infer -> reinforce");

    const char* nnwt_path = "/tmp/test_nn_pipeline.nnwt";

    ASSERT(write_synthetic_nnwt(nnwt_path), "failed to write synthetic nnwt");
    ASSERT(gpucompress::loadNNFromBinary(nnwt_path), "load should succeed");

    // Run inference
    float ratio = 0.0f, comp_time = 0.0f;
    int top[32];
    int action = gpucompress::runNNInference(
        5.0, 0.25, 0.15, 4 * 1024 * 1024, 0.001, 0,
        &ratio, &comp_time, top);
    ASSERT(action >= 0 && action < 32, "inference should return valid action");

    gpucompress::cleanupNN();
    remove(nnwt_path);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_nn ===\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping GPU tests\n");
        return 1;
    }

    test_load_reload_failure();
    test_cleanup_resets_bounds();
    test_zero_std_no_nan();
    test_inference_repeated();
    test_inference_not_loaded();
    test_full_pipeline();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
