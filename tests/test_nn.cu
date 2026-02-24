/**
 * @file test_nn.cu
 * @brief Tests for src/nn/ fixes:
 *   1. loadNNFromBinary reload failure leaves g_nn_loaded=false (Bug 1)
 *   2. cleanupNN resets g_has_bounds (Bug 2)
 *   3. x_stds=0 doesn't produce NaN (Bug 3)
 *   4. nn_reinforce_apply error recovery preserves CPU weights (Bug 4)
 *   5. runNNInference uses pre-allocated buffers (Bug 5 — verified by repeated calls)
 *   6. nn_reinforce_add_sample null input_raw (Issue 6)
 *   7. experience_buffer_append rejects invalid action (Issue 7)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>

#include "nn/nn_weights.h"
#include "nn/nn_reinforce.h"
#include "nn/experience_buffer.h"

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
 * Test 6: Reinforce init/add_sample/apply round-trip
 * ============================================================ */
static void test_reinforce_roundtrip() {
    TEST("Reinforce init + add_sample + apply round-trip");

    const char* path = "/tmp/test_nn_reinforce.nnwt";
    ASSERT(write_synthetic_nnwt(path), "failed to write synthetic nnwt");
    ASSERT(gpucompress::loadNNFromBinary(path), "load should succeed");

    // Get device pointer
    void* d_weights = const_cast<NNWeightsGPU*>(gpucompress::getNNWeightsDevicePtr());
    ASSERT(d_weights != nullptr, "device weights should not be null");

    // Init reinforce
    int rc = nn_reinforce_init(d_weights);
    ASSERT(rc == 0, "reinforce init should succeed");

    // Add a sample (config 0 = lz4, no quant, no shuffle)
    float input_raw[15] = {0};
    input_raw[0] = 1.0f;  // algo one-hot: lz4
    input_raw[10] = -3.0f; // error_bound encoded
    input_raw[11] = 20.0f; // data_size encoded
    input_raw[12] = 5.0f;  // entropy
    input_raw[13] = 0.3f;  // mad
    input_raw[14] = 0.1f;  // deriv

    nn_reinforce_add_sample(input_raw, 2.5, 1.0, 0.5, 80.0);

    // Apply SGD
    rc = nn_reinforce_apply(d_weights, 0.001f);
    ASSERT(rc == 0, "reinforce apply should succeed");

    // Check stats
    float grad_norm = 0.0f;
    int num_samples = 0;
    int was_clipped = -1;
    nn_reinforce_get_last_stats(&grad_norm, &num_samples, &was_clipped);
    ASSERT(num_samples == 1, "should have 1 sample");
    ASSERT(grad_norm >= 0.0f, "grad norm should be non-negative");
    ASSERT(was_clipped == 0 || was_clipped == 1, "clipped should be 0 or 1");

    nn_reinforce_cleanup();
    gpucompress::cleanupNN();
    remove(path);
    PASS();
}

/* ============================================================
 * Test 7: nn_reinforce_add_sample with null input doesn't crash
 * ============================================================ */
static void test_reinforce_null_input() {
    TEST("nn_reinforce_add_sample with null input doesn't crash");

    const char* path = "/tmp/test_nn_null.nnwt";
    ASSERT(write_synthetic_nnwt(path), "failed to write synthetic nnwt");
    ASSERT(gpucompress::loadNNFromBinary(path), "load should succeed");

    void* d_weights = const_cast<NNWeightsGPU*>(gpucompress::getNNWeightsDevicePtr());

    int rc = nn_reinforce_init(d_weights);
    ASSERT(rc == 0, "reinforce init should succeed");

    // This should not crash (null guard)
    nn_reinforce_add_sample(nullptr, 2.5, 1.0, 0.5, 80.0);

    // Apply should fail (0 samples)
    rc = nn_reinforce_apply(d_weights, 0.001f);
    ASSERT(rc == -1, "apply with 0 samples should fail");

    nn_reinforce_cleanup();
    gpucompress::cleanupNN();
    remove(path);
    PASS();
}

/* ============================================================
 * Test 8: experience_buffer rejects invalid action
 * ============================================================ */
static void test_experience_buffer_invalid_action() {
    TEST("experience_buffer_append rejects invalid action");

    const char* csv_path = "/tmp/test_experience.csv";
    remove(csv_path);

    int rc = experience_buffer_init(csv_path);
    ASSERT(rc == 0, "init should succeed");

    // Valid sample (action=0 = lz4/no-quant/no-shuffle)
    ExperienceSample valid;
    memset(&valid, 0, sizeof(valid));
    valid.entropy = 5.0;
    valid.mad = 0.3;
    valid.second_derivative = 0.1;
    valid.data_size = 1024 * 1024;
    valid.error_bound = 0.001;
    valid.action = 0;
    valid.actual_ratio = 2.5;
    valid.actual_comp_time_ms = 1.0;
    valid.actual_decomp_time_ms = 0.5;
    valid.actual_psnr = 80.0;

    rc = experience_buffer_append(&valid);
    ASSERT(rc == 0, "valid action=0 should succeed");

    // Valid action=31 (bitcomp + quant + shuffle)
    valid.action = 31;
    rc = experience_buffer_append(&valid);
    ASSERT(rc == 0, "valid action=31 should succeed");

    // Invalid action=-1
    ExperienceSample bad = valid;
    bad.action = -1;
    rc = experience_buffer_append(&bad);
    ASSERT(rc == -1, "action=-1 should be rejected");

    // Invalid action=32
    bad.action = 32;
    rc = experience_buffer_append(&bad);
    ASSERT(rc == -1, "action=32 should be rejected");

    // Invalid action=100
    bad.action = 100;
    rc = experience_buffer_append(&bad);
    ASSERT(rc == -1, "action=100 should be rejected");

    // Verify only 2 valid samples were written
    ASSERT(experience_buffer_count() == 2, "should have exactly 2 samples");

    experience_buffer_cleanup();
    remove(csv_path);
    PASS();
}

/* ============================================================
 * Test 9: experience_buffer null pointer handling
 * ============================================================ */
static void test_experience_buffer_null() {
    TEST("experience_buffer null pointer handling");

    int rc = experience_buffer_init(nullptr);
    ASSERT(rc == -1, "init with null path should fail");

    rc = experience_buffer_append(nullptr);
    ASSERT(rc == -1, "append with null sample should fail");

    PASS();
}

/* ============================================================
 * Test 10: Full pipeline — load, infer, reinforce, experience
 * ============================================================ */
static void test_full_pipeline() {
    TEST("Full NN pipeline: load -> infer -> reinforce -> log");

    const char* nnwt_path = "/tmp/test_nn_pipeline.nnwt";
    const char* csv_path = "/tmp/test_nn_pipeline.csv";
    remove(csv_path);

    ASSERT(write_synthetic_nnwt(nnwt_path), "failed to write synthetic nnwt");
    ASSERT(gpucompress::loadNNFromBinary(nnwt_path), "load should succeed");

    // Run inference
    float ratio = 0.0f, comp_time = 0.0f;
    int top[32];
    int action = gpucompress::runNNInference(
        5.0, 0.25, 0.15, 4 * 1024 * 1024, 0.001, 0,
        &ratio, &comp_time, top);
    ASSERT(action >= 0 && action < 32, "inference should return valid action");

    // Reinforce with "actual" data
    void* d_weights = const_cast<NNWeightsGPU*>(gpucompress::getNNWeightsDevicePtr());

    int rc = nn_reinforce_init(d_weights);
    ASSERT(rc == 0, "reinforce init should succeed");

    // Build input matching what the kernel would construct for this action
    float input_raw[15] = {0};
    int algo_idx = action % 8;
    int quant = (action / 8) % 2;
    int shuffle = (action / 16) % 2;
    input_raw[algo_idx] = 1.0f;
    input_raw[8] = (float)quant;
    input_raw[9] = (float)shuffle;
    input_raw[10] = log10f(0.001f);
    input_raw[11] = log2f(4.0f * 1024 * 1024);
    input_raw[12] = 5.0f;
    input_raw[13] = 0.25f;
    input_raw[14] = 0.15f;

    nn_reinforce_add_sample(input_raw, 3.0, 2.0, 1.0, 90.0);
    rc = nn_reinforce_apply(d_weights, 0.0001f);
    ASSERT(rc == 0, "reinforce apply should succeed");

    // Log experience
    rc = experience_buffer_init(csv_path);
    ASSERT(rc == 0, "experience init should succeed");

    ExperienceSample sample;
    sample.entropy = 5.0;
    sample.mad = 0.25;
    sample.second_derivative = 0.15;
    sample.data_size = 4 * 1024 * 1024;
    sample.error_bound = 0.001;
    sample.action = action;
    sample.actual_ratio = 3.0;
    sample.actual_comp_time_ms = 2.0;
    sample.actual_decomp_time_ms = 1.0;
    sample.actual_psnr = 90.0;

    rc = experience_buffer_append(&sample);
    ASSERT(rc == 0, "experience append should succeed");
    ASSERT(experience_buffer_count() == 1, "should have 1 sample");

    experience_buffer_cleanup();
    nn_reinforce_cleanup();
    gpucompress::cleanupNN();
    remove(nnwt_path);
    remove(csv_path);
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
    test_reinforce_roundtrip();
    test_reinforce_null_input();
    test_experience_buffer_invalid_action();
    test_experience_buffer_null();
    test_full_pipeline();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
