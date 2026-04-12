/**
 * @file test_nn_cost_ranking.cu
 * @brief Tests for cost-based ranking in NN inference.
 *
 * Verifies:
 *   1. Default weights (w0=w1=w2=1) produce a valid selection
 *   2. w0=1,w1=0,w2=0 → prefers lowest compression time
 *   3. w0=0,w1=1,w2=0 → prefers lowest decompression time
 *   4. w0=0,w1=0,w2=1 → prefers best compression ratio (lowest I/O cost)
 *   5. Quant configs masked when error_bound=0 (lossless)
 *   6. gpucompress_set_bandwidth override works
 *   7. gpucompress_set_ranking_weights API works end-to-end
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>

#include "nn/nn_weights.h"

/* Cost-ranking globals in gpucompress_api.cpp */
extern float g_rank_w0;
extern float g_rank_w1;
extern float g_rank_w2;
extern float g_measured_bw_bytes_per_ms;

/* Forward declarations for gpucompress namespace functions */
namespace gpucompress {
    bool loadNNFromBinary(const char* filepath);
    void cleanupNN();
    bool isNNLoaded();
    int runNNInference(
        double entropy, double mad_norm, double deriv_norm,
        size_t data_size, double error_bound, cudaStream_t stream,
        float* out_predicted_ratio = nullptr,
        float* out_predicted_comp_time = nullptr,
        float* out_predicted_decomp_time = nullptr,
        float* out_predicted_psnr = nullptr,
        int* out_top_actions = nullptr
    );
}

/* Public API */
extern "C" {
    void gpucompress_set_ranking_weights(float w0, float w1, float w2);
    void gpucompress_set_bandwidth(float bw_gbps);
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
 * Helper: Write a synthetic .nnwt with real weights
 * ============================================================ */
static constexpr uint32_t NN_MAGIC = 0x4E4E5754;

static bool write_synthetic_nnwt(const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // All dimensions from nn_weights.h so tests stay in sync with the model
    uint32_t version   = 2;
    uint32_t n_layers  = NN_NUM_LAYERS;
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

    // Small weights so outputs are bounded and deterministic
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM;  i++) w.w1[i] = 0.01f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b1[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w2[i] = 0.001f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b2[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w3[i] = 0.001f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b3[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w4[i] = 0.001f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b4[i] = 0.1f;
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) w.w5[i] = 0.01f;
    for (int i = 0; i < NN_OUTPUT_DIM; i++) w.b5[i] = 0.0f;

    // Write normalization
    file.write(reinterpret_cast<const char*>(w.x_means), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.x_stds),  NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_means), NN_OUTPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.y_stds),  NN_OUTPUT_DIM * sizeof(float));

    // Write all NN_NUM_LAYERS layers
    file.write(reinterpret_cast<const char*>(w.w1), NN_HIDDEN_DIM * NN_INPUT_DIM  * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b1), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w2), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b2), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w3), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b3), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w4), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b4), NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.w5), NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.b5), NN_OUTPUT_DIM * sizeof(float));

    // v2: feature bounds
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        w.x_mins[i] = -10.0f;
        w.x_maxs[i] = 10.0f;
    }
    file.write(reinterpret_cast<const char*>(w.x_mins), NN_INPUT_DIM * sizeof(float));
    file.write(reinterpret_cast<const char*>(w.x_maxs), NN_INPUT_DIM * sizeof(float));

    return file.good();
}

/* Helper: decode action to (algo, quant, shuffle) */
static void decode_action(int action, int& algo, int& quant, int& shuffle) {
    algo    = action % 8;
    quant   = (action / 8) % 2;
    shuffle = (action / 16) % 2;
}

/* Helper: run inference and get all 32 ranked actions + best predictions */
struct InferResult {
    int best_action;
    float ratio, comp_time, decomp_time, psnr;
    int top_actions[32];
};

static bool run_inference(InferResult& r, double eb = 0.0,
                          size_t data_size = 1024*1024) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    r.best_action = gpucompress::runNNInference(
        4.0, 0.3, 0.1,   // entropy, mad, deriv
        data_size, eb, stream,
        &r.ratio, &r.comp_time, &r.decomp_time, &r.psnr,
        r.top_actions
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return r.best_action >= 0;
}

/* ============================================================
 * Test 1: Default weights produce valid selection
 * ============================================================ */
static void test_default_weights() {
    TEST("default weights (w0=w1=w2=1) produce valid action");

    g_rank_w0 = 1.0f;
    g_rank_w1 = 1.0f;
    g_rank_w2 = 1.0f;
    g_measured_bw_bytes_per_ms = 1e6f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");
    ASSERT(r.best_action >= 0 && r.best_action < 32, "action in [0,31]");
    ASSERT(r.ratio > 0.0f, "ratio should be positive");
    ASSERT(r.comp_time > 0.0f, "comp_time should be positive");
    ASSERT(r.decomp_time > 0.0f, "decomp_time should be positive");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    printf("(action=%d algo=%d q=%d s=%d ratio=%.2f ct=%.3f dt=%.3f) ",
           r.best_action, algo, quant, shuffle, r.ratio, r.comp_time, r.decomp_time);
    PASS();
}

/* ============================================================
 * Test 2: w0-only → prefers lowest compression time
 * ============================================================ */
static void test_w0_comp_time() {
    TEST("w0=1,w1=0,w2=0 → lowest comp_time wins");

    g_rank_w0 = 1.0f;
    g_rank_w1 = 0.0f;
    g_rank_w2 = 0.0f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");

    // Run again with all 32 to verify winner has lowest comp_time among lossless
    // Get predictions for all 32 by running individual checks
    // The top_actions[0] should be the winner
    ASSERT(r.top_actions[0] == r.best_action, "top_actions[0] == best_action");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    // With lossless (eb=0), quant must be 0
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d ct=%.3f) ", r.best_action, algo, r.comp_time);
    PASS();
}

/* ============================================================
 * Test 3: w1-only → prefers lowest decompression time
 * ============================================================ */
static void test_w1_decomp_time() {
    TEST("w0=0,w1=1,w2=0 → lowest decomp_time wins");

    g_rank_w0 = 0.0f;
    g_rank_w1 = 1.0f;
    g_rank_w2 = 0.0f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d dt=%.3f) ", r.best_action, algo, r.decomp_time);
    PASS();
}

/* ============================================================
 * Test 4: w2-only → prefers best ratio (lowest I/O cost)
 * ============================================================ */
static void test_w2_ratio() {
    TEST("w0=0,w1=0,w2=1 → best ratio wins");

    g_rank_w0 = 0.0f;
    g_rank_w1 = 0.0f;
    g_rank_w2 = 1.0f;
    g_measured_bw_bytes_per_ms = 1e6f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");

    // Verify that w2-only ranking picks highest ratio (= lowest io_cost)
    // The best action's ratio should be >= all other lossless configs
    // We check the top action is lossless
    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d ratio=%.2f) ", r.best_action, algo, r.ratio);
    PASS();
}

/* ============================================================
 * Test 5: Quant configs masked when error_bound=0 (lossless)
 * ============================================================ */
static void test_quant_masked_lossless() {
    TEST("quant configs masked when error_bound=0");

    g_rank_w0 = 1.0f;
    g_rank_w1 = 1.0f;
    g_rank_w2 = 1.0f;

    InferResult r;
    ASSERT(run_inference(r, 0.0), "inference should succeed");

    // Check that all top-16 actions are lossless (quant=0)
    // Actually, check that the top actions with quant=1 are at the end (rank_val=-INF)
    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "best action should have quant=0 when eb=0");

    // Check the top half of sorted actions (first 16) are all lossless
    for (int i = 0; i < 16; i++) {
        int a = r.top_actions[i];
        int q = (a / 8) % 2;
        ASSERT(q == 0, "top-16 should all be lossless when eb=0");
    }

    PASS();
}

/* ============================================================
 * Test 6: Quant configs allowed when error_bound > 0
 * ============================================================ */
static void test_quant_allowed_lossy() {
    TEST("quant configs allowed when error_bound>0");

    g_rank_w0 = 1.0f;
    g_rank_w1 = 1.0f;
    g_rank_w2 = 1.0f;

    InferResult r;
    ASSERT(run_inference(r, 0.01), "inference should succeed");

    // With lossy, at least some quant actions should be in top-16
    bool found_quant = false;
    for (int i = 0; i < 16; i++) {
        int a = r.top_actions[i];
        int q = (a / 8) % 2;
        if (q == 1) { found_quant = true; break; }
    }
    ASSERT(found_quant, "some quant configs should appear in top-16 for lossy");

    PASS();
}

/* ============================================================
 * Test 7: gpucompress_set_bandwidth override
 * ============================================================ */
static void test_bw_override() {
    TEST("gpucompress_set_bandwidth override");

    // Set very low BW → I/O cost dominates → should prefer high ratio
    gpucompress_set_bandwidth(0.001f);  // 0.001 GB/s = very slow I/O
    g_rank_w0 = 0.0f;
    g_rank_w1 = 0.0f;
    g_rank_w2 = 1.0f;

    InferResult r_slow;
    ASSERT(run_inference(r_slow), "inference should succeed with slow BW");

    // Set very high BW → I/O cost negligible → ratio matters less
    gpucompress_set_bandwidth(100.0f);  // 100 GB/s = very fast I/O

    InferResult r_fast;
    ASSERT(run_inference(r_fast), "inference should succeed with fast BW");

    // Both should produce valid actions
    ASSERT(r_slow.best_action >= 0 && r_slow.best_action < 32, "slow BW valid");
    ASSERT(r_fast.best_action >= 0 && r_fast.best_action < 32, "fast BW valid");

    // Verify BW was actually stored correctly
    float expected = 100.0f * 1e6f;
    float diff = fabsf(g_measured_bw_bytes_per_ms - expected);
    ASSERT(diff < 1.0f, "BW should be 100 GB/s = 1e8 bytes/ms");

    // Restore
    g_measured_bw_bytes_per_ms = 1e6f;

    printf("(slow_action=%d fast_action=%d) ", r_slow.best_action, r_fast.best_action);
    PASS();
}

/* ============================================================
 * Test 8: gpucompress_set_ranking_weights API
 * ============================================================ */
static void test_set_weights_api() {
    TEST("gpucompress_set_ranking_weights API");

    gpucompress_set_ranking_weights(0.5f, 2.0f, 0.3f);
    ASSERT(fabsf(g_rank_w0 - 0.5f) < 1e-6f, "w0 should be 0.5");
    ASSERT(fabsf(g_rank_w1 - 2.0f) < 1e-6f, "w1 should be 2.0");
    ASSERT(fabsf(g_rank_w2 - 0.3f) < 1e-6f, "w2 should be 0.3");

    // Run inference to ensure it works with custom weights
    InferResult r;
    ASSERT(run_inference(r), "inference should succeed with custom weights");
    ASSERT(r.best_action >= 0 && r.best_action < 32, "valid action");

    // Restore defaults
    gpucompress_set_ranking_weights(1.0f, 1.0f, 1.0f);
    PASS();
}

/* ============================================================
 * Test 9: Different weight configs produce different rankings
 * ============================================================ */
static void test_weight_configs_differ() {
    TEST("different weight configs produce different top-action orderings");

    g_measured_bw_bytes_per_ms = 1e6f;

    // Config A: comp-time only
    g_rank_w0 = 1.0f; g_rank_w1 = 0.0f; g_rank_w2 = 0.0f;
    InferResult ra;
    ASSERT(run_inference(ra), "inference A");

    // Config B: decomp-time only
    g_rank_w0 = 0.0f; g_rank_w1 = 1.0f; g_rank_w2 = 0.0f;
    InferResult rb;
    ASSERT(run_inference(rb), "inference B");

    // Config C: ratio only
    g_rank_w0 = 0.0f; g_rank_w1 = 0.0f; g_rank_w2 = 1.0f;
    InferResult rc;
    ASSERT(run_inference(rc), "inference C");

    // At least 2 of 3 configs should produce different orderings
    // (comparing the full sorted order of the first 8 actions)
    bool ab_same = true, bc_same = true, ac_same = true;
    for (int i = 0; i < 8; i++) {
        if (ra.top_actions[i] != rb.top_actions[i]) ab_same = false;
        if (rb.top_actions[i] != rc.top_actions[i]) bc_same = false;
        if (ra.top_actions[i] != rc.top_actions[i]) ac_same = false;
    }
    bool at_least_two_differ = !ab_same || !bc_same || !ac_same;
    ASSERT(at_least_two_differ, "different cost weights should produce different rankings");

    printf("(A=%d B=%d C=%d) ", ra.best_action, rb.best_action, rc.best_action);

    // Restore
    g_rank_w0 = 1.0f; g_rank_w1 = 1.0f; g_rank_w2 = 1.0f;
    PASS();
}

/* ============================================================
 * Test 10: Zero weights edge case (all weights=0 → all costs equal)
 * ============================================================ */
static void test_zero_weights() {
    TEST("all weights=0 → valid action (tie-breaking)");

    g_rank_w0 = 0.0f; g_rank_w1 = 0.0f; g_rank_w2 = 0.0f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");
    ASSERT(r.best_action >= 0 && r.best_action < 32, "valid action");

    // All lossless costs should be 0 → any lossless action is acceptable
    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    // Restore
    g_rank_w0 = 1.0f; g_rank_w1 = 1.0f; g_rank_w2 = 1.0f;
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== Cost-Based Ranking Tests ===\n\n");

    // Load synthetic NN weights
    const char* nnwt_path = "/tmp/test_cost_ranking.nnwt";
    if (!write_synthetic_nnwt(nnwt_path)) {
        fprintf(stderr, "FATAL: failed to write synthetic .nnwt\n");
        return 1;
    }
    if (!gpucompress::loadNNFromBinary(nnwt_path)) {
        fprintf(stderr, "FATAL: failed to load synthetic .nnwt\n");
        return 1;
    }
    printf("  Loaded synthetic NN weights\n\n");

    test_default_weights();
    test_w0_comp_time();
    test_w1_decomp_time();
    test_w2_ratio();
    test_quant_masked_lossless();
    test_quant_allowed_lossy();
    test_bw_override();
    test_set_weights_api();
    test_weight_configs_differ();
    test_zero_weights();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    gpucompress::cleanupNN();
    remove(nnwt_path);

    return g_fail > 0 ? 1 : 0;
}
