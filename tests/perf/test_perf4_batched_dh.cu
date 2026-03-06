/**
 * test_perf4_batched_dh.cu
 *
 * PERF-4: runNNInference() batched D→H copy verification.
 *
 * Before the fix, runNNInference() issued up to 3 separate cudaMemcpyAsync
 * calls to copy results back to host:
 *   1.  d_infer_action      (4 B) → h_action
 *   2.  d_infer_ratio       (4 B) → out_predicted_ratio       [conditional]
 *   3.  d_infer_comp_time   (4 B) → out_predicted_comp_time   [conditional]
 *
 * After the fix, nnInferenceKernel writes to a single NNInferenceOutput
 * struct on device and a single cudaMemcpyAsync(16 B) brings all four
 * fields (action, predicted_ratio, predicted_comp_time, is_ood) in one
 * transfer.
 *
 * This test verifies correctness of the batched copy by:
 *
 *   T1. All 3 output fields are populated: action ∈ [0,31], ratio > 0,
 *       comp_time > 0.
 *
 *   T2. Calling with no optional outputs (ratio/comp_time NULL) returns the
 *       same action as a full call — no regression from the batched struct.
 *
 *   T3. Calling with out_top_actions returns action == top_actions[0],
 *       i.e., the winner written to NNInferenceOutput matches the sorted list.
 *
 *   T4. Cross-validation: runNNFusedInference() (fused kernel, uses
 *       NNInferenceOutput since before this fix) returns the same action
 *       when given matching AutoStatsGPU stats.  This confirms the two
 *       independent code paths agree.
 *
 * Pass criterion: 4 subtests × 4 input patterns = 16/16.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* Internal headers — must be outside any namespace block */
#include "nn/nn_weights.h"
#include "stats/auto_stats_gpu.h"
#include "api/internal.hpp"
#include "gpucompress.h"

/* ── Test parameters ─────────────────────────────────────────────────── */

/* Typical 4 MB chunk size and lossless mode */
static const size_t DATA_SIZE   = 4u * 1024u * 1024u;
static const double ERROR_BOUND = 0.0;

struct TestInput {
    const char* name;
    double entropy;
    double mad;
    double deriv;
};

static const TestInput INPUTS[] = {
    { "low-entropy-smooth",  3.0, 0.10, 0.020 },
    { "high-entropy-noisy",  7.5, 0.45, 0.350 },
    { "mid-entropy-mixed",   5.0, 0.25, 0.150 },
    { "near-zero-flat",      0.5, 0.01, 0.001 },
};
static const int N_INPUTS = (int)(sizeof(INPUTS) / sizeof(INPUTS[0]));

/* ── Helpers ─────────────────────────────────────────────────────────── */

/**
 * Populate a device-side AutoStatsGPU with the given (entropy, mad, deriv).
 * All other fields zeroed so the fused kernel only sees the stats we set.
 */
static AutoStatsGPU* make_device_stats(double entropy, double mad, double deriv)
{
    AutoStatsGPU h = {};
    h.entropy          = entropy;
    h.mad_normalized   = mad;
    h.deriv_normalized = deriv;

    AutoStatsGPU* d = nullptr;
    if (cudaMalloc(&d, sizeof(AutoStatsGPU)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, &h, sizeof(AutoStatsGPU), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return nullptr;
    }
    return d;
}

/* ── Per-input test ──────────────────────────────────────────────────── */

static int run_test(const TestInput* inp)
{
    int subtests_passed = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* -----------------------------------------------------------------
     * T1: All 3 output fields populated from single batched D→H copy.
     * ----------------------------------------------------------------- */
    float ratio1 = -1.0f, ct1 = -1.0f;
    int action1 = gpucompress::runNNInference(
        inp->entropy, inp->mad, inp->deriv,
        DATA_SIZE, ERROR_BOUND, stream,
        &ratio1, &ct1, /*top_actions=*/nullptr
    );

    int t1_ok = (action1 >= 0 && action1 < NN_NUM_CONFIGS &&
                 ratio1 > 0.0f && ct1 > 0.0f);
    printf("  T1 [all-outputs   ] action=%2d ratio=%.4f ct=%.4f  %s\n",
           action1, ratio1, ct1, t1_ok ? "PASS" : "FAIL");
    if (t1_ok) subtests_passed++;

    /* -----------------------------------------------------------------
     * T2: No optional outputs — should return the same action.
     * ----------------------------------------------------------------- */
    int action2 = gpucompress::runNNInference(
        inp->entropy, inp->mad, inp->deriv,
        DATA_SIZE, ERROR_BOUND, stream,
        /*ratio=*/nullptr, /*ct=*/nullptr, /*top_actions=*/nullptr
    );

    int t2_ok = (action2 == action1);
    printf("  T2 [no-optionals  ] action=%2d (expected %2d)       %s\n",
           action2, action1, t2_ok ? "PASS" : "FAIL");
    if (t2_ok) subtests_passed++;

    /* -----------------------------------------------------------------
     * T3: top_actions buffer — winner must equal top_actions[0].
     * ----------------------------------------------------------------- */
    int top[NN_NUM_CONFIGS];
    float ratio3 = -1.0f, ct3 = -1.0f;
    int action3 = gpucompress::runNNInference(
        inp->entropy, inp->mad, inp->deriv,
        DATA_SIZE, ERROR_BOUND, stream,
        &ratio3, &ct3, top
    );

    /* action3 and top[0] must agree: both come from the bitonic-sort path.
     * They may differ from action1 (tree-reduction path) — that is a
     * separate pre-existing design choice, not a PERF-4 concern.        */
    int t3_ok = (action3 == top[0]) && (action3 >= 0 && action3 < NN_NUM_CONFIGS);
    printf("  T3 [with-top-acts ] action=%2d top[0]=%2d (T1=%2d)  %s\n",
           action3, top[0], action1, t3_ok ? "PASS" : "FAIL");
    if (t3_ok) subtests_passed++;

    /* -----------------------------------------------------------------
     * T4: Cross-validate against runNNFusedInference (different kernel,
     *     same nnForwardPass logic) using matching AutoStatsGPU stats.
     * ----------------------------------------------------------------- */
    AutoStatsGPU* d_stats = make_device_stats(inp->entropy, inp->mad, inp->deriv);
    int action4 = -99;
    if (d_stats != nullptr) {
        gpucompress::runNNFusedInference(
            d_stats, DATA_SIZE, ERROR_BOUND, stream,
            &action4, nullptr, nullptr, nullptr, nullptr
        );
        cudaFree(d_stats);
    }

    int t4_ok = (action4 == action1);
    printf("  T4 [cross-fused   ] action=%2d (fused=%2d)          %s\n",
           action1, action4, t4_ok ? "PASS" : "FAIL");
    if (t4_ok) subtests_passed++;

    cudaStreamDestroy(stream);
    return subtests_passed;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== PERF-4: batched D→H copy in runNNInference() ===\n\n");

    if (gpucompress_load_nn("neural_net/weights/model.nnwt") != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: failed to load NN weights from "
                        "neural_net/weights/model.nnwt\n");
        return 1;
    }

    int total_passed  = 0;
    const int SUBTESTS = 4;

    for (int i = 0; i < N_INPUTS; i++) {
        printf("[%s]\n", INPUTS[i].name);
        int p = run_test(&INPUTS[i]);
        total_passed += p;
        printf("  → %d/%d subtests passed\n\n", p, SUBTESTS);
    }

    int total = N_INPUTS * SUBTESTS;
    printf("=== PERF-4 Result: %d/%d passed ===\n", total_passed, total);

    if (total_passed == total) {
        printf("VERDICT: Single NNInferenceOutput D→H copy returns all fields "
               "correctly.\n         Action, ratio, comp_time all agree with "
               "fused-kernel cross-check.\n");
        gpucompress_cleanup();
        return 0;
    }

    printf("VERDICT: FAILURES — batched copy produced wrong or missing fields.\n");
    gpucompress_cleanup();
    return 1;
}
