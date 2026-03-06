/**
 * @file test_nn_reinforce.cpp
 * @brief NN forward/backward correctness tests.
 *
 * Two sequential tests:
 *   1. Forward pass parity (CPU vs GPU)
 *   2. Numerical gradient check (CPU analytical vs finite-difference)
 *
 * Requires a .nnwt weights file (auto-discovered or passed via argv[1]).
 * Link against libgpucompress.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpucompress.h"
#include "nn/nn_weights.h"

/* ============================================================
 * External symbols from libgpucompress.so
 * ============================================================ */

extern "C" void* gpucompress_nn_get_device_ptr_impl(void);

namespace gpucompress {
int runNNInference(double entropy, double mad_norm, double deriv_norm,
                   size_t data_size, double error_bound,
                   cudaStream_t stream,
                   float* out_predicted_ratio = nullptr,
                   float* out_predicted_comp_time = nullptr,
                   int* out_top_actions = nullptr);
}

/* ============================================================
 * Test parameters
 * ============================================================ */

static const double TEST_ENTROPY     = 4.0;
static const double TEST_MAD         = 0.3;
static const double TEST_DERIV       = 0.2;
static const size_t TEST_DATA_SIZE   = 4096;
static const double TEST_ERROR_BOUND = 1e-7;

/* ============================================================
 * Helpers
 * ============================================================ */

/** Build the 15-element raw feature vector for a given action ID. */
static void build_input(int action, double error_bound, size_t data_size,
                         double entropy, double mad, double deriv,
                         float input_raw[15]) {
    int algo    = action % 8;
    int quant   = (action / 8) % 2;
    int shuffle = (action / 16) % 2;

    for (int i = 0; i < 8; i++)
        input_raw[i] = (i == algo) ? 1.0f : 0.0f;
    input_raw[8]  = static_cast<float>(quant);
    input_raw[9]  = static_cast<float>(shuffle);

    double eb = error_bound;
    if (eb < 1e-7) eb = 1e-7;
    input_raw[10] = static_cast<float>(log10(eb));

    double ds = static_cast<double>(data_size);
    if (ds < 1.0) ds = 1.0;
    input_raw[11] = static_cast<float>(log2(ds));

    input_raw[12] = static_cast<float>(entropy);
    input_raw[13] = static_cast<float>(mad);
    input_raw[14] = static_cast<float>(deriv);
}

/** CPU forward pass returning normalized outputs y[NN_OUTPUT_DIM]. */
static void cpu_forward(const NNWeightsGPU& w, const float input_raw[15],
                         float y_out[NN_OUTPUT_DIM]) {
    float x[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++)
        x[i] = (input_raw[i] - w.x_means[i]) / w.x_stds[i];

    float h1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++)
            sum += w.w1[j * NN_INPUT_DIM + i] * x[i];
        h1[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float h2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w2[j * NN_HIDDEN_DIM + i] * h1[i];
        h2[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = w.b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w3[j * NN_HIDDEN_DIM + i] * h2[i];
        y_out[j] = sum;
    }
}

/** Compute MSE loss on output index 2 in normalized space. */
static float compute_loss(const NNWeightsGPU& w, const float input_raw[15],
                           float target_norm) {
    float y[NN_OUTPUT_DIM];
    cpu_forward(w, input_raw, y);
    float diff = y[2] - target_norm;
    return 0.5f * diff * diff;
}

/**
 * CPU backward pass returning analytical gradients for all weights.
 * MSE loss on output index 2 only.
 */
static void cpu_backward(const NNWeightsGPU& w, const float input_raw[15],
                          float target_norm,
                          float gw1[], float gb1[],
                          float gw2[], float gb2[],
                          float gw3[], float gb3[]) {
    /* ---- Forward pass with intermediates ---- */
    float x[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++)
        x[i] = (input_raw[i] - w.x_means[i]) / w.x_stds[i];

    float z1[NN_HIDDEN_DIM], h1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++)
            sum += w.w1[j * NN_INPUT_DIM + i] * x[i];
        z1[j] = sum;
        h1[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float z2[NN_HIDDEN_DIM], h2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w2[j * NN_HIDDEN_DIM + i] * h1[i];
        z2[j] = sum;
        h2[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float y[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = w.b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w3[j * NN_HIDDEN_DIM + i] * h2[i];
        y[j] = sum;
    }

    /* ---- Backward (MSE on output[2]) ---- */
    float d3 = y[2] - target_norm;

    memset(gw1, 0, NN_HIDDEN_DIM * NN_INPUT_DIM * sizeof(float));
    memset(gb1, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw2, 0, NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb2, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw3, 0, NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb3, 0, NN_OUTPUT_DIM * sizeof(float));

    // Layer 3 gradients (only output 2)
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        gw3[2 * NN_HIDDEN_DIM + i] = d3 * h2[i];
    gb3[2] = d3;

    // Backprop to h2
    float dh2[NN_HIDDEN_DIM];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        dh2[i] = w.w3[2 * NN_HIDDEN_DIM + i] * d3;

    // ReLU backward layer 2
    float dz2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz2[j] = (z2[j] > 0.0f) ? dh2[j] : 0.0f;

    // Layer 2 gradients
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            gw2[j * NN_HIDDEN_DIM + i] = dz2[j] * h1[i];
        gb2[j] = dz2[j];
    }

    // Backprop to h1
    float dh1[NN_HIDDEN_DIM];
    memset(dh1, 0, sizeof(dh1));
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            dh1[i] += w.w2[j * NN_HIDDEN_DIM + i] * dz2[j];

    // ReLU backward layer 1
    float dz1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz1[j] = (z1[j] > 0.0f) ? dh1[j] : 0.0f;

    // Layer 1 gradients
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_INPUT_DIM; i++)
            gw1[j * NN_INPUT_DIM + i] = dz1[j] * x[i];
        gb1[j] = dz1[j];
    }
}

/* ============================================================
 * Test 1: Forward pass parity (CPU vs GPU)
 * ============================================================ */

static bool test_forward_parity(const NNWeightsGPU& h_weights) {
    printf("\n[Test 1] Forward pass parity (CPU vs GPU)\n");

    // Run GPU inference — returns winner and its predicted ratio
    float gpu_ratio = 0.0f;
    int top_actions[NN_NUM_CONFIGS];
    int winner = gpucompress::runNNInference(
        TEST_ENTROPY, TEST_MAD, TEST_DERIV,
        TEST_DATA_SIZE, TEST_ERROR_BOUND,
        nullptr, &gpu_ratio, nullptr, top_actions);

    if (winner < 0) {
        printf("  GPU inference failed\n  FAIL\n");
        return false;
    }

    printf("  GPU winner: action=%d, predicted_ratio=%.6f\n", winner, gpu_ratio);

    // CPU forward pass for the same winning action
    float input_raw[NN_INPUT_DIM];
    build_input(winner, TEST_ERROR_BOUND, TEST_DATA_SIZE,
                TEST_ENTROPY, TEST_MAD, TEST_DERIV, input_raw);

    float y[NN_OUTPUT_DIM];
    cpu_forward(h_weights, input_raw, y);

    float cpu_ratio = expm1f(y[2] * h_weights.y_stds[2] + h_weights.y_means[2]);
    float delta = fabsf(cpu_ratio - gpu_ratio);

    printf("  CPU forward for action %d: ratio=%.6f\n", winner, cpu_ratio);
    printf("  Delta: %.6f\n", delta);

    if (delta < 1e-4f) {
        printf("  PASS\n");
        return true;
    } else {
        printf("  FAIL (delta %.6f >= 1e-4)\n", delta);
        return false;
    }
}

/* ============================================================
 * Test 2: Numerical gradient check
 * ============================================================ */

static bool test_gradient_check(const NNWeightsGPU& h_weights) {
    printf("\n[Test 2] Numerical gradient check\n");

    // Fixed input (action 0 = algo 0, quant 0, shuffle 0)
    float input_raw[NN_INPUT_DIM];
    build_input(0, TEST_ERROR_BOUND, TEST_DATA_SIZE,
                TEST_ENTROPY, TEST_MAD, TEST_DERIV, input_raw);

    double target_ratio = 3.0;
    float target_norm = static_cast<float>(
        (log1p(target_ratio) - h_weights.y_means[2]) / h_weights.y_stds[2]);

    // Compute analytical gradients
    float gw1[NN_HIDDEN_DIM * NN_INPUT_DIM];
    float gb1[NN_HIDDEN_DIM];
    float gw2[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float gb2[NN_HIDDEN_DIM];
    float gw3[NN_OUTPUT_DIM * NN_HIDDEN_DIM];
    float gb3[NN_OUTPUT_DIM];

    cpu_backward(h_weights, input_raw, target_norm,
                  gw1, gb1, gw2, gb2, gw3, gb3);

    // Weight candidate descriptor
    struct WeightCheck {
        const char* name;
        int layer;
        bool is_bias;
        int index;
        float analytical;
    };

    WeightCheck checks[20];
    int n = 0;

    // 5 from w1 (spread across the array via stride 37)
    for (int i = 0; i < 5; i++) {
        int idx = (i * 37) % (NN_HIDDEN_DIM * NN_INPUT_DIM);
        checks[n].name = "w1"; checks[n].layer = 1; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw1[idx];
        n++;
    }
    // 5 from w2 (stride 131)
    for (int i = 0; i < 5; i++) {
        int idx = (i * 131) % (NN_HIDDEN_DIM * NN_HIDDEN_DIM);
        checks[n].name = "w2"; checks[n].layer = 2; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw2[idx];
        n++;
    }
    // 5 from w3 (row 2 — the only row with non-zero gradient)
    for (int i = 0; i < 5; i++) {
        int idx = 2 * NN_HIDDEN_DIM + i;
        checks[n].name = "w3"; checks[n].layer = 3; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw3[idx];
        n++;
    }
    // 2 from b1
    for (int i = 0; i < 2; i++) {
        int idx = i * 31;
        checks[n].name = "b1"; checks[n].layer = 1; checks[n].is_bias = true;
        checks[n].index = idx; checks[n].analytical = gb1[idx];
        n++;
    }
    // 2 from b2
    for (int i = 0; i < 2; i++) {
        int idx = i * 31;
        checks[n].name = "b2"; checks[n].layer = 2; checks[n].is_bias = true;
        checks[n].index = idx; checks[n].analytical = gb2[idx];
        n++;
    }
    // 1 from b3 (output 2)
    checks[n].name = "b3"; checks[n].layer = 3; checks[n].is_bias = true;
    checks[n].index = 2; checks[n].analytical = gb3[2];
    n++;

    printf("  Checking %d weights...\n", n);

    const float eps = 1e-4f;
    int pass_count = 0;
    int checked = 0;
    int kink_skipped = 0;

    float loss_center = compute_loss(h_weights, input_raw, target_norm);

    for (int c = 0; c < n; c++) {
        NNWeightsGPU w_perturb;
        memcpy(&w_perturb, &h_weights, sizeof(NNWeightsGPU));

        // Get pointer to the weight to perturb
        float* w_ptr = nullptr;
        switch (checks[c].layer) {
            case 1: w_ptr = checks[c].is_bias ? w_perturb.b1 : w_perturb.w1; break;
            case 2: w_ptr = checks[c].is_bias ? w_perturb.b2 : w_perturb.w2; break;
            case 3: w_ptr = checks[c].is_bias ? w_perturb.b3 : w_perturb.w3; break;
        }

        float orig = w_ptr[checks[c].index];

        // L(w + eps)
        w_ptr[checks[c].index] = orig + eps;
        float loss_plus = compute_loss(w_perturb, input_raw, target_norm);

        // L(w - eps)
        w_ptr[checks[c].index] = orig - eps;
        float loss_minus = compute_loss(w_perturb, input_raw, target_norm);

        // Detect ReLU kink crossing: if one-sided gradients disagree
        // significantly, the perturbation crosses a ReLU boundary and
        // the central-difference approximation is unreliable.
        float grad_right = (loss_plus - loss_center) / eps;
        float grad_left  = (loss_center - loss_minus) / eps;
        float denom = fabsf(grad_right) + fabsf(grad_left) + 1e-10f;
        float disagreement = fabsf(grad_right - grad_left) / denom;

        if (disagreement > 0.1f) {
            printf("  %s[%d]: SKIP (ReLU kink, one-sided grads disagree %.0f%%)\n",
                   checks[c].name, checks[c].index, disagreement * 100);
            kink_skipped++;
            continue;
        }

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        float analytical = checks[c].analytical;
        float rel_err = fabsf(analytical - numerical) / (fabsf(analytical) + 1e-7f);

        // 5e-3 tolerance accounts for float32 accumulation error across
        // 128-wide dot products (standard for single-precision grad checks)
        const char* status = (rel_err < 5e-3f) ? "OK" : "FAIL";
        if (rel_err < 5e-3f) pass_count++;
        checked++;

        printf("  %s[%d]: analytical=%.6f numerical=%.6f rel_err=%.2e %s\n",
               checks[c].name, checks[c].index,
               analytical, numerical, rel_err, status);
    }

    printf("  %d checked, %d skipped (ReLU kink)\n", checked, kink_skipped);

    if (pass_count == checked) {
        printf("  PASS (%d/%d within tolerance)\n", pass_count, checked);
        return true;
    } else {
        printf("  FAIL (%d/%d within tolerance)\n", pass_count, checked);
        return false;
    }
}


/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    // Find weights file
    const char* nnwt_path = nullptr;
    if (argc > 1) {
        nnwt_path = argv[1];
    } else {
        FILE* f = fopen("neural_net/weights/model.nnwt", "rb");
        if (f) {
            fclose(f);
            nnwt_path = "neural_net/weights/model.nnwt";
        }
    }

    if (!nnwt_path) {
        fprintf(stderr, "Usage: %s [path/to/model.nnwt]\n", argv[0]);
        fprintf(stderr, "  Or place weights at neural_net/weights/model.nnwt\n");
        return 1;
    }

    printf("Loading weights from: %s\n", nnwt_path);

    // Initialize library (loads NN weights to GPU)
    gpucompress_error_t err = gpucompress_init(nnwt_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(err));
        return 1;
    }

    // Get device pointer and copy weights to host
    void* d_weights = gpucompress_nn_get_device_ptr_impl();
    if (!d_weights) {
        fprintf(stderr, "No NN device weights (is the .nnwt file valid?)\n");
        gpucompress_cleanup();
        return 1;
    }

    NNWeightsGPU h_weights;
    cudaError_t cerr = cudaMemcpy(&h_weights, d_weights,
                                    sizeof(NNWeightsGPU), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(cerr));
        gpucompress_cleanup();
        return 1;
    }

    int pass_count = 0;
    int total = 2;

    if (test_forward_parity(h_weights)) pass_count++;
    if (test_gradient_check(h_weights))  pass_count++;

    printf("\n");
    if (pass_count == total)
        printf("All tests passed.\n");
    else
        printf("%d/%d tests passed.\n", pass_count, total);

    gpucompress_cleanup();
    return (pass_count == total) ? 0 : 1;
}
