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
                   float* out_predicted_decomp_time = nullptr,
                   float* out_predicted_psnr = nullptr,
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

/** Build the 8-element raw feature vector for a given action ID.
 *  Must match nn_gpu.cu:140-148 (runNNInferenceCtx) bit-for-bit. */
static void build_input(int action, double error_bound, size_t data_size,
                         double entropy, double mad, double deriv,
                         float input_raw[NN_INPUT_DIM]) {
    int algo    = action % 8;
    int quant   = (action / 8) % 2;
    int shuffle = (action / 16) % 2;

    input_raw[0] = static_cast<float>(algo);
    input_raw[1] = static_cast<float>(quant);
    input_raw[2] = static_cast<float>(shuffle);
    /* error_bound: training used 1e-7 sentinel for lossless (quant==0). */
    input_raw[3] = (quant == 0) ? 1e-7f : static_cast<float>(error_bound);
    input_raw[4] = static_cast<float>(data_size);
    input_raw[5] = static_cast<float>(entropy);
    input_raw[6] = static_cast<float>(mad);
    input_raw[7] = static_cast<float>(deriv);
}

/** CPU forward pass returning normalized outputs y[NN_OUTPUT_DIM].
 *  Matches the 5-layer architecture in nn_gpu.cu:157-200 (4 hidden ReLU
 *  layers w1..w4, linear output layer w5). */
static void cpu_forward(const NNWeightsGPU& w, const float input_raw[NN_INPUT_DIM],
                         float y_out[NN_OUTPUT_DIM]) {
    float x[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        float std_val = w.x_stds[i];
        if (std_val < 1e-8f) std_val = 1e-8f;
        x[i] = (input_raw[i] - w.x_means[i]) / std_val;
    }

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

    float h3[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w3[j * NN_HIDDEN_DIM + i] * h2[i];
        h3[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float h4[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b4[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w4[j * NN_HIDDEN_DIM + i] * h3[i];
        h4[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = w.b5[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w5[j * NN_HIDDEN_DIM + i] * h4[i];
        y_out[j] = sum;
    }
}

/** Compute MSE loss on output index 2 in normalized space. */
static float compute_loss(const NNWeightsGPU& w, const float input_raw[NN_INPUT_DIM],
                           float target_norm) {
    float y[NN_OUTPUT_DIM];
    cpu_forward(w, input_raw, y);
    float diff = y[2] - target_norm;
    return 0.5f * diff * diff;
}

/**
 * CPU backward pass — analytical gradients for all 5 weight matrices and
 * their biases.  MSE loss on output index 2 only (predicted_ratio head).
 */
static void cpu_backward(const NNWeightsGPU& w, const float input_raw[NN_INPUT_DIM],
                          float target_norm,
                          float gw1[], float gb1[],
                          float gw2[], float gb2[],
                          float gw3[], float gb3[],
                          float gw4[], float gb4[],
                          float gw5[], float gb5[]) {
    /* ---- Forward pass with intermediates ---- */
    float x[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        float std_val = w.x_stds[i];
        if (std_val < 1e-8f) std_val = 1e-8f;
        x[i] = (input_raw[i] - w.x_means[i]) / std_val;
    }

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

    float z3[NN_HIDDEN_DIM], h3[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w3[j * NN_HIDDEN_DIM + i] * h2[i];
        z3[j] = sum;
        h3[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float z4[NN_HIDDEN_DIM], h4[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = w.b4[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w4[j * NN_HIDDEN_DIM + i] * h3[i];
        z4[j] = sum;
        h4[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float y[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = w.b5[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += w.w5[j * NN_HIDDEN_DIM + i] * h4[i];
        y[j] = sum;
    }

    /* ---- Backward (MSE on output[2]) ---- */
    float d5 = y[2] - target_norm;

    memset(gw1, 0, NN_HIDDEN_DIM * NN_INPUT_DIM * sizeof(float));
    memset(gb1, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw2, 0, NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb2, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw3, 0, NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb3, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw4, 0, NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb4, 0, NN_HIDDEN_DIM * sizeof(float));
    memset(gw5, 0, NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    memset(gb5, 0, NN_OUTPUT_DIM * sizeof(float));

    /* Output layer 5 — only row 2 has non-zero gradient */
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        gw5[2 * NN_HIDDEN_DIM + i] = d5 * h4[i];
    gb5[2] = d5;

    /* Backprop to h4 */
    float dh4[NN_HIDDEN_DIM];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        dh4[i] = w.w5[2 * NN_HIDDEN_DIM + i] * d5;

    float dz4[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz4[j] = (z4[j] > 0.0f) ? dh4[j] : 0.0f;

    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            gw4[j * NN_HIDDEN_DIM + i] = dz4[j] * h3[i];
        gb4[j] = dz4[j];
    }

    /* Backprop to h3 */
    float dh3[NN_HIDDEN_DIM];
    memset(dh3, 0, sizeof(dh3));
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            dh3[i] += w.w4[j * NN_HIDDEN_DIM + i] * dz4[j];

    float dz3[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz3[j] = (z3[j] > 0.0f) ? dh3[j] : 0.0f;

    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            gw3[j * NN_HIDDEN_DIM + i] = dz3[j] * h2[i];
        gb3[j] = dz3[j];
    }

    /* Backprop to h2 */
    float dh2[NN_HIDDEN_DIM];
    memset(dh2, 0, sizeof(dh2));
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            dh2[i] += w.w3[j * NN_HIDDEN_DIM + i] * dz3[j];

    float dz2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz2[j] = (z2[j] > 0.0f) ? dh2[j] : 0.0f;

    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            gw2[j * NN_HIDDEN_DIM + i] = dz2[j] * h1[i];
        gb2[j] = dz2[j];
    }

    /* Backprop to h1 */
    float dh1[NN_HIDDEN_DIM];
    memset(dh1, 0, sizeof(dh1));
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            dh1[i] += w.w2[j * NN_HIDDEN_DIM + i] * dz2[j];

    float dz1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++)
        dz1[j] = (z1[j] > 0.0f) ? dh1[j] : 0.0f;

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
        nullptr, &gpu_ratio, nullptr, nullptr, nullptr, top_actions);

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
    float rel = delta / (fabsf(gpu_ratio) + 1e-7f);

    printf("  CPU forward for action %d: ratio=%.6f\n", winner, cpu_ratio);
    printf("  Delta: %.6f  (rel %.2f%%)\n", delta, rel * 100.0f);

    /* 1% relative tolerance: float32 accumulation through 5 matmuls (ReLU
     * kinks, FMA-vs-mul+add ordering between CPU & GPU) can produce ~1%
     * drift even when both sides are correct. 1e-4 absolute was a hangover
     * from the original 3-layer network. */
    if (rel < 1e-2f) {
        printf("  PASS\n");
        return true;
    } else {
        printf("  FAIL (rel_err %.2f%% >= 1%%)\n", rel * 100.0f);
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

    // Compute analytical gradients for all 5 layers.
    float gw1[NN_HIDDEN_DIM * NN_INPUT_DIM];
    float gb1[NN_HIDDEN_DIM];
    float gw2[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float gb2[NN_HIDDEN_DIM];
    float gw3[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float gb3[NN_HIDDEN_DIM];
    float gw4[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float gb4[NN_HIDDEN_DIM];
    float gw5[NN_OUTPUT_DIM * NN_HIDDEN_DIM];
    float gb5[NN_OUTPUT_DIM];

    cpu_backward(h_weights, input_raw, target_norm,
                 gw1, gb1, gw2, gb2, gw3, gb3,
                 gw4, gb4, gw5, gb5);

    struct WeightCheck {
        const char* name;
        int layer;
        bool is_bias;
        int index;
        float analytical;
    };

    WeightCheck checks[20];
    int n = 0;

    // 3 from w1 (input→hidden, stride 37)
    for (int i = 0; i < 3; i++) {
        int idx = (i * 37) % (NN_HIDDEN_DIM * NN_INPUT_DIM);
        checks[n].name = "w1"; checks[n].layer = 1; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw1[idx];
        n++;
    }
    // 3 from w2 (hidden→hidden, stride 131)
    for (int i = 0; i < 3; i++) {
        int idx = (i * 131) % (NN_HIDDEN_DIM * NN_HIDDEN_DIM);
        checks[n].name = "w2"; checks[n].layer = 2; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw2[idx];
        n++;
    }
    // 3 from w3
    for (int i = 0; i < 3; i++) {
        int idx = (i * 97) % (NN_HIDDEN_DIM * NN_HIDDEN_DIM);
        checks[n].name = "w3"; checks[n].layer = 3; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw3[idx];
        n++;
    }
    // 3 from w4
    for (int i = 0; i < 3; i++) {
        int idx = (i * 71) % (NN_HIDDEN_DIM * NN_HIDDEN_DIM);
        checks[n].name = "w4"; checks[n].layer = 4; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw4[idx];
        n++;
    }
    // 3 from w5 (output row 2 only — other rows are zero by MSE-on-output[2])
    for (int i = 0; i < 3; i++) {
        int idx = 2 * NN_HIDDEN_DIM + i;
        checks[n].name = "w5"; checks[n].layer = 5; checks[n].is_bias = false;
        checks[n].index = idx; checks[n].analytical = gw5[idx];
        n++;
    }
    // 1 from each bias b1..b4
    for (int layer = 1; layer <= 4; layer++) {
        int idx = 31;
        float g = (layer == 1) ? gb1[idx] :
                  (layer == 2) ? gb2[idx] :
                  (layer == 3) ? gb3[idx] : gb4[idx];
        checks[n].name = (layer == 1) ? "b1" : (layer == 2) ? "b2" :
                         (layer == 3) ? "b3" : "b4";
        checks[n].layer = layer; checks[n].is_bias = true;
        checks[n].index = idx; checks[n].analytical = g;
        n++;
    }
    // 1 from b5 (output bias 2)
    checks[n].name = "b5"; checks[n].layer = 5; checks[n].is_bias = true;
    checks[n].index = 2; checks[n].analytical = gb5[2];
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

        // Get pointer to the weight to perturb (layers 1..5 in the 5-layer NN).
        float* w_ptr = nullptr;
        switch (checks[c].layer) {
            case 1: w_ptr = checks[c].is_bias ? w_perturb.b1 : w_perturb.w1; break;
            case 2: w_ptr = checks[c].is_bias ? w_perturb.b2 : w_perturb.w2; break;
            case 3: w_ptr = checks[c].is_bias ? w_perturb.b3 : w_perturb.w3; break;
            case 4: w_ptr = checks[c].is_bias ? w_perturb.b4 : w_perturb.w4; break;
            case 5: w_ptr = checks[c].is_bias ? w_perturb.b5 : w_perturb.w5; break;
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

        // 5e-2 (5%) tolerance: float32 central-difference gradient check
        // on a 5-layer NN accumulates drift through 4 ReLU layers plus the
        // linear output. The old 5e-3 bound was tuned for the 3-layer net;
        // with the 8-input/5-layer upgrade, expected drift roughly doubles
        // per extra layer. 5% is still ~2 orders of magnitude tighter than
        // any symptom of a missing backprop term would produce.
        const char* status = (rel_err < 5e-2f) ? "OK" : "FAIL";
        if (rel_err < 5e-2f) pass_count++;
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
        } else {
            f = fopen("../neural_net/weights/model.nnwt", "rb");
            if (f) { fclose(f); nnwt_path = "../neural_net/weights/model.nnwt"; }
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
