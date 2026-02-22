/**
 * @file nn_reinforce.cpp
 * @brief Online NN reinforcement — CPU forward/backward pass + SGD.
 *
 * Pure C++ math plus one cudaMemcpy call in nn_reinforce_apply().
 * Trains on all 4 outputs: index 0 (comp_time), 1 (decomp_time), 2 (ratio), 3 (psnr).
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "nn_weights.h"
#include "nn_reinforce.h"

/* ============================================================
 * Static state
 * ============================================================ */

static NNWeightsGPU h_weights;
static bool g_initialized = false;

// Gradient accumulators (same layout as weights)
static float dw1[NN_HIDDEN_DIM * NN_INPUT_DIM];
static float db1[NN_HIDDEN_DIM];
static float dw2[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
static float db2[NN_HIDDEN_DIM];
static float dw3[NN_OUTPUT_DIM * NN_HIDDEN_DIM];
static float db3[NN_OUTPUT_DIM];

static int sample_count = 0;

// Last-apply stats for external query
static float g_last_grad_norm = 0.0f;
static int   g_last_sample_count = 0;
static bool  g_last_clipped = false;

/* ============================================================
 * Helper: zero all gradient accumulators
 * ============================================================ */

static void zero_gradients() {
    memset(dw1, 0, sizeof(dw1));
    memset(db1, 0, sizeof(db1));
    memset(dw2, 0, sizeof(dw2));
    memset(db2, 0, sizeof(db2));
    memset(dw3, 0, sizeof(dw3));
    memset(db3, 0, sizeof(db3));
    sample_count = 0;
}

/* ============================================================
 * Public API
 * ============================================================ */

extern "C" int nn_reinforce_init(const void* d_weights) {
    if (d_weights == nullptr) return -1;

    cudaError_t err = cudaMemcpy(&h_weights, d_weights,
                                  sizeof(NNWeightsGPU),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "nn_reinforce_init: cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    zero_gradients();
    g_initialized = true;
    return 0;
}

extern "C" void nn_reinforce_add_sample(const float input_raw[15],
                                         double actual_ratio,
                                         double actual_comp_time,
                                         double actual_decomp_time,
                                         double actual_psnr) {
    if (!g_initialized) return;

    // ---- Standardize input ----
    float x[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        x[i] = (input_raw[i] - h_weights.x_means[i]) / h_weights.x_stds[i];
    }

    // ---- Layer 1: Linear(15,128) + ReLU ----
    float z1[NN_HIDDEN_DIM], h1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = h_weights.b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            sum += h_weights.w1[j * NN_INPUT_DIM + i] * x[i];
        }
        z1[j] = sum;
        h1[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    // ---- Layer 2: Linear(128,128) + ReLU ----
    float z2[NN_HIDDEN_DIM], h2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = h_weights.b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += h_weights.w2[j * NN_HIDDEN_DIM + i] * h1[i];
        }
        z2[j] = sum;
        h2[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    // ---- Layer 3: Linear(128,4) ----
    float y[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = h_weights.b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += h_weights.w3[j * NN_HIDDEN_DIM + i] * h2[i];
        }
        y[j] = sum;
    }

    // ---- Targets in normalized space ----
    // Output index 2 = ratio (log1p-transformed)
    float target_ratio = static_cast<float>(
        (log1p(actual_ratio) - h_weights.y_means[2]) / h_weights.y_stds[2]);

    // Output index 0 = comp_time (log1p-transformed), only if available
    bool train_comp_time = (actual_comp_time > 0.0);
    float target_ct = 0.0f;
    if (train_comp_time) {
        target_ct = static_cast<float>(
            (log1p(actual_comp_time) - h_weights.y_means[0]) / h_weights.y_stds[0]);
    }

    // Output index 1 = decomp_time (log1p-transformed), only if available
    bool train_decomp_time = (actual_decomp_time > 0.0);
    float target_dt = 0.0f;
    if (train_decomp_time) {
        target_dt = static_cast<float>(
            (log1p(actual_decomp_time) - h_weights.y_means[1]) / h_weights.y_stds[1]);
    }

    // Output index 3 = psnr (clamped, not logged), only if available
    bool train_psnr = (actual_psnr > 0.0);
    float target_psnr = 0.0f;
    if (train_psnr) {
        float clamped = static_cast<float>(fmin(actual_psnr, 120.0));
        target_psnr = (clamped - h_weights.y_means[3]) / h_weights.y_stds[3];
    }

    // ---- Backward pass (MSE loss on active outputs) ----
    // dL/dy for each output (factor of 2 absorbed into lr)
    float d3_out[NN_OUTPUT_DIM];
    memset(d3_out, 0, sizeof(d3_out));
    d3_out[2] = y[2] - target_ratio;
    if (train_comp_time) {
        d3_out[0] = y[0] - target_ct;
    }
    if (train_decomp_time) {
        d3_out[1] = y[1] - target_dt;
    }
    if (train_psnr) {
        d3_out[3] = y[3] - target_psnr;
    }

    // Layer 3 gradients (active outputs only)
    for (int out = 0; out < NN_OUTPUT_DIM; out++) {
        if (d3_out[out] == 0.0f) continue;
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            dw3[out * NN_HIDDEN_DIM + i] += d3_out[out] * h2[i];
        }
        db3[out] += d3_out[out];
    }

    // Backprop through layer 3 to h2
    float dh2[NN_HIDDEN_DIM];
    memset(dh2, 0, sizeof(dh2));
    for (int out = 0; out < NN_OUTPUT_DIM; out++) {
        if (d3_out[out] == 0.0f) continue;
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            dh2[i] += h_weights.w3[out * NN_HIDDEN_DIM + i] * d3_out[out];
        }
    }

    // ReLU backward for layer 2
    float dz2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        dz2[j] = (z2[j] > 0.0f) ? dh2[j] : 0.0f;
    }

    // Layer 2 gradients
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            dw2[j * NN_HIDDEN_DIM + i] += dz2[j] * h1[i];
        }
        db2[j] += dz2[j];
    }

    // Backprop through layer 2 to h1
    float dh1[NN_HIDDEN_DIM];
    memset(dh1, 0, sizeof(dh1));
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            dh1[i] += h_weights.w2[j * NN_HIDDEN_DIM + i] * dz2[j];
        }
    }

    // ReLU backward for layer 1
    float dz1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        dz1[j] = (z1[j] > 0.0f) ? dh1[j] : 0.0f;
    }

    // Layer 1 gradients
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            dw1[j * NN_INPUT_DIM + i] += dz1[j] * x[i];
        }
        db1[j] += dz1[j];
    }

    sample_count++;
}

extern "C" int nn_reinforce_apply(void* d_weights, float learning_rate) {
    if (!g_initialized || d_weights == nullptr || sample_count == 0) return -1;

    float inv_n = 1.0f / static_cast<float>(sample_count);

    // ---- Average gradients ----
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) dw1[i] *= inv_n;
    for (int i = 0; i < NN_HIDDEN_DIM; i++)               db1[i] *= inv_n;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) dw2[i] *= inv_n;
    for (int i = 0; i < NN_HIDDEN_DIM; i++)               db2[i] *= inv_n;
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) dw3[i] *= inv_n;
    for (int i = 0; i < NN_OUTPUT_DIM; i++)               db3[i] *= inv_n;

    // ---- Gradient clipping (L2 norm, max 1.0) ----
    float norm_sq = 0.0f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) norm_sq += dw1[i] * dw1[i];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)               norm_sq += db1[i] * db1[i];
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) norm_sq += dw2[i] * dw2[i];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)               norm_sq += db2[i] * db2[i];
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) norm_sq += dw3[i] * dw3[i];
    for (int i = 0; i < NN_OUTPUT_DIM; i++)               norm_sq += db3[i] * db3[i];

    float norm = sqrtf(norm_sq);
    g_last_grad_norm = norm;
    g_last_sample_count = sample_count;
    g_last_clipped = (norm > 1.0f);

    if (norm > 1.0f) {
        float scale = 1.0f / norm;
        for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) dw1[i] *= scale;
        for (int i = 0; i < NN_HIDDEN_DIM; i++)               db1[i] *= scale;
        for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) dw2[i] *= scale;
        for (int i = 0; i < NN_HIDDEN_DIM; i++)               db2[i] *= scale;
        for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++) dw3[i] *= scale;
        for (int i = 0; i < NN_OUTPUT_DIM; i++)               db3[i] *= scale;
    }

    // ---- SGD step: w -= lr * grad ----
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++)
        h_weights.w1[i] -= learning_rate * dw1[i];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        h_weights.b1[i] -= learning_rate * db1[i];
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++)
        h_weights.w2[i] -= learning_rate * dw2[i];
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        h_weights.b2[i] -= learning_rate * db2[i];
    for (int i = 0; i < NN_OUTPUT_DIM * NN_HIDDEN_DIM; i++)
        h_weights.w3[i] -= learning_rate * dw3[i];
    for (int i = 0; i < NN_OUTPUT_DIM; i++)
        h_weights.b3[i] -= learning_rate * db3[i];

    // ---- Copy updated weights back to GPU ----
    cudaError_t err = cudaMemcpy(d_weights, &h_weights,
                                  sizeof(NNWeightsGPU),
                                  cudaMemcpyHostToDevice);

    // Always reset gradients to avoid stale accumulation on next call
    zero_gradients();

    if (err != cudaSuccess) {
        fprintf(stderr, "nn_reinforce_apply: cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

extern "C" void nn_reinforce_get_last_stats(float* grad_norm, int* num_samples,
                                              int* was_clipped) {
    if (grad_norm)   *grad_norm   = g_last_grad_norm;
    if (num_samples) *num_samples = g_last_sample_count;
    if (was_clipped) *was_clipped = g_last_clipped ? 1 : 0;
}

extern "C" void nn_reinforce_cleanup(void) {
    zero_gradients();
    g_initialized = false;
}
