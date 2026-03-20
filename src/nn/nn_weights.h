/**
 * @file nn_weights.h
 * @brief Shared NN weight layout and dimension constants.
 *
 * Used by both nn_gpu.cu (GPU inference) and nn_reinforce.cpp (CPU backward pass).
 */

#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

static constexpr int NN_INPUT_DIM  = 15;
static constexpr int NN_HIDDEN_DIM = 128;
static constexpr int NN_OUTPUT_DIM = 4;
static constexpr int NN_NUM_CONFIGS = 32;  // 8 algos x 2 quant x 2 shuffle
static constexpr int NN_MAX_SGD_SAMPLES = 8;

/** Total number of gradient floats for one gradient region. */
static constexpr int NN_SGD_GRAD_REGION =
    NN_HIDDEN_DIM * NN_INPUT_DIM + NN_HIDDEN_DIM +   // w1 + b1: 1920 + 128
    NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +  // w2 + b2: 16384 + 128
    NN_OUTPUT_DIM * NN_HIDDEN_DIM + NN_OUTPUT_DIM;   // w3 + b3: 512 + 4 = 19076

/** Total SGD buffer size: 3 regions (accumulator + workspace + EMA gradient). */
static constexpr int NN_SGD_GRAD_SIZE = 3 * NN_SGD_GRAD_REGION;

/** Input sample for GPU SGD kernel. */
struct SGDSample {
    int action;
    float actual_ratio;
    float actual_comp_time;
    float actual_decomp_time;
    float actual_psnr;
};

/** Input sample for batched deferred decomp head-only SGD. */
struct DeferredDecompSample {
    int   action;
    float entropy, mad_normalized, deriv_normalized;
    float error_bound_enc, data_size_enc;
    float actual_decomp_ms;
};

/** Output from GPU SGD kernel. */
struct SGDOutput {
    float grad_norm;
    int was_clipped;
    int sample_count;
};

/** Output from fused NN inference kernel. */
struct NNInferenceOutput {
    int action;
    float predicted_ratio;
    float predicted_comp_time;
    float predicted_decomp_time;
    float predicted_psnr;
    int is_ood;
};

/** Per-config debug output (32 entries, one per thread). */
struct NNDebugPerConfig {
    float ratio;
    float comp_time;
    float decomp_time;
    float cost;       /* w0*ct + w1*dt + w2*ds/(ratio*bw) */
};

/** All neural net weights packed contiguously (mirrors GPU layout). */
struct NNWeightsGPU {
    // Normalization parameters
    float x_means[NN_INPUT_DIM];
    float x_stds[NN_INPUT_DIM];
    float y_means[NN_OUTPUT_DIM];
    float y_stds[NN_OUTPUT_DIM];

    // Layer 1: 15 -> 128 (weight stored as [128][15])
    float w1[NN_HIDDEN_DIM * NN_INPUT_DIM];
    float b1[NN_HIDDEN_DIM];

    // Layer 2: 128 -> 128 (weight stored as [128][128])
    float w2[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float b2[NN_HIDDEN_DIM];

    // Layer 3: 128 -> 4 (weight stored as [4][128])
    float w3[NN_OUTPUT_DIM * NN_HIDDEN_DIM];
    float b3[NN_OUTPUT_DIM];

    // Feature bounds for OOD detection (v2+)
    float x_mins[NN_INPUT_DIM];
    float x_maxs[NN_INPUT_DIM];

    // Per-output learned log-variance for uncertainty weighting (Kendall 2018).
    // Each output's error is scaled by exp(-0.5 * log_var[o]) before backprop.
    // Noisy/hard outputs automatically learn a larger log_var, reducing their
    // gradient contribution to shared W1/W2 without hard-cutting them.
    // Regularization term 0.5*log_var prevents log_var → +∞ (silencing all).
    float log_var[NN_OUTPUT_DIM];

    // Previous gradient dot-product signature for anti-flip damping.
    // Stores the dot(g_current, g_prev) sign from the last SGD call.
    // When the gradient reverses direction (dot < 0), the update is damped
    // to prevent oscillation between two weight configurations.
    float prev_grad_dot;   // last gradient direction signature (for anti-flip)
    int   sgd_call_count;  // number of SGD calls (for warmup)
};

#endif /* NN_WEIGHTS_H */
