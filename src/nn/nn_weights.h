/**
 * @file nn_weights.h
 * @brief Shared NN weight layout and dimension constants.
 *
 * Used by both nn_gpu.cu (GPU inference) and nn_reinforce.cpp (CPU backward pass).
 */

#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

static constexpr int NN_INPUT_DIM    = 8;
static constexpr int NN_HIDDEN_DIM   = 64;
static constexpr int NN_OUTPUT_DIM   = 8;
static constexpr int NN_NUM_LAYERS   = 5;  // 4 hidden + 1 output linear layers
static constexpr int NN_INFER_OUTPUTS = 4; // only first 4 used for action selection
static constexpr int NN_NUM_CONFIGS  = 32;  // 8 algos x 2 quant x 2 shuffle
static constexpr int NN_MAX_SGD_SAMPLES = 8;

/** Total number of gradient floats for one gradient region.
 *  dW1:64×8=512  dB1:64
 *  dW2:64×64=4096 dB2:64
 *  dW3:64×64=4096 dB3:64
 *  dW4:64×64=4096 dB4:64
 *  dW5:8×64=512   dB5:8
 *  Total = 512+64+4096+64+4096+64+4096+64+512+8 = 13576
 */
static constexpr int NN_SGD_GRAD_REGION =
    NN_HIDDEN_DIM * NN_INPUT_DIM  + NN_HIDDEN_DIM +  // w1 + b1: 512  + 64
    NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +  // w2 + b2: 4096 + 64
    NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +  // w3 + b3: 4096 + 64
    NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +  // w4 + b4: 4096 + 64
    NN_OUTPUT_DIM * NN_HIDDEN_DIM + NN_OUTPUT_DIM;   // w5 + b5: 512  + 8 = 13576

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
    float error_bound_enc;  /* raw error_bound (NN input feature, z-score normalized) */
    float data_size_enc;    /* raw data_size in bytes (NN input feature, z-score normalized) */
    float actual_decomp_ms;
};

/** Output from GPU SGD kernel. */
struct SGDOutput {
    float grad_norm;
    int was_clipped;
    int sample_count;
};

/** Output from fused NN inference kernel — winner config only. */
struct NNInferenceOutput {
    int   action;
    /* Performance predictions */
    float predicted_ratio;
    float predicted_comp_time;
    float predicted_decomp_time;
    float predicted_psnr;
    /* Data-quality predictions (outputs 4-7) */
    float predicted_rmse;
    float predicted_max_error;
    float predicted_mae;
    float predicted_ssim;     /* in [0,1]: 1 - exp(-ssim_nlog) */
};

/** Per-config debug output (32 entries, one per thread). */
struct NNDebugPerConfig {
    float ratio;
    float comp_time;
    float decomp_time;
    float cost;           /* w0*ct + w1*dt + w2*ds/(ratio*bw) */
    float psnr;           /* predicted PSNR (dB) */
    float rmse;
    float max_error;
    float mae;
    float ssim;
};

/** All neural net weights packed contiguously (mirrors GPU layout). */
struct NNWeightsGPU {
    // Normalization parameters
    float x_means[NN_INPUT_DIM];
    float x_stds[NN_INPUT_DIM];
    float y_means[NN_OUTPUT_DIM];
    float y_stds[NN_OUTPUT_DIM];

    // Layer 1: 8 -> 64 (weight stored as [64][8])
    float w1[NN_HIDDEN_DIM * NN_INPUT_DIM];
    float b1[NN_HIDDEN_DIM];

    // Layer 2: 64 -> 64 (weight stored as [64][64])
    float w2[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float b2[NN_HIDDEN_DIM];

    // Layer 3: 64 -> 64 (weight stored as [64][64])
    float w3[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float b3[NN_HIDDEN_DIM];

    // Layer 4: 64 -> 64 (weight stored as [64][64])
    float w4[NN_HIDDEN_DIM * NN_HIDDEN_DIM];
    float b4[NN_HIDDEN_DIM];

    // Layer 5 (output): 64 -> 8 (weight stored as [8][64])
    float w5[NN_OUTPUT_DIM * NN_HIDDEN_DIM];
    float b5[NN_OUTPUT_DIM];

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
