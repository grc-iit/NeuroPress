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

/** Total number of gradient floats for GPU SGD buffer. */
static constexpr int NN_SGD_GRAD_SIZE =
    NN_HIDDEN_DIM * NN_INPUT_DIM + NN_HIDDEN_DIM +   // w1 + b1: 1920 + 128
    NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +  // w2 + b2: 16384 + 128
    NN_OUTPUT_DIM * NN_HIDDEN_DIM + NN_OUTPUT_DIM;   // w3 + b3: 512 + 4 = 19076

/** Input sample for GPU SGD kernel. */
struct SGDSample {
    int action;
    float actual_ratio;
    float actual_comp_time;
    float actual_decomp_time;
    float actual_psnr;
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
    int is_ood;
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
};

#endif /* NN_WEIGHTS_H */
