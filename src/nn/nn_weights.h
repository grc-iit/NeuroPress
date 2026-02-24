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
