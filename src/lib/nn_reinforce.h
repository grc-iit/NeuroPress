/**
 * @file nn_reinforce.h
 * @brief Online reinforcement for NN weights via single-sample SGD on CPU.
 *
 * When a prediction is badly wrong (MAPE > threshold), the caller:
 *   1. Calls nn_reinforce_init() once to snapshot GPU weights to host
 *   2. Calls nn_reinforce_add_sample() for each explored config
 *   3. Calls nn_reinforce_apply() to average gradients, SGD step, cudaMemcpy back
 */

#ifndef NN_REINFORCE_H
#define NN_REINFORCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Copy current GPU weights to host and zero gradient accumulators.
 * @param d_weights  Device pointer to NNWeightsGPU
 * @return 0 on success, -1 on error
 */
int nn_reinforce_init(const void* d_weights);

/**
 * CPU forward + backward pass for one (config, actual_ratio) pair.
 * Accumulates gradients into internal buffers.
 * @param input_raw    15-element raw feature vector (same encoding as GPU kernel)
 * @param actual_ratio Ground-truth compression ratio
 */
void nn_reinforce_add_sample(const float input_raw[15], double actual_ratio);

/**
 * Average accumulated gradients, clip, SGD step, cudaMemcpy weights to GPU.
 * @param d_weights      Device pointer to NNWeightsGPU
 * @param learning_rate  SGD step size
 * @return 0 on success, -1 on error
 */
int nn_reinforce_apply(void* d_weights, float learning_rate);

/**
 * Reset internal state (zero gradients, mark uninitialized).
 */
void nn_reinforce_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* NN_REINFORCE_H */
