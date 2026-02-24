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
 * CPU forward + backward pass for one (config, actual) pair.
 * Accumulates gradients into internal buffers.
 * Trains on output index 2 (ratio), and optionally outputs 0 (comp_time),
 * 1 (decomp_time), and 3 (psnr).
 * @param input_raw          15-element raw feature vector (same encoding as GPU kernel)
 * @param actual_ratio       Ground-truth compression ratio
 * @param actual_comp_time   Ground-truth compression time in ms (0 to skip comp_time loss)
 * @param actual_decomp_time Ground-truth decompression time in ms (0 to skip)
 * @param actual_psnr        Ground-truth PSNR in dB (0 to skip)
 */
void nn_reinforce_add_sample(const float input_raw[15], double actual_ratio,
                             double actual_comp_time,
                             double actual_decomp_time,
                             double actual_psnr);

/**
 * Average accumulated gradients, clip, SGD step, cudaMemcpy weights to GPU.
 * @param d_weights      Device pointer to NNWeightsGPU
 * @param learning_rate  SGD step size
 * @return 0 on success, -1 on error
 */
int nn_reinforce_apply(void* d_weights, float learning_rate);

/**
 * Query stats from the last nn_reinforce_apply() call.
 * @param grad_norm    Output: gradient L2 norm before clipping (NULL to skip)
 * @param num_samples  Output: number of samples in the batch (NULL to skip)
 * @param was_clipped  Output: 1 if gradient was clipped, 0 otherwise (NULL to skip)
 */
void nn_reinforce_get_last_stats(float* grad_norm, int* num_samples,
                                  int* was_clipped);

/**
 * Reset internal state (zero gradients, mark uninitialized).
 */
void nn_reinforce_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* NN_REINFORCE_H */
