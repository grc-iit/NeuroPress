/**
 * @file nn_gpu.cu
 * @brief GPU Neural Network inference for compression algorithm selection
 *
 * Loads trained neural network weights from binary file (.nnwt) and runs
 * inference entirely on GPU. Replaces Q-Table lookup with a learned model
 * that predicts compression metrics for all 32 configurations and ranks them.
 *
 * Architecture: 15 → 128 (ReLU) → 128 (ReLU) → 4
 * Inputs:  [algo_onehot(8), quant, shuffle, error_bound, data_size, entropy, mad, deriv]
 * Outputs: [comp_time_log, decomp_time_log, ratio_log, psnr_clamped]
 *
 * Binary format (.nnwt):
 *   Header: magic(4) version(4) n_layers(4) input_dim(4) hidden_dim(4) output_dim(4)
 *   Normalization: x_means(15×4) x_stds(15×4) y_means(4×4) y_stds(4×4)
 *   Layer 1: W(128×15×4) b(128×4)
 *   Layer 2: W(128×128×4) b(128×4)
 *   Layer 3: W(4×128×4) b(4×4)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

#include "stats/auto_stats_gpu.h"
#include "nn/nn_weights.h"
#include "api/internal.hpp"

/* SGD synchronization globals — defined in gpucompress_api.cpp at file scope */
extern cudaStream_t g_sgd_stream;
extern cudaEvent_t  g_sgd_done;
extern bool         g_sgd_ever_fired;

namespace gpucompress {

/* ============================================================
 * Constants and shared weight layout
 * ============================================================ */

static constexpr uint32_t NN_MAGIC = 0x4E4E5754;  // "NNWT"

/** Ranking criterion for selecting best config */
enum NNRankCriterion {
    NN_RANK_BY_RATIO = 0,       // Highest compression ratio (default)
    NN_RANK_BY_COMP_TIME = 1,   // Lowest compression time
    NN_RANK_BY_DECOMP_TIME = 2, // Lowest decompression time
    NN_RANK_BY_PSNR = 3         // Highest PSNR
};

/* ============================================================
 * Static state
 * ============================================================ */

static NNWeightsGPU* d_nn_weights = nullptr;
static bool g_nn_loaded = false;
static NNRankCriterion g_rank_criterion = NN_RANK_BY_RATIO;

/** Pre-allocated inference output buffers (avoids per-call cudaMalloc) */
static int*   d_infer_action = nullptr;
static float* d_infer_ratio = nullptr;
static float* d_infer_comp_time = nullptr;
static int*   d_infer_top_actions = nullptr;

static void allocInferenceBuffers() {
    if (d_infer_action == nullptr)
        cudaMalloc(&d_infer_action, sizeof(int));
    if (d_infer_ratio == nullptr)
        cudaMalloc(&d_infer_ratio, sizeof(float));
    if (d_infer_comp_time == nullptr)
        cudaMalloc(&d_infer_comp_time, sizeof(float));
    if (d_infer_top_actions == nullptr)
        cudaMalloc(&d_infer_top_actions, NN_NUM_CONFIGS * sizeof(int));
}

static void freeInferenceBuffers() {
    if (d_infer_action) { cudaFree(d_infer_action); d_infer_action = nullptr; }
    if (d_infer_ratio) { cudaFree(d_infer_ratio); d_infer_ratio = nullptr; }
    if (d_infer_comp_time) { cudaFree(d_infer_comp_time); d_infer_comp_time = nullptr; }
    if (d_infer_top_actions) { cudaFree(d_infer_top_actions); d_infer_top_actions = nullptr; }
}

/** Host-side copy of feature bounds for OOD detection */
static float g_x_mins[NN_INPUT_DIM];
static float g_x_maxs[NN_INPUT_DIM];
static bool g_has_bounds = false;

/* ============================================================
 * Neural Network Inference Kernel
 * ============================================================ */

/**
 * GPU kernel: Run forward pass for all 32 configs in parallel.
 *
 * Each of the 32 threads represents one (algorithm, quant, shuffle) config.
 * Thread mapping:
 *   algo_idx  = tid % 8
 *   quant     = (tid / 8) % 2
 *   shuffle   = (tid / 16) % 2
 *
 * The kernel constructs the 15-feature input vector, runs the full
 * forward pass, and writes the predicted metric to shared memory.
 * Then a parallel reduction finds the best action.
 *
 * @param weights       NN weights in GPU global memory
 * @param entropy       Shannon entropy of data (raw, not standardized)
 * @param mad_norm      Normalized MAD (already divided by range)
 * @param deriv_norm    Normalized 2nd derivative (already divided by range)
 * @param data_size     Data size in bytes
 * @param error_bound   Error bound (0 for lossless)
 * @param criterion         Which output to rank by
 * @param out_action        Output: best action ID (0-31)
 * @param out_predicted_ratio Output: predicted compression ratio for winner (nullable)
 * @param out_top_actions   Output: all 32 action IDs sorted by rank (nullable)
 */
__global__ void nnInferenceKernel(
    const NNWeightsGPU* __restrict__ weights,
    double entropy,
    double mad_norm,
    double deriv_norm,
    size_t data_size,
    double error_bound,
    int criterion,
    int* __restrict__ out_action,
    float* __restrict__ out_predicted_ratio,
    float* __restrict__ out_predicted_comp_time,
    int* __restrict__ out_top_actions
) {
    int tid = threadIdx.x;
    if (tid >= NN_NUM_CONFIGS) return;

    // ---- Decode this thread's configuration ----
    int algo_idx = tid % 8;
    int quant    = (tid / 8) % 2;
    int shuffle  = (tid / 16) % 2;

    // ---- Build raw input features ----
    float input_raw[NN_INPUT_DIM];

    // One-hot algorithm encoding (features 0-7)
    for (int i = 0; i < 8; i++) {
        input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
    }

    // Quantization binary (feature 8)
    input_raw[8] = static_cast<float>(quant);

    // Shuffle binary (feature 9)
    input_raw[9] = static_cast<float>(shuffle);

    // Error bound: log10(clip(error_bound, 1e-7)) (feature 10)
    double eb_clipped = error_bound;
    if (eb_clipped < 1e-7) eb_clipped = 1e-7;
    input_raw[10] = static_cast<float>(log10(eb_clipped));

    // Data size: log2(data_size) (feature 11)
    double ds = static_cast<double>(data_size);
    if (ds < 1.0) ds = 1.0;
    input_raw[11] = static_cast<float>(log2(ds));

    // Entropy (feature 12) - raw value, will be standardized
    input_raw[12] = static_cast<float>(entropy);

    // MAD (feature 13) - already normalized by range
    input_raw[13] = static_cast<float>(mad_norm);

    // Second derivative (feature 14) - already normalized by range
    input_raw[14] = static_cast<float>(deriv_norm);

    // ---- Standardize input features ----
    // One-hot and binary features have mean=0, std=1 in the normalization
    // (they were not standardized during training), but the stored means/stds
    // handle this correctly (means=0, stds=1 for those indices)
    float input[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        float std_val = weights->x_stds[i];
        if (std_val < 1e-8f) std_val = 1e-8f;
        input[i] = (input_raw[i] - weights->x_means[i]) / std_val;
    }

    // ---- Layer 1: Linear(15, 128) + ReLU ----
    float hidden1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            sum += weights->w1[j * NN_INPUT_DIM + i] * input[i];
        }
        hidden1[j] = (sum > 0.0f) ? sum : 0.0f;  // ReLU
    }

    // ---- Layer 2: Linear(128, 128) + ReLU ----
    float hidden2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += weights->w2[j * NN_HIDDEN_DIM + i] * hidden1[i];
        }
        hidden2[j] = (sum > 0.0f) ? sum : 0.0f;  // ReLU
    }

    // ---- Layer 3: Linear(128, 4) ----
    float output_norm[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = weights->b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += weights->w3[j * NN_HIDDEN_DIM + i] * hidden2[i];
        }
        output_norm[j] = sum;
    }

    // ---- De-normalize outputs ----
    // output_raw[0] = comp_time_log   → expm1() gives comp_time_ms
    // output_raw[1] = decomp_time_log → expm1() gives decomp_time_ms
    // output_raw[2] = ratio_log       → expm1() gives compression_ratio
    // output_raw[3] = psnr_clamped    → direct (clamped, not logged)
    float output_raw[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];
    }

    // Convert log-space back to original scale
    float comp_time  = expm1f(output_raw[0]);  // ms
    float decomp_time = expm1f(output_raw[1]); // ms
    float ratio      = expm1f(output_raw[2]);  // ratio
    float psnr       = output_raw[3];          // dB

    // ---- Select ranking metric ----
    float rank_val;
    bool higher_is_better;

    switch (criterion) {
        case NN_RANK_BY_RATIO:
            rank_val = ratio;
            higher_is_better = true;
            break;
        case NN_RANK_BY_COMP_TIME:
            rank_val = comp_time;
            higher_is_better = false;
            break;
        case NN_RANK_BY_DECOMP_TIME:
            rank_val = decomp_time;
            higher_is_better = false;
            break;
        case NN_RANK_BY_PSNR:
            rank_val = psnr;
            higher_is_better = true;
            break;
        default:
            rank_val = ratio;
            higher_is_better = true;
    }

    // For "lower is better" metrics, negate so we can always do max-reduction
    if (!higher_is_better) {
        rank_val = -rank_val;
    }

    // Mask out quantization configs in lossless mode — their predicted
    // ratios assume quant is applied, but it won't be (error_bound <= 0).
    if (quant == 1 && error_bound <= 0.0) {
        rank_val = -INFINITY;
    }

    // ---- Store per-thread predicted ratio and comp_time for later retrieval ----
    __shared__ float s_ratios[NN_NUM_CONFIGS];
    __shared__ float s_comp_times[NN_NUM_CONFIGS];
    s_ratios[tid] = ratio;
    s_comp_times[tid] = comp_time;

    // ---- Parallel reduction to find best config ----
    __shared__ float s_vals[NN_NUM_CONFIGS];
    __shared__ int   s_idxs[NN_NUM_CONFIGS];

    s_vals[tid] = rank_val;
    s_idxs[tid] = tid;
    __syncthreads();

    // If top-K actions requested, do a full sort (simple insertion sort by thread 0)
    // Otherwise, just do tree reduction for the winner
    if (out_top_actions != nullptr) {
        // Thread 0 sorts all 32 actions by rank_val (descending)
        if (tid == 0) {
            // Copy to local arrays for sorting
            float local_vals[NN_NUM_CONFIGS];
            int   local_idxs[NN_NUM_CONFIGS];
            for (int i = 0; i < NN_NUM_CONFIGS; i++) {
                local_vals[i] = s_vals[i];
                local_idxs[i] = s_idxs[i];
            }

            // Insertion sort (descending by value) - 32 elements, fast enough
            for (int i = 1; i < NN_NUM_CONFIGS; i++) {
                float key_val = local_vals[i];
                int   key_idx = local_idxs[i];
                int j = i - 1;
                while (j >= 0 && local_vals[j] < key_val) {
                    local_vals[j + 1] = local_vals[j];
                    local_idxs[j + 1] = local_idxs[j];
                    j--;
                }
                local_vals[j + 1] = key_val;
                local_idxs[j + 1] = key_idx;
            }

            // Write winner
            *out_action = local_idxs[0];

            // Write all sorted action IDs
            for (int i = 0; i < NN_NUM_CONFIGS; i++) {
                out_top_actions[i] = local_idxs[i];
            }

            // Write predicted ratio and comp_time for the winner
            if (out_predicted_ratio != nullptr) {
                *out_predicted_ratio = s_ratios[local_idxs[0]];
            }
            if (out_predicted_comp_time != nullptr) {
                *out_predicted_comp_time = s_comp_times[local_idxs[0]];
            }
        }
    } else {
        // Tree reduction (32 → 16 → 8 → 4 → 2 → 1)
        for (int s = NN_NUM_CONFIGS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_vals[tid + s] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + s];
                    s_idxs[tid] = s_idxs[tid + s];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes the result
        if (tid == 0) {
            *out_action = s_idxs[0];

            // Write predicted ratio and comp_time for the winner
            if (out_predicted_ratio != nullptr) {
                *out_predicted_ratio = s_ratios[s_idxs[0]];
            }
            if (out_predicted_comp_time != nullptr) {
                *out_predicted_comp_time = s_comp_times[s_idxs[0]];
            }
        }
    }
}

/* ============================================================
 * Fused Inference Kernel (reads stats from device pointer)
 * ============================================================ */

/**
 * Fused GPU kernel: same as nnInferenceKernel but reads stats from
 * AutoStatsGPU device pointer instead of scalar params.
 * Thread 0 also performs OOD detection on the 5 continuous features.
 */
__global__ void nnFusedInferenceKernel(
    const NNWeightsGPU* __restrict__ weights,
    const AutoStatsGPU* __restrict__ d_stats,
    size_t data_size,
    double error_bound,
    int criterion,
    NNInferenceOutput* __restrict__ out_result,
    int* __restrict__ out_top_actions
) {
    int tid = threadIdx.x;
    if (tid >= NN_NUM_CONFIGS) return;

    // Read stats from device memory
    double entropy = d_stats->entropy;
    double mad_norm = d_stats->mad_normalized;
    double deriv_norm = d_stats->deriv_normalized;

    // Decode this thread's configuration
    int algo_idx = tid % 8;
    int quant    = (tid / 8) % 2;
    int shuffle  = (tid / 16) % 2;

    // Build raw input features
    float input_raw[NN_INPUT_DIM];
    for (int i = 0; i < 8; i++) {
        input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
    }
    input_raw[8] = static_cast<float>(quant);
    input_raw[9] = static_cast<float>(shuffle);

    double eb_clipped = error_bound;
    if (eb_clipped < 1e-7) eb_clipped = 1e-7;
    input_raw[10] = static_cast<float>(log10(eb_clipped));

    double ds = static_cast<double>(data_size);
    if (ds < 1.0) ds = 1.0;
    input_raw[11] = static_cast<float>(log2(ds));

    input_raw[12] = static_cast<float>(entropy);
    input_raw[13] = static_cast<float>(mad_norm);
    input_raw[14] = static_cast<float>(deriv_norm);

    // Standardize
    float input[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        float std_val = weights->x_stds[i];
        if (std_val < 1e-8f) std_val = 1e-8f;
        input[i] = (input_raw[i] - weights->x_means[i]) / std_val;
    }

    // Layer 1: Linear(15, 128) + ReLU
    float hidden1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            sum += weights->w1[j * NN_INPUT_DIM + i] * input[i];
        }
        hidden1[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    // Layer 2: Linear(128, 128) + ReLU
    float hidden2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += weights->w2[j * NN_HIDDEN_DIM + i] * hidden1[i];
        }
        hidden2[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    // Layer 3: Linear(128, 4)
    float output_norm[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        float sum = weights->b3[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            sum += weights->w3[j * NN_HIDDEN_DIM + i] * hidden2[i];
        }
        output_norm[j] = sum;
    }

    // De-normalize outputs
    float output_raw[NN_OUTPUT_DIM];
    for (int j = 0; j < NN_OUTPUT_DIM; j++) {
        output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];
    }

    float comp_time  = expm1f(output_raw[0]);
    float decomp_time = expm1f(output_raw[1]);
    float ratio      = expm1f(output_raw[2]);
    float psnr       = output_raw[3];

    // Select ranking metric
    float rank_val;
    bool higher_is_better;
    switch (criterion) {
        case NN_RANK_BY_RATIO:      rank_val = ratio;       higher_is_better = true;  break;
        case NN_RANK_BY_COMP_TIME:  rank_val = comp_time;   higher_is_better = false; break;
        case NN_RANK_BY_DECOMP_TIME:rank_val = decomp_time; higher_is_better = false; break;
        case NN_RANK_BY_PSNR:       rank_val = psnr;        higher_is_better = true;  break;
        default:                    rank_val = ratio;       higher_is_better = true;
    }
    if (!higher_is_better) rank_val = -rank_val;
    if (quant == 1 && error_bound <= 0.0) rank_val = -INFINITY;

    __shared__ float s_ratios[NN_NUM_CONFIGS];
    __shared__ float s_comp_times[NN_NUM_CONFIGS];
    s_ratios[tid] = ratio;
    s_comp_times[tid] = comp_time;

    __shared__ float s_vals[NN_NUM_CONFIGS];
    __shared__ int   s_idxs[NN_NUM_CONFIGS];
    s_vals[tid] = rank_val;
    s_idxs[tid] = tid;
    __syncthreads();

    if (out_top_actions != nullptr) {
        // Thread 0 sorts all 32 actions
        if (tid == 0) {
            float local_vals[NN_NUM_CONFIGS];
            int   local_idxs[NN_NUM_CONFIGS];
            for (int i = 0; i < NN_NUM_CONFIGS; i++) {
                local_vals[i] = s_vals[i];
                local_idxs[i] = s_idxs[i];
            }
            for (int i = 1; i < NN_NUM_CONFIGS; i++) {
                float key_val = local_vals[i];
                int   key_idx = local_idxs[i];
                int j = i - 1;
                while (j >= 0 && local_vals[j] < key_val) {
                    local_vals[j + 1] = local_vals[j];
                    local_idxs[j + 1] = local_idxs[j];
                    j--;
                }
                local_vals[j + 1] = key_val;
                local_idxs[j + 1] = key_idx;
            }

            int winner = local_idxs[0];
            out_result->action = winner;
            out_result->predicted_ratio = s_ratios[winner];
            out_result->predicted_comp_time = s_comp_times[winner];
            for (int i = 0; i < NN_NUM_CONFIGS; i++) {
                out_top_actions[i] = local_idxs[i];
            }

            // OOD detection on continuous features (indices 10-14)
            int ood = 0;
            float features[5] = {input_raw[10], input_raw[11], input_raw[12],
                                 input_raw[13], input_raw[14]};
            int indices[5] = {10, 11, 12, 13, 14};
            for (int i = 0; i < 5; i++) {
                int idx = indices[i];
                float range = weights->x_maxs[idx] - weights->x_mins[idx];
                float margin = range * 0.10f;
                if (features[i] < weights->x_mins[idx] - margin ||
                    features[i] > weights->x_maxs[idx] + margin) {
                    ood = 1;
                    break;
                }
            }
            out_result->is_ood = ood;
        }
    } else {
        // Tree reduction
        for (int s = NN_NUM_CONFIGS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_vals[tid + s] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + s];
                    s_idxs[tid] = s_idxs[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int winner = s_idxs[0];
            out_result->action = winner;
            out_result->predicted_ratio = s_ratios[winner];
            out_result->predicted_comp_time = s_comp_times[winner];

            // OOD detection
            int ood = 0;
            float features[5] = {input_raw[10], input_raw[11], input_raw[12],
                                 input_raw[13], input_raw[14]};
            int indices[5] = {10, 11, 12, 13, 14};
            for (int i = 0; i < 5; i++) {
                int idx = indices[i];
                float range = weights->x_maxs[idx] - weights->x_mins[idx];
                float margin = range * 0.10f;
                if (features[i] < weights->x_mins[idx] - margin ||
                    features[i] > weights->x_maxs[idx] + margin) {
                    ood = 1;
                    break;
                }
            }
            out_result->is_ood = ood;
        }
    }
}

/* ============================================================
 * Pre-allocated fused inference buffers
 * ============================================================ */

static NNInferenceOutput* d_fused_infer_output = nullptr;
static int* d_fused_top_actions = nullptr;

static void allocFusedInferenceBuffers() {
    if (d_fused_infer_output == nullptr)
        cudaMalloc(&d_fused_infer_output, sizeof(NNInferenceOutput));
    if (d_fused_top_actions == nullptr)
        cudaMalloc(&d_fused_top_actions, NN_NUM_CONFIGS * sizeof(int));
}

static void freeFusedInferenceBuffers() {
    if (d_fused_infer_output) { cudaFree(d_fused_infer_output); d_fused_infer_output = nullptr; }
    if (d_fused_top_actions) { cudaFree(d_fused_top_actions); d_fused_top_actions = nullptr; }
}

/* ============================================================
 * GPU SGD Kernel
 * ============================================================ */

// Total gradient buffer size in floats
static constexpr int SGD_GRAD_SIZE =
    NN_HIDDEN_DIM * NN_INPUT_DIM +   // dw1: 1920
    NN_HIDDEN_DIM +                    // db1: 128
    NN_HIDDEN_DIM * NN_HIDDEN_DIM +   // dw2: 16384
    NN_HIDDEN_DIM +                    // db2: 128
    NN_OUTPUT_DIM * NN_HIDDEN_DIM +   // dw3: 512
    NN_OUTPUT_DIM;                     // db3: 4
// = 19076 floats = ~76KB

// Offsets into gradient buffer
static constexpr int SGD_OFF_DW1 = 0;
static constexpr int SGD_OFF_DB1 = NN_HIDDEN_DIM * NN_INPUT_DIM;                    // 1920
static constexpr int SGD_OFF_DW2 = SGD_OFF_DB1 + NN_HIDDEN_DIM;                     // 2048
static constexpr int SGD_OFF_DB2 = SGD_OFF_DW2 + NN_HIDDEN_DIM * NN_HIDDEN_DIM;     // 18432
static constexpr int SGD_OFF_DW3 = SGD_OFF_DB2 + NN_HIDDEN_DIM;                     // 18560
static constexpr int SGD_OFF_DB3 = SGD_OFF_DW3 + NN_OUTPUT_DIM * NN_HIDDEN_DIM;     // 19072

/**
 * GPU SGD kernel: forward + backward pass + weight update, all on GPU.
 *
 * Launch: <<<1, 128>>>
 * Thread t (0-127) owns hidden neuron t.
 *
 * Shared memory layout (~3.5KB):
 *   s_x[15], s_h1[128], s_z1[128], s_h2[128], s_z2[128],
 *   s_y[4], s_d3[4], s_dz2[128], s_reduce[128]
 */
__global__ void nnSGDKernel(
    NNWeightsGPU* __restrict__ weights,
    const AutoStatsGPU* __restrict__ d_stats,
    const SGDSample* __restrict__ samples,
    int num_samples,
    size_t data_size,
    double error_bound,
    float learning_rate,
    float* __restrict__ d_grad_buffer,
    SGDOutput* __restrict__ out_result
) {
    int t = threadIdx.x;  // 0..127

    // Shared memory
    __shared__ float s_x[NN_INPUT_DIM];
    __shared__ float s_h1[NN_HIDDEN_DIM];
    __shared__ float s_z1[NN_HIDDEN_DIM];
    __shared__ float s_h2[NN_HIDDEN_DIM];
    __shared__ float s_z2[NN_HIDDEN_DIM];
    __shared__ float s_y[NN_OUTPUT_DIM];
    __shared__ float s_d3[NN_OUTPUT_DIM];
    __shared__ float s_dz2[NN_HIDDEN_DIM];
    __shared__ float s_reduce[NN_HIDDEN_DIM];

    // Read stats once
    float stat_entropy = static_cast<float>(d_stats->entropy);
    float stat_mad = static_cast<float>(d_stats->mad_normalized);
    float stat_deriv = static_cast<float>(d_stats->deriv_normalized);

    // Encode data_size and error_bound
    double eb_c = error_bound;
    if (eb_c < 1e-7) eb_c = 1e-7;
    float eb_enc = static_cast<float>(log10(eb_c));
    double ds = static_cast<double>(data_size);
    if (ds < 1.0) ds = 1.0;
    float ds_enc = static_cast<float>(log2(ds));

    // Zero this thread's gradient slice
    // Thread t owns: dw1[t*15..t*15+14], db1[t],
    //                dw2[t*128..t*128+127], db2[t],
    //                dw3[out*128+t] for out=0..3, db3[t] (only t<4)
    for (int i = 0; i < NN_INPUT_DIM; i++)
        d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i] = 0.0f;
    d_grad_buffer[SGD_OFF_DB1 + t] = 0.0f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++)
        d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] = 0.0f;
    d_grad_buffer[SGD_OFF_DB2 + t] = 0.0f;
    for (int out = 0; out < NN_OUTPUT_DIM; out++)
        d_grad_buffer[SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t] = 0.0f;
    if (t < NN_OUTPUT_DIM)
        d_grad_buffer[SGD_OFF_DB3 + t] = 0.0f;
    __syncthreads();

    // Per-sample loop
    for (int si = 0; si < num_samples; si++) {
        SGDSample sample = samples[si];

        // Thread 0 builds standardized input → s_x[15]
        if (t == 0) {
            int algo_idx = sample.action % 8;
            int quant = (sample.action / 8) % 2;
            int shuffle = (sample.action / 16) % 2;

            float raw[NN_INPUT_DIM];
            for (int i = 0; i < 8; i++) raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
            raw[8] = static_cast<float>(quant);
            raw[9] = static_cast<float>(shuffle);
            raw[10] = eb_enc;
            raw[11] = ds_enc;
            raw[12] = stat_entropy;
            raw[13] = stat_mad;
            raw[14] = stat_deriv;

            for (int i = 0; i < NN_INPUT_DIM; i++) {
                float std_val = weights->x_stds[i];
                if (std_val < 1e-8f) std_val = 1e-8f;
                s_x[i] = (raw[i] - weights->x_means[i]) / std_val;
            }
        }
        __syncthreads();

        // Forward L1: thread t computes h1[t]
        {
            float sum = weights->b1[t];
            for (int i = 0; i < NN_INPUT_DIM; i++) {
                sum += weights->w1[t * NN_INPUT_DIM + i] * s_x[i];
            }
            s_z1[t] = sum;
            s_h1[t] = (sum > 0.0f) ? sum : 0.0f;
        }
        __syncthreads();

        // Forward L2: thread t computes h2[t]
        {
            float sum = weights->b2[t];
            for (int i = 0; i < NN_HIDDEN_DIM; i++) {
                sum += weights->w2[t * NN_HIDDEN_DIM + i] * s_h1[i];
            }
            s_z2[t] = sum;
            s_h2[t] = (sum > 0.0f) ? sum : 0.0f;
        }
        __syncthreads();

        // Forward L3: threads 0-3 compute y[out]
        if (t < NN_OUTPUT_DIM) {
            float sum = weights->b3[t];
            for (int i = 0; i < NN_HIDDEN_DIM; i++) {
                sum += weights->w3[t * NN_HIDDEN_DIM + i] * s_h2[i];
            }
            s_y[t] = sum;
        }
        __syncthreads();

        // Thread 0 computes targets + output errors → s_d3[4]
        if (t == 0) {
            // Targets in normalized space
            float y_std2 = weights->y_stds[2];
            if (y_std2 < 1e-8f) y_std2 = 1e-8f;
            float target_ratio = (log1pf(sample.actual_ratio) - weights->y_means[2]) / y_std2;

            s_d3[2] = s_y[2] - target_ratio;

            // Comp time (output 0)
            if (sample.actual_comp_time > 0.0f) {
                float y_std0 = weights->y_stds[0];
                if (y_std0 < 1e-8f) y_std0 = 1e-8f;
                s_d3[0] = s_y[0] - (log1pf(sample.actual_comp_time) - weights->y_means[0]) / y_std0;
            } else {
                s_d3[0] = 0.0f;
            }

            // Decomp time (output 1)
            if (sample.actual_decomp_time > 0.0f) {
                float y_std1 = weights->y_stds[1];
                if (y_std1 < 1e-8f) y_std1 = 1e-8f;
                s_d3[1] = s_y[1] - (log1pf(sample.actual_decomp_time) - weights->y_means[1]) / y_std1;
            } else {
                s_d3[1] = 0.0f;
            }

            // PSNR (output 3)
            if (sample.actual_psnr > 0.0f) {
                float clamped = fminf(sample.actual_psnr, 120.0f);
                float y_std3 = weights->y_stds[3];
                if (y_std3 < 1e-8f) y_std3 = 1e-8f;
                s_d3[3] = s_y[3] - (clamped - weights->y_means[3]) / y_std3;
            } else {
                s_d3[3] = 0.0f;
            }
        }
        __syncthreads();

        // Layer 3 gradients: thread t handles dw3[out][t] for out=0..3
        for (int out = 0; out < NN_OUTPUT_DIM; out++) {
            if (s_d3[out] != 0.0f) {
                d_grad_buffer[SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t] += s_d3[out] * s_h2[t];
            }
        }
        if (t < NN_OUTPUT_DIM) {
            d_grad_buffer[SGD_OFF_DB3 + t] += s_d3[t];
        }

        // Backward L3→L2: dh2[t] = sum(w3[out][t]*d3[out])
        {
            float dh2_t = 0.0f;
            for (int out = 0; out < NN_OUTPUT_DIM; out++) {
                dh2_t += weights->w3[out * NN_HIDDEN_DIM + t] * s_d3[out];
            }
            // ReLU backward
            s_dz2[t] = (s_z2[t] > 0.0f) ? dh2_t : 0.0f;
        }
        __syncthreads();

        // Layer 2 gradients
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] += s_dz2[t] * s_h1[i];
        }
        d_grad_buffer[SGD_OFF_DB2 + t] += s_dz2[t];

        // Backward L2→L1: dh1[t] = sum(w2[j][t]*dz2[j])
        float dz1_t;
        {
            float dh1_t = 0.0f;
            for (int j = 0; j < NN_HIDDEN_DIM; j++) {
                dh1_t += weights->w2[j * NN_HIDDEN_DIM + t] * s_dz2[j];
            }
            dz1_t = (s_z1[t] > 0.0f) ? dh1_t : 0.0f;
        }

        // Layer 1 gradients
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i] += dz1_t * s_x[i];
        }
        d_grad_buffer[SGD_OFF_DB1 + t] += dz1_t;

        __syncthreads();
    }

    // --- After all samples: average, clip, SGD step ---
    float inv_n = 1.0f / static_cast<float>(num_samples);

    // Average and compute local norm contribution
    float local_norm_sq = 0.0f;

    // dw1 owned by thread t
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        int idx = SGD_OFF_DW1 + t * NN_INPUT_DIM + i;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }
    // db1
    {
        int idx = SGD_OFF_DB1 + t;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }
    // dw2
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        int idx = SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }
    // db2
    {
        int idx = SGD_OFF_DB2 + t;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }
    // dw3 (thread t owns [out*128+t] for out=0..3)
    for (int out = 0; out < NN_OUTPUT_DIM; out++) {
        int idx = SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }
    // db3 (threads 0-3)
    if (t < NN_OUTPUT_DIM) {
        int idx = SGD_OFF_DB3 + t;
        d_grad_buffer[idx] *= inv_n;
        local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
    }

    // Shared memory reduction for total norm
    s_reduce[t] = local_norm_sq;
    __syncthreads();
    for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
        if (t < s) s_reduce[t] += s_reduce[t + s];
        __syncthreads();
    }

    float norm = sqrtf(s_reduce[0]);
    float clip_scale = (norm > 1.0f) ? (1.0f / norm) : 1.0f;
    float lr_scaled = learning_rate * clip_scale;

    // SGD step: update weights in-place
    // w1
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        weights->w1[t * NN_INPUT_DIM + i] -=
            lr_scaled * d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i];
    }
    weights->b1[t] -= lr_scaled * d_grad_buffer[SGD_OFF_DB1 + t];

    // w2
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        weights->w2[t * NN_HIDDEN_DIM + i] -=
            lr_scaled * d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i];
    }
    weights->b2[t] -= lr_scaled * d_grad_buffer[SGD_OFF_DB2 + t];

    // w3 (4 outputs, thread t updates its column)
    for (int out = 0; out < NN_OUTPUT_DIM; out++) {
        weights->w3[out * NN_HIDDEN_DIM + t] -=
            lr_scaled * d_grad_buffer[SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t];
    }
    if (t < NN_OUTPUT_DIM) {
        weights->b3[t] -= lr_scaled * d_grad_buffer[SGD_OFF_DB3 + t];
    }

    // Thread 0 writes output
    if (t == 0) {
        out_result->grad_norm = norm;
        out_result->was_clipped = (norm > 1.0f) ? 1 : 0;
        out_result->sample_count = num_samples;
    }
}

/* ============================================================
 * Pre-allocated SGD buffers
 * ============================================================ */

static float* d_sgd_grad_buffer = nullptr;
static SGDOutput* d_sgd_output = nullptr;
static SGDSample* d_sgd_samples = nullptr;

static void allocSGDBuffers() {
    if (d_sgd_grad_buffer == nullptr)
        cudaMalloc(&d_sgd_grad_buffer, SGD_GRAD_SIZE * sizeof(float));
    if (d_sgd_output == nullptr)
        cudaMalloc(&d_sgd_output, sizeof(SGDOutput));
    if (d_sgd_samples == nullptr)
        cudaMalloc(&d_sgd_samples, NN_MAX_SGD_SAMPLES * sizeof(SGDSample));
}

static void freeSGDBuffers() {
    if (d_sgd_grad_buffer) { cudaFree(d_sgd_grad_buffer); d_sgd_grad_buffer = nullptr; }
    if (d_sgd_output) { cudaFree(d_sgd_output); d_sgd_output = nullptr; }
    if (d_sgd_samples) { cudaFree(d_sgd_samples); d_sgd_samples = nullptr; }
}

/* ============================================================
 * Host Functions
 * ============================================================ */

/**
 * Load neural network weights from binary file (.nnwt).
 *
 * @param filepath Path to .nnwt file
 * @return true on success
 */
bool loadNNFromBinary(const char* filepath) {
    if (filepath == nullptr) return false;

    // Immediately mark as not loaded — prevents using stale/corrupted weights
    // if any step below fails during a reload attempt
    g_nn_loaded = false;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "NN: Failed to open weights file: %s\n", filepath);
        return false;
    }

    // Read header
    uint32_t magic, version, n_layers, input_dim, hidden_dim, output_dim;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&n_layers), 4);
    file.read(reinterpret_cast<char*>(&input_dim), 4);
    file.read(reinterpret_cast<char*>(&hidden_dim), 4);
    file.read(reinterpret_cast<char*>(&output_dim), 4);

    if (magic != NN_MAGIC) {
        fprintf(stderr, "NN: Invalid magic number: 0x%08X (expected 0x%08X)\n", magic, NN_MAGIC);
        return false;
    }

    if (version != 1 && version != 2) {
        fprintf(stderr, "NN: Unsupported version: %u\n", version);
        return false;
    }

    if (n_layers != 3 || input_dim != NN_INPUT_DIM ||
        hidden_dim != NN_HIDDEN_DIM || output_dim != NN_OUTPUT_DIM) {
        fprintf(stderr, "NN: Architecture mismatch: %u layers, %u→%u→%u (expected 3, %d→%d→%d)\n",
                n_layers, input_dim, hidden_dim, output_dim,
                NN_INPUT_DIM, NN_HIDDEN_DIM, NN_OUTPUT_DIM);
        return false;
    }

    // Read all weights into host struct
    NNWeightsGPU h_weights;

    // Normalization parameters
    file.read(reinterpret_cast<char*>(h_weights.x_means), NN_INPUT_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.x_stds), NN_INPUT_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.y_means), NN_OUTPUT_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.y_stds), NN_OUTPUT_DIM * sizeof(float));

    // Layer 1
    file.read(reinterpret_cast<char*>(h_weights.w1), NN_HIDDEN_DIM * NN_INPUT_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.b1), NN_HIDDEN_DIM * sizeof(float));

    // Layer 2
    file.read(reinterpret_cast<char*>(h_weights.w2), NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.b2), NN_HIDDEN_DIM * sizeof(float));

    // Layer 3
    file.read(reinterpret_cast<char*>(h_weights.w3), NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float));
    file.read(reinterpret_cast<char*>(h_weights.b3), NN_OUTPUT_DIM * sizeof(float));

    if (!file) {
        fprintf(stderr, "NN: Failed to read weights data\n");
        return false;
    }

    // Feature bounds (v2+)
    if (version >= 2) {
        file.read(reinterpret_cast<char*>(h_weights.x_mins), NN_INPUT_DIM * sizeof(float));
        file.read(reinterpret_cast<char*>(h_weights.x_maxs), NN_INPUT_DIM * sizeof(float));
        if (!file) {
            fprintf(stderr, "NN: Failed to read feature bounds\n");
            return false;
        }
        memcpy(g_x_mins, h_weights.x_mins, sizeof(g_x_mins));
        memcpy(g_x_maxs, h_weights.x_maxs, sizeof(g_x_maxs));
        g_has_bounds = true;
    } else {
        // v1: no bounds, set defaults (no OOD detection)
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            h_weights.x_mins[i] = -1e30f;
            h_weights.x_maxs[i] = 1e30f;
            g_x_mins[i] = -1e30f;
            g_x_maxs[i] = 1e30f;
        }
        g_has_bounds = false;
    }

    // Allocate GPU memory and copy
    if (d_nn_weights == nullptr) {
        cudaError_t err = cudaMalloc(&d_nn_weights, sizeof(NNWeightsGPU));
        if (err != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    fprintf(stderr, "[XFER H→D] NN weights load (%zu B)\n", sizeof(NNWeightsGPU));
    cudaError_t err = cudaMemcpy(d_nn_weights, &h_weights, sizeof(NNWeightsGPU),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "NN: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    g_nn_loaded = true;

    // Pre-allocate inference output buffers
    allocInferenceBuffers();
    allocFusedInferenceBuffers();
    allocSGDBuffers();

    size_t total_params = NN_HIDDEN_DIM * NN_INPUT_DIM + NN_HIDDEN_DIM +
                          NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +
                          NN_OUTPUT_DIM * NN_HIDDEN_DIM + NN_OUTPUT_DIM;
    printf("NN: Loaded %zu parameters from %s (%.1f KB on GPU)\n",
           total_params, filepath, sizeof(NNWeightsGPU) / 1024.0f);

    return true;
}

/**
 * Check if input features are out-of-distribution.
 *
 * Checks the 5 continuous features (indices 10-14: error_bound_enc,
 * data_size_enc, entropy, mad, deriv) against training data bounds
 * with a 10% margin.
 *
 * @return true if any feature is outside bounds (OOD)
 */
bool isInputOOD(double entropy, double mad, double deriv,
                size_t data_size, double error_bound) {
    if (!g_has_bounds) return false;

    // Build raw feature values for continuous features (indices 10-14)
    // Same encoding as the kernel
    double eb_clipped = error_bound;
    if (eb_clipped < 1e-7) eb_clipped = 1e-7;
    float error_bound_enc = static_cast<float>(log10(eb_clipped));

    double ds = static_cast<double>(data_size);
    if (ds < 1.0) ds = 1.0;
    float data_size_enc = static_cast<float>(log2(ds));

    float features[5] = {
        error_bound_enc,                    // index 10
        data_size_enc,                      // index 11
        static_cast<float>(entropy),        // index 12
        static_cast<float>(mad),            // index 13
        static_cast<float>(deriv)           // index 14
    };
    int indices[5] = {10, 11, 12, 13, 14};

    for (int i = 0; i < 5; i++) {
        int idx = indices[i];
        float range = g_x_maxs[idx] - g_x_mins[idx];
        float margin = range * 0.10f;  // 10% margin
        if (features[i] < g_x_mins[idx] - margin ||
            features[i] > g_x_maxs[idx] + margin) {
            return true;
        }
    }
    return false;
}

/**
 * Free neural network GPU memory.
 */
void cleanupNN() {
    if (d_nn_weights != nullptr) {
        cudaFree(d_nn_weights);
        d_nn_weights = nullptr;
    }
    freeInferenceBuffers();
    freeFusedInferenceBuffers();
    freeSGDBuffers();
    g_nn_loaded = false;
    g_has_bounds = false;
}

/**
 * Check if neural network is loaded.
 */
bool isNNLoaded() {
    return g_nn_loaded;
}

/**
 * Get device pointer to NN weights.
 */
const NNWeightsGPU* getNNWeightsDevicePtr() {
    return d_nn_weights;
}

/**
 * Set the ranking criterion for neural network inference.
 */
void setNNRankCriterion(NNRankCriterion criterion) {
    g_rank_criterion = criterion;
}

/**
 * Get current ranking criterion.
 */
NNRankCriterion getNNRankCriterion() {
    return g_rank_criterion;
}

/**
 * Run neural network inference to find best compression config.
 *
 * Launches 32 threads (one per config), each runs a full forward pass
 * and participates in a parallel reduction to find the best action.
 *
 * @param entropy             Shannon entropy (0-8)
 * @param mad_norm            Normalized MAD (0-1)
 * @param deriv_norm          Normalized 2nd derivative (0-1)
 * @param data_size           Data size in bytes
 * @param error_bound         Error bound (0 for lossless)
 * @param stream              CUDA stream
 * @param out_predicted_ratio [out] Nullable, predicted compression ratio for winner
 * @param out_top_actions     [out] Nullable, all 32 action IDs sorted by rank
 * @return Best action ID (0-31), or -1 on error
 */
int runNNInference(
    double entropy,
    double mad_norm,
    double deriv_norm,
    size_t data_size,
    double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio,
    float* out_predicted_comp_time,
    int* out_top_actions
) {
    if (!g_nn_loaded || d_nn_weights == nullptr) {
        return -1;  // Error: NN not loaded
    }

    // Ensure pre-allocated buffers exist
    if (d_infer_action == nullptr) {
        allocInferenceBuffers();
        if (d_infer_action == nullptr) return -1;
    }

    int h_action = 0;

    nnInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
        d_nn_weights,
        entropy, mad_norm, deriv_norm,
        data_size, error_bound,
        static_cast<int>(g_rank_criterion),
        d_infer_action,
        out_predicted_ratio ? d_infer_ratio : nullptr,
        out_predicted_comp_time ? d_infer_comp_time : nullptr,
        out_top_actions ? d_infer_top_actions : nullptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    fprintf(stderr, "[XFER D→H] NN inference (old): action (%zu B)\n", sizeof(int));
    err = cudaMemcpyAsync(&h_action, d_infer_action, sizeof(int),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    if (out_predicted_ratio) {
        fprintf(stderr, "[XFER D→H] NN inference (old): predicted_ratio (%zu B)\n", sizeof(float));
        err = cudaMemcpyAsync(out_predicted_ratio, d_infer_ratio, sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    if (out_predicted_comp_time) {
        fprintf(stderr, "[XFER D→H] NN inference (old): predicted_comp_time (%zu B)\n", sizeof(float));
        err = cudaMemcpyAsync(out_predicted_comp_time, d_infer_comp_time, sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    if (out_top_actions) {
        fprintf(stderr, "[XFER D→H] NN inference (old): top_actions (%zu B)\n", NN_NUM_CONFIGS * sizeof(int));
        err = cudaMemcpyAsync(out_top_actions, d_infer_top_actions,
                               NN_NUM_CONFIGS * sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    return h_action;
}

/**
 * Fused NN inference: reads stats directly from device pointer.
 * Single D→H copy of NNInferenceOutput (16B), optional top_actions.
 */
int runNNFusedInference(
    const AutoStatsGPU* d_stats,
    size_t data_size,
    double error_bound,
    cudaStream_t stream,
    int* out_action,
    float* out_ratio,
    float* out_comp_time,
    int* out_is_ood,
    int* out_top_actions
) {
    if (!g_nn_loaded || d_nn_weights == nullptr || d_stats == nullptr) {
        return -1;
    }

    // Ensure fused buffers exist
    if (d_fused_infer_output == nullptr) {
        allocFusedInferenceBuffers();
        if (d_fused_infer_output == nullptr) return -1;
    }

    nnFusedInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
        d_nn_weights,
        d_stats,
        data_size, error_bound,
        static_cast<int>(g_rank_criterion),
        d_fused_infer_output,
        out_top_actions ? d_fused_top_actions : nullptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    // Single D→H of NNInferenceOutput (16B)
    NNInferenceOutput h_result;
    fprintf(stderr, "[XFER D→H] NN fused inference: NNInferenceOutput (%zu B)\n", sizeof(NNInferenceOutput));
    err = cudaMemcpyAsync(&h_result, d_fused_infer_output, sizeof(NNInferenceOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    if (out_top_actions) {
        fprintf(stderr, "[XFER D→H] NN fused inference: top_actions (%zu B)\n", NN_NUM_CONFIGS * sizeof(int));
        err = cudaMemcpyAsync(out_top_actions, d_fused_top_actions,
                               NN_NUM_CONFIGS * sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_action) *out_action = h_result.action;
    if (out_ratio) *out_ratio = h_result.predicted_ratio;
    if (out_comp_time) *out_comp_time = h_result.predicted_comp_time;
    if (out_is_ood) *out_is_ood = h_result.is_ood;

    return h_result.action;
}

/**
 * GPU-native SGD: forward/backward + weight update entirely on GPU.
 * Copies only samples H→D (60B) and SGDOutput D→H (12B).
 */
int runNNSGD(
    const AutoStatsGPU* d_stats,
    const SGDSample* samples,
    int num_samples,
    size_t data_size,
    double error_bound,
    float learning_rate,
    cudaStream_t stream,
    float* out_grad_norm,
    int* out_clipped,
    int* out_count
) {
    if (!g_nn_loaded || d_nn_weights == nullptr || d_stats == nullptr) return -1;
    if (samples == nullptr || num_samples <= 0) return -1;
    if (num_samples > NN_MAX_SGD_SAMPLES) num_samples = NN_MAX_SGD_SAMPLES;

    // Ensure SGD buffers exist
    if (d_sgd_grad_buffer == nullptr) {
        allocSGDBuffers();
        if (d_sgd_grad_buffer == nullptr) return -1;
    }

    // H→D: copy samples (tiny: num_samples * 20B)
    fprintf(stderr, "[XFER H→D] SGD samples (%d × %zu B = %zu B)\n",
            num_samples, sizeof(SGDSample), (size_t)num_samples * sizeof(SGDSample));
    cudaError_t err = cudaMemcpyAsync(d_sgd_samples, samples,
                                       num_samples * sizeof(SGDSample),
                                       cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return -1;

    // Launch SGD kernel
    nnSGDKernel<<<1, NN_HIDDEN_DIM, 0, stream>>>(
        d_nn_weights,
        d_stats,
        d_sgd_samples,
        num_samples,
        data_size, error_bound,
        learning_rate,
        d_sgd_grad_buffer,
        d_sgd_output
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    // D→H: copy SGDOutput (12B)
    SGDOutput h_result;
    fprintf(stderr, "[XFER D→H] SGD output: grad_norm + clipped + count (%zu B)\n", sizeof(SGDOutput));
    err = cudaMemcpyAsync(&h_result, d_sgd_output, sizeof(SGDOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_grad_norm) *out_grad_norm = h_result.grad_norm;
    if (out_clipped) *out_clipped = h_result.was_clipped;
    if (out_count) *out_count = h_result.sample_count;

    fprintf(stderr, "[SGD] GPU: %d samples, grad_norm=%.4f%s\n",
            h_result.sample_count, h_result.grad_norm,
            h_result.was_clipped ? " (clipped)" : "");

    return 0;
}

/**
 * Context-aware fused NN inference.
 * Uses ctx->d_fused_infer_output / d_fused_top_actions instead of globals.
 * Inserts a GPU-level dependency on g_sgd_done so inference waits for
 * any in-flight SGD on g_sgd_stream.
 */
int runNNFusedInferenceCtx(
    const AutoStatsGPU* d_stats,
    size_t data_size,
    double error_bound,
    cudaStream_t stream,
    CompContext* ctx,
    int* out_action,
    float* out_ratio,
    float* out_comp_time,
    int* out_is_ood,
    int* out_top_actions
) {
    if (!g_nn_loaded || d_nn_weights == nullptr || d_stats == nullptr || ctx == nullptr) {
        return -1;
    }

    /* GPU-level barrier: wait for last SGD on g_sgd_stream before reading weights */
    if (g_sgd_ever_fired)
        cudaStreamWaitEvent(stream, g_sgd_done, 0);

    nnFusedInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
        d_nn_weights,
        d_stats,
        data_size, error_bound,
        static_cast<int>(g_rank_criterion),
        ctx->d_fused_infer_output,
        out_top_actions ? ctx->d_fused_top_actions : nullptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    NNInferenceOutput h_result;
    err = cudaMemcpyAsync(&h_result, ctx->d_fused_infer_output, sizeof(NNInferenceOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    if (out_top_actions) {
        err = cudaMemcpyAsync(out_top_actions, ctx->d_fused_top_actions,
                               NN_NUM_CONFIGS * sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_action)   *out_action   = h_result.action;
    if (out_ratio)    *out_ratio    = h_result.predicted_ratio;
    if (out_comp_time)*out_comp_time = h_result.predicted_comp_time;
    if (out_is_ood)   *out_is_ood   = h_result.is_ood;

    return h_result.action;
}

/**
 * Context-aware GPU SGD.
 * Launches nnSGDKernel on g_sgd_stream (not ctx->stream) so all SGD
 * updates are serialized on one stream.  Uses ctx->d_sgd_* for workspace.
 * Records g_sgd_done after kernel; sets g_sgd_ever_fired = true.
 * Callers must hold g_sgd_mutex before calling this function.
 */
int runNNSGDCtx(
    const AutoStatsGPU* d_stats,
    const SGDSample* samples,
    int num_samples,
    size_t data_size,
    double error_bound,
    float learning_rate,
    CompContext* ctx,
    float* out_grad_norm,
    int* out_clipped,
    int* out_count
) {
    if (!g_nn_loaded || d_nn_weights == nullptr || d_stats == nullptr || ctx == nullptr) return -1;
    if (samples == nullptr || num_samples <= 0) return -1;
    if (num_samples > NN_MAX_SGD_SAMPLES) num_samples = NN_MAX_SGD_SAMPLES;

    /* H→D: copy samples on g_sgd_stream */
    cudaError_t err = cudaMemcpyAsync(ctx->d_sgd_samples, samples,
                                       num_samples * sizeof(SGDSample),
                                       cudaMemcpyHostToDevice, g_sgd_stream);
    if (err != cudaSuccess) return -1;

    /* Launch SGD kernel on dedicated g_sgd_stream */
    nnSGDKernel<<<1, NN_HIDDEN_DIM, 0, g_sgd_stream>>>(
        d_nn_weights,
        d_stats,
        ctx->d_sgd_samples,
        num_samples,
        data_size, error_bound,
        learning_rate,
        ctx->d_sgd_grad_buffer,
        ctx->d_sgd_output
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    /* Record completion event so future inference calls can wait */
    cudaEventRecord(g_sgd_done, g_sgd_stream);

    /* D→H: copy SGDOutput (12B) */
    SGDOutput h_result;
    err = cudaMemcpyAsync(&h_result, ctx->d_sgd_output, sizeof(SGDOutput),
                           cudaMemcpyDeviceToHost, g_sgd_stream);
    if (err != cudaSuccess) return -1;

    err = cudaStreamSynchronize(g_sgd_stream);
    if (err != cudaSuccess) return -1;

    g_sgd_ever_fired = true;

    if (out_grad_norm) *out_grad_norm = h_result.grad_norm;
    if (out_clipped)   *out_clipped   = h_result.was_clipped;
    if (out_count)     *out_count     = h_result.sample_count;

    fprintf(stderr, "[SGD-CTX] slot=%d %d samples, grad_norm=%.4f%s\n",
            ctx->slot_id, h_result.sample_count, h_result.grad_norm,
            h_result.was_clipped ? " (clipped)" : "");

    return 0;
}

} // namespace gpucompress

/* ============================================================
 * C API Wrappers
 * ============================================================ */

extern "C" {

int gpucompress_nn_load_impl(const char* filepath) {
    return gpucompress::loadNNFromBinary(filepath) ? 0 : -1;
}

int gpucompress_nn_is_loaded_impl(void) {
    return gpucompress::isNNLoaded() ? 1 : 0;
}

void gpucompress_nn_cleanup_impl(void) {
    gpucompress::cleanupNN();
}

void gpucompress_nn_set_criterion_impl(int criterion) {
    gpucompress::setNNRankCriterion(
        static_cast<gpucompress::NNRankCriterion>(criterion));
}

void* gpucompress_nn_get_device_ptr_impl(void) {
    return static_cast<void*>(gpucompress::d_nn_weights);
}

} // extern "C"
