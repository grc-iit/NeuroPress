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

#include <atomic>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <mutex>
#include <vector>

#include "stats/auto_stats_gpu.h"
#include "nn/nn_weights.h"
#include "api/internal.hpp"

/* SGD synchronization globals — defined in gpucompress_api.cpp at file scope */
extern cudaStream_t g_sgd_stream;
extern cudaEvent_t  g_sgd_done;
extern std::atomic<bool> g_sgd_ever_fired;

/* Cost-based ranking globals — defined in gpucompress_api.cpp */
extern float g_rank_w0;
extern float g_rank_w1;
extern float g_rank_w2;
extern float g_measured_bw_bytes_per_ms;

/* Debug flag — defined in gpucompress_api.cpp, set via GPUCOMPRESS_DEBUG_NN=1 */
extern bool g_debug_nn;

namespace gpucompress {

/* ============================================================
 * Constants and shared weight layout
 * ============================================================ */

static constexpr uint32_t NN_MAGIC = 0x4E4E5754;  // "NNWT"

/* ============================================================
 * Static state
 * ============================================================ */

static NNWeightsGPU* d_nn_weights = nullptr;
static std::atomic<bool> g_nn_loaded{false};

/** Protects d_nn_weights pointer: held during kernel launch (read) and swap (write).
 *  Prevents use-after-free when gpucompress_reload_nn swaps the pointer
 *  while concurrent inference is between reading the pointer and launching. */
static std::mutex g_nn_ptr_mutex;

/** Pre-allocated inference output buffers (avoids per-call cudaMalloc) */
static NNInferenceOutput* d_infer_output = nullptr;
static int*               d_infer_top_actions = nullptr;

static bool allocInferenceBuffers() {
    if (d_infer_output == nullptr) {
        if (cudaMalloc(&d_infer_output, sizeof(NNInferenceOutput)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for inference output\n");
            return false;
        }
    }
    if (d_infer_top_actions == nullptr) {
        if (cudaMalloc(&d_infer_top_actions, NN_NUM_CONFIGS * sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for top actions\n");
            return false;
        }
    }
    return true;
}

static void freeInferenceBuffers() {
    if (d_infer_output) { cudaFree(d_infer_output); d_infer_output = nullptr; }
    if (d_infer_top_actions) { cudaFree(d_infer_top_actions); d_infer_top_actions = nullptr; }
}


/* ============================================================
 * Shared forward-pass device function
 * ============================================================ */

/**
 * 15→128→128→4 forward pass for one (algo, quant, shuffle) config.
 * Shared by nnInferenceKernel and nnFusedInferenceKernel to eliminate ~125
 * lines of duplication. Stats (entropy/mad/deriv) are passed as scalars so
 * the caller decides whether to read them from registers or device memory.
 *
 * Outputs rank_val (negated for lower-is-better; -INF for masked lossless),
 * ratio, and comp_time for the shared-memory reduction that follows.
 */
__device__ static void nnForwardPass(
    const NNWeightsGPU* __restrict__ weights,
    int   tid,
    float eb_enc,       // log10(clip(error_bound, 1e-7)) — precomputed by caller
    float ds_enc,       // log2(max(data_size, 1))        — precomputed by caller
    float entropy_f,    // (float)entropy                  — precomputed by caller
    float mad_f,        // (float)mad_norm                 — precomputed by caller
    float deriv_f,      // (float)deriv_norm               — precomputed by caller
    double error_bound, // kept for quant-mask check only
    float data_size_bytes, float w0, float w1, float w2, float bw,
    float* out_rank_val,
    float* out_ratio,
    float* out_comp_time,
    float* out_decomp_time,
    float* out_psnr
) {
    int algo_idx = tid % 8;
    int quant    = (tid / 8) % 2;
    int shuffle  = (tid / 16) % 2;

    float input_raw[NN_INPUT_DIM];
    for (int i = 0; i < 8; i++) input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
    input_raw[8]  = static_cast<float>(quant);
    input_raw[9]  = static_cast<float>(shuffle);
    input_raw[10] = eb_enc;
    input_raw[11] = ds_enc;
    input_raw[12] = entropy_f;
    input_raw[13] = mad_f;
    input_raw[14] = deriv_f;

    float input[NN_INPUT_DIM];
    for (int i = 0; i < NN_INPUT_DIM; i++) {
        float std_val = weights->x_stds[i];
        if (std_val < 1e-8f) std_val = 1e-8f;
        input[i] = (input_raw[i] - weights->x_means[i]) / std_val;
    }

    float hidden1[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++)
            sum += weights->w1[j * NN_INPUT_DIM + i] * input[i];
        hidden1[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float hidden2[NN_HIDDEN_DIM];
    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float sum = weights->b2[j];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            sum += weights->w2[j * NN_HIDDEN_DIM + i] * hidden1[i];
        hidden2[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    float comp_time   = weights->b3[0];
    float decomp_time = weights->b3[1];
    float ratio       = weights->b3[2];
    float psnr        = weights->b3[3];
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        comp_time   += weights->w3[0 * NN_HIDDEN_DIM + i] * hidden2[i];
        decomp_time += weights->w3[1 * NN_HIDDEN_DIM + i] * hidden2[i];
        ratio       += weights->w3[2 * NN_HIDDEN_DIM + i] * hidden2[i];
        psnr        += weights->w3[3 * NN_HIDDEN_DIM + i] * hidden2[i];
    }
    comp_time   = expm1f(comp_time   * weights->y_stds[0] + weights->y_means[0]);
    decomp_time = expm1f(decomp_time * weights->y_stds[1] + weights->y_means[1]);
    ratio       = expm1f(ratio       * weights->y_stds[2] + weights->y_means[2]);
    psnr        =        psnr        * weights->y_stds[3] + weights->y_means[3];

    /* Clamp expm1f outputs to sane ranges — prevents SGD weight drift
       from producing INF/NaN that corrupt action selection and MAPE. */
    comp_time   = fmaxf(1e-6f, fminf(comp_time,   1e6f));
    decomp_time = fmaxf(1e-6f, fminf(decomp_time, 1e6f));
    ratio       = fmaxf(0.1f,  fminf(ratio,        1e5f));

    /* Policy clamps: ct/dt floor at 5ms, ratio cap at 100x. */
    comp_time   = fmaxf(5.0f, comp_time);
    decomp_time = fmaxf(5.0f, decomp_time);
    ratio       = fminf(100.0f, ratio);

    /* Cost = w0*ct + w1*dt + w2*ds/(ratio*bw).
       Lower cost is better → rank_val = -cost (reduction finds max). */
    float io_cost = data_size_bytes / (ratio * bw);
    float cost = w0 * comp_time + w1 * decomp_time + w2 * io_cost;
    float rank_val = -cost;
    if (quant == 1 && error_bound <= 0.0) rank_val = -INFINITY;

    *out_rank_val   = rank_val;
    *out_ratio      = ratio;
    *out_comp_time  = comp_time;
    *out_decomp_time = decomp_time;
    *out_psnr       = psnr;
}

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
 * @param w0/w1/w2/bw       Cost-based ranking weights and bandwidth
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
    float w0, float w1, float w2, float bw,
    NNInferenceOutput* __restrict__ out_result,
    int* __restrict__ out_top_actions
) {
    int tid = threadIdx.x;
    if (tid >= NN_NUM_CONFIGS) return;

    // ---- Precompute scalars shared across all 32 threads (thread 0 only) ----
    __shared__ float s_enc[5];  // [eb_enc, ds_enc, entropy_f, mad_f, deriv_f]
    if (tid == 0) {
        double eb_c = (error_bound < 1e-7) ? 1e-7 : error_bound;
        s_enc[0] = static_cast<float>(log10(eb_c));
        double ds = static_cast<double>(data_size);
        if (ds < 1.0) ds = 1.0;
        s_enc[1] = static_cast<float>(log2(ds));
        s_enc[2] = static_cast<float>(entropy);
        s_enc[3] = static_cast<float>(mad_norm);
        s_enc[4] = static_cast<float>(deriv_norm);
    }
    __syncthreads();

    // ---- Forward pass via shared device function ----
    float rank_val, ratio, comp_time, decomp_time, psnr;
    nnForwardPass(weights, tid,
                  s_enc[0], s_enc[1], s_enc[2], s_enc[3], s_enc[4],
                  error_bound,
                  static_cast<float>(data_size), w0, w1, w2, bw,
                  &rank_val, &ratio, &comp_time, &decomp_time, &psnr);

    // ---- Store per-thread predictions for later retrieval ----
    __shared__ float s_ratios[NN_NUM_CONFIGS];
    __shared__ float s_comp_times[NN_NUM_CONFIGS];
    __shared__ float s_decomp_times[NN_NUM_CONFIGS];
    __shared__ float s_psnrs[NN_NUM_CONFIGS];
    s_ratios[tid]      = ratio;
    s_comp_times[tid]  = comp_time;
    s_decomp_times[tid] = decomp_time;
    s_psnrs[tid]       = psnr;

    // ---- Parallel reduction to find best config ----
    __shared__ float s_vals[NN_NUM_CONFIGS];
    __shared__ int   s_idxs[NN_NUM_CONFIGS];

    s_vals[tid] = rank_val;
    s_idxs[tid] = tid;
    __syncthreads();

    // If top-K actions requested, do a full sort (simple insertion sort by thread 0)
    // Otherwise, just do tree reduction for the winner
    if (out_top_actions != nullptr) {
        // Parallel bitonic sort (descending) — all 32 threads participate.
        // NN_NUM_CONFIGS==32==warp size: warp shuffles replace shared memory.
        float my_val = s_vals[tid];
        int   my_idx = s_idxs[tid];
        for (int k = 2; k <= NN_NUM_CONFIGS; k <<= 1) {
            for (int j = k >> 1; j >= 1; j >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFFu, my_val, j);
                int   other_idx = __shfl_xor_sync(0xFFFFFFFFu, my_idx, j);
                bool  is_lower  = ((tid ^ j) > tid);
                bool  swap;
                if (is_lower) {
                    swap = ((tid & k) == 0) ? (my_val < other_val)
                                            : (my_val > other_val);
                } else {
                    swap = ((tid & k) == 0) ? (my_val > other_val)
                                            : (my_val < other_val);
                }
                if (swap) {
                    my_val = other_val;
                    my_idx = other_idx;
                }
            }
        }
        // All threads write their sorted slot in parallel; thread 0 is the winner.
        out_top_actions[tid] = my_idx;
        if (tid == 0) {
            out_result->action              = my_idx;
            out_result->predicted_ratio     = s_ratios[my_idx];
            out_result->predicted_comp_time = s_comp_times[my_idx];
            out_result->predicted_decomp_time = s_decomp_times[my_idx];
            out_result->predicted_psnr      = s_psnrs[my_idx];
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

        // Thread 0 writes the result as a single NNInferenceOutput
        if (tid == 0) {
            int winner = s_idxs[0];
            out_result->action              = winner;
            out_result->predicted_ratio     = s_ratios[winner];
            out_result->predicted_comp_time = s_comp_times[winner];
            out_result->predicted_decomp_time = s_decomp_times[winner];
            out_result->predicted_psnr      = s_psnrs[winner];
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
    float w0, float w1, float w2, float bw,
    NNInferenceOutput* __restrict__ out_result,
    int* __restrict__ out_top_actions,
    NNDebugPerConfig* __restrict__ out_debug  /* nullable: per-config costs */
) {
    int tid = threadIdx.x;
    if (tid >= NN_NUM_CONFIGS) return;

    // ---- Precompute scalars shared across all 32 threads (thread 0 only) ----
    // Reads d_stats once, computes log10/log2 once — 31 redundant ops eliminated.
    __shared__ float s_enc[5];  // [eb_enc, ds_enc, entropy_f, mad_f, deriv_f]
    if (tid == 0) {
        double eb_c = (error_bound < 1e-7) ? 1e-7 : error_bound;
        s_enc[0] = static_cast<float>(log10(eb_c));
        double ds = static_cast<double>(data_size);
        if (ds < 1.0) ds = 1.0;
        s_enc[1] = static_cast<float>(log2(ds));
        s_enc[2] = static_cast<float>(d_stats->entropy);
        s_enc[3] = static_cast<float>(d_stats->mad_normalized);
        s_enc[4] = static_cast<float>(d_stats->deriv_normalized);
    }
    __syncthreads();

    // ---- Forward pass via shared device function ----
    float rank_val, ratio, comp_time, decomp_time, psnr;
    nnForwardPass(weights, tid,
                  s_enc[0], s_enc[1], s_enc[2], s_enc[3], s_enc[4],
                  error_bound,
                  static_cast<float>(data_size), w0, w1, w2, bw,
                  &rank_val, &ratio, &comp_time, &decomp_time, &psnr);

    /* Write per-config debug output (each thread writes its own slot) */
    if (out_debug) {
        float io_cost = static_cast<float>(data_size) / (ratio * bw);
        out_debug[tid].ratio      = ratio;
        out_debug[tid].comp_time  = comp_time;
        out_debug[tid].decomp_time = decomp_time;
        out_debug[tid].cost       = w0 * comp_time + w1 * decomp_time + w2 * io_cost;
    }

    __shared__ float s_ratios[NN_NUM_CONFIGS];
    __shared__ float s_comp_times[NN_NUM_CONFIGS];
    __shared__ float s_decomp_times[NN_NUM_CONFIGS];
    __shared__ float s_psnrs[NN_NUM_CONFIGS];
    s_ratios[tid]      = ratio;
    s_comp_times[tid]  = comp_time;
    s_decomp_times[tid] = decomp_time;
    s_psnrs[tid]       = psnr;

    __shared__ float s_vals[NN_NUM_CONFIGS];
    __shared__ int   s_idxs[NN_NUM_CONFIGS];
    s_vals[tid] = rank_val;
    s_idxs[tid] = tid;
    __syncthreads();

    if (out_top_actions != nullptr) {
        // Parallel bitonic sort (descending) — all 32 threads participate.
        float my_val = s_vals[tid];
        int   my_idx = s_idxs[tid];
        for (int k = 2; k <= NN_NUM_CONFIGS; k <<= 1) {
            for (int j = k >> 1; j >= 1; j >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFFu, my_val, j);
                int   other_idx = __shfl_xor_sync(0xFFFFFFFFu, my_idx, j);
                bool  is_lower  = ((tid ^ j) > tid);
                bool  swap;
                if (is_lower) {
                    swap = ((tid & k) == 0) ? (my_val < other_val)
                                            : (my_val > other_val);
                } else {
                    swap = ((tid & k) == 0) ? (my_val > other_val)
                                            : (my_val < other_val);
                }
                if (swap) {
                    my_val = other_val;
                    my_idx = other_idx;
                }
            }
        }
        // All threads write their sorted slot in parallel; thread 0 is the winner.
        out_top_actions[tid] = my_idx;
        if (tid == 0) {
            int winner = my_idx;
            out_result->action = winner;
            out_result->predicted_ratio = s_ratios[winner];
            out_result->predicted_comp_time = s_comp_times[winner];
            out_result->predicted_decomp_time = s_decomp_times[winner];
            out_result->predicted_psnr = s_psnrs[winner];
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
            out_result->predicted_decomp_time = s_decomp_times[winner];
            out_result->predicted_psnr = s_psnrs[winner];
        }
    }
}

/* ============================================================
 * Pre-allocated fused inference buffers
 * ============================================================ */

static NNInferenceOutput* d_fused_infer_output = nullptr;
static int* d_fused_top_actions = nullptr;

static bool allocFusedInferenceBuffers() {
    if (d_fused_infer_output == nullptr) {
        if (cudaMalloc(&d_fused_infer_output, sizeof(NNInferenceOutput)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for fused inference output\n");
            return false;
        }
    }
    if (d_fused_top_actions == nullptr) {
        if (cudaMalloc(&d_fused_top_actions, NN_NUM_CONFIGS * sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for fused top actions\n");
            return false;
        }
    }
    return true;
}

static void freeFusedInferenceBuffers() {
    if (d_fused_infer_output) { cudaFree(d_fused_infer_output); d_fused_infer_output = nullptr; }
    if (d_fused_top_actions) { cudaFree(d_fused_top_actions); d_fused_top_actions = nullptr; }
}

/* ============================================================
 * GPU SGD Kernel
 * ============================================================ */

// Total gradient buffer size in floats (from nn_weights.h)
static constexpr int SGD_GRAD_SIZE = NN_SGD_GRAD_SIZE; // 3 x 19076 floats = ~228KB
static constexpr int SGD_REGION = NN_SGD_GRAD_REGION;  // 19076 floats per region
static constexpr int EMA_REGION = 2 * SGD_REGION;      // region 2: EMA gradient

// Offsets into gradient buffer (region 0 = accumulator, region 1 = per-output workspace)
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
 * Shared memory layout (~3.6KB):
 *   s_x[15], s_h1[128], s_z1[128], s_h2[128], s_z2[128],
 *   s_y[4], s_d3[4], s_dz2[128], s_reduce[128],
 *   s_d3_all[8][4] (cached per-sample errors for per-output backward)
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
    __shared__ float s_d3_all[NN_MAX_SGD_SAMPLES][NN_OUTPUT_DIM];
    __shared__ float s_d3_raw[NN_MAX_SGD_SAMPLES][NN_OUTPUT_DIM];  // unclamped errors for UW
    __shared__ float s_uw[NN_OUTPUT_DIM];  // uncertainty weight: exp(-0.5 * log_var[o])
    __shared__ float s_dz2_all[NN_OUTPUT_DIM][NN_HIDDEN_DIM];  // per-output L2 grad for PCGrad
    __shared__ float s_pcgrad_dot[NN_OUTPUT_DIM];  // reduction workspace for dot products

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
            // Clamp actual values to sane ranges BEFORE log1p encoding.
            // This prevents extreme out-of-distribution samples from
            // producing unbounded normalized targets that cause gradient
            // explosion and weight oscillation (Issue #1).
            float clamped_ratio     = fmaxf(0.5f,  fminf(sample.actual_ratio,   10000.0f));
            float clamped_comp_time = fmaxf(0.01f, fminf(sample.actual_comp_time, 5000.0f));
            float clamped_decomp    = fmaxf(0.01f, fminf(sample.actual_decomp_time, 5000.0f));

            // Targets in normalized space
            float y_std2 = weights->y_stds[2];
            if (y_std2 < 1e-8f) y_std2 = 1e-8f;
            float target_ratio = (log1pf(clamped_ratio) - weights->y_means[2]) / y_std2;

            s_d3[2] = s_y[2] - target_ratio;

            // Comp time (output 0)
            if (sample.actual_comp_time > 0.0f) {
                float y_std0 = weights->y_stds[0];
                if (y_std0 < 1e-8f) y_std0 = 1e-8f;
                s_d3[0] = s_y[0] - (log1pf(clamped_comp_time) - weights->y_means[0]) / y_std0;
            } else {
                s_d3[0] = 0.0f;
            }

            // Decomp time (output 1)
            if (sample.actual_decomp_time > 0.0f) {
                float y_std1 = weights->y_stds[1];
                if (y_std1 < 1e-8f) y_std1 = 1e-8f;
                s_d3[1] = s_y[1] - (log1pf(clamped_decomp) - weights->y_means[1]) / y_std1;
            } else {
                s_d3[1] = 0.0f;
            }

            // PSNR (output 3) — treat psnr<=0 as 120 dB (lossless = perfect reconstruction)
            {
                float psnr_val = (sample.actual_psnr <= 0.0f) ? 120.0f : sample.actual_psnr;
                float clamped_psnr = fminf(psnr_val, 120.0f);
                float y_std3 = weights->y_stds[3];
                if (y_std3 < 1e-8f) y_std3 = 1e-8f;
                s_d3[3] = s_y[3] - (clamped_psnr - weights->y_means[3]) / y_std3;
            }

            // Clamp per-output error signals (Huber-style gradient clipping).
            //
            // Without clamping, OOD samples produce normalized errors of
            // magnitude 5-10+, which after backprop amplification cause
            // catastrophic weight oscillation: SGD overshoots, a different
            // action wins, the predicted_ratio jumps 100x, repeat.
            //
            // We use a tight delta of 0.5 (half a standard deviation) so
            // each SGD step makes small corrections rather than large jumps.
            // This matches PPO/DQN practice of constraining update magnitude.
            // Step 8 (GPT guide): Noise gating for comp_time (output 0).
            // If |error| < threshold, zero it out — it's GPU timing jitter, not
            // a real prediction error.  Prevents noisy comp_time gradients from
            // destabilizing shared layers.
            constexpr float NOISE_GATE_THRESH = 0.10f;  // ~10% of a std-dev
            if (fabsf(s_d3[0]) < NOISE_GATE_THRESH) s_d3[0] = 0.0f;

            // Cache RAW (unclamped) errors for uncertainty weighting (Phase 1.5).
            // UW needs the true error magnitude to distinguish noisy from clean outputs.
            for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                s_d3_raw[si][o] = s_d3[o];
            }

            constexpr float SGD_ERROR_DELTA = 0.5f;
            for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                s_d3[o] = fmaxf(-SGD_ERROR_DELTA, fminf(s_d3[o], SGD_ERROR_DELTA));
            }

            // Cache clamped errors for Phase 2 backward pass.
            for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                s_d3_all[si][o] = s_d3[o];
            }
        }
        __syncthreads();
    }

    // ================================================================
    // Phase 1.5: Uncertainty weighting (Kendall et al., 2018).
    //
    // Each output has a learned log_var[o] = log(σ²_o).  The precision
    // weight is w_o = exp(-0.5 * log_var[o]) = 1/σ_o.
    //
    // Scaling error signals by w_o before backprop means:
    //   - Noisy outputs (large σ) contribute less to shared W1/W2 gradients
    //   - Clean outputs (small σ) contribute more
    //
    // The log_var update rule is derived from the Gaussian negative
    // log-likelihood: L_o = 0.5 * exp(-log_var[o]) * error² + 0.5 * log_var[o]
    //   ∂L/∂log_var = 0.5 * (1 - exp(-log_var[o]) * error²)
    //
    // This balances: pushing log_var up (reducing gradient influence) vs
    // the regularizer pulling it down (preventing total silencing).
    // ================================================================
    constexpr float UW_LR        = 0.01f;   // slow LR for log_var (stability)
    constexpr float UW_LOG_VAR_MIN = -2.0f;  // exp(-2) ≈ 0.14: floor prevents over-weighting
    constexpr float UW_LOG_VAR_MAX =  4.0f;  // exp(4) ≈ 55: ceiling prevents total silencing

    if (t == 0) {
        for (int o = 0; o < NN_OUTPUT_DIM; o++) {
            float lv = weights->log_var[o];
            float precision = expf(fmaxf(-20.0f, fminf(20.0f, -lv)));  // safe exp(-log_var)

            // Use RAW (unclamped) errors for log_var gradient.
            // Clamped errors are all ≤0.5², so UW can't distinguish outputs.
            // Raw errors preserve the true magnitude: a 2.0 raw error vs 0.05
            // raw error will drive very different log_var updates.
            float raw_mse = 0.0f;
            for (int si = 0; si < num_samples; si++) {
                float e = s_d3_raw[si][o];
                raw_mse += e * e;
            }
            raw_mse /= static_cast<float>(num_samples);

            // Gradient of NLL w.r.t. log_var: 0.5 * (1 - precision * raw_mse)
            // When raw_mse >> σ²: grad is negative → lv increases → downweight
            // When raw_mse << σ²: grad is positive → lv decreases → upweight
            float grad_lv = 0.5f * (1.0f - precision * raw_mse);
            lv -= UW_LR * grad_lv;  // gradient descent on log_var
            lv = fmaxf(UW_LOG_VAR_MIN, fminf(lv, UW_LOG_VAR_MAX));
            weights->log_var[o] = lv;

            // Precision weight for scaling CLAMPED error signals: 1/σ = exp(-0.5 * log_var)
            s_uw[o] = expf(fmaxf(-20.0f, fminf(20.0f, -0.5f * lv)));

            // Scale clamped errors by the uncertainty weight for backprop
            for (int si = 0; si < num_samples; si++)
                s_d3_all[si][o] *= s_uw[o];
        }
    }
    __syncthreads();

    // ================================================================
    // Phase 2: Per-output backward passes through W2/W1
    //
    // Each output's error is backpropagated independently through the
    // shared hidden layers, with per-output gradient clipping.
    // Error signals are already scaled by uncertainty weights (Phase 1.5),
    // so noisy outputs naturally contribute less to W1/W2 updates.
    //
    // Region 0 (d_grad_buffer[0..SGD_REGION-1]) = accumulator for
    //   clipped weight deltas across all 4 outputs.
    // Region 1 (d_grad_buffer[SGD_REGION..2*SGD_REGION-1]) = per-output
    //   workspace, zeroed and reused for each output.
    //
    // Weights are updated ONCE at the end from the accumulated deltas,
    // so forward recomputation always sees the original weights.
    // ================================================================
    constexpr float GRAD_CLIP_THRESHOLD = 0.1f;
    float total_norm = 0.0f;

    // Region 1 workspace offset
    const int WS = SGD_REGION;

    // Region 0 is already zeroed from the initial gradient zero (lines above).
    // Phase 1 no longer accumulates W3 gradients, so region 0 is clean.
    // It will accumulate clipped weight deltas from all 4 outputs.

    for (int target_out = 0; target_out < NN_OUTPUT_DIM; target_out++) {

        // Zero region 1 workspace for this output
        for (int i = 0; i < NN_INPUT_DIM; i++)
            d_grad_buffer[WS + SGD_OFF_DW1 + t * NN_INPUT_DIM + i] = 0.0f;
        d_grad_buffer[WS + SGD_OFF_DB1 + t] = 0.0f;
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            d_grad_buffer[WS + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] = 0.0f;
        d_grad_buffer[WS + SGD_OFF_DB2 + t] = 0.0f;
        d_grad_buffer[WS + SGD_OFF_DW3 + target_out * NN_HIDDEN_DIM + t] = 0.0f;
        if (t == 0)
            d_grad_buffer[WS + SGD_OFF_DB3 + target_out] = 0.0f;
        __syncthreads();

        for (int si = 0; si < num_samples; si++) {
            SGDSample sample = samples[si];

            // Recompute forward pass to recover activations
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

            // Forward L1
            {
                float sum = weights->b1[t];
                for (int i = 0; i < NN_INPUT_DIM; i++)
                    sum += weights->w1[t * NN_INPUT_DIM + i] * s_x[i];
                s_z1[t] = sum;
                s_h1[t] = (sum > 0.0f) ? sum : 0.0f;
            }
            __syncthreads();

            // Forward L2
            {
                float sum = weights->b2[t];
                for (int i = 0; i < NN_HIDDEN_DIM; i++)
                    sum += weights->w2[t * NN_HIDDEN_DIM + i] * s_h1[i];
                s_z2[t] = sum;
                s_h2[t] = (sum > 0.0f) ? sum : 0.0f;
            }
            __syncthreads();

            // ────────────────────────────────────────────────────────
            // PCGrad-Lite: compute L3→L2 backward for ALL outputs,
            // project conflicting gradients, then continue backward.
            // ────────────────────────────────────────────────────────

            // Step 1: Compute s_dz2 for ALL outputs and normalize.
            //
            // Normalization makes projection scale-invariant: a large-magnitude
            // output can't dominate the projection just because its gradient is
            // bigger.  After projection, we rescale back to the original magnitude.
            {
                for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                    float es = s_d3_all[si][o];
                    float dh2_t = weights->w3[o * NN_HIDDEN_DIM + t] * es;
                    s_dz2_all[o][t] = (s_z2[t] > 0.0f) ? dh2_t : 0.0f;
                }
            }
            __syncthreads();

            // Normalize per-output gradients to unit vectors for PCGrad.
            // After projection, rescale by |error| (not gradient norm) so the
            // update magnitude reflects how wrong each output is, not how
            // large W3 happens to make the gradient.
            for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                float local_nsq = s_dz2_all[o][t] * s_dz2_all[o][t];
                s_reduce[t] = local_nsq;
                __syncthreads();
                for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
                    if (t < s) s_reduce[t] += s_reduce[t + s];
                    __syncthreads();
                }
                float norm_o = sqrtf(s_reduce[0]) + 1e-6f;
                if (t < NN_OUTPUT_DIM && t == o)
                    s_pcgrad_dot[o] = fabsf(s_d3_all[si][o]);  // error magnitude for rescaling
                s_dz2_all[o][t] /= norm_o;    // normalize to unit vector
                __syncthreads();
            }

            // Step 2: PCGrad projection with SOFT conflict threshold.
            //
            // Only project when cosine similarity < -ε (true conflict).
            // With normalized vectors, dot(g_i, g_j) IS the cosine similarity.
            // ε = 0.1 means we ignore conflicts within ±0.1 cosine — avoids
            // reacting to noise in single-sample SGD.
            constexpr float PCGRAD_COS_THRESH = -0.1f;  // only project if cos < -0.1
            {
                float my_dz2 = s_dz2_all[target_out][t];

                for (int j = 0; j < NN_OUTPUT_DIM; j++) {
                    if (j == target_out) continue;

                    // Cosine similarity (vectors are unit-normalized)
                    float local_dot = my_dz2 * s_dz2_all[j][t];
                    s_reduce[t] = local_dot;
                    __syncthreads();
                    for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
                        if (t < s) s_reduce[t] += s_reduce[t + s];
                        __syncthreads();
                    }
                    float cos_ij = s_reduce[0];

                    // Soft threshold: only project for strong conflicts
                    if (cos_ij < PCGRAD_COS_THRESH) {
                        // Vectors are unit-normalized → ||g_j||² = 1.0
                        // So projection simplifies to: g_i -= dot * g_j
                        my_dz2 -= cos_ij * s_dz2_all[j][t];
                    }
                    __syncthreads();
                }

                // Rescale projected gradient back to original magnitude
                s_dz2[t] = my_dz2 * s_pcgrad_dot[target_out];
            }
            __syncthreads();

            // Step 3: Accumulate W3 gradient (uses ORIGINAL error_signal, not projected)
            {
                float error_signal = s_d3_all[si][target_out];
                d_grad_buffer[WS + SGD_OFF_DW3 + target_out * NN_HIDDEN_DIM + t] +=
                    error_signal * s_h2[t];
                if (t == 0) {
                    d_grad_buffer[WS + SGD_OFF_DB3 + target_out] += error_signal;
                }
            }

            // Step 4: Accumulate L2 gradients using PROJECTED s_dz2
            for (int i = 0; i < NN_HIDDEN_DIM; i++) {
                d_grad_buffer[WS + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] += s_dz2[t] * s_h1[i];
            }
            d_grad_buffer[WS + SGD_OFF_DB2 + t] += s_dz2[t];

            // Step 5: Backward L2→L1 using PROJECTED s_dz2
            float dz1_t;
            {
                float dh1_t = 0.0f;
                for (int j = 0; j < NN_HIDDEN_DIM; j++)
                    dh1_t += weights->w2[j * NN_HIDDEN_DIM + t] * s_dz2[j];
                dz1_t = (s_z1[t] > 0.0f) ? dh1_t : 0.0f;
            }

            // Step 6: Accumulate L1 gradients
            for (int i = 0; i < NN_INPUT_DIM; i++) {
                d_grad_buffer[WS + SGD_OFF_DW1 + t * NN_INPUT_DIM + i] += dz1_t * s_x[i];
            }
            d_grad_buffer[WS + SGD_OFF_DB1 + t] += dz1_t;

            __syncthreads();
        }

        // Average this output's gradients over samples and compute norm
        float inv_n = 1.0f / static_cast<float>(num_samples);
        float local_norm_sq = 0.0f;

        for (int i = 0; i < NN_INPUT_DIM; i++) {
            int idx = WS + SGD_OFF_DW1 + t * NN_INPUT_DIM + i;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }
        {
            int idx = WS + SGD_OFF_DB1 + t;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            int idx = WS + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }
        {
            int idx = WS + SGD_OFF_DB2 + t;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }

        // Include W3 gradient for this output (from workspace)
        {
            int idx = WS + SGD_OFF_DW3 + target_out * NN_HIDDEN_DIM + t;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }
        if (t == target_out) {
            int idx = WS + SGD_OFF_DB3 + target_out;
            d_grad_buffer[idx] *= inv_n;
            local_norm_sq += d_grad_buffer[idx] * d_grad_buffer[idx];
        }

        // Per-output gradient norm reduction
        s_reduce[t] = local_norm_sq;
        __syncthreads();
        for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
            if (t < s) s_reduce[t] += s_reduce[t + s];
            __syncthreads();
        }

        float out_norm = sqrtf(s_reduce[0]) + 1e-8f;
        float clip_scale = (out_norm > GRAD_CLIP_THRESHOLD) ? (GRAD_CLIP_THRESHOLD / out_norm) : 1.0f;
        float lr_out = learning_rate * clip_scale;
        total_norm += out_norm;

        // Accumulate clipped weight deltas into region 0 (deferred update)
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i] +=
                lr_out * d_grad_buffer[WS + SGD_OFF_DW1 + t * NN_INPUT_DIM + i];
        }
        d_grad_buffer[SGD_OFF_DB1 + t] +=
            lr_out * d_grad_buffer[WS + SGD_OFF_DB1 + t];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] +=
                lr_out * d_grad_buffer[WS + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i];
        }
        d_grad_buffer[SGD_OFF_DB2 + t] +=
            lr_out * d_grad_buffer[WS + SGD_OFF_DB2 + t];

        // W3/b3 for this output
        d_grad_buffer[SGD_OFF_DW3 + target_out * NN_HIDDEN_DIM + t] +=
            lr_out * d_grad_buffer[WS + SGD_OFF_DW3 + target_out * NN_HIDDEN_DIM + t];
        if (t == target_out) {
            d_grad_buffer[SGD_OFF_DB3 + target_out] +=
                lr_out * d_grad_buffer[WS + SGD_OFF_DB3 + target_out];
        }

        __syncthreads();
    }

    // ================================================================
    // Steps 5-9: Trust-region scaling, EMA smoothing, anti-flip damping.
    //
    // Region 0 now holds the combined gradient from all outputs.
    // Region 2 (EMA_REGION) stores the exponential moving average.
    // ================================================================
    constexpr float EMA_DECAY      = 0.85f;   // EMA smoothing factor
    constexpr float TRUST_K        = 0.08f;   // step = k * avg_error
    constexpr float MAX_STEP       = 0.02f;   // maximum step size
    constexpr float MIN_STEP       = 1e-4f;   // minimum step (avoid stall)
    constexpr float ANTI_FLIP_DAMP = 0.5f;    // halve update on direction reversal

    {
        // Step 5: Compute gradient norm for trust-region scaling
        float local_norm_sq = 0.0f;
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            float v = d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i];
            local_norm_sq += v * v;
        }
        local_norm_sq += d_grad_buffer[SGD_OFF_DB1 + t] * d_grad_buffer[SGD_OFF_DB1 + t];
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            float v = d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i];
            local_norm_sq += v * v;
        }
        local_norm_sq += d_grad_buffer[SGD_OFF_DB2 + t] * d_grad_buffer[SGD_OFF_DB2 + t];
        for (int out = 0; out < NN_OUTPUT_DIM; out++) {
            float v = d_grad_buffer[SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t];
            local_norm_sq += v * v;
        }
        if (t < NN_OUTPUT_DIM) {
            local_norm_sq += d_grad_buffer[SGD_OFF_DB3 + t] * d_grad_buffer[SGD_OFF_DB3 + t];
        }

        s_reduce[t] = local_norm_sq;
        __syncthreads();
        for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
            if (t < s) s_reduce[t] += s_reduce[t + s];
            __syncthreads();
        }
        float g_norm = sqrtf(s_reduce[0]) + 1e-8f;

        // Normalize gradient to unit vector (trust-region: decouple direction from magnitude)
        float inv_norm = 1.0f / g_norm;
        for (int i = 0; i < NN_INPUT_DIM; i++)
            d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i] *= inv_norm;
        d_grad_buffer[SGD_OFF_DB1 + t] *= inv_norm;
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i] *= inv_norm;
        d_grad_buffer[SGD_OFF_DB2 + t] *= inv_norm;
        for (int out = 0; out < NN_OUTPUT_DIM; out++)
            d_grad_buffer[SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t] *= inv_norm;
        if (t < NN_OUTPUT_DIM)
            d_grad_buffer[SGD_OFF_DB3 + t] *= inv_norm;
        __syncthreads();

        // Step 5 continued: Trust-region step size = k * avg_error, clamped
        // Compute avg |error| across active outputs and samples
        __shared__ float s_avg_err;
        if (t == 0) {
            float sum_err = 0.0f;
            int count = 0;
            for (int si = 0; si < num_samples; si++) {
                for (int o = 0; o < NN_OUTPUT_DIM; o++) {
                    float e = fabsf(s_d3_raw[si][o]);
                    if (e > 0.0f) { sum_err += e; count++; }
                }
            }
            s_avg_err = (count > 0) ? sum_err / static_cast<float>(count) : 0.0f;
        }
        __syncthreads();

        // Step 7: Minimum step to avoid stall
        float step = fmaxf(MIN_STEP, fminf(MAX_STEP, TRUST_K * s_avg_err));

        // Step 9: Anti-flip damping — dot current gradient with EMA (previous direction)
        // If they point in opposite directions, damp the step.
        // Use region 2 (EMA) as the "previous direction" reference.
        float local_dot_ema = 0.0f;
        for (int i = 0; i < NN_INPUT_DIM; i++)
            local_dot_ema += d_grad_buffer[SGD_OFF_DW1 + t * NN_INPUT_DIM + i]
                           * d_grad_buffer[EMA_REGION + SGD_OFF_DW1 + t * NN_INPUT_DIM + i];
        local_dot_ema += d_grad_buffer[SGD_OFF_DB1 + t]
                       * d_grad_buffer[EMA_REGION + SGD_OFF_DB1 + t];
        for (int i = 0; i < NN_HIDDEN_DIM; i++)
            local_dot_ema += d_grad_buffer[SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i]
                           * d_grad_buffer[EMA_REGION + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i];
        local_dot_ema += d_grad_buffer[SGD_OFF_DB2 + t]
                       * d_grad_buffer[EMA_REGION + SGD_OFF_DB2 + t];

        s_reduce[t] = local_dot_ema;
        __syncthreads();
        for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
            if (t < s) s_reduce[t] += s_reduce[t + s];
            __syncthreads();
        }
        float dot_ema = s_reduce[0];

        // Only apply anti-flip after warmup (first few SGD calls have no meaningful EMA)
        bool warmed_up = (weights->sgd_call_count > 3);
        if (warmed_up && dot_ema < 0.0f) {
            step *= ANTI_FLIP_DAMP;  // halve step when reversing direction
        }

        // Step 6: EMA smoothing — blend current gradient into EMA buffer
        // g_ema = decay * g_ema + (1-decay) * g_current
        // Sanitize: if current gradient has NaN (from corrupted forward pass),
        // zero it out to prevent permanent EMA poisoning.
        float ema_new = 1.0f - EMA_DECAY;
        {
            // Sanitize region 0 (current gradient) — zero any NaN/Inf entries
            for (int i = 0; i < NN_INPUT_DIM; i++) {
                int idx = SGD_OFF_DW1 + t * NN_INPUT_DIM + i;
                if (!isfinite(d_grad_buffer[idx])) d_grad_buffer[idx] = 0.0f;
            }
            if (!isfinite(d_grad_buffer[SGD_OFF_DB1 + t]))
                d_grad_buffer[SGD_OFF_DB1 + t] = 0.0f;
            for (int i = 0; i < NN_HIDDEN_DIM; i++) {
                int idx = SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i;
                if (!isfinite(d_grad_buffer[idx])) d_grad_buffer[idx] = 0.0f;
            }
            if (!isfinite(d_grad_buffer[SGD_OFF_DB2 + t]))
                d_grad_buffer[SGD_OFF_DB2 + t] = 0.0f;
            for (int out = 0; out < NN_OUTPUT_DIM; out++) {
                int idx = SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t;
                if (!isfinite(d_grad_buffer[idx])) d_grad_buffer[idx] = 0.0f;
            }
            if (t < NN_OUTPUT_DIM && !isfinite(d_grad_buffer[SGD_OFF_DB3 + t]))
                d_grad_buffer[SGD_OFF_DB3 + t] = 0.0f;
        }
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            int idx = SGD_OFF_DW1 + t * NN_INPUT_DIM + i;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        {
            int idx = SGD_OFF_DB1 + t;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            int idx = SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        {
            int idx = SGD_OFF_DB2 + t;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        for (int out = 0; out < NN_OUTPUT_DIM; out++) {
            int idx = SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        if (t < NN_OUTPUT_DIM) {
            int idx = SGD_OFF_DB3 + t;
            d_grad_buffer[EMA_REGION + idx] = EMA_DECAY * d_grad_buffer[EMA_REGION + idx]
                                            + ema_new * d_grad_buffer[idx];
        }
        __syncthreads();

        // NaN/Inf guard: skip weight update entirely if gradient is corrupted.
        // This prevents a single bad sample from permanently destroying the network.
        if (!isfinite(step) || !isfinite(s_reduce[0])) {
            if (t == 0) weights->sgd_call_count++;
            total_norm = 0.0f;
            // Fall through to output — skip weight update
        } else {
            // Apply EMA-smoothed gradient with trust-region step size
            constexpr float W_CLAMP = 10.0f;  // prevent weight explosion
            for (int i = 0; i < NN_INPUT_DIM; i++) {
                float w = weights->w1[t * NN_INPUT_DIM + i] -
                    step * d_grad_buffer[EMA_REGION + SGD_OFF_DW1 + t * NN_INPUT_DIM + i];
                weights->w1[t * NN_INPUT_DIM + i] = fmaxf(-W_CLAMP, fminf(W_CLAMP, w));
            }
            {
                float b = weights->b1[t] - step * d_grad_buffer[EMA_REGION + SGD_OFF_DB1 + t];
                weights->b1[t] = fmaxf(-W_CLAMP, fminf(W_CLAMP, b));
            }
            for (int i = 0; i < NN_HIDDEN_DIM; i++) {
                float w = weights->w2[t * NN_HIDDEN_DIM + i] -
                    step * d_grad_buffer[EMA_REGION + SGD_OFF_DW2 + t * NN_HIDDEN_DIM + i];
                weights->w2[t * NN_HIDDEN_DIM + i] = fmaxf(-W_CLAMP, fminf(W_CLAMP, w));
            }
            {
                float b = weights->b2[t] - step * d_grad_buffer[EMA_REGION + SGD_OFF_DB2 + t];
                weights->b2[t] = fmaxf(-W_CLAMP, fminf(W_CLAMP, b));
            }
            for (int out = 0; out < NN_OUTPUT_DIM; out++) {
                // Skip decomp_time head (output 1) — owned by batched deferred SGD
                if (out == 1) continue;
                float w = weights->w3[out * NN_HIDDEN_DIM + t] -
                    step * d_grad_buffer[EMA_REGION + SGD_OFF_DW3 + out * NN_HIDDEN_DIM + t];
                weights->w3[out * NN_HIDDEN_DIM + t] = fmaxf(-W_CLAMP, fminf(W_CLAMP, w));
            }
            if (t < NN_OUTPUT_DIM && t != 1) {
                float b = weights->b3[t] - step * d_grad_buffer[EMA_REGION + SGD_OFF_DB3 + t];
                weights->b3[t] = fmaxf(-W_CLAMP, fminf(W_CLAMP, b));
            }

            // Update call counter for warmup
            if (t == 0) weights->sgd_call_count++;
        }

        total_norm = g_norm;
    }

    // Thread 0 writes output
    if (t == 0) {
        out_result->grad_norm = total_norm;
        out_result->was_clipped = (total_norm > GRAD_CLIP_THRESHOLD) ? 1 : 0;
        out_result->sample_count = num_samples;
    }
}

/* ============================================================
 * Pre-allocated SGD buffers
 * ============================================================ */

static float* d_sgd_grad_buffer = nullptr;
static SGDOutput* d_sgd_output = nullptr;
static SGDSample* d_sgd_samples = nullptr;

static bool allocSGDBuffers() {
    if (d_sgd_grad_buffer == nullptr) {
        if (cudaMalloc(&d_sgd_grad_buffer, SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for SGD gradient buffer\n");
            d_sgd_grad_buffer = nullptr;
            return false;
        }
        // Zero the entire buffer (includes EMA region 2)
        if (cudaMemset(d_sgd_grad_buffer, 0, SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMemset failed for SGD gradient buffer\n");
        }
    }
    if (d_sgd_output == nullptr) {
        if (cudaMalloc(&d_sgd_output, sizeof(SGDOutput)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for SGD output\n");
            d_sgd_output = nullptr;
            return false;
        }
    }
    if (d_sgd_samples == nullptr) {
        if (cudaMalloc(&d_sgd_samples, NN_MAX_SGD_SAMPLES * sizeof(SGDSample)) != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed for SGD samples\n");
            d_sgd_samples = nullptr;
            return false;
        }
    }
    return true;
}

static void freeSGDBuffers() {
    if (d_sgd_grad_buffer) { cudaFree(d_sgd_grad_buffer); d_sgd_grad_buffer = nullptr; }
    if (d_sgd_output) { cudaFree(d_sgd_output); d_sgd_output = nullptr; }
    if (d_sgd_samples) { cudaFree(d_sgd_samples); d_sgd_samples = nullptr; }
}

/* Forward declarations for batched decomp SGD (defined after runNNSGDCtx) */
static void freeDecompSGDBuffers();

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
    g_nn_loaded.store(false);

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "NN: Failed to open weights file: %s\n", filepath);
        return false;
    }

    // Read header as a single block so gcount() covers all 6 fields
    uint32_t magic, version, n_layers, input_dim, hidden_dim, output_dim;
    {
        uint32_t hdr[6] = {};
        file.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if ((size_t)file.gcount() != sizeof(hdr)) {
            fprintf(stderr, "NN: Truncated header (%zd/24 bytes) in %s\n",
                    (size_t)file.gcount(), filepath);
            return false;
        }
        magic = hdr[0]; version = hdr[1]; n_layers = hdr[2];
        input_dim = hdr[3]; hidden_dim = hdr[4]; output_dim = hdr[5];
    }

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

    // Read all weights into host struct; check gcount() after every read
    // so a truncated file is caught immediately with a precise error message.
#define NN_READ(ptr, nbytes, name) \
    do { \
        size_t _nb = (size_t)(nbytes); \
        file.read(reinterpret_cast<char*>(ptr), (std::streamsize)_nb); \
        if ((size_t)file.gcount() != _nb) { \
            fprintf(stderr, "NN: Truncated %s (%zd/%zu B) in %s\n", \
                    (name), (size_t)file.gcount(), _nb, filepath); \
            return false; \
        } \
    } while (0)

    NNWeightsGPU h_weights;

    // Normalization parameters
    NN_READ(h_weights.x_means, NN_INPUT_DIM * sizeof(float),  "x_means");
    NN_READ(h_weights.x_stds,  NN_INPUT_DIM * sizeof(float),  "x_stds");
    NN_READ(h_weights.y_means, NN_OUTPUT_DIM * sizeof(float), "y_means");
    NN_READ(h_weights.y_stds,  NN_OUTPUT_DIM * sizeof(float), "y_stds");

    // Layer 1
    NN_READ(h_weights.w1, NN_HIDDEN_DIM * NN_INPUT_DIM  * sizeof(float), "w1");
    NN_READ(h_weights.b1, NN_HIDDEN_DIM                 * sizeof(float), "b1");

    // Layer 2
    NN_READ(h_weights.w2, NN_HIDDEN_DIM * NN_HIDDEN_DIM * sizeof(float), "w2");
    NN_READ(h_weights.b2, NN_HIDDEN_DIM                 * sizeof(float), "b2");

    // Layer 3
    NN_READ(h_weights.w3, NN_OUTPUT_DIM * NN_HIDDEN_DIM * sizeof(float), "w3");
    NN_READ(h_weights.b3, NN_OUTPUT_DIM                 * sizeof(float), "b3");

#undef NN_READ

    // Feature bounds (v2+)
    if (version >= 2) {
        size_t bounds_sz = NN_INPUT_DIM * sizeof(float);
        file.read(reinterpret_cast<char*>(h_weights.x_mins), (std::streamsize)bounds_sz);
        if ((size_t)file.gcount() != bounds_sz) {
            fprintf(stderr, "NN: Truncated x_mins (%zd/%zu B) in %s\n",
                    (size_t)file.gcount(), bounds_sz, filepath);
            return false;
        }
        file.read(reinterpret_cast<char*>(h_weights.x_maxs), (std::streamsize)bounds_sz);
        if ((size_t)file.gcount() != bounds_sz) {
            fprintf(stderr, "NN: Truncated x_maxs (%zd/%zu B) in %s\n",
                    (size_t)file.gcount(), bounds_sz, filepath);
            return false;
        }
    } else {
        // v1: no bounds, set defaults
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            h_weights.x_mins[i] = -1e30f;
            h_weights.x_maxs[i] = 1e30f;
        }
    }

    // Initialize per-output log-variance to 0.0 (σ²=1, neutral weighting).
    // SGD will learn to increase log_var for noisy outputs, reducing their
    // gradient contribution to shared layers.
    for (int o = 0; o < NN_OUTPUT_DIM; o++)
        h_weights.log_var[o] = 0.0f;

    // Initialize anti-flip damping and call counter
    h_weights.prev_grad_dot = 0.0f;
    h_weights.sgd_call_count = 0;

    // Allocate new GPU buffer and copy weights into it
    NNWeightsGPU* d_new = nullptr;
    {
        cudaError_t err = cudaMalloc(&d_new, sizeof(NNWeightsGPU));
        if (err != cudaSuccess) {
            fprintf(stderr, "NN: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }

    {
        cudaError_t err = cudaMemcpy(d_new, &h_weights, sizeof(NNWeightsGPU),
                                      cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "NN: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_new);
            return false;
        }
    }

    // Atomically swap the weight pointer under lock, then free old after sync
    NNWeightsGPU* d_old = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        d_old = d_nn_weights;
        d_nn_weights = d_new;
    }
    g_nn_loaded.store(true);

    if (d_old != nullptr) {
        /* S3 fix: drain only CompContext + SGD streams, not the entire device */
        gpucompress::syncAllCompContextStreams();
        cudaFree(d_old);
    }

    // Pre-allocate inference output buffers
    if (!allocInferenceBuffers() || !allocFusedInferenceBuffers()) {
        fprintf(stderr, "NN: Failed to allocate inference buffers\n");
        return false;
    }
    allocSGDBuffers();

    size_t total_params = NN_HIDDEN_DIM * NN_INPUT_DIM + NN_HIDDEN_DIM +
                          NN_HIDDEN_DIM * NN_HIDDEN_DIM + NN_HIDDEN_DIM +
                          NN_OUTPUT_DIM * NN_HIDDEN_DIM + NN_OUTPUT_DIM;
    printf("NN: Loaded %zu parameters from %s (%.1f KB on GPU)\n",
           total_params, filepath, sizeof(NNWeightsGPU) / 1024.0f);

    return true;
}


/**
 * Free neural network GPU memory.
 */
void cleanupNN() {
    NNWeightsGPU* d_old = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        d_old = d_nn_weights;
        d_nn_weights = nullptr;
    }
    g_nn_loaded.store(false);
    if (d_old != nullptr) {
        /* S3 fix: drain only CompContext + SGD streams */
        gpucompress::syncAllCompContextStreams();
        cudaFree(d_old);
    }
    freeInferenceBuffers();
    freeFusedInferenceBuffers();
    freeSGDBuffers();
    freeDecompSGDBuffers();
}

/**
 * Check if neural network is loaded.
 */
bool isNNLoaded() {
    return g_nn_loaded.load();
}

/**
 * Get device pointer to NN weights.
 */
const NNWeightsGPU* getNNWeightsDevicePtr() {
    std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
    return d_nn_weights;
}

/**
 * Save a snapshot of the current NN weights from device to host buffer.
 */
extern "C" gpucompress_error_t gpucompress_nn_save_snapshot(void* dst) {
    if (!dst) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
    if (!d_nn_weights) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    cudaError_t err = cudaMemcpy(dst, d_nn_weights, sizeof(NNWeightsGPU),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_CUDA_FAILED;
}

/**
 * Restore NN weights from a host snapshot buffer to device.
 */
extern "C" gpucompress_error_t gpucompress_nn_restore_snapshot(const void* src) {
    if (!src) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
    if (!d_nn_weights) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    cudaError_t err = cudaMemcpy(d_nn_weights, src, sizeof(NNWeightsGPU),
                                  cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_CUDA_FAILED;
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
    float* out_predicted_decomp_time,
    float* out_predicted_psnr,
    int* out_top_actions
) {
    // Ensure pre-allocated buffers exist (outside pointer lock)
    if (d_infer_output == nullptr) {
        if (!allocInferenceBuffers()) return -1;
    }

    // Lock pointer, launch kernel, then release — prevents use-after-free during reload
    cudaError_t err;
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) {
            return -1;  // NN not loaded
        }

        /* GPU-level barrier: wait for last SGD before reading weights */
        if (g_sgd_ever_fired.load(std::memory_order_acquire))
            cudaStreamWaitEvent(stream, g_sgd_done, 0);

        nnInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
            d_nn_weights,
            entropy, mad_norm, deriv_norm,
            data_size, error_bound,
            g_rank_w0, g_rank_w1, g_rank_w2, g_measured_bw_bytes_per_ms,
            d_infer_output,
            out_top_actions ? d_infer_top_actions : nullptr
        );

        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return -1;

    // Single D→H copy of NNInferenceOutput (16B) — replaces 3 separate transfers
    NNInferenceOutput h_result;
    err = cudaMemcpyAsync(&h_result, d_infer_output, sizeof(NNInferenceOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    if (out_top_actions) {
        err = cudaMemcpyAsync(out_top_actions, d_infer_top_actions,
                               NN_NUM_CONFIGS * sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_predicted_ratio)      *out_predicted_ratio      = h_result.predicted_ratio;
    if (out_predicted_comp_time)  *out_predicted_comp_time  = h_result.predicted_comp_time;
    if (out_predicted_decomp_time)*out_predicted_decomp_time = h_result.predicted_decomp_time;
    if (out_predicted_psnr)       *out_predicted_psnr       = h_result.predicted_psnr;

    return h_result.action;
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
    float* out_decomp_time,
    float* out_psnr,
    int* out_top_actions,
    cudaEvent_t nn_stop_event
) {
    if (d_stats == nullptr) return -1;

    // Ensure fused buffers exist (outside pointer lock)
    if (d_fused_infer_output == nullptr) {
        if (!allocFusedInferenceBuffers()) return -1;
    }

    cudaError_t err;
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) return -1;

        /* GPU-level barrier: wait for last SGD before reading weights */
        if (g_sgd_ever_fired.load(std::memory_order_acquire))
            cudaStreamWaitEvent(stream, g_sgd_done, 0);

        nnFusedInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
            d_nn_weights,
            d_stats,
            data_size, error_bound,
            g_rank_w0, g_rank_w1, g_rank_w2, g_measured_bw_bytes_per_ms,
            d_fused_infer_output,
            out_top_actions ? d_fused_top_actions : nullptr,
            nullptr  /* no debug output in non-ctx path */
        );

        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return -1;

    // Single D→H of NNInferenceOutput (16B)
    NNInferenceOutput h_result;
    err = cudaMemcpyAsync(&h_result, d_fused_infer_output, sizeof(NNInferenceOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    if (out_top_actions) {
        err = cudaMemcpyAsync(out_top_actions, d_fused_top_actions,
                               NN_NUM_CONFIGS * sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return -1;
    }

    if (nn_stop_event) cudaEventRecord(nn_stop_event, stream);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_action)     *out_action     = h_result.action;
    if (out_ratio)      *out_ratio      = h_result.predicted_ratio;
    if (out_comp_time)  *out_comp_time  = h_result.predicted_comp_time;
    if (out_decomp_time)*out_decomp_time = h_result.predicted_decomp_time;
    if (out_psnr)       *out_psnr       = h_result.predicted_psnr;

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
    if (d_stats == nullptr) return -1;
    if (samples == nullptr || num_samples <= 0) return -1;
    if (num_samples > NN_MAX_SGD_SAMPLES) num_samples = NN_MAX_SGD_SAMPLES;

    // Ensure SGD buffers exist (outside pointer lock)
    if (d_sgd_grad_buffer == nullptr) {
        allocSGDBuffers();
        if (d_sgd_grad_buffer == nullptr) return -1;
    }

    // H→D: copy samples (tiny: num_samples * 20B)
    cudaError_t err = cudaMemcpyAsync(d_sgd_samples, samples,
                                       num_samples * sizeof(SGDSample),
                                       cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return -1;

    // Lock pointer, launch kernel, then release
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) return -1;

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
    }
    if (err != cudaSuccess) return -1;

    /* Record completion so future inference waits for this SGD */
    cudaEventRecord(g_sgd_done, stream);
    g_sgd_ever_fired.store(true, std::memory_order_release);

    // D→H: copy SGDOutput (12B)
    SGDOutput h_result;
    err = cudaMemcpyAsync(&h_result, d_sgd_output, sizeof(SGDOutput),
                           cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return -1;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (out_grad_norm) *out_grad_norm = h_result.grad_norm;
    if (out_clipped) *out_clipped = h_result.was_clipped;
    if (out_count) *out_count = h_result.sample_count;

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
    float* out_decomp_time,
    float* out_psnr,
    int* out_top_actions,
    cudaEvent_t nn_stop_event
) {
    if (d_stats == nullptr || ctx == nullptr) return -1;

    /* Debug: allocate per-config output buffer on first use */
    NNDebugPerConfig* d_debug = nullptr;
    static NNDebugPerConfig* s_d_debug = nullptr;
    if (g_debug_nn) {
        if (!s_d_debug) cudaMalloc(&s_d_debug, NN_NUM_CONFIGS * sizeof(NNDebugPerConfig));
        d_debug = s_d_debug;
    }

    cudaError_t err;
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) return -1;

        /* GPU-level barrier: wait for last SGD on g_sgd_stream before reading weights */
        if (g_sgd_ever_fired.load(std::memory_order_acquire))
            cudaStreamWaitEvent(stream, g_sgd_done, 0);

        nnFusedInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(
            d_nn_weights,
            d_stats,
            data_size, error_bound,
            g_rank_w0, g_rank_w1, g_rank_w2, g_measured_bw_bytes_per_ms,
            ctx->d_fused_infer_output,
            out_top_actions ? ctx->d_fused_top_actions : nullptr,
            d_debug
        );

        err = cudaGetLastError();
    }
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

    /* Debug: copy back and print per-config costs */
    NNDebugPerConfig h_debug[NN_NUM_CONFIGS];
    if (d_debug) {
        cudaMemcpyAsync(h_debug, d_debug, NN_NUM_CONFIGS * sizeof(NNDebugPerConfig),
                         cudaMemcpyDeviceToHost, stream);
    }

    if (nn_stop_event) cudaEventRecord(nn_stop_event, stream);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return -1;

    if (d_debug) {
        float dbg_entropy = -1.0f, dbg_mad = -1.0f, dbg_deriv = -1.0f;
        if (d_stats) {
            AutoStatsGPU h_stats;
            cudaMemcpy(&h_stats, d_stats, sizeof(AutoStatsGPU), cudaMemcpyDeviceToHost);
            dbg_entropy = (float)h_stats.entropy;
            dbg_mad     = (float)h_stats.mad_normalized;
            dbg_deriv   = (float)h_stats.deriv_normalized;
        }
        gpucompress::printNNDebugRanking(h_debug, h_result.action,
                                          dbg_entropy, dbg_mad, dbg_deriv);
    }

    if (out_action)      *out_action      = h_result.action;
    if (out_ratio)       *out_ratio       = h_result.predicted_ratio;
    if (out_comp_time)   *out_comp_time   = h_result.predicted_comp_time;
    if (out_decomp_time) *out_decomp_time = h_result.predicted_decomp_time;
    if (out_psnr)        *out_psnr        = h_result.predicted_psnr;

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
    if (d_stats == nullptr || ctx == nullptr) return -1;
    if (samples == nullptr || num_samples <= 0) return -1;
    if (num_samples > NN_MAX_SGD_SAMPLES) num_samples = NN_MAX_SGD_SAMPLES;

    /* H→D: copy samples on g_sgd_stream */
    cudaError_t err = cudaMemcpyAsync(ctx->d_sgd_samples, samples,
                                       num_samples * sizeof(SGDSample),
                                       cudaMemcpyHostToDevice, g_sgd_stream);
    if (err != cudaSuccess) return -1;

    /* Lock pointer, launch SGD kernel, then release */
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) return -1;

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
    }
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

    g_sgd_ever_fired.store(true, std::memory_order_release);

    if (out_grad_norm) *out_grad_norm = h_result.grad_norm;
    if (out_clipped)   *out_clipped   = h_result.was_clipped;
    if (out_count)     *out_count     = h_result.sample_count;

    return 0;
}

/* ============================================================
 * Batched Deferred Decomp Head-Only SGD
 *
 * Called ONCE per timestep after all chunks are read.
 * Receives N samples with actual decomp times, computes mean
 * gradient across all samples, applies ONE bounded update to
 * W3[1] + b3[1].  No shared-layer changes.
 * ============================================================ */

static DeferredDecompSample* d_decomp_samples = nullptr;
static SGDOutput*            d_decomp_sgd_output = nullptr;
static int                   d_decomp_samples_cap = 0;

static bool allocDecompSGDBuffers(int n) {
    if (d_decomp_samples == nullptr || n > d_decomp_samples_cap) {
        if (d_decomp_samples) cudaFree(d_decomp_samples);
        if (cudaMalloc(&d_decomp_samples, n * sizeof(DeferredDecompSample)) != cudaSuccess) {
            d_decomp_samples = nullptr;
            d_decomp_samples_cap = 0;
            return false;
        }
        d_decomp_samples_cap = n;
    }
    if (d_decomp_sgd_output == nullptr) {
        if (cudaMalloc(&d_decomp_sgd_output, sizeof(SGDOutput)) != cudaSuccess) {
            d_decomp_sgd_output = nullptr;
            return false;
        }
    }
    return true;
}

static void freeDecompSGDBuffers() {
    if (d_decomp_samples)    { cudaFree(d_decomp_samples);    d_decomp_samples = nullptr; }
    if (d_decomp_sgd_output) { cudaFree(d_decomp_sgd_output); d_decomp_sgd_output = nullptr; }
    d_decomp_samples_cap = 0;
}

/**
 * Batched decomp head-only SGD kernel.
 * Processes N samples: for each, recomputes forward pass through frozen W1/W2,
 * accumulates mean gradient for W3[1]+b3[1], applies ONE update.
 *
 * Launch: <<<1, 128>>>
 */
__global__ void nnBatchedDecompSGDKernel(
    NNWeightsGPU* __restrict__ weights,
    const DeferredDecompSample* __restrict__ samples,
    int num_samples,
    float learning_rate,
    SGDOutput* __restrict__ out_result
) {
    int t = threadIdx.x;  // 0..127

    __shared__ float s_x[NN_INPUT_DIM];
    __shared__ float s_h1[NN_HIDDEN_DIM];
    __shared__ float s_h2[NN_HIDDEN_DIM];
    __shared__ float s_reduce[NN_HIDDEN_DIM];
    __shared__ float s_err;

    // Accumulate mean gradient across all samples
    float acc_gw = 0.0f;   // accumulated dw3 for thread t
    float acc_gb = 0.0f;   // accumulated db3 (thread 0 only)
    float acc_abs_err = 0.0f; // accumulated |error| for trust-region (thread 0 only)
    int   valid  = 0;

    for (int si = 0; si < num_samples; si++) {
        const DeferredDecompSample& samp = samples[si];

        // Thread 0 builds standardized input
        if (t == 0) {
            int algo_idx = samp.action % 8;
            int quant    = (samp.action / 8) % 2;
            int shuffle  = (samp.action / 16) % 2;

            float raw[NN_INPUT_DIM];
            for (int i = 0; i < 8; i++) raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
            raw[8]  = static_cast<float>(quant);
            raw[9]  = static_cast<float>(shuffle);
            raw[10] = samp.error_bound_enc;
            raw[11] = samp.data_size_enc;
            raw[12] = samp.entropy;
            raw[13] = samp.mad_normalized;
            raw[14] = samp.deriv_normalized;

            for (int i = 0; i < NN_INPUT_DIM; i++) {
                float std_val = weights->x_stds[i];
                if (std_val < 1e-8f) std_val = 1e-8f;
                s_x[i] = (raw[i] - weights->x_means[i]) / std_val;
            }
        }
        __syncthreads();

        // Forward L1 (read-only)
        {
            float sum = weights->b1[t];
            for (int i = 0; i < NN_INPUT_DIM; i++)
                sum += weights->w1[t * NN_INPUT_DIM + i] * s_x[i];
            s_h1[t] = (sum > 0.0f) ? sum : 0.0f;
        }
        __syncthreads();

        // Forward L2 (read-only)
        {
            float sum = weights->b2[t];
            for (int i = 0; i < NN_HIDDEN_DIM; i++)
                sum += weights->w2[t * NN_HIDDEN_DIM + i] * s_h1[i];
            s_h2[t] = (sum > 0.0f) ? sum : 0.0f;
        }
        __syncthreads();

        // Forward L3 output 1 only, compute error in log-space
        if (t == 0) {
            float y_norm = weights->b3[1];
            for (int i = 0; i < NN_HIDDEN_DIM; i++)
                y_norm += weights->w3[1 * NN_HIDDEN_DIM + i] * s_h2[i];

            float y_std1 = weights->y_stds[1];
            if (y_std1 < 1e-8f) y_std1 = 1e-8f;
            float pred_log = y_norm * y_std1 + weights->y_means[1];

            float clamped = fmaxf(0.01f, fminf(samp.actual_decomp_ms, 5000.0f));
            float target_log = log1pf(clamped);

            float err_log = pred_log - target_log;

            // Clamp error in log-space
            err_log = fmaxf(-2.0f, fminf(2.0f, err_log));

            // Convert to normalized-space for gradient
            s_err = err_log / y_std1;
        }
        __syncthreads();

        // Skip if error is tiny (noise gate)
        if (fabsf(s_err) < 0.05f) { __syncthreads(); continue; }

        // Accumulate gradient: dw3[t] += err * h2[t], db3 += err
        acc_gw += s_err * s_h2[t];
        if (t == 0) {
            acc_gb += s_err;
            acc_abs_err += fabsf(s_err);
        }
        valid++;
        __syncthreads();
    }

    // Average gradient
    if (valid == 0) {
        if (t == 0) {
            out_result->grad_norm = 0.0f;
            out_result->was_clipped = 0;
            out_result->sample_count = 0;
        }
        return;
    }

    float inv_n = 1.0f / static_cast<float>(valid);
    acc_gw *= inv_n;
    if (t == 0) acc_gb *= inv_n;

    // Compute gradient norm for diagnostic
    s_reduce[t] = acc_gw * acc_gw;
    __syncthreads();
    for (int s = NN_HIDDEN_DIM / 2; s > 0; s >>= 1) {
        if (t < s) s_reduce[t] += s_reduce[t + s];
        __syncthreads();
    }
    float g_norm = sqrtf(s_reduce[0] + (t == 0 ? acc_gb * acc_gb : 0.0f)) + 1e-8f;

    // Compute mean absolute error for trust-region scaling.
    // The step size is proportional to the error: large error → large step,
    // small error → small step.  This prevents overshooting when the head
    // is already well-calibrated, eliminating the oscillation that occurs
    // with a constant step size.
    __shared__ float s_mean_abs_err;
    if (t == 0) {
        s_mean_abs_err = acc_abs_err * inv_n;
    }
    __syncthreads();

    // Trust-region: step = min(lr, k * |mean_error|), capped
    constexpr float DECOMP_TRUST_K = 0.15f;
    constexpr float DECOMP_MAX_STEP = 0.05f;
    constexpr float DECOMP_MIN_STEP = 1e-4f;
    float step = fmaxf(DECOMP_MIN_STEP, fminf(DECOMP_MAX_STEP,
                        DECOMP_TRUST_K * s_mean_abs_err));

    // Normalize gradient, apply bounded step
    float gw_normed = acc_gw / g_norm;
    float gb_normed = (t == 0) ? acc_gb / g_norm : 0.0f;

    // Apply with weight clamp
    constexpr float W_CLAMP = 5.0f;
    float new_w = weights->w3[1 * NN_HIDDEN_DIM + t] - step * gw_normed;
    weights->w3[1 * NN_HIDDEN_DIM + t] = fmaxf(-W_CLAMP, fminf(W_CLAMP, new_w));
    if (t == 0) {
        float new_b = weights->b3[1] - step * gb_normed;
        weights->b3[1] = fmaxf(-W_CLAMP, fminf(W_CLAMP, new_b));
    }

    if (t == 0) {
        out_result->grad_norm = g_norm;
        out_result->was_clipped = 0;
        out_result->sample_count = valid;
    }
}

/**
 * Host wrapper for batched decomp SGD.
 */
int runBatchedDecompSGD(
    const DeferredDecompSample* samples,
    int num_samples,
    float learning_rate,
    float* out_grad_norm
) {
    if (samples == nullptr || num_samples <= 0) return -1;

    if (!allocDecompSGDBuffers(num_samples)) return -1;

    // H→D: copy all samples
    cudaError_t err = cudaMemcpyAsync(d_decomp_samples, samples,
                                       num_samples * sizeof(DeferredDecompSample),
                                       cudaMemcpyHostToDevice, g_sgd_stream);
    if (err != cudaSuccess) return -1;

    // Launch kernel under NN pointer lock
    {
        std::lock_guard<std::mutex> lock(g_nn_ptr_mutex);
        if (!g_nn_loaded.load() || d_nn_weights == nullptr) return -1;

        nnBatchedDecompSGDKernel<<<1, NN_HIDDEN_DIM, 0, g_sgd_stream>>>(
            d_nn_weights,
            d_decomp_samples,
            num_samples,
            learning_rate,
            d_decomp_sgd_output
        );
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return -1;

    cudaEventRecord(g_sgd_done, g_sgd_stream);
    g_sgd_ever_fired.store(true, std::memory_order_release);

    SGDOutput h_result;
    err = cudaMemcpyAsync(&h_result, d_decomp_sgd_output, sizeof(SGDOutput),
                           cudaMemcpyDeviceToHost, g_sgd_stream);
    if (err != cudaSuccess) return -1;

    err = cudaStreamSynchronize(g_sgd_stream);
    if (err != cudaSuccess) return -1;

    if (out_grad_norm) *out_grad_norm = h_result.grad_norm;

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

void* gpucompress_nn_get_device_ptr_impl(void) {
    std::lock_guard<std::mutex> lock(gpucompress::g_nn_ptr_mutex);
    return static_cast<void*>(gpucompress::d_nn_weights);
}

} // extern "C"
