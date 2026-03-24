/**
 * @file internal.hpp
 * @brief Internal C++ interface for GPUCompress library
 *
 * This header provides internal C++ utilities used by the library.
 * Not part of the public API.
 */

#ifndef INTERNAL_HPP
#define INTERNAL_HPP

#include <cuda_runtime.h>

#include "gpucompress.h"
#include "compression/compression_factory.hpp"
#include "compression/compression_header.h"
#include "stats/auto_stats_gpu.h"
#include "nn/nn_weights.h"

/* ============================================================
 * CompContext pool — per-compression-slot GPU state
 * ============================================================ */

/** Number of concurrent compression slots.
 *  9 = 8 worker slots + 1 dedicated inference slot (used by VOL Stage 1). */
static constexpr int N_COMP_CTX = 9;

/**
 * Per-slot GPU state for concurrent compression.
 * Eliminates sharing of global stats/NN/SGD buffers between concurrent calls.
 */
struct CompContext {
    int           slot_id;
    cudaStream_t  stream;
    cudaEvent_t   t_start, t_stop;
    cudaEvent_t   nn_start, nn_stop;
    cudaEvent_t   stats_start, stats_stop;

    /* Stats workspace (~1.1 KB) */
    void*          d_stats_workspace;
    AutoStatsGPU*  d_stats;
    unsigned int*  d_histogram;

    /* Fused inference buffers */
    NNInferenceOutput* d_fused_infer_output;
    int*               d_fused_top_actions;
    float*             d_fused_costs;       /* 32 per-config predicted costs */

    /* SGD buffers (~75 KB) */
    float*      d_sgd_grad_buffer;
    SGDOutput*  d_sgd_output;
    SGDSample*  d_sgd_samples;

    /* Quantization range buffers (8B each).
     * Replaces the static d_range_min/d_range_max globals in quantization_kernels.cu.
     * Each slot owns its own pair so concurrent quantize_simple() calls on different
     * streams never alias each other's min/max reduction targets. */
    void* d_range_min;
    void* d_range_max;

    static constexpr int N_COMP_ALGOS = 8;

    /* LRU-3 nvcomp manager cache — up to 3 managers per slot.
     * Raw pointers: CompContext is memset-zeroed in initCompContextPool(),
     * which would corrupt non-trivial dtors (unique_ptr).
     * Deleted safely in destroyCompContextSlot() after cudaStreamSynchronize.
     *
     * comp_mgr[i]      : cached nvcomp manager (nullptr = empty slot)
     * comp_mgr_algo[i] : CompressionAlgorithm index (0-7), or -1 = empty
     * comp_mgr_tick[i] : access recency (higher = more recent)
     * comp_mgr_clock   : monotonic tick counter for LRU eviction */
    static constexpr int LRU_DEPTH = 3;
    nvcomp::nvcompManagerBase* comp_mgr[LRU_DEPTH];
    int                        comp_mgr_algo[LRU_DEPTH];
    int                        comp_mgr_tick[LRU_DEPTH];
    int                        comp_mgr_clock;
};

namespace gpucompress {

/* ============================================================
 * Internal Constants
 * ============================================================ */

/** Default chunk size for nvcomp (64KB) */
constexpr size_t DEFAULT_CHUNK_SIZE = 1 << 16;

/** Chunk size for byte shuffle (256KB) */
constexpr size_t SHUFFLE_CHUNK_SIZE = 256 * 1024;

/** Alignment for GPU memory (4KB for optimal performance) */
constexpr size_t GPU_ALIGNMENT = 4096;

/* ============================================================
 * Algorithm Mapping
 * ============================================================ */

/**
 * Map gpucompress_algorithm_t to internal CompressionAlgorithm.
 */
inline CompressionAlgorithm toInternalAlgorithm(gpucompress_algorithm_t algo) {
    switch (algo) {
        case GPUCOMPRESS_ALGO_LZ4:      return CompressionAlgorithm::LZ4;
        case GPUCOMPRESS_ALGO_SNAPPY:   return CompressionAlgorithm::SNAPPY;
        case GPUCOMPRESS_ALGO_DEFLATE:  return CompressionAlgorithm::DEFLATE;
        case GPUCOMPRESS_ALGO_GDEFLATE: return CompressionAlgorithm::GDEFLATE;
        case GPUCOMPRESS_ALGO_ZSTD:     return CompressionAlgorithm::ZSTD;
        case GPUCOMPRESS_ALGO_ANS:      return CompressionAlgorithm::ANS;
        case GPUCOMPRESS_ALGO_CASCADED: return CompressionAlgorithm::CASCADED;
        case GPUCOMPRESS_ALGO_BITCOMP:  return CompressionAlgorithm::BITCOMP;
        default:                        return CompressionAlgorithm::LZ4;
    }
}

/**
 * Map internal CompressionAlgorithm to gpucompress_algorithm_t.
 */
inline gpucompress_algorithm_t fromInternalAlgorithm(CompressionAlgorithm algo) {
    switch (algo) {
        case CompressionAlgorithm::LZ4:      return GPUCOMPRESS_ALGO_LZ4;
        case CompressionAlgorithm::SNAPPY:   return GPUCOMPRESS_ALGO_SNAPPY;
        case CompressionAlgorithm::DEFLATE:  return GPUCOMPRESS_ALGO_DEFLATE;
        case CompressionAlgorithm::GDEFLATE: return GPUCOMPRESS_ALGO_GDEFLATE;
        case CompressionAlgorithm::ZSTD:     return GPUCOMPRESS_ALGO_ZSTD;
        case CompressionAlgorithm::ANS:      return GPUCOMPRESS_ALGO_ANS;
        case CompressionAlgorithm::CASCADED: return GPUCOMPRESS_ALGO_CASCADED;
        case CompressionAlgorithm::BITCOMP:  return GPUCOMPRESS_ALGO_BITCOMP;
        default:                             return GPUCOMPRESS_ALGO_LZ4;
    }
}

/* ============================================================
 * Action Encoding/Decoding
 * ============================================================ */

/**
 * Decoded action configuration.
 */
struct DecodedAction {
    int algorithm;          /**< Algorithm index 0-7 */
    bool use_quantization;  /**< Whether to apply quantization */
    int shuffle_size;       /**< Shuffle element size (0 or 4) */
};

/**
 * Decode action ID to action configuration.
 *
 * Encoding: action = algorithm + quant*8 + shuffle*16
 */
inline DecodedAction decodeAction(int action_id) {
    DecodedAction action;
    action.algorithm = action_id % 8;
    action.use_quantization = ((action_id / 8) % 2) != 0;
    action.shuffle_size = ((action_id / 16) % 2) ? 4 : 0;
    return action;
}

/* ============================================================
 * CUDA Utilities
 * ============================================================ */

/**
 * Align size to specified alignment.
 */
inline size_t alignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * CUDA error checking wrapper.
 */
inline gpucompress_error_t checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Entropy Calculation (GPU)
 * ============================================================ */

/**
 * Launch entropy kernels asynchronously without D->H copy.
 *
 * Fire-and-forget variant that writes entropy to a device pointer.
 *
 * @param d_data         Data buffer (GPU memory)
 * @param num_bytes      Size in bytes
 * @param d_histogram    Pre-allocated histogram buffer (256 uints)
 * @param d_entropy_out  Device pointer to write entropy result
 * @param stream         CUDA stream
 */
int launchEntropyKernelsAsync(
    const void* d_data,
    size_t num_bytes,
    unsigned int* d_histogram,
    double* d_entropy_out,
    cudaStream_t stream
);

/* ============================================================
 * Neural Network GPU Functions
 * ============================================================ */

/**
 * Load neural network weights from binary file (.nnwt).
 *
 * @param filepath Path to .nnwt weights file
 * @return true on success
 */
bool loadNNFromBinary(const char* filepath);

/**
 * Free neural network GPU memory.
 */
void cleanupNN();

/**
 * Check if neural network is loaded.
 */
bool isNNLoaded();

/**
 * Print all 32 NN config rankings to stderr (debug output).
 */
void printNNDebugRanking(const NNDebugPerConfig* h_debug, int winner_action,
                         float entropy = -1.0f, float mad = -1.0f, float deriv = -1.0f);

/**
 * Input struct for recordChunkDiagnostic — avoids long parameter lists.
 */
struct ChunkDiagInput {
    int nn_action, nn_original_action;
    bool exploration_triggered, sgd_fired;
    float nn_inference_ms, stats_ms, preprocessing_ms;
    float compression_ms, compression_ms_raw, exploration_ms, sgd_ms;
    size_t input_size, primary_compressed_size, compressed_size;
    float predicted_ratio, predicted_comp_time, predicted_decomp_time, predicted_psnr;
    double error_bound;
    const AutoStatsGPU* d_stats_ptr;
    /* Cost model */
    float cost_model_error_pct, actual_cost, predicted_cost;
    /* Original config metrics (before exploration swap) */
    float orig_actual_ratio, orig_comp_ms, orig_cost;
    /* Exploration results */
    int explore_n_alternatives;
    int explore_alternatives[31];
    float explore_ratios[31];
    float explore_comp_ms[31];
    float explore_costs[31];
    /* NN predicted ranking and costs (all 32 configs) */
    const int* top_actions;       /* nullptr if non-AUTO path */
    int top_actions_count;        /* 32 if AUTO, 0 otherwise */
    const float* predicted_costs; /* nullptr if non-AUTO path, indexed by action ID */
};

/**
 * Record per-chunk diagnostic entry to history buffer.
 */
void recordChunkDiagnostic(const ChunkDiagInput& d);

/**
 * Run neural network inference to find best compression config.
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
    float* out_predicted_ratio = nullptr,
    float* out_predicted_comp_time = nullptr,
    float* out_predicted_decomp_time = nullptr,
    float* out_predicted_psnr = nullptr,
    int* out_top_actions = nullptr
);

/**
 * Run stats-only pipeline on GPU (no NN inference or Q-Table lookup).
 *
 * Computes entropy, normalized MAD, and normalized second derivative
 * entirely on GPU.
 *
 * @param d_input      Float data already on GPU
 * @param input_size   Size in bytes
 * @param stream       CUDA stream
 * @param out_entropy  [out] Shannon entropy (0-8 bits)
 * @param out_mad      [out] Normalized MAD (0-1)
 * @param out_deriv    [out] Normalized second derivative (0-1)
 * @return 0 on success, -1 on error
 */
int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
);

/**
 * Run stats kernels without D→H copy or sync.
 * Returns device pointer to stats buffer for fused pipeline.
 */
AutoStatsGPU* runStatsKernelsNoSync(const void* d_input, size_t input_size, cudaStream_t stream);

/* ============================================================
 * CompContext pool API
 * ============================================================ */

int          initCompContextPool();
void         flushCompManagerCache();   ///< evict all cached nvcomp managers (lightweight)
void         destroyCompContextPool();
CompContext* acquireCompContext();   ///< blocks until a slot is free
void         releaseCompContext(CompContext*);

/** S3 fix: synchronize all CompContext streams without draining the entire device.
 *  Used by NN load/cleanup to ensure no in-flight kernels reference old weights. */
void         syncAllCompContextStreams();

/** Create a fresh compression manager for this context+algorithm.
 *  Caller owns the returned unique_ptr.  Used for exploration alternatives. */
std::unique_ptr<nvcomp::nvcompManagerBase> createCompManagerForCtx(CompContext* ctx, CompressionAlgorithm algo);

/** LRU-3: return a cached manager if algo is in the 3-deep cache, else evict
 *  the least-recently-used entry and create a new one.  Thread-safe because
 *  each CompContext is exclusively owned by one thread at a time. */
nvcomp::nvcompManagerBase* getOrCreateCompManager(CompContext* ctx, CompressionAlgorithm algo);

/* ============================================================
 * Context-aware inference / SGD overloads (used by pool)
 * ============================================================ */

/** ctx overload: uses ctx->d_stats / d_histogram / d_stats_workspace */
AutoStatsGPU* runStatsKernelsNoSync(const void* d_input, size_t input_size,
    cudaStream_t stream, CompContext* ctx);

/** ctx overload: uses ctx->d_fused_infer_output / d_fused_top_actions;
 *  inserts cudaStreamWaitEvent on g_sgd_done if SGD has ever fired. */
int runNNFusedInferenceCtx(const AutoStatsGPU* d_stats, size_t data_size,
    double error_bound, cudaStream_t stream, CompContext* ctx,
    int* out_action, float* out_ratio = nullptr, float* out_comp_time = nullptr,
    float* out_decomp_time = nullptr, float* out_psnr = nullptr,
    int* out_top_actions = nullptr, float* out_predicted_costs = nullptr,
    cudaEvent_t nn_stop_event = nullptr);

/** ctx overload: launches nnSGDKernel on g_sgd_stream (not ctx->stream),
 *  uses ctx->d_sgd_* buffers, records g_sgd_done, syncs g_sgd_stream. */
int runNNSGDCtx(const AutoStatsGPU* d_stats, const SGDSample* samples,
    int num_samples, size_t data_size, double error_bound, float learning_rate,
    CompContext* ctx, float* out_grad_norm = nullptr,
    int* out_clipped = nullptr, int* out_count = nullptr);

/**
 * Fused NN inference: reads stats directly from device pointer.
 * Eliminates stats D→H→GPU round-trip.
 */
int runNNFusedInference(const AutoStatsGPU* d_stats, size_t data_size, double error_bound,
    cudaStream_t stream, int* out_action, float* out_ratio = nullptr,
    float* out_comp_time = nullptr, float* out_decomp_time = nullptr,
    float* out_psnr = nullptr,
    int* out_top_actions = nullptr, cudaEvent_t nn_stop_event = nullptr);

/**
 * GPU-native SGD: forward/backward pass + weight update entirely on GPU.
 * Eliminates ~152KB D→H + H→D round-trip per SGD call.
 */
int runNNSGD(const AutoStatsGPU* d_stats, const SGDSample* samples, int num_samples,
    size_t data_size, double error_bound, float learning_rate, cudaStream_t stream,
    float* out_grad_norm = nullptr, int* out_clipped = nullptr, int* out_count = nullptr);

/**
 * Batched deferred decomp head-only SGD.
 * Takes N samples with actual decomp times, computes mean gradient,
 * applies ONE update to W3[1] + b3[1]. No shared layer changes.
 */
int runBatchedDecompSGD(const DeferredDecompSample* samples, int num_samples,
    float learning_rate, float* out_grad_norm = nullptr);

/** Get device pointer to current NN weights (read-only). */
const NNWeightsGPU* getNNWeightsDevicePtr();

} // namespace gpucompress

/* ============================================================
 * Split-phase API for sequential-inference pipeline (VOL)
 * ============================================================ */

/**
 * Phase A: stats + NN inference only.
 * Runs on the provided CompContext's stream.  Returns the NN action and
 * predictions so they can be forwarded to gpucompress_compress_with_action_gpu().
 *
 * @param d_input           GPU-resident input data
 * @param input_size        Size in bytes
 * @param cfg               Compression config (only error_bound used)
 * @param stats             Optional stats struct (entropy/mad/deriv filled)
 * @param ctx               CompContext to run stats+inference on
 * @param out_action        [out] NN action (0-31), or -1 on error
 * @param out_predicted_ratio       [out] predicted ratio
 * @param out_predicted_comp_time   [out] predicted compression time
 * @param out_predicted_decomp_time [out] predicted decompression time
 * @param out_predicted_psnr        [out] predicted PSNR
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_infer_gpu(
    const void* d_input, size_t input_size,
    const gpucompress_config_t* cfg,
    gpucompress_stats_t* stats,
    CompContext* ctx,
    int* out_action,
    float* out_predicted_ratio,
    float* out_predicted_comp_time,
    float* out_predicted_decomp_time,
    float* out_predicted_psnr,
    int* out_top_actions = nullptr,
    float* out_predicted_costs = nullptr);

/**
 * Phase B: preprocess + compress + exploration + SGD.
 * Uses the pre-computed action from gpucompress_infer_gpu() instead of
 * running its own stats+inference.  Acquires its own CompContext from the pool.
 *
 * @param d_input           GPU-resident input data
 * @param input_size        Size in bytes
 * @param d_output          GPU output buffer
 * @param output_size       [in] max size, [out] actual compressed size
 * @param config            Compression config
 * @param stats             Optional stats struct
 * @param stream_arg        Optional caller CUDA stream
 * @param action            Pre-computed NN action from Phase A
 * @param predicted_ratio   Predicted ratio from Phase A
 * @param predicted_comp_time   Predicted compression time from Phase A
 * @param predicted_decomp_time Predicted decompression time from Phase A
 * @param predicted_psnr    Predicted PSNR from Phase A
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_with_action_gpu(
    const void* d_input, size_t input_size,
    void* d_output, size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats,
    void* stream_arg,
    int action,
    float predicted_ratio,
    float predicted_comp_time,
    float predicted_decomp_time,
    float predicted_psnr,
    const int* top_actions = nullptr,
    const float* predicted_costs = nullptr,
    float stage1_nn_ms = 0.0f,
    float stage1_stats_ms = 0.0f,
    AutoStatsGPU* d_precomputed_stats = nullptr);

#endif /* INTERNAL_HPP */
