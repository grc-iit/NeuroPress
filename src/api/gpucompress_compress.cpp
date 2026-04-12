/**
 * @file gpucompress_compress.cpp
 * @brief GPU-path compression, inference, exploration, and decompression.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>

#include "gpucompress.h"
#include "api/internal.hpp"
#include "api/gpucompress_state.hpp"
#include "api/diagnostics_store.hpp"
#include "compression/compression_factory.hpp"
#include "compression/compression_header.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

/* Analytical PSNR for linear quantization.
 * Uses expected MSE = eb²/3 (uniform error in [-eb, eb]),
 * matching the real MSE formula: PSNR = 10*log10(range²/MSE). */
static inline double analytical_psnr(double data_range, double error_bound) {
    if (data_range <= 0.0 || error_bound <= 0.0) return -1.0;
    double mse_expected = (error_bound * error_bound) / 3.0;
    return fmin(10.0 * log10((data_range * data_range) / mse_expected), 120.0);
}

/* Detailed timing helpers — zero overhead when g_detailed_timing is false */
#define DT_START(var) \
    auto var = g_detailed_timing ? std::chrono::steady_clock::now() \
                                 : std::chrono::steady_clock::time_point{}
#define DT_MS(start) \
    (g_detailed_timing ? std::chrono::duration<float, std::milli>( \
        std::chrono::steady_clock::now() - (start)).count() : 0.0f)

extern "C" {
    int gpucompress_nn_is_loaded_impl(void);
}

extern "C" gpucompress_error_t gpucompress_compress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats,
    void* stream_arg
) {
    if (!g_initialized.load()) return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    if (!d_input || !d_output || !output_size) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (input_size == 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_config_t cfg = config ? *config : gpucompress_default_config();

    /* ---- Non-AUTO: encode explicit algo as action, delegate directly ---- */
    if (cfg.algorithm != GPUCOMPRESS_ALGO_AUTO) {
        int action = (cfg.algorithm - 1) % 8;  /* algo index 0-7 */
        if (cfg.preprocessing & GPUCOMPRESS_PREPROC_QUANTIZE)
            action += 8;
        if (GPUCOMPRESS_GET_SHUFFLE_SIZE(cfg.preprocessing) > 0)
            action += 16;
        return gpucompress_compress_with_action_gpu(
            d_input, input_size, d_output, output_size, &cfg, stats, stream_arg,
            action, 0.0f, 0.0f, 0.0f, 0.0f, nullptr, nullptr, 0.0f, 0.0f, nullptr);
    }

    /* ---- ALGO_AUTO + NN: acquire context, run inference, delegate ----
     * NOTE: This holds the infer context while _with_action_gpu acquires a
     * second context for compression.  Two slots are occupied simultaneously.
     * This is safe for typical usage (sequential per-chunk calls) but could
     * cause pool exhaustion if >4 threads call compress_gpu concurrently. */
    ContextGuard infer_guard{gpucompress::acquireCompContext()};
    if (!infer_guard.ctx) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    CompContext* infer_ctx = infer_guard.ctx;

    int action = -1;
    float pred_ratio = 0.0f, pred_ct = 0.0f, pred_dt = 0.0f, pred_psnr = 0.0f;
    int top_actions[32] = {};
    float predicted_costs[32] = {};

    gpucompress_error_t ie = gpucompress_infer_gpu(
        d_input, input_size, &cfg, stats, infer_ctx,
        &action, &pred_ratio, &pred_ct, &pred_dt, &pred_psnr,
        top_actions, predicted_costs);

    if (ie != GPUCOMPRESS_SUCCESS || action < 0) {
        fprintf(stderr, "gpucompress ERROR: ALGO_AUTO requested but NN inference failed "
                "(weights not loaded or inference error). Load weights via gpucompress_init() "
                "or use an explicit algorithm.\n");
        return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
    }

    /* Capture timing from inference context */
    float nn_ms = 0.0f, stats_ms = 0.0f;
    cudaEventElapsedTime(&nn_ms, infer_ctx->nn_start, infer_ctx->nn_stop);
    cudaEventElapsedTime(&stats_ms, infer_ctx->stats_start, infer_ctx->stats_stop);

    /* Pre-computed stats live in infer_ctx->d_stats — pass to _with_action_gpu
     * so it can reuse them for SGD without recomputing. */
    AutoStatsGPU* d_precomputed_stats = infer_ctx->d_stats;

    /* Release inference context before delegating (frees pool slot for worker) */
    /* NOTE: d_precomputed_stats points into infer_ctx->d_stats which remains
     * valid until the context is reused.  _with_action_gpu copies it into its
     * own ctx->d_stats buffer before releasing the infer context's slot. */

    return gpucompress_compress_with_action_gpu(
        d_input, input_size, d_output, output_size, &cfg, stats, stream_arg,
        action, pred_ratio, pred_ct, pred_dt, pred_psnr,
        top_actions, predicted_costs, nn_ms, stats_ms, d_precomputed_stats);
}

/* ============================================================
 * Split-phase API: Phase A — stats + inference only
 * ============================================================ */
gpucompress_error_t gpucompress_infer_gpu(
    const void* d_input, size_t input_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats,
    CompContext* ctx,
    int* out_action,
    float* out_predicted_ratio,
    float* out_predicted_comp_time,
    float* out_predicted_decomp_time,
    float* out_predicted_psnr,
    int* out_top_actions,
    float* out_predicted_costs,
    float* out_predicted_rmse,
    float* out_predicted_max_error,
    float* out_predicted_mae,
    float* out_predicted_ssim,
    NNDebugPerConfig* out_per_config)
{
    if (!g_initialized.load()) return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    if (!d_input || !out_action) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (input_size == 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_config_t cfg = config ? *config : gpucompress_default_config();
    cudaStream_t stream = ctx->stream;

    *out_action = -1;
    if (out_predicted_ratio)       *out_predicted_ratio = 0.0f;
    if (out_predicted_comp_time)   *out_predicted_comp_time = 0.0f;
    if (out_predicted_decomp_time) *out_predicted_decomp_time = 0.0f;
    if (out_predicted_psnr)        *out_predicted_psnr = 0.0f;
    if (out_predicted_rmse)        *out_predicted_rmse = 0.0f;
    if (out_predicted_max_error)   *out_predicted_max_error = 0.0f;
    if (out_predicted_mae)         *out_predicted_mae = 0.0f;
    if (out_predicted_ssim)        *out_predicted_ssim = 0.0f;

    size_t num_elements = input_size / sizeof(float);
    if (num_elements == 0 || !gpucompress_nn_is_loaded_impl())
        return GPUCOMPRESS_ERROR_NN_NOT_LOADED;

    /* Stats timing */
    cudaEventRecord(ctx->stats_start, stream);

    AutoStatsGPU* d_stats_ptr = gpucompress::runStatsKernelsNoSync(d_input, input_size, stream, ctx);

    /* Stats remain on GPU — NN inference reads d_stats_ptr directly on device. */
    cudaEventRecord(ctx->stats_stop, stream);

    if (!d_stats_ptr)
        return GPUCOMPRESS_ERROR_NN_NOT_LOADED;



    float pred_ratio = 0.0f, pred_ct = 0.0f, pred_dt = 0.0f, pred_psnr = 0.0f;
    float pred_rmse = 0.0f, pred_max_error = 0.0f, pred_mae = 0.0f, pred_ssim = 0.0f;

    int local_top[32] = {0};
    float local_costs[32] = {0};

    cudaEventRecord(ctx->nn_start, stream);
    int action = gpucompress::runNNFusedInferenceCtx(
        d_stats_ptr, input_size, cfg.error_bound, stream, ctx,
        &action, &pred_ratio, &pred_ct, &pred_dt, &pred_psnr,
        local_top, local_costs,
        ctx->nn_stop,
        &pred_rmse, &pred_max_error, &pred_mae, &pred_ssim,
        out_per_config);

    /* runNNFusedInferenceCtx does internal cudaStreamSynchronize */

    if (action < 0)
        return GPUCOMPRESS_ERROR_NN_NOT_LOADED;

    *out_action = action;
    if (out_predicted_ratio)       *out_predicted_ratio       = pred_ratio;
    if (out_predicted_comp_time)   *out_predicted_comp_time   = pred_ct;
    if (out_predicted_decomp_time) *out_predicted_decomp_time = pred_dt;
    if (out_predicted_psnr)        *out_predicted_psnr        = pred_psnr;
    if (out_predicted_rmse)        *out_predicted_rmse        = pred_rmse;
    if (out_predicted_max_error)   *out_predicted_max_error   = pred_max_error;
    if (out_predicted_mae)         *out_predicted_mae         = pred_mae;
    if (out_predicted_ssim)        *out_predicted_ssim        = pred_ssim;
    if (out_top_actions)           memcpy(out_top_actions, local_top, sizeof(local_top));
    if (out_predicted_costs)       memcpy(out_predicted_costs, local_costs, sizeof(local_costs));

    if (stats != nullptr) {
        stats->entropy_bits = 0.0;
        stats->mad = 0.0;
        stats->second_derivative = 0.0;
        stats->predicted_ratio = static_cast<double>(pred_ratio);
        stats->predicted_comp_time_ms = static_cast<double>(pred_ct);
        stats->predicted_decomp_time_ms = static_cast<double>(pred_dt);
        stats->predicted_psnr_db = static_cast<double>(pred_psnr);
    }

    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Split-phase API: Phase B — preprocess + compress + SGD
 * Uses pre-computed action from gpucompress_infer_gpu().
 * ============================================================ */
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
    const int* top_actions,
    const float* predicted_costs,
    float stage1_nn_ms,
    float stage1_stats_ms,
    AutoStatsGPU* d_precomputed_stats,
    int* out_diag_slot,
    float predicted_rmse,
    float predicted_max_error,
    float predicted_mae,
    float predicted_ssim)
{
    if (!g_initialized.load()) return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    if (!d_input || !d_output || !output_size) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (input_size == 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_config_t cfg = config ? *config : gpucompress_default_config();

    /* Acquire a per-slot context (blocks until one is free) */
    DT_START(_dt_ctx);
    ContextGuard guard{gpucompress::acquireCompContext()};
    float dt_ctx_acquire = DT_MS(_dt_ctx);
    if (!guard.ctx) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    CompContext* ctx = guard.ctx;
    cudaStream_t stream = ctx->stream;
    cudaStream_t caller_stream = stream_arg ? static_cast<cudaStream_t>(stream_arg) : nullptr;
    if (caller_stream) {
        cudaEventRecord(ctx->t_start, caller_stream);
        cudaStreamWaitEvent(stream, ctx->t_start, 0);
    }

    /* NN action → algo + preprocessing */
    int nn_action = action;
    int nn_original_action = action;
    bool sgd_fired = false, exploration_triggered = false;
    AutoStatsGPU* d_stats_ptr = nullptr;
    /* Primary algorithm's actual metrics — saved before exploration may overwrite.
     * Used in chunk diagnostics for fair MAPE reporting. */
    size_t primary_compressed_size = 0;
    float primary_actual_psnr = -1.0f;  /* -1.0 = lossless (skip PSNR MAPE) */

    /* Cost model diagnostics (populated when online learning enabled) */
    double error_pct = 0.0, actual_cost = 0.0, predicted_cost = 0.0;
    struct ExploredResult { int action; double ratio; double comp_time_ms;
                            double decomp_time_ms; double psnr; double cost; };
    std::vector<ExploredResult> explored_samples;

    /* Per-chunk timing breakdown.
     * stage1_nn_ms / stage1_stats_ms are passed from the VOL Stage 1
     * sequential-inference loop where stats+NN ran on a dedicated context. */
    bool timing_ok = (ctx->t_start != nullptr && ctx->t_stop != nullptr);
    float diag_nn_inference_ms  = stage1_nn_ms;
    float diag_stats_ms         = stage1_stats_ms;
    float diag_preprocessing_ms = 0.0f;
    float diag_compression_ms   = 0.0f;
    float diag_exploration_ms   = 0.0f;
    float diag_sgd_ms           = 0.0f;
    double diag_ratio_mape       = 0.0;
    double diag_comp_time_mape   = 0.0;
    double diag_decomp_time_mape = 0.0;
    double diag_regret           = -1.0;  /* -1 = no exploration, no regret available */

    gpucompress_algorithm_t algo_to_use;
    unsigned int preproc_to_use = 0;

    /* Decode action to algorithm + preprocessing */
    gpucompress::DecodedAction decoded = gpucompress::decodeAction(action);
    algo_to_use = static_cast<gpucompress_algorithm_t>(decoded.algorithm + 1);
    if (decoded.shuffle_size > 0)
        preproc_to_use |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
    if (decoded.use_quantization)
        preproc_to_use |= GPUCOMPRESS_PREPROC_QUANTIZE;

    g_last_nn_action.store(action);

    /* ---- Preprocessing (P1: uses pre-allocated buffers from CompContext) ---- */
    const uint8_t* d_compress_input = static_cast<const uint8_t*>(d_input);
    size_t compress_input_size = input_size;
    uint8_t* d_quantized = nullptr;
    uint8_t* d_shuffled  = nullptr;
    bool owns_quantized = true, owns_shuffled = true;
    QuantizationResult quant_result;

    auto t_preproc_start = std::chrono::steady_clock::now();

    if ((preproc_to_use & GPUCOMPRESS_PREPROC_QUANTIZE) && cfg.error_bound > 0.0) {
        size_t num_elements = input_size / sizeof(float);
        if (num_elements > 0) {
            QuantizationConfig quant_cfg(
                QuantizationType::LINEAR, cfg.error_bound,
                num_elements, sizeof(float));
            quant_result = quantize_simple(
                const_cast<uint8_t*>(d_compress_input), num_elements, sizeof(float), quant_cfg,
                ctx->d_range_min, ctx->d_range_max,
                ctx->d_preproc_quant, ctx->preproc_quant_cap,
                ctx->d_cub_temp, ctx->cub_temp_cap,
                stream);
            if (quant_result.isValid()) {
                d_quantized = static_cast<uint8_t*>(quant_result.d_quantized);
                owns_quantized = quant_result.owns_output;
                d_compress_input = d_quantized;
                compress_input_size = quant_result.quantized_bytes;
            } else {
                return GPUCOMPRESS_ERROR_COMPRESSION;
            }
        }
    }

    unsigned int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(preproc_to_use);
    if (shuffle_size > 0) {
        d_shuffled = byte_shuffle_simple(
            const_cast<uint8_t*>(d_compress_input), compress_input_size,
            shuffle_size, gpucompress::SHUFFLE_CHUNK_SIZE, stream,
            ctx->d_preproc_shuffle, ctx->preproc_shuffle_cap,
            &owns_shuffled);
        if (d_shuffled) {
            d_compress_input = d_shuffled;
        } else {
            if (d_quantized && owns_quantized) cudaFree(d_quantized);
            return GPUCOMPRESS_ERROR_COMPRESSION;
        }
    }

    cudaError_t cuda_err = cudaSuccess;

    if (d_quantized || d_shuffled) {
        cudaStreamSynchronize(stream);  /* ensure preprocessing kernels complete before timing */
        diag_preprocessing_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_preproc_start).count();
    }

    /* ---- Compression ---- */
    DT_START(_dt_mgr);
    CompressionAlgorithm internal_algo = gpucompress::toInternalAlgorithm(algo_to_use);
    auto* compressor = gpucompress::getOrCreateCompManager(ctx, internal_algo);
    float dt_mgr_acquire = DT_MS(_dt_mgr);
    if (!compressor) {
        if (d_quantized && owns_quantized) cudaFree(d_quantized);
        if (d_shuffled && owns_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    DT_START(_dt_cfg);
    CompressionConfig comp_config = compressor->configure_compression(compress_input_size);
    float dt_configure_comp = DT_MS(_dt_cfg);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;

    if (max_compressed_size == 0) {
        if (d_quantized && owns_quantized) cudaFree(d_quantized);
        if (d_shuffled && owns_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    size_t header_size = GPUCOMPRESS_HEADER_SIZE;
    size_t total_max_needed = header_size + max_compressed_size;
    if (total_max_needed < header_size) {
        if (d_quantized && owns_quantized) cudaFree(d_quantized);
        if (d_shuffled && owns_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    uint8_t* d_out      = static_cast<uint8_t*>(d_output);
    uint8_t* d_temp_out = nullptr;
    uint8_t* d_comp_target;

    float dt_temp_alloc = 0.0f;
    if (total_max_needed > *output_size) {
        DT_START(_dt_ta);
        if (cudaMalloc(&d_temp_out, total_max_needed) != cudaSuccess) {
            if (d_quantized && owns_quantized) cudaFree(d_quantized);
            if (d_shuffled && owns_shuffled)   cudaFree(d_shuffled);
            return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
        }
        d_comp_target = d_temp_out + header_size;
        dt_temp_alloc = DT_MS(_dt_ta);
    } else {
        d_comp_target = d_out + header_size;
    }

    float primary_comp_time_ms = 0.0f;
    bool need_timing = timing_ok && (stats != nullptr || g_online_learning_enabled);
    if (need_timing) cudaEventRecord(ctx->t_start, stream);

    DT_START(_dt_launch);
    try {
        compressor->compress(d_compress_input, d_comp_target, comp_config);
    } catch (const std::exception& e) {
        if (d_temp_out)  cudaFree(d_temp_out);
        if (d_quantized && owns_quantized) cudaFree(d_quantized);
        if (d_shuffled && owns_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    } catch (...) {
        if (d_temp_out)  cudaFree(d_temp_out);
        if (d_quantized && owns_quantized) cudaFree(d_quantized);
        if (d_shuffled && owns_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    float dt_compress_launch = DT_MS(_dt_launch);

    if (need_timing) {
        cudaEventRecord(ctx->t_stop, stream);
    }
    /* Always sync the stream so that get_compressed_output_size() can safely
     * read the compressed output metadata.  cudaEventSynchronize alone is
     * insufficient when nvcomp's get_compressed_output_size() does an
     * internal cudaMemcpy on a different (default) stream — the event sync
     * blocks the host but does not establish cross-stream memory visibility
     * in per-thread default-stream mode. */
    DT_START(_dt_sync);
    cudaStreamSynchronize(stream);
    float dt_stream_sync = DT_MS(_dt_sync);
    if (need_timing) {
        cudaEventElapsedTime(&primary_comp_time_ms, ctx->t_start, ctx->t_stop);
    }
    diag_compression_ms = std::max(1.0f, primary_comp_time_ms);  /* clamped for MAPE */
    float diag_compression_ms_raw = primary_comp_time_ms;       /* unclamped for breakdown */

    DT_START(_dt_gcs);
    size_t compressed_size = compressor->get_compressed_output_size(d_comp_target);
    float dt_get_comp_size = DT_MS(_dt_gcs);
    size_t total_size      = header_size + compressed_size;

    float primary_decomp_time_ms = 0.0f;

    /* Build header */
    CompressionHeader header;
    header.magic               = COMPRESSION_MAGIC;
    header.version             = COMPRESSION_HEADER_VERSION;
    header.shuffle_element_size = shuffle_size;
    header.original_size       = input_size;
    header.compressed_size     = compressed_size;
    if (d_quantized && quant_result.isValid()) {
        header.setQuantizationFlags(
            static_cast<uint32_t>(quant_result.type),
            quant_result.actual_precision, true);
        header.quant_error_bound = quant_result.error_bound;
        header.quant_scale       = quant_result.scale_factor;
        header.data_min          = quant_result.data_min;
        header.data_max          = quant_result.data_max;
    } else {
        header.quant_flags       = 0;
        header.quant_error_bound = 0.0;
        header.quant_scale       = 0.0;
        header.data_min          = 0.0;
        header.data_max          = 0.0;
    }
    header.setAlgorithmId((uint8_t)algo_to_use);

    DT_START(_dt_hdr);
    if (d_temp_out) {
        cuda_err = cudaMemcpyAsync(d_temp_out, &header, sizeof(CompressionHeader),
                                   cudaMemcpyHostToDevice, stream);
        if (cuda_err == cudaSuccess) {
            cuda_err = cudaMemcpyAsync(d_out, d_temp_out, total_size,
                                       cudaMemcpyDeviceToDevice, stream);
        }
        if (cuda_err == cudaSuccess)
            cuda_err = cudaStreamSynchronize(stream);
        cudaFree(d_temp_out);
    } else {
        cuda_err = cudaMemcpyAsync(d_out, &header, sizeof(CompressionHeader),
                                   cudaMemcpyHostToDevice, stream);
    }
    float dt_header_write = DT_MS(_dt_hdr);

    /* P1: only free if we allocated (not pre-allocated from CompContext) */
    if (d_quantized && owns_quantized) { cudaFree(d_quantized); d_quantized = nullptr; }
    if (d_shuffled && owns_shuffled)   { cudaFree(d_shuffled);  d_shuffled  = nullptr; }

    if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;

    size_t output_capacity = *output_size;
    *output_size = total_size;

    /* ---- Exploration + SGD (same as gpucompress_compress_gpu) ---- */
    /* We need stats for SGD — run stats on this ctx for the SGD/exploration phase */
    auto t_explore_start = std::chrono::steady_clock::now();

    /* Copy pre-computed stats for diagnostics (always) and SGD (when learning enabled). */
    DT_START(_dt_sc);
    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && d_precomputed_stats) {
        cuda_err = cudaMemcpyAsync(ctx->d_stats, d_precomputed_stats, sizeof(AutoStatsGPU),
                                   cudaMemcpyDeviceToDevice, stream);
        if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;
        cuda_err = cudaStreamSynchronize(stream);
        if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;
        d_stats_ptr = ctx->d_stats;
    }
    float dt_stats_copy = DT_MS(_dt_sc);

    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && g_online_learning_enabled) {

        double actual_ratio = static_cast<double>(input_size) /
                              static_cast<double>(compressed_size);

        /* Save primary algorithm's actual metrics BEFORE exploration may overwrite them.
         * These are used for MAPE reporting in chunk diagnostics so that prediction
         * accuracy reflects the NN's chosen algorithm, not the exploration winner. */
        primary_compressed_size = compressed_size;  /* save before exploration overwrites */

        /* Save primary's actual PSNR before exploration may overwrite d_quantized/quant_result.
         * Lossless primary → PSNR undefined (-1.0 sentinel, MAPE skips it).
         * Lossy primary → analytical PSNR from quantization range. */
        if (d_quantized && quant_result.isValid()) {
            double range = quant_result.data_max - quant_result.data_min;
            double psnr = analytical_psnr(range, quant_result.error_bound);
            if (psnr > 0.0)
                primary_actual_psnr = static_cast<float>(psnr);
        }

        double ds = static_cast<double>(input_size);
        double bw = static_cast<double>(g_measured_bw_bytes_per_ms);
        double w0 = static_cast<double>(g_rank_w0);
        double w1 = static_cast<double>(g_rank_w1);
        double w2 = static_cast<double>(g_rank_w2);
        /* Policy clamps: ct/dt floor at 1ms, ratio cap at 100x. */
        double pred_dt = std::max(1.0, static_cast<double>(predicted_decomp_time));
        double pred_ct = std::max(1.0, static_cast<double>(predicted_comp_time));
        double pred_r  = std::min(100.0, static_cast<double>(predicted_ratio));
        double act_ct  = std::max(1.0, static_cast<double>(primary_comp_time_ms));
        double act_dt  = (primary_decomp_time_ms > 0.0f)
                       ? std::max(1.0, static_cast<double>(primary_decomp_time_ms)) : pred_dt;
        double act_r   = std::min(100.0, actual_ratio);

        actual_cost = w0 * act_ct + w1 * act_dt
                    + ((act_r > 0.0) ? w2 * ds / (act_r * bw) : 0.0);
        predicted_cost = w0 * pred_ct + w1 * pred_dt
                       + ((pred_r > 0.0) ? w2 * ds / (pred_r * bw) : 0.0);
        error_pct = (actual_cost > 0.0)
            ? std::abs(actual_cost - predicted_cost) / actual_cost : 0.0;

        /* Per-statistic MAPE (clamped values) */
        diag_ratio_mape      = (act_r > 0.0)   ? std::abs(act_r   - pred_r)   / act_r   : 0.0;
        diag_comp_time_mape  = (act_ct > 0.0)  ? std::abs(act_ct  - pred_ct)  / act_ct  : 0.0;
        diag_decomp_time_mape= (act_dt > 0.0)  ? std::abs(act_dt  - pred_dt)  / act_dt  : 0.0;

        // cost = w0*ct + w1*dt + w2*ds/(ratio*bw) — same formula as NN ranking
        // Policy clamps applied inside: ct/dt floor 1ms, ratio cap 100x.
        auto compute_cost = [&](double ct, double dt, double r) -> double {
            double c = std::max(1.0, ct), d = std::max(1.0, dt);
            double rc = std::min(100.0, r);
            return w0 * c + w1 * d + ((rc > 0.0) ? w2 * ds / (rc * bw) : 1e30);
        };
        double primary_cost = compute_cost(
            static_cast<double>(primary_comp_time_ms),
            (primary_decomp_time_ms > 0.0f) ? static_cast<double>(primary_decomp_time_ms) : pred_dt,
            actual_ratio);
        explored_samples.push_back({nn_action, actual_ratio,
            static_cast<double>(primary_comp_time_ms),
            (primary_decomp_time_ms > 0.0f) ? static_cast<double>(primary_decomp_time_ms) : 0.0,
            120.0, primary_cost});

        /* ── SGD Phase 1: learn from PRIMARY result immediately ── */
        float sgd_phase1_ms = 0.0f, sgd_phase2_ms = 0.0f;
        auto t_sgd1_start = std::chrono::steady_clock::now();
        if (error_pct > static_cast<double>(g_reinforce_mape_threshold) && d_stats_ptr && !g_best_mode.load()) {
            SGDSample primary_sgd[1];
            primary_sgd[0].action            = explored_samples[0].action;
            primary_sgd[0].actual_ratio      = static_cast<float>(explored_samples[0].ratio);
            primary_sgd[0].actual_comp_time  = static_cast<float>(explored_samples[0].comp_time_ms);
            primary_sgd[0].actual_decomp_time= static_cast<float>(explored_samples[0].decomp_time_ms);
            primary_sgd[0].actual_psnr       = static_cast<float>(explored_samples[0].psnr);
            if (d_quantized && quant_result.isValid()) {
                double range = quant_result.data_max - quant_result.data_min;
                double psnr = analytical_psnr(range, quant_result.error_bound);
                if (psnr > 0.0)
                    primary_sgd[0].actual_psnr = static_cast<float>(psnr);
            }
            {
                std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                float gn = 0; int gc = 0, gs = 0;
                if (gpucompress::runNNSGDCtx(d_stats_ptr, primary_sgd, 1,
                        input_size, cfg.error_bound, g_reinforce_lr, ctx,
                        &gn, &gc, &gs) == 0) {
                    sgd_fired = true;
                }
            }
        }
        if (sgd_fired) {
            sgd_phase1_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_sgd1_start).count();
        }

        if (g_exploration_enabled && (g_best_mode.load() || error_pct > g_exploration_threshold) && top_actions) {
            exploration_triggered = true;
            int K;
            if (g_exploration_k_override > 0) {
                K = g_exploration_k_override;
            } else {
                K = 3;  // Top-3 alternatives (limits GPU memory pressure)
            }

            double best_cost = primary_cost;

            /* ---- Parallel exploration: K alternatives on K separate streams ----
             * Phase 1: launch all K compressions in parallel.
             * Phase 2: sync all, read sizes, pick winner by cost.
             * Phase 3: cleanup per-slot resources. */

            int actual_K = std::min(K, 31);

            struct ExploreSlot {
                int       action;
                int       algo;             /* gpucompress_algorithm_t */
                unsigned  preproc;
                unsigned  shuf_size;
                bool      has_quant;
                QuantizationResult quant_result;
                cudaStream_t       stream;
                cudaEvent_t        ev_start;        /* compression timing */
                cudaEvent_t        ev_stop;
                uint8_t*  d_out;            /* compressed output */
                uint8_t*  d_quant;          /* quantized buffer (owned) */
                uint8_t*  d_shuf;           /* shuffled buffer (owned) */
                void*     d_range_min;      /* per-slot quantization range */
                void*     d_range_max;
                std::unique_ptr<nvcomp::nvcompManagerBase> comp_mgr;
                size_t    comp_size;        /* filled after sync */
                double    ratio;
                float     comp_time_ms;     /* measured compression time */
                bool      valid;            /* compression succeeded */
            };

            ExploreSlot slots[31];  /* max possible: 32 configs - 1 primary */
            int n_slots = 0;
            size_t explore_max_comp = gpucompress_max_compressed_size(input_size);

            /* ---- Phase 1: prepare and launch K compressions in parallel ---- */
            for (int i = 1; i <= actual_K && i < 32; i++) {
                int alt_action = top_actions[i];
                if (alt_action == nn_action) continue;

                gpucompress::DecodedAction alt = gpucompress::decodeAction(alt_action);
                if (alt.use_quantization && cfg.error_bound <= 0.0) continue;

                ExploreSlot& s = slots[n_slots];
                memset(&s, 0, sizeof(s));
                s.action    = alt_action;
                s.algo      = alt.algorithm + 1;
                s.shuf_size = (alt.shuffle_size > 0) ? 4 : 0;
                s.has_quant = alt.use_quantization;
                s.preproc   = 0;
                if (s.shuf_size > 0) s.preproc |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                if (s.has_quant)     s.preproc |= GPUCOMPRESS_PREPROC_QUANTIZE;
                s.valid = false;

                /* Create per-slot stream, timing events, and output buffer */
                if (cudaStreamCreate(&s.stream) != cudaSuccess) {
                    fprintf(stderr, "gpucompress ERROR: exploration stream create failed\n");
                    cuda_err = cudaErrorUnknown; break;
                }
                cudaEventCreate(&s.ev_start);
                cudaEventCreate(&s.ev_stop);
                if (cudaMalloc(&s.d_out, explore_max_comp) != cudaSuccess) {
                    fprintf(stderr, "gpucompress ERROR: exploration output alloc failed\n");
                    cudaStreamDestroy(s.stream); s.stream = nullptr;
                    cuda_err = cudaErrorUnknown; break;
                }

                /* Per-slot quantization range buffers (8B each) */
                if (s.has_quant) {
                    if (cudaMalloc(&s.d_range_min, sizeof(double)) != cudaSuccess ||
                        cudaMalloc(&s.d_range_max, sizeof(double)) != cudaSuccess) {
                        fprintf(stderr, "gpucompress ERROR: exploration range buf alloc failed\n");
                        cudaFree(s.d_out); s.d_out = nullptr;
                        cudaStreamDestroy(s.stream); s.stream = nullptr;
                        if (s.d_range_min) { cudaFree(s.d_range_min); s.d_range_min = nullptr; }
                        cuda_err = cudaErrorUnknown; break;
                    }
                }

                /* Preprocessing on per-slot stream */
                uint8_t* d_alt_input = const_cast<uint8_t*>(
                    static_cast<const uint8_t*>(d_input));
                size_t alt_compress_size = input_size;

                if (s.has_quant && cfg.error_bound > 0.0) {
                    size_t num_el = input_size / sizeof(float);
                    if (num_el > 0) {
                        QuantizationConfig qcfg(QuantizationType::LINEAR,
                            cfg.error_bound, num_el, sizeof(float));
                        s.quant_result = quantize_simple(d_alt_input, num_el,
                            sizeof(float), qcfg, s.d_range_min, s.d_range_max,
                            s.stream);
                        if (s.quant_result.isValid()) {
                            s.d_quant = static_cast<uint8_t*>(s.quant_result.d_quantized);
                            d_alt_input = s.d_quant;
                            alt_compress_size = s.quant_result.quantized_bytes;
                        }
                    }
                }

                if (s.shuf_size > 0) {
                    uint8_t* shuf = byte_shuffle_simple(d_alt_input,
                        alt_compress_size, s.shuf_size,
                        gpucompress::SHUFFLE_CHUNK_SIZE, s.stream);
                    if (shuf) { s.d_shuf = shuf; d_alt_input = shuf; }
                    else {
                        fprintf(stderr, "gpucompress ERROR: exploration shuffle failed "
                                "(action=%d, size=%zu)\n", alt_action, alt_compress_size);
                        n_slots++;  /* include in Phase 3 cleanup */
                        /* Phase 3 will clean up all slots; return error after cleanup */
                        cuda_err = cudaErrorUnknown;
                        break;
                    }
                }

                /* Create fresh compression manager on per-slot stream */
                CompressionAlgorithm alt_internal = gpucompress::toInternalAlgorithm(
                    static_cast<gpucompress_algorithm_t>(s.algo));
                s.comp_mgr = createCompressionManager(alt_internal,
                    gpucompress::DEFAULT_CHUNK_SIZE, s.stream, nullptr);

                if (s.comp_mgr) {
                    try {
                        CompressionConfig alt_cc =
                            s.comp_mgr->configure_compression(alt_compress_size);
                        if (alt_cc.max_compressed_buffer_size <= explore_max_comp) {
                            cudaEventRecord(s.ev_start, s.stream);
                            s.comp_mgr->compress(d_alt_input, s.d_out, alt_cc);
                            cudaEventRecord(s.ev_stop, s.stream);
                            s.valid = true;
                        }
                    } catch (...) { /* compression failed, s.valid stays false */ }
                }

                n_slots++;
            }

            /* ---- Phase 2: sync all streams, read sizes, pick winner ---- */
            for (int k = 0; k < n_slots; k++) {
                ExploreSlot& s = slots[k];
                if (!s.valid) continue;

                cudaStreamSynchronize(s.stream);
                s.comp_size = s.comp_mgr->get_compressed_output_size(s.d_out);
                s.ratio = static_cast<double>(input_size) /
                          static_cast<double>(s.comp_size);

                /* Measured compression time from per-slot CUDA events */
                s.comp_time_ms = 0.0f;
                cudaEventElapsedTime(&s.comp_time_ms, s.ev_start, s.ev_stop);

                /* PSNR: analytical from error bound (no D->H needed).
                 * For non-quantized actions, use -1.0 as sentinel — the SGD
                 * kernel skips PSNR gradient when actual_psnr < 0. Using 0.0
                 * doesn't work because nn_gpu.cu:689 remaps <=0 to 120. */
                double alt_psnr = -1.0;
                if (s.has_quant && s.quant_result.isValid()) {
                    double range = s.quant_result.data_max - s.quant_result.data_min;
                    alt_psnr = analytical_psnr(range, s.quant_result.error_bound);
                }

                /* Use measured comp_time; decomp_time uses NN prediction (no decomp at write) */
                double alt_cost = compute_cost(
                    static_cast<double>(s.comp_time_ms), pred_dt, s.ratio);
                explored_samples.push_back({s.action, s.ratio,
                    static_cast<double>(s.comp_time_ms), 0.0, alt_psnr, alt_cost});

                if (alt_cost < best_cost) {
                    best_cost = alt_cost;

                    /* Write winner to output */
                    size_t hdr_sz = GPUCOMPRESS_HEADER_SIZE;
                    size_t alt_total = hdr_sz + s.comp_size;
                    if (alt_total <= output_capacity) {
                        nn_action = s.action;
                        g_last_nn_action.store(s.action);
                        CompressionHeader alt_hdr;
                        alt_hdr.magic = COMPRESSION_MAGIC;
                        alt_hdr.version = COMPRESSION_HEADER_VERSION;
                        alt_hdr.shuffle_element_size = s.shuf_size;
                        alt_hdr.original_size = input_size;
                        alt_hdr.compressed_size = s.comp_size;

                        if (s.d_quant && s.quant_result.isValid()) {
                            alt_hdr.setQuantizationFlags(
                                static_cast<uint32_t>(s.quant_result.type),
                                s.quant_result.actual_precision, true);
                            alt_hdr.quant_error_bound = s.quant_result.error_bound;
                            alt_hdr.quant_scale = s.quant_result.scale_factor;
                            alt_hdr.data_min = s.quant_result.data_min;
                            alt_hdr.data_max = s.quant_result.data_max;
                        } else {
                            alt_hdr.quant_flags = 0;
                            alt_hdr.quant_error_bound = 0.0;
                            alt_hdr.quant_scale = 0.0;
                            alt_hdr.data_min = 0.0;
                            alt_hdr.data_max = 0.0;
                        }
                        alt_hdr.setAlgorithmId((uint8_t)s.algo);

                        /* Copy winner header to output. Use synchronous copy because
                         * alt_hdr is stack-allocated and goes out of scope before
                         * cudaStreamSynchronize — async would read freed stack memory
                         * if multiple winners are found in the loop. */
                        cudaMemcpy(d_out, &alt_hdr, sizeof(CompressionHeader),
                                   cudaMemcpyHostToDevice);
                        cudaMemcpyAsync(d_out + hdr_sz, s.d_out, s.comp_size,
                                        cudaMemcpyDeviceToDevice, stream);
                        *output_size = alt_total;
                        total_size = alt_total;
                        compressed_size = s.comp_size;
                        actual_ratio = s.ratio;
                        algo_to_use = static_cast<gpucompress_algorithm_t>(s.algo);
                        preproc_to_use = s.preproc;
                        /* Update quant state so diagnostics record the winner's PSNR.
                         * Only quant_result (host struct) matters — d_quantized is
                         * used purely as a boolean flag, not dereferenced. */
                        if (s.d_quant && s.quant_result.isValid()) {
                            d_quantized = s.d_quant;
                            quant_result = s.quant_result;
                        } else {
                            d_quantized = nullptr;
                            quant_result = {};
                        }
                    }
                }
            }

            /* Drain stream before freeing slot buffers — the winner copy
             * (cudaMemcpyAsync from s.d_out) may still be in-flight. */
            cudaStreamSynchronize(stream);

            /* ---- Phase 3: cleanup per-slot resources ---- */
            for (int k = 0; k < n_slots; k++) {
                ExploreSlot& s = slots[k];
                s.comp_mgr.reset();  /* release manager before destroying stream */
                if (s.d_quant)     cudaFree(s.d_quant);
                if (s.d_shuf)      cudaFree(s.d_shuf);
                if (s.d_out)       cudaFree(s.d_out);
                if (s.d_range_min) cudaFree(s.d_range_min);
                if (s.d_range_max) cudaFree(s.d_range_max);
                if (s.ev_start)    cudaEventDestroy(s.ev_start);
                if (s.ev_stop)     cudaEventDestroy(s.ev_stop);
                if (s.stream)      cudaStreamDestroy(s.stream);
            }

            /* Regret: how much worse was the primary vs. the best explored config?
             * regret = (primary_cost - best_cost) / best_cost  (0 if primary was optimal) */
            if (best_cost > 0.0)
                diag_regret = (primary_cost - best_cost) / best_cost;

            if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_COMPRESSION;
        }

        if (exploration_triggered) {
            diag_exploration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_explore_start).count();
        }

        /* ── SGD Phase 2: learn from EXPLORATION results separately ── */
        auto t_sgd2_start = std::chrono::steady_clock::now();
        if (exploration_triggered && explored_samples.size() > 1 && d_stats_ptr && !g_best_mode.load()) {
            std::sort(explored_samples.begin() + 1, explored_samples.end(),
                      [](const ExploredResult& a, const ExploredResult& b) {
                          return a.cost < b.cost;
                      });
            SGDSample explore_sgd[NN_MAX_SGD_SAMPLES];
            int efill = 0;
            size_t emax = std::min(explored_samples.size(),
                                    static_cast<size_t>(NN_MAX_SGD_SAMPLES));
            for (size_t ei = 1; ei < emax; ei++) {
                explore_sgd[efill].action            = explored_samples[ei].action;
                explore_sgd[efill].actual_ratio      = static_cast<float>(explored_samples[ei].ratio);
                explore_sgd[efill].actual_comp_time  = static_cast<float>(explored_samples[ei].comp_time_ms);
                explore_sgd[efill].actual_decomp_time= static_cast<float>(explored_samples[ei].decomp_time_ms);
                explore_sgd[efill].actual_psnr       = static_cast<float>(explored_samples[ei].psnr);
                efill++;
            }
            if (efill > 0) {
                std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                float gn = 0; int gc = 0, gs = 0;
                int rc = gpucompress::runNNSGDCtx(d_stats_ptr, explore_sgd,
                    efill, input_size, cfg.error_bound, g_reinforce_lr, ctx,
                    &gn, &gc, &gs);
                if (rc == 0) sgd_fired = true;
            }
        }
        /* Bug fix: sgd_ms now measures Phase 1 + Phase 2 only,
         * excluding exploration time (was double-counted with explore_ms). */
        if (sgd_fired || exploration_triggered) {
            sgd_phase2_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_sgd2_start).count();
            diag_sgd_ms = sgd_phase1_ms + sgd_phase2_ms;
        }
    }

    /* Synchronize ctx->stream before releasing context */
    DT_START(_dt_fsync);
    cuda_err = cudaStreamSynchronize(stream);
    float dt_final_sync = DT_MS(_dt_fsync);
    if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;

    /* Fill stats */
    if (stats != nullptr) {
        stats->original_size = input_size;
        stats->compressed_size = total_size;
        stats->compression_ratio = static_cast<double>(input_size) /
            static_cast<double>(total_size > 0 ? total_size : 1);
        stats->algorithm_used = algo_to_use;
        stats->preprocessing_used = preproc_to_use;
        stats->throughput_mbps = (primary_comp_time_ms > 0.0f) ?
            (input_size / (1024.0 * 1024.0)) / (primary_comp_time_ms / 1000.0) : 0.0;
        stats->predicted_ratio = static_cast<double>(predicted_ratio);
        stats->predicted_comp_time_ms = static_cast<double>(predicted_comp_time);
        stats->predicted_decomp_time_ms = static_cast<double>(predicted_decomp_time);
        stats->predicted_psnr_db = static_cast<double>(predicted_psnr);
        stats->actual_comp_time_ms = static_cast<double>(primary_comp_time_ms);
        stats->sgd_fired = sgd_fired ? 1 : 0;
        stats->exploration_triggered = exploration_triggered ? 1 : 0;
        stats->nn_original_action = nn_original_action;
        stats->nn_final_action = nn_action;
    }

    /* Update globals */
    g_last_nn_action.store(nn_action);
    g_last_nn_original_action.store(nn_original_action);
    g_last_exploration_triggered.store(exploration_triggered ? 1 : 0);
    g_last_sgd_fired.store(sgd_fired ? 1 : 0);

    /* Append to per-chunk diagnostic history */
    {
        gpucompress::ChunkDiagInput di = {};
        di.nn_action = nn_action;
        di.nn_original_action = nn_original_action;
        di.exploration_triggered = exploration_triggered;
        di.sgd_fired = sgd_fired;
        di.nn_inference_ms = diag_nn_inference_ms;
        di.stats_ms = diag_stats_ms;
        di.preprocessing_ms = diag_preprocessing_ms;
        di.compression_ms = diag_compression_ms;
        di.compression_ms_raw = diag_compression_ms_raw;
        di.exploration_ms = diag_exploration_ms;
        di.sgd_ms = diag_sgd_ms;
        di.input_size = input_size;
        di.primary_compressed_size = primary_compressed_size;
        di.compressed_size = compressed_size;
        di.predicted_ratio = predicted_ratio;
        di.predicted_comp_time = predicted_comp_time;
        di.predicted_decomp_time = predicted_decomp_time;
        di.predicted_psnr = predicted_psnr;
        di.predicted_rmse = predicted_rmse;
        di.predicted_max_error = predicted_max_error;
        di.predicted_mae = predicted_mae;
        di.predicted_ssim = predicted_ssim;
        /* actual_psnr: uses PRIMARY algorithm's PSNR (for fair MAPE reporting).
         * The exploration winner's PSNR is computed from the post-swap
         * d_quantized/quant_result and stored in the public struct as actual_psnr
         * by recordChunkDiagnostic. */
        di.actual_psnr = primary_actual_psnr;
        di.error_bound = cfg.error_bound;
        di.d_stats_ptr = d_stats_ptr;
        /* P4 fix: copy stats features to host BEFORE entering the diagnostic
         * mutex.  The stream is already synced (line 814), so d_stats_ptr is
         * stable.  This eliminates a synchronous cudaMemcpy under lock. */
        if (d_stats_ptr) {
            AutoStatsGPU h_stats;
            if (cudaMemcpy(&h_stats, d_stats_ptr, sizeof(AutoStatsGPU),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                di.h_feat_entropy = static_cast<float>(h_stats.entropy);
                di.h_feat_mad     = static_cast<float>(h_stats.mad_normalized);
                di.h_feat_deriv   = static_cast<float>(h_stats.deriv_normalized);
                di.h_stats_valid  = true;
            }
        }
        /* Cost model diagnostics (only valid when online learning enabled) */
        if (g_online_learning_enabled) {
            di.cost_model_error_pct  = static_cast<float>(error_pct);
            di.actual_cost           = static_cast<float>(actual_cost);
            di.predicted_cost        = static_cast<float>(predicted_cost);
            di.ratio_mape            = static_cast<float>(diag_ratio_mape);
            di.comp_time_mape        = static_cast<float>(diag_comp_time_mape);
            di.decomp_time_mape      = static_cast<float>(diag_decomp_time_mape);
            di.regret                = static_cast<float>(diag_regret);
            /* Original config metrics (explored_samples[0] is the primary) */
            if (!explored_samples.empty()) {
                di.orig_actual_ratio = static_cast<float>(explored_samples[0].ratio);
                di.orig_comp_ms = static_cast<float>(explored_samples[0].comp_time_ms);
                di.orig_cost = static_cast<float>(explored_samples[0].cost);
            }
            /* Exploration alternatives */
            int n = std::min((int)explored_samples.size() - 1, 31);
            di.explore_n_alternatives = (n > 0) ? n : 0;
            for (int i = 0; i < n; i++) {
                auto& e = explored_samples[i + 1]; /* skip primary at index 0 */
                di.explore_alternatives[i] = e.action;
                di.explore_ratios[i] = static_cast<float>(e.ratio);
                di.explore_comp_ms[i] = static_cast<float>(e.comp_time_ms);
                di.explore_costs[i] = static_cast<float>(e.cost);
            }
        }
        /* NN predicted ranking and costs (all 32 configs) */
        di.top_actions = top_actions;
        di.top_actions_count = top_actions ? 32 : 0;
        di.predicted_costs = predicted_costs;
        /* Detailed timing fields */
        di.ctx_acquire_ms     = dt_ctx_acquire;
        di.mgr_acquire_ms     = dt_mgr_acquire;
        di.configure_comp_ms  = dt_configure_comp;
        di.temp_alloc_ms      = dt_temp_alloc;
        di.compress_launch_ms = dt_compress_launch;
        di.stream_sync_ms     = dt_stream_sync;
        di.get_comp_size_ms   = dt_get_comp_size;
        di.header_write_ms    = dt_header_write;
        di.stats_copy_ms      = dt_stats_copy;
        di.final_sync_ms      = dt_final_sync;

        DT_START(_dt_diag);
        int diag_slot = gpucompress::recordChunkDiagnostic(di);
        float dt_diag_record = DT_MS(_dt_diag);
        if (g_detailed_timing && diag_slot >= 0)
            gpucompress::DiagnosticsStore::instance().setDiagRecordMs(diag_slot, dt_diag_record);
        /* Propagate slot index to caller for VOL timing writeback */
        if (out_diag_slot) *out_diag_slot = diag_slot;
    }

    if (caller_stream) {
        cudaEventRecord(ctx->t_stop, stream);
        cudaStreamWaitEvent(caller_stream, ctx->t_stop, 0);
    }

    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_decompress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    void* stream_arg
) {
    if (!g_initialized.load()) return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    if (!d_input || !d_output || !output_size) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (input_size < GPUCOMPRESS_HEADER_SIZE) return GPUCOMPRESS_ERROR_INVALID_HEADER;

    cudaStream_t stream = stream_arg ? static_cast<cudaStream_t>(stream_arg) : g_default_stream;

    /* Read header from GPU (64B D→H) */
    CompressionHeader header;
    cudaError_t cuda_err = readHeaderFromDevice(d_input, header, stream);
    if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    if (!header.isValid())       return GPUCOMPRESS_ERROR_INVALID_HEADER;

    if (header.original_size > *output_size) {
        *output_size = header.original_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    size_t compressed_size = header.compressed_size;
    size_t hdr_plus_comp = GPUCOMPRESS_HEADER_SIZE + compressed_size;
    if (hdr_plus_comp < GPUCOMPRESS_HEADER_SIZE || input_size < hdr_plus_comp)
        return GPUCOMPRESS_ERROR_INVALID_HEADER;

    const uint8_t* d_compressed_data =
        static_cast<const uint8_t*>(d_input) + GPUCOMPRESS_HEADER_SIZE;

    /* nvcomp decompression */
    auto decompressor = createDecompressionManager(d_compressed_data, stream);
    if (!decompressor) return GPUCOMPRESS_ERROR_DECOMPRESSION;

    DecompressionConfig decomp_config =
        decompressor->configure_decompression(d_compressed_data);
    size_t decompressed_size = decomp_config.decomp_data_size;

    uint8_t* d_decompressed = nullptr;
    if (cudaMalloc(&d_decompressed, decompressed_size) != cudaSuccess)
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;

    try {
        decompressor->decompress(d_decompressed, d_compressed_data, decomp_config);
    } catch (...) {
        cudaFree(d_decompressed);
        return GPUCOMPRESS_ERROR_DECOMPRESSION;
    }

    /* Reverse preprocessing */
    uint8_t* d_result    = d_decompressed;
    uint8_t* d_unshuffled = nullptr;

    if (header.hasShuffleApplied()) {
        d_unshuffled = byte_unshuffle_simple(
            d_decompressed, decompressed_size,
            header.shuffle_element_size,
            gpucompress::SHUFFLE_CHUNK_SIZE, stream);
        if (d_unshuffled) {
            d_result = d_unshuffled;
        } else {
            fprintf(stderr, "gpucompress ERROR: byte unshuffle failed during GPU decompression "
                    "(element_size=%u, size=%zu)\n", header.shuffle_element_size, decompressed_size);
            cudaFree(d_decompressed);
            return GPUCOMPRESS_ERROR_DECOMPRESSION;
        }
    }

    void* d_dequantized = nullptr;
    if (header.hasQuantizationApplied()) {
        QuantizationResult qr;
        qr.scale_factor          = header.quant_scale;
        qr.data_min              = header.data_min;
        qr.data_max              = header.data_max;
        qr.error_bound           = header.quant_error_bound;
        qr.type                  = static_cast<QuantizationType>(header.getQuantizationType());
        qr.actual_precision      = header.getQuantizationPrecision();
        qr.num_elements          = header.original_size / sizeof(float);
        qr.original_element_size = sizeof(float);
        qr.d_quantized           = d_result;
        qr.quantized_bytes       = decompressed_size;
        d_dequantized = dequantize_simple(d_result, qr, stream);
        if (d_dequantized) {
            if (d_unshuffled) { cudaFree(d_unshuffled); d_unshuffled = nullptr; }
            d_result = static_cast<uint8_t*>(d_dequantized);
        } else {
            fprintf(stderr, "gpucompress ERROR: dequantization failed during GPU decompression "
                    "(precision=%d, num_elements=%zu)\n",
                    qr.actual_precision, qr.num_elements);
            if (d_unshuffled) cudaFree(d_unshuffled);
            cudaFree(d_decompressed);
            return GPUCOMPRESS_ERROR_DECOMPRESSION;
        }
    }

    /* D→D copy to caller's output buffer */
    cuda_err = cudaMemcpyAsync(d_output, d_result, header.original_size,
                               cudaMemcpyDeviceToDevice, stream);
    if (cuda_err == cudaSuccess)
        cuda_err = cudaStreamSynchronize(stream);

    if (d_dequantized) cudaFree(d_dequantized);
    if (d_unshuffled)  cudaFree(d_unshuffled);
    cudaFree(d_decompressed);

    if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;

    *output_size = header.original_size;
    return GPUCOMPRESS_SUCCESS;
}
