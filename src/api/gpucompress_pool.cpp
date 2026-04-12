/**
 * @file gpucompress_pool.cpp
 * @brief CompContext pool — per-slot GPU state for concurrent compression.
 */

#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <memory>

#include "gpucompress.h"
#include "api/internal.hpp"
#include "api/gpucompress_state.hpp"
#include "api/diagnostics_store.hpp"
#include "compression/compression_factory.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

/* ---- Pool-private state ---- */

static CompContext g_comp_pool[N_COMP_CTX];
static bool g_pool_slot_free[N_COMP_CTX];
static int  g_pool_free_count = 0;
static std::mutex g_pool_mutex;
static std::condition_variable g_pool_cv;

namespace gpucompress {

static constexpr size_t kPoolStatsWSZ =
    sizeof(AutoStatsGPU) + 256 * sizeof(unsigned int) + sizeof(int);

static void destroyCompContextSlot(CompContext& ctx) {
    if (ctx.stream)
        cudaStreamSynchronize(ctx.stream);
    for (int i = 0; i < CompContext::LRU_DEPTH; i++) {
        delete ctx.comp_mgr[i];
        ctx.comp_mgr[i]      = nullptr;
        ctx.comp_mgr_algo[i] = -1;
        ctx.comp_mgr_tick[i] = 0;
    }
    ctx.comp_mgr_clock = 0;

    if (ctx.stream)              { cudaStreamDestroy(ctx.stream); ctx.stream = nullptr; }
    if (ctx.t_start)             { cudaEventDestroy(ctx.t_start);     ctx.t_start     = nullptr; }
    if (ctx.t_stop)              { cudaEventDestroy(ctx.t_stop);      ctx.t_stop      = nullptr; }
    if (ctx.nn_start)            { cudaEventDestroy(ctx.nn_start);    ctx.nn_start    = nullptr; }
    if (ctx.nn_stop)             { cudaEventDestroy(ctx.nn_stop);     ctx.nn_stop     = nullptr; }
    if (ctx.stats_start)         { cudaEventDestroy(ctx.stats_start); ctx.stats_start = nullptr; }
    if (ctx.stats_stop)          { cudaEventDestroy(ctx.stats_stop);  ctx.stats_stop  = nullptr; }
    if (ctx.d_stats_workspace)   { cudaFree(ctx.d_stats_workspace);
                                   ctx.d_stats_workspace = nullptr;
                                   ctx.d_stats = nullptr; ctx.d_histogram = nullptr; }
    if (ctx.d_fused_infer_output){ cudaFree(ctx.d_fused_infer_output);
                                   ctx.d_fused_infer_output = nullptr; }
    if (ctx.d_fused_top_actions) { cudaFree(ctx.d_fused_top_actions);
                                   ctx.d_fused_top_actions = nullptr; }
    if (ctx.d_fused_costs)      { cudaFree(ctx.d_fused_costs);
                                   ctx.d_fused_costs = nullptr; }
    if (ctx.d_sgd_grad_buffer)   { cudaFree(ctx.d_sgd_grad_buffer);
                                   ctx.d_sgd_grad_buffer = nullptr; }
    if (ctx.d_sgd_output)        { cudaFree(ctx.d_sgd_output); ctx.d_sgd_output = nullptr; }
    if (ctx.d_sgd_samples)       { cudaFree(ctx.d_sgd_samples); ctx.d_sgd_samples = nullptr; }
    if (ctx.d_range_min)         { cudaFree(ctx.d_range_min); ctx.d_range_min = nullptr; }
    if (ctx.d_range_max)         { cudaFree(ctx.d_range_max); ctx.d_range_max = nullptr; }
    /* P1: preprocessing buffers */
    if (ctx.d_preproc_quant)     { cudaFree(ctx.d_preproc_quant); ctx.d_preproc_quant = nullptr; ctx.preproc_quant_cap = 0; }
    if (ctx.d_preproc_shuffle)   { cudaFree(ctx.d_preproc_shuffle); ctx.d_preproc_shuffle = nullptr; ctx.preproc_shuffle_cap = 0; }
    if (ctx.d_cub_temp)          { cudaFree(ctx.d_cub_temp); ctx.d_cub_temp = nullptr; ctx.cub_temp_cap = 0; }
}

int initCompContextPool() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    memset(g_comp_pool, 0, sizeof(g_comp_pool));
    for (int s = 0; s < N_COMP_CTX; s++)
        for (int d = 0; d < CompContext::LRU_DEPTH; d++)
            g_comp_pool[s].comp_mgr_algo[d] = -1;

    g_pool_free_count = 0;
    for (int i = 0; i < N_COMP_CTX; i++) {
        CompContext& ctx = g_comp_pool[i];
        ctx.slot_id = i;
        if (cudaStreamCreate(&ctx.stream)   != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.t_start)     != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.t_stop)      != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.nn_start)    != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.nn_stop)     != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.stats_start) != cudaSuccess) goto fail;
        if (cudaEventCreate (&ctx.stats_stop)  != cudaSuccess) goto fail;

        if (cudaMalloc(&ctx.d_stats_workspace, kPoolStatsWSZ) != cudaSuccess) goto fail;
        ctx.d_stats     = static_cast<AutoStatsGPU*>(ctx.d_stats_workspace);
        ctx.d_histogram = reinterpret_cast<unsigned int*>(
            static_cast<uint8_t*>(ctx.d_stats_workspace) + sizeof(AutoStatsGPU));

        if (cudaMalloc(&ctx.d_fused_infer_output,
                       sizeof(NNInferenceOutput)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_fused_top_actions,
                       NN_NUM_CONFIGS * sizeof(int)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_fused_costs,
                       NN_NUM_CONFIGS * sizeof(float)) != cudaSuccess) goto fail;

        if (cudaMalloc(&ctx.d_sgd_grad_buffer,
                       NN_SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess) goto fail;
        cudaMemset(ctx.d_sgd_grad_buffer, 0, NN_SGD_GRAD_SIZE * sizeof(float));
        if (cudaMalloc(&ctx.d_sgd_output, sizeof(SGDOutput)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_sgd_samples,
                       NN_MAX_SGD_SAMPLES * sizeof(SGDSample)) != cudaSuccess) goto fail;

        if (cudaMalloc(&ctx.d_range_min, sizeof(double)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_range_max, sizeof(double)) != cudaSuccess) goto fail;

        /* P1: Pre-allocate preprocessing buffers.
         * 16MB covers the typical max chunk size. Quantized output may be
         * smaller (int8/int16) but we allocate for worst case (1:1).
         * CUB temp is typically a few KB — 64KB is generous. */
        {
            static constexpr size_t kPreallocChunk = 16UL << 20;  /* 16 MB */
            static constexpr size_t kCubTemp       = 64UL << 10;  /* 64 KB */
            if (cudaMalloc(&ctx.d_preproc_quant, kPreallocChunk) != cudaSuccess) goto fail;
            ctx.preproc_quant_cap = kPreallocChunk;
            if (cudaMalloc(&ctx.d_preproc_shuffle, kPreallocChunk) != cudaSuccess) goto fail;
            ctx.preproc_shuffle_cap = kPreallocChunk;
            if (cudaMalloc(&ctx.d_cub_temp, kCubTemp) != cudaSuccess) goto fail;
            ctx.cub_temp_cap = kCubTemp;
        }

        g_pool_slot_free[i] = true;
        g_pool_free_count++;
    }
    return 0;

fail:
    for (int j = 0; j < N_COMP_CTX; j++) {
        destroyCompContextSlot(g_comp_pool[j]);
        g_pool_slot_free[j] = false;
    }
    g_pool_free_count = 0;
    return -1;
}

void flushCompManagerCache() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    for (int i = 0; i < N_COMP_CTX; i++) {
        CompContext& ctx = g_comp_pool[i];
        if (ctx.stream) cudaStreamSynchronize(ctx.stream);
        for (int d = 0; d < CompContext::LRU_DEPTH; d++) {
            delete ctx.comp_mgr[d];
            ctx.comp_mgr[d]      = nullptr;
            ctx.comp_mgr_algo[d] = -1;
            ctx.comp_mgr_tick[d] = 0;
        }
        ctx.comp_mgr_clock = 0;
    }
}

void destroyCompContextPool() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    for (int i = 0; i < N_COMP_CTX; i++) {
        destroyCompContextSlot(g_comp_pool[i]);
        g_pool_slot_free[i] = false;
    }
    g_pool_free_count = 0;
}

CompContext* acquireCompContext() {
    std::unique_lock<std::mutex> lk(g_pool_mutex);
    g_pool_cv.wait(lk, []{ return g_pool_free_count > 0; });
    for (int i = 0; i < N_COMP_CTX; i++) {
        if (g_pool_slot_free[i]) {
            g_pool_slot_free[i] = false;
            g_pool_free_count--;
            return &g_comp_pool[i];
        }
    }
    return nullptr;
}

void releaseCompContext(CompContext* ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_pool_slot_free[ctx->slot_id] = true;
    g_pool_free_count++;
    g_pool_cv.notify_one();
}

void syncAllCompContextStreams() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    for (int i = 0; i < N_COMP_CTX; i++) {
        if (g_comp_pool[i].stream)
            cudaStreamSynchronize(g_comp_pool[i].stream);
    }
    if (::g_sgd_stream)
        cudaStreamSynchronize(::g_sgd_stream);
}

void resetAllSGDEMABuffers() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    /* Zero region 2 (EMA gradient) of every context's SGD grad buffer.
     * Called on weight reload to prevent stale EMA from a prior phase
     * contaminating the anti-flip damping of the new phase. */
    size_t ema_offset = 2 * NN_SGD_GRAD_REGION * sizeof(float);
    size_t ema_bytes  = NN_SGD_GRAD_REGION * sizeof(float);
    for (int i = 0; i < N_COMP_CTX; i++) {
        if (g_comp_pool[i].d_sgd_grad_buffer)
            cudaMemset(g_comp_pool[i].d_sgd_grad_buffer + 2 * NN_SGD_GRAD_REGION,
                       0, ema_bytes);
    }
}

std::unique_ptr<nvcomp::nvcompManagerBase> createCompManagerForCtx(CompContext* ctx, CompressionAlgorithm algo) {
    int idx = static_cast<int>(algo);
    if (idx < 0 || idx >= CompContext::N_COMP_ALGOS) return nullptr;
    gpucompress::DiagnosticsStore::instance().incrementCacheMiss();
    return createCompressionManager(algo, gpucompress::DEFAULT_CHUNK_SIZE, ctx->stream, nullptr);
}

nvcomp::nvcompManagerBase* getOrCreateCompManager(CompContext* ctx, CompressionAlgorithm algo) {
    int idx = static_cast<int>(algo);
    if (idx < 0 || idx >= CompContext::N_COMP_ALGOS) return nullptr;

    for (int i = 0; i < CompContext::LRU_DEPTH; i++) {
        if (ctx->comp_mgr[i] && ctx->comp_mgr_algo[i] == idx) {
            ctx->comp_mgr_tick[i] = ++ctx->comp_mgr_clock;
            gpucompress::DiagnosticsStore::instance().incrementCacheHit();
            return ctx->comp_mgr[i];
        }
    }

    int victim = -1;
    for (int i = 0; i < CompContext::LRU_DEPTH; i++) {
        if (!ctx->comp_mgr[i]) { victim = i; break; }
    }
    if (victim < 0) {
        int min_tick = ctx->comp_mgr_tick[0];
        victim = 0;
        for (int i = 1; i < CompContext::LRU_DEPTH; i++) {
            if (ctx->comp_mgr_tick[i] < min_tick) {
                min_tick = ctx->comp_mgr_tick[i];
                victim = i;
            }
        }
        delete ctx->comp_mgr[victim];
    }

    auto mgr = createCompressionManager(
        algo, gpucompress::DEFAULT_CHUNK_SIZE, ctx->stream, nullptr);
    ctx->comp_mgr[victim]      = mgr.release();
    ctx->comp_mgr_algo[victim] = idx;
    ctx->comp_mgr_tick[victim] = ++ctx->comp_mgr_clock;
    gpucompress::DiagnosticsStore::instance().incrementCacheMiss();
    return ctx->comp_mgr[victim];
}

} // namespace gpucompress
