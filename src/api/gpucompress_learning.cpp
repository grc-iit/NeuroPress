/**
 * @file gpucompress_learning.cpp
 * @brief Online learning controls, NN management, chunk history, SGD dispatch.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <vector>

#include "gpucompress.h"
#include "api/internal.hpp"
#include "api/gpucompress_state.hpp"
#include "nn/nn_weights.h"

extern "C" {
    int gpucompress_nn_load_impl(const char* filepath);
    int gpucompress_nn_is_loaded_impl(void);
}

/* ============================================================
 * NN Management
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_load_nn(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }
    int result = gpucompress_nn_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_INVALID_INPUT;
}

extern "C" int gpucompress_nn_is_loaded(void) {
    return gpucompress_nn_is_loaded_impl();
}

extern "C" gpucompress_error_t gpucompress_reload_nn(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }
    std::lock_guard<std::mutex> lock(g_init_mutex);
    int result = gpucompress_nn_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_INVALID_INPUT;
}

extern "C" size_t gpucompress_nn_weights_size(void) {
    return sizeof(NNWeightsGPU);
}

extern "C" gpucompress_error_t gpucompress_nn_save_weights(void* host_buffer, size_t buffer_size) {
    if (!host_buffer || buffer_size < sizeof(NNWeightsGPU))
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    const NNWeightsGPU* d_ptr = gpucompress::getNNWeightsDevicePtr();
    if (!d_ptr) return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
    cudaError_t err = cudaMemcpy(host_buffer, d_ptr, sizeof(NNWeightsGPU),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_CUDA_FAILED;
}

extern "C" gpucompress_error_t gpucompress_nn_restore_weights(const void* host_buffer, size_t buffer_size) {
    if (!host_buffer || buffer_size < sizeof(NNWeightsGPU))
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    const NNWeightsGPU* d_ptr = gpucompress::getNNWeightsDevicePtr();
    if (!d_ptr) return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
    cudaError_t err = cudaMemcpy(const_cast<NNWeightsGPU*>(d_ptr), host_buffer,
                                  sizeof(NNWeightsGPU), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_CUDA_FAILED;
}

/* ============================================================
 * Query Functions
 * ============================================================ */

extern "C" int gpucompress_get_last_nn_action(void) {
    return g_last_nn_action.load();
}

extern "C" int gpucompress_get_last_nn_original_action(void) {
    return g_last_nn_original_action.load();
}

extern "C" int gpucompress_get_last_exploration_triggered(void) {
    return g_last_exploration_triggered.load();
}

extern "C" int gpucompress_get_last_sgd_fired(void) {
    return g_last_sgd_fired.load();
}

extern "C" void gpucompress_reset_chunk_history(void) {
    g_chunk_history_count.store(0);
}

extern "C" void gpucompress_reset_cache_stats(void) {
    g_mgr_cache_hits.store(0);
    g_mgr_cache_misses.store(0);
}

extern "C" void gpucompress_get_cache_stats(int *hits, int *misses) {
    if (hits)   *hits   = g_mgr_cache_hits.load();
    if (misses) *misses = g_mgr_cache_misses.load();
}

extern "C" int gpucompress_get_chunk_history_count(void) {
    return g_chunk_history_count.load();
}

extern "C" int gpucompress_get_chunk_diag(int idx, gpucompress_chunk_diag_t *out) {
    if (idx < 0 || out == NULL)
        return -1;
    std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
    if (idx >= g_chunk_history_count.load() || idx >= g_chunk_history_cap)
        return -1;
    *out = g_chunk_history[idx];
    return 0;
}

extern "C" void gpucompress_record_chunk_decomp_ms(int idx, float ms) {
    std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
    if (idx >= 0 && idx < g_chunk_history_count.load() && idx < g_chunk_history_cap)
        g_chunk_history[idx].decompression_ms = ms;
}

/* ============================================================
 * Batched Decomp SGD
 * ============================================================ */

extern "C" void gpucompress_batched_decomp_sgd(void) {
    if (!g_online_learning_enabled) return;

    std::vector<DeferredDecompSample> batch;
    {
        std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
        int n = g_chunk_history_count.load();
        if (n > g_chunk_history_cap) n = g_chunk_history_cap;
        for (int i = 0; i < n; i++) {
            const gpucompress_chunk_diag_t& h = g_chunk_history[i];
            if (h.decompression_ms <= 0.0f) continue;
            if (h.feat_ds_enc <= 0.0f) continue;
            DeferredDecompSample s;
            s.action           = h.feat_action;
            s.entropy          = h.feat_entropy;
            s.mad_normalized   = h.feat_mad;
            s.deriv_normalized = h.feat_deriv;
            s.error_bound_enc  = h.feat_eb_enc;
            s.data_size_enc    = h.feat_ds_enc;
            s.actual_decomp_ms = h.decompression_ms;
            batch.push_back(s);
        }
    }
    if (batch.empty()) return;

    {
        std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
        gpucompress::runBatchedDecompSGD(batch.data(),
            static_cast<int>(batch.size()), g_reinforce_lr);
    }
}

/* ============================================================
 * Online Learning Controls
 * ============================================================ */

extern "C" void gpucompress_enable_online_learning(void) {
    g_online_learning_enabled = true;
}

extern "C" void gpucompress_disable_online_learning(void) {
    g_online_learning_enabled = false;
    g_exploration_enabled = false;
}

extern "C" int gpucompress_online_learning_enabled(void) {
    return g_online_learning_enabled ? 1 : 0;
}

extern "C" void gpucompress_set_exploration(int enable) {
    g_exploration_enabled = (enable != 0);
}

extern "C" gpucompress_error_t gpucompress_enable_active_learning(void) {
    gpucompress_enable_online_learning();
    g_exploration_enabled = true;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" void gpucompress_disable_active_learning(void) {
    gpucompress_disable_online_learning();
    g_exploration_enabled = false;
}

extern "C" int gpucompress_active_learning_enabled(void) {
    return g_online_learning_enabled ? 1 : 0;
}

extern "C" void gpucompress_set_exploration_threshold(double threshold) {
    if (threshold >= 0.0 && threshold < 1.0) {
        g_exploration_threshold = threshold;
    }
}

extern "C" void gpucompress_set_exploration_k(int k) {
    g_exploration_k_override = (k > 0 && k <= 31) ? k : -1;
}

extern "C" void gpucompress_set_reinforcement(int /*enable*/, float learning_rate,
                                               float mape_threshold,
                                               float /*ct_mape_threshold*/) {
    if (learning_rate > 0.0f) g_reinforce_lr = learning_rate;
    if (mape_threshold > 0.0f) g_reinforce_mape_threshold = mape_threshold;
}

extern "C" void gpucompress_set_ranking_weights(float w0, float w1, float w2) {
    g_rank_w0 = w0;
    g_rank_w1 = w1;
    g_rank_w2 = w2;
}

extern "C" void gpucompress_set_bandwidth(float bw_gbps) {
    if (bw_gbps > 0.0f)
        g_measured_bw_bytes_per_ms = bw_gbps * 1e6f;
}
