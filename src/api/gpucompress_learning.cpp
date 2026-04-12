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
#include "api/diagnostics_store.hpp"
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

/* ============================================================
 * Query Functions
 * ============================================================ */

extern "C" int gpucompress_get_last_nn_action(void) {
    return g_last_nn_action.load();
}

extern "C" int gpucompress_get_last_exploration_triggered(void) {
    return g_last_exploration_triggered.load();
}

extern "C" int gpucompress_get_last_sgd_fired(void) {
    return g_last_sgd_fired.load();
}


/* ============================================================
 * Batched Decomp SGD
 * ============================================================ */

extern "C" void gpucompress_batched_decomp_sgd(void) {
    if (!g_online_learning_enabled) return;

    std::vector<DeferredDecompSample> batch;
    {
        auto& store = gpucompress::DiagnosticsStore::instance();
        int n = store.count();
        for (int i = 0; i < n; i++) {
            gpucompress_chunk_diag_t h;
            if (store.getDiag(i, &h) != 0) continue;
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
 * Algorithm Selection Mode
 * ============================================================ */

extern "C" void gpucompress_set_selection_mode(gpucompress_selection_mode_t mode) {
    g_selection_mode.store(static_cast<int>(mode));
}

extern "C" gpucompress_selection_mode_t gpucompress_get_selection_mode(void) {
    return static_cast<gpucompress_selection_mode_t>(g_selection_mode.load());
}

/* ============================================================
 * Best (Exhaustive Search) Mode
 * ============================================================ */

static float s_saved_w0 = 1.0f, s_saved_w1 = 1.0f, s_saved_w2 = 1.0f;

extern "C" void gpucompress_set_best_mode(int enable) {
    g_best_mode.store(enable != 0);
    if (enable) {
        /* Force exploration on every chunk with all alternatives */
        g_exploration_enabled = true;
        g_exploration_k_override = 31;
        /* Save current weights and force ratio-only selection */
        s_saved_w0 = g_rank_w0;
        s_saved_w1 = g_rank_w1;
        s_saved_w2 = g_rank_w2;
        g_rank_w0 = 0.0f;
        g_rank_w1 = 0.0f;
        g_rank_w2 = 1.0f;
    } else {
        /* Restore defaults */
        g_exploration_enabled = false;
        g_exploration_k_override = -1;
        g_rank_w0 = s_saved_w0;
        g_rank_w1 = s_saved_w1;
        g_rank_w2 = s_saved_w2;
    }
}

extern "C" int gpucompress_best_mode_enabled(void) {
    return g_best_mode.load() ? 1 : 0;
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
    if (threshold >= 0.0) {
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

extern "C" void gpucompress_set_min_psnr(float min_psnr_db) {
    g_min_psnr_db = min_psnr_db;
}

extern "C" void gpucompress_set_bandwidth(float bw_gbps) {
    if (bw_gbps > 0.0f)
        g_measured_bw_bytes_per_ms = bw_gbps * 1e6f;
}

extern "C" float gpucompress_get_bandwidth_bytes_per_ms(void) {
    return g_measured_bw_bytes_per_ms;
}
