/**
 * @file gpucompress_diagnostics.cpp
 * @brief Consolidated diagnostics: chunk history, debug output, cache stats.
 *
 * All diagnostic/observability code lives here, keeping core logic files
 * (gpucompress_compress.cpp, nn_gpu.cu) focused on compression and inference.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "gpucompress.h"
#include "api/internal.hpp"
#include "api/gpucompress_state.hpp"
#include "nn/nn_weights.h"

/* Cost ranking globals (for debug print) */
extern float g_rank_w0;
extern float g_rank_w1;
extern float g_rank_w2;
extern float g_measured_bw_bytes_per_ms;
extern bool g_debug_nn;

static const char* s_algo_names[] = {
    "lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"};

static void action_str(int action, char* buf, size_t bufsz) {
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo = action % 8, quant = (action/8)%2, shuf = (action/16)%2;
    snprintf(buf, bufsz, "%s%s%s", s_algo_names[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

/* Debug context: phase name and timestep for log output */
static char g_dbg_phase[64] = "";
static int  g_dbg_timestep = -1;
static int  g_dbg_chunk = 0;

/* ============================================================
 * Chunk History Accessors
 * ============================================================ */

extern "C" void gpucompress_reset_chunk_history(void) {
    g_chunk_history_count.store(0);
    g_dbg_chunk = 0;
}

extern "C" void gpucompress_set_debug_context(const char* phase, int timestep) {
    if (phase) snprintf(g_dbg_phase, sizeof(g_dbg_phase), "%s", phase);
    else g_dbg_phase[0] = '\0';
    g_dbg_timestep = timestep;
    g_dbg_chunk = 0;
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
    if (idx >= 0 && idx < g_chunk_history_count.load() && idx < g_chunk_history_cap) {
        g_chunk_history[idx].decompression_ms = std::max(5.0f, ms);  /* clamped for MAPE */
        g_chunk_history[idx].decompression_ms_raw = ms;              /* unclamped for breakdown */
    }
}

/* ============================================================
 * Cache Statistics
 * ============================================================ */

extern "C" void gpucompress_flush_manager_cache(void) {
    gpucompress::flushCompManagerCache();
}

extern "C" void gpucompress_reset_cache_stats(void) {
    g_mgr_cache_hits.store(0);
    g_mgr_cache_misses.store(0);
}

extern "C" void gpucompress_get_cache_stats(int *hits, int *misses) {
    if (hits)   *hits   = g_mgr_cache_hits.load();
    if (misses) *misses = g_mgr_cache_misses.load();
}

/* ============================================================
 * Chunk Diagnostic Recording
 *
 * Called from gpucompress_compress_with_action_gpu() after
 * compression completes. Appends one entry to the history buffer.
 * ============================================================ */

namespace gpucompress {

int recordChunkDiagnostic(const ChunkDiagInput& d)
{
    std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
    int idx = g_chunk_history_count.fetch_add(1);
    if (idx >= g_chunk_history_cap) {
        int new_cap = (g_chunk_history_cap == 0) ? 4096 : g_chunk_history_cap * 2;
        auto* p = static_cast<gpucompress_chunk_diag_t*>(
            realloc(g_chunk_history, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
        if (p) { g_chunk_history = p; g_chunk_history_cap = new_cap; }
    }
    if (idx < g_chunk_history_cap) {
        gpucompress_chunk_diag_t *h = &g_chunk_history[idx];
        memset(h, 0, sizeof(*h));
        h->nn_action             = d.nn_action;
        h->nn_original_action    = d.nn_original_action;
        h->exploration_triggered = d.exploration_triggered ? 1 : 0;
        h->sgd_fired             = d.sgd_fired ? 1 : 0;
        h->nn_inference_ms       = d.nn_inference_ms;
        h->stats_ms              = d.stats_ms;
        h->preprocessing_ms      = d.preprocessing_ms;
        h->compression_ms        = d.compression_ms;
        h->compression_ms_raw   = d.compression_ms_raw;
        h->exploration_ms        = d.exploration_ms;
        h->sgd_update_ms         = d.sgd_ms;
        /* actual_ratio: uses primary compressed size (NN's pick) for fair MAPE reporting */
        size_t primary_sz = (d.primary_compressed_size > 0) ? d.primary_compressed_size : d.compressed_size;
        h->actual_ratio          = (primary_sz > 0)
            ? std::min(100.0f, static_cast<float>(d.input_size) / static_cast<float>(primary_sz))
            : 0.0f;
        /* final_ratio: uses post-exploration compressed size (what was written to disk) */
        h->final_ratio           = (d.compressed_size > 0)
            ? std::min(100.0f, static_cast<float>(d.input_size) / static_cast<float>(d.compressed_size))
            : 0.0f;
        h->predicted_ratio       = d.predicted_ratio;
        h->predicted_comp_time   = d.predicted_comp_time;
        h->predicted_decomp_time = d.predicted_decomp_time;
        h->predicted_psnr        = d.predicted_psnr;
        h->actual_psnr           = d.actual_psnr;

        /* Cost model diagnostics */
        h->cost_model_error_pct  = d.cost_model_error_pct;
        h->actual_cost           = d.actual_cost;
        h->predicted_cost        = d.predicted_cost;

        /* Original config metrics (before exploration swap) */
        h->orig_actual_ratio     = d.orig_actual_ratio;
        h->orig_comp_ms          = d.orig_comp_ms;
        h->orig_cost             = d.orig_cost;

        /* Exploration results */
        h->explore_n_alternatives = d.explore_n_alternatives;
        int n = std::min(d.explore_n_alternatives, 31);
        for (int i = 0; i < n; i++) {
            h->explore_alternatives[i] = d.explore_alternatives[i];
            h->explore_ratios[i]       = d.explore_ratios[i];
            h->explore_comp_ms[i]      = d.explore_comp_ms[i];
            h->explore_costs[i]        = d.explore_costs[i];
        }

        /* NN predicted ranking and per-config costs */
        h->predicted_ranking_count = d.top_actions_count;
        if (d.top_actions && d.top_actions_count > 0) {
            int nc = std::min(d.top_actions_count, 32);
            for (int i = 0; i < nc; i++)
                h->predicted_ranking[i] = d.top_actions[i];
        }
        if (d.predicted_costs)
            memcpy(h->predicted_costs, d.predicted_costs, 32 * sizeof(float));

        /* Features for deferred decomp SGD */
        h->feat_action   = d.nn_original_action;
        h->feat_eb_enc   = static_cast<float>(log10(fmax(d.error_bound, 1e-7)));
        h->feat_ds_enc   = static_cast<float>(log2(fmax((double)d.input_size, 1.0)));
        /* P4 fix: use pre-copied host-side stats instead of synchronous
         * cudaMemcpy D->H under mutex.  The caller copies stats to host
         * before entering this function, eliminating GPU stalls under lock. */
        if (d.h_stats_valid) {
            h->feat_entropy = d.h_feat_entropy;
            h->feat_mad     = d.h_feat_mad;
            h->feat_deriv   = d.h_feat_deriv;
        }

        /* Detailed timing breakdown (zero when g_detailed_timing is off) */
        h->ctx_acquire_ms     = d.ctx_acquire_ms;
        h->mgr_acquire_ms     = d.mgr_acquire_ms;
        h->configure_comp_ms  = d.configure_comp_ms;
        h->temp_alloc_ms      = d.temp_alloc_ms;
        h->compress_launch_ms = d.compress_launch_ms;
        h->stream_sync_ms     = d.stream_sync_ms;
        h->get_comp_size_ms   = d.get_comp_size_ms;
        h->header_write_ms    = d.header_write_ms;
        h->stats_copy_ms      = d.stats_copy_ms;
        h->final_sync_ms      = d.final_sync_ms;
        /* VOL Stage 1 timing (carried through WorkItem) */
        h->vol_stats_malloc_ms = d.vol_stats_malloc_ms;
        h->vol_stats_copy_ms   = d.vol_stats_copy_ms;
        h->vol_wq_post_wait_ms = d.vol_wq_post_wait_ms;
        /* diag_record_ms set post-hoc by caller after this function returns */

        /* Debug: print per-chunk predicted vs actual summary.
         * Skip non-AUTO compressions (predicted_ratio==0) to suppress noise
         * from Kendall tau profiler and fixed-algo phases. */
        if (g_debug_nn && h->predicted_ratio > 0.0f) {
            char final_str[40], orig_str[40];
            action_str(d.nn_action, final_str, sizeof(final_str));
            action_str(d.nn_original_action, orig_str, sizeof(orig_str));
            bool swapped = (d.nn_action != d.nn_original_action);
            if (swapped) {
                /* Show initial (NN pick) and final (exploration winner) */
                fprintf(stderr, "[NN-DBG] [%s T=%d] Chunk %d | initial: %s → final: %s | "
                        "cost_err=%.1f%% [SGD] [EXP]\n",
                        g_dbg_phase[0] ? g_dbg_phase : "?", g_dbg_timestep, idx,
                        orig_str, final_str,
                        (double)d.cost_model_error_pct * 100.0);
                fprintf(stderr, "           initial: ratio=%.2f ct=%.3f cost=%.3f\n",
                        (double)h->orig_actual_ratio,
                        (double)h->orig_comp_ms,
                        (double)h->orig_cost);
                fprintf(stderr, "             final: ratio=%.2f (primary=%.2f) ct=%.3f cost=%.3f\n",
                        (double)h->final_ratio,
                        (double)h->actual_ratio,
                        (double)h->compression_ms,
                        (double)h->actual_cost);
            } else {
                fprintf(stderr, "[NN-DBG] [%s T=%d] Chunk %d | %s | "
                        "ratio: pred=%.2f actual=%.2f | "
                        "ct: pred=%.3f actual=%.3f | "
                        "dt: pred=%.3f actual=%.3f | "
                        "cost_err=%.1f%%%s%s\n",
                        g_dbg_phase[0] ? g_dbg_phase : "?", g_dbg_timestep, idx,
                        final_str,
                        h->actual_ratio > 0 ? (double)h->predicted_ratio : 0.0,
                        (double)h->actual_ratio,
                        (double)h->predicted_comp_time, (double)h->compression_ms,
                        (double)h->predicted_decomp_time, 0.0,  /* decomp not measured at write */
                        (double)d.cost_model_error_pct * 100.0,
                        d.sgd_fired ? " [SGD]" : "",
                        d.exploration_triggered ? " [EXP]" : "");
            }
        }
    }
    return idx;
}

extern "C" void gpucompress_record_chunk_vol_timing(int idx,
    float pool_acquire_ms, float d2h_copy_ms, float io_queue_wait_ms)
{
    std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
    if (idx >= 0 && idx < g_chunk_history_count.load() && idx < g_chunk_history_cap) {
        g_chunk_history[idx].vol_pool_acquire_ms  = pool_acquire_ms;
        g_chunk_history[idx].vol_d2h_copy_ms      = d2h_copy_ms;
        g_chunk_history[idx].vol_io_queue_wait_ms = io_queue_wait_ms;
    }
}

extern "C" void gpucompress_record_chunk_s1_timing(int idx,
    float stats_malloc_ms, float stats_copy_ms, float wq_post_wait_ms)
{
    std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
    if (idx >= 0 && idx < g_chunk_history_count.load() && idx < g_chunk_history_cap) {
        g_chunk_history[idx].vol_stats_malloc_ms  = stats_malloc_ms;
        g_chunk_history[idx].vol_stats_copy_ms    = stats_copy_ms;
        g_chunk_history[idx].vol_wq_post_wait_ms  = wq_post_wait_ms;
    }
}

/* ============================================================
 * NN Debug: Print Full Cost Ranking
 * ============================================================ */

void printNNDebugRanking(const NNDebugPerConfig* h_debug, int winner_action,
                         float entropy, float mad, float deriv) {
    static const char* algo_names[] = {
        "lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"};

    int order[NN_NUM_CONFIGS];
    for (int i = 0; i < NN_NUM_CONFIGS; i++) order[i] = i;
    for (int i = 0; i < NN_NUM_CONFIGS - 1; i++)
        for (int j = i + 1; j < NN_NUM_CONFIGS; j++)
            if (h_debug[order[j]].cost < h_debug[order[i]].cost)
                { int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }

    fprintf(stderr, "[NN-DBG] ---- [%s T=%d] Chunk %d | Cost ranking (w0=%.1f w1=%.1f w2=%.1f bw=%.0f) ----\n",
            g_dbg_phase[0] ? g_dbg_phase : "?", g_dbg_timestep, g_dbg_chunk++,
            g_rank_w0, g_rank_w1, g_rank_w2, g_measured_bw_bytes_per_ms);
    if (entropy >= 0.0f)
        fprintf(stderr, "[NN-DBG]   features: entropy=%.4f  MAD=%.6f  deriv=%.6f\n",
                entropy, mad, deriv);
    fprintf(stderr, "[NN-DBG] %4s %-22s %8s %8s %8s %10s\n",
            "Rank", "Config", "CompT", "DecT", "Ratio", "COST");
    for (int i = 0; i < NN_NUM_CONFIGS; i++) {
        int a = order[i];
        int algo = a % 8, quant = (a/8)%2, shuf = (a/16)%2;
        char name[32];
        snprintf(name, sizeof(name), "%s%s%s", algo_names[algo],
                 shuf ? "+shuf" : "", quant ? "+quant" : "");
        fprintf(stderr, "[NN-DBG] %4d %-22s %8.3f %8.3f %8.2f %10.3f%s\n",
                i + 1, name,
                h_debug[a].comp_time, h_debug[a].decomp_time,
                h_debug[a].ratio, h_debug[a].cost,
                (a == winner_action) ? "  <- WINNER" : "");
    }
    fprintf(stderr, "[NN-DBG] ----\n");
}

} // namespace gpucompress
