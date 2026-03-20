#ifndef GPUCOMPRESS_STATE_HPP
#define GPUCOMPRESS_STATE_HPP

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include "gpucompress.h"

struct CompContext;  /* forward declaration — full definition in internal.hpp */

/* Shared mutable state for the GPUCompress library.
 *
 * All variables are defined in gpucompress_api.cpp (single definition site)
 * and extern-declared here for use by the split implementation files:
 *   gpucompress_pool.cpp, gpucompress_compress.cpp,
 *   gpucompress_explore.cpp, gpucompress_learning.cpp.
 *
 * The 7 globals with external linkage (g_sgd_stream, g_rank_w0, etc.)
 * are also extern'd from nn_gpu.cu — they stay defined in gpucompress_api.cpp. */

/* ---- Library lifecycle ---- */
extern std::atomic<bool> g_initialized;
extern std::mutex        g_init_mutex;
extern cudaStream_t      g_default_stream;

/* ---- Online learning ---- */
extern std::atomic<bool> g_online_learning_enabled;
extern std::atomic<bool> g_exploration_enabled;
extern double            g_exploration_threshold;
extern int               g_exploration_k_override;
extern float             g_reinforce_lr;
extern float             g_reinforce_mape_threshold;

/* ---- Last NN action (per-call query) ---- */
extern std::atomic<int>  g_last_nn_action;
extern std::atomic<int>  g_last_nn_original_action;
extern std::atomic<int>  g_last_exploration_triggered;
extern std::atomic<int>  g_last_sgd_fired;

/* ---- Chunk diagnostic history ---- */
extern gpucompress_chunk_diag_t* g_chunk_history;
extern int                       g_chunk_history_cap;
extern std::atomic<int>          g_chunk_history_count;
extern std::mutex                g_chunk_history_mutex;

/* ---- nvcomp manager cache stats ---- */
extern std::atomic<int>  g_mgr_cache_hits;
extern std::atomic<int>  g_mgr_cache_misses;

/* ---- SGD serialization ---- */
extern std::mutex        g_sgd_mutex;

/* ---- SGD stream/event (also extern'd from nn_gpu.cu) ---- */
extern cudaStream_t      g_sgd_stream;
extern cudaEvent_t       g_sgd_done;
extern std::atomic<bool> g_sgd_ever_fired;

/* ---- Cost ranking weights (also extern'd from nn_gpu.cu) ---- */
extern float g_rank_w0;
extern float g_rank_w1;
extern float g_rank_w2;
extern float g_measured_bw_bytes_per_ms;

/* ---- ContextGuard RAII (used by compress and pool callers) ---- */

namespace gpucompress { void releaseCompContext(CompContext*); }

struct ContextGuard {
    CompContext* ctx;
    explicit ContextGuard(CompContext* c) : ctx(c) {}
    ~ContextGuard() { if (ctx) { gpucompress::releaseCompContext(ctx); ctx = nullptr; } }
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;
};

#endif /* GPUCOMPRESS_STATE_HPP */
