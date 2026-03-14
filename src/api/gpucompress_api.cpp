/**
 * @file gpucompress_api.cpp
 * @brief GPUCompress C API Implementation
 *
 * Implements the public C API defined in gpucompress.h.
 * Wraps existing C++ components for external library integration.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <strings.h>
#include <cstdio>
#include <cinttypes>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>

#include "gpucompress.h"
#include "api/internal.hpp"
#include "compression/compression_factory.hpp"
#include "compression/compression_header.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

#include "nvcomp.hpp"
#include "xfer_tracker.h"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

/* ============================================================
 * External Functions from CUDA files
 * ============================================================ */

extern "C" {
    // From entropy_kernel.cu
    double gpucompress_entropy_gpu_impl(const void* d_data, size_t num_bytes, void* stream);

    // From nn_gpu.cu
    int gpucompress_nn_load_impl(const char* filepath);
    int gpucompress_nn_is_loaded_impl(void);
    void gpucompress_nn_cleanup_impl(void);
    void* gpucompress_nn_get_device_ptr_impl(void);

    // From stats_kernel.cu
    void gpucompress_free_stats_workspace(void);

}

/* ============================================================
 * SGD stream / event globals — extern-referenced from nn_gpu.cu
 * Must be at file scope (external linkage, not anonymous namespace)
 * ============================================================ */

cudaStream_t g_sgd_stream    = nullptr;
cudaEvent_t  g_sgd_done      = nullptr;
std::atomic<bool> g_sgd_ever_fired{false};

/** Verbose transfer/SGD logging — off by default, set via gpucompress_set_verbose(). */
bool g_gc_verbose = false;

/** Cost-based ranking weights and measured bandwidth — extern-referenced from nn_gpu.cu */
float g_rank_w0 = 1.0f;                   // weight on compression time
float g_rank_w1 = 1.0f;                   // weight on decompression time
float g_rank_w2 = 1.0f;                   // weight on I/O cost (data_size / (ratio * bw))
float g_measured_bw_bytes_per_ms = 1e6f;   // default 1 GB/s = 1e6 bytes/ms

#define GC_LOG(fmt, ...) do { if (g_gc_verbose) fprintf(stderr, fmt, ##__VA_ARGS__); } while(0)

/* ============================================================
 * Global State
 * ============================================================ */

namespace {

/** Library initialization state */
std::atomic<bool> g_initialized{false};

/** Reference count for init/cleanup */
std::atomic<int> g_ref_count{0};

/** Mutex for thread-safe initialization */
std::mutex g_init_mutex;

/** Default CUDA stream */
cudaStream_t g_default_stream = nullptr;

/** CUDA device */
int g_cuda_device = 0;

/** Version string */
const char* VERSION_STRING = "1.0.0";

/** Online learning state */
std::atomic<bool> g_online_learning_enabled{false};   // master switch
std::atomic<bool> g_exploration_enabled{false};       // OFF by default
double g_exploration_threshold = 0.20;    // 20% MAPE
int g_exploration_k_override = -1;        // -1 = use dynamic default

/** Last NN action and exploration state (for query after compress).
 *  Under concurrent use values are undefined per-call; per-call stats struct
 *  carries the correct value. Atomic to prevent UB. */
std::atomic<int> g_last_nn_action{-1};
std::atomic<int> g_last_nn_original_action{-1};
std::atomic<int> g_last_exploration_triggered{0};
std::atomic<int> g_last_sgd_fired{0};

/** Per-chunk diagnostic history — dynamic, unbounded */
static gpucompress_chunk_diag_t* g_chunk_history      = nullptr;
static int                       g_chunk_history_cap   = 0;
std::atomic<int>                 g_chunk_history_count{0};
static std::mutex                g_chunk_history_mutex;

/** Force-algorithm queue: per-chunk algorithm overrides for ALGO_AUTO */
struct ForcedAlgo {
    int algorithm;      // gpucompress_algorithm_t enum (1-8)
    int shuffle;        // 1 = enable byte shuffle, 0 = disable
    int quant;          // 1 = enable quantization, 0 = disable
    double error_bound; // error bound for quantization (ignored when quant=0)
};
static ForcedAlgo*   g_force_algo_queue    = nullptr;
static int           g_force_algo_cap      = 0;
static int           g_force_algo_count    = 0;
static std::atomic<int> g_force_algo_idx{0};
static std::mutex    g_force_algo_mutex;

/* ---- CompContext pool ---- */
static CompContext g_comp_pool[N_COMP_CTX];
static bool g_pool_slot_free[N_COMP_CTX];
static int  g_pool_free_count = 0;
static std::mutex g_pool_mutex;
static std::condition_variable g_pool_cv;

/* ---- SGD serialization mutex (CPU-side) ---- */
static std::mutex g_sgd_mutex;

/* g_auto_mutex removed (E1 fix): host-path now uses per-slot CompContext
 * buffers for stats + NN inference, eliminating serialization. */

/** Thread-local CUDA timing events — each thread gets its own pair,
 *  lazy-initialized on first use. Avoids the race on global events
 *  while keeping the zero-contention benefit of per-thread storage.
 *  Cleanup handled by CUDA context teardown (single GPU node). */
/* Thread-local timing events removed for host path (E1 fix):
 * gpucompress_compress now uses per-slot CompContext events.
 * These remain only for any non-pool callers that need thread-local events. */

/** Online reinforcement (SGD) state */
float g_reinforce_lr = 0.01f;
float g_reinforce_mape_threshold = 0.20f;

/** Algorithm names */
const char* ALGORITHM_NAMES[] = {
    "auto",
    "lz4",
    "snappy",
    "deflate",
    "gdeflate",
    "zstd",
    "ans",
    "cascaded",
    "bitcomp"
};

/** Error messages */
const char* ERROR_MESSAGES[] = {
    "Success",
    "Invalid input parameter",
    "CUDA operation failed",
    "Compression failed",
    "Decompression failed",
    "Out of memory",
    "Reserved",
    "Invalid compression header",
    "Library not initialized",
    "Output buffer too small"
};

} // anonymous namespace

/* ============================================================
 * CompContext pool implementation
 * ============================================================ */

namespace gpucompress {

// Layout: [AutoStatsGPU | histogram(256 uint) | init_flag(int)]
static constexpr size_t kPoolStatsWSZ =
    sizeof(AutoStatsGPU) + 256 * sizeof(unsigned int) + sizeof(int);

/** Destroy a single CompContext slot (null-safe). Caller must hold g_pool_mutex. */
static void destroyCompContextSlot(CompContext& ctx) {
    /* N1 fix: leak cached managers on shutdown — their GPU workspace
     * is cleaned up by CUDA context teardown. nvcomp manager dtors
     * crash if called during late cleanup (CUDA primary ctx gone). */
    for (int a = 0; a < CompContext::N_COMP_ALGOS; a++)
        ctx.comp_mgr_cache[a] = nullptr;  /* intentional leak */
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
    if (ctx.d_sgd_grad_buffer)   { cudaFree(ctx.d_sgd_grad_buffer);
                                   ctx.d_sgd_grad_buffer = nullptr; }
    if (ctx.d_sgd_output)        { cudaFree(ctx.d_sgd_output); ctx.d_sgd_output = nullptr; }
    if (ctx.d_sgd_samples)       { cudaFree(ctx.d_sgd_samples); ctx.d_sgd_samples = nullptr; }
    if (ctx.d_range_min)         { cudaFree(ctx.d_range_min); ctx.d_range_min = nullptr; }
    if (ctx.d_range_max)         { cudaFree(ctx.d_range_max); ctx.d_range_max = nullptr; }
}

int initCompContextPool() {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    /* N1 fix: delete cached managers before memset (raw ptrs, safe to zero after) */
    for (int i = 0; i < N_COMP_CTX; i++) {
        for (int a = 0; a < CompContext::N_COMP_ALGOS; a++) {
            delete g_comp_pool[i].comp_mgr_cache[a];
        }
    }
    memset(g_comp_pool, 0, sizeof(g_comp_pool));
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

        if (cudaMalloc(&ctx.d_sgd_grad_buffer,
                       NN_SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_sgd_output, sizeof(SGDOutput)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_sgd_samples,
                       NN_MAX_SGD_SAMPLES * sizeof(SGDSample)) != cudaSuccess) goto fail;

        /* Per-slot quantization range buffers — eliminates shared static globals */
        if (cudaMalloc(&ctx.d_range_min, sizeof(double)) != cudaSuccess) goto fail;
        if (cudaMalloc(&ctx.d_range_max, sizeof(double)) != cudaSuccess) goto fail;

        g_pool_slot_free[i] = true;
        g_pool_free_count++;
    }
    return 0;

fail:
    /* Clean up all slots (partially-initialized ones have non-null pointers,
     * untouched ones are zeroed from memset — destroyCompContextSlot is null-safe). */
    for (int j = 0; j < N_COMP_CTX; j++) {
        destroyCompContextSlot(g_comp_pool[j]);
        g_pool_slot_free[j] = false;
    }
    g_pool_free_count = 0;
    return -1;
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
    /* S3 fix: synchronize only the N_COMP_CTX pool streams + g_sgd_stream,
     * rather than draining the entire device with cudaDeviceSynchronize.
     * Safe to call under g_pool_mutex since streams are never destroyed
     * while the pool is live. */
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    for (int i = 0; i < N_COMP_CTX; i++) {
        if (g_comp_pool[i].stream)
            cudaStreamSynchronize(g_comp_pool[i].stream);
    }
    /* g_sgd_stream is at file scope (not in gpucompress namespace) */
    if (::g_sgd_stream)
        cudaStreamSynchronize(::g_sgd_stream);
}

nvcomp::nvcompManagerBase* getCachedCompManager(CompContext* ctx, CompressionAlgorithm algo) {
    int idx = static_cast<int>(algo);
    if (idx < 0 || idx >= CompContext::N_COMP_ALGOS) return nullptr;
    if (!ctx->comp_mgr_cache[idx]) {
        auto mgr = createCompressionManager(
            algo, gpucompress::DEFAULT_CHUNK_SIZE, ctx->stream, nullptr);
        ctx->comp_mgr_cache[idx] = mgr.release();  /* transfer ownership to cache */
    }
    return ctx->comp_mgr_cache[idx];
}

} // namespace gpucompress

/* RAII guard: releases CompContext on scope exit */
namespace {
struct ContextGuard {
    CompContext* ctx;
    explicit ContextGuard(CompContext* c) : ctx(c) {}
    ~ContextGuard() { if (ctx) { gpucompress::releaseCompContext(ctx); ctx = nullptr; } }
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;
};
} // anonymous namespace

/* ============================================================
 * Initialization and Cleanup
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_init(const char* weights_path) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Increment reference count
    int old_ref = g_ref_count.fetch_add(1);

    if (old_ref > 0) {
        // Already initialized
        return GPUCOMPRESS_SUCCESS;
    }

    // Initialize CUDA
    cudaError_t err = cudaSetDevice(g_cuda_device);
    if (err != cudaSuccess) {
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Create default stream
    err = cudaStreamCreate(&g_default_stream);
    if (err != cudaSuccess) {
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Timing events are now thread_local — no global creation needed

    // SGD stream + event for GPU-level weight-update ordering
    err = cudaStreamCreate(&g_sgd_stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(g_default_stream);
        g_default_stream = nullptr;
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }
    err = cudaEventCreate(&g_sgd_done);
    if (err != cudaSuccess) {
        cudaStreamDestroy(g_sgd_stream);
        g_sgd_stream = nullptr;
        cudaStreamDestroy(g_default_stream);
        g_default_stream = nullptr;
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Init CompContext pool
    if (gpucompress::initCompContextPool() != 0) {
        gpucompress::destroyCompContextPool();
        cudaEventDestroy(g_sgd_done);
        g_sgd_done = nullptr;
        cudaStreamDestroy(g_sgd_stream);
        g_sgd_stream = nullptr;
        cudaStreamDestroy(g_default_stream);
        g_default_stream = nullptr;
        g_ref_count--;
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Bandwidth probe: write+read 16MB temp file to measure storage BW
    {
        const size_t probe_sz = 16 * 1024 * 1024;
        char tmp_path[] = "/tmp/gpucompress_bw_XXXXXX";
        int fd = mkstemp(tmp_path);
        if (fd >= 0) {
            std::vector<char> buf(probe_sz, 0x42);
            auto t_bw0 = std::chrono::steady_clock::now();
            ssize_t wr = write(fd, buf.data(), probe_sz);
            fsync(fd);
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            lseek(fd, 0, SEEK_SET);
            ssize_t rd = read(fd, buf.data(), probe_sz);
            auto t_bw1 = std::chrono::steady_clock::now();
            close(fd);
            unlink(tmp_path);
            if (wr == (ssize_t)probe_sz && rd == (ssize_t)probe_sz) {
                double ms = std::chrono::duration<double, std::milli>(t_bw1 - t_bw0).count();
                if (ms > 0.0) {
                    // Use min of write and read BW (conservative)
                    g_measured_bw_bytes_per_ms = static_cast<float>(
                        static_cast<double>(probe_sz) / ms);
                    GC_LOG("[BW PROBE] 16 MB write+read in %.1f ms => %.2f GB/s\n",
                           ms, g_measured_bw_bytes_per_ms / 1e6);
                }
            }
        }
    }

    // Load NN weights if path provided
    if (weights_path != nullptr && weights_path[0] != '\0') {
        if (gpucompress_nn_load_impl(weights_path) != 0) {
            fprintf(stderr, "Warning: Failed to load NN weights from %s\n", weights_path);
        }
    }

    g_initialized.store(true);
    return GPUCOMPRESS_SUCCESS;
}

extern "C" void gpucompress_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Guard against underflow: if ref_count is already 0, nothing to do
    if (g_ref_count.load() <= 0) return;

    int old_ref = g_ref_count.fetch_sub(1);

    if (old_ref <= 1) {
        // Dump transfer summary before shutdown
        xfer_tracker_dump();

        // Last reference, cleanup
        g_online_learning_enabled = false;
        g_exploration_enabled = false;
        gpucompress_nn_cleanup_impl();
        gpucompress_free_stats_workspace();
        gpucompress::destroyCompContextPool();
        free_range_bufs();
        if (g_sgd_stream) { cudaStreamDestroy(g_sgd_stream); g_sgd_stream = nullptr; }
        if (g_sgd_done)   { cudaEventDestroy(g_sgd_done);   g_sgd_done   = nullptr; }
        g_sgd_ever_fired.store(false, std::memory_order_relaxed);
        // Timing events are thread_local — cleaned up by CUDA context teardown
        if (g_default_stream != nullptr) {
            cudaStreamDestroy(g_default_stream);
            g_default_stream = nullptr;
        }
        g_initialized.store(false);
        g_ref_count.store(0);
        /* Free dynamic chunk history */
        {
            std::lock_guard<std::mutex> lk(g_chunk_history_mutex);
            if (g_chunk_history) {
                free(g_chunk_history);
                g_chunk_history    = nullptr;
                g_chunk_history_cap = 0;
            }
            g_chunk_history_count.store(0);
        }
        /* Free force-algorithm queue */
        {
            std::lock_guard<std::mutex> fq_lock(g_force_algo_mutex);
            if (g_force_algo_queue) {
                free(g_force_algo_queue);
                g_force_algo_queue = nullptr;
                g_force_algo_cap = 0;
                g_force_algo_count = 0;
                g_force_algo_idx.store(0);
            }
        }
    }
}

extern "C" int gpucompress_is_initialized(void) {
    return g_initialized.load() ? 1 : 0;
}

extern "C" gpucompress_config_t gpucompress_default_config(void) {
    gpucompress_config_t config;
    config.algorithm = GPUCOMPRESS_ALGO_LZ4;
    config.preprocessing = GPUCOMPRESS_PREPROC_NONE;
    config.error_bound = 0.0;
    config.cuda_device = -1;
    config.cuda_stream = nullptr;
    return config;
}

/* ============================================================
 * Core Compression
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_compress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (input == nullptr || output == nullptr || output_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (input_size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    // Use default config if not provided
    gpucompress_config_t cfg = config ? *config : gpucompress_default_config();

    /* Acquire a per-slot context for thread-safe stats+NN (E1 fix) */
    ContextGuard guard{gpucompress::acquireCompContext()};
    if (!guard.ctx) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    CompContext* ctx = guard.ctx;
    cudaStream_t stream = ctx->stream;

    cudaError_t cuda_err;

    // Allocate GPU input buffer
    size_t aligned_input_size = gpucompress::alignSize(input_size, gpucompress::GPU_ALIGNMENT);
    uint8_t* d_input = nullptr;

    cuda_err = cudaMalloc(&d_input, aligned_input_size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy input to GPU
    GC_LOG("[XFER H→D] compress: input data (%zu B)\n", input_size);
    XFER_TRACK("compress: H->D input data", input_size, cudaMemcpyHostToDevice);
    cuda_err = cudaMemcpyAsync(d_input, input, input_size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Variables for compression
    gpucompress_algorithm_t algo_to_use = cfg.algorithm;
    unsigned int preproc_to_use = cfg.preprocessing;
    double entropy = 0.0;

    // Auto algorithm selection
    double mad = 0.0;
    double second_derivative = 0.0;
    bool nn_was_used = false;
    bool sgd_fired = false;
    bool exploration_triggered = false;
    int nn_action = 0;
    int nn_original_action = -1;
    float predicted_ratio = 0.0f;
    float predicted_comp_time = 0.0f;
    float predicted_decomp_time = 0.0f;
    float predicted_psnr = 0.0f;
    int top_actions[32] = {0};
    bool is_ood = false;
    AutoStatsGPU* d_stats_ptr = nullptr;

    /* Per-chunk timing breakdown */
    float diag_nn_inference_ms  = 0.0f;
    float diag_stats_ms         = 0.0f;
    float diag_preprocessing_ms = 0.0f;
    float diag_compression_ms   = 0.0f;
    float diag_exploration_ms   = 0.0f;
    float diag_sgd_ms           = 0.0f;

    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO) {
        size_t num_elements = input_size / sizeof(float);
        int action = 0;
        int rc = -1;

        if (num_elements > 0 && gpucompress_nn_is_loaded_impl()) {
            bool nn_timed = false;

            /* Stats timing — use per-context events (no mutex needed, E1 fix) */
            cudaEventRecord(ctx->stats_start, stream);

            /* Use ctx-aware stats (per-slot buffers, no global contention) */
            d_stats_ptr = gpucompress::runStatsKernelsNoSync(d_input, input_size, stream, ctx);

            if (d_stats_ptr && stats != nullptr) {
                XFER_TRACK("compress/auto: D->H stats entropy", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&entropy, &d_stats_ptr->entropy, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
                XFER_TRACK("compress/auto: D->H stats mad", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&mad, &d_stats_ptr->mad_normalized, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
                XFER_TRACK("compress/auto: D->H stats deriv", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&second_derivative, &d_stats_ptr->deriv_normalized, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
            }
            cudaEventRecord(ctx->stats_stop, stream);

            if (d_stats_ptr) {
                float* p_ratio = &predicted_ratio;
                float* p_comp_time = &predicted_comp_time;
                float* p_decomp_time = &predicted_decomp_time;
                float* p_psnr = &predicted_psnr;
                int* p_top = g_online_learning_enabled ? top_actions : nullptr;
                int fused_ood = 0;

                /* NN timing — use ctx-aware inference (per-slot buffers, E1 fix) */
                cudaEventRecord(ctx->nn_start, stream);
                action = gpucompress::runNNFusedInferenceCtx(
                    d_stats_ptr, input_size, cfg.error_bound, stream, ctx,
                    &action, p_ratio, p_comp_time, p_decomp_time, p_psnr,
                    g_online_learning_enabled ? &fused_ood : nullptr, p_top,
                    ctx->nn_stop);
                nn_timed = true;
                rc = (action >= 0) ? 0 : -1;

                if (rc == 0) {
                    is_ood = (fused_ood != 0);
                }
            }

            if (rc == 0) {
                nn_was_used = true;
                nn_action = action;
                nn_original_action = action;
                g_last_nn_action.store(action);
            }

            /* Events already synchronized by runNNFusedInferenceCtx's internal sync */
            if (nn_timed) {
                cudaEventElapsedTime(&diag_nn_inference_ms, ctx->nn_start, ctx->nn_stop);
                cudaEventElapsedTime(&diag_stats_ms, ctx->stats_start, ctx->stats_stop);
            }
        }

        if (rc == 0) {
            gpucompress::DecodedAction decoded = gpucompress::decodeAction(action);
            algo_to_use = static_cast<gpucompress_algorithm_t>(decoded.algorithm + 1);
            preproc_to_use = 0;
            if (decoded.shuffle_size > 0)
                preproc_to_use |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
            if (decoded.use_quantization)
                preproc_to_use |= GPUCOMPRESS_PREPROC_QUANTIZE;

            GC_LOG("NN: chose %s%s%s (action=%d, predicted_ratio=%.2f)\n",
                   gpucompress_algorithm_name(algo_to_use),
                   (preproc_to_use & GPUCOMPRESS_PREPROC_SHUFFLE_4) ? " +shuffle" : "",
                   (preproc_to_use & GPUCOMPRESS_PREPROC_QUANTIZE)  ? " +quant"   : "",
                   action, predicted_ratio);
        } else {
            fprintf(stderr, "gpucompress ERROR: ALGO_AUTO requested but NN inference failed "
                    "(weights not loaded or inference error). Load weights via gpucompress_init() "
                    "or use an explicit algorithm.\n");
            cudaFree(d_input);
            return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
        }
    }

    // Pointer to data to compress (may change with preprocessing)
    uint8_t* d_compress_input = d_input;
    size_t compress_input_size = input_size;
    uint8_t* d_quantized = nullptr;
    uint8_t* d_shuffled = nullptr;
    QuantizationResult quant_result;

    auto t_preproc_start = std::chrono::steady_clock::now();

    // Apply quantization if requested
    if ((preproc_to_use & GPUCOMPRESS_PREPROC_QUANTIZE) && cfg.error_bound > 0.0) {
        size_t element_size = sizeof(float);  // Default to float
        size_t num_elements = input_size / element_size;

        if (num_elements > 0) {
            QuantizationConfig quant_cfg(
                QuantizationType::LINEAR,
                cfg.error_bound,
                num_elements,
                element_size
            );

            quant_result = quantize_simple(d_compress_input, num_elements,
                                           element_size, quant_cfg, stream);

            if (quant_result.isValid()) {
                d_quantized = static_cast<uint8_t*>(quant_result.d_quantized);
                d_compress_input = d_quantized;
                compress_input_size = quant_result.quantized_bytes;
            } else {
                fprintf(stderr, "gpucompress ERROR: quantization failed but was requested "
                        "(error_bound=%.6e, num_elements=%zu)\n", cfg.error_bound, num_elements);
                cudaFree(d_input);
                return GPUCOMPRESS_ERROR_COMPRESSION;
            }
        }
    }

    // Apply byte shuffle if requested
    unsigned int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(preproc_to_use);
    if (shuffle_size > 0) {
        d_shuffled = byte_shuffle_simple(
            d_compress_input,
            compress_input_size,
            shuffle_size,
            gpucompress::SHUFFLE_CHUNK_SIZE,
            stream
        );

        if (d_shuffled != nullptr) {
            d_compress_input = d_shuffled;
        } else {
            fprintf(stderr, "gpucompress ERROR: byte shuffle failed but was requested "
                    "(element_size=%u, input_size=%zu)\n", shuffle_size, compress_input_size);
            if (d_quantized) cudaFree(d_quantized);
            cudaFree(d_input);
            return GPUCOMPRESS_ERROR_COMPRESSION;
        }
    }

    /* No cudaStreamSynchronize here — compression runs on the same stream,
     * so GPU ordering guarantees preprocessing completes first. Any async
     * errors will surface at the final sync after compression. (H2 fix) */
    if (d_quantized || d_shuffled) {
        /* Approximate timing — may include compression setup overhead,
         * but avoids a blocking sync purely for diagnostics. */
        diag_preprocessing_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_preproc_start).count();
    }

    /* N1 fix: use cached compression manager */
    CompressionAlgorithm internal_algo = gpucompress::toInternalAlgorithm(algo_to_use);
    auto* compressor = gpucompress::getCachedCompManager(ctx, internal_algo);

    if (!compressor) {
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    // Configure compression
    CompressionConfig comp_config = compressor->configure_compression(compress_input_size);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;

    if (max_compressed_size == 0) {
        fprintf(stderr, "gpucompress ERROR: configure_compression returned "
                "max_compressed_buffer_size=0 (algo=%d, input_size=%zu)\n",
                (int)algo_to_use, compress_input_size);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    // Allocate output buffer on GPU (with header space)
    size_t header_size = GPUCOMPRESS_HEADER_SIZE;
    size_t total_max_size = header_size + max_compressed_size;
    if (total_max_size < header_size) {
        // Overflow in header + compressed size addition
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    uint8_t* d_output = nullptr;
    cuda_err = cudaMalloc(&d_output, total_max_size);
    if (cuda_err != cudaSuccess) {
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Pointer to compressed data (after header)
    uint8_t* d_compressed = d_output + header_size;

    // Compress (with timing for reinforcement)
    float primary_comp_time_ms = 0.0f;
    bool timing_ok = (ctx->t_start != nullptr && ctx->t_stop != nullptr);
    if (timing_ok) cudaEventRecord(ctx->t_start, stream);

    try {
        compressor->compress(d_compress_input, d_compressed, comp_config);
    } catch (...) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    if (timing_ok) {
        cudaEventRecord(ctx->t_stop, stream);
        cudaEventSynchronize(ctx->t_stop);
        cudaEventElapsedTime(&primary_comp_time_ms, ctx->t_start, ctx->t_stop);
    }
    diag_compression_ms = primary_comp_time_ms;

    // Get compressed size
    size_t compressed_size = compressor->get_compressed_output_size(d_compressed);
    size_t total_size = header_size + compressed_size;

    // Measure primary decomp time (RL-gated: only when SGD can learn from it)
    float primary_decomp_time_ms = 0.0f;
    if (g_reinforce_lr > 0.0f && nn_was_used && timing_ok) {
        auto rt_decomp = createDecompressionManager(d_compressed, stream);
        if (rt_decomp) {
            DecompressionConfig rt_dc = rt_decomp->configure_decompression(d_compressed);
            size_t rt_decomp_size = rt_dc.decomp_data_size;
            uint8_t* d_rt_buf = nullptr;
            if (cudaMalloc(&d_rt_buf, rt_decomp_size) == cudaSuccess) {
                cudaEventRecord(ctx->t_start, stream);
                try {
                    rt_decomp->decompress(d_rt_buf, d_compressed, rt_dc);
                    cudaEventRecord(ctx->t_stop, stream);
                    cudaEventSynchronize(ctx->t_stop);
                    cudaEventElapsedTime(&primary_decomp_time_ms, ctx->t_start, ctx->t_stop);
                } catch (...) {
                    // Decompress failed — leave at 0
                }
                cudaFree(d_rt_buf);
            }
        }
    }

    // Check output buffer size
    if (total_size > *output_size) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        *output_size = total_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    // Build compression header
    CompressionHeader header;
    header.magic = COMPRESSION_MAGIC;
    header.version = COMPRESSION_HEADER_VERSION;
    header.shuffle_element_size = shuffle_size;
    header.original_size = input_size;
    header.compressed_size = compressed_size;

    if (d_quantized && quant_result.isValid()) {
        header.setQuantizationFlags(
            static_cast<uint32_t>(quant_result.type),
            quant_result.actual_precision,
            true
        );
        header.quant_error_bound = quant_result.error_bound;
        header.quant_scale = quant_result.scale_factor;
        header.data_min = quant_result.data_min;
        header.data_max = quant_result.data_max;
    } else {
        header.quant_flags = 0;
        header.quant_error_bound = 0.0;
        header.quant_scale = 0.0;
        header.data_min = 0.0;
        header.data_max = 0.0;
    }
    header.setAlgorithmId((uint8_t)algo_to_use);

    // Write header directly to host output (no GPU round-trip for 64 bytes)
    memcpy(output, &header, sizeof(CompressionHeader));

    // Copy only compressed payload from GPU to host (skip header region)
    uint8_t* host_payload = static_cast<uint8_t*>(output) + header_size;
    GC_LOG("[XFER D→H] compress: payload (%zu B)\n", compressed_size);
    XFER_TRACK("compress: D->H compressed payload", compressed_size, cudaMemcpyDeviceToHost);
    cuda_err = cudaMemcpyAsync(host_payload, d_compressed, compressed_size,
                               cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cuda_err = cudaStreamSynchronize(stream);

    // Cleanup GPU output and preprocessing buffers (but keep d_input for exploration)
    cudaFree(d_output);
    if (d_quantized) cudaFree(d_quantized);
    if (d_shuffled) cudaFree(d_shuffled);

    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Set output size
    *output_size = total_size;

    // Active learning: Level 1 passive collection + Level 2 exploration
    auto t_explore_start = std::chrono::steady_clock::now();
    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && nn_was_used &&
        g_online_learning_enabled) {
        double actual_ratio = static_cast<double>(input_size) /
                              static_cast<double>(compressed_size);

        // Cost-based prediction error: compare predicted vs actual cost.
        // Now uses measured decomp time when available (RL-gated decomp-at-write).
        double ds = static_cast<double>(input_size);
        double bw = static_cast<double>(g_measured_bw_bytes_per_ms);
        double w0 = static_cast<double>(g_rank_w0);
        double w1 = static_cast<double>(g_rank_w1);
        double w2 = static_cast<double>(g_rank_w2);
        double pred_dt = static_cast<double>(predicted_decomp_time);
        double pred_ct = static_cast<double>(predicted_comp_time);
        double pred_r  = static_cast<double>(predicted_ratio);
        double act_ct  = static_cast<double>(primary_comp_time_ms);
        double act_dt  = (primary_decomp_time_ms > 0.0f)
                       ? static_cast<double>(primary_decomp_time_ms) : pred_dt;

        double actual_cost = w0 * act_ct + w1 * act_dt
                           + ((actual_ratio > 0.0) ? w2 * ds / (actual_ratio * bw) : 0.0);
        double predicted_cost = w0 * pred_ct + w1 * pred_dt
                              + ((pred_r > 0.0) ? w2 * ds / (pred_r * bw) : 0.0);
        double error_pct = (actual_cost > 0.0)
            ? std::abs(actual_cost - predicted_cost) / actual_cost : 0.0;

        // Collect explored (action, ratio, comp_time) for reinforcement
        struct ExploredResult { int action; double ratio; double comp_time_ms;
                                double decomp_time_ms; double psnr; };
        std::vector<ExploredResult> explored_samples;
        double primary_psnr = 120.0;
        if (d_quantized && quant_result.isValid()) {
            double range = quant_result.data_max - quant_result.data_min;
            if (range > 0.0 && quant_result.error_bound > 0.0) {
                primary_psnr = fmin(20.0 * log10(range / quant_result.error_bound), 120.0);
            }
        }
        explored_samples.push_back({nn_action, actual_ratio,
                                    static_cast<double>(primary_comp_time_ms),
                                    static_cast<double>(primary_decomp_time_ms),
                                    primary_psnr});

        if (g_exploration_enabled && error_pct > g_exploration_threshold) {
            exploration_triggered = true;
            // Level 2: Explore alternatives
            // Determine K (number of alternatives to try)
            // OOD increases K but no longer forces exploration independently
            int K;
            if (g_exploration_k_override > 0) {
                K = g_exploration_k_override;
            } else {
                K = 3;  // Top-3 alternatives (limits GPU memory pressure)
            }

            gpucompress::DecodedAction primary_dec = gpucompress::decodeAction(nn_action);
            GC_LOG("[EXPLORE] Chunk %zuB | primary=action%d (%s%s%s) ratio=%.3f | "
                    "cost_err=%.1f%% %s| K=%d\n",
                    input_size, nn_action,
                    ALGORITHM_NAMES[primary_dec.algorithm + 1],
                    primary_dec.shuffle_size > 0 ? "+shuf" : "",
                    primary_dec.use_quantization ? "+quant" : "",
                    actual_ratio,
                    error_pct * 100.0,
                    is_ood ? "OOD " : "",
                    K);

            // Try top-K alternative configs
            double best_ratio = actual_ratio;

            for (int i = 1; i <= K && i < 32; i++) {
                int alt_action = top_actions[i];
                if (alt_action == nn_action) continue;

                gpucompress::DecodedAction alt = gpucompress::decodeAction(alt_action);
                if (alt.use_quantization && cfg.error_bound <= 0.0) continue;
                gpucompress_algorithm_t alt_algo =
                    static_cast<gpucompress_algorithm_t>(alt.algorithm + 1);
                unsigned int alt_preproc = 0;
                if (alt.shuffle_size > 0)
                    alt_preproc |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                if (alt.use_quantization)
                    alt_preproc |= GPUCOMPRESS_PREPROC_QUANTIZE;

                // Compress with alternative config
                uint8_t* d_alt_input = d_input;
                size_t alt_compress_size = input_size;
                uint8_t* d_alt_quant = nullptr;
                uint8_t* d_alt_shuf = nullptr;
                QuantizationResult alt_quant_result;

                // Apply quantization
                if ((alt_preproc & GPUCOMPRESS_PREPROC_QUANTIZE) && cfg.error_bound > 0.0) {
                    size_t num_el = input_size / sizeof(float);
                    if (num_el > 0) {
                        QuantizationConfig qcfg(
                            QuantizationType::LINEAR, cfg.error_bound,
                            num_el, sizeof(float));
                        alt_quant_result = quantize_simple(
                            d_alt_input, num_el, sizeof(float), qcfg, stream);
                        if (alt_quant_result.isValid()) {
                            d_alt_quant = static_cast<uint8_t*>(alt_quant_result.d_quantized);
                            d_alt_input = d_alt_quant;
                            alt_compress_size = alt_quant_result.quantized_bytes;
                        }
                    }
                }

                // Apply shuffle
                unsigned int alt_shuf_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(alt_preproc);
                if (alt_shuf_size > 0) {
                    uint8_t* shuf = byte_shuffle_simple(
                        d_alt_input, alt_compress_size, alt_shuf_size,
                        gpucompress::SHUFFLE_CHUNK_SIZE, stream);
                    if (shuf) {
                        d_alt_shuf = shuf;
                        d_alt_input = d_alt_shuf;
                    }
                }

                /* No sync needed — compression runs on the same stream (H2 fix) */

                /* N1 fix: use cached manager for exploration alternatives */
                CompressionAlgorithm alt_internal =
                    gpucompress::toInternalAlgorithm(alt_algo);
                auto* alt_comp = gpucompress::getCachedCompManager(ctx, alt_internal);

                size_t alt_comp_size = 0;

                if (alt_comp) {
                    uint8_t* d_alt_out = nullptr;
                    try {
                        CompressionConfig alt_cc =
                            alt_comp->configure_compression(alt_compress_size);
                        size_t alt_max = alt_cc.max_compressed_buffer_size;
                        if (cudaMalloc(&d_alt_out, alt_max) == cudaSuccess) {
                            float alt_ct_ms = 0.0f;
                            if (timing_ok) cudaEventRecord(ctx->t_start, stream);

                            alt_comp->compress(d_alt_input, d_alt_out, alt_cc);

                            if (timing_ok) {
                                cudaEventRecord(ctx->t_stop, stream);
                                cudaEventSynchronize(ctx->t_stop);
                                cudaEventElapsedTime(&alt_ct_ms, ctx->t_start, ctx->t_stop);
                            }

                            alt_comp_size = alt_comp->get_compressed_output_size(d_alt_out);

                            double alt_ratio = static_cast<double>(input_size) /
                                               static_cast<double>(alt_comp_size);

                            // --- Round-trip decompress + PSNR ---
                            double alt_decomp_ms = 0.0;
                            double alt_psnr = 0.0;
                            {
                                {
                                    // Decompress on GPU
                                    auto rt_decomp = createDecompressionManager(d_alt_out, stream);
                                    if (rt_decomp) {
                                        DecompressionConfig rt_dc = rt_decomp->configure_decompression(d_alt_out);
                                        size_t rt_decomp_size = rt_dc.decomp_data_size;
                                        uint8_t* d_rt_decompressed = nullptr;
                                        if (cudaMalloc(&d_rt_decompressed, rt_decomp_size) == cudaSuccess) {
                                            if (timing_ok) cudaEventRecord(ctx->t_start, stream);

                                            try {
                                                rt_decomp->decompress(d_rt_decompressed, d_alt_out, rt_dc);
                                            } catch (...) {
                                                // Decompress failed, leave decomp_time and psnr at 0
                                                cudaFree(d_rt_decompressed);
                                                goto skip_roundtrip;
                                            }

                                            // Apply unshuffle if needed
                                            uint8_t* d_rt_result = d_rt_decompressed;
                                            uint8_t* d_rt_unshuf = nullptr;
                                            if (alt_shuf_size > 0) {
                                                d_rt_unshuf = byte_unshuffle_simple(
                                                    d_rt_decompressed, rt_decomp_size,
                                                    alt_shuf_size, gpucompress::SHUFFLE_CHUNK_SIZE,
                                                    stream);
                                                if (d_rt_unshuf) d_rt_result = d_rt_unshuf;
                                            }

                                            // Apply dequantization if needed
                                            void* d_rt_dequant = nullptr;
                                            if (d_alt_quant && alt_quant_result.isValid()) {
                                                QuantizationResult rt_qr;
                                                rt_qr.scale_factor = alt_quant_result.scale_factor;
                                                rt_qr.data_min = alt_quant_result.data_min;
                                                rt_qr.data_max = alt_quant_result.data_max;
                                                rt_qr.error_bound = alt_quant_result.error_bound;
                                                rt_qr.type = alt_quant_result.type;
                                                rt_qr.actual_precision = alt_quant_result.actual_precision;
                                                rt_qr.num_elements = input_size / sizeof(float);
                                                rt_qr.original_element_size = sizeof(float);
                                                rt_qr.d_quantized = d_rt_result;
                                                rt_qr.quantized_bytes = rt_decomp_size;
                                                d_rt_dequant = dequantize_simple(d_rt_result, rt_qr, stream);
                                                if (d_rt_dequant) d_rt_result = static_cast<uint8_t*>(d_rt_dequant);
                                            }

                                            if (timing_ok) {
                                                cudaEventRecord(ctx->t_stop, stream);
                                                cudaEventSynchronize(ctx->t_stop);
                                                float dt_ms = 0.0f;
                                                cudaEventElapsedTime(&dt_ms, ctx->t_start, ctx->t_stop);
                                                alt_decomp_ms = static_cast<double>(dt_ms);
                                            }

                                            // Compute PSNR (CPU-side for simplicity)
                                            if (!d_alt_quant) {
                                                // Lossless: hardcode 120.0
                                                alt_psnr = 120.0;
                                            } else {
                                                // Lossy: copy both original and decompressed to host, compute MSE
                                                size_t num_floats = input_size / sizeof(float);
                                                std::vector<float> h_orig(num_floats);
                                                std::vector<float> h_dec(num_floats);
                                                GC_LOG("[XFER D→H] explore PSNR: original data (%zu B)\n", input_size);
                                                XFER_TRACK("explore PSNR: D->H original (UNNECESSARY if GPU PSNR kernel existed)", input_size, cudaMemcpyDeviceToHost);
                                                cudaMemcpy(h_orig.data(), d_input, input_size, cudaMemcpyDeviceToHost);
                                                GC_LOG("[XFER D→H] explore PSNR: decompressed data (%zu B)\n", input_size);
                                                XFER_TRACK("explore PSNR: D->H decompressed (UNNECESSARY if GPU PSNR kernel existed)", input_size, cudaMemcpyDeviceToHost);
                                                cudaMemcpy(h_dec.data(), d_rt_result, input_size, cudaMemcpyDeviceToHost);

                                                double mse = 0.0;
                                                float dmin = h_orig[0], dmax = h_orig[0];
                                                for (size_t fi = 0; fi < num_floats; fi++) {
                                                    double diff = static_cast<double>(h_orig[fi]) - static_cast<double>(h_dec[fi]);
                                                    mse += diff * diff;
                                                    if (h_orig[fi] < dmin) dmin = h_orig[fi];
                                                    if (h_orig[fi] > dmax) dmax = h_orig[fi];
                                                }
                                                mse /= static_cast<double>(num_floats);
                                                double range = static_cast<double>(dmax) - static_cast<double>(dmin);
                                                if (mse > 0.0 && range > 0.0) {
                                                    alt_psnr = 10.0 * log10(range * range / mse);
                                                } else {
                                                    alt_psnr = 120.0;
                                                }
                                                alt_psnr = fmin(alt_psnr, 120.0);
                                            }

                                            // Cleanup round-trip buffers
                                            if (d_rt_dequant) cudaFree(d_rt_dequant);
                                            if (d_rt_unshuf) cudaFree(d_rt_unshuf);
                                            cudaFree(d_rt_decompressed);
                                        }
                                    }
                                }
                            }
                            skip_roundtrip:

                            explored_samples.push_back({alt_action, alt_ratio,
                                                        static_cast<double>(alt_ct_ms),
                                                        alt_decomp_ms, alt_psnr});

                            GC_LOG("[EXPLORE]   alt %d/%d: action%d (%s%s%s) "
                                    "ratio=%.3f comp=%.2fms%s\n",
                                    i, K, alt_action,
                                    ALGORITHM_NAMES[alt.algorithm + 1],
                                    alt.shuffle_size > 0 ? "+shuf" : "",
                                    alt.use_quantization ? "+quant" : "",
                                    alt_ratio, alt_ct_ms,
                                    (alt_ratio > best_ratio) ? " << NEW BEST" : "");

                            // Track if this is better than current best
                            if (alt_ratio > best_ratio) {
                                best_ratio = alt_ratio;
                                nn_action = alt_action;  // update so diagnostics & SGD see the real winner
                                g_last_nn_action.store(alt_action);
                                diag_compression_ms = alt_ct_ms;

                                // Copy better result to output
                                size_t header_sz = GPUCOMPRESS_HEADER_SIZE;
                                size_t alt_total = header_sz + alt_comp_size;
                                if (alt_total <= *output_size) {
                                    // Build header for alternative
                                    CompressionHeader alt_hdr;
                                    alt_hdr.magic = COMPRESSION_MAGIC;
                                    alt_hdr.version = COMPRESSION_HEADER_VERSION;
                                    alt_hdr.shuffle_element_size = alt_shuf_size;
                                    alt_hdr.original_size = input_size;
                                    alt_hdr.compressed_size = alt_comp_size;

                                    if (d_alt_quant && alt_quant_result.isValid()) {
                                        alt_hdr.setQuantizationFlags(
                                            static_cast<uint32_t>(alt_quant_result.type),
                                            alt_quant_result.actual_precision, true);
                                        alt_hdr.quant_error_bound = alt_quant_result.error_bound;
                                        alt_hdr.quant_scale = alt_quant_result.scale_factor;
                                        alt_hdr.data_min = alt_quant_result.data_min;
                                        alt_hdr.data_max = alt_quant_result.data_max;
                                    } else {
                                        alt_hdr.quant_flags = 0;
                                        alt_hdr.quant_error_bound = 0.0;
                                        alt_hdr.quant_scale = 0.0;
                                        alt_hdr.data_min = 0.0;
                                        alt_hdr.data_max = 0.0;
                                    }
                                    alt_hdr.setAlgorithmId((uint8_t)alt_algo);

                                    // Write header + compressed data to host output
                                    memcpy(output, &alt_hdr, sizeof(CompressionHeader));
                                    GC_LOG("[XFER D→H] explore winner: alt compressed payload (%zu B)\n", alt_comp_size);
                                    XFER_TRACK("explore winner: D->H alt payload", alt_comp_size, cudaMemcpyDeviceToHost);
                                    cudaMemcpy(
                                        static_cast<uint8_t*>(output) + header_sz,
                                        d_alt_out, alt_comp_size,
                                        cudaMemcpyDeviceToHost);
                                    *output_size = alt_total;
                                    total_size = alt_total;
                                    compressed_size = alt_comp_size;
                                    actual_ratio = static_cast<double>(input_size) / static_cast<double>(compressed_size);
                                    algo_to_use = alt_algo;
                                    preproc_to_use = alt_preproc;
                                }
                            }
                        }
                    } catch (...) {
                        // Compression failed for this config, skip it
                    }
                    if (d_alt_out) cudaFree(d_alt_out);
                }

                if (d_alt_quant) cudaFree(d_alt_quant);
                if (d_alt_shuf) cudaFree(d_alt_shuf);
            }

            // Summary: did exploration find something better?
            gpucompress_algorithm_t final_algo = algo_to_use;
            unsigned int final_preproc = preproc_to_use;
            double final_ratio = static_cast<double>(input_size) / static_cast<double>(compressed_size);
            if (final_algo != static_cast<gpucompress_algorithm_t>(primary_dec.algorithm + 1) ||
                final_preproc != (primary_dec.shuffle_size > 0 ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0u)) {
                GC_LOG("[EXPLORE] >> Switched to %s%s ratio=%.3f (was %.3f)\n",
                        gpucompress_algorithm_name(final_algo),
                        (final_preproc & GPUCOMPRESS_PREPROC_SHUFFLE_4) ? "+shuf" : "",
                        final_ratio, actual_ratio);
            } else {
                GC_LOG("[EXPLORE] >> Kept primary (ratio=%.3f)\n", actual_ratio);
            }

        }

        if (exploration_triggered) {
            diag_exploration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_explore_start).count();
        }

        // Online reinforcement: trigger GPU SGD when cost prediction error exceeds threshold.
        auto t_sgd_start = std::chrono::steady_clock::now();
        if (error_pct > static_cast<double>(g_reinforce_mape_threshold) && d_stats_ptr) {
            // Sort explored samples by ratio (descending) and feed top 3 to GPU SGD.
            std::sort(explored_samples.begin(), explored_samples.end(),
                      [](const ExploredResult& a, const ExploredResult& b) {
                          return a.ratio > b.ratio;
                      });
            size_t sgd_limit = std::min(explored_samples.size(), static_cast<size_t>(3));

            SGDSample sgd_samples[NN_MAX_SGD_SAMPLES];
            for (size_t ei = 0; ei < sgd_limit; ei++) {
                sgd_samples[ei].action = explored_samples[ei].action;
                sgd_samples[ei].actual_ratio = static_cast<float>(explored_samples[ei].ratio);
                sgd_samples[ei].actual_comp_time = static_cast<float>(explored_samples[ei].comp_time_ms);
                sgd_samples[ei].actual_decomp_time = static_cast<float>(explored_samples[ei].decomp_time_ms);
                sgd_samples[ei].actual_psnr = static_cast<float>(explored_samples[ei].psnr);
            }

            float sgd_grad_norm = 0.0f;
            int sgd_clipped = 0, sgd_count = 0;
            {
                std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                if (gpucompress::runNNSGDCtx(d_stats_ptr, sgd_samples,
                        static_cast<int>(sgd_limit), input_size, cfg.error_bound,
                        g_reinforce_lr, ctx,
                        &sgd_grad_norm, &sgd_clipped, &sgd_count) == 0) {
                    sgd_fired = true;
                }
            }
        }
        if (sgd_fired) {
            diag_sgd_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_sgd_start).count();
        }
    }

    // Now free d_input
    cudaFree(d_input);

    // Fill stats if requested
    if (stats != nullptr) {
        stats->original_size = input_size;
        stats->compressed_size = total_size;
        stats->compression_ratio = static_cast<double>(input_size) /
            static_cast<double>(total_size > 0 ? total_size : 1);
        stats->entropy_bits = entropy;
        stats->mad = mad;
        stats->second_derivative = second_derivative;
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

    /* Update globals for query after HDF5 filter path */
    g_last_nn_action.store(nn_action);
    g_last_nn_original_action.store(nn_original_action);
    g_last_exploration_triggered.store(exploration_triggered ? 1 : 0);
    g_last_sgd_fired.store(sgd_fired ? 1 : 0);

    /* Append to per-chunk history — grow array as needed */
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
            h->nn_action             = nn_action;
            h->nn_original_action    = nn_original_action;
            h->exploration_triggered = exploration_triggered ? 1 : 0;
            h->sgd_fired             = sgd_fired ? 1 : 0;
            h->nn_inference_ms       = diag_nn_inference_ms;
            h->stats_ms             = diag_stats_ms;
            h->preprocessing_ms      = diag_preprocessing_ms;
            h->compression_ms        = diag_compression_ms;
            h->exploration_ms        = diag_exploration_ms;
            h->sgd_update_ms         = diag_sgd_ms;
            h->actual_ratio          = (compressed_size > 0)
                ? static_cast<float>(input_size) / static_cast<float>(compressed_size)
                : 0.0f;
            h->predicted_ratio       = predicted_ratio;
            h->predicted_comp_time   = predicted_comp_time;
            h->predicted_decomp_time = predicted_decomp_time;
            h->predicted_psnr        = predicted_psnr;
            h->decompression_ms      = primary_decomp_time_ms;  /* measured at write when RL active, updated during read */
        }
    }

    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (input == nullptr || output == nullptr || output_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (input_size < GPUCOMPRESS_HEADER_SIZE) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    cudaStream_t stream = g_default_stream;
    cudaError_t cuda_err;

    // Read header from input
    CompressionHeader header;
    memcpy(&header, input, sizeof(CompressionHeader));

    // Validate header (checks both magic and version range)
    if (!header.isValid()) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    // Handle zero compressed_size
    size_t compressed_size = header.compressed_size;
    if (compressed_size == 0) {
        if (header.original_size == 0) {
            *output_size = 0;
            return GPUCOMPRESS_SUCCESS;
        }
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    // Check output buffer size
    if (header.original_size > *output_size) {
        *output_size = header.original_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    // Validate that input_size can hold header + claimed compressed data
    size_t header_size = GPUCOMPRESS_HEADER_SIZE;
    size_t header_plus_compressed = header_size + compressed_size;
    if (header_plus_compressed < header_size || input_size < header_plus_compressed) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    uint8_t* d_compressed_data = nullptr;
    cuda_err = cudaMalloc(&d_compressed_data, compressed_size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy only the compressed payload to GPU (skip header)
    const uint8_t* host_payload = static_cast<const uint8_t*>(input) + header_size;
    GC_LOG("[XFER H→D] decompress: compressed payload (%zu B)\n", compressed_size);
    XFER_TRACK("decompress: H->D compressed payload", compressed_size, cudaMemcpyHostToDevice);
    cuda_err = cudaMemcpyAsync(d_compressed_data, host_payload, compressed_size,
                               cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Create decompression manager (auto-detects algorithm)
    auto decompressor = createDecompressionManager(d_compressed_data, stream);
    if (!decompressor) {
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_DECOMPRESSION;
    }

    // Configure decompression
    DecompressionConfig decomp_config = decompressor->configure_decompression(d_compressed_data);
    size_t decompressed_size = decomp_config.decomp_data_size;

    // Allocate GPU buffer for decompressed data
    uint8_t* d_decompressed = nullptr;
    cuda_err = cudaMalloc(&d_decompressed, decompressed_size);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Decompress
    try {
        decompressor->decompress(d_decompressed, d_compressed_data, decomp_config);
    } catch (...) {
        cudaFree(d_decompressed);
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_DECOMPRESSION;
    }

    // Apply unshuffle if needed
    uint8_t* d_result = d_decompressed;
    uint8_t* d_unshuffled = nullptr;

    if (header.hasShuffleApplied()) {
        d_unshuffled = byte_unshuffle_simple(
            d_decompressed,
            decompressed_size,
            header.shuffle_element_size,
            gpucompress::SHUFFLE_CHUNK_SIZE,
            stream
        );

        if (d_unshuffled != nullptr) {
            d_result = d_unshuffled;
        } else {
            fprintf(stderr, "gpucompress ERROR: byte unshuffle failed during decompression "
                    "(element_size=%u, size=%zu)\n", header.shuffle_element_size, decompressed_size);
            cudaFree(d_decompressed);
            cudaFree(d_compressed_data);
            return GPUCOMPRESS_ERROR_DECOMPRESSION;
        }
    }

    // Apply dequantization if needed
    void* d_dequantized = nullptr;

    if (header.hasQuantizationApplied()) {
        // Reconstruct quantization result from header
        QuantizationResult quant_result;
        quant_result.scale_factor = header.quant_scale;
        quant_result.data_min = header.data_min;
        quant_result.data_max = header.data_max;
        quant_result.error_bound = header.quant_error_bound;
        quant_result.type = static_cast<QuantizationType>(header.getQuantizationType());
        quant_result.actual_precision = header.getQuantizationPrecision();
        quant_result.num_elements = header.original_size / sizeof(float);
        quant_result.original_element_size = sizeof(float);
        quant_result.d_quantized = d_result;
        quant_result.quantized_bytes = decompressed_size;

        d_dequantized = dequantize_simple(d_result, quant_result, stream);

        if (d_dequantized != nullptr) {
            if (d_unshuffled != nullptr) {
                cudaFree(d_unshuffled);
                d_unshuffled = nullptr;
            }
            d_result = static_cast<uint8_t*>(d_dequantized);
        } else {
            fprintf(stderr, "gpucompress ERROR: dequantization failed during decompression "
                    "(precision=%d, num_elements=%zu)\n",
                    quant_result.actual_precision, quant_result.num_elements);
            if (d_unshuffled) cudaFree(d_unshuffled);
            cudaFree(d_decompressed);
            cudaFree(d_compressed_data);
            return GPUCOMPRESS_ERROR_DECOMPRESSION;
        }
    }

    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        if (d_dequantized) cudaFree(d_dequantized);
        if (d_unshuffled) cudaFree(d_unshuffled);
        cudaFree(d_decompressed);
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Copy result to host
    GC_LOG("[XFER D→H] decompress: result (%" PRIu64 " B)\n", (uint64_t)header.original_size);
    XFER_TRACK("decompress: D->H decompressed result", header.original_size, cudaMemcpyDeviceToHost);
    cuda_err = cudaMemcpyAsync(output, d_result, header.original_size,
                               cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        if (d_dequantized) cudaFree(d_dequantized);
        if (d_unshuffled) cudaFree(d_unshuffled);
        cudaFree(d_decompressed);
        cudaFree(d_compressed_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cuda_err = cudaStreamSynchronize(stream);

    // Cleanup
    if (d_dequantized) cudaFree(d_dequantized);
    if (d_unshuffled) cudaFree(d_unshuffled);
    cudaFree(d_decompressed);
    cudaFree(d_compressed_data);

    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *output_size = header.original_size;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" size_t gpucompress_max_compressed_size(size_t input_size) {
    // Worst case: no compression + header + some margin
    // Guard against overflow: each addend must not wrap
    size_t margin = input_size / 8;
    size_t sum = input_size + margin;
    if (sum < input_size) return 0;  // overflow
    sum += 1024;
    if (sum < 1024) return 0;       // overflow
    sum += GPUCOMPRESS_HEADER_SIZE;
    if (sum < GPUCOMPRESS_HEADER_SIZE) return 0;  // overflow
    return sum;
}

extern "C" gpucompress_error_t gpucompress_get_original_size(
    const void* compressed,
    size_t* original_size
) {
    if (compressed == nullptr || original_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    const CompressionHeader* header = static_cast<const CompressionHeader*>(compressed);

    if (header->magic != COMPRESSION_MAGIC) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    *original_size = header->original_size;
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Entropy Calculation
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_calculate_entropy(
    const void* data,
    size_t size,
    double* entropy_out
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (data == nullptr || entropy_out == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    cudaStream_t stream = g_default_stream;
    cudaError_t cuda_err;

    // Allocate GPU buffer
    uint8_t* d_data = nullptr;
    cuda_err = cudaMalloc(&d_data, size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy to GPU
    GC_LOG("[XFER H→D] compute_stats: input data (%zu B)\n", size);
    XFER_TRACK("calculate_entropy: H->D input data", size, cudaMemcpyHostToDevice);
    cuda_err = cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Calculate entropy
    double entropy = gpucompress_entropy_gpu_impl(d_data, size, stream);

    cudaFree(d_data);

    if (entropy < 0.0) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *entropy_out = entropy;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_calculate_entropy_gpu(
    const void* d_data,
    size_t size,
    double* entropy_out,
    void* stream
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (d_data == nullptr || entropy_out == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : g_default_stream;
    double entropy = gpucompress_entropy_gpu_impl(d_data, size, cuda_stream);

    if (entropy < 0.0) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *entropy_out = entropy;
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Statistical Analysis
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_compute_stats(
    const void* data,
    size_t size,
    double* entropy,
    double* mad,
    double* second_derivative
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (data == nullptr || entropy == nullptr || mad == nullptr ||
        second_derivative == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (size % sizeof(float) != 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    // Upload to GPU and compute stats via GPU kernels
    void* d_data = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_data, size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }
    XFER_TRACK("compute_stats: H->D input data", size, cudaMemcpyHostToDevice);
    cuda_err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    int rc = gpucompress::runStatsOnlyPipeline(d_data, size, g_default_stream,
                                                entropy, mad, second_derivative);
    cudaFree(d_data);
    if (rc != 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Neural Network Management
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

/* ============================================================
 * Utility Functions
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
    /* keep allocated buffer for reuse */
}

extern "C" void gpucompress_force_algorithm_reset(void) {
    std::lock_guard<std::mutex> lock(g_force_algo_mutex);
    g_force_algo_count = 0;
    g_force_algo_idx.store(0);
}

extern "C" void gpucompress_force_algorithm_push(int algorithm, int shuffle,
                                                  int quant, double error_bound) {
    std::lock_guard<std::mutex> lock(g_force_algo_mutex);
    if (g_force_algo_count >= g_force_algo_cap) {
        int new_cap = (g_force_algo_cap == 0) ? 256 : g_force_algo_cap * 2;
        auto* p = static_cast<ForcedAlgo*>(
            realloc(g_force_algo_queue, (size_t)new_cap * sizeof(ForcedAlgo)));
        if (!p) return;
        g_force_algo_queue = p;
        g_force_algo_cap = new_cap;
    }
    g_force_algo_queue[g_force_algo_count++] = {algorithm, shuffle, quant, error_bound};
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

extern "C" const char* gpucompress_algorithm_name(gpucompress_algorithm_t algorithm) {
    int idx = static_cast<int>(algorithm);
    if (idx >= 0 && idx <= 8) {
        return ALGORITHM_NAMES[idx];
    }
    return "unknown";
}

extern "C" gpucompress_algorithm_t gpucompress_algorithm_from_string(const char* name) {
    if (name == nullptr) {
        return GPUCOMPRESS_ALGO_LZ4;
    }

    for (int i = 0; i <= 8; i++) {
        if (strcasecmp(name, ALGORITHM_NAMES[i]) == 0) {
            return static_cast<gpucompress_algorithm_t>(i);
        }
    }

    return GPUCOMPRESS_ALGO_LZ4;
}

extern "C" const char* gpucompress_error_string(gpucompress_error_t error) {
    int idx = -static_cast<int>(error);
    if (idx >= 0 && idx <= 9) {
        return ERROR_MESSAGES[idx];
    }
    return "Unknown error";
}

extern "C" const char* gpucompress_version(void) {
    return VERSION_STRING;
}

/* ============================================================
 * Active Learning API
 * ============================================================ */

/* ============================================================
 * Online Learning API (new granular controls)
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

/* ============================================================
 * Backward-compatible Active Learning API
 * ============================================================ */

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
    if (threshold > 0.0 && threshold < 1.0) {
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

extern "C" void gpucompress_set_verbose(int enable) {
    g_gc_verbose = (enable != 0);
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

extern "C" gpucompress_error_t gpucompress_reload_nn(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Load new weights — loadNNFromBinary handles the atomic pointer swap
    // and frees the old buffer after draining in-flight kernels.
    // No cleanupNN() call needed; that would free inference/SGD buffers
    // and create a window where d_nn_weights is null.
    int result = gpucompress_nn_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_INVALID_INPUT;
}

/* ============================================================
 * GPU Memory API (stubs for now)
 * ============================================================ */

extern "C" int gpucompress_is_device_ptr(const void* ptr) {
    if (!ptr) return 0;
    struct cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError(); /* clear error state */
        return 0;
    }
    return (attrs.type == cudaMemoryTypeDevice) ? 1 : 0;
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

    /* Acquire a per-slot context (blocks until one is free) */
    ContextGuard guard{gpucompress::acquireCompContext()};
    if (!guard.ctx) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    CompContext* ctx = guard.ctx;
    cudaStream_t stream = ctx->stream;
    cudaStream_t caller_stream = stream_arg ? static_cast<cudaStream_t>(stream_arg) : nullptr;
    if (caller_stream) {
        cudaEventRecord(ctx->t_start, caller_stream);
        cudaStreamWaitEvent(stream, ctx->t_start, 0);
    }

    /* d_input is already on GPU — no H→D copy */
    gpucompress_algorithm_t algo_to_use = cfg.algorithm;
    unsigned int preproc_to_use = cfg.preprocessing;

    /* Variables for NN/exploration/SGD (mirrors gpucompress_compress) */
    double entropy = 0.0, mad = 0.0, second_derivative = 0.0;
    bool nn_was_used = false, sgd_fired = false, exploration_triggered = false;
    int nn_action = 0, nn_original_action = -1;
    float predicted_ratio = 0.0f, predicted_comp_time = 0.0f,
          predicted_decomp_time = 0.0f, predicted_psnr = 0.0f;
    int top_actions[32] = {0};
    bool is_ood = false;
    AutoStatsGPU* d_stats_ptr = nullptr;

    /* Per-chunk timing breakdown */
    bool timing_ok = (ctx->t_start != nullptr && ctx->t_stop != nullptr);
    float diag_nn_inference_ms  = 0.0f;
    float diag_stats_ms         = 0.0f;
    float diag_preprocessing_ms = 0.0f;
    float diag_compression_ms   = 0.0f;
    float diag_exploration_ms   = 0.0f;
    float diag_sgd_ms           = 0.0f;

    if (algo_to_use == GPUCOMPRESS_ALGO_AUTO) {
        /* Check force-algorithm queue first — oracle/exhaustive uses this
         * to inject per-chunk algorithm choices without NN inference. */
        {
            std::lock_guard<std::mutex> fq_lock(g_force_algo_mutex);
            int fidx = g_force_algo_idx.fetch_add(1);
            if (fidx < g_force_algo_count && g_force_algo_queue) {
                algo_to_use = static_cast<gpucompress_algorithm_t>(
                    g_force_algo_queue[fidx].algorithm);
                preproc_to_use = 0;
                if (g_force_algo_queue[fidx].shuffle)
                    preproc_to_use |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                if (g_force_algo_queue[fidx].quant) {
                    preproc_to_use |= GPUCOMPRESS_PREPROC_QUANTIZE;
                    cfg.error_bound = g_force_algo_queue[fidx].error_bound;
                }
                goto skip_nn;
            }
            /* No forced entry — revert the index bump and fall through to NN */
            g_force_algo_idx.fetch_sub(1);
        }

        size_t num_elements = input_size / sizeof(float);
        int action = 0;
        int rc = -1;

        if (num_elements > 0 && gpucompress_nn_is_loaded_impl()) {
            bool nn_timed = false;

            /* Stats timing */
            cudaEventRecord(ctx->stats_start, stream);

            /* d_input is already on GPU — use ctx-aware overload */
            d_stats_ptr = gpucompress::runStatsKernelsNoSync(d_input, input_size, stream, ctx);

            if (d_stats_ptr && stats != nullptr) {
                /* Enqueue stats D→H copies now (before inference kernel) so
                 * the single cudaStreamSynchronize inside runNNFusedInferenceCtx
                 * drains both stats copies and inference D→H in one sync. */
                XFER_TRACK("compress_gpu/auto: D->H stats entropy", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&entropy, &d_stats_ptr->entropy, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
                XFER_TRACK("compress_gpu/auto: D->H stats mad", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&mad, &d_stats_ptr->mad_normalized, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
                XFER_TRACK("compress_gpu/auto: D->H stats deriv", sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&second_derivative, &d_stats_ptr->deriv_normalized, sizeof(double),
                                cudaMemcpyDeviceToHost, stream);
            }
            cudaEventRecord(ctx->stats_stop, stream);

            if (d_stats_ptr) {
                float* p_ratio = &predicted_ratio;  // always capture — no extra cost
                float* p_comp_time = &predicted_comp_time;
                float* p_decomp_time = &predicted_decomp_time;
                float* p_psnr = &predicted_psnr;
                int* p_top = g_online_learning_enabled ? top_actions : nullptr;
                int fused_ood = 0;

                /* NN timing — nn_start recorded here, nn_stop inside runNNFusedInferenceCtx */
                cudaEventRecord(ctx->nn_start, stream);
                action = gpucompress::runNNFusedInferenceCtx(
                    d_stats_ptr, input_size, cfg.error_bound, stream, ctx,
                    &action, p_ratio, p_comp_time, p_decomp_time, p_psnr,
                    g_online_learning_enabled ? &fused_ood : nullptr, p_top,
                    ctx->nn_stop);
                nn_timed = true;
                rc = (action >= 0) ? 0 : -1;
                if (rc == 0) is_ood = (fused_ood != 0);
            }

            if (rc == 0) {
                nn_was_used = true;
                nn_action = action;
                nn_original_action = action;
                g_last_nn_action.store(action);
            }

            /* Events already synchronized by runNNFusedInferenceCtx's internal sync */
            if (nn_timed) {
                cudaEventElapsedTime(&diag_nn_inference_ms, ctx->nn_start, ctx->nn_stop);
                cudaEventElapsedTime(&diag_stats_ms, ctx->stats_start, ctx->stats_stop);
            }
        }

        if (rc == 0) {
            gpucompress::DecodedAction decoded = gpucompress::decodeAction(action);
            algo_to_use = static_cast<gpucompress_algorithm_t>(decoded.algorithm + 1);
            preproc_to_use = 0;
            if (decoded.shuffle_size > 0)
                preproc_to_use |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
            if (decoded.use_quantization)
                preproc_to_use |= GPUCOMPRESS_PREPROC_QUANTIZE;
        } else {
            fprintf(stderr, "gpucompress ERROR: ALGO_AUTO requested but NN inference failed "
                    "(weights not loaded or inference error). Load weights via gpucompress_init() "
                    "or use an explicit algorithm.\n");
            return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
        }
    }
skip_nn:

    /* Apply preprocessing on GPU (same kernels as host path) */
    const uint8_t* d_compress_input = static_cast<const uint8_t*>(d_input);
    size_t compress_input_size = input_size;
    uint8_t* d_quantized = nullptr;
    uint8_t* d_shuffled  = nullptr;
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
                ctx->d_range_min, ctx->d_range_max, stream);
            if (quant_result.isValid()) {
                d_quantized = static_cast<uint8_t*>(quant_result.d_quantized);
                d_compress_input = d_quantized;
                compress_input_size = quant_result.quantized_bytes;
            } else {
                fprintf(stderr, "gpucompress ERROR: quantization failed but was requested "
                        "(error_bound=%.6e, num_elements=%zu)\n", cfg.error_bound, num_elements);
                return GPUCOMPRESS_ERROR_COMPRESSION;
            }
        }
    }

    unsigned int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(preproc_to_use);
    if (shuffle_size > 0) {
        d_shuffled = byte_shuffle_simple(
            const_cast<uint8_t*>(d_compress_input), compress_input_size,
            shuffle_size, gpucompress::SHUFFLE_CHUNK_SIZE, stream);
        if (d_shuffled) {
            d_compress_input = d_shuffled;
        } else {
            fprintf(stderr, "gpucompress ERROR: byte shuffle failed but was requested "
                    "(element_size=%u, input_size=%zu)\n", shuffle_size, compress_input_size);
            if (d_quantized) cudaFree(d_quantized);
            return GPUCOMPRESS_ERROR_COMPRESSION;
        }
    }

    /* No sync needed — subsequent compression runs on the same stream,
     * so GPU ordering guarantees preprocessing completes first.
     * Any async errors surface at the final stream sync. (H2 fix) */
    cudaError_t cuda_err = cudaSuccess;

    if (d_quantized || d_shuffled) {
        diag_preprocessing_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_preproc_start).count();
    }

    /* N1 fix: use cached compression manager (avoids per-call workspace alloc) */
    CompressionAlgorithm internal_algo = gpucompress::toInternalAlgorithm(algo_to_use);
    auto* compressor = gpucompress::getCachedCompManager(ctx, internal_algo);
    if (!compressor) {
        fprintf(stderr, "gpucompress ERROR: getCachedCompManager failed for algo=%d "
                "(GPU OOM or unsupported algo, input_size=%zu)\n",
                (int)algo_to_use, compress_input_size);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    CompressionConfig comp_config = compressor->configure_compression(compress_input_size);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;

    if (max_compressed_size == 0) {
        fprintf(stderr, "gpucompress ERROR: configure_compression returned "
                "max_compressed_buffer_size=0 (algo=%d, input_size=%zu)\n",
                (int)algo_to_use, compress_input_size);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    size_t header_size = GPUCOMPRESS_HEADER_SIZE;
    size_t total_max_needed = header_size + max_compressed_size;
    if (total_max_needed < header_size) {
        // Overflow in header + compressed size addition
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    /* If the NN-chosen algorithm's worst-case buffer exceeds the caller's
     * allocation (configure_compression returns a conservative upper bound
     * that can be 2-4× input for CASCADED/ANS/BITCOMP), allocate a
     * temporary device buffer.  After compression the actual result is
     * D→D copied back — actual compressed sizes are always << max, so
     * they always fit.  This preserves the NN's algorithm choice. */
    uint8_t* d_out      = static_cast<uint8_t*>(d_output);
    uint8_t* d_temp_out = nullptr;
    uint8_t* d_comp_target;   /* where nvcomp writes payload */

    if (total_max_needed > *output_size) {
        if (cudaMalloc(&d_temp_out, total_max_needed) != cudaSuccess) {
            fprintf(stderr, "gpucompress ERROR: cudaMalloc failed for temp compression buffer "
                    "(needed=%zu bytes, algo=%d)\n", total_max_needed, (int)algo_to_use);
            if (d_quantized) cudaFree(d_quantized);
            if (d_shuffled)  cudaFree(d_shuffled);
            return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
        }
        d_comp_target = d_temp_out + header_size;
    } else {
        d_comp_target = d_out + header_size;
    }

    float primary_comp_time_ms = 0.0f;
    bool need_timing = timing_ok && (stats != nullptr || g_online_learning_enabled);
    if (need_timing) cudaEventRecord(ctx->t_start, stream);

    try {
        compressor->compress(d_compress_input, d_comp_target, comp_config);
    } catch (const std::exception& e) {
        fprintf(stderr, "gpucompress ERROR: nvcomp compress threw exception: %s "
                "(algo=%d, input_size=%zu)\n", e.what(), (int)algo_to_use, compress_input_size);
        if (d_temp_out)  cudaFree(d_temp_out);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    } catch (...) {
        fprintf(stderr, "gpucompress ERROR: nvcomp compress threw unknown exception "
                "(algo=%d, input_size=%zu)\n", (int)algo_to_use, compress_input_size);
        if (d_temp_out)  cudaFree(d_temp_out);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled)  cudaFree(d_shuffled);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    if (need_timing) {
        cudaEventRecord(ctx->t_stop, stream);
        cudaEventSynchronize(ctx->t_stop);
        cudaEventElapsedTime(&primary_comp_time_ms, ctx->t_start, ctx->t_stop);
    }
    diag_compression_ms = primary_comp_time_ms;

    size_t compressed_size = compressor->get_compressed_output_size(d_comp_target);
    size_t total_size      = header_size + compressed_size;

    // Measure primary decomp time (RL-gated: only when SGD can learn from it)
    float primary_decomp_time_ms = 0.0f;
    if (g_reinforce_lr > 0.0f && nn_was_used && timing_ok) {
        auto rt_decomp = createDecompressionManager(d_comp_target, stream);
        if (rt_decomp) {
            DecompressionConfig rt_dc = rt_decomp->configure_decompression(d_comp_target);
            size_t rt_decomp_size = rt_dc.decomp_data_size;
            uint8_t* d_rt_buf = nullptr;
            if (cudaMalloc(&d_rt_buf, rt_decomp_size) == cudaSuccess) {
                cudaEventRecord(ctx->t_start, stream);
                try {
                    rt_decomp->decompress(d_rt_buf, d_comp_target, rt_dc);
                    cudaEventRecord(ctx->t_stop, stream);
                    cudaEventSynchronize(ctx->t_stop);
                    cudaEventElapsedTime(&primary_decomp_time_ms, ctx->t_start, ctx->t_stop);
                } catch (...) {
                    // Decompress failed — leave at 0
                }
                cudaFree(d_rt_buf);
            }
        }
    }

    /* Build header and write to d_output[0..63] via H→D */
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

    if (d_temp_out) {
        /* Write header into temp buffer, then D→D copy full result to caller */
        XFER_TRACK("compress_gpu: H->D header to temp", sizeof(CompressionHeader), cudaMemcpyHostToDevice);
        cuda_err = cudaMemcpyAsync(d_temp_out, &header, sizeof(CompressionHeader),
                                   cudaMemcpyHostToDevice, stream);
        if (cuda_err == cudaSuccess) {
            XFER_TRACK("compress_gpu: D->D temp->output (header+payload)", total_size, cudaMemcpyDeviceToDevice);
            cuda_err = cudaMemcpyAsync(d_out, d_temp_out, total_size,
                                       cudaMemcpyDeviceToDevice, stream);
        }
        if (cuda_err == cudaSuccess)
            cuda_err = cudaStreamSynchronize(stream);
        cudaFree(d_temp_out);
    } else {
        /* S1 fix: header is 64B from non-pinned stack — cudaMemcpyAsync is
         * internally synchronous for non-pinned host memory, so the extra
         * cudaStreamSynchronize was redundant. Final sync at scope exit. */
        XFER_TRACK("compress_gpu: H->D header (64B struct from host)", sizeof(CompressionHeader), cudaMemcpyHostToDevice);
        cuda_err = cudaMemcpyAsync(d_out, &header, sizeof(CompressionHeader),
                                   cudaMemcpyHostToDevice, stream);
    }

    if (d_quantized) cudaFree(d_quantized);
    if (d_shuffled)  cudaFree(d_shuffled);

    if (cuda_err != cudaSuccess) return GPUCOMPRESS_ERROR_CUDA_FAILED;

    *output_size = total_size;

    /* ---- Exploration + SGD (mirrors gpucompress_compress, GPU-path adaptations noted) ---- */
    auto t_explore_start = std::chrono::steady_clock::now();

    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && nn_was_used &&
        g_online_learning_enabled) {
        double actual_ratio = static_cast<double>(input_size) /
                              static_cast<double>(compressed_size);

        // Cost-based prediction error (mirrors non-ctx path, uses measured decomp time)
        double ds = static_cast<double>(input_size);
        double bw = static_cast<double>(g_measured_bw_bytes_per_ms);
        double w0 = static_cast<double>(g_rank_w0);
        double w1 = static_cast<double>(g_rank_w1);
        double w2 = static_cast<double>(g_rank_w2);
        double pred_dt = static_cast<double>(predicted_decomp_time);
        double pred_ct = static_cast<double>(predicted_comp_time);
        double pred_r  = static_cast<double>(predicted_ratio);
        double act_ct  = static_cast<double>(primary_comp_time_ms);
        double act_dt  = (primary_decomp_time_ms > 0.0f)
                       ? static_cast<double>(primary_decomp_time_ms) : pred_dt;

        double actual_cost = w0 * act_ct + w1 * act_dt
                           + ((actual_ratio > 0.0) ? w2 * ds / (actual_ratio * bw) : 0.0);
        double predicted_cost = w0 * pred_ct + w1 * pred_dt
                              + ((pred_r > 0.0) ? w2 * ds / (pred_r * bw) : 0.0);
        double error_pct = (actual_cost > 0.0)
            ? std::abs(actual_cost - predicted_cost) / actual_cost : 0.0;

        struct ExploredResult { int action; double ratio; double comp_time_ms;
                                double decomp_time_ms; double psnr; };
        std::vector<ExploredResult> explored_samples;
        double primary_psnr = 120.0;
        if (d_quantized && quant_result.isValid()) {
            double range = quant_result.data_max - quant_result.data_min;
            if (range > 0.0 && quant_result.error_bound > 0.0) {
                primary_psnr = fmin(20.0 * log10(range / quant_result.error_bound), 120.0);
            }
        }
        explored_samples.push_back({nn_action, actual_ratio,
                                    static_cast<double>(primary_comp_time_ms),
                                    static_cast<double>(primary_decomp_time_ms),
                                    primary_psnr});

        if (g_exploration_enabled && error_pct > g_exploration_threshold) {
            exploration_triggered = true;
            int K;
            if (g_exploration_k_override > 0) {
                K = g_exploration_k_override;
            } else {
                K = 3;  // Top-3 alternatives (limits GPU memory pressure)
            }

            gpucompress::DecodedAction primary_dec = gpucompress::decodeAction(nn_action);
            GC_LOG("[EXPLORE-GPU] Chunk %zuB | primary=action%d (%s%s%s) ratio=%.3f | "
                    "cost_err=%.1f%% %s| K=%d\n",
                    input_size, nn_action,
                    ALGORITHM_NAMES[primary_dec.algorithm + 1],
                    primary_dec.shuffle_size > 0 ? "+shuf" : "",
                    primary_dec.use_quantization ? "+quant" : "",
                    actual_ratio,
                    error_pct * 100.0,
                    is_ood ? "OOD " : "",
                    K);

            double best_ratio = actual_ratio;

            /* ---- Parallel exploration: K alternatives on K separate streams ----
             * Phase 1: launch all K compressions in parallel.
             * Phase 2: sync all, read sizes, pick winner by ratio.
             * Phase 3: decompress+verify only the winner (if any). */

            constexpr int MAX_EXPLORE_K = 8;
            int actual_K = std::min(K, MAX_EXPLORE_K);

            struct ExploreSlot {
                int       action;
                int       algo;             /* gpucompress_algorithm_t */
                unsigned  preproc;
                unsigned  shuf_size;
                bool      has_quant;
                QuantizationResult quant_result;
                cudaStream_t       stream;
                uint8_t*  d_out;            /* compressed output */
                uint8_t*  d_quant;          /* quantized buffer (owned) */
                uint8_t*  d_shuf;           /* shuffled buffer (owned) */
                void*     d_range_min;      /* per-slot quantization range */
                void*     d_range_max;
                std::unique_ptr<nvcomp::nvcompManagerBase> comp_mgr;
                size_t    comp_size;        /* filled after sync */
                double    ratio;
                bool      valid;            /* compression succeeded */
            };

            ExploreSlot slots[MAX_EXPLORE_K];
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

                /* Create per-slot stream and output buffer */
                if (cudaStreamCreate(&s.stream) != cudaSuccess) continue;
                if (cudaMalloc(&s.d_out, explore_max_comp) != cudaSuccess) {
                    cudaStreamDestroy(s.stream); s.stream = nullptr; continue;
                }

                /* Per-slot quantization range buffers (8B each) */
                if (s.has_quant) {
                    if (cudaMalloc(&s.d_range_min, sizeof(double)) != cudaSuccess ||
                        cudaMalloc(&s.d_range_max, sizeof(double)) != cudaSuccess) {
                        cudaFree(s.d_out); cudaStreamDestroy(s.stream);
                        if (s.d_range_min) cudaFree(s.d_range_min);
                        continue;
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
                            s.comp_mgr->compress(d_alt_input, s.d_out, alt_cc);
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

                /* PSNR: analytical from error bound (no D→H needed) */
                double alt_psnr = 120.0;
                if (s.has_quant && s.quant_result.isValid()) {
                    double range = s.quant_result.data_max - s.quant_result.data_min;
                    if (range > 0.0 && s.quant_result.error_bound > 0.0)
                        alt_psnr = fmin(20.0 * log10(range / s.quant_result.error_bound), 120.0);
                }

                explored_samples.push_back({s.action, s.ratio, 0.0, 0.0, alt_psnr});

                GC_LOG("[EXPLORE-GPU]   slot %d: action%d ratio=%.3f%s\n",
                        k, s.action, s.ratio,
                        (s.ratio > best_ratio) ? " << NEW BEST" : "");

                if (s.ratio > best_ratio) {
                    best_ratio = s.ratio;
                    nn_action = s.action;
                    g_last_nn_action.store(s.action);

                    /* Write winner to output */
                    size_t hdr_sz = GPUCOMPRESS_HEADER_SIZE;
                    size_t alt_total = hdr_sz + s.comp_size;
                    if (alt_total <= *output_size) {
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

                        /* Copy winner to output on ctx->stream (sync with caller) */
                        cudaMemcpyAsync(d_out, &alt_hdr, sizeof(CompressionHeader),
                                        cudaMemcpyHostToDevice, stream);
                        cudaMemcpyAsync(d_out + hdr_sz, s.d_out, s.comp_size,
                                        cudaMemcpyDeviceToDevice, stream);
                        *output_size = alt_total;
                        total_size = alt_total;
                        compressed_size = s.comp_size;
                        actual_ratio = s.ratio;
                        algo_to_use = static_cast<gpucompress_algorithm_t>(s.algo);
                        preproc_to_use = s.preproc;
                    }
                }
            }

            /* ---- Phase 3: cleanup per-slot resources ---- */
            for (int k = 0; k < n_slots; k++) {
                ExploreSlot& s = slots[k];
                s.comp_mgr.reset();  /* release manager before destroying stream */
                if (s.d_quant)     cudaFree(s.d_quant);
                if (s.d_shuf)      cudaFree(s.d_shuf);
                if (s.d_out)       cudaFree(s.d_out);
                if (s.d_range_min) cudaFree(s.d_range_min);
                if (s.d_range_max) cudaFree(s.d_range_max);
                if (s.stream)      cudaStreamDestroy(s.stream);
            }
        }

        if (exploration_triggered) {
            diag_exploration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_explore_start).count();
        }

        /* GPU SGD: fire when cost prediction error exceeds threshold */
        auto t_sgd_start = std::chrono::steady_clock::now();
        if (error_pct > static_cast<double>(g_reinforce_mape_threshold) && d_stats_ptr) {
            std::sort(explored_samples.begin(), explored_samples.end(),
                      [](const ExploredResult& a, const ExploredResult& b) {
                          return a.ratio > b.ratio;
                      });
            size_t sgd_limit = std::min(explored_samples.size(), static_cast<size_t>(3));

            SGDSample sgd_samples[NN_MAX_SGD_SAMPLES];
            for (size_t ei = 0; ei < sgd_limit; ei++) {
                sgd_samples[ei].action = explored_samples[ei].action;
                sgd_samples[ei].actual_ratio = static_cast<float>(explored_samples[ei].ratio);
                sgd_samples[ei].actual_comp_time = static_cast<float>(explored_samples[ei].comp_time_ms);
                sgd_samples[ei].actual_decomp_time = static_cast<float>(explored_samples[ei].decomp_time_ms);
                sgd_samples[ei].actual_psnr = static_cast<float>(explored_samples[ei].psnr);
            }

            float sgd_grad_norm = 0.0f;
            int sgd_clipped = 0, sgd_count = 0;
            {
                std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                if (gpucompress::runNNSGDCtx(d_stats_ptr, sgd_samples,
                        static_cast<int>(sgd_limit), input_size, cfg.error_bound,
                        g_reinforce_lr, ctx,
                        &sgd_grad_norm, &sgd_clipped, &sgd_count) == 0) {
                    sgd_fired = true;
                }
            }
        }
        if (sgd_fired) {
            diag_sgd_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_sgd_start).count();
        }
    }

    /* Synchronize ctx->stream before releasing context — ensures d_output is fully written */
    cudaStreamSynchronize(stream);

    /* Fill stats */
    if (stats != nullptr) {
        stats->original_size = input_size;
        stats->compressed_size = total_size;
        stats->compression_ratio = static_cast<double>(input_size) /
            static_cast<double>(total_size > 0 ? total_size : 1);
        stats->entropy_bits = entropy;
        stats->mad = mad;
        stats->second_derivative = second_derivative;
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

    /* Update globals for query after HDF5 filter path */
    g_last_nn_action.store(nn_action);
    g_last_nn_original_action.store(nn_original_action);
    g_last_exploration_triggered.store(exploration_triggered ? 1 : 0);
    g_last_sgd_fired.store(sgd_fired ? 1 : 0);

    /* Append to per-chunk history — grow array as needed */
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
            h->nn_action             = nn_action;
            h->nn_original_action    = nn_original_action;
            h->exploration_triggered = exploration_triggered ? 1 : 0;
            h->sgd_fired             = sgd_fired ? 1 : 0;
            h->nn_inference_ms       = diag_nn_inference_ms;
            h->stats_ms             = diag_stats_ms;
            h->preprocessing_ms      = diag_preprocessing_ms;
            h->compression_ms        = diag_compression_ms;
            h->exploration_ms        = diag_exploration_ms;
            h->sgd_update_ms         = diag_sgd_ms;
            h->actual_ratio          = (compressed_size > 0)
                ? static_cast<float>(input_size) / static_cast<float>(compressed_size)
                : 0.0f;
            h->predicted_ratio       = predicted_ratio;
            h->predicted_comp_time   = predicted_comp_time;
            h->predicted_decomp_time = predicted_decomp_time;
            h->predicted_psnr        = predicted_psnr;
            h->decompression_ms      = primary_decomp_time_ms;  /* measured at write when RL active, updated during read */
        }
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

    GC_LOG("[decompress_gpu] algo_id=%d  shuffle=%s  orig=%zuB  comp=%zuB\n",
           header.getAlgorithmId(),
           header.hasShuffleApplied() ? "yes" : "no",
           (size_t)header.original_size,
           (size_t)header.compressed_size);

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
    XFER_TRACK("decompress_gpu: D->D result to output", header.original_size, cudaMemcpyDeviceToDevice);
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
