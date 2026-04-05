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
#include "api/gpucompress_state.hpp"
#include "compression/compression_factory.hpp"
#include "compression/compression_header.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

/* ============================================================
 * External Functions from CUDA files
 * ============================================================ */

extern "C" {
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


/** Cost-based ranking weights and measured bandwidth — extern-referenced from nn_gpu.cu */
float g_rank_w0 = 1.0f;                   // weight on compression time
float g_rank_w1 = 1.0f;                   // weight on decompression time
float g_rank_w2 = 1.0f;                   // weight on I/O cost (data_size / (ratio * bw))
float g_min_psnr_db = 0.0f;               // PSNR quality floor (0 = no filtering)
float g_measured_bw_bytes_per_ms = 5e6f;   // 5 GB/s = 5e6 bytes/ms (representative HPC storage)


/* ============================================================
 * Shared Global State (definitions — extern-declared in gpucompress_state.hpp)
 * ============================================================ */

std::atomic<bool> g_initialized{false};
std::mutex        g_init_mutex;
cudaStream_t      g_default_stream = nullptr;

std::atomic<bool> g_online_learning_enabled{false};
std::atomic<bool> g_exploration_enabled{false};
double            g_exploration_threshold = 0.20;
int               g_exploration_k_override = -1;

std::atomic<int>  g_last_nn_action{-1};
std::atomic<int>  g_last_nn_original_action{-1};
std::atomic<int>  g_last_exploration_triggered{0};
std::atomic<int>  g_last_sgd_fired{0};

gpucompress_chunk_diag_t* g_chunk_history      = nullptr;
int                       g_chunk_history_cap   = 0;
std::atomic<int>          g_chunk_history_count{0};
std::mutex                g_chunk_history_mutex;

std::atomic<int>  g_mgr_cache_hits{0};
std::atomic<int>  g_mgr_cache_misses{0};

std::mutex        g_sgd_mutex;

std::atomic<int> g_selection_mode{GPUCOMPRESS_SELECT_NN};
std::atomic<bool> g_best_mode{false};

bool g_debug_nn = false;
bool g_detailed_timing = false;

float g_reinforce_lr = 0.01f;
float g_reinforce_mape_threshold = 0.10f;

/* ============================================================
 * File-Private State (not shared across translation units)
 * ============================================================ */

namespace {

std::atomic<int> g_ref_count{0};
int g_cuda_device = 0;
const char* VERSION_STRING = "1.0.0";

/* CompContext pool moved to gpucompress_pool.cpp */

const char* ALGORITHM_NAMES[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

const char* ERROR_MESSAGES[] = {
    "Success", "Invalid input parameter", "CUDA operation failed",
    "Compression failed", "Decompression failed", "Out of memory",
    "Reserved", "Invalid compression header", "Library not initialized",
    "Output buffer too small"
};

} // anonymous namespace

/* CompContext pool implementation moved to gpucompress_pool.cpp */
/* ContextGuard defined in gpucompress_state.hpp */

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

    // Allow per-rank GPU selection via GPUCOMPRESS_DEVICE env var.
    // Primary multi-GPU mechanism is CUDA_VISIBLE_DEVICES from the mpirun
    // wrapper, but this provides a fallback/override for non-MPI use cases.
    // WARNING: GPUCOMPRESS_DEVICE and CUDA_VISIBLE_DEVICES are mutually exclusive.
    // When CUDA_VISIBLE_DEVICES is set, device indices are remapped so that device 0
    // is the only (or first) visible GPU. Setting GPUCOMPRESS_DEVICE to a value > 0
    // under CUDA_VISIBLE_DEVICES will fail or select the wrong physical device.
    const char* env_dev = getenv("GPUCOMPRESS_DEVICE");
    if (env_dev) {
        if (getenv("CUDA_VISIBLE_DEVICES")) {
            fprintf(stderr, "gpucompress: WARNING: both GPUCOMPRESS_DEVICE and "
                    "CUDA_VISIBLE_DEVICES are set. These are mutually exclusive — "
                    "GPUCOMPRESS_DEVICE will be ignored; CUDA_VISIBLE_DEVICES remaps "
                    "device indices and device 0 is the correct target.\n");
        } else {
            int requested = atoi(env_dev);
            int dev_count = 0;
            cudaGetDeviceCount(&dev_count);
            if (requested >= 0 && requested < dev_count) {
                g_cuda_device = requested;
            } else {
                fprintf(stderr, "gpucompress: GPUCOMPRESS_DEVICE=%d out of range (0..%d), using 0\n",
                        requested, dev_count - 1);
                g_cuda_device = 0;
            }
        }
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

    // Storage bandwidth: static 5 GB/s (representative HPC parallel filesystem).
    // Override at runtime via gpucompress_set_bandwidth() or GPUCOMPRESS_BW_GBPS env var.
    {
        const char* bw_env = getenv("GPUCOMPRESS_BW_GBPS");
        if (bw_env) {
            float bw_gbps = (float)atof(bw_env);
            if (bw_gbps > 0.0f)
                g_measured_bw_bytes_per_ms = bw_gbps * 1e6f;
        }
        fprintf(stderr, "gpucompress: storage bandwidth: %.1f GB/s\n",
                g_measured_bw_bytes_per_ms / 1e6);
    }

    // Load NN weights if path provided
    if (weights_path != nullptr && weights_path[0] != '\0') {
        if (gpucompress_nn_load_impl(weights_path) != 0) {
            fprintf(stderr, "Warning: Failed to load NN weights from %s\n", weights_path);
        }
    }

    /* Check debug env var */
    const char* dbg_env = getenv("GPUCOMPRESS_DEBUG_NN");
    if (dbg_env && (dbg_env[0] == '1' || dbg_env[0] == 'y' || dbg_env[0] == 'Y'))
        g_debug_nn = true;

    /* Check detailed timing env var */
    const char* dt_env = getenv("GPUCOMPRESS_DETAILED_TIMING");
    if (dt_env && (dt_env[0] == '1' || dt_env[0] == 'y' || dt_env[0] == 'Y'))
        g_detailed_timing = true;

    g_initialized.store(true);
    return GPUCOMPRESS_SUCCESS;
}

extern "C" void gpucompress_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Guard against underflow: if ref_count is already 0, nothing to do
    if (g_ref_count.load() <= 0) return;

    int old_ref = g_ref_count.fetch_sub(1);

    if (old_ref <= 1) {
        // Last reference, cleanup
        g_online_learning_enabled = false;
        g_exploration_enabled = false;
        g_selection_mode.store(GPUCOMPRESS_SELECT_NN);
        g_best_mode.store(false);
        gpucompress_nn_cleanup_impl();
        gpucompress_free_stats_workspace();
        /* P0: free cached VOL buffers while CUDA context is still alive.
         * Weak reference avoids circular dependency (gpucompress ↔ H5VLgpucompress). */
        extern void H5VL_gpucompress_release_buf_cache(void) __attribute__((weak));
        if (H5VL_gpucompress_release_buf_cache)
            H5VL_gpucompress_release_buf_cache();
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
    if (!input || !output || !output_size || input_size == 0)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    /* Allocate GPU input buffer and upload */
    void* d_input = nullptr;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess)
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    if (cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    /* Allocate GPU output buffer */
    size_t max_comp = gpucompress_max_compressed_size(input_size);
    void* d_output = nullptr;
    if (cudaMalloc(&d_output, max_comp) != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    /* Compress on GPU */
    size_t comp_size = max_comp;
    gpucompress_error_t err = gpucompress_compress_gpu(
        d_input, input_size,
        d_output, &comp_size,
        config, stats, nullptr);

    cudaFree(d_input);

    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_output);
        return err;
    }

    /* Check output buffer is large enough */
    if (comp_size > *output_size) {
        cudaFree(d_output);
        *output_size = comp_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    /* Download compressed data to host */
    if (cudaMemcpy(output, d_output, comp_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_output);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cudaFree(d_output);
    *output_size = comp_size;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size
) {
    if (!input || !output || !output_size || input_size == 0)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    /* Read original size from header to know decompressed buffer size */
    size_t original_size = 0;
    gpucompress_error_t err = gpucompress_get_original_size(input, &original_size);
    if (err != GPUCOMPRESS_SUCCESS)
        return err;

    if (original_size > *output_size) {
        *output_size = original_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    /* Upload compressed data to GPU */
    void* d_input = nullptr;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess)
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    if (cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    /* Allocate GPU output buffer */
    void* d_output = nullptr;
    if (cudaMalloc(&d_output, original_size) != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    /* Decompress on GPU */
    size_t decomp_size = original_size;
    err = gpucompress_decompress_gpu(
        d_input, input_size,
        d_output, &decomp_size, nullptr);

    cudaFree(d_input);

    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_output);
        return err;
    }

    /* Download decompressed data to host */
    if (cudaMemcpy(output, d_output, decomp_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_output);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cudaFree(d_output);
    *output_size = decomp_size;
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

/* NN management, query, chunk history, learning controls → gpucompress_learning.cpp */

/* ============================================================
 * String Helpers
 * ============================================================ */

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
    if (idx >= 0 && idx <= 8) {
        return ERROR_MESSAGES[idx];
    }
    return "Unknown error";
}

extern "C" const char* gpucompress_version(void) {
    return VERSION_STRING;
}

/* ============================================================
 * GPU Memory API
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

