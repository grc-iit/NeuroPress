/**
 * @file gpucompress.h
 * @brief GPUCompress C API - GPU-accelerated compression library
 *
 * This library provides GPU-accelerated compression with neural network-based
 * algorithm selection. It wraps nvcomp algorithms with preprocessing
 * (quantization, byte shuffle) and supports HDF5 filter integration.
 *
 * Usage:
 *   1. Call gpucompress_init() once at program start
 *   2. Use gpucompress_compress()/gpucompress_decompress() for data
 *   3. Call gpucompress_cleanup() at program end
 */

#ifndef GPUCOMPRESS_H
#define GPUCOMPRESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version and Constants
 * ============================================================ */

#define GPUCOMPRESS_VERSION_MAJOR 1
#define GPUCOMPRESS_VERSION_MINOR 0
#define GPUCOMPRESS_VERSION_PATCH 0

/** Size of compression header prepended to compressed data */
#define GPUCOMPRESS_HEADER_SIZE 64

/** Magic number for identifying GPUCompress data ("GPUC" = 0x43555047) */
#define GPUCOMPRESS_MAGIC 0x43555047

/* ============================================================
 * Algorithm Selection
 * ============================================================ */

/**
 * Compression algorithm identifiers.
 * Use GPUCOMPRESS_ALGO_AUTO for neural network-based automatic selection.
 */
typedef enum {
    GPUCOMPRESS_ALGO_AUTO     = 0,  /**< NN-based automatic selection */
    GPUCOMPRESS_ALGO_LZ4      = 1,  /**< Fast compression, general purpose */
    GPUCOMPRESS_ALGO_SNAPPY   = 2,  /**< Fastest compression, lower ratio */
    GPUCOMPRESS_ALGO_DEFLATE  = 3,  /**< Better ratio, slower */
    GPUCOMPRESS_ALGO_GDEFLATE = 4,  /**< GPU-optimized deflate */
    GPUCOMPRESS_ALGO_ZSTD     = 5,  /**< Best ratio, configurable */
    GPUCOMPRESS_ALGO_ANS      = 6,  /**< Entropy coding, numerical data */
    GPUCOMPRESS_ALGO_CASCADED = 7,  /**< High compression for floating-point */
    GPUCOMPRESS_ALGO_BITCOMP  = 8   /**< Lossless for scientific data */
} gpucompress_algorithm_t;

/* ============================================================
 * Preprocessing Options
 * ============================================================ */

/**
 * Preprocessing flags (can be OR'd together).
 */
typedef enum {
    GPUCOMPRESS_PREPROC_NONE       = 0x00,  /**< No preprocessing */
    GPUCOMPRESS_PREPROC_SHUFFLE_4  = 0x02,  /**< 4-byte element shuffle (float32) */
    GPUCOMPRESS_PREPROC_QUANTIZE   = 0x10   /**< Enable quantization (lossy) */
} gpucompress_preproc_t;

/**
 * Extract shuffle element size from preprocessing flags.
 * Returns 0 or 4.
 */
#define GPUCOMPRESS_GET_SHUFFLE_SIZE(flags) \
    (((flags) & GPUCOMPRESS_PREPROC_SHUFFLE_4) ? 4 : 0)

/* ============================================================
 * Error Codes
 * ============================================================ */

/**
 * Error codes returned by GPUCompress functions.
 */
typedef enum {
    GPUCOMPRESS_SUCCESS               =  0,  /**< Operation succeeded */
    GPUCOMPRESS_ERROR_INVALID_INPUT   = -1,  /**< Invalid input parameter */
    GPUCOMPRESS_ERROR_CUDA_FAILED     = -2,  /**< CUDA operation failed */
    GPUCOMPRESS_ERROR_COMPRESSION     = -3,  /**< Compression failed */
    GPUCOMPRESS_ERROR_DECOMPRESSION   = -4,  /**< Decompression failed */
    GPUCOMPRESS_ERROR_OUT_OF_MEMORY   = -5,  /**< Memory allocation failed */
    GPUCOMPRESS_ERROR_RESERVED_6      = -6,  /**< Reserved (formerly Q-Table) */
    GPUCOMPRESS_ERROR_INVALID_HEADER  = -7,  /**< Invalid compression header */
    GPUCOMPRESS_ERROR_NOT_INITIALIZED = -8,  /**< Library not initialized */
    GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL = -9, /**< Output buffer too small */
    GPUCOMPRESS_ERROR_NN_NOT_LOADED    = -10 /**< ALGO_AUTO requires NN weights but none are loaded */
} gpucompress_error_t;

/**
 * Get human-readable error message.
 */
const char* gpucompress_error_string(gpucompress_error_t error);

/* ============================================================
 * Configuration Structures
 * ============================================================ */

/**
 * Compression configuration.
 */
typedef struct {
    gpucompress_algorithm_t algorithm;  /**< Algorithm (AUTO for NN selection) */
    unsigned int preprocessing;          /**< Bitmask of gpucompress_preproc_t */
    double error_bound;                  /**< Quantization error bound (0 = lossless) */
    int cuda_device;                     /**< CUDA device (-1 = default) */
    void* cuda_stream;                   /**< CUDA stream (NULL = default) */
} gpucompress_config_t;

/**
 * Compression statistics (output from compress).
 */
typedef struct {
    size_t original_size;                /**< Original data size in bytes */
    size_t compressed_size;              /**< Compressed size in bytes */
    double compression_ratio;            /**< original_size / compressed_size */
    double entropy_bits;                 /**< Calculated data entropy (0-8 bits) */
    double mad;                          /**< Mean Absolute Deviation */
    double second_derivative;            /**< Mean absolute second derivative */
    gpucompress_algorithm_t algorithm_used; /**< Algorithm actually used */
    unsigned int preprocessing_used;     /**< Preprocessing actually applied */
    double throughput_mbps;              /**< Compression throughput (MB/s) */
    double predicted_ratio;              /**< NN-predicted compression ratio (0.0 if not ALGO_AUTO/NN) */
    double predicted_comp_time_ms;       /**< NN-predicted compression time in ms (0.0 if not ALGO_AUTO/NN) */
    double predicted_decomp_time_ms;     /**< NN-predicted decompression time in ms (0.0 if not ALGO_AUTO/NN) */
    double predicted_psnr_db;            /**< NN-predicted PSNR in dB (0.0 if not ALGO_AUTO/NN) */
    double actual_comp_time_ms;          /**< Actual GPU compression time in ms (CUDA event timing) */
    int    sgd_fired;                    /**< 1 if online reinforcement (SGD) was triggered, 0 otherwise */
    int    exploration_triggered;         /**< 1 if Level 2 exploration was triggered, 0 otherwise */
    int    nn_original_action;           /**< NN's primary action before exploration (-1 if not ALGO_AUTO) */
    int    nn_final_action;              /**< Final action used (may differ from original if exploration found better) */
} gpucompress_stats_t;

/* ============================================================
 * Initialization and Cleanup
 * ============================================================ */

/**
 * Initialize the GPUCompress library.
 *
 * Must be called before any other GPUCompress functions.
 * Thread-safe: can be called multiple times (uses reference counting).
 *
 * @param weights_path Path to .nnwt weights file for NN-based selection.
 *                     Use NULL if not using ALGO_AUTO.
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_init(const char* weights_path);

/**
 * Clean up GPUCompress library resources.
 *
 * Should be called when done using the library.
 * Thread-safe: uses reference counting, cleanup only when last reference.
 */
void gpucompress_cleanup(void);

/**
 * Check if library is initialized.
 *
 * @return 1 if initialized, 0 otherwise
 */
int gpucompress_is_initialized(void);

/**
 * Get default configuration.
 *
 * Default: LZ4 algorithm, no preprocessing, lossless, default CUDA device.
 *
 * @return Default configuration struct
 */
gpucompress_config_t gpucompress_default_config(void);

/* ============================================================
 * Core Compression/Decompression API
 * ============================================================ */

/**
 * Compress data from host memory.
 *
 * Transfers data to GPU, compresses, and returns result to host.
 * Output includes 64-byte header with metadata.
 *
 * @param input       Input buffer (host memory)
 * @param input_size  Size of input in bytes
 * @param output      Output buffer (host memory, preallocated)
 * @param output_size [in] Max output buffer size
 *                    [out] Actual compressed size including header
 * @param config      Compression configuration (NULL for default)
 * @param stats       Optional output statistics (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats
);

/**
 * Decompress data from host memory.
 *
 * Reads metadata from header, auto-detects algorithm and preprocessing.
 *
 * @param input       Compressed input buffer (host memory)
 * @param input_size  Size of compressed data including header
 * @param output      Output buffer (host memory, preallocated)
 * @param output_size [in] Max output buffer size
 *                    [out] Actual decompressed size
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size
);

/**
 * Get maximum possible compressed size for given input.
 *
 * Use this to preallocate output buffer for compress().
 * Includes space for header.
 *
 * @param input_size Size of input data in bytes
 * @return Maximum compressed size (input may not compress)
 */
size_t gpucompress_max_compressed_size(size_t input_size);

/**
 * Get original (decompressed) size from compressed data header.
 *
 * @param compressed     Compressed data buffer
 * @param original_size  Output: original uncompressed size
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_get_original_size(
    const void* compressed,
    size_t* original_size
);

/* ============================================================
 * GPU Memory API (Advanced)
 * ============================================================ */

/**
 * Compress data already in GPU memory.
 *
 * Avoids host-GPU transfer when data is already on GPU.
 * Output is written to GPU memory, caller must transfer to host if needed.
 *
 * @param d_input      Input buffer (GPU memory)
 * @param input_size   Size of input in bytes
 * @param d_output     Output buffer (GPU memory, preallocated)
 * @param output_size  [in] Max output size, [out] Actual compressed size
 * @param config       Compression configuration
 * @param stats        Optional output statistics
 * @param stream       CUDA stream for async operation
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats,
    void* stream
);

/**
 * Decompress data in GPU memory.
 *
 * @param d_input      Compressed input buffer (GPU memory)
 * @param input_size   Size of compressed data
 * @param d_output     Output buffer (GPU memory, preallocated)
 * @param output_size  [in] Max output size, [out] Actual decompressed size
 * @param stream       CUDA stream for async operation
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_decompress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    void* stream
);

/**
 * Check if a pointer is a CUDA device pointer.
 *
 * Uses cudaPointerGetAttributes internally.
 *
 * @param ptr  Pointer to test
 * @return 1 if device pointer, 0 if host/unknown
 */
int gpucompress_is_device_ptr(const void* ptr);

/* ============================================================
 * Entropy Analysis (GPU-accelerated)
 * ============================================================ */

/**
 * Calculate Shannon entropy of data (GPU-accelerated).
 *
 * Computes byte-level entropy using histogram approach on GPU.
 * Result range: 0.0 (all same byte) to 8.0 (uniform distribution).
 *
 * @param data        Data buffer (host memory)
 * @param size        Size in bytes
 * @param entropy_out Output: entropy in bits (0.0 to 8.0)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_calculate_entropy(
    const void* data,
    size_t size,
    double* entropy_out
);

/**
 * Calculate entropy of data in GPU memory.
 *
 * @param d_data      Data buffer (GPU memory)
 * @param size        Size in bytes
 * @param entropy_out Output: entropy in bits
 * @param stream      CUDA stream
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_calculate_entropy_gpu(
    const void* d_data,
    size_t size,
    double* entropy_out,
    void* stream
);

/* ============================================================
 * Statistical Analysis (GPU-accelerated)
 * ============================================================ */

/**
 * Compute data statistics on GPU: entropy, MAD, second derivative.
 *
 * Interprets data as float32 array for MAD and derivative computation.
 * Entropy is computed at byte level (0.0 to 8.0 bits).
 * MAD and second derivative are normalized by data range to [0, 1].
 *
 * @param data               Data buffer (host memory, float32 array)
 * @param size               Size in bytes (must be multiple of 4)
 * @param entropy            Output: Shannon entropy in bits (0.0 to 8.0)
 * @param mad                Output: normalized Mean Absolute Deviation (0.0 to 1.0)
 * @param second_derivative  Output: normalized mean |x[i+1]-2x[i]+x[i-1]| (0.0 to 1.0)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compute_stats(
    const void* data,
    size_t size,
    double* entropy,
    double* mad,
    double* second_derivative
);

/* ============================================================
 * Neural Network Model Management
 * ============================================================ */

/**
 * Load neural network weights for ML-based algorithm selection.
 *
 * When loaded, ALGO_AUTO will use the neural network instead of Q-Table.
 * The NN predicts compression metrics for all configs and ranks them.
 *
 * @param filepath Path to .nnwt weights file
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_load_nn(const char* filepath);

/**
 * Check if neural network is loaded.
 *
 * @return 1 if NN is loaded, 0 otherwise
 */
int gpucompress_nn_is_loaded(void);

/* ============================================================
 * Online Learning API
 * ============================================================ */

/**
 * Enable online learning (master switch). Activates SGD reinforcement.
 * Exploration and experience logging are off by default — enable separately.
 */
void gpucompress_enable_online_learning(void);

/**
 * Disable online learning entirely. Stops SGD, exploration, and logging.
 */
void gpucompress_disable_online_learning(void);

/**
 * Check if online learning is enabled.
 *
 * @return 1 if enabled, 0 otherwise
 */
int gpucompress_online_learning_enabled(void);

/**
 * Enable/disable exploration. Only effective when online learning is on.
 * Off by default.
 *
 * @param enable 1 to enable, 0 to disable
 */
void gpucompress_set_exploration(int enable);

/* ============================================================
 * Active Learning API (backward-compatible convenience)
 * ============================================================ */

/**
 * Enable active learning.
 * Convenience function: enables online learning and exploration.
 *
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_enable_active_learning(void);

/**
 * Disable active learning. Calls gpucompress_disable_online_learning().
 */
void gpucompress_disable_active_learning(void);

/**
 * Check if online learning is currently enabled.
 *
 * @return 1 if enabled, 0 otherwise
 */
int gpucompress_active_learning_enabled(void);

/**
 * Set the exploration threshold for active learning.
 *
 * When the NN's predicted compression ratio differs from actual ratio
 * by more than this fraction, Level 2 exploration triggers.
 *
 * @param threshold Fractional threshold (default 0.20 = 20% MAPE)
 */
void gpucompress_set_exploration_threshold(double threshold);

/**
 * Override the number of alternative configs explored (K) during Level-2
 * exploration.  Pass k <= 0 or k > 31 to restore the dynamic default.
 *
 * @param k  Number of alternatives to try (1-31), or -1 for dynamic default.
 */
void gpucompress_set_exploration_k(int k);

/**
 * Set SGD reinforcement parameters.
 * SGD is always active when online learning is on — the enable param is ignored.
 *
 * @param enable            Ignored (kept for backward compatibility)
 * @param learning_rate     SGD step size (default 1e-4)
 * @param mape_threshold    Ratio MAPE threshold to trigger reinforcement (default 0.20)
 * @param ct_mape_threshold Ignored (kept for backward compatibility)
 */
void gpucompress_set_reinforcement(int enable, float learning_rate,
                                   float mape_threshold, float ct_mape_threshold);

/**
 * Enable/disable verbose transfer and SGD logging to stderr.
 * Off by default. Enable when debugging data-transfer timing or SGD behavior.
 *
 * @param enable 1 to enable verbose logging, 0 to disable
 */
void gpucompress_set_verbose(int enable);

/* ============================================================
 * Cost-Based Ranking Configuration
 *
 * The NN predicts (comp_time, decomp_time, ratio) for 32 configs.
 * The ranking formula is:
 *   cost = w0 * comp_time + w1 * decomp_time + w2 * data_size / (ratio * bw)
 * The config with the lowest cost is selected.
 *
 * Default: w0=1, w1=1, w2=1 with auto-probed bandwidth from gpucompress_init().
 * ============================================================ */

/**
 * Set the cost-based ranking weights for NN algorithm selection.
 *
 * @param w0  Weight on predicted compression time (ms)
 * @param w1  Weight on predicted decompression time (ms)
 * @param w2  Weight on I/O cost: data_size / (ratio * bandwidth)
 */
void gpucompress_set_ranking_weights(float w0, float w1, float w2);

/**
 * Override the auto-probed storage bandwidth used in cost-based ranking.
 *
 * @param bw_gbps  Bandwidth in GB/s (e.g. 3.0 for NVMe, 0.2 for HDD)
 */
void gpucompress_set_bandwidth(float bw_gbps);

/**
 * Hot-reload neural network weights from a new .nnwt file.
 *
 * Replaces the current NN model without restarting the library.
 * Thread-safe: holds the init mutex during reload.
 *
 * @param filepath Path to new .nnwt weights file
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_reload_nn(const char* filepath);

/* ============================================================
 * Utility Functions
 * ============================================================ */

/**
 * Get algorithm name string.
 *
 * @param algorithm Algorithm enum value
 * @return Human-readable algorithm name
 */
const char* gpucompress_algorithm_name(gpucompress_algorithm_t algorithm);

/**
 * Get the last NN action chosen during ALGO_AUTO compression.
 * Action encodes: algorithm (action % 8), quantize ((action/8) % 2), shuffle ((action/16) % 2).
 * Returns -1 if no NN prediction has been made yet.
 */
int gpucompress_get_last_nn_action(void);

/**
 * Get the NN's original action before exploration may have changed it.
 * Returns -1 if no NN prediction has been made yet.
 */
int gpucompress_get_last_nn_original_action(void);

/**
 * Check if Level 2 exploration was triggered during the last ALGO_AUTO compress.
 * Returns 1 if exploration was triggered, 0 otherwise.
 */
int gpucompress_get_last_exploration_triggered(void);

/**
 * Check if online SGD (reinforcement) fired during the last ALGO_AUTO compress.
 * Returns 1 if SGD was applied, 0 otherwise.
 */
int gpucompress_get_last_sgd_fired(void);

/* ── Per-chunk diagnostic history ─────────────────────────────────── */

/**
 * Per-chunk diagnostic snapshot recorded during H5Dwrite(H5S_ALL).
 */
typedef struct {
    int    nn_action;
    int    nn_original_action;
    int    exploration_triggered;
    int    sgd_fired;

    /* Per-chunk timing breakdown (ms, 0.0 if not applicable) */
    float  nn_inference_ms;      /* NN forward pass kernel + result D→H   */
    float  stats_ms;             /* stats kernels + stats D→H copies      */
    float  preprocessing_ms;     /* quantization + byte shuffle           */
    float  compression_ms;       /* primary nvCOMP kernel only            */
    float  exploration_ms;       /* exploration loop (0 if not triggered) */
    float  sgd_update_ms;        /* SGD weight update (0 if not fired)    */

    /* Per-chunk ratio and prediction accuracy */
    float  actual_ratio;         /* input_size / compressed_size          */
    float  predicted_ratio;      /* NN-predicted ratio (0 if not AUTO)    */
    float  predicted_comp_time;  /* NN-predicted compression time ms      */
    float  predicted_decomp_time;/* NN-predicted decompression time ms    */
    float  predicted_psnr;       /* NN-predicted PSNR in dB               */

    /* Filled during read (decompression) — 0 until VOL read completes   */
    float  decompression_ms;     /* actual decompression time (nvCOMP)    */
} gpucompress_chunk_diag_t;

/**
 * Reset the per-chunk diagnostic history (call before H5Dwrite).
 */
void gpucompress_reset_chunk_history(void);

/**
 * Return the number of chunk diagnostics recorded since the last reset.
 */
int  gpucompress_get_chunk_history_count(void);

/**
 * Copy the diagnostic record for chunk @p idx into @p out.
 * @return 0 on success, -1 if idx is out of range or out is NULL.
 */
int  gpucompress_get_chunk_diag(int idx, gpucompress_chunk_diag_t *out);

/**
 * Record actual decompression time for chunk @p idx (called from VOL read).
 */
void gpucompress_record_chunk_decomp_ms(int idx, float ms);

/* ============================================================
 * Force-Algorithm Queue
 *
 * Allows callers to override ALGO_AUTO's NN-based algorithm selection
 * on a per-chunk basis.  Push one entry per chunk before calling
 * H5Dwrite; each call to gpucompress_compress_gpu() with ALGO_AUTO
 * will dequeue the next entry instead of running NN inference.
 * ============================================================ */

/**
 * Reset (clear) the force-algorithm queue.
 */
void gpucompress_force_algorithm_reset(void);

/**
 * Push one per-chunk algorithm override onto the queue.
 *
 * @param algorithm    Algorithm enum (1-8, e.g. GPUCOMPRESS_ALGO_ZSTD)
 * @param shuffle      1 to enable byte-shuffle, 0 to disable
 * @param quant        1 to enable lossy quantization, 0 to disable
 * @param error_bound  Error bound for quantization (ignored when quant=0)
 */
void gpucompress_force_algorithm_push(int algorithm, int shuffle,
                                      int quant, double error_bound);

/**
 * Parse algorithm from string.
 *
 * @param name Algorithm name (e.g., "lz4", "zstd", "auto")
 * @return Algorithm enum value, or GPUCOMPRESS_ALGO_LZ4 if unknown
 */
gpucompress_algorithm_t gpucompress_algorithm_from_string(const char* name);

/**
 * Get library version string.
 *
 * @return Version string (e.g., "1.0.0")
 */
const char* gpucompress_version(void);

/* ============================================================
 * nvcomp Manager Cache Statistics
 * ============================================================ */

/**
 * Reset the nvcomp manager cache hit/miss counters to zero.
 * Call before a benchmark or test to get clean counts.
 */
void gpucompress_reset_cache_stats(void);

/**
 * Read the nvcomp manager cache hit/miss counters.
 * A "hit" means an existing cached manager was reused (zero alloc).
 * A "miss" means a new manager was created (workspace alloc on GPU).
 * Any pointer may be NULL if that value is not needed.
 */
void gpucompress_get_cache_stats(int *hits, int *misses);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_H */
