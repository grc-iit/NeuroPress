/**
 * @file gpucompress.h
 * @brief GPUCompress C API - GPU-accelerated compression library
 *
 * This library provides GPU-accelerated compression with optional RL-based
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
 * Use GPUCOMPRESS_ALGO_AUTO for RL-based automatic selection.
 */
typedef enum {
    GPUCOMPRESS_ALGO_AUTO     = 0,  /**< RL-based automatic selection */
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
    GPUCOMPRESS_PREPROC_SHUFFLE_2  = 0x01,  /**< 2-byte element shuffle */
    GPUCOMPRESS_PREPROC_SHUFFLE_4  = 0x02,  /**< 4-byte element shuffle */
    GPUCOMPRESS_PREPROC_SHUFFLE_8  = 0x04,  /**< 8-byte element shuffle */
    GPUCOMPRESS_PREPROC_QUANTIZE   = 0x10   /**< Enable quantization (lossy) */
} gpucompress_preproc_t;

/**
 * Extract shuffle element size from preprocessing flags.
 * Returns 0, 2, 4, or 8.
 */
#define GPUCOMPRESS_GET_SHUFFLE_SIZE(flags) \
    (((flags) & GPUCOMPRESS_PREPROC_SHUFFLE_8) ? 8 : \
     ((flags) & GPUCOMPRESS_PREPROC_SHUFFLE_4) ? 4 : \
     ((flags) & GPUCOMPRESS_PREPROC_SHUFFLE_2) ? 2 : 0)

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
    GPUCOMPRESS_ERROR_QTABLE_NOT_LOADED = -6, /**< Q-Table not loaded for AUTO */
    GPUCOMPRESS_ERROR_INVALID_HEADER  = -7,  /**< Invalid compression header */
    GPUCOMPRESS_ERROR_NOT_INITIALIZED = -8,  /**< Library not initialized */
    GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL = -9  /**< Output buffer too small */
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
    gpucompress_algorithm_t algorithm;  /**< Algorithm (AUTO for RL selection) */
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
 * @param qtable_path Path to Q-Table file for RL-based selection.
 *                    Use NULL for default path or if not using ALGO_AUTO.
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_init(const char* qtable_path);

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
 * Q-Table / RL Management
 * ============================================================ */

/**
 * Load Q-Table from file for RL-based algorithm selection.
 *
 * Supports JSON and binary formats. Q-Table is loaded to GPU
 * constant memory for fast inference.
 *
 * @param filepath Path to Q-Table file (.json or .bin)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_load_qtable(const char* filepath);

/**
 * Check if Q-Table is loaded.
 *
 * @return 1 if Q-Table is loaded, 0 otherwise
 */
int gpucompress_qtable_is_loaded(void);

/**
 * Get recommended configuration based on data characteristics.
 *
 * Uses loaded Q-Table to recommend algorithm and preprocessing
 * based on entropy, error bound level, MAD, and second derivative.
 *
 * @param entropy            Data entropy in bits (0.0 to 8.0)
 * @param error_bound        Desired error bound (0 for lossless)
 * @param mad                Mean Absolute Deviation (0.0 if unknown)
 * @param second_derivative  Mean absolute second derivative (0.0 if unknown)
 * @param algorithm_out      Output: recommended algorithm
 * @param preprocessing_out  Output: recommended preprocessing flags
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_recommend_config(
    double entropy,
    double error_bound,
    double mad,
    double second_derivative,
    gpucompress_algorithm_t* algorithm_out,
    unsigned int* preprocessing_out
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
 * Active Learning API
 * ============================================================ */

/**
 * Enable active learning with experience collection.
 *
 * When enabled, each ALGO_AUTO compression call will:
 *   - Store the (data_stats, config, actual_result) as a CSV row
 *   - Check if the NN prediction was accurate
 *   - If prediction error exceeds threshold, explore alternative configs
 *
 * Off by default. Call gpucompress_disable_active_learning() to stop.
 *
 * @param experience_path Path to CSV file for storing experience samples
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_enable_active_learning(
    const char* experience_path);

/**
 * Disable active learning and close the experience file.
 */
void gpucompress_disable_active_learning(void);

/**
 * Check if active learning is currently enabled.
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
 * Get the number of experience samples collected this session.
 *
 * @return Number of samples written since active learning was enabled
 */
size_t gpucompress_experience_count(void);

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

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_H */
