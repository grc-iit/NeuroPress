/**
 * @file gpucompress_internal.hpp
 * @brief Internal C++ interface for GPUCompress library
 *
 * This header provides internal C++ utilities used by the library.
 * Not part of the public API.
 */

#ifndef GPUCOMPRESS_INTERNAL_HPP
#define GPUCOMPRESS_INTERNAL_HPP

#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <atomic>

#include "gpucompress.h"
#include "CompressionFactory.hpp"
#include "compression_header.h"

namespace gpucompress {

/* ============================================================
 * Internal Constants
 * ============================================================ */

/** Default chunk size for nvcomp (64KB) */
constexpr size_t DEFAULT_CHUNK_SIZE = 1 << 16;

/** Chunk size for byte shuffle (256KB) */
constexpr size_t SHUFFLE_CHUNK_SIZE = 256 * 1024;

/** Alignment for GPU memory (4KB for optimal performance) */
constexpr size_t GPU_ALIGNMENT = 4096;

/** Number of entropy bins for Q-Table state */
constexpr int NUM_ENTROPY_BINS = 10;

/** Number of error bound levels */
constexpr int NUM_ERROR_LEVELS = 3;

/** Number of Q-Table states (entropy_bins * error_levels) */
constexpr int NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS;

/** Number of Q-Table actions (quantization * shuffle * algorithms) */
constexpr int NUM_ACTIONS = 32;  // 2 * 2 * 8

/* ============================================================
 * Algorithm Mapping
 * ============================================================ */

/**
 * Map gpucompress_algorithm_t to internal CompressionAlgorithm.
 */
inline CompressionAlgorithm toInternalAlgorithm(gpucompress_algorithm_t algo) {
    switch (algo) {
        case GPUCOMPRESS_ALGO_LZ4:      return CompressionAlgorithm::LZ4;
        case GPUCOMPRESS_ALGO_SNAPPY:   return CompressionAlgorithm::SNAPPY;
        case GPUCOMPRESS_ALGO_DEFLATE:  return CompressionAlgorithm::DEFLATE;
        case GPUCOMPRESS_ALGO_GDEFLATE: return CompressionAlgorithm::GDEFLATE;
        case GPUCOMPRESS_ALGO_ZSTD:     return CompressionAlgorithm::ZSTD;
        case GPUCOMPRESS_ALGO_ANS:      return CompressionAlgorithm::ANS;
        case GPUCOMPRESS_ALGO_CASCADED: return CompressionAlgorithm::CASCADED;
        case GPUCOMPRESS_ALGO_BITCOMP:  return CompressionAlgorithm::BITCOMP;
        default:                        return CompressionAlgorithm::LZ4;
    }
}

/**
 * Map internal CompressionAlgorithm to gpucompress_algorithm_t.
 */
inline gpucompress_algorithm_t fromInternalAlgorithm(CompressionAlgorithm algo) {
    switch (algo) {
        case CompressionAlgorithm::LZ4:      return GPUCOMPRESS_ALGO_LZ4;
        case CompressionAlgorithm::SNAPPY:   return GPUCOMPRESS_ALGO_SNAPPY;
        case CompressionAlgorithm::DEFLATE:  return GPUCOMPRESS_ALGO_DEFLATE;
        case CompressionAlgorithm::GDEFLATE: return GPUCOMPRESS_ALGO_GDEFLATE;
        case CompressionAlgorithm::ZSTD:     return GPUCOMPRESS_ALGO_ZSTD;
        case CompressionAlgorithm::ANS:      return GPUCOMPRESS_ALGO_ANS;
        case CompressionAlgorithm::CASCADED: return GPUCOMPRESS_ALGO_CASCADED;
        case CompressionAlgorithm::BITCOMP:  return GPUCOMPRESS_ALGO_BITCOMP;
        default:                             return GPUCOMPRESS_ALGO_LZ4;
    }
}

/* ============================================================
 * Q-Table Action Encoding/Decoding
 * ============================================================ */

/**
 * Decoded Q-Table action.
 */
struct QTableAction {
    int algorithm;          /**< Algorithm index 0-7 */
    bool use_quantization;  /**< Whether to apply quantization */
    int shuffle_size;       /**< Shuffle element size (0 or 4) */
};

/**
 * Encode Q-Table state from entropy and error level.
 */
inline int encodeState(double entropy, int error_level) {
    int entropy_bin = static_cast<int>(entropy);
    if (entropy_bin < 0) entropy_bin = 0;
    if (entropy_bin >= NUM_ENTROPY_BINS) entropy_bin = NUM_ENTROPY_BINS - 1;
    return entropy_bin * NUM_ERROR_LEVELS + error_level;
}

/**
 * Decode action ID to action configuration.
 *
 * Encoding: action = algorithm + quant*8 + shuffle*16
 */
inline QTableAction decodeAction(int action_id) {
    QTableAction action;
    action.algorithm = action_id % 8;
    action.use_quantization = ((action_id / 8) % 2) != 0;
    action.shuffle_size = ((action_id / 16) % 2) ? 4 : 0;
    return action;
}

/**
 * Map error bound to error level (0, 1, or 2).
 *
 * Level 0: error_bound >= 0.01 (aggressive)
 * Level 1: 0.001 <= error_bound < 0.01 (balanced)
 * Level 2: error_bound < 0.001 or lossless (precise)
 */
inline int errorBoundToLevel(double error_bound) {
    if (error_bound <= 0.0) return 2;  // Lossless = most precise
    if (error_bound >= 0.01) return 0;
    if (error_bound >= 0.001) return 1;
    return 2;
}

/* ============================================================
 * CUDA Utilities
 * ============================================================ */

/**
 * Align size to specified alignment.
 */
inline size_t alignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * CUDA error checking wrapper.
 */
inline gpucompress_error_t checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Global Library State
 * ============================================================ */

/**
 * Library global state (thread-safe singleton).
 */
class LibraryState {
public:
    static LibraryState& instance() {
        static LibraryState instance;
        return instance;
    }

    bool initialize(const char* qtable_path);
    void cleanup();
    bool isInitialized() const { return initialized_.load(); }

    // Q-Table access
    bool loadQTable(const char* path);
    bool qtableLoaded() const { return qtable_loaded_.load(); }
    int getBestAction(int state) const;

    // CUDA resources
    cudaStream_t getDefaultStream() const { return default_stream_; }
    int getDevice() const { return cuda_device_; }

private:
    LibraryState() = default;
    ~LibraryState() { cleanup(); }
    LibraryState(const LibraryState&) = delete;
    LibraryState& operator=(const LibraryState&) = delete;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> qtable_loaded_{false};
    std::atomic<int> ref_count_{0};
    std::mutex mutex_;

    cudaStream_t default_stream_ = nullptr;
    int cuda_device_ = 0;

    // Q-Table data (loaded to host, copied to GPU constant memory)
    float* h_qtable_ = nullptr;
};

/* ============================================================
 * Entropy Calculation (GPU)
 * ============================================================ */

/**
 * Calculate Shannon entropy on GPU.
 *
 * @param d_data    Data buffer (GPU memory)
 * @param num_bytes Size in bytes
 * @param stream    CUDA stream
 * @return Entropy in bits (0.0 to 8.0)
 */
double calculateEntropyGPU(const void* d_data, size_t num_bytes, cudaStream_t stream);

/* ============================================================
 * Q-Table GPU Functions
 * ============================================================ */

/**
 * Load Q-Table to GPU constant memory.
 *
 * @param h_qtable Host array of Q-Table values (NUM_STATES * NUM_ACTIONS floats)
 * @return CUDA error code
 */
cudaError_t loadQTableToGPU(const float* h_qtable);

/**
 * Get best action from Q-Table on GPU.
 *
 * @param state     Q-Table state index
 * @param stream    CUDA stream
 * @return Best action ID
 */
int getBestActionGPU(int state, cudaStream_t stream);

} // namespace gpucompress

#endif /* GPUCOMPRESS_INTERNAL_HPP */
