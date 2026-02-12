/**
 * @file internal.hpp
 * @brief Internal C++ interface for GPUCompress library
 *
 * This header provides internal C++ utilities used by the library.
 * Not part of the public API.
 */

#ifndef INTERNAL_HPP
#define INTERNAL_HPP

#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <atomic>

#include "gpucompress.h"
#include "core/compression_factory.hpp"
#include "core/compression_header.h"

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

/** Number of entropy bins for Q-Table state (0.5-width bins) */
constexpr int NUM_ENTROPY_BINS = 16;

/** Number of error bound levels (aggressive, balanced, precise, lossless) */
constexpr int NUM_ERROR_LEVELS = 4;

/** Number of MAD (Mean Absolute Deviation) bins */
constexpr int NUM_MAD_BINS = 4;

/** Number of first derivative bins */
constexpr int NUM_DERIV_BINS = 4;

/** Number of Q-Table states (entropy * error * MAD * derivative) */
constexpr int NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS * NUM_MAD_BINS * NUM_DERIV_BINS;

/** Number of Q-Table actions (quantization * shuffle * algorithms) */
constexpr int NUM_ACTIONS = 32;  // 2 * 2 * 8

/** MAD bin thresholds (3 thresholds → 4 bins) */
constexpr double MAD_BIN_THRESHOLDS[3] = {0.05, 0.15, 0.30};

/** First derivative bin thresholds (3 thresholds → 4 bins) */
constexpr double DERIV_BIN_THRESHOLDS[3] = {0.05, 0.15, 0.35};

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
 * Map a value to a bin using thresholds array.
 */
inline int valueToBin(double value, const double* thresholds, int n_thresholds) {
    for (int i = 0; i < n_thresholds; i++) {
        if (value < thresholds[i]) return i;
    }
    return n_thresholds;
}

/**
 * Map MAD value to bin index (0-3).
 */
inline int madToBin(double mad) {
    return valueToBin(mad, MAD_BIN_THRESHOLDS, 3);
}

/**
 * Map first derivative value to bin index (0-3).
 */
inline int derivToBin(double first_derivative) {
    return valueToBin(first_derivative, DERIV_BIN_THRESHOLDS, 3);
}

/**
 * Encode Q-Table state from entropy, error level, MAD, and first derivative.
 */
inline int encodeState(double entropy, int error_level,
                       double mad = 0.0, double first_derivative = 0.0) {
    int entropy_bin = static_cast<int>(entropy * 2);
    if (entropy_bin < 0) entropy_bin = 0;
    if (entropy_bin >= NUM_ENTROPY_BINS) entropy_bin = NUM_ENTROPY_BINS - 1;
    int mad_bin = madToBin(mad);
    int deriv_bin = derivToBin(first_derivative);
    return ((entropy_bin * NUM_ERROR_LEVELS + error_level)
            * NUM_MAD_BINS + mad_bin) * NUM_DERIV_BINS + deriv_bin;
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
 * Map error bound to error level (0, 1, 2, or 3).
 *
 * Level 0: error_bound >= 0.01        (aggressive lossy)
 * Level 1: 0.01 <= error_bound < 0.1  (moderate lossy)
 * Level 2: 0.001 <= error_bound < 0.01 (precise lossy)
 * Level 3: error_bound <= 0           (lossless, no quantization)
 */
inline int errorBoundToLevel(double error_bound) {
    if (error_bound <= 0.0) return 3;   // Lossless
    if (error_bound >= 0.1) return 0;   // Aggressive
    if (error_bound >= 0.01) return 1;  // Moderate
    if (error_bound >= 0.001) return 2; // Precise
    return 3;                           // Below 0.001 treated as lossless
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
 * Load Q-Table to GPU global memory.
 *
 * @param h_qtable Host array of Q-Table values (NUM_STATES * NUM_ACTIONS floats)
 * @return CUDA error code
 */
cudaError_t loadQTableToGPU(const float* h_qtable);

/**
 * Free GPU Q-Table memory. Called during library cleanup.
 */
void cleanupQTable();

/**
 * Get best action from Q-Table on GPU.
 *
 * @param state     Q-Table state index
 * @param stream    CUDA stream
 * @return Best action ID
 */
int getBestActionGPU(int state, cudaStream_t stream);

/* ============================================================
 * Auto-Stats GPU Pipeline
 * ============================================================ */

/**
 * Run the complete ALGO_AUTO statistics pipeline on GPU.
 *
 * Computes entropy, MAD, first derivative, encodes state, and
 * performs Q-Table argmax lookup entirely on GPU. Only copies
 * back the final action int (4 bytes) plus optional stats.
 *
 * @param d_input      Float data already on GPU
 * @param input_size   Size in bytes
 * @param error_level  From errorBoundToLevel(), 0-3
 * @param d_qtable     Q-Table in GPU memory
 * @param stream       CUDA stream
 * @param out_action   [out] Best action index
 * @param out_entropy  [out] Nullable, entropy for stats reporting
 * @param out_mad      [out] Nullable, normalized MAD for stats reporting
 * @param out_deriv    [out] Nullable, normalized derivative for stats reporting
 * @return 0 on success, -1 on error
 */
int runAutoStatsPipeline(
    const void* d_input,
    size_t input_size,
    int error_level,
    const float* d_qtable,
    cudaStream_t stream,
    int* out_action,
    double* out_entropy = nullptr,
    double* out_mad = nullptr,
    double* out_deriv = nullptr
);

/**
 * Launch entropy kernels asynchronously without D->H copy.
 *
 * Fire-and-forget variant that writes entropy to a device pointer.
 *
 * @param d_data         Data buffer (GPU memory)
 * @param num_bytes      Size in bytes
 * @param d_histogram    Pre-allocated histogram buffer (256 uints)
 * @param d_entropy_out  Device pointer to write entropy result
 * @param stream         CUDA stream
 */
void launchEntropyKernelsAsync(
    const void* d_data,
    size_t num_bytes,
    unsigned int* d_histogram,
    double* d_entropy_out,
    cudaStream_t stream
);

/**
 * Get device pointer to Q-Table in GPU global memory.
 *
 * @return Device pointer to Q-Table, or nullptr if not loaded
 */
const float* getQTableDevicePtr();

} // namespace gpucompress

#endif /* INTERNAL_HPP */
