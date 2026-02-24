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

#include "gpucompress.h"
#include "compression/compression_factory.hpp"
#include "compression/compression_header.h"

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
 * Action Encoding/Decoding
 * ============================================================ */

/**
 * Decoded action configuration.
 */
struct DecodedAction {
    int algorithm;          /**< Algorithm index 0-7 */
    bool use_quantization;  /**< Whether to apply quantization */
    int shuffle_size;       /**< Shuffle element size (0 or 4) */
};

/**
 * Decode action ID to action configuration.
 *
 * Encoding: action = algorithm + quant*8 + shuffle*16
 */
inline DecodedAction decodeAction(int action_id) {
    DecodedAction action;
    action.algorithm = action_id % 8;
    action.use_quantization = ((action_id / 8) % 2) != 0;
    action.shuffle_size = ((action_id / 16) % 2) ? 4 : 0;
    return action;
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
int launchEntropyKernelsAsync(
    const void* d_data,
    size_t num_bytes,
    unsigned int* d_histogram,
    double* d_entropy_out,
    cudaStream_t stream
);

/* ============================================================
 * Neural Network GPU Functions
 * ============================================================ */

/**
 * Load neural network weights from binary file (.nnwt).
 *
 * @param filepath Path to .nnwt weights file
 * @return true on success
 */
bool loadNNFromBinary(const char* filepath);

/**
 * Free neural network GPU memory.
 */
void cleanupNN();

/**
 * Check if neural network is loaded.
 */
bool isNNLoaded();

/**
 * Check if input features are out-of-distribution.
 *
 * @return true if any continuous feature is outside training bounds
 */
bool isInputOOD(double entropy, double mad, double deriv,
                size_t data_size, double error_bound);

/**
 * Run neural network inference to find best compression config.
 *
 * @param entropy             Shannon entropy (0-8)
 * @param mad_norm            Normalized MAD (0-1)
 * @param deriv_norm          Normalized 2nd derivative (0-1)
 * @param data_size           Data size in bytes
 * @param error_bound         Error bound (0 for lossless)
 * @param stream              CUDA stream
 * @param out_predicted_ratio [out] Nullable, predicted compression ratio for winner
 * @param out_top_actions     [out] Nullable, all 32 action IDs sorted by rank
 * @return Best action ID (0-31), or -1 on error
 */
int runNNInference(
    double entropy,
    double mad_norm,
    double deriv_norm,
    size_t data_size,
    double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio = nullptr,
    float* out_predicted_comp_time = nullptr,
    int* out_top_actions = nullptr
);

/**
 * Run stats-only pipeline on GPU (no NN inference or Q-Table lookup).
 *
 * Computes entropy, normalized MAD, and normalized second derivative
 * entirely on GPU. Used by gpucompress_compute_stats() public API.
 *
 * @param d_input      Float data already on GPU
 * @param input_size   Size in bytes
 * @param stream       CUDA stream
 * @param out_entropy  [out] Shannon entropy (0-8 bits)
 * @param out_mad      [out] Normalized MAD (0-1)
 * @param out_deriv    [out] Normalized second derivative (0-1)
 * @return 0 on success, -1 on error
 */
int runStatsOnlyPipeline(
    const void* d_input,
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,
    double* out_mad,
    double* out_deriv
);

/**
 * Run the complete ALGO_AUTO statistics + NN inference pipeline on GPU.
 *
 * Same stats computation as runAutoStatsPipeline but uses neural network
 * instead of Q-Table for action selection.
 *
 * @param d_input      Float data already on GPU
 * @param input_size   Size in bytes
 * @param error_bound  Raw error bound value
 * @param stream       CUDA stream
 * @param out_action   [out] Best action index
 * @param out_entropy  [out] Nullable, entropy for stats reporting
 * @param out_mad      [out] Nullable, normalized MAD for stats reporting
 * @param out_deriv    [out] Nullable, normalized derivative for stats reporting
 * @return 0 on success, -1 on error
 */
int runAutoStatsNNPipeline(
    const void* d_input,
    size_t input_size,
    double error_bound,
    cudaStream_t stream,
    int* out_action,
    double* out_entropy = nullptr,
    double* out_mad = nullptr,
    double* out_deriv = nullptr,
    float* out_predicted_ratio = nullptr,
    float* out_predicted_comp_time = nullptr,
    int* out_top_actions = nullptr
);

} // namespace gpucompress

#endif /* INTERNAL_HPP */
