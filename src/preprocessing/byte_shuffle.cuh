#ifndef BYTE_SHUFFLE_CUH
#define BYTE_SHUFFLE_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "compression/util.h"

/**
 * @file byte_shuffle.cuh
 * @brief GPU Byte Shuffle for float32 compression preprocessing
 *
 * Reorganizes data by grouping bytes at same position across elements:
 *   INPUT:  [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]
 *   OUTPUT: [A0 B0 C0][A1 B1 C1][A2 B2 C2][A3 B3 C3]
 *
 * Architecture: One warp per chunk, compile-time specialized for 4-byte elements.
 */

#define WARP_SIZE 32

// ============================================================================
// High-Level API (RECOMMENDED)
// ============================================================================

/**
 * @brief Shuffle a device buffer for better compression
 *
 * Allocates output, chunks the data on GPU, and shuffles in parallel.
 * Specialized for 4-byte (float32) elements.
 *
 * @param device_input   Pointer to input data on device
 * @param total_bytes    Total size of input data in bytes
 * @param element_size   Size of each element in bytes (4 for float32)
 * @param chunk_bytes    Chunk size for processing (default: 256KB)
 * @param stream         CUDA stream (default: 0)
 * @return Pointer to shuffled output on device (caller must cudaFree!)
 */
uint8_t* byte_shuffle_simple(
    void* device_input,
    size_t total_bytes,
    unsigned element_size,
    size_t chunk_bytes = 256 * 1024,
    cudaStream_t stream = 0
);

/**
 * @brief Unshuffle a device buffer (reverse of byte_shuffle_simple)
 *
 * @param device_input   Pointer to shuffled data on device
 * @param total_bytes    Total size of data in bytes
 * @param element_size   Size of each element in bytes (4 for float32)
 * @param chunk_bytes    Same chunk size used in shuffle
 * @param stream         CUDA stream (default: 0)
 * @return Pointer to unshuffled output on device (caller must cudaFree!)
 */
uint8_t* byte_unshuffle_simple(
    void* device_input,
    size_t total_bytes,
    unsigned element_size,
    size_t chunk_bytes = 256 * 1024,
    cudaStream_t stream = 0
);

// ============================================================================
// Low-Level API
// ============================================================================

/**
 * @brief Launch shuffle kernel (specialized<4> for float32)
 */
cudaError_t launch_byte_shuffle(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size,
    cudaStream_t stream = 0
);

/**
 * @brief Launch unshuffle kernel (specialized<4> for float32)
 */
cudaError_t launch_byte_unshuffle(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size,
    cudaStream_t stream = 0
);

#endif // BYTE_SHUFFLE_CUH
