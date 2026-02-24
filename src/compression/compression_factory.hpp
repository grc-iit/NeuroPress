/*
 * Compression Factory - Header
 * 
 * Factory design pattern for creating nvcomp compression managers
 * Provides a clean interface for algorithm selection and configuration
 */

#ifndef COMPRESSION_COMPRESSION_FACTORY_HPP
#define COMPRESSION_COMPRESSION_FACTORY_HPP

#include <memory>
#include <string>
#include <cuda_runtime.h>

// nvCOMP includes
#include "nvcomp.hpp"
#include "nvcomp/nvcompManager.hpp"

// Compression algorithm enumeration
enum class CompressionAlgorithm {
    LZ4,
    SNAPPY,
    DEFLATE,
    GDEFLATE,
    ZSTD,
    ANS,
    CASCADED,
    BITCOMP
};

// Helper functions
std::string toLowerCase(const std::string& str);
std::string getAlgorithmName(CompressionAlgorithm algo);
CompressionAlgorithm parseCompressionAlgorithm(const std::string& algo_str);

// Factory function to create compression manager
std::unique_ptr<nvcomp::nvcompManagerBase> createCompressionManager(
    CompressionAlgorithm algo,
    size_t chunk_size,
    cudaStream_t stream,
    const void* sample_input = nullptr
);

// Factory function to create decompression manager from compressed data
// Algorithm is auto-detected from the compressed data header
// Note: Accepts const void* for convenience, internally casts to uint8_t*
// Returns shared_ptr (as returned by nvcomp's create_manager)
std::shared_ptr<nvcomp::nvcompManagerBase> createDecompressionManager(
    const void* d_compressed,
    cudaStream_t stream
);

#endif // COMPRESSION_COMPRESSION_FACTORY_HPP

