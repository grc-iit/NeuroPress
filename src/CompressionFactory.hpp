/*
 * Compression Factory - Header
 * 
 * Factory design pattern for creating nvcomp compression managers
 * Provides a clean interface for algorithm selection and configuration
 */

#ifndef COMPRESSION_FACTORY_HPP
#define COMPRESSION_FACTORY_HPP

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
    GZIP,
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

#endif // COMPRESSION_FACTORY_HPP

