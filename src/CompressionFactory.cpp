/*
 * Compression Factory - Implementation
 * 
 * Factory design pattern for creating nvcomp compression managers
 * Supports multiple compression algorithms with default configurations
 */

#include "CompressionFactory.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

// nvCOMP includes
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/deflate.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/zstd.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/bitcomp.hpp"

using namespace nvcomp;

// Helper: Convert string to lowercase
std::string toLowerCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Helper: Get algorithm name as string
std::string getAlgorithmName(CompressionAlgorithm algo) {
    switch (algo) {
        case CompressionAlgorithm::LZ4:      return "LZ4";
        case CompressionAlgorithm::SNAPPY:   return "Snappy";
        case CompressionAlgorithm::DEFLATE:  return "Deflate";
        case CompressionAlgorithm::GDEFLATE: return "Gdeflate";
        case CompressionAlgorithm::ZSTD:     return "Zstd";
        case CompressionAlgorithm::ANS:      return "ANS";
        case CompressionAlgorithm::CASCADED: return "Cascaded";
        case CompressionAlgorithm::BITCOMP:  return "Bitcomp";
        default: return "Unknown";
    }
}

// Parse algorithm from command-line string
CompressionAlgorithm parseCompressionAlgorithm(const std::string& algo_str) {
    std::string lower = toLowerCase(algo_str);
    
    if (lower == "lz4")       return CompressionAlgorithm::LZ4;
    if (lower == "snappy")    return CompressionAlgorithm::SNAPPY;
    if (lower == "deflate")   return CompressionAlgorithm::DEFLATE;
    if (lower == "gdeflate")  return CompressionAlgorithm::GDEFLATE;
    if (lower == "zstd")      return CompressionAlgorithm::ZSTD;
    if (lower == "ans")       return CompressionAlgorithm::ANS;
    if (lower == "cascaded")  return CompressionAlgorithm::CASCADED;
    if (lower == "bitcomp")   return CompressionAlgorithm::BITCOMP;
    
    throw std::runtime_error("Unknown compression algorithm: " + algo_str);
}

// Factory function to create compression manager
std::unique_ptr<nvcomp::nvcompManagerBase> createCompressionManager(
    CompressionAlgorithm algo,
    size_t chunk_size,
    cudaStream_t stream,
    const void* sample_input
) {
    (void)sample_input; // Unused for now, reserved for future AUTO mode
    
    switch (algo) {
        case CompressionAlgorithm::LZ4: {
            nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
            // * - NVCOMP_TYPE_(U)CHAR: 1-byte, generic data type
            // * - NVCOMP_TYPE_FLOAT16: 2-byte floating-point data type. Applicable to all half-precision data formats.
            opts.data_type = NVCOMP_TYPE_CHAR; 
            return std::make_unique<nvcomp::LZ4Manager>(
                chunk_size, opts, nvcompBatchedLZ4DecompressDefaultOpts, stream);
        }
        
        case CompressionAlgorithm::SNAPPY: {
            return std::make_unique<nvcomp::SnappyManager>(
                chunk_size, 
                nvcompBatchedSnappyCompressDefaultOpts,
                nvcompBatchedSnappyDecompressDefaultOpts,
                stream);
        }
        
        case CompressionAlgorithm::DEFLATE: {
            nvcompBatchedDeflateCompressOpts_t opts = nvcompBatchedDeflateCompressDefaultOpts;
            return std::make_unique<nvcomp::DeflateManager>(
                chunk_size, opts, nvcompBatchedDeflateDecompressDefaultOpts, stream);
        }
        
        case CompressionAlgorithm::GDEFLATE: {
            return std::make_unique<nvcomp::GdeflateManager>(
                chunk_size,
                nvcompBatchedGdeflateCompressDefaultOpts,
                nvcompBatchedGdeflateDecompressDefaultOpts,
                stream);
        }
        
        case CompressionAlgorithm::ZSTD: {
            return std::make_unique<nvcomp::ZstdManager>(
                chunk_size,
                nvcompBatchedZstdCompressDefaultOpts,
                nvcompBatchedZstdDecompressDefaultOpts,
                stream);
        }
        
        case CompressionAlgorithm::ANS: {
            return std::make_unique<nvcomp::ANSManager>(
                chunk_size,
                nvcompBatchedANSCompressDefaultOpts,
                nvcompBatchedANSDecompressDefaultOpts,
                stream);
        }
        
        case CompressionAlgorithm::CASCADED: {
            nvcompBatchedCascadedCompressOpts_t opts = nvcompBatchedCascadedCompressDefaultOpts;
            opts.type = NVCOMP_TYPE_LONGLONG; // Good for floating-point/scientific data
            return std::make_unique<nvcomp::CascadedManager>(
                chunk_size, opts, nvcompBatchedCascadedDecompressDefaultOpts, stream);
        }
        
        case CompressionAlgorithm::BITCOMP: {
            nvcompBatchedBitcompCompressOpts_t opts = nvcompBatchedBitcompCompressDefaultOpts;
            opts.data_type = NVCOMP_TYPE_LONGLONG; // Good for scientific data
            opts.algorithm = 0; // Default algorithm
            return std::make_unique<nvcomp::BitcompManager>(
                chunk_size, opts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
        }
        
        default:
            throw std::runtime_error("Unsupported compression algorithm");
    }
}


// Factory function to create decompression manager from compressed data
// The algorithm is automatically detected from the compressed data header
std::shared_ptr<nvcomp::nvcompManagerBase> createDecompressionManager(
    const void* d_compressed,
    cudaStream_t stream
) {
    // create_manager automatically detects the compression algorithm
    // from the metadata stored in the compressed data header
    return create_manager(static_cast<const uint8_t*>(d_compressed), stream);
}

