/*
 * GPU-accelerated file compression using NVIDIA GDS and nvcomp
 * 
 * This program:
 * 1. Reads data from file directly to GPU memory using GDS (bypassing CPU)
 * 2. Compresses data on GPU using dynamically selected nvcomp algorithm
 * 3. Writes compressed data back to file using GDS
 */

#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <memory>
#include <algorithm>
#include <cctype>

#include <cuda_runtime.h>
#include <cufile.h>
#include <nvtx3/nvToolsExt.h>

// nvCOMP base
#include "nvcomp.hpp"

// Individual compression algorithm headers
#include "nvcomp/lz4.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/deflate.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/zstd.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/bitcomp.hpp"

using namespace nvcomp;

// CUDA error checking macro
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cerr << "CUDA API call failure \"" #func "\" with " << rt          \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      throw;                                                                   \
    }                                                                          \
  } while (0)

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
        case CompressionAlgorithm::GZIP:     return "Gzip";
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
    switch (lower) {
        case "lz4"       :  return CompressionAlgorithm::LZ4;
        case "snappy"    :  return CompressionAlgorithm::SNAPPY;
        case "deflate"   :  return CompressionAlgorithm::DEFLATE;
        case "gzip"      :  return CompressionAlgorithm::GZIP;
        case "zstd"      :  return CompressionAlgorithm::ZSTD;
        case "ans"       :  return CompressionAlgorithm::ANS;
        case "cascaded"  :  return CompressionAlgorithm::CASCADED;
        case "bitcomp"   :  return CompressionAlgorithm::BITCOMP;
        default:   
            throw std::runtime_error("Unknown compression algorithm: " + algo_str);
    }
}

// Usage information
void usage(const char* prog) {
    printf("Usage: %s <input_file> <output_file> [algorithm]\n", prog);
    printf("\nReads uncompressed data from input_file using GDS,\n");
    printf("compresses it on GPU with nvcomp, and writes\n");
    printf("compressed data to output_file using GDS.\n");
    printf("\nAvailable algorithms:\n");
    printf("  lz4       - Fast compression, general purpose (default)\n");
    printf("  snappy    - Fastest compression, lower ratio\n");
    printf("  deflate   - Better ratio, slower\n");
    printf("  gzip      - Standard compression, compatible\n");
    printf("  zstd      - Best ratio, configurable\n");
    printf("  ans       - Entropy coding, numerical data\n");
    printf("  cascaded  - High compression for floating-point\n");
    printf("  bitcomp   - Lossless for scientific data\n");
    printf("\nExample:\n");
    printf("  %s noisy_pattern.bin output.bin.lz4 lz4\n", prog);
    printf("  %s noisy_pattern.bin output.bin.zst zstd\n", prog);
    exit(1);
}

// Factory function to create compression manager
std::unique_ptr<nvcomp::nvcompManagerBase> createCompressionManager(
    CompressionAlgorithm algo,
    size_t chunk_size,
    cudaStream_t stream,
    const void* sample_input = nullptr
) {
    (void)sample_input; // Unused for now, reserved for future AUTO mode
    
    switch (algo) {
        case CompressionAlgorithm::LZ4: {
            nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
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
            opts.algorithm = 0; // High throughput mode
            return std::make_unique<nvcomp::DeflateManager>(
                chunk_size, opts, nvcompBatchedDeflateDecompressDefaultOpts, stream);
        }
        
        case CompressionAlgorithm::GZIP: {
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

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    
    // Default to LZ4 if no algorithm specified
    CompressionAlgorithm algo = CompressionAlgorithm::LZ4;
    if (argc == 4) {
        algo = parseCompressionAlgorithm(argv[3]);
    }

    printf("========================================\n");
    printf("   GPU Direct Storage Compression\n");
    printf("========================================\n");
    printf("Algorithm: %s\n\n", getAlgorithmName(algo).c_str());

    // ========== Step 1: Open input file for reading ==========
    nvtxRangePushA("Open Input File");
    
    int fd_input = open(input_file, O_RDONLY | O_DIRECT);
    if (fd_input == -1) {
        printf("Error: Cannot open input file: %s\n", input_file);
        perror("open");
        return -1;
    }

    // Get file size
    struct stat st;
    if (fstat(fd_input, &st) != 0) {
        printf("Error: Cannot get file stats\n");
        close(fd_input);
        return -1;
    }
    size_t file_size = st.st_size;
    
    printf("Input file: %s\n", input_file);
    printf("File size: %lu bytes (%.2f MB)\n", file_size, file_size / (1024.0 * 1024.0));
    
    nvtxRangePop();

    // ========== Step 2: Initialize GPU ==========
    nvtxRangePushA("GPU Initialization");
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int smcount = deviceProp.multiProcessorCount;
    printf("GPU: %s (%d SMs)\n", deviceProp.name, smcount);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    nvtxRangePop();

    // ========== Step 3: Allocate GPU memory for input data ==========
    nvtxRangePushA("Allocate GPU Memory");
    
    // Align to 4KB for GDS optimal performance
    size_t aligned_input_size = ((file_size + 4095) / 4096) * 4096;
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, aligned_input_size));
    printf("\nAllocated %lu bytes (%.2f MB) on GPU for input\n", 
           aligned_input_size, aligned_input_size / (1024.0 * 1024.0));
    
    nvtxRangePop();

    // ========== Step 4: Initialize GDS (cuFile) ==========
    nvtxRangePushA("GDS Setup");
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        printf("Error: cuFileDriverOpen failed (%d)\n", status.err);
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }
    printf("\n✓ GDS driver initialized\n");

    // Register input file with GDS
    CUfileDescr_t cf_descr_in;
    memset(&cf_descr_in, 0, sizeof(CUfileDescr_t));
    cf_descr_in.handle.fd = fd_input;
    cf_descr_in.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileHandle_t cf_handle_in;
    status = cuFileHandleRegister(&cf_handle_in, &cf_descr_in);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Error: cuFileHandleRegister failed (%d)\n", status.err);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }
    printf("✓ Input file registered with GDS\n");

    // Register input buffer (optional but recommended for best performance)
    bool input_buf_registered = true;
    status = cuFileBufRegister(d_input, aligned_input_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Warning: Input buffer registration failed - will use bounce buffer\n");
        input_buf_registered = false;
    } else {
        printf("✓ Input buffer registered\n");
    }
    
    nvtxRangePop();

    // ========== Step 5: Read data from file to GPU using GDS ==========
    nvtxRangePushA("GDS Read");
    
    printf("\n--- Reading data from file to GPU via GDS ---\n");
    ssize_t bytes_read = cuFileRead(cf_handle_in, d_input, aligned_input_size, 0, 0);
    if (bytes_read < 0) {
        printf("Error: cuFileRead failed with return value %ld\n", bytes_read);
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }
    printf("✓ Read %ld bytes directly to GPU (bypassed CPU!)\n", bytes_read);
    
    nvtxRangePop();

    // ========== Step 6: Setup compression ==========
    nvtxRangePushA("Compression Setup");
    
    printf("\n--- Setting up %s compression ---\n", getAlgorithmName(algo).c_str());
    
    // Create compression manager using factory
    const size_t CHUNK_SIZE = 1 << 16; // 64KB chunks
    auto compressor = createCompressionManager(algo, CHUNK_SIZE, stream, d_input);
    
    // Configure compression and get max output size
    const CompressionConfig comp_config = compressor->configure_compression(file_size);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;
    
    // Align compressed output to 4KB for GDS
    size_t aligned_compressed_size = ((max_compressed_size + 4095) / 4096) * 4096;
    
    uint8_t* d_compressed;
    CUDA_CHECK(cudaMalloc(&d_compressed, aligned_compressed_size));
    
    printf("Max compressed size: %lu bytes (%.2f MB)\n", 
           max_compressed_size, max_compressed_size / (1024.0 * 1024.0));
    printf("Aligned compressed size: %lu bytes (%.2f MB)\n",
           aligned_compressed_size, aligned_compressed_size / (1024.0 * 1024.0));
    
    nvtxRangePop();

    // ========== Step 7: Compress data on GPU ==========
    nvtxRangePushA("GPU Compression");
    
    printf("\n--- Compressing data on GPU ---\n");
    compressor->compress(d_input, d_compressed, comp_config);
    
    // Get actual compressed size
    const size_t compressed_size = compressor->get_compressed_output_size(d_compressed);
    size_t final_aligned_size = ((compressed_size + 4095) / 4096) * 4096;
    
    printf("✓ Compressed %lu bytes -> %lu bytes\n", file_size, compressed_size);
    printf("  Compression ratio: %.2fx\n", (double)file_size / compressed_size);
    printf("  Aligned for GDS write: %lu bytes\n", final_aligned_size);
    
    nvtxRangePop();

    // ========== Step 8: Open output file and write compressed data ==========
    nvtxRangePushA("GDS Write");
    
    printf("\n--- Writing compressed data via GDS ---\n");
    
    // Open output file with O_DIRECT for GDS
    int fd_out = open(output_file, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0666);
    if (fd_out == -1) {
        printf("Error: Cannot create output file: %s\n", output_file);
        perror("open");
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_compressed));
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }

    // Register output file with GDS
    CUfileDescr_t cf_descr_out;
    memset(&cf_descr_out, 0, sizeof(CUfileDescr_t));
    cf_descr_out.handle.fd = fd_out;
    cf_descr_out.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileHandle_t cf_handle_out;
    status = cuFileHandleRegister(&cf_handle_out, &cf_descr_out);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Error: cuFileHandleRegister for output failed (%d)\n", status.err);
        close(fd_out);
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_compressed));
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }

    // Register output buffer
    bool output_buf_registered = true;
    status = cuFileBufRegister(d_compressed, aligned_compressed_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Warning: Output buffer registration failed\n");
        output_buf_registered = false;
    }

    // Write compressed data from GPU to file using GDS
    ssize_t bytes_written = cuFileWrite(cf_handle_out, d_compressed, final_aligned_size, 0, 0);
    if (bytes_written != (ssize_t)final_aligned_size) {
        printf("Error: cuFileWrite returned %ld instead of %lu\n", bytes_written, final_aligned_size);
        if (output_buf_registered) cuFileBufDeregister(d_compressed);
        cuFileHandleDeregister(cf_handle_out);
        close(fd_out);
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_compressed));
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }
    
    printf("✓ Wrote %ld bytes to %s via GDS\n", bytes_written, output_file);
    
    // Truncate file to actual compressed size (remove padding)
    ftruncate(fd_out, compressed_size);
    
    nvtxRangePop();

    // ========== Step 9: Cleanup ==========
    nvtxRangePushA("Cleanup");
    
    if (output_buf_registered) cuFileBufDeregister(d_compressed);
    if (input_buf_registered) cuFileBufDeregister(d_input);
    
    cuFileHandleDeregister(cf_handle_out);
    cuFileHandleDeregister(cf_handle_in);
    cuFileDriverClose();
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_compressed));
    CUDA_CHECK(cudaFree(d_input));
    
    close(fd_out);
    close(fd_input);
    
    printf("✓ Cleanup complete\n");
    
    nvtxRangePop();

    // ========== Summary ==========
    printf("\n========================================\n");
    printf("           SUCCESS!\n");
    printf("========================================\n");
    printf("Algorithm: %s\n", getAlgorithmName(algo).c_str());
    printf("Input:  %s (%.2f MB)\n", input_file, file_size / (1024.0 * 1024.0));
    printf("Output: %s (%.2f MB)\n", output_file, compressed_size / (1024.0 * 1024.0));
    printf("Compression ratio: %.2fx\n", (double)file_size / compressed_size);
    printf("Space saved: %.2f MB (%.1f%%)\n", 
           (file_size - compressed_size) / (1024.0 * 1024.0),
           100.0 * (1.0 - (double)compressed_size / file_size));
    printf("\n");

    return 0;
}
