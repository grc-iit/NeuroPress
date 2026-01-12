/*
 * GPU-accelerated file decompression using NVIDIA GDS and nvcomp
 * 
 * This program:
 * 1. Reads compressed data from file directly to GPU memory using GDS (bypassing CPU)
 * 2. Decompresses data on GPU using nvcomp (algorithm auto-detected)
 * 3. Writes decompressed data back to file using GDS
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
// #include <nvtx3/nvToolsExt.h>  // Commented out - profiling disabled

// nvCOMP base
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

// Compression factory for algorithm selection
#include "CompressionFactory.hpp"

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

// Usage information
void usage(const char* prog) {
    printf("Usage: %s <compressed_input_file> <decompressed_output_file>\n", prog);
    printf("\nReads compressed data from input_file using GDS,\n");
    printf("decompresses it on GPU with nvcomp (auto-detects algorithm), and writes\n");
    printf("decompressed data to output_file using GDS.\n");
    printf("\nSupported compression formats:\n");
    printf("  - LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp\n");
    printf("  - Algorithm is automatically detected from compressed data header\n");
    printf("\nExample:\n");
    printf("  %s compressed.bin.lz4 decompressed.bin\n", prog);
    printf("  %s compressed.bin.zst decompressed.bin\n", prog);
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        usage(argv[0]);
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    printf("========================================\n");
    printf("   GPU Direct Storage Decompression\n");
    printf("========================================\n");
    printf("Algorithm: Auto-detected from compressed data\n\n");

    // ========== Step 1: Open input file for reading ==========
    
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
    
    printf("Input file (compressed): %s\n", input_file);
    printf("Compressed file size: %lu bytes (%.2f MB)\n", file_size, file_size / (1024.0 * 1024.0));
    
    // ========== Step 2: Initialize GPU ==========
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ========== Step 3: Allocate GPU memory for compressed input data ==========
    
    // Align to 4KB for GDS optimal performance
    size_t aligned_input_size = ((file_size + 4095) / 4096) * 4096;
    
    uint8_t* d_compressed;
    CUDA_CHECK(cudaMalloc(&d_compressed, aligned_input_size));
    printf("\nAllocated %lu bytes (%.2f MB) on GPU for compressed input\n", 
           aligned_input_size, aligned_input_size / (1024.0 * 1024.0));
    
    // ========== Step 4: Initialize GDS (cuFile) ==========
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        printf("Error: cuFileDriverOpen failed (%d)\n", status.err);
        CUDA_CHECK(cudaFree(d_compressed));
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
        CUDA_CHECK(cudaFree(d_compressed));
        close(fd_input);
        return -1;
    }
    printf("✓ Input file registered with GDS\n");

    // Register compressed input buffer (optional but recommended for best performance)
    bool input_buf_registered = true;
    status = cuFileBufRegister(d_compressed, aligned_input_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Warning: Compressed input buffer registration failed - will use bounce buffer\n");
        input_buf_registered = false;
    } else {
        printf("✓ Compressed input buffer registered\n");
    }
    
    // ========== Step 5: Read compressed data from file to GPU using GDS ==========
    
    printf("\n--- Reading compressed data from file to GPU via GDS ---\n");
    ssize_t bytes_read = cuFileRead(cf_handle_in, d_compressed, aligned_input_size, 0, 0);
    if (bytes_read < 0) {
        printf("Error: cuFileRead failed with return value %ld\n", bytes_read);
        if (input_buf_registered) cuFileBufDeregister(d_compressed);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_compressed));
        close(fd_input);
        return -1;
    }
    printf("✓ Read %ld bytes of compressed data directly to GPU (bypassed CPU!)\n", bytes_read);
    
    // ========== Step 6: Setup decompression ==========
    
    printf("\n--- Setting up decompression (auto-detecting algorithm) ---\n");
    
    // Create decompression manager - algorithm is auto-detected from compressed data
    auto decompressor = createDecompressionManager(d_compressed, stream);

    // Configure decompression and get output size
    const DecompressionConfig decomp_config = decompressor->configure_decompression(d_compressed);

    size_t decompressed_size = decomp_config.decomp_data_size;
    
    printf("Decompressed size: %lu bytes (%.2f MB)\n",
        decompressed_size, decompressed_size / (1024.0 * 1024.0));
    
    // Align decompressed output to 4KB for GDS
    size_t aligned_decompressed_size = ((decompressed_size + 4095) / 4096) * 4096;
    
    uint8_t* d_decompressed;
    CUDA_CHECK(cudaMalloc(&d_decompressed, aligned_decompressed_size));
    
    printf("Aligned decompressed size for GDS: %lu bytes (%.2f MB)\n",
        aligned_decompressed_size, aligned_decompressed_size / (1024.0 * 1024.0));
    

    // ========== Step 7: Decompress data on GPU ==========
    
    printf("\n--- Decompressing data on GPU ---\n");
    decompressor->decompress(d_decompressed, d_compressed, decomp_config);
    
    printf("✓ Decompressed %lu bytes -> %lu bytes\n", file_size, decompressed_size);
    printf("  Decompression ratio: %.2fx\n", (double)decompressed_size / file_size);
    
    size_t final_aligned_size = ((decompressed_size + 4095) / 4096) * 4096;
    printf("  Aligned for GDS write: %lu bytes\n", final_aligned_size);

    // ========== Step 8: Open output file and write decompressed data ==========
    
    printf("\n--- Writing decompressed data via GDS ---\n");
    
    // Open output file with O_DIRECT for GDS
    int fd_out = open(output_file, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0666);
    if (fd_out == -1) {
        printf("Error: Cannot create output file: %s\n", output_file);
        perror("open");
        if (input_buf_registered) cuFileBufDeregister(d_compressed);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_decompressed));
        CUDA_CHECK(cudaFree(d_compressed));
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
        if (input_buf_registered) cuFileBufDeregister(d_compressed);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_decompressed));
        CUDA_CHECK(cudaFree(d_compressed));
        close(fd_input);
        return -1;
    }

    // Register output buffer (decompressed data)
    bool output_buf_registered = true;
    status = cuFileBufRegister(d_decompressed, aligned_decompressed_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Warning: Output buffer registration failed\n");
        output_buf_registered = false;
    }

    // Write decompressed data from GPU to file using GDS
    ssize_t bytes_written = cuFileWrite(cf_handle_out, d_decompressed, final_aligned_size, 0, 0);
    if (bytes_written != (ssize_t)final_aligned_size) {
        printf("Error: cuFileWrite returned %ld instead of %lu\n", bytes_written, final_aligned_size);
        if (output_buf_registered) cuFileBufDeregister(d_decompressed);
        cuFileHandleDeregister(cf_handle_out);
        close(fd_out);
        if (input_buf_registered) cuFileBufDeregister(d_compressed);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_decompressed));
        CUDA_CHECK(cudaFree(d_compressed));
        close(fd_input);
        return -1;
    }
    
    printf("✓ Wrote %ld bytes of decompressed data to %s via GDS\n", bytes_written, output_file);
    
    // Truncate file to actual decompressed size (remove padding)
    ftruncate(fd_out, decompressed_size);
    
    // nvtxRangePop();

    // ========== Step 9: Cleanup ==========
    
    if (output_buf_registered) cuFileBufDeregister(d_decompressed);
    if (input_buf_registered) cuFileBufDeregister(d_compressed);
    
    cuFileHandleDeregister(cf_handle_out);
    cuFileHandleDeregister(cf_handle_in);
    cuFileDriverClose();
    
    // IMPORTANT: Destroy decompressor BEFORE destroying the stream it uses
    decompressor.reset();  // Manually destroy the nvcomp manager
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_decompressed));
    CUDA_CHECK(cudaFree(d_compressed));
    
    close(fd_out);
    close(fd_input);
    
    printf("✓ Cleanup complete\n");

    // ========== Summary ==========
    printf("\n========================================\n");
    printf("           SUCCESS!\n");
    printf("========================================\n");
    printf("Algorithm: Auto-detected\n");
    printf("Input (compressed):  %s (%.2f MB)\n", input_file, file_size / (1024.0 * 1024.0));
    printf("Output (decompressed): %s (%.2f MB)\n", output_file, decompressed_size / (1024.0 * 1024.0));
    printf("Decompression ratio: %.2fx\n", (double)decompressed_size / file_size);
    printf("Data expanded: %.2f MB\n", 
           (decompressed_size - file_size) / (1024.0 * 1024.0));
    printf("\n");

    return 0;
}
