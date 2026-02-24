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
// #include <nvtx3/nvToolsExt.h>  // Commented out - profiling disabled

// nvCOMP base
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

// Compression factory for algorithm selection
#include "compression/compression_factory.hpp"

// Byte shuffle for preprocessing
#include "preprocessing/byte_shuffle.cuh"
#include "compression/util.h"

// Compression header for metadata
#include "compression/compression_header.h"

// Quantization for lossy preprocessing
#include "preprocessing/quantization.cuh"

using namespace nvcomp;

// CUDA error checking macro
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cerr << "CUDA API call failure \"" #func "\" with " << rt          \
                << " (" << cudaGetErrorString(rt) << ")"                       \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Kernel to compare two buffers for data integrity verification
// Sets *invalid = 1 if any mismatch is found
__global__ void compare_buffers(const uint8_t* ref, const uint8_t* val, int* invalid, size_t n)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  
  for (size_t i = idx; i < n; i += stride) {
    if (ref[i] != val[i]) {
      *invalid = 1;
      return;  // Exit early on first mismatch
    }
  }
}

// Usage information
void usage(const char* prog) {
    printf("Usage: %s <input_file> <output_file> [algorithm] [options]\n", prog);
    printf("\nReads uncompressed data from input_file using GDS,\n");
    printf("optionally applies quantization and/or byte shuffle preprocessing,\n");
    printf("compresses it on GPU with nvcomp, and writes\n");
    printf("compressed data to output_file using GDS.\n");
    printf("\nAvailable algorithms:\n");
    printf("  lz4       - Fast compression, general purpose (default)\n");
    printf("  snappy    - Fastest compression, lower ratio\n");
    printf("  deflate   - Better ratio, slower\n");
    printf("  gdeflate  - Standard compression, compatible\n");
    printf("  zstd      - Best ratio, configurable\n");
    printf("  ans       - Entropy coding, numerical data\n");
    printf("  cascaded  - High compression for floating-point\n");
    printf("  bitcomp   - Lossless for scientific data\n");
    printf("\nOptions:\n");
    printf("  --shuffle <size>            Byte shuffle element size: 4 (float32) (default: 0 = disabled)\n");
    printf("  --quant-type <type>         Quantization method: linear (default: none)\n");
    printf("  --error-bound <value>       Absolute error bound (required if quant-type set)\n");
    printf("\nQuantization:\n");
    printf("  Linear quantization trades controlled precision loss for better compression.\n");
    printf("  Use with --error-bound to set max allowed deviation from original values.\n");
    printf("\nExamples:\n");
    printf("  %s input.bin output.bin lz4                                    # No preprocessing\n", prog);
    printf("  %s input.bin output.bin lz4 --shuffle 4                        # With 4-byte shuffle\n", prog);
    printf("  %s input.bin output.bin lz4 --quant-type linear --error-bound 0.001\n", prog);
    printf("  %s input.bin output.bin zstd --quant-type linear --error-bound 0.0001 --shuffle 4\n", prog);
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        usage(argv[0]);
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    // Default settings
    CompressionAlgorithm algo = CompressionAlgorithm::LZ4;
    unsigned shuffle_element_size = 0;  // 0 = no shuffle
    QuantizationType quant_type = QuantizationType::NONE;
    double error_bound = 0.0;
    size_t quant_element_size = sizeof(float);  // Default to float32

    // Parse arguments
    int arg_idx = 3;

    // Check if next argument is an algorithm (not starting with --)
    if (argc > 3 && argv[3][0] != '-') {
        algo = parseCompressionAlgorithm(argv[3]);
        arg_idx = 4;
    }

    // Parse options
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "--shuffle") == 0 && arg_idx + 1 < argc) {
            shuffle_element_size = atoi(argv[arg_idx + 1]);
            if (shuffle_element_size != 0 && shuffle_element_size != 4) {
                printf("Error: Only 4-byte shuffle is supported (float32)\n");
                return -1;
            }
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--quant-type") == 0 && arg_idx + 1 < argc) {
            quant_type = parseQuantizationType(argv[arg_idx + 1]);
            if (quant_type == QuantizationType::NONE) {
                printf("Error: Unknown quantization type '%s'\n", argv[arg_idx + 1]);
                printf("Valid type: linear\n");
                return -1;
            }
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--error-bound") == 0 && arg_idx + 1 < argc) {
            error_bound = atof(argv[arg_idx + 1]);
            if (error_bound <= 0.0) {
                printf("Error: Error bound must be positive, got %s\n", argv[arg_idx + 1]);
                return -1;
            }
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--element-size") == 0 && arg_idx + 1 < argc) {
            quant_element_size = atoi(argv[arg_idx + 1]);
            if (quant_element_size != 4 && quant_element_size != 8) {
                printf("Error: Element size must be 4 (float) or 8 (double), got %s\n", argv[arg_idx + 1]);
                return -1;
            }
            arg_idx += 2;
        } else {
            printf("Error: Unknown option '%s'\n", argv[arg_idx]);
            usage(argv[0]);
        }
    }

    // Validate quantization settings
    if (quant_type != QuantizationType::NONE && error_bound <= 0.0) {
        printf("Error: --error-bound is required when using --quant-type\n");
        return -1;
    }

    // Print header
    printf("========================================\n");
    printf("   GPU Direct Storage Compression\n");
    if (quant_type != QuantizationType::NONE || shuffle_element_size > 0) {
        printf("   WITH ");
        if (quant_type != QuantizationType::NONE) {
            printf("QUANTIZATION");
            if (shuffle_element_size > 0) printf(" + ");
        }
        if (shuffle_element_size > 0) {
            printf("BYTE SHUFFLE");
        }
        printf(" PREPROCESSING\n");
    }
    printf("========================================\n");
    printf("Algorithm: %s\n", getAlgorithmName(algo).c_str());
    if (shuffle_element_size > 0) {
        printf("Shuffle element size: %u bytes\n", shuffle_element_size);
    }
    if (quant_type != QuantizationType::NONE) {
        printf("Quantization: %s (error bound: %.2e)\n",
               getQuantizationTypeName(quant_type), error_bound);
    }
    printf("\n");

    // ========== Step 1: Open input file for reading ==========
    printf("\n[START] Step 1: Open input file for reading\n");

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
    printf("[END] Step 1: Open input file for reading\n");

    // ========== Step 2: Initialize GPU ==========
    printf("\n[START] Step 2: Initialize GPU\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    printf("[END] Step 2: Initialize GPU\n");

    // ========== Step 3: Allocate GPU memory for input data ==========
    printf("\n[START] Step 3: Allocate GPU memory for input data\n");
    
    // Align to 4KB for GDS optimal performance
    size_t aligned_input_size = ((file_size + 4095) / 4096) * 4096;
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, aligned_input_size));
    printf("\nAllocated %lu bytes (%.2f MB) on GPU for input\n",
           aligned_input_size, aligned_input_size / (1024.0 * 1024.0));
    printf("[END] Step 3: Allocate GPU memory for input data\n");

    // ========== Step 4: Initialize GDS (cuFile) ==========
    printf("\n[START] Step 4: Initialize GDS (cuFile)\n");
    
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
    printf("[END] Step 4: Initialize GDS (cuFile)\n");

    // ========== Step 5: Read data from file to GPU using GDS ==========
    printf("\n[START] Step 5: Read data from file to GPU using GDS\n");
    
    printf("\n###### Reading data from file to GPU via GDS ######\n");
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
    printf("Read %ld bytes directly to GPU (bypassed CPU!)\n", bytes_read);
    printf("[END] Step 5: Read data from file to GPU using GDS\n");

    // ========== Step 5.4: Apply quantization if requested ==========

    uint8_t* d_compress_input = d_input;  // Default: compress original input
    uint8_t* d_quantized = nullptr;
    QuantizationResult quant_result;
    size_t data_size_for_compression = file_size;  // Size to use for compression

    if (quant_type != QuantizationType::NONE) {
        printf("\n[START] Step 5.4: Apply quantization preprocessing\n");
        printf("###### Applying %s quantization preprocessing ######\n",
               getQuantizationTypeName(quant_type));
        printf("Error bound: %.2e\n", error_bound);
        printf("Element size: %zu bytes (%s)\n", quant_element_size,
               quant_element_size == 4 ? "float32" : "float64");

        size_t num_elements = file_size / quant_element_size;

        QuantizationConfig quant_config(
            quant_type,
            error_bound,
            num_elements,
            quant_element_size
        );

        quant_result = quantize_simple(
            d_input,
            num_elements,
            quant_element_size,
            quant_config,
            stream
        );

        if (!quant_result.isValid()) {
            printf("Error: Quantization failed!\n");
            if (input_buf_registered) cuFileBufDeregister(d_input);
            cuFileHandleDeregister(cf_handle_in);
            cuFileDriverClose();
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaStreamDestroy(stream));
            close(fd_input);
            return -1;
        }

        printf("Quantization complete:\n");
        printf("  Data range: [%.6e, %.6e]\n", quant_result.data_min, quant_result.data_max);
        printf("  Precision: %d bits\n", quant_result.actual_precision);
        printf("  Quantized size: %zu bytes (%.2fx reduction)\n",
               quant_result.quantized_bytes, quant_result.getQuantizationRatio());

        d_quantized = static_cast<uint8_t*>(quant_result.d_quantized);
        d_compress_input = d_quantized;
        data_size_for_compression = quant_result.quantized_bytes;
        printf("[END] Step 5.4: Apply quantization preprocessing\n");
    }

    // ========== Step 5.5: Apply byte shuffle if requested ==========

    uint8_t* d_shuffled = nullptr;

    if (shuffle_element_size > 0) {
        printf("\n[START] Step 5.5: Apply byte shuffle preprocessing\n");
        printf("###### Applying byte shuffle preprocessing ######\n");
        printf("Element size: %u bytes\n", shuffle_element_size);

        // Apply shuffle using the simple API
        const size_t SHUFFLE_CHUNK_SIZE = 256 * 1024;  // 256KB chunks
        d_shuffled = byte_shuffle_simple(
            d_compress_input,
            data_size_for_compression,
            shuffle_element_size,
            SHUFFLE_CHUNK_SIZE,
            stream
        );

        if (d_shuffled == nullptr) {
            printf("Error: Byte shuffle failed!\n");
            if (d_quantized) CUDA_CHECK(cudaFree(d_quantized));
            if (input_buf_registered) cuFileBufDeregister(d_input);
            cuFileHandleDeregister(cf_handle_in);
            cuFileDriverClose();
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaStreamDestroy(stream));
            close(fd_input);
            return -1;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("Byte shuffle complete - data reorganized for better compression\n");
        printf("[END] Step 5.5: Apply byte shuffle preprocessing\n");

        // Compress the shuffled data instead
        d_compress_input = d_shuffled;
    }

    // ========== Step 6: Setup compression ==========
    printf("\n[START] Step 6: Setup compression\n");

    printf("\n###### Setting up %s compression ######\n", getAlgorithmName(algo).c_str());

    // Create compression manager using factory
    const size_t CHUNK_SIZE = 1 << 16; // 64KB chunks
    auto compressor = createCompressionManager(algo, CHUNK_SIZE, stream, d_compress_input);

    // Configure compression and get max output size
    // Use data_size_for_compression which accounts for quantization
    const CompressionConfig comp_config = compressor->configure_compression(data_size_for_compression);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;
    
    // Add space for compression header (64 bytes for version 2)
    size_t header_size = sizeof(CompressionHeader);
    size_t max_total_size = header_size + max_compressed_size;
    
    // Align total size to 4KB for GDS
    size_t aligned_total_size = ((max_total_size + 4095) / 4096) * 4096;
    
    uint8_t* d_output_buffer;
    CUDA_CHECK(cudaMalloc(&d_output_buffer, aligned_total_size));
    
    // Pointer to compressed data section (after header)
    uint8_t* d_compressed = d_output_buffer + header_size;
    
    printf("Max compressed size: %lu bytes (%.2f MB)\n", 
           max_compressed_size, max_compressed_size / (1024.0 * 1024.0));
    printf("Header size: %lu bytes\n", header_size);
    printf("Max total size (header + compressed): %lu bytes (%.2f MB)\n",
           max_total_size, max_total_size / (1024.0 * 1024.0));
    printf("Aligned total size for GDS: %lu bytes (%.2f MB)\n",
           aligned_total_size, aligned_total_size / (1024.0 * 1024.0));
    printf("[END] Step 6: Setup compression\n");

    // ========== Step 7: Compress data on GPU ==========
    printf("\n[START] Step 7: Compress data on GPU\n");

    printf("\n###### Compressing data on GPU ######\n");
    compressor->compress(d_compress_input, d_compressed, comp_config);

    // Get actual compressed size
    const size_t compressed_size = compressor->get_compressed_output_size(d_compressed);

    printf("Compressed %lu bytes -> %lu bytes\n", data_size_for_compression, compressed_size);
    printf("  Compression ratio: %.2fx\n", (double)data_size_for_compression / compressed_size);
    if (quant_type != QuantizationType::NONE) {
        printf("  Total reduction (with quantization): %.2fx\n", (double)file_size / compressed_size);
    }
    printf("[END] Step 7: Compress data on GPU\n");

    // ========== Step 7.3: Write compression header with metadata ==========
    printf("\n[START] Step 7.3: Write compression header with metadata\n");

    printf("\n###### Writing compression header with metadata ######\n");

    CompressionHeader header;
    header.magic = COMPRESSION_MAGIC;
    header.version = COMPRESSION_HEADER_VERSION;
    header.shuffle_element_size = shuffle_element_size;
    header.original_size = file_size;
    header.compressed_size = compressed_size;

    // Set quantization metadata if enabled
    if (quant_type != QuantizationType::NONE) {
        header.setQuantizationFlags(
            static_cast<uint32_t>(quant_type),
            quant_result.actual_precision,
            true  // enabled
        );
        header.quant_error_bound = error_bound;
        header.quant_scale = quant_result.scale_factor;
        header.data_min = quant_result.data_min;
        header.data_max = quant_result.data_max;
    } else {
        header.quant_flags = 0;
        header.quant_error_bound = 0.0;
        header.quant_scale = 0.0;
        header.data_min = 0.0;
        header.data_max = 0.0;
    }

    // Write header to device memory (at beginning of output buffer)
    writeHeaderToDevice(d_output_buffer, header, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("Header written:\n");
    header.print();
    
    // Calculate total size to write (header + compressed data)
    size_t total_output_size = header_size + compressed_size;
    size_t final_aligned_size = ((total_output_size + 4095) / 4096) * 4096;
    printf("\nTotal output size: %lu bytes (%.2f MB)\n", 
           total_output_size, total_output_size / (1024.0 * 1024.0));
    printf("Aligned for GDS write: %lu bytes\n", final_aligned_size);
    printf("[END] Step 7.3: Write compression header with metadata\n");

    // ========== Step 7.5: Verify compression with decompression ==========
    printf("\n[START] Step 7.5: Verify compression with decompression\n");
    printf("###### Verifying data integrity ######\n");

    // Configure decompression using the same compressor manager
    DecompressionConfig decomp_config = compressor->configure_decompression(d_compressed);

    printf("Decompressed size: %lu bytes\n", decomp_config.decomp_data_size);

    // Allocate buffer for decompressed data
    uint8_t* d_decompressed;
    CUDA_CHECK(cudaMalloc(&d_decompressed, decomp_config.decomp_data_size));

    // Decompress the data
    compressor->decompress(d_decompressed, d_compressed, decomp_config);

    // ========== Step 7.6: Apply unshuffle if shuffle was used ==========

    uint8_t* d_verify_data = d_decompressed;
    uint8_t* d_unshuffled = nullptr;
    size_t verify_data_size = decomp_config.decomp_data_size;

    if (shuffle_element_size > 0) {
        printf("\n[START] Step 7.6: Apply unshuffle for verification\n");
        printf("Applying byte unshuffle to restore original data format...\n");

        // Unshuffle the decompressed data
        const size_t SHUFFLE_CHUNK_SIZE = 256 * 1024;
        d_unshuffled = byte_unshuffle_simple(
            d_decompressed,
            verify_data_size,
            shuffle_element_size,
            SHUFFLE_CHUNK_SIZE,
            stream
        );

        if (d_unshuffled == nullptr) {
            printf("Error: Byte unshuffle failed!\n");
            CUDA_CHECK(cudaFree(d_decompressed));
            if (d_quantized) CUDA_CHECK(cudaFree(d_quantized));
            if (input_buf_registered) cuFileBufDeregister(d_input);
            cuFileHandleDeregister(cf_handle_in);
            cuFileDriverClose();
            CUDA_CHECK(cudaFree(d_output_buffer));
            if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaStreamDestroy(stream));
            close(fd_input);
            return -1;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("Byte unshuffle complete\n");

        CUDA_CHECK(cudaFree(d_decompressed));
        d_verify_data = d_unshuffled;
        printf("[END] Step 7.6: Apply unshuffle for verification\n");
    }

    // ========== Step 7.7: Apply dequantization if quantization was used ==========

    void* d_dequantized = nullptr;

    if (quant_type != QuantizationType::NONE) {
        printf("\n[START] Step 7.7: Apply dequantization for verification\n");
        printf("Applying dequantization to restore original values...\n");

        d_dequantized = dequantize_simple(d_verify_data, quant_result, stream);

        if (d_dequantized == nullptr) {
            printf("Error: Dequantization failed!\n");
            if (d_unshuffled) CUDA_CHECK(cudaFree(d_unshuffled));
            if (d_quantized) CUDA_CHECK(cudaFree(d_quantized));
            if (input_buf_registered) cuFileBufDeregister(d_input);
            cuFileHandleDeregister(cf_handle_in);
            cuFileDriverClose();
            CUDA_CHECK(cudaFree(d_output_buffer));
            if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaStreamDestroy(stream));
            close(fd_input);
            return -1;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("Dequantization complete\n");

        // Free intermediate buffer
        if (d_unshuffled) {
            CUDA_CHECK(cudaFree(d_unshuffled));
            d_unshuffled = nullptr;
        } else {
            CUDA_CHECK(cudaFree(d_decompressed));
        }
        d_verify_data = static_cast<uint8_t*>(d_dequantized);
        printf("[END] Step 7.7: Apply dequantization for verification\n");
    }

    // ========== Step 7.8: Verify data integrity ==========
    printf("\n[START] Step 7.8: Verify data integrity\n");

    bool verification_passed = false;

    if (quant_type != QuantizationType::NONE) {
        // For lossy quantization: verify error bound is respected
        printf("Verifying error bound (lossy quantization)...\n");

        double max_error = 0.0;
        size_t num_elements = file_size / quant_element_size;

        bool error_bound_ok = verify_error_bound(
            d_input,
            d_verify_data,
            num_elements,
            quant_element_size,
            error_bound,
            stream,
            &max_error
        );

        if (error_bound_ok) {
            printf("PASSED: Error bound verified!\n");
            printf("  Max error: %.6e (bound: %.6e)\n", max_error, error_bound);
            printf("  All %lu elements within specified error bound\n", num_elements);
            verification_passed = true;
        } else {
            printf("FAILED: Error bound violated!\n");
            printf("  Max error: %.6e exceeds bound: %.6e\n", max_error, error_bound);
            verification_passed = false;
        }
    } else {
        // For lossless: verify exact match
        int* d_invalid;
        CUDA_CHECK(cudaMalloc(&d_invalid, sizeof(int)));
        int h_invalid = 0;
        CUDA_CHECK(cudaMemsetAsync(d_invalid, 0, sizeof(int), stream));

        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        int sm_count = deviceProp.multiProcessorCount;

        compare_buffers<<<2 * sm_count, 1024, 0, stream>>>(
            d_input, d_verify_data, d_invalid, file_size);

        CUDA_CHECK(cudaMemcpyAsync(&h_invalid, d_invalid, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_invalid) {
            if (shuffle_element_size > 0) {
                printf("FAILED: Shuffle -> Compress -> Decompress -> Unshuffle did NOT restore original data!\n");
            } else {
                printf("FAILED: Decompressed data does NOT match original input!\n");
            }
            verification_passed = false;
        } else {
            if (shuffle_element_size > 0) {
                printf("PASSED: Full round-trip verification successful!\n");
                printf("  Shuffle -> Compress -> Decompress -> Unshuffle restored original data perfectly!\n");
            } else {
                printf("PASSED: Decompressed data matches original input perfectly!\n");
            }
            printf("  Data integrity verified: %lu bytes verified byte-by-byte\n", file_size);
            verification_passed = true;
        }

        CUDA_CHECK(cudaFree(d_invalid));
    }
    printf("[END] Step 7.8: Verify data integrity\n");
    printf("[END] Step 7.5: Verify compression with decompression\n");

    // Clean up verification resources
    if (d_dequantized) CUDA_CHECK(cudaFree(d_dequantized));
    else if (d_unshuffled) CUDA_CHECK(cudaFree(d_unshuffled));
    else CUDA_CHECK(cudaFree(d_decompressed));

    if (!verification_passed) {
        if (d_quantized) CUDA_CHECK(cudaFree(d_quantized));
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_output_buffer));
        if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaStreamDestroy(stream));
        close(fd_input);
        return -1;
    }

    // ========== Step 8: Open output file and write compressed data ==========
    printf("\n[START] Step 8: Open output file and write compressed data\n");
    printf("###### Writing compressed data via GDS ######\n");
    
    // Open output file with O_DIRECT for GDS
    int fd_out = open(output_file, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0666);
    if (fd_out == -1) {
        printf("Error: Cannot create output file: %s\n", output_file);
        perror("open");
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_output_buffer));
        if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
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
        CUDA_CHECK(cudaFree(d_output_buffer));
        if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }

    // Register output buffer (header + compressed data)
    bool output_buf_registered = true;
    status = cuFileBufRegister(d_output_buffer, aligned_total_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("Warning: Output buffer registration failed\n");
        output_buf_registered = false;
    }

    // Write entire buffer (header + compressed data) from GPU to file using GDS
    ssize_t bytes_written = cuFileWrite(cf_handle_out, d_output_buffer, final_aligned_size, 0, 0);
    if (bytes_written != (ssize_t)final_aligned_size) {
        printf("Error: cuFileWrite returned %ld instead of %lu\n", bytes_written, final_aligned_size);
        if (output_buf_registered) cuFileBufDeregister(d_output_buffer);
        cuFileHandleDeregister(cf_handle_out);
        close(fd_out);
        if (input_buf_registered) cuFileBufDeregister(d_input);
        cuFileHandleDeregister(cf_handle_in);
        cuFileDriverClose();
        CUDA_CHECK(cudaFree(d_output_buffer));
        if (d_shuffled) CUDA_CHECK(cudaFree(d_shuffled));
        CUDA_CHECK(cudaFree(d_input));
        close(fd_input);
        return -1;
    }
    
    printf("✓ Wrote %ld bytes (header + compressed data) to %s via GDS\n", bytes_written, output_file);
    
    // Truncate file to actual size (header + compressed, remove padding)
    if (ftruncate(fd_out, total_output_size) != 0) {
        perror("Warning: ftruncate failed — output file may have trailing padding");
    }
    printf("[END] Step 8: Open output file and write compressed data\n");

    // ========== Step 9: Cleanup ==========
    printf("\n[START] Step 9: Cleanup\n");
    // nvtxRangePushA("Cleanup");
    
    if (output_buf_registered) cuFileBufDeregister(d_output_buffer);
    if (input_buf_registered) cuFileBufDeregister(d_input);
    
    cuFileHandleDeregister(cf_handle_out);
    cuFileHandleDeregister(cf_handle_in);
    cuFileDriverClose();
    
    // IMPORTANT: Destroy compressor BEFORE destroying the stream it uses
    compressor.reset();  // Manually destroy the nvcomp manager
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_output_buffer));  // Free output buffer (contains header + compressed)
    if (d_shuffled != nullptr) {
        CUDA_CHECK(cudaFree(d_shuffled));
    }
    if (d_quantized != nullptr) {
        CUDA_CHECK(cudaFree(d_quantized));
    }
    CUDA_CHECK(cudaFree(d_input));

    close(fd_out);
    close(fd_input);

    printf("Cleanup complete\n");
    printf("[END] Step 9: Cleanup\n");

    // ========== Summary ==========
    printf("\n========================================\n");
    printf("           SUCCESS!\n");
    printf("========================================\n");
    printf("Algorithm: %s\n", getAlgorithmName(algo).c_str());
    if (quant_type != QuantizationType::NONE) {
        printf("Quantization: %s (error bound: %.2e)\n",
               getQuantizationTypeName(quant_type), error_bound);
    } else {
        printf("Quantization: None (lossless)\n");
    }
    printf("Shuffle: %s\n", shuffle_element_size > 0 ?
           (std::to_string(shuffle_element_size) + "-byte").c_str() : "None");
    printf("Input:  %s (%.2f MB)\n", input_file, file_size / (1024.0 * 1024.0));
    printf("Output: %s (%.2f MB, includes metadata header)\n",
           output_file, total_output_size / (1024.0 * 1024.0));
    printf("Compressed data only: %.2f MB\n", compressed_size / (1024.0 * 1024.0));
    printf("Compression ratio: %.2fx\n", (double)file_size / compressed_size);
    printf("Space saved: %.2f MB (%.1f%%)\n",
           (file_size - compressed_size) / (1024.0 * 1024.0),
           100.0 * (1.0 - (double)compressed_size / file_size));
    printf("\n");

    return 0;
}
