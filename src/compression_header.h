/**
 * @file compression_header.h
 * @brief Header format for compressed data with shuffle metadata
 * 
 * This header is prepended to compressed data to store metadata about
 * whether byte shuffle was applied and what element size was used.
 */

#ifndef COMPRESSION_HEADER_H
#define COMPRESSION_HEADER_H

#include <stdint.h>
#include <cstring>

/**
 * @brief Magic number to identify our compressed format
 * ASCII: "GPUC" (GPU Compression)
 */
#define COMPRESSION_MAGIC 0x43555047  // "GPUC" in little-endian

/**
 * @brief Current version of the header format
 */
#define COMPRESSION_HEADER_VERSION 1

/**
 * @brief Header structure for compressed data with shuffle metadata
 * 
 * This header is placed at the beginning of the compressed buffer to store
 * metadata that the decompressor needs to correctly restore the original data.
 * 
 * Layout (32 bytes, aligned):
 * - 4 bytes: Magic number (0x43555047 = "GPUC")
 * - 4 bytes: Version (currently 1)
 * - 4 bytes: Shuffle element size (0 = no shuffle, 2/4/8/etc = shuffled)
 * - 4 bytes: Reserved for future use
 * - 8 bytes: Original uncompressed size
 * - 8 bytes: Compressed data size (excludes this header)
 */
struct CompressionHeader {
    uint32_t magic;               // Magic number for format identification
    uint32_t version;             // Header version
    uint32_t shuffle_element_size; // Element size used for shuffle (0 = none)
    uint32_t reserved;            // Reserved for future use (e.g., checksum)
    uint64_t original_size;       // Original uncompressed data size
    uint64_t compressed_size;     // Size of compressed data (after header)
    
    /**
     * @brief Initialize header with default values
     */
    CompressionHeader() 
        : magic(COMPRESSION_MAGIC)
        , version(COMPRESSION_HEADER_VERSION)
        , shuffle_element_size(0)
        , reserved(0)
        , original_size(0)
        , compressed_size(0)
    {}
    
    /**
     * @brief Check if header is valid
     */
    bool isValid() const {
        return (magic == COMPRESSION_MAGIC) && (version == COMPRESSION_HEADER_VERSION);
    }
    
    /**
     * @brief Check if shuffle was applied
     */
    bool hasShuffleApplied() const {
        return shuffle_element_size > 0;
    }
    
    /**
     * @brief Get total file size (header + compressed data)
     */
    uint64_t getTotalSize() const {
        return sizeof(CompressionHeader) + compressed_size;
    }
    
    /**
     * @brief Print header information
     */
    void print() const {
        printf("Compression Header:\n");
        printf("  Magic: 0x%08X %s\n", magic, 
               (magic == COMPRESSION_MAGIC) ? "✓ Valid" : "✗ Invalid");
        printf("  Version: %u\n", version);
        printf("  Shuffle: %s", hasShuffleApplied() ? "Yes" : "No");
        if (hasShuffleApplied()) {
            printf(" (%u-byte elements)\n", shuffle_element_size);
        } else {
            printf("\n");
        }
        printf("  Original size: %lu bytes (%.2f MB)\n", 
               original_size, original_size / (1024.0 * 1024.0));
        printf("  Compressed size: %lu bytes (%.2f MB)\n",
               compressed_size, compressed_size / (1024.0 * 1024.0));
        printf("  Compression ratio: %.2fx\n", 
               (double)original_size / compressed_size);
        printf("  Header size: %lu bytes\n", sizeof(CompressionHeader));
        printf("  Total size: %lu bytes (%.2f MB)\n",
               getTotalSize(), getTotalSize() / (1024.0 * 1024.0));
    }
} __attribute__((packed));  // Ensure no padding

// Compile-time check that header is expected size
static_assert(sizeof(CompressionHeader) == 32, 
              "CompressionHeader must be exactly 32 bytes");

/**
 * @brief Write header to device memory
 * 
 * @param d_buffer Device buffer where header will be written
 * @param header Header structure to write
 * @param stream CUDA stream for async operation
 */
inline void writeHeaderToDevice(
    void* d_buffer,
    const CompressionHeader& header,
    cudaStream_t stream = 0
) {
    cudaMemcpyAsync(d_buffer, &header, sizeof(CompressionHeader), 
                    cudaMemcpyHostToDevice, stream);
}

/**
 * @brief Read header from device memory
 * 
 * @param d_buffer Device buffer containing header
 * @param header Header structure to populate
 * @param stream CUDA stream for async operation
 */
inline void readHeaderFromDevice(
    const void* d_buffer,
    CompressionHeader& header,
    cudaStream_t stream = 0
) {
    cudaMemcpyAsync(&header, d_buffer, sizeof(CompressionHeader),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

/**
 * @brief Get pointer to compressed data (after header)
 * 
 * @param d_buffer Device buffer containing header + compressed data
 * @return Pointer to compressed data section
 */
inline uint8_t* getCompressedDataPtr(void* d_buffer) {
    return static_cast<uint8_t*>(d_buffer) + sizeof(CompressionHeader);
}

/**
 * @brief Get const pointer to compressed data (after header)
 * 
 * @param d_buffer Device buffer containing header + compressed data
 * @return Const pointer to compressed data section
 */
inline const uint8_t* getCompressedDataPtr(const void* d_buffer) {
    return static_cast<const uint8_t*>(d_buffer) + sizeof(CompressionHeader);
}

#endif // COMPRESSION_HEADER_H
