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
#include <inttypes.h>
#include <cstring>

/**
 * @brief Magic number to identify our compressed format
 * ASCII: "GPUC" (GPU Compression)
 */
#define COMPRESSION_MAGIC 0x43555047  // "GPUC" in little-endian

/**
 * @brief Current version of the header format
 * Version 1: Original format with shuffle support
 * Version 2: Extended with quantization metadata
 */
#define COMPRESSION_HEADER_VERSION 2

/**
 * @brief Minimum header version that supports quantization
 */
#define COMPRESSION_HEADER_VERSION_QUANT 2

/**
 * @brief Header structure for compressed data with shuffle and quantization metadata
 *
 * This header is placed at the beginning of the compressed buffer to store
 * metadata that the decompressor needs to correctly restore the original data.
 *
 * Layout (64 bytes, aligned):
 * - 4 bytes: Magic number (0x43555047 = "GPUC")
 * - 4 bytes: Version (currently 2)
 * - 4 bytes: Shuffle element size (0 = no shuffle, 2/4/8/etc = shuffled)
 * - 4 bytes: Quantization flags (bits [0-3]=type, [4-7]=precision, [8]=enabled)
 * - 8 bytes: Original uncompressed size
 * - 8 bytes: Compressed data size (excludes this header)
 * --- Extended fields for version >= 2: ---
 * - 8 bytes: Quantization error bound (absolute)
 * - 8 bytes: Quantization scale factor
 * - 8 bytes: Data minimum value (for dequantization)
 * - 8 bytes: Data maximum value
 */
struct CompressionHeader {
    // Core fields (32 bytes) - compatible with version 1
    uint32_t magic;               // Magic number for format identification
    uint32_t version;             // Header version
    uint32_t shuffle_element_size; // Element size used for shuffle (0 = none)
    uint32_t quant_flags;         // Quantization flags: bits [0-3]=type, [4-7]=precision, [8]=enabled
    uint64_t original_size;       // Original uncompressed data size
    uint64_t compressed_size;     // Size of compressed data (after header)

    // Extended fields for quantization (32 bytes) - version 2+
    double   quant_error_bound;   // Absolute error bound used for quantization
    double   quant_scale;         // Scale factor = 1 / (2 * error_bound)
    double   data_min;            // Minimum value in original data
    double   data_max;            // Maximum value in original data

    /**
     * @brief Initialize header with default values
     */
    CompressionHeader()
        : magic(COMPRESSION_MAGIC)
        , version(COMPRESSION_HEADER_VERSION)
        , shuffle_element_size(0)
        , quant_flags(0)
        , original_size(0)
        , compressed_size(0)
        , quant_error_bound(0.0)
        , quant_scale(0.0)
        , data_min(0.0)
        , data_max(0.0)
    {}
    
    /**
     * @brief Check if header is valid
     * Supports both version 1 and version 2 headers
     */
    bool isValid() const {
        return (magic == COMPRESSION_MAGIC) &&
               (version >= 1 && version <= COMPRESSION_HEADER_VERSION);
    }

    /**
     * @brief Check if shuffle was applied
     */
    bool hasShuffleApplied() const {
        return shuffle_element_size > 0;
    }

    /**
     * @brief Check if quantization was applied
     */
    bool hasQuantizationApplied() const {
        return (quant_flags & 0x100) != 0;  // bit 8 = enabled
    }

    /**
     * @brief Get quantization type from flags
     * @return 0=NONE, 1=LINEAR
     */
    uint32_t getQuantizationType() const {
        return quant_flags & 0x0F;
    }

    /**
     * @brief Get quantization precision from flags
     * @return 8, 16, or 32 bits
     */
    int getQuantizationPrecision() const {
        uint32_t prec_code = (quant_flags >> 4) & 0x0F;
        switch (prec_code) {
            case 1: return 8;
            case 2: return 16;
            case 3: return 32;
            default: return 32;
        }
    }

    /**
     * @brief Set quantization flags
     * @param type 0=NONE, 1=LINEAR
     * @param precision 8, 16, or 32 bits
     * @param enabled Whether quantization is enabled
     */
    void setQuantizationFlags(uint32_t type, int precision, bool enabled) {
        quant_flags = 0;
        quant_flags |= (type & 0x0F);           // bits 0-3
        uint32_t prec_code = (precision == 8 ? 1 : (precision == 16 ? 2 : 3));
        quant_flags |= (prec_code << 4);        // bits 4-7
        quant_flags |= (enabled ? (1 << 8) : 0); // bit 8
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
               (magic == COMPRESSION_MAGIC) ? "(Valid)" : "(Invalid)");
        printf("  Version: %u\n", version);
        printf("  Shuffle: %s", hasShuffleApplied() ? "Yes" : "No");
        if (hasShuffleApplied()) {
            printf(" (%u-byte elements)\n", shuffle_element_size);
        } else {
            printf("\n");
        }

        // Print quantization info if enabled
        if (hasQuantizationApplied()) {
            const char* type_str = (getQuantizationType() == 1) ? "Linear" : "Unknown";
            printf("  Quantization: Yes (%s, %d-bit)\n", type_str, getQuantizationPrecision());
            printf("    Error bound: %.6e\n", quant_error_bound);
            printf("    Scale factor: %.6e\n", quant_scale);
            printf("    Data range: [%.6e, %.6e]\n", data_min, data_max);
        } else {
            printf("  Quantization: No\n");
        }

        printf("  Original size: %" PRIu64 " bytes (%.2f MB)\n",
               original_size, original_size / (1024.0 * 1024.0));
        printf("  Compressed size: %" PRIu64 " bytes (%.2f MB)\n",
               compressed_size, compressed_size / (1024.0 * 1024.0));
        if (compressed_size > 0) {
            printf("  Compression ratio: %.2fx\n",
                   (double)original_size / compressed_size);
        } else {
            printf("  Compression ratio: N/A (no compressed data)\n");
        }
        printf("  Header size: %zu bytes\n", sizeof(CompressionHeader));
        printf("  Total size: %" PRIu64 " bytes (%.2f MB)\n",
               getTotalSize(), getTotalSize() / (1024.0 * 1024.0));
    }
} __attribute__((packed));  // Ensure no padding

// Compile-time check that header is expected size (64 bytes for version 2)
static_assert(sizeof(CompressionHeader) == 64,
              "CompressionHeader must be exactly 64 bytes");

/**
 * @brief Write header to device memory
 * 
 * @param d_buffer Device buffer where header will be written
 * @param header Header structure to write
 * @param stream CUDA stream for async operation
 */
inline cudaError_t writeHeaderToDevice(
    void* d_buffer,
    const CompressionHeader& header,
    cudaStream_t stream = 0
) {
    return cudaMemcpyAsync(d_buffer, &header, sizeof(CompressionHeader),
                           cudaMemcpyHostToDevice, stream);
}

/**
 * @brief Read header from device memory
 * 
 * @param d_buffer Device buffer containing header
 * @param header Header structure to populate
 * @param stream CUDA stream for async operation
 */
inline cudaError_t readHeaderFromDevice(
    const void* d_buffer,
    CompressionHeader& header,
    cudaStream_t stream = 0
) {
    cudaError_t err = cudaMemcpyAsync(&header, d_buffer, sizeof(CompressionHeader),
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    return cudaStreamSynchronize(stream);
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
