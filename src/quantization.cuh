/**
 * @file quantization.cuh
 * @brief Error-Bound Quantization Preprocessor for GPU Compression
 *
 * Provides GPU-accelerated quantization as a preprocessing step to significantly
 * improve compression ratios when using nvcomp lossless algorithms. This trades
 * controlled precision loss for much higher compression ratios.
 *
 * Supported quantization methods:
 * 1. LINEAR: Simple quantization using round(value / (2 * error_bound))
 * 2. LORENZO_1D: Prediction-based quantization for smooth data
 * 3. BLOCK_TRANSFORM: ZFP-style 4-element block orthogonal transform
 *
 * Error bound guarantee:
 * - ABS (Absolute): |original - decompressed| <= error_bound
 *
 * Supported data types: float32, float64
 * Output precision: Adaptive (int8/16/32 based on error bound & data range)
 */

#ifndef QUANTIZATION_CUH
#define QUANTIZATION_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cctype>
#include <cstring>

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Quantization method types
 */
enum class QuantizationType : uint32_t {
    NONE = 0,           // No quantization applied
    LINEAR = 1,         // Simple linear quantization: round(value / (2*eb))
    LORENZO_1D = 2,     // Lorenzo 1D prediction-based quantization
    BLOCK_TRANSFORM = 3 // ZFP-style 4-element block transform
};

/**
 * @brief Output precision for quantized values
 */
enum class QuantizationPrecision : uint32_t {
    AUTO = 0,   // Automatically select based on data range and error bound
    INT8 = 1,   // Force 8-bit output (range: -128 to 127)
    INT16 = 2,  // Force 16-bit output (range: -32768 to 32767)
    INT32 = 3   // Force 32-bit output (full precision)
};

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * @brief Configuration for quantization operation
 */
struct QuantizationConfig {
    QuantizationType type;          // Quantization method to use
    QuantizationPrecision precision; // Output precision (AUTO recommended)
    double error_bound;             // Absolute error bound (must be > 0)
    size_t num_elements;            // Number of elements to quantize
    size_t element_size;            // sizeof(float) = 4 or sizeof(double) = 8

    /**
     * @brief Default constructor with sensible defaults
     */
    QuantizationConfig()
        : type(QuantizationType::LINEAR)
        , precision(QuantizationPrecision::AUTO)
        , error_bound(0.001)
        , num_elements(0)
        , element_size(sizeof(float))
    {}

    /**
     * @brief Convenience constructor
     */
    QuantizationConfig(
        QuantizationType t,
        double eb,
        size_t n_elements,
        size_t elem_size,
        QuantizationPrecision prec = QuantizationPrecision::AUTO
    )
        : type(t)
        , precision(prec)
        , error_bound(eb)
        , num_elements(n_elements)
        , element_size(elem_size)
    {}

    /**
     * @brief Validate configuration
     */
    bool isValid() const {
        return (type != QuantizationType::NONE) &&
               (error_bound > 0.0) &&
               (num_elements > 0) &&
               (element_size == 4 || element_size == 8);
    }
};

/**
 * @brief Result of quantization operation with metadata for dequantization
 */
struct QuantizationResult {
    void* d_quantized;              // Device pointer to quantized data
    size_t quantized_bytes;         // Output size in bytes
    int actual_precision;           // 8, 16, or 32 bits used
    double data_min;                // Minimum value in original data
    double data_max;                // Maximum value in original data
    double scale_factor;            // Quantization scale = 1 / (2 * error_bound)
    double error_bound;             // Error bound used
    QuantizationType type;          // Quantization method used
    size_t num_elements;            // Number of elements
    size_t original_element_size;   // Original element size (4 or 8)

    /**
     * @brief Default constructor
     */
    QuantizationResult()
        : d_quantized(nullptr)
        , quantized_bytes(0)
        , actual_precision(0)
        , data_min(0.0)
        , data_max(0.0)
        , scale_factor(0.0)
        , error_bound(0.0)
        , type(QuantizationType::NONE)
        , num_elements(0)
        , original_element_size(0)
    {}

    /**
     * @brief Check if result is valid
     */
    bool isValid() const {
        return (d_quantized != nullptr) &&
               (quantized_bytes > 0) &&
               (actual_precision == 8 || actual_precision == 16 || actual_precision == 32);
    }

    /**
     * @brief Get compression ratio from quantization alone
     */
    double getQuantizationRatio() const {
        if (quantized_bytes == 0) return 0.0;
        return (double)(num_elements * original_element_size) / quantized_bytes;
    }

    /**
     * @brief Print result information
     */
    void print() const {
        const char* type_str = "Unknown";
        switch (type) {
            case QuantizationType::NONE: type_str = "None"; break;
            case QuantizationType::LINEAR: type_str = "Linear"; break;
            case QuantizationType::LORENZO_1D: type_str = "Lorenzo 1D"; break;
            case QuantizationType::BLOCK_TRANSFORM: type_str = "Block Transform"; break;
        }

        printf("Quantization Result:\n");
        printf("  Method: %s\n", type_str);
        printf("  Error bound: %.6e\n", error_bound);
        printf("  Data range: [%.6e, %.6e]\n", data_min, data_max);
        printf("  Scale factor: %.6e\n", scale_factor);
        printf("  Precision: %d bits\n", actual_precision);
        printf("  Original size: %lu bytes\n", num_elements * original_element_size);
        printf("  Quantized size: %lu bytes\n", quantized_bytes);
        printf("  Quantization ratio: %.2fx\n", getQuantizationRatio());
    }
};

// ============================================================================
// Precision Selection
// ============================================================================

/**
 * @brief Compute required precision based on data range and error bound
 *
 * Determines the minimum bits needed to represent all quantized values
 * without overflow.
 *
 * @param data_range Maximum - minimum value in data
 * @param error_bound Absolute error bound
 * @return 8, 16, or 32 bits required
 */
inline int compute_required_precision(double data_range, double error_bound) {
    // Number of quantization bins needed
    double num_bins = data_range / (2.0 * error_bound);

    // Add margin for safety and rounding
    num_bins = num_bins * 1.1;  // 10% safety margin

    if (num_bins <= 127) {
        return 8;   // int8 sufficient (-128 to 127)
    } else if (num_bins <= 32767) {
        return 16;  // int16 sufficient (-32768 to 32767)
    } else {
        return 32;  // int32 needed
    }
}

/**
 * @brief Get bytes per quantized element for given precision
 */
inline size_t precision_to_bytes(int precision) {
    switch (precision) {
        case 8:  return 1;
        case 16: return 2;
        case 32: return 4;
        default: return 4;  // Default to 32-bit
    }
}

// ============================================================================
// Simple High-Level API (RECOMMENDED)
// ============================================================================

/**
 * @brief Quantize a device buffer with automatic precision selection
 *
 * This is the simplest API - just pass your device buffer!
 *
 * How it works:
 * 1. Computes data range (min/max) on GPU
 * 2. Determines optimal precision based on range and error bound
 * 3. Allocates output buffer
 * 4. Applies quantization kernel
 * 5. Returns result with metadata for dequantization
 *
 * @param d_input       Pointer to input data on device (float or double)
 * @param num_elements  Number of elements (not bytes!)
 * @param element_size  Size of each element: sizeof(float)=4 or sizeof(double)=8
 * @param config        Quantization configuration
 * @param stream        CUDA stream (default: 0)
 *
 * @return QuantizationResult with output pointer and metadata
 *
 * Example:
 *   float* d_data = ...;
 *   QuantizationConfig config(QuantizationType::LINEAR, 0.001, n_elements, sizeof(float));
 *   QuantizationResult result = quantize_simple(d_data, n_elements, sizeof(float), config);
 *   // Use result.d_quantized for compression...
 *   cudaFree(result.d_quantized);
 */
QuantizationResult quantize_simple(
    void* d_input,
    size_t num_elements,
    size_t element_size,
    QuantizationConfig config,
    cudaStream_t stream = 0
);

/**
 * @brief Dequantize a device buffer back to original data type
 *
 * Reverses the quantization process using metadata from QuantizationResult.
 *
 * @param d_quantized   Pointer to quantized data on device
 * @param metadata      QuantizationResult from quantize_simple()
 * @param stream        CUDA stream (default: 0)
 *
 * @return Pointer to dequantized output on device (caller must cudaFree!)
 *         Returns nullptr on error.
 *
 * Example:
 *   void* d_restored = dequantize_simple(result.d_quantized, result);
 *   // d_restored contains float/double data restored within error bounds
 *   cudaFree(d_restored);
 */
void* dequantize_simple(
    void* d_quantized,
    const QuantizationResult& metadata,
    cudaStream_t stream = 0
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Parse quantization type from string
 *
 * @param str String representation ("linear", "lorenzo", "block")
 * @return QuantizationType enum value, NONE if invalid
 */
inline QuantizationType parseQuantizationType(const char* str) {
    if (str == nullptr) return QuantizationType::NONE;

    // Convert to lowercase for comparison
    char lower[64];
    size_t len = strlen(str);
    if (len >= sizeof(lower)) len = sizeof(lower) - 1;
    for (size_t i = 0; i < len; i++) {
        lower[i] = static_cast<char>(tolower(static_cast<unsigned char>(str[i])));
    }
    lower[len] = '\0';

    if (strcmp(lower, "linear") == 0) return QuantizationType::LINEAR;
    if (strcmp(lower, "lorenzo") == 0 || strcmp(lower, "lorenzo1d") == 0 ||
        strcmp(lower, "lorenzo_1d") == 0) return QuantizationType::LORENZO_1D;
    if (strcmp(lower, "block") == 0 || strcmp(lower, "blocktransform") == 0 ||
        strcmp(lower, "block_transform") == 0) return QuantizationType::BLOCK_TRANSFORM;

    return QuantizationType::NONE;
}

/**
 * @brief Get string name for quantization type
 */
inline const char* getQuantizationTypeName(QuantizationType type) {
    switch (type) {
        case QuantizationType::NONE: return "None";
        case QuantizationType::LINEAR: return "Linear";
        case QuantizationType::LORENZO_1D: return "Lorenzo 1D";
        case QuantizationType::BLOCK_TRANSFORM: return "Block Transform";
        default: return "Unknown";
    }
}

/**
 * @brief Verify error bound is respected after round-trip
 *
 * Compares original and restored data, checks max error <= error_bound.
 *
 * @param d_original    Original data on device
 * @param d_restored    Restored data on device (after dequantization)
 * @param num_elements  Number of elements
 * @param element_size  sizeof(float) or sizeof(double)
 * @param error_bound   Error bound that should be respected
 * @param stream        CUDA stream
 * @param max_error_out Optional output: maximum error found
 *
 * @return true if all errors are within bound, false otherwise
 */
bool verify_error_bound(
    const void* d_original,
    const void* d_restored,
    size_t num_elements,
    size_t element_size,
    double error_bound,
    cudaStream_t stream = 0,
    double* max_error_out = nullptr
);

// ============================================================================
// Header Metadata Helpers (for compression_header.h integration)
// ============================================================================

/**
 * @brief Pack quantization flags for header storage
 *
 * Layout: bits [0-3]=type, [4-7]=precision, [8]=enabled
 */
inline uint32_t pack_quant_flags(QuantizationType type, int precision, bool enabled) {
    uint32_t flags = 0;
    flags |= (static_cast<uint32_t>(type) & 0x0F);           // bits 0-3
    flags |= ((precision == 8 ? 1 : (precision == 16 ? 2 : 3)) << 4);  // bits 4-7
    flags |= (enabled ? (1 << 8) : 0);                        // bit 8
    return flags;
}

/**
 * @brief Unpack quantization type from header flags
 */
inline QuantizationType unpack_quant_type(uint32_t flags) {
    return static_cast<QuantizationType>(flags & 0x0F);
}

/**
 * @brief Unpack precision from header flags
 */
inline int unpack_quant_precision(uint32_t flags) {
    uint32_t prec_code = (flags >> 4) & 0x0F;
    switch (prec_code) {
        case 1: return 8;
        case 2: return 16;
        case 3: return 32;
        default: return 32;
    }
}

/**
 * @brief Check if quantization is enabled from header flags
 */
inline bool is_quant_enabled(uint32_t flags) {
    return (flags & 0x100) != 0;
}

#endif // QUANTIZATION_CUH
