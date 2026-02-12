/**
 * @file gpucompress_api.cpp
 * @brief GPUCompress C API Implementation
 *
 * Implements the public C API defined in gpucompress.h.
 * Wraps existing C++ components for external library integration.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "gpucompress.h"
#include "internal.hpp"
#include "core/compression_factory.hpp"
#include "core/compression_header.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"

#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

/* ============================================================
 * External Functions from CUDA files
 * ============================================================ */

extern "C" {
    // From entropy_kernel.cu
    double gpucompress_entropy_gpu_impl(const void* d_data, size_t num_bytes, void* stream);

    // From qtable_gpu.cu
    int gpucompress_qtable_load_impl(const char* filepath);
    int gpucompress_qtable_is_loaded_impl(void);
    int gpucompress_qtable_get_best_action_impl(int state);
    void gpucompress_qtable_init_default_impl(void);
    void gpucompress_qtable_cleanup_impl(void);
}

/* ============================================================
 * Global State
 * ============================================================ */

namespace {

/** Library initialization state */
std::atomic<bool> g_initialized{false};

/** Reference count for init/cleanup */
std::atomic<int> g_ref_count{0};

/** Mutex for thread-safe initialization */
std::mutex g_init_mutex;

/** Default CUDA stream */
cudaStream_t g_default_stream = nullptr;

/** CUDA device */
int g_cuda_device = 0;

/** Version string */
const char* VERSION_STRING = "1.0.0";

/** Algorithm names */
const char* ALGORITHM_NAMES[] = {
    "auto",
    "lz4",
    "snappy",
    "deflate",
    "gdeflate",
    "zstd",
    "ans",
    "cascaded",
    "bitcomp"
};

/** Error messages */
const char* ERROR_MESSAGES[] = {
    "Success",
    "Invalid input parameter",
    "CUDA operation failed",
    "Compression failed",
    "Decompression failed",
    "Out of memory",
    "Q-Table not loaded",
    "Invalid compression header",
    "Library not initialized",
    "Output buffer too small"
};

/**
 * Calculate normalized Mean Absolute Deviation of float data on host.
 * Normalized by (max - min) to produce values in [0, 1] range,
 * matching the benchmark and Q-Table bin thresholds.
 */
double calculateMADHost(const float* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;
    double sum = 0.0;
    double vmin = static_cast<double>(data[0]);
    double vmax = static_cast<double>(data[0]);
    for (size_t i = 0; i < num_elements; i++) {
        double v = static_cast<double>(data[i]);
        sum += v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    double data_range = vmax - vmin;
    if (data_range == 0.0) return 0.0;
    double mean = sum / static_cast<double>(num_elements);
    double mad_sum = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        mad_sum += std::abs(static_cast<double>(data[i]) - mean);
    }
    return (mad_sum / static_cast<double>(num_elements)) / data_range;
}

/**
 * Calculate normalized mean absolute first derivative of float data on host.
 * Normalized by (max - min) to produce values in [0, 1] range,
 * matching the benchmark and Q-Table bin thresholds.
 */
double calculateFirstDerivativeHost(const float* data, size_t num_elements) {
    if (num_elements < 2) return 0.0;
    double vmin = static_cast<double>(data[0]);
    double vmax = static_cast<double>(data[0]);
    for (size_t i = 0; i < num_elements; i++) {
        double v = static_cast<double>(data[i]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    double data_range = vmax - vmin;
    if (data_range == 0.0) return 0.0;
    double sum = 0.0;
    for (size_t i = 1; i < num_elements; i++) {
        sum += std::abs(static_cast<double>(data[i]) - static_cast<double>(data[i - 1]));
    }
    return (sum / static_cast<double>(num_elements - 1)) / data_range;
}

} // anonymous namespace

/* ============================================================
 * Initialization and Cleanup
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_init(const char* qtable_path) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Increment reference count
    int old_ref = g_ref_count.fetch_add(1);

    if (old_ref > 0) {
        // Already initialized
        return GPUCOMPRESS_SUCCESS;
    }

    // Initialize CUDA
    cudaError_t err = cudaSetDevice(g_cuda_device);
    if (err != cudaSuccess) {
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Create default stream
    err = cudaStreamCreate(&g_default_stream);
    if (err != cudaSuccess) {
        g_ref_count--;
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Load Q-Table if path provided
    if (qtable_path != nullptr && qtable_path[0] != '\0') {
        if (gpucompress_qtable_load_impl(qtable_path) != 0) {
            // Warning but don't fail - can still use without Q-Table
            fprintf(stderr, "Warning: Failed to load Q-Table from %s\n", qtable_path);
        }
    } else {
        // Initialize default Q-Table
        gpucompress_qtable_init_default_impl();
    }

    g_initialized.store(true);
    return GPUCOMPRESS_SUCCESS;
}

extern "C" void gpucompress_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    int old_ref = g_ref_count.fetch_sub(1);

    if (old_ref <= 1) {
        // Last reference, cleanup
        gpucompress_qtable_cleanup_impl();
        if (g_default_stream != nullptr) {
            cudaStreamDestroy(g_default_stream);
            g_default_stream = nullptr;
        }
        g_initialized.store(false);
        g_ref_count.store(0);
    }
}

extern "C" int gpucompress_is_initialized(void) {
    return g_initialized.load() ? 1 : 0;
}

extern "C" gpucompress_config_t gpucompress_default_config(void) {
    gpucompress_config_t config;
    config.algorithm = GPUCOMPRESS_ALGO_LZ4;
    config.preprocessing = GPUCOMPRESS_PREPROC_NONE;
    config.error_bound = 0.0;
    config.cuda_device = -1;
    config.cuda_stream = nullptr;
    return config;
}

/* ============================================================
 * Core Compression
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_compress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (input == nullptr || output == nullptr || output_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (input_size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    // Use default config if not provided
    gpucompress_config_t cfg = config ? *config : gpucompress_default_config();

    // Get CUDA stream
    cudaStream_t stream = cfg.cuda_stream ?
                          static_cast<cudaStream_t>(cfg.cuda_stream) :
                          g_default_stream;

    cudaError_t cuda_err;

    // Allocate GPU input buffer
    size_t aligned_input_size = gpucompress::alignSize(input_size, gpucompress::GPU_ALIGNMENT);
    uint8_t* d_input = nullptr;

    cuda_err = cudaMalloc(&d_input, aligned_input_size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy input to GPU
    cuda_err = cudaMemcpyAsync(d_input, input, input_size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Variables for compression
    gpucompress_algorithm_t algo_to_use = cfg.algorithm;
    unsigned int preproc_to_use = cfg.preprocessing;
    double entropy = 0.0;

    // Auto algorithm selection using Q-Table
    double mad = 0.0;
    double first_derivative = 0.0;

    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO) {
        size_t num_elements = input_size / sizeof(float);
        if (num_elements > 0 && gpucompress_qtable_is_loaded_impl()) {
            int error_level = gpucompress::errorBoundToLevel(cfg.error_bound);
            int action = 0;
            int rc = gpucompress::runAutoStatsPipeline(
                d_input, input_size, error_level,
                gpucompress::getQTableDevicePtr(), stream,
                &action,
                stats ? &entropy : nullptr,
                stats ? &mad : nullptr,
                stats ? &first_derivative : nullptr);
            if (rc == 0) {
                gpucompress::QTableAction decoded = gpucompress::decodeAction(action);
                algo_to_use = static_cast<gpucompress_algorithm_t>(decoded.algorithm + 1);
                preproc_to_use = 0;
                if (decoded.shuffle_size > 0)
                    preproc_to_use |= (decoded.shuffle_size == 4) ?
                        GPUCOMPRESS_PREPROC_SHUFFLE_4 : GPUCOMPRESS_PREPROC_SHUFFLE_2;
                if (decoded.use_quantization)
                    preproc_to_use |= GPUCOMPRESS_PREPROC_QUANTIZE;
            } else {
                algo_to_use = GPUCOMPRESS_ALGO_LZ4;
            }
        } else {
            algo_to_use = GPUCOMPRESS_ALGO_LZ4;
        }
    }

    // Pointer to data to compress (may change with preprocessing)
    uint8_t* d_compress_input = d_input;
    size_t compress_input_size = input_size;
    uint8_t* d_quantized = nullptr;
    uint8_t* d_shuffled = nullptr;
    QuantizationResult quant_result;

    // Apply quantization if requested
    if ((preproc_to_use & GPUCOMPRESS_PREPROC_QUANTIZE) && cfg.error_bound > 0.0) {
        size_t element_size = sizeof(float);  // Default to float
        size_t num_elements = input_size / element_size;

        if (num_elements > 0) {
            QuantizationConfig quant_cfg(
                QuantizationType::LINEAR,
                cfg.error_bound,
                num_elements,
                element_size
            );

            quant_result = quantize_simple(d_compress_input, num_elements,
                                           element_size, quant_cfg, stream);

            if (quant_result.isValid()) {
                d_quantized = static_cast<uint8_t*>(quant_result.d_quantized);
                d_compress_input = d_quantized;
                compress_input_size = quant_result.quantized_bytes;
            }
        }
    }

    // Apply byte shuffle if requested
    unsigned int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(preproc_to_use);
    if (shuffle_size > 0) {
        d_shuffled = byte_shuffle_simple(
            d_compress_input,
            compress_input_size,
            shuffle_size,
            gpucompress::SHUFFLE_CHUNK_SIZE,
            ShuffleKernelType::AUTO,
            stream
        );

        if (d_shuffled != nullptr) {
            d_compress_input = d_shuffled;
        }
    }

    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Create compression manager
    CompressionAlgorithm internal_algo = gpucompress::toInternalAlgorithm(algo_to_use);
    auto compressor = createCompressionManager(
        internal_algo, gpucompress::DEFAULT_CHUNK_SIZE, stream, d_compress_input);

    if (!compressor) {
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    // Configure compression
    CompressionConfig comp_config = compressor->configure_compression(compress_input_size);
    size_t max_compressed_size = comp_config.max_compressed_buffer_size;

    // Allocate output buffer on GPU (with header space)
    size_t header_size = GPUCOMPRESS_HEADER_SIZE;
    size_t total_max_size = header_size + max_compressed_size;

    uint8_t* d_output = nullptr;
    cuda_err = cudaMalloc(&d_output, total_max_size);
    if (cuda_err != cudaSuccess) {
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Pointer to compressed data (after header)
    uint8_t* d_compressed = d_output + header_size;

    // Compress
    try {
        compressor->compress(d_compress_input, d_compressed, comp_config);
    } catch (...) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    // Get compressed size
    size_t compressed_size = compressor->get_compressed_output_size(d_compressed);
    size_t total_size = header_size + compressed_size;

    // Check output buffer size
    if (total_size > *output_size) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        *output_size = total_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    // Build compression header
    CompressionHeader header;
    header.magic = COMPRESSION_MAGIC;
    header.version = COMPRESSION_HEADER_VERSION;
    header.shuffle_element_size = shuffle_size;
    header.original_size = input_size;
    header.compressed_size = compressed_size;

    if (d_quantized && quant_result.isValid()) {
        header.setQuantizationFlags(
            static_cast<uint32_t>(quant_result.type),
            quant_result.actual_precision,
            true
        );
        header.quant_error_bound = quant_result.error_bound;
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

    // Write header to GPU output
    writeHeaderToDevice(d_output, header, stream);

    // Copy output to host
    cuda_err = cudaMemcpyAsync(output, d_output, total_size, cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cuda_err = cudaStreamSynchronize(stream);

    // Cleanup GPU memory
    cudaFree(d_output);
    if (d_quantized) cudaFree(d_quantized);
    if (d_shuffled) cudaFree(d_shuffled);
    cudaFree(d_input);

    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Set output size
    *output_size = total_size;

    // Fill stats if requested
    if (stats != nullptr) {
        stats->original_size = input_size;
        stats->compressed_size = total_size;
        stats->compression_ratio = static_cast<double>(input_size) / compressed_size;
        stats->entropy_bits = entropy;
        stats->mad = mad;
        stats->first_derivative = first_derivative;
        stats->algorithm_used = algo_to_use;
        stats->preprocessing_used = preproc_to_use;
        stats->throughput_mbps = 0.0;  // Would need timing to calculate
    }

    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (input == nullptr || output == nullptr || output_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (input_size < GPUCOMPRESS_HEADER_SIZE) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    cudaStream_t stream = g_default_stream;
    cudaError_t cuda_err;

    // Read header from input
    CompressionHeader header;
    memcpy(&header, input, sizeof(CompressionHeader));

    // Validate header
    if (header.magic != COMPRESSION_MAGIC) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    // Check output buffer size
    if (header.original_size > *output_size) {
        *output_size = header.original_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }

    // Allocate GPU buffer for compressed input
    size_t compressed_size = header.compressed_size;
    size_t header_size = GPUCOMPRESS_HEADER_SIZE;

    uint8_t* d_compressed = nullptr;
    cuda_err = cudaMalloc(&d_compressed, input_size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy compressed data to GPU
    cuda_err = cudaMemcpyAsync(d_compressed, input, input_size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Pointer to actual compressed data (after header)
    uint8_t* d_compressed_data = d_compressed + header_size;

    // Create decompression manager (auto-detects algorithm)
    auto decompressor = createDecompressionManager(d_compressed_data, stream);
    if (!decompressor) {
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_DECOMPRESSION;
    }

    // Configure decompression
    DecompressionConfig decomp_config = decompressor->configure_decompression(d_compressed_data);
    size_t decompressed_size = decomp_config.decomp_data_size;

    // Allocate GPU buffer for decompressed data
    uint8_t* d_decompressed = nullptr;
    cuda_err = cudaMalloc(&d_decompressed, decompressed_size);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Decompress
    try {
        decompressor->decompress(d_decompressed, d_compressed_data, decomp_config);
    } catch (...) {
        cudaFree(d_decompressed);
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_DECOMPRESSION;
    }

    // Apply unshuffle if needed
    uint8_t* d_result = d_decompressed;
    uint8_t* d_unshuffled = nullptr;

    if (header.hasShuffleApplied()) {
        d_unshuffled = byte_unshuffle_simple(
            d_decompressed,
            decompressed_size,
            header.shuffle_element_size,
            gpucompress::SHUFFLE_CHUNK_SIZE,
            ShuffleKernelType::AUTO,
            stream
        );

        if (d_unshuffled != nullptr) {
            d_result = d_unshuffled;
        }
    }

    // Apply dequantization if needed
    void* d_dequantized = nullptr;

    if (header.hasQuantizationApplied()) {
        // Reconstruct quantization result from header
        QuantizationResult quant_result;
        quant_result.scale_factor = header.quant_scale;
        quant_result.data_min = header.data_min;
        quant_result.data_max = header.data_max;
        quant_result.error_bound = header.quant_error_bound;
        quant_result.type = static_cast<QuantizationType>(header.getQuantizationType());
        quant_result.actual_precision = header.getQuantizationPrecision();
        quant_result.num_elements = header.original_size / sizeof(float);
        quant_result.original_element_size = sizeof(float);
        quant_result.d_quantized = d_result;
        quant_result.quantized_bytes = decompressed_size;

        d_dequantized = dequantize_simple(d_result, quant_result, stream);

        if (d_dequantized != nullptr) {
            if (d_unshuffled != nullptr) {
                cudaFree(d_unshuffled);
                d_unshuffled = nullptr;
            }
            d_result = static_cast<uint8_t*>(d_dequantized);
        }
    }

    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        if (d_dequantized) cudaFree(d_dequantized);
        if (d_unshuffled) cudaFree(d_unshuffled);
        cudaFree(d_decompressed);
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Copy result to host
    cuda_err = cudaMemcpyAsync(output, d_result, header.original_size,
                               cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        if (d_dequantized) cudaFree(d_dequantized);
        if (d_unshuffled) cudaFree(d_unshuffled);
        cudaFree(d_decompressed);
        cudaFree(d_compressed);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    cuda_err = cudaStreamSynchronize(stream);

    // Cleanup
    if (d_dequantized) cudaFree(d_dequantized);
    if (d_unshuffled) cudaFree(d_unshuffled);
    cudaFree(d_decompressed);
    cudaFree(d_compressed);

    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *output_size = header.original_size;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" size_t gpucompress_max_compressed_size(size_t input_size) {
    // Worst case: no compression + header + some margin
    return GPUCOMPRESS_HEADER_SIZE + input_size + (input_size / 8) + 1024;
}

extern "C" gpucompress_error_t gpucompress_get_original_size(
    const void* compressed,
    size_t* original_size
) {
    if (compressed == nullptr || original_size == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    const CompressionHeader* header = static_cast<const CompressionHeader*>(compressed);

    if (header->magic != COMPRESSION_MAGIC) {
        return GPUCOMPRESS_ERROR_INVALID_HEADER;
    }

    *original_size = header->original_size;
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Entropy Calculation
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_calculate_entropy(
    const void* data,
    size_t size,
    double* entropy_out
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (data == nullptr || entropy_out == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    cudaStream_t stream = g_default_stream;
    cudaError_t cuda_err;

    // Allocate GPU buffer
    uint8_t* d_data = nullptr;
    cuda_err = cudaMalloc(&d_data, size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    // Copy to GPU
    cuda_err = cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Calculate entropy
    double entropy = gpucompress_entropy_gpu_impl(d_data, size, stream);

    cudaFree(d_data);

    if (entropy < 0.0) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *entropy_out = entropy;
    return GPUCOMPRESS_SUCCESS;
}

extern "C" gpucompress_error_t gpucompress_calculate_entropy_gpu(
    const void* d_data,
    size_t size,
    double* entropy_out,
    void* stream
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (d_data == nullptr || entropy_out == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : g_default_stream;
    double entropy = gpucompress_entropy_gpu_impl(d_data, size, cuda_stream);

    if (entropy < 0.0) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    *entropy_out = entropy;
    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Q-Table Management
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_load_qtable(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    int result = gpucompress_qtable_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_QTABLE_NOT_LOADED;
}

extern "C" int gpucompress_qtable_is_loaded(void) {
    return gpucompress_qtable_is_loaded_impl();
}

extern "C" gpucompress_error_t gpucompress_recommend_config(
    double entropy,
    double error_bound,
    double mad,
    double first_derivative,
    gpucompress_algorithm_t* algorithm_out,
    unsigned int* preprocessing_out
) {
    if (algorithm_out == nullptr || preprocessing_out == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (!gpucompress_qtable_is_loaded_impl()) {
        // Return defaults if Q-Table not loaded
        *algorithm_out = GPUCOMPRESS_ALGO_LZ4;
        *preprocessing_out = GPUCOMPRESS_PREPROC_NONE;
        return GPUCOMPRESS_ERROR_QTABLE_NOT_LOADED;
    }

    int error_level = gpucompress::errorBoundToLevel(error_bound);
    int state = gpucompress::encodeState(entropy, error_level, mad, first_derivative);
    int action = gpucompress_qtable_get_best_action_impl(state);
    gpucompress::QTableAction decoded = gpucompress::decodeAction(action);

    *algorithm_out = static_cast<gpucompress_algorithm_t>(decoded.algorithm + 1);

    *preprocessing_out = 0;
    if (decoded.shuffle_size > 0) {
        *preprocessing_out |= (decoded.shuffle_size == 4) ?
                              GPUCOMPRESS_PREPROC_SHUFFLE_4 :
                              GPUCOMPRESS_PREPROC_SHUFFLE_2;
    }
    if (decoded.use_quantization) {
        *preprocessing_out |= GPUCOMPRESS_PREPROC_QUANTIZE;
    }

    return GPUCOMPRESS_SUCCESS;
}

/* ============================================================
 * Utility Functions
 * ============================================================ */

extern "C" const char* gpucompress_algorithm_name(gpucompress_algorithm_t algorithm) {
    int idx = static_cast<int>(algorithm);
    if (idx >= 0 && idx <= 8) {
        return ALGORITHM_NAMES[idx];
    }
    return "unknown";
}

extern "C" gpucompress_algorithm_t gpucompress_algorithm_from_string(const char* name) {
    if (name == nullptr) {
        return GPUCOMPRESS_ALGO_LZ4;
    }

    for (int i = 0; i <= 8; i++) {
        if (strcmp(name, ALGORITHM_NAMES[i]) == 0) {
            return static_cast<gpucompress_algorithm_t>(i);
        }
    }

    return GPUCOMPRESS_ALGO_LZ4;
}

extern "C" const char* gpucompress_error_string(gpucompress_error_t error) {
    int idx = -static_cast<int>(error);
    if (idx >= 0 && idx <= 9) {
        return ERROR_MESSAGES[idx];
    }
    return "Unknown error";
}

extern "C" const char* gpucompress_version(void) {
    return VERSION_STRING;
}

/* ============================================================
 * GPU Memory API (stubs for now)
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_compress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats,
    void* stream
) {
    // TODO: Implement direct GPU-to-GPU compression
    (void)d_input;
    (void)input_size;
    (void)d_output;
    (void)output_size;
    (void)config;
    (void)stats;
    (void)stream;
    return GPUCOMPRESS_ERROR_INVALID_INPUT;
}

extern "C" gpucompress_error_t gpucompress_decompress_gpu(
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t* output_size,
    void* stream
) {
    // TODO: Implement direct GPU-to-GPU decompression
    (void)d_input;
    (void)input_size;
    (void)d_output;
    (void)output_size;
    (void)stream;
    return GPUCOMPRESS_ERROR_INVALID_INPUT;
}
