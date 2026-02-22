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
#include <utility>
#include <vector>

#include "gpucompress.h"
#include "internal.hpp"
#include "core/compression_factory.hpp"
#include "core/compression_header.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"
#include "experience_buffer.h"

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

    // From nn_gpu.cu
    int gpucompress_nn_load_impl(const char* filepath);
    int gpucompress_nn_is_loaded_impl(void);
    void gpucompress_nn_cleanup_impl(void);
    void gpucompress_nn_set_criterion_impl(int criterion);
    void* gpucompress_nn_get_device_ptr_impl(void);

    // From nn_reinforce.cpp
    int nn_reinforce_init(const void* d_weights);
    void nn_reinforce_add_sample(const float input_raw[15], double actual_ratio,
                                 double actual_comp_time);
    int nn_reinforce_apply(void* d_weights, float learning_rate);
    void nn_reinforce_cleanup(void);
    void nn_reinforce_get_last_stats(float* grad_norm, int* num_samples,
                                      int* was_clipped);
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

/** Active learning state */
bool g_active_learning_enabled = false;
double g_exploration_threshold = 0.20;  // 20% MAPE
char g_experience_path[512] = "";

/** Online reinforcement state */
bool g_reinforce_enabled = false;
float g_reinforce_lr = 1e-4f;
double g_reinforce_threshold = 0.60;  // 60% ratio MAPE
double g_reinforce_ct_threshold = 0.60;  // 60% comp_time MAPE
bool g_reinforce_initialized = false;

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

    // Load model if path provided (auto-detect .nnwt vs Q-Table)
    if (qtable_path != nullptr && qtable_path[0] != '\0') {
        std::string path(qtable_path);
        bool loaded = false;

        // Try NN weights if .nnwt extension
        if (path.size() >= 5 && path.substr(path.size() - 5) == ".nnwt") {
            if (gpucompress_nn_load_impl(qtable_path) == 0) {
                loaded = true;
            } else {
                fprintf(stderr, "Warning: Failed to load NN weights from %s\n", qtable_path);
            }
        }

        // Try Q-Table if not .nnwt or NN load failed
        if (!loaded) {
            if (gpucompress_qtable_load_impl(qtable_path) != 0) {
                fprintf(stderr, "Warning: Failed to load Q-Table from %s\n", qtable_path);
            }
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
        nn_reinforce_cleanup();
        g_reinforce_initialized = false;
        experience_buffer_cleanup();
        g_active_learning_enabled = false;
        gpucompress_nn_cleanup_impl();
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

    // Auto algorithm selection
    double mad = 0.0;
    double second_derivative = 0.0;
    bool nn_was_used = false;
    bool sgd_fired = false;
    int nn_action = 0;
    float predicted_ratio = 0.0f;
    float predicted_comp_time = 0.0f;
    int top_actions[32] = {0};
    bool is_ood = false;

    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO) {
        size_t num_elements = input_size / sizeof(float);
        int action = 0;
        int rc = -1;

        if (num_elements > 0 && gpucompress_nn_is_loaded_impl()) {
            // Neural Network inference (preferred)
            // Always get stats when active learning is enabled
            bool need_stats = (stats != nullptr) || g_active_learning_enabled;
            float* p_ratio = (stats != nullptr || g_active_learning_enabled) ? &predicted_ratio : nullptr;
            float* p_comp_time = p_ratio ? &predicted_comp_time : nullptr;
            int* p_top = g_active_learning_enabled ? top_actions : nullptr;

            rc = gpucompress::runAutoStatsNNPipeline(
                d_input, input_size, cfg.error_bound, stream,
                &action,
                need_stats ? &entropy : nullptr,
                need_stats ? &mad : nullptr,
                need_stats ? &second_derivative : nullptr,
                p_ratio,
                p_comp_time,
                p_top);

            if (rc == 0) {
                nn_was_used = true;
                nn_action = action;

                // Check OOD
                if (g_active_learning_enabled) {
                    is_ood = gpucompress::isInputOOD(
                        entropy, mad, second_derivative,
                        input_size, cfg.error_bound);
                }
            }
        } else if (num_elements > 0 && gpucompress_qtable_is_loaded_impl()) {
            // Q-Table fallback
            int error_level = gpucompress::errorBoundToLevel(cfg.error_bound);
            rc = gpucompress::runAutoStatsPipeline(
                d_input, input_size, error_level,
                gpucompress::getQTableDevicePtr(), stream,
                &action,
                stats ? &entropy : nullptr,
                stats ? &mad : nullptr,
                stats ? &second_derivative : nullptr);
        }

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

    // Compress (with timing for reinforcement)
    cudaEvent_t t_start, t_stop;
    float primary_comp_time_ms = 0.0f;
    bool timing_ok = (cudaEventCreate(&t_start) == cudaSuccess &&
                      cudaEventCreate(&t_stop) == cudaSuccess);
    if (timing_ok) cudaEventRecord(t_start, stream);

    try {
        compressor->compress(d_compress_input, d_compressed, comp_config);
    } catch (...) {
        if (timing_ok) { cudaEventDestroy(t_start); cudaEventDestroy(t_stop); }
        cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_shuffled) cudaFree(d_shuffled);
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_COMPRESSION;
    }

    if (timing_ok) {
        cudaEventRecord(t_stop, stream);
        cudaEventSynchronize(t_stop);
        cudaEventElapsedTime(&primary_comp_time_ms, t_start, t_stop);
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);
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

    // Cleanup GPU output and preprocessing buffers (but keep d_input for exploration)
    cudaFree(d_output);
    if (d_quantized) cudaFree(d_quantized);
    if (d_shuffled) cudaFree(d_shuffled);

    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Set output size
    *output_size = total_size;

    // Active learning: Level 1 passive collection + Level 2 exploration
    if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && nn_was_used &&
        g_active_learning_enabled) {
        double actual_ratio = static_cast<double>(input_size) /
                              static_cast<double>(compressed_size);

        // Level 1: Store passive sample
        ExperienceSample sample;
        sample.entropy = entropy;
        sample.mad = mad;
        sample.second_derivative = second_derivative;
        sample.data_size = input_size;
        sample.error_bound = cfg.error_bound;
        sample.action = nn_action;
        sample.actual_ratio = actual_ratio;
        sample.actual_comp_time_ms = 0.0;  // Not timed in this path
        experience_buffer_append(&sample);

        // Check prediction error
        double pred_ratio_d = static_cast<double>(predicted_ratio);
        double error_pct = (actual_ratio > 0.0) ?
            std::abs(pred_ratio_d - actual_ratio) / actual_ratio : 0.0;

        if (error_pct > g_exploration_threshold || is_ood) {
            // Level 2: Explore alternatives
            // Determine K (number of alternatives to try)
            int K;
            if (is_ood) {
                K = 31;  // Try all 32 configs (minus original)
            } else if (error_pct > 0.50) {
                K = 9;
            } else {
                K = 4;
            }

            // Collect explored (action, ratio, comp_time) for reinforcement
            struct ExploredResult { int action; double ratio; double comp_time_ms; };
            std::vector<ExploredResult> explored_samples;
            explored_samples.push_back({nn_action, actual_ratio,
                                        static_cast<double>(primary_comp_time_ms)});

            // Try top-K alternative configs
            double best_ratio = actual_ratio;
            (void)best_ratio;  // Updated in loop, used for tracking

            for (int i = 1; i <= K && i < 32; i++) {
                int alt_action = top_actions[i];
                if (alt_action == nn_action) continue;

                gpucompress::QTableAction alt = gpucompress::decodeAction(alt_action);
                gpucompress_algorithm_t alt_algo =
                    static_cast<gpucompress_algorithm_t>(alt.algorithm + 1);
                unsigned int alt_preproc = 0;
                if (alt.shuffle_size > 0)
                    alt_preproc |= (alt.shuffle_size == 4) ?
                        GPUCOMPRESS_PREPROC_SHUFFLE_4 : GPUCOMPRESS_PREPROC_SHUFFLE_2;
                if (alt.use_quantization)
                    alt_preproc |= GPUCOMPRESS_PREPROC_QUANTIZE;

                // Compress with alternative config
                uint8_t* d_alt_input = d_input;
                size_t alt_compress_size = input_size;
                uint8_t* d_alt_quant = nullptr;
                uint8_t* d_alt_shuf = nullptr;
                QuantizationResult alt_quant_result;

                // Apply quantization
                if ((alt_preproc & GPUCOMPRESS_PREPROC_QUANTIZE) && cfg.error_bound > 0.0) {
                    size_t num_el = input_size / sizeof(float);
                    if (num_el > 0) {
                        QuantizationConfig qcfg(
                            QuantizationType::LINEAR, cfg.error_bound,
                            num_el, sizeof(float));
                        alt_quant_result = quantize_simple(
                            d_alt_input, num_el, sizeof(float), qcfg, stream);
                        if (alt_quant_result.isValid()) {
                            d_alt_quant = static_cast<uint8_t*>(alt_quant_result.d_quantized);
                            d_alt_input = d_alt_quant;
                            alt_compress_size = alt_quant_result.quantized_bytes;
                        }
                    }
                }

                // Apply shuffle
                unsigned int alt_shuf_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(alt_preproc);
                if (alt_shuf_size > 0) {
                    uint8_t* shuf = byte_shuffle_simple(
                        d_alt_input, alt_compress_size, alt_shuf_size,
                        gpucompress::SHUFFLE_CHUNK_SIZE,
                        ShuffleKernelType::AUTO, stream);
                    if (shuf) {
                        d_alt_shuf = shuf;
                        d_alt_input = d_alt_shuf;
                    }
                }

                cudaStreamSynchronize(stream);

                // Compress
                CompressionAlgorithm alt_internal =
                    gpucompress::toInternalAlgorithm(alt_algo);
                auto alt_comp = createCompressionManager(
                    alt_internal, gpucompress::DEFAULT_CHUNK_SIZE,
                    stream, d_alt_input);

                size_t alt_comp_size = 0;

                if (alt_comp) {
                    try {
                        CompressionConfig alt_cc =
                            alt_comp->configure_compression(alt_compress_size);
                        size_t alt_max = alt_cc.max_compressed_buffer_size;
                        uint8_t* d_alt_out = nullptr;
                        if (cudaMalloc(&d_alt_out, alt_max) == cudaSuccess) {
                            cudaEvent_t at0, at1;
                            float alt_ct_ms = 0.0f;
                            bool at_ok = (cudaEventCreate(&at0) == cudaSuccess &&
                                          cudaEventCreate(&at1) == cudaSuccess);
                            if (at_ok) cudaEventRecord(at0, stream);

                            alt_comp->compress(d_alt_input, d_alt_out, alt_cc);

                            if (at_ok) {
                                cudaEventRecord(at1, stream);
                                cudaEventSynchronize(at1);
                                cudaEventElapsedTime(&alt_ct_ms, at0, at1);
                                cudaEventDestroy(at0);
                                cudaEventDestroy(at1);
                            }

                            alt_comp_size = alt_comp->get_compressed_output_size(d_alt_out);

                            double alt_ratio = static_cast<double>(input_size) /
                                               static_cast<double>(alt_comp_size);

                            explored_samples.push_back({alt_action, alt_ratio,
                                                        static_cast<double>(alt_ct_ms)});

                            // Store alternative experience
                            ExperienceSample alt_sample;
                            alt_sample.entropy = entropy;
                            alt_sample.mad = mad;
                            alt_sample.second_derivative = second_derivative;
                            alt_sample.data_size = input_size;
                            alt_sample.error_bound = cfg.error_bound;
                            alt_sample.action = alt_action;
                            alt_sample.actual_ratio = alt_ratio;
                            alt_sample.actual_comp_time_ms = 0.0;
                            experience_buffer_append(&alt_sample);

                            // Track if this is better than current best
                            if (alt_ratio > best_ratio) {
                                best_ratio = alt_ratio;

                                // Copy better result to output
                                size_t header_sz = GPUCOMPRESS_HEADER_SIZE;
                                size_t alt_total = header_sz + alt_comp_size;
                                if (alt_total <= *output_size) {
                                    // Build header for alternative
                                    CompressionHeader alt_hdr;
                                    alt_hdr.magic = COMPRESSION_MAGIC;
                                    alt_hdr.version = COMPRESSION_HEADER_VERSION;
                                    alt_hdr.shuffle_element_size = alt_shuf_size;
                                    alt_hdr.original_size = input_size;
                                    alt_hdr.compressed_size = alt_comp_size;

                                    if (d_alt_quant && alt_quant_result.isValid()) {
                                        alt_hdr.setQuantizationFlags(
                                            static_cast<uint32_t>(alt_quant_result.type),
                                            alt_quant_result.actual_precision, true);
                                        alt_hdr.quant_error_bound = alt_quant_result.error_bound;
                                        alt_hdr.quant_scale = alt_quant_result.scale_factor;
                                        alt_hdr.data_min = alt_quant_result.data_min;
                                        alt_hdr.data_max = alt_quant_result.data_max;
                                    } else {
                                        alt_hdr.quant_flags = 0;
                                        alt_hdr.quant_error_bound = 0.0;
                                        alt_hdr.quant_scale = 0.0;
                                        alt_hdr.data_min = 0.0;
                                        alt_hdr.data_max = 0.0;
                                    }

                                    // Write header + compressed data to host output
                                    memcpy(output, &alt_hdr, sizeof(CompressionHeader));
                                    cudaMemcpy(
                                        static_cast<uint8_t*>(output) + header_sz,
                                        d_alt_out, alt_comp_size,
                                        cudaMemcpyDeviceToHost);
                                    *output_size = alt_total;
                                    total_size = alt_total;
                                    compressed_size = alt_comp_size;
                                    algo_to_use = alt_algo;
                                    preproc_to_use = alt_preproc;
                                }
                            }
                            cudaFree(d_alt_out);
                        }
                    } catch (...) {
                        // Compression failed for this config, skip it
                    }
                }

                if (d_alt_quant) cudaFree(d_alt_quant);
                if (d_alt_shuf) cudaFree(d_alt_shuf);
            }

        }

        // Online reinforcement: if ratio OR comp_time MAPE exceeds threshold,
        // do CPU backward pass and update GPU weights.
        // This runs independently of exploration so comp_time can be reinforced
        // even when ratio predictions are already good.
        double ct_error_pct = 0.0;
        if (primary_comp_time_ms > 0.0f && predicted_comp_time > 0.0f) {
            double pred_ct = static_cast<double>(predicted_comp_time);
            double actual_ct = static_cast<double>(primary_comp_time_ms);
            ct_error_pct = std::abs(pred_ct - actual_ct) / actual_ct;
        }

        if (g_reinforce_enabled &&
            (error_pct > g_reinforce_threshold ||
             ct_error_pct > g_reinforce_ct_threshold)) {
            void* d_weights = gpucompress_nn_get_device_ptr_impl();
            if (d_weights) {
                // Lazy-init: copy GPU weights to host on first trigger
                if (!g_reinforce_initialized) {
                    if (nn_reinforce_init(d_weights) == 0) {
                        g_reinforce_initialized = true;
                    }
                }

                if (g_reinforce_initialized) {
                    // Build input for primary config (always available)
                    int algo_idx = nn_action % 8;
                    int quant_flag = (nn_action / 8) % 2;
                    int shuffle_flag = (nn_action / 16) % 2;

                    float input_raw[15];
                    for (int f = 0; f < 8; f++)
                        input_raw[f] = (f == algo_idx) ? 1.0f : 0.0f;
                    input_raw[8] = static_cast<float>(quant_flag);
                    input_raw[9] = static_cast<float>(shuffle_flag);

                    double eb_c = cfg.error_bound;
                    if (eb_c < 1e-7) eb_c = 1e-7;
                    input_raw[10] = static_cast<float>(log10(eb_c));

                    double ds = static_cast<double>(input_size);
                    if (ds < 1.0) ds = 1.0;
                    input_raw[11] = static_cast<float>(log2(ds));

                    input_raw[12] = static_cast<float>(entropy);
                    input_raw[13] = static_cast<float>(mad);
                    input_raw[14] = static_cast<float>(second_derivative);

                    nn_reinforce_add_sample(input_raw, actual_ratio,
                                            static_cast<double>(primary_comp_time_ms));

                    nn_reinforce_apply(d_weights, g_reinforce_lr);
                    sgd_fired = true;
                }
            }
        }
    }

    // Now free d_input
    cudaFree(d_input);

    // Fill stats if requested
    if (stats != nullptr) {
        stats->original_size = input_size;
        stats->compressed_size = total_size;
        stats->compression_ratio = static_cast<double>(input_size) / compressed_size;
        stats->entropy_bits = entropy;
        stats->mad = mad;
        stats->second_derivative = second_derivative;
        stats->algorithm_used = algo_to_use;
        stats->preprocessing_used = preproc_to_use;
        stats->throughput_mbps = 0.0;  // Would need timing to calculate
        stats->predicted_ratio = static_cast<double>(predicted_ratio);
        stats->predicted_comp_time_ms = static_cast<double>(predicted_comp_time);
        stats->sgd_fired = sgd_fired ? 1 : 0;
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
 * Statistical Analysis
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_compute_stats(
    const void* data,
    size_t size,
    double* entropy,
    double* mad,
    double* second_derivative
) {
    if (!g_initialized.load()) {
        return GPUCOMPRESS_ERROR_NOT_INITIALIZED;
    }

    if (data == nullptr || entropy == nullptr || mad == nullptr ||
        second_derivative == nullptr || size == 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    if (size % sizeof(float) != 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    cudaStream_t stream = g_default_stream;
    cudaError_t cuda_err;

    // Allocate GPU buffer and copy data
    uint8_t* d_data = nullptr;
    cuda_err = cudaMalloc(&d_data, size);
    if (cuda_err != cudaSuccess) {
        return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;
    }

    cuda_err = cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_data);
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

    // Run stats-only pipeline on GPU
    int rc = gpucompress::runStatsOnlyPipeline(
        d_data, size, stream, entropy, mad, second_derivative);

    cudaFree(d_data);

    if (rc != 0) {
        return GPUCOMPRESS_ERROR_CUDA_FAILED;
    }

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
    double second_derivative,
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
    int state = gpucompress::encodeState(entropy, error_level, mad, second_derivative);
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
 * Neural Network Management
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_load_nn(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    int result = gpucompress_nn_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_INVALID_INPUT;
}

extern "C" int gpucompress_nn_is_loaded(void) {
    return gpucompress_nn_is_loaded_impl();
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
 * Active Learning API
 * ============================================================ */

extern "C" gpucompress_error_t gpucompress_enable_active_learning(
    const char* experience_path
) {
    if (experience_path == nullptr || experience_path[0] == '\0') {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Initialize experience buffer
    if (experience_buffer_init(experience_path) != 0) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    strncpy(g_experience_path, experience_path, sizeof(g_experience_path) - 1);
    g_experience_path[sizeof(g_experience_path) - 1] = '\0';
    g_active_learning_enabled = true;

    return GPUCOMPRESS_SUCCESS;
}

extern "C" void gpucompress_disable_active_learning(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    g_active_learning_enabled = false;
    experience_buffer_cleanup();
    g_experience_path[0] = '\0';
}

extern "C" int gpucompress_active_learning_enabled(void) {
    return g_active_learning_enabled ? 1 : 0;
}

extern "C" void gpucompress_set_exploration_threshold(double threshold) {
    if (threshold > 0.0 && threshold < 1.0) {
        g_exploration_threshold = threshold;
    }
}

extern "C" void gpucompress_set_reinforcement(int enable, float learning_rate,
                                               float mape_threshold,
                                               float ct_mape_threshold) {
    g_reinforce_enabled = (enable != 0);
    if (learning_rate > 0.0f) g_reinforce_lr = learning_rate;
    if (mape_threshold > 0.0f) g_reinforce_threshold = static_cast<double>(mape_threshold);
    if (ct_mape_threshold > 0.0f) g_reinforce_ct_threshold = static_cast<double>(ct_mape_threshold);

    if (!g_reinforce_enabled) {
        nn_reinforce_cleanup();
        g_reinforce_initialized = false;
    }
}

extern "C" void gpucompress_reinforce_last_stats(float* grad_norm,
                                                    int* num_samples,
                                                    int* was_clipped) {
    nn_reinforce_get_last_stats(grad_norm, num_samples, was_clipped);
}

extern "C" size_t gpucompress_experience_count(void) {
    return experience_buffer_count();
}

extern "C" gpucompress_error_t gpucompress_reload_nn(const char* filepath) {
    if (filepath == nullptr) {
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Clean up reinforcement state (weights are about to change)
    if (g_reinforce_initialized) {
        nn_reinforce_cleanup();
        g_reinforce_initialized = false;
    }

    // Clean up old weights
    gpucompress_nn_cleanup_impl();

    // Load new weights
    int result = gpucompress_nn_load_impl(filepath);
    return (result == 0) ? GPUCOMPRESS_SUCCESS : GPUCOMPRESS_ERROR_INVALID_INPUT;
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
