/**
 * @file experience_buffer.cpp
 * @brief Thread-safe CSV experience buffer implementation
 *
 * Appends experience samples from compression calls as CSV rows.
 * Column format matches benchmark_results.csv so the retrain script
 * can concatenate directly.
 *
 * CSV columns:
 *   entropy, mad, second_derivative, original_size, error_bound,
 *   algorithm, quantization, shuffle, compression_ratio, compression_time_ms,
 *   decompression_time_ms, psnr_db
 */

#include "experience_buffer.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <atomic>

namespace {

/** Algorithm names matching the library's indexing (0-7) */
static const char* ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static FILE* g_file = nullptr;
static std::mutex g_mutex;
static std::atomic<size_t> g_count{0};

static const char* CSV_HEADER =
    "entropy,mad,second_derivative,original_size,error_bound,"
    "algorithm,quantization,shuffle,compression_ratio,compression_time_ms,"
    "decompression_time_ms,psnr_db\n";

} // anonymous namespace

extern "C" int experience_buffer_init(const char* csv_path) {
    if (csv_path == nullptr) return -1;

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_file != nullptr) {
        fclose(g_file);
        g_file = nullptr;
    }

    // Check if file exists and is non-empty
    bool write_header = true;
    FILE* check = fopen(csv_path, "r");
    if (check != nullptr) {
        fseek(check, 0, SEEK_END);
        if (ftell(check) > 0) {
            write_header = false;
        }
        fclose(check);
    }

    g_file = fopen(csv_path, "a");
    if (g_file == nullptr) {
        return -1;
    }

    if (write_header) {
        fputs(CSV_HEADER, g_file);
        fflush(g_file);
    }

    g_count.store(0);
    return 0;
}

extern "C" int experience_buffer_append(const ExperienceSample* sample) {
    if (sample == nullptr) return -1;

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_file == nullptr) return -1;

    // Decode action: algo + quant*8 + shuffle*16
    int algo_idx = sample->action % 8;
    int quant    = (sample->action / 8) % 2;
    int shuffle  = (sample->action / 16) % 2;

    const char* algo_name = (algo_idx >= 0 && algo_idx < 8) ?
                            ALGO_NAMES[algo_idx] : "lz4";
    const char* quant_str = quant ? "linear" : "none";
    int shuffle_val = shuffle ? 4 : 0;

    fprintf(g_file,
            "%.6f,%.6f,%.6f,%zu,%.10g,%s,%s,%d,%.6f,%.6f,%.6f,%.6f\n",
            sample->entropy,
            sample->mad,
            sample->second_derivative,
            sample->data_size,
            sample->error_bound,
            algo_name,
            quant_str,
            shuffle_val,
            sample->actual_ratio,
            sample->actual_comp_time_ms,
            sample->actual_decomp_time_ms,
            sample->actual_psnr);

    fflush(g_file);
    g_count.fetch_add(1);
    return 0;
}

extern "C" size_t experience_buffer_count(void) {
    return g_count.load();
}

extern "C" void experience_buffer_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_file != nullptr) {
        fclose(g_file);
        g_file = nullptr;
    }
    g_count.store(0);
}
