/**
 * GPUCompress C++ Benchmark
 *
 * Systematically evaluates all compression configurations across synthetic
 * datasets. Links directly against libgpucompress.so via the C API.
 *
 * Benchmark combination space (64 configs per file):
 *   8 algorithms x 2 shuffle options x 4 quantization options = 64
 *
 * Usage:
 *   ./build/benchmark \
 *     --data-dir syntheticGeneration/datasets/ \
 *     --output benchmark_results.csv \
 *     --workers 4 \
 *     --max-files 1563
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include "gpucompress.h"

// ============================================================
// Constants
// ============================================================

static const char* ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};
static const int ALGO_IDS[] = {1, 2, 3, 4, 5, 6, 7, 8};
static const int NUM_ALGOS = 8;

struct ShuffleOpt {
    int shuffle_size;
    unsigned int flag;
};
static const ShuffleOpt SHUFFLE_OPTS[] = {{0, 0x00}, {4, 0x02}};
static const int NUM_SHUFFLE = 2;

struct QuantOpt {
    bool enabled;
    double error_bound;
};
static const QuantOpt QUANT_OPTS[] = {
    {false, 0.0}, {true, 0.1}, {true, 0.01}, {true, 0.001}
};
static const int NUM_QUANT = 4;

static const unsigned int PREPROC_QUANTIZE = 0x10;

static const char* CSV_HEADER =
    "file,dtype,palette,perturbation,bin_width,fill_mode,"
    "algorithm,shuffle,quantization,error_bound,"
    "original_size,compressed_size,compression_ratio,"
    "compression_time_ms,decompression_time_ms,"
    "compression_throughput_mbps,decompression_throughput_mbps,"
    "psnr_db,max_error,rmse,"
    "entropy,mad,second_derivative,"
    "lib_throughput_mbps,exact_match,success\n";

// ============================================================
// Structures
// ============================================================

struct BenchConfig {
    std::string algo_name;
    int algo_id;
    int shuffle_size;
    unsigned int preprocessing;
    std::string quant_label;
    double error_bound;
};

struct FileMetadata {
    std::string dtype;
    std::string palette;
    double bin_width;
    double perturbation;
    std::string fill_mode;
    std::string size_label;  // e.g. "16kb", "1mb" (empty if single-size batch)
};

struct QualityMetrics {
    double psnr;
    double max_error;
    double rmse;
};

struct BenchmarkRow {
    std::string file;
    std::string dtype;
    std::string palette;
    double perturbation;
    double bin_width;
    std::string fill_mode;
    std::string algorithm;
    int shuffle;
    std::string quantization;
    double error_bound;
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double compression_time_ms;
    double decompression_time_ms;
    double compression_throughput_mbps;
    double decompression_throughput_mbps;
    std::string psnr_db;
    std::string max_error_str;
    std::string rmse_str;
    double entropy;
    double mad;
    double second_derivative;
    double lib_throughput_mbps;
    std::string exact_match;
    bool success;
};

struct CLIConfig {
    std::string data_dir;
    std::vector<std::string> files;
    std::string output = "benchmark_results.csv";
    int workers = 4;
    int max_files = 0;
};

// ============================================================
// Config generation (64 configs)
// ============================================================

static std::vector<BenchConfig> generate_configs() {
    std::vector<BenchConfig> configs;
    configs.reserve(NUM_ALGOS * NUM_SHUFFLE * NUM_QUANT);
    for (int a = 0; a < NUM_ALGOS; ++a) {
        for (int s = 0; s < NUM_SHUFFLE; ++s) {
            for (int q = 0; q < NUM_QUANT; ++q) {
                BenchConfig cfg;
                cfg.algo_name = ALGO_NAMES[a];
                cfg.algo_id = ALGO_IDS[a];
                cfg.shuffle_size = SHUFFLE_OPTS[s].shuffle_size;
                cfg.preprocessing = SHUFFLE_OPTS[s].flag;
                if (QUANT_OPTS[q].enabled) {
                    cfg.preprocessing |= PREPROC_QUANTIZE;
                    cfg.error_bound = QUANT_OPTS[q].error_bound;
                    cfg.quant_label = "linear";
                } else {
                    cfg.error_bound = 0.0;
                    cfg.quant_label = "none";
                }
                configs.push_back(cfg);
            }
        }
    }
    return configs;
}

// ============================================================
// Filename metadata parsing
// ============================================================

static FileMetadata parse_filename(const std::string& filename) {
    FileMetadata meta;
    meta.dtype = "unknown";
    meta.palette = "unknown";
    meta.bin_width = 0.0;
    meta.perturbation = 0.0;
    meta.fill_mode = "unknown";

    // Strip directory and extension
    std::string stem = filename;
    size_t slash = stem.rfind('/');
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    size_t dot = stem.rfind('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);

    // Split by '_'
    std::vector<std::string> parts;
    std::istringstream iss(stem);
    std::string token;
    while (std::getline(iss, token, '_')) {
        parts.push_back(token);
    }
    if (parts.empty()) return meta;

    meta.dtype = parts[0];

    // Helper: check if string (after first char) is all digits
    auto all_digits_after = [](const std::string& s, size_t start) -> bool {
        if (s.size() <= start) return false;
        for (size_t j = start; j < s.size(); ++j) {
            if (!isdigit(s[j])) return false;
        }
        return true;
    };

    // Helper: check if string (after first char) is a valid number (digits and dots)
    auto is_number_after = [](const std::string& s, size_t start) -> bool {
        if (s.size() <= start) return false;
        bool has_digit = false;
        for (size_t j = start; j < s.size(); ++j) {
            if (isdigit(s[j])) has_digit = true;
            else if (s[j] != '.') return false;
        }
        return has_digit;
    };

    // Format 1 (generator.py batch): {dtype}_{palette}_w{bw*100}_p{pert*1000}_{fill}.bin
    // Look for w{digits} marker first
    int w_idx = -1, p_idx = -1;
    for (int i = 1; i < (int)parts.size(); ++i) {
        if (w_idx < 0 && parts[i].size() > 1 && parts[i][0] == 'w' &&
            all_digits_after(parts[i], 1)) {
            w_idx = i;
        }
        if (w_idx >= 0 && p_idx < 0 && parts[i].size() > 1 && parts[i][0] == 'p' &&
            all_digits_after(parts[i], 1)) {
            p_idx = i;
        }
    }

    if (w_idx >= 0 && p_idx >= 0) {
        // Full batch format: palette = parts[1..w_idx), bw/100, pert/1000, fill after p
        std::string palette;
        for (int i = 1; i < w_idx; ++i) {
            if (!palette.empty()) palette += "_";
            palette += parts[i];
        }
        meta.palette = palette;
        meta.bin_width = std::stoi(parts[w_idx].substr(1)) / 100.0;
        meta.perturbation = std::stoi(parts[p_idx].substr(1)) / 1000.0;

        // Check if last part is a size suffix (e.g. "16kb", "256kb", "1mb", "4mb")
        int end_idx = (int)parts.size();
        if (end_idx > p_idx + 1) {
            const std::string& last = parts[end_idx - 1];
            bool is_size_label = false;
            if (last.size() >= 3) {
                std::string suffix = last.substr(last.size() - 2);
                if (suffix == "kb" || suffix == "mb" || suffix == "gb") {
                    // Check that everything before the suffix is digits
                    bool all_digits = true;
                    for (size_t ci = 0; ci < last.size() - 2; ++ci) {
                        if (!isdigit(last[ci])) { all_digits = false; break; }
                    }
                    is_size_label = all_digits;
                }
            }
            if (is_size_label) {
                meta.size_label = last;
                end_idx--;
            }
        }

        std::string fill_mode;
        for (int i = p_idx + 1; i < end_idx; ++i) {
            if (!fill_mode.empty()) fill_mode += "_";
            fill_mode += parts[i];
        }
        if (!fill_mode.empty()) meta.fill_mode = fill_mode;

        return meta;
    }

    // Format 2 (training_data): {dtype}_{palette}_p{float}.bin
    // No w marker; look for p{number} where number may contain dots
    int p_float_idx = -1;
    for (int i = 1; i < (int)parts.size(); ++i) {
        if (parts[i].size() > 1 && parts[i][0] == 'p' &&
            is_number_after(parts[i], 1)) {
            p_float_idx = i;
            break;
        }
    }

    if (p_float_idx >= 0) {
        std::string palette;
        for (int i = 1; i < p_float_idx; ++i) {
            if (!palette.empty()) palette += "_";
            palette += parts[i];
        }
        meta.palette = palette;
        meta.perturbation = std::stod(parts[p_float_idx].substr(1));

        std::string fill_mode;
        for (int i = p_float_idx + 1; i < (int)parts.size(); ++i) {
            if (!fill_mode.empty()) fill_mode += "_";
            fill_mode += parts[i];
        }
        if (!fill_mode.empty()) meta.fill_mode = fill_mode;
    }

    return meta;
}

// ============================================================
// Quality metrics (PSNR, max_error, RMSE)
// ============================================================

static QualityMetrics compute_quality(const float* orig, const float* decomp, size_t n) {
    QualityMetrics qm;
    qm.psnr = std::numeric_limits<double>::infinity();
    qm.max_error = 0.0;
    qm.rmse = 0.0;

    if (n == 0) return qm;

    double mse = 0.0;
    double max_err = 0.0;
    double vmin = static_cast<double>(orig[0]);
    double vmax = static_cast<double>(orig[0]);

    for (size_t i = 0; i < n; ++i) {
        double o = static_cast<double>(orig[i]);
        double d = static_cast<double>(decomp[i]);
        double diff = o - d;
        mse += diff * diff;
        double adiff = std::abs(diff);
        if (adiff > max_err) max_err = adiff;
        if (o < vmin) vmin = o;
        if (o > vmax) vmax = o;
    }
    mse /= static_cast<double>(n);

    qm.max_error = max_err;
    qm.rmse = std::sqrt(mse);

    if (mse == 0.0) {
        qm.psnr = std::numeric_limits<double>::infinity();
        return qm;
    }

    double data_range = vmax - vmin;
    if (data_range > 0.0) {
        qm.psnr = 10.0 * std::log10((data_range * data_range) / mse);
    } else {
        qm.psnr = std::numeric_limits<double>::infinity();
    }
    return qm;
}

// ============================================================
// File discovery
// ============================================================

static std::vector<std::string> discover_files(const std::string& dir, int max_files) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::cerr << "Error: cannot open directory " << dir << std::endl;
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            std::string path = dir;
            if (!path.empty() && path.back() != '/') path += '/';
            path += name;
            files.push_back(path);
        }
    }
    closedir(d);

    std::sort(files.begin(), files.end());
    if (max_files > 0 && (int)files.size() > max_files) {
        files.resize(max_files);
    }
    return files;
}

// ============================================================
// Helper: format double for CSV
// ============================================================

static std::string fmt_double(double v, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

// ============================================================
// CSV row formatting
// ============================================================

static std::string format_row(const BenchmarkRow& r) {
    std::ostringstream oss;
    oss << r.file << ","
        << r.dtype << ","
        << r.palette << ","
        << r.perturbation << ","
        << r.bin_width << ","
        << r.fill_mode << ","
        << r.algorithm << ","
        << r.shuffle << ","
        << r.quantization << ","
        << r.error_bound << ","
        << r.original_size << ","
        << r.compressed_size << ","
        << fmt_double(r.compression_ratio, 6) << ","
        << fmt_double(r.compression_time_ms, 4) << ","
        << fmt_double(r.decompression_time_ms, 4) << ","
        << fmt_double(r.compression_throughput_mbps, 2) << ","
        << fmt_double(r.decompression_throughput_mbps, 2) << ","
        << r.psnr_db << ","
        << r.max_error_str << ","
        << r.rmse_str << ","
        << fmt_double(r.entropy, 6) << ","
        << fmt_double(r.mad, 6) << ","
        << fmt_double(r.second_derivative, 6) << ","
        << fmt_double(r.lib_throughput_mbps, 2) << ","
        << r.exact_match << ","
        << (r.success ? "True" : "False") << "\n";
    return oss.str();
}

// ============================================================
// Benchmark a single file across all configs
// ============================================================

static std::vector<BenchmarkRow> benchmark_file(
    const std::string& filepath,
    const std::vector<BenchConfig>& configs)
{
    std::vector<BenchmarkRow> rows;

    // Extract just the filename
    std::string filename = filepath;
    size_t slash = filepath.rfind('/');
    if (slash != std::string::npos) filename = filepath.substr(slash + 1);

    FileMetadata meta = parse_filename(filepath);

    // Load file
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "Warning: cannot open " << filepath << std::endl;
        return rows;
    }
    size_t file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);

    std::vector<char> raw(file_size);
    ifs.read(raw.data(), file_size);
    ifs.close();

    const float* data = reinterpret_cast<const float*>(raw.data());
    size_t num_elements = file_size / sizeof(float);
    size_t input_size = file_size;

    // Compute stats once (GPU-accelerated via public API)
    double entropy, mad, second_deriv;
    gpucompress_error_t stats_err = gpucompress_compute_stats(
        data, num_elements * sizeof(float),
        &entropy, &mad, &second_deriv);
    if (stats_err != GPUCOMPRESS_SUCCESS) {
        std::cerr << "Warning: gpucompress_compute_stats failed for "
                  << filepath << std::endl;
        return rows;
    }

    // Allocate buffers
    size_t max_out = gpucompress_max_compressed_size(input_size);
    std::vector<char> comp_buf(max_out);
    std::vector<char> decomp_buf(input_size);

    for (const auto& cfg : configs) {
        BenchmarkRow row;
        row.file = filename;
        row.dtype = meta.dtype;
        row.palette = meta.palette;
        row.perturbation = meta.perturbation;
        row.bin_width = meta.bin_width;
        row.fill_mode = meta.fill_mode;
        row.algorithm = cfg.algo_name;
        row.shuffle = cfg.shuffle_size;
        row.quantization = cfg.quant_label;
        row.error_bound = cfg.error_bound;
        row.original_size = input_size;
        row.entropy = entropy;
        row.mad = mad;
        row.second_derivative = second_deriv;

        // Setup config
        gpucompress_config_t c_cfg = gpucompress_default_config();
        c_cfg.algorithm = static_cast<gpucompress_algorithm_t>(cfg.algo_id);
        c_cfg.preprocessing = cfg.preprocessing;
        c_cfg.error_bound = cfg.error_bound;
        c_cfg.cuda_device = -1;
        c_cfg.cuda_stream = nullptr;

        size_t out_size = max_out;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        // Compress
        auto t0 = std::chrono::high_resolution_clock::now();
        int rc = gpucompress_compress(
            raw.data(), input_size,
            comp_buf.data(), &out_size,
            &c_cfg, &stats);
        auto t1 = std::chrono::high_resolution_clock::now();
        double comp_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0) {
            row.compressed_size = 0;
            row.compression_ratio = 0.0;
            row.compression_time_ms = comp_time;
            row.decompression_time_ms = 0.0;
            row.compression_throughput_mbps = 0.0;
            row.decompression_throughput_mbps = 0.0;
            row.psnr_db = "";
            row.max_error_str = "";
            row.rmse_str = "";
            row.lib_throughput_mbps = 0.0;
            row.exact_match = "";
            row.success = false;
            rows.push_back(row);
            continue;
        }

        size_t compressed_size = out_size;
        double comp_ratio = (compressed_size > 0)
            ? static_cast<double>(input_size) / static_cast<double>(compressed_size)
            : 0.0;
        double comp_tp = (comp_time > 0.0)
            ? (static_cast<double>(input_size) / (1024.0 * 1024.0)) / (comp_time / 1000.0)
            : 0.0;

        // Decompress
        size_t decomp_size = input_size;
        auto t2 = std::chrono::high_resolution_clock::now();
        int rc_d = gpucompress_decompress(
            comp_buf.data(), compressed_size,
            decomp_buf.data(), &decomp_size);
        auto t3 = std::chrono::high_resolution_clock::now();
        double decomp_time = std::chrono::duration<double, std::milli>(t3 - t2).count();

        double decomp_tp = (decomp_time > 0.0)
            ? (static_cast<double>(input_size) / (1024.0 * 1024.0)) / (decomp_time / 1000.0)
            : 0.0;

        // Quality metrics and data integrity
        if (rc_d == 0 && cfg.quant_label == "linear") {
            // Lossy: compute quality metrics
            const float* decomp_data = reinterpret_cast<const float*>(decomp_buf.data());
            size_t decomp_elements = decomp_size / sizeof(float);
            if (decomp_elements == num_elements) {
                QualityMetrics qm = compute_quality(data, decomp_data, num_elements);
                if (std::isinf(qm.psnr)) {
                    row.psnr_db = "inf";
                } else {
                    row.psnr_db = fmt_double(qm.psnr, 6);
                }
                row.max_error_str = fmt_double(qm.max_error, 6);
                row.rmse_str = fmt_double(qm.rmse, 6);
                // Lossy compression: check if result happens to be exact
                row.exact_match = (decomp_size == input_size &&
                    memcmp(raw.data(), decomp_buf.data(), input_size) == 0)
                    ? "True" : "False";
            } else {
                row.psnr_db = "";
                row.max_error_str = "";
                row.rmse_str = "";
                row.exact_match = "False";
            }
        } else if (rc_d == 0) {
            // Lossless: verify data integrity via memcmp
            bool match = (decomp_size == input_size &&
                memcmp(raw.data(), decomp_buf.data(), input_size) == 0);
            row.exact_match = match ? "True" : "False";
            if (match) {
                row.psnr_db = "inf";
                row.max_error_str = fmt_double(0.0, 1);
                row.rmse_str = fmt_double(0.0, 1);
            } else {
                // Decompressed data does not match original
                row.psnr_db = "";
                row.max_error_str = "";
                row.rmse_str = "";
                row.success = false;
                rows.push_back(row);
                continue;
            }
        } else {
            // Decompression failed
            row.psnr_db = "";
            row.max_error_str = "";
            row.rmse_str = "";
            row.exact_match = "";
        }

        row.compressed_size = compressed_size;
        row.compression_ratio = comp_ratio;
        row.compression_time_ms = comp_time;
        row.decompression_time_ms = decomp_time;
        row.compression_throughput_mbps = comp_tp;
        row.decompression_throughput_mbps = decomp_tp;
        row.lib_throughput_mbps = stats.throughput_mbps;
        row.success = true;
        rows.push_back(row);
    }

    return rows;
}

// ============================================================
// CLI parsing
// ============================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --data-dir <path>   Directory containing .bin files\n"
              << "  -f, --file <path>       Single .bin file (repeatable)\n"
              << "  -o, --output <path>     CSV output path (default: benchmark_results.csv)\n"
              << "  -w, --workers <n>       Thread count (default: 4)\n"
              << "  -m, --max-files <n>     Cap files processed (0 = all, default: 0)\n"
              << "  -h, --help              Show this help\n"
              << "\nEither --data-dir or --file is required. Both can be combined.\n";
}

static CLIConfig parse_args(int argc, char** argv) {
    CLIConfig config;
    static struct option long_options[] = {
        {"data-dir",  required_argument, 0, 'd'},
        {"file",      required_argument, 0, 'f'},
        {"output",    required_argument, 0, 'o'},
        {"workers",   required_argument, 0, 'w'},
        {"max-files", required_argument, 0, 'm'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "d:f:o:w:m:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd': config.data_dir = optarg; break;
            case 'f': config.files.push_back(optarg); break;
            case 'o': config.output = optarg; break;
            case 'w': config.workers = std::atoi(optarg); break;
            case 'm': config.max_files = std::atoi(optarg); break;
            case 'h': print_usage(argv[0]); exit(0);
            default: break;
        }
    }

    if (config.data_dir.empty() && config.files.empty()) {
        std::cerr << "Error: --data-dir or --file is required\n";
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    CLIConfig config = parse_args(argc, argv);

    // Discover files from --data-dir and/or --file
    std::vector<std::string> files;
    if (!config.data_dir.empty()) {
        files = discover_files(config.data_dir, config.max_files);
    }
    for (const auto& f : config.files) {
        struct stat st;
        if (stat(f.c_str(), &st) != 0) {
            std::cerr << "Error: cannot stat " << f << std::endl;
            return 1;
        }
        if (S_ISDIR(st.st_mode)) {
            std::cerr << "Error: " << f << " is a directory, use --data-dir instead" << std::endl;
            return 1;
        }
        files.push_back(f);
    }
    if (files.empty()) {
        std::cerr << "Error: no .bin files found" << std::endl;
        return 1;
    }

    std::vector<BenchConfig> configs = generate_configs();
    size_t total_rows = files.size() * configs.size();

    std::cout << "GPUCompress Benchmark" << std::endl;
    std::cout << "  Files:   " << files.size() << std::endl;
    std::cout << "  Configs: " << configs.size() << " per file" << std::endl;
    std::cout << "  Total:   " << total_rows << " rows" << std::endl;
    std::cout << "  Workers: " << config.workers << std::endl;
    std::cout << "  Output:  " << config.output << std::endl;
    std::cout << std::endl;

    // Initialize library once
    int rc = gpucompress_init(nullptr);
    if (rc != 0) {
        std::cerr << "Error: gpucompress_init failed with code " << rc << std::endl;
        return 1;
    }

    // Open output CSV
    std::ofstream csv_out(config.output);
    if (!csv_out) {
        std::cerr << "Error: cannot open " << config.output << " for writing" << std::endl;
        gpucompress_cleanup();
        return 1;
    }
    csv_out << CSV_HEADER;

    // Work queue
    std::queue<size_t> work_queue;
    std::mutex queue_mutex;
    std::mutex csv_mutex;
    std::atomic<size_t> files_done{0};

    for (size_t i = 0; i < files.size(); ++i) {
        work_queue.push(i);
    }

    auto start_time = std::chrono::steady_clock::now();

    // Worker function
    auto worker = [&]() {
        while (true) {
            size_t file_idx;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (work_queue.empty()) break;
                file_idx = work_queue.front();
                work_queue.pop();
            }

            std::vector<BenchmarkRow> rows = benchmark_file(files[file_idx], configs);

            // Write rows atomically
            {
                std::lock_guard<std::mutex> lock(csv_mutex);
                for (const auto& row : rows) {
                    csv_out << format_row(row);
                }
            }

            size_t done = ++files_done;
            if (done % 10 == 0 || done == files.size()) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double rate = (elapsed > 0) ? done / elapsed : 0;
                double eta = (rate > 0) ? (files.size() - done) / rate : 0;
                std::cerr << "\r  [" << done << "/" << files.size() << "] "
                          << std::fixed << std::setprecision(1)
                          << (done * 100.0 / files.size()) << "% "
                          << "(" << std::setprecision(1) << rate << " files/s, ETA "
                          << std::setprecision(0) << eta << "s)" << std::flush;
            }
        }
    };

    // Launch threads
    int n_workers = std::min(config.workers, static_cast<int>(files.size()));
    std::vector<std::thread> threads;
    for (int t = 0; t < n_workers; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }

    csv_out.close();
    std::cerr << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Count successes
    // Re-read is wasteful; count during writing instead.
    // We'll just print totals.
    size_t total_written = files_done.load() * configs.size();

    std::cout << "\nBenchmark complete" << std::endl;
    std::cout << "  Time:      " << std::fixed << std::setprecision(1) << total_time << "s" << std::endl;
    std::cout << "  Rows:      " << total_written << std::endl;
    std::cout << "  Output:    " << config.output << std::endl;

    gpucompress_cleanup();
    return 0;
}
