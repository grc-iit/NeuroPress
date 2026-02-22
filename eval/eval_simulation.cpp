/**
 * GPUCompress Simulation Evaluation
 *
 * Processes .bin files sequentially through ALGO_AUTO with NN + active learning.
 * Reports per-file metrics and aggregate adaptation statistics.
 *
 * Single-threaded, ALGO_AUTO only — designed to observe the active learning
 * adaptation loop: initial poor predictions → experience collection → retrain →
 * improved predictions.
 *
 * Usage:
 *   ./build/eval_simulation \
 *     --data-dir eval/data/ \
 *     --weights model.nnwt \
 *     --experience exp.csv \
 *     --output results.csv \
 *     --error-bound 0.0 \
 *     --threshold 0.20
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include "gpucompress.h"

// ============================================================
// CSV header
// ============================================================

static const char* CSV_HEADER =
    "file,field,scenario,timestep,entropy,mad,second_derivative,"
    "original_size,algorithm,shuffle,quantization,error_bound,"
    "compressed_size,compression_ratio,compress_time_ms,throughput_mbps,"
    "experience_count,experience_delta,predicted_ratio,mape\n";

// ============================================================
// CLI configuration
// ============================================================

struct EvalConfig {
    std::string data_dir;
    std::string weights;
    std::string experience = "experience.csv";
    std::string output = "results.csv";
    double error_bound = 0.0;
    double threshold = 0.20;
    int max_files = 0;
    bool reinforce = false;
    float reinforce_lr = 1e-4f;
    float reinforce_threshold = 0.60f;
    bool verbose = false;
};

// ============================================================
// Parse NSM filename metadata
// ============================================================

struct NSMMetadata {
    std::string field;
    int scenario;
    int timestep;
};

static NSMMetadata parse_nsm_filename(const std::string& filepath) {
    NSMMetadata meta;
    meta.field = "unknown";
    meta.scenario = -1;
    meta.timestep = -1;

    // Extract basename
    std::string name = filepath;
    size_t slash = name.rfind('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);

    // Strip extension
    size_t dot = name.rfind('.');
    if (dot != std::string::npos) name = name.substr(0, dot);

    // Expected: float32_nsm_{field}_s{NN}_t{NNN}
    // Split by '_'
    std::vector<std::string> parts;
    std::istringstream iss(name);
    std::string token;
    while (std::getline(iss, token, '_')) {
        parts.push_back(token);
    }

    // Find sNN and tNNN parts from the end
    int s_idx = -1, t_idx = -1;
    for (int i = (int)parts.size() - 1; i >= 0; --i) {
        if (t_idx < 0 && parts[i].size() > 1 && parts[i][0] == 't') {
            bool all_digits = true;
            for (size_t j = 1; j < parts[i].size(); ++j) {
                if (!isdigit(parts[i][j])) { all_digits = false; break; }
            }
            if (all_digits) t_idx = i;
        }
        if (s_idx < 0 && parts[i].size() > 1 && parts[i][0] == 's') {
            bool all_digits = true;
            for (size_t j = 1; j < parts[i].size(); ++j) {
                if (!isdigit(parts[i][j])) { all_digits = false; break; }
            }
            if (all_digits) s_idx = i;
        }
    }

    if (s_idx >= 0) meta.scenario = std::stoi(parts[s_idx].substr(1));
    if (t_idx >= 0) meta.timestep = std::stoi(parts[t_idx].substr(1));

    // Field name: parts between "nsm" marker and sNN
    // e.g. float32_nsm_internal_energy_s00_t000 → field = "internal_energy"
    int nsm_idx = -1;
    for (int i = 0; i < (int)parts.size(); ++i) {
        if (parts[i] == "nsm") { nsm_idx = i; break; }
    }

    int field_end = (s_idx >= 0) ? s_idx : (int)parts.size();
    int field_start = (nsm_idx >= 0) ? nsm_idx + 1 : 1;

    if (field_start < field_end) {
        std::string field;
        for (int i = field_start; i < field_end; ++i) {
            if (!field.empty()) field += "_";
            field += parts[i];
        }
        meta.field = field;
    }

    return meta;
}

// ============================================================
// File discovery (from benchmark.cpp pattern)
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

    // Natural sort gives chronological order for NSM naming convention
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
// CLI parsing
// ============================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --data-dir <path>      Directory with .bin files (required)\n"
              << "  -w, --weights <path>       NN weights file (.nnwt) (required)\n"
              << "  -e, --experience <path>    Experience CSV output (default: experience.csv)\n"
              << "  -o, --output <path>        Results CSV output (default: results.csv)\n"
              << "  -b, --error-bound <float>  Quantization error bound (default: 0.0 = lossless)\n"
              << "  -t, --threshold <float>    Exploration threshold (default: 0.20)\n"
              << "  -m, --max-files <n>        Max files to process (0 = all, default: 0)\n"
              << "      --reinforce            Enable online reinforcement learning\n"
              << "      --reinforce-lr <f>     Reinforcement learning rate (default: 1e-4)\n"
              << "      --reinforce-threshold <f>  MAPE threshold for reinforcement (default: 0.60)\n"
              << "  -v, --verbose              Detailed per-file reinforcement trace\n"
              << "  -h, --help                 Show this help\n";
}

static EvalConfig parse_args(int argc, char** argv) {
    EvalConfig config;
    enum { OPT_REINFORCE = 256, OPT_REINFORCE_LR, OPT_REINFORCE_THRESH };
    static struct option long_options[] = {
        {"data-dir",              required_argument, 0, 'd'},
        {"weights",               required_argument, 0, 'w'},
        {"experience",            required_argument, 0, 'e'},
        {"output",                required_argument, 0, 'o'},
        {"error-bound",           required_argument, 0, 'b'},
        {"threshold",             required_argument, 0, 't'},
        {"max-files",             required_argument, 0, 'm'},
        {"reinforce",             no_argument,       0, OPT_REINFORCE},
        {"reinforce-lr",          required_argument, 0, OPT_REINFORCE_LR},
        {"reinforce-threshold",   required_argument, 0, OPT_REINFORCE_THRESH},
        {"verbose",               no_argument,       0, 'v'},
        {"help",                  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "d:w:e:o:b:t:m:vh",
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd': config.data_dir = optarg; break;
            case 'w': config.weights = optarg; break;
            case 'e': config.experience = optarg; break;
            case 'o': config.output = optarg; break;
            case 'b': config.error_bound = std::atof(optarg); break;
            case 't': config.threshold = std::atof(optarg); break;
            case 'm': config.max_files = std::atoi(optarg); break;
            case OPT_REINFORCE: config.reinforce = true; break;
            case OPT_REINFORCE_LR: config.reinforce_lr = std::atof(optarg); break;
            case OPT_REINFORCE_THRESH: config.reinforce_threshold = std::atof(optarg); break;
            case 'v': config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: break;
        }
    }

    if (config.data_dir.empty()) {
        std::cerr << "Error: --data-dir is required\n";
        print_usage(argv[0]);
        exit(1);
    }
    if (config.weights.empty()) {
        std::cerr << "Error: --weights is required\n";
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    EvalConfig config = parse_args(argc, argv);

    // Discover files
    std::vector<std::string> files = discover_files(config.data_dir, config.max_files);
    if (files.empty()) {
        std::cerr << "Error: no .bin files found in " << config.data_dir << std::endl;
        return 1;
    }

    std::cout << "GPUCompress Simulation Evaluation" << std::endl;
    std::cout << "  Files:       " << files.size() << std::endl;
    std::cout << "  Weights:     " << config.weights << std::endl;
    std::cout << "  Experience:  " << config.experience << std::endl;
    std::cout << "  Output:      " << config.output << std::endl;
    std::cout << "  Error bound: " << config.error_bound << std::endl;
    std::cout << "  Threshold:   " << config.threshold << std::endl;
    std::cout << std::endl;

    // 1. Initialize library with NN weights
    int rc = gpucompress_init(config.weights.c_str());
    if (rc != GPUCOMPRESS_SUCCESS) {
        std::cerr << "Error: gpucompress_init failed: "
                  << gpucompress_error_string((gpucompress_error_t)rc) << std::endl;
        return 1;
    }

    // Load NN if init didn't load it via the weights path
    if (!gpucompress_nn_is_loaded()) {
        rc = gpucompress_load_nn(config.weights.c_str());
        if (rc != GPUCOMPRESS_SUCCESS) {
            std::cerr << "Error: gpucompress_load_nn failed: "
                      << gpucompress_error_string((gpucompress_error_t)rc) << std::endl;
            gpucompress_cleanup();
            return 1;
        }
    }
    std::cout << "  NN loaded: " << (gpucompress_nn_is_loaded() ? "yes" : "no") << std::endl;

    // 2. Enable active learning
    rc = gpucompress_enable_active_learning(config.experience.c_str());
    if (rc != GPUCOMPRESS_SUCCESS) {
        std::cerr << "Error: gpucompress_enable_active_learning failed: "
                  << gpucompress_error_string((gpucompress_error_t)rc) << std::endl;
        gpucompress_cleanup();
        return 1;
    }
    gpucompress_set_exploration_threshold(config.threshold);
    std::cout << "  Active learning: enabled" << std::endl;

    // 2b. Enable reinforcement if requested
    if (config.reinforce) {
        gpucompress_set_reinforcement(1, config.reinforce_lr,
                                      config.reinforce_threshold);
        std::cout << "  Reinforcement:   enabled (lr="
                  << config.reinforce_lr
                  << ", threshold=" << config.reinforce_threshold << ")"
                  << std::endl;
    }
    std::cout << std::endl;

    // 3. Open output CSV
    std::ofstream csv_out(config.output);
    if (!csv_out) {
        std::cerr << "Error: cannot open " << config.output << " for writing" << std::endl;
        gpucompress_cleanup();
        return 1;
    }
    csv_out << CSV_HEADER;

    // 4. Warm-up: load first file and do a throwaway compress
    {
        std::ifstream ifs(files[0], std::ios::binary | std::ios::ate);
        if (ifs) {
            size_t sz = static_cast<size_t>(ifs.tellg());
            ifs.seekg(0);
            std::vector<char> buf(sz);
            ifs.read(buf.data(), sz);
            ifs.close();

            size_t max_out = gpucompress_max_compressed_size(sz);
            std::vector<char> comp(max_out);
            size_t out_sz = max_out;

            gpucompress_config_t warmup_cfg = gpucompress_default_config();
            warmup_cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
            gpucompress_stats_t warmup_stats;
            memset(&warmup_stats, 0, sizeof(warmup_stats));

            gpucompress_compress(buf.data(), sz, comp.data(), &out_sz,
                                 &warmup_cfg, &warmup_stats);
            std::cout << "  Warm-up complete" << std::endl << std::endl;
        }
    }

    // Reset experience count baseline after warm-up
    size_t prev_experience = gpucompress_experience_count();

    // 5. Process each file
    size_t total_explorations = 0;
    double total_ratio = 0.0;
    size_t files_processed = 0;

    // MAPE tracking
    double total_mape = 0.0;

    // Rolling MAPE (circular buffer of last 20)
    const size_t ROLLING_WINDOW = 20;
    std::vector<double> rolling_mape(ROLLING_WINDOW, 0.0);
    size_t rolling_idx = 0;
    size_t rolling_filled = 0;

    auto wall_start = std::chrono::steady_clock::now();

    for (size_t fi = 0; fi < files.size(); ++fi) {
        const std::string& filepath = files[fi];

        // Extract filename
        std::string filename = filepath;
        size_t slash = filepath.rfind('/');
        if (slash != std::string::npos) filename = filepath.substr(slash + 1);

        NSMMetadata meta = parse_nsm_filename(filepath);

        // Load file
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        if (!ifs) {
            std::cerr << "Warning: cannot open " << filepath << std::endl;
            continue;
        }
        size_t file_size = static_cast<size_t>(ifs.tellg());
        ifs.seekg(0);
        std::vector<char> raw(file_size);
        ifs.read(raw.data(), file_size);
        ifs.close();

        // Compute stats
        double entropy, mad, second_deriv;
        gpucompress_error_t stats_err = gpucompress_compute_stats(
            raw.data(), file_size, &entropy, &mad, &second_deriv);
        if (stats_err != GPUCOMPRESS_SUCCESS) {
            std::cerr << "Warning: stats failed for " << filename << std::endl;
            continue;
        }

        // Allocate output buffer
        size_t max_out = gpucompress_max_compressed_size(file_size);
        std::vector<char> comp_buf(max_out);

        // Compress with ALGO_AUTO
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        cfg.error_bound = config.error_bound;

        size_t out_size = max_out;
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        auto t0 = std::chrono::high_resolution_clock::now();
        int comp_rc = gpucompress_compress(
            raw.data(), file_size,
            comp_buf.data(), &out_size,
            &cfg, &stats);
        auto t1 = std::chrono::high_resolution_clock::now();
        double comp_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Track experience delta
        size_t curr_experience = gpucompress_experience_count();
        size_t exp_delta = curr_experience - prev_experience;
        prev_experience = curr_experience;

        bool explored = (exp_delta > 1);
        if (explored) total_explorations++;

        if (comp_rc != GPUCOMPRESS_SUCCESS) {
            std::cerr << "Warning: compress failed for " << filename
                      << ": " << gpucompress_error_string((gpucompress_error_t)comp_rc)
                      << std::endl;
            continue;
        }

        double ratio = stats.compression_ratio;
        double throughput = (comp_time > 0.0)
            ? (static_cast<double>(file_size) / (1024.0 * 1024.0)) / (comp_time / 1000.0)
            : 0.0;

        // Compute MAPE from NN prediction
        double pred_ratio = stats.predicted_ratio;
        double mape = (ratio > 0.0)
            ? std::abs(pred_ratio - ratio) / ratio
            : 0.0;

        total_mape += mape;

        // Rolling MAPE buffer
        rolling_mape[rolling_idx % ROLLING_WINDOW] = mape;
        rolling_idx++;
        if (rolling_filled < ROLLING_WINDOW) rolling_filled++;

        total_ratio += ratio;
        files_processed++;

        // Decode algorithm/shuffle/quantization from stats
        const char* algo_name = gpucompress_algorithm_name(stats.algorithm_used);
        int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(stats.preprocessing_used);
        const char* quant_str = (stats.preprocessing_used & GPUCOMPRESS_PREPROC_QUANTIZE)
                                ? "linear" : "none";

        // Print per-file summary
        bool reinforced = config.reinforce &&
                          (mape > static_cast<double>(config.reinforce_threshold));

        std::cout << "  [" << std::setw(3) << (fi + 1) << "/" << files.size() << "] "
                  << std::setw(50) << std::left << filename << std::right
                  << "  algo=" << std::setw(9) << algo_name
                  << "  ratio=" << std::fixed << std::setprecision(3) << ratio
                  << "  mape=" << std::setprecision(1) << (mape * 100.0) << "%"
                  << "  exp_delta=" << exp_delta
                  << (explored ? " *EXPLORE*" : "")
                  << (reinforced ? " *SGD*" : "")
                  << std::endl;

        // Verbose: detailed reinforcement trace
        if (config.verbose) {
            std::cout << "           predicted=" << std::setprecision(4) << pred_ratio
                      << "  actual=" << std::setprecision(4) << ratio
                      << "  error=" << std::setprecision(1) << (mape * 100.0) << "%"
                      << "  threshold=" << (config.reinforce_threshold * 100.0) << "%"
                      << std::endl;

            if (reinforced) {
                float grad_norm = 0.0f;
                int num_samples = 0, was_clipped = 0;
                gpucompress_reinforce_last_stats(&grad_norm, &num_samples,
                                                  &was_clipped);
                std::cout << "           >>> SGD fired: "
                          << num_samples << " samples"
                          << "  grad_norm=" << std::setprecision(4) << grad_norm
                          << (was_clipped ? " (CLIPPED)" : "")
                          << "  lr=" << config.reinforce_lr
                          << std::endl;
            }
            std::cout << std::endl;
        }

        // Print rolling MAPE every ROLLING_WINDOW files
        if (rolling_idx > 0 && rolling_idx % ROLLING_WINDOW == 0) {
            double rolling_sum = 0.0;
            for (size_t ri = 0; ri < rolling_filled; ++ri)
                rolling_sum += rolling_mape[ri];
            double rolling_avg = rolling_sum / rolling_filled;
            std::cout << "  Rolling MAPE (last " << ROLLING_WINDOW << "): "
                      << std::fixed << std::setprecision(1)
                      << (rolling_avg * 100.0) << "%" << std::endl;
        }

        // Write CSV row
        csv_out << filename << ","
                << meta.field << ","
                << meta.scenario << ","
                << meta.timestep << ","
                << fmt_double(entropy, 6) << ","
                << fmt_double(mad, 6) << ","
                << fmt_double(second_deriv, 6) << ","
                << file_size << ","
                << algo_name << ","
                << shuffle_size << ","
                << quant_str << ","
                << config.error_bound << ","
                << out_size << ","
                << fmt_double(ratio, 6) << ","
                << fmt_double(comp_time, 4) << ","
                << fmt_double(throughput, 2) << ","
                << curr_experience << ","
                << exp_delta << ","
                << fmt_double(pred_ratio, 6) << ","
                << fmt_double(mape, 6) << "\n";
    }

    csv_out.close();

    auto wall_end = std::chrono::steady_clock::now();
    double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();

    // 6. Print aggregate summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Evaluation Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Files processed:     " << files_processed << std::endl;
    std::cout << "  Total experience:    " << gpucompress_experience_count() << std::endl;
    std::cout << "  Exploration events:  " << total_explorations
              << " (" << std::fixed << std::setprecision(1)
              << (files_processed > 0 ? 100.0 * total_explorations / files_processed : 0.0)
              << "% of files)" << std::endl;
    std::cout << "  Mean ratio:          " << std::setprecision(4)
              << (files_processed > 0 ? total_ratio / files_processed : 0.0) << std::endl;
    std::cout << "  Mean MAPE:           " << std::setprecision(1)
              << (files_processed > 0 ? 100.0 * total_mape / files_processed : 0.0)
              << "%" << std::endl;
    std::cout << "  Wall time:           " << std::setprecision(1) << wall_time << "s" << std::endl;
    std::cout << "  Output:              " << config.output << std::endl;
    std::cout << "  Experience:          " << config.experience << std::endl;
    std::cout << "========================================" << std::endl;

    gpucompress_disable_active_learning();
    gpucompress_cleanup();
    return 0;
}
