/**
 * synthetic/main.cu
 *
 * GPU Synthetic Benchmark — generates float32 chunks on the GPU using the
 * palette-based 32-bin system and applies all 64 compression configurations
 * (8 algorithms x 2 shuffle x 4 quantization levels) to each chunk.
 *
 * Output: CSV with one row per (chunk x config) trial.
 *
 * Usage:
 *   ./synthetic_benchmark [options]
 *
 * Options:
 *   -n, --num-chunks <N>    Number of chunks to generate (default: 1000)
 *   -o, --output <path>     Output CSV path (default: synthetic_results.csv)
 *   -s, --seed <N>          RNG seed (default: 42)
 *   -d, --device <N>        CUDA device (default: 0)
 *   -h, --help              Show help
 */

#include <cuda_runtime.h>
#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gpucompress.h"
#include "configs.hpp"
#include "generator.cuh"
#include "quality.hpp"

// ============================================================
// CSV header
// ============================================================

static const char* CSV_HEADER =
    "chunk_id,"
    "num_elements,"
    "size_bytes,"
    "palette,"
    "bin_width,"
    "perturbation,"
    "fill_mode,"
    "algorithm,"
    "shuffle_bytes,"
    "quantized,"
    "error_bound,"
    "entropy_bits,"
    "mad,"
    "second_derivative,"
    "data_range,"
    "comp_time_ms,"
    "decomp_time_ms,"
    "lib_comp_time_ms,"
    "original_bytes,"
    "compressed_bytes,"
    "compression_ratio,"
    "comp_throughput_mbps,"
    "decomp_throughput_mbps,"
    "psnr_db,"
    "ssim,"
    "max_abs_error,"
    "rmse,"
    "mean_abs_error,"
    "bit_rate,"
    "success\n";

// ============================================================
// Helpers
// ============================================================

static std::string fmt(double v, int prec = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

static std::string fmt_psnr(double v) {
    if (std::isinf(v)) return "inf";
    return fmt(v, 4);
}

// ============================================================
// CLI config
// ============================================================

struct CLIArgs {
    int         num_chunks = 1000;
    std::string output     = "synthetic_results.csv";
    uint64_t    seed       = 42;
    int         device     = 0;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -n, --num-chunks <N>   Chunks to generate (default: 1000)\n"
        "  -o, --output <path>    CSV output (default: synthetic_results.csv)\n"
        "  -s, --seed <N>         RNG seed (default: 42)\n"
        "  -d, --device <N>       CUDA device (default: 0)\n"
        "  -h, --help             Show this help\n",
        prog);
}

static CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;
    static struct option long_opts[] = {
        {"num-chunks", required_argument, 0, 'n'},
        {"output",     required_argument, 0, 'o'},
        {"seed",       required_argument, 0, 's'},
        {"device",     required_argument, 0, 'd'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "n:o:s:d:h", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'n': args.num_chunks = std::atoi(optarg);  break;
            case 'o': args.output     = optarg;             break;
            case 's': args.seed       = (uint64_t)std::stoull(optarg); break;
            case 'd': args.device     = std::atoi(optarg);  break;
            case 'h': print_usage(argv[0]); exit(0);
            default:  break;
        }
    }
    return args;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    CLIArgs args = parse_args(argc, argv);

    // Select CUDA device
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (args.device >= n_devices) {
        fprintf(stderr, "Error: device %d not available (%d devices found)\n",
                args.device, n_devices);
        return 1;
    }
    cudaSetDevice(args.device);

    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, args.device);
        printf("Device %d: %s\n", args.device, prop.name);
    }

    // Initialize gpucompress (no NN weights needed for explicit algorithm selection)
    int rc_init = gpucompress_init(nullptr);
    if (rc_init != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "Error: gpucompress_init failed (%d)\n", rc_init);
        return 1;
    }

    // Build compression config table (64 configs)
    std::vector<BenchConfig> configs = make_bench_configs();
    printf("Compression configs: %zu\n", configs.size());
    printf("Chunks to generate:  %d\n", args.num_chunks);
    printf("Expected CSV rows:   %zu\n", (size_t)args.num_chunks * configs.size());
    printf("Output:              %s\n\n", args.output.c_str());

    // Open output CSV
    std::ofstream csv(args.output);
    if (!csv) {
        fprintf(stderr, "Error: cannot open %s for writing\n", args.output.c_str());
        gpucompress_cleanup();
        return 1;
    }
    csv << CSV_HEADER;

    // Pre-allocate GPU buffers sized for largest chunk (4MB = 1048576 floats)
    const size_t MAX_ELEMENTS  = 1048576;             // 4MB / 4 bytes
    const size_t MAX_IN_BYTES  = MAX_ELEMENTS * sizeof(float);
    const size_t MAX_COMP_BYTES= gpucompress_max_compressed_size(MAX_IN_BYTES);

    void* d_comp  = nullptr;
    void* d_decomp= nullptr;
    cudaMalloc(&d_comp,   MAX_COMP_BYTES);
    cudaMalloc(&d_decomp, MAX_IN_BYTES);

    // Host decompression buffer (sized for largest chunk)
    std::vector<float> h_orig(MAX_ELEMENTS);
    std::vector<float> h_decomp(MAX_ELEMENTS);

    // CUDA timing events (reused across all trials)
    cudaEvent_t ev_c0, ev_c1, ev_d0, ev_d1;
    cudaEventCreate(&ev_c0); cudaEventCreate(&ev_c1);
    cudaEventCreate(&ev_d0); cudaEventCreate(&ev_d1);

    // RNG for chunk parameter selection
    std::mt19937_64 rng(args.seed);
    uint64_t gen_seed = args.seed ^ 0xDEADBEEFULL;

    // Row buffer for batched CSV writes
    std::string row_buf;
    row_buf.reserve(4 * 1024 * 1024); // 4MB buffer

    auto wall_start = std::chrono::steady_clock::now();

    for (int chunk_id = 0; chunk_id < args.num_chunks; ++chunk_id) {
        // ── Random chunk parameters ────────────────────────────────────────
        ChunkParams params;
        params.palette     = (Palette)(rng() % PAL_COUNT);
        params.bin_width   = TRAINING_BIN_WIDTHS   [rng() % N_BIN_WIDTHS];
        params.perturbation= TRAINING_PERTURBATIONS[rng() % N_PERTURBATIONS];
        params.fill_mode   = (FillMode)(rng() % FILL_COUNT);
        params.num_elements= TRAINING_SIZES[rng() % N_SIZES];

        size_t input_bytes = params.num_elements * sizeof(float);

        // ── Generate chunk on GPU ──────────────────────────────────────────
        float* d_chunk = generate_chunk_gpu(params, gen_seed + (uint64_t)chunk_id);
        cudaDeviceSynchronize();

        // Copy to host for quality computation reference
        cudaMemcpy(h_orig.data(), d_chunk, input_bytes, cudaMemcpyDeviceToHost);

        // ── Per-chunk stats (computed once, independent of compression config)
        double entropy = 0.0, mad = 0.0, second_deriv = 0.0;
        {
            gpucompress_error_t rc = gpucompress_compute_stats_gpu(
                d_chunk, input_bytes, &entropy, &mad, &second_deriv);
            if (rc != GPUCOMPRESS_SUCCESS) {
                fprintf(stderr, "Warning: stats computation failed for chunk %d\n", chunk_id);
            }
        }

        // Data range (vmax - vmin) computed on host from already-copied data
        float h_vmin = *std::min_element(h_orig.begin(), h_orig.begin() + params.num_elements);
        float h_vmax = *std::max_element(h_orig.begin(), h_orig.begin() + params.num_elements);
        double data_range = static_cast<double>(h_vmax) - static_cast<double>(h_vmin);

        // ── Apply all 64 compression configurations ────────────────────────
        for (const auto& cfg : configs) {
            gpucompress_config_t gc;
            gc.algorithm    = static_cast<gpucompress_algorithm_t>(cfg.algo_id);
            gc.preprocessing= cfg.preprocessing;
            gc.error_bound  = cfg.error_bound;
            gc.cuda_device  = args.device;
            gc.cuda_stream  = nullptr;

            size_t comp_size = MAX_COMP_BYTES;
            gpucompress_stats_t stats;
            memset(&stats, 0, sizeof(stats));

            // ── Compression (CUDA event timed) ────────────────────────────
            cudaEventRecord(ev_c0);
            int rc_c = gpucompress_compress_gpu(
                d_chunk, input_bytes,
                d_comp, &comp_size,
                &gc, &stats, nullptr);
            cudaEventRecord(ev_c1);
            cudaEventSynchronize(ev_c1);

            float comp_ms = 0.0f;
            cudaEventElapsedTime(&comp_ms, ev_c0, ev_c1);

            bool success = (rc_c == GPUCOMPRESS_SUCCESS);

            // ── Decompression (CUDA event timed) ──────────────────────────
            size_t decomp_size = input_bytes;
            float  decomp_ms   = 0.0f;
            int    rc_d        = -1;

            if (success) {
                cudaEventRecord(ev_d0);
                rc_d = gpucompress_decompress_gpu(
                    d_comp, comp_size,
                    d_decomp, &decomp_size, nullptr);
                cudaEventRecord(ev_d1);
                cudaEventSynchronize(ev_d1);
                cudaEventElapsedTime(&decomp_ms, ev_d0, ev_d1);
            }

            bool decomp_ok = (rc_d == GPUCOMPRESS_SUCCESS);
            success = success && decomp_ok;

            // ── Quality metrics ────────────────────────────────────────────
            QualityMetrics qm{};
            if (decomp_ok && decomp_size == input_bytes) {
                cudaMemcpy(h_decomp.data(), d_decomp, input_bytes,
                           cudaMemcpyDeviceToHost);

                if (!cfg.quantized) {
                    // Lossless: verify exact match
                    bool exact = (memcmp(h_orig.data(), h_decomp.data(),
                                        input_bytes) == 0);
                    if (exact) {
                        qm = perfect_quality(params.num_elements, comp_size);
                    } else {
                        // Unexpected mismatch — compute actual error
                        qm = compute_quality(h_orig.data(), h_decomp.data(),
                                             params.num_elements, comp_size);
                        success = false;
                    }
                } else {
                    // Lossy: compute quality metrics
                    qm = compute_quality(h_orig.data(), h_decomp.data(),
                                        params.num_elements, comp_size);
                }
            }

            // ── Throughput ────────────────────────────────────────────────
            double mb_in = static_cast<double>(input_bytes) / (1024.0 * 1024.0);
            double comp_tp  = (comp_ms  > 0.0f) ? mb_in / (comp_ms  / 1000.0) : 0.0;
            double decomp_tp= (decomp_ms > 0.0f) ? mb_in / (decomp_ms / 1000.0) : 0.0;
            double comp_ratio = (comp_size > 0 && success)
                ? static_cast<double>(input_bytes) / static_cast<double>(comp_size)
                : 0.0;

            // ── Write CSV row ──────────────────────────────────────────────
            std::ostringstream row;
            row << chunk_id                         << ","
                << params.num_elements              << ","
                << input_bytes                      << ","
                << palette_name(params.palette)     << ","
                << params.bin_width                 << ","
                << params.perturbation              << ","
                << fillmode_name(params.fill_mode)  << ","
                << cfg.algo_name                    << ","
                << cfg.shuffle_bytes                << ","
                << (cfg.quantized ? "True" : "False") << ","
                << cfg.error_bound                  << ","
                << fmt(entropy,      6)             << ","
                << fmt(mad,          6)             << ","
                << fmt(second_deriv, 6)             << ","
                << fmt(data_range,   6)             << ","
                << fmt(comp_ms,      4)             << ","
                << fmt(decomp_ms,    4)             << ","
                << fmt(stats.actual_comp_time_ms, 4) << ","
                << input_bytes                      << ","
                << (success ? (long long)comp_size : 0LL) << ","
                << fmt(comp_ratio,   6)             << ","
                << fmt(comp_tp,      2)             << ","
                << fmt(decomp_tp,    2)             << ","
                << fmt_psnr(qm.psnr)                << ","
                << fmt(qm.ssim,          8)         << ","
                << fmt(qm.max_abs_error, 6)         << ","
                << fmt(qm.rmse,          6)         << ","
                << fmt(qm.mean_abs_err,  6)         << ","
                << fmt(qm.bit_rate,      4)         << ","
                << (success ? "True" : "False")     << "\n";

            row_buf += row.str();
        }

        // Flush row buffer every chunk
        csv << row_buf;
        row_buf.clear();

        // Free generated chunk
        cudaFree(d_chunk);

        // Progress report every 50 chunks
        if ((chunk_id + 1) % 50 == 0 || chunk_id + 1 == args.num_chunks) {
            auto now     = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            double rate  = (elapsed > 0.0) ? (chunk_id + 1) / elapsed : 0.0;
            double eta   = (rate > 0.0) ? (args.num_chunks - chunk_id - 1) / rate : 0.0;
            fprintf(stderr, "\r  [%d/%d] %.1f%% — %.1f chunks/s, ETA %.0fs   ",
                    chunk_id + 1, args.num_chunks,
                    (chunk_id + 1) * 100.0 / args.num_chunks,
                    rate, eta);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n");

    // Cleanup
    cudaEventDestroy(ev_c0); cudaEventDestroy(ev_c1);
    cudaEventDestroy(ev_d0); cudaEventDestroy(ev_d1);
    cudaFree(d_comp);
    cudaFree(d_decomp);
    csv.close();
    gpucompress_cleanup();

    auto wall_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(wall_end - wall_start).count();
    size_t total_rows = (size_t)args.num_chunks * configs.size();

    printf("\nDone.\n");
    printf("  Chunks:  %d\n", args.num_chunks);
    printf("  Rows:    %zu\n", total_rows);
    printf("  Time:    %.1f s\n", total_s);
    printf("  Output:  %s\n", args.output.c_str());

    return 0;
}
