/**
 * @file nyx_amrex_bridge.hpp
 * @brief Header-only bridge between AMReX MultiFab and GPUCompress
 *
 * Writes compressed HDF5 using GPUCompress VOL connector — follows the
 * exact same pattern as VPIC's write_gpu_to_hdf5() in vpic_compress_deck.cxx.
 * HDF5 handles chunking, VOL intercepts H5Dwrite with GPU pointers.
 */

#ifndef NYX_AMREX_BRIDGE_HPP
#define NYX_AMREX_BRIDGE_HPP

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
/* Kendall tau profiler for ranking CSV — path relative to GPUCompress root.
 * The build must add GPUCompress/benchmarks to the include path. */
#ifndef KENDALL_TAU_PROFILER_PATH
#include "kendall_tau_profiler.cuh"
#else
#include KENDALL_TAU_PROFILER_PATH
#endif

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_VisMF.H>

#include <hdf5.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdio>
#include <cmath>

namespace gpucompress_nyx_bridge {

/* ── Per-chunk CSV diagnostic logger ─────────────────────────────
 * Mirrors the VPIC benchmark CSV format so the cross-workload
 * regret plotter can consume them uniformly.
 * ─────────────────────────────────────────────────────────────── */

static inline void action_to_str(int action, char *buf, size_t bufsz) {
    static const char* algo_names[] = {"lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"};
    int algo = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    const char* a = (algo < 8) ? algo_names[algo] : "?";
    if (quant && shuf) snprintf(buf, bufsz, "%s+shuf+quant", a);
    else if (quant)    snprintf(buf, bufsz, "%s+quant", a);
    else if (shuf)     snprintf(buf, bufsz, "%s+shuf", a);
    else               snprintf(buf, bufsz, "%s", a);
}

struct DiagLogger {
    FILE* tc_csv;           /* timestep_chunks CSV */
    FILE* ranking_csv;      /* ranking CSV (top1_regret) */
    FILE* ranking_costs_csv;
    int   total_writes;     /* total expected writes (for milestone check) */
    float w0, w1, w2;       /* cost model weights */
    double error_bound;
    size_t chunk_bytes;
    bool  enabled;

    DiagLogger() : tc_csv(NULL), ranking_csv(NULL), ranking_costs_csv(NULL),
                   total_writes(0), w0(1), w1(1), w2(1), error_bound(0),
                   chunk_bytes(4*1024*1024), enabled(false) {}

    void open(const char* results_dir, const char* prefix,
              int n_writes, float cw0, float cw1, float cw2,
              double eb, size_t cb) {
        total_writes = n_writes;
        w0 = cw0; w1 = cw1; w2 = cw2;
        error_bound = eb;
        chunk_bytes = cb;
        enabled = true;

        char path[600];
        snprintf(path, sizeof(path), "%s/benchmark_%s_timestep_chunks.csv",
                 results_dir, prefix);
        tc_csv = fopen(path, "w");
        if (tc_csv) {
            fprintf(tc_csv, "rank,phase,timestep,chunk,action,action_orig,"
                    "predicted_ratio,actual_ratio,"
                    "predicted_comp_ms,actual_comp_ms_raw,"
                    "predicted_decomp_ms,actual_decomp_ms_raw,"
                    "predicted_psnr_db,actual_psnr_db,"
                    "mape_ratio,mape_comp,mape_decomp,mape_psnr,"
                    "sgd_fired,exploration_triggered,"
                    "cost_model_error_pct,actual_cost,predicted_cost,"
                    "explore_n_alt");
            for (int ei = 0; ei < 31; ei++)
                fprintf(tc_csv, ",explore_alt_%d,explore_ratio_%d,"
                        "explore_comp_ms_%d,explore_cost_%d", ei, ei, ei, ei);
            fprintf(tc_csv, ",feat_entropy,feat_mad,feat_deriv\n");
        }

        snprintf(path, sizeof(path), "%s/benchmark_%s_ranking.csv",
                 results_dir, prefix);
        ranking_csv = fopen(path, "w");
        if (ranking_csv) write_ranking_csv_header(ranking_csv);

        snprintf(path, sizeof(path), "%s/benchmark_%s_ranking_costs.csv",
                 results_dir, prefix);
        ranking_costs_csv = fopen(path, "w");
        if (ranking_costs_csv) write_ranking_costs_csv_header(ranking_costs_csv);
    }

    void close() {
        if (tc_csv) { fclose(tc_csv); tc_csv = NULL; }
        if (ranking_csv) { fclose(ranking_csv); ranking_csv = NULL; }
        if (ranking_costs_csv) { fclose(ranking_costs_csv); ranking_costs_csv = NULL; }
    }

    /* Write per-chunk diagnostics after H5Dwrite completes */
    void log_chunks(const char* phase, int timestep) {
        if (!tc_csv) return;
        int n_hist = gpucompress_get_chunk_history_count();
        for (int ci = 0; ci < n_hist; ci++) {
            gpucompress_chunk_diag_t dd;
            if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;

            double mr = 0, mc = 0, md = 0;
            if (dd.actual_ratio > 0)
                mr = fmin(200.0, fabs(dd.predicted_ratio - dd.actual_ratio)
                          / fabs(dd.actual_ratio) * 100.0);
            if (dd.compression_ms > 0)
                mc = fmin(200.0, fabs(dd.predicted_comp_time - dd.compression_ms)
                          / fabs(dd.compression_ms) * 100.0);
            if (dd.decompression_ms > 0)
                md = fmin(200.0, fabs(dd.predicted_decomp_time - dd.decompression_ms)
                          / fabs(dd.decompression_ms) * 100.0);
            /* PSNR MAPE: lossless filter via the actual_psnr = -1.0 sentinel
             * set in gpucompress_compress.cpp:259. For lossless chunks emit
             * NaN (not 0) so cross-timestep aggregators using nanmean skip
             * them entirely rather than averaging zeros (which silently
             * scales down the reported PSNR MAPE). Same gate used by VPIC,
             * WarpX, and the LAMMPS fix. */
            double mp;
            double pred_psnr_out, actual_psnr_out;
            if (dd.predicted_psnr > 0.0f
                && std::isfinite(dd.actual_psnr)
                && dd.actual_psnr > 0.0f) {
                mp = fmin(200.0, fabs((double)dd.predicted_psnr - (double)dd.actual_psnr)
                                  / fabs((double)dd.actual_psnr) * 100.0);
                pred_psnr_out   = (double)dd.predicted_psnr;
                actual_psnr_out = (double)dd.actual_psnr;
            } else {
                mp              = std::nan("");
                pred_psnr_out   = std::nan("");
                actual_psnr_out = std::nan("");
            }

            char action_str[40], orig_str[40];
            action_to_str(dd.nn_action, action_str, sizeof(action_str));
            action_to_str(dd.nn_original_action, orig_str, sizeof(orig_str));

            fprintf(tc_csv,
                    "0,%s,%d,%d,%s,%s,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                    "%.2f,%.2f,"
                    "%.2f,%.2f,%.2f,%.2f,%d,%d,"
                    "%.4f,%.4f,%.4f,%d",
                    phase, timestep, ci, action_str, orig_str,
                    (double)dd.predicted_ratio, (double)dd.actual_ratio,
                    (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                    (double)dd.predicted_decomp_time, (double)dd.decompression_ms_raw,
                    pred_psnr_out, actual_psnr_out,
                    mr, mc, md, mp,
                    dd.sgd_fired, dd.exploration_triggered,
                    (double)dd.cost_model_error_pct,
                    (double)dd.actual_cost, (double)dd.predicted_cost,
                    dd.explore_n_alternatives);

            for (int ei = 0; ei < 31; ei++) {
                if (ei < dd.explore_n_alternatives) {
                    char ea_str[40];
                    action_to_str(dd.explore_alternatives[ei], ea_str, sizeof(ea_str));
                    fprintf(tc_csv, ",%s,%.4f,%.4f,%.4f",
                            ea_str, (double)dd.explore_ratios[ei],
                            (double)dd.explore_comp_ms[ei],
                            (double)dd.explore_costs[ei]);
                } else {
                    fprintf(tc_csv, ",,,,");
                }
            }
            fprintf(tc_csv, ",%.4f,%.6f,%.6f\n",
                    (double)dd.feat_entropy, (double)dd.feat_mad,
                    (double)dd.feat_deriv);
        }
        fflush(tc_csv);
    }

    /* Run ranking profiler at milestone timesteps */
    void log_ranking(const void* d_data, size_t total_bytes,
                     const char* phase, int timestep) {
        if (!ranking_csv) return;
        if (!is_ranking_milestone(timestep, total_writes)) return;

        float bw = gpucompress_get_bandwidth_bytes_per_ms();
        RankingMilestoneResult result = {};
        run_ranking_profiler(d_data, total_bytes, chunk_bytes,
                             error_bound, w0, w1, w2, bw,
                             3, ranking_csv, ranking_costs_csv,
                             phase, timestep, &result);
        fprintf(stderr, "    [ranking] T=%d: tau=%.3f regret=%.3fx (%.0fms)\n",
                timestep, result.mean_tau, result.mean_regret,
                result.profiling_ms);
    }
};

/**
 * Initialize GPUCompress + HDF5 VOL connector. Call once.
 * Returns the FAPL with VOL configured, or H5I_INVALID_HID on failure.
 */
inline hid_t init(const char* weights_path)
{
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) return H5I_INVALID_HID;

    H5Z_gpucompress_register();
    hid_t vol_id = H5VL_gpucompress_register();
    (void)vol_id;

    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    return fapl;
}

/**
 * Write GPU-resident data to compressed HDF5 — same pattern as VPIC deck.
 * HDF5 chunking + VOL connector handles compression transparently.
 */
inline void write_gpu_to_hdf5(const char* filename, const char* dset_name,
                               const void* d_data, size_t n_elements,
                               hid_t h5type, size_t chunk_elements,
                               hid_t fapl,
                               gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
                               double error_bound = 0.0)
{
    if (chunk_elements > n_elements) chunk_elements = n_elements;
    hsize_t dims[1]  = { (hsize_t)n_elements };
    hsize_t cdims[1] = { (hsize_t)chunk_elements };

    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    unsigned int preproc = 0;
    if (error_bound > 0.0) preproc = GPUCOMPRESS_PREPROC_QUANTIZE;
    H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);

    hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset >= 0) {
        /* d_data is a CUDA device pointer — VOL detects this and
         * compresses each chunk on GPU, writes pre-compressed bytes */
        H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
    }

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
}

/**
 * Read back a compressed HDF5 file via VOL and verify bitwise against original GPU data.
 * Returns 0 on success (bitwise match), -1 on mismatch or error.
 */
inline int verify_gpu_hdf5(const char* filename, const char* dset_name,
                            const void* d_original, size_t n_elements,
                            hid_t h5type, hid_t fapl)
{
    hid_t fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    if (fid < 0) return -1;

    hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    size_t elem_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    size_t total_bytes = n_elements * elem_size;

    /* Allocate GPU buffer for decompressed readback */
    void* d_readback = nullptr;
    cudaMalloc(&d_readback, total_bytes);

    /* VOL reads compressed chunks from disk, decompresses on GPU */
    H5Dread(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    cudaDeviceSynchronize();

    H5Dclose(dset);
    H5Fclose(fid);

    /* Bitwise comparison on host */
    std::vector<char> h_orig(total_bytes);
    std::vector<char> h_read(total_bytes);
    cudaMemcpy(h_orig.data(), d_original, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read.data(), d_readback, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_readback);

    int result = memcmp(h_orig.data(), h_read.data(), total_bytes);
    return (result == 0) ? 0 : -1;
}

/**
 * Write an entire MultiFab to compressed HDF5 files via VOL connector.
 * One HDF5 file per FArrayBox, each with chunked+compressed dataset.
 *
 * @param dir          Output directory
 * @param mf           MultiFab (GPU-resident data)
 * @param varnames     Variable names
 * @param fapl         File access property list from init()
 * @param chunk_bytes  HDF5 chunk size in bytes (default 4 MiB)
 * @param algo         Compression algorithm
 * @param error_bound  0.0 = lossless
 * @return Total compressed bytes
 */
inline long write_multifab_compressed(
    const std::string& dir,
    const amrex::MultiFab& mf,
    const amrex::Vector<std::string>& varnames,
    hid_t fapl,
    size_t chunk_bytes = 4 * 1024 * 1024,
    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
    double error_bound = 0.0,
    bool verify = false,
    DiagLogger* logger = nullptr,
    const char* phase = "nn-rl+exp50",
    int timestep = -1)
{
    amrex::Gpu::streamSynchronize();

    int ncomp = mf.nComp();
    long total_compressed = 0;
    long total_original = 0;

    if (amrex::ParallelDescriptor::IOProcessor())
        amrex::UtilCreateDirectory(dir, 0755);
    amrex::ParallelDescriptor::Barrier();

    hid_t h5type = (sizeof(amrex::Real) == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    size_t elem_size = sizeof(amrex::Real);
    size_t chunk_elems = chunk_bytes / elem_size;

    int fab_idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi, ++fab_idx)
    {
        const amrex::FArrayBox& fab = mf[mfi];
        long ncells = mfi.validbox().numPts();
        size_t n_elements = (size_t)ncells * ncomp;
        size_t fab_bytes = n_elements * elem_size;

        const void* d_ptr = (const void*)fab.dataPtr();

        char fname[256];
        snprintf(fname, sizeof(fname), "%s/fab_%04d.h5", dir.c_str(), fab_idx);

        gpucompress_reset_chunk_history();
        H5VL_gpucompress_reset_stats();
        write_gpu_to_hdf5(fname, "data", d_ptr, n_elements,
                          h5type, chunk_elems, fapl, algo, error_bound);

        /* Sync after each file to avoid async conflicts with AMReX */
        cudaDeviceSynchronize();

        int n_comp_chunks = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, &n_comp_chunks, NULL);

        /* ── Always read back for decomp-time profiling ──
         *
         * Reopen the file and H5Dread it. The read routes through the
         * GPUCompress VOL, which times the nvCOMP decompress kernel and
         * updates decompression_ms_raw / decompression_ms in the same
         * chunk-history slots that H5Dwrite filled. Without this, the
         * mape_decomp column in the chunks CSV is permanently zero.
         * Same pattern as VPIC (vpic_benchmark_deck.cxx:1584), the
         * WarpX FlushFormat patch, and the LAMMPS udf. Must run BEFORE
         * logger->log_chunks() so the slots are populated when the
         * logger walks the chunk history. */
        {
            hid_t rfid = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
            if (rfid >= 0) {
                hid_t rdset = H5Dopen2(rfid, "data", H5P_DEFAULT);
                if (rdset >= 0) {
                    void* d_readback = nullptr;
                    cudaMalloc(&d_readback, fab_bytes);
                    H5Dread(rdset, h5type, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_readback);
                    cudaDeviceSynchronize();

                    if (verify) {
                        std::vector<char> h_orig(fab_bytes);
                        std::vector<char> h_read(fab_bytes);
                        cudaMemcpy(h_orig.data(), d_ptr, fab_bytes,
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_read.data(), d_readback, fab_bytes,
                                   cudaMemcpyDeviceToHost);
                        if (memcmp(h_orig.data(), h_read.data(),
                                   fab_bytes) != 0) {
                            amrex::Print()
                                << "[GPUCompress] VERIFY FAILED: fab_"
                                << fab_idx << " bitwise mismatch!\n";
                            cudaFree(d_readback);
                            H5Dclose(rdset);
                            H5Fclose(rfid);
                            amrex::Abort("GPUCompress lossless verification failed");
                        }
                    }
                    cudaFree(d_readback);
                    H5Dclose(rdset);
                }
                H5Fclose(rfid);
            }
        }

        /* Log per-chunk diagnostics + ranking profiler at milestones.
         * Must come AFTER the read-back above so dd.decompression_ms_raw
         * is populated when log_chunks() walks the chunk history. */
        if (logger && logger->enabled && amrex::ParallelDescriptor::IOProcessor()) {
            int ts = (timestep >= 0) ? timestep : fab_idx;
            logger->log_chunks(phase, ts);
            logger->log_ranking(d_ptr, fab_bytes, phase, ts);
        }

        total_original += fab_bytes;
    }

    /* Get total compressed from VOL pipeline timing */
    double stage1_ms = 0, drain_ms = 0, io_drain_ms = 0, total_ms = 0;
    H5VL_gpucompress_get_stage_timing(&stage1_ms, &drain_ms, &io_drain_ms, &total_ms);

    return total_original;  /* caller can check file sizes for actual compression */
}

/**
 * Print compression statistics.
 */
inline void print_stats(const std::string& label,
                        long original_bytes, long compressed_bytes,
                        double elapsed_ms = 0.0)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    double ratio = (compressed_bytes > 0)
                 ? (double)original_bytes / compressed_bytes : 0.0;
    double orig_mb = original_bytes / (1024.0 * 1024.0);
    double comp_mb = compressed_bytes / (1024.0 * 1024.0);

    amrex::Print() << "[GPUCompress] " << label << ": "
                   << orig_mb << " MB -> " << comp_mb << " MB"
                   << " (ratio " << ratio << "x)";
    if (elapsed_ms > 0.0)
        amrex::Print() << " in " << elapsed_ms << " ms"
                       << " (" << (orig_mb / (elapsed_ms / 1000.0)) << " MB/s)";
    amrex::Print() << "\n";
}

} /* namespace gpucompress_nyx_bridge */

#endif /* NYX_AMREX_BRIDGE_HPP */
