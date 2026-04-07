#include "FlushFormatGPUCompress.H"

#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_Gpu.H>

#ifdef WARPX_USE_GPUCOMPRESS
#   include <gpucompress.h>
#   include <gpucompress_warpx.h>
#   include <gpucompress_hdf5_vol.h>
#   include <gpucompress_hdf5.h>
#   include <hdf5.h>
#   include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <sys/stat.h>

#ifdef WARPX_USE_GPUCOMPRESS
#include "kendall_tau_profiler.cuh"

/* Per-chunk CSV logger state — persists across WriteToFile calls */
static FILE* s_tc_csv = nullptr;
static FILE* s_ranking_csv = nullptr;
static FILE* s_ranking_costs_csv = nullptr;
static int   s_write_count = 0;
static int   s_total_writes = 0;  /* estimated, for milestone check */
static float s_w0 = 1, s_w1 = 1, s_w2 = 1;
static double s_error_bound = 0.0;
static int   s_chunk_bytes = 4 * 1024 * 1024;
static bool  s_learning_initialized = false;

/* In-situ decompression profiling. When set, every H5Dwrite is followed
 * by an in-process H5Dread of the same file (page-cache hit, no real
 * disk I/O). The H5Dread routes through the GPUCompress VOL, which
 * times the nvCOMP decompress kernel with cudaEventRecord and updates
 * the same chunk-history slot the write filled — so each slot ends up
 * with both predicted_decomp_time (set at write) and decompression_ms_raw
 * (set at read), enabling per-chunk decomp MAPE in the CSV.
 *
 * Enabled by env var WARPX_PROFILE_DECOMP=1. The threshold sweep
 * always sets it; production WarpX runs leave it off and pay nothing. */
static bool  s_profile_decomp = false;

static void action_to_str_warpx(int action, char *buf, size_t bufsz) {
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
#endif

FlushFormatGPUCompress::FlushFormatGPUCompress ()
{
#ifdef WARPX_USE_GPUCOMPRESS
    /* Read optional config from inputs file:
     *   gpucompress.weights_path = "path/to/weights.nnwt"
     *   gpucompress.algorithm    = "auto"    (or lz4, zstd, etc.)
     *   gpucompress.error_bound  = 0.0       (0 = lossless)
     *   gpucompress.chunk_bytes  = 4194304   (4 MiB)
     *   gpucompress.verify       = 0
     */
    std::string weights_path;
    amrex::ParmParse pp("gpucompress");
    pp.query("weights_path", weights_path);

    gpucompress_error_t err = gpucompress_init(
        weights_path.empty() ? nullptr : weights_path.c_str());
    if (err == GPUCOMPRESS_SUCCESS) {
        H5Z_gpucompress_register();
        H5VL_gpucompress_register();
        m_initialized = true;

        /* Set policy weights */
        std::string policy = "balanced";
        pp.query("policy", policy);
        if (policy == "speed")         gpucompress_set_ranking_weights(1.0f, 0.0f, 0.0f);
        else if (policy == "balanced") gpucompress_set_ranking_weights(1.0f, 1.0f, 1.0f);
        else                           gpucompress_set_ranking_weights(0.0f, 0.0f, 1.0f);

        /* Enable online SGD learning + exploration */
        if (!s_learning_initialized) {
            float sgd_lr = 0.2f, sgd_mape = 0.10f, explore_thresh = 0.20f;
            int explore_k = 4;
            pp.query("sgd_lr", sgd_lr);
            pp.query("sgd_mape", sgd_mape);
            pp.query("explore_k", explore_k);
            pp.query("explore_thresh", explore_thresh);

            gpucompress_enable_online_learning();
            gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, 0.0f);
            gpucompress_set_exploration(1);
            gpucompress_set_exploration_k(explore_k);
            gpucompress_set_exploration_threshold(explore_thresh);
            s_learning_initialized = true;

            amrex::Print() << "[GPUCompress] Online learning: lr=" << sgd_lr
                           << " mape=" << sgd_mape << " explore_k=" << explore_k
                           << " explore_thresh=" << explore_thresh << "\n";
        }

        /* Parse in-situ decomp profiling toggle (off by default; the
         * threshold sweep always enables it). Done once at constructor
         * time so the WriteToFile hot path can branch on a static. */
        {
            const char* pd_env = std::getenv("WARPX_PROFILE_DECOMP");
            s_profile_decomp = (pd_env && atoi(pd_env) != 0);
            if (s_profile_decomp && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "[GPUCompress] In-situ decompression profiling enabled "
                                  "(every H5Dwrite is followed by an in-process H5Dread "
                                  "to populate decompression_ms_raw in chunk history)\n";
            }
        }

        /* Open per-chunk CSV + ranking CSV if WARPX_LOG_DIR is set */
        const char* log_dir = std::getenv("WARPX_LOG_DIR");
        if (log_dir && log_dir[0] && !s_tc_csv && amrex::ParallelDescriptor::IOProcessor()) {
            mkdir(log_dir, 0755);

            /* Save cost model params for ranking profiler */
            double eb = 0.0;
            pp.query("error_bound", eb);
            s_error_bound = eb;
            pp.query("chunk_bytes", s_chunk_bytes);
            if (policy == "speed")         { s_w0=1; s_w1=0; s_w2=0; }
            else if (policy == "balanced") { s_w0=1; s_w1=1; s_w2=1; }
            else                           { s_w0=0; s_w1=0; s_w2=1; }

            /* Estimate total writes for milestone check */
            const char* ms_env = std::getenv("WARPX_MAX_STEP");
            const char* di_env = std::getenv("WARPX_DIAG_INTERVAL");
            int est_ms = ms_env ? atoi(ms_env) : 100;
            int est_di = di_env ? atoi(di_env) : 10;
            s_total_writes = (est_di > 0) ? (est_ms / est_di + 1) : 10;

            char csv_path[700];
            snprintf(csv_path, sizeof(csv_path),
                     "%s/benchmark_warpx_timestep_chunks.csv", log_dir);
            s_tc_csv = fopen(csv_path, "w");
            if (s_tc_csv) {
                /* PSNR columns (predicted_psnr_db, actual_psnr_db, mape_psnr)
                 * are populated only for chunks where the NN selected a
                 * quantized config. Lossless chunks emit `nan` for all three
                 * so cross-timestep aggregators using nanmean correctly skip
                 * them rather than averaging zeros (which silently scales
                 * down the reported PSNR MAPE). */
                fprintf(s_tc_csv, "rank,phase,timestep,chunk,"
                        "predicted_ratio,actual_ratio,"
                        "predicted_comp_ms,actual_comp_ms_raw,"
                        "mape_ratio,mape_comp,"
                        "sgd_fired,exploration_triggered,"
                        "cost_model_error_pct,actual_cost,predicted_cost,"
                        "predicted_decomp_ms,actual_decomp_ms_raw,mape_decomp,"
                        "predicted_psnr_db,actual_psnr_db,mape_psnr\n");
            }

            snprintf(csv_path, sizeof(csv_path),
                     "%s/benchmark_warpx_ranking.csv", log_dir);
            s_ranking_csv = fopen(csv_path, "w");
            if (s_ranking_csv) write_ranking_csv_header(s_ranking_csv);

            snprintf(csv_path, sizeof(csv_path),
                     "%s/benchmark_warpx_ranking_costs.csv", log_dir);
            s_ranking_costs_csv = fopen(csv_path, "w");
            if (s_ranking_costs_csv) write_ranking_costs_csv_header(s_ranking_costs_csv);

            amrex::Print() << "[GPUCompress] Chunk + ranking CSV logging to " << log_dir << "\n";
        }
    } else {
        amrex::Print() << "[GPUCompress] WARNING: init failed (error "
                       << err << "), falling back to uncompressed HDF5\n";
    }
#else
    amrex::Abort("WarpX was not built with GPUCompress support. "
                 "Rebuild with -DWarpX_GPUCOMPRESS=ON");
#endif
}

FlushFormatGPUCompress::~FlushFormatGPUCompress ()
{
#ifdef WARPX_USE_GPUCOMPRESS
    if (m_initialized) {
        gpucompress_cleanup();
    }
#endif
}

void
FlushFormatGPUCompress::WriteToFile (
    const amrex::Vector<std::string>& varnames,
    const amrex::Vector<amrex::MultiFab>& mf,
    amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<int> iteration, const double time,
    const amrex::Vector<ParticleDiag>& /*particle_diags*/, int nlev,
    const std::string prefix, int file_min_digits,
    bool /*plot_raw_fields*/,
    bool /*plot_raw_fields_guards*/,
    int verbose,
    const bool /*use_pinned_pc*/,
    bool /*isBTD*/, int /*snapshotID*/, int /*bufferID*/, int /*numBuffers*/,
    const amrex::Geometry& /*full_BTD_snapshot*/,
    bool /*isLastBTDFlush*/) const
{
#ifdef WARPX_USE_GPUCOMPRESS
    ABLASTR_PROFILE("FlushFormatGPUCompress::WriteToFile()");

    const std::string& dirname = amrex::Concatenate(prefix, iteration[0], file_min_digits);

    if (verbose > 0) {
        amrex::Print() << Utils::TextMsg::Info(
            "Writing GPUCompress output " + dirname);
    }

    /* Read per-call settings from inputs file */
    std::string algo_str = "auto";
    double error_bound = 0.0;
    int chunk_bytes = 4 * 1024 * 1024;
    int do_verify = 0;
    {
        amrex::ParmParse pp("gpucompress");
        pp.query("algorithm", algo_str);
        pp.query("error_bound", error_bound);
        pp.query("chunk_bytes", chunk_bytes);
        pp.query("verify", do_verify);
    }

    gpucompress_algorithm_t algo = gpucompress_algorithm_from_string(algo_str.c_str());
    const bool verify = (do_verify != 0);

    /* Create output directory */
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::UtilCreateDirectory(dirname, 0755);
    }
    amrex::ParallelDescriptor::Barrier();

    /* Set up HDF5 file access with VOL connector */
    hid_t fapl = H5P_DEFAULT;
    if (m_initialized) {
        hid_t native_id = H5VLget_connector_id_by_name("native");
        fapl = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_gpucompress(fapl, native_id, NULL);
        H5VLclose(native_id);
    }

    hid_t h5type = (sizeof(amrex::Real) == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    size_t elem_size = sizeof(amrex::Real);
    size_t chunk_elems = static_cast<size_t>(chunk_bytes) / elem_size;

    long total_original = 0;

    /* Iterate over AMR levels */
    for (int lev = 0; lev < nlev; ++lev) {
        amrex::Gpu::streamSynchronize();

        int ncomp = mf[lev].nComp();

        /* Write each component separately for clarity and variable naming */
        for (int icomp = 0; icomp < ncomp; ++icomp) {
            const std::string& vname = (icomp < static_cast<int>(varnames.size()))
                                     ? varnames[icomp] : "field_" + std::to_string(icomp);

            int fab_idx = 0;
            for (amrex::MFIter mfi(mf[lev]); mfi.isValid(); ++mfi, ++fab_idx) {
                const amrex::FArrayBox& fab = mf[lev][mfi];
                long ncells = mfi.validbox().numPts();
                size_t n_elements = static_cast<size_t>(ncells);
                size_t fab_bytes = n_elements * elem_size;

                /* Borrow device pointer directly from AMReX FArrayBox */
                const void* d_ptr = static_cast<const void*>(fab.dataPtr(icomp));

                char fname[512];
                std::snprintf(fname, sizeof(fname),
                    "%s/lev%d_%s_fab%04d.h5",
                    dirname.c_str(), lev, vname.c_str(), fab_idx);

                /* Write compressed HDF5 via VOL connector */
                size_t ce = (chunk_elems > n_elements) ? n_elements : chunk_elems;
                hsize_t dims[1]  = { static_cast<hsize_t>(n_elements) };
                hsize_t cdims[1] = { static_cast<hsize_t>(ce) };

                hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
                if (fid < 0) { continue; }

                hid_t space = H5Screate_simple(1, dims, nullptr);
                hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
                H5Pset_chunk(dcpl, 1, cdims);

                if (m_initialized) {
                    unsigned int shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
                    unsigned int preproc = 0;
                    if (error_bound > 0.0) { preproc = GPUCOMPRESS_PREPROC_QUANTIZE; }
                    H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);
                }

                hid_t dset = H5Dcreate2(fid, "data", h5type, space,
                                         H5P_DEFAULT, dcpl, H5P_DEFAULT);
                gpucompress_reset_chunk_history();

                if (dset >= 0) {
                    /* d_ptr is a GPU device pointer -- VOL detects and
                     * compresses each chunk on GPU before writing */
                    H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_ptr);

                    /* ── Optional in-situ read-back ──
                     *
                     * If decomp profiling or verify is enabled, reopen the
                     * file we just wrote and H5Dread it. The read routes
                     * through the GPUCompress VOL, which times the nvCOMP
                     * decompress kernel with cudaEventRecord and updates
                     * decompression_ms_raw / decompression_ms in the SAME
                     * chunk-history slots that H5Dwrite filled. After this
                     * block returns, every slot that compressed will also
                     * have its decomp timing populated.
                     *
                     * Note: chunk history is not reset between write and
                     * read — the read updates existing slots in place. See
                     * src/hdf5/H5VLgpucompress.cu:2181 and
                     * src/api/gpucompress_diagnostics.cpp:74. This is the
                     * same pattern VPIC and sdrbench use. */
                    /* Always read back so decompression_ms is recorded for
                     * every chunk. The bitwise verify gate (if true) is
                     * applied inside the block on the host-side memcmp
                     * only. Same pattern as VPIC (vpic_benchmark_deck.cxx
                     * line 1584: "Always read back for timing"). The
                     * s_profile_decomp env knob is now redundant and only
                     * kept for backward compat with the threshold sweep
                     * scripts that still set it. */
                    if (m_initialized) {
                        cudaDeviceSynchronize();
                        /* Close the writer's handles so the file is fully
                         * flushed before we reopen it. */
                        H5Dclose(dset);  dset  = H5I_INVALID_HID;
                        H5Pclose(dcpl);  dcpl  = H5I_INVALID_HID;
                        H5Sclose(space); space = H5I_INVALID_HID;
                        H5Fclose(fid);   fid   = H5I_INVALID_HID;

                        hid_t rfid = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
                        if (rfid >= 0) {
                            hid_t rdset = H5Dopen2(rfid, "data", H5P_DEFAULT);
                            if (rdset >= 0) {
                                void* d_readback = nullptr;
                                cudaMalloc(&d_readback, fab_bytes);
                                H5Dread(rdset, h5type, H5S_ALL, H5S_ALL,
                                        H5P_DEFAULT, d_readback);
                                cudaDeviceSynchronize();

                                /* Bitwise verify (host-side memcmp) only when
                                 * gpucompress.verify=1 was requested in the
                                 * inputs deck. Decomp profiling alone skips
                                 * this — the chunk-history update happens
                                 * inside H5Dread regardless. */
                                if (verify) {
                                    std::vector<char> h_orig(fab_bytes);
                                    std::vector<char> h_read(fab_bytes);
                                    cudaMemcpy(h_orig.data(), d_ptr, fab_bytes,
                                               cudaMemcpyDeviceToHost);
                                    cudaMemcpy(h_read.data(), d_readback, fab_bytes,
                                               cudaMemcpyDeviceToHost);
                                    if (std::memcmp(h_orig.data(), h_read.data(),
                                                    fab_bytes) != 0) {
                                        amrex::Print()
                                            << "[GPUCompress] VERIFY FAILED: "
                                            << vname << " lev" << lev
                                            << " fab" << fab_idx << "\n";
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

                    /* ── Per-chunk CSV row ──
                     *
                     * Walk chunk history once. By this point each slot has:
                     *   - compress-time fields from H5Dwrite (predicted_*,
                     *     compression_ms_raw, actual_ratio, …)
                     *   - decomp-time fields from H5Dread (decompression_ms_raw,
                     *     decompression_ms), if the read-back ran. Otherwise
                     *     decompression_ms_raw is 0 and the row's mape_decomp
                     *     comes out as 0, which the aggregator skips. */
                    if (s_tc_csv && amrex::ParallelDescriptor::IOProcessor()) {
                        int n_hist = gpucompress_get_chunk_history_count();
                        for (int ci = 0; ci < n_hist; ci++) {
                            gpucompress_chunk_diag_t dd;
                            if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;
                            /* Per-chunk MAPE in percent.
                             *
                             * Both heads of the NN clamp their outputs to a
                             * 5 ms floor (training artifact), so the
                             * comparison must use the same clamped denominator
                             * — otherwise we'd be comparing a clamped
                             * prediction against a sub-millisecond actual and
                             * the MAPE explodes. Gate on the unclamped raw
                             * wall-clock so every measured chunk gets a row,
                             * but use max(raw, 5) as the denominator. */
                            double mr = 0, mc = 0, md = 0;
                            if (dd.actual_ratio > 0)
                                mr = std::fabs(dd.predicted_ratio - dd.actual_ratio)
                                     / std::fabs(dd.actual_ratio) * 100.0;
                            if (dd.compression_ms_raw > 0) {
                                double clamped =
                                    std::max((double)dd.compression_ms_raw, 5.0);
                                mc = std::fabs((double)dd.predicted_comp_time - clamped)
                                     / clamped * 100.0;
                            }
                            if (dd.decompression_ms_raw > 0) {
                                double clamped =
                                    std::max((double)dd.decompression_ms_raw, 5.0);
                                md = std::fabs((double)dd.predicted_decomp_time - clamped)
                                     / clamped * 100.0;
                            }
                            /* PSNR MAPE: lossless filter via the actual_psnr
                             * = -1.0 sentinel set in gpucompress_compress.cpp:259.
                             * For lossless chunks we emit NaN (not 0) so the
                             * cross-timestep aggregator can use nanmean and
                             * skip them entirely. The compression-side analytical
                             * PSNR is only set when quantize_simple actually
                             * ran, so this gate is the same one VPIC and the
                             * NYX bridge use. */
                            double mp;
                            double pred_psnr_out, actual_psnr_out;
                            if (dd.predicted_psnr > 0.0f
                                && std::isfinite(dd.actual_psnr)
                                && dd.actual_psnr > 0.0f) {
                                mp = std::fabs((double)dd.predicted_psnr
                                               - (double)dd.actual_psnr)
                                     / std::fabs((double)dd.actual_psnr) * 100.0;
                                if (mp > 200.0) mp = 200.0;
                                pred_psnr_out   = (double)dd.predicted_psnr;
                                actual_psnr_out = (double)dd.actual_psnr;
                            } else {
                                mp              = std::nan("");
                                pred_psnr_out   = std::nan("");
                                actual_psnr_out = std::nan("");
                            }
                            char act_s[40], orig_s[40];
                            action_to_str_warpx(dd.nn_action, act_s, sizeof(act_s));
                            action_to_str_warpx(dd.nn_original_action, orig_s, sizeof(orig_s));
                            fprintf(s_tc_csv,
                                    "0,nn-rl+exp50,%d,%d,"
                                    "%.4f,%.4f,%.4f,%.4f,"
                                    "%.2f,%.2f,%d,%d,"
                                    "%.4f,%.4f,%.4f,"
                                    "%.4f,%.4f,%.2f,"
                                    "%.4f,%.4f,%.2f\n",
                                    s_write_count, ci,
                                    (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                    (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                                    mr, mc, dd.sgd_fired, dd.exploration_triggered,
                                    (double)dd.cost_model_error_pct,
                                    (double)dd.actual_cost, (double)dd.predicted_cost,
                                    (double)dd.predicted_decomp_time,
                                    (double)dd.decompression_ms_raw, md,
                                    pred_psnr_out, actual_psnr_out, mp);
                        }
                        fflush(s_tc_csv);
                    }
                }

                /* Unified cleanup. The read-back branch above closes its own
                 * handles and sets them to H5I_INVALID_HID, so the guards
                 * here are no-ops in that case. */
                if (dset  >= 0) H5Dclose(dset);
                if (dcpl  >= 0) H5Pclose(dcpl);
                if (space >= 0) H5Sclose(space);
                if (fid   >= 0) H5Fclose(fid);

                total_original += fab_bytes;
            }
        }
    }

    if (fapl != H5P_DEFAULT) {
        H5Pclose(fapl);
    }

    /* Summary */
    if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        double orig_mb = total_original / (1024.0 * 1024.0);
        amrex::Print() << "[GPUCompress] Wrote " << orig_mb
                       << " MB original data to " << dirname << "\n";
    }

    /* ── Raw field dump for standalone benchmark (WARPX_DUMP_FIELDS=1) ── */
    {
        static int warpx_dump_raw = -1;
        if (warpx_dump_raw < 0) {
            const char* env = std::getenv("WARPX_DUMP_FIELDS");
            warpx_dump_raw = (env && std::atoi(env)) ? 1 : 0;
        }
        if (warpx_dump_raw && amrex::ParallelDescriptor::IOProcessor()) {
            const char* raw_dir = std::getenv("WARPX_DUMP_DIR");
            if (raw_dir && raw_dir[0]) {
                char ts_dir[700];
                std::snprintf(ts_dir, sizeof(ts_dir), "%s/diag%05d",
                              raw_dir, iteration[0]);
                mkdir(ts_dir, 0755);

                for (int lev = 0; lev < nlev; ++lev) {
                    int fab_idx = 0;
                    for (amrex::MFIter mfi(mf[lev]); mfi.isValid(); ++mfi, ++fab_idx) {
                        const amrex::FArrayBox& fab = mf[lev][mfi];
                        long ncells = mfi.validbox().numPts();

                        for (int icomp = 0; icomp < mf[lev].nComp(); ++icomp) {
                            const void* d_ptr = static_cast<const void*>(
                                fab.dataPtr(icomp));
                            size_t nbytes = (size_t)ncells * elem_size;
                            void* h_buf = std::malloc(nbytes);
                            cudaMemcpy(h_buf, d_ptr, nbytes,
                                       cudaMemcpyDeviceToHost);

                            const std::string& vn =
                                (icomp < static_cast<int>(varnames.size()))
                                    ? varnames[icomp]
                                    : "field_" + std::to_string(icomp);
                            char fpath[800];
                            std::snprintf(fpath, sizeof(fpath),
                                "%s/lev%d_%s_fab%04d.f32",
                                ts_dir, lev, vn.c_str(), fab_idx);
                            FILE* fp = std::fopen(fpath, "wb");
                            if (fp) {
                                if (elem_size == 8) {
                                    /* amrex::Real is double — downcast to float32 */
                                    float* f32 = (float*)std::malloc(
                                        (size_t)ncells * sizeof(float));
                                    const double* src = (const double*)h_buf;
                                    for (long i = 0; i < ncells; i++)
                                        f32[i] = (float)src[i];
                                    std::fwrite(f32, sizeof(float),
                                                (size_t)ncells, fp);
                                    std::free(f32);
                                } else {
                                    std::fwrite(h_buf, sizeof(float),
                                                (size_t)ncells, fp);
                                }
                                std::fclose(fp);
                            }
                            std::free(h_buf);
                        }
                    }
                }
                if (s_write_count == 0)
                    amrex::Print() << "[GPUCompress] Raw field dump → "
                                   << raw_dir << "\n";
            }
        }
    }

    /* Run ranking profiler at milestone writes */
    if (s_ranking_csv && amrex::ParallelDescriptor::IOProcessor()
        && is_ranking_milestone(s_write_count, s_total_writes)) {
        /* Use the last FAB's data for ranking (representative of this write) */
        for (int lev = 0; lev < nlev; ++lev) {
            for (amrex::MFIter mfi(mf[lev]); mfi.isValid(); ++mfi) {
                const amrex::FArrayBox& fab = mf[lev][mfi];
                long ncells = mfi.validbox().numPts();
                size_t fab_bytes = (size_t)ncells * mf[lev].nComp() * elem_size;
                const void* d_ptr = static_cast<const void*>(fab.dataPtr());
                float bw = gpucompress_get_bandwidth_bytes_per_ms();
                RankingMilestoneResult result = {};
                run_ranking_profiler(d_ptr, fab_bytes, (size_t)s_chunk_bytes,
                                     s_error_bound, s_w0, s_w1, s_w2, bw,
                                     3, s_ranking_csv, s_ranking_costs_csv,
                                     "nn-rl+exp50", s_write_count, &result);
                amrex::Print() << "    [ranking] T=" << s_write_count
                               << ": tau=" << result.mean_tau
                               << " regret=" << result.mean_regret << "x"
                               << " (" << result.profiling_ms << "ms)\n";
                break; /* one FAB is enough for ranking */
            }
            break; /* level 0 only */
        }
    }

    s_write_count++;

    /* Write metadata file with geometry info for downstream tools */
    if (amrex::ParallelDescriptor::IOProcessor()) {
        char meta_fname[512];
        std::snprintf(meta_fname, sizeof(meta_fname),
            "%s/gpucompress_metadata.txt", dirname.c_str());
        FILE* fp = std::fopen(meta_fname, "w");
        if (fp) {
            std::fprintf(fp, "time = %.17e\n", time);
            std::fprintf(fp, "iteration = %d\n", iteration[0]);
            std::fprintf(fp, "nlev = %d\n", nlev);
            std::fprintf(fp, "ncomp = %d\n",
                (nlev > 0) ? mf[0].nComp() : 0);
            std::fprintf(fp, "precision = %zu\n", elem_size);
            std::fprintf(fp, "algorithm = %s\n", algo_str.c_str());
            std::fprintf(fp, "error_bound = %.17e\n", error_bound);
            for (int lev = 0; lev < nlev; ++lev) {
                const auto& lo = geom[lev].ProbLo();
                const auto& hi = geom[lev].ProbHi();
                std::fprintf(fp, "lev%d_lo = %.17e %.17e %.17e\n",
                    lev, lo[0],
                    AMREX_SPACEDIM > 1 ? lo[1] : 0.0,
                    AMREX_SPACEDIM > 2 ? lo[2] : 0.0);
                std::fprintf(fp, "lev%d_hi = %.17e %.17e %.17e\n",
                    lev, hi[0],
                    AMREX_SPACEDIM > 1 ? hi[1] : 0.0,
                    AMREX_SPACEDIM > 2 ? hi[2] : 0.0);
            }
            for (int i = 0; i < static_cast<int>(varnames.size()); ++i) {
                std::fprintf(fp, "varname_%d = %s\n", i, varnames[i].c_str());
            }
            std::fclose(fp);
        }
    }

#else
    amrex::ignore_unused(varnames, mf, geom, iteration, time,
        nlev, prefix, file_min_digits, verbose);
    amrex::Abort("WarpX was not built with GPUCompress support.");
#endif
}
