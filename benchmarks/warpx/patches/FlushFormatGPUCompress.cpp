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
/* Per-chunk CSV logger state — persists across WriteToFile calls */
static FILE* s_tc_csv = nullptr;
static int   s_write_count = 0;
static bool  s_learning_initialized = false;

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

        /* Open per-chunk CSV if WARPX_LOG_DIR is set */
        const char* log_dir = std::getenv("WARPX_LOG_DIR");
        if (log_dir && log_dir[0] && !s_tc_csv && amrex::ParallelDescriptor::IOProcessor()) {
            mkdir(log_dir, 0755);
            char csv_path[700];
            snprintf(csv_path, sizeof(csv_path),
                     "%s/benchmark_warpx_timestep_chunks.csv", log_dir);
            s_tc_csv = fopen(csv_path, "w");
            if (s_tc_csv) {
                fprintf(s_tc_csv, "rank,phase,timestep,chunk,"
                        "predicted_ratio,actual_ratio,"
                        "predicted_comp_ms,actual_comp_ms_raw,"
                        "mape_ratio,mape_comp,"
                        "sgd_fired,exploration_triggered,"
                        "cost_model_error_pct,actual_cost,predicted_cost\n");
                amrex::Print() << "[GPUCompress] Chunk CSV: " << csv_path << "\n";
            }
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

                    /* Log per-chunk diagnostics to CSV */
                    if (s_tc_csv && amrex::ParallelDescriptor::IOProcessor()) {
                        int n_hist = gpucompress_get_chunk_history_count();
                        for (int ci = 0; ci < n_hist; ci++) {
                            gpucompress_chunk_diag_t dd;
                            if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;
                            double mr = 0, mc = 0;
                            if (dd.actual_ratio > 0)
                                mr = std::fabs(dd.predicted_ratio - dd.actual_ratio)
                                     / std::fabs(dd.actual_ratio) * 100.0;
                            if (dd.compression_ms > 0)
                                mc = std::fabs(dd.predicted_comp_time - dd.compression_ms)
                                     / std::fabs(dd.compression_ms) * 100.0;
                            char act_s[40], orig_s[40];
                            action_to_str_warpx(dd.nn_action, act_s, sizeof(act_s));
                            action_to_str_warpx(dd.nn_original_action, orig_s, sizeof(orig_s));
                            fprintf(s_tc_csv,
                                    "0,nn-rl+exp50,%d,%d,"
                                    "%.4f,%.4f,%.4f,%.4f,"
                                    "%.2f,%.2f,%d,%d,"
                                    "%.4f,%.4f,%.4f\n",
                                    s_write_count, ci,
                                    (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                    (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                                    mr, mc, dd.sgd_fired, dd.exploration_triggered,
                                    (double)dd.cost_model_error_pct,
                                    (double)dd.actual_cost, (double)dd.predicted_cost);
                        }
                        fflush(s_tc_csv);
                    }

                    /* Verify round-trip if requested */
                    if (verify && m_initialized) {
                        cudaDeviceSynchronize();
                        H5Dclose(dset);
                        H5Pclose(dcpl);
                        H5Sclose(space);
                        H5Fclose(fid);

                        /* Read back and compare bitwise */
                        hid_t rfid = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
                        if (rfid >= 0) {
                            hid_t rdset = H5Dopen2(rfid, "data", H5P_DEFAULT);
                            if (rdset >= 0) {
                                void* d_readback = nullptr;
                                cudaMalloc(&d_readback, fab_bytes);
                                H5Dread(rdset, h5type, H5S_ALL, H5S_ALL,
                                        H5P_DEFAULT, d_readback);
                                cudaDeviceSynchronize();

                                /* Host-side comparison */
                                std::vector<char> h_orig(fab_bytes);
                                std::vector<char> h_read(fab_bytes);
                                cudaMemcpy(h_orig.data(), d_ptr, fab_bytes,
                                           cudaMemcpyDeviceToHost);
                                cudaMemcpy(h_read.data(), d_readback, fab_bytes,
                                           cudaMemcpyDeviceToHost);
                                cudaFree(d_readback);

                                if (std::memcmp(h_orig.data(), h_read.data(), fab_bytes) != 0) {
                                    amrex::Print()
                                        << "[GPUCompress] VERIFY FAILED: "
                                        << vname << " lev" << lev
                                        << " fab" << fab_idx << "\n";
                                    amrex::Abort("GPUCompress lossless verification failed");
                                }
                                H5Dclose(rdset);
                            }
                            H5Fclose(rfid);
                        }
                        /* Skip normal cleanup since we did it above */
                        total_original += fab_bytes;
                        continue;
                    }

                    H5Dclose(dset);
                }

                H5Pclose(dcpl);
                H5Sclose(space);
                H5Fclose(fid);

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
