/**
 * @file vpic_benchmark_deck.cxx
 * @brief VPIC Benchmark Deck: No-Comp vs Oracle vs NN+SGD
 *
 * A real VPIC-Kokkos Harris sheet input deck that benchmarks GPU-resident
 * field data compression through the GPUCompress VOL connector under five
 * compression phases — matching the benchmark_vol_gpu / benchmark_grayscott_vol
 * structure.
 *
 * Pipeline per phase:
 *   VPIC simulation (GPU) → field_array->k_f_d → H5Dwrite(d_ptr) → VOL →
 *   GPU compress → HDF5 file → H5Dread(d_ptr) → GPU decompress → bitwise verify
 *
 * Phases (run sequentially after simulation reaches steady state):
 *   1. no-comp      : H5Dwrite via VOL (no filter, VOL-2 D→H fallback)
 *   2. oracle       : Exhaustive search (8 algos × 2 shuffle per chunk)
 *   3. nn           : H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *   4. nn-rl        : ALGO_AUTO + online SGD (MAPE≥20%, LR=0.4)
 *   5. nn-rl+exp50  : ALGO_AUTO + online SGD + Level-2 exploration (MAPE≥50%)
 *
 * BUILD:
 *   See benchmarks/vpic-kokkos/build_vpic_benchmark.sh
 *
 * RUN:
 *   export LD_LIBRARY_PATH=$GPU_DIR/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
 *   export GPUCOMPRESS_WEIGHTS=$GPU_DIR/neural_net/weights/model.nnwt
 *   mpirun -np 1 ./vpic_benchmark_deck.Linux
 *
 * OUTPUT:
 *   Console: per-phase ratio, write/read MB/s, verification, SGD/exploration stats
 *   Files:   /tmp/bm_vpic_*.h5 (temporary, removed after each phase)
 */

// ============================================================
// GPUCompress + HDF5 headers
// ============================================================
#include "gpucompress.h"
#include "gpucompress_vpic.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
#include "vpic_kokkos_bridge.hpp"

#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// ============================================================
// Constants
// ============================================================
#define REINFORCE_LR        0.4f
#define REINFORCE_MAPE      0.20f

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define N_PHASES 5

#define TMP_NOCOMP   "/tmp/bm_vpic_nocomp.h5"
#define TMP_NN       "/tmp/bm_vpic_nn.h5"
#define TMP_NN_RL    "/tmp/bm_vpic_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_vpic_nn_rlexp.h5"

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                sim_steps;         // Steps to run before benchmarking
    size_t             chunk_bytes;       // HDF5 chunk size in bytes
    gpucompress_vpic_t vpic_fields_h;     // Adapter handle for fields
    hid_t              vol_fapl;          // File access property list with VOL
    hid_t              vol_id;            // VOL connector ID
    int                gpucompress_ready; // 1 if init succeeded
    int                benchmark_done;    // 1 after benchmark has run

    // GPU buffers for benchmark
    float*             d_read;            // Read-back buffer
    unsigned long long* d_count;          // Mismatch counter
};

// ============================================================
// Helper: timing
// ============================================================
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================
// Helper: file size
// ============================================================
static size_t get_file_size(const char* path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

static void drop_pagecache(const char* path)
{
    int fd = open(path, O_RDWR);
    if (fd < 0) return;
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

// ============================================================
// Helper: pack double into two unsigned ints for HDF5 cd_values
// ============================================================
static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

// ============================================================
// Result struct
// ============================================================
struct PhaseResult {
    char   phase[20];
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    double read_mbps;
    unsigned long long mismatches;
    int    sgd_fires;
    int    explorations;
    int    n_chunks;
};

// ============================================================
// VOL FAPL
// ============================================================
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

// ============================================================
// DCPL builders (1D chunked)
// ============================================================
static hid_t make_dcpl_nocomp(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    return dcpl;
}

static hid_t make_dcpl_auto(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; // ALGO_AUTO
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

// ============================================================
// GPU-side byte-exact comparison (in gpu_compare.cu, avoids preprocessor)
// ============================================================
extern "C" unsigned long long gpu_compare(const float* d_a, const float* d_b,
                                          size_t n, unsigned long long* d_count);

// ============================================================
// Oracle: exhaustive best-config per chunk (1D flat data)
// Tries 8 algos × 2 shuffle = 16 configs per chunk.
// ============================================================
static int run_oracle_pass(const float* d_data, size_t n_floats,
                           size_t chunk_floats, int n_chunks,
                           PhaseResult* r)
{
    size_t chunk_bytes = chunk_floats * sizeof(float);
    size_t out_buf_size = chunk_bytes + 65536;
    size_t total_bytes = n_floats * sizeof(float);

    void* d_comp_buf = NULL;
    void* d_decomp_buf = NULL;
    if (cudaMalloc(&d_comp_buf, out_buf_size) != cudaSuccess) return 1;
    if (cudaMalloc(&d_decomp_buf, chunk_bytes) != cudaSuccess) {
        cudaFree(d_comp_buf); return 1;
    }

    static const char* ALGO_NAMES[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    size_t total_original = 0, total_best_compressed = 0;
    double total_best_comp_ms = 0;
    double total_best_decomp_ms = 0;

    for (int c = 0; c < n_chunks; c++) {
        size_t offset = (size_t)c * chunk_floats;
        size_t this_floats = chunk_floats;
        if (offset + this_floats > n_floats)
            this_floats = n_floats - offset;
        size_t this_bytes = this_floats * sizeof(float);
        const float* d_chunk = d_data + offset;

        double best_ratio = 0;
        size_t best_compressed = this_bytes;
        double best_comp_ms = 0;
        double best_throughput = 0;
        int best_algo_enum = 1;
        int best_shuffle = 0;
        const char* best_algo = "none";

        for (int algo_enum = 1; algo_enum <= 8; algo_enum++) {
            for (int shuffle = 0; shuffle <= 1; shuffle++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)algo_enum;
                cfg.preprocessing = 0;
                if (shuffle) cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                cfg.error_bound = 0.0;

                size_t out_size = out_buf_size;
                gpucompress_stats_t stats;
                gpucompress_error_t err = gpucompress_compress_gpu(
                    d_chunk, this_bytes,
                    d_comp_buf, &out_size, &cfg, &stats, NULL);

                double ratio = (err == GPUCOMPRESS_SUCCESS)
                    ? stats.compression_ratio : 0;

                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_compressed = stats.compressed_size;
                    best_comp_ms = stats.actual_comp_time_ms;
                    best_throughput = stats.throughput_mbps;
                    best_algo_enum = algo_enum;
                    best_shuffle = shuffle;
                    best_algo = ALGO_NAMES[algo_enum];
                }
            }
        }

        /* Re-compress with best config for decompression benchmark */
        gpucompress_config_t best_cfg = gpucompress_default_config();
        best_cfg.algorithm = (gpucompress_algorithm_t)best_algo_enum;
        best_cfg.preprocessing = 0;
        if (best_shuffle) best_cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
        best_cfg.error_bound = 0.0;

        size_t comp_size = out_buf_size;
        gpucompress_stats_t best_stats;
        gpucompress_compress_gpu(d_chunk, this_bytes,
                                d_comp_buf, &comp_size, &best_cfg, &best_stats, NULL);

        /* Decompress and measure time */
        size_t decomp_size = chunk_bytes;
        cudaDeviceSynchronize();
        double dt0 = now_ms();
        gpucompress_decompress_gpu(d_comp_buf, comp_size,
                                   d_decomp_buf, &decomp_size, NULL);
        cudaDeviceSynchronize();
        double decomp_ms = now_ms() - dt0;

        printf("  chunk %3d/%d: best=%s%s  ratio=%.2fx  comp=%.1f MiB/s (%.2f ms)  decomp=%.1f MiB/s (%.2f ms)\n",
               c + 1, n_chunks, best_algo,
               best_shuffle ? "+shuf" : "      ",
               best_ratio, best_throughput, best_comp_ms,
               decomp_ms > 0 ? ((double)this_bytes / (1 << 20)) / (decomp_ms / 1000.0) : 0,
               decomp_ms);

        total_original += this_bytes;
        total_best_compressed += best_compressed;
        total_best_comp_ms += best_comp_ms;
        total_best_decomp_ms += decomp_ms;
    }

    memset(r, 0, sizeof(*r));
    snprintf(r->phase, sizeof(r->phase), "oracle");
    r->orig_bytes = total_bytes;
    r->file_bytes = (total_best_compressed > 0) ? total_best_compressed : total_bytes;
    r->ratio = (total_best_compressed > 0)
        ? (double)total_original / (double)total_best_compressed : 1.0;
    r->write_ms = total_best_comp_ms;
    r->read_ms  = total_best_decomp_ms;
    r->write_mbps = (total_best_comp_ms > 0)
        ? ((double)total_original / (1 << 20)) / (total_best_comp_ms / 1000.0) : 0;
    r->read_mbps = (total_best_decomp_ms > 0)
        ? ((double)total_original / (1 << 20)) / (total_best_decomp_ms / 1000.0) : 0;
    r->mismatches = 0;
    r->n_chunks = n_chunks;

    printf("  [oracle] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "comp=%.0f ms  decomp=%.0f ms\n",
           r->ratio, r->write_mbps, r->read_mbps,
           total_best_comp_ms, total_best_decomp_ms);

    cudaFree(d_comp_buf);
    cudaFree(d_decomp_buf);
    return 0;
}

// ============================================================
// Run one benchmark phase: write → read → verify
// ============================================================
static int run_phase(const char* phase_name, const char* tmp_file,
                     float* d_data, float* d_read,
                     unsigned long long* d_count,
                     size_t n_floats, int n_chunks, hid_t dcpl,
                     PhaseResult* r)
{
    size_t total_bytes = n_floats * sizeof(float);
    hsize_t dims[1] = { (hsize_t)n_floats };

    // VOL write
    printf("  [%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file); return 1; }

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "fields", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0   = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    double t1   = now_ms();
    printf("%.0f ms\n", t1 - t0);

    if (wret < 0) { fprintf(stderr, "  [%s] H5Dwrite failed\n", phase_name); return 1; }

    drop_pagecache(tmp_file);

    // VOL read
    printf("  [%s] H5Dread (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "fields", H5P_DEFAULT);

    double t2   = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    double t3   = now_ms();
    H5Dclose(dset);
    H5Fclose(file);
    printf("%.0f ms\n", t3 - t2);

    if (rret < 0) { fprintf(stderr, "  [%s] H5Dread failed\n", phase_name); return 1; }

    // GPU bitwise verify
    unsigned long long mm = gpu_compare(d_data, d_read, n_floats, d_count);

    // Collect per-chunk diagnostics
    int sgd_fires    = 0;
    int explorations = 0;
    int n_hist       = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            sgd_fires    += d.sgd_fired;
            explorations += d.exploration_triggered;
        }
    }

    size_t fbytes = get_file_size(tmp_file);

    r->write_ms     = t1 - t0;
    r->read_ms      = t3 - t2;
    r->file_bytes   = fbytes;
    r->orig_bytes   = total_bytes;
    r->ratio        = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps   = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps    = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches   = mm;
    r->sgd_fires    = sgd_fires;
    r->explorations = explorations;
    r->n_chunks     = n_chunks;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    printf("  [%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  sgd=%d expl=%d/%d  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), sgd_fires, explorations, n_hist, mm);

    remove(tmp_file);
    return (mm == 0) ? 0 : 1;
}

// ============================================================
// Initialization: Harris sheet + GPUCompress + VOL setup
// ============================================================
begin_initialization {
    // ---- Physics (Harris sheet reconnection) ----
    double L    = 1;
    double ec   = 1;
    double me   = 1;
    double c    = 1;
    double eps0 = 1;

    double mi_me   = 25;
    double rhoi_L  = 1;
    double Ti_Te   = 1;
    double wpe_wce = 3;

    // Grid: 320x320x320 cells, 2 particles/cell
    // 320^3 + ghost = 322^3 voxels × 16 vars × 4B ≈ 2.03 GB field data
    // Total GPU memory ~8 GB (fields + grid neighbor + particles)
    double Lx   = 80*L;
    double Ly   = 80*L;
    double Lz   = 80*L;
    double nx   = 128;
    double ny   = 128;
    double nz   = 128;
    double nppc = 2;

    double damp      = 0.001;
    double cfl_req   = 0.99;
    double wpedt_max = 0.36;

    double mi   = me*mi_me;
    double kTe  = me*c*c/(2*wpe_wce*wpe_wce*(1+Ti_Te));
    double kTi  = kTe*Ti_Te;
    double vthi = sqrt(2*kTi/mi);
    double wci  = vthi/(rhoi_L*L);
    double wce  = wci*mi_me;
    double wpe  = wce*wpe_wce;
    double vdre = c*c*wce/(wpe*wpe*L*(1+Ti_Te));
    double vdri = -Ti_Te*vdre;
    double b0   = me*wce/ec;
    double n0   = me*eps0*wpe*wpe/(ec*ec);
    double Npe  = 2*n0*Ly*Lz*L*tanh(0.5*Lx/L);
    double Npi  = Npe;
    double Ne   = 0.5*nppc*nx*ny*nz;
    Ne = trunc_granular(Ne, nproc());
    double Ni   = Ne;
    double we   = Npe/Ne;
    double wi   = Npi/Ni;
    double gdri = 1/sqrt(1 - vdri*vdri/(c*c));
    double gdre = 1/sqrt(1 - vdre*vdre/(c*c));
    double udri = vdri*gdri;
    double udre = vdre*gdre;
    double uthi = sqrt(kTi/mi)/c;
    double uthe = sqrt(kTe/me)/c;

    double dg = courant_length(Lx, Ly, Lz, nx, ny, nz);
    double dt = cfl_req*dg/c;
    if (wpe*dt > wpedt_max) dt = wpedt_max/wpe;

    // Run 100 steps to develop physics, then benchmark on step 100
    num_step        = 101;   // 100 warmup + 1 diagnostic trigger
    status_interval = 50;
    clean_div_e_interval = 10;
    clean_div_b_interval = 10;

    global->sim_steps     = 100;   // Step at which benchmark triggers
    global->chunk_bytes   = 2 * 1024 * 1024;  // 2 MB chunks
    global->benchmark_done = 0;

    // Grid setup
    define_units(c, eps0);
    define_timestep(dt);
    define_periodic_grid(-0.5*Lx, 0, 0,
                          0.5*Lx, Ly, Lz,
                          nx, ny, nz,
                          1, nproc(), 1);

    set_domain_field_bc(BOUNDARY(-1,0,0), pec_fields);
    set_domain_field_bc(BOUNDARY( 1,0,0), pec_fields);
    set_domain_particle_bc(BOUNDARY(-1,0,0), reflect_particles);
    set_domain_particle_bc(BOUNDARY( 1,0,0), reflect_particles);

    define_material("vacuum", 1);
    define_field_array(NULL, damp);

    species_t* ion      = define_species("ion",       ec, mi, 1.5*Ni/nproc(), -1, 5, 1);
    species_t* electron = define_species("electron", -ec, me, 1.5*Ne/nproc(), -1, 5, 1);

    // Load fields (Harris current sheet)
    set_region_field(everywhere,
                     0, 0, 0,
                     0, 0, b0*tanh(x/L));

    // Load particles (drifting Maxwellians)
    double ymin = rank()*Ly/nproc();
    double ymax = (rank()+1)*Ly/nproc();

    repeat(Ni/nproc()) {
        double px, py, pz, ux, uy, uz, d0;
        do { px = L*atanh(uniform(rng(0), -1, 1)); }
        while (px <= -0.5*Lx || px >= 0.5*Lx);
        py = uniform(rng(0), ymin, ymax);
        pz = uniform(rng(0), 0, Lz);

        ux = normal(rng(0), 0, uthi);
        uy = normal(rng(0), 0, uthi);
        uz = normal(rng(0), 0, uthi);
        d0 = gdri*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udri;
        uy = d0; uz = uz;
        inject_particle(ion, px, py, pz, ux, uy, uz, wi, 0, 0);

        ux = normal(rng(0), 0, uthe);
        uy = normal(rng(0), 0, uthe);
        uz = normal(rng(0), 0, uthe);
        d0 = gdre*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udre;
        uy = d0; uz = uz;
        inject_particle(electron, px, py, pz, ux, uy, uz, we, 0, 0);
    }

    // ---- GPUCompress + HDF5 VOL initialization ----
    global->gpucompress_ready = 0;
    global->d_read  = NULL;
    global->d_count = NULL;

    const char* weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    gpucompress_error_t gerr = gpucompress_init(weights_path);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        sim_log("FATAL: gpucompress_init failed (" << gerr << ")");
        return;
    }

    if (weights_path && !gpucompress_nn_is_loaded()) {
        sim_log("WARNING: NN weights not loaded from " << weights_path);
    }

    H5Z_gpucompress_register();
    global->vol_id = H5VL_gpucompress_register();

    hid_t native_id = H5VLget_connector_id_by_name("native");
    global->vol_fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(global->vol_fapl, native_id, NULL);
    H5VLclose(native_id);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    H5VL_gpucompress_set_trace(0);
    global->gpucompress_ready = 1;

    // Create VPIC adapter handle for fields
    VpicSettings fs = vpic_default_settings();
    fs.data_type = VPIC_DATA_FIELDS;
    fs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_fields_h, &fs);

    size_t field_bytes = (size_t)grid->nv * 16 * sizeof(float);
    sim_log("=== VPIC Benchmark Deck: Harris Sheet Reconnection ===");
    sim_log("  Grid     : " << (int)nx << "x" << (int)ny << "x" << (int)nz
            << " = " << grid->nv << " voxels");
    sim_log("  Fields   : " << field_bytes / (1024*1024) << " MB (16 vars x "
            << grid->nv << " cells)");
    sim_log("  Chunks   : " << global->chunk_bytes / (1024*1024) << " MB each");
    sim_log("  Particles: " << nppc << " per cell");
    sim_log("  Warmup   : " << global->sim_steps << " steps");
    sim_log("  Weights  : " << (weights_path ? weights_path : "(none, fallback to LZ4)"));
    sim_log("  NN loaded: " << (gpucompress_nn_is_loaded() ? "yes" : "no"));

    // Allocate GPU buffers for read-back and mismatch counter
    cudaMalloc(&global->d_read,  field_bytes);
    cudaMalloc(&global->d_count, sizeof(unsigned long long));
};

// ============================================================
// Diagnostics: run benchmark after warmup steps
// ============================================================
begin_diagnostics {
    if (global->benchmark_done) return;
    if (step() < global->sim_steps) return;
    if (!global->gpucompress_ready) return;

    global->benchmark_done = 1;

    // Attach GPU-resident field data
    vpic_attach_fields(global->vpic_fields_h, field_array->k_f_d);

    float*  d_fields = NULL;
    size_t  nbytes_f = 0;
    gpucompress_vpic_get_device_ptrs(global->vpic_fields_h,
                                     &d_fields, NULL, &nbytes_f, NULL);
    size_t n_floats    = nbytes_f / sizeof(float);
    size_t chunk_floats = global->chunk_bytes / sizeof(float);
    int    n_chunks    = (int)((n_floats + chunk_floats - 1) / chunk_floats);

    sim_log("");
    sim_log("╔═══════════════════════════════════════════════════════════════════════════╗");
    sim_log("║  VPIC Benchmark: No-Comp vs Oracle vs NN+SGD (Real Harris Sheet Data)   ║");
    sim_log("╚═══════════════════════════════════════════════════════════════════════════╝");
    sim_log("");
    sim_log("  Step " << step() << ": field data on GPU, " << nbytes_f / (1024*1024)
            << " MB, " << n_chunks << " chunks of "
            << global->chunk_bytes / (1024*1024) << " MB");
    sim_log("");

    PhaseResult results[N_PHASES];
    int n_phases = 0;
    int any_fail = 0;

    // ── Phase 1: no-comp ──────────────────────────────────────────
    sim_log("── Phase 1/5: no-comp (GPU→Host→HDF5, VOL-2 fallback) ────────");
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_nocomp((hsize_t)chunk_floats);
        int rc = run_phase("no-comp", TMP_NOCOMP,
                           d_fields, global->d_read, global->d_count,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        if (rc) any_fail = 1;
        n_phases++;
    }

    // ── Phase 2: oracle (exhaustive best-config per chunk) ────────
    sim_log("── Phase 2/5: oracle (exhaustive 8 algos × 2 shuffle per chunk) ──");
    {
        int rc = run_oracle_pass(d_fields, n_floats, chunk_floats, n_chunks,
                                 &results[n_phases]);
        if (rc) any_fail = 1;
        n_phases++;
    }

    // ── Phase 3: nn (inference-only) ──────────────────────────────
    sim_log("── Phase 3/5: nn (VOL, ALGO_AUTO, inference-only) ───────────");
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats);
        int rc = run_phase("nn", TMP_NN,
                           d_fields, global->d_read, global->d_count,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        if (rc) any_fail = 1;
        n_phases++;
    }

    // ── Phase 4: nn-rl (SGD, no exploration) ──────────────────────
    sim_log("── Phase 4/5: nn-rl (ALGO_AUTO + SGD, MAPE>=20%, LR=0.4) ───");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats);
        int rc = run_phase("nn-rl", TMP_NN_RL,
                           d_fields, global->d_read, global->d_count,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        n_phases++;
    }

    // ── Phase 5: nn-rl+exp50 (SGD + exploration) ──────────────────
    sim_log("── Phase 5/5: nn-rl+exp50 (ALGO_AUTO + SGD + expl@MAPE>=50%) ");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.50);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats);
        int rc = run_phase("nn-rl+exp50", TMP_NN_RLEXP,
                           d_fields, global->d_read, global->d_count,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        n_phases++;
    }

    // ── Summary table ─────────────────────────────────────────────
    sim_log("");
    printf("\n╔══════════════╦══════════╦══════════╦═══════╦══════════╦═════════════╗\n");
    printf("║  Phase       ║ Write    ║ Read     ║ Ratio ║ File MiB ║ Verify      ║\n");
    printf("║              ║ (MiB/s)  ║ (MiB/s)  ║       ║          ║             ║\n");
    printf("╠══════════════╬══════════╬══════════╬═══════╬══════════╬═════════════╣\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult* r = &results[i];
        const char* verdict = (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %8.0f ║ %8.0f ║ %5.2fx ║ %8.0f ║ %-11s ║\n",
               r->phase, r->write_mbps, r->read_mbps,
               r->ratio, (double)r->file_bytes / (1 << 20), verdict);
    }
    printf("╚══════════════╩══════════╩══════════╩═══════╩══════════╩═════════════╝\n");

    // NN SGD/exploration detail
    for (int i = 0; i < n_phases; i++) {
        if (strncmp(results[i].phase, "nn", 2) == 0 && results[i].n_chunks > 0) {
            printf("\n  %-14s SGD fired: %d/%d chunks  Explorations: %d/%d chunks\n",
                   results[i].phase,
                   results[i].sgd_fires,   results[i].n_chunks,
                   results[i].explorations, results[i].n_chunks);
        }
    }

    printf("\n=== VPIC Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");

    // Write CSV results
    {
        const char* csv_dir  = "benchmark_vpic_deck_results";
        const char* csv_path = "benchmark_vpic_deck_results/benchmark_vpic_deck.csv";
        mkdir(csv_dir, 0755);
        FILE* csv = fopen(csv_path, "w");
        if (csv) {
            fprintf(csv, "source,phase,write_ms,read_ms,file_mib,orig_mib,ratio,"
                         "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks\n");
            for (int i = 0; i < n_phases; i++) {
                PhaseResult* r = &results[i];
                fprintf(csv, "vpic,%s,%.2f,%.2f,%.2f,%.2f,%.4f,"
                             "%.1f,%.1f,%llu,%d,%d,%d\n",
                        r->phase, r->write_ms, r->read_ms,
                        (double)r->file_bytes / (1 << 20),
                        (double)r->orig_bytes / (1 << 20), r->ratio,
                        r->write_mbps, r->read_mbps,
                        r->mismatches, r->sgd_fires, r->explorations, r->n_chunks);
            }
            fclose(csv);
            printf("CSV written to: %s\n", csv_path);
        }
    }

    // Cleanup GPU benchmark buffers
    cudaFree(global->d_read);
    cudaFree(global->d_count);
    global->d_read  = NULL;
    global->d_count = NULL;

    gpucompress_vpic_destroy(global->vpic_fields_h);
    H5Pclose(global->vol_fapl);
    H5VLclose(global->vol_id);
    gpucompress_cleanup();
};

begin_particle_injection {
    // No injection
}

begin_current_injection {
    // No injection
}

begin_field_injection {
    // No injection
}

begin_particle_collisions {
    // No collisions
}