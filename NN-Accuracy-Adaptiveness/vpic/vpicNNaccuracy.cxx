/**
 * @file benchmark_vpic_sim.cxx
 * @brief Simulation-Based VPIC Benchmark: Per-Timestep Compression Tracking
 *
 * Runs an actual VPIC-Kokkos Harris sheet simulation and tracks compression
 * performance at each diagnostic timestep as the plasma physics evolves.
 *
 * NN Baseline and NN+SGD compress every simulation step (realistic I/O).
 * Exhaustive runs at the diagnostic interval (+ first and last step).
 *
 * Strategies:
 *   0. Exhaustive   : exhaustive best-config per chunk (at diag interval)
 *   1. NN Baseline  : ALGO_AUTO inference-only (every step)
 *   2. NN + SGD     : ALGO_AUTO with online learning (every step, persistent)
 *
 * PHASE SELECTION:
 *   GPUCOMPRESS_PHASES env var — comma-separated list of phases to run.
 *   Valid names: exhaustive, nn, nn-sgd
 *   Default: all phases.  Example: GPUCOMPRESS_PHASES=nn,nn-sgd
 *
 * OUTPUT:
 *   benchmarks/vpic/sim_timestep_metrics.csv   — per-timestep ratio/MAPE
 *   benchmarks/vpic/sim_chunk_metrics.csv      — per-chunk per-timestep detail
 *   benchmarks/vpic/sim_upper_bound.csv        — exhaustive search results
 *   benchmarks/vpic/sim_aggregate.csv          — aggregate summary
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
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// ============================================================
// Constants (defaults, overridable via environment variables)
//
//   VPIC_NX                    — grid side length         (default: 128)
//   GPUCOMPRESS_SGD_LR         — SGD learning rate       (default: 0.5)
//   GPUCOMPRESS_SGD_MAPE       — MAPE threshold          (default: 0.25)
//   GPUCOMPRESS_DIAG_INTERVAL  — exhaustive + logging every N steps (default: 20)
//   GPUCOMPRESS_NUM_STEPS      — total simulation steps  (default: 1000)
//   GPUCOMPRESS_CHUNK_MB       — chunk size in MB        (default: 8)
// ============================================================
#define DEFAULT_SGD_LR          0.05f
#define DEFAULT_SGD_MAPE_THRESH 0.25f
#define DEFAULT_DIAG_INTERVAL   20
#define DEFAULT_NUM_STEPS       1000
#define DEFAULT_CHUNK_MB        8

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

// Phase bitmask constants
enum { P_EXHAUSTIVE = 1, P_NN = 2, P_NN_SGD = 4 };

#define TMP_NN       "/tmp/bm_vpic_sim_nn.h5"
#define TMP_SGD      "/tmp/bm_vpic_sim_sgd.h5"

#define OUT_DIR       "NN-Accuracy-Adaptiveness/vpic/results"
#define OUT_TIMESTEP  OUT_DIR "/sim_timestep_metrics.csv"
#define OUT_CHUNKS    OUT_DIR "/sim_chunk_metrics.csv"
#define OUT_UB_CSV    OUT_DIR "/sim_upper_bound.csv"
#define OUT_AGG_CSV   OUT_DIR "/sim_aggregate.csv"

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                gpucompress_ready;
    int                diag_count;
    gpucompress_vpic_t vpic_fields_h;
    size_t             chunk_bytes;

    /* Host buffers for CPU-side bitwise verification */
    float*             h_orig;
    float*             h_read;
    size_t             h_buf_size;

    /* GPU readback buffer */
    float*             d_read;

    /* Runtime-configurable parameters */
    float              sgd_lr;
    float              sgd_mape_thresh;
    int                diag_interval;
    int                num_steps;
    unsigned int       phase_mask;
};

// ============================================================
// Timing
// ============================================================
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================
// File helpers
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
// CPU-side bitwise comparison (avoids CUDA kernel in .cxx deck)
// ============================================================
static unsigned long long cpu_compare(const float* a, const float* b, size_t n)
{
    unsigned long long mm = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned int ua, ub;
        memcpy(&ua, &a[i], sizeof(unsigned int));
        memcpy(&ub, &b[i], sizeof(unsigned int));
        if (ua != ub) mm++;
    }
    return mm;
}

// ============================================================
// NN action → human-readable string
// ============================================================
static const char* ACTION_ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_str(int action, char *buf, size_t bufsz)
{
    if (action < 0 || action > 31) {
        snprintf(buf, bufsz, "act%d", action);
        return;
    }
    int algo    = action % 8;
    int quant   = (action / 8) % 2;
    int shuffle = (action / 16) % 2;
    int n = snprintf(buf, bufsz, "%s", ACTION_ALGO_NAMES[algo]);
    if (shuffle) n += snprintf(buf + n, bufsz - n, "+shuf");
    if (quant)   snprintf(buf + n, bufsz - n, "+quant");
}

// ============================================================
// HDF5 helpers
// ============================================================
static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}


static hid_t make_dcpl_auto(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

// ============================================================
// Per-step result
// ============================================================
struct StepResult {
    int    timestep;
    char   phase[32];
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    unsigned long long mismatches;
    int    sgd_fires;
    int    n_chunks;
    double ratio_min;
    double ratio_max;
    double ratio_stddev;
    double mean_mape;

    /* Timing breakdown */
    double total_nn_inference_ms;
    double total_sgd_ms;
    double total_compression_ms;
};

// ============================================================
// Run one compression pass: write -> read -> verify
// ============================================================
static int run_pass(const char* phase_name, const char* tmp_file,
                    float* d_data, float* d_read,
                    float* h_orig, float* h_read,
                    size_t n_floats, int n_chunks, hid_t dcpl,
                    StepResult* r, int timestep)
{
    size_t total_bytes = n_floats * sizeof(float);
    hsize_t dims[1] = { (hsize_t)n_floats };

    remove(tmp_file);
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) return 1;

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

    if (wret < 0) return 1;
    drop_pagecache(tmp_file);

    /* Read back to GPU */
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

    if (rret < 0) return 1;

    /* CPU-side bitwise verification */
    cudaMemcpy(h_orig, d_data, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read, d_read, total_bytes, cudaMemcpyDeviceToHost);
    unsigned long long mm = cpu_compare(h_orig, h_read, n_floats);

    /* Collect per-chunk diagnostics */
    int sgd_fires = 0;
    double rmin = 1e30, rmax = 0, rsum = 0, rsum2 = 0;
    double mape_sum = 0;
    int mape_count = 0, n_valid = 0;
    double sum_nn = 0, sum_sgd = 0, sum_comp = 0;

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            sgd_fires += d.sgd_fired;
            sum_nn   += d.nn_inference_ms;
            sum_sgd  += d.sgd_update_ms;
            sum_comp += d.compression_ms;
            if (d.actual_ratio > 0) {
                if (d.actual_ratio < rmin) rmin = d.actual_ratio;
                if (d.actual_ratio > rmax) rmax = d.actual_ratio;
                rsum  += d.actual_ratio;
                rsum2 += (double)d.actual_ratio * d.actual_ratio;
                n_valid++;
            }
            if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
                mape_sum += fabs((double)d.predicted_ratio - d.actual_ratio)
                            / d.actual_ratio;
                mape_count++;
            }
        }
    }

    size_t fbytes = get_file_size(tmp_file);

    r->timestep    = timestep;
    r->write_ms    = t1 - t0;
    r->read_ms     = t3 - t2;
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps  = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->mismatches  = mm;
    r->sgd_fires   = sgd_fires;
    r->n_chunks    = n_chunks;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    r->ratio_min = (n_valid > 0) ? rmin : 0;
    r->ratio_max = (n_valid > 0) ? rmax : 0;
    if (n_valid > 1) {
        double mean = rsum / n_valid;
        r->ratio_stddev = sqrt((rsum2 / n_valid) - mean * mean);
    } else {
        r->ratio_stddev = 0;
    }
    r->mean_mape = (mape_count > 0)
        ? (mape_sum / mape_count) * 100.0 : 0;
    r->total_nn_inference_ms = sum_nn;
    r->total_sgd_ms          = sum_sgd;
    r->total_compression_ms  = sum_comp;

    remove(tmp_file);
    return (mm == 0) ? 0 : 1;
}

// ============================================================
// Write per-chunk detail CSV
// ============================================================
static void write_chunk_detail(int timestep, const char* phase_name, int n_chunks)
{
    FILE* f = fopen(OUT_CHUNKS, "a");
    if (!f) return;

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist && i < n_chunks; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            char action_str[40];
            action_to_str(d.nn_action, action_str, sizeof(action_str));
            fprintf(f, "%d,%s,%d,%s,%d,"
                       "%.3f,%.3f,%.3f,%.3f,"
                       "%.4f,%.4f\n",
                    timestep, phase_name, i,
                    action_str, d.sgd_fired,
                    d.nn_inference_ms, d.preprocessing_ms,
                    d.compression_ms, d.sgd_update_ms,
                    d.actual_ratio, d.predicted_ratio);
        }
    }
    fclose(f);
}

// ============================================================
// Oracle: exhaustive best-config per chunk
// ============================================================
static double run_oracle_pass(const float* d_data, size_t n_floats,
                              size_t chunk_floats, int n_chunks,
                              int timestep, FILE* f_ub)
{
    static const char* ALGO_NAMES[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    size_t max_chunk_bytes = chunk_floats * sizeof(float);
    size_t out_buf_size = max_chunk_bytes + 65536;

    float* d_chunk_buf = NULL;
    void*  d_output    = NULL;
    if (cudaMalloc(&d_chunk_buf, max_chunk_bytes) != cudaSuccess) return 1.0;
    if (cudaMalloc(&d_output, out_buf_size) != cudaSuccess) {
        cudaFree(d_chunk_buf); return 1.0;
    }

    size_t total_original = 0, total_best_compressed = 0;

    for (int c = 0; c < n_chunks; c++) {
        size_t offset = (size_t)c * chunk_floats;
        size_t this_chunk_floats = chunk_floats;
        if (offset + this_chunk_floats > n_floats)
            this_chunk_floats = n_floats - offset;
        size_t chunk_bytes = this_chunk_floats * sizeof(float);

        cudaMemcpy(d_chunk_buf, d_data + offset, chunk_bytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        double best_ratio = 0;
        size_t best_compressed = chunk_bytes;

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
                    d_chunk_buf, chunk_bytes,
                    d_output, &out_size, &cfg, &stats, NULL);

                double ratio = (err == GPUCOMPRESS_SUCCESS)
                    ? stats.compression_ratio : 0;

                if (f_ub) {
                    fprintf(f_ub, "%d,%d,%s,%d,0,0,%.4f,%zu,%zu,%s\n",
                            timestep, c, ALGO_NAMES[algo_enum],
                            shuffle ? 4 : 0, ratio,
                            (err == GPUCOMPRESS_SUCCESS) ? stats.compressed_size : 0,
                            chunk_bytes,
                            (err == GPUCOMPRESS_SUCCESS) ? "ok" : "fail");
                }

                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_compressed = stats.compressed_size;
                }
            }
        }

        total_original += chunk_bytes;
        total_best_compressed += best_compressed;
    }

    cudaFree(d_chunk_buf);
    cudaFree(d_output);

    return (total_best_compressed > 0)
        ? (double)total_original / (double)total_best_compressed : 1.0;
}

// ============================================================
// Initialization: Harris sheet + GPUCompress
// ============================================================
begin_initialization {
    // ---- Harris sheet physics ----
    double L    = 1;
    double ec   = 1;
    double me   = 1;
    double c    = 1;
    double eps0 = 1;

    double mi_me   = 25;
    double rhoi_L  = 1;
    double Ti_Te   = 1;
    double wpe_wce = 3;

    /* Grid size configurable via VPIC_NX env var (default 128)
     * Field data = (nx+2)^3 * 16 * 4 bytes
     *   64 → ~18 MB   128 → ~134 MB   160 → ~264 MB   256 → ~1.1 GB */
    const char* env_nx = getenv("VPIC_NX");
    int grid_n = env_nx ? atoi(env_nx) : 128;
    if (grid_n < 16) grid_n = 16;

    double Lx   = 32*L;
    double Ly   = 32*L;
    double Lz   = 32*L;
    double nx   = grid_n;
    double ny   = grid_n;
    double nz   = grid_n;
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

    // Read runtime parameters from environment
    const char* env_val;

    float sgd_lr = DEFAULT_SGD_LR;
    env_val = getenv("GPUCOMPRESS_SGD_LR");
    if (env_val) sgd_lr = (float)atof(env_val);

    float sgd_mape = DEFAULT_SGD_MAPE_THRESH;
    env_val = getenv("GPUCOMPRESS_SGD_MAPE");
    if (env_val) sgd_mape = (float)atof(env_val);

    int diag_interval = DEFAULT_DIAG_INTERVAL;
    env_val = getenv("GPUCOMPRESS_DIAG_INTERVAL");
    if (env_val) diag_interval = atoi(env_val);

    int total_steps = DEFAULT_NUM_STEPS;
    env_val = getenv("GPUCOMPRESS_NUM_STEPS");
    if (env_val) total_steps = atoi(env_val);

    int chunk_mb = DEFAULT_CHUNK_MB;
    env_val = getenv("GPUCOMPRESS_CHUNK_MB");
    if (env_val) chunk_mb = atoi(env_val);

    /* Parse GPUCOMPRESS_PHASES: comma-separated list (exhaustive,nn,nn-sgd) */
    unsigned int phase_mask = 0;
    env_val = getenv("GPUCOMPRESS_PHASES");
    if (env_val) {
        char buf[256];
        strncpy(buf, env_val, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char* tok = strtok(buf, ",");
        while (tok) {
            while (*tok == ' ') tok++;  /* trim leading spaces */
            if      (strcmp(tok, "exhaustive") == 0) phase_mask |= P_EXHAUSTIVE;
            else if (strcmp(tok, "nn") == 0)         phase_mask |= P_NN;
            else if (strcmp(tok, "nn-sgd") == 0)     phase_mask |= P_NN_SGD;
            else fprintf(stderr, "Warning: unknown phase '%s' (valid: exhaustive, nn, nn-sgd)\n", tok);
            tok = strtok(NULL, ",");
        }
    }
    if (phase_mask == 0) phase_mask = P_EXHAUSTIVE | P_NN | P_NN_SGD;

    global->sgd_lr          = sgd_lr;
    global->sgd_mape_thresh = sgd_mape;
    global->diag_interval   = diag_interval;
    global->num_steps       = total_steps;
    global->phase_mask      = phase_mask;

    num_step        = total_steps + 1;
    status_interval = 50;
    clean_div_e_interval = 10;
    clean_div_b_interval = 10;

    global->chunk_bytes = (size_t)chunk_mb * 1024 * 1024;
    global->gpucompress_ready = 0;
    global->diag_count = 0;
    global->d_read  = NULL;
    global->h_orig  = NULL;
    global->h_read  = NULL;
    global->h_buf_size = 0;

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

    set_region_field(everywhere, 0, 0, 0, 0, 0, b0*tanh(x/L));

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
        uy = d0;
        inject_particle(ion, px, py, pz, ux, uy, uz, wi, 0, 0);

        ux = normal(rng(0), 0, uthe);
        uy = normal(rng(0), 0, uthe);
        uz = normal(rng(0), 0, uthe);
        d0 = gdre*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udre;
        uy = d0;
        inject_particle(electron, px, py, pz, ux, uy, uz, we, 0, 0);
    }

    // ---- GPUCompress initialization ----
    const char* weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    gpucompress_error_t gerr = gpucompress_init(weights_path);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        sim_log("FATAL: gpucompress_init failed (" << gerr << ")");
        return;
    }

    H5Z_gpucompress_register();
    H5VL_gpucompress_register();
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    H5VL_gpucompress_set_trace(0);

    VpicSettings fs = vpic_default_settings();
    fs.data_type = VPIC_DATA_FIELDS;
    fs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_fields_h, &fs);

    size_t field_bytes = (size_t)grid->nv * 16 * sizeof(float);
    cudaMalloc(&global->d_read, field_bytes);
    global->h_orig = (float*)malloc(field_bytes);
    global->h_read = (float*)malloc(field_bytes);
    global->h_buf_size = field_bytes;

    global->gpucompress_ready = 1;

    // Write CSV headers
    mkdir(OUT_DIR, 0755);
    {
        FILE* f = fopen(OUT_TIMESTEP, "w");
        if (f) {
            fprintf(f, "timestep,phase,ratio,write_ms,read_ms,write_mibps,"
                       "file_mib,orig_mib,sgd_fires,n_chunks,"
                       "ratio_min,ratio_max,ratio_stddev,mean_mape,mismatches\n");
            fclose(f);
        }
    }
    {
        FILE* f = fopen(OUT_CHUNKS, "w");
        if (f) {
            fprintf(f, "timestep,phase,chunk,nn_action,sgd_fired,"
                       "nn_inference_ms,preprocessing_ms,compression_ms,"
                       "sgd_update_ms,actual_ratio,predicted_ratio\n");
            fclose(f);
        }
    }
    {
        FILE* f = fopen(OUT_UB_CSV, "w");
        if (f) {
            fprintf(f, "timestep,chunk,algorithm,shuffle,quant,error_bound,"
                       "ratio,compressed_bytes,original_bytes,status\n");
            fclose(f);
        }
    }

    sim_log("╔═══════════════════════════════════════════════════════════════════════════╗");
    sim_log("║  Simulation-Based VPIC Benchmark: Exhaustive + NN + SGD Per-Timestep     ║");
    sim_log("╚═══════════════════════════════════════════════════════════════════════════╝");
    sim_log("");
    sim_log("  Grid        : " << grid_n << "x" << grid_n << "x" << grid_n
            << " = " << grid->nv << " voxels (VPIC_NX=" << grid_n << ")");
    sim_log("  Fields      : " << field_bytes / (1024*1024) << " MB");
    sim_log("  Chunks      : " << global->chunk_bytes / (1024*1024) << " MB each ("
            << (field_bytes / global->chunk_bytes) << " chunks)");
    sim_log("  Diag every  : " << global->diag_interval << " steps (exhaustive + logging)");
    sim_log("  Total steps : " << global->num_steps);
    sim_log("  SGD         : lr=" << global->sgd_lr << " mt=" << global->sgd_mape_thresh);
    sim_log("  Weights     : " << (weights_path ? weights_path : "(fallback)"));
    sim_log("  NN loaded   : " << (gpucompress_nn_is_loaded() ? "yes" : "no"));
    sim_log("");
};

// ============================================================
// Diagnostics: NN passes every step, oracle at interval
//
// - NN Baseline and NN+SGD compress every step (realistic I/O)
// - Exhaustive runs at diag_interval steps (+ first & last)
// ============================================================
begin_diagnostics {
    if (!global->gpucompress_ready) return;
    if (step() == 0) return;

    int ts = step();
    int is_last = (ts == global->num_steps);
    int is_diag = (ts == 1) || is_last
                  || (ts % global->diag_interval == 0);

    /* Attach live GPU field data */
    vpic_attach_fields(global->vpic_fields_h, field_array->k_f_d);

    float*  d_fields = NULL;
    size_t  nbytes_f = 0;
    gpucompress_vpic_get_device_ptrs(global->vpic_fields_h,
                                     &d_fields, NULL, &nbytes_f, NULL);
    size_t n_floats     = nbytes_f / sizeof(float);
    size_t chunk_floats = global->chunk_bytes / sizeof(float);
    int    n_chunks     = (int)((n_floats + chunk_floats - 1) / chunk_floats);

    StepResult results[3];
    int n_results = 0;
    double oracle_ratio = 0;

    if (is_diag) {
        global->diag_count++;
        sim_log("── Step " << ts << " (diag " << global->diag_count
                << ") ────────");
    }

    /* ── Exhaustive: only at diag interval, first step, and last step ── */
    if ((global->phase_mask & P_EXHAUSTIVE) && is_diag) {
        FILE* f_ub = fopen(OUT_UB_CSV, "a");
        oracle_ratio = run_oracle_pass(d_fields, n_floats,
                                       chunk_floats, n_chunks,
                                       ts, f_ub);
        if (f_ub) fclose(f_ub);

        StepResult r;
        memset(&r, 0, sizeof(r));
        r.timestep = ts;
        snprintf(r.phase, sizeof(r.phase), "exhaustive");
        r.ratio = oracle_ratio;
        r.n_chunks = n_chunks;
        r.orig_bytes = n_floats * sizeof(float);
        results[n_results++] = r;

        if (is_diag) {
            printf("  exhaustive: ratio=%.2fx\n", oracle_ratio);
        }
    }

    /* ── NN Baseline: every step (fresh weights) ── */
    if (global->phase_mask & P_NN) {
        const char* wp = getenv("GPUCOMPRESS_WEIGHTS");
        if (wp) gpucompress_reload_nn(wp);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);

        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats);
        StepResult r;
        memset(&r, 0, sizeof(r));
        run_pass("nn_baseline", TMP_NN,
                 d_fields, global->d_read,
                 global->h_orig, global->h_read,
                 n_floats, n_chunks, dcpl, &r, ts);
        H5Pclose(dcpl);

        write_chunk_detail(ts, "nn_baseline", n_chunks);
        results[n_results++] = r;

        if (is_diag) {
            printf("  nn_base   : ratio=%.2fx  mape=%.1f%%  %s\n",
                   r.ratio, r.mean_mape, r.mismatches == 0 ? "PASS" : "FAIL");
        }
    }

    /* ── NN + SGD: every step (persistent learning) ── */
    if (global->phase_mask & P_NN_SGD) {
        if (step() == 1) {
            const char* wp = getenv("GPUCOMPRESS_WEIGHTS");
            if (wp) gpucompress_reload_nn(wp);
        }
        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, global->sgd_lr, global->sgd_mape_thresh, global->sgd_mape_thresh);
        gpucompress_set_exploration(0);

        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats);
        StepResult r;
        memset(&r, 0, sizeof(r));
        run_pass("nn_sgd", TMP_SGD,
                 d_fields, global->d_read,
                 global->h_orig, global->h_read,
                 n_floats, n_chunks, dcpl, &r, ts);
        H5Pclose(dcpl);

        write_chunk_detail(ts, "nn_sgd", n_chunks);
        results[n_results++] = r;

        if (is_diag) {
            printf("  nn_sgd    : ratio=%.2fx  mape=%.1f%%  sgd=%d/%d  %s\n",
                   r.ratio, r.mean_mape, r.sgd_fires, n_chunks,
                   r.mismatches == 0 ? "PASS" : "FAIL");
        }
    }

    /* Write timestep CSV */
    {
        FILE* f = fopen(OUT_TIMESTEP, "a");
        if (f) {
            for (int i = 0; i < n_results; i++) {
                StepResult* r = &results[i];
                fprintf(f, "%d,%s,%.4f,%.2f,%.2f,%.1f,"
                           "%.2f,%.2f,%d,%d,"
                           "%.4f,%.4f,%.4f,%.2f,%llu\n",
                        r->timestep, r->phase, r->ratio,
                        r->write_ms, r->read_ms, r->write_mbps,
                        (double)r->file_bytes / (1 << 20),
                        (double)r->orig_bytes / (1 << 20),
                        r->sgd_fires, r->n_chunks,
                        r->ratio_min, r->ratio_max, r->ratio_stddev,
                        r->mean_mape, r->mismatches);
            }
            fclose(f);
        }
    }

    /* ── Write aggregate CSV + cleanup on last step ── */
    if (is_last) {
        FILE* f = fopen(OUT_AGG_CSV, "w");
        if (f) {
            fprintf(f, "phase,ratio,write_ms,read_ms,file_mib,orig_mib,"
                       "write_mibps,read_mibps,mismatches,sgd_fires,n_chunks,"
                       "ratio_min,ratio_max,ratio_stddev,mean_prediction_error_pct\n");
            for (int i = 0; i < n_results; i++) {
                StepResult* r = &results[i];
                fprintf(f, "%s,%.4f,%.2f,%.2f,%.2f,%.2f,"
                           "%.1f,0,%llu,%d,%d,"
                           "%.4f,%.4f,%.4f,%.2f\n",
                        r->phase, r->ratio,
                        r->write_ms, r->read_ms,
                        (double)r->file_bytes / (1 << 20),
                        (double)r->orig_bytes / (1 << 20),
                        r->write_mbps,
                        r->mismatches, r->sgd_fires,
                        r->n_chunks,
                        r->ratio_min, r->ratio_max,
                        r->ratio_stddev, r->mean_mape);
            }
            fclose(f);
        }

        sim_log("");
        sim_log("╔═══════════════════════════════════════════════════════════════════════════╗");
        sim_log("║  Simulation Benchmark Complete                                           ║");
        sim_log("╠═══════════════════════════════════════════════════════════════════════════╣");
        sim_log("  Total steps      : " << global->num_steps);
        sim_log("  Exhaustive intervals : " << global->diag_count);
        sim_log("  Timestep CSV     : " << OUT_TIMESTEP);
        sim_log("  Per-chunk CSV    : " << OUT_CHUNKS);
        sim_log("  Upper bound CSV  : " << OUT_UB_CSV);
        sim_log("  Aggregate CSV    : " << OUT_AGG_CSV);
        sim_log("╚═══════════════════════════════════════════════════════════════════════════╝");

        /* Cleanup */
        cudaFree(global->d_read);
        free(global->h_orig);
        free(global->h_read);
        global->d_read = NULL;
        global->h_orig = NULL;
        global->h_read = NULL;
        if (global->vpic_fields_h) {
            gpucompress_vpic_destroy(global->vpic_fields_h);
            global->vpic_fields_h = NULL;
        }
        gpucompress_cleanup();
    }
};

begin_particle_injection {
}

begin_current_injection {
}

begin_field_injection {
}

begin_particle_collisions {
}
