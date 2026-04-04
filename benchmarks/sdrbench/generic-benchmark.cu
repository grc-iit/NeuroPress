/**
 * @file generic-benchmark.cu
 * @brief Generic SDRBench VOL Benchmark: No-Comp vs NN+SGD
 *
 * Loads raw binary float32 files from disk into GPU memory, then benchmarks
 * writing them to HDF5 via the GPUCompress VOL connector.
 * Supports any SDRBench dataset (Hurricane Isabel, Nyx, CESM-ATM, etc.).
 *
 * Each field file is treated as one "timestep" for multi-timestep mode,
 * enabling SGD online learning across fields with different statistics.
 *
 * Single-shot phases (run on first field):
 *   1. no-comp      : GPU ptr → H5Dwrite via VOL (no compression, baseline)
 *   2. nn           : GPU ptr → H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *
 * Multi-timestep phases (across all fields):
 *   3. nn-rl        : ALGO_AUTO + online SGD (learning curve over N fields)
 *   4. nn-rl+exp50  : ALGO_AUTO + online SGD + exploration
 *
 * Usage:
 *   ./build/generic_benchmark model.nnwt \
 *       --data-dir data/sdrbench/nyx/SDRBENCH-EXASKY-NYX-512x512x512 \
 *       --dims 512,512,512 \
 *       [--ext .f32] [--chunk-mb 4] [--runs 5] [--phase <name>] ...
 *
 * Options:
 *   model.nnwt       Path to NN weights (required, or GPUCOMPRESS_WEIGHTS env)
 *   --data-dir DIR   Directory containing raw binary field files
 *   --dims X,Y,Z     Dimensions of each field (e.g. 512,512,512 or 1800,3600)
 *   --ext EXT        File extension filter (default: .f32, also: .dat, .bin.f32)
 *   --chunk-mb N     Chunk size in MB (default 4)
 *   --runs N         Repeat single-shot phases N times (mean +/- std)
 *   --phase <name>   Run only specified phase(s). Repeat for multiple.
 *                    Valid: no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd,
 *                           fixed-lz4+shuf, fixed-gdeflate+shuf, fixed-zstd+shuf,
 *                           nn, nn-rl, nn-rl+exp50
 *   --out-dir DIR    Output directory for CSVs (default: benchmarks/sdrbench/results)
 *   --w0/w1/w2 F     Cost model weights (default 1/1/1)
 *
 * Dataset reference:
 *   Hurricane Isabel: --dims 100,500,500 --ext .bin.f32
 *   Nyx:             --dims 512,512,512 --ext .f32
 *   CESM-ATM:        --dims 1800,3600   --ext .dat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>
#include <dirent.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#ifdef GPUCOMPRESS_USE_MPI
#include <mpi.h>
#endif

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "../kendall_tau_profiler.cuh"

/* MPI rank/size — defaults to single-process when MPI is not used */
static int g_mpi_rank = 0;
static int g_mpi_size = 1;

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define DEFAULT_CHUNK_MB    4
#define DEFAULT_EXT         ".f32"

#define REINFORCE_LR        0.2f
#define REINFORCE_MAPE      0.10f

/* Temporary HDF5 files — rank-suffixed to avoid collisions on multi-GPU nodes. */
static char TMP_NOCOMP[256];
static char TMP_FIX_LZ4[256];
static char TMP_FIX_SNAPPY[256];
static char TMP_FIX_DEFL[256];
static char TMP_FIX_GDEFL[256];
static char TMP_FIX_ZSTD[256];
static char TMP_FIX_ANS[256];
static char TMP_FIX_CASC[256];
static char TMP_FIX_BITCOMP[256];
static char TMP_NN[256];
static char TMP_NN_RL[256];
static char TMP_NN_RLEXP[256];

static void init_tmp_paths(int mpi_rank) {
    snprintf(TMP_NOCOMP,      sizeof(TMP_NOCOMP),      "/tmp/bm_generic_nocomp_rank%d.h5",      mpi_rank);
    snprintf(TMP_FIX_LZ4,     sizeof(TMP_FIX_LZ4),     "/tmp/bm_generic_fix_lz4_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_SNAPPY,   sizeof(TMP_FIX_SNAPPY),   "/tmp/bm_generic_fix_snappy_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_DEFL,    sizeof(TMP_FIX_DEFL),    "/tmp/bm_generic_fix_deflate_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_GDEFL,   sizeof(TMP_FIX_GDEFL),   "/tmp/bm_generic_fix_gdefl_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_ZSTD,    sizeof(TMP_FIX_ZSTD),    "/tmp/bm_generic_fix_zstd_rank%d.h5",    mpi_rank);
    snprintf(TMP_FIX_ANS,     sizeof(TMP_FIX_ANS),     "/tmp/bm_generic_fix_ans_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_CASC,    sizeof(TMP_FIX_CASC),    "/tmp/bm_generic_fix_cascaded_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_BITCOMP,  sizeof(TMP_FIX_BITCOMP),  "/tmp/bm_generic_fix_bitcomp_rank%d.h5",  mpi_rank);
    snprintf(TMP_NN,          sizeof(TMP_NN),          "/tmp/bm_generic_nn_rank%d.h5",          mpi_rank);
    snprintf(TMP_NN_RL,       sizeof(TMP_NN_RL),       "/tmp/bm_generic_nn_rl_rank%d.h5",       mpi_rank);
    snprintf(TMP_NN_RLEXP,    sizeof(TMP_NN_RLEXP),    "/tmp/bm_generic_nn_rlexp_rank%d.h5",    mpi_rank);
}

#define DEFAULT_OUT_DIR "benchmarks/sdrbench/results"
#define MAX_FIELDS      128
#define MAX_PATH_LEN    512

/* Phase bitmask */
#define P_NOCOMP      0x001
#define P_FIX_LZ4     0x002
#define P_FIX_SNAPPY  0x004
#define P_FIX_DEFL    0x008
#define P_FIX_GDEFL   0x010
#define P_FIX_ZSTD    0x020
#define P_FIX_ANS     0x200
#define P_FIX_CASC    0x400
#define P_FIX_BITCOMP 0x800
#define P_NN          0x040
#define P_NNRL        0x080
#define P_NNRLEXP     0x100

/* Output paths */
static char OUT_DIR[MAX_PATH_LEN];
static char OUT_CSV[MAX_PATH_LEN];
static char OUT_CHUNKS[MAX_PATH_LEN];
static char OUT_TSTEP[MAX_PATH_LEN];
static char OUT_TSTEP_CHUNKS[MAX_PATH_LEN];
static char OUT_RANKING[MAX_PATH_LEN];
static char OUT_RANKING_COSTS[MAX_PATH_LEN];

/* HDF5 filter ID */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * Result struct (matches grayscott-benchmark.cu)
 * ============================================================ */

typedef struct {
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
    double mape_ratio_pct;
    double mape_comp_pct;
    double mape_decomp_pct;
    double mae_ratio;       /* Mean Absolute Error: mean(|pred - actual|) for ratio */
    double mae_comp_ms;     /* MAE for compression time (ms) */
    double mae_decomp_ms;   /* MAE for decompression time (ms) */
    double mape_psnr_pct;   /* MAPE for PSNR prediction (lossy only) */
    double mae_psnr_db;     /* MAE for PSNR prediction (dB) */
    double r2_ratio;        /* R² for ratio prediction */
    double r2_comp;         /* R² for compression time prediction */
    double r2_decomp;       /* R² for decompression time prediction */
    double r2_psnr;         /* R² for PSNR prediction */
    double comp_gbps;
    double decomp_gbps;
    /* Per-component timing (cumulative across chunks, unclamped) */
    double nn_ms;
    double stats_ms;
    double preproc_ms;
    double comp_ms;
    double decomp_ms;
    double explore_ms;
    double sgd_ms;
    /* Additive pipeline timing from VOL (stage1 + drain + io_drain = total) */
    double stage1_ms;     /* stats + NN inference (main thread) */
    double drain_ms;      /* worker drain: S1 end → workers joined */
    double io_drain_ms;   /* I/O drain: workers joined → I/O joined */
    /* Overlapping busy times (NOT additive with above) */
    double s2_busy_ms;    /* bottleneck worker wall time */
    double s3_busy_ms;    /* I/O write time (serial, accumulated) */
    /* Quality metrics (lossy only — populated when error_bound > 0) */
    double psnr_db;
    double rmse;
    double max_abs_err;
    double mean_abs_err;
    double ssim;
    double bit_rate;
    /* Standard deviations (populated when --runs N > 1) */
    double write_ms_std;
    double read_ms_std;
    double comp_gbps_std;
    double decomp_gbps_std;
    int    n_runs;
} PhaseResult;

/* Global error bound — set from main(), used by run_phase_* for PSNR */
static double g_error_bound = 0.0;

/* Global file extension — set from main(), used by load_field() */
static const char *g_ext = ".f32";

/* ============================================================
 * Timing
 * ============================================================ */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * File helpers
 * ============================================================ */

static size_t get_file_size(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

static void drop_pagecache(const char *path)
{
    int fd = open(path, O_RDWR);
    if (fd < 0) return;
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

static void mkdirs(const char *path)
{
    char tmp[MAX_PATH_LEN];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* ============================================================
 * PSNR / quality metrics (implemented in vpic_psnr.cu)
 * ============================================================ */
extern "C" double vpic_compute_quality_gpu(
    const float* d_original, const float* d_decompressed, size_t n_floats,
    double* out_rmse, double* out_max_err, double* out_mean_err, double* out_ssim);

/* ============================================================
 * GPU comparison kernel
 * ============================================================ */

__global__ void count_mismatches_kernel(const float * __restrict__ a,
                                        const float * __restrict__ b,
                                        size_t n,
                                        unsigned long long *count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local++;
    atomicAdd(count, local);
}

static unsigned long long gpu_compare(const float *a, const float *b,
                                      size_t n_floats,
                                      unsigned long long *d_count)
{
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    count_mismatches_kernel<<<512, 256>>>(a, b, n_floats, d_count);
    cudaDeviceSynchronize();
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost);
    return h_count;
}

/* ============================================================
 * Load binary file to GPU
 * ============================================================ */

static int load_field_to_gpu(const char *filepath, float *d_buf,
                             size_t expected_bytes)
{
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", filepath);
        return 1;
    }

    float *h_buf = (float *)malloc(expected_bytes);
    if (!h_buf) {
        fprintf(stderr, "ERROR: malloc failed for %zu bytes\n", expected_bytes);
        fclose(f);
        return 1;
    }

    size_t nread = fread(h_buf, 1, expected_bytes, f);
    fclose(f);

    if (nread != expected_bytes) {
        fprintf(stderr, "ERROR: %s: read %zu bytes, expected %zu\n",
                filepath, nread, expected_bytes);
        free(h_buf);
        return 1;
    }

    cudaMemcpy(d_buf, h_buf, expected_bytes, cudaMemcpyHostToDevice);
    free(h_buf);
    return 0;
}

/* Forward declaration */
static hid_t make_vol_fapl(void);

/* ============================================================
 * Load field from HDF5 file (VOL decompresses directly to GPU)
 * ============================================================ */

static int load_field_from_hdf5(const char *filepath, float *d_buf,
                                 size_t expected_bytes)
{
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        fprintf(stderr, "ERROR: H5Fopen failed: %s\n", filepath);
        return 1;
    }
    /* Try "data" first (from --hdf5-direct training), then "field" (from benchmark) */
    hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0)
        dset = H5Dopen2(file, "field", H5P_DEFAULT);
    if (dset < 0) {
        fprintf(stderr, "ERROR: H5Dopen2 failed (tried 'data' and 'field'): %s\n", filepath);
        H5Fclose(file);
        return 1;
    }
    herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_buf);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    if (rc < 0) {
        fprintf(stderr, "ERROR: H5Dread failed: %s\n", filepath);
        return 1;
    }
    return 0;
}

/* Dispatch: load from .f32 (raw) or .h5 (HDF5 VOL) based on extension */
static int load_field(const char *filepath, float *d_buf,
                      size_t expected_bytes, const char *ext)
{
    if (strcmp(ext, ".h5") == 0)
        return load_field_from_hdf5(filepath, d_buf, expected_bytes);
    return load_field_to_gpu(filepath, d_buf, expected_bytes);
}

/* ============================================================
 * Discover field files in a directory
 * ============================================================ */

static int discover_fields(const char *data_dir, const char *ext,
                           char fields[][MAX_PATH_LEN], int max_fields)
{
    DIR *d = opendir(data_dir);
    if (!d) {
        fprintf(stderr, "ERROR: cannot open directory %s\n", data_dir);
        return 0;
    }

    int n = 0;
    size_t ext_len = strlen(ext);
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && n < max_fields) {
        size_t name_len = strlen(ent->d_name);
        if (name_len >= ext_len &&
            strcmp(ent->d_name + name_len - ext_len, ext) == 0) {
            snprintf(fields[n], MAX_PATH_LEN, "%s/%s", data_dir, ent->d_name);
            n++;
        }
    }
    closedir(d);

    /* Sort alphabetically for reproducibility */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (strcmp(fields[i], fields[j]) > 0) {
                char tmp[MAX_PATH_LEN];
                memcpy(tmp, fields[i], MAX_PATH_LEN);
                memcpy(fields[i], fields[j], MAX_PATH_LEN);
                memcpy(fields[j], tmp, MAX_PATH_LEN);
            }

    return n;
}

/* ============================================================
 * HDF5 helpers
 * ============================================================ */

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_dcpl_auto(const hsize_t *dims, int ndims,
                            const hsize_t *chunk_dims, double eb)
{
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, ndims, chunk_dims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(eb, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static hid_t make_dcpl_fixed(int ndims, const hsize_t *chunk_dims,
                             gpucompress_algorithm_t algo,
                             unsigned int preproc = 0)
{
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, ndims, chunk_dims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo;
    cd[1] = preproc;
    cd[2] = 0;
    cd[3] = 0; cd[4] = 0; /* error_bound = 0.0 */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static hid_t make_dcpl_nocomp(int ndims, const hsize_t *chunk_dims)
{
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, ndims, chunk_dims);
    return dcpl;
}

static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

/* ============================================================
 * Collect per-chunk diagnostics into PhaseResult
 * ============================================================ */

static void collect_chunk_metrics(PhaseResult *r)
{
    int n_hist = gpucompress_get_chunk_history_count();
    double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0, mape_p_sum = 0;
    double mae_r_sum = 0, mae_c_sum = 0, mae_d_sum = 0, mae_p_sum = 0;
    int mcnt_r = 0, mcnt_c = 0, mcnt_d = 0, mcnt_p = 0;
    int sgd_t = 0, expl_t = 0;
    /* R² accumulators: one-pass via Σactual, Σactual², Σ(actual-predicted)² */
    double r2_r_sum = 0, r2_r_sum2 = 0, r2_r_ss = 0; int r2_r_n = 0;
    double r2_c_sum = 0, r2_c_sum2 = 0, r2_c_ss = 0; int r2_c_n = 0;
    double r2_d_sum = 0, r2_d_sum2 = 0, r2_d_ss = 0; int r2_d_n = 0;
    double r2_p_sum = 0, r2_p_sum2 = 0, r2_p_ss = 0; int r2_p_n = 0;

    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
        if (diag.sgd_fired) sgd_t++;
        if (diag.exploration_triggered) expl_t++;

        /* Ratio: MAPE (clamped denominator), MAE + R² (raw) */
        if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
            double a = diag.actual_ratio, p = diag.predicted_ratio;
            mape_r_sum += fabs(p - a) / fabs(a);
            mae_r_sum  += fabs(p - a);
            r2_r_sum += a; r2_r_sum2 += a*a; r2_r_ss += (a-p)*(a-p); r2_r_n++;
            mcnt_r++;
        }
        /* Compression time: MAPE uses clamped (5ms floor, matches NN target);
         * MAE and R² use unclamped raw time (measures real variance). */
        if (diag.compression_ms > 0) {
            double p = diag.predicted_comp_time;
            mape_c_sum += fabs(p - diag.compression_ms) / fabs(diag.compression_ms);
            mcnt_c++;
        }
        if (diag.compression_ms_raw > 0) {
            double a = diag.compression_ms_raw, p = diag.predicted_comp_time;
            mae_c_sum  += fabs(p - a);
            r2_c_sum += a; r2_c_sum2 += a*a; r2_c_ss += (a-p)*(a-p); r2_c_n++;
        }
        /* Decompression time: same split — MAPE on clamped, MAE/R² on raw. */
        if (diag.decompression_ms > 0) {
            double p = diag.predicted_decomp_time;
            mape_d_sum += fabs(p - diag.decompression_ms) / fabs(diag.decompression_ms);
            mcnt_d++;
        }
        if (diag.decompression_ms_raw > 0) {
            double a = diag.decompression_ms_raw, p = diag.predicted_decomp_time;
            mae_d_sum  += fabs(p - a);
            r2_d_sum += a; r2_d_sum2 += a*a; r2_d_ss += (a-p)*(a-p); r2_d_n++;
        }
        /* PSNR: MAPE/MAE/R² (skip lossless: actual_psnr=inf, no variance) */
        if (diag.predicted_psnr > 0.0f && diag.actual_psnr > 0.0f && std::isfinite(diag.actual_psnr)) {
            double a = diag.actual_psnr, p = diag.predicted_psnr;
            mape_p_sum += fabs(p - a) / fabs(a);
            mae_p_sum  += fabs(p - a);
            r2_p_sum += a; r2_p_sum2 += a*a; r2_p_ss += (a-p)*(a-p); r2_p_n++;
            mcnt_p++;
        }
    }

    /* Compute R² = 1 - SS_res/SS_tot where SS_tot = Σx² - (Σx)²/n */
    auto compute_r2 = [](double sum, double sum2, double ss_res, int n) -> double {
        if (n < 2) return 0.0;
        double ss_tot = sum2 - (sum * sum) / n;
        if (ss_tot < 1e-12) return 0.0;
        return 1.0 - ss_res / ss_tot;
    };

    r->sgd_fires   = sgd_t;
    r->explorations = expl_t;
    r->mape_ratio_pct   = fmin(200.0, mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0);
    r->mape_comp_pct    = fmin(200.0, mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0);
    r->mape_decomp_pct  = fmin(200.0, mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0);
    r->mape_psnr_pct    = fmin(200.0, mcnt_p ? (mape_p_sum / mcnt_p) * 100.0 : 0.0);
    r->mae_ratio     = mcnt_r ? mae_r_sum / mcnt_r : 0.0;
    r->mae_comp_ms   = mcnt_c ? mae_c_sum / mcnt_c : 0.0;
    r->mae_decomp_ms = mcnt_d ? mae_d_sum / mcnt_d : 0.0;
    r->mae_psnr_db   = mcnt_p ? mae_p_sum / mcnt_p : 0.0;
    r->r2_ratio  = compute_r2(r2_r_sum, r2_r_sum2, r2_r_ss, r2_r_n);
    r->r2_comp   = compute_r2(r2_c_sum, r2_c_sum2, r2_c_ss, r2_c_n);
    r->r2_decomp = compute_r2(r2_d_sum, r2_d_sum2, r2_d_ss, r2_d_n);
    r->r2_psnr   = compute_r2(r2_p_sum, r2_p_sum2, r2_p_ss, r2_p_n);

    /* Compute per-component timing and isolated throughput from chunk diagnostics. */
    double total_comp_ms = 0, total_decomp_ms = 0;
    double total_nn_ms = 0, total_stats_ms = 0, total_preproc_ms = 0;
    double total_explore_ms = 0, total_sgd_ms = 0;
    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
        total_comp_ms    += diag.compression_ms_raw;  /* unclamped for accurate breakdown */
        total_decomp_ms  += diag.decompression_ms_raw;  /* unclamped for breakdown */
        total_nn_ms      += diag.nn_inference_ms;
        total_stats_ms   += diag.stats_ms;
        total_preproc_ms += diag.preprocessing_ms;
        total_explore_ms += diag.exploration_ms;
        total_sgd_ms     += diag.sgd_update_ms;
    }
    r->comp_gbps   = (total_comp_ms > 0)
        ? (double)r->orig_bytes / total_comp_ms / 1e6 : 0.0;
    r->decomp_gbps = (total_decomp_ms > 0)
        ? (double)r->orig_bytes / total_decomp_ms / 1e6 : 0.0;
    r->nn_ms       = total_nn_ms;
    r->stats_ms    = total_stats_ms;
    r->preproc_ms  = total_preproc_ms;
    r->comp_ms     = total_comp_ms;
    r->decomp_ms   = total_decomp_ms;
    r->explore_ms  = total_explore_ms;
    r->sgd_ms      = total_sgd_ms;
}

/* ============================================================
 * Phase: no-comp
 * ============================================================ */

static int run_phase_nocomp(float *d_data, float *d_read,
                            unsigned long long *d_count,
                            size_t n_floats, int ndims, const hsize_t *dims,
                            const hsize_t *chunk_dims,
                            PhaseResult *r)
{
    memset(r, 0, sizeof(*r));
    size_t total_bytes = n_floats * sizeof(float);

    /* Warmup */
    {
        const char *warmup_file = "/tmp/bm_generic_warmup.h5";
        remove(warmup_file);
        hid_t wdcpl = make_dcpl_nocomp(ndims, chunk_dims);
        hid_t wfapl = make_vol_fapl();
        hid_t wfile = H5Fcreate(warmup_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
        H5Pclose(wfapl);
        if (wfile >= 0) {
            hid_t wfsp  = H5Screate_simple(ndims, dims, NULL);
            hid_t wdset = H5Dcreate2(wfile, "field", H5T_NATIVE_FLOAT,
                                      wfsp, H5P_DEFAULT, wdcpl, H5P_DEFAULT);
            H5Sclose(wfsp);
            if (wdset >= 0) {
                H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
                cudaDeviceSynchronize();
                H5Dclose(wdset);
            }
            H5Fclose(wfile);
            remove(warmup_file);
        }
        H5Pclose(wdcpl);
    }

    /* Write */
    printf("[no-comp] H5Dwrite... "); fflush(stdout);
    remove(TMP_NOCOMP);
    hid_t dcpl = make_dcpl_nocomp(ndims, chunk_dims);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_NOCOMP, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    assert(file >= 0);
    H5Pclose(fapl);
    hid_t fsp  = H5Screate_simple(ndims, dims, NULL);
    hid_t dset = H5Dcreate2(file, "field", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    assert(dset >= 0);
    H5Sclose(fsp);
    H5Pclose(dcpl);

    double t0 = now_ms();
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();
    printf("%.0f ms\n", t1 - t0);

    drop_pagecache(TMP_NOCOMP);

    /* Read */
    printf("[no-comp] H5Dread... "); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_NOCOMP, H5F_ACC_RDONLY, fapl);
    assert(file >= 0);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "field", H5P_DEFAULT);
    assert(dset >= 0);

    double t2 = now_ms();
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = now_ms();
    printf("%.0f ms\n", t3 - t2);

    unsigned long long mm = gpu_compare(d_data, d_read, n_floats, d_count);

    size_t fbytes = get_file_size(TMP_NOCOMP);
    r->write_ms   = t1 - t0;
    r->read_ms    = t3 - t2;
    r->file_bytes = fbytes;
    r->orig_bytes = total_bytes;
    r->ratio      = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps  = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches = mm;
    r->n_runs     = 1;
    snprintf(r->phase, sizeof(r->phase), "no-comp");

    if (g_error_bound > 0.0) {
        r->psnr_db = vpic_compute_quality_gpu(d_data, d_read, n_floats,
                                              &r->rmse, &r->max_abs_err,
                                              &r->mean_abs_err, &r->ssim);
        r->bit_rate = (fbytes > 0) ? (double)(fbytes * 8) / (double)n_floats : 0.0;
    } else {
        r->ssim = 1.0;  /* lossless: identical signals */
    }

    printf("[no-comp] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  mismatches=%llu\n",
           r->ratio, r->write_mbps, r->read_mbps, mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Phase: VOL (nn / nn-rl / nn-rl+exp)
 * ============================================================ */

static int run_phase_vol(float *d_data, float *d_read,
                         unsigned long long *d_count,
                         size_t n_floats, int ndims, const hsize_t *dims,
                         const hsize_t *chunk_dims,
                         const char *phase_name, const char *tmp_file,
                         hid_t dcpl, PhaseResult *r)
{
    memset(r, 0, sizeof(*r));
    size_t total_bytes = n_floats * sizeof(float);

    /* Warmup (learning disabled) */
    {
        const char *warmup_file = "/tmp/bm_generic_warmup.h5";
        int save_learning = gpucompress_online_learning_enabled();
        gpucompress_disable_online_learning();

        remove(warmup_file);
        hid_t wfapl = make_vol_fapl();
        hid_t wfile = H5Fcreate(warmup_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
        H5Pclose(wfapl);
        if (wfile >= 0) {
            hid_t wfsp  = H5Screate_simple(ndims, dims, NULL);
            hid_t wdset = H5Dcreate2(wfile, "field", H5T_NATIVE_FLOAT,
                                      wfsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(wfsp);
            if (wdset >= 0) {
                H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
                cudaDeviceSynchronize();
                H5Dclose(wdset);
            }
            H5Fclose(wfile);
            remove(warmup_file);
        }

        if (save_learning) gpucompress_enable_online_learning();
        gpucompress_flush_manager_cache();  /* cold-start the timed run */
    }

    /* Write */
    printf("[%s] H5Dwrite... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
    gpucompress_set_debug_context(phase_name, -1);
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file); return 1; }

    hid_t fsp  = H5Screate_simple(ndims, dims, NULL);
    hid_t dset = H5Dcreate2(file, "field", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0 = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();
    printf("%.0f ms\n", t1 - t0);

    if (wret < 0) { fprintf(stderr, "[%s] H5Dwrite failed\n", phase_name); return 1; }

    /* Capture additive pipeline timing + busy times from VOL */
    H5VL_gpucompress_get_stage_timing(&r->stage1_ms, &r->drain_ms, &r->io_drain_ms, NULL);
    H5VL_gpucompress_get_busy_timing(&r->s2_busy_ms, &r->s3_busy_ms);
    drop_pagecache(tmp_file);

    /* Read */
    printf("[%s] H5Dread... ", phase_name); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    assert(file >= 0);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "field", H5P_DEFAULT);
    assert(dset >= 0);

    double t2 = now_ms();
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = now_ms();
    printf("%.0f ms\n", t3 - t2);

    unsigned long long mm = gpu_compare(d_data, d_read, n_floats, d_count);

    size_t fbytes = get_file_size(tmp_file);
    r->write_ms   = t1 - t0;
    r->read_ms    = t3 - t2;
    r->file_bytes = fbytes;
    r->orig_bytes = total_bytes;
    r->ratio      = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps  = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches = mm;
    r->n_runs     = 1;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    if (g_error_bound > 0.0) {
        r->psnr_db = vpic_compute_quality_gpu(d_data, d_read, n_floats,
                                              &r->rmse, &r->max_abs_err,
                                              &r->mean_abs_err, &r->ssim);
        r->bit_rate = (fbytes > 0) ? (double)(fbytes * 8) / (double)n_floats : 0.0;
    } else {
        r->ssim = 1.0;  /* lossless: identical signals */
    }

    collect_chunk_metrics(r);

    printf("[%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "MAPE_R=%.1f%%  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           r->mape_ratio_pct, mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Multi-run averaging
 * ============================================================ */

static void merge_phase_results(PhaseResult *runs, int n, PhaseResult *out)
{
    *out = runs[0];
    if (n <= 1) return;

    double sum_w = 0, sum_r = 0, sum_cg = 0, sum_dg = 0;
    for (int i = 0; i < n; i++) {
        sum_w  += runs[i].write_ms;
        sum_r  += runs[i].read_ms;
        sum_cg += runs[i].comp_gbps;
        sum_dg += runs[i].decomp_gbps;
    }
    out->write_ms   = sum_w / n;
    out->read_ms    = sum_r / n;
    out->comp_gbps  = sum_cg / n;
    out->decomp_gbps = sum_dg / n;

    /* Recompute derived */
    out->write_mbps = (double)out->orig_bytes / (1 << 20) / (out->write_ms / 1000.0);
    out->read_mbps  = (double)out->orig_bytes / (1 << 20) / (out->read_ms / 1000.0);

    /* Standard deviations */
    double var_w = 0, var_r = 0, var_cg = 0, var_dg = 0;
    for (int i = 0; i < n; i++) {
        var_w  += (runs[i].write_ms - out->write_ms) * (runs[i].write_ms - out->write_ms);
        var_r  += (runs[i].read_ms - out->read_ms) * (runs[i].read_ms - out->read_ms);
        var_cg += (runs[i].comp_gbps - out->comp_gbps) * (runs[i].comp_gbps - out->comp_gbps);
        var_dg += (runs[i].decomp_gbps - out->decomp_gbps) * (runs[i].decomp_gbps - out->decomp_gbps);
    }
    double denom = (n > 1) ? (n - 1) : 1;  /* sample std dev */
    out->write_ms_std   = sqrt(var_w / denom);
    out->read_ms_std    = sqrt(var_r / denom);
    out->comp_gbps_std  = sqrt(var_cg / denom);
    out->decomp_gbps_std = sqrt(var_dg / denom);
    out->n_runs = n;
}

/* ============================================================
 * Multi-field averaging for single-shot phases.
 * Runs a VOL phase on every field and averages the results,
 * so single-shot and multi-timestep phases are comparable.
 * ============================================================ */

typedef int (*run_phase_fn)(float *d_data, float *d_read,
                            unsigned long long *d_count,
                            size_t n_floats, int ndims,
                            const hsize_t *dims, const hsize_t *chunk_dims,
                            const char *phase_name, const char *tmp_file,
                            hid_t dcpl, PhaseResult *r);

typedef int (*run_phase_nocomp_fn)(float *d_data, float *d_read,
                                   unsigned long long *d_count,
                                   size_t n_floats, int ndims,
                                   const hsize_t *dims, const hsize_t *chunk_dims,
                                   PhaseResult *r);

static void run_phase_all_fields(
    run_phase_fn fn, float *d_data, float *d_read,
    unsigned long long *d_count, size_t n_floats, size_t total_bytes,
    int ndims, const hsize_t *dims, const hsize_t *chunk_dims,
    const char *phase_name, const char *tmp_file, hid_t dcpl,
    char fields[][MAX_PATH_LEN], int n_fields,
    PhaseResult *out, int *any_fail)
{
    /* Per-field arrays for std computation */
    const int MAX_F = 256;
    double *f_write = (double*)calloc(MAX_F, sizeof(double));
    double *f_read  = (double*)calloc(MAX_F, sizeof(double));
    double *f_cgbps = (double*)calloc(MAX_F, sizeof(double));
    double *f_dgbps = (double*)calloc(MAX_F, sizeof(double));

    double sum_ratio = 0, sum_write_ms = 0, sum_read_ms = 0;
    double sum_comp_gbps = 0, sum_decomp_gbps = 0;
    double sum_nn_ms = 0, sum_stats_ms = 0, sum_preproc_ms = 0;
    double sum_comp_ms = 0, sum_decomp_ms = 0, sum_explore_ms = 0, sum_sgd_ms = 0;
    double sum_stage1 = 0, sum_stage2 = 0, sum_stage3 = 0;
    double sum_s2_busy = 0, sum_s3_busy = 0;
    size_t sum_file_bytes = 0;
    int sum_sgd_fires = 0, sum_explorations = 0;
    double sum_mae_r = 0, sum_mae_c = 0, sum_mae_d = 0;
    double sum_mape_r = 0, sum_mape_c = 0, sum_mape_d = 0, sum_mape_p = 0;
    double sum_r2_r = 0, sum_r2_c = 0, sum_r2_d = 0, sum_r2_p = 0;
    double agg_mse = 0, agg_range_sq = 0, agg_maxerr = 0, agg_bitrate = 0;
    double agg_meanerr = 0, agg_ssim = 0;
    double sum_mae_p = 0;
    int count = 0;
    int first_n_chunks = 0;

    for (int fi = 0; fi < n_fields; fi++) {
        if (load_field(fields[fi], d_data, total_bytes, g_ext)) {
            printf("  field %d: SKIP (load failed)\n", fi);
            continue;
        }

        PhaseResult fr;
        int rc = fn(d_data, d_read, d_count, n_floats, ndims, dims,
                    chunk_dims, phase_name, tmp_file, dcpl, &fr);
        if (rc) *any_fail = 1;

        if (count == 0) first_n_chunks = fr.n_chunks;

        const char *fname = strrchr(fields[fi], '/');
        fname = fname ? fname + 1 : fields[fi];
        printf("  field %d/%d: %-30s ratio=%.2fx  write=%.0f MiB/s\n",
               fi + 1, n_fields, fname, fr.ratio, fr.write_mbps);

        if (count < MAX_F) {
            f_write[count] = fr.write_ms;
            f_read[count]  = fr.read_ms;
            f_cgbps[count] = fr.comp_gbps;
            f_dgbps[count] = fr.decomp_gbps;
        }

        sum_ratio        += fr.ratio;
        sum_write_ms     += fr.write_ms;
        sum_read_ms      += fr.read_ms;
        sum_file_bytes   += fr.file_bytes;
        sum_comp_gbps    += fr.comp_gbps;
        sum_decomp_gbps  += fr.decomp_gbps;
        sum_nn_ms        += fr.nn_ms;
        sum_stats_ms     += fr.stats_ms;
        sum_preproc_ms   += fr.preproc_ms;
        sum_comp_ms      += fr.comp_ms;
        sum_decomp_ms    += fr.decomp_ms;
        sum_explore_ms   += fr.explore_ms;
        sum_sgd_ms       += fr.sgd_ms;
        sum_stage1       += fr.stage1_ms;
        sum_stage2       += fr.drain_ms;
        sum_stage3       += fr.io_drain_ms;
        sum_s2_busy      += fr.s2_busy_ms;
        sum_s3_busy      += fr.s3_busy_ms;
        sum_sgd_fires    += fr.sgd_fires;
        sum_explorations += fr.explorations;
        sum_mape_r       += fr.mape_ratio_pct;
        sum_mape_c       += fr.mape_comp_pct;
        sum_mape_d       += fr.mape_decomp_pct;
        sum_mape_p       += fr.mape_psnr_pct;
        sum_mae_r        += fr.mae_ratio;
        sum_mae_c        += fr.mae_comp_ms;
        sum_mae_d        += fr.mae_decomp_ms;
        sum_mae_p        += fr.mae_psnr_db;
        sum_r2_r         += fr.r2_ratio;
        sum_r2_c         += fr.r2_comp;
        sum_r2_d         += fr.r2_decomp;
        sum_r2_p         += fr.r2_psnr;
        agg_mse          += fr.rmse * fr.rmse;  /* accumulate MSE = RMSE² */
        if (std::isfinite(fr.psnr_db) && fr.psnr_db < 300.0 && fr.rmse > 0.0)
            agg_range_sq += fr.rmse * fr.rmse * pow(10.0, fr.psnr_db / 10.0);
        agg_maxerr        = (fr.max_abs_err > agg_maxerr) ? fr.max_abs_err : agg_maxerr;
        agg_meanerr      += fr.mean_abs_err;
        agg_ssim         += fr.ssim;
        agg_bitrate      += fr.bit_rate;
        count++;
    }

    if (count > 0) {
        *out = {};
        snprintf(out->phase, sizeof(out->phase), "%s", phase_name);
        out->file_bytes   = sum_file_bytes / count;
        out->ratio        = (sum_file_bytes > 0)
            ? (double)(count * total_bytes) / (double)sum_file_bytes : 1.0;
        out->write_ms     = sum_write_ms / count;
        out->read_ms      = sum_read_ms / count;
        out->orig_bytes   = total_bytes;
        out->write_mbps   = (double)total_bytes / (1 << 20) / (out->write_ms / 1000.0);
        out->read_mbps    = (double)total_bytes / (1 << 20) / (out->read_ms / 1000.0);
        out->comp_gbps    = sum_comp_gbps / count;
        out->decomp_gbps  = sum_decomp_gbps / count;
        out->nn_ms        = sum_nn_ms / count;
        out->stats_ms     = sum_stats_ms / count;
        out->preproc_ms   = sum_preproc_ms / count;
        out->comp_ms      = sum_comp_ms / count;
        out->decomp_ms    = sum_decomp_ms / count;
        out->explore_ms   = sum_explore_ms / count;
        out->sgd_ms       = sum_sgd_ms / count;
        out->stage1_ms    = sum_stage1 / count;
        out->drain_ms     = sum_stage2 / count;
        out->io_drain_ms  = sum_stage3 / count;
        out->s2_busy_ms   = sum_s2_busy / count;
        out->s3_busy_ms   = sum_s3_busy / count;
        out->sgd_fires    = sum_sgd_fires;
        out->explorations = sum_explorations;
        out->mape_ratio_pct  = sum_mape_r / count;
        out->mape_comp_pct   = sum_mape_c / count;
        out->mape_decomp_pct = sum_mape_d / count;
        out->mape_psnr_pct   = sum_mape_p / count;
        out->mae_ratio       = sum_mae_r / count;
        out->mae_comp_ms     = sum_mae_c / count;
        out->mae_decomp_ms   = sum_mae_d / count;
        out->mae_psnr_db     = sum_mae_p / count;
        out->r2_ratio        = sum_r2_r / count;
        out->r2_comp         = sum_r2_c / count;
        out->r2_decomp       = sum_r2_d / count;
        out->r2_psnr         = sum_r2_p / count;
        out->n_runs       = count;
        out->n_chunks     = first_n_chunks;
        if (g_error_bound > 0.0) {
            double avg_mse = agg_mse / count;
            out->rmse         = sqrt(avg_mse);
            out->psnr_db      = (avg_mse > 0.0 && agg_range_sq > 0.0)
                ? 10.0 * log10((agg_range_sq / count) / avg_mse) : INFINITY;
            out->max_abs_err  = agg_maxerr;
            out->mean_abs_err = agg_meanerr / count;
            out->ssim         = agg_ssim / count;
            out->bit_rate     = agg_bitrate / count;
        }

        /* Compute std across fields (sample std, n-1) */
        int n = (count < MAX_F) ? count : MAX_F;
        double var_w = 0, var_r = 0, var_cg = 0, var_dg = 0;
        for (int i = 0; i < n; i++) {
            var_w  += (f_write[i] - out->write_ms) * (f_write[i] - out->write_ms);
            var_r  += (f_read[i]  - out->read_ms)  * (f_read[i]  - out->read_ms);
            var_cg += (f_cgbps[i] - out->comp_gbps) * (f_cgbps[i] - out->comp_gbps);
            var_dg += (f_dgbps[i] - out->decomp_gbps) * (f_dgbps[i] - out->decomp_gbps);
        }
        double denom = (n > 1) ? (n - 1) : 1;
        out->write_ms_std    = sqrt(var_w / denom);
        out->read_ms_std     = sqrt(var_r / denom);
        out->comp_gbps_std   = sqrt(var_cg / denom);
        out->decomp_gbps_std = sqrt(var_dg / denom);
    }

    free(f_write); free(f_read); free(f_cgbps); free(f_dgbps);
}

/* ============================================================
 * Write summary CSV
 * ============================================================ */

static void write_summary_csv(const char *dataset_name,
                              PhaseResult *results, int n_phases,
                              size_t total_bytes, int n_chunks)
{
    FILE *f = fopen(OUT_CSV, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", OUT_CSV); return; }

    fprintf(f, "rank,source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,"
            "file_mib,orig_mib,ratio,"
            "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks,"
            "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
            "comp_gbps,decomp_gbps,"
            "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,"
            "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
            "r2_ratio,r2_comp,r2_decomp,r2_psnr,"
            "comp_gbps_std,decomp_gbps_std,"
            "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
            "vol_s2_busy_ms,vol_s3_busy_ms,"
            "psnr_db,rmse,max_abs_err,mean_abs_err,ssim,bit_rate\n");

    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &results[i];
        fprintf(f, "%d,%s,%s,%d,%.2f,%.2f,%.2f,%.2f,"
                "%.2f,%.2f,%.4f,"
                "%.1f,%.1f,%llu,%d,%d,%d,"
                "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                "%.4f,%.4f,"
                "%.2f,%.2f,%.2f,%.2f,"
                "%.4f,%.4f,%.4f,%.4f,"
                "%.4f,%.4f,%.4f,%.4f,"
                "%.4f,%.4f,"
                "%.2f,%.2f,%.2f,"
                "%.2f,%.2f,"
                "%.2f,%.6f,%.6f,%.6f,%.8f,%.4f\n",
                g_mpi_rank, dataset_name, r->phase, r->n_runs,
                r->write_ms, r->write_ms_std, r->read_ms, r->read_ms_std,
                (double)r->file_bytes / (1 << 20),
                (double)r->orig_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps, r->mismatches,
                r->sgd_fires, r->explorations, r->n_chunks,
                r->nn_ms, r->stats_ms, r->preproc_ms,
                r->comp_ms, r->decomp_ms, r->explore_ms, r->sgd_ms,
                r->comp_gbps, r->decomp_gbps,
                r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct, r->mape_psnr_pct,
                r->mae_ratio, r->mae_comp_ms, r->mae_decomp_ms, r->mae_psnr_db,
                r->r2_ratio, r->r2_comp, r->r2_decomp, r->r2_psnr,
                r->comp_gbps_std, r->decomp_gbps_std,
                r->stage1_ms, r->drain_ms, r->io_drain_ms,
                r->s2_busy_ms, r->s3_busy_ms,
                r->psnr_db, r->rmse, r->max_abs_err,
                r->mean_abs_err, r->ssim, r->bit_rate);
    }
    fclose(f);
    printf("\nSummary CSV: %s\n", OUT_CSV);
}

/* ============================================================
 * Write per-chunk CSV (delegates to chunk history)
 * ============================================================ */

static void write_chunk_csv(const char *phase_name)
{
    int n_hist = gpucompress_get_chunk_history_count();
    if (n_hist <= 0) return;

    int append = (get_file_size(OUT_CHUNKS) > 0);
    FILE *f = fopen(OUT_CHUNKS, append ? "a" : "w");
    if (!f) return;

    if (!append) {
        fprintf(f, "phase,chunk,action,"
                "predicted_ratio,actual_ratio,final_ratio,"
                "predicted_comp_ms,actual_comp_ms_raw,"
                "predicted_decomp_ms,actual_decomp_ms_raw,"
                "mape_ratio,mape_comp,mape_decomp,"
                "sgd_fired,exploration_triggered,"
                "feat_entropy,feat_mad,feat_deriv\n");
    }

    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;

        double mr = 0, mc = 0, md = 0;
        if (diag.actual_ratio > 0)
            mr = fmin(200.0, fabs(diag.predicted_ratio - diag.actual_ratio) / fabs(diag.actual_ratio) * 100.0);
        if (diag.compression_ms > 0)
            mc = fmin(200.0, fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms) * 100.0);
        if (diag.decompression_ms > 0)
            md = fmin(200.0, fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms) * 100.0);

        fprintf(f, "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.2f,%.2f,%.2f,%d,%d,"
                "%.4f,%.6f,%.6f\n",
                phase_name, ci, diag.nn_action,
                diag.predicted_ratio, diag.actual_ratio, diag.final_ratio,
                diag.predicted_comp_time, diag.compression_ms_raw,
                diag.predicted_decomp_time, diag.decompression_ms_raw,
                mr, mc, md,
                diag.sgd_fired, diag.exploration_triggered,
                diag.feat_entropy, diag.feat_mad, diag.feat_deriv);
    }
    fclose(f);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
#ifdef GPUCOMPRESS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
#endif

    /* Initialize rank-suffixed /tmp paths before any CUDA or HDF5 calls */
    init_tmp_paths(g_mpi_rank);

    /* ── Parse arguments ── */
    const char *weights_path = NULL;
    const char *data_dir     = NULL;
    const char *ext          = DEFAULT_EXT;
    const char *out_dir_override = NULL;
    const char *name_override = NULL;
    int dims[3] = {0, 0, 0};
    int ndims = 0;
    int chunk_mb = DEFAULT_CHUNK_MB;
    int n_runs   = 1;
    unsigned int phase_mask = 0;
    float rank_w0 = 1.0f, rank_w1 = 1.0f, rank_w2 = 1.0f;
    float sgd_lr = REINFORCE_LR;
    float sgd_mape = REINFORCE_MAPE;
    int   explore_k = 4;
    float explore_thresh = 0.20f;
    int   do_verify = 1;
    double error_bound = 0.0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !weights_path) {
            weights_path = argv[i];
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) {
            i++;
            ndims = 0;
            char *tok = strtok(argv[i], ",x");
            while (tok && ndims < 3) {
                dims[ndims++] = atoi(tok);
                tok = strtok(NULL, ",x");
            }
        } else if (strcmp(argv[i], "--ext") == 0 && i + 1 < argc) {
            ext = argv[++i];
            g_ext = ext;
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "no-comp") == 0) phase_mask |= P_NOCOMP;
            else if (strcmp(argv[i], "lz4") == 0 || strcmp(argv[i], "fixed-lz4") == 0) phase_mask |= P_FIX_LZ4;
            else if (strcmp(argv[i], "snappy") == 0 || strcmp(argv[i], "fixed-snappy") == 0) phase_mask |= P_FIX_SNAPPY;
            else if (strcmp(argv[i], "deflate") == 0 || strcmp(argv[i], "fixed-deflate") == 0) phase_mask |= P_FIX_DEFL;
            else if (strcmp(argv[i], "gdeflate") == 0 || strcmp(argv[i], "fixed-gdeflate") == 0) phase_mask |= P_FIX_GDEFL;
            else if (strcmp(argv[i], "zstd") == 0 || strcmp(argv[i], "fixed-zstd") == 0) phase_mask |= P_FIX_ZSTD;
            else if (strcmp(argv[i], "ans") == 0 || strcmp(argv[i], "fixed-ans") == 0) phase_mask |= P_FIX_ANS;
            else if (strcmp(argv[i], "cascaded") == 0 || strcmp(argv[i], "fixed-cascaded") == 0) phase_mask |= P_FIX_CASC;
            else if (strcmp(argv[i], "bitcomp") == 0 || strcmp(argv[i], "fixed-bitcomp") == 0) phase_mask |= P_FIX_BITCOMP;
            else if (strcmp(argv[i], "nn") == 0) phase_mask |= P_NN;
            else if (strcmp(argv[i], "nn-rl") == 0) phase_mask |= P_NNRL;
            else if (strcmp(argv[i], "nn-rl+exp50") == 0) phase_mask |= P_NNRLEXP;
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir_override = argv[++i];
        } else if (strcmp(argv[i], "--w0") == 0 && i + 1 < argc) {
            rank_w0 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--w1") == 0 && i + 1 < argc) {
            rank_w1 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--w2") == 0 && i + 1 < argc) {
            rank_w2 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            sgd_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--mape") == 0 && i + 1 < argc) {
            sgd_mape = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--explore-k") == 0 && i + 1 < argc) {
            explore_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--explore-thresh") == 0 && i + 1 < argc) {
            explore_thresh = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            do_verify = 0;
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc) {
            error_bound = atof(argv[++i]);
            g_error_bound = error_bound;
        } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            name_override = argv[++i];
        }
    }

    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");

    if (!weights_path || !data_dir || ndims < 2) {
        fprintf(stderr,
            "Usage: %s <weights.nnwt> --data-dir DIR --dims X,Y,Z [options]\n\n"
            "Required:\n"
            "  weights.nnwt       Path to NN weights\n"
            "  --data-dir DIR     Directory with raw binary field files\n"
            "  --dims X,Y,Z       Dimensions (2D or 3D)\n\n"
            "Options:\n"
            "  --ext EXT          File extension filter (default: %s)\n"
            "  --chunk-mb N       Chunk size in MB (default: %d)\n"
            "  --runs N           Repeat single-shot phases N times\n"
            "  --phase <name>     Run specific phase(s): no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd,\n"
            "                     fixed-lz4+shuf, fixed-gdeflate+shuf, fixed-zstd+shuf, nn, nn-rl, nn-rl+exp50\n"
            "  --out-dir DIR      Output directory (default: %s)\n"
            "  --w0/w1/w2 F       Cost model weights\n"
            "  --lr F             SGD learning rate (default: %.2f)\n"
            "  --error-bound F    Error bound for lossy (default: 0.0 = lossless)\n"
            "  --name NAME        Dataset name for CSV filenames (default: auto from data-dir)\n\n"
            "Examples:\n"
            "  %s model.nnwt --data-dir data/sdrbench/nyx/SDRBENCH-EXASKY-NYX-512x512x512 "
            "--dims 512,512,512 --ext .f32\n"
            "  %s model.nnwt --data-dir data/sdrbench/cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600 "
            "--dims 1800,3600 --ext .dat\n",
            argv[0], DEFAULT_EXT, DEFAULT_CHUNK_MB, DEFAULT_OUT_DIR,
            REINFORCE_LR, argv[0], argv[0]);
        return 1;
    }

    if (phase_mask == 0) phase_mask = P_NOCOMP | P_FIX_LZ4 | P_FIX_SNAPPY | P_FIX_DEFL
                                    | P_FIX_GDEFL | P_FIX_ZSTD | P_FIX_ANS
                                    | P_FIX_CASC | P_FIX_BITCOMP
                                    | P_NN | P_NNRL | P_NNRLEXP;

    /* ── Discover field files ── */
    static char fields[MAX_FIELDS][MAX_PATH_LEN];
    int n_fields = discover_fields(data_dir, ext, fields, MAX_FIELDS);
    if (n_fields == 0) {
        fprintf(stderr, "ERROR: no *%s files found in %s\n", ext, data_dir);
        return 1;
    }

    /* ── Compute sizes ── */
    size_t n_floats = 1;
    for (int i = 0; i < ndims; i++) n_floats *= (size_t)dims[i];
    size_t total_bytes = n_floats * sizeof(float);
    size_t expected_file_size = total_bytes;

    /* Verify first file matches expected size (skip for .h5 — compressed on disk) */
    if (strcmp(g_ext, ".h5") != 0) {
        size_t first_size = get_file_size(fields[0]);
        if (first_size != expected_file_size) {
            fprintf(stderr, "ERROR: %s is %zu bytes, expected %zu (dims=",
                    fields[0], first_size, expected_file_size);
            for (int i = 0; i < ndims; i++)
                fprintf(stderr, "%s%d", i ? "x" : "", dims[i]);
            fprintf(stderr, " * 4 bytes)\n");
            return 1;
        }
    }

    /* ── Set up HDF5 dimensions and chunking ── */
    hsize_t h5_dims[3];
    hsize_t chunk_dims[3];
    for (int i = 0; i < ndims; i++) h5_dims[i] = (hsize_t)dims[i];

    /* Chunking: last dimension is split to achieve target chunk_mb */
    size_t slice_floats = 1;
    for (int i = 0; i < ndims - 1; i++) slice_floats *= (size_t)dims[i];
    int last_chunk = (int)((size_t)chunk_mb * 1024 * 1024 / (slice_floats * sizeof(float)));
    if (last_chunk < 1) last_chunk = 1;
    if (last_chunk > dims[ndims - 1]) last_chunk = dims[ndims - 1];

    for (int i = 0; i < ndims - 1; i++) chunk_dims[i] = h5_dims[i];
    chunk_dims[ndims - 1] = (hsize_t)last_chunk;

    int n_chunks = (dims[ndims - 1] + last_chunk - 1) / last_chunk;
    double dataset_mb = (double)total_bytes / (1 << 20);
    double cmb = (double)(slice_floats * last_chunk * sizeof(float)) / (1024.0 * 1024.0);

    /* ── Extract dataset name from --name override or data_dir ── */
    const char *dataset_name;
    if (name_override) {
        dataset_name = name_override;
    } else {
        dataset_name = strrchr(data_dir, '/');
        dataset_name = dataset_name ? dataset_name + 1 : data_dir;
        /* If leaf is purely digits and separators (e.g. "100x500x500"),
           use the parent directory name instead for readability. */
        bool leaf_is_dims = true;
        for (const char *p = dataset_name; *p; p++) {
            if (!(*p >= '0' && *p <= '9') && *p != 'x' && *p != 'X') {
                leaf_is_dims = false;
                break;
            }
        }
        if (leaf_is_dims && dataset_name > data_dir) {
            /* Walk back to find parent directory name */
            const char *end = dataset_name - 1; /* points to '/' before leaf */
            const char *start = end - 1;
            while (start > data_dir && *start != '/') start--;
            if (*start == '/') start++;
            static char parent_name[256];
            int len = (int)(end - start);
            if (len > 0 && len < 255) {
                memcpy(parent_name, start, len);
                parent_name[len] = '\0';
                dataset_name = parent_name;
            }
        }
    }

    /* ── Print banner ── */
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Generic SDRBench VOL Benchmark: No-Comp vs NN+SGD                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Dataset  : %s\n", dataset_name);
    printf("  Data dir : %s\n", data_dir);
    printf("  Fields   : %d files (*%s)\n", n_fields, ext);
    printf("  Dims     : ");
    for (int i = 0; i < ndims; i++) printf("%s%d", i ? " x " : "", dims[i]);
    printf("  (%.1f MB per field)\n", dataset_mb);
    printf("  Chunks   : ");
    for (int i = 0; i < ndims; i++) printf("%s%llu", i ? " x " : "", (unsigned long long)chunk_dims[i]);
    printf("  (%d chunks, %.1f MB each)\n", n_chunks, cmb);
    printf("  Cost w   : %.2f / %.2f / %.2f\n", rank_w0, rank_w1, rank_w2);
    if (n_runs > 1) printf("  Runs     : %d (mean +/- std)\n", n_runs > 32 ? 32 : n_runs);
    printf("  Weights  : %s\n\n", weights_path);

    gpucompress_set_ranking_weights(rank_w0, rank_w1, rank_w2);

    /* ── Set up output paths (rank-suffixed for multi-GPU) ── */
    {
        const char *od = out_dir_override ? out_dir_override : DEFAULT_OUT_DIR;
        snprintf(OUT_DIR, sizeof(OUT_DIR), "%s", od);
        if (g_mpi_size > 1) {
            snprintf(OUT_CSV, sizeof(OUT_CSV), "%s/benchmark_%s_rank%d.csv", od, dataset_name, g_mpi_rank);
            snprintf(OUT_CHUNKS, sizeof(OUT_CHUNKS), "%s/benchmark_%s_chunks_rank%d.csv", od, dataset_name, g_mpi_rank);
            snprintf(OUT_TSTEP, sizeof(OUT_TSTEP), "%s/benchmark_%s_timesteps_rank%d.csv", od, dataset_name, g_mpi_rank);
            snprintf(OUT_TSTEP_CHUNKS, sizeof(OUT_TSTEP_CHUNKS),
                     "%s/benchmark_%s_timestep_chunks_rank%d.csv", od, dataset_name, g_mpi_rank);
            snprintf(OUT_RANKING, sizeof(OUT_RANKING),
                     "%s/benchmark_%s_ranking_rank%d.csv", od, dataset_name, g_mpi_rank);
            snprintf(OUT_RANKING_COSTS, sizeof(OUT_RANKING_COSTS),
                     "%s/benchmark_%s_ranking_costs_rank%d.csv", od, dataset_name, g_mpi_rank);
        } else {
            snprintf(OUT_CSV, sizeof(OUT_CSV), "%s/benchmark_%s.csv", od, dataset_name);
            snprintf(OUT_CHUNKS, sizeof(OUT_CHUNKS), "%s/benchmark_%s_chunks.csv", od, dataset_name);
            snprintf(OUT_TSTEP, sizeof(OUT_TSTEP), "%s/benchmark_%s_timesteps.csv", od, dataset_name);
            snprintf(OUT_TSTEP_CHUNKS, sizeof(OUT_TSTEP_CHUNKS),
                     "%s/benchmark_%s_timestep_chunks.csv", od, dataset_name);
            snprintf(OUT_RANKING, sizeof(OUT_RANKING),
                     "%s/benchmark_%s_ranking.csv", od, dataset_name);
            snprintf(OUT_RANKING_COSTS, sizeof(OUT_RANKING_COSTS),
                     "%s/benchmark_%s_ranking_costs.csv", od, dataset_name);
        }
    }
    mkdirs(OUT_DIR);
    remove(OUT_CHUNKS);

    /* ── Init gpucompress ── */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded from %s\n", weights_path);
        gpucompress_cleanup(); return 1;
    }

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup(); return 1;
    }
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* ── Allocate GPU buffers ── */
    float *d_data = NULL, *d_read = NULL;
    unsigned long long *d_count = NULL;
    cudaMalloc(&d_data, total_bytes);
    cudaMalloc(&d_read, total_bytes);
    cudaMalloc(&d_count, sizeof(unsigned long long));

    if (!d_data || !d_read || !d_count) {
        fprintf(stderr, "FATAL: cudaMalloc failed for %.1f MB\n", dataset_mb);
        gpucompress_cleanup(); return 1;
    }

    /* ── Load first field ── */
    printf("Loading %s... ", fields[0]); fflush(stdout);
    if (load_field(fields[0], d_data, total_bytes, g_ext)) {
        fprintf(stderr, "FATAL: cannot load first field\n");
        cudaFree(d_data); cudaFree(d_read); cudaFree(d_count);
        gpucompress_cleanup(); return 1;
    }
    printf("OK (%.1f MB on GPU)\n\n", dataset_mb);

    PhaseResult results[12];
    int n_phases = 0;
    int any_fail = 0;

    /* ── Phase 1: no-comp ── */
    if (phase_mask & P_NOCOMP) {
        printf("── Phase %d: no-comp ─────────────────────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        if (n_fields > 1) {
            /* Run on all fields and average for fair comparison */
            const int MAX_F = 256;
            double *f_wr = (double*)calloc(MAX_F, sizeof(double));
            double *f_rd = (double*)calloc(MAX_F, sizeof(double));
            double sum_wr = 0, sum_rd = 0, sum_rat = 0;
            double sum_mse = 0, sum_range_sq = 0, sum_meanerr = 0, sum_ssim = 0, sum_bitrate = 0;
            double max_maxerr = 0;
            size_t sum_fb = 0;
            int count = 0;
            for (int fi = 0; fi < n_fields; fi++) {
                if (load_field(fields[fi], d_data, total_bytes, g_ext)) continue;
                PhaseResult fr;
                run_phase_nocomp(d_data, d_read, d_count,
                                 n_floats, ndims, h5_dims, chunk_dims, &fr);
                sum_wr += fr.write_ms;
                sum_rd += fr.read_ms;
                sum_rat += fr.ratio;
                sum_fb += fr.file_bytes;
                if (g_error_bound > 0.0) {
                    sum_mse      += fr.rmse * fr.rmse;
                    if (std::isfinite(fr.psnr_db) && fr.psnr_db < 300.0 && fr.rmse > 0.0)
                        sum_range_sq += fr.rmse * fr.rmse * pow(10.0, fr.psnr_db / 10.0);
                    max_maxerr   = (fr.max_abs_err > max_maxerr) ? fr.max_abs_err : max_maxerr;
                    sum_meanerr += fr.mean_abs_err;
                    sum_ssim    += fr.ssim;
                    sum_bitrate += fr.bit_rate;
                }
                if (count < MAX_F) {
                    f_wr[count] = fr.write_ms;
                    f_rd[count] = fr.read_ms;
                }
                count++;
                const char *fname = strrchr(fields[fi], '/');
                fname = fname ? fname + 1 : fields[fi];
                printf("  field %d/%d: %-30s ratio=%.2fx  write=%.0f MiB/s\n",
                       fi + 1, n_fields, fname, fr.ratio, fr.write_mbps);
            }
            if (count > 0) {
                results[n_phases] = {};
                snprintf(results[n_phases].phase, sizeof(results[n_phases].phase), "no-comp");
                results[n_phases].write_ms = sum_wr / count;
                results[n_phases].read_ms = sum_rd / count;
                results[n_phases].ratio = sum_rat / count;
                results[n_phases].file_bytes = sum_fb / count;
                results[n_phases].orig_bytes = total_bytes;
                results[n_phases].write_mbps = (double)total_bytes / (1 << 20) / (results[n_phases].write_ms / 1000.0);
                results[n_phases].read_mbps = (double)total_bytes / (1 << 20) / (results[n_phases].read_ms / 1000.0);
                results[n_phases].n_runs = count;
                results[n_phases].n_chunks = n_chunks;
                if (g_error_bound > 0.0) {
                    double avg_mse = sum_mse / count;
                    results[n_phases].rmse          = sqrt(avg_mse);
                    results[n_phases].psnr_db       = (avg_mse > 0.0 && sum_range_sq > 0.0)
                        ? 10.0 * log10((sum_range_sq / count) / avg_mse) : INFINITY;
                    results[n_phases].max_abs_err   = max_maxerr;
                    results[n_phases].mean_abs_err  = sum_meanerr / count;
                    results[n_phases].ssim          = sum_ssim / count;
                    results[n_phases].bit_rate      = sum_bitrate / count;
                }
                /* Compute std-dev across fields (sample std, n-1) */
                int n = (count < MAX_F) ? count : MAX_F;
                double var_w = 0, var_r = 0;
                for (int i = 0; i < n; i++) {
                    var_w += (f_wr[i] - results[n_phases].write_ms) * (f_wr[i] - results[n_phases].write_ms);
                    var_r += (f_rd[i] - results[n_phases].read_ms)  * (f_rd[i] - results[n_phases].read_ms);
                }
                double denom = (n > 1) ? (n - 1) : 1;
                results[n_phases].write_ms_std = sqrt(var_w / denom);
                results[n_phases].read_ms_std  = sqrt(var_r / denom);
            }
            free(f_wr); free(f_rd);
        } else {
            PhaseResult runs_buf[32];
            int eff_runs = (n_runs > 32) ? 32 : n_runs;
            for (int run = 0; run < eff_runs; run++) {
                if (eff_runs > 1) printf("  Run %d/%d\n", run + 1, eff_runs);
                run_phase_nocomp(d_data, d_read, d_count,
                                 n_floats, ndims, h5_dims, chunk_dims,
                                 &runs_buf[run]);
            }
            if (eff_runs > 1)
                merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
            else
                results[n_phases] = runs_buf[0];
            results[n_phases].n_runs = eff_runs;
            results[n_phases].n_chunks = n_chunks;
        }
        n_phases++;
    }

    /* ── Phases: fixed-algo baselines ── */
    {
        struct FixedAlgoPhase {
            unsigned int mask;
            const char *name;
            const char *tmp_file;
            gpucompress_algorithm_t algo;
            unsigned int preproc;
        };
        FixedAlgoPhase fixed_phases[] = {
            { P_FIX_LZ4,     "fixed-lz4",           TMP_FIX_LZ4,                    GPUCOMPRESS_ALGO_LZ4,      0 },
            { P_FIX_SNAPPY,  "fixed-snappy",         TMP_FIX_SNAPPY,                 GPUCOMPRESS_ALGO_SNAPPY,   0 },
            { P_FIX_DEFL,    "fixed-deflate",        TMP_FIX_DEFL,                   GPUCOMPRESS_ALGO_DEFLATE,  0 },
            { P_FIX_GDEFL,   "fixed-gdeflate",       TMP_FIX_GDEFL,                  GPUCOMPRESS_ALGO_GDEFLATE, 0 },
            { P_FIX_ZSTD,    "fixed-zstd",           TMP_FIX_ZSTD,                   GPUCOMPRESS_ALGO_ZSTD,     0 },
            { P_FIX_ANS,     "fixed-ans",            TMP_FIX_ANS,                    GPUCOMPRESS_ALGO_ANS,      0 },
            { P_FIX_CASC,    "fixed-cascaded",       TMP_FIX_CASC,                   GPUCOMPRESS_ALGO_CASCADED, 0 },
            { P_FIX_BITCOMP, "fixed-bitcomp",        TMP_FIX_BITCOMP,                GPUCOMPRESS_ALGO_BITCOMP,  0 },
        };
        for (int fi = 0; fi < 8; fi++) {
            if (!(phase_mask & fixed_phases[fi].mask)) continue;
            printf("\n── Phase %d: %s ─────────────────────────────────────\n",
                   n_phases + 1, fixed_phases[fi].name);
            gpucompress_disable_online_learning();
            gpucompress_set_exploration(0);
            hid_t dcpl_f = make_dcpl_fixed(ndims, chunk_dims, fixed_phases[fi].algo, fixed_phases[fi].preproc);
            if (n_fields > 1) {
                /* Run on all fields and average for fair comparison */
                run_phase_all_fields(run_phase_vol, d_data, d_read, d_count,
                    n_floats, total_bytes, ndims, h5_dims, chunk_dims,
                    fixed_phases[fi].name, fixed_phases[fi].tmp_file, dcpl_f,
                    fields, n_fields,
                    &results[n_phases], &any_fail);
            } else {
                PhaseResult runs_buf[32];
                int eff_runs = (n_runs > 32) ? 32 : n_runs;
                for (int run = 0; run < eff_runs; run++) {
                    if (eff_runs > 1) printf("  Run %d/%d\n", run + 1, eff_runs);
                    int rc = run_phase_vol(d_data, d_read, d_count,
                                           n_floats, ndims, h5_dims, chunk_dims,
                                           fixed_phases[fi].name, fixed_phases[fi].tmp_file,
                                           dcpl_f, &runs_buf[run]);
                    if (rc) any_fail = 1;
                }
                if (eff_runs > 1)
                    merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
                else
                    results[n_phases] = runs_buf[0];
                results[n_phases].n_runs = eff_runs;
            }
            results[n_phases].n_chunks = n_chunks;
            H5Pclose(dcpl_f);
            n_phases++;
        }
    }

    /* ── Phase: nn ── */
    if (phase_mask & P_NN) {
        printf("\n── Phase %d: nn ──────────────────────────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_nn = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);
        if (n_fields > 1) {
            run_phase_all_fields(run_phase_vol, d_data, d_read, d_count,
                n_floats, total_bytes, ndims, h5_dims, chunk_dims,
                "nn", TMP_NN, dcpl_nn,
                fields, n_fields,
                &results[n_phases], &any_fail);
        } else {
            PhaseResult runs_buf[32];
            int eff_runs = (n_runs > 32) ? 32 : n_runs;
            for (int run = 0; run < eff_runs; run++) {
                if (eff_runs > 1) printf("  Run %d/%d\n", run + 1, eff_runs);
                int rc = run_phase_vol(d_data, d_read, d_count,
                                       n_floats, ndims, h5_dims, chunk_dims,
                                       "nn", TMP_NN, dcpl_nn, &runs_buf[run]);
                if (rc) any_fail = 1;
            }
            if (eff_runs > 1)
                merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
            else
                results[n_phases] = runs_buf[0];
            results[n_phases].n_runs = eff_runs;
        }
        results[n_phases].n_chunks = n_chunks;
        H5Pclose(dcpl_nn);
        write_chunk_csv("nn");
        n_phases++;
    }

    /* ── nn-rl / nn-rl+exp50: SGD learns per-chunk within each field ── */
    if (phase_mask & (P_NNRL | P_NNRLEXP)) {
        struct TsPhase { const char *name; int sgd; int explore; unsigned int mask; };
        TsPhase ts_all[] = {
            { "nn-rl",       1, 0, P_NNRL },
            { "nn-rl+exp50", 1, 1, P_NNRLEXP },
        };

        hid_t dcpl_ts = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);

        /* Open timestep CSV */
        FILE *ts_csv = fopen(OUT_TSTEP, "w");
        if (ts_csv) {
            fprintf(ts_csv, "rank,phase,field_idx,field_name,write_ms,read_ms,ratio,"
                    "mape_ratio,mape_comp,mape_decomp,mape_psnr,"
                    "sgd_fires,explorations,n_chunks,mismatches,"
                    "write_mbps,read_mbps,"
                    "file_bytes,"
                    "stats_ms,nn_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                    "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
                    "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
                    "vol_s2_busy_ms,vol_s3_busy_ms,"
                    "h5dwrite_ms,cuda_sync_ms,h5dclose_ms,h5fclose_ms,"
                    "vol_setup_ms,vol_pipeline_ms,"
                    "psnr_db,rmse,max_abs_err,mean_abs_err,ssim,bit_rate,"
                    "r2_ratio,r2_comp,r2_decomp,r2_psnr\n");
        }

        /* Open timestep-chunks CSV for milestone fields */
        FILE *tc_csv = fopen(OUT_TSTEP_CHUNKS, "w");
        if (tc_csv) {
            fprintf(tc_csv, "rank,phase,field_idx,field_name,chunk,action,action_orig,"
                    "predicted_ratio,actual_ratio,final_ratio,"
                    "predicted_comp_ms,actual_comp_ms_raw,"
                    "predicted_decomp_ms,actual_decomp_ms_raw,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fired,exploration_triggered,"
                    "feat_entropy,feat_mad,feat_deriv\n");
        }
        /* Open ranking quality CSV (skip if NO_RANKING=1) */
        bool skip_ranking = false;
        { const char* nr = getenv("NO_RANKING"); if (nr && atoi(nr)) skip_ranking = true; }
        if (skip_ranking)
            printf("  [NO_RANKING=1] Kendall tau ranking profiler disabled.\n");
        FILE *ranking_csv = skip_ranking ? nullptr : fopen(OUT_RANKING, "w");
        if (ranking_csv)
            write_ranking_csv_header(ranking_csv);
        FILE *ranking_costs_csv = skip_ranking ? nullptr : fopen(OUT_RANKING_COSTS, "w");
        if (ranking_costs_csv)
            write_ranking_costs_csv_header(ranking_costs_csv);

        /* Milestone indices: 0%, 25%, 50%, 75%, last */
        int milestones[5] = {0, n_fields/4, n_fields/2, 3*n_fields/4, n_fields-1};

        for (int pi = 0; pi < 2; pi++) {
            if (!(phase_mask & ts_all[pi].mask)) continue;

            const char *phase_name = ts_all[pi].name;

            printf("\n══════════════════════════════════════════════════════════════\n");
            printf("  Multi-field [%s]: %d fields (SGD=%s, Explore=%s)\n",
                   phase_name, n_fields,
                   ts_all[pi].sgd ? "on" : "off",
                   ts_all[pi].explore ? "on" : "off");
            printf("══════════════════════════════════════════════════════════════\n\n");

            /* Reload NN for fresh start */
            gpucompress_reload_nn(weights_path);
            gpucompress_flush_manager_cache();

            if (ts_all[pi].sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, sgd_mape);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(ts_all[pi].explore);
            if (ts_all[pi].explore) {
                gpucompress_set_exploration_threshold(explore_thresh);
                gpucompress_set_exploration_k(explore_k);
            }

            printf("  %-4s  %-30s  %-7s  %-7s  %-7s  %-8s  %-4s  %-4s\n",
                   "F#", "Field", "WrMs", "RdMs", "Ratio", "MAPE_R", "SGD", "EXP");
            printf("  ----  %-30s  -------  -------  -------  --------  ----  ----\n", "-----");

            const int WARMUP_SKIP = 0;
            const int MAX_F = 256;
            double *f_write_ms = (double*)calloc(MAX_F, sizeof(double));
            double *f_read_ms  = (double*)calloc(MAX_F, sizeof(double));
            double *f_cgbps    = (double*)calloc(MAX_F, sizeof(double));
            double *f_dgbps    = (double*)calloc(MAX_F, sizeof(double));
            double sum_write_ms = 0, sum_read_ms = 0;
            double sum_ratio = 0, sum_file_sz = 0;
            double sum_mape_r = 0, sum_mape_c = 0, sum_mape_d = 0, sum_mape_p = 0;
            double sum_mae_r = 0, sum_mae_c = 0, sum_mae_d = 0, sum_mae_p = 0;
            double sum_r2_r = 0, sum_r2_c = 0, sum_r2_d = 0, sum_r2_p = 0;
            double sum_nn_ms = 0, sum_stats_ms = 0, sum_preproc_ms = 0;
            double sum_comp_ms = 0, sum_decomp_ms = 0, sum_explore_ms = 0, sum_sgd_ms = 0;
            double sum_comp_gbps = 0, sum_decomp_gbps = 0;
            double sum_mse = 0, sum_range_sq = 0, sum_maxerr = 0, sum_bitrate = 0;
            double sum_meanerr = 0, sum_ssim = 0;
            int    sum_sgd = 0, sum_expl = 0;
            size_t last_file_sz = 0;
            int n_steady = 0;

            for (int fi = 0; fi < n_fields; fi++) {
                /* Load field to GPU */
                if (load_field(fields[fi], d_data, total_bytes, g_ext)) {
                    printf("  %-4d  SKIP (load failed)\n", fi);
                    continue;
                }

                gpucompress_flush_manager_cache();  /* cold-start each field, matching single-shot phases */
                gpucompress_reset_chunk_history();
                gpucompress_set_debug_context(phase_name, fi);
                H5VL_gpucompress_reset_stats();
                remove(TMP_NN_RL);

                hid_t fapl = make_vol_fapl();
                hid_t file = H5Fcreate(TMP_NN_RL, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
                H5Pclose(fapl);
                hid_t fsp  = H5Screate_simple(ndims, h5_dims, NULL);
                hid_t dset = H5Dcreate2(file, "field", H5T_NATIVE_FLOAT,
                                         fsp, H5P_DEFAULT, dcpl_ts, H5P_DEFAULT);
                H5Sclose(fsp);

                double tw0 = now_ms();
                H5Dwrite(dset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
                double tw_after_write = now_ms();
                cudaDeviceSynchronize();
                double tw_after_sync = now_ms();
                H5Dclose(dset);
                double tw_after_dclose = now_ms();
                H5Fclose(file);
                double tw1 = now_ms();

                /* Fine-grained write path breakdown */
                double h5dwrite_ms_t  = tw_after_write - tw0;
                double cuda_sync_ms_t = tw_after_sync - tw_after_write;
                double h5dclose_ms_t  = tw_after_dclose - tw_after_sync;
                double h5fclose_ms_t  = tw1 - tw_after_dclose;

                /* VOL additive pipeline timing + busy times */
                double vol_s1 = 0, vol_drain = 0, vol_io_drain = 0, vol_total = 0;
                H5VL_gpucompress_get_stage_timing(&vol_s1, &vol_drain, &vol_io_drain, &vol_total);
                double vol_s2_busy = 0, vol_s3_busy = 0;
                H5VL_gpucompress_get_busy_timing(&vol_s2_busy, &vol_s3_busy);
                double vol_setup = 0;
                H5VL_gpucompress_get_vol_func_timing(&vol_setup, NULL);

                /* Always read back for timing; only validate if do_verify */
                drop_pagecache(TMP_NN_RL);
                fapl = make_vol_fapl();
                file = H5Fopen(TMP_NN_RL, H5F_ACC_RDONLY, fapl);
                H5Pclose(fapl);
                dset = H5Dopen2(file, "field", H5P_DEFAULT);

                double tr0 = now_ms();
                H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
                cudaDeviceSynchronize();
                H5Dclose(dset); H5Fclose(file);
                double tr1 = now_ms();
                double read_ms_t = tr1 - tr0;

                unsigned long long mm = 0;
                if (do_verify) {
                    mm = gpu_compare(d_data, d_read, n_floats, d_count);
                }
                double field_psnr = 0.0, field_rmse = 0.0, field_maxerr = 0.0;
                double field_meanerr = 0.0, field_ssim = 1.0;
                if (g_error_bound > 0.0) {
                    field_psnr = vpic_compute_quality_gpu(d_data, d_read, n_floats,
                                                         &field_rmse, &field_maxerr,
                                                         &field_meanerr, &field_ssim);
                }
                size_t file_sz = get_file_size(TMP_NN_RL);
                double ratio_t = (file_sz > 0) ? (double)total_bytes / (double)file_sz : 1.0;
                double write_ms_t = tw1 - tw0;
                double field_bitrate = (file_sz > 0) ? (double)(file_sz * 8) / (double)n_floats : 0.0;

                /* Collect MAPE for this field */
                PhaseResult field_r;
                memset(&field_r, 0, sizeof(field_r));
                field_r.orig_bytes = total_bytes;
                collect_chunk_metrics(&field_r);

                if (fi >= WARMUP_SKIP) {
                    sum_write_ms += write_ms_t;
                    sum_read_ms  += read_ms_t;
                    sum_ratio    += ratio_t;
                    sum_file_sz  += (double)file_sz;
                    last_file_sz  = file_sz;
                    if (n_steady < MAX_F) {
                        f_write_ms[n_steady] = write_ms_t;
                        f_read_ms[n_steady]  = read_ms_t;
                        f_cgbps[n_steady]    = field_r.comp_gbps;
                        f_dgbps[n_steady]    = field_r.decomp_gbps;
                    }
                    n_steady++;
                    sum_mape_r += field_r.mape_ratio_pct;
                    sum_mape_c += field_r.mape_comp_pct;
                    sum_mape_d += field_r.mape_decomp_pct;
                    sum_mape_p += field_r.mape_psnr_pct;
                    sum_mae_r  += field_r.mae_ratio;
                    sum_mae_c  += field_r.mae_comp_ms;
                    sum_mae_d  += field_r.mae_decomp_ms;
                    sum_mae_p  += field_r.mae_psnr_db;
                    sum_r2_r   += field_r.r2_ratio;
                    sum_r2_c   += field_r.r2_comp;
                    sum_r2_d   += field_r.r2_decomp;
                    sum_r2_p   += field_r.r2_psnr;
                    sum_nn_ms       += field_r.nn_ms;
                    sum_stats_ms    += field_r.stats_ms;
                    sum_preproc_ms  += field_r.preproc_ms;
                    sum_comp_ms     += field_r.comp_ms;
                    sum_decomp_ms   += field_r.decomp_ms;
                    sum_explore_ms  += field_r.explore_ms;
                    sum_sgd_ms      += field_r.sgd_ms;
                    sum_comp_gbps   += field_r.comp_gbps;
                    sum_decomp_gbps += field_r.decomp_gbps;
                    sum_sgd    += field_r.sgd_fires;
                    sum_expl   += field_r.explorations;
                    sum_mse    += field_rmse * field_rmse;
                    if (std::isfinite(field_psnr) && field_psnr < 300.0 && field_rmse > 0.0)
                        sum_range_sq += field_rmse * field_rmse * pow(10.0, field_psnr / 10.0);
                    sum_maxerr  = (field_maxerr > sum_maxerr) ? field_maxerr : sum_maxerr;
                    sum_meanerr += field_meanerr;
                    sum_ssim    += field_ssim;
                    sum_bitrate += field_bitrate;
                }

                /* Extract just the filename */
                const char *fname = strrchr(fields[fi], '/');
                fname = fname ? fname + 1 : fields[fi];

                printf("  %-4d  %-30s  %7.0f  %7.0f  %7.2f  %8.1f  %4d  %4d",
                       fi, fname, write_ms_t, read_ms_t, ratio_t,
                       field_r.mape_ratio_pct,
                       field_r.sgd_fires, field_r.explorations);
                if (mm > 0) printf("  MISMATCH=%llu", mm);
                printf("\n");

                /* Write to timestep CSV */
                if (ts_csv) {
                    double wr_mbps = (write_ms_t > 0) ? (double)total_bytes / (1 << 20) / (write_ms_t / 1000.0) : 0;
                    double rd_mbps = (read_ms_t > 0) ? (double)total_bytes / (1 << 20) / (read_ms_t / 1000.0) : 0;
                    fprintf(ts_csv, "%d,%s,%d,%s,%.2f,%.2f,%.4f,"
                            "%.2f,%.2f,%.2f,%.2f,"
                            "%d,%d,%d,%llu,%.1f,%.1f,%zu,"
                            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                            "%.4f,%.4f,%.4f,%.4f,"
                            "%.3f,%.3f,%.3f,"
                            "%.3f,%.3f,"
                            "%.3f,%.3f,%.3f,%.3f,"
                            "%.3f,%.3f,"
                            "%.2f,%.6f,%.6f,%.6f,%.8f,%.4f,"
                            "%.4f,%.4f,%.4f,%.4f\n",
                            g_mpi_rank, phase_name, fi, fname,
                            write_ms_t, read_ms_t, ratio_t,
                            field_r.mape_ratio_pct, field_r.mape_comp_pct,
                            field_r.mape_decomp_pct, field_r.mape_psnr_pct,
                            field_r.sgd_fires, field_r.explorations,
                            n_chunks, mm, wr_mbps, rd_mbps, file_sz,
                            field_r.stats_ms, field_r.nn_ms, field_r.preproc_ms,
                            field_r.comp_ms, field_r.decomp_ms, field_r.explore_ms, field_r.sgd_ms,
                            field_r.mae_ratio, field_r.mae_comp_ms, field_r.mae_decomp_ms,
                            field_r.mae_psnr_db,
                            vol_s1, vol_drain, vol_io_drain,
                            vol_s2_busy, vol_s3_busy,
                            h5dwrite_ms_t, cuda_sync_ms_t, h5dclose_ms_t, h5fclose_ms_t,
                            vol_setup, vol_total,
                            field_psnr, field_rmse, field_maxerr,
                            field_meanerr, field_ssim, field_bitrate,
                            field_r.r2_ratio, field_r.r2_comp, field_r.r2_decomp, field_r.r2_psnr);
                }

                /* Write per-chunk detail at milestone fields */
                bool is_milestone = false;
                for (int mi = 0; mi < 5; mi++)
                    if (fi == milestones[mi]) { is_milestone = true; break; }
                if (is_milestone && tc_csv) {
                    int nh = gpucompress_get_chunk_history_count();
                    for (int ci = 0; ci < nh; ci++) {
                        gpucompress_chunk_diag_t d;
                        if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;
                        double mr = 0, mc = 0, md = 0;
                        if (d.actual_ratio > 0 && d.predicted_ratio > 0)
                            mr = fmin(fabs(d.predicted_ratio-d.actual_ratio)/fabs(d.actual_ratio)*100.0, 200.0);
                        if (d.compression_ms > 0 && d.predicted_comp_time > 0)
                            mc = fmin(fabs(d.predicted_comp_time-d.compression_ms)/fabs(d.compression_ms)*100.0, 200.0);
                        if (d.decompression_ms > 0 && d.predicted_decomp_time > 0)
                            md = fmin(fabs(d.predicted_decomp_time-d.decompression_ms)/fabs(d.decompression_ms)*100.0, 200.0);
                        fprintf(tc_csv, "%d,%s,%d,%s,%d,%d,%d,"
                                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                                "%.2f,%.2f,%.2f,"
                                "%d,%d,%.4f,%.6f,%.6f\n",
                                g_mpi_rank, phase_name, fi, fname, ci, d.nn_action, d.nn_original_action,
                                d.predicted_ratio, d.actual_ratio, d.final_ratio,
                                d.predicted_comp_time, d.compression_ms_raw,
                                d.predicted_decomp_time, d.decompression_ms_raw,
                                mr, mc, md,
                                d.sgd_fired, d.exploration_triggered,
                                d.feat_entropy, d.feat_mad, d.feat_deriv);
                    }
                }

                /* Kendall τ ranking quality at milestones */
                if (ranking_csv && is_ranking_milestone(fi, n_fields)) {
                    size_t sdr_chunk_bytes = slice_floats * last_chunk * sizeof(float);
                    float bw = gpucompress_get_bandwidth_bytes_per_ms();
                    RankingMilestoneResult tau_result = {};
                    run_ranking_profiler(
                        d_data, total_bytes, sdr_chunk_bytes,
                        error_bound, rank_w0, rank_w1, rank_w2, bw,
                        3, ranking_csv, ranking_costs_csv,
                        phase_name, fi, &tau_result);
                    printf("    [τ] F=%d: τ=%.3f  regret=%.3fx  (%.0fms)\n",
                           fi, tau_result.mean_tau,
                           tau_result.mean_regret,
                           tau_result.profiling_ms);
                }
            }

            /* Store final result for summary */
            PhaseResult *pr = &results[n_phases];
            memset(pr, 0, sizeof(*pr));
            snprintf(pr->phase, sizeof(pr->phase), "%s", phase_name);
            pr->orig_bytes = total_bytes;
            pr->n_chunks = n_chunks;
            pr->n_runs = n_steady;
            if (n_steady > 0) {
                pr->write_ms = sum_write_ms / n_steady;
                pr->read_ms  = sum_read_ms / n_steady;
                pr->write_mbps = (double)total_bytes / (1 << 20) / (pr->write_ms / 1000.0);
                pr->read_mbps  = (double)total_bytes / (1 << 20) / (pr->read_ms / 1000.0);
                pr->file_bytes = (size_t)(sum_file_sz / n_steady);
                pr->ratio      = (sum_file_sz > 0)
                    ? (double)(n_steady * total_bytes) / sum_file_sz : 1.0;
                pr->mape_ratio_pct  = sum_mape_r / n_steady;
                pr->mape_comp_pct   = sum_mape_c / n_steady;
                pr->mape_decomp_pct = sum_mape_d / n_steady;
                pr->mape_psnr_pct   = sum_mape_p / n_steady;
                pr->mae_ratio       = sum_mae_r / n_steady;
                pr->mae_comp_ms     = sum_mae_c / n_steady;
                pr->mae_decomp_ms   = sum_mae_d / n_steady;
                pr->mae_psnr_db     = sum_mae_p / n_steady;
                pr->r2_ratio        = sum_r2_r / n_steady;
                pr->r2_comp         = sum_r2_c / n_steady;
                pr->r2_decomp       = sum_r2_d / n_steady;
                pr->r2_psnr         = sum_r2_p / n_steady;
                pr->nn_ms       = sum_nn_ms / n_steady;
                pr->stats_ms    = sum_stats_ms / n_steady;
                pr->preproc_ms  = sum_preproc_ms / n_steady;
                pr->comp_ms     = sum_comp_ms / n_steady;
                pr->decomp_ms   = sum_decomp_ms / n_steady;
                pr->explore_ms  = sum_explore_ms / n_steady;
                pr->sgd_ms      = sum_sgd_ms / n_steady;
                pr->comp_gbps   = sum_comp_gbps / n_steady;
                pr->decomp_gbps = sum_decomp_gbps / n_steady;
                pr->sgd_fires   = sum_sgd;
                pr->explorations = sum_expl;
                if (g_error_bound > 0.0) {
                    double avg_mse   = sum_mse / n_steady;
                    pr->rmse         = sqrt(avg_mse);
                    pr->psnr_db      = (avg_mse > 0.0 && sum_range_sq > 0.0)
                        ? 10.0 * log10((sum_range_sq / n_steady) / avg_mse) : INFINITY;
                    pr->max_abs_err  = sum_maxerr;  /* max across fields, not average */
                    pr->mean_abs_err = sum_meanerr / n_steady;
                    pr->ssim         = sum_ssim / n_steady;
                    pr->bit_rate     = sum_bitrate / n_steady;
                }

                /* Compute std across steady-state fields */
                int ns = (n_steady < MAX_F) ? n_steady : MAX_F;
                if (ns > 1) {
                    double var_w = 0, var_r = 0, var_cg = 0, var_dg = 0;
                    for (int i = 0; i < ns; i++) {
                        var_w  += (f_write_ms[i] - pr->write_ms) * (f_write_ms[i] - pr->write_ms);
                        var_r  += (f_read_ms[i]  - pr->read_ms)  * (f_read_ms[i]  - pr->read_ms);
                        var_cg += (f_cgbps[i] - pr->comp_gbps) * (f_cgbps[i] - pr->comp_gbps);
                        var_dg += (f_dgbps[i] - pr->decomp_gbps) * (f_dgbps[i] - pr->decomp_gbps);
                    }
                    double d = ns - 1;
                    pr->write_ms_std    = sqrt(var_w / d);
                    pr->read_ms_std     = sqrt(var_r / d);
                    pr->comp_gbps_std   = sqrt(var_cg / d);
                    pr->decomp_gbps_std = sqrt(var_dg / d);
                }
            }
            n_phases++;
            free(f_write_ms); free(f_read_ms); free(f_cgbps); free(f_dgbps);
        }

        H5Pclose(dcpl_ts);
        if (ts_csv) fclose(ts_csv);
        if (tc_csv) fclose(tc_csv);
        if (ranking_csv) { fclose(ranking_csv); printf("  Ranking quality CSV: %s\n", OUT_RANKING); }
        if (ranking_costs_csv) { fclose(ranking_costs_csv); printf("  Ranking costs CSV: %s\n", OUT_RANKING_COSTS); }
    }

    /* ── Write summary CSV ── */
    write_summary_csv(dataset_name, results, n_phases, total_bytes, n_chunks);

    /* ── Cleanup ── */
    cudaFree(d_data);
    cudaFree(d_read);
    cudaFree(d_count);
    remove(TMP_NOCOMP);
    remove(TMP_FIX_LZ4);
    remove(TMP_FIX_SNAPPY);
    remove(TMP_FIX_DEFL);
    remove(TMP_FIX_GDEFL);
    remove(TMP_FIX_ZSTD);
    remove(TMP_FIX_ANS);
    remove(TMP_FIX_CASC);
    remove(TMP_FIX_BITCOMP);
    remove(TMP_NN);
    remove(TMP_NN_RL);
    remove(TMP_NN_RLEXP);

    H5VLclose(vol_id);
    gpucompress_cleanup();

    if (g_mpi_rank == 0)
        printf("\nBenchmark %s\n", any_fail ? "FAILED" : "PASSED");

#ifdef GPUCOMPRESS_USE_MPI
    MPI_Finalize();
#endif
    return any_fail;
}
