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
 *                    Valid: no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd, entropy-heuristic, best, nn, nn-rl, nn-rl+exp50
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

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define DEFAULT_CHUNK_MB    4
#define DEFAULT_EXT         ".f32"

#define REINFORCE_LR        0.1f
#define REINFORCE_MAPE      0.10f

#define TMP_NOCOMP    "/tmp/bm_generic_nocomp.h5"
#define TMP_FIX_LZ4   "/tmp/bm_generic_fix_lz4.h5"
#define TMP_FIX_GDEFL "/tmp/bm_generic_fix_gdefl.h5"
#define TMP_FIX_ZSTD  "/tmp/bm_generic_fix_zstd.h5"
#define TMP_HEUR      "/tmp/bm_generic_heur.h5"
#define TMP_NN        "/tmp/bm_generic_nn.h5"
#define TMP_NN_RL     "/tmp/bm_generic_nn_rl.h5"
#define TMP_NN_RLEXP  "/tmp/bm_generic_nn_rlexp.h5"

#define DEFAULT_OUT_DIR "benchmarks/sdrbench/results"
#define MAX_FIELDS      128
#define MAX_PATH_LEN    512

/* Phase bitmask */
#define P_NOCOMP      0x001
#define P_FIX_LZ4     0x002
#define P_FIX_GDEFL   0x004
#define P_FIX_ZSTD    0x008
#define P_HEUR        0x010
#define P_BEST        0x020
#define P_NN          0x040
#define P_NNRL        0x080
#define P_NNRLEXP     0x100

/* Output paths */
static char OUT_DIR[MAX_PATH_LEN];
static char OUT_CSV[MAX_PATH_LEN];
static char OUT_CHUNKS[MAX_PATH_LEN];
static char OUT_TSTEP[MAX_PATH_LEN];
static char OUT_TSTEP_CHUNKS[MAX_PATH_LEN];

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
    double smape_ratio_pct;
    double smape_comp_pct;
    double smape_decomp_pct;
    double mape_ratio_pct;
    double mape_comp_pct;
    double mape_decomp_pct;
    double comp_gbps;
    double decomp_gbps;
    double write_ms_std;
    double read_ms_std;
    double comp_gbps_std;
    double decomp_gbps_std;
    int    n_runs;
} PhaseResult;

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
    double ape_r = 0, ape_c = 0, ape_d = 0;
    double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0;
    int cnt_r = 0, cnt_c = 0, cnt_d = 0;
    int mcnt_r = 0, mcnt_c = 0, mcnt_d = 0;
    int sgd_t = 0, expl_t = 0;

    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
        if (diag.sgd_fired) sgd_t++;
        if (diag.exploration_triggered) expl_t++;

        /* sMAPE */
        if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
            double denom = (fabs(diag.actual_ratio) + fabs(diag.predicted_ratio)) / 2.0;
            if (denom > 0) { ape_r += fabs(diag.actual_ratio - diag.predicted_ratio) / denom; cnt_r++; }
        }
        if (diag.compression_ms > 0 && diag.predicted_comp_time > 0) {
            double denom = (fabs(diag.compression_ms) + fabs(diag.predicted_comp_time)) / 2.0;
            if (denom > 0) { ape_c += fabs(diag.compression_ms - diag.predicted_comp_time) / denom; cnt_c++; }
        }
        if (diag.decompression_ms > 0 && diag.predicted_decomp_time > 0) {
            double denom = (fabs(diag.decompression_ms) + fabs(diag.predicted_decomp_time)) / 2.0;
            if (denom > 0) { ape_d += fabs(diag.decompression_ms - diag.predicted_decomp_time) / denom; cnt_d++; }
        }

        /* MAPE */
        if (diag.actual_ratio > 0) {
            mape_r_sum += fabs(diag.predicted_ratio - diag.actual_ratio) / fabs(diag.actual_ratio);
            mcnt_r++;
        }
        if (diag.compression_ms > 0) {
            mape_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms);
            mcnt_c++;
        }
        if (diag.decompression_ms > 0) {
            mape_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms);
            mcnt_d++;
        }
    }

    r->sgd_fires   = sgd_t;
    r->explorations = expl_t;
    r->smape_ratio_pct  = cnt_r ? (ape_r / cnt_r) * 100.0 : 0.0;
    r->smape_comp_pct   = cnt_c ? (ape_c / cnt_c) * 100.0 : 0.0;
    r->smape_decomp_pct = cnt_d ? (ape_d / cnt_d) * 100.0 : 0.0;
    r->mape_ratio_pct   = mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0;
    r->mape_comp_pct    = mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0;
    r->mape_decomp_pct  = mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0;

    /* Compute isolated compression throughput from per-chunk timing.
     * orig_bytes is set by the caller; total comp/decomp time is summed
     * from chunk diagnostics. */
    double total_comp_ms = 0, total_decomp_ms = 0;
    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
        total_comp_ms   += diag.compression_ms;
        total_decomp_ms += diag.decompression_ms;
    }
    r->comp_gbps   = (total_comp_ms > 0)
        ? (double)r->orig_bytes / total_comp_ms / 1e6 : 0.0;
    r->decomp_gbps = (total_decomp_ms > 0)
        ? (double)r->orig_bytes / total_decomp_ms / 1e6 : 0.0;
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
    out->write_ms_std   = sqrt(var_w / n);
    out->read_ms_std    = sqrt(var_r / n);
    out->comp_gbps_std  = sqrt(var_cg / n);
    out->decomp_gbps_std = sqrt(var_dg / n);
    out->n_runs = n;
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

    fprintf(f, "dataset,phase,write_ms,read_ms,ratio,file_bytes,orig_bytes,"
            "write_mbps,read_mbps,mismatches,"
            "smape_ratio,smape_comp,smape_decomp,"
            "mape_ratio,mape_comp,mape_decomp,"
            "sgd_fires,explorations,n_chunks,"
            "comp_gbps,decomp_gbps,"
            "write_ms_std,read_ms_std,comp_gbps_std,decomp_gbps_std,n_runs\n");

    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &results[i];
        fprintf(f, "%s,%s,%.2f,%.2f,%.4f,%zu,%zu,"
                "%.1f,%.1f,%llu,"
                "%.2f,%.2f,%.2f,"
                "%.2f,%.2f,%.2f,"
                "%d,%d,%d,"
                "%.4f,%.4f,"
                "%.2f,%.2f,%.4f,%.4f,%d\n",
                dataset_name, r->phase, r->write_ms, r->read_ms,
                r->ratio, r->file_bytes, r->orig_bytes,
                r->write_mbps, r->read_mbps, r->mismatches,
                r->smape_ratio_pct, r->smape_comp_pct, r->smape_decomp_pct,
                r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct,
                r->sgd_fires, r->explorations, r->n_chunks,
                r->comp_gbps, r->decomp_gbps,
                r->write_ms_std, r->read_ms_std,
                r->comp_gbps_std, r->decomp_gbps_std, r->n_runs);
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
                "predicted_ratio,actual_ratio,"
                "predicted_comp_ms,actual_comp_ms,"
                "predicted_decomp_ms,actual_decomp_ms,"
                "smape_ratio,smape_comp,smape_decomp,"
                "mape_ratio,mape_comp,mape_decomp,"
                "sgd_fired,exploration_triggered\n");
    }

    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;

        double sr = 0, sc = 0, sd = 0, mr = 0, mc = 0, md = 0;
        if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
            double denom = (fabs(diag.actual_ratio) + fabs(diag.predicted_ratio)) / 2.0;
            if (denom > 0) sr = fabs(diag.actual_ratio - diag.predicted_ratio) / denom * 100.0;
        }
        if (diag.compression_ms > 0 && diag.predicted_comp_time > 0) {
            double denom = (fabs(diag.compression_ms) + fabs(diag.predicted_comp_time)) / 2.0;
            if (denom > 0) sc = fabs(diag.compression_ms - diag.predicted_comp_time) / denom * 100.0;
        }
        if (diag.decompression_ms > 0 && diag.predicted_decomp_time > 0) {
            double denom = (fabs(diag.decompression_ms) + fabs(diag.predicted_decomp_time)) / 2.0;
            if (denom > 0) sd = fabs(diag.decompression_ms - diag.predicted_decomp_time) / denom * 100.0;
        }
        if (diag.actual_ratio > 0)
            mr = fabs(diag.predicted_ratio - diag.actual_ratio) / fabs(diag.actual_ratio) * 100.0;
        if (diag.compression_ms > 0)
            mc = fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms) * 100.0;
        if (diag.decompression_ms > 0)
            md = fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms) * 100.0;

        fprintf(f, "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
                phase_name, ci, diag.nn_action,
                diag.predicted_ratio, diag.actual_ratio,
                diag.predicted_comp_time, diag.compression_ms,
                diag.predicted_decomp_time, diag.decompression_ms,
                sr, sc, sd, mr, mc, md,
                diag.sgd_fired, diag.exploration_triggered);
    }
    fclose(f);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    /* ── Parse arguments ── */
    const char *weights_path = NULL;
    const char *data_dir     = NULL;
    const char *ext          = DEFAULT_EXT;
    const char *out_dir_override = NULL;
    int dims[3] = {0, 0, 0};
    int ndims = 0;
    int chunk_mb = DEFAULT_CHUNK_MB;
    int n_runs   = 1;
    unsigned int phase_mask = 0;
    float rank_w0 = 1.0f, rank_w1 = 1.0f, rank_w2 = 1.0f;
    float sgd_lr = REINFORCE_LR;
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
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "no-comp") == 0) phase_mask |= P_NOCOMP;
            else if (strcmp(argv[i], "fixed-lz4") == 0) phase_mask |= P_FIX_LZ4;
            else if (strcmp(argv[i], "fixed-gdeflate") == 0) phase_mask |= P_FIX_GDEFL;
            else if (strcmp(argv[i], "fixed-zstd") == 0) phase_mask |= P_FIX_ZSTD;
            else if (strcmp(argv[i], "entropy-heuristic") == 0) phase_mask |= P_HEUR;
            else if (strcmp(argv[i], "best") == 0) phase_mask |= P_BEST;
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
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc) {
            error_bound = atof(argv[++i]);
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
            "  --phase <name>     Run specific phase(s): no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd, entropy-heuristic, best, nn, nn-rl, nn-rl+exp50\n"
            "  --out-dir DIR      Output directory (default: %s)\n"
            "  --w0/w1/w2 F       Cost model weights\n"
            "  --lr F             SGD learning rate (default: %.2f)\n"
            "  --error-bound F    Error bound for lossy (default: 0.0 = lossless)\n\n"
            "Examples:\n"
            "  %s model.nnwt --data-dir data/sdrbench/nyx/SDRBENCH-EXASKY-NYX-512x512x512 "
            "--dims 512,512,512 --ext .f32\n"
            "  %s model.nnwt --data-dir data/sdrbench/cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600 "
            "--dims 1800,3600 --ext .dat\n",
            argv[0], DEFAULT_EXT, DEFAULT_CHUNK_MB, DEFAULT_OUT_DIR,
            REINFORCE_LR, argv[0], argv[0]);
        return 1;
    }

    if (phase_mask == 0) phase_mask = P_NOCOMP | P_FIX_LZ4 | P_FIX_GDEFL | P_FIX_ZSTD
                                    | P_HEUR | P_BEST | P_NN | P_NNRL | P_NNRLEXP;

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

    /* Verify first file matches expected size */
    size_t first_size = get_file_size(fields[0]);
    if (first_size != expected_file_size) {
        fprintf(stderr, "ERROR: %s is %zu bytes, expected %zu (dims=",
                fields[0], first_size, expected_file_size);
        for (int i = 0; i < ndims; i++)
            fprintf(stderr, "%s%d", i ? "x" : "", dims[i]);
        fprintf(stderr, " * 4 bytes)\n");
        return 1;
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

    /* ── Extract dataset name from data_dir ── */
    const char *dataset_name = strrchr(data_dir, '/');
    dataset_name = dataset_name ? dataset_name + 1 : data_dir;

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

    /* ── Set up output paths ── */
    {
        const char *od = out_dir_override ? out_dir_override : DEFAULT_OUT_DIR;
        snprintf(OUT_DIR, sizeof(OUT_DIR), "%s", od);
        snprintf(OUT_CSV, sizeof(OUT_CSV), "%s/benchmark_%s.csv", od, dataset_name);
        snprintf(OUT_CHUNKS, sizeof(OUT_CHUNKS), "%s/benchmark_%s_chunks.csv", od, dataset_name);
        snprintf(OUT_TSTEP, sizeof(OUT_TSTEP), "%s/benchmark_%s_timesteps.csv", od, dataset_name);
        snprintf(OUT_TSTEP_CHUNKS, sizeof(OUT_TSTEP_CHUNKS),
                 "%s/benchmark_%s_timestep_chunks.csv", od, dataset_name);
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
    if (load_field_to_gpu(fields[0], d_data, total_bytes)) {
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
            { P_FIX_GDEFL,   "fixed-gdeflate",       TMP_FIX_GDEFL,                  GPUCOMPRESS_ALGO_GDEFLATE, 0 },
            { P_FIX_ZSTD,    "fixed-zstd",           TMP_FIX_ZSTD,                   GPUCOMPRESS_ALGO_ZSTD,     0 },
        };
        for (int fi = 0; fi < 3; fi++) {
            if (!(phase_mask & fixed_phases[fi].mask)) continue;
            printf("\n── Phase %d: %s ─────────────────────────────────────\n",
                   n_phases + 1, fixed_phases[fi].name);
            gpucompress_disable_online_learning();
            gpucompress_set_exploration(0);
            hid_t dcpl_f = make_dcpl_fixed(ndims, chunk_dims, fixed_phases[fi].algo, fixed_phases[fi].preproc);
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
        results[n_phases].n_chunks = n_chunks;
            H5Pclose(dcpl_f);
            n_phases++;
        }
    }

    /* ── Phase 5: entropy-heuristic ── */
    if (phase_mask & P_HEUR) {
        printf("\n── Phase %d: entropy-heuristic ───────────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_HEURISTIC);
        hid_t dcpl_h = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);
        PhaseResult runs_buf[32];
        int eff_runs = (n_runs > 32) ? 32 : n_runs;
        for (int run = 0; run < eff_runs; run++) {
            if (eff_runs > 1) printf("  Run %d/%d\n", run + 1, eff_runs);
            int rc = run_phase_vol(d_data, d_read, d_count,
                                   n_floats, ndims, h5_dims, chunk_dims,
                                   "entropy-heuristic", TMP_HEUR, dcpl_h, &runs_buf[run]);
            if (rc) any_fail = 1;
        }
        if (eff_runs > 1)
            merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
        else
            results[n_phases] = runs_buf[0];
        results[n_phases].n_runs = eff_runs;
        results[n_phases].n_chunks = n_chunks;
        H5Pclose(dcpl_h);
        write_chunk_csv("entropy-heuristic");
        gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_NN);  /* restore */
        n_phases++;
    }

    /* ── Phase: best (exhaustive, 32 configs/chunk) ── */
    if (phase_mask & P_BEST) {
        printf("\n── Phase %d: best (exhaustive, 32 configs/chunk) ────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(1);
        gpucompress_set_best_mode(1);
        hid_t dcpl_b = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);
        printf("  Write + Read + Verify... "); fflush(stdout);
        PhaseResult best_r;
        int rc = run_phase_vol(d_data, d_read, d_count,
                               n_floats, ndims, h5_dims, chunk_dims,
                               "best", "/tmp/bm_generic_best.h5", dcpl_b, &best_r);
        if (rc) any_fail = 1;
        results[n_phases] = best_r;
        results[n_phases].n_runs = 1;
        results[n_phases].n_chunks = n_chunks;
        H5Pclose(dcpl_b);
        write_chunk_csv("best");
        gpucompress_set_best_mode(0);
        n_phases++;
    }

    /* ── Phase: nn ── */
    if (phase_mask & P_NN) {
        printf("\n── Phase %d: nn ──────────────────────────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_nn = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);
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
        results[n_phases].n_chunks = n_chunks;
        H5Pclose(dcpl_nn);
        write_chunk_csv("nn");
        n_phases++;
    }

    /* ── Multi-field mode (nn-rl, nn-rl+exp50) ── */
    if (n_fields > 1 && (phase_mask & (P_NNRL | P_NNRLEXP))) {
        struct TsPhase { const char *name; int sgd; int explore; unsigned int mask; };
        TsPhase ts_all[] = {
            { "nn-rl",       1, 0, P_NNRL },
            { "nn-rl+exp50", 1, 1, P_NNRLEXP },
        };

        hid_t dcpl_ts = make_dcpl_auto(h5_dims, ndims, chunk_dims, error_bound);

        /* Open timestep CSV */
        FILE *ts_csv = fopen(OUT_TSTEP, "w");
        if (ts_csv) {
            fprintf(ts_csv, "phase,field_idx,field_name,write_ms,read_ms,ratio,"
                    "smape_ratio,smape_comp,smape_decomp,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fires,explorations,n_chunks,mismatches,"
                    "write_mbps,read_mbps\n");
        }

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
                gpucompress_set_reinforcement(1, sgd_lr, REINFORCE_MAPE, REINFORCE_MAPE);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(ts_all[pi].explore);
            if (ts_all[pi].explore) {
                gpucompress_set_exploration_threshold(0.20);
                gpucompress_set_exploration_k(31);
            }

            printf("  %-4s  %-30s  %-7s  %-7s  %-7s  %-8s  %-4s  %-4s\n",
                   "F#", "Field", "WrMs", "RdMs", "Ratio", "MAPE_R", "SGD", "EXP");
            printf("  ----  %-30s  -------  -------  -------  --------  ----  ----\n", "-----");

            const int WARMUP_SKIP = 3;
            double sum_write_ms = 0, sum_read_ms = 0;
            double sum_ratio = 0;
            double sum_mape_r = 0, sum_mape_c = 0, sum_mape_d = 0;
            int    sum_sgd = 0, sum_expl = 0;
            size_t last_file_sz = 0;
            int n_steady = 0;

            for (int fi = 0; fi < n_fields; fi++) {
                /* Load field to GPU */
                if (load_field_to_gpu(fields[fi], d_data, total_bytes)) {
                    printf("  %-4d  SKIP (load failed)\n", fi);
                    continue;
                }

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
                cudaDeviceSynchronize();
                H5Dclose(dset); H5Fclose(file);
                double tw1 = now_ms();

                drop_pagecache(TMP_NN_RL);

                /* Read back */
                fapl = make_vol_fapl();
                file = H5Fopen(TMP_NN_RL, H5F_ACC_RDONLY, fapl);
                H5Pclose(fapl);
                dset = H5Dopen2(file, "field", H5P_DEFAULT);

                double tr0 = now_ms();
                H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
                cudaDeviceSynchronize();
                H5Dclose(dset); H5Fclose(file);
                double tr1 = now_ms();

                unsigned long long mm = gpu_compare(d_data, d_read, n_floats, d_count);
                size_t file_sz = get_file_size(TMP_NN_RL);
                double ratio_t = (file_sz > 0) ? (double)total_bytes / (double)file_sz : 1.0;
                double write_ms_t = tw1 - tw0;
                double read_ms_t = tr1 - tr0;

                if (fi >= WARMUP_SKIP) {
                    sum_write_ms += write_ms_t;
                    sum_read_ms  += read_ms_t;
                    sum_ratio    += ratio_t;
                    last_file_sz  = file_sz;
                    n_steady++;
                }

                /* Collect MAPE for this field */
                PhaseResult field_r;
                memset(&field_r, 0, sizeof(field_r));
                collect_chunk_metrics(&field_r);

                if (fi >= WARMUP_SKIP) {
                    sum_mape_r += field_r.mape_ratio_pct;
                    sum_mape_c += field_r.mape_comp_pct;
                    sum_mape_d += field_r.mape_decomp_pct;
                    sum_sgd    += field_r.sgd_fires;
                    sum_expl   += field_r.explorations;
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
                    fprintf(ts_csv, "%s,%d,%s,%.2f,%.2f,%.4f,"
                            "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                            "%d,%d,%d,%llu,%.1f,%.1f\n",
                            phase_name, fi, fname,
                            write_ms_t, read_ms_t, ratio_t,
                            field_r.smape_ratio_pct, field_r.smape_comp_pct,
                            field_r.smape_decomp_pct,
                            field_r.mape_ratio_pct, field_r.mape_comp_pct,
                            field_r.mape_decomp_pct,
                            field_r.sgd_fires, field_r.explorations,
                            n_chunks, mm,
                            (double)total_bytes / (1 << 20) / (write_ms_t / 1000.0),
                            (double)total_bytes / (1 << 20) / (read_ms_t / 1000.0));
                }
            }

            /* Store final result for summary */
            PhaseResult *pr = &results[n_phases];
            memset(pr, 0, sizeof(*pr));
            snprintf(pr->phase, sizeof(pr->phase), "%s", phase_name);
            pr->orig_bytes = total_bytes;
            pr->n_chunks = n_chunks;
            pr->n_runs = 1;
            if (n_steady > 0) {
                pr->write_ms = sum_write_ms / n_steady;
                pr->read_ms  = sum_read_ms / n_steady;
                pr->write_mbps = (double)total_bytes / (1 << 20) / (pr->write_ms / 1000.0);
                pr->read_mbps  = (double)total_bytes / (1 << 20) / (pr->read_ms / 1000.0);
                pr->ratio      = sum_ratio / n_steady;
                pr->file_bytes = last_file_sz;
                pr->mape_ratio_pct  = sum_mape_r / n_steady;
                pr->mape_comp_pct   = sum_mape_c / n_steady;
                pr->mape_decomp_pct = sum_mape_d / n_steady;
                pr->sgd_fires   = sum_sgd;
                pr->explorations = sum_expl;
            }
            n_phases++;
        }

        H5Pclose(dcpl_ts);
        if (ts_csv) fclose(ts_csv);
    }

    /* ── Write summary CSV ── */
    write_summary_csv(dataset_name, results, n_phases, total_bytes, n_chunks);

    /* ── Cleanup ── */
    cudaFree(d_data);
    cudaFree(d_read);
    cudaFree(d_count);
    remove(TMP_NOCOMP);
    remove(TMP_FIX_LZ4);
    remove(TMP_FIX_GDEFL);
    remove(TMP_FIX_ZSTD);
    remove(TMP_HEUR);
    remove("/tmp/bm_generic_best.h5");
    remove(TMP_NN);
    remove(TMP_NN_RL);
    remove(TMP_NN_RLEXP);

    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\nBenchmark %s\n", any_fail ? "FAILED" : "PASSED");
    return any_fail;
}
