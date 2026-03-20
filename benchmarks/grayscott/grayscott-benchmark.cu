/**
 * @file grayscott-benchmark.cu
 * @brief Gray-Scott VOL Benchmark: No-Comp vs NN+SGD
 *
 * Runs a Gray-Scott reaction-diffusion simulation on the GPU, then benchmarks
 * writing the 3D field to HDF5 via the GPUCompress VOL connector under four
 * compression phases.  Data stays GPU-resident throughout.
 *
 * Phases:
 *   1. no-comp      : GPU sim → H5Dwrite via VOL (no filter, VOL-2 D→H fallback)
 *   2. nn           : GPU sim → H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *   3. nn-rl        : ALGO_AUTO + online SGD (MAPE≥20%, LR=0.4, no exploration)
 *   4. nn-rl+exp50  : ALGO_AUTO + online SGD + Level-2 exploration (MAPE≥50%)
 *
 * The simulation runs once; all phases reuse the same GPU-resident V-field.
 * Read-back goes into the unused U-field buffer (scratch), so the reference
 * V-field stays intact across phases.  A GPU kernel performs bitwise
 * comparison after each read-back.
 *
 * Usage:
 *   ./build/grayscott_benchmark model.nnwt \
 *       [--L 512] [--steps 1000] [--chunk-mb 8] \
 *       [--F 0.04] [--k 0.06075] [--phase <name>] ...
 *
 * Options:
 *   model.nnwt       Path to NN weights (required, or GPUCOMPRESS_WEIGHTS env)
 *   --L N            Grid side length (dataset = N^3 floats, default 512 ≈ 512 MB)
 *   --steps N        Simulation time-steps before snapshot (default 1000)
 *   --chunk-mb N     Chunk size in MB (auto-calculates chunk_z, default 8)
 *   --chunk-z Z      Set Z-dimension of chunk directly (overrides --chunk-mb)
 *   --F val          Feed rate (default 0.04)
 *   --k val          Kill rate (default 0.06075)
 *   --phase <name>   Run only the specified phase(s). Can be repeated to
 *                    select multiple phases. If omitted, all 4 phases run.
 *                    Valid names: no-comp, nn, nn-rl, nn-rl+exp50
 *
 * Examples:
 *   # Run all phases with 1 GB dataset and 64 MB chunks:
 *   ./build/grayscott_benchmark weights.nnwt --L 640 --chunk-mb 64
 *
 *   # Run only the nn-rl+exp50 phase:
 *   ./build/grayscott_benchmark weights.nnwt --L 256 --phase nn-rl+exp50
 *
 * Dataset size reference:
 *   --L 128 →   8 MB     --L 640  → 1 GB
 *   --L 256 →  64 MB     --L 1000 → 4 GB
 *   --L 512 → 512 MB     --L 1260 → 8 GB
 *
 * Gray-Scott pattern reference (F, k):
 *   --F 0.04  --k 0.06075  Spots (default)
 *   --F 0.035 --k 0.065    Stripes/labyrinth
 *   --F 0.014 --k 0.045    Chaos (high entropy, hard to compress)
 *   --F 0.04  --k 0.065    Sparse spots (very compressible)
 *
 * Output:
 *   tests/benchmark_grayscott_vol_results/benchmark_grayscott_vol.csv
 *   tests/benchmark_grayscott_vol_results/benchmark_grayscott_vol_chunks.csv
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

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_grayscott.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define DEFAULT_L           512
#define DEFAULT_STEPS       1000
#define DEFAULT_CHUNK_MB    8
#define DEFAULT_TIMESTEPS   1
#define DEFAULT_F           0.04f
#define DEFAULT_K           0.06075f

#define REINFORCE_LR        0.1f
#define REINFORCE_MAPE      0.10f

#define TMP_NOCOMP   "/tmp/bm_gs_nocomp.h5"
#define TMP_NN       "/tmp/bm_gs_nn.h5"
#define TMP_NN_RL    "/tmp/bm_gs_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_gs_nn_rlexp.h5"

#define OUT_DIR     "benchmarks/grayscott/results"
#define OUT_CSV     OUT_DIR "/benchmark_grayscott_vol.csv"
#define OUT_CHUNKS  OUT_DIR "/benchmark_grayscott_vol_chunks.csv"
#define OUT_TSTEP   OUT_DIR "/benchmark_grayscott_timesteps.csv"
#define OUT_TSTEP_CHUNKS OUT_DIR "/benchmark_grayscott_timestep_chunks.csv"

/* HDF5 filter ID */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * Helper: decode NN action to readable string
 * ============================================================ */
static const char *ACTION_ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_str(int action, char *buf, size_t bufsz)
{
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ACTION_ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

/* ============================================================
 * GPU-side byte-exact comparison
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
 * Timing
 * ============================================================ */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * File size & page-cache helpers
 * ============================================================ */

static size_t file_size(const char *path)
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

/* 3D chunked DCPL for ALGO_AUTO lossless */
static hid_t make_dcpl_auto(int L, int chunk_z, double eb = 0.0)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(eb, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

/* 3D chunked DCPL for no-comp: chunked, no filter */
static hid_t make_dcpl_nocomp(int L, int chunk_z)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    return dcpl;
}

/* VOL FAPL */
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

/* ============================================================
 * Result struct
 * ============================================================ */

typedef struct {
    char   phase[20];
    double sim_ms;
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
    /* Per-component GPU-time (cumulative across chunks) */
    double nn_ms;
    double stats_ms;
    double preproc_ms;
    double comp_ms;
    double explore_ms;
    double sgd_ms;
    /* Multi-timestep averaged I/O (set after multi-timestep loop; 0 if single-shot only) */
    double write_ms_avg;
    double read_ms_avg;
} PhaseResult;

/* ============================================================
 * Run simulation once and capture V-field and U-field pointers
 *
 * After simulation completes, d_u is unused scratch space that
 * serves as the read-back buffer for all phases.  The V-field
 * (d_v) stays intact as the reference for verification.
 * ============================================================ */

static double run_simulation(gpucompress_grayscott_t sim, int steps,
                             float **d_v_out, float **d_u_out)
{
    gpucompress_grayscott_init(sim);
    double t0 = now_ms();
    gpucompress_grayscott_run(sim, steps);
    cudaDeviceSynchronize();
    double dt = now_ms() - t0;

    float *d_u = NULL, *d_v = NULL;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);
    *d_v_out = d_v;
    if (d_u_out) *d_u_out = d_u;
    return dt;
}

/* ============================================================
 * Phase: no-comp (GPU ptr, VOL-2 fallback)
 * ============================================================ */

static int run_phase_nocomp(float *d_v, float *d_read,
                            unsigned long long *d_count,
                            size_t n_floats, int L, int chunk_z,
                            PhaseResult *r)
{
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* Write */
    printf("[no-comp] H5Dwrite (GPU ptr, VOL-2 fallback D->H)... "); fflush(stdout);
    remove(TMP_NOCOMP);
    hid_t dcpl = make_dcpl_nocomp(L, chunk_z);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_NOCOMP, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    assert(file >= 0 && "[no-comp] H5Fcreate failed");
    H5Pclose(fapl);
    hid_t fsp  = H5Screate_simple(3, dims, NULL);
    hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    assert(dset >= 0 && "[no-comp] H5Dcreate2 failed");
    H5Sclose(fsp);
    H5Pclose(dcpl);

    double t0 = now_ms();
    herr_t wrc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
    assert(wrc >= 0 && "[no-comp] H5Dwrite failed");
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();
    printf("%.0f ms\n", t1 - t0);

    drop_pagecache(TMP_NOCOMP);

    /* Read */
    printf("[no-comp] H5Dread (GPU ptr, VOL-2 fallback H->D)... "); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_NOCOMP, H5F_ACC_RDONLY, fapl);
    assert(file >= 0 && "[no-comp] H5Fopen failed");
    H5Pclose(fapl);
    dset = H5Dopen2(file, "V", H5P_DEFAULT);
    assert(dset >= 0 && "[no-comp] H5Dopen2 failed");

    double t2 = now_ms();
    herr_t rrc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    assert(rrc >= 0 && "[no-comp] H5Dread failed");
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = now_ms();
    printf("%.0f ms\n", t3 - t2);

    /* Compare d_v (original) vs d_read (d_u, read-back) — same sim run */
    unsigned long long mm = gpu_compare(d_v, d_read, n_floats, d_count);

    size_t fbytes = file_size(TMP_NOCOMP);
    r->sim_ms      = 0;  /* set by caller */
    r->write_ms    = t1 - t0;
    r->read_ms     = t3 - t2;
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps  = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps   = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches  = mm;
    r->sgd_fires   = 0;
    r->explorations = 0;
    r->n_chunks    = n_chunks;
    r->mape_ratio_pct  = 0.0;
    r->mape_comp_pct   = 0.0;
    r->mape_decomp_pct = 0.0;
    snprintf(r->phase, sizeof(r->phase), "no-comp");

    printf("[no-comp] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  mismatches=%llu\n",
           r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Phase: VOL (nn / nn-rl / nn-rl+exp)
 * ============================================================ */

static int run_phase_vol(float *d_v, float *d_read,
                         unsigned long long *d_count,
                         size_t n_floats, int L, int chunk_z,
                         const char *phase_name, const char *tmp_file,
                         hid_t dcpl,
                         PhaseResult *r)
{
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* VOL write */
    printf("[%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file); return 1; }

    hid_t fsp  = H5Screate_simple(3, dims, NULL);
    hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0   = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
    H5Dclose(dset);
    H5Fclose(file);
    double t1   = now_ms();
    printf("%.0f ms\n", t1 - t0);

    if (wret < 0) { fprintf(stderr, "[%s] H5Dwrite failed\n", phase_name); return 1; }

    drop_pagecache(tmp_file);

    /* VOL read — into d_read (= sim's d_u, unused after sim) */
    printf("[%s] H5Dread (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "V", H5P_DEFAULT);

    double t2   = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3   = now_ms();
    printf("%.0f ms\n", t3 - t2);

    if (rret < 0) { fprintf(stderr, "[%s] H5Dread failed\n", phase_name); return 1; }

    /* Compare d_v (original) vs d_read (d_u, read-back) — same sim run */
    unsigned long long mm = gpu_compare(d_v, d_read, n_floats, d_count);

    /* Collect per-chunk diagnostics */
    int sgd_fires  = 0;
    int explorations = 0;
    double ape_ratio_sum = 0.0, ape_comp_sum = 0.0, ape_decomp_sum = 0.0;
    int    ape_ratio_cnt = 0,   ape_comp_cnt = 0,   ape_decomp_cnt = 0;
    double total_nn_ms     = 0.0;
    double total_stats_ms  = 0.0;
    double total_preproc_ms = 0.0;
    double total_comp_ms   = 0.0;
    double total_explore_ms = 0.0;
    double total_sgd_ms    = 0.0;
    int n_hist     = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            sgd_fires    += d.sgd_fired;
            explorations += d.exploration_triggered;
            total_nn_ms      += d.nn_inference_ms;
            total_stats_ms   += d.stats_ms;
            total_preproc_ms += d.preprocessing_ms;
            total_comp_ms    += d.compression_ms;
            total_explore_ms += d.exploration_ms;
            total_sgd_ms     += d.sgd_update_ms;

            /* Per-metric sMAPE (symmetric, bounded 0–200%) */
            double denom_r = (fabs(d.actual_ratio) + fabs(d.predicted_ratio)) / 2.0;
            double denom_c = (fabs(d.compression_ms) + fabs(d.predicted_comp_time)) / 2.0;
            double denom_d = (fabs(d.decompression_ms) + fabs(d.predicted_decomp_time)) / 2.0;
            double mape_ratio = (denom_r > 0)
                ? fabs(d.predicted_ratio - d.actual_ratio) / denom_r * 100.0 : 0.0;
            double mape_comp = (denom_c > 0)
                ? fabs(d.predicted_comp_time - d.compression_ms) / denom_c * 100.0 : 0.0;
            double mape_decomp = (denom_d > 0)
                ? fabs(d.predicted_decomp_time - d.decompression_ms) / denom_d * 100.0 : 0.0;

            if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
                ape_ratio_sum += mape_ratio; ape_ratio_cnt++;
            }
            if (d.compression_ms > 0) {
                ape_comp_sum += mape_comp; ape_comp_cnt++;
            }
            if (d.decompression_ms > 0) {
                ape_decomp_sum += mape_decomp; ape_decomp_cnt++;
            }

            /* Per-chunk detail omitted for single-write phases (high MAPE expected) */
        }
    }

    double write_ms = t1 - t0;
    size_t fbytes = file_size(tmp_file);
    r->sim_ms      = 0;  /* set by caller */
    r->write_ms    = write_ms;
    r->read_ms     = t3 - t2;
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps  = (double)total_bytes / (1 << 20) / (write_ms / 1000.0);
    r->read_mbps   = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches  = mm;
    r->sgd_fires   = sgd_fires;
    r->explorations = explorations;
    r->n_chunks    = n_chunks;
    r->mape_ratio_pct  = (ape_ratio_cnt > 0) ? ape_ratio_sum / ape_ratio_cnt : 0.0;
    r->mape_comp_pct   = (ape_comp_cnt > 0) ? ape_comp_sum / ape_comp_cnt : 0.0;
    r->mape_decomp_pct = (ape_decomp_cnt > 0) ? ape_decomp_sum / ape_decomp_cnt : 0.0;

    printf("\n  MAPE avg:  ratio=%.1f%%  comp_time=%.1f%%  decomp_time=%.1f%%\n",
           r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct);
    r->nn_ms       = total_nn_ms;
    r->stats_ms    = total_stats_ms;
    r->preproc_ms  = total_preproc_ms;
    r->comp_ms     = total_comp_ms;
    r->explore_ms  = total_explore_ms;
    r->sgd_ms      = total_sgd_ms;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    printf("[%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  sgd=%d expl=%d/%d  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), sgd_fires, explorations, n_hist, mm);

    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * CSV output
 * ============================================================ */

static void write_aggregate_csv(PhaseResult *res, int n_phases,
                                int L, int steps, int chunk_z,
                                float F, float k)
{
    FILE *f = fopen(OUT_CSV, "w");
    if (!f) { perror("fopen " OUT_CSV); return; }

    fprintf(f, "phase,L,steps,F,k,chunk_z,n_chunks,"
               "sim_ms,write_ms,read_ms,file_mib,orig_mib,ratio,"
               "write_mibps,read_mibps,mismatches,sgd_fires,explorations,"
               "mape_ratio_pct,mape_comp_pct,mape_decomp_pct\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        fprintf(f, "%s,%d,%d,%.4f,%.5f,%d,%d,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,"
                   "%.1f,%.1f,%llu,%d,%d,%.2f,%.2f,%.2f\n",
                r->phase, L, steps, F, k, chunk_z, r->n_chunks,
                r->sim_ms, r->write_ms, r->read_ms,
                (double)r->file_bytes / (1 << 20),
                (double)r->orig_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps,
                r->mismatches, r->sgd_fires, r->explorations,
                r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct);
    }
    fclose(f);
    printf("\nAggregate CSV written to: %s\n", OUT_CSV);
}

static void write_chunk_csv(const char *phase_name, int n_chunks)
{
    struct stat csv_st;
    bool need_hdr = (stat(OUT_CHUNKS, &csv_st) != 0 || csv_st.st_size == 0);
    FILE *f = fopen(OUT_CHUNKS, "a");
    if (!f) { perror("fopen " OUT_CHUNKS); return; }
    if (need_hdr) {
        fprintf(f, "phase,chunk,action_final,action_orig,"
                   "actual_ratio,predicted_ratio,mape_ratio,"
                   "actual_comp_ms,predicted_comp_ms,mape_comp,"
                   "actual_decomp_ms,predicted_decomp_ms,mape_decomp,"
                   "sgd_fired,exploration_triggered,"
                   "nn_inference_ms,preprocessing_ms,compression_ms,"
                   "exploration_ms,sgd_update_ms\n");
    }

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist && i < n_chunks; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            char final_str[40], orig_str[40];
            action_to_str(d.nn_action, final_str, sizeof(final_str));
            action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));
            double dr = (fabs(d.actual_ratio) + fabs(d.predicted_ratio)) / 2.0;
            double dc = (fabs(d.compression_ms) + fabs(d.predicted_comp_time)) / 2.0;
            double dd = (fabs(d.decompression_ms) + fabs(d.predicted_decomp_time)) / 2.0;
            double mape_ratio = (dr > 0)
                ? fabs(d.predicted_ratio - d.actual_ratio) / dr * 100.0 : 0.0;
            double mape_comp = (dc > 0)
                ? fabs(d.predicted_comp_time - d.compression_ms) / dc * 100.0 : 0.0;
            double mape_decomp = (dd > 0)
                ? fabs(d.predicted_decomp_time - d.decompression_ms) / dd * 100.0 : 0.0;
            fprintf(f, "%s,%d,%s,%s,%.4f,%.4f,%.1f,"
                       "%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,"
                       "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    phase_name, i, final_str, orig_str,
                    (double)d.actual_ratio, (double)d.predicted_ratio, mape_ratio,
                    (double)d.compression_ms, (double)d.predicted_comp_time, mape_comp,
                    (double)d.decompression_ms, (double)d.predicted_decomp_time, mape_decomp,
                    d.sgd_fired, d.exploration_triggered,
                    (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                    (double)d.compression_ms, (double)d.exploration_ms,
                    (double)d.sgd_update_ms);
        } else {
            fprintf(f, "%s,%d,none,none,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n", phase_name, i);
        }
    }
    fclose(f);
}

/* ============================================================
 * Console summary table
 * ============================================================ */

static void print_summary(PhaseResult *res, int n_phases,
                          int L, int steps, int chunk_z,
                          float F, float k)
{
    double dataset_mb = (double)L * L * L * sizeof(float) / (1 << 20);
    double chunk_mb   = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);
    int n_chunks      = (L + chunk_z - 1) / chunk_z;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Gray-Scott VOL Benchmark Summary                                       ║\n");
    printf("║  Grid: %d^3 (%.0f MB)  Chunks: %d x %.1f MB  Steps: %d               \n",
           L, dataset_mb, n_chunks, chunk_mb, steps);
    printf("║  Pattern: F=%.4f  k=%.5f                                              \n", F, k);
    printf("╠══════════════╦══════════╦══════════╦═══════╦══════════╦══════╦═════════════╣\n");
    printf("║  Phase       ║  Sim     ║ Write    ║ Read  ║ Ratio   ║ File ║ Verify      ║\n");
    printf("║              ║  (ms)    ║ (MiB/s)  ║(MiB/s)║         ║(MiB) ║             ║\n");
    printf("╠══════════════╬══════════╬══════════╬═══════╬═════════╬══════╬═════════════╣\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        const char *verdict = (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %8.0f ║ %8.0f ║ %5.0f ║ %5.2fx  ║ %4.0f ║ %-11s ║\n",
               r->phase, r->sim_ms, r->write_mbps, r->read_mbps,
               r->ratio, (double)r->file_bytes / (1 << 20), verdict);
    }
    printf("╚══════════════╩══════════╩══════════╩═══════╩═════════╩══════╩═════════════╝\n");

    /* NN phase SGD/exploration summary */
    for (int i = 0; i < n_phases; i++) {
        if (strncmp(res[i].phase, "nn", 2) == 0 && res[i].n_chunks > 0) {
            printf("\n  %-14s SGD: %d/%d  Expl: %d/%d  MAPE: ratio=%.1f%% comp=%.1f%% decomp=%.1f%%\n",
                   res[i].phase,
                   res[i].sgd_fires,   res[i].n_chunks,
                   res[i].explorations, res[i].n_chunks,
                   res[i].mape_ratio_pct, res[i].mape_comp_pct, res[i].mape_decomp_pct);
        }
    }

}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    /* Defaults */
    const char *weights_path = NULL;
    int L        = DEFAULT_L;
    int steps    = DEFAULT_STEPS;
    int timesteps = DEFAULT_TIMESTEPS;  /* multi-timestep mode: >1 loops nn-rl */
    int chunk_z  = 0;    /* 0 = auto from chunk_mb */
    int chunk_mb = DEFAULT_CHUNK_MB;
    float F      = DEFAULT_F;
    float k      = DEFAULT_K;
    float  sgd_lr    = REINFORCE_LR;  /* overridable via --lr */
    double error_bound = 0.0;  /* 0 = lossless, >0 = lossy quantization */
    const char *out_dir_override = NULL;  /* overridable via --out-dir */
    int verbose_chunks = 0;  /* --verbose-chunks: print per-chunk detail at milestones */

    /* Phase selection: bit flags */
    enum { P_NOCOMP = 1, P_NN = 4, P_NNRL = 8, P_NNRLEXP = 16 };
    unsigned int phase_mask = 0;  /* 0 = run all */

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--timesteps") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
            if (timesteps < 1) timesteps = 1;
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
            chunk_z = 0;  /* will compute from chunk_mb */
        } else if (strcmp(argv[i], "--chunk-z") == 0 && i + 1 < argc) {
            chunk_z = atoi(argv[++i]);
            chunk_mb = 0; /* explicit chunk_z overrides */
        } else if (strcmp(argv[i], "--F") == 0 && i + 1 < argc) {
            F = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc) {
            error_bound = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            sgd_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir_override = argv[++i];
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            const char *p = argv[++i];
            if      (strcmp(p, "no-comp") == 0)      phase_mask |= P_NOCOMP;
            else if (strcmp(p, "nn") == 0)            phase_mask |= P_NN;
            else if (strcmp(p, "nn-rl") == 0)         phase_mask |= P_NNRL;
            else if (strcmp(p, "nn-rl+exp50") == 0)   phase_mask |= P_NNRLEXP;
            else { fprintf(stderr, "Unknown phase: %s\n"
                           "  Valid: no-comp, nn, nn-rl, nn-rl+exp50\n", p);
                   return 1; }
        } else if (strcmp(argv[i], "--verbose-chunks") == 0) {
            verbose_chunks = 1;
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--steps N] "
                "[--chunk-mb N] [--chunk-z Z] [--F val] [--k val] "
                "[--phase <name>] ...\n"
                "  Phases: no-comp, nn, nn-rl, nn-rl+exp50\n"
                "  Use --phase multiple times to select specific phases.\n", argv[0]);
        return 1;
    }
    if (phase_mask == 0) phase_mask = P_NOCOMP | P_NN | P_NNRL | P_NNRLEXP;

    /* Compute chunk_z from chunk_mb if not explicitly set */
    if (chunk_z <= 0 && chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    }
    if (chunk_z < 1) chunk_z = L / 4;
    if (chunk_z < 1) chunk_z = 1;

    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int    n_chunks    = (L + chunk_z - 1) / chunk_z;
    double dataset_mb  = (double)total_bytes / (1 << 20);
    double cmb         = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);

    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Gray-Scott VOL Benchmark: No-Comp vs NN+SGD                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Steps    : %d per timestep\n", steps);
    if (timesteps > 1)
        printf("  Timesteps: %d (multi-timestep mode: nn-rl across %d writes)\n",
               timesteps, timesteps);
    printf("  Pattern  : F=%.4f  k=%.5f\n", F, k);
    printf("  SGD LR   : %.4f\n", sgd_lr);
    printf("  Weights  : %s\n\n", weights_path);

    /* Create output directory and clear stale chunks CSV */
    mkdir(OUT_DIR, 0755);
    remove(OUT_CHUNKS);

    /* Init gpucompress */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded from %s\n", weights_path);
        gpucompress_cleanup(); return 1;
    }

    /* Register VOL */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup(); return 1;
    }
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Create Gray-Scott simulation */
    GrayScottSettings s = gray_scott_default_settings();
    s.L = L;
    s.F = F;
    s.k = k;
    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);

    /* Allocate GPU mismatch counter (read-back reuses sim's d_u buffer) */
    unsigned long long *d_count = NULL;

    if (cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc for mismatch counter failed\n");
        gpucompress_grayscott_destroy(sim);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    /* ── Run simulation once — all phases reuse the same data ─────── */
    printf("\n── Running Gray-Scott simulation once (%d steps)... ", steps);
    fflush(stdout);
    float *d_v = NULL, *d_read = NULL;
    double sim_ms = run_simulation(sim, steps, &d_v, &d_read);
    printf("%.0f ms\n", sim_ms);

    PhaseResult results[4];
    int n_phases = 0;
    int any_fail = 0;

    int rc = 0;

    /* ── Phase 1: no-comp ──────────────────────────────────────────── */
    if (phase_mask & P_NOCOMP) {
        printf("\n── Phase %d: no-comp (GPU->Host->HDF5) ────────────────────────\n", n_phases + 1);
        printf("  Writing... "); fflush(stdout);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        double t0 = now_ms();
        rc = run_phase_nocomp(d_v, d_read, d_count,
                                  n_floats, L, chunk_z,
                                  &results[n_phases]);
        printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
        results[n_phases].sim_ms = sim_ms;
        if (rc) any_fail = 1;
        n_phases++;
    }

    /* ── Phase 2: nn (VOL, ALGO_AUTO, inference-only) ────────────── */
    if (phase_mask & P_NN) {
        printf("\n── Phase %d: nn (VOL, ALGO_AUTO, inference-only) ────────────\n", n_phases + 1);
        printf("  Write + Read + Verify... "); fflush(stdout);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_nn = make_dcpl_auto(L, chunk_z, error_bound);
        double t0 = now_ms();
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn", TMP_NN, dcpl_nn,
                           &results[n_phases]);
        printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_nn);
        if (rc) any_fail = 1;
        write_chunk_csv("nn", n_chunks);
        n_phases++;
    }

    /* ── Phase 3: nn-rl (ALGO_AUTO + SGD, no exploration) ─────────── */
    if (phase_mask & P_NNRL) {
        printf("\n── Phase %d: nn-rl (ALGO_AUTO + SGD, MAPE>=%.0f%%, LR=%.4f) ────\n",
               n_phases + 1, REINFORCE_MAPE * 100.0f, sgd_lr);
        printf("  Write + Read + Verify... "); fflush(stdout);
        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, sgd_lr, REINFORCE_MAPE, REINFORCE_MAPE);
        gpucompress_set_exploration(0);
        hid_t dcpl_rl = make_dcpl_auto(L, chunk_z, error_bound);
        double t0 = now_ms();
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn-rl", TMP_NN_RL, dcpl_rl,
                           &results[n_phases]);
        printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_rl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        write_chunk_csv("nn-rl", n_chunks);
        n_phases++;
    }

    /* ── Phase 4: nn-rl+exp50 (ALGO_AUTO + SGD + exploration) ────── */
    if (phase_mask & P_NNRLEXP) {
        /* Reset NN weights so phase 4 starts from original trained weights,
         * not the SGD-modified weights from phase 3. */
        gpucompress_reload_nn(weights_path);
        printf("\n── Phase %d: nn-rl+exp50 (ALGO_AUTO + SGD + expl@MAPE>=20%%, K=4) ─\n", n_phases + 1);
        printf("  Write + Read + Verify... "); fflush(stdout);
        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, sgd_lr, REINFORCE_MAPE, REINFORCE_MAPE);
        gpucompress_set_exploration(1);
        gpucompress_set_exploration_threshold(0.20);
        gpucompress_set_exploration_k(4);
        hid_t dcpl_rlexp = make_dcpl_auto(L, chunk_z, error_bound);
        double t0 = now_ms();
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn-rl+exp50", TMP_NN_RLEXP, dcpl_rlexp,
                           &results[n_phases]);
        printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_rlexp);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        write_chunk_csv("nn-rl+exp50", n_chunks);
        n_phases++;
    }

    /* ── Multi-timestep mode ─────────────────────────────────────── */
    if (timesteps > 1) {

        /* Phase configs: name, sgd_enabled, exploration_enabled */
        struct TsPhase {
            const char *name;
            int sgd;
            int explore;
        };
        TsPhase ts_phases[] = {
            { "nn-rl",       1, 0 },
            { "nn-rl+exp50", 1, 1 },
        };
        int n_ts_phases = 2;

        hid_t dcpl_ts = make_dcpl_auto(L, chunk_z, error_bound);
        hsize_t dims3[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

        /* Open timestep CSV (shared across phases) */
        FILE *ts_csv = fopen(OUT_TSTEP, "w");
        if (ts_csv) {
            fprintf(ts_csv, "phase,timestep,sim_step,write_ms,read_ms,ratio,"
                    "smape_ratio,smape_comp,smape_decomp,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fires,explorations,n_chunks,mismatches,"
                    "write_mbps,read_mbps,cache_hits,cache_misses\n");
        }

        /* Open per-chunk milestone CSV (shared across phases) */
        FILE *tc_csv = fopen(OUT_TSTEP_CHUNKS, "w");
        if (tc_csv) {
            fprintf(tc_csv, "phase,timestep,chunk,action,predicted_ratio,actual_ratio,"
                    "predicted_comp_ms,actual_comp_ms,"
                    "predicted_decomp_ms,actual_decomp_ms,"
                    "smape_ratio,smape_comp,smape_decomp,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fired,exploration_triggered\n");
        }

        for (int pi = 0; pi < n_ts_phases; pi++) {
            const char *phase_name = ts_phases[pi].name;
            int do_sgd   = ts_phases[pi].sgd;
            int do_expl  = ts_phases[pi].explore;

            printf("\n══════════════════════════════════════════════════════════════\n");
            printf("  Multi-timestep [%s]: %d writes (SGD=%s, Explore=%s)\n",
                   phase_name, timesteps,
                   do_sgd ? "on" : "off", do_expl ? "on" : "off");
            printf("  Each timestep: %d sim steps + H5Dwrite + verify\n", steps);
            printf("══════════════════════════════════════════════════════════════\n\n");

            /* Reload NN and re-init simulation to start fresh */
            gpucompress_grayscott_destroy(sim);
            sim = NULL;
            gpucompress_grayscott_create(&sim, &s);

            gpucompress_reload_nn(weights_path);
            if (do_sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, sgd_lr, REINFORCE_MAPE, REINFORCE_MAPE);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(do_expl);
            if (do_expl) {
                gpucompress_set_exploration_threshold(0.20);
                gpucompress_set_exploration_k(4);
            }

            printf("  %-4s  %-8s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %-4s  %-4s\n",
                   "T", "SimStep", "WrMs", "RdMs", "Ratio",
                   "sMAPE_R", "sMAPE_C", "sMAPE_D",
                   "MAPE_R", "MAPE_C", "MAPE_D", "SGD", "EXP");
            printf("  ----  --------  -------  -------  -------  "
                   "--------  --------  --------  "
                   "--------  --------  --------  ----  ----\n");

            /* Accumulators for averaged throughput (skip first WARMUP_SKIP timesteps) */
            const int WARMUP_SKIP = 5;
            double sum_write_ms = 0, sum_read_ms = 0;
            int    n_steady = 0;
            /* Final timestep MAPE (overwritten each iteration, last value is final) */
            double final_mape_r = 0, final_mape_c = 0, final_mape_d = 0;
            int    final_sgd = 0, final_expl = 0;
            /* Cumulative SGD/exploration counts across all timesteps */
            int    cum_sgd = 0, cum_expl = 0;

            for (int t = 0; t < timesteps; t++) {
                int cum_sim_step = (t + 1) * steps;

                /* ── Progress bar ── */
                {
                    int bar_w = 30;
                    int filled = (t * bar_w) / timesteps;
                    printf("\r  [");
                    for (int b = 0; b < bar_w; b++)
                        putchar(b < filled ? '#' : '.');
                    printf("] %d/%d    ", t, timesteps);
                    fflush(stdout);
                }

                /* Advance simulation by `steps` */
                gpucompress_grayscott_run(sim, steps);
                cudaDeviceSynchronize();
                gpucompress_grayscott_get_device_ptrs(sim, &d_read, &d_v);

                /* Write via VOL */
                gpucompress_reset_chunk_history();
                gpucompress_reset_cache_stats();
                remove(TMP_NN_RL);

                hid_t fapl = make_vol_fapl();
                hid_t file = H5Fcreate(TMP_NN_RL, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
                H5Pclose(fapl);

                hid_t fsp  = H5Screate_simple(3, dims3, NULL);
                hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                                         fsp, H5P_DEFAULT, dcpl_ts, H5P_DEFAULT);
                H5Sclose(fsp);

                double tw0  = now_ms();
                herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                                        H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
                H5Dclose(dset); H5Fclose(file);
                double tw1  = now_ms();
                double write_ms_t = tw1 - tw0;

                if (wret < 0) {
                    printf("  %-4d  H5Dwrite failed\n", t);
                    continue;
                }

                /* Read back + verify */
                fapl = make_vol_fapl();
                file = H5Fopen(TMP_NN_RL, H5F_ACC_RDONLY, fapl);
                H5Pclose(fapl);
                dset = H5Dopen2(file, "V", H5P_DEFAULT);

                double tr0  = now_ms();
                H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
                cudaDeviceSynchronize();
                H5Dclose(dset); H5Fclose(file);
                double tr1  = now_ms();
                double read_ms_t = tr1 - tr0;

                /* Accumulate for averaged throughput (skip warmup) */
                if (t >= WARMUP_SKIP) {
                    sum_write_ms += write_ms_t;
                    sum_read_ms  += read_ms_t;
                    n_steady++;
                }

                unsigned long long mm = gpu_compare(d_v, d_read, n_floats, d_count);

                /* File size for ratio */
                size_t file_sz = file_size(TMP_NN_RL);
                double ratio_t = (file_sz > 0) ? (double)total_bytes / (double)file_sz : 1.0;

                /* Collect per-chunk sMAPE (symmetric, bounded 0–200%) AND real MAPE */
                int n_hist = gpucompress_get_chunk_history_count();
                double ape_r = 0, ape_c = 0, ape_d = 0;
                double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0;
                int    cnt_r = 0, cnt_c = 0, cnt_d = 0;
                int    mcnt_r = 0, mcnt_c = 0, mcnt_d = 0;
                int    sgd_t = 0, expl_t = 0;
                for (int ci = 0; ci < n_hist; ci++) {
                    gpucompress_chunk_diag_t diag;
                    if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
                    if (diag.sgd_fired) sgd_t++;
                    if (diag.exploration_triggered) expl_t++;
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
                double mape_r = cnt_r ? (ape_r / cnt_r) * 100.0 : 0.0;
                double mape_c = cnt_c ? (ape_c / cnt_c) * 100.0 : 0.0;
                double mape_d = cnt_d ? (ape_d / cnt_d) * 100.0 : 0.0;
                double real_mape_r = mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0;
                double real_mape_c = mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0;
                double real_mape_d = mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0;

                /* Track final timestep values for summary table */
                final_mape_r = mape_r;
                final_mape_c = mape_c;
                final_mape_d = mape_d;
                final_sgd    = sgd_t;
                final_expl   = expl_t;
                cum_sgd     += sgd_t;
                cum_expl    += expl_t;

                double wr_mbps = (write_ms_t > 0) ? dataset_mb / (write_ms_t / 1000.0) : 0;
                double rd_mbps = (read_ms_t > 0)  ? dataset_mb / (read_ms_t  / 1000.0) : 0;

                int c_hits = 0, c_misses = 0;
                gpucompress_get_cache_stats(&c_hits, &c_misses);

                /* Print summary row every 5 timesteps, first and last */
                bool print_row = (t % 5 == 0 || t == timesteps - 1);
                if (print_row) {
                    printf("\r%80s\r", "");  /* clear progress bar */
                    printf("  %-4d  %-8d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%  %3d  %3d\n",
                           t, cum_sim_step, write_ms_t, read_ms_t,
                           ratio_t, mape_r, mape_c, mape_d,
                           real_mape_r, real_mape_c, real_mape_d, sgd_t, expl_t);
                }

                /* Per-chunk detail at milestone timesteps.
                 * Console output requires --verbose-chunks; CSV always written. */
                {
                    bool is_milestone = (t == 0 || t == timesteps / 4 ||
                                         t == timesteps / 2 || t == (timesteps * 3) / 4 ||
                                         t == timesteps - 1);
                    if (is_milestone) {
                        if (verbose_chunks)
                            printf("    ── Per-chunk detail for T=%d ──\n", t);
                        int n_detail = n_hist;
                        int stride = (n_detail > 20) ? n_detail / 10 : 1;
                        for (int ci = 0; ci < n_detail; ci++) {
                            gpucompress_chunk_diag_t dd;
                            if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;

                            double sr = 0, sc = 0, sd = 0;
                            double mr = 0, mc = 0, md = 0;
                            double den_r = (fabs(dd.actual_ratio) + fabs(dd.predicted_ratio)) / 2.0;
                            double den_c = (fabs(dd.compression_ms) + fabs(dd.predicted_comp_time)) / 2.0;
                            double den_d = (fabs(dd.decompression_ms) + fabs(dd.predicted_decomp_time)) / 2.0;
                            if (den_r > 0) sr = fabs(dd.predicted_ratio - dd.actual_ratio) / den_r * 100.0;
                            if (den_c > 0) sc = fabs(dd.predicted_comp_time - dd.compression_ms) / den_c * 100.0;
                            if (den_d > 0) sd = fabs(dd.predicted_decomp_time - dd.decompression_ms) / den_d * 100.0;
                            if (dd.actual_ratio > 0)
                                mr = fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) * 100.0;
                            if (dd.compression_ms > 0)
                                mc = fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) * 100.0;
                            if (dd.decompression_ms > 0)
                                md = fabs(dd.predicted_decomp_time - dd.decompression_ms) / fabs(dd.decompression_ms) * 100.0;

                            if (verbose_chunks) {
                                bool print_it = (ci < 3 || ci >= n_detail - 3 || ci % stride == 0);
                                if (print_it) {
                                    printf("      C%3d  pred_d=%6.3fms  act_d=%5.3fms  MAPE_D=%6.1f%%"
                                           "  pred_r=%6.1fx  act_r=%6.1fx  MAPE_R=%5.1f%%%s%s\n",
                                           ci + 1,
                                           (double)dd.predicted_decomp_time,
                                           (double)dd.decompression_ms, md,
                                           (double)dd.predicted_ratio,
                                           (double)dd.actual_ratio, mr,
                                           dd.sgd_fired ? " [SGD]" : "",
                                           dd.exploration_triggered ? " [EXP]" : "");
                                }
                            }

                            if (tc_csv) {
                                char action_str[40];
                                action_to_str(dd.nn_action, action_str, sizeof(action_str));
                                fprintf(tc_csv, "%s,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                                        "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
                                        phase_name, t, ci, action_str,
                                        (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                        (double)dd.predicted_comp_time, (double)dd.compression_ms,
                                        (double)dd.predicted_decomp_time, (double)dd.decompression_ms,
                                        sr, sc, sd, mr, mc, md,
                                        dd.sgd_fired, dd.exploration_triggered);
                            }
                        }
                        if (verbose_chunks)
                            printf("    ── end T=%d ──\n", t);
                    }
                }

                if (ts_csv) {
                    fprintf(ts_csv, "%s,%d,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%llu,%.1f,%.1f,%d,%d\n",
                            phase_name, t, cum_sim_step, write_ms_t, read_ms_t, ratio_t,
                            mape_r, mape_c, mape_d,
                            real_mape_r, real_mape_c, real_mape_d,
                            sgd_t, expl_t, n_hist,
                            (unsigned long long)mm, wr_mbps, rd_mbps,
                            c_hits, c_misses);
                }

                remove(TMP_NN_RL);
            } /* end timestep loop */

            /* Final progress bar */
            {
                printf("\r  [");
                for (int b = 0; b < 30; b++) putchar('#');
                printf("] %d/%d              \n\n", timesteps, timesteps);
            }

            /* Store averaged throughput and final-timestep MAPE in the matching
             * single-shot PhaseResult so the summary table reflects multi-timestep state. */
            for (int ri = 0; ri < n_phases; ri++) {
                if (strcmp(results[ri].phase, phase_name) == 0) {
                    if (n_steady > 0) {
                        double avg_wr = sum_write_ms / n_steady;
                        double avg_rd = sum_read_ms  / n_steady;
                        results[ri].write_ms_avg = avg_wr;
                        results[ri].read_ms_avg  = avg_rd;
                        results[ri].write_mbps = dataset_mb / (avg_wr / 1000.0);
                        results[ri].read_mbps  = dataset_mb / (avg_rd / 1000.0);
                    }
                    /* Overwrite MAPE with final timestep values (not single-shot) */
                    results[ri].mape_ratio_pct  = final_mape_r;
                    results[ri].mape_comp_pct   = final_mape_c;
                    results[ri].mape_decomp_pct = final_mape_d;
                    /* Cumulative SGD/exploration across all timesteps */
                    results[ri].sgd_fires   = cum_sgd;
                    results[ri].explorations = cum_expl;
                    results[ri].n_chunks    = n_chunks * timesteps;
                    break;
                }
            }
        } /* end phase loop */

        if (ts_csv) {
            fclose(ts_csv);
            printf("\n  Timestep CSV: %s\n", OUT_TSTEP);
        }
        if (tc_csv) {
            fclose(tc_csv);
            printf("  Timestep chunks CSV: %s\n", OUT_TSTEP_CHUNKS);
        }

        gpucompress_disable_online_learning();
        H5Pclose(dcpl_ts);
    }

    /* ── Summary ───────────────────────────────────────────────────── */
    print_summary(results, n_phases, L, steps, chunk_z, F, k);
    write_aggregate_csv(results, n_phases, L, steps, chunk_z, F, k);
    printf("Per-chunk CSV written to: %s\n", OUT_CHUNKS);

    /* ── Cleanup ───────────────────────────────────────────────────── */
    cudaFree(d_count);
    gpucompress_grayscott_destroy(sim);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    remove(TMP_NOCOMP);
    remove(TMP_NN);
    remove(TMP_NN_RL);
    remove(TMP_NN_RLEXP);

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}