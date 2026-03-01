/**
 * @file benchmark_vol_gpu.cu
 * @brief GPU-native VOL Benchmark: No-Comp vs Static vs NN+SGD
 *
 * Generates float32 data entirely on the GPU using 8 deterministic patterns,
 * writes each phase to HDF5, reads back, and verifies byte-exact correctness
 * with a GPU-side mismatch counter.
 *
 * Phases:
 *   1. no-comp      : GPU fill → H5Dwrite via VOL (GPU ptr, no filter → VOL-2 D→H fallback)
 *   2. static       : GPU fill → H5Dwrite via VOL (zstd + byte-shuffle)
 *   3. nn           : GPU fill → H5Dwrite via VOL (ALGO_AUTO, inference-only, no RL)
 *   4. nn-rl        : ALGO_AUTO + online SGD (MAPE≥20%, LR=0.4, no exploration)
 *   5. nn-rl+exp50  : ALGO_AUTO + online SGD + Level-2 exploration (MAPE≥50%)
 *
 * GPU buffer is regenerated per phase to ensure a clean reference.
 *
 * Fill modes:
 *   --mode uniform     : all chunks filled with ramp (pattern 2)
 *   --mode contiguous  : each pattern fills N/8 consecutive chunks
 *   --mode cycling     : pattern = chunk_index % 8   [default]
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/benchmark_vol_gpu neural_net/weights/model.nnwt \
 *     [--dataset-mb 4096] [--chunk-mb 8] [--mode cycling]
 *
 * Output:
 *   tests/benchmark_vol_gpu_results/benchmark_vol_gpu.csv
 *   tests/benchmark_vol_gpu_results/benchmark_vol_gpu_chunks.csv
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
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define DEFAULT_DATASET_MB  4096
#define DEFAULT_CHUNK_MB    8
#define DEFAULT_MODE        "cycling"
#define N_PATTERNS          8
#define REINFORCE_LR        0.4f
#define REINFORCE_MAPE      0.20f

#define TMP_NOCOMP   "/tmp/bm_vol_nocomp.h5"
#define TMP_STATIC   "/tmp/bm_vol_static.h5"
#define TMP_NN       "/tmp/bm_vol_nn.h5"
#define TMP_NN_RL    "/tmp/bm_vol_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_vol_nn_rlexp.h5"

#define OUT_DIR     "tests/benchmark_vol_gpu_results"
#define OUT_CSV     OUT_DIR "/benchmark_vol_gpu.csv"
#define OUT_CHUNKS  OUT_DIR "/benchmark_vol_gpu_chunks.csv"

/* HDF5 filter ID */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * GPU fill kernel — 8 deterministic patterns
 *
 * Uses a per-element splitmix64 hash for pseudo-random patterns.
 * Each element is computed independently — safe for any thread count.
 * seed_base encodes (chunk_index) for per-chunk determinism.
 * ============================================================ */

__device__ static float elem_rand(unsigned long long seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 33;
    return (float)(seed & 0xFFFFFFu) / (float)0x1000000u;
}

/**
 * fill_chunk_kernel — fills buf[0..n) according to pattern_id.
 * @param seed_base   chunk_index — ensures different chunks have different data
 *                    even for the same pattern.
 */
__global__ void fill_chunk_kernel(float * __restrict__ buf, size_t n,
                                  int pattern_id, unsigned long long seed_base)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned long long s = seed_base ^ ((unsigned long long)i * 6364136223846793005ULL);
        float r = elem_rand(s);
        float v = 0.0f;
        switch (pattern_id) {
        case 0: /* constant — best-case for compressor */
            v = 42.0f;
            break;
        case 1: /* smooth_sine — 1 full cycle per chunk */
            v = 1000.0f * sinf(2.0f * 3.14159265358979f * (float)i / (float)n);
            break;
        case 2: /* ramp — linear 0..1 */
            v = (float)i / (float)n;
            break;
        case 3: /* sparse — 99% zero, 1% random spikes */
            v = (r < 0.01f) ? (r * 10000.0f - 50.0f) : 0.0f;
            break;
        case 4: /* step — 8 discrete levels */
            v = (float)(i / (n / 8u)) * 100.0f;
            break;
        case 5: /* exp_decay — 1000 * exp(-5*t) */
            v = 1000.0f * expf(-5.0f * (float)i / (float)n);
            break;
        case 6: /* sawtooth — 16 repeating ramps per chunk */
        {
            size_t period = n / 16u;
            v = (float)(i % period) / (float)period * 1000.0f;
            break;
        }
        case 7: /* impulse_train — narrow spikes every 1024 elements */
            v = ((i % 1024u) < 4u) ? (5000.0f + (r - 0.5f) * 200.0f) : 0.0f;
            break;
        default:
            v = 0.0f;
            break;
        }
        buf[i] = v;
    }
}

/* GPU-side byte-exact comparison — accumulates mismatch count */
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

/* ============================================================
 * Pattern names
 * ============================================================ */

static const char *PATTERN_NAMES[N_PATTERNS] = {
    "constant", "smooth_sine", "ramp", "sparse",
    "step", "exp_decay", "sawtooth", "impulse_train"
};

/* ============================================================
 * Fill mode helpers
 * ============================================================ */

#define MODE_CYCLING    0
#define MODE_UNIFORM    1
#define MODE_CONTIGUOUS 2
#define UNIFORM_PATTERN 2  /* ramp */

static int get_pattern(int chunk_idx, int fill_mode, int n_chunks)
{
    switch (fill_mode) {
    case MODE_UNIFORM:    return UNIFORM_PATTERN;
    case MODE_CONTIGUOUS: return (chunk_idx / (n_chunks / N_PATTERNS)) % N_PATTERNS;
    default:              return chunk_idx % N_PATTERNS;
    }
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
 * GPU fill — entire dataset d_full[0..n_floats)
 * ============================================================ */

static void fill_dataset(float *d_full, size_t n_floats, size_t chunk_floats,
                         int fill_mode)
{
    int n_chunks = (int)(n_floats / chunk_floats);
    int threads  = 256;
    int blocks   = 512;   /* enough for good occupancy at any chunk size */

    for (int c = 0; c < n_chunks; c++) {
        int pattern = get_pattern(c, fill_mode, n_chunks);
        fill_chunk_kernel<<<blocks, threads>>>(d_full + c * chunk_floats,
                                              chunk_floats, pattern,
                                              (unsigned long long)c);
    }
    cudaDeviceSynchronize();
}

/* ============================================================
 * GPU compare — returns number of mismatches
 * ============================================================ */

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
 * File size helper
 * ============================================================ */

static size_t file_size(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

/* Flush file to disk and evict from page cache so reads measure real I/O. */
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

/* DCPL for ALGO_AUTO lossless */
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

/* DCPL for static: lz4 + 4-byte shuffle */
static hid_t make_dcpl_static(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 1; /* GPUCOMPRESS_ALGO_LZ4 */
    cd[1] = 2; /* GPUCOMPRESS_PREPROC_SHUFFLE_4 */
    cd[2] = 4; /* 4-byte shuffle element size (float32) */
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

/* DCPL for no-comp: chunked, no filter */
static hid_t make_dcpl_nocomp(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
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
    char   phase[20];     /* "no-comp" / "static" / "nn" / "nn-rl" / "nn-rl+exp50" */
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    double read_mbps;
    unsigned long long mismatches;
    int    sgd_fires;     /* nn phases with online learning */
    int    explorations;  /* chunks where Level-2 exploration triggered */
    int    n_chunks;
} PhaseResult;

/* ============================================================
 * Phase 1: no-comp
 *   GPU fill → H5Dwrite (GPU ptr, VOL-2 fallback: D→H internally, no filter)
 *           → H5Dread  (GPU ptr, VOL-2 fallback: read natively, H→D internally)
 *           → GPU compare
 *
 * Tests VOL-2 fix: VOL detects GPU ptr + no gpucompress filter, performs the
 * D→H / H→D copies internally and forwards to the native connector.
 * A WARNING line is printed to stderr by the VOL for each call.
 * ============================================================ */

static int run_phase_nocomp(float *d_full, float *d_read,
                            unsigned long long *d_count,
                            float *h_buf, size_t n_floats,
                            size_t chunk_floats, int fill_mode,
                            PhaseResult *r)
{
    (void)h_buf;   /* no longer used — VOL-2 fallback handles D<->H internally */

    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (int)(n_floats / chunk_floats);

    printf("[no-comp] Filling GPU buffer...\n");
    fill_dataset(d_full, n_floats, chunk_floats, fill_mode);

    /* Write: pass GPU pointer directly — no gpucompress filter on DCPL.
     * VOL-2 fallback activates: D→H copy + native write, all inside H5Dwrite. */
    printf("[no-comp] H5Dwrite (GPU ptr, VOL-2 fallback D->H)... "); fflush(stdout);
    double t0 = now_ms();

    remove(TMP_NOCOMP);
    hsize_t dims[1] = { (hsize_t)n_floats };
    hid_t dcpl = make_dcpl_nocomp((hsize_t)chunk_floats);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_NOCOMP, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    assert(file >= 0 && "[no-comp] H5Fcreate failed");
    H5Pclose(fapl);
    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    assert(dset >= 0 && "[no-comp] H5Dcreate2 failed");
    H5Sclose(fsp);
    H5Pclose(dcpl);
    herr_t wrc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
    assert(wrc >= 0 && "[no-comp] H5Dwrite(GPU ptr) failed — VOL-2 fallback broken");
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();
    printf("%.0f ms\n", t1 - t0);

    /* Evict from page cache so the read measures real disk I/O */
    drop_pagecache(TMP_NOCOMP);

    /* Read: pass GPU pointer directly — VOL-2 fallback activates:
     * native read into host staging buffer, H→D copy, all inside H5Dread. */
    printf("[no-comp] H5Dread (GPU ptr, VOL-2 fallback H->D)... "); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_NOCOMP, H5F_ACC_RDONLY, fapl);
    assert(file >= 0 && "[no-comp] H5Fopen failed");
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    assert(dset >= 0 && "[no-comp] H5Dopen2 failed");
    double t2 = now_ms();
    herr_t rrc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    assert(rrc >= 0 && "[no-comp] H5Dread(GPU ptr) failed — VOL-2 fallback broken");
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = now_ms();
    printf("%.0f ms\n", t3 - t2);

    /* Regenerate reference in d_full, then compare */
    fill_dataset(d_full, n_floats, chunk_floats, fill_mode);
    unsigned long long mm = gpu_compare(d_full, d_read, n_floats, d_count);

    size_t fbytes = file_size(TMP_NOCOMP);
    r->write_ms    = t1 - t0;
    r->read_ms     = t3 - t2;
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps  = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps   = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches  = mm;
    r->sgd_fires   = 0;
    r->n_chunks    = n_chunks;
    snprintf(r->phase, sizeof(r->phase), "no-comp");

    printf("[no-comp] ratio=%.2fx  write=%.0f MB/s  read=%.0f MB/s  "
           "file=%.0f MiB  mismatches=%llu\n",
           r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Phase 2/3: VOL phases (static or nn)
 *   GPU fill → H5Dwrite (VOL) → H5Dread (VOL) → GPU compare
 * ============================================================ */

static int run_phase_vol(float *d_full, float *d_read,
                         unsigned long long *d_count,
                         size_t n_floats, size_t chunk_floats, int fill_mode,
                         const char *phase_name, const char *tmp_file,
                         hid_t dcpl,
                         PhaseResult *r)
{
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (int)(n_floats / chunk_floats);

    printf("[%s] Filling GPU buffer...\n", phase_name);
    fill_dataset(d_full, n_floats, chunk_floats, fill_mode);

    /* VOL write */
    printf("[%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);
    hsize_t dims[1] = { (hsize_t)n_floats };

    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file); return 1; }

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0   = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
    H5Dclose(dset);
    H5Fclose(file);
    double t1   = now_ms();
    printf("%.0f ms\n", t1 - t0);

    if (wret < 0) { fprintf(stderr, "[%s] H5Dwrite failed\n", phase_name); return 1; }

    /* Evict from page cache so the read measures real disk I/O */
    drop_pagecache(tmp_file);

    /* VOL read: open file outside timer, measure only H5Dread + GPU sync */
    printf("[%s] H5Dread (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    double t2   = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    double t3   = now_ms();
    H5Dclose(dset);
    H5Fclose(file);
    printf("%.0f ms\n", t3 - t2);

    if (rret < 0) { fprintf(stderr, "[%s] H5Dread failed\n", phase_name); return 1; }

    /* Regenerate reference in d_full, then GPU compare */
    fill_dataset(d_full, n_floats, chunk_floats, fill_mode);
    unsigned long long mm = gpu_compare(d_full, d_read, n_floats, d_count);

    /* Collect per-chunk diagnostics */
    int sgd_fires  = 0;
    int explorations = 0;
    int n_hist     = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            sgd_fires    += d.sgd_fired;
            explorations += d.exploration_triggered;
        }
    }

    size_t fbytes = file_size(tmp_file);
    r->write_ms    = t1 - t0;
    r->read_ms     = t3 - t2;
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps  = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps   = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches  = mm;
    r->sgd_fires   = sgd_fires;
    r->explorations = explorations;
    r->n_chunks    = n_chunks;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    printf("[%s] ratio=%.2fx  write=%.0f MB/s  read=%.0f MB/s  "
           "file=%.0f MiB  sgd=%d expl=%d/%d  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), sgd_fires, explorations, n_hist, mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * CSV output
 * ============================================================ */

static void write_aggregate_csv(PhaseResult *res, int n_phases,
                                int dataset_mb, int chunk_mb,
                                const char *mode_name)
{
    FILE *f = fopen(OUT_CSV, "w");
    if (!f) { perror("fopen " OUT_CSV); return; }

    fprintf(f, "phase,dataset_mb,chunk_mb,fill_mode,"
               "write_ms,read_ms,file_mb,ratio,"
               "write_mbps,read_mbps,mismatches,sgd_fires,explorations,n_chunks\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        fprintf(f, "%s,%d,%d,%s,"
                   "%.2f,%.2f,%.2f,%.4f,"
                   "%.1f,%.1f,%llu,%d,%d,%d\n",
                r->phase, dataset_mb, chunk_mb, mode_name,
                r->write_ms, r->read_ms,
                (double)r->file_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps,
                r->mismatches, r->sgd_fires, r->explorations, r->n_chunks);
    }
    fclose(f);
    printf("\nAggregate CSV written to: %s\n", OUT_CSV);
}

static void write_chunk_csv(const char *phase_name,
                            int fill_mode, int n_chunks)
{
    FILE *f;
    static int header_written = 0;

    if (!header_written) {
        f = fopen(OUT_CHUNKS, "w");
        if (!f) { perror("fopen " OUT_CHUNKS); return; }
        fprintf(f, "phase,chunk,pattern,nn_action,explored,sgd_fired\n");
        fclose(f);
        header_written = 1;
    }

    f = fopen(OUT_CHUNKS, "a");
    if (!f) { perror("fopen " OUT_CHUNKS); return; }

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist && i < n_chunks; i++) {
        gpucompress_chunk_diag_t d;
        int pattern = get_pattern(i, fill_mode, n_chunks);
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            fprintf(f, "%s,%d,%s,%d,%d,%d\n",
                    phase_name, i, PATTERN_NAMES[pattern],
                    d.nn_action, d.exploration_triggered, d.sgd_fired);
        } else {
            fprintf(f, "%s,%d,%s,-1,0,0\n",
                    phase_name, i, PATTERN_NAMES[pattern]);
        }
    }
    fclose(f);
}

/* ============================================================
 * Console summary table
 * ============================================================ */

static void print_summary(PhaseResult *res, int n_phases,
                          int dataset_mb, int chunk_mb, const char *mode_name)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  GPU-Native VOL Benchmark Summary                                    ║\n");
    printf("║  Dataset: %4d MB  Chunk: %d MB  Fill: %-10s                   ║\n",
           dataset_mb, chunk_mb, mode_name);
    printf("╠══════════════╦══════════╦══════════╦═══════╦══════════╦═════════════╣\n");
    printf("║  Phase       ║ Write    ║ Read     ║ Ratio ║ File MB  ║ Verify      ║\n");
    printf("║              ║ (MB/s)   ║ (MB/s)   ║       ║          ║             ║\n");
    printf("╠══════════════╬══════════╬══════════╬═══════╬══════════╬═════════════╣\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        const char *verdict = (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %8.0f ║ %8.0f ║ %5.2fx ║ %8.0f ║ %-11s ║\n",
               r->phase, r->write_mbps, r->read_mbps,
               r->ratio, (double)r->file_bytes / (1 << 20), verdict);
    }
    printf("╚══════════════╩══════════╩══════════╩═══════╩══════════╩═════════════╝\n");

    /* NN phase SGD/exploration summary */
    for (int i = 0; i < n_phases; i++) {
        if (strncmp(res[i].phase, "nn", 2) == 0 && res[i].n_chunks > 0) {
            printf("\n  %-14s SGD fired: %d/%d chunks  Explorations: %d/%d chunks",
                   res[i].phase,
                   res[i].sgd_fires,   res[i].n_chunks,
                   res[i].explorations, res[i].n_chunks);
            printf("\n");
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
    int dataset_mb = DEFAULT_DATASET_MB;
    int chunk_mb   = DEFAULT_CHUNK_MB;
    int fill_mode  = MODE_CYCLING;
    const char *mode_name = DEFAULT_MODE;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset-mb") == 0 && i + 1 < argc) {
            dataset_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_name = argv[++i];
            if (strcmp(mode_name, "uniform") == 0)     fill_mode = MODE_UNIFORM;
            else if (strcmp(mode_name, "contiguous") == 0) fill_mode = MODE_CONTIGUOUS;
            else                                        fill_mode = MODE_CYCLING;
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--dataset-mb N] "
                "[--chunk-mb N] [--mode cycling|uniform|contiguous]\n", argv[0]);
        return 1;
    }

    size_t chunk_floats  = (size_t)chunk_mb   * 1024 * 1024 / sizeof(float);
    size_t n_floats      = (size_t)dataset_mb * 1024 * 1024 / sizeof(float);
    size_t total_bytes   = n_floats * sizeof(float);
    int    n_chunks      = (int)(n_floats / chunk_floats);

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  GPU-Native VOL Benchmark: No-Comp vs Static vs NN+SGD              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Dataset  : %d MB (%zu floats, %d chunks of %d MB)\n",
           dataset_mb, n_floats, n_chunks, chunk_mb);
    printf("  Fill mode: %s\n", mode_name);
    printf("  Weights  : %s\n", weights_path);
    printf("  Patterns : 8 GPU-generated (constant/sine/ramp/sparse/step/\n");
    printf("             exp_decay/sawtooth/impulse_train)\n\n");

    /* Create output directory */
    mkdir(OUT_DIR, 0755);

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

    /* Allocate GPU buffers */
    float *d_full = NULL, *d_read = NULL;
    unsigned long long *d_count = NULL;

    printf("[Alloc] Allocating %.0f MiB on GPU (x2 for write+read)...\n",
           (double)total_bytes / (1 << 20));
    if (cudaMalloc(&d_full,  total_bytes) != cudaSuccess ||
        cudaMalloc(&d_read,  total_bytes) != cudaSuccess ||
        cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed — need %.0f MiB free GPU memory\n",
                2.0 * total_bytes / (1 << 20));
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    /* Host buffer for no-comp phase */
    float *h_buf = (float *)malloc(total_bytes);
    if (!h_buf) {
        fprintf(stderr, "FATAL: malloc(%.0f MiB) failed\n",
                (double)total_bytes / (1 << 20));
        cudaFree(d_full); cudaFree(d_read); cudaFree(d_count);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }
    printf("[Alloc] Host buffer: %.0f MiB\n", (double)total_bytes / (1 << 20));

    PhaseResult results[5];
    int n_phases = 0;
    int any_fail = 0;

    /* ── Phase 1: no-comp ──────────────────────────────────────────── */
    printf("\n── Phase 1/5: no-comp (GPU→Host→HDF5) ────────────────────────\n");
    int rc = run_phase_nocomp(d_full, d_read, d_count, h_buf,
                              n_floats, chunk_floats, fill_mode,
                              &results[n_phases]);
    if (rc) any_fail = 1;
    n_phases++;

    /* ── Phase 2: static (VOL, lz4+shuffle) ───────────────────────── */
    printf("\n── Phase 2/5: static (VOL, lz4+4B-shuffle) ───────────────────\n");
    hid_t dcpl_static = make_dcpl_static((hsize_t)chunk_floats);
    rc = run_phase_vol(d_full, d_read, d_count, n_floats, chunk_floats,
                       fill_mode, "static", TMP_STATIC, dcpl_static,
                       &results[n_phases]);
    H5Pclose(dcpl_static);
    if (rc) any_fail = 1;
    write_chunk_csv("static", fill_mode, n_chunks);
    n_phases++;

    /* ── Phase 3: nn (VOL, ALGO_AUTO, no online learning) ─────────── */
    printf("\n── Phase 3/5: nn (VOL, ALGO_AUTO, inference-only) ────────────\n");
    /* Online learning is intentionally OFF here: pure NN routing, no SGD,
     * no exploration. Establishes the baseline for per-chunk algorithm selection
     * without any gradient updates to the model weights. */
    hid_t dcpl_nn = make_dcpl_auto((hsize_t)chunk_floats);
    rc = run_phase_vol(d_full, d_read, d_count, n_floats, chunk_floats,
                       fill_mode, "nn", TMP_NN, dcpl_nn,
                       &results[n_phases]);
    H5Pclose(dcpl_nn);
    if (rc) any_fail = 1;
    write_chunk_csv("nn", fill_mode, n_chunks);
    n_phases++;

    /* ── Phase 4: nn-rl (ALGO_AUTO + SGD, no exploration) ─────────── */
    /* Online learning ON, exploration OFF.
     * SGD fires whenever ratio MAPE > 20% (REINFORCE_MAPE).
     * Only the primary action's ratio is fed to the SGD kernel
     * (explored_samples has exactly one entry).
     * NOTE: model weights are updated in-place on GPU during this phase,
     * so Phase 5 starts from Phase 4's final weights. */
    printf("\n── Phase 4/5: nn-rl (ALGO_AUTO + SGD, MAPE≥20%%, LR=0.4) ────\n");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    hid_t dcpl_rl = make_dcpl_auto((hsize_t)chunk_floats);
    rc = run_phase_vol(d_full, d_read, d_count, n_floats, chunk_floats,
                       fill_mode, "nn-rl", TMP_NN_RL, dcpl_rl,
                       &results[n_phases]);
    H5Pclose(dcpl_rl);
    gpucompress_disable_online_learning();   /* stop SGD; reset exploration flag */
    if (rc) any_fail = 1;
    write_chunk_csv("nn-rl", fill_mode, n_chunks);
    n_phases++;

    /* ── Phase 5: nn-rl+exp50 (ALGO_AUTO + SGD + Level-2 exploration) */
    /* Same SGD as Phase 4, plus Level-2 exploration when MAPE > 50%.
     * When exploration triggers, up to K alternative configs are compressed
     * and timed; the top 3 by ratio are fed to GPU SGD instead of just
     * the primary action, giving richer gradient signal.
     * Model weights carry over from Phase 4's updates.
     * Exploration threshold 50% (vs SGD threshold 20%) means SGD fires broadly
     * but exploration only kicks in for the worst-predicted chunks. */
    printf("\n── Phase 5/5: nn-rl+exp50 (ALGO_AUTO + SGD + expl@MAPE≥50%%) ─\n");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.50);
    hid_t dcpl_rlexp = make_dcpl_auto((hsize_t)chunk_floats);
    rc = run_phase_vol(d_full, d_read, d_count, n_floats, chunk_floats,
                       fill_mode, "nn-rl+exp50", TMP_NN_RLEXP, dcpl_rlexp,
                       &results[n_phases]);
    H5Pclose(dcpl_rlexp);
    gpucompress_disable_online_learning();
    if (rc) any_fail = 1;
    write_chunk_csv("nn-rl+exp50", fill_mode, n_chunks);
    n_phases++;

    /* ── Summary ───────────────────────────────────────────────────── */
    print_summary(results, n_phases, dataset_mb, chunk_mb, mode_name);
    write_aggregate_csv(results, n_phases, dataset_mb, chunk_mb, mode_name);
    printf("Per-chunk CSV written to: %s\n", OUT_CHUNKS);

    /* ── Cleanup ───────────────────────────────────────────────────── */
    free(h_buf);
    cudaFree(d_full);
    cudaFree(d_read);
    cudaFree(d_count);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    remove(TMP_NOCOMP);
    remove(TMP_STATIC);
    remove(TMP_NN);
    remove(TMP_NN_RL);
    remove(TMP_NN_RLEXP);

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}
