/**
 * @file benchmark_nn_adaptiveness.cu
 * @brief NN Model Adaptiveness Benchmark: SGD & Exploration Hyperparameter Study
 *
 * Generates 5 different GPU-resident data patterns (~3.2GB each at L=928)
 * and runs hyperparameter grid searches over SGD learning rate / MAPE threshold
 * and exploration threshold / K alternatives.
 *
 * Usage:
 *   ./build/benchmark_nn_adaptiveness model.nnwt \
 *       [--L 928] [--chunk-mb 64] [--quick]
 *
 * Options:
 *   model.nnwt       Path to NN weights (required, or GPUCOMPRESS_WEIGHTS env)
 *   --L N            Grid side length (dataset = N^3 floats, default 928)
 *   --chunk-mb N     Chunk size in MB (default 64)
 *   --quick          Reduced hyperparameter grid (~55 runs vs ~255)
 *
 * Output:
 *   benchmarks/nn_adaptiveness/sgd_study.csv
 *   benchmarks/nn_adaptiveness/exploration_study.csv
 *   benchmarks/nn_adaptiveness/chunks_detail.csv
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

#include "data_generators.cuh"

/* ============================================================
 * Constants
 * ============================================================ */

#define DEFAULT_L          928
#define DEFAULT_CHUNK_MB   64

#define TMP_FILE           "/tmp/bm_nn_adapt.h5"

#define OUT_DIR            "benchmarks/nn_adaptiveness"
#define OUT_SGD_CSV        OUT_DIR "/sgd_study.csv"
#define OUT_EXPL_CSV       OUT_DIR "/exploration_study.csv"
#define OUT_CHUNKS_CSV     OUT_DIR "/chunks_detail.csv"
#define OUT_UB_CSV         OUT_DIR "/upper_bound.csv"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define N_PATTERNS 5
#define GS_CHAOS_F 0.014f
#define GS_CHAOS_K 0.045f
#define GS_CHAOS_STEPS 5000

/* ============================================================
 * Action decoding (mirrors nnInferenceKernel thread mapping)
 *
 * Action ID encoding (0-31):
 *   algo_idx = action % 8    (0=lz4, 1=snappy, 2=deflate, 3=zstd,
 *                              4=ans, 5=cascaded, 6=bitcomp, 7=gdeflate)
 *   quant    = (action/8) % 2 (0=off, 1=on)
 *   shuffle  = (action/16) % 2 (0=off, 1=4-byte shuffle)
 * ============================================================ */

static const char *ACTION_ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_string(int action, char *buf, size_t bufsz)
{
    if (action < 0 || action > 31) {
        snprintf(buf, bufsz, "act%d", action);
        return;
    }
    int algo    = action % 8;
    int quant   = (action / 8) % 2;
    int shuffle = (action / 16) % 2;
    const char *algo_name = ACTION_ALGO_NAMES[algo];

    int n = snprintf(buf, bufsz, "%s", algo_name);
    if (shuffle) n += snprintf(buf + n, bufsz - n, "+shuf4");
    if (quant)   snprintf(buf + n, bufsz - n, "+quant");
}

static void print_action_summary(void)
{
    int n_hist = gpucompress_get_chunk_history_count();
    if (n_hist == 0) return;

    /* Count action frequencies */
    int counts[32] = {0};
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0 && d.nn_action >= 0 && d.nn_action < 32)
            counts[d.nn_action]++;
    }

    /* Find top-3 actions */
    int top[3] = {-1, -1, -1};
    for (int t = 0; t < 3; t++) {
        int best = -1, best_count = 0;
        for (int a = 0; a < 32; a++) {
            if (counts[a] > best_count) {
                int already = 0;
                for (int tt = 0; tt < t; tt++)
                    if (top[tt] == a) already = 1;
                if (!already) { best = a; best_count = counts[a]; }
            }
        }
        top[t] = best;
    }

    printf("    actions: ");
    for (int t = 0; t < 3; t++) {
        if (top[t] < 0 || counts[top[t]] == 0) break;
        char buf[64];
        action_to_string(top[t], buf, sizeof(buf));
        printf("%s=%d/%d(%.0f%%)  ", buf, counts[top[t]], n_hist,
               100.0 * counts[top[t]] / n_hist);
    }
    int n_unique = 0;
    for (int a = 0; a < 32; a++) if (counts[a] > 0) n_unique++;
    printf("[%d unique]\n", n_unique);
}

static const char* chunk_pattern_name(int chunk_idx, int n_chunks);

static void print_per_chunk_ratios(int n_chunks)
{
    int n_hist = gpucompress_get_chunk_history_count();
    if (n_hist == 0) return;

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        char abuf[64];
        action_to_string(d.nn_action, abuf, sizeof(abuf));
        printf("    chunk %3d/%-3d [%-18s] action=%-16s ratio=%6.2fx  pred=%6.2fx\n",
               i + 1, n_hist, chunk_pattern_name(i, n_chunks),
               abuf, d.actual_ratio, d.predicted_ratio);
    }
}

/* Forward declaration — set from --error-bound, used by gpu_compare */
static double g_error_bound;

/* ============================================================
 * GPU-side comparison (exact if error_bound==0, tolerance otherwise)
 * ============================================================ */

__global__ void count_mismatches_kernel(const float * __restrict__ a,
                                        const float * __restrict__ b,
                                        size_t n,
                                        float tolerance,
                                        unsigned long long *count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (tolerance == 0.0f) {
            if (a[i] != b[i]) local++;
        } else {
            float diff = a[i] - b[i];
            if (diff < 0) diff = -diff;
            if (diff > tolerance) local++;
        }
    }
    atomicAdd(count, local);
}

static unsigned long long gpu_compare(const float *a, const float *b,
                                      size_t n_floats,
                                      unsigned long long *d_count)
{
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    count_mismatches_kernel<<<512, 256>>>(a, b, n_floats,
                                          (float)g_error_bound, d_count);
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

static hid_t make_dcpl_auto(int L, int chunk_z)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(g_error_bound, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
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
 * Composite dataset generator
 *
 * Fills d_data (L^3 floats) with all 5 patterns laid out along Z:
 *   z = [0, slab)          → pattern 0: ocean waves
 *   z = [slab, 2*slab)     → pattern 1: heating surface
 *   z = [2*slab, 3*slab)   → pattern 2: turbulent flow (Gray-Scott)
 *   z = [3*slab, 4*slab)   → pattern 3: geological strata
 *   z = [4*slab, L)        → pattern 4: particle shower
 *
 * Chunks (size L×L×chunk_z) traverse Z, so the NN encounters
 * pattern transitions within a single contiguous write.
 *
 * d_scratch must be at least L*L*slab floats (reuse d_readback).
 * ============================================================ */

static void generate_all_patterns(float *d_data, float *d_scratch, int L,
                                  unsigned int seed,
                                  gpucompress_grayscott_t gs_sim, int gs_L)
{
    size_t N = (size_t)L * L * L;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    if (blocks > 2048) blocks = 2048;

    /* Generate patterns 0,1,3,4 inline; pattern 2 region filled as zeros */
    gen_composite<<<blocks, threads>>>(d_data, L, seed);
    cudaDeviceSynchronize();

    /* Pattern 2: Gray-Scott chaos.
     * The sim runs at gs_L (= L/5) to save memory.
     * Copy gs_L×gs_L worth of Z-slices into the pattern-2 region.
     * For (x,y) within [0, gs_L), copy slab floats from gs_v to d_data. */
    int slab = L / 5;
    int copy_xy = (gs_L < L) ? gs_L : L;   /* min(gs_L, L) */
    int copy_z  = (slab < gs_L) ? slab : gs_L;  /* min(slab, gs_L) */

    gpucompress_grayscott_init(gs_sim);
    gpucompress_grayscott_run(gs_sim, GS_CHAOS_STEPS);
    cudaDeviceSynchronize();

    float *d_u = NULL, *d_v = NULL;
    gpucompress_grayscott_get_device_ptrs(gs_sim, &d_u, &d_v);

    /* Copy row-by-row: source is gs_L^3, destination is L^3 */
    size_t row_bytes = (size_t)copy_z * sizeof(float);
    for (int x = 0; x < copy_xy; x++) {
        for (int y = 0; y < copy_xy; y++) {
            size_t src_off = (size_t)x * gs_L * gs_L + (size_t)y * gs_L;
            size_t dst_off = (size_t)x * L * L + (size_t)y * L + 2 * slab;
            cudaMemcpy(d_data + dst_off, d_v + src_off,
                       row_bytes, cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();

    printf("    Generated composite dataset: %d^3, slab=%d (%.1f MB/pattern)\n",
           L, slab, (double)L * L * slab * sizeof(float) / (1 << 20));
    for (int p = 0; p < N_PATTERNS; p++) {
        int z0 = p * slab;
        int z1 = (p == 4) ? L : (p + 1) * slab;
        printf("      z=[%d,%d) → %s\n", z0, z1, PATTERN_NAMES[p]);
    }
}

/* ============================================================
 * Single run: write + read + verify + collect diagnostics
 *
 * Returns 0 on success (mismatches == 0), 1 on failure.
 * ============================================================ */

typedef struct {
    double  ratio;
    double  ratio_min;
    double  ratio_max;
    double  ratio_stddev;
    double  final_mape;
    int     convergence_chunks;
    int     n_chunks;
    int     sgd_fires;
    int     explorations;
    double  total_write_ms;
    double  total_sgd_ms;
    double  total_nn_inference_ms;
    double  total_compression_ms;
    double  total_exploration_ms;
    double  write_mibps;
    double  file_mib;
    unsigned long long mismatches;
} RunResult;

static int do_single_run(float *d_data, float *d_readback, size_t n_floats,
                          int L, int chunk_z, unsigned long long *d_count,
                          RunResult *result)
{
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    /* Write */
    remove(TMP_FILE);
    hid_t dcpl = make_dcpl_auto(L, chunk_z);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { H5Pclose(dcpl); return 1; }

    hid_t fsp  = H5Screate_simple(3, dims, NULL);
    hid_t dset = H5Dcreate2(file, "D", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);
    H5Pclose(dcpl);

    double t0 = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();
    if (wret < 0) return 1;

    drop_pagecache(TMP_FILE);

    /* Read */
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "D", H5P_DEFAULT);

    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    if (rret < 0) return 1;

    /* Verify */
    unsigned long long mm = gpu_compare(d_data, d_readback, n_floats, d_count);

    /* Collect diagnostics from chunk history */
    int n_hist = gpucompress_get_chunk_history_count();
    double sum_nn = 0, sum_comp = 0, sum_expl = 0, sum_sgd = 0;
    double r_min = 1e30, r_max = 0, r_sum = 0, r_sum2 = 0;
    double mape_sum = 0;
    int mape_count = 0, n_valid = 0;
    int sgd_fires = 0, explorations = 0;

    /* For convergence tracking: sliding window MAPE over last CONV_WINDOW chunks.
     * A cumulative average never recovers from early high-error chunks, so we
     * use a ring buffer of the most recent MAPE values instead. */
#define CONV_WINDOW 10
#define CONV_THRESHOLD 0.15  /* 15% — realistic for NN ratio prediction */
    double conv_ring[CONV_WINDOW];
    int conv_ring_pos = 0;
    int conv_ring_count = 0;
    int convergence_chunk = 0;  /* first chunk where sliding MAPE < threshold */

    /* Final MAPE: over last 10% of chunks */
    int last_10pct_start = n_hist - (n_hist / 10);
    if (last_10pct_start < 0) last_10pct_start = 0;
    double final_mape_sum = 0;
    int final_mape_count = 0;

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        sum_nn   += d.nn_inference_ms;
        sum_comp += d.compression_ms;
        sum_expl += d.exploration_ms;
        sum_sgd  += d.sgd_update_ms;
        sgd_fires    += d.sgd_fired;
        explorations += d.exploration_triggered;

        if (d.actual_ratio > 0) {
            if (d.actual_ratio < r_min) r_min = d.actual_ratio;
            if (d.actual_ratio > r_max) r_max = d.actual_ratio;
            r_sum  += d.actual_ratio;
            r_sum2 += (double)d.actual_ratio * d.actual_ratio;
            n_valid++;
        }

        if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
            double chunk_mape = fabs((double)d.predicted_ratio - d.actual_ratio)
                                / d.actual_ratio;
            mape_sum += chunk_mape;
            mape_count++;

            conv_ring[conv_ring_pos] = chunk_mape;
            conv_ring_pos = (conv_ring_pos + 1) % CONV_WINDOW;
            if (conv_ring_count < CONV_WINDOW) conv_ring_count++;
            if (convergence_chunk == 0 && conv_ring_count == CONV_WINDOW) {
                double window_sum = 0;
                for (int w = 0; w < CONV_WINDOW; w++) window_sum += conv_ring[w];
                if (window_sum / CONV_WINDOW < CONV_THRESHOLD) {
                    convergence_chunk = i + 1;
                }
            }

            if (i >= last_10pct_start) {
                final_mape_sum += chunk_mape;
                final_mape_count++;
            }
        }
    }

    size_t fbytes = file_size(TMP_FILE);
    double write_ms = t1 - t0;

    result->ratio          = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    result->ratio_min      = (n_valid > 0) ? r_min : 0;
    result->ratio_max      = (n_valid > 0) ? r_max : 0;
    if (n_valid > 1) {
        double mean = r_sum / n_valid;
        result->ratio_stddev = sqrt((r_sum2 / n_valid) - mean * mean);
    } else {
        result->ratio_stddev = 0;
    }
    result->final_mape          = (final_mape_count > 0)
                                    ? (final_mape_sum / final_mape_count) * 100.0 : 0;
    result->convergence_chunks  = convergence_chunk;
    result->n_chunks            = n_chunks;
    result->sgd_fires           = sgd_fires;
    result->explorations        = explorations;
    result->total_write_ms      = write_ms;
    result->total_sgd_ms        = sum_sgd;
    result->total_nn_inference_ms = sum_nn;
    result->total_compression_ms  = sum_comp;
    result->total_exploration_ms  = sum_expl;
    result->write_mibps         = (double)total_bytes / (1 << 20) / (write_ms / 1000.0);
    result->file_mib            = (double)fbytes / (1 << 20);
    result->mismatches          = mm;

    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Best-Static Upper Bound: try all configs per chunk exhaustively
 * ============================================================ */

typedef struct {
    int    best_action;
    double best_ratio;
    size_t best_compressed_size;
} OracleChunkResult;

static void action_to_config(int action, gpucompress_config_t *cfg)
{
    *cfg = gpucompress_default_config();
    int algo_idx = action % 8;
    int quant    = (action / 8) % 2;
    int shuffle  = (action / 16) % 2;

    /* algo_idx 0..7 maps to enum values 1..8 */
    cfg->algorithm = (gpucompress_algorithm_t)(algo_idx + 1);
    cfg->preprocessing = 0;
    if (shuffle) cfg->preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
    if (quant)   cfg->preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
    cfg->error_bound = g_error_bound;
}

/*
 * Gather one HDF5 chunk from d_data into d_chunk_buf (contiguous).
 * Dataset is [L][L][L], chunk is [L][L][actual_cz].
 * In C-order, each (x,y) row has L contiguous z-values;
 * we copy actual_cz of them per row, for L*L rows.
 */
static void gather_chunk(float *d_chunk_buf, const float *d_data,
                          int L, int chunk_z, int chunk_idx)
{
    int z0 = chunk_idx * chunk_z;
    int actual_cz = chunk_z;
    if (z0 + actual_cz > L) actual_cz = L - z0;

    cudaMemcpy2D(d_chunk_buf, (size_t)actual_cz * sizeof(float),
                 d_data + z0,  (size_t)L * sizeof(float),
                 (size_t)actual_cz * sizeof(float),
                 (size_t)L * L,
                 cudaMemcpyDeviceToDevice);
}

static int run_oracle_pass(const float *d_data, int L, int chunk_z,
                            int n_chunks,
                            OracleChunkResult *results,
                            double *oracle_ratio_out,
                            FILE *f_ub)
{
    size_t max_chunk_floats = (size_t)L * L * chunk_z;
    size_t max_chunk_bytes  = max_chunk_floats * sizeof(float);
    size_t out_buf_size = max_chunk_bytes + 65536;

    float *d_chunk_buf = NULL;
    void  *d_output    = NULL;

    if (cudaMalloc(&d_chunk_buf, max_chunk_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&d_output, out_buf_size) != cudaSuccess) {
        cudaFree(d_chunk_buf); return 1;
    }

    /* Algorithm names indexed by gpucompress_algorithm_t enum (1-8) */
    static const char *ALGO_NAMES[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    size_t total_original   = 0;
    size_t total_best_compressed = 0;

    for (int c = 0; c < n_chunks; c++) {
        int z0 = c * chunk_z;
        int actual_cz = chunk_z;
        if (z0 + actual_cz > L) actual_cz = L - z0;
        size_t chunk_bytes = (size_t)L * L * actual_cz * sizeof(float);

        gather_chunk(d_chunk_buf, d_data, L, chunk_z, c);
        cudaDeviceSynchronize();

        double best_ratio = 0;
        size_t best_compressed = chunk_bytes;
        const char *best_algo = "none";
        int best_shuffle = 0;
        int best_quant = 0;

        /* Try every algorithm × preprocessing combination directly */
        for (int algo_enum = 1; algo_enum <= 8; algo_enum++) {
            for (int shuffle = 0; shuffle <= 1; shuffle++) {
                for (int quant = 0; quant <= 1; quant++) {
                    /* Skip quantization in lossless mode */
                    if (quant && g_error_bound <= 0.0)
                        continue;

                    gpucompress_config_t cfg = gpucompress_default_config();
                    cfg.algorithm = (gpucompress_algorithm_t)algo_enum;
                    cfg.preprocessing = 0;
                    if (shuffle) cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                    if (quant)   cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
                    cfg.error_bound = g_error_bound;

                    size_t out_size = out_buf_size;
                    gpucompress_stats_t stats;
                    gpucompress_error_t err = gpucompress_compress_gpu(
                        d_chunk_buf, chunk_bytes,
                        d_output, &out_size, &cfg, &stats, NULL);

                    double ratio = (err == GPUCOMPRESS_SUCCESS)
                        ? stats.compression_ratio : 0;

                    /* Write every result to CSV */
                    if (f_ub) {
                        fprintf(f_ub, "%d,%s,%s,%d,%d,%g,%.4f,%zu,%zu,%s\n",
                                c, chunk_pattern_name(c, n_chunks),
                                ALGO_NAMES[algo_enum],
                                shuffle ? 4 : 0,
                                quant,
                                g_error_bound,
                                ratio,
                                (err == GPUCOMPRESS_SUCCESS) ? stats.compressed_size : 0,
                                chunk_bytes,
                                (err == GPUCOMPRESS_SUCCESS) ? "ok" : "fail");
                    }

                    if (ratio > best_ratio) {
                        best_ratio      = ratio;
                        best_compressed = stats.compressed_size;
                        best_algo       = ALGO_NAMES[algo_enum];
                        best_shuffle    = shuffle ? 4 : 0;
                        best_quant      = quant;
                    }
                }
            }
        }

        results[c].best_action          = -1; /* not used */
        results[c].best_ratio           = best_ratio;
        results[c].best_compressed_size = best_compressed;
        total_original       += chunk_bytes;
        total_best_compressed += best_compressed;

        printf("    chunk %2d/%-2d [%-18s] best=%-10s shuf=%d quant=%d  ratio=%6.2fx  eb=%g\n",
               c + 1, n_chunks, chunk_pattern_name(c, n_chunks),
               best_algo, best_shuffle, best_quant, best_ratio, g_error_bound);
    }

    *oracle_ratio_out = (total_best_compressed > 0)
        ? (double)total_original / (double)total_best_compressed : 1.0;

    cudaFree(d_chunk_buf);
    cudaFree(d_output);
    return 0;
}

/* ============================================================
 * Write chunk detail rows for current run
 * ============================================================ */

static const char* chunk_pattern_name(int chunk_idx, int n_chunks)
{
    if (n_chunks <= 0) return "unknown";
    int band = chunk_idx * N_PATTERNS / n_chunks;
    if (band >= N_PATTERNS) band = N_PATTERNS - 1;
    return PATTERN_NAMES[band];
}

static void write_chunk_detail(FILE *f, const char *study, int n_chunks,
                                float lr, float mape_thresh,
                                float expl_thresh, int K)
{
    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        fprintf(f, "%s,%s,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f\n",
                study, chunk_pattern_name(i, n_chunks),
                lr, mape_thresh, expl_thresh, K,
                i, d.nn_action, d.exploration_triggered, d.sgd_fired,
                d.actual_ratio, d.predicted_ratio,
                d.nn_inference_ms, d.compression_ms,
                d.exploration_ms, d.sgd_update_ms);
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
    int chunk_mb = DEFAULT_CHUNK_MB;
    int chunk_z  = 0;
    int quick    = 0;
    int no_explore = 0;
    int no_sgd = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc) {
            g_error_bound = atof(argv[++i]);
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--no-explore") == 0) {
            no_explore = 1;
        } else if (strcmp(argv[i], "--no-sgd") == 0) {
            no_sgd = 1;
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--chunk-mb N] [--error-bound E] [--quick] [--no-explore] [--no-sgd]\n",
                argv[0]);
        return 1;
    }

    /* Compute chunk_z from chunk_mb */
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
    printf("║  NN Adaptiveness Benchmark: SGD & Exploration Hyperparameter Study       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Mode     : %s\n", quick ? "QUICK" : "FULL");
    printf("  ErrBound : %g%s\n", g_error_bound, g_error_bound == 0.0 ? " (lossless)" : " (lossy)");
    printf("  Weights  : %s\n\n", weights_path);

    /* Hyperparameter grids */
    float sgd_lrs_full[]  = {0.25f, 0.5f, 0.75f};
    float sgd_mapes_full[] = {0.15f, 0.20f, 0.25f};
    float sgd_lrs_quick[]  = {0.05f, 0.2f, 0.8f};
    float sgd_mapes_quick[] = {0.20f, 0.30f};

    float expl_thresholds_full[]  = {0.10f, 0.20f, 0.30f, 0.50f};
    int   expl_ks_full[]          = {2, 4, 8, 16, 31};
    float expl_thresholds_quick[] = {0.10f, 0.30f};
    int   expl_ks_quick[]         = {4, 16};

    float *sgd_lrs   = quick ? sgd_lrs_quick   : sgd_lrs_full;
    float *sgd_mapes = quick ? sgd_mapes_quick  : sgd_mapes_full;
    int n_lrs   = no_sgd ? 0 : (quick ? 3 : 3);
    int n_mapes = no_sgd ? 0 : (quick ? 2 : 3);

    float *expl_thresholds = quick ? expl_thresholds_quick : expl_thresholds_full;
    int   *expl_ks         = quick ? expl_ks_quick         : expl_ks_full;
    int n_thresholds = no_explore ? 0 : (quick ? 2 : 4);
    int n_ks         = no_explore ? 0 : (quick ? 2 : 5);

    int total_runs = 1                                      /* baseline */
                   + n_lrs * n_mapes                       /* SGD study */
                   + n_thresholds * n_ks;                  /* exploration study */
    printf("  Total runs: %d (1 baseline + %d SGD + %d exploration)\n",
           total_runs,
           n_lrs * n_mapes,
           n_thresholds * n_ks);
    printf("  Dataset   : composite (5 patterns along Z, each %d slices)\n\n",
           L / 5);

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

    /* Declare all variables before first goto target */
    float *d_data = NULL, *d_readback = NULL;
    unsigned long long *d_count = NULL;
    gpucompress_grayscott_t gs_sim = NULL;
    FILE *f_sgd = NULL, *f_expl = NULL, *f_chunks = NULL, *f_ub = NULL;
    int run_num = 0;
    int any_fail = 0;
    double baseline_ratio = 0;
    double oracle_ratio = 0;
    OracleChunkResult *oracle_results = NULL;
    int gs_L = L / 5;
    if (gs_L < 32) gs_L = 32;

    /* Allocate GPU buffers */
    if (cudaMalloc(&d_data, n_floats * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc d_data (%.1f MB) failed\n", dataset_mb);
        goto cleanup;
    }
    if (cudaMalloc(&d_readback, n_floats * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc d_readback (%.1f MB) failed\n", dataset_mb);
        goto cleanup;
    }
    if (cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc d_count failed\n");
        goto cleanup;
    }

    /* Create Gray-Scott sim for pattern 2.
     * Only L/5 Z-slices are needed, so run at L/5 to save ~2×(L^3 - (L/5)^3)
     * bytes of GPU memory (e.g., ~15 GB saved at L=1260). */
    {
        GrayScottSettings gs = gray_scott_default_settings();
        gs.L = gs_L;
        gs.F = GS_CHAOS_F;
        gs.k = GS_CHAOS_K;
        gpucompress_grayscott_create(&gs_sim, &gs);
    }

    /* Open CSV files */
    f_sgd = fopen(OUT_SGD_CSV, "w");
    f_expl = fopen(OUT_EXPL_CSV, "w");
    f_chunks = fopen(OUT_CHUNKS_CSV, "w");
    f_ub = fopen(OUT_UB_CSV, "w");
    if (!f_sgd || !f_expl || !f_chunks || !f_ub) {
        fprintf(stderr, "FATAL: Cannot open output CSV files in %s\n", OUT_DIR);
        goto cleanup;
    }

    /* Write CSV headers */
    fprintf(f_sgd, "lr,mape_threshold,n_chunks,final_mape,convergence_chunks,"
                   "ratio,ratio_vs_upper_bound,sgd_fire_rate,total_write_ms,total_sgd_ms,"
                   "total_nn_inference_ms,total_compression_ms,"
                   "ratio_min,ratio_max,ratio_stddev,write_mibps,file_mib,mismatches\n");

    fprintf(f_expl, "expl_threshold,K,n_chunks,ratio,ratio_vs_baseline,ratio_vs_upper_bound,"
                    "exploration_trigger_rate,total_write_ms,total_exploration_ms,"
                    "total_sgd_ms,total_nn_inference_ms,total_compression_ms,"
                    "ratio_min,ratio_max,ratio_stddev,write_mibps,file_mib,mismatches\n");

    fprintf(f_chunks, "study,pattern_region,lr,mape_threshold,expl_threshold,K,"
                      "chunk_idx,nn_action,explored,sgd_fired,"
                      "actual_ratio,predicted_ratio,"
                      "nn_inference_ms,compression_ms,exploration_ms,sgd_update_ms\n");

    fprintf(f_ub, "chunk_idx,pattern_region,algorithm,shuffle_bytes,quantization,"
                  "error_bound,compression_ratio,compressed_bytes,original_bytes,status\n");

    /* Generate composite dataset once — d_data stays constant across all runs */
    printf("═══ GENERATING COMPOSITE DATASET ═══\n\n");
    generate_all_patterns(d_data, d_readback, L, 42, gs_sim, gs_L);

    /* Best-Static Upper Bound: exhaustive search per chunk */
    printf("\n═══ BEST-STATIC UPPER BOUND (exhaustive, %d configs x %d chunks, error_bound=%g%s) ═══\n\n",
           g_error_bound > 0 ? 32 : 16, n_chunks,
           g_error_bound, g_error_bound == 0.0 ? " lossless" : " lossy");
    oracle_results = (OracleChunkResult *)calloc(n_chunks, sizeof(OracleChunkResult));
    if (oracle_results) {
        int orc = run_oracle_pass(d_data, L, chunk_z, n_chunks,
                                  oracle_results, &oracle_ratio, f_ub);
        if (orc == 0) {
            printf("\n    Best-Static Upper Bound: %.2fx\n", oracle_ratio);
        } else {
            fprintf(stderr, "WARNING: Best-static upper bound pass failed\n");
        }
    }

    printf("\n═══ BASELINE (inference-only, no SGD/exploration) ═══\n\n");

    {
        run_num++;
        printf("  [%d/%d] Baseline: composite dataset ... ", run_num, total_runs);
        fflush(stdout);

        gpucompress_reload_nn(weights_path);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        gpucompress_set_exploration_k(-1);

        RunResult res;
        int rc = do_single_run(d_data, d_readback, n_floats, L, chunk_z, d_count, &res);
        if (rc) any_fail = 1;

        baseline_ratio = res.ratio;
        printf("ratio=%.2fx  file=%.0fMiB  %s\n",
               res.ratio, res.file_mib,
               res.mismatches == 0 ? "PASS" : "FAIL");
        print_action_summary();
        print_per_chunk_ratios(n_chunks);
        write_chunk_detail(f_chunks, "baseline", n_chunks, 0, 0, 0, 0);
    }

    /* ────────────────────────────────────────────────────────────────────
     * SGD STUDY
     * ──────────────────────────────────────────────────────────────────── */

    printf("\n═══ SGD HYPERPARAMETER STUDY ═══\n\n");

    for (int li = 0; li < n_lrs; li++) {
        for (int mi = 0; mi < n_mapes; mi++) {
            float lr = sgd_lrs[li];
            float mape = sgd_mapes[mi];

            run_num++;
            printf("  [%d/%d] SGD lr=%.3f mape=%.2f ... ",
                   run_num, total_runs, lr, mape);
            fflush(stdout);

            gpucompress_reload_nn(weights_path);
            gpucompress_enable_online_learning();
            gpucompress_set_reinforcement(1, lr, mape, mape);
            gpucompress_set_exploration(0);
            gpucompress_set_exploration_k(-1);

            RunResult res;
            int rc = do_single_run(d_data, d_readback, n_floats,
                                   L, chunk_z, d_count, &res);
            if (rc) any_fail = 1;

            double sgd_fire_rate = (res.n_chunks > 0)
                ? (double)res.sgd_fires / res.n_chunks : 0;
            double ratio_vs_oracle = (oracle_ratio > 0)
                ? res.ratio / oracle_ratio : 0;
            fprintf(f_sgd, "%.3f,%.3f,%d,%.4f,%d,"
                           "%.4f,%.4f,%.4f,%.2f,%.2f,"
                           "%.2f,%.2f,"
                           "%.4f,%.4f,%.4f,%.1f,%.2f,%llu\n",
                    lr, mape, res.n_chunks,
                    res.final_mape, res.convergence_chunks,
                    res.ratio, ratio_vs_oracle, sgd_fire_rate,
                    res.total_write_ms, res.total_sgd_ms,
                    res.total_nn_inference_ms, res.total_compression_ms,
                    res.ratio_min, res.ratio_max, res.ratio_stddev,
                    res.write_mibps, res.file_mib, res.mismatches);

            write_chunk_detail(f_chunks, "sgd", n_chunks, lr, mape, 0, 0);

            printf("ratio=%.2fx  mape=%.1f%%  conv=%d  %s\n",
                   res.ratio, res.final_mape, res.convergence_chunks,
                   res.mismatches == 0 ? "PASS" : "FAIL");
            print_action_summary();
            print_per_chunk_ratios(n_chunks);
        }
    }
    fflush(f_sgd);

    /* ────────────────────────────────────────────────────────────────────
     * EXPLORATION STUDY
     * ──────────────────────────────────────────────────────────────────── */

    printf("\n═══ EXPLORATION HYPERPARAMETER STUDY ═══\n\n");

    for (int ti = 0; ti < n_thresholds; ti++) {
        for (int ki = 0; ki < n_ks; ki++) {
            float thresh = expl_thresholds[ti];
            int K = expl_ks[ki];

            run_num++;
            printf("  [%d/%d] Expl thresh=%.2f K=%d ... ",
                   run_num, total_runs, thresh, K);
            fflush(stdout);

            gpucompress_reload_nn(weights_path);
            gpucompress_enable_online_learning();
            gpucompress_set_reinforcement(1, 0.1f, 0.20f, 0.20f);
            gpucompress_set_exploration(1);
            gpucompress_set_exploration_threshold((double)thresh);
            gpucompress_set_exploration_k(K);

            RunResult res;
            int rc = do_single_run(d_data, d_readback, n_floats,
                                   L, chunk_z, d_count, &res);
            if (rc) any_fail = 1;

            double ratio_vs_baseline = (baseline_ratio > 0)
                ? res.ratio / baseline_ratio : 1.0;
            double expl_ratio_vs_oracle = (oracle_ratio > 0)
                ? res.ratio / oracle_ratio : 0;
            double expl_trigger_rate = (res.n_chunks > 0)
                ? (double)res.explorations / res.n_chunks : 0;

            fprintf(f_expl, "%.3f,%d,%d,%.4f,%.4f,%.4f,"
                            "%.4f,%.2f,%.2f,"
                            "%.2f,%.2f,%.2f,"
                            "%.4f,%.4f,%.4f,%.1f,%.2f,%llu\n",
                    thresh, K, res.n_chunks,
                    res.ratio, ratio_vs_baseline, expl_ratio_vs_oracle,
                    expl_trigger_rate,
                    res.total_write_ms, res.total_exploration_ms,
                    res.total_sgd_ms, res.total_nn_inference_ms,
                    res.total_compression_ms,
                    res.ratio_min, res.ratio_max, res.ratio_stddev,
                    res.write_mibps, res.file_mib, res.mismatches);

            write_chunk_detail(f_chunks, "exploration", n_chunks, 0.1f, 0.20f, thresh, K);

            printf("ratio=%.2fx (%.0f%% vs base)  expl=%.0f%%  %s\n",
                   res.ratio, (ratio_vs_baseline - 1.0) * 100.0,
                   expl_trigger_rate * 100.0,
                   res.mismatches == 0 ? "PASS" : "FAIL");
            print_action_summary();
        }
    }

    /* ────────────────────────────────────────────────────────────────────
     * Console summary
     * ──────────────────────────────────────────────────────────────────── */

    printf("\n╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NN Adaptiveness Benchmark Complete                                      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Upper Bound (best-static per chunk): %.2fx                               ║\n",
           oracle_ratio);
    printf("║  Baseline    (NN inference-only):     %.2fx                               ║\n",
           baseline_ratio);
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Output files:                                                           ║\n");
    printf("║    %s\n", OUT_SGD_CSV);
    printf("║    %s\n", OUT_EXPL_CSV);
    printf("║    %s\n", OUT_CHUNKS_CSV);
    printf("║    %s\n", OUT_UB_CSV);
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    if (any_fail) {
        printf("\nWARNING: Some runs had data mismatches!\n");
    }

cleanup:
    if (oracle_results) free(oracle_results);
    if (f_sgd)    fclose(f_sgd);
    if (f_expl)   fclose(f_expl);
    if (f_chunks) fclose(f_chunks);
    if (f_ub)     fclose(f_ub);
    if (gs_sim) gpucompress_grayscott_destroy(gs_sim);
    if (d_data)     cudaFree(d_data);
    if (d_readback) cudaFree(d_readback);
    if (d_count)    cudaFree(d_count);
    remove(TMP_FILE);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    return any_fail;
}
