/**
 * @file synthetic-nn-benchmark.cu
 * @brief Synthetic Data NN Prediction Benchmark
 *
 * Generates 8 diverse synthetic float32 patterns directly on the GPU,
 * then writes each through the HDF5 VOL connector with ALGO_AUTO so the
 * NN picks the best compression config per chunk.  Collects per-chunk
 * diagnostics (predicted ratio, actual ratio, algorithm chosen, MAPE,
 * timing breakdown) and prints a summary table.
 *
 * Usage:
 *   ./build/synthetic_nn_benchmark model.nnwt [--L 128] [--chunk-mb 4]
 *
 * Dataset size reference (per pattern):
 *   --L  64 →   1 MB      --L 256 →  64 MB
 *   --L 128 →   8 MB      --L 320 → 128 MB
 *
 * All 8 patterns × L^3 floats stay under 1 GB with --L 256.
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

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Defaults
 * ============================================================ */

#define DEFAULT_L        128
#define DEFAULT_CHUNK_MB 4

#define TMP_DIR  "/tmp"

#define OUT_DIR  "benchmarks/synthetic/results"
#define OUT_CSV  OUT_DIR "/synthetic_nn_predictions.csv"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * Action → string
 * ============================================================ */

static const char *ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_str(int action, char *buf, size_t bufsz)
{
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
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
 * GPU kernels: generate synthetic data patterns
 * ============================================================ */

/* 1. Constant value */
__global__ void gen_constant(float *out, size_t n, float val)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = val;
}

/* 2. Linear ramp: value = index * scale */
__global__ void gen_linear_ramp(float *out, size_t n, float scale)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = (float)i * scale;
}

/* 3. Smooth 3D sine wave */
__global__ void gen_sine_3d(float *out, int L, float freq)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)L * L * L;
    for (; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        int k = idx % L;
        int j = (idx / L) % L;
        int i = idx / ((size_t)L * L);
        float x = (float)i / (float)L;
        float y = (float)j / (float)L;
        float z = (float)k / (float)L;
        out[idx] = sinf(freq * x) * cosf(freq * y) * sinf(freq * 0.7f * z);
    }
}

/* 4. Uniform random via curand */
__global__ void gen_random_uniform(float *out, size_t n, unsigned long long seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = curand_uniform(&state) * 2.0f - 1.0f;
}

/* 5. Gaussian random */
__global__ void gen_random_gaussian(float *out, size_t n, unsigned long long seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = curand_normal(&state);
}

/* 6. Step function: 3D blocks of constant value */
__global__ void gen_step_blocks(float *out, int L, int block_size,
                                unsigned long long seed)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)L * L * L;
    for (; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        int k = idx % L;
        int j = (idx / L) % L;
        int i = idx / ((size_t)L * L);
        /* Hash block coords to get a deterministic "random" value */
        int bi = i / block_size;
        int bj = j / block_size;
        int bk = k / block_size;
        unsigned int h = (unsigned int)(bi * 7919 + bj * 6271 + bk * 4507 + seed);
        h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
        out[idx] = (float)(int)(h % 201) - 100.0f;  /* [-100, 100] */
    }
}

/* 7. Sparse spiky: mostly zero with rare peaks */
__global__ void gen_sparse(float *out, size_t n, float density,
                           unsigned long long seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float r = curand_uniform(&state);
        out[i] = (r < density) ? curand_normal(&state) * 50.0f : 0.0f;
    }
}

/* 8. Multi-frequency turbulence */
__global__ void gen_turbulence(float *out, int L)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)L * L * L;
    for (; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        int k = idx % L;
        int j = (idx / L) % L;
        int i = idx / ((size_t)L * L);
        float x = (float)i / (float)L;
        float y = (float)j / (float)L;
        float z = (float)k / (float)L;
        float v = 0.0f;
        v += 1.0f  * sinf(2.0f * 3.14159f * 1.0f * x);
        v += 0.7f  * sinf(2.0f * 3.14159f * 3.0f * y + 0.5f);
        v += 0.4f  * sinf(2.0f * 3.14159f * 7.0f * z + 1.2f);
        v += 0.2f  * sinf(2.0f * 3.14159f * 15.0f * x + 2.1f);
        v += 0.1f  * sinf(2.0f * 3.14159f * 31.0f * y + 3.7f);
        v += 0.05f * sinf(2.0f * 3.14159f * 63.0f * z + 0.3f);
        out[idx] = v;
    }
}

/* ============================================================
 * GPU compare (bitwise verification)
 * ============================================================ */

__global__ void count_mismatches_kernel(const float *a, const float *b,
                                        size_t n, unsigned long long *count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local++;
    atomicAdd(count, local);
}

static unsigned long long gpu_compare(const float *a, const float *b,
                                      size_t n, unsigned long long *d_count)
{
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    count_mismatches_kernel<<<512, 256>>>(a, b, n, d_count);
    cudaDeviceSynchronize();
    unsigned long long h = 0;
    cudaMemcpy(&h, d_count, sizeof(h), cudaMemcpyDeviceToHost);
    return h;
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

static hid_t make_dcpl_auto(int L, int chunk_z, double eb)
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

static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

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
 * Pattern descriptor
 * ============================================================ */

typedef struct {
    const char *name;
    const char *description;
} PatternInfo;

static PatternInfo PATTERNS[] = {
    { "constant",         "All values = 3.14159 (trivially compressible)" },
    { "linear_ramp",      "Linearly increasing values (smooth, low 2nd deriv)" },
    { "smooth_sine",      "3D sinusoidal field (periodic, moderate entropy)" },
    { "uniform_random",   "Uniform random [-1,1] (high entropy, hard to compress)" },
    { "gaussian_noise",   "Gaussian N(0,1) noise (moderate-high entropy)" },
    { "step_blocks",      "3D blocks of constant value (sharp boundaries)" },
    { "sparse_spiky",     "99% zeros, 1% Gaussian peaks (very sparse)" },
    { "turbulence",       "Sum of sines at 6 frequencies (multi-scale)" },
};
#define NUM_PATTERNS 8

/* Generate pattern i into d_buf (L^3 floats, already allocated) */
static void generate_pattern(int pattern_idx, float *d_buf, int L)
{
    size_t n = (size_t)L * L * L;
    int blocks = (int)((n + 255) / 256);
    if (blocks > 2048) blocks = 2048;

    switch (pattern_idx) {
    case 0: gen_constant<<<blocks, 256>>>(d_buf, n, 3.14159f); break;
    case 1: gen_linear_ramp<<<blocks, 256>>>(d_buf, n, 1.0f / (float)n); break;
    case 2: gen_sine_3d<<<blocks, 256>>>(d_buf, L, 6.2832f * 4.0f); break;
    case 3: gen_random_uniform<<<blocks, 256>>>(d_buf, n, 12345ULL); break;
    case 4: gen_random_gaussian<<<blocks, 256>>>(d_buf, n, 67890ULL); break;
    case 5: gen_step_blocks<<<blocks, 256>>>(d_buf, L, 16, 42ULL); break;
    case 6: gen_sparse<<<blocks, 256>>>(d_buf, n, 0.01f, 99999ULL); break;
    case 7: gen_turbulence<<<blocks, 256>>>(d_buf, L); break;
    }
    cudaDeviceSynchronize();
}

/* ============================================================
 * Per-pattern result
 * ============================================================ */

typedef struct {
    const char *name;
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    double read_mbps;
    unsigned long long mismatches;
    int    n_chunks;
    double mape_pct;
    int    sgd_fires;
    int    explorations;

    /* Data stats (from first chunk) */
    double entropy;
    double mad;
    double second_deriv;
} PatternResult;

/* ============================================================
 * Run one pattern: generate → VOL write → VOL read → verify
 * ============================================================ */

static int run_pattern(int pat_idx, float *d_buf, float *d_read,
                       unsigned long long *d_count,
                       int L, int chunk_z, double error_bound,
                       PatternResult *r, FILE *f_csv)
{
    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks       = (L + chunk_z - 1) / chunk_z;

    r->name       = PATTERNS[pat_idx].name;
    r->orig_bytes = total_bytes;
    r->n_chunks   = n_chunks;

    /* Generate data on GPU */
    printf("\n── Pattern %d/%d: %s ──\n", pat_idx + 1, NUM_PATTERNS,
           PATTERNS[pat_idx].name);
    printf("  %s\n", PATTERNS[pat_idx].description);

    double t_gen = now_ms();
    generate_pattern(pat_idx, d_buf, L);
    double gen_ms = now_ms() - t_gen;
    printf("  Generated in %.1f ms on GPU\n", gen_ms);

    /* Compute data stats via gpucompress API */
    {
        double ent = 0.0, m = 0.0, sd = 0.0;
        gpucompress_error_t serr = gpucompress_compute_stats(d_buf, total_bytes,
                                                              &ent, &m, &sd);
        if (serr == GPUCOMPRESS_SUCCESS) {
            r->entropy      = ent;
            r->mad          = m;
            r->second_deriv = sd;
            printf("  Stats: entropy=%.4f bits  MAD=%.6f  2nd_deriv=%.6f\n",
                   ent, m, sd);
        } else {
            r->entropy = r->mad = r->second_deriv = -1.0;
            printf("  Stats: compute failed (err=%d)\n", serr);
        }
    }

    /* Reset NN state and chunk history */
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    char tmp_file[256];
    snprintf(tmp_file, sizeof(tmp_file), TMP_DIR "/bm_synth_%s.h5",
             PATTERNS[pat_idx].name);

    /* VOL write */
    printf("  H5Dwrite (GPU ptr → VOL → HDF5)... "); fflush(stdout);
    remove(tmp_file);
    {
        hid_t dcpl = make_dcpl_auto(L, chunk_z, error_bound);
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file < 0) {
            fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file);
            H5Pclose(dcpl);
            return 1;
        }

        hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
        hid_t fsp  = H5Screate_simple(3, dims, NULL);
        hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                                 fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);

        double t0   = now_ms();
        herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_buf);
        H5Dclose(dset);
        H5Fclose(file);
        H5Pclose(dcpl);
        r->write_ms = now_ms() - t0;
        printf("%.0f ms\n", r->write_ms);

        if (wret < 0) {
            fprintf(stderr, "  H5Dwrite failed\n");
            return 1;
        }
    }

    drop_pagecache(tmp_file);

    /* VOL read */
    printf("  H5Dread  (HDF5 → VOL → GPU ptr)... "); fflush(stdout);
    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);

        double t0   = now_ms();
        herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();
        H5Dclose(dset);
        H5Fclose(file);
        r->read_ms = now_ms() - t0;
        printf("%.0f ms\n", r->read_ms);

        if (rret < 0) {
            fprintf(stderr, "  H5Dread failed\n");
            return 1;
        }
    }

    /* Bitwise verification */
    unsigned long long mm = gpu_compare(d_buf, d_read, n_floats, d_count);
    r->mismatches = mm;

    size_t fbytes = file_size(tmp_file);
    r->file_bytes = fbytes;
    r->ratio      = (fbytes > 0) ? (double)total_bytes / (double)fbytes : 1.0;
    r->write_mbps = (double)total_bytes / (1 << 20) / (r->write_ms / 1000.0);
    r->read_mbps  = (double)total_bytes / (1 << 20) / (r->read_ms / 1000.0);

    /* Collect per-chunk NN diagnostics */
    int n_hist = gpucompress_get_chunk_history_count();
    int sgd_fires = 0, explorations = 0;
    double ape_ratio_sum = 0.0, ape_comp_sum = 0.0, ape_decomp_sum = 0.0;
    int ape_ratio_cnt = 0, ape_comp_cnt = 0, ape_decomp_cnt = 0;

    for (int c = 0; c < n_hist; c++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(c, &d) != 0) continue;

        char final_str[40], orig_str[40];
        action_to_str(d.nn_action, final_str, sizeof(final_str));
        action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));

        int lossy = (d.nn_action >= 0) ? ((d.nn_action / 8) % 2) : 0;

        /* Per-metric MAPE */
        double mape_ratio = (d.actual_ratio > 0)
            ? fabs(d.predicted_ratio - d.actual_ratio) / d.actual_ratio * 100.0 : 0.0;
        double mape_comp = (d.compression_ms > 0)
            ? fabs(d.predicted_comp_time - d.compression_ms) / d.compression_ms * 100.0 : 0.0;
        double mape_decomp = (d.decompression_ms > 0)
            ? fabs(d.predicted_decomp_time - d.decompression_ms) / d.decompression_ms * 100.0 : 0.0;

        sgd_fires    += d.sgd_fired;
        explorations += d.exploration_triggered;
        if (d.actual_ratio > 0 && d.predicted_ratio > 0) {
            ape_ratio_sum += mape_ratio; ape_ratio_cnt++;
        }
        if (d.compression_ms > 0) {
            ape_comp_sum += mape_comp; ape_comp_cnt++;
        }
        if (d.decompression_ms > 0) {
            ape_decomp_sum += mape_decomp; ape_decomp_cnt++;
        }

        /* Print chunk header */
        printf("\n  Chunk %3d/%d  [%s]%s%s%s\n",
               c + 1, n_chunks, final_str,
               lossy ? " (lossy)" : "",
               d.exploration_triggered ? " [EXPLORE]" : "",
               d.sgd_fired ? " [SGD]" : "");
        printf("  %-14s  %12s  %12s  %9s\n",
               "Metric", "Predicted", "Actual", "MAPE");
        printf("  ─────────────────────────────────────────────────\n");

        printf("  %-14s  %10.3fx  %10.3fx  %6.1f%%\n",
               "ratio", (double)d.predicted_ratio, (double)d.actual_ratio, mape_ratio);
        printf("  %-14s  %9.3f ms  %9.3f ms  %6.1f%%\n",
               "comp_time", (double)d.predicted_comp_time, (double)d.compression_ms, mape_comp);
        if (d.decompression_ms > 0) {
            printf("  %-14s  %9.3f ms  %9.3f ms  %6.1f%%\n",
                   "decomp_time", (double)d.predicted_decomp_time,
                   (double)d.decompression_ms, mape_decomp);
        } else {
            printf("  %-14s  %9.3f ms  %12s\n",
                   "decomp_time", (double)d.predicted_decomp_time, "n/a");
        }
        if (lossy) {
            printf("  %-14s  %8.1f dB  %12s\n",
                   "psnr", (double)d.predicted_psnr, "—");
        }

        /* Write per-chunk CSV row */
        if (f_csv) {
            fprintf(f_csv, "%s,%d,%s,%s,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.1f,"
                           "%.1f,%.1f,%.1f,"
                           "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,"
                           "%.4f,%.6f,%.6f\n",
                    PATTERNS[pat_idx].name, c, final_str, orig_str,
                    (double)d.actual_ratio, (double)d.predicted_ratio,
                    (double)d.compression_ms, (double)d.predicted_comp_time,
                    (double)d.decompression_ms, (double)d.predicted_decomp_time,
                    (double)d.predicted_psnr,
                    mape_ratio, mape_comp, mape_decomp,
                    d.sgd_fired, d.exploration_triggered,
                    (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                    (double)d.compression_ms, (double)d.exploration_ms,
                    (double)d.sgd_update_ms,
                    r->entropy, r->mad, r->second_deriv);
        }
    }

    r->sgd_fires   = sgd_fires;
    r->explorations = explorations;
    r->mape_pct    = (ape_ratio_cnt > 0) ? ape_ratio_sum / ape_ratio_cnt : 0.0;
    double mape_comp_avg  = (ape_comp_cnt > 0) ? ape_comp_sum / ape_comp_cnt : 0.0;
    double mape_decomp_avg = (ape_decomp_cnt > 0) ? ape_decomp_sum / ape_decomp_cnt : 0.0;

    printf("\n  Summary: ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.1f MiB  mismatches=%llu\n",
           r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), mm);
    printf("  MAPE:  ratio=%.1f%%  comp_time=%.1f%%  decomp_time=%.1f%%  "
           "SGD=%d  Explore=%d\n",
           r->mape_pct, mape_comp_avg, mape_decomp_avg,
           sgd_fires, explorations);

    /* Clean up temp file */
    remove(tmp_file);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * Summary table
 * ============================================================ */

static void print_summary(PatternResult *res, int n, int L, int chunk_z)
{
    double dataset_mb = (double)L * L * L * sizeof(float) / (1 << 20);
    double chunk_mb   = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);
    int n_chunks      = (L + chunk_z - 1) / chunk_z;

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Synthetic NN Prediction Benchmark Summary                                                                   ║\n");
    printf("║  Grid: %d^3 (%.0f MB per pattern)  Chunks: %d x %.1f MB                                                    \n",
           L, dataset_mb, n_chunks, chunk_mb);
    printf("╠══════════════════╦═════════╦═══════╦═════════╦═══════════╦══════════╦═══════╦══════════╦════════╦═════════════╣\n");
    printf("║  Pattern         ║ Entropy ║  MAD  ║ 2ndDeriv║  Ratio    ║ Write    ║ Read  ║ MAPE     ║ SGD/   ║ Verify      ║\n");
    printf("║                  ║  (bits) ║       ║         ║           ║ (MiB/s)  ║(MiB/s)║          ║ Expl   ║             ║\n");
    printf("╠══════════════════╬═════════╬═══════╬═════════╬═══════════╬══════════╬═══════╬══════════╬════════╬═════════════╣\n");

    for (int i = 0; i < n; i++) {
        PatternResult *r = &res[i];
        const char *verdict = (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-16s║  %5.2f  ║ %5.3f ║  %5.3f  ║ %7.2fx  ║ %7.0f  ║ %5.0f ║  %5.1f%%  ║ %2d/%2d  ║ %-11s ║\n",
               r->name, r->entropy, r->mad, r->second_deriv,
               r->ratio, r->write_mbps, r->read_mbps,
               r->mape_pct, r->sgd_fires, r->explorations,
               verdict);
    }
    printf("╚══════════════════╩═════════╩═══════╩═════════╩═══════════╩══════════╩═══════╩══════════╩════════╩═════════════╝\n");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    int L        = DEFAULT_L;
    int chunk_z  = 0;
    int chunk_mb = DEFAULT_CHUNK_MB;
    double error_bound = 0.0;
    float rl_lr          = 0.25f;
    float mape_threshold = 0.20f;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc)
            L = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]); chunk_z = 0;
        } else if (strcmp(argv[i], "--chunk-z") == 0 && i + 1 < argc) {
            chunk_z = atoi(argv[++i]); chunk_mb = 0;
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc)
            error_bound = atof(argv[++i]);
        else if (strcmp(argv[i], "--rl-lr") == 0 && i + 1 < argc)
            rl_lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--mape-threshold") == 0 && i + 1 < argc)
            mape_threshold = (float)atof(argv[++i]);
        else if (argv[i][0] != '-')
            weights_path = argv[i];
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--chunk-mb N] "
                "[--error-bound val]\n", argv[0]);
        return 1;
    }

    /* Compute chunk_z */
    if (chunk_z <= 0 && chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    }
    if (chunk_z < 1) chunk_z = L / 4;
    if (chunk_z < 1) chunk_z = 1;

    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks       = (L + chunk_z - 1) / chunk_z;
    double dataset_mb  = (double)total_bytes / (1 << 20);
    double cmb         = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);

    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Synthetic Data NN Prediction Benchmark                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB per pattern)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Patterns : %d\n", NUM_PATTERNS);
    printf("  Total    : %.1f MB across all patterns\n", dataset_mb * NUM_PATTERNS);
    printf("  Weights  : %s\n", weights_path);
    if (error_bound > 0.0)
        printf("  ErrBound : %.6f (lossy quantization)\n", error_bound);
    else
        printf("  Mode     : lossless\n");
    printf("\n");

    /* Ensure total stays under 1 GB */
    if (dataset_mb * NUM_PATTERNS > 1024.0) {
        fprintf(stderr, "WARNING: Total data %.0f MB exceeds 1 GB. "
                "Reduce --L (try --L %d).\n",
                dataset_mb * NUM_PATTERNS,
                (int)cbrt(1024.0 * 1024.0 * 1024.0 / (NUM_PATTERNS * sizeof(float))));
    }

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

    /* Allocate GPU buffers: d_buf (data) + d_read (verification) + d_count */
    float *d_buf = NULL, *d_read = NULL;
    unsigned long long *d_count = NULL;
    if (cudaMalloc(&d_buf,   total_bytes) != cudaSuccess ||
        cudaMalloc(&d_read,  total_bytes) != cudaSuccess ||
        cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed (need %.0f MB)\n",
                2.0 * dataset_mb);
        gpucompress_cleanup(); return 1;
    }

    /* Online learning: configurable via CLI */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_reinforcement(1, rl_lr, mape_threshold, 0.0f);
    gpucompress_set_exploration_threshold((double)mape_threshold);
    printf("  RL LR    : %.4f\n", rl_lr);
    printf("  MAPE thr : %.0f%%\n", mape_threshold * 100.0f);

    /* Open CSV */
    mkdir(OUT_DIR, 0755);
    FILE *f_csv = fopen(OUT_CSV, "w");
    if (f_csv) {
        fprintf(f_csv, "pattern,chunk,action_final,action_orig,"
                       "actual_ratio,predicted_ratio,"
                       "actual_comp_time_ms,predicted_comp_time_ms,"
                       "actual_decomp_time_ms,predicted_decomp_time_ms,"
                       "predicted_psnr_db,"
                       "mape_ratio,mape_comp,mape_decomp,"
                       "sgd_fired,exploration_triggered,"
                       "nn_inference_ms,preprocessing_ms,compression_ms,"
                       "exploration_ms,sgd_update_ms,"
                       "entropy,mad,second_derivative\n");
    }

    /* Run all patterns */
    PatternResult results[NUM_PATTERNS];
    int any_fail = 0;

    for (int p = 0; p < NUM_PATTERNS; p++) {
        int rc = run_pattern(p, d_buf, d_read, d_count,
                             L, chunk_z, error_bound,
                             &results[p], f_csv);
        if (rc) any_fail = 1;
    }

    if (f_csv) fclose(f_csv);

    /* Summary */
    print_summary(results, NUM_PATTERNS, L, chunk_z);
    printf("\nPer-chunk CSV: %s\n", OUT_CSV);

    /* Cleanup */
    cudaFree(d_buf);
    cudaFree(d_read);
    cudaFree(d_count);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}
