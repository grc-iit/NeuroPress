/**
 * @file grayscott-benchmark.cu
 * @brief Gray-Scott VOL Benchmark: No-Comp vs Oracle vs NN+SGD
 *
 * Runs a Gray-Scott reaction-diffusion simulation on the GPU, then benchmarks
 * writing the 3D field to HDF5 via the GPUCompress VOL connector under five
 * compression phases.  Data stays GPU-resident throughout.
 *
 * Phases:
 *   1. no-comp      : GPU sim → H5Dwrite via VOL (no filter, VOL-2 D→H fallback)
 *   2. oracle       : Exhaustive search (8 algos × 2 shuffle per chunk)
 *   3. nn           : GPU sim → H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *   4. nn-rl        : ALGO_AUTO + online SGD (MAPE≥20%, LR=0.4, no exploration)
 *   5. nn-rl+exp50  : ALGO_AUTO + online SGD + Level-2 exploration (MAPE≥50%)
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
 *                    select multiple phases. If omitted, all 5 phases run.
 *                    Valid names: no-comp, exhaustive, nn, nn-rl, nn-rl+exp50
 *
 * Examples:
 *   # Run all phases with 1 GB dataset and 64 MB chunks:
 *   ./build/grayscott_benchmark weights.nnwt --L 640 --chunk-mb 64
 *
 *   # Run only the nn-rl+exp50 phase:
 *   ./build/grayscott_benchmark weights.nnwt --L 256 --phase nn-rl+exp50
 *
 *   # Run nn and nn-rl phases together:
 *   ./build/grayscott_benchmark weights.nnwt --phase nn --phase nn-rl
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
#define DEFAULT_F           0.04f
#define DEFAULT_K           0.06075f

#define REINFORCE_LR        0.05f
#define REINFORCE_MAPE      0.20f

#define TMP_NOCOMP   "/tmp/bm_gs_nocomp.h5"
#define TMP_ORACLE   "/tmp/bm_gs_oracle.h5"
#define TMP_NN       "/tmp/bm_gs_nn.h5"
#define TMP_NN_RL    "/tmp/bm_gs_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_gs_nn_rlexp.h5"

#define OUT_DIR     "benchmarks/grayscott/results"
#define OUT_CSV     OUT_DIR "/benchmark_grayscott_vol.csv"
#define OUT_CHUNKS  OUT_DIR "/benchmark_grayscott_vol_chunks.csv"
#define OUT_UB_CSV  OUT_DIR "/upper_bound.csv"

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
 * Gather chunk: extract a Z-slab from 3D L×L×L field
 * ============================================================ */

__global__ void gather_chunk_kernel(float *dst, const float *src,
                                    int L, int actual_cz, int k0)
{
    size_t chunk_floats = (size_t)L * L * actual_cz;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_floats; idx += (size_t)gridDim.x * blockDim.x) {
        size_t lk  = idx % actual_cz;
        size_t rem = idx / actual_cz;
        size_t j   = rem % L;
        size_t i   = rem / L;
        size_t src_idx = i * (size_t)L * L + j * (size_t)L + (k0 + lk);
        dst[idx] = src[src_idx];
    }
}

static void gather_chunk(float *dst, const float *src, int L, int chunk_z, int actual_cz, int chunk_idx)
{
    size_t chunk_floats = (size_t)L * L * actual_cz;
    int blocks = (int)((chunk_floats + 255) / 256);
    if (blocks > 2048) blocks = 2048;
    int k0 = chunk_idx * chunk_z;
    gather_chunk_kernel<<<blocks, 256>>>(dst, src, L, actual_cz, k0);
}

/* Inverse of gather: scatter chunk data back to 3D field */
__global__ void scatter_chunk_kernel(float *dst, const float *src,
                                     int L, int actual_cz, int k0)
{
    size_t chunk_floats = (size_t)L * L * actual_cz;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_floats; idx += (size_t)gridDim.x * blockDim.x) {
        size_t lk  = idx % actual_cz;
        size_t rem = idx / actual_cz;
        size_t j   = rem % L;
        size_t i   = rem / L;
        size_t dst_idx = i * (size_t)L * L + j * (size_t)L + (k0 + lk);
        dst[dst_idx] = src[idx];
    }
}

static void scatter_chunk(float *dst, const float *src, int L, int chunk_z, int actual_cz, int chunk_idx)
{
    size_t chunk_floats = (size_t)L * L * actual_cz;
    int blocks = (int)((chunk_floats + 255) / 256);
    if (blocks > 2048) blocks = 2048;
    int k0 = chunk_idx * chunk_z;
    scatter_chunk_kernel<<<blocks, 256>>>(dst, src, L, actual_cz, k0);
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

/* 3D chunked DCPL for a specific algorithm + shuffle */
static hid_t make_dcpl_fixed(int L, int chunk_z, int algo_enum, int shuffle)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo_enum;
    cd[1] = shuffle ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
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
    double mape_pct;
    /* Per-component GPU-time (cumulative across chunks) */
    double nn_ms;
    double stats_ms;
    double preproc_ms;
    double comp_ms;
    double explore_ms;
    double sgd_ms;
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
 * Oracle: exhaustive best-config per chunk (8 algos × 2 shuffle)
 * ============================================================ */

static int run_oracle_pass(const float *d_data, int L, int chunk_z,
                           int n_chunks, FILE *f_ub,
                           const char *tmp_file,
                           float *d_decomp_field,
                           unsigned long long *d_count,
                           double error_bound,
                           PhaseResult *r)
{
    size_t max_chunk_floats = (size_t)L * L * chunk_z;
    size_t max_chunk_bytes  = max_chunk_floats * sizeof(float);
    size_t out_buf_size = max_chunk_bytes + 65536;
    size_t total_bytes = (size_t)L * L * L * sizeof(float);
    size_t n_floats    = (size_t)L * L * L;

    float *d_chunk_buf = NULL;
    void  *d_comp_buf  = NULL;
    void  *h_comp_buf  = NULL;
    if (cudaMalloc(&d_chunk_buf, max_chunk_bytes) != cudaSuccess) return 1;
    if (cudaMalloc(&d_comp_buf, out_buf_size) != cudaSuccess) {
        cudaFree(d_chunk_buf); return 1;
    }
    h_comp_buf = malloc(out_buf_size);
    if (!h_comp_buf) {
        cudaFree(d_chunk_buf); cudaFree(d_comp_buf); return 1;
    }

    static const char *ALGO_NAMES[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    /* ── Stage 1: Exhaustive search — find best config per chunk ─── */
    /* Try 8 algos × 2 shuffle × (1 or 2 quant) configs per chunk.
     * Quantization is only tried when error_bound > 0. */
    int n_quant = (error_bound > 0.0) ? 2 : 1;
    int *best_algos    = (int *)calloc(n_chunks, sizeof(int));
    int *best_shuffles = (int *)calloc(n_chunks, sizeof(int));
    int *best_quants   = (int *)calloc(n_chunks, sizeof(int));
    size_t total_best_compressed = 0;

    for (int c = 0; c < n_chunks; c++) {
        int actual_cz = chunk_z;
        if (c * chunk_z + actual_cz > L) actual_cz = L - c * chunk_z;
        size_t chunk_bytes = (size_t)L * L * actual_cz * sizeof(float);

        gather_chunk(d_chunk_buf, d_data, L, chunk_z, actual_cz, c);
        cudaDeviceSynchronize();

        double best_ratio = 0;
        size_t best_compressed = chunk_bytes;
        double best_comp_ms = 0;
        double best_throughput = 0;
        int best_algo_enum = 1;
        int best_shuffle = 0;
        int best_quant = 0;
        const char *best_algo = "none";

        for (int algo_enum = 1; algo_enum <= 8; algo_enum++) {
            for (int shuffle = 0; shuffle <= 1; shuffle++) {
                for (int quant = 0; quant < n_quant; quant++) {
                    gpucompress_config_t cfg = gpucompress_default_config();
                    cfg.algorithm = (gpucompress_algorithm_t)algo_enum;
                    cfg.preprocessing = 0;
                    if (shuffle) cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                    if (quant) cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
                    cfg.error_bound = quant ? error_bound : 0.0;

                    size_t out_size = out_buf_size;
                    gpucompress_stats_t stats;
                    gpucompress_error_t err = gpucompress_compress_gpu(
                        d_chunk_buf, chunk_bytes,
                        d_comp_buf, &out_size, &cfg, &stats, NULL);

                    double ratio = (err == GPUCOMPRESS_SUCCESS)
                        ? stats.compression_ratio : 0;

                    if (f_ub) {
                        fprintf(f_ub, "%d,grayscott,%s,%d,%d,%.6f,%.4f,%zu,%zu,%.3f,%.1f,%s\n",
                                c, ALGO_NAMES[algo_enum],
                                shuffle ? 4 : 0, quant, quant ? error_bound : 0.0,
                                ratio,
                                (err == GPUCOMPRESS_SUCCESS) ? stats.compressed_size : 0,
                                chunk_bytes,
                                (err == GPUCOMPRESS_SUCCESS) ? stats.actual_comp_time_ms : 0,
                                (err == GPUCOMPRESS_SUCCESS) ? stats.throughput_mbps : 0,
                                (err == GPUCOMPRESS_SUCCESS) ? "ok" : "fail");
                    }

                    if (ratio > best_ratio) {
                        best_ratio = ratio;
                        best_compressed = stats.compressed_size;
                        best_comp_ms = stats.actual_comp_time_ms;
                        best_throughput = stats.throughput_mbps;
                        best_algo_enum = algo_enum;
                        best_shuffle = shuffle;
                        best_quant = quant;
                        best_algo = ALGO_NAMES[algo_enum];
                    }
                }
            }
        }

        best_algos[c]    = best_algo_enum;
        best_shuffles[c] = best_shuffle;
        best_quants[c]   = best_quant;
        total_best_compressed += best_compressed;

        printf("  chunk %3d/%d: best=%s%s%s  ratio=%.2fx  comp=%.1f MiB/s (%.2f ms)\n",
               c + 1, n_chunks, best_algo,
               best_shuffle ? "+shuf" : "",
               best_quant   ? "+quant" : "",
               best_ratio, best_throughput, best_comp_ms);

        /* Write oracle row to per-chunk CSV */
        {
            struct stat csv_st;
            bool need_hdr = (stat(OUT_CHUNKS, &csv_st) != 0 || csv_st.st_size == 0);
            FILE *cf = fopen(OUT_CHUNKS, "a");
            if (cf) {
                if (need_hdr)
                    fprintf(cf, "phase,chunk,action_final,action_orig,actual_ratio,"
                                "predicted_ratio,sgd_fired,exploration_triggered,"
                                "nn_inference_ms,preprocessing_ms,compression_ms,"
                                "exploration_ms,sgd_update_ms\n");
                char algo_str[40];
                snprintf(algo_str, sizeof(algo_str), "%s%s%s",
                         best_algo, best_shuffle ? "+shuf" : "",
                         best_quant ? "+quant" : "");
                fprintf(cf, "exhaustive,%d,%s,%s,%.4f,0.0000,0,0,"
                            "0.000,0.000,%.3f,0.000,0.000\n",
                        c, algo_str, algo_str, best_ratio, best_comp_ms);
                fclose(cf);
            }
        }
    }

    double oracle_ratio = (total_best_compressed > 0)
        ? (double)total_bytes / (double)total_best_compressed : 1.0;

    /* Push per-chunk best configs into force-algorithm queue */
    gpucompress_force_algorithm_reset();
    for (int c = 0; c < n_chunks; c++)
        gpucompress_force_algorithm_push(best_algos[c], best_shuffles[c],
                                         best_quants[c], error_bound);
    printf("[exhaustive] VOL E2E using per-chunk best via force-algorithm queue\n");

    /* ── Stage 2: E2E write via HDF5 VOL (ALGO_AUTO + force queue) ── */
    printf("[exhaustive] E2E write (GPU ptr → VOL → HDF5)... "); fflush(stdout);
    remove(tmp_file);
    {
        hid_t dcpl_ex = make_dcpl_auto(L, chunk_z, error_bound);
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file < 0) { free(best_algos); free(best_shuffles); free(best_quants);
                         free(h_comp_buf); cudaFree(d_chunk_buf); cudaFree(d_comp_buf);
                         H5Pclose(dcpl_ex); return 1; }

        hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
        hid_t fsp  = H5Screate_simple(3, dims, NULL);
        hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                                 fsp, H5P_DEFAULT, dcpl_ex, H5P_DEFAULT);
        H5Sclose(fsp);

        double t_write_start = now_ms();
        herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
        H5Fclose(file);
        H5Pclose(dcpl_ex);
        double write_ms = now_ms() - t_write_start;
        printf("%.0f ms\n", write_ms);

        if (wret < 0) { fprintf(stderr, "[exhaustive] H5Dwrite failed\n");
                         free(best_algos); free(best_shuffles); free(best_quants);
                         free(h_comp_buf); cudaFree(d_chunk_buf); cudaFree(d_comp_buf); return 1; }

        r->write_ms   = write_ms;
        r->write_mbps = (double)total_bytes / (1 << 20) / (write_ms / 1000.0);
    }

    drop_pagecache(tmp_file);

    /* ── Stage 3: E2E read via HDF5 VOL ── */
    printf("[exhaustive] E2E read  (HDF5 → VOL → GPU ptr)... "); fflush(stdout);
    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        hid_t dset = H5Dopen2(file, "V", H5P_DEFAULT);

        double t_read_start = now_ms();
        herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_decomp_field);
        cudaDeviceSynchronize();
        H5Dclose(dset);
        H5Fclose(file);
        double read_ms = now_ms() - t_read_start;
        printf("%.0f ms\n", read_ms);

        if (rret < 0) { fprintf(stderr, "[exhaustive] H5Dread failed\n");
                         free(best_algos); free(best_shuffles); free(best_quants);
                         free(h_comp_buf); cudaFree(d_chunk_buf); cudaFree(d_comp_buf); return 1; }

        r->read_ms   = read_ms;
        r->read_mbps = (double)total_bytes / (1 << 20) / (read_ms / 1000.0);
    }

    /* Bitwise verification */
    unsigned long long mm = gpu_compare(d_data, d_decomp_field, n_floats, d_count);

    /* Fill result — write_ms/read_ms/write_mbps/read_mbps already set above */
    size_t fbytes = file_size(tmp_file);
    snprintf(r->phase, sizeof(r->phase), "exhaustive");
    r->sim_ms      = 0;  /* set by caller */
    r->file_bytes  = fbytes;
    r->orig_bytes  = total_bytes;
    r->ratio       = (fbytes > 0) ? (double)total_bytes / (double)fbytes : 1.0;
    r->mismatches  = mm;
    r->n_chunks    = n_chunks;
    r->sgd_fires   = 0;
    r->explorations = 0;
    r->mape_pct    = 0.0;

    printf("[exhaustive] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  mismatches=%llu\n",
           r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), mm);

    free(best_algos);
    free(best_shuffles);
    free(best_quants);
    free(h_comp_buf);
    cudaFree(d_chunk_buf);
    cudaFree(d_comp_buf);
    return (mm == 0) ? 0 : 1;
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
    r->mape_pct    = 0.0;
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
    double ape_sum = 0.0;
    int    ape_cnt = 0;
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
            if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
                ape_sum += fabs(d.predicted_ratio - d.actual_ratio)
                           / d.actual_ratio;
                ape_cnt++;
            }
            char final_str[40], orig_str[40];
            action_to_str(d.nn_action, final_str, sizeof(final_str));
            action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));
            double chunk_mape = (d.actual_ratio > 0.0f)
                ? fabs((double)d.predicted_ratio - (double)d.actual_ratio)
                  / (double)d.actual_ratio * 100.0
                : 0.0;
            printf("  chunk %3d/%d: algo=%-18s ratio=%.2fx  pred=%.2fx  MAPE=%.1f%%",
                   i + 1, n_chunks, final_str, (double)d.actual_ratio,
                   (double)d.predicted_ratio, chunk_mape);
            if (d.exploration_triggered)
                printf("  (orig=%s, explored)", orig_str);
            if (d.sgd_fired)
                printf("  [SGD]");
            printf("\n");
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
    r->mape_pct    = (ape_cnt > 0) ? 100.0 * ape_sum / ape_cnt : 0.0;
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

    /* Per-component overhead breakdown.
     * Per-chunk timings are cumulative (sum across all chunks).  With 8
     * concurrent VOL workers, the wall-clock time is less than the sum.
     * Report both the cumulative GPU-time and the share of total GPU-time,
     * so users can see which component dominates the pipeline. */
    double total_tracked = total_nn_ms + total_stats_ms + total_preproc_ms
                         + total_comp_ms + total_explore_ms + total_sgd_ms;
    printf("[%s] Overhead breakdown (%d chunks, write=%.1f ms, total GPU-time=%.1f ms):\n",
           phase_name, n_hist, write_ms, total_tracked);
    printf("  Stats compute: %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_stats_ms, total_tracked > 0 ? 100.0 * total_stats_ms / total_tracked : 0.0);
    printf("  NN inference : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_nn_ms, total_tracked > 0 ? 100.0 * total_nn_ms / total_tracked : 0.0);
    printf("  Preprocessing: %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_preproc_ms, total_tracked > 0 ? 100.0 * total_preproc_ms / total_tracked : 0.0);
    printf("  Compression  : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_comp_ms, total_tracked > 0 ? 100.0 * total_comp_ms / total_tracked : 0.0);
    printf("  Exploration  : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_explore_ms, total_tracked > 0 ? 100.0 * total_explore_ms / total_tracked : 0.0);
    printf("  SGD update   : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_sgd_ms, total_tracked > 0 ? 100.0 * total_sgd_ms / total_tracked : 0.0);

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
               "mean_prediction_error_pct\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        fprintf(f, "%s,%d,%d,%.4f,%.5f,%d,%d,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,"
                   "%.1f,%.1f,%llu,%d,%d,%.2f\n",
                r->phase, L, steps, F, k, chunk_z, r->n_chunks,
                r->sim_ms, r->write_ms, r->read_ms,
                (double)r->file_bytes / (1 << 20),
                (double)r->orig_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps,
                r->mismatches, r->sgd_fires, r->explorations,
                r->mape_pct);
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
        fprintf(f, "phase,chunk,action_final,action_orig,actual_ratio,"
                   "predicted_ratio,mape_pct,sgd_fired,exploration_triggered,"
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
            double chunk_mape = (d.actual_ratio > 0.0f)
                ? fabs((double)d.predicted_ratio - (double)d.actual_ratio)
                  / (double)d.actual_ratio * 100.0
                : 0.0;
            fprintf(f, "%s,%d,%s,%s,%.4f,%.4f,%.1f,%d,%d,"
                       "%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    phase_name, i, final_str, orig_str,
                    (double)d.actual_ratio, (double)d.predicted_ratio,
                    chunk_mape,
                    d.sgd_fired, d.exploration_triggered,
                    (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                    (double)d.compression_ms, (double)d.exploration_ms,
                    (double)d.sgd_update_ms);
        } else {
            fprintf(f, "%s,%d,none,none,0,0,0.0,0,0,0,0,0,0,0\n", phase_name, i);
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
            printf("\n  %-14s SGD fired: %d/%d chunks  Explorations: %d/%d chunks  MAPE: %.1f%%",
                   res[i].phase,
                   res[i].sgd_fires,   res[i].n_chunks,
                   res[i].explorations, res[i].n_chunks,
                   res[i].mape_pct);
            printf("\n");
        }
    }

    /* GPU-time overhead breakdown for NN phases */
    bool has_nn = false;
    for (int i = 0; i < n_phases; i++)
        if (strncmp(res[i].phase, "nn", 2) == 0) { has_nn = true; break; }

    if (has_nn) {
        printf("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  GPU-Time Overhead Breakdown (cumulative across chunks, 8 concurrent workers)                  ║\n");
        printf("╠══════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦════════════════╣\n");
        printf("║  Phase       ║ Stats    ║ NN Infer ║ Preproc  ║ Compress ║ Explore  ║ SGD      ║ Total GPU-time ║\n");
        printf("╠══════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬════════════════╣\n");
        for (int i = 0; i < n_phases; i++) {
            if (strncmp(res[i].phase, "nn", 2) != 0) continue;
            PhaseResult *r = &res[i];
            double total_gpu = r->stats_ms + r->nn_ms + r->preproc_ms + r->comp_ms
                             + r->explore_ms + r->sgd_ms;
            printf("║  %-12s║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║   %7.0f ms   ║\n",
                   r->phase, r->stats_ms, r->nn_ms, r->preproc_ms, r->comp_ms,
                   r->explore_ms, r->sgd_ms, total_gpu);
        }
        printf("╠══════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬════════════════╣\n");
        printf("║  (%% of GPU)  ║          ║          ║          ║          ║          ║          ║                ║\n");
        for (int i = 0; i < n_phases; i++) {
            if (strncmp(res[i].phase, "nn", 2) != 0) continue;
            PhaseResult *r = &res[i];
            double total_gpu = r->stats_ms + r->nn_ms + r->preproc_ms + r->comp_ms
                             + r->explore_ms + r->sgd_ms;
            if (total_gpu <= 0) total_gpu = 1.0;
            printf("║  %-12s║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║   wall: %4.0f ms ║\n",
                   r->phase,
                   100.0 * r->stats_ms / total_gpu,
                   100.0 * r->nn_ms / total_gpu,
                   100.0 * r->preproc_ms / total_gpu,
                   100.0 * r->comp_ms / total_gpu,
                   100.0 * r->explore_ms / total_gpu,
                   100.0 * r->sgd_ms / total_gpu,
                   r->write_ms);
        }
        printf("╚══════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩════════════════╝\n");
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
    int chunk_z  = 0;    /* 0 = auto from chunk_mb */
    int chunk_mb = DEFAULT_CHUNK_MB;
    float F      = DEFAULT_F;
    float k      = DEFAULT_K;
    double error_bound = 0.0;  /* 0 = lossless, >0 = lossy quantization */

    /* Phase selection: bit flags */
    enum { P_NOCOMP = 1, P_EXHAUSTIVE = 2, P_NN = 4, P_NNRL = 8, P_NNRLEXP = 16 };
    unsigned int phase_mask = 0;  /* 0 = run all */

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
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
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            const char *p = argv[++i];
            if      (strcmp(p, "no-comp") == 0)      phase_mask |= P_NOCOMP;
            else if (strcmp(p, "exhaustive") == 0)    phase_mask |= P_EXHAUSTIVE;
            else if (strcmp(p, "nn") == 0)            phase_mask |= P_NN;
            else if (strcmp(p, "nn-rl") == 0)         phase_mask |= P_NNRL;
            else if (strcmp(p, "nn-rl+exp50") == 0)   phase_mask |= P_NNRLEXP;
            else { fprintf(stderr, "Unknown phase: %s\n"
                           "  Valid: no-comp, exhaustive, nn, nn-rl, nn-rl+exp50\n", p);
                   return 1; }
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--steps N] "
                "[--chunk-mb N] [--chunk-z Z] [--F val] [--k val] "
                "[--phase <name>] ...\n"
                "  Phases: no-comp, exhaustive, nn, nn-rl, nn-rl+exp50\n"
                "  Use --phase multiple times to select specific phases.\n", argv[0]);
        return 1;
    }
    if (phase_mask == 0) phase_mask = P_NOCOMP | P_EXHAUSTIVE | P_NN | P_NNRL | P_NNRLEXP;

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
    printf("║  Gray-Scott VOL Benchmark: No-Comp vs Exhaustive vs NN+SGD              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Steps    : %d\n", steps);
    printf("  Pattern  : F=%.4f  k=%.5f\n", F, k);
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

    PhaseResult results[5];
    int n_phases = 0;
    int any_fail = 0;

    int rc = 0;

    /* ── Phase 1: no-comp ──────────────────────────────────────────── */
    if (phase_mask & P_NOCOMP) {
        printf("\n── Phase %d: no-comp (GPU→Host→HDF5) ────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        rc = run_phase_nocomp(d_v, d_read, d_count,
                                  n_floats, L, chunk_z,
                                  &results[n_phases]);
        results[n_phases].sim_ms = sim_ms;
        if (rc) any_fail = 1;
        n_phases++;
    }

    /* ── Phase 2: oracle (exhaustive search + end-to-end I/O) ────── */
    if (phase_mask & P_EXHAUSTIVE) {
        printf("\n── Phase %d: exhaustive search (8 algos × 2 shuffle per chunk) ──\n", n_phases + 1);
        {
            FILE *f_ub = fopen(OUT_UB_CSV, "w");
            if (f_ub) {
                fprintf(f_ub, "chunk,pattern,algorithm,shuffle,quant,error_bound,"
                               "ratio,compressed_bytes,original_bytes,"
                               "comp_time_ms,throughput_mbps,status\n");
            }

            rc = run_oracle_pass(d_v, L, chunk_z, n_chunks, f_ub,
                                 TMP_ORACLE, d_read, d_count,
                                 error_bound, &results[n_phases]);
            if (f_ub) fclose(f_ub);
            results[n_phases].sim_ms = sim_ms;
            if (rc) any_fail = 1;
            printf("  Upper-bound CSV: %s\n", OUT_UB_CSV);
        }
        n_phases++;
    }

    /* ── Phase 3: nn (VOL, ALGO_AUTO, inference-only) ─────────────── */
    if (phase_mask & P_NN) {
        printf("\n── Phase %d: nn (VOL, ALGO_AUTO, inference-only) ────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_nn = make_dcpl_auto(L, chunk_z, error_bound);
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn", TMP_NN, dcpl_nn,
                           &results[n_phases]);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_nn);
        if (rc) any_fail = 1;
        write_chunk_csv("nn", n_chunks);
        n_phases++;
    }

    /* ── Phase 4: nn-rl (ALGO_AUTO + SGD, no exploration) ─────────── */
    if (phase_mask & P_NNRL) {
        printf("\n── Phase %d: nn-rl (ALGO_AUTO + SGD, MAPE>=20%%, LR=0.4) ────\n", n_phases + 1);
        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
        gpucompress_set_exploration(0);
        hid_t dcpl_rl = make_dcpl_auto(L, chunk_z, error_bound);
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn-rl", TMP_NN_RL, dcpl_rl,
                           &results[n_phases]);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_rl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        write_chunk_csv("nn-rl", n_chunks);
        n_phases++;
    }

    /* ── Phase 5: nn-rl+exp50 (ALGO_AUTO + SGD + exploration) ────── */
    if (phase_mask & P_NNRLEXP) {
        printf("\n── Phase %d: nn-rl+exp50 (ALGO_AUTO + SGD + expl@MAPE>=50%%) ─\n", n_phases + 1);
        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
        gpucompress_set_exploration(1);
        gpucompress_set_exploration_threshold(0.50);
        gpucompress_set_exploration_k(1);  // K=1 to avoid GPU OOM with large chunks + 8 workers
        hid_t dcpl_rlexp = make_dcpl_auto(L, chunk_z, error_bound);
        rc = run_phase_vol(d_v, d_read, d_count,
                           n_floats, L, chunk_z,
                           "nn-rl+exp50", TMP_NN_RLEXP, dcpl_rlexp,
                           &results[n_phases]);
        results[n_phases].sim_ms = sim_ms;
        H5Pclose(dcpl_rlexp);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        write_chunk_csv("nn-rl+exp50", n_chunks);
        n_phases++;
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
    remove(TMP_ORACLE);
    remove(TMP_NN);
    remove(TMP_NN_RL);
    remove(TMP_NN_RLEXP);

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}