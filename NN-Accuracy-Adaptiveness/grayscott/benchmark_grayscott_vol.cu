/**
 * @file benchmark_grayscott_vol.cu
 * @brief Gray-Scott VOL Benchmark: No-Comp vs Static vs NN+SGD
 *
 * Runs a Gray-Scott reaction-diffusion simulation on the GPU, then benchmarks
 * writing the 3D field to HDF5 via the GPUCompress VOL connector under five
 * compression phases.  Data stays GPU-resident throughout.
 *
 * Phases:
 *   1. Oracle        : exhaustive per-chunk best-config search (8 algos × 2 shuffle)
 *   2. Baseline NN   : ALGO_AUTO inference-only (no SGD)
 *   3. SGD Study     : sweep LR {0.25, 0.5, 0.75} × MT {0.15, 0.20, 0.25}
 *
 * The simulation is run once to produce GPU-resident data; the same data
 * is used for all phases (oracle operates on raw GPU buffers, baseline/SGD
 * write through HDF5 VOL).
 *
 * Usage:
 *   ./build/benchmark_grayscott_vol model.nnwt \
 *       [--L 256] [--steps 1000] [--chunk-mb 4] \
 *       [--F 0.04] [--k 0.06075] [--no-sgd]
 *
 * Options:
 *   model.nnwt       Path to NN weights (required, or GPUCOMPRESS_WEIGHTS env)
 *   --L N            Grid side length (dataset = N^3 floats, default 256)
 *   --steps N        Simulation time-steps before snapshot (default 1000)
 *   --chunk-mb N     Chunk size in MB (auto-calculates chunk_z, default 4)
 *   --chunk-z Z      Set Z-dimension of chunk directly (overrides --chunk-mb)
 *   --F val          Feed rate (default 0.04)
 *   --k val          Kill rate (default 0.06075)
 *   --no-sgd         Skip SGD hyperparameter sweep
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
 *   benchmarks/grayscott/benchmark_grayscott_vol.csv
 *   benchmarks/grayscott/benchmark_grayscott_vol_chunks.csv
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

#define DEFAULT_L           256
#define DEFAULT_STEPS       1000
#define DEFAULT_CHUNK_MB    4
#define DEFAULT_F           0.04f
#define DEFAULT_K           0.06075f

#define TMP_BASELINE "/tmp/bm_gs_baseline.h5"
#define TMP_SGD      "/tmp/bm_gs_sgd.h5"

#define OUT_DIR        "benchmarks/grayscott"
#define OUT_CSV        OUT_DIR "/benchmark_grayscott_vol.csv"
#define OUT_CHUNKS     OUT_DIR "/benchmark_grayscott_vol_chunks.csv"
#define OUT_SGD_CSV    OUT_DIR "/sgd_study.csv"
#define OUT_UB_CSV     OUT_DIR "/upper_bound.csv"

/* HDF5 filter ID */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

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

/**
 * Gather a chunk from the 3D field for oracle compression.
 *
 * HDF5 dataset [L,L,L] with chunk dims [L,L,chunk_z] splits along the
 * last dimension (k).  The simulation stores data in ZYX order:
 *   gs_idx(x,y,z) = x + y*L + z*L*L
 * so HDF5's element [i][j][k] (= flat offset i*L*L+j*L+k) corresponds
 * to simulation coordinates (x=k, y=j, z=i).
 *
 * Chunk c covers k in [c*chunk_z, (c+1)*chunk_z), i.e., sim x in that
 * range.  These elements are NOT contiguous — they're 16-element runs
 * separated by stride L.
 *
 * We gather them into dst in HDF5's row-major order within the chunk:
 *   dst[i*L*chunk_z + j*chunk_z + lk] = src[i*L*L + j*L + (k0+lk)]
 * where k0 = c * chunk_z.
 */
__global__ void gather_chunk_kernel(float *dst, const float *src,
                                    int L, int chunk_z, int k0)
{
    size_t chunk_floats = (size_t)L * L * chunk_z;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_floats; idx += (size_t)gridDim.x * blockDim.x) {
        /* dst layout: [L][L][chunk_z] row-major => i*L*cz + j*cz + lk */
        size_t lk  = idx % chunk_z;
        size_t rem = idx / chunk_z;
        size_t j   = rem % L;
        size_t i   = rem / L;
        size_t src_idx = i * (size_t)L * L + j * (size_t)L + (k0 + lk);
        dst[idx] = src[src_idx];
    }
}

static void gather_chunk(float *dst, const float *src, int L, int chunk_z, int chunk_idx)
{
    size_t chunk_floats = (size_t)L * L * chunk_z;
    int blocks = (int)((chunk_floats + 255) / 256);
    if (blocks > 2048) blocks = 2048;
    int k0 = chunk_idx * chunk_z;
    gather_chunk_kernel<<<blocks, 256>>>(dst, src, L, chunk_z, k0);
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
static hid_t make_dcpl_auto(int L, int chunk_z)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
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

    /* Per-phase timing breakdown (summed across chunks) */
    double total_nn_inference_ms;
    double total_preprocessing_ms;
    double total_compression_ms;
    double total_exploration_ms;
    double total_sgd_ms;

    /* Per-phase ratio distribution */
    double ratio_min;
    double ratio_max;
    double ratio_stddev;

    /* NN prediction accuracy */
    double mean_prediction_error_pct;  /* MAPE */
} PhaseResult;

/* ============================================================
 * Run simulation and capture V-field and U-field pointers
 *
 * After simulation completes, d_u is unused scratch space that
 * can serve as the read-back buffer — saving 1× dataset of GPU
 * memory (e.g. 8 GB for L=1260).
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
 * Phase: VOL write/read with compression
 * ============================================================ */

static int run_phase_vol(gpucompress_grayscott_t sim, int steps,
                         unsigned long long *d_count,
                         size_t n_floats, int L, int chunk_z,
                         const char *phase_name, const char *tmp_file,
                         hid_t dcpl,
                         PhaseResult *r)
{
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    printf("[%s] Running simulation (%d steps)...\n", phase_name, steps);
    float *d_v = NULL, *d_read = NULL;
    double sim_ms = run_simulation(sim, steps, &d_v, &d_read);
    printf("[%s] Simulation: %.0f ms\n", phase_name, sim_ms);

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
    double t3   = now_ms();
    H5Dclose(dset);
    H5Fclose(file);
    printf("%.0f ms\n", t3 - t2);

    if (rret < 0) { fprintf(stderr, "[%s] H5Dread failed\n", phase_name); return 1; }

    /* Compare d_v (original) vs d_read (d_u, read-back) — same sim run */
    unsigned long long mm = gpu_compare(d_v, d_read, n_floats, d_count);

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
    r->sim_ms      = sim_ms;
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

    printf("[%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  sgd=%d expl=%d/%d  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), sgd_fires, explorations, n_hist, mm);
    return (mm == 0) ? 0 : 1;
}

/* ============================================================
 * CSV output
 * ============================================================ */

static void write_chunk_csv(const char *phase_name, int n_chunks,
                            PhaseResult *r)
{
    FILE *f;
    static int header_written = 0;

    if (!header_written) {
        f = fopen(OUT_CHUNKS, "w");
        if (!f) { perror("fopen " OUT_CHUNKS); return; }
        fprintf(f, "phase,chunk,nn_action,explored,sgd_fired,"
                   "nn_inference_ms,preprocessing_ms,compression_ms,"
                   "exploration_ms,sgd_update_ms,"
                   "actual_ratio,predicted_ratio\n");
        fclose(f);
        header_written = 1;
    }

    f = fopen(OUT_CHUNKS, "a");
    if (!f) { perror("fopen " OUT_CHUNKS); return; }

    /* Aggregate accumulators */
    double sum_nn = 0, sum_pp = 0, sum_comp = 0, sum_expl = 0, sum_sgd = 0;
    double r_min = 1e30, r_max = 0, r_sum = 0, r_sum2 = 0;
    double mape_sum = 0;
    int    mape_count = 0;
    int    n_valid = 0;

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist && i < n_chunks; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            fprintf(f, "%s,%d,%d,%d,%d,"
                       "%.3f,%.3f,%.3f,%.3f,%.3f,"
                       "%.4f,%.4f\n",
                    phase_name, i,
                    d.nn_action, d.exploration_triggered, d.sgd_fired,
                    d.nn_inference_ms, d.preprocessing_ms, d.compression_ms,
                    d.exploration_ms, d.sgd_update_ms,
                    d.actual_ratio, d.predicted_ratio);

            sum_nn   += d.nn_inference_ms;
            sum_pp   += d.preprocessing_ms;
            sum_comp += d.compression_ms;
            sum_expl += d.exploration_ms;
            sum_sgd  += d.sgd_update_ms;

            if (d.actual_ratio > 0) {
                if (d.actual_ratio < r_min) r_min = d.actual_ratio;
                if (d.actual_ratio > r_max) r_max = d.actual_ratio;
                r_sum  += d.actual_ratio;
                r_sum2 += (double)d.actual_ratio * d.actual_ratio;
                n_valid++;
            }
            if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
                mape_sum += fabs((double)d.predicted_ratio - d.actual_ratio)
                            / d.actual_ratio;
                mape_count++;
            }
        } else {
            fprintf(f, "%s,%d,-1,0,0,0,0,0,0,0,0,0\n", phase_name, i);
        }
    }
    fclose(f);

    /* Fill aggregate fields in PhaseResult */
    if (r) {
        r->total_nn_inference_ms  = sum_nn;
        r->total_preprocessing_ms = sum_pp;
        r->total_compression_ms   = sum_comp;
        r->total_exploration_ms   = sum_expl;
        r->total_sgd_ms           = sum_sgd;
        r->ratio_min = (n_valid > 0) ? r_min : 0;
        r->ratio_max = (n_valid > 0) ? r_max : 0;
        if (n_valid > 1) {
            double mean = r_sum / n_valid;
            r->ratio_stddev = sqrt((r_sum2 / n_valid) - mean * mean);
        } else {
            r->ratio_stddev = 0;
        }
        r->mean_prediction_error_pct = (mape_count > 0)
            ? (mape_sum / mape_count) * 100.0 : 0;
    }
}

/* ============================================================
 * Best-Static Oracle: try all configs per chunk exhaustively
 * ============================================================ */

static int run_oracle_pass(const float *d_data, int L, int chunk_z,
                           int n_chunks, FILE *f_ub, double *oracle_ratio_out)
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

    static const char *ALGO_NAMES[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate",
        "zstd", "ans", "cascaded", "bitcomp"
    };

    size_t total_original = 0, total_best_compressed = 0;

    for (int c = 0; c < n_chunks; c++) {
        int actual_cz = chunk_z;
        if (c * chunk_z + actual_cz > L) actual_cz = L - c * chunk_z;
        size_t chunk_bytes = (size_t)L * L * actual_cz * sizeof(float);

        gather_chunk(d_chunk_buf, d_data, L, chunk_z, c);
        cudaDeviceSynchronize();

        double best_ratio = 0;
        size_t best_compressed = chunk_bytes;

        for (int algo_enum = 1; algo_enum <= 8; algo_enum++) {
            for (int shuffle = 0; shuffle <= 1; shuffle++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)algo_enum;
                cfg.preprocessing = 0;
                if (shuffle) cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                cfg.error_bound = 0.0;  /* lossless */

                size_t out_size = out_buf_size;
                gpucompress_stats_t stats;
                gpucompress_error_t err = gpucompress_compress_gpu(
                    d_chunk_buf, chunk_bytes,
                    d_output, &out_size, &cfg, &stats, NULL);

                double ratio = (err == GPUCOMPRESS_SUCCESS)
                    ? stats.compression_ratio : 0;

                if (f_ub) {
                    fprintf(f_ub, "%d,grayscott,%s,%d,0,0,%.4f,%zu,%zu,%s\n",
                            c, ALGO_NAMES[algo_enum],
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

    *oracle_ratio_out = (total_best_compressed > 0)
        ? (double)total_original / (double)total_best_compressed : 1.0;

    cudaFree(d_chunk_buf);
    cudaFree(d_output);
    return 0;
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
    int no_sgd   = 0;

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
        } else if (strcmp(argv[i], "--no-sgd") == 0) {
            no_sgd = 1;
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--steps N] "
                "[--chunk-mb N] [--chunk-z Z] [--F val] [--k val] [--no-sgd]\n", argv[0]);
        return 1;
    }

    /* Compute chunk_z from chunk_mb if not explicitly set */
    if (chunk_z <= 0 && chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    }
    if (chunk_z < 1) chunk_z = L / 4;
    if (chunk_z < 1) chunk_z = 1;

    /* SGD hyperparameter grid */
    float sgd_lrs[]   = {0.25f, 0.5f, 0.75f};
    float sgd_mapes[] = {0.15f, 0.20f, 0.25f};
    int n_lrs   = no_sgd ? 0 : 3;
    int n_mapes = no_sgd ? 0 : 3;

    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int    n_chunks    = (L + chunk_z - 1) / chunk_z;
    double dataset_mb  = (double)total_bytes / (1 << 20);
    double cmb         = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);

    int total_runs = 2 + n_lrs * n_mapes;  /* oracle + baseline + SGD */

    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Gray-Scott Adaptiveness Benchmark: Oracle vs Baseline vs SGD            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Steps    : %d\n", steps);
    printf("  Pattern  : F=%.4f  k=%.5f\n", F, k);
    printf("  Weights  : %s\n", weights_path);
    printf("  Total runs: %d (1 oracle + 1 baseline + %d SGD)\n\n",
           total_runs, n_lrs * n_mapes);

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

    /* Create Gray-Scott simulation */
    GrayScottSettings s = gray_scott_default_settings();
    s.L = L;
    s.F = F;
    s.k = k;
    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);

    /* Allocate GPU mismatch counter */
    unsigned long long *d_count = NULL;
    if (cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc for mismatch counter failed\n");
        gpucompress_grayscott_destroy(sim);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    int any_fail = 0;
    int run_num  = 0;

    /* ────────────────────────────────────────────────────────────────────
     * Run simulation once — data stays GPU-resident for all phases
     * ──────────────────────────────────────────────────────────────────── */
    printf("\n═══ RUNNING GRAY-SCOTT SIMULATION (%d steps) ═══\n\n", steps);
    float *d_v = NULL, *d_u = NULL;
    double sim_ms = run_simulation(sim, steps, &d_v, &d_u);
    printf("  Simulation completed in %.0f ms\n", sim_ms);

    /* ────────────────────────────────────────────────────────────────────
     * PHASE 1: ORACLE — exhaustive best-config per chunk
     * ──────────────────────────────────────────────────────────────────── */
    run_num++;
    printf("\n═══ [%d/%d] ORACLE: Best-Static Per-Chunk Search ═══\n\n", run_num, total_runs);

    FILE *f_ub = fopen(OUT_UB_CSV, "w");
    if (f_ub) {
        fprintf(f_ub, "chunk,pattern,algorithm,shuffle,quant,error_bound,"
                       "ratio,compressed_bytes,original_bytes,status\n");
    }

    double oracle_ratio = 1.0;
    int orc = run_oracle_pass(d_v, L, chunk_z, n_chunks, f_ub, &oracle_ratio);
    if (f_ub) fclose(f_ub);
    if (orc) {
        fprintf(stderr, "  Oracle pass failed\n");
        any_fail = 1;
    }
    printf("\n  Oracle aggregate ratio: %.2fx\n", oracle_ratio);
    printf("  Upper-bound CSV written to: %s\n", OUT_UB_CSV);

    /* ────────────────────────────────────────────────────────────────────
     * PHASE 2: BASELINE — NN inference-only (no SGD)
     * ──────────────────────────────────────────────────────────────────── */
    run_num++;
    printf("\n═══ [%d/%d] BASELINE: NN Inference-Only ═══\n\n", run_num, total_runs);

    gpucompress_reload_nn(weights_path);
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    PhaseResult baseline_res;
    memset(&baseline_res, 0, sizeof(baseline_res));
    hid_t dcpl_nn = make_dcpl_auto(L, chunk_z);
    int rc = run_phase_vol(sim, steps, d_count,
                           n_floats, L, chunk_z,
                           "baseline", TMP_BASELINE, dcpl_nn,
                           &baseline_res);
    H5Pclose(dcpl_nn);
    if (rc) any_fail = 1;

    /* Reset chunk CSV header flag */
    write_chunk_csv("baseline", n_chunks, &baseline_res);

    double baseline_ratio = baseline_res.ratio;
    printf("\n  Baseline ratio: %.2fx  (oracle: %.2fx, gap: %.1f%%)\n",
           baseline_ratio, oracle_ratio,
           oracle_ratio > 0 ? (1.0 - baseline_ratio / oracle_ratio) * 100.0 : 0);

    /* ────────────────────────────────────────────────────────────────────
     * PHASE 3: SGD HYPERPARAMETER SWEEP
     * ──────────────────────────────────────────────────────────────────── */
    FILE *f_sgd = fopen(OUT_SGD_CSV, "w");
    if (f_sgd) {
        fprintf(f_sgd, "lr,mape_threshold,n_chunks,final_mape,convergence_chunks,"
                        "ratio,ratio_vs_oracle,sgd_fire_rate,"
                        "total_write_ms,total_sgd_ms,"
                        "total_nn_inference_ms,total_compression_ms,"
                        "ratio_min,ratio_max,ratio_stddev,"
                        "write_mibps,file_mib,mismatches\n");
    }

    if (n_lrs > 0)
        printf("\n═══ SGD HYPERPARAMETER STUDY ═══\n\n");

    PhaseResult best_sgd_res;
    memset(&best_sgd_res, 0, sizeof(best_sgd_res));
    double best_sgd_ratio = 0;
    float  best_sgd_lr = 0, best_sgd_mt = 0;

    for (int li = 0; li < n_lrs; li++) {
        for (int mi = 0; mi < n_mapes; mi++) {
            float lr   = sgd_lrs[li];
            float mape = sgd_mapes[mi];

            run_num++;
            printf("  [%d/%d] SGD lr=%.3f mt=%.2f ... ", run_num, total_runs, lr, mape);
            fflush(stdout);

            gpucompress_reload_nn(weights_path);
            gpucompress_enable_online_learning();
            gpucompress_set_reinforcement(1, lr, mape, mape);
            gpucompress_set_exploration(0);
            gpucompress_set_exploration_k(-1);

            PhaseResult sgd_res;
            memset(&sgd_res, 0, sizeof(sgd_res));

            char phase_name[32];
            snprintf(phase_name, sizeof(phase_name), "sgd_%.2f_%.2f", lr, mape);
            char tmp_file[64];
            snprintf(tmp_file, sizeof(tmp_file), "/tmp/bm_gs_sgd_%.2f_%.2f.h5", lr, mape);

            hid_t dcpl_sgd = make_dcpl_auto(L, chunk_z);
            rc = run_phase_vol(sim, steps, d_count,
                               n_floats, L, chunk_z,
                               phase_name, tmp_file, dcpl_sgd,
                               &sgd_res);
            H5Pclose(dcpl_sgd);
            gpucompress_disable_online_learning();
            if (rc) any_fail = 1;

            write_chunk_csv(phase_name, n_chunks, &sgd_res);

            /* Compute convergence and final MAPE from chunk history */
            int convergence_chunks = 0;
            double final_mape_val = 0;
            int n_hist = gpucompress_get_chunk_history_count();
            {
                double mape_sum = 0;
                int mape_count = 0;
                int converged = 0;
                for (int i = 0; i < n_hist; i++) {
                    gpucompress_chunk_diag_t d;
                    if (gpucompress_get_chunk_diag(i, &d) != 0) continue;
                    if (d.actual_ratio > 0 && d.predicted_ratio > 0) {
                        double err = fabs((double)d.predicted_ratio - d.actual_ratio)
                                     / d.actual_ratio;
                        mape_sum += err;
                        mape_count++;
                        if (!converged && err < (double)mape) {
                            convergence_chunks = i + 1;
                            converged = 1;
                        }
                    }
                }
                final_mape_val = (mape_count > 0) ? (mape_sum / mape_count) * 100.0 : 0;
            }

            double sgd_fire_rate = (sgd_res.n_chunks > 0)
                ? (double)sgd_res.sgd_fires / sgd_res.n_chunks : 0;
            double ratio_vs_oracle = (oracle_ratio > 0)
                ? sgd_res.ratio / oracle_ratio : 0;

            if (f_sgd) {
                fprintf(f_sgd, "%.3f,%.3f,%d,%.4f,%d,"
                               "%.4f,%.4f,%.4f,%.2f,%.2f,"
                               "%.2f,%.2f,"
                               "%.4f,%.4f,%.4f,%.1f,%.2f,%llu\n",
                        lr, mape, sgd_res.n_chunks,
                        final_mape_val, convergence_chunks,
                        sgd_res.ratio, ratio_vs_oracle, sgd_fire_rate,
                        sgd_res.write_ms, sgd_res.total_sgd_ms,
                        sgd_res.total_nn_inference_ms, sgd_res.total_compression_ms,
                        sgd_res.ratio_min, sgd_res.ratio_max, sgd_res.ratio_stddev,
                        sgd_res.write_mbps, (double)sgd_res.file_bytes / (1 << 20),
                        sgd_res.mismatches);
            }

            printf("ratio=%.2fx  mape=%.1f%%  conv=%d  sgd=%d/%d  %s\n",
                   sgd_res.ratio, final_mape_val, convergence_chunks,
                   sgd_res.sgd_fires, sgd_res.n_chunks,
                   sgd_res.mismatches == 0 ? "PASS" : "FAIL");

            /* Track best SGD */
            if (sgd_res.ratio > best_sgd_ratio) {
                best_sgd_ratio = sgd_res.ratio;
                best_sgd_res   = sgd_res;
                best_sgd_lr    = lr;
                best_sgd_mt    = mape;
            }

            remove(tmp_file);
        }
    }

    if (f_sgd) fclose(f_sgd);

    /* ── Summary ───────────────────────────────────────────────────── */
    printf("\n╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Gray-Scott Adaptiveness Summary                                         ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Grid: %d^3 (%.0f MB)  Chunks: %d x %.1f MB  Steps: %d\n",
           L, dataset_mb, n_chunks, cmb, steps);
    printf("║  Pattern: F=%.4f  k=%.5f\n", F, k);
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Oracle (best-static):   %.2fx\n", oracle_ratio);
    printf("║  Baseline (NN only):     %.2fx  (%.1f%% of oracle)\n",
           baseline_ratio, oracle_ratio > 0 ? baseline_ratio / oracle_ratio * 100.0 : 0);
    if (n_lrs > 0) {
        printf("║  Best SGD (lr=%.2f mt=%.2f): %.2fx  (%.1f%% of oracle)\n",
               best_sgd_lr, best_sgd_mt, best_sgd_ratio,
               oracle_ratio > 0 ? best_sgd_ratio / oracle_ratio * 100.0 : 0);
    }
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    /* Write aggregate CSV */
    {
        FILE *f = fopen(OUT_CSV, "w");
        if (f) {
            fprintf(f, "phase,ratio,sim_ms,write_ms,read_ms,file_mib,orig_mib,"
                       "write_mibps,read_mibps,mismatches,sgd_fires,n_chunks,"
                       "ratio_min,ratio_max,ratio_stddev,mean_prediction_error_pct\n");
            fprintf(f, "oracle,%.4f,%.2f,0,0,0,%.2f,0,0,0,0,%d,0,0,0,0\n",
                    oracle_ratio, sim_ms, dataset_mb, n_chunks);
            fprintf(f, "baseline,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                       "%.1f,%.1f,%llu,%d,%d,"
                       "%.4f,%.4f,%.4f,%.2f\n",
                    baseline_res.ratio, baseline_res.sim_ms,
                    baseline_res.write_ms, baseline_res.read_ms,
                    (double)baseline_res.file_bytes / (1 << 20),
                    (double)baseline_res.orig_bytes / (1 << 20),
                    baseline_res.write_mbps, baseline_res.read_mbps,
                    baseline_res.mismatches, baseline_res.sgd_fires,
                    baseline_res.n_chunks,
                    baseline_res.ratio_min, baseline_res.ratio_max,
                    baseline_res.ratio_stddev,
                    baseline_res.mean_prediction_error_pct);
            if (n_lrs > 0) {
                fprintf(f, "best_sgd,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                           "%.1f,%.1f,%llu,%d,%d,"
                           "%.4f,%.4f,%.4f,%.2f\n",
                        best_sgd_res.ratio, best_sgd_res.sim_ms,
                        best_sgd_res.write_ms, best_sgd_res.read_ms,
                        (double)best_sgd_res.file_bytes / (1 << 20),
                        (double)best_sgd_res.orig_bytes / (1 << 20),
                        best_sgd_res.write_mbps, best_sgd_res.read_mbps,
                        best_sgd_res.mismatches, best_sgd_res.sgd_fires,
                        best_sgd_res.n_chunks,
                        best_sgd_res.ratio_min, best_sgd_res.ratio_max,
                        best_sgd_res.ratio_stddev,
                        best_sgd_res.mean_prediction_error_pct);
            }
            fclose(f);
            printf("\nAggregate CSV written to: %s\n", OUT_CSV);
        }
    }
    if (n_lrs > 0)
        printf("SGD study CSV written to: %s\n", OUT_SGD_CSV);
    printf("Per-chunk CSV written to: %s\n", OUT_CHUNKS);

    /* ── Cleanup ───────────────────────────────────────────────────── */
    cudaFree(d_count);
    gpucompress_grayscott_destroy(sim);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    remove(TMP_BASELINE);
    remove(TMP_SGD);

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}
