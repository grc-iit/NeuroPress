/**
 * @file grayscott-benchmark.cu
 * @brief Gray-Scott VOL Benchmark: No-Comp vs NN+SGD
 *
 * Runs a Gray-Scott reaction-diffusion simulation on the GPU, then benchmarks
 * writing the 3D field to HDF5 via the GPUCompress VOL connector.
 * Data stays GPU-resident throughout.
 *
 * Single-shot phases (run once):
 *   1. no-comp      : GPU sim → H5Dwrite via VOL (no compression, baseline)
 *   2. nn           : GPU sim → H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *
 * Multi-timestep phases (require --timesteps N):
 *   3. nn-rl        : ALGO_AUTO + online SGD (learning curve over N writes)
 *   4. nn-rl+exp50  : ALGO_AUTO + online SGD + exploration
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
 *                    Valid names: no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd,
 *                                fixed-lz4+shuf, fixed-gdeflate+shuf, fixed-zstd+shuf,
 *                                nn, nn-rl, nn-rl+exp50
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

#ifdef GPUCOMPRESS_USE_MPI
#include <mpi.h>
#endif

#include "gpucompress.h"
#include "gpucompress_grayscott.h"
#include "gpucompress_hdf5_vol.h"
#include "../kendall_tau_profiler.cuh"

/* MPI rank/size — defaults to single-process when MPI is not used */
static int g_mpi_rank = 0;
static int g_mpi_size = 1;

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define DEFAULT_L           512
#define DEFAULT_STEPS       1000
#define DEFAULT_CHUNK_MB    8
#define DEFAULT_TIMESTEPS   1
#define DEFAULT_F           0.04f
#define DEFAULT_K           0.06075f

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
static char TMP_WARMUP[256];

static void init_tmp_paths(int mpi_rank) {
    snprintf(TMP_NOCOMP,      sizeof(TMP_NOCOMP),      "/tmp/bm_gs_nocomp_rank%d.h5",      mpi_rank);
    snprintf(TMP_FIX_LZ4,     sizeof(TMP_FIX_LZ4),     "/tmp/bm_gs_fix_lz4_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_SNAPPY,   sizeof(TMP_FIX_SNAPPY),   "/tmp/bm_gs_fix_snappy_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_DEFL,    sizeof(TMP_FIX_DEFL),    "/tmp/bm_gs_fix_deflate_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_GDEFL,   sizeof(TMP_FIX_GDEFL),   "/tmp/bm_gs_fix_gdefl_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_ZSTD,    sizeof(TMP_FIX_ZSTD),    "/tmp/bm_gs_fix_zstd_rank%d.h5",    mpi_rank);
    snprintf(TMP_FIX_ANS,     sizeof(TMP_FIX_ANS),     "/tmp/bm_gs_fix_ans_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_CASC,    sizeof(TMP_FIX_CASC),    "/tmp/bm_gs_fix_cascaded_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_BITCOMP,  sizeof(TMP_FIX_BITCOMP),  "/tmp/bm_gs_fix_bitcomp_rank%d.h5",  mpi_rank);
    snprintf(TMP_NN,          sizeof(TMP_NN),          "/tmp/bm_gs_nn_rank%d.h5",          mpi_rank);
    snprintf(TMP_NN_RL,       sizeof(TMP_NN_RL),       "/tmp/bm_gs_nn_rl_rank%d.h5",       mpi_rank);
    snprintf(TMP_NN_RLEXP,    sizeof(TMP_NN_RLEXP),    "/tmp/bm_gs_nn_rlexp_rank%d.h5",    mpi_rank);
    snprintf(TMP_WARMUP,      sizeof(TMP_WARMUP),      "/tmp/bm_gs_warmup_rank%d.h5",      mpi_rank);
}

#define DEFAULT_OUT_DIR "benchmarks/grayscott/results"

/* Output paths — populated at runtime from out_dir */
static char OUT_DIR[512];
static char OUT_CSV[512];
static char OUT_CHUNKS[512];
static char OUT_TSTEP[512];
static char OUT_TSTEP_CHUNKS[512];
static char OUT_RANKING[512];
static char OUT_RANKING_COSTS[512];

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
 * GPU kernel: inject contrasting data patterns into specific
 * Z-slices of the field.  Each injected region gets a different
 * pattern so the NN sees chunks with very different statistics.
 *
 * Patterns (by region):
 *   0: random noise    — high entropy, incompressible
 *   1: constant zero   — trivially compressible
 *   2: smooth gradient — low entropy, compressible with delta encoding
 *   3: periodic sine   — medium entropy, repetitive structure
 *   4: sparse spikes   — mostly zero with rare large values
 * ============================================================ */

__global__ void inject_patterns_kernel(float *data, int L, int chunk_z,
                                        int n_chunks, unsigned long long seed)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slice_size = (size_t)L * L * chunk_z;
    size_t total = (size_t)L * L * L;

    for (size_t i = idx; i < total; i += (size_t)gridDim.x * blockDim.x) {
        int chunk_id = (int)(i / slice_size);
        /* Only inject into the middle third of chunks */
        int first_inject = n_chunks / 3;
        int last_inject  = (2 * n_chunks) / 3;
        if (chunk_id < first_inject || chunk_id >= last_inject)
            continue;

        int pattern = (chunk_id - first_inject) % 5;
        size_t local = i - (size_t)chunk_id * slice_size;

        /* Simple hash for pseudo-random */
        unsigned long long h = (seed ^ (i * 6364136223846793005ULL)) + 1442695040888963407ULL;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        float rnd = (float)(h & 0xFFFFFF) / (float)0xFFFFFF;

        float val;
        switch (pattern) {
            case 0: /* random noise: uniform [0, 1000] */
                val = rnd * 1000.0f;
                break;
            case 1: /* constant zero */
                val = 0.0f;
                break;
            case 2: /* smooth gradient: small integers */
                val = (float)(local % 256);
                break;
            case 3: /* periodic sine */
                val = sinf((float)local * 0.01f) * 500.0f;
                break;
            case 4: /* sparse spikes: 99% zeros, 1% large */
                val = (rnd < 0.01f) ? 99999.0f : 0.0f;
                break;
            default:
                val = data[i];
                break;
        }
        data[i] = val;
    }
}

static void inject_contrasting_patterns(float *d_v, int L, int chunk_z)
{
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    int first = n_chunks / 3;
    int last  = (2 * n_chunks) / 3;
    int n_injected = last - first;
    if (n_injected <= 0) return;

    printf("  Injecting %d contrasting pattern chunks (%d-%d of %d) into V-field...\n",
           n_injected, first, last - 1, n_chunks);

    size_t total = (size_t)L * L * L;
    int blocks = (int)((total + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    inject_patterns_kernel<<<blocks, 256>>>(d_v, L, chunk_z, n_chunks, 42ULL);
    cudaDeviceSynchronize();
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

/* 3D chunked DCPL for a fixed algorithm with optional preprocessing */
static hid_t make_dcpl_fixed(int L, int chunk_z, gpucompress_algorithm_t algo,
                              unsigned int preproc = 0)
{
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo;
    cd[1] = preproc;
    cd[2] = 0;
    cd[3] = 0; cd[4] = 0; /* error_bound = 0.0 */
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
    double mae_ratio;
    double mae_comp_ms;
    double mae_decomp_ms;
    /* Per-component GPU-time (cumulative across chunks) */
    double nn_ms;
    double stats_ms;
    double preproc_ms;
    double comp_ms;
    double decomp_ms;
    double explore_ms;
    double sgd_ms;
    /* Isolated compression/decompression throughput (GB/s) */
    double comp_gbps;
    double decomp_gbps;
    /* Standard deviations (populated when --runs N > 1) */
    double write_ms_std;
    double read_ms_std;
    double comp_gbps_std;
    double decomp_gbps_std;
    int    n_runs;  /* number of runs used for this result */
    /* VOL pipeline stage timing (averaged across timesteps) */
    double stage1_ms;
    double drain_ms;
    double io_drain_ms;
    double s2_busy_ms;
    double s3_busy_ms;
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
 * Phase: no-comp (GPU ptr, VOL fallback D→H + native HDF5)
 *
 * Measures uncompressed I/O through the same VOL connector stack
 * used by compressed phases.  The VOL detects no gpucompress
 * filter and falls back to: cudaMemcpy D→H (full dataset) →
 * pinned host buffer → native HDF5 chunked write.
 *
 * This is the correct baseline for an apples-to-apples comparison:
 * both compressed and uncompressed paths go through the same VOL
 * connector, HDF5 file format, and storage layer.  The throughput
 * difference reflects the actual benefit of GPU compression —
 * reduced D→H transfer volume and reduced disk I/O.
 * ============================================================ */

static int run_phase_nocomp(float *d_v, float *d_read,
                            unsigned long long *d_count,
                            size_t n_floats, int L, int chunk_z,
                            PhaseResult *r)
{
    memset(r, 0, sizeof(*r));
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* Warmup write: primes VOL fallback path and HDF5 infrastructure.
     * Results discarded — only the timed run counts. */
    {
        const char *warmup_file = TMP_WARMUP;
        remove(warmup_file);
        hid_t wdcpl = make_dcpl_nocomp(L, chunk_z);
        hid_t wfapl = make_vol_fapl();
        hid_t wfile = H5Fcreate(warmup_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
        H5Pclose(wfapl);
        if (wfile >= 0) {
            hid_t wfsp  = H5Screate_simple(3, dims, NULL);
            hid_t wdset = H5Dcreate2(wfile, "V", H5T_NATIVE_FLOAT,
                                      wfsp, H5P_DEFAULT, wdcpl, H5P_DEFAULT);
            H5Sclose(wfsp);
            if (wdset >= 0) {
                H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
                cudaDeviceSynchronize();
                H5Dclose(wdset);
            }
            H5Fclose(wfile);
            remove(warmup_file);
        }
        H5Pclose(wdcpl);
    }

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
    r->mape_ratio_pct   = 0.0;
    r->mape_comp_pct    = 0.0;
    r->mape_decomp_pct  = 0.0;
    r->comp_gbps   = 0.0;
    r->decomp_gbps = 0.0;
    r->write_ms_std = 0.0;
    r->read_ms_std  = 0.0;
    r->comp_gbps_std  = 0.0;
    r->decomp_gbps_std = 0.0;
    r->n_runs      = 1;
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
    memset(r, 0, sizeof(*r));
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* Warmup write: primes VOL write context, nvCOMP manager cache,
     * and NN inference JIT.  Results discarded — only the timed run counts.
     * Learning is temporarily disabled to prevent SGD from modifying weights
     * before the timed run. */
    {
        const char *warmup_file = TMP_WARMUP;
        int save_learning = gpucompress_online_learning_enabled();
        gpucompress_disable_online_learning();

        remove(warmup_file);
        hid_t wfapl = make_vol_fapl();
        hid_t wfile = H5Fcreate(warmup_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
        H5Pclose(wfapl);
        if (wfile >= 0) {
            hid_t wfsp  = H5Screate_simple(3, dims, NULL);
            hid_t wdset = H5Dcreate2(wfile, "V", H5T_NATIVE_FLOAT,
                                      wfsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(wfsp);
            if (wdset >= 0) {
                H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
                cudaDeviceSynchronize();
                H5Dclose(wdset);
            }
            H5Fclose(wfile);
            remove(warmup_file);
        }

        if (save_learning) gpucompress_enable_online_learning();
        gpucompress_flush_manager_cache();  /* cold-start the timed run */
    }

    /* VOL write */
    printf("[%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
    gpucompress_set_debug_context(phase_name, -1);
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
    cudaDeviceSynchronize();
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
    /* MAPE, MAE accumulators */
    double mape_r_sum = 0.0, mape_c_sum = 0.0, mape_d_sum = 0.0;
    double mae_r_sum = 0.0, mae_c_sum = 0.0, mae_d_sum = 0.0;
    int    mape_r_cnt = 0,   mape_c_cnt = 0,   mape_d_cnt = 0;
    double total_nn_ms     = 0.0;
    double total_stats_ms  = 0.0;
    double total_preproc_ms = 0.0;
    double total_comp_ms   = 0.0;
    double total_decomp_ms = 0.0;
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
            total_comp_ms    += d.compression_ms_raw;  /* unclamped for breakdown */
            total_decomp_ms  += d.decompression_ms_raw;  /* unclamped for breakdown */
            total_explore_ms += d.exploration_ms;
            total_sgd_ms     += d.sgd_update_ms;

            /* MAPE, MAE */
            if (d.actual_ratio > 0 && d.predicted_ratio > 0) {
                double diff = d.predicted_ratio - d.actual_ratio;
                mape_r_sum += fabs(diff) / fabs(d.actual_ratio);
                mae_r_sum  += fabs(diff);
                mape_r_cnt++;
            }
            if (d.compression_ms > 0) {
                mape_c_sum += fabs(d.predicted_comp_time - d.compression_ms) / fabs(d.compression_ms);
                mae_c_sum  += fabs(d.predicted_comp_time - d.compression_ms);
                mape_c_cnt++;
            }
            if (d.decompression_ms > 0) {
                mape_d_sum += fabs(d.predicted_decomp_time - d.decompression_ms) / fabs(d.decompression_ms);
                mae_d_sum  += fabs(d.predicted_decomp_time - d.decompression_ms);
                mape_d_cnt++;
            }
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
    r->mape_ratio_pct   = fmin(200.0, (mape_r_cnt > 0)  ? mape_r_sum / mape_r_cnt * 100.0 : 0.0);
    r->mape_comp_pct    = fmin(200.0, (mape_c_cnt > 0)  ? mape_c_sum / mape_c_cnt * 100.0 : 0.0);
    r->mape_decomp_pct  = fmin(200.0, (mape_d_cnt > 0)  ? mape_d_sum / mape_d_cnt * 100.0 : 0.0);
    r->mae_ratio     = mape_r_cnt ? mae_r_sum / mape_r_cnt : 0.0;
    r->mae_comp_ms   = mape_c_cnt ? mae_c_sum / mape_c_cnt : 0.0;
    r->mae_decomp_ms = mape_d_cnt ? mae_d_sum / mape_d_cnt : 0.0;

    printf("  MAPE avg:  ratio=%.1f%%  comp=%.1f%%  decomp=%.1f%%\n",
           r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct);
    r->nn_ms       = total_nn_ms;
    r->stats_ms    = total_stats_ms;
    r->preproc_ms  = total_preproc_ms;
    r->comp_ms     = total_comp_ms;
    r->decomp_ms   = total_decomp_ms;
    r->explore_ms  = total_explore_ms;
    r->sgd_ms      = total_sgd_ms;
    /* Isolated compression/decompression throughput (GB/s) */
    r->comp_gbps   = (total_comp_ms > 0)
        ? (double)total_bytes / total_comp_ms / 1e6 : 0.0;
    r->decomp_gbps = (total_decomp_ms > 0)
        ? (double)total_bytes / total_decomp_ms / 1e6 : 0.0;
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
    if (!f) { perror(OUT_CSV); return; }

    fprintf(f, "rank,source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,"
               "file_mib,orig_mib,ratio,"
               "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks,"
               "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
               "comp_gbps,decomp_gbps,"
               "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,"
               "mae_ratio,mae_comp_ms,mae_decomp_ms,"
               "comp_gbps_std,decomp_gbps_std,"
               "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
               "vol_s2_busy_ms,vol_s3_busy_ms,"
               "L,steps,F,k,chunk_z,sim_ms\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        fprintf(f, "%d,grayscott,%s,%d,%.2f,%.2f,%.2f,%.2f,"
                   "%.2f,%.2f,%.4f,"
                   "%.1f,%.1f,%llu,%d,%d,%d,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                   "%.4f,%.4f,"
                   "%.2f,%.2f,%.2f,"
                   "%.4f,%.4f,%.4f,"
                   "%.4f,%.4f,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,"
                   "%d,%d,%.4f,%.5f,%d,%.2f\n",
                g_mpi_rank, r->phase, r->n_runs,
                r->write_ms, r->write_ms_std, r->read_ms, r->read_ms_std,
                (double)r->file_bytes / (1 << 20),
                (double)r->orig_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps, r->mismatches,
                r->sgd_fires, r->explorations, r->n_chunks,
                r->nn_ms, r->stats_ms, r->preproc_ms,
                r->comp_ms, r->decomp_ms, r->explore_ms, r->sgd_ms,
                r->comp_gbps, r->decomp_gbps,
                r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct,
                r->mae_ratio, r->mae_comp_ms, r->mae_decomp_ms,
                r->comp_gbps_std, r->decomp_gbps_std,
                r->stage1_ms, r->drain_ms, r->io_drain_ms,
                r->s2_busy_ms, r->s3_busy_ms,
                L, steps, F, k, chunk_z, r->sim_ms);
    }
    fclose(f);
    printf("\nAggregate CSV written to: %s\n", OUT_CSV);
}

static void write_chunk_csv(const char *phase_name, int n_chunks)
{
    struct stat csv_st;
    bool need_hdr = (stat(OUT_CHUNKS, &csv_st) != 0 || csv_st.st_size == 0);
    FILE *f = fopen(OUT_CHUNKS, "a");
    if (!f) { perror(OUT_CHUNKS); return; }
    if (need_hdr) {
        fprintf(f, "phase,chunk,action_final,action_orig,"
                   "actual_ratio,predicted_ratio,mape_ratio,"
                   "actual_comp_ms_raw,predicted_comp_ms,mape_comp,"
                   "actual_decomp_ms_raw,predicted_decomp_ms,mape_decomp,"
                   "sgd_fired,exploration_triggered,"
                   "cost_model_error_pct,actual_cost,predicted_cost,"
                   "explore_n_alt");
        for (int ei = 0; ei < 31; ei++)
            fprintf(f, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d",
                    ei, ei, ei, ei);
        fprintf(f, ","
                   "stats_ms,nn_inference_ms,preprocessing_ms,compression_ms_raw,"
                   "exploration_ms,sgd_update_ms,"
                   "feat_entropy,feat_mad,feat_deriv,feat_eb_enc,feat_ds_enc\n");
    }

    int n_hist = gpucompress_get_chunk_history_count();
    for (int i = 0; i < n_hist && i < n_chunks; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            char final_str[40], orig_str[40];
            action_to_str(d.nn_action, final_str, sizeof(final_str));
            action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));
            /* MAPE per chunk */
            double mape_r = fmin(200.0, (d.actual_ratio > 0)
                ? fabs(d.predicted_ratio - d.actual_ratio) / fabs(d.actual_ratio) * 100.0 : 0.0);
            double mape_c = fmin(200.0, (d.compression_ms > 0)
                ? fabs(d.predicted_comp_time - d.compression_ms) / fabs(d.compression_ms) * 100.0 : 0.0);
            double mape_d = fmin(200.0, (d.decompression_ms > 0)
                ? fabs(d.predicted_decomp_time - d.decompression_ms) / fabs(d.decompression_ms) * 100.0 : 0.0);
            fprintf(f, "%s,%d,%s,%s,"
                       "%.4f,%.4f,%.1f,"
                       "%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,"
                       "%d,%d,"
                       "%.4f,%.4f,%.4f,%d",
                    phase_name, i, final_str, orig_str,
                    (double)d.actual_ratio, (double)d.predicted_ratio, mape_r,
                    (double)d.compression_ms_raw, (double)d.predicted_comp_time, mape_c,
                    (double)d.decompression_ms_raw, (double)d.predicted_decomp_time, mape_d,
                    d.sgd_fired, d.exploration_triggered,
                    (double)d.cost_model_error_pct,
                    (double)d.actual_cost, (double)d.predicted_cost,
                    d.explore_n_alternatives);
            /* Exploration alternatives (31 slots) */
            for (int ei = 0; ei < 31; ei++) {
                if (ei < d.explore_n_alternatives) {
                    char alt_str[40];
                    action_to_str(d.explore_alternatives[ei], alt_str, sizeof(alt_str));
                    fprintf(f, ",%s,%.4f,%.4f,%.4f",
                            alt_str,
                            (double)d.explore_ratios[ei],
                            (double)d.explore_comp_ms[ei],
                            (double)d.explore_costs[ei]);
                } else {
                    fprintf(f, ",,,,");
                }
            }
            fprintf(f, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                       "%.4f,%.6f,%.6f,%.4f,%.4f\n",
                    (double)d.stats_ms,
                    (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                    (double)d.compression_ms_raw, (double)d.exploration_ms,
                    (double)d.sgd_update_ms,
                    (double)d.feat_entropy, (double)d.feat_mad, (double)d.feat_deriv,
                    (double)d.feat_eb_enc, (double)d.feat_ds_enc);
        } else {
            fprintf(f, "%s,%d,none,none", phase_name, i);
            for (int z = 0; z < 44; z++) fprintf(f, ",0");
            fprintf(f, "\n");
        }
    }
    fclose(f);
}

/* ============================================================
 * Console summary table
 * ============================================================ */

static void print_summary(PhaseResult *res, int n_phases,
                          int L, int steps, int chunk_z,
                          float F, float k, int do_verify)
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
    printf("╠══════════════╦══════════╦══════════╦═══════╦══════════╦══════════╦══════╦═════════════╣\n");
    printf("║  Phase       ║  Sim     ║ Write    ║ Read  ║ Comp     ║ Ratio   ║ File ║ Verify      ║\n");
    printf("║              ║  (ms)    ║ (MiB/s)  ║(MiB/s)║ (GB/s)   ║         ║(MiB) ║             ║\n");
    printf("╠══════════════╬══════════╬══════════╬═══════╬══════════╬═════════╬══════╬═════════════╣\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult *r = &res[i];
        const char *verdict = !do_verify ? "SKIP"
                             : (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %8.0f ║ %8.0f ║ %5.0f ║ %7.2f  ║ %5.2fx  ║ %4.0f ║ %-11s ║\n",
               r->phase, r->sim_ms, r->write_mbps, r->read_mbps,
               r->comp_gbps, r->ratio, (double)r->file_bytes / (1 << 20), verdict);
    }
    printf("╚══════════════╩══════════╩══════════╩═══════╩══════════╩═════════╩══════╩═════════════╝\n");

    /* NN phase SGD/exploration summary */
    for (int i = 0; i < n_phases; i++) {
        if (strncmp(res[i].phase, "nn", 2) == 0 && res[i].n_chunks > 0) {
            printf("\n  %-14s SGD: %d/%d  Expl: %d/%d\n"
                   "                 MAPE: ratio=%.1f%% comp=%.1f%% decomp=%.1f%%\n",
                   res[i].phase,
                   res[i].sgd_fires,   res[i].n_chunks,
                   res[i].explorations, res[i].n_chunks,
                   res[i].mape_ratio_pct, res[i].mape_comp_pct, res[i].mape_decomp_pct);
        }
    }

}

/* ============================================================
 * Helper: compute mean and std from an array of doubles
 * ============================================================ */

static void compute_mean_std(const double *vals, int n, double *mean, double *std)
{
    if (n <= 0) { *mean = 0; *std = 0; return; }
    double sum = 0;
    for (int i = 0; i < n; i++) sum += vals[i];
    *mean = sum / n;
    if (n <= 1) { *std = 0; return; }
    double var = 0;
    for (int i = 0; i < n; i++) {
        double d = vals[i] - *mean;
        var += d * d;
    }
    *std = sqrt(var / (n - 1));  /* sample std */
}

/* ============================================================
 * Helper: merge N PhaseResults into one with mean ± std
 * Uses the last run's non-timing fields (ratio, mismatches, etc.)
 * ============================================================ */

static void merge_phase_results(PhaseResult *runs, int n, PhaseResult *out)
{
    /* Copy last run as base (ratio, mismatches, sgd_fires, etc.) */
    *out = runs[n - 1];
    out->n_runs = n;

    double wr_ms[32], rd_ms[32], cg[32], dg[32];
    int cap = (n > 32) ? 32 : n;
    for (int i = 0; i < cap; i++) {
        wr_ms[i] = runs[i].write_ms;
        rd_ms[i] = runs[i].read_ms;
        cg[i]    = runs[i].comp_gbps;
        dg[i]    = runs[i].decomp_gbps;
    }

    double m, s;
    compute_mean_std(wr_ms, cap, &m, &s);
    out->write_ms = m; out->write_ms_std = s;
    compute_mean_std(rd_ms, cap, &m, &s);
    out->read_ms = m; out->read_ms_std = s;
    compute_mean_std(cg, cap, &m, &s);
    out->comp_gbps = m; out->comp_gbps_std = s;
    compute_mean_std(dg, cap, &m, &s);
    out->decomp_gbps = m; out->decomp_gbps_std = s;

    /* Recompute throughput from averaged times */
    double total_bytes = (double)out->orig_bytes;
    out->write_mbps = (out->write_ms > 0)
        ? total_bytes / (1 << 20) / (out->write_ms / 1000.0) : 0;
    out->read_mbps = (out->read_ms > 0)
        ? total_bytes / (1 << 20) / (out->read_ms / 1000.0) : 0;
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

    /* Defaults */
    const char *weights_path = NULL;
    int L        = DEFAULT_L;
    int steps    = DEFAULT_STEPS;
    int timesteps = DEFAULT_TIMESTEPS;  /* multi-timestep mode: >1 loops nn-rl */
    int n_runs   = 1;    /* --runs N: repeat each single-shot phase N times for error bars */
    int chunk_z  = 0;    /* 0 = auto from chunk_mb */
    int chunk_mb = DEFAULT_CHUNK_MB;
    float F      = DEFAULT_F;
    float k      = DEFAULT_K;
    float  sgd_lr    = REINFORCE_LR;  /* overridable via --lr */
    float  sgd_mape  = REINFORCE_MAPE; /* overridable via --mape */
    int    explore_k = 4;              /* overridable via --explore-k */
    float  explore_thresh = 0.20f;     /* overridable via --explore-thresh */
    float  rank_w0 = 1.0f, rank_w1 = 1.0f, rank_w2 = 1.0f;  /* cost model weights */
    double error_bound = 0.0;  /* 0 = lossless, >0 = lossy quantization */
    const char *out_dir_override = NULL;  /* overridable via --out-dir */
    int do_verify = 1;       /* --no-verify: skip bitwise validation (read-back still runs for timing) */
    int verbose_chunks = 0;  /* --verbose-chunks: print per-chunk detail at milestones */
    int inject_patterns = 0; /* --inject-patterns: overwrite middle chunks with contrasting data */

    /* Phase selection: bit flags */
    enum { P_NOCOMP = 0x001, P_FIX_LZ4 = 0x002, P_FIX_GDEFL = 0x004, P_FIX_ZSTD = 0x008,
           P_FIX_SNAPPY = 0x010, P_FIX_DEFL = 0x020, P_FIX_ANS = 0x200,
           P_FIX_CASC = 0x400, P_FIX_BITCOMP = 0x800,
           P_NN = 0x040, P_NNRL = 0x080, P_NNRLEXP = 0x100 };
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
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
            if (n_runs < 1) n_runs = 1;
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
        } else if (strcmp(argv[i], "--mape") == 0 && i + 1 < argc) {
            sgd_mape = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--explore-k") == 0 && i + 1 < argc) {
            explore_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--explore-thresh") == 0 && i + 1 < argc) {
            explore_thresh = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--w0") == 0 && i + 1 < argc) {
            rank_w0 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--w1") == 0 && i + 1 < argc) {
            rank_w1 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--w2") == 0 && i + 1 < argc) {
            rank_w2 = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir_override = argv[++i];
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            const char *p = argv[++i];
            if      (strcmp(p, "no-comp") == 0)              phase_mask |= P_NOCOMP;
            else if (strcmp(p, "lz4") == 0)           phase_mask |= P_FIX_LZ4;
            else if (strcmp(p, "snappy") == 0)        phase_mask |= P_FIX_SNAPPY;
            else if (strcmp(p, "deflate") == 0)       phase_mask |= P_FIX_DEFL;
            else if (strcmp(p, "gdeflate") == 0)      phase_mask |= P_FIX_GDEFL;
            else if (strcmp(p, "zstd") == 0)          phase_mask |= P_FIX_ZSTD;
            else if (strcmp(p, "ans") == 0)           phase_mask |= P_FIX_ANS;
            else if (strcmp(p, "cascaded") == 0)      phase_mask |= P_FIX_CASC;
            else if (strcmp(p, "bitcomp") == 0)       phase_mask |= P_FIX_BITCOMP;
            else if (strcmp(p, "nn") == 0)                  phase_mask |= P_NN;
            else if (strcmp(p, "nn-rl") == 0)               phase_mask |= P_NNRL;
            else if (strcmp(p, "nn-rl+exp50") == 0)         phase_mask |= P_NNRLEXP;
            else { fprintf(stderr, "Unknown phase: %s\n"
                           "  Valid: no-comp, fixed-lz4, fixed-snappy, fixed-deflate,\n"
                           "         fixed-gdeflate, fixed-zstd, fixed-ans, fixed-cascaded,\n"
                           "         fixed-bitcomp, nn, nn-rl, nn-rl+exp50\n", p);
                   return 1; }
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            do_verify = 0;
        } else if (strcmp(argv[i], "--verbose-chunks") == 0) {
            verbose_chunks = 1;
        } else if (strcmp(argv[i], "--inject-patterns") == 0) {
            inject_patterns = 1;
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt> [--L N] [--steps N] "
                "[--chunk-mb N] [--chunk-z Z] [--F val] [--k val] "
                "[--runs N] [--phase <name>] [--w0 F] [--w1 F] [--w2 F] ...\n"
                "  --runs N    Repeat each single-shot phase N times (mean ± std)\n"
                "  --w0/w1/w2  Cost model weights (default 1/1/1)\n"
                "              w0=comp_time, w1=decomp_time, w2=IO_cost\n"
                "              e.g. --w0 0 --w1 0 --w2 1  (ratio-only ranking)\n"
                "  Phases: no-comp, fixed-lz4, fixed-gdeflate, fixed-zstd,\n"
                "         fixed-lz4+shuf, fixed-gdeflate+shuf, fixed-zstd+shuf,\n"
                "         nn, nn-rl, nn-rl+exp50\n"
                "  Use --phase multiple times to select specific phases.\n", argv[0]);
        return 1;
    }
    if (phase_mask == 0) phase_mask = P_NOCOMP | P_FIX_LZ4 | P_FIX_SNAPPY | P_FIX_DEFL
                                    | P_FIX_GDEFL | P_FIX_ZSTD | P_FIX_ANS
                                    | P_FIX_CASC | P_FIX_BITCOMP
                                    | P_NN | P_NNRL | P_NNRLEXP;

    /* Compute chunk_z from chunk_mb if not explicitly set */
    if (chunk_z <= 0 && chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    }
    if (chunk_z < 1) chunk_z = L / 4;
    if (chunk_z < 1) chunk_z = 1;

    /* Alignment fix: nvcomp algorithms (bitcomp, ANS, etc.) require the chunk
     * element count to be even so internal uint2/uint4 loads stay 8-byte aligned.
     * When L is odd, L*L is odd, so L*L*chunk_z is odd iff chunk_z is odd —
     * bump chunk_z up by 1 to make the product even without exceeding L. */
    if (((size_t)L * L * (size_t)chunk_z) % 2 != 0) {
        int new_z = chunk_z + 1;
        if (new_z <= L) chunk_z = new_z;
        /* else: already at L, cannot go higher — fall through and hope for the best */
    }

    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int    n_chunks    = (L + chunk_z - 1) / chunk_z;
    double dataset_mb  = (double)total_bytes / (1 << 20);
    double cmb         = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);

    if (g_mpi_rank == 0) {
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Gray-Scott VOL Benchmark: No-Comp vs NN+SGD                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n\n");
    if (g_mpi_size > 1)
        printf("  MPI ranks: %d\n", g_mpi_size);
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, cmb);
    printf("  Steps    : %d per timestep\n", steps);
    if (timesteps > 1)
        printf("  Timesteps: %d (multi-timestep mode: nn-rl across %d writes)\n",
               timesteps, timesteps);
    printf("  Pattern  : F=%.4f  k=%.5f\n", F, k);
    printf("  SGD LR   : %.4f\n", sgd_lr);
    printf("  Cost w0/w1/w2: %.2f / %.2f / %.2f\n", rank_w0, rank_w1, rank_w2);
    gpucompress_set_ranking_weights(rank_w0, rank_w1, rank_w2);
    if (n_runs > 1) {
        int eff = (n_runs > 32) ? 32 : n_runs;
        printf("  Runs     : %d (mean ± std reported)\n", eff);
        if (n_runs > 32)
            printf("  WARNING  : --runs %d capped to 32\n", n_runs);
    }
    printf("  Weights  : %s\n\n", weights_path);
    } /* end if (g_mpi_rank == 0) */

    /* Set up output paths (rank-suffixed for multi-GPU) */
    {
        const char *od = out_dir_override ? out_dir_override : DEFAULT_OUT_DIR;
        snprintf(OUT_DIR, sizeof(OUT_DIR), "%s", od);
        if (g_mpi_size > 1) {
            snprintf(OUT_CSV, sizeof(OUT_CSV), "%s/benchmark_grayscott_vol_rank%d.csv", od, g_mpi_rank);
            snprintf(OUT_CHUNKS, sizeof(OUT_CHUNKS), "%s/benchmark_grayscott_vol_chunks_rank%d.csv", od, g_mpi_rank);
            snprintf(OUT_TSTEP, sizeof(OUT_TSTEP), "%s/benchmark_grayscott_timesteps_rank%d.csv", od, g_mpi_rank);
            snprintf(OUT_TSTEP_CHUNKS, sizeof(OUT_TSTEP_CHUNKS), "%s/benchmark_grayscott_timestep_chunks_rank%d.csv", od, g_mpi_rank);
            snprintf(OUT_RANKING, sizeof(OUT_RANKING), "%s/benchmark_grayscott_ranking_rank%d.csv", od, g_mpi_rank);
            snprintf(OUT_RANKING_COSTS, sizeof(OUT_RANKING_COSTS), "%s/benchmark_grayscott_ranking_costs_rank%d.csv", od, g_mpi_rank);
        } else {
            snprintf(OUT_CSV, sizeof(OUT_CSV), "%s/benchmark_grayscott_vol.csv", od);
            snprintf(OUT_CHUNKS, sizeof(OUT_CHUNKS), "%s/benchmark_grayscott_vol_chunks.csv", od);
            snprintf(OUT_TSTEP, sizeof(OUT_TSTEP), "%s/benchmark_grayscott_timesteps.csv", od);
            snprintf(OUT_TSTEP_CHUNKS, sizeof(OUT_TSTEP_CHUNKS), "%s/benchmark_grayscott_timestep_chunks.csv", od);
            snprintf(OUT_RANKING, sizeof(OUT_RANKING), "%s/benchmark_grayscott_ranking.csv", od);
            snprintf(OUT_RANKING_COSTS, sizeof(OUT_RANKING_COSTS), "%s/benchmark_grayscott_ranking_costs.csv", od);
        }
    }

    /* Create output directory (recursive) and clear stale chunks CSV */
    {
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "%s", OUT_DIR);
        for (char *p = tmp + 1; *p; p++) {
            if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
        }
        mkdir(tmp, 0755);
    }
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

    /* ── Phase-major: unified loop for ALL phases ─────────────────── */
    /* Each phase: re-create simulation → run all timesteps → record.
     * All phases see identical data (deterministic PDE). */

    float *d_v = NULL, *d_read = NULL;
    PhaseResult results[12];
    int n_phases = 0;
    int any_fail = 0;
    int rc = 0;

    struct Phase {
        const char *name;
        const char *tmp_file;
        int sgd;
        int explore;
        int is_nocomp;
        gpucompress_algorithm_t algo;
        unsigned int preproc;
        unsigned int mask;
    };
    Phase all_phases[] = {
        { "no-comp",    TMP_NOCOMP,     0, 0, 1, (gpucompress_algorithm_t)0, 0, P_NOCOMP },
        { "lz4",        TMP_FIX_LZ4,    0, 0, 0, GPUCOMPRESS_ALGO_LZ4,      0, P_FIX_LZ4 },
        { "snappy",     TMP_FIX_SNAPPY,  0, 0, 0, GPUCOMPRESS_ALGO_SNAPPY,   0, P_FIX_SNAPPY },
        { "deflate",    TMP_FIX_DEFL,    0, 0, 0, GPUCOMPRESS_ALGO_DEFLATE,  0, P_FIX_DEFL },
        { "gdeflate",   TMP_FIX_GDEFL,   0, 0, 0, GPUCOMPRESS_ALGO_GDEFLATE, 0, P_FIX_GDEFL },
        { "zstd",       TMP_FIX_ZSTD,    0, 0, 0, GPUCOMPRESS_ALGO_ZSTD,     0, P_FIX_ZSTD },
        { "ans",        TMP_FIX_ANS,     0, 0, 0, GPUCOMPRESS_ALGO_ANS,      0, P_FIX_ANS },
        { "cascaded",   TMP_FIX_CASC,    0, 0, 0, GPUCOMPRESS_ALGO_CASCADED,  0, P_FIX_CASC },
        { "bitcomp",    TMP_FIX_BITCOMP, 0, 0, 0, GPUCOMPRESS_ALGO_BITCOMP,   0, P_FIX_BITCOMP },
        { "nn",         TMP_NN,          0, 0, 0, GPUCOMPRESS_ALGO_AUTO,     0, P_NN },
        { "nn-rl",      TMP_NN_RL,       1, 0, 0, GPUCOMPRESS_ALGO_AUTO,     0, P_NNRL },
        { "nn-rl+exp50",TMP_NN_RLEXP,    1, 1, 0, GPUCOMPRESS_ALGO_AUTO,     0, P_NNRLEXP },
    };
    int n_all_phases = sizeof(all_phases) / sizeof(all_phases[0]);

    /* CSV files for per-timestep data */
    FILE *ts_csv = NULL, *tc_csv = NULL;
    FILE *ranking_csv = NULL, *ranking_costs_csv = NULL;

    /* Open timestep CSV */
    ts_csv = fopen(OUT_TSTEP, "w");
    if (ts_csv) {
        fprintf(ts_csv, "rank,phase,timestep,sim_step,write_ms,read_ms,ratio,"
                "mape_ratio,mape_comp,mape_decomp,"
                "sgd_fires,explorations,n_chunks,mismatches,"
                "write_mibps,read_mibps,file_bytes,"
                "stats_ms,nn_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                "mae_ratio,mae_comp_ms,mae_decomp_ms,"
                "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
                "vol_s2_busy_ms,vol_s3_busy_ms,"
                "h5dwrite_ms,cuda_sync_ms,h5dclose_ms,h5fclose_ms,"
                "vol_setup_ms,vol_pipeline_ms\n");
    }
    /* Open timestep-chunks CSV */
    tc_csv = fopen(OUT_TSTEP_CHUNKS, "w");
    if (tc_csv) {
        fprintf(tc_csv, "rank,phase,timestep,chunk,action,action_orig,"
                "predicted_ratio,actual_ratio,predicted_comp_ms,actual_comp_ms_raw,"
                "predicted_decomp_ms,actual_decomp_ms_raw,"
                "mape_ratio,mape_comp,mape_decomp,"
                "sgd_fired,exploration_triggered,"
                "cost_model_error_pct,actual_cost,predicted_cost,explore_n_alt");
        for (int i = 0; i < 31; i++)
            fprintf(tc_csv, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d", i, i, i, i);
        fprintf(tc_csv, ",feat_entropy,feat_mad,feat_deriv\n");
    }
    /* Open ranking CSVs */
    ranking_csv = fopen(OUT_RANKING, "w");
    ranking_costs_csv = fopen(OUT_RANKING_COSTS, "w");
    if (ranking_csv)
        write_ranking_csv_header(ranking_csv);
    if (ranking_costs_csv)
        write_ranking_costs_csv_header(ranking_costs_csv);

    hsize_t dims3[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
    double orig_mib = (double)total_bytes / (1 << 20);

    /* ══════════════════════════════════════════════════════════════
     * UNIFIED PHASE LOOP: every phase runs all timesteps
     * ══════════════════════════════════════════════════════════════ */
    for (int pi = 0; pi < n_all_phases; pi++) {
        if (!(phase_mask & all_phases[pi].mask)) continue;

        const char *phase_name = all_phases[pi].name;
        int do_sgd  = all_phases[pi].sgd;
        int do_expl = all_phases[pi].explore;
        bool is_nocomp = all_phases[pi].is_nocomp;
        bool is_nn  = (all_phases[pi].algo == GPUCOMPRESS_ALGO_AUTO && !is_nocomp);

        printf("\n══════════════════════════════════════════════════════════════\n");
        printf("  [%s]: %d timesteps (SGD=%s, Explore=%s)\n",
               phase_name, timesteps,
               do_sgd ? "on" : "off", do_expl ? "on" : "off");
        printf("══════════════════════════════════════════════════════════════\n\n");
        fflush(stdout);

        /* Re-create simulation for fresh start */
        gpucompress_grayscott_destroy(sim);
        sim = NULL;
        gpucompress_grayscott_create(&sim, &s);
        gpucompress_grayscott_init(sim);

        /* Reload NN weights and configure learning */
        gpucompress_reload_nn(weights_path);
        gpucompress_flush_manager_cache();
        if (do_sgd) {
            gpucompress_enable_online_learning();
            gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, sgd_mape);
        } else {
            gpucompress_disable_online_learning();
        }
        gpucompress_set_exploration(do_expl);
        if (do_expl) {
            gpucompress_set_exploration_threshold(explore_thresh);
            gpucompress_set_exploration_k(explore_k);
        }

        /* Build DCPL for this phase */
        hid_t dcpl;
        if (is_nocomp) {
            dcpl = make_dcpl_nocomp(L, chunk_z);
        } else if (all_phases[pi].algo != GPUCOMPRESS_ALGO_AUTO) {
            dcpl = make_dcpl_fixed(L, chunk_z, all_phases[pi].algo, all_phases[pi].preproc);
        } else {
            dcpl = make_dcpl_auto(L, chunk_z, error_bound);
        }

        /* Accumulators */
        double sum_write_ms = 0, sum_read_ms = 0;
        double sum_nn_ms = 0, sum_stats_ms = 0, sum_preproc_ms = 0;
        double sum_comp_ms = 0, sum_decomp_ms = 0, sum_explore_ms = 0, sum_sgd_ms = 0;
        double sum_comp_gbps = 0, sum_decomp_gbps = 0;
        double sum_mae_r = 0, sum_mae_c = 0, sum_mae_d = 0;
        double sum_mape_r = 0, sum_mape_c = 0, sum_mape_d = 0;
        double sum_file_sz = 0;
        double sum_vol_s1 = 0, sum_vol_drain = 0, sum_vol_io_drain = 0;
        double sum_vol_s2_busy = 0, sum_vol_s3_busy = 0;
        size_t last_file_sz = 0;
        int n_steady = 0;
        int cum_sgd = 0, cum_expl = 0;

        printf("  %-4s  %-8s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-4s\n",
               "T", "SimStep", "WrMs", "RdMs", "Ratio",
               "MAPE_R", "MAPE_C", "MAPE_D", "SGD");
        printf("  ----  --------  -------  -------  -------  "
               "--------  --------  --------  ----\n");
        fflush(stdout);

        for (int t = 0; t < timesteps; t++) {
            int cum_sim_step = (t + 1) * steps;

            /* Advance simulation */
            gpucompress_grayscott_run(sim, steps);
            cudaDeviceSynchronize();
            gpucompress_grayscott_get_device_ptrs(sim, &d_read, &d_v);

            /* Write via VOL */
            gpucompress_flush_manager_cache();
            gpucompress_reset_chunk_history();
            H5VL_gpucompress_reset_stats();
            gpucompress_set_debug_context(phase_name, t);
            remove(all_phases[pi].tmp_file);

            hid_t fapl = make_vol_fapl();
            hid_t file = H5Fcreate(all_phases[pi].tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
            H5Pclose(fapl);
            hid_t fsp  = H5Screate_simple(3, dims3, NULL);
            hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                                     fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(fsp);

            double tw0  = now_ms();
            herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                                    H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
            double tw_after_write = now_ms();
            cudaDeviceSynchronize();
            double tw_after_sync = now_ms();
            H5Dclose(dset);
            double tw_after_dclose = now_ms();
            H5Fclose(file);
            double tw1  = now_ms();
            double write_ms_t = tw1 - tw0;

            /* Fine-grained write path breakdown */
            double h5dwrite_ms_t  = tw_after_write - tw0;
            double cuda_sync_ms_t = tw_after_sync - tw_after_write;
            double h5dclose_ms_t  = tw_after_dclose - tw_after_sync;
            double h5fclose_ms_t  = tw1 - tw_after_dclose;

            /* VOL pipeline stage timing */
            double vol_s1 = 0, vol_drain = 0, vol_io_drain = 0, vol_total = 0;
            H5VL_gpucompress_get_stage_timing(&vol_s1, &vol_drain, &vol_io_drain, &vol_total);
            double vol_s2_busy = 0, vol_s3_busy = 0;
            H5VL_gpucompress_get_busy_timing(&vol_s2_busy, &vol_s3_busy);
            double vol_setup = 0;
            H5VL_gpucompress_get_vol_func_timing(&vol_setup, NULL);

            if (wret < 0) { printf("  %-4d  H5Dwrite failed\n", t); continue; }

            /* Always read back for timing; only validate if do_verify */
            drop_pagecache(all_phases[pi].tmp_file);
            fapl = make_vol_fapl();
            file = H5Fopen(all_phases[pi].tmp_file, H5F_ACC_RDONLY, fapl);
            H5Pclose(fapl);
            dset = H5Dopen2(file, "V", H5P_DEFAULT);
            double tr0  = now_ms();
            H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
            cudaDeviceSynchronize();
            H5Dclose(dset); H5Fclose(file);
            double tr1  = now_ms();
            double read_ms_t = tr1 - tr0;

            unsigned long long mm = 0;
            if (do_verify) {
                mm = gpu_compare(d_v, d_read, n_floats, d_count);
            }

            /* Accumulate */
            sum_write_ms += write_ms_t;
            sum_read_ms  += read_ms_t;
            n_steady++;
            size_t file_sz = file_size(all_phases[pi].tmp_file);
            last_file_sz = file_sz;
            double ratio_t = (file_sz > 0) ? (double)total_bytes / (double)file_sz : 1.0;

            /* Per-chunk diagnostics */
            int n_hist = gpucompress_get_chunk_history_count();
            double ts_stats = 0, ts_nn = 0, ts_preproc = 0;
            double ts_comp = 0, ts_decomp = 0, ts_explore = 0, ts_sgd = 0;
            double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0;
            double mae_r_sum = 0, mae_c_sum = 0, mae_d_sum = 0;
            int mcnt = 0, sgd_t = 0, expl_t = 0;
            double pred_sum = 0, act_sum = 0, pred_sq = 0, act_sq = 0, pred_act = 0;

            for (int ci = 0; ci < n_hist; ci++) {
                gpucompress_chunk_diag_t diag;
                if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;

                ts_stats   += diag.stats_ms;
                ts_nn      += diag.nn_inference_ms;
                ts_preproc += diag.preprocessing_ms;
                ts_comp    += diag.compression_ms_raw;
                ts_decomp  += diag.decompression_ms_raw;
                ts_explore += diag.exploration_ms;
                ts_sgd     += diag.sgd_update_ms;
                sgd_t  += diag.sgd_fired;
                expl_t += diag.exploration_triggered;

                if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
                    double ar = diag.actual_ratio, pr = diag.predicted_ratio;
                    mape_r_sum += fabs(pr - ar) / fabs(ar);
                    mae_r_sum  += fabs(pr - ar);
                    pred_sum += pr; act_sum += ar;
                    pred_sq += pr*pr; act_sq += ar*ar; pred_act += pr*ar;
                    mcnt++;
                }
                if (diag.compression_ms > 0 && diag.predicted_comp_time > 0) {
                    mape_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms);
                    mae_c_sum  += fabs(diag.predicted_comp_time - diag.compression_ms_raw);
                }
                if (diag.decompression_ms > 0 && diag.predicted_decomp_time > 0) {
                    mape_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms);
                    mae_d_sum  += fabs(diag.predicted_decomp_time - diag.decompression_ms_raw);
                }
            }

            double real_mape_r = (mcnt > 0) ? 100.0 * mape_r_sum / mcnt : 0;
            double real_mape_c = (mcnt > 0) ? 100.0 * mape_c_sum / mcnt : 0;
            double real_mape_d = (mcnt > 0) ? 100.0 * mape_d_sum / mcnt : 0;
            double ts_mae_r = (mcnt > 0) ? mae_r_sum / mcnt : 0;
            double ts_mae_c = (mcnt > 0) ? mae_c_sum / mcnt : 0;
            double ts_mae_d = (mcnt > 0) ? mae_d_sum / mcnt : 0;
            cum_sgd += sgd_t;
            cum_expl += expl_t;
            sum_stats_ms   += ts_stats;
            sum_nn_ms      += ts_nn;
            sum_preproc_ms += ts_preproc;
            sum_comp_ms    += ts_comp;
            sum_decomp_ms  += ts_decomp;
            sum_explore_ms += ts_explore;
            sum_sgd_ms     += ts_sgd;
            sum_mape_r     += real_mape_r;
            sum_mape_c     += real_mape_c;
            sum_mape_d     += real_mape_d;
            sum_mae_r      += ts_mae_r;
            sum_mae_c      += ts_mae_c;
            sum_mae_d      += ts_mae_d;
            sum_file_sz    += (double)file_sz;
            sum_vol_s1       += vol_s1;
            sum_vol_drain    += vol_drain;
            sum_vol_io_drain += vol_io_drain;
            sum_vol_s2_busy  += vol_s2_busy;
            sum_vol_s3_busy  += vol_s3_busy;

            double wr_mbps = (write_ms_t > 0) ? orig_mib / (write_ms_t / 1000.0) : 0;
            double rd_mbps = (read_ms_t > 0) ? orig_mib / (read_ms_t / 1000.0) : 0;

            if (ts_comp > 0) sum_comp_gbps += (double)total_bytes / 1e9 / (ts_comp / 1000.0);
            if (ts_decomp > 0) sum_decomp_gbps += (double)total_bytes / 1e9 / (ts_decomp / 1000.0);

            /* Print row (flush so stderr VOL warnings don't interleave) */
            printf("  %-4d  %-8d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %3d\n",
                   t, cum_sim_step, write_ms_t, read_ms_t, ratio_t,
                   real_mape_r, real_mape_c, real_mape_d, sgd_t);
            fflush(stdout);

            /* Write timestep CSV row */
            if (ts_csv) {
                fprintf(ts_csv,
                    "%d,%s,%d,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d,%llu,%.1f,%.1f,"
                    "%zu,"
                    "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                    "%.4f,%.4f,%.4f,"
                    "%.3f,%.3f,%.3f,"
                    "%.3f,%.3f,"
                    "%.3f,%.3f,%.3f,%.3f,"
                    "%.3f,%.3f\n",
                    g_mpi_rank, phase_name, t, cum_sim_step, write_ms_t, read_ms_t, ratio_t,
                    real_mape_r, real_mape_c, real_mape_d,
                    sgd_t, expl_t, n_hist, (unsigned long long)mm, wr_mbps, rd_mbps,
                    file_sz,
                    ts_stats, ts_nn, ts_preproc, ts_comp, ts_decomp, ts_explore, ts_sgd,
                    ts_mae_r, ts_mae_c, ts_mae_d,
                    vol_s1, vol_drain, vol_io_drain,
                    vol_s2_busy, vol_s3_busy,
                    h5dwrite_ms_t, cuda_sync_ms_t, h5dclose_ms_t, h5fclose_ms_t,
                    vol_setup, vol_total);
                fflush(ts_csv);
            }

            /* Write per-chunk milestone CSV */
            if (tc_csv && (t == 0 || t == timesteps/4 || t == timesteps/2 ||
                           t == 3*timesteps/4 || t == timesteps - 1)) {
                for (int ci = 0; ci < n_hist; ci++) {
                    gpucompress_chunk_diag_t dd;
                    if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;
                    char act_str[40], orig_str[40];
                    action_to_str(dd.nn_action, act_str, sizeof(act_str));
                    action_to_str(dd.nn_original_action, orig_str, sizeof(orig_str));
                    fprintf(tc_csv, "%d,%s,%d,%d,%s,%s,"
                            "%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,"
                            "%.1f,%.1f,%.1f,%d,%d,"
                            "%.4f,%.4f,%.4f,%d",
                            g_mpi_rank, phase_name, t, ci, act_str, orig_str,
                            dd.predicted_ratio, dd.actual_ratio,
                            dd.predicted_comp_time, dd.compression_ms_raw,
                            dd.predicted_decomp_time, dd.decompression_ms_raw,
                            (dd.actual_ratio > 0 && dd.predicted_ratio > 0) ?
                                100.0 * fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) : 0.0,
                            (dd.compression_ms > 0 && dd.predicted_comp_time > 0) ?
                                100.0 * fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) : 0.0,
                            (dd.decompression_ms > 0 && dd.predicted_decomp_time > 0) ?
                                100.0 * fabs(dd.predicted_decomp_time - dd.decompression_ms) / fabs(dd.decompression_ms) : 0.0,
                            dd.sgd_fired, dd.exploration_triggered,
                            dd.cost_model_error_pct, dd.actual_cost, dd.predicted_cost,
                            dd.explore_n_alternatives);
                    for (int ei = 0; ei < 31; ei++)
                        fprintf(tc_csv, ",%d,%.4f,%.3f,%.4f",
                                dd.explore_alternatives[ei], dd.explore_ratios[ei],
                                dd.explore_comp_ms[ei], dd.explore_costs[ei]);
                    fprintf(tc_csv, ",%.4f,%.6f,%.6f\n",
                            dd.feat_entropy, dd.feat_mad, dd.feat_deriv);
                }
                fflush(tc_csv);
            }

            /* Kendall tau ranking at milestones (NN phases only) */
            if (is_nn && ranking_csv && is_ranking_milestone(t, timesteps)) {
                float bw = gpucompress_get_bandwidth_bytes_per_ms();
                RankingMilestoneResult tau_result = {};
                run_ranking_profiler(
                    d_v, total_bytes, (size_t)chunk_z * L * L * sizeof(float),
                    error_bound,
                    rank_w0, rank_w1, rank_w2, bw,
                    3, ranking_csv, ranking_costs_csv,
                    phase_name, t, &tau_result);
                printf("    [τ] T=%d: τ=%.3f  regret=%.3fx  (%.0fms)\n",
                       t, tau_result.mean_tau,
                       tau_result.mean_regret,
                       tau_result.profiling_ms);
            }

            remove(all_phases[pi].tmp_file);
        } /* end timestep loop */

        /* Compute averages → store in PhaseResult */
        int n = n_steady;
        if (n > 0) {
            PhaseResult &r = results[n_phases];
            memset(&r, 0, sizeof(r));
            snprintf(r.phase, sizeof(r.phase), "%s", phase_name);
            r.write_ms   = sum_write_ms / n;
            r.read_ms    = sum_read_ms / n;
            r.orig_bytes = total_bytes;
            r.file_bytes = (sum_file_sz > 0) ? (size_t)(sum_file_sz / n) : last_file_sz;
            r.ratio      = (r.file_bytes > 0) ? (double)total_bytes / (double)r.file_bytes : 1.0;
            r.write_mbps = (r.write_ms > 0) ? orig_mib / (r.write_ms / 1000.0) : 0;
            r.read_mbps  = (r.read_ms > 0) ? orig_mib / (r.read_ms / 1000.0) : 0;
            r.mismatches = 0;
            r.sgd_fires  = cum_sgd;
            r.explorations = cum_expl;
            r.n_chunks   = gpucompress_get_chunk_history_count();
            r.stats_ms   = sum_stats_ms / n;
            r.nn_ms      = sum_nn_ms / n;
            r.preproc_ms = sum_preproc_ms / n;
            r.comp_ms    = sum_comp_ms / n;
            r.decomp_ms  = sum_decomp_ms / n;
            r.explore_ms = sum_explore_ms / n;
            r.sgd_ms     = sum_sgd_ms / n;
            r.comp_gbps  = sum_comp_gbps / n;
            r.decomp_gbps = sum_decomp_gbps / n;
            r.mape_ratio_pct = sum_mape_r / n;
            r.mape_comp_pct  = sum_mape_c / n;
            r.mape_decomp_pct = sum_mape_d / n;
            r.mae_ratio  = sum_mae_r / n;
            r.mae_comp_ms = sum_mae_c / n;
            r.mae_decomp_ms = sum_mae_d / n;
            r.stage1_ms   = sum_vol_s1 / n;
            r.drain_ms    = sum_vol_drain / n;
            r.io_drain_ms = sum_vol_io_drain / n;
            r.s2_busy_ms  = sum_vol_s2_busy / n;
            r.s3_busy_ms  = sum_vol_s3_busy / n;
            r.sim_ms     = (double)timesteps * steps;  /* placeholder */
            r.n_runs     = n;
        }
        n_phases++;

        H5Pclose(dcpl);
        gpucompress_disable_online_learning();

        printf("\n  [%s] Complete: %d timesteps, ratio=%.2fx, write=%.0f MiB/s\n",
               phase_name, n_steady,
               results[n_phases - 1].ratio, results[n_phases - 1].write_mbps);
    } /* end phase loop */

    /* Close CSV files */
    if (ts_csv) { fclose(ts_csv); printf("\n  Timestep CSV: %s\n", OUT_TSTEP); }
    if (tc_csv) { fclose(tc_csv); printf("  Timestep chunks CSV: %s\n", OUT_TSTEP_CHUNKS); }
    if (ranking_csv) { fclose(ranking_csv); printf("  Ranking CSV: %s\n", OUT_RANKING); }
    if (ranking_costs_csv) { fclose(ranking_costs_csv); printf("  Ranking costs CSV: %s\n", OUT_RANKING_COSTS); }

/* ── OLD CODE REMOVED: run_single_shot_multi_ts, per-phase single-shot, multi-ts nn-rl loop ── */
#if 0

    auto run_single_shot_multi_ts = [&](const char *phase_name,
                                         const char *tmp_file,
                                         hid_t dcpl,
                                         int is_nocomp,
                                         PhaseResult *out) -> int
    {
        /* Re-create simulation for fresh start */
        gpucompress_grayscott_destroy(sim);
        sim = NULL;
        gpucompress_grayscott_create(&sim, &s);
        gpucompress_grayscott_init(sim);

        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);

        /* Store per-timestep values for std computation */
        const int MAX_TS = 1024;
        double *ts_write = (double*)calloc(MAX_TS, sizeof(double));
        double *ts_read  = (double*)calloc(MAX_TS, sizeof(double));
        double *ts_cgbps = (double*)calloc(MAX_TS, sizeof(double));
        double *ts_dgbps = (double*)calloc(MAX_TS, sizeof(double));

        double sum_ratio = 0, sum_write_ms = 0, sum_read_ms = 0;
        double sum_nn_ms = 0, sum_stats_ms = 0, sum_preproc_ms = 0;
        double sum_comp_ms = 0, sum_decomp_ms = 0, sum_explore_ms = 0, sum_sgd_ms = 0;
        double sum_comp_gbps = 0, sum_decomp_gbps = 0;
        size_t sum_file_bytes = 0;
        int sum_sgd_fires = 0, sum_explorations = 0;
        int count = 0;
        int fail = 0;

        for (int t = 0; t < timesteps && t < MAX_TS; t++) {
            gpucompress_grayscott_run(sim, steps);
            cudaDeviceSynchronize();
            gpucompress_grayscott_get_device_ptrs(sim, &d_read, &d_v);

            PhaseResult tr;
            int trc;
            if (is_nocomp) {
                trc = run_phase_nocomp(d_v, d_read, d_count,
                                       n_floats, L, chunk_z, &tr);
            } else {
                trc = run_phase_vol(d_v, d_read, d_count,
                                    n_floats, L, chunk_z,
                                    phase_name, tmp_file, dcpl, &tr);
            }
            if (trc) fail = 1;

            printf("  timestep %d/%d: ratio=%.2fx  write=%.0f MiB/s\n",
                   t + 1, timesteps, tr.ratio, tr.write_mbps);

            ts_write[count] = tr.write_ms;
            ts_read[count]  = tr.read_ms;
            ts_cgbps[count] = tr.comp_gbps;
            ts_dgbps[count] = tr.decomp_gbps;

            sum_ratio       += tr.ratio;
            sum_write_ms    += tr.write_ms;
            sum_read_ms     += tr.read_ms;
            sum_file_bytes  += tr.file_bytes;
            sum_nn_ms       += tr.nn_ms;
            sum_stats_ms    += tr.stats_ms;
            sum_preproc_ms  += tr.preproc_ms;
            sum_comp_ms     += tr.comp_ms;
            sum_decomp_ms   += tr.decomp_ms;
            sum_explore_ms  += tr.explore_ms;
            sum_sgd_ms      += tr.sgd_ms;
            sum_comp_gbps   += tr.comp_gbps;
            sum_decomp_gbps += tr.decomp_gbps;
            sum_sgd_fires   += tr.sgd_fires;
            sum_explorations += tr.explorations;
            count++;
        }

        if (count > 0) {
            memset(out, 0, sizeof(*out));
            snprintf(out->phase, sizeof(out->phase), "%s", phase_name);
            out->file_bytes  = sum_file_bytes / count;
            out->ratio       = (sum_file_bytes > 0)
                ? (double)(count * total_bytes) / (double)sum_file_bytes : 1.0;
            out->write_ms    = sum_write_ms / count;
            out->read_ms     = sum_read_ms / count;
            out->orig_bytes  = total_bytes;
            out->write_mbps  = (double)total_bytes / (1 << 20) / (out->write_ms / 1000.0);
            out->read_mbps   = (double)total_bytes / (1 << 20) / (out->read_ms / 1000.0);
            out->nn_ms       = sum_nn_ms / count;
            out->stats_ms    = sum_stats_ms / count;
            out->preproc_ms  = sum_preproc_ms / count;
            out->comp_ms     = sum_comp_ms / count;
            out->decomp_ms   = sum_decomp_ms / count;
            out->explore_ms  = sum_explore_ms / count;
            out->sgd_ms      = sum_sgd_ms / count;
            out->comp_gbps   = sum_comp_gbps / count;
            out->decomp_gbps = sum_decomp_gbps / count;
            out->sgd_fires   = sum_sgd_fires;
            out->explorations = sum_explorations;
            out->n_runs      = count;
            out->n_chunks    = n_chunks;

            /* Compute std across timesteps (sample std, n-1) */
            double var_w = 0, var_r = 0, var_cg = 0, var_dg = 0;
            for (int i = 0; i < count; i++) {
                var_w  += (ts_write[i] - out->write_ms) * (ts_write[i] - out->write_ms);
                var_r  += (ts_read[i]  - out->read_ms)  * (ts_read[i]  - out->read_ms);
                var_cg += (ts_cgbps[i] - out->comp_gbps) * (ts_cgbps[i] - out->comp_gbps);
                var_dg += (ts_dgbps[i] - out->decomp_gbps) * (ts_dgbps[i] - out->decomp_gbps);
            }
            double denom = (count > 1) ? (count - 1) : 1;
            out->write_ms_std    = sqrt(var_w / denom);
            out->read_ms_std     = sqrt(var_r / denom);
            out->comp_gbps_std   = sqrt(var_cg / denom);
            out->decomp_gbps_std = sqrt(var_dg / denom);
        }

        free(ts_write); free(ts_read); free(ts_cgbps); free(ts_dgbps);
        return fail;
    };

    /* ── Phase 1: no-comp ──────────────────────────────────────────── */
    if (phase_mask & P_NOCOMP) {
        printf("\n── Phase %d: no-comp (GPU->Host->HDF5) ────────────────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        if (timesteps > 1) {
            printf("  Running across %d timesteps for fair comparison...\n", timesteps);
            rc = run_single_shot_multi_ts("no-comp", NULL, H5I_INVALID_HID, 1, &results[n_phases]);
            if (rc) any_fail = 1;
        } else {
            PhaseResult runs_buf[32];
            int eff_runs = (n_runs > 32) ? 32 : n_runs;
            for (int run = 0; run < eff_runs; run++) {
                if (eff_runs > 1) printf("  Run %d/%d... ", run + 1, eff_runs);
                else printf("  Writing... ");
                fflush(stdout);
                double t0 = now_ms();
                rc = run_phase_nocomp(d_v, d_read, d_count,
                                          n_floats, L, chunk_z,
                                          &runs_buf[run]);
                printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
                runs_buf[run].sim_ms = sim_ms;
                if (rc) any_fail = 1;
            }
            if (eff_runs > 1)
                merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
            else
                results[n_phases] = runs_buf[0];
            results[n_phases].n_runs = eff_runs;
        }
        n_phases++;
    }

    /* ── Phases: fixed-algo baselines ── */
    struct FixedAlgoPhase {
        unsigned int mask;
        const char *name;
        const char *tmp_file;
        gpucompress_algorithm_t algo;
        unsigned int preproc;
    };
    FixedAlgoPhase fixed_phases[] = {
        { P_FIX_LZ4,      "lz4",           TMP_FIX_LZ4,             GPUCOMPRESS_ALGO_LZ4,      0 },
        { P_FIX_SNAPPY,   "snappy",         TMP_FIX_SNAPPY,          GPUCOMPRESS_ALGO_SNAPPY,   0 },
        { P_FIX_DEFL,     "deflate",        TMP_FIX_DEFL,            GPUCOMPRESS_ALGO_DEFLATE,  0 },
        { P_FIX_GDEFL,    "gdeflate",       TMP_FIX_GDEFL,           GPUCOMPRESS_ALGO_GDEFLATE, 0 },
        { P_FIX_ZSTD,     "zstd",           TMP_FIX_ZSTD,            GPUCOMPRESS_ALGO_ZSTD,     0 },
        { P_FIX_ANS,      "ans",            TMP_FIX_ANS,             GPUCOMPRESS_ALGO_ANS,      0 },
        { P_FIX_CASC,     "cascaded",       TMP_FIX_CASC,            GPUCOMPRESS_ALGO_CASCADED, 0 },
        { P_FIX_BITCOMP,  "bitcomp",        TMP_FIX_BITCOMP,         GPUCOMPRESS_ALGO_BITCOMP,  0 },
    };
    for (int fi = 0; fi < 8; fi++) {
        if (!(phase_mask & fixed_phases[fi].mask)) continue;
        printf("\n── Phase %d: %s ─────────────────────────────────────\n",
               n_phases + 1, fixed_phases[fi].name);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_f = make_dcpl_fixed(L, chunk_z, fixed_phases[fi].algo, fixed_phases[fi].preproc);
        if (timesteps > 1) {
            printf("  Running across %d timesteps for fair comparison...\n", timesteps);
            rc = run_single_shot_multi_ts(fixed_phases[fi].name, fixed_phases[fi].tmp_file,
                                           dcpl_f, 0, &results[n_phases]);
            if (rc) any_fail = 1;
        } else {
            PhaseResult runs_buf[32];
            int eff_runs = (n_runs > 32) ? 32 : n_runs;
            for (int run = 0; run < eff_runs; run++) {
                if (eff_runs > 1) printf("  Run %d/%d... ", run + 1, eff_runs);
                else printf("  Write + Read + Verify... ");
                fflush(stdout);
                double t0 = now_ms();
                rc = run_phase_vol(d_v, d_read, d_count,
                                   n_floats, L, chunk_z,
                                   fixed_phases[fi].name, fixed_phases[fi].tmp_file,
                                   dcpl_f, &runs_buf[run]);
                printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
                runs_buf[run].sim_ms = sim_ms;
                if (rc) any_fail = 1;
            }
            if (eff_runs > 1)
                merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
            else
                results[n_phases] = runs_buf[0];
            results[n_phases].n_runs = eff_runs;
        }
        H5Pclose(dcpl_f);
        n_phases++;
    }

    /* ── Phase: nn (VOL, ALGO_AUTO, inference-only) ────────────── */
    if (phase_mask & P_NN) {
        printf("\n── Phase %d: nn (VOL, ALGO_AUTO, inference-only) ────────────\n", n_phases + 1);
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);
        hid_t dcpl_nn = make_dcpl_auto(L, chunk_z, error_bound);
        if (timesteps > 1) {
            printf("  Running across %d timesteps for fair comparison...\n", timesteps);
            rc = run_single_shot_multi_ts("nn", TMP_NN, dcpl_nn, 0, &results[n_phases]);
            if (rc) any_fail = 1;
        } else {
            PhaseResult runs_buf[32];
            int eff_runs = (n_runs > 32) ? 32 : n_runs;
            for (int run = 0; run < eff_runs; run++) {
                if (eff_runs > 1) printf("  Run %d/%d... ", run + 1, eff_runs);
                else printf("  Write + Read + Verify... ");
                fflush(stdout);
                double t0 = now_ms();
                rc = run_phase_vol(d_v, d_read, d_count,
                                   n_floats, L, chunk_z,
                                   "nn", TMP_NN, dcpl_nn,
                                   &runs_buf[run]);
                printf("done (%.1fs)\n", (now_ms() - t0) / 1000.0);
                runs_buf[run].sim_ms = sim_ms;
                if (rc) any_fail = 1;
            }
            if (eff_runs > 1)
                merge_phase_results(runs_buf, eff_runs, &results[n_phases]);
            else
                results[n_phases] = runs_buf[0];
            results[n_phases].n_runs = eff_runs;
        }
        H5Pclose(dcpl_nn);
        write_chunk_csv("nn", n_chunks);  /* last run's chunks */
        n_phases++;
    }

    /* ── Multi-timestep mode (nn-rl, nn-rl+exp50) ───────────────── */
    if (timesteps > 1) {

        /* Phase configs: name, sgd_enabled, exploration_enabled */
        struct TsPhase {
            const char *name;
            int sgd;
            int explore;
            unsigned int mask;
        };
        TsPhase ts_phases_all[] = {
            { "nn-rl",       1, 0, P_NNRL },
            { "nn-rl+exp50", 1, 1, P_NNRLEXP },
        };
        /* Only run multi-timestep for phases selected via --phase mask */
        TsPhase ts_phases[2];
        int n_ts_phases = 0;
        for (int i = 0; i < 2; i++) {
            if (phase_mask & ts_phases_all[i].mask)
                ts_phases[n_ts_phases++] = ts_phases_all[i];
        }

        hid_t dcpl_ts = make_dcpl_auto(L, chunk_z, error_bound);
        hsize_t dims3[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

        /* Open timestep CSV (shared across phases) */
        FILE *ts_csv = fopen(OUT_TSTEP, "w");
        if (ts_csv) {
            fprintf(ts_csv, "rank,phase,timestep,sim_step,write_ms,read_ms,ratio,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fires,explorations,n_chunks,mismatches,"
                    "write_mibps,read_mibps,"
                    "file_bytes,"
                    "stats_ms,nn_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                    "mae_ratio,mae_comp_ms,mae_decomp_ms\n");
        }

        /* Open per-chunk milestone CSV (shared across phases) */
        FILE *tc_csv = fopen(OUT_TSTEP_CHUNKS, "w");
        if (tc_csv) {
            fprintf(tc_csv, "rank,phase,timestep,chunk,action,action_orig,"
                    "predicted_ratio,actual_ratio,"
                    "predicted_comp_ms,actual_comp_ms_raw,"
                    "predicted_decomp_ms,actual_decomp_ms_raw,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fired,exploration_triggered,"
                    "cost_model_error_pct,actual_cost,predicted_cost,"
                    "explore_n_alt");
            for (int ei = 0; ei < 31; ei++)
                fprintf(tc_csv, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d",
                        ei, ei, ei, ei);
            fprintf(tc_csv, ",feat_entropy,feat_mad,feat_deriv\n");
        }

        /* Open ranking quality CSVs (shared across phases) */
        FILE *ranking_csv = fopen(OUT_RANKING, "w");
        if (ranking_csv)
            write_ranking_csv_header(ranking_csv);
        FILE *ranking_costs_csv = fopen(OUT_RANKING_COSTS, "w");
        if (ranking_costs_csv)
            write_ranking_costs_csv_header(ranking_costs_csv);

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
            gpucompress_grayscott_init(sim);

            gpucompress_reload_nn(weights_path);
            gpucompress_flush_manager_cache();
            if (do_sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, sgd_mape);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(do_expl);
            if (do_expl) {
                gpucompress_set_exploration_threshold(explore_thresh);
                gpucompress_set_exploration_k(explore_k);
            }

            printf("  %-4s  %-8s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-4s  %-4s\n",
                   "T", "SimStep", "WrMs", "RdMs", "Ratio",
                   "MAPE_R", "MAPE_C", "MAPE_D", "SGD", "EXP");
            printf("  ----  --------  -------  -------  -------  "
                   "--------  --------  --------  ----  ----\n");

            /* Accumulators for averaged throughput */
            const int WARMUP_SKIP = 0;
            double sum_write_ms = 0, sum_read_ms = 0;
            double sum_nn_ms = 0, sum_stats_ms = 0, sum_preproc_ms = 0;
            double sum_comp_ms = 0, sum_decomp_ms = 0, sum_explore_ms = 0, sum_sgd_ms = 0;
            double sum_comp_gbps = 0, sum_decomp_gbps = 0;
            double sum_mae_r = 0, sum_mae_c = 0, sum_mae_d = 0;
            double sum_mape_r = 0, sum_mape_c = 0, sum_mape_d = 0;
            double sum_file_sz = 0;
            double sum_vol_s1 = 0, sum_vol_drain = 0, sum_vol_io_drain = 0;
            double sum_vol_s2_busy = 0, sum_vol_s3_busy = 0;
            size_t last_file_sz = 0;
            int    n_steady = 0;
            /* Per-timestep MAPE (tracked for console output) */
            double latest_mape_r = 0, latest_mape_c = 0, latest_mape_d = 0;
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
                gpucompress_flush_manager_cache();  /* cold-start each timestep, matching single-shot phases */
                gpucompress_reset_chunk_history();
                gpucompress_set_debug_context(phase_name, t);
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
                cudaDeviceSynchronize();
                H5Dclose(dset); H5Fclose(file);
                double tw1  = now_ms();
                double write_ms_t = tw1 - tw0;

                /* VOL pipeline stage timing */
                double vol_s1 = 0, vol_drain = 0, vol_io_drain = 0, vol_total = 0;
                H5VL_gpucompress_get_stage_timing(&vol_s1, &vol_drain, &vol_io_drain, &vol_total);
                double vol_s2_busy = 0, vol_s3_busy = 0;
                H5VL_gpucompress_get_busy_timing(&vol_s2_busy, &vol_s3_busy);

                if (wret < 0) {
                    printf("  %-4d  H5Dwrite failed\n", t);
                    continue;
                }

                drop_pagecache(TMP_NN_RL);

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
                last_file_sz = file_sz;
                double ratio_t = (file_sz > 0) ? (double)total_bytes / (double)file_sz : 1.0;

                /* Collect per-chunk MAPE */
                int n_hist = gpucompress_get_chunk_history_count();
                double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0;
                int    mcnt_r = 0, mcnt_c = 0, mcnt_d = 0;
                int    sgd_t = 0, expl_t = 0;
                double ts_nn_ms = 0, ts_stats_ms = 0, ts_preproc_ms = 0;
                double ts_comp_ms = 0, ts_decomp_ms = 0, ts_explore_ms = 0, ts_sgd_ms = 0;
                double ts_mae_r_sum = 0, ts_mae_c_sum = 0, ts_mae_d_sum = 0;
                int    mae_r_cnt = 0;
                for (int ci = 0; ci < n_hist; ci++) {
                    gpucompress_chunk_diag_t diag;
                    if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
                    if (diag.sgd_fired) sgd_t++;
                    if (diag.exploration_triggered) expl_t++;
                    ts_nn_ms      += diag.nn_inference_ms;
                    ts_stats_ms   += diag.stats_ms;
                    ts_preproc_ms += diag.preprocessing_ms;
                    ts_comp_ms    += diag.compression_ms_raw;  /* unclamped for breakdown */
                    ts_decomp_ms  += diag.decompression_ms_raw;  /* unclamped */
                    ts_explore_ms += diag.exploration_ms;
                    ts_sgd_ms     += diag.sgd_update_ms;
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
                    /* MAE accumulators */
                    if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
                        ts_mae_r_sum += fabs(diag.predicted_ratio - diag.actual_ratio);
                        mae_r_cnt++;
                    }
                    if (diag.compression_ms_raw > 0)
                        ts_mae_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms);
                    if (diag.decompression_ms_raw > 0)
                        ts_mae_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms);
                }
                double real_mape_r = fmin(200.0, mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0);
                double real_mape_c = fmin(200.0, mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0);
                double real_mape_d = fmin(200.0, mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0);
                /* Per-timestep MAE */
                double ts_mae_r = mae_r_cnt ? ts_mae_r_sum / mae_r_cnt : 0.0;
                double ts_mae_c = mcnt_c ? ts_mae_c_sum / mcnt_c : 0.0;
                double ts_mae_d = mcnt_d ? ts_mae_d_sum / mcnt_d : 0.0;

                /* Track latest timestep values for console output */
                latest_mape_r = real_mape_r;
                latest_mape_c = real_mape_c;
                latest_mape_d = real_mape_d;
                final_sgd    = sgd_t;
                final_expl   = expl_t;
                cum_sgd     += sgd_t;
                cum_expl    += expl_t;

                /* Accumulate per-component timing, file size, and MAE (skip warmup) */
                if (t >= WARMUP_SKIP) {
                    sum_file_sz     += (double)file_sz;
                    sum_nn_ms       += ts_nn_ms;
                    sum_stats_ms    += ts_stats_ms;
                    sum_preproc_ms  += ts_preproc_ms;
                    sum_comp_ms     += ts_comp_ms;
                    sum_decomp_ms   += ts_decomp_ms;
                    sum_explore_ms  += ts_explore_ms;
                    sum_sgd_ms      += ts_sgd_ms;
                    sum_mape_r      += real_mape_r;
                    sum_mape_c      += real_mape_c;
                    sum_mape_d      += real_mape_d;
                    sum_mae_r       += ts_mae_r;
                    sum_mae_c       += ts_mae_c;
                    sum_mae_d       += ts_mae_d;
                    sum_vol_s1       += vol_s1;
                    sum_vol_drain    += vol_drain;
                    sum_vol_io_drain += vol_io_drain;
                    sum_vol_s2_busy  += vol_s2_busy;
                    sum_vol_s3_busy  += vol_s3_busy;
                    double ts_comp_gbps = (ts_comp_ms > 0)
                        ? (double)total_bytes / ts_comp_ms / 1e6 : 0.0;
                    double ts_decomp_gbps = (ts_decomp_ms > 0)
                        ? (double)total_bytes / ts_decomp_ms / 1e6 : 0.0;
                    sum_comp_gbps   += ts_comp_gbps;
                    sum_decomp_gbps += ts_decomp_gbps;
                }

                double wr_mbps = (write_ms_t > 0) ? dataset_mb / (write_ms_t / 1000.0) : 0;
                double rd_mbps = (read_ms_t > 0)  ? dataset_mb / (read_ms_t  / 1000.0) : 0;

                /* Print summary row every 5 timesteps, first and last */
                bool print_row = (t % 5 == 0 || t == timesteps - 1);
                if (print_row) {
                    printf("\r%80s\r", "");  /* clear progress bar */
                    printf("  %-4d  %-8d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %3d  %3d\n",
                           t, cum_sim_step, write_ms_t, read_ms_t,
                           ratio_t, real_mape_r, real_mape_c, real_mape_d, sgd_t, expl_t);
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

                            double mr = 0, mc = 0, md = 0;
                            if (dd.actual_ratio > 0)
                                mr = fmin(200.0, fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) * 100.0);
                            if (dd.compression_ms > 0)
                                mc = fmin(200.0, fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) * 100.0);
                            if (dd.decompression_ms > 0)
                                md = fmin(200.0, fabs(dd.predicted_decomp_time - dd.decompression_ms) / fabs(dd.decompression_ms) * 100.0);

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
                                char action_str[40], orig_str[40];
                                action_to_str(dd.nn_action, action_str, sizeof(action_str));
                                action_to_str(dd.nn_original_action, orig_str, sizeof(orig_str));
                                fprintf(tc_csv, "%d,%s,%d,%d,%s,%s,"
                                        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                                        "%.2f,%.2f,%.2f,%d,%d,"
                                        "%.4f,%.4f,%.4f,%d",
                                        g_mpi_rank, phase_name, t, ci, action_str, orig_str,
                                        (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                        (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                                        (double)dd.predicted_decomp_time, (double)dd.decompression_ms_raw,
                                        mr, mc, md,
                                        dd.sgd_fired, dd.exploration_triggered,
                                        (double)dd.cost_model_error_pct,
                                        (double)dd.actual_cost, (double)dd.predicted_cost,
                                        dd.explore_n_alternatives);
                                /* Exploration alternatives (31 slots) */
                                for (int ei = 0; ei < 31; ei++) {
                                    if (ei < dd.explore_n_alternatives) {
                                        char alt_str[40];
                                        action_to_str(dd.explore_alternatives[ei], alt_str, sizeof(alt_str));
                                        fprintf(tc_csv, ",%s,%.4f,%.4f,%.4f",
                                                alt_str,
                                                (double)dd.explore_ratios[ei],
                                                (double)dd.explore_comp_ms[ei],
                                                (double)dd.explore_costs[ei]);
                                    } else {
                                        fprintf(tc_csv, ",,,,");
                                    }
                                }
                                /* Features */
                                fprintf(tc_csv, ",%.4f,%.6f,%.6f\n",
                                        (double)dd.feat_entropy,
                                        (double)dd.feat_mad,
                                        (double)dd.feat_deriv);
                            }
                        }
                        if (verbose_chunks)
                            printf("    ── end T=%d ──\n", t);
                    }
                }

                /* ── Kendall τ ranking quality at milestones ── */
                if (ranking_csv && is_ranking_milestone(t, timesteps)) {
                    size_t gs_chunk_bytes = (size_t)L * L * chunk_z * sizeof(float);
                    size_t gs_total_bytes = (size_t)L * L * L * sizeof(float);
                    float bw = gpucompress_get_bandwidth_bytes_per_ms();
                    RankingMilestoneResult tau_result = {};
                    run_ranking_profiler(
                        d_v, gs_total_bytes, gs_chunk_bytes,
                        error_bound, rank_w0, rank_w1, rank_w2, bw,
                        3, ranking_csv, ranking_costs_csv,
                        phase_name, t, &tau_result);
                    printf("    [τ] T=%d: τ=%.3f  regret=%.3fx  (%.0fms)\n",
                           t, tau_result.mean_tau,
                           tau_result.mean_regret,
                           tau_result.profiling_ms);
                }

                if (ts_csv) {
                    fprintf(ts_csv, "%d,%s,%d,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d,%llu,%.1f,%.1f,%zu,"
                            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                            "%.4f,%.4f,%.4f\n",
                            g_mpi_rank, phase_name, t, cum_sim_step, write_ms_t, read_ms_t, ratio_t,
                            real_mape_r, real_mape_c, real_mape_d,
                            sgd_t, expl_t, n_hist,
                            (unsigned long long)mm, wr_mbps, rd_mbps,
                            file_sz,
                            ts_stats_ms, ts_nn_ms, ts_preproc_ms,
                            ts_comp_ms, ts_decomp_ms, ts_explore_ms, ts_sgd_ms,
                            ts_mae_r, ts_mae_c, ts_mae_d);
                }

                remove(TMP_NN_RL);
            } /* end timestep loop */

            /* Final progress bar */
            {
                printf("\r  [");
                for (int b = 0; b < 30; b++) putchar('#');
                printf("] %d/%d              \n\n", timesteps, timesteps);
            }

            /* Store multi-timestep results: create a new PhaseResult entry */
            {
                PhaseResult *pr = &results[n_phases];
                memset(pr, 0, sizeof(*pr));
                snprintf(pr->phase, sizeof(pr->phase), "%s", phase_name);
                pr->orig_bytes = total_bytes;
                pr->n_chunks   = n_chunks;
                pr->n_runs     = timesteps;
                if (n_steady > 0) {
                    double avg_wr = sum_write_ms / n_steady;
                    double avg_rd = sum_read_ms  / n_steady;
                    pr->write_ms    = avg_wr;
                    pr->read_ms     = avg_rd;
                    pr->write_mbps  = dataset_mb / (avg_wr / 1000.0);
                    pr->read_mbps   = dataset_mb / (avg_rd / 1000.0);
                    double avg_file_sz = sum_file_sz / n_steady;
                    pr->ratio       = (avg_file_sz > 0) ? (double)total_bytes / avg_file_sz : 1.0;
                    pr->file_bytes  = (size_t)avg_file_sz;
                    pr->nn_ms       = sum_nn_ms / n_steady;
                    pr->stats_ms    = sum_stats_ms / n_steady;
                    pr->preproc_ms  = sum_preproc_ms / n_steady;
                    pr->comp_ms     = sum_comp_ms / n_steady;
                    pr->decomp_ms   = sum_decomp_ms / n_steady;
                    pr->explore_ms  = sum_explore_ms / n_steady;
                    pr->sgd_ms      = sum_sgd_ms / n_steady;
                    pr->comp_gbps   = sum_comp_gbps / n_steady;
                    pr->decomp_gbps = sum_decomp_gbps / n_steady;
                }
                pr->mape_ratio_pct   = n_steady > 0 ? sum_mape_r / n_steady : latest_mape_r;
                pr->mape_comp_pct    = n_steady > 0 ? sum_mape_c / n_steady : latest_mape_c;
                pr->mape_decomp_pct  = n_steady > 0 ? sum_mape_d / n_steady : latest_mape_d;
                if (n_steady > 0) {
                    pr->mae_ratio     = sum_mae_r / n_steady;
                    pr->mae_comp_ms   = sum_mae_c / n_steady;
                    pr->mae_decomp_ms = sum_mae_d / n_steady;
                    pr->stage1_ms     = sum_vol_s1 / n_steady;
                    pr->drain_ms      = sum_vol_drain / n_steady;
                    pr->io_drain_ms   = sum_vol_io_drain / n_steady;
                    pr->s2_busy_ms    = sum_vol_s2_busy / n_steady;
                    pr->s3_busy_ms    = sum_vol_s3_busy / n_steady;
                }
                pr->sgd_fires   = cum_sgd;
                pr->explorations = cum_expl;
                n_phases++;
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
        if (ranking_csv) {
            fclose(ranking_csv);
            printf("  Ranking quality CSV: %s\n", OUT_RANKING);
        }
        if (ranking_costs_csv) {
            fclose(ranking_costs_csv);
            printf("  Ranking costs CSV: %s\n", OUT_RANKING_COSTS);
        }

        gpucompress_disable_online_learning();
        H5Pclose(dcpl_ts);
    }

#endif /* OLD CODE */

    /* ── Summary ───────────────────────────────────────────────────── */
    print_summary(results, n_phases, L, steps, chunk_z, F, k, do_verify);
    write_aggregate_csv(results, n_phases, L, steps, chunk_z, F, k);
    printf("Per-chunk CSV written to: %s\n", OUT_CHUNKS);

    /* ── Cleanup ───────────────────────────────────────────────────── */
    cudaFree(d_count);
    gpucompress_grayscott_destroy(sim);
    H5VLclose(vol_id);
    gpucompress_cleanup();

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

    if (g_mpi_rank == 0)
        printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");

#ifdef GPUCOMPRESS_USE_MPI
    MPI_Finalize();
#endif
    return any_fail ? 1 : 0;
}