/**
 * @file benchmark_hdf5.c
 * @brief HDF5 Lossless Benchmark: No Compression vs Static vs NN
 *
 * Generates data patterns (float32, chunked), writes each to HDF5
 * under three modes, and outputs CSV + console summary + per-chunk adaptation log.
 *
 * Fill modes:
 *   --mode uniform     : every chunk filled with the same pattern (ramp)
 *   --mode contiguous  : each pattern fills a contiguous block of chunks
 *   --mode cycling     : patterns cycle every chunk
 *
 * Sizes (configurable):
 *   --dataset-mb N     : total dataset size in MB (default 16384 = 16 GB)
 *   --chunk-mb N       : chunk size in MB (default 4)
 *
 * Compression modes:
 *   1. No compression — chunked, no filter (baseline)
 *   2. Static — LZ4+shuffle (best static config)
 *   3. NN — ALGO_AUTO lossless via bulk H5Dwrite with per-chunk diagnostic history
 *
 * Usage:
 *   ./build/benchmark_hdf5 neural_net/weights/model.nnwt [--mode contiguous] [--dataset-mb 16384] [--chunk-mb 4]
 *   GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt ./build/benchmark_hdf5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <libgen.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ============================================================
 * Compile-time constants
 * ============================================================ */

#define N_ALGOS        8
#define N_STATIC       1
#define MAX_RESULTS    512
#define WINDOW_SIZE    256

#define DEFAULT_CHUNK_MB   4
#define DEFAULT_DATASET_MB 16384   /* 16 GB */
#define DEFAULT_SWEEP_MB   128     /* small sweep dataset */

#define TMP_HDF5       "/tmp/bm_hdf5_tmp.h5"
#define DEFAULT_CSV    "tests/benchmark_hdf5_results/benchmark_hdf5_results.csv"
#define DEFAULT_CHUNK_CSV "tests/benchmark_hdf5_results/benchmark_hdf5_chunks.csv"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * Runtime sizing (set once in main, read everywhere)
 * ============================================================ */

static size_t g_chunk_floats;    /* floats per chunk */
static size_t g_chunk_bytes;     /* bytes per chunk  */
static size_t g_dataset_floats;  /* total floats     */
static size_t g_dataset_bytes;   /* total bytes      */
static size_t g_n_chunks;        /* dataset / chunk  */
static size_t g_n_windows;       /* n_chunks / WINDOW_SIZE */

/* ============================================================
 * Fill Modes & Pattern Table
 * ============================================================ */

#define MODE_UNIFORM    0
#define MODE_CONTIGUOUS 1   /* each pattern fills a contiguous block */
#define MODE_CYCLING    2   /* patterns cycle every chunk */
#define UNIFORM_PATTERN 2   /* ramp */

#define N_FILL_PATTERNS 12

static const char *fill_pattern_names[N_FILL_PATTERNS] = {
    "constant",       /* 0  — calibration/baseline */
    "smooth_sine",    /* 1  — seismic/acoustic wave */
    "ramp",           /* 2  — sensor sweep/ADC ramp */
    "gaussian",       /* 3  — detector noise/Monte Carlo */
    "sparse",         /* 4  — particle physics events */
    "step",           /* 5  — quantized sensor/state machine */
    "hf_sine_noise",  /* 6  — turbulence/vibration */
    "exp_decay",      /* 7  — radioactive decay/RC circuit */
    "sawtooth",       /* 8  — DAC output/timing signals */
    "mixed_freq",     /* 9  — climate/multi-scale physics */
    "lognormal",      /* 10 — fluid dynamics/finance */
    "impulse_train",  /* 11 — neural spike trains/radar */
};

/* ============================================================
 * PRNG (simple LCG)
 * ============================================================ */

static uint32_t lcg_state;
static void     lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void)       { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
static float    lcg_float(void)      { return (float)(lcg_next() >> 8) / 16777216.0f; }

/* ============================================================
 * Timing
 * ============================================================ */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Per-Chunk Data Pattern Generator
 *
 * Fills exactly g_chunk_floats with a self-contained pattern.
 * Each pattern is deterministic (seeded per pattern+chunk).
 * ============================================================ */

static void fill_chunk(float *buf, int pattern_id, int chunk_idx) {
    const size_t N = g_chunk_floats;
    /* Seed LCG deterministically per pattern+chunk for reproducibility */
    lcg_seed(0xDEADBEEF ^ ((uint32_t)pattern_id * 7919u)
                         ^ ((uint32_t)chunk_idx * 104729u));

    switch (pattern_id) {
    case 0: /* constant — all 42.0f */
        for (size_t i = 0; i < N; i++)
            buf[i] = 42.0f;
        break;

    case 1: /* smooth_sine — 1 full cycle per chunk */
        for (size_t i = 0; i < N; i++)
            buf[i] = 1000.0f * sinf(2.0f * (float)M_PI * (float)i / (float)N);
        break;

    case 2: /* ramp — linear 0..1 within chunk */
        for (size_t i = 0; i < N; i++)
            buf[i] = (float)i / (float)N;
        break;

    case 3: /* gaussian — Box-Muller, mean=0, std=500 */
        for (size_t i = 0; i < N; i += 2) {
            float u1 = lcg_float() * 0.9999f + 0.0001f;
            float u2 = lcg_float();
            float mag = sqrtf(-2.0f * logf(u1));
            buf[i] = mag * cosf(2.0f * (float)M_PI * u2) * 500.0f;
            if (i + 1 < N)
                buf[i + 1] = mag * sinf(2.0f * (float)M_PI * u2) * 500.0f;
        }
        break;

    case 4: /* sparse — 99% zero, 1% random spikes */
        for (size_t i = 0; i < N; i++) {
            if ((lcg_next() % 100) == 0)
                buf[i] = lcg_float() * 10000.0f - 5000.0f;
            else
                buf[i] = 0.0f;
        }
        break;

    case 5: /* step — 8 discrete levels */
        for (size_t i = 0; i < N; i++)
            buf[i] = (float)(i / (N / 8)) * 100.0f;
        break;

    case 6: /* hf_sine_noise — high-freq sine + 50% noise */
        for (size_t i = 0; i < N; i++)
            buf[i] = 500.0f * sinf(2.0f * (float)M_PI * (float)i * 0.3f)
                   + (lcg_float() - 0.5f) * 500.0f;
        break;

    case 7: /* exp_decay — exponential decay with noise */
        for (size_t i = 0; i < N; i++)
            buf[i] = 1000.0f * expf(-5.0f * (float)i / (float)N)
                   + (lcg_float() - 0.5f) * 50.0f;
        break;

    case 8: /* sawtooth — 16 repeating ramps per chunk */
        for (size_t i = 0; i < N; i++) {
            size_t period = N / 16;
            buf[i] = (float)(i % period) / (float)period * 1000.0f;
        }
        break;

    case 9: /* mixed_freq — sum of 3 sine waves */
        for (size_t i = 0; i < N; i++) {
            float t = (float)i / (float)N;
            buf[i] = 300.0f * sinf(2.0f * (float)M_PI * t * 3.0f)
                   + 200.0f * sinf(2.0f * (float)M_PI * t * 17.0f)
                   + 100.0f * sinf(2.0f * (float)M_PI * t * 97.0f);
        }
        break;

    case 10: /* lognormal — exp(gaussian) */
        for (size_t i = 0; i < N; i += 2) {
            float u1 = lcg_float() * 0.9999f + 0.0001f;
            float u2 = lcg_float();
            float mag = sqrtf(-2.0f * logf(u1));
            float z1 = mag * cosf(2.0f * (float)M_PI * u2);
            buf[i] = expf(z1 * 0.5f + 2.0f);
            if (i + 1 < N) {
                float z2 = mag * sinf(2.0f * (float)M_PI * u2);
                buf[i + 1] = expf(z2 * 0.5f + 2.0f);
            }
        }
        break;

    case 11: /* impulse_train — periodic narrow spikes on zero background */
        for (size_t i = 0; i < N; i++) {
            if ((i % 1024) < 4)
                buf[i] = 5000.0f + (lcg_float() - 0.5f) * 200.0f;
            else
                buf[i] = 0.0f;
        }
        break;

    default:
        memset(buf, 0, N * sizeof(float));
        break;
    }
}

/* ============================================================
 * Result Storage
 * ============================================================ */

typedef struct {
    char         pattern[32];
    char         mode[16];           /* "none", "static", "nn" */
    char         algorithm[32];
    unsigned int shuffle;
    size_t       file_bytes;
    size_t       original_bytes;
    double       ratio;
    double       write_ms;
    double       read_ms;
    double       write_mbps;
    double       read_mbps;
    int          nn_original_action;
    int          nn_final_action;
    int          explored;
    int          sgd_fired;
} result_t;

static result_t g_results[MAX_RESULTS];
static int      g_nresults = 0;

/* Per-chunk CSV for adaptation visualization */
static FILE *g_chunk_csv = NULL;

/* ============================================================
 * Decode NN action to human-readable label
 * ============================================================ */

static void action_label(int action, char *buf, size_t bufsz) {
    if (action < 0) {
        snprintf(buf, bufsz, "?");
        return;
    }
    int algo_idx = action % 8;
    int use_shuf = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s",
             gpucompress_algorithm_name((gpucompress_algorithm_t)(algo_idx + 1)),
             use_shuf ? "+shuf" : "");
}

/* ============================================================
 * Per-chunk NN diagnostics
 * ============================================================ */

typedef struct {
    int    explored;
    int    sgd_fired;
    int    original_action;
    int    final_action;
    double write_ms;
} chunk_diag_t;

/* ============================================================
 * Test Runner — single lossless write+read cycle (for none/static)
 * ============================================================ */

static int run_test(const float *data, size_t n_floats,
                    int use_filter, gpucompress_algorithm_t algo,
                    unsigned int shuffle_sz, result_t *r)
{
    herr_t hs;
    size_t orig_bytes = n_floats * sizeof(float);

    remove(TMP_HDF5);

    hid_t file = H5Fcreate(TMP_HDF5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return -1;

    hsize_t dims[1]       = { n_floats };
    hsize_t chunk_dims[1] = { g_chunk_floats };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);

    if (use_filter) {
        hs = H5Pset_gpucompress(dcpl, algo, 0, shuffle_sz, 0.0);
        if (hs < 0) {
            H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
            remove(TMP_HDF5);
            return -1;
        }
    }

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5);
        return -1;
    }

    double t0 = time_ms();
    hs = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (hs < 0) {
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5);
        return -1;
    }
    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file);
    double t1 = time_ms();

    /* Read back */
    file = H5Fopen(TMP_HDF5, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { remove(TMP_HDF5); return -1; }

    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(file); remove(TMP_HDF5); return -1; }

    hsize_t storage = H5Dget_storage_size(dset);

    float *rbuf = (float *)malloc(orig_bytes);
    if (!rbuf) { H5Dclose(dset); H5Fclose(file); remove(TMP_HDF5); return -1; }

    double t2 = time_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = time_ms();

    if (hs < 0) { free(rbuf); remove(TMP_HDF5); return -1; }

    /* Verify lossless */
    for (size_t i = 0; i < n_floats; i++) {
        if (data[i] != rbuf[i]) {
            fprintf(stderr, "VERIFY FAIL: lossless mismatch at [%zu] "
                    "%.6g != %.6g\n", i, data[i], rbuf[i]);
            free(rbuf);
            remove(TMP_HDF5);
            return -1;
        }
    }
    free(rbuf);

    r->original_bytes = orig_bytes;
    r->file_bytes     = (size_t)storage;
    r->ratio          = (storage > 0) ? (double)orig_bytes / (double)storage : 0.0;
    r->write_ms       = t1 - t0;
    r->read_ms        = t3 - t2;
    double mb         = orig_bytes / (1024.0 * 1024.0);
    r->write_mbps     = (r->write_ms > 0) ? mb / (r->write_ms / 1000.0) : 0.0;
    r->read_mbps      = (r->read_ms  > 0) ? mb / (r->read_ms  / 1000.0) : 0.0;
    r->shuffle        = shuffle_sz;
    r->nn_original_action = -1;
    r->nn_final_action    = -1;
    r->explored           = 0;
    r->sgd_fired          = 0;

    remove(TMP_HDF5);
    return 0;
}

/* ============================================================
 * Ensure parent directory exists
 * ============================================================ */

static void ensure_parent_dir(const char *path) {
    char *tmp = strdup(path);
    char *dir = dirname(tmp);
    if (dir && strlen(dir) > 0 && strcmp(dir, ".") != 0) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
        (void)system(cmd);
    }
    free(tmp);
}

/* ============================================================
 * CSV Output
 * ============================================================ */

static void write_csv(const char *path) {
    ensure_parent_dir(path);
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }

    fprintf(fp, "pattern,mode,algorithm,shuffle,"
                "file_bytes,original_bytes,compression_ratio,"
                "write_ms,read_ms,write_mbps,read_mbps,"
                "nn_original_action,nn_final_action,explored,sgd_fired\n");

    for (int i = 0; i < g_nresults; i++) {
        result_t *r = &g_results[i];
        fprintf(fp, "%s,%s,%s,%u,"
                    "%zu,%zu,%.4f,"
                    "%.2f,%.2f,%.1f,%.1f,"
                    "%d,%d,%d,%d\n",
                r->pattern, r->mode, r->algorithm, r->shuffle,
                r->file_bytes, r->original_bytes, r->ratio,
                r->write_ms, r->read_ms, r->write_mbps, r->read_mbps,
                r->nn_original_action, r->nn_final_action,
                r->explored, r->sgd_fired);
    }

    fclose(fp);
    printf("\nCSV written: %s (%d rows)\n", path, g_nresults);
}

/* ============================================================
 * Summary Table
 * ============================================================ */

static void print_summary(const char *pat_name) {
    double best_ratio = 0.0, nn_ratio = 0.0;
    char   best_algo[64], nn_algo[64], nn_orig_algo[64];
    int    nn_explored = 0, nn_sgd = 0;

    best_algo[0] = '\0'; nn_algo[0] = '\0'; nn_orig_algo[0] = '\0';

    for (int i = 0; i < g_nresults; i++) {
        result_t *r = &g_results[i];
        if (strcmp(r->pattern, pat_name) != 0) continue;

        if (strcmp(r->mode, "static") == 0 && r->ratio > best_ratio) {
            best_ratio = r->ratio;
            snprintf(best_algo, sizeof(best_algo), "%s%s",
                     r->algorithm, r->shuffle ? "+shuf" : "");
        } else if (strcmp(r->mode, "nn") == 0) {
            nn_ratio = r->ratio;
            snprintf(nn_algo, sizeof(nn_algo), "%s", r->algorithm);
            nn_explored = r->explored;
            nn_sgd = r->sgd_fired;
            action_label(r->nn_original_action,
                         nn_orig_algo, sizeof(nn_orig_algo));
        }
    }

    double pct = (best_ratio > 0) ? (nn_ratio / best_ratio * 100.0) : 0.0;
    int changed = nn_explored && strcmp(nn_orig_algo, nn_algo) != 0;

    printf("\n=== Lossless Compression Ratios ===\n\n");
    printf("%-14s | %10s  %-14s | %-14s  %-14s  %10s | %6s | %4s  %3s\n",
           "Pattern", "BestStatic", "Config",
           "NN Initial", "NN Final", "Ratio",
           "NN/Best", "Expl", "SGD");
    printf("%-14s-+-%10s--%-14s-+-%-14s--%-14s--%10s-+-%6s-+-%4s--%3s\n",
           "--------------", "----------", "--------------",
           "--------------", "--------------", "----------",
           "------", "----", "---");

    printf("%-14s | %8.2fx  %-14s | %-14s  %-14s  %8.2fx | %5.0f%% | %-4s  %-3s\n",
           pat_name,
           best_ratio, best_algo,
           nn_orig_algo, changed ? nn_algo : "-", nn_ratio,
           pct,
           nn_explored ? "yes" : "no",
           nn_sgd ? "yes" : "no");

    printf("\n  NN/Best: %.0f%%  |  Exploration: %s  |  SGD: %s  |  Changed: %s\n",
           pct,
           nn_explored ? "yes" : "no",
           nn_sgd ? "yes" : "no",
           changed ? "yes" : "no");

    printf("\n=== Verification ===\n");
    printf("  All configs verified lossless (bit-exact) during write+read cycle.\n");
}

/* ============================================================
 * Static config type
 * ============================================================ */

typedef struct {
    gpucompress_algorithm_t algo;
    unsigned int            shuffle_sz;
} static_cfg_t;

/* ============================================================
 * Sweep all 16 static configs (8 algos × 2 shuffle) on a small
 * dataset, return the best config by compression ratio.
 * ============================================================ */

static static_cfg_t sweep_all_static(const float *data, size_t n_floats,
                                     size_t sweep_bytes)
{
    static_cfg_t all_cfgs[N_ALGOS * 2];
    int ncfg = 0;
    for (int a = 1; a <= N_ALGOS; a++) {
        all_cfgs[ncfg].algo = (gpucompress_algorithm_t)a;
        all_cfgs[ncfg].shuffle_sz = 0;
        ncfg++;
        all_cfgs[ncfg].algo = (gpucompress_algorithm_t)a;
        all_cfgs[ncfg].shuffle_sz = 4;
        ncfg++;
    }

    printf("============================================================\n");
    printf("  PHASE 1: Static Sweep (%d configs, %zu MB)\n", ncfg,
           sweep_bytes / (1024 * 1024));
    printf("============================================================\n\n");
    printf("  %-12s | %4s | %8s | %10s | %10s\n",
           "Algorithm", "Shuf", "Ratio", "Write MB/s", "Read MB/s");
    printf("  -------------+------+----------+------------+------------\n");

    double best_ratio = 0.0;
    static_cfg_t best_cfg = { GPUCOMPRESS_ALGO_ZSTD, 4 };  /* fallback */

    for (int c = 0; c < ncfg; c++) {
        result_t r;
        memset(&r, 0, sizeof(r));
        if (run_test(data, n_floats, 1,
                     all_cfgs[c].algo, all_cfgs[c].shuffle_sz, &r) == 0) {
            const char *aname = gpucompress_algorithm_name(all_cfgs[c].algo);
            printf("  %-12s | %4u | %8.2fx | %10.1f | %10.1f\n",
                   aname, all_cfgs[c].shuffle_sz,
                   r.ratio, r.write_mbps, r.read_mbps);
            if (r.ratio > best_ratio) {
                best_ratio = r.ratio;
                best_cfg = all_cfgs[c];
            }
        } else {
            const char *aname = gpucompress_algorithm_name(all_cfgs[c].algo);
            printf("  %-12s | %4u | %8s | %10s | %10s\n",
                   aname, all_cfgs[c].shuffle_sz, "FAIL", "-", "-");
        }
    }

    printf("\n  Best: %.2fx (%s%s)\n\n", best_ratio,
           gpucompress_algorithm_name(best_cfg.algo),
           best_cfg.shuffle_sz ? "+shuf" : "");

    return best_cfg;
}

/* ============================================================
 * NN benchmark phase — standalone
 * Writes entire dataset with ALGO_AUTO, captures diagnostics.
 * ============================================================ */

static void bench_nn(const char *pat_name, const float *data,
                     size_t n_floats, const int *chunk_patterns)
{
    if (g_nresults >= MAX_RESULTS) return;

    result_t *r;
    remove(TMP_HDF5);

    hid_t nn_file = H5Fcreate(TMP_HDF5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (nn_file < 0) { printf("  NN: FAILED (file create)\n"); return; }

    hsize_t dims[1]       = { n_floats };
    hsize_t chunk_dims[1] = { g_chunk_floats };
    hid_t nn_dspace = H5Screate_simple(1, dims, NULL);
    hid_t nn_dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(nn_dcpl, 1, chunk_dims);

    if (H5Pset_gpucompress(nn_dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) < 0) {
        H5Pclose(nn_dcpl); H5Sclose(nn_dspace); H5Fclose(nn_file);
        remove(TMP_HDF5);
        printf("  NN: FAILED (filter setup)\n");
        return;
    }

    hid_t nn_dset = H5Dcreate2(nn_file, "data", H5T_NATIVE_FLOAT,
                                 nn_dspace, H5P_DEFAULT, nn_dcpl, H5P_DEFAULT);
    if (nn_dset < 0) {
        H5Pclose(nn_dcpl); H5Sclose(nn_dspace); H5Fclose(nn_file);
        remove(TMP_HDF5);
        printf("  NN: FAILED (dataset create)\n");
        return;
    }

    printf("\n  --- NN Adaptation (%zu x %zu MB chunks, single HDF5 file) ---\n",
           g_n_chunks, g_chunk_bytes / (1024 * 1024));
    printf("  %5s | %-14s | %-14s | %-14s | %3s %3s\n",
           "Chunk", "Pattern", "NN Prediction", "Final Action", "Exp", "SGD");
    printf("  ------+----------------+----------------+----------------+--------\n");

    gpucompress_reset_chunk_history();
    double t0 = time_ms();
    herr_t hs = H5Dwrite(nn_dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, data);

    if (hs < 0) {
        fprintf(stderr, "  FAILED: H5Dwrite\n");
        H5Dclose(nn_dset); H5Pclose(nn_dcpl);
        H5Sclose(nn_dspace); H5Fclose(nn_file);
        remove(TMP_HDF5);
        return;
    }

    H5Dclose(nn_dset);
    H5Pclose(nn_dcpl);
    H5Sclose(nn_dspace);
    H5Fclose(nn_file);
    double t1 = time_ms();
    double total_write_ms = t1 - t0;

    /* Retrieve per-chunk diagnostics from history buffer */
    int n_hist = gpucompress_get_chunk_history_count();
    chunk_diag_t *diags = (chunk_diag_t *)calloc(g_n_chunks, sizeof(chunk_diag_t));
    int *win_explored = (int *)calloc(g_n_windows, sizeof(int));
    int *win_sgd      = (int *)calloc(g_n_windows, sizeof(int));
    int *win_changed  = (int *)calloc(g_n_windows, sizeof(int));
    if (!diags || !win_explored || !win_sgd || !win_changed) {
        fprintf(stderr, "  FAILED: alloc diags\n");
        free(diags); free(win_explored); free(win_sgd); free(win_changed);
        remove(TMP_HDF5);
        return;
    }

    int total_explored = 0, total_sgd = 0, total_changed = 0;
    int last_final_action = -1;

    for (size_t i = 0; i < g_n_chunks; i++) {
        chunk_diag_t *d = &diags[i];
        gpucompress_chunk_diag_t hd;
        if ((int)i < n_hist && gpucompress_get_chunk_diag((int)i, &hd) == 0) {
            d->original_action = hd.nn_original_action;
            d->final_action    = hd.nn_action;
            d->explored        = hd.exploration_triggered;
            d->sgd_fired       = hd.sgd_fired;
        } else {
            d->original_action = -1;
            d->final_action    = -1;
        }
        d->write_ms = 0.0;

        size_t w = i / WINDOW_SIZE;
        if (w < g_n_windows) {
            win_explored[w] += d->explored;
            win_sgd[w]      += d->sgd_fired;
            if (d->explored && d->original_action != d->final_action)
                win_changed[w]++;
        }

        total_explored += d->explored;
        total_sgd      += d->sgd_fired;
        if (d->explored && d->original_action != d->final_action)
            total_changed++;
        last_final_action = d->final_action;

        if (i < 32 || d->explored) {
            char orig[32], final[32];
            action_label(d->original_action, orig, sizeof(orig));
            action_label(d->final_action, final, sizeof(final));
            int chg = d->explored && d->original_action != d->final_action;
            printf("  %5zu | %-14s | %-14s | %-14s | %s %s%s\n",
                   i + 1,
                   fill_pattern_names[chunk_patterns[i]],
                   orig,
                   chg ? final : "-",
                   d->explored ? " E " : "   ",
                   d->sgd_fired ? " S " : "   ",
                   i >= 32 ? "  *" : "");
        }
    }

    /* Reopen: get storage size, per-chunk sizes, and timed read-back */
    size_t nn_orig_bytes = n_floats * sizeof(float);
    size_t nn_storage = 0;
    double read_ms = 0.0;
    double *chunk_ratios = (double *)calloc(g_n_chunks, sizeof(double));

    nn_file = H5Fopen(TMP_HDF5, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (nn_file >= 0) {
        nn_dset = H5Dopen2(nn_file, "data", H5P_DEFAULT);
        if (nn_dset >= 0) {
            nn_storage = (size_t)H5Dget_storage_size(nn_dset);

            hid_t cspace = H5Dget_space(nn_dset);
            for (size_t i = 0; i < g_n_chunks; i++) {
                hsize_t coff; unsigned fmask; haddr_t caddr; hsize_t csz = 0;
                if (H5Dget_chunk_info(nn_dset, cspace, (hsize_t)i,
                                      &coff, &fmask, &caddr, &csz) >= 0 && csz > 0) {
                    chunk_ratios[i] = (double)g_chunk_bytes / (double)csz;
                }
            }
            H5Sclose(cspace);

            float *rbuf = (float *)malloc(nn_orig_bytes);
            if (rbuf) {
                double t2 = time_ms();
                H5Dread(nn_dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, rbuf);
                double t3 = time_ms();
                read_ms = t3 - t2;
                free(rbuf);
            }
            H5Dclose(nn_dset);
        }
        H5Fclose(nn_file);
    }
    remove(TMP_HDF5);

    double nn_ratio = (nn_storage > 0)
                    ? (double)nn_orig_bytes / (double)nn_storage : 0.0;

    /* Write per-chunk CSV */
    if (g_chunk_csv) {
        for (size_t i = 0; i < g_n_chunks; i++) {
            chunk_diag_t *d = &diags[i];
            char ol[32], fl[32];
            action_label(d->original_action, ol, sizeof(ol));
            action_label(d->final_action, fl, sizeof(fl));
            fprintf(g_chunk_csv, "%s,%zu,%s,%d,%d,%d,%d,%s,%s,%.4f,%.2f\n",
                    pat_name, i,
                    fill_pattern_names[chunk_patterns[i]],
                    d->explored, d->sgd_fired,
                    d->original_action, d->final_action,
                    ol, fl, chunk_ratios[i], d->write_ms);
        }
    }

    /* Windowed summary */
    printf("\n  --- Windowed summary (%d-chunk windows) ---\n\n", WINDOW_SIZE);
    printf("  %-11s | %4s | %4s | %4s | %8s\n",
           "Chunks", "Expl", "SGD", "Chgd", "Expl%");
    printf("  -----------+------+------+------+----------\n");

    for (size_t w = 0; w < g_n_windows; w++) {
        printf("  %3zu - %3zu  | %2d/%d | %2d/%d | %2d/%d | %6.1f%%\n",
               w * WINDOW_SIZE + 1, (w + 1) * WINDOW_SIZE,
               win_explored[w], WINDOW_SIZE,
               win_sgd[w], WINDOW_SIZE,
               win_changed[w], WINDOW_SIZE,
               100.0 * win_explored[w] / WINDOW_SIZE);
    }

    printf("  -----------+------+------+------+----------\n");
    printf("  TOTAL      | %3d  | %3d  | %3d  | %6.1f%%\n",
           total_explored, total_sgd, total_changed,
           100.0 * total_explored / g_n_chunks);
    printf("  out of %zu chunks\n", g_n_chunks);

    if (total_explored == 0)
        printf("\n  Model predicted accurately from the start.\n");
    else if (g_n_windows > 0 && win_explored[g_n_windows - 1] == 0)
        printf("\n  Model adapted: exploration dropped to 0%% by the end.\n");
    else
        printf("\n  Model still exploring at the end — may need more chunks or tuning.\n");

    /* Record aggregate NN result */
    r = &g_results[g_nresults];
    snprintf(r->pattern, sizeof(r->pattern), "%s", pat_name);
    snprintf(r->mode,    sizeof(r->mode),    "nn");
    action_label(last_final_action, r->algorithm, sizeof(r->algorithm));
    double total_mb       = (double)nn_orig_bytes / (1024.0 * 1024.0);
    r->original_bytes     = nn_orig_bytes;
    r->file_bytes         = nn_storage;
    r->ratio              = nn_ratio;
    r->write_ms           = total_write_ms;
    r->read_ms            = read_ms;
    r->write_mbps         = (total_write_ms > 0) ? total_mb / (total_write_ms / 1000.0) : 0.0;
    r->read_mbps          = (read_ms > 0)        ? total_mb / (read_ms / 1000.0) : 0.0;
    r->shuffle            = 0;
    r->nn_original_action = -1;
    r->nn_final_action    = last_final_action;
    r->explored           = total_explored;
    r->sgd_fired          = total_sgd;
    g_nresults++;

    printf("\n  NN: %.2fx (%s), %zu -> %zu bytes\n",
           nn_ratio, r->algorithm, nn_orig_bytes, nn_storage);
    printf("  Write: %.1f ms (%.1f MB/s), Read: %.1f ms (%.1f MB/s)\n",
           total_write_ms, r->write_mbps, read_ms, r->read_mbps);
    printf("  Explored %d/%zu, SGD %d/%zu, Changed %d/%zu\n",
           total_explored, g_n_chunks, total_sgd, g_n_chunks, total_changed, g_n_chunks);

    free(diags);
    free(win_explored);
    free(win_sgd);
    free(win_changed);
    free(chunk_ratios);
}

/* ============================================================
 * Benchmark one dataset: none + static + NN
 * ============================================================ */

static void bench_dataset(const char *pat_name, const float *data,
                          size_t n_floats,
                          const static_cfg_t *cfgs, int ncfg,
                          const int *chunk_patterns)
{
    result_t *r;

    /* 1. No compression */
    if (g_nresults >= MAX_RESULTS) return;
    r = &g_results[g_nresults];
    if (run_test(data, n_floats, 0, 0, 0, r) == 0) {
        snprintf(r->pattern,   sizeof(r->pattern),   "%s", pat_name);
        snprintf(r->mode,      sizeof(r->mode),      "none");
        snprintf(r->algorithm, sizeof(r->algorithm), "none");
        printf("  No compression: %.2fx, Write: %.1f ms (%.1f MB/s), Read: %.1f ms (%.1f MB/s)\n",
               r->ratio, r->write_ms, r->write_mbps, r->read_ms, r->read_mbps);
        g_nresults++;
    } else {
        printf("  No compression: FAILED\n");
    }

    /* 2. Static configs (LZ4+shuffle) */
    int ok = 0, fail = 0;
    double best_ratio = 0.0;
    int    best_idx   = -1;

    for (int c = 0; c < ncfg; c++) {
        if (g_nresults >= MAX_RESULTS) break;
        r = &g_results[g_nresults];
        if (run_test(data, n_floats, 1,
                     cfgs[c].algo, cfgs[c].shuffle_sz, r) == 0) {
            snprintf(r->pattern,   sizeof(r->pattern),   "%s", pat_name);
            snprintf(r->mode,      sizeof(r->mode),      "static");
            snprintf(r->algorithm, sizeof(r->algorithm), "%s",
                     gpucompress_algorithm_name(cfgs[c].algo));
            if (r->ratio > best_ratio) {
                best_ratio = r->ratio;
                best_idx   = g_nresults;
            }
            g_nresults++;
            ok++;
        } else {
            fail++;
        }
    }

    if (best_idx >= 0) {
        result_t *br = &g_results[best_idx];
        printf("  Static: %d/%d ok, best %.2fx (%s%s)\n",
               ok, ncfg, best_ratio, br->algorithm,
               br->shuffle ? "+shuf" : "");
    }
    if (fail > 0) printf("  Static: %d configs failed\n", fail);

    /* 3. NN lossless */
    bench_nn(pat_name, data, n_floats, chunk_patterns);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv) {
    const char *csv_path       = DEFAULT_CSV;
    const char *chunk_csv_path = DEFAULT_CHUNK_CSV;
    const char *weights_path   = NULL;
    int         fill_mode      = MODE_CONTIGUOUS; /* default: contiguous */
    int         dataset_mb     = DEFAULT_DATASET_MB;
    int         chunk_mb       = DEFAULT_CHUNK_MB;
    int         sweep_mode     = 0;
    int         sweep_mb       = DEFAULT_SWEEP_MB;
    gpucompress_algorithm_t static_algo = GPUCOMPRESS_ALGO_ZSTD;
    unsigned int static_shuffle = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--chunk-csv") == 0 && i + 1 < argc)
            chunk_csv_path = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "uniform") == 0)
                fill_mode = MODE_UNIFORM;
            else if (strcmp(argv[i], "contiguous") == 0)
                fill_mode = MODE_CONTIGUOUS;
            else if (strcmp(argv[i], "cycling") == 0)
                fill_mode = MODE_CYCLING;
            else {
                fprintf(stderr, "ERROR: unknown mode '%s' (use 'uniform', 'contiguous', or 'cycling')\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--dataset-mb") == 0 && i + 1 < argc) {
            dataset_mb = atoi(argv[++i]);
            if (dataset_mb < 1) {
                fprintf(stderr, "ERROR: --dataset-mb must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]);
            if (chunk_mb < 1) {
                fprintf(stderr, "ERROR: --chunk-mb must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--static-algo") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "lz4") == 0)
                static_algo = GPUCOMPRESS_ALGO_LZ4;
            else if (strcmp(argv[i], "snappy") == 0)
                static_algo = GPUCOMPRESS_ALGO_SNAPPY;
            else if (strcmp(argv[i], "deflate") == 0)
                static_algo = GPUCOMPRESS_ALGO_DEFLATE;
            else if (strcmp(argv[i], "gdeflate") == 0)
                static_algo = GPUCOMPRESS_ALGO_GDEFLATE;
            else if (strcmp(argv[i], "zstd") == 0)
                static_algo = GPUCOMPRESS_ALGO_ZSTD;
            else if (strcmp(argv[i], "ans") == 0)
                static_algo = GPUCOMPRESS_ALGO_ANS;
            else if (strcmp(argv[i], "cascaded") == 0)
                static_algo = GPUCOMPRESS_ALGO_CASCADED;
            else if (strcmp(argv[i], "bitcomp") == 0)
                static_algo = GPUCOMPRESS_ALGO_BITCOMP;
            else {
                fprintf(stderr, "ERROR: unknown algo '%s' "
                        "(use lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp)\n",
                        argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--sweep") == 0) {
            sweep_mode = 1;
        } else if (strcmp(argv[i], "--sweep-mb") == 0 && i + 1 < argc) {
            sweep_mb = atoi(argv[++i]);
            if (sweep_mb < 1) {
                fprintf(stderr, "ERROR: --sweep-mb must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--no-shuffle") == 0) {
            static_shuffle = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [weights.nnwt] [options]\n"
                   "  --mode uniform|contiguous|cycling  Fill mode (default: contiguous)\n"
                   "  --dataset-mb N                     Dataset size in MB (default: %d)\n"
                   "  --chunk-mb N                       Chunk size in MB (default: %d)\n"
                   "  --sweep                            3-phase: sweep small -> best static -> NN\n"
                   "  --sweep-mb N                       Sweep dataset size in MB (default: %d)\n"
                   "  --static-algo ALGO                 Static baseline algorithm (default: zstd)\n"
                   "       ALGO: lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp\n"
                   "  --no-shuffle                       Disable shuffle for static baseline\n"
                   "  --output FILE                      Aggregate CSV path\n"
                   "  --chunk-csv FILE                   Per-chunk CSV path\n"
                   "  Or set GPUCOMPRESS_WEIGHTS env var\n",
                   argv[0], DEFAULT_DATASET_MB, DEFAULT_CHUNK_MB, DEFAULT_SWEEP_MB);
            return 0;
        } else if (!weights_path)
            weights_path = argv[i];
    }

    if (!weights_path)
        weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s [weights.nnwt] [--mode uniform|contiguous|cycling] "
                        "[--dataset-mb N] [--chunk-mb N]\n"
                        "  Or set GPUCOMPRESS_WEIGHTS env var\n", argv[0]);
        return 1;
    }

    if (dataset_mb < chunk_mb) {
        fprintf(stderr, "ERROR: --dataset-mb (%d) must be >= --chunk-mb (%d)\n",
                dataset_mb, chunk_mb);
        return 1;
    }

    /* Compute runtime sizes */
    g_chunk_floats   = (size_t)chunk_mb * 1024 * 1024 / sizeof(float);
    g_chunk_bytes    = g_chunk_floats * sizeof(float);
    g_dataset_floats = (size_t)dataset_mb * 1024 * 1024 / sizeof(float);
    g_dataset_bytes  = g_dataset_floats * sizeof(float);
    g_n_chunks       = g_dataset_floats / g_chunk_floats;
    g_n_windows      = g_n_chunks / WINDOW_SIZE;

    if (g_n_chunks == 0) {
        fprintf(stderr, "ERROR: dataset (%d MB) too small for chunk size (%d MB)\n",
                dataset_mb, chunk_mb);
        return 1;
    }

    const char *mode_str, *pat_name;
    switch (fill_mode) {
    case MODE_UNIFORM:    mode_str = "uniform";    pat_name = "uniform_ramp";  break;
    case MODE_CONTIGUOUS: mode_str = "contiguous"; pat_name = "contiguous";    break;
    case MODE_CYCLING:    mode_str = "cycling";    pat_name = "cycling";       break;
    default:              mode_str = "unknown";    pat_name = "unknown";       break;
    }

    printf("=== HDF5 Lossless Benchmark: No Compression vs Static vs NN ===\n\n");
    printf("Weights:  %s\n", weights_path);
    if (sweep_mode)
        printf("Sweep:    %d MB (finding best static config)\n", sweep_mb);
    else
        printf("Static:   %s%s\n", gpucompress_algorithm_name(static_algo),
               static_shuffle ? "+shuffle" : "");
    printf("Mode:     %s\n", mode_str);
    printf("CSV:      %s\n", csv_path);
    printf("ChunkCSV: %s\n", chunk_csv_path);
    printf("Chunk:    %zu MB float32 (%zu floats)\n",
           g_chunk_bytes / (1024 * 1024), g_chunk_floats);
    printf("Dataset:  %zu MB float32 (%zu floats, %zu chunks)\n",
           g_dataset_bytes / (1024 * 1024), g_dataset_floats, g_n_chunks);
    if (fill_mode == MODE_CONTIGUOUS)
        printf("Patterns: %d contiguous blocks (~%zu chunks each)\n\n",
               N_FILL_PATTERNS, g_n_chunks / N_FILL_PATTERNS);
    else if (fill_mode == MODE_CYCLING)
        printf("Patterns: %d cycling every chunk (%zu chunks = ~%zu full cycles)\n\n",
               N_FILL_PATTERNS, g_n_chunks, g_n_chunks / N_FILL_PATTERNS);
    else
        printf("Pattern:  %s (uniform across all %zu chunks)\n\n",
               fill_pattern_names[UNIFORM_PATTERN], g_n_chunks);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    gpucompress_error_t rc = gpucompress_init(weights_path);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights did not load from %s\n", weights_path);
        gpucompress_cleanup();
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.4f, 0.2f, 0.0f);

    /* Open per-chunk CSV */
    ensure_parent_dir(chunk_csv_path);
    g_chunk_csv = fopen(chunk_csv_path, "w");
    if (g_chunk_csv) {
        fprintf(g_chunk_csv, "pattern,chunk_id,chunk_pattern,explored,sgd_fired,"
                             "original_action,final_action,"
                             "orig_label,final_label,ratio,write_ms\n");
    } else {
        fprintf(stderr, "WARNING: cannot write chunk CSV %s\n", chunk_csv_path);
    }

    if (sweep_mode) {
        /* ============================================================
         * 3-Phase Sweep Workflow
         * ============================================================ */

        /* --- Phase 1: Small sweep to find best static config --- */
        size_t sweep_floats = (size_t)sweep_mb * 1024 * 1024 / sizeof(float);
        size_t sweep_bytes  = sweep_floats * sizeof(float);
        size_t sweep_chunks = sweep_floats / g_chunk_floats;
        if (sweep_chunks == 0) {
            fprintf(stderr, "ERROR: --sweep-mb (%d) too small for chunk size (%d MB)\n",
                    sweep_mb, chunk_mb);
            gpucompress_cleanup();
            return 1;
        }
        /* Round to whole chunks */
        sweep_floats = sweep_chunks * g_chunk_floats;
        sweep_bytes  = sweep_floats * sizeof(float);

        printf("Phase 1: Allocating %zu MB sweep buffer...\n",
               sweep_bytes / (1024 * 1024));
        float *sweep_data = (float *)malloc(sweep_bytes);
        if (!sweep_data) { perror("malloc sweep"); gpucompress_cleanup(); return 1; }

        printf("Filling %zu sweep chunks (%s mode)...\n", sweep_chunks, mode_str);
        double fill_t0 = time_ms();
        size_t chunks_per_pat = sweep_chunks / N_FILL_PATTERNS;
        if (chunks_per_pat == 0) chunks_per_pat = 1;
        for (size_t c = 0; c < sweep_chunks; c++) {
            int pat_id;
            switch (fill_mode) {
            case MODE_CONTIGUOUS:
                pat_id = (int)(c / chunks_per_pat);
                if (pat_id >= N_FILL_PATTERNS) pat_id = N_FILL_PATTERNS - 1;
                break;
            case MODE_CYCLING:
                pat_id = (int)(c % N_FILL_PATTERNS);
                break;
            default:
                pat_id = UNIFORM_PATTERN;
                break;
            }
            fill_chunk(sweep_data + c * g_chunk_floats, pat_id, (int)c);
        }
        double fill_t1 = time_ms();
        printf("Sweep fill complete: %.1f ms\n\n", fill_t1 - fill_t0);

        static_cfg_t best_cfg = sweep_all_static(sweep_data, sweep_floats, sweep_bytes);
        free(sweep_data);

        /* --- Phase 2: Full-scale static benchmark (no-compress + best) --- */
        printf("============================================================\n");
        printf("  PHASE 2: Full Static (%d MB, %s%s)\n",
               dataset_mb, gpucompress_algorithm_name(best_cfg.algo),
               best_cfg.shuffle_sz ? "+shuf" : "");
        printf("============================================================\n\n");

        printf("Allocating %zu MB full buffer...\n", g_dataset_bytes / (1024 * 1024));
        float *data = (float *)malloc(g_dataset_bytes);
        if (!data) { perror("malloc full"); gpucompress_cleanup(); return 1; }

        int *chunk_patterns = (int *)malloc(g_n_chunks * sizeof(int));
        if (!chunk_patterns) { perror("malloc"); free(data); gpucompress_cleanup(); return 1; }

        printf("Filling %zu chunks (%s mode)...\n", g_n_chunks, mode_str);
        fill_t0 = time_ms();
        chunks_per_pat = g_n_chunks / N_FILL_PATTERNS;
        if (chunks_per_pat == 0) chunks_per_pat = 1;
        for (size_t c = 0; c < g_n_chunks; c++) {
            int pat_id;
            switch (fill_mode) {
            case MODE_CONTIGUOUS:
                pat_id = (int)(c / chunks_per_pat);
                if (pat_id >= N_FILL_PATTERNS) pat_id = N_FILL_PATTERNS - 1;
                break;
            case MODE_CYCLING:
                pat_id = (int)(c % N_FILL_PATTERNS);
                break;
            default:
                pat_id = UNIFORM_PATTERN;
                break;
            }
            chunk_patterns[c] = pat_id;
            fill_chunk(data + c * g_chunk_floats, pat_id, (int)c);
        }
        fill_t1 = time_ms();
        printf("Fill complete: %.1f ms\n\n", fill_t1 - fill_t0);

        /* No compression baseline */
        result_t *r;
        if (g_nresults < MAX_RESULTS) {
            r = &g_results[g_nresults];
            if (run_test(data, g_dataset_floats, 0, 0, 0, r) == 0) {
                snprintf(r->pattern,   sizeof(r->pattern),   "%s", pat_name);
                snprintf(r->mode,      sizeof(r->mode),      "none");
                snprintf(r->algorithm, sizeof(r->algorithm), "none");
                printf("  No compression: %.2fx, Write: %.1f ms (%.1f MB/s), Read: %.1f ms (%.1f MB/s)\n",
                       r->ratio, r->write_ms, r->write_mbps, r->read_ms, r->read_mbps);
                g_nresults++;
            } else {
                printf("  No compression: FAILED\n");
            }
        }

        /* Best static config from sweep */
        if (g_nresults < MAX_RESULTS) {
            r = &g_results[g_nresults];
            if (run_test(data, g_dataset_floats, 1,
                         best_cfg.algo, best_cfg.shuffle_sz, r) == 0) {
                snprintf(r->pattern,   sizeof(r->pattern),   "%s", pat_name);
                snprintf(r->mode,      sizeof(r->mode),      "static");
                snprintf(r->algorithm, sizeof(r->algorithm), "%s",
                         gpucompress_algorithm_name(best_cfg.algo));
                printf("  Static (%s%s): %.2fx, Write: %.1f ms (%.1f MB/s), Read: %.1f ms (%.1f MB/s)\n",
                       r->algorithm, r->shuffle ? "+shuf" : "",
                       r->ratio, r->write_ms, r->write_mbps, r->read_ms, r->read_mbps);
                g_nresults++;
            } else {
                printf("  Static (%s%s): FAILED\n",
                       gpucompress_algorithm_name(best_cfg.algo),
                       best_cfg.shuffle_sz ? "+shuf" : "");
            }
        }

        /* --- Phase 3: Full-scale NN benchmark --- */
        printf("\n============================================================\n");
        printf("  PHASE 3: Full NN (%d MB, ALGO_AUTO)\n", dataset_mb);
        printf("============================================================\n");

        bench_nn(pat_name, data, g_dataset_floats, chunk_patterns);

        free(data);
        free(chunk_patterns);

    } else {
        /* ============================================================
         * Original single-config flow (no sweep)
         * ============================================================ */

        printf("Allocating %zu MB buffer...\n", g_dataset_bytes / (1024 * 1024));
        float *data = (float *)malloc(g_dataset_bytes);
        if (!data) { perror("malloc"); gpucompress_cleanup(); return 1; }

        int *chunk_patterns = (int *)malloc(g_n_chunks * sizeof(int));
        if (!chunk_patterns) { perror("malloc"); free(data); gpucompress_cleanup(); return 1; }

        printf("Filling %zu chunks (%s mode)...\n", g_n_chunks, mode_str);
        double fill_t0 = time_ms();
        size_t chunks_per_pat = g_n_chunks / N_FILL_PATTERNS;
        if (chunks_per_pat == 0) chunks_per_pat = 1;
        for (size_t c = 0; c < g_n_chunks; c++) {
            int pat_id;
            switch (fill_mode) {
            case MODE_CONTIGUOUS:
                pat_id = (int)(c / chunks_per_pat);
                if (pat_id >= N_FILL_PATTERNS) pat_id = N_FILL_PATTERNS - 1;
                break;
            case MODE_CYCLING:
                pat_id = (int)(c % N_FILL_PATTERNS);
                break;
            default:
                pat_id = UNIFORM_PATTERN;
                break;
            }
            chunk_patterns[c] = pat_id;
            fill_chunk(data + c * g_chunk_floats, pat_id, (int)c);
        }
        double fill_t1 = time_ms();
        printf("Fill complete: %.1f ms\n\n", fill_t1 - fill_t0);

        static_cfg_t cfgs[1];
        cfgs[0].algo       = static_algo;
        cfgs[0].shuffle_sz = static_shuffle;
        int ncfg = 1;

        printf("Dataset: %s (%zu MB, %zu x %zu MB chunks)\n",
               pat_name, g_dataset_bytes / (1024 * 1024),
               g_n_chunks, g_chunk_bytes / (1024 * 1024));

        bench_dataset(pat_name, data, g_dataset_floats, cfgs, ncfg, chunk_patterns);
        free(data);
        free(chunk_patterns);
    }

    /* Output */
    write_csv(csv_path);
    if (g_chunk_csv) {
        fclose(g_chunk_csv);
        g_chunk_csv = NULL;
        printf("Chunk CSV written: %s\n", chunk_csv_path);
    }
    print_summary(pat_name);

    gpucompress_cleanup();
    printf("\nDone. %d results recorded.\n", g_nresults);
    return 0;
}
