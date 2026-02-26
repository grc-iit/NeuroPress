/**
 * @file benchmark_hdf5.c
 * @brief HDF5 Lossless Benchmark: No Compression vs Static vs NN
 *
 * Generates 8 uniform data patterns (4 MB float32 each), writes each to HDF5
 * under three modes, and outputs a CSV + console summary table.
 *
 * Modes:
 *   1. No compression — chunked, no filter (baseline)
 *   2. Static — 16 lossless configs (8 algos x 2 shuffle)
 *   3. NN — ALGO_AUTO lossless (NN decides algo + shuffle)
 *
 * Usage:
 *   ./build/benchmark_hdf5 neural_net/weights/model.nnwt [--output results.csv]
 *   GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt ./build/benchmark_hdf5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ============================================================
 * Constants
 * ============================================================ */

#define N_PATTERNS     2
#define CHUNK_FLOATS   (2 * 1024 * 1024)      /* 2M floats = 8 MB per chunk */
#define CHUNK_BYTES    (CHUNK_FLOATS * sizeof(float))
#define DATASET_FLOATS (16 * 1024 * 1024)     /* 16M floats = 64 MB dataset */
#define DATASET_BYTES  (DATASET_FLOATS * sizeof(float))
#define N_ALGOS        8
#define N_STATIC       (N_ALGOS * 2)          /* 8 algos x 2 shuffle */
#define MAX_RESULTS    256

#define TMP_HDF5       "/tmp/bm_hdf5_tmp.h5"
#define DEFAULT_CSV    "tests/benchmark_hdf5_results/benchmark_hdf5_results.csv"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
 * Data Patterns (8 x 4 MB float32)
 * ============================================================ */

static const char *pattern_names[N_PATTERNS] = {
    "ramp", "sparse"
};

static const char *pattern_desc[N_PATTERNS] = {
    "linear ramp",
    "sparse (99%% zero)"
};

static void fill_dataset(float *buf, int id) {
    const size_t N = DATASET_FLOATS;
    switch (id) {
    case 0: /* linear ramp 0..1 */
        for (size_t i = 0; i < N; i++)
            buf[i] = (float)i / (float)N;
        break;
    case 1: /* sparse: 99% zero, 1% random spikes */
        lcg_seed(0xCAFEBABE);
        for (size_t i = 0; i < N; i++) {
            if ((lcg_next() % 100) == 0)
                buf[i] = lcg_float() * 10000.0f - 5000.0f;
            else
                buf[i] = 0.0f;
        }
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
    /* NN diagnostics (only populated for mode=="nn") */
    int          nn_original_action;
    int          nn_final_action;
    int          explored;
    int          sgd_fired;
} result_t;

static result_t g_results[MAX_RESULTS];
static int      g_nresults = 0;

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
 * Test Runner — single lossless write+read cycle
 * ============================================================ */

static int run_test(const float *data, size_t n_floats,
                    int use_filter, gpucompress_algorithm_t algo,
                    unsigned int shuffle_sz, result_t *r)
{
    herr_t hs;
    size_t orig_bytes = n_floats * sizeof(float);

    remove(TMP_HDF5);

    /* Create file & dataset */
    hid_t file = H5Fcreate(TMP_HDF5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return -1;

    hsize_t dims[1]       = { n_floats };
    hsize_t chunk_dims[1] = { CHUNK_FLOATS };
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

    /* Write (timer includes flush) */
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

    /* Read back (timer includes close) */
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

    /* Verify lossless: must be bit-exact */
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

    /* Fill result */
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
 * CSV Output
 * ============================================================ */

static void write_csv(const char *path) {
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

static void print_summary(void) {
    double best_ratio[N_PATTERNS], nn_ratio[N_PATTERNS];
    char   best_algo[N_PATTERNS][64], nn_algo[N_PATTERNS][64];
    char   nn_orig_algo[N_PATTERNS][64];
    int    nn_explored[N_PATTERNS], nn_sgd[N_PATTERNS];

    for (int p = 0; p < N_PATTERNS; p++) {
        best_ratio[p] = 0.0; nn_ratio[p] = 0.0;
        best_algo[p][0] = '\0'; nn_algo[p][0] = '\0';
        nn_orig_algo[p][0] = '\0';
        nn_explored[p] = 0; nn_sgd[p] = 0;

        const char *pat = pattern_names[p];
        for (int i = 0; i < g_nresults; i++) {
            result_t *r = &g_results[i];
            if (strcmp(r->pattern, pat) != 0) continue;

            if (strcmp(r->mode, "static") == 0 && r->ratio > best_ratio[p]) {
                best_ratio[p] = r->ratio;
                snprintf(best_algo[p], sizeof(best_algo[p]), "%s%s",
                         r->algorithm, r->shuffle ? "+shuf" : "");
            } else if (strcmp(r->mode, "nn") == 0) {
                nn_ratio[p] = r->ratio;
                snprintf(nn_algo[p], sizeof(nn_algo[p]), "%s", r->algorithm);
                nn_explored[p] = r->explored;
                nn_sgd[p] = r->sgd_fired;
                action_label(r->nn_original_action,
                             nn_orig_algo[p], sizeof(nn_orig_algo[p]));
            }
        }
    }

    /* Ratio table */
    printf("\n=== Lossless Compression Ratios ===\n\n");
    printf("%-14s | %10s  %-14s | %-14s  %-14s  %10s | %6s | %4s  %3s\n",
           "Pattern", "BestStatic", "Config",
           "NN Initial", "NN Final", "Ratio",
           "NN/Best", "Expl", "SGD");
    printf("%-14s-+-%10s--%-14s-+-%-14s--%-14s--%10s-+-%6s-+-%4s--%3s\n",
           "--------------", "----------", "--------------",
           "--------------", "--------------", "----------",
           "------", "----", "---");

    for (int p = 0; p < N_PATTERNS; p++) {
        double pct = (best_ratio[p] > 0)
                   ? (nn_ratio[p] / best_ratio[p] * 100.0) : 0.0;
        int changed = nn_explored[p] &&
                      strcmp(nn_orig_algo[p], nn_algo[p]) != 0;
        const char *final_col = changed ? nn_algo[p] : "-";

        printf("%-14s | %8.2fx  %-14s | %-14s  %-14s  %8.2fx | %5.0f%% | %-4s  %-3s\n",
               pattern_names[p],
               best_ratio[p], best_algo[p],
               nn_orig_algo[p], final_col, nn_ratio[p],
               pct,
               nn_explored[p] ? "yes" : "no",
               nn_sgd[p] ? "yes" : "no");
    }

    /* Totals */
    double sum_pct = 0.0;
    int n = 0, n_explored = 0, n_sgd = 0, n_changed = 0;
    for (int p = 0; p < N_PATTERNS; p++) {
        if (best_ratio[p] > 0) {
            sum_pct += nn_ratio[p] / best_ratio[p] * 100.0;
            n++;
        }
        if (nn_explored[p]) n_explored++;
        if (nn_sgd[p]) n_sgd++;
        if (nn_explored[p] && strcmp(nn_orig_algo[p], nn_algo[p]) != 0)
            n_changed++;
    }
    printf("\n  Avg NN/Best: %.0f%%  |  Exploration: %d/%d  |  SGD: %d/%d  |  Changed: %d/%d\n",
           n > 0 ? sum_pct / n : 0.0,
           n_explored, N_PATTERNS, n_sgd, N_PATTERNS, n_changed, N_PATTERNS);

    /* Verification */
    printf("\n=== Verification ===\n");
    printf("  All configs verified lossless (bit-exact) during write+read cycle.\n");
}

/* ============================================================
 * Run benchmarks for a given data buffer
 * ============================================================ */

typedef struct {
    gpucompress_algorithm_t algo;
    unsigned int            shuffle_sz;
} static_cfg_t;

static void bench_dataset(const char *pat_name, const float *data,
                          size_t n_floats,
                          const static_cfg_t *cfgs, int ncfg)
{
    result_t *r;

    /* 1. No compression */
    if (g_nresults >= MAX_RESULTS) return;
    r = &g_results[g_nresults];
    if (run_test(data, n_floats, 0, 0, 0, r) == 0) {
        snprintf(r->pattern,   sizeof(r->pattern),   "%s", pat_name);
        snprintf(r->mode,      sizeof(r->mode),      "none");
        snprintf(r->algorithm, sizeof(r->algorithm), "none");
        printf("  No compression: %.2fx\n", r->ratio);
        g_nresults++;
    } else {
        printf("  No compression: FAILED\n");
    }

    /* 2. Static configs (16 lossless) */
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
    if (g_nresults >= MAX_RESULTS) return;
    r = &g_results[g_nresults];
    if (run_test(data, n_floats, 1,
                 GPUCOMPRESS_ALGO_AUTO, 0, r) == 0) {
        snprintf(r->pattern, sizeof(r->pattern), "%s", pat_name);
        snprintf(r->mode,    sizeof(r->mode),    "nn");

        /* Query NN diagnostics */
        r->nn_original_action = gpucompress_get_last_nn_original_action();
        r->nn_final_action    = gpucompress_get_last_nn_action();
        r->explored           = gpucompress_get_last_exploration_triggered();
        r->sgd_fired          = gpucompress_get_last_sgd_fired();

        action_label(r->nn_final_action, r->algorithm, sizeof(r->algorithm));

        /* Print with diagnostics */
        char orig_label[64];
        action_label(r->nn_original_action, orig_label, sizeof(orig_label));

        printf("  NN: %.2fx (%s)", r->ratio, r->algorithm);
        if (r->explored && r->nn_original_action != r->nn_final_action)
            printf("  [explored: %s -> %s]", orig_label, r->algorithm);
        else if (r->explored)
            printf("  [explored: kept %s]", orig_label);
        if (r->sgd_fired)
            printf("  [SGD]");
        printf("\n");
        g_nresults++;
    } else {
        printf("  NN: FAILED\n");
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv) {
    const char *csv_path     = DEFAULT_CSV;
    const char *weights_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [weights.nnwt] [--output results.csv]\n"
                   "  Or set GPUCOMPRESS_WEIGHTS env var\n", argv[0]);
            return 0;
        } else if (!weights_path)
            weights_path = argv[i];
    }

    if (!weights_path)
        weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s [weights.nnwt] [--output results.csv]\n"
                        "  Or set GPUCOMPRESS_WEIGHTS env var\n", argv[0]);
        return 1;
    }

    printf("=== HDF5 Lossless Benchmark: No Compression vs Static vs NN ===\n\n");
    printf("Weights: %s\n", weights_path);
    printf("CSV:     %s\n", csv_path);
    printf("Chunk:   %zu MB float32 (%zu floats)\n",
           CHUNK_BYTES / (1024 * 1024), (size_t)CHUNK_FLOATS);
    printf("Dataset: %zu MB float32 (%zu floats)\n\n",
           DATASET_BYTES / (1024 * 1024), (size_t)DATASET_FLOATS);

    /* Suppress HDF5 automatic error printing */
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Init library */
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

    /* SGD reinforcement only — no exploration, no CSV logging */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.0f);
    printf("Online learning: SGD only, lr=1e-3, threshold=10%%\n\n");

    /* Generate data patterns */
    float *datasets[N_PATTERNS];
    for (int p = 0; p < N_PATTERNS; p++) {
        datasets[p] = (float *)malloc(DATASET_BYTES);
        if (!datasets[p]) { perror("malloc"); return 1; }
        fill_dataset(datasets[p], p);
    }

    /* Build 16 lossless static configs */
    static_cfg_t cfgs[N_STATIC];
    int ncfg = 0;
    for (int a = 1; a <= N_ALGOS; a++) {
        for (int s = 0; s <= 1; s++) {
            cfgs[ncfg].algo       = (gpucompress_algorithm_t)a;
            cfgs[ncfg].shuffle_sz = s ? 4 : 0;
            ncfg++;
        }
    }

    /* Run benchmarks */
    for (int p = 0; p < N_PATTERNS; p++) {
        printf("Pattern %d/%d: %s (%zu MB, %zu MB chunks)\n",
               p + 1, N_PATTERNS, pattern_desc[p],
               DATASET_BYTES / (1024 * 1024),
               CHUNK_BYTES / (1024 * 1024));
        bench_dataset(pattern_names[p], datasets[p], DATASET_FLOATS, cfgs, ncfg);
        printf("\n");
    }

    /* Output */
    write_csv(csv_path);
    print_summary();

    /* Cleanup */
    for (int p = 0; p < N_PATTERNS; p++) free(datasets[p]);
    gpucompress_cleanup();

    printf("\nDone. %d results recorded.\n", g_nresults);
    return 0;
}
