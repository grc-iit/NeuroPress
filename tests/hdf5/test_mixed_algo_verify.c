/**
 * @file test_mixed_algo_verify.c
 * @brief Verify lossless round-trip when NN picks different algorithms per chunk
 *
 * Creates a single HDF5 dataset with 8 chunks (4 MB each), where each chunk
 * has radically different data characteristics to force the NN to choose
 * different compression algorithms. Then reads back the entire dataset and
 * verifies bit-exact match per chunk.
 *
 * Usage:
 *   ./build/test_mixed_algo_verify neural_net/weights/model.nnwt
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

#define N_CHUNKS       8
#define CHUNK_FLOATS   (1024 * 1024)             /* 1M floats = 4 MB */
#define CHUNK_BYTES    (CHUNK_FLOATS * sizeof(float))
#define TOTAL_FLOATS   (CHUNK_FLOATS * N_CHUNKS) /* 8M floats = 32 MB */
#define TOTAL_BYTES    (TOTAL_FLOATS * sizeof(float))
#define HDF5_FILE      "/tmp/test_mixed_algo.h5"
#define DUMP_ORIGINAL  "/tmp/test_mixed_original.txt"
#define DUMP_READBACK  "/tmp/test_mixed_readback.txt"
#define DUMP_SAMPLES   32   /* floats to dump from head and tail of each chunk */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * PRNG
 * ============================================================ */

static uint32_t lcg_state;
static void     lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void)       { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
static float    lcg_float(void)      { return (float)(lcg_next() >> 8) / 16777216.0f; }
static float    lcg_gaussian(void) {
    float u1 = lcg_float() * 0.9999f + 0.0001f;
    float u2 = lcg_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ============================================================
 * Timing
 * ============================================================ */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Chunk names and descriptions
 * ============================================================ */

static const char *chunk_names[N_CHUNKS] = {
    "constant",    "sine",      "random",   "ramp",
    "sparse",      "gaussian",  "cycle",    "step"
};

/* ============================================================
 * Generate 8 chunks with radically different characteristics
 * ============================================================ */

static void generate_data(float *buf) {
    size_t C = CHUNK_FLOATS;

    /* Chunk 0: Constant — entropy ~0, perfectly compressible */
    for (size_t i = 0; i < C; i++)
        buf[i] = 42.0f;

    /* Chunk 1: Smooth sine — low entropy, smooth derivative */
    for (size_t i = 0; i < C; i++)
        buf[C + i] = 1000.0f * sinf(2.0f * (float)M_PI * i / (float)C);

    /* Chunk 2: Uniform random [-1000, 1000] — max entropy */
    lcg_seed(0xDEADBEEF);
    for (size_t i = 0; i < C; i++)
        buf[2*C + i] = lcg_float() * 2000.0f - 1000.0f;

    /* Chunk 3: Linear ramp 0..1 — moderate entropy */
    for (size_t i = 0; i < C; i++)
        buf[3*C + i] = (float)i / (float)C;

    /* Chunk 4: Sparse — 99.9% zero, rare spikes */
    lcg_seed(0xCAFEBABE);
    for (size_t i = 0; i < C; i++) {
        if ((lcg_next() % 1000) == 0)
            buf[4*C + i] = lcg_float() * 10000.0f - 5000.0f;
        else
            buf[4*C + i] = 0.0f;
    }

    /* Chunk 5: Gaussian noise N(0, 500) — high entropy, concentrated range */
    lcg_seed(0x12345678);
    for (size_t i = 0; i < C; i++)
        buf[5*C + i] = lcg_gaussian() * 500.0f;

    /* Chunk 6: Repeating 4-value cycle — extremely compressible */
    for (size_t i = 0; i < C; i++)
        buf[6*C + i] = (float)((i % 4) + 1);

    /* Chunk 7: Step function — 256 plateaus, sharp edges */
    for (size_t i = 0; i < C; i++)
        buf[7*C + i] = (float)(i / (C / 256)) * 3.9215686f;
}

/* ============================================================
 * Per-chunk byte entropy (for display)
 * ============================================================ */

static double byte_entropy(const float *chunk, size_t nfloats) {
    size_t hist[256];
    memset(hist, 0, sizeof(hist));
    const unsigned char *bytes = (const unsigned char *)chunk;
    size_t nbytes = nfloats * sizeof(float);
    for (size_t i = 0; i < nbytes; i++)
        hist[bytes[i]]++;
    double ent = 0.0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] == 0) continue;
        double p = (double)hist[i] / nbytes;
        ent -= p * log2(p);
    }
    return ent;
}

/* ============================================================
 * Dump data to text file (head + tail samples per chunk)
 * ============================================================ */

static void dump_data(const char *path, const char *label, const float *data) {
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return; }

    fprintf(fp, "# %s\n", label);
    fprintf(fp, "# %d chunks x %d floats (%zu MB total)\n",
            N_CHUNKS, CHUNK_FLOATS, TOTAL_BYTES / (1024*1024));
    fprintf(fp, "# Showing first %d and last %d floats per chunk\n\n",
            DUMP_SAMPLES, DUMP_SAMPLES);

    for (int c = 0; c < N_CHUNKS; c++) {
        size_t base = (size_t)c * CHUNK_FLOATS;
        fprintf(fp, "=== Chunk %d: %s (%d floats) ===\n", c, chunk_names[c], CHUNK_FLOATS);

        fprintf(fp, "  [HEAD] index 0..%d:\n", DUMP_SAMPLES - 1);
        for (int i = 0; i < DUMP_SAMPLES; i++)
            fprintf(fp, "    [%d] %.10g\n", i, data[base + i]);

        fprintf(fp, "  [TAIL] index %d..%d:\n",
                CHUNK_FLOATS - DUMP_SAMPLES, CHUNK_FLOATS - 1);
        for (int i = CHUNK_FLOATS - DUMP_SAMPLES; i < CHUNK_FLOATS; i++)
            fprintf(fp, "    [%d] %.10g\n", i, data[base + i]);

        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("  Dumped to: %s\n", path);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv) {
    const char *weights_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!weights_path) weights_path = argv[i];
    }
    if (!weights_path)
        weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s <weights.nnwt>\n", argv[0]);
        return 1;
    }

    printf("=== Mixed-Algorithm Lossless Verification ===\n\n");
    printf("Weights:  %s\n", weights_path);
    printf("Dataset:  %d chunks x %zu MB = %zu MB\n",
           N_CHUNKS, CHUNK_BYTES / (1024*1024), TOTAL_BYTES / (1024*1024));
    printf("Mode:     ALGO_AUTO (NN picks algorithm per chunk)\n\n");

    /* Suppress HDF5 error printing */
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Init library */
    gpucompress_error_t rc = gpucompress_init(weights_path);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded\n");
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        return 1;
    }

    /* Generate test data */
    printf("Generating %d chunks with different data patterns...\n", N_CHUNKS);
    float *original = (float *)malloc(TOTAL_BYTES);
    if (!original) { perror("malloc"); return 1; }
    generate_data(original);

    /* Print per-chunk entropy */
    printf("\n  %-3s  %-10s  %7s\n", "#", "Pattern", "Entropy");
    printf("  ---  ----------  -------\n");
    for (int c = 0; c < N_CHUNKS; c++) {
        double ent = byte_entropy(original + c * CHUNK_FLOATS, CHUNK_FLOATS);
        printf("  [%d]  %-10s  %.4f bits\n", c, chunk_names[c], ent);
    }

    /* ========================================
     * WRITE: single dataset, all 8 chunks
     * ======================================== */
    printf("\n--- Writing dataset with ALGO_AUTO ---\n");
    remove(HDF5_FILE);

    hid_t file = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[1]       = { TOTAL_FLOATS };
    hsize_t chunk_dims[1] = { CHUNK_FLOATS };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0);

    hid_t dset = H5Dcreate2(file, "mixed_data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Z_gpucompress_reset_chunk_tracking();

    double t_write_start = time_ms();
    herr_t hs = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, original);
    double t_write_end = time_ms();

    if (hs < 0) {
        fprintf(stderr, "FATAL: H5Dwrite failed\n");
        return 1;
    }

    /* Write per-chunk algo attribute */
    H5Z_gpucompress_write_chunk_attr(dset);

    /* Print per-chunk algorithm choices */
    int nc = H5Z_gpucompress_get_chunk_count();
    printf("\n  Per-chunk algorithm choices (NN):\n");
    printf("  %-3s  %-10s  %-16s\n", "#", "Pattern", "Algorithm");
    printf("  ---  ----------  ----------------\n");

    int n_unique_algos = 0;
    int seen_algos[32];
    memset(seen_algos, 0, sizeof(seen_algos));

    for (int c = 0; c < nc && c < N_CHUNKS; c++) {
        int a = H5Z_gpucompress_get_chunk_algorithm(c);
        printf("  [%d]  %-10s  %s\n", c, chunk_names[c],
               gpucompress_algorithm_name((gpucompress_algorithm_t)a));
        if (a >= 0 && a < 32 && !seen_algos[a]) {
            seen_algos[a] = 1;
            n_unique_algos++;
        }
    }

    hsize_t storage = H5Dget_storage_size(dset);
    double ratio = (storage > 0) ? (double)TOTAL_BYTES / (double)storage : 0.0;

    printf("\n  Unique algorithms used: %d\n", n_unique_algos);
    printf("  Write time:   %.1f ms (%.0f MB/s)\n",
           t_write_end - t_write_start,
           (TOTAL_BYTES / (1024.0*1024.0)) / ((t_write_end - t_write_start) / 1000.0));
    printf("  Storage:      %.1f MB (%.2fx compression)\n",
           storage / (1024.0*1024.0), ratio);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file);

    /* ========================================
     * READ: read back entire dataset
     * ======================================== */
    printf("\n--- Reading back and verifying ---\n");

    file = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, "mixed_data", H5P_DEFAULT);

    float *readback = (float *)malloc(TOTAL_BYTES);
    if (!readback) { perror("malloc"); return 1; }
    memset(readback, 0xAA, TOTAL_BYTES);  /* fill with garbage to ensure real read */

    double t_read_start = time_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, readback);
    double t_read_end = time_ms();

    H5Dclose(dset);
    H5Fclose(file);

    if (hs < 0) {
        fprintf(stderr, "FATAL: H5Dread failed\n");
        return 1;
    }

    printf("  Read time:    %.1f ms (%.0f MB/s)\n",
           t_read_end - t_read_start,
           (TOTAL_BYTES / (1024.0*1024.0)) / ((t_read_end - t_read_start) / 1000.0));

    /* ========================================
     * VERIFY: per-chunk bit-exact comparison
     * ======================================== */
    printf("\n--- Per-chunk lossless verification ---\n\n");
    printf("  %-3s  %-10s  %10s  %s\n", "#", "Pattern", "Floats", "Status");
    printf("  ---  ----------  ----------  ------\n");

    int all_pass = 1;
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t offset = (size_t)c * CHUNK_FLOATS;
        int chunk_ok = 1;
        size_t first_mismatch = 0;

        if (memcmp(original + offset, readback + offset, CHUNK_BYTES) != 0) {
            chunk_ok = 0;
            all_pass = 0;
            /* Find first mismatch */
            for (size_t j = 0; j < CHUNK_FLOATS; j++) {
                if (original[offset + j] != readback[offset + j]) {
                    first_mismatch = j;
                    break;
                }
            }
        }

        if (chunk_ok) {
            printf("  [%d]  %-10s  %10zu  PASS (bit-exact)\n",
                   c, chunk_names[c], (size_t)CHUNK_FLOATS);
        } else {
            printf("  [%d]  %-10s  %10zu  FAIL at offset %zu: "
                   "expected %.8g, got %.8g\n",
                   c, chunk_names[c], (size_t)CHUNK_FLOATS,
                   first_mismatch,
                   original[offset + first_mismatch],
                   readback[offset + first_mismatch]);
        }
    }

    /* Dump original and readback data to text files */
    printf("\n--- Dumping data samples ---\n");
    dump_data(DUMP_ORIGINAL, "ORIGINAL data (before compression)", original);
    dump_data(DUMP_READBACK, "READBACK data (after decompress)", readback);

    /* Final verdict */
    printf("\n========================================\n");
    if (all_pass && n_unique_algos > 1) {
        printf("  RESULT: PASS\n");
        printf("  %d chunks, %d different algorithms, all bit-exact.\n",
               N_CHUNKS, n_unique_algos);
        printf("  Mixed-algorithm decompression verified.\n");
    } else if (all_pass && n_unique_algos <= 1) {
        printf("  RESULT: PASS (but only %d unique algorithm used)\n",
               n_unique_algos);
        printf("  Lossless OK, but NN chose the same algo for all chunks.\n");
        printf("  Try patterns with more diverse characteristics.\n");
    } else {
        printf("  RESULT: FAIL\n");
        printf("  One or more chunks did NOT round-trip correctly.\n");
    }
    printf("========================================\n");

    /* Cleanup */
    free(original);
    free(readback);
    // remove(HDF5_FILE);
    gpucompress_cleanup();

    return all_pass ? 0 : 1;
}
