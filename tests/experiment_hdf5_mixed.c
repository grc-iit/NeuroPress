/**
 * @file experiment_hdf5_mixed.c
 * @brief HDF5 experiment with 8 extremely different 16MB chunks
 *
 * Each chunk has radically different data characteristics so the NN
 * sees different entropy/MAD/derivative and picks different algorithms.
 *
 * Chunk patterns (16 MB each, 4M floats):
 *   0: Constant (42.0)          — entropy≈0, MAD=0, deriv=0
 *   1: Smooth sine wave         — low entropy, smooth derivative
 *   2: Uniform random [0,1000]  — max entropy, high MAD
 *   3: Linear ramp              — moderate entropy, zero 2nd deriv
 *   4: Repeating short pattern  — very low entropy (4-value cycle)
 *   5: Gaussian noise N(0,1)    — high entropy, low MAD
 *   6: Sparse (99.9% zero)      — near-zero entropy, tiny MAD
 *   7: Step function (256 steps)— moderate entropy, spiky derivative
 *
 * Usage:
 *   GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt \
 *   GPUCOMPRESS_VERBOSE=1 \
 *   LD_LIBRARY_PATH=/tmp/lib:build ./build/experiment_hdf5_mixed
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

#define CHUNK_FLOATS  (4 * 1024 * 1024)   /* 16 MB per chunk */
#define N_CHUNKS      8
#define TOTAL_FLOATS  (CHUNK_FLOATS * N_CHUNKS)  /* 128 MB total */
#define TOTAL_BYTES   (TOTAL_FLOATS * sizeof(float))
#define HDF5_FILE     "/tmp/experiment_hdf5_mixed.h5"

/* Simple LCG PRNG (avoids rand() which varies across platforms) */
static uint32_t g_seed = 12345;
static float prng_uniform(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)(g_seed >> 8) / 16777216.0f;
}

/* Box-Muller for gaussian */
static float prng_gaussian(void) {
    float u1 = prng_uniform();
    float u2 = prng_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Generate 8 extremely different chunks
 * ============================================================ */
static void generate_mixed_data(float* buf) {
    size_t C = CHUNK_FLOATS;

    /* Chunk 0: Constant value — near-zero entropy */
    printf("  Chunk 0: constant (42.0)\n");
    for (size_t i = 0; i < C; i++)
        buf[i] = 42.0f;

    /* Chunk 1: Smooth sine wave — low entropy, smooth derivatives */
    printf("  Chunk 1: smooth sine wave\n");
    for (size_t i = 0; i < C; i++)
        buf[C + i] = sinf((float)i * 0.0001f) * 100.0f;

    /* Chunk 2: Uniform random [0, 1000] — maximum byte entropy */
    printf("  Chunk 2: uniform random [0, 1000]\n");
    g_seed = 99999;
    for (size_t i = 0; i < C; i++)
        buf[2*C + i] = prng_uniform() * 1000.0f;

    /* Chunk 3: Linear ramp — moderate entropy, zero 2nd derivative */
    printf("  Chunk 3: linear ramp\n");
    for (size_t i = 0; i < C; i++)
        buf[3*C + i] = (float)i * 0.001f;

    /* Chunk 4: Repeating short pattern [1,2,3,4] — extremely compressible */
    printf("  Chunk 4: repeating 4-value cycle\n");
    for (size_t i = 0; i < C; i++)
        buf[4*C + i] = (float)((i % 4) + 1);

    /* Chunk 5: Gaussian noise N(0,1) — high entropy, concentrated range */
    printf("  Chunk 5: gaussian noise N(0,1)\n");
    g_seed = 77777;
    for (size_t i = 0; i < C; i++)
        buf[5*C + i] = prng_gaussian();

    /* Chunk 6: Sparse — 99.9% zeros, occasional spikes */
    printf("  Chunk 6: sparse (99.9%% zero)\n");
    g_seed = 55555;
    for (size_t i = 0; i < C; i++) {
        if (prng_uniform() < 0.001f)
            buf[6*C + i] = prng_uniform() * 10000.0f;
        else
            buf[6*C + i] = 0.0f;
    }

    /* Chunk 7: Step function — 256 plateaus, sharp edges */
    printf("  Chunk 7: step function (256 levels)\n");
    {
        size_t step_size = C / 256;
        for (size_t i = 0; i < C; i++) {
            size_t level = i / step_size;
            buf[7*C + i] = (float)level * 3.9215686f; /* 0..1000 in 256 steps */
        }
    }

    printf("\n");
}

/* ============================================================
 * Per-chunk entropy (quick CPU estimate for display)
 * ============================================================ */
static double chunk_byte_entropy(const float* chunk, size_t n) {
    size_t hist[256];
    memset(hist, 0, sizeof(hist));
    const unsigned char* bytes = (const unsigned char*)chunk;
    size_t nbytes = n * sizeof(float);
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
 * Write dataset, read back, verify
 * ============================================================ */
static int run_test(
    const float* data,
    const char* dset_name,
    gpucompress_algorithm_t algo,
    unsigned int preproc,
    unsigned int shuffle_size,
    double error_bound
) {
    double t0, t1, t2, t3;

    hid_t file_id = H5Fopen(HDF5_FILE, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0)
        file_id = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "  FAIL: open/create\n"); return -1; }

    hsize_t dims[1] = {TOTAL_FLOATS};
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK_FLOATS};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    if (H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound) < 0) {
        fprintf(stderr, "  FAIL: H5Pset_gpucompress\n");
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file_id);
        return -1;
    }

    hid_t dset = H5Dcreate2(file_id, dset_name, H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        fprintf(stderr, "  FAIL: H5Dcreate2\n");
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file_id);
        return -1;
    }

    H5Z_gpucompress_reset_chunk_tracking();

    t0 = time_ms();
    herr_t status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data);
    t1 = time_ms();

    if (status < 0) {
        fprintf(stderr, "  FAIL: H5Dwrite\n");
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file_id);
        return -1;
    }

    H5Z_gpucompress_write_chunk_attr(dset);

    /* Print per-chunk algorithms */
    int nc = H5Z_gpucompress_get_chunk_count();
    printf("  Per-chunk algorithms:\n");
    const char* chunk_names[] = {
        "constant", "sine", "random", "ramp",
        "cycle", "gaussian", "sparse", "step"
    };
    for (int i = 0; i < nc && i < N_CHUNKS; i++) {
        int a = H5Z_gpucompress_get_chunk_algorithm(i);
        printf("    [%d] %-10s -> %s\n", i, chunk_names[i],
               gpucompress_algorithm_name((gpucompress_algorithm_t)a));
    }

    H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file_id);

    /* Read back */
    file_id = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    hsize_t storage = H5Dget_storage_size(dset);

    float* read_buf = (float*)malloc(TOTAL_BYTES);
    memset(read_buf, 0, TOTAL_BYTES);

    t2 = time_ms();
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, read_buf);
    t3 = time_ms();

    H5Dclose(dset); H5Fclose(file_id);

    if (status < 0) {
        fprintf(stderr, "  FAIL: H5Dread\n");
        free(read_buf);
        return -1;
    }

    /* Verify */
    int ok = 1;
    if (error_bound > 0.0) {
        double max_err = 0.0;
        for (size_t i = 0; i < TOTAL_FLOATS; i++) {
            double err = fabs((double)data[i] - (double)read_buf[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > error_bound * 1.01) { ok = 0; }
        printf("  Verify:  %s (max_err=%.2e, bound=%.2e)\n",
               ok ? "OK" : "FAIL", max_err, error_bound);
    } else {
        if (memcmp(data, read_buf, TOTAL_BYTES) != 0) {
            ok = 0;
            for (size_t i = 0; i < TOTAL_FLOATS; i++) {
                if (data[i] != read_buf[i]) {
                    fprintf(stderr, "  Mismatch at [%zu] chunk %zu: %.8g vs %.8g\n",
                            i, i / CHUNK_FLOATS, data[i], read_buf[i]);
                    break;
                }
            }
        }
        printf("  Verify:  %s (lossless)\n", ok ? "OK" : "FAIL");
    }

    double ratio = (storage > 0) ? (double)TOTAL_BYTES / (double)storage : 0.0;
    printf("  Storage: %llu KB (%.2fx)\n", (unsigned long long)storage / 1024, ratio);
    printf("  Write:   %.1f ms (%.0f MB/s)\n", t1 - t0,
           TOTAL_BYTES / (1024.0*1024.0) / ((t1-t0)/1000.0));
    printf("  Read:    %.1f ms (%.0f MB/s)\n", t3 - t2,
           TOTAL_BYTES / (1024.0*1024.0) / ((t3-t2)/1000.0));

    free(read_buf);
    return ok ? 0 : -1;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    int has_nn = (weights != NULL && weights[0] != '\0');

    printf("=== HDF5 Mixed-Data Experiment (128 MB, 8 x 16 MB chunks) ===\n");
    printf("NN weights: %s\n\n", has_nn ? weights : "(none)");

    /* Init */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n", gpucompress_error_string(rc));
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register\n");
        return 1;
    }

    /* Generate data */
    printf("Generating 8 chunks with extremely different patterns...\n");
    float* data = (float*)malloc(TOTAL_BYTES);
    if (!data) { perror("malloc"); return 1; }
    generate_mixed_data(data);

    /* Print per-chunk entropy */
    printf("Per-chunk byte entropy:\n");
    for (int c = 0; c < N_CHUNKS; c++) {
        double ent = chunk_byte_entropy(data + c * CHUNK_FLOATS, CHUNK_FLOATS);
        const char* names[] = {"constant","sine","random","ramp",
                               "cycle","gaussian","sparse","step"};
        printf("  [%d] %-10s entropy=%.4f bits\n", c, names[c], ent);
    }
    printf("\n");

    remove(HDF5_FILE);

    int pass = 0, fail = 0;

    /* Test 1: ALGO_AUTO + NN */
    if (has_nn && gpucompress_nn_is_loaded()) {
        printf("--- [1] ALGO_AUTO + NN (per-chunk selection) ---\n");
        if (run_test(data, "auto_nn", GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) == 0)
            pass++; else fail++;
        printf("\n");

        printf("--- [2] ALGO_AUTO + NN + shuffle ---\n");
        if (run_test(data, "auto_nn_shuffle", GPUCOMPRESS_ALGO_AUTO,
                     GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
            pass++; else fail++;
        printf("\n");
    } else {
        printf("--- [1] ALGO_AUTO (no NN — fallback to LZ4) ---\n");
        if (run_test(data, "auto_fallback", GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) == 0)
            pass++; else fail++;
        printf("\n");
    }

    /* Test 3: Manual best-for-each comparison baselines */
    printf("--- [3] LZ4 + shuffle (baseline) ---\n");
    if (run_test(data, "lz4_shuffle", GPUCOMPRESS_ALGO_LZ4,
                 GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    printf("--- [4] Zstd + shuffle (baseline) ---\n");
    if (run_test(data, "zstd_shuffle", GPUCOMPRESS_ALGO_ZSTD,
                 GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    printf("--- [5] Bitcomp (baseline) ---\n");
    if (run_test(data, "bitcomp", GPUCOMPRESS_ALGO_BITCOMP, 0, 0, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    printf("--- [6] Cascaded (baseline) ---\n");
    if (run_test(data, "cascaded", GPUCOMPRESS_ALGO_CASCADED, 0, 0, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    printf("=== Results: %d passed, %d failed ===\n", pass, fail);

    struct stat st;
    if (stat(HDF5_FILE, &st) == 0)
        printf("HDF5 file: %.1f MB (%s)\n", st.st_size / (1024.0*1024.0), HDF5_FILE);

    gpucompress_cleanup();
    free(data);
    return fail > 0 ? 1 : 0;
}
