/**
 * @file test_design6_chunk_tracker.c
 * @brief Verifies DESIGN-6 fix: dynamic chunk algorithm tracker handles >256 chunks.
 *
 * Before fix: g_chunk_algorithms[256] was a static array — any dataset with >256
 * chunks silently stopped recording algorithm choices after chunk 255.
 *
 * After fix: dynamic int* grown with realloc() — unbounded.
 *
 * Tests:
 *   1. 300-chunk round-trip: all 300 chunk algorithms tracked (proves array grew past 256)
 *   2. 512-chunk round-trip: scales to larger counts, all data verified byte-exact
 *   3. Reset reuse:          reset then re-track verifies g_chunk_count resets to 0
 *
 * Usage: ./build/test_design6_chunk_tracker [weights_path]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ── test framework ─────────────────────────────────────────── */
static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, fmt, ...) do {                        \
    if (!(cond)) {                                         \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__);        \
        g_fail++;                                          \
        return -1;                                         \
    }                                                      \
} while (0)

#define PASS() do { printf("  PASS\n"); g_pass++; return 0; } while(0)

/* ── helpers ────────────────────────────────────────────────── */

/* Chunk size in floats — small so we can create many chunks cheaply */
#define FLOATS_PER_CHUNK 1024   /* 4 KB */

/**
 * Create an HDF5 file with N 1-D chunks of FLOATS_PER_CHUNK floats each,
 * compress using the GPUCompress filter (ZSTD lossless), write, read back,
 * verify byte-exact, and check that the tracker recorded exactly N algorithms.
 *
 * Returns 0 on success, -1 on failure.
 */
static int run_chunk_test(int n_chunks, const char* test_name) {
    printf("[TEST] %s (%d chunks) ...\n", test_name, n_chunks);

    hsize_t total_floats = (hsize_t)n_chunks * FLOATS_PER_CHUNK;
    hsize_t chunk_floats = FLOATS_PER_CHUNK;

    /* Allocate and fill reference data */
    float* ref = (float*)malloc(total_floats * sizeof(float));
    ASSERT(ref != NULL, "malloc ref");
    for (hsize_t i = 0; i < total_floats; i++)
        ref[i] = (float)(i % 1000) * 0.001f;   /* deterministic ramp */

    float* readback = (float*)malloc(total_floats * sizeof(float));
    ASSERT(readback != NULL, "malloc readback");

    /* Build temp HDF5 file name */
    char fname[128];
    snprintf(fname, sizeof(fname), "/tmp/test_design6_%s.h5", test_name);

    /* ── Write ─────────────────────────────────────────────── */
    H5Z_gpucompress_reset_chunk_tracking();

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t fid  = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    ASSERT(fid >= 0, "H5Fcreate");

    hid_t space = H5Screate_simple(1, &total_floats, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, &chunk_floats);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    ASSERT(dset >= 0, "H5Dcreate2");

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, ref);
    ASSERT(rc >= 0, "H5Dwrite");

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);

    /* ── Check tracker ─────────────────────────────────────── */
    int tracked = H5Z_gpucompress_get_chunk_count();
    ASSERT(tracked == n_chunks,
           "chunk_count=%d expected=%d", tracked, n_chunks);

    /* Verify every tracked slot has a valid (non-negative) algorithm ID */
    int bad_slots = 0;
    for (int i = 0; i < tracked; i++) {
        if (H5Z_gpucompress_get_chunk_algorithm(i) < 0)
            bad_slots++;
    }
    ASSERT(bad_slots == 0, "%d slots returned -1 algorithm", bad_slots);

    /* Verify get_chunk_algorithm returns -1 for out-of-range index */
    ASSERT(H5Z_gpucompress_get_chunk_algorithm(tracked)     == -1,
           "expected -1 for index==tracked");
    ASSERT(H5Z_gpucompress_get_chunk_algorithm(-1)          == -1,
           "expected -1 for index==-1");
    ASSERT(H5Z_gpucompress_get_chunk_algorithm(tracked + 99) == -1,
           "expected -1 for large out-of-range index");

    /* Specifically check indices > 255 were recorded (the DESIGN-6 bug) */
    if (n_chunks > 256) {
        ASSERT(H5Z_gpucompress_get_chunk_algorithm(255) >= 0,
               "chunk 255 not tracked");
        ASSERT(H5Z_gpucompress_get_chunk_algorithm(256) >= 0,
               "chunk 256 not tracked (DESIGN-6 regression)");
        ASSERT(H5Z_gpucompress_get_chunk_algorithm(n_chunks - 1) >= 0,
               "last chunk not tracked");
    }

    /* ── Read back and verify ──────────────────────────────── */
    fid  = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT(fid >= 0, "H5Fopen");
    dset = H5Dopen2(fid, "data", H5P_DEFAULT);
    ASSERT(dset >= 0, "H5Dopen2");

    rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, readback);
    ASSERT(rc >= 0, "H5Dread");

    H5Dclose(dset);
    H5Fclose(fid);

    /* Byte-exact comparison */
    int mismatches = 0;
    for (hsize_t i = 0; i < total_floats; i++) {
        if (ref[i] != readback[i]) mismatches++;
    }
    ASSERT(mismatches == 0, "%d mismatches in round-trip", mismatches);

    remove(fname);
    free(ref);
    free(readback);
    PASS();
}

/**
 * Test 3: reset then re-track — verifies g_chunk_count resets to 0
 * and a second write re-populates from 0.
 */
static int test_reset_reuse(void) {
    printf("[TEST] reset_reuse (reset then write 10 chunks) ...\n");

    hsize_t n_chunks     = 10;
    hsize_t chunk_floats = FLOATS_PER_CHUNK;
    hsize_t total_floats = n_chunks * chunk_floats;

    float* buf = (float*)malloc(total_floats * sizeof(float));
    ASSERT(buf != NULL, "malloc");
    for (hsize_t i = 0; i < total_floats; i++) buf[i] = (float)i;

    /* First write: 300 chunks (to force array to grow) */
    {
        hsize_t big = 300 * chunk_floats;
        float* big_buf = (float*)malloc(big * sizeof(float));
        ASSERT(big_buf != NULL, "malloc big");
        for (hsize_t i = 0; i < big; i++) big_buf[i] = (float)i;

        H5Z_gpucompress_reset_chunk_tracking();
        hid_t fid  = H5Fcreate("/tmp/test_design6_reset_big.h5",
                                H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        ASSERT(fid >= 0, "H5Fcreate big");
        hid_t sp   = H5Screate_simple(1, &big, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, &chunk_floats);
        H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0);
        hid_t ds   = H5Dcreate2(fid, "d", H5T_NATIVE_FLOAT, sp,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, big_buf);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcpl); H5Fclose(fid);
        free(big_buf);
        remove("/tmp/test_design6_reset_big.h5");
    }
    ASSERT(H5Z_gpucompress_get_chunk_count() == 300, "pre-reset count should be 300");

    /* Reset */
    H5Z_gpucompress_reset_chunk_tracking();
    ASSERT(H5Z_gpucompress_get_chunk_count() == 0, "count after reset should be 0");

    /* Second write: 10 chunks */
    H5Z_gpucompress_reset_chunk_tracking();
    hid_t fid  = H5Fcreate("/tmp/test_design6_reset2.h5",
                            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT(fid >= 0, "H5Fcreate");
    hid_t sp   = H5Screate_simple(1, &total_floats, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, &chunk_floats);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0);
    hid_t ds   = H5Dcreate2(fid, "d", H5T_NATIVE_FLOAT, sp,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(ds); H5Sclose(sp); H5Pclose(dcpl); H5Fclose(fid);

    ASSERT((int)H5Z_gpucompress_get_chunk_count() == (int)n_chunks,
           "post-reset count=%d expected=%d",
           H5Z_gpucompress_get_chunk_count(), (int)n_chunks);

    remove("/tmp/test_design6_reset2.h5");
    free(buf);
    PASS();
}

/* ── main ───────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    printf("=== test_design6_chunk_tracker ===\n\n");

    /* Optional weights path for ALGO_AUTO (not needed here — using ZSTD) */
    if (argc > 1) setenv("GPUCOMPRESS_WEIGHTS", argv[1], 1);

    /* Register filter */
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: failed to register H5Zgpucompress filter\n");
        return 1;
    }

    run_chunk_test(300, "300_chunks");   /* crosses the old 256 boundary */
    run_chunk_test(512, "512_chunks");   /* 2× original limit */
    test_reset_reuse();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
