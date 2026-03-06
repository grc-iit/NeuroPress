/**
 * @file test_hdf5_filter.c
 * @brief HDF5 filter integration test for GPUCompress
 *
 * Creates a 16 MB float32 dataset with 4 MB chunks, compressed through
 * the gpucompress HDF5 filter using ALGO_AUTO (NN-based per-chunk
 * algorithm selection). Reads back and verifies lossless round-trip.
 *
 * Build:
 *   cmake --build build --target test_hdf5_filter
 *
 * Run:
 *   GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt \
 *   LD_LIBRARY_PATH=/tmp/lib:build ./build/test_hdf5_filter
 *
 * Inspect:
 *   h5dump -pH test_hdf5_filter.h5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ---------------- Configuration ---------------- */
#define TOTAL_FLOATS  (4 * 1024 * 1024)   /* 16 MB / 4 = 4M floats */
#define CHUNK_FLOATS  (1 * 1024 * 1024)   /* 4 MB / 4 = 1M floats */
#define TEST_FILE     "test_hdf5_filter.h5"
#define DATASET_NAME  "auto_mixed"

/* ---------------- Data Generator --------------- */

/**
 * Fill each 4 MB chunk with a distinct data pattern so the NN
 * observes different entropy/MAD/derivative characteristics per chunk
 * and selects different compression algorithms.
 *
 * Chunk 0: constant value  — near-zero entropy, zero MAD
 * Chunk 1: smooth sine     — low entropy, low derivative
 * Chunk 2: random uniform  — high entropy, high MAD
 * Chunk 3: linear ramp     — moderate entropy, zero 2nd derivative
 */
static void gen_mixed(float* buf, size_t n) {
    size_t chunk = CHUNK_FLOATS;

    /* Chunk 0: constant */
    for (size_t i = 0; i < chunk && i < n; i++)
        buf[i] = 42.0f;

    /* Chunk 1: smooth sine wave */
    for (size_t i = chunk; i < 2 * chunk && i < n; i++)
        buf[i] = sinf((float)(i - chunk) * 0.001f) * 100.0f;

    /* Chunk 2: random uniform */
    srand(12345);
    for (size_t i = 2 * chunk; i < 3 * chunk && i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX * 1000.0f;

    /* Chunk 3: linear ramp */
    for (size_t i = 3 * chunk; i < 4 * chunk && i < n; i++)
        buf[i] = (float)(i - 3 * chunk) * 0.01f;
}

/* ---------------- Main ------------------------- */

int main(void) {
    int ret = 0;

    float* write_buf = (float*)malloc(TOTAL_FLOATS * sizeof(float));
    float* read_buf  = (float*)malloc(TOTAL_FLOATS * sizeof(float));
    if (!write_buf || !read_buf) {
        fprintf(stderr, "FATAL: malloc failed\n");
        return 1;
    }

    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed: %d (%s)\n",
                rc, gpucompress_error_string(rc));
        free(write_buf); free(read_buf);
        return 1;
    }

    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        free(write_buf); free(read_buf);
        return 1;
    }

    printf("=== HDF5 GPUCompress Filter Test ===\n");
    printf("  Data size : 16 MB (%d floats)\n", TOTAL_FLOATS);
    printf("  Chunk size: 4 MB (%d floats)\n", CHUNK_FLOATS);
    printf("  Algorithm : ALGO_AUTO (NN per-chunk selection)\n");
    printf("  Test file : %s\n\n", TEST_FILE);

    /* Generate mixed data: each chunk has different characteristics */
    gen_mixed(write_buf, TOTAL_FLOATS);

    /* ---- Write ---- */
    printf("--- Writing dataset: %s ---\n", DATASET_NAME);

    hid_t file_id = H5Fcreate(TEST_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "FATAL: H5Fcreate failed\n");
        ret = 1; goto cleanup;
    }

    hsize_t dims[1] = {TOTAL_FLOATS};
    hid_t dspace_id = H5Screate_simple(1, dims, NULL);

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK_FLOATS};
    H5Pset_chunk(dcpl_id, 1, chunk_dims);

    herr_t status = H5Pset_gpucompress(dcpl_id, GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0);
    if (status < 0) {
        fprintf(stderr, "FAIL: H5Pset_gpucompress\n");
        H5Pclose(dcpl_id); H5Sclose(dspace_id); H5Fclose(file_id);
        ret = 1; goto cleanup;
    }

    hid_t dset_id = H5Dcreate2(file_id, DATASET_NAME, H5T_NATIVE_FLOAT,
                                dspace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (dset_id < 0) {
        fprintf(stderr, "FAIL: H5Dcreate2\n");
        H5Pclose(dcpl_id); H5Sclose(dspace_id); H5Fclose(file_id);
        ret = 1; goto cleanup;
    }

    H5Z_gpucompress_reset_chunk_tracking();

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, write_buf);

    /* Write per-chunk algorithm attribute before closing dataset */
    if (status >= 0) {
        H5Z_gpucompress_write_chunk_attr(dset_id);
    }

    H5Dclose(dset_id);
    H5Pclose(dcpl_id);
    H5Sclose(dspace_id);
    H5Fclose(file_id);

    if (status < 0) {
        fprintf(stderr, "FAIL: H5Dwrite\n");
        ret = 1; goto cleanup;
    }
    printf("  Write OK\n\n");

    /* ---- Read back and verify ---- */
    printf("--- Reading and verifying ---\n");

    file_id = H5Fopen(TEST_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "FAIL: H5Fopen\n");
        ret = 1; goto cleanup;
    }

    dset_id = H5Dopen2(file_id, DATASET_NAME, H5P_DEFAULT);
    if (dset_id < 0) {
        fprintf(stderr, "FAIL: H5Dopen2\n");
        H5Fclose(file_id);
        ret = 1; goto cleanup;
    }

    memset(read_buf, 0, TOTAL_FLOATS * sizeof(float));
    status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, read_buf);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    if (status < 0) {
        fprintf(stderr, "FAIL: H5Dread\n");
        ret = 1; goto cleanup;
    }

    /* Lossless verification */
    for (size_t i = 0; i < TOTAL_FLOATS; i++) {
        if (write_buf[i] != read_buf[i]) {
            fprintf(stderr, "FAIL: mismatch at index %zu: wrote %.8g, read %.8g\n",
                    i, write_buf[i], read_buf[i]);
            ret = 1; goto cleanup;
        }
    }

    printf("  PASS: %d floats match exactly (lossless round-trip)\n", TOTAL_FLOATS);
    printf("\nHDF5 file kept at: %s\n", TEST_FILE);

cleanup:
    gpucompress_cleanup();
    free(write_buf);
    free(read_buf);
    return ret;
}
