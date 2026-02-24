/**
 * @file experiment_hdf5_128mb.c
 * @brief HDF5 experiment: write 128MB float32 dataset with 16MB chunks
 *
 * Tests compression through the HDF5 filter pipeline with several configs:
 *   1. LZ4 (manual)
 *   2. Zstd (manual)
 *   3. LZ4 + 4-byte shuffle
 *   4. Zstd + 4-byte shuffle
 *   5. ALGO_AUTO without NN (fallback)
 *   6. ALGO_AUTO with NN (set GPUCOMPRESS_WEIGHTS env)
 *
 * Usage:
 *   # Without NN:
 *   LD_LIBRARY_PATH=/tmp/lib:build ./build/experiment_hdf5_128mb /tmp/test_128mb.bin
 *
 *   # With NN:
 *   GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt \
 *   LD_LIBRARY_PATH=/tmp/lib:build ./build/experiment_hdf5_128mb /tmp/test_128mb.bin
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

/* 128 MB of float32 = 32M floats */
#define TOTAL_FLOATS  (32 * 1024 * 1024)
#define TOTAL_BYTES   (TOTAL_FLOATS * sizeof(float))

/* 16 MB chunks = 4M floats per chunk => 8 chunks */
#define CHUNK_FLOATS  (4 * 1024 * 1024)
#define N_CHUNKS      (TOTAL_FLOATS / CHUNK_FLOATS)

#define HDF5_FILE     "/tmp/experiment_hdf5_128mb.h5"

/* ============================================================
 * Timing helper
 * ============================================================ */
static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Write a dataset with the given filter config, read back, verify
 * ============================================================ */
static int run_config(
    const float* data,
    const char* dset_name,
    gpucompress_algorithm_t algo,
    unsigned int preproc,
    unsigned int shuffle_size,
    double error_bound
) {
    int ret = 0;
    double t0, t1, t2, t3;

    /* --- Write --- */
    hid_t file_id = H5Fopen(HDF5_FILE, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        /* First dataset: create the file */
        file_id = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (file_id < 0) {
        fprintf(stderr, "  FAIL: cannot open/create %s\n", HDF5_FILE);
        return -1;
    }

    hsize_t dims[1] = {TOTAL_FLOATS};
    hid_t dspace_id = H5Screate_simple(1, dims, NULL);

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK_FLOATS};
    H5Pset_chunk(dcpl_id, 1, chunk_dims);

    herr_t status = H5Pset_gpucompress(dcpl_id, algo, preproc, shuffle_size, error_bound);
    if (status < 0) {
        fprintf(stderr, "  FAIL: H5Pset_gpucompress\n");
        H5Pclose(dcpl_id); H5Sclose(dspace_id); H5Fclose(file_id);
        return -1;
    }

    hid_t dset_id = H5Dcreate2(file_id, dset_name, H5T_NATIVE_FLOAT,
                                dspace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (dset_id < 0) {
        fprintf(stderr, "  FAIL: H5Dcreate2 for '%s'\n", dset_name);
        H5Pclose(dcpl_id); H5Sclose(dspace_id); H5Fclose(file_id);
        return -1;
    }

    H5Z_gpucompress_reset_chunk_tracking();

    t0 = time_ms();
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    t1 = time_ms();

    if (status < 0) {
        fprintf(stderr, "  FAIL: H5Dwrite for '%s'\n", dset_name);
        H5Dclose(dset_id); H5Pclose(dcpl_id); H5Sclose(dspace_id); H5Fclose(file_id);
        return -1;
    }

    /* Write per-chunk algorithm attribute */
    H5Z_gpucompress_write_chunk_attr(dset_id);

    /* Print per-chunk algorithms */
    int n_chunks = H5Z_gpucompress_get_chunk_count();
    printf("  Chunks: ");
    for (int i = 0; i < n_chunks; i++) {
        int a = H5Z_gpucompress_get_chunk_algorithm(i);
        printf("%s%s", (i > 0) ? ", " : "",
               gpucompress_algorithm_name((gpucompress_algorithm_t)a));
    }
    printf(" (%d chunks)\n", n_chunks);

    H5Dclose(dset_id);
    H5Pclose(dcpl_id);
    H5Sclose(dspace_id);
    H5Fclose(file_id);

    /* --- Get file-level storage size --- */
    file_id = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    hsize_t storage = H5Dget_storage_size(dset_id);

    /* --- Read back --- */
    float* read_buf = (float*)malloc(TOTAL_BYTES);
    if (!read_buf) {
        fprintf(stderr, "  FAIL: malloc\n");
        H5Dclose(dset_id); H5Fclose(file_id);
        return -1;
    }
    memset(read_buf, 0, TOTAL_BYTES);

    t2 = time_ms();
    status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, read_buf);
    t3 = time_ms();

    H5Dclose(dset_id);
    H5Fclose(file_id);

    if (status < 0) {
        fprintf(stderr, "  FAIL: H5Dread for '%s'\n", dset_name);
        free(read_buf);
        return -1;
    }

    /* --- Verify --- */
    const char* verify_str = "OK";
    if (error_bound > 0.0) {
        double max_err = 0.0;
        for (size_t i = 0; i < TOTAL_FLOATS; i++) {
            double err = fabs((double)data[i] - (double)read_buf[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > error_bound * 1.01) {
            verify_str = "BOUND_EXCEEDED";
            ret = -1;
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "OK (max_err=%.2e)", max_err);
            printf("  Verify: %s\n", buf);
        }
    } else {
        if (memcmp(data, read_buf, TOTAL_BYTES) != 0) {
            verify_str = "MISMATCH";
            ret = -1;
            /* Find first mismatch */
            for (size_t i = 0; i < TOTAL_FLOATS; i++) {
                if (data[i] != read_buf[i]) {
                    fprintf(stderr, "  First mismatch at [%zu]: wrote %.8g, read %.8g\n",
                            i, data[i], read_buf[i]);
                    break;
                }
            }
        }
    }

    double ratio = (storage > 0) ? (double)TOTAL_BYTES / (double)storage : 0.0;
    double write_ms = t1 - t0;
    double read_ms = t3 - t2;
    double write_tp = TOTAL_BYTES / (1024.0 * 1024.0) / (write_ms / 1000.0);
    double read_tp = TOTAL_BYTES / (1024.0 * 1024.0) / (read_ms / 1000.0);

    printf("  Storage: %llu KB (%.2fx ratio)\n", (unsigned long long)storage / 1024, ratio);
    printf("  Write:   %.1f ms (%.0f MB/s)\n", write_ms, write_tp);
    printf("  Read:    %.1f ms (%.0f MB/s)\n", read_ms, read_tp);
    if (ret == 0 && error_bound <= 0.0)
        printf("  Verify:  %s (lossless)\n", verify_str);
    else if (ret != 0)
        printf("  Verify:  %s\n", verify_str);

    free(read_buf);
    return ret;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_128mb.bin>\n", argv[0]);
        return 1;
    }

    /* Read input file */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if ((size_t)file_size < TOTAL_BYTES) {
        fprintf(stderr, "Input file too small: %ld bytes, need %zu\n",
                file_size, (size_t)TOTAL_BYTES);
        fclose(f);
        return 1;
    }

    float* data = (float*)malloc(TOTAL_BYTES);
    if (!data) { perror("malloc"); fclose(f); return 1; }
    size_t nread = fread(data, 1, TOTAL_BYTES, f);
    fclose(f);

    if (nread != TOTAL_BYTES) {
        fprintf(stderr, "Short read: %zu / %zu\n", nread, (size_t)TOTAL_BYTES);
        free(data);
        return 1;
    }

    /* Check if NN weights are available */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    int has_nn = (weights != NULL && weights[0] != '\0');

    printf("=== HDF5 GPUCompress 128MB Experiment ===\n");
    printf("Input:      %s (%.1f MB)\n", argv[1], TOTAL_BYTES / (1024.0 * 1024.0));
    printf("Chunks:     %d x %d MB = %d MB\n", N_CHUNKS,
           (int)(CHUNK_FLOATS * sizeof(float) / (1024 * 1024)), (int)(TOTAL_BYTES / (1024 * 1024)));
    printf("HDF5 file:  %s\n", HDF5_FILE);
    printf("NN weights: %s\n\n", has_nn ? weights : "(none — ALGO_AUTO will fallback to LZ4)");

    /* Initialize library */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n", gpucompress_error_string(rc));
        free(data);
        return 1;
    }

    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        free(data);
        return 1;
    }

    /* Remove old file */
    remove(HDF5_FILE);

    int pass = 0, fail = 0;

    /* --- Config 1: LZ4 (no preprocessing) --- */
    printf("--- [1] LZ4 (lossless, no preprocessing) ---\n");
    if (run_config(data, "lz4_plain", GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    /* --- Config 2: Zstd (no preprocessing) --- */
    printf("--- [2] Zstd (lossless, no preprocessing) ---\n");
    if (run_config(data, "zstd_plain", GPUCOMPRESS_ALGO_ZSTD, 0, 0, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    /* --- Config 3: LZ4 + 4-byte shuffle --- */
    printf("--- [3] LZ4 + 4-byte shuffle ---\n");
    if (run_config(data, "lz4_shuffle", GPUCOMPRESS_ALGO_LZ4,
                   GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    /* --- Config 4: Zstd + 4-byte shuffle --- */
    printf("--- [4] Zstd + 4-byte shuffle ---\n");
    if (run_config(data, "zstd_shuffle", GPUCOMPRESS_ALGO_ZSTD,
                   GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
        pass++; else fail++;
    printf("\n");

    /* --- Config 5: ALGO_AUTO (without NN = LZ4 fallback) --- */
    if (!has_nn) {
        printf("--- [5] ALGO_AUTO (no NN — fallback) ---\n");
        if (run_config(data, "auto_no_nn", GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) == 0)
            pass++; else fail++;
        printf("\n");
    }

    /* --- Config 6: ALGO_AUTO with NN --- */
    if (has_nn) {
        if (gpucompress_nn_is_loaded()) {
            printf("--- [5] ALGO_AUTO + NN (per-chunk selection) ---\n");
            if (run_config(data, "auto_nn", GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) == 0)
                pass++; else fail++;
            printf("\n");

            printf("--- [6] ALGO_AUTO + NN + 4-byte shuffle ---\n");
            if (run_config(data, "auto_nn_shuffle", GPUCOMPRESS_ALGO_AUTO,
                           GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0) == 0)
                pass++; else fail++;
            printf("\n");
        } else {
            printf("--- [5/6] NN weights file set but not loaded, skipping ---\n\n");
        }
    }

    /* --- Config 7: LZ4 + quantization (lossy) --- */
    printf("--- [%d] LZ4 + quantization (error_bound=0.001) ---\n", has_nn ? 7 : 6);
    if (run_config(data, "lz4_quant", GPUCOMPRESS_ALGO_LZ4,
                   GPUCOMPRESS_PREPROC_QUANTIZE, 0, 0.001) == 0)
        pass++; else fail++;
    printf("\n");

    /* Summary */
    printf("=== Results: %d passed, %d failed ===\n", pass, fail);

    /* Show file size */
    struct stat st;
    if (stat(HDF5_FILE, &st) == 0) {
        printf("HDF5 file size: %.1f MB (%s)\n",
               st.st_size / (1024.0 * 1024.0), HDF5_FILE);
    }

    gpucompress_cleanup();
    free(data);
    return fail > 0 ? 1 : 0;
}
