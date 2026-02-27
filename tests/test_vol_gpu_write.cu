/**
 * test_vol_gpu_write.cu
 *
 * Validates the GPUCompress HDF5 VOL connector end-to-end:
 *   1. GPU pointer write path: H5Dwrite(d_ptr) → GPU compress → H5Dwrite_chunk
 *   2. GPU pointer read path:  H5Dread(d_ptr) → H5Dread_chunk → GPU decompress
 *   3. Host pointer fallback:  H5Dwrite/H5Dread(h_ptr) → native HDF5 filter path
 *   4. Multi-chunk 1D dataset
 *   5. 2D chunked dataset
 *   6. Lossy (quantized) round-trip
 *
 * Links against HDF5 2.x (for VOL API) — does NOT link H5Zgpucompress.
 * Filter cd_values are set directly via H5Pset_filter().
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Filter constants (mirror of H5Zgpucompress.h)                        */
/* ------------------------------------------------------------------ */
#define H5Z_FILTER_GPUCOMPRESS   305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Encode a double into two uint32s (little-endian IEEE 754) */
static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/**
 * Set gpucompress filter on a DCPL.
 *   algorithm: GPUCOMPRESS_ALGO_* value (int)
 *   preprocessing: 0=none, 2=shuffle4, 18=shuffle4+quantize
 *   shuffle_size: 4 for float32
 *   error_bound: 0.0 for lossless
 */
static herr_t set_gpucompress_filter(hid_t dcpl,
                                     unsigned int algorithm,
                                     unsigned int preprocessing,
                                     unsigned int shuffle_size,
                                     double error_bound)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algorithm;
    cd[1] = preprocessing;
    cd[2] = shuffle_size;
    pack_double(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* ------------------------------------------------------------------ */
/* Test infrastructure                                                  */
/* ------------------------------------------------------------------ */

static int pass_count = 0;
static int fail_count = 0;

#define PASS() do { printf("  PASS\n"); pass_count++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); fail_count++; } while(0)

static void fill_pattern(float* h, size_t n, float seed) {
    for (size_t i = 0; i < n; i++)
        h[i] = seed + (float)(i % 256) * 0.01f;
}

static int verify_close(const float* a, const float* b, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "    mismatch at [%zu]: %.6f vs %.6f (tol=%.6f)\n",
                    i, a[i], b[i], tol);
            return 0;
        }
    }
    return 1;
}

/* Open/create a VOL-enabled HDF5 file */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) { fprintf(stderr, "H5VL_gpucompress_register() failed\n"); return H5I_INVALID_HID; }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL) < 0) {
        fprintf(stderr, "H5Pset_fapl_gpucompress() failed\n");
        H5Pclose(fapl); H5VLclose(vol_id);
        return H5I_INVALID_HID;
    }

    hid_t fid;
    if (flags & H5F_ACC_TRUNC)
        fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    else
        fid = H5Fopen(path, H5F_ACC_RDONLY, fapl);

    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

/* ------------------------------------------------------------------ */
/* Test 1: VOL registration idempotent                                  */
/* ------------------------------------------------------------------ */
static void test_registration_idempotent(void) {
    printf("[Test 1] VOL registration idempotent ... ");
    fflush(stdout);

    hid_t id1 = H5VL_gpucompress_register();
    hid_t id2 = H5VL_gpucompress_register();

    if (id1 < 0 || id2 < 0) {
        FAIL("registration returned invalid ID");
        if (id1 >= 0) H5VLclose(id1);
        if (id2 >= 0) H5VLclose(id2);
        return;
    }

    /* Check while IDs are still open (closing would unregister) */
    htri_t reg = H5VLis_connector_registered_by_value(H5VL_GPUCOMPRESS_VALUE);

    H5VLclose(id1);
    H5VLclose(id2);

    if (reg > 0) { PASS(); } else { FAIL("connector value 512 not registered"); }
}

/* ------------------------------------------------------------------ */
/* Test 2: 1D single-chunk GPU write + read (LZ4, lossless)             */
/* ------------------------------------------------------------------ */
static void test_1d_single_chunk(void) {
    printf("[Test 2] 1D single-chunk GPU write + read (LZ4) ... ");
    fflush(stdout);

    const hsize_t N     = 1024;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol_1d_single.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 1.0f);

    float* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) { FAIL("cudaMalloc"); free(h_ref); free(h_out); return; }
    cudaMemcpy(d_data, h_ref, bytes, cudaMemcpyHostToDevice);

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open_vol_file write"); goto done1; }

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 1 /*LZ4*/, 2 /*shuffle4*/, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done1; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data) < 0) {
            FAIL("H5Dwrite GPU"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done1;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open_vol_file read"); goto done1; }

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2"); H5Fclose(fid); goto done1; }

        float* d_read;
        cudaMalloc(&d_read, bytes);

        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read) < 0) {
            FAIL("H5Dread GPU"); cudaFree(d_read); H5Dclose(dset); H5Fclose(fid); goto done1;
        }

        cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, 1e-5f)) { PASS(); } else { FAIL("data mismatch"); }

done1:
    cudaFree(d_data); free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 3: 1D multi-chunk GPU write + read (Zstd)                       */
/* ------------------------------------------------------------------ */
static void test_1d_multi_chunk(void) {
    printf("[Test 3] 1D multi-chunk GPU write + read (Zstd) ... ");
    fflush(stdout);

    const hsize_t N      = 8192;
    const hsize_t CSIZE  = 1024;
    const size_t  bytes  = N * sizeof(float);
    const char*   fname  = "/tmp/test_vol_1d_multi.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 2.0f);

    float* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) { FAIL("cudaMalloc"); free(h_ref); free(h_out); return; }
    cudaMemcpy(d_data, h_ref, bytes, cudaMemcpyHostToDevice);

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open write"); goto done3; }

        hsize_t dims[1] = { N }, chunk[1] = { CSIZE };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 5 /*ZSTD*/, 2 /*shuffle4*/, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done3; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data) < 0) {
            FAIL("H5Dwrite"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done3;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open read"); goto done3; }

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2"); H5Fclose(fid); goto done3; }

        float* d_read;
        cudaMalloc(&d_read, bytes);

        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read) < 0) {
            FAIL("H5Dread"); cudaFree(d_read); H5Dclose(dset); H5Fclose(fid); goto done3;
        }

        cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, 1e-5f)) { PASS(); } else { FAIL("data mismatch"); }

done3:
    cudaFree(d_data); free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 4: Host pointer fallback (native filter path)                   */
/* ------------------------------------------------------------------ */
static void test_host_pointer_fallback(void) {
    printf("[Test 4] Host pointer fallback (native filter path) ... ");
    fflush(stdout);

    const hsize_t N    = 2048;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol_host_fallback.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 3.0f);

    /* Write with HOST pointer through VOL — falls through to native */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open write"); goto done4; }

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        /* No gpucompress filter — just chunked storage */

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done4; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_ref) < 0) {
            FAIL("H5Dwrite host"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done4;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read with HOST pointer */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open read"); goto done4; }

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2"); H5Fclose(fid); goto done4; }

        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_out) < 0) {
            FAIL("H5Dread host"); H5Dclose(dset); H5Fclose(fid); goto done4;
        }

        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, 1e-5f)) { PASS(); } else { FAIL("data mismatch"); }

done4:
    free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 5: 2D chunked dataset                                           */
/* ------------------------------------------------------------------ */
static void test_2d_chunked(void) {
    printf("[Test 5] 2D chunked GPU write + read ... ");
    fflush(stdout);

    const hsize_t ROWS = 64, COLS = 128;
    const hsize_t CROW = 32, CCOL = 128;
    const size_t  N    = ROWS * COLS;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol_2d.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 4.0f);

    float* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) { FAIL("cudaMalloc"); free(h_ref); free(h_out); return; }
    cudaMemcpy(d_data, h_ref, bytes, cudaMemcpyHostToDevice);

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open write 2D"); goto done5; }

        hsize_t dims[2]  = { ROWS, COLS };
        hsize_t chunk[2] = { CROW, CCOL };
        hid_t space = H5Screate_simple(2, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, chunk);
        set_gpucompress_filter(dcpl, 1 /*LZ4*/, 2 /*shuffle4*/, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data2d", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2 2D"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done5; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data) < 0) {
            FAIL("H5Dwrite 2D"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done5;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open read 2D"); goto done5; }

        hid_t dset = H5Dopen2(fid, "data2d", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2 2D"); H5Fclose(fid); goto done5; }

        float* d_read;
        cudaMalloc(&d_read, bytes);

        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read) < 0) {
            FAIL("H5Dread 2D"); cudaFree(d_read); H5Dclose(dset); H5Fclose(fid); goto done5;
        }

        cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, 1e-5f)) { PASS(); } else { FAIL("data mismatch 2D"); }

done5:
    cudaFree(d_data); free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 6 (extra): 2D dataset where chunk has partial trailing dim      */
/*  Previously the contiguity check was inverted and would have used    */
/*  a direct (wrong) pointer offset for this layout.                    */
/* ------------------------------------------------------------------ */
static void test_2d_partial_col_chunk(void) {
    printf("[Test 6b] 2D partial-column chunk (regression) ... ");
    fflush(stdout);

    /* Dataset [64, 128], chunk [64, 64] — full row-count, half col-count.
     * With the old (wrong) check: chunk_dims[0]==dset_dims[0] → "contiguous"
     * → direct pointer offset → data corruption.
     * With the fix: chunk_dims[1]=64 != dset_dims[1]=128 → gather kernel → correct. */
    const hsize_t ROWS = 64, COLS = 128;
    const hsize_t CROW = 64, CCOL = 64;
    const size_t  N    = ROWS * COLS;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol_2d_partcol.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 7.0f);

    float* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) { FAIL("cudaMalloc"); free(h_ref); free(h_out); return; }
    cudaMemcpy(d_data, h_ref, bytes, cudaMemcpyHostToDevice);

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open write"); goto done6b; }

        hsize_t dims[2]  = { ROWS, COLS };
        hsize_t chunk[2] = { CROW, CCOL };
        hid_t space = H5Screate_simple(2, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, chunk);
        set_gpucompress_filter(dcpl, 1 /*LZ4*/, 2 /*shuffle4*/, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "partcol", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done6b; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data) < 0) {
            FAIL("H5Dwrite"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done6b;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open read"); goto done6b; }

        hid_t dset = H5Dopen2(fid, "partcol", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2"); H5Fclose(fid); goto done6b; }

        float* d_read;
        cudaMalloc(&d_read, bytes);
        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read) < 0) {
            FAIL("H5Dread"); cudaFree(d_read); H5Dclose(dset); H5Fclose(fid); goto done6b;
        }

        cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, 1e-5f)) { PASS(); } else { FAIL("data mismatch (contiguity bug?)"); }

done6b:
    cudaFree(d_data); free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 6: Lossy (quantized) write + read                               */
/* ------------------------------------------------------------------ */
static void test_lossy_error_bound(void) {
    printf("[Test 6] Lossy (quantized) GPU write + read (error_bound=0.01) ... ");
    fflush(stdout);

    const hsize_t N          = 4096;
    const size_t  bytes       = N * sizeof(float);
    const float   error_bound = 0.01f;
    const char*   fname       = "/tmp/test_vol_lossy.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    fill_pattern(h_ref, N, 5.0f);

    float* d_data;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) { FAIL("cudaMalloc"); free(h_ref); free(h_out); return; }
    cudaMemcpy(d_data, h_ref, bytes, cudaMemcpyHostToDevice);

    /* Write: preprocessing = shuffle4 (0x02) | quantize (0x10) = 0x12 = 18 */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open write lossy"); goto done6; }

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 1 /*LZ4*/, 0x12 /*shuffle4+quant*/, 4, (double)error_bound);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dcreate2 lossy"); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done6; }

        if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data) < 0) {
            FAIL("H5Dwrite lossy"); H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid); goto done6;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { FAIL("open read lossy"); goto done6; }

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        if (dset < 0) { FAIL("H5Dopen2 lossy"); H5Fclose(fid); goto done6; }

        float* d_read;
        cudaMalloc(&d_read, bytes);

        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read) < 0) {
            FAIL("H5Dread lossy"); cudaFree(d_read); H5Dclose(dset); H5Fclose(fid); goto done6;
        }

        cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_close(h_ref, h_out, N, error_bound * 2.0f)) { PASS(); } else { FAIL("exceeds error bound"); }

done6:
    cudaFree(d_data); free(h_ref); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Main                                                                  */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    printf("=== GPUCompress HDF5 VOL Connector Tests ===\n\n");

    /* Suppress HDF5 error stack output */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    /* Init gpucompress (uses global state; re-init per test is fine) */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init() failed\n");
        return 1;
    }

    test_registration_idempotent();
    test_1d_single_chunk();
    test_1d_multi_chunk();
    test_host_pointer_fallback();
    test_2d_chunked();
    test_2d_partial_col_chunk();
    test_lossy_error_bound();

    gpucompress_cleanup();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
