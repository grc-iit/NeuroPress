/**
 * test_vol2_gpu_fallback.cu
 *
 * Regression test for VOL-2:
 *   When a GPU (device) pointer is passed to H5Dwrite()/H5Dread() on a dataset
 *   that does NOT have the gpucompress filter, the VOL connector must:
 *     - NOT return an error (-1)
 *     - Perform a D→H copy, then write via the native VOL (write path)
 *     - Perform a native read, then an H→D copy (read path)
 *     - Preserve data exactly (lossless round-trip)
 *
 * Tests:
 *   [1] GPU write → GPU read on dataset WITHOUT gpucompress filter (fallback path)
 *   [2] GPU write → host read on dataset WITHOUT gpucompress filter
 *   [3] Host write → GPU read on dataset WITHOUT gpucompress filter
 *   [4] GPU write → GPU read on dataset WITH gpucompress filter (normal path, no regression)
 *   [5] Host write → host read on dataset WITHOUT gpucompress filter (pure native, no regression)
 *   [6] Multi-chunk dataset without gpucompress filter (GPU write/read fallback)
 *
 * Requires HDF5 2.x built at /tmp/hdf5-install/.
 * Run: LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH ./build/test_vol2_gpu_fallback
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Filter constants                                                     */
/* ------------------------------------------------------------------ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

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

static int verify_exact(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "    mismatch at [%zu]: %.8f vs %.8f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

/* Open/create a VOL-enabled HDF5 file */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    assert(vol_id >= 0 && "H5VL_gpucompress_register() failed");

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl >= 0 && "H5Pcreate(H5P_FILE_ACCESS) failed");

    herr_t rc = H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    assert(rc >= 0 && "H5Pset_fapl_gpucompress() failed");

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
/* Test 1: GPU write + GPU read — dataset WITHOUT gpucompress filter   */
/* ------------------------------------------------------------------ */
static void test_gpu_write_gpu_read_no_filter(void) {
    printf("[Test 1] GPU write + GPU read, no gpucompress filter (fallback D→H→disk→H→D) ... ");
    fflush(stdout);

    const hsize_t N     = 2048;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol2_no_filter_gpu_rw.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 2.5f);

    float* d_write;
    float* d_read;
    cudaError_t ce;

    ce = cudaMalloc(&d_write, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_write failed");
    ce = cudaMalloc(&d_read, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_read failed");

    ce = cudaMemcpy(d_write, h_ref, bytes, cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess && "H→D copy for reference data failed");

    /* Write: GPU ptr, dataset WITHOUT gpucompress filter */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        /* Note: intentionally NO gpucompress filter — only standard chunking */

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        /* This must NOT return -1 (the VOL-2 bug) */
        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_write);
        if (ret < 0) {
            FAIL("H5Dwrite returned -1 for GPU ptr on non-gpucompress dataset (VOL-2 regression)");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            goto done;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read: GPU ptr, same dataset — fallback native read + H→D copy */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        cudaMemset(d_read, 0, bytes);  /* ensure we're reading fresh data */

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        if (ret < 0) {
            FAIL("H5Dread returned -1 for GPU ptr on non-gpucompress dataset (VOL-2 regression)");
            H5Dclose(dset); H5Fclose(fid);
            goto done;
        }

        ce = cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        assert(ce == cudaSuccess && "D→H copy of read data failed");

        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch after round-trip"); }

done:
    cudaFree(d_write);
    cudaFree(d_read);
    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 2: GPU write + host read — dataset WITHOUT gpucompress filter  */
/* ------------------------------------------------------------------ */
static void test_gpu_write_host_read_no_filter(void) {
    printf("[Test 2] GPU write + host read, no gpucompress filter ... ");
    fflush(stdout);

    const hsize_t N     = 1024;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol2_no_filter_gpu_write_host_read.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 7.0f);

    float* d_write;
    cudaError_t ce = cudaMalloc(&d_write, bytes);
    assert(ce == cudaSuccess && "cudaMalloc failed");
    ce = cudaMemcpy(d_write, h_ref, bytes, cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess && "H→D copy failed");

    /* Write via GPU fallback */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_write);
        if (ret < 0) {
            FAIL("H5Dwrite returned -1 for GPU ptr on non-gpucompress dataset");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read via host pointer (native HDF5 path, no VOL intercept needed) */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, h_out);
        if (ret < 0) {
            FAIL("H5Dread(host) failed");
            H5Dclose(dset); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch"); }

done:
    cudaFree(d_write);
    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 3: host write + GPU read — dataset WITHOUT gpucompress filter  */
/* ------------------------------------------------------------------ */
static void test_host_write_gpu_read_no_filter(void) {
    printf("[Test 3] Host write + GPU read, no gpucompress filter (H→D fallback on read) ... ");
    fflush(stdout);

    const hsize_t N     = 1024;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol2_no_filter_host_write_gpu_read.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 3.14f);

    float* d_read;
    cudaError_t ce = cudaMalloc(&d_read, bytes);
    assert(ce == cudaSuccess && "cudaMalloc failed");

    /* Write via host ptr (native path — no fallback needed) */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, h_ref);
        assert(ret >= 0 && "H5Dwrite(host) failed");

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read via GPU ptr — triggers H→D fallback */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        if (ret < 0) {
            FAIL("H5Dread returned -1 for GPU ptr on non-gpucompress dataset (VOL-2 regression)");
            H5Dclose(dset); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Fclose(fid);
    }

    ce = cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
    assert(ce == cudaSuccess && "D→H copy of read result failed");

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch"); }

done:
    cudaFree(d_read);
    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 4: GPU write + GPU read — dataset WITH gpucompress filter      */
/* (normal GPU compression path — must not regress)                    */
/* ------------------------------------------------------------------ */
static void test_gpu_write_gpu_read_with_filter(void) {
    printf("[Test 4] GPU write + GPU read, WITH gpucompress filter (normal path, no regression) ... ");
    fflush(stdout);

    const hsize_t N     = 2048;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol2_with_filter_gpu_rw.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 5.0f);

    float* d_write;
    float* d_read;
    cudaError_t ce;

    ce = cudaMalloc(&d_write, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_write failed");
    ce = cudaMalloc(&d_read, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_read failed");
    ce = cudaMemcpy(d_write, h_ref, bytes, cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess && "H→D copy failed");

    /* Write: GPU ptr WITH gpucompress filter → normal GPU compress path */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 1 /*LZ4*/, 2 /*shuffle4*/, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_write);
        if (ret < 0) {
            FAIL("H5Dwrite returned -1 (gpucompress filter path regression)");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read: GPU ptr WITH gpucompress filter → normal GPU decompress path */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        if (ret < 0) {
            FAIL("H5Dread returned -1 (gpucompress filter path regression)");
            H5Dclose(dset); H5Fclose(fid);
            goto done;
        }

        ce = cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
        assert(ce == cudaSuccess && "D→H copy of read data failed");

        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch"); }

done:
    cudaFree(d_write);
    cudaFree(d_read);
    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 5: host write + host read, no filter (pure native, no VOL      */
/* interception — must not regress)                                    */
/* ------------------------------------------------------------------ */
static void test_host_write_host_read_no_filter(void) {
    printf("[Test 5] Host write + host read, no filter (pure native path, no regression) ... ");
    fflush(stdout);

    const hsize_t N     = 512;
    const size_t  bytes = N * sizeof(float);
    const char*   fname = "/tmp/test_vol2_host_native.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 1.11f);

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1] = { N }, chunk[1] = { N };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, h_ref);
        assert(ret >= 0 && "H5Dwrite(host) failed");

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, h_out);
        assert(ret >= 0 && "H5Dread(host) failed");

        H5Dclose(dset); H5Fclose(fid);
    }

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch"); }

    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Test 6: GPU write + GPU read on multi-chunk dataset, no filter      */
/* ------------------------------------------------------------------ */
static void test_gpu_multichunk_no_filter(void) {
    printf("[Test 6] GPU write + GPU read, multi-chunk, no filter (fallback) ... ");
    fflush(stdout);

    const hsize_t CHUNK  = 512;
    const hsize_t NCHUNK = 4;
    const hsize_t N      = CHUNK * NCHUNK;
    const size_t  bytes  = N * sizeof(float);
    const char*   fname  = "/tmp/test_vol2_no_filter_multichunk.h5";

    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)calloc(N, sizeof(float));
    assert(h_ref && h_out && "host buffer allocation failed");
    fill_pattern(h_ref, N, 9.9f);

    float* d_write;
    float* d_read;
    cudaError_t ce;

    ce = cudaMalloc(&d_write, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_write failed");
    ce = cudaMalloc(&d_read, bytes);
    assert(ce == cudaSuccess && "cudaMalloc d_read failed");
    ce = cudaMemcpy(d_write, h_ref, bytes, cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess && "H→D copy failed");

    /* Write */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        assert(fid >= 0 && "open_vol_file(write) failed");

        hsize_t dims[1]  = { N };
        hsize_t chunk[1] = { CHUNK };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        /* No gpucompress filter */

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_write);
        if (ret < 0) {
            FAIL("H5Dwrite returned -1 for GPU ptr on multi-chunk dataset without filter");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Read */
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        assert(fid >= 0 && "open_vol_file(read) failed");

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        if (ret < 0) {
            FAIL("H5Dread returned -1 for GPU ptr on multi-chunk dataset without filter");
            H5Dclose(dset); H5Fclose(fid);
            goto done;
        }
        H5Dclose(dset); H5Fclose(fid);
    }

    ce = cudaMemcpy(h_out, d_read, bytes, cudaMemcpyDeviceToHost);
    assert(ce == cudaSuccess && "D→H copy of read result failed");

    if (verify_exact(h_ref, h_out, N)) { PASS(); } else { FAIL("data mismatch"); }

done:
    cudaFree(d_write);
    cudaFree(d_read);
    free(h_ref);
    free(h_out);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    printf("=== VOL-2 GPU Fallback Tests (GPU ptr on non-gpucompress dataset) ===\n\n");

    /* Suppress HDF5 error stack output so expected failures are quiet */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init() failed\n");
        return 1;
    }

    test_gpu_write_gpu_read_no_filter();
    test_gpu_write_host_read_no_filter();
    test_host_write_gpu_read_no_filter();
    test_gpu_write_gpu_read_with_filter();
    test_host_write_host_read_no_filter();
    test_gpu_multichunk_no_filter();

    gpucompress_cleanup();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
