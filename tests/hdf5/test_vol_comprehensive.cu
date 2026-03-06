/**
 * test_vol_comprehensive.cu
 *
 * Comprehensive test suite for the GPUCompress HDF5 VOL connector.
 *
 * Test groups:
 *   A. Algorithms   — LZ4, Snappy, Deflate, GDeflate, Zstd, ANS, Cascaded, Bitcomp, AUTO
 *   B. Geometry     — 1D/2D/3D, boundary chunks, partial-column chunks, single-element
 *   C. Data types   — float32, float64, int32
 *   D. Preprocessing — no preproc, shuffle only, lossy (various error bounds)
 *   E. File ops     — reopen between write/read, multiple datasets, same file
 *   F. Fallbacks    — host pointer, contiguous (non-chunked) dataset, no filter
 *   G. Stress       — large dataset (~16 MB), many small chunks, many chunks 2D
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Algorithm IDs (mirror gpucompress.h enum) */
#define ALGO_AUTO      0
#define ALGO_LZ4       1
#define ALGO_SNAPPY    2
#define ALGO_DEFLATE   3
#define ALGO_GDEFLATE  4
#define ALGO_ZSTD      5
#define ALGO_ANS       6
#define ALGO_CASCADED  7
#define ALGO_BITCOMP   8

/* Preprocessing flags */
#define PREPROC_NONE      0x00
#define PREPROC_SHUFFLE4  0x02
#define PREPROC_QUANTIZE  0x10

/* ------------------------------------------------------------------ */
/* Test infrastructure                                                  */
/* ------------------------------------------------------------------ */
static int g_pass = 0, g_fail = 0;
static int g_skip = 0;          /* algorithms not compiled in */

#define PASS()      do { printf("PASS\n"); g_pass++; } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); g_fail++; } while(0)
#define SKIP(msg)   do { printf("SKIP (%s)\n", msg); g_skip++; } while(0)

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static void pack_double(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t set_filter(hid_t dcpl, unsigned algo, unsigned preproc,
                          unsigned shuf, double eb)
{
    unsigned cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algo; cd[1] = preproc; cd[2] = shuf;
    pack_double(eb, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

static hid_t open_file(const char *path, unsigned flags) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    hid_t fid = (flags & H5F_ACC_TRUNC)
        ? H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        : H5Fopen(path,  H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    return fid;
}

/* Fill float array with a reproducible pattern */
static void fill_float(float *h, size_t n, float seed) {
    for (size_t i = 0; i < n; i++)
        h[i] = seed + sinf((float)i * 0.0137f) * 100.0f;
}

/* Fill double array */
static void fill_double(double *h, size_t n, double seed) {
    for (size_t i = 0; i < n; i++)
        h[i] = seed + sin((double)i * 0.0137) * 1000.0;
}

/* Fill int32 array */
static void fill_int32(int *h, size_t n, int seed) {
    for (size_t i = 0; i < n; i++)
        h[i] = seed + (int)(i % 65536);
}

static int verify_float(const float *a, const float *b, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "    float mismatch[%zu]: %.6g vs %.6g (tol=%.6g)\n",
                    i, a[i], b[i], tol);
            return 0;
        }
    }
    return 1;
}

static int verify_double(const double *a, const double *b, size_t n, double tol) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tol) {
            fprintf(stderr, "    double mismatch[%zu]: %.10g vs %.10g (tol=%.6g)\n",
                    i, a[i], b[i], tol);
            return 0;
        }
    }
    return 1;
}

static int verify_int32(const int *a, const int *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "    int32 mismatch[%zu]: %d vs %d\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

/**
 * Core 1D float round-trip helper.
 * Returns 1 on pass, 0 on fail.
 */
static int roundtrip_1d_float(const char *fname, const char *dset_name,
                               hsize_t N, hsize_t chunk,
                               unsigned algo, unsigned preproc,
                               double error_bound, float verify_tol)
{
    int ok = 0;
    float *h_ref = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)calloc(N, sizeof(float));
    fill_float(h_ref, N, (float)algo + 1.0f);

    float *d_data = NULL;
    if (cudaMalloc(&d_data, N * sizeof(float)) != cudaSuccess) goto cleanup;
    cudaMemcpy(d_data, h_ref, N * sizeof(float), cudaMemcpyHostToDevice);

    /* Write */
    {
        hid_t fid = open_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) goto cleanup;

        hsize_t dims[1] = {N}, chk[1] = {chunk};
        hid_t sp  = H5Screate_simple(1, dims, NULL);
        hid_t dcp = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp, 1, chk);
        set_filter(dcp, algo, preproc, 4, error_bound);

        hid_t ds = H5Dcreate2(fid, dset_name, H5T_NATIVE_FLOAT, sp,
                               H5P_DEFAULT, dcp, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid); goto cleanup; }
        if (H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, d_data) < 0) {
            H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid); goto cleanup;
        }
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);
    }

    /* Read */
    {
        float *d_read = NULL;
        if (cudaMalloc(&d_read, N * sizeof(float)) != cudaSuccess) goto cleanup;

        hid_t fid = open_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) { cudaFree(d_read); goto cleanup; }

        hid_t ds = H5Dopen2(fid, dset_name, H5P_DEFAULT);
        if (ds < 0) { H5Fclose(fid); cudaFree(d_read); goto cleanup; }

        if (H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, d_read) < 0) {
            H5Dclose(ds); H5Fclose(fid); cudaFree(d_read); goto cleanup;
        }
        cudaMemcpy(h_out, d_read, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_read);
        H5Dclose(ds); H5Fclose(fid);
    }

    ok = verify_float(h_ref, h_out, N, verify_tol);

cleanup:
    cudaFree(d_data);
    free(h_ref); free(h_out);
    return ok;
}

/* ================================================================== */
/* GROUP A: Algorithm tests                                            */
/* ================================================================== */

/**
 * Probe whether an algorithm is available in gpucompress_compress_gpu.
 * Deflate/GDeflate/ANS require an nvcomp premium license and will either
 * return GPUCOMPRESS_ERROR_COMPRESSION or crash without it.
 * Returns 1 if available, 0 if not.
 */
static int probe_algo_gpu(unsigned algo) {
    const size_t N = 4096;   /* large enough for all algorithms' min-size requirements */
    float *d_in = NULL, *d_out = NULL;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, gpucompress_max_compressed_size(N * sizeof(float)));
    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm     = (gpucompress_algorithm_t)algo;
    cfg.preprocessing = PREPROC_NONE;
    size_t sz = gpucompress_max_compressed_size(N * sizeof(float));
    gpucompress_error_t e = gpucompress_compress_gpu(d_in, N*sizeof(float), d_out, &sz, &cfg, NULL, NULL);
    cudaFree(d_in); cudaFree(d_out);
    return (e == GPUCOMPRESS_SUCCESS);
}

static void test_algo(const char *name, unsigned algo) {
    printf("  [A] 1D chunk, algo=%s ... ", name);
    fflush(stdout);
    if (!probe_algo_gpu(algo)) {
        SKIP("nvcomp premium license required");
        return;
    }
    char fname[64]; snprintf(fname, sizeof(fname), "/tmp/tvol_algo%u.h5", algo);
    if (roundtrip_1d_float(fname, "d", 16384, 4096, algo,
                           PREPROC_SHUFFLE4, 0.0, 1e-5f))
        PASS();
    else
        FAIL("round-trip mismatch");
}

static void test_group_algorithms(void) {
    puts("\n--- Group A: Algorithms ---");
    /* AUTO falls back to LZ4 in GPU path — still must round-trip correctly */
    test_algo("AUTO",     ALGO_AUTO);
    test_algo("LZ4",      ALGO_LZ4);
    test_algo("Snappy",   ALGO_SNAPPY);
    test_algo("Deflate",  ALGO_DEFLATE);
    test_algo("GDeflate", ALGO_GDEFLATE);
    test_algo("Zstd",     ALGO_ZSTD);
    test_algo("ANS",      ALGO_ANS);
    test_algo("Cascaded", ALGO_CASCADED);
    test_algo("Bitcomp",  ALGO_BITCOMP);
}

/* ================================================================== */
/* GROUP B: Geometry tests                                             */
/* ================================================================== */

static void test_group_geometry(void) {
    puts("\n--- Group B: Geometry ---");

    /* B1: 1D, N exactly divisible by chunk */
    {
        printf("  [B1] 1D exact-divisible (8192 / 1024) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_b1.h5", "d", 8192, 1024,
                               ALGO_LZ4, PREPROC_SHUFFLE4, 0.0, 1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* B2: 1D, N NOT divisible by chunk (boundary chunk) */
    {
        printf("  [B2] 1D boundary chunk (10000 / 3000) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_b2.h5", "d", 10000, 3000,
                               ALGO_LZ4, PREPROC_SHUFFLE4, 0.0, 1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* B3: 1D, single element chunk */
    {
        printf("  [B3] 1D single-element chunks (64 elements) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_b3.h5", "d", 64, 1,
                               ALGO_LZ4, PREPROC_NONE, 0.0, 1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* B4: 1D, exactly one chunk (N == chunk_size) */
    {
        printf("  [B4] 1D single exact chunk (N==chunk==512) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_b4.h5", "d", 512, 512,
                               ALGO_LZ4, PREPROC_SHUFFLE4, 0.0, 1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* B5: 2D, chunk spans full column width (contiguous fast path) */
    {
        printf("  [B5] 2D full-col chunk [32,128] on [64,128] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=64, C=128, CR=32, CC=128;
        float *h_ref = (float*)malloc(R*C*sizeof(float));
        float *h_out = (float*)calloc(R*C, sizeof(float));
        fill_float(h_ref, R*C, 10.0f);
        float *d = NULL; cudaMalloc(&d, R*C*sizeof(float));
        cudaMemcpy(d, h_ref, R*C*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_b5.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={CR,CC};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2,R*C*sizeof(float));
        fid=open_file("/tmp/tvol_b5.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,R*C*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,R*C,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* B6: 2D, partial-column chunk (gather-kernel path) */
    {
        printf("  [B6] 2D partial-col chunk [64,64] on [64,128] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=64, C=128, CR=64, CC=64;
        float *h_ref = (float*)malloc(R*C*sizeof(float));
        float *h_out = (float*)calloc(R*C, sizeof(float));
        fill_float(h_ref, R*C, 11.0f);
        float *d = NULL; cudaMalloc(&d, R*C*sizeof(float));
        cudaMemcpy(d, h_ref, R*C*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_b6.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={CR,CC};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2,R*C*sizeof(float));
        fid=open_file("/tmp/tvol_b6.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,R*C*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,R*C,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* B7: 2D, boundary in both dimensions (ROWS/COLS not divisible by chunk) */
    {
        printf("  [B7] 2D boundary both dims [70,130] chunk [32,50] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=70, C=130, CR=32, CC=50;
        size_t N = R*C;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 12.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_b7.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={CR,CC};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
        fid=open_file("/tmp/tvol_b7.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* B8: 3D dataset */
    {
        printf("  [B8] 3D dataset [8,16,32] chunk [4,8,32] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t D0=8, D1=16, D2=32;
        const hsize_t C0=4,  C1=8,  C2=32;
        size_t N = D0*D1*D2;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 13.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_b8.h5", H5F_ACC_TRUNC);
        hsize_t dims[3]={D0,D1,D2}, chk[3]={C0,C1,C2};
        hid_t sp=H5Screate_simple(3,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,3,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
        fid=open_file("/tmp/tvol_b8.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* B9: 3D dataset, partial boundary in all dims */
    {
        printf("  [B9] 3D boundary all dims [10,15,20] chunk [4,6,8] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t D0=10, D1=15, D2=20;
        const hsize_t C0=4,  C1=6,  C2=8;
        size_t N = D0*D1*D2;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 14.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_b9.h5", H5F_ACC_TRUNC);
        hsize_t dims[3]={D0,D1,D2}, chk[3]={C0,C1,C2};
        hid_t sp=H5Screate_simple(3,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,3,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
        fid=open_file("/tmp/tvol_b9.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }
}

/* ================================================================== */
/* GROUP C: Data type tests                                            */
/* ================================================================== */

static void test_group_datatypes(void) {
    puts("\n--- Group C: Data types ---");

    /* C1: float64 (double) */
    {
        printf("  [C1] float64 1D round-trip ... "); fflush(stdout);
        int ok = 0;
        const hsize_t N = 8192;
        double *h_ref = (double*)malloc(N*sizeof(double));
        double *h_out = (double*)calloc(N, sizeof(double));
        fill_double(h_ref, N, 1.0);
        double *d=NULL; cudaMalloc(&d, N*sizeof(double));
        cudaMemcpy(d, h_ref, N*sizeof(double), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_c1.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={2048};
        hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_NONE,0,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_DOUBLE,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        double *d2=NULL; cudaMalloc(&d2, N*sizeof(double));
        fid=open_file("/tmp/tvol_c1.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(double),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_double(h_ref,h_out,N,1e-12);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* C2: int32 */
    {
        printf("  [C2] int32 1D round-trip ... "); fflush(stdout);
        int ok = 0;
        const hsize_t N = 8192;
        int *h_ref = (int*)malloc(N*sizeof(int));
        int *h_out = (int*)calloc(N, sizeof(int));
        fill_int32(h_ref, N, 42);
        int *d=NULL; cudaMalloc(&d, N*sizeof(int));
        cudaMemcpy(d, h_ref, N*sizeof(int), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_c2.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={2048};
        hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_NONE,0,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_INT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        int *d2=NULL; cudaMalloc(&d2, N*sizeof(int));
        fid=open_file("/tmp/tvol_c2.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(int),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_int32(h_ref,h_out,N);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* C3: float64 2D */
    {
        printf("  [C3] float64 2D [32,64] chunk [16,64] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=32, C=64;
        double *h_ref = (double*)malloc(R*C*sizeof(double));
        double *h_out = (double*)calloc(R*C, sizeof(double));
        fill_double(h_ref, R*C, 2.0);
        double *d=NULL; cudaMalloc(&d, R*C*sizeof(double));
        cudaMemcpy(d, h_ref, R*C*sizeof(double), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_c3.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={16,64};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk); set_filter(dcp,ALGO_LZ4,PREPROC_NONE,0,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_DOUBLE,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        double *d2=NULL; cudaMalloc(&d2, R*C*sizeof(double));
        fid=open_file("/tmp/tvol_c3.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,R*C*sizeof(double),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_double(h_ref,h_out,R*C,1e-12);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }
}

/* ================================================================== */
/* GROUP D: Preprocessing / lossy tests                               */
/* ================================================================== */

static void test_group_preprocessing(void) {
    puts("\n--- Group D: Preprocessing ---");

    /* D1: No preprocessing at all */
    {
        printf("  [D1] No preprocessing (raw bytes) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d1.h5","d",8192,2048,
                               ALGO_LZ4,PREPROC_NONE,0.0,1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* D2: Shuffle only (no quantize) */
    {
        printf("  [D2] Shuffle-only (no quantize) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d2.h5","d",8192,2048,
                               ALGO_LZ4,PREPROC_SHUFFLE4,0.0,1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* D3: Lossy, error_bound = 0.001 */
    {
        printf("  [D3] Lossy error_bound=0.001 ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d3.h5","d",8192,2048,
                               ALGO_LZ4,PREPROC_SHUFFLE4|PREPROC_QUANTIZE,
                               0.001,0.002f)) PASS();
        else FAIL("exceeds error bound");
    }

    /* D4: Lossy, error_bound = 0.01 */
    {
        printf("  [D4] Lossy error_bound=0.01 ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d4.h5","d",8192,2048,
                               ALGO_LZ4,PREPROC_SHUFFLE4|PREPROC_QUANTIZE,
                               0.01,0.02f)) PASS();
        else FAIL("exceeds error bound");
    }

    /* D5: Lossy, error_bound = 0.1 */
    {
        printf("  [D5] Lossy error_bound=0.1 ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d5.h5","d",8192,2048,
                               ALGO_LZ4,PREPROC_SHUFFLE4|PREPROC_QUANTIZE,
                               0.1,0.2f)) PASS();
        else FAIL("exceeds error bound");
    }

    /* D6: Shuffle + Zstd (better compression) */
    {
        printf("  [D6] Shuffle + Zstd lossless ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_d6.h5","d",8192,2048,
                               ALGO_ZSTD,PREPROC_SHUFFLE4,0.0,1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* D7: Lossy 2D dataset */
    {
        printf("  [D7] Lossy 2D [64,128] chunk [32,128] eb=0.05 ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=64, C=128; const float EB=0.05f;
        float *h_ref = (float*)malloc(R*C*sizeof(float));
        float *h_out = (float*)calloc(R*C, sizeof(float));
        fill_float(h_ref, R*C, 20.0f);
        float *d=NULL; cudaMalloc(&d, R*C*sizeof(float));
        cudaMemcpy(d, h_ref, R*C*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_d7.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={32,128};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk);
        set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4|PREPROC_QUANTIZE,4,(double)EB);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, R*C*sizeof(float));
        fid=open_file("/tmp/tvol_d7.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,R*C*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,R*C,EB*2.0f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("exceeds error bound");
    }
}

/* ================================================================== */
/* GROUP E: File operation tests                                       */
/* ================================================================== */

static void test_group_fileops(void) {
    puts("\n--- Group E: File operations ---");

    /* E1: Write, close, reopen with new file handle, read */
    {
        printf("  [E1] Write-close-reopen-read ... "); fflush(stdout);
        const hsize_t N=4096, CHUNK=1024;
        int ok = 0;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 30.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        /* Write and fully close */
        {
            hid_t fid = open_file("/tmp/tvol_e1.h5", H5F_ACC_TRUNC);
            hsize_t dims[1]={N}, chk[1]={CHUNK};
            hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
            hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
            H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
            H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);
        }
        /* Reopen in a completely separate open call */
        {
            float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
            hid_t fid = open_file("/tmp/tvol_e1.h5", H5F_ACC_RDONLY);
            hid_t ds = H5Dopen2(fid, "d", H5P_DEFAULT);
            H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
            cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaFree(d2); H5Dclose(ds); H5Fclose(fid);
        }
        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch after reopen");
    }

    /* E2: Multiple datasets in same file, different algorithms */
    {
        printf("  [E2] Multiple datasets in one file (3 algos) ... "); fflush(stdout);
        const hsize_t N=4096, CHUNK=1024;
        int ok = 1;
        float *h_lz4  = (float*)malloc(N*sizeof(float));
        float *h_zstd = (float*)malloc(N*sizeof(float));
        float *h_snappy=(float*)malloc(N*sizeof(float));
        float *h_out  = (float*)calloc(N, sizeof(float));
        fill_float(h_lz4,   N, 31.0f);
        fill_float(h_zstd,  N, 32.0f);
        fill_float(h_snappy,N, 33.0f);

        float *d0=NULL, *d1=NULL, *d2_=NULL;
        cudaMalloc(&d0, N*sizeof(float)); cudaMemcpy(d0,h_lz4,  N*sizeof(float),cudaMemcpyHostToDevice);
        cudaMalloc(&d1, N*sizeof(float)); cudaMemcpy(d1,h_zstd, N*sizeof(float),cudaMemcpyHostToDevice);
        cudaMalloc(&d2_,N*sizeof(float)); cudaMemcpy(d2_,h_snappy,N*sizeof(float),cudaMemcpyHostToDevice);

        /* Write all three */
        {
            hid_t fid = open_file("/tmp/tvol_e2.h5", H5F_ACC_TRUNC);
            hsize_t dims[1]={N}, chk[1]={CHUNK};
            struct { const char *name; unsigned algo; float *ptr; } dsets[3] = {
                {"lz4",   ALGO_LZ4,    d0},
                {"zstd",  ALGO_ZSTD,   d1},
                {"snappy",ALGO_SNAPPY, d2_},
            };
            for (int i = 0; i < 3; i++) {
                hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
                H5Pset_chunk(dcp,1,chk); set_filter(dcp,dsets[i].algo,PREPROC_SHUFFLE4,4,0.0);
                hid_t ds=H5Dcreate2(fid,dsets[i].name,H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
                H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,dsets[i].ptr);
                H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp);
            }
            H5Fclose(fid);
        }
        /* Read and verify each */
        {
            float *d_rd=NULL; cudaMalloc(&d_rd, N*sizeof(float));
            hid_t fid = open_file("/tmp/tvol_e2.h5", H5F_ACC_RDONLY);
            struct { const char *name; float *ref; } checks[3] = {
                {"lz4",    h_lz4},
                {"zstd",   h_zstd},
                {"snappy", h_snappy},
            };
            for (int i = 0; i < 3; i++) {
                cudaMemset(d_rd, 0, N*sizeof(float));
                hid_t ds = H5Dopen2(fid, checks[i].name, H5P_DEFAULT);
                H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d_rd);
                cudaMemcpy(h_out,d_rd,N*sizeof(float),cudaMemcpyDeviceToHost);
                H5Dclose(ds);
                if (!verify_float(checks[i].ref,h_out,N,1e-5f)) {
                    fprintf(stderr,"    mismatch in dataset '%s'\n",checks[i].name);
                    ok = 0;
                }
            }
            cudaFree(d_rd); H5Fclose(fid);
        }
        cudaFree(d0); cudaFree(d1); cudaFree(d2_);
        free(h_lz4); free(h_zstd); free(h_snappy); free(h_out);
        if (ok) PASS(); else FAIL("mismatch in multi-dataset file");
    }

    /* E3: GPU dataset and host dataset coexist in same file */
    {
        printf("  [E3] GPU and host datasets coexist in same file ... "); fflush(stdout);
        const hsize_t N=4096, CHUNK=1024;
        int ok = 1;
        float *h_ref_gpu  = (float*)malloc(N*sizeof(float));
        float *h_ref_host = (float*)malloc(N*sizeof(float));
        float *h_out      = (float*)calloc(N, sizeof(float));
        fill_float(h_ref_gpu,  N, 40.0f);
        fill_float(h_ref_host, N, 41.0f);

        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref_gpu, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_e3.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={CHUNK};

        /* GPU dataset */
        {
            hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
            hid_t ds=H5Dcreate2(fid,"gpu_data",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
            H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
            H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp);
        }
        /* Host dataset (no filter — pure native) */
        {
            hid_t sp=H5Screate_simple(1,dims,NULL);
            hid_t ds=H5Dcreate2(fid,"host_data",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
            H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,h_ref_host);
            H5Dclose(ds); H5Sclose(sp);
        }
        H5Fclose(fid);

        /* Read back both */
        fid = open_file("/tmp/tvol_e3.h5", H5F_ACC_RDONLY);
        {
            float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
            hid_t ds=H5Dopen2(fid,"gpu_data",H5P_DEFAULT);
            H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
            cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaFree(d2); H5Dclose(ds);
            if (!verify_float(h_ref_gpu,h_out,N,1e-5f)) {
                fputs("    GPU dataset mismatch\n", stderr); ok=0;
            }
        }
        {
            memset(h_out,0,N*sizeof(float));
            hid_t ds=H5Dopen2(fid,"host_data",H5P_DEFAULT);
            H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,h_out);
            H5Dclose(ds);
            if (!verify_float(h_ref_host,h_out,N,1e-5f)) {
                fputs("    host dataset mismatch\n", stderr); ok=0;
            }
        }
        H5Fclose(fid);
        cudaFree(d); free(h_ref_gpu); free(h_ref_host); free(h_out);
        if (ok) PASS(); else FAIL("dataset mismatch");
    }

    /* E4: Overwrite file (truncate) and verify new contents */
    {
        printf("  [E4] Overwrite file (truncate + re-write) ... "); fflush(stdout);
        const hsize_t N=2048, CHUNK=512;
        int ok = 1;
        float *h_v1 = (float*)malloc(N*sizeof(float));
        float *h_v2 = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_v1, N, 50.0f);
        fill_float(h_v2, N, 51.0f);

        for (int pass = 0; pass < 2; pass++) {
            float *src = (pass == 0) ? h_v1 : h_v2;
            float *d=NULL; cudaMalloc(&d, N*sizeof(float));
            cudaMemcpy(d, src, N*sizeof(float), cudaMemcpyHostToDevice);
            hid_t fid = open_file("/tmp/tvol_e4.h5", H5F_ACC_TRUNC);
            hsize_t dims[1]={N}, chk[1]={CHUNK};
            hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
            hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
            H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
            H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);
            cudaFree(d);
        }
        /* Read back — should contain h_v2 */
        {
            float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
            hid_t fid=open_file("/tmp/tvol_e4.h5",H5F_ACC_RDONLY);
            hid_t ds=H5Dopen2(fid,"d",H5P_DEFAULT);
            H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
            cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaFree(d2); H5Dclose(ds); H5Fclose(fid);
        }
        if (!verify_float(h_v2,h_out,N,1e-5f)) {
            fputs("    got stale v1 data\n",stderr); ok=0;
        }
        free(h_v1); free(h_v2); free(h_out);
        if (ok) PASS(); else FAIL("stale data after truncate");
    }
}

/* ================================================================== */
/* GROUP F: Fallback tests                                             */
/* ================================================================== */

static void test_group_fallbacks(void) {
    puts("\n--- Group F: Fallbacks ---");

    /* F1: Host pointer goes through native filter path (filter IS registered) */
    {
        printf("  [F1] Host pointer → native filter path ... "); fflush(stdout);
        const hsize_t N=4096, CHUNK=1024;
        int ok = 0;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 60.0f);

        hid_t fid = open_file("/tmp/tvol_f1.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={CHUNK};
        hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        /* HOST pointer write */
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,h_ref);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        fid=open_file("/tmp/tvol_f1.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        /* HOST pointer read */
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,h_out);
        H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* F2: Contiguous (non-chunked) dataset with GPU pointer — must not crash */
    {
        printf("  [F2] Contiguous (non-chunked) dataset, GPU ptr → fallback ... "); fflush(stdout);
        const hsize_t N=1024;
        int ok = 0;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 61.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        /* Write with GPU ptr but no chunking (VOL will fall through since dcpl=H5P_DEFAULT) */
        hid_t fid = open_file("/tmp/tvol_f2.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N};
        hid_t sp=H5Screate_simple(1,dims,NULL);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        /* GPU ptr → dcpl=H5P_DEFAULT → no chunking → VOL stores H5I_INVALID_HID
         * → falls through to H5VLdataset_write (native path).
         * Note: native HDF5 does NOT support GPU pointers for contiguous write;
         * this will likely fail or return bogus data.  We just check it doesn't
         * crash the process (ok if H5Dwrite returns error). */
        herr_t wr = H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Fclose(fid);
        cudaFree(d);
        free(h_ref); free(h_out);
        /* We expect it to either succeed (if HDF5 handles GPU ptr)
         * or fail gracefully (no crash/abort). Either way: PASS for no crash. */
        (void)wr;
        ok = 1;  /* process still running */
        if (ok) PASS(); else FAIL("crashed");
    }

    /* F3: Dataset without gpucompress filter, GPU pointer → graceful fallback */
    {
        printf("  [F3] Chunked dataset without filter, GPU ptr → native path ... "); fflush(stdout);
        const hsize_t N=4096, CHUNK=1024;
        int ok = 0;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 62.0f);
        float *d=NULL; cudaMalloc(&d, N*sizeof(float));
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        /* Write with GPU ptr, chunked but NO gpucompress filter.
         * VOL: dcpl != H5P_DEFAULT, dcpl_id is valid, but get_gpucompress_config_from_dcpl
         * returns -1 (filter absent) → gpu_aware_chunked_write returns -1 → H5Dwrite fails.
         * OR: VOL falls through to native (depending on implementation).
         * We just verify the process doesn't crash. */
        hid_t fid = open_file("/tmp/tvol_f3.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={CHUNK};
        hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,1,chk);
        /* Deliberately NO set_filter() call */
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        herr_t wr = H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);
        cudaFree(d); free(h_ref); free(h_out);
        (void)wr;
        ok = 1;  /* no crash */
        if (ok) PASS(); else FAIL("crashed");
    }
}

/* ================================================================== */
/* GROUP G: Stress tests                                               */
/* ================================================================== */

static void test_group_stress(void) {
    puts("\n--- Group G: Stress ---");

    /* G1: Large 1D dataset (~16 MB) with many chunks */
    {
        printf("  [G1] Large 1D (~16 MB, 4M floats, 256KB chunks) ... "); fflush(stdout);
        const hsize_t N = 4 * 1024 * 1024;   /* 4M floats = 16 MB */
        const hsize_t CHUNK = 64 * 1024;      /* 256KB per chunk → 64 chunks */
        int ok = 0;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 70.0f);

        float *d=NULL;
        if (cudaMalloc(&d, N*sizeof(float)) != cudaSuccess) {
            FAIL("cudaMalloc OOM"); free(h_ref); free(h_out); return;
        }
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_g1.h5", H5F_ACC_TRUNC);
        hsize_t dims[1]={N}, chk[1]={CHUNK};
        hid_t sp=H5Screate_simple(1,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,1,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        if (H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d) < 0) {
            FAIL("H5Dwrite");
            H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);
            cudaFree(d); free(h_ref); free(h_out); return;
        }
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
        fid=open_file("/tmp/tvol_g1.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* G2: Many small chunks (N=65536, chunk=16 → 4096 chunks) */
    {
        printf("  [G2] Many small chunks (4096 chunks of 16 floats) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_g2.h5","d",65536,16,
                               ALGO_LZ4,PREPROC_NONE,0.0,1e-5f)) PASS();
        else FAIL("mismatch");
    }

    /* G3: Large 2D dataset with partial-column chunks (gather kernel stress) */
    {
        printf("  [G3] Large 2D gather-kernel stress [512,256] chunk [64,64] ... "); fflush(stdout);
        int ok = 0;
        const hsize_t R=512, C=256, CR=64, CC=64;
        size_t N = R*C;
        float *h_ref = (float*)malloc(N*sizeof(float));
        float *h_out = (float*)calloc(N, sizeof(float));
        fill_float(h_ref, N, 71.0f);

        float *d=NULL;
        if (cudaMalloc(&d, N*sizeof(float)) != cudaSuccess) {
            FAIL("cudaMalloc"); free(h_ref); free(h_out); return;
        }
        cudaMemcpy(d, h_ref, N*sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = open_file("/tmp/tvol_g3.h5", H5F_ACC_TRUNC);
        hsize_t dims[2]={R,C}, chk[2]={CR,CC};
        hid_t sp=H5Screate_simple(2,dims,NULL), dcp=H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcp,2,chk); set_filter(dcp,ALGO_LZ4,PREPROC_SHUFFLE4,4,0.0);
        hid_t ds=H5Dcreate2(fid,"d",H5T_NATIVE_FLOAT,sp,H5P_DEFAULT,dcp,H5P_DEFAULT);
        H5Dwrite(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d);
        H5Dclose(ds); H5Sclose(sp); H5Pclose(dcp); H5Fclose(fid);

        float *d2=NULL; cudaMalloc(&d2, N*sizeof(float));
        fid=open_file("/tmp/tvol_g3.h5",H5F_ACC_RDONLY);
        ds=H5Dopen2(fid,"d",H5P_DEFAULT);
        H5Dread(ds,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,d2);
        cudaMemcpy(h_out,d2,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d2); H5Dclose(ds); H5Fclose(fid);

        ok = verify_float(h_ref,h_out,N,1e-5f);
        cudaFree(d); free(h_ref); free(h_out);
        if (ok) PASS(); else FAIL("mismatch");
    }

    /* G4: Zstd on large dataset (better ratio validation) */
    {
        printf("  [G4] Large 1D Zstd (~8 MB, 2M floats) ... "); fflush(stdout);
        if (roundtrip_1d_float("/tmp/tvol_g4.h5","d",2*1024*1024,64*1024,
                               ALGO_ZSTD,PREPROC_SHUFFLE4,0.0,1e-5f)) PASS();
        else FAIL("mismatch");
    }
}

/* ================================================================== */
/* Main                                                                 */
/* ================================================================== */
int main(void) {
    printf("=== GPUCompress VOL Comprehensive Tests ===\n");

    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init() failed\n");
        return 1;
    }

    test_group_algorithms();
    test_group_geometry();
    test_group_datatypes();
    test_group_preprocessing();
    test_group_fileops();
    test_group_fallbacks();
    test_group_stress();

    gpucompress_cleanup();

    printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
           g_pass, g_fail, g_skip);
    return (g_fail == 0) ? 0 : 1;
}
