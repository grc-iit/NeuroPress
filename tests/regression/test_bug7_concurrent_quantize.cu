/**
 * tests/test_bug7_concurrent_quantize.cu
 *
 * Regression test for BUG-7:
 *   d_range_min / d_range_max were static globals in quantization_kernels.cu,
 *   shared across all concurrent CompContext workers.  When ≥2 chunks needed
 *   quantization simultaneously, their GPU min/max reductions aliased the same
 *   device addresses → corrupted scale_factor → silently wrong quantized output.
 *
 * Fix: each CompContext slot now owns ctx->d_range_min / ctx->d_range_max
 * (allocated at gpucompress_init time).  The new quantize_simple() overload
 * accepts these as explicit parameters so concurrent calls never alias.
 *
 * Test strategy:
 *   - Dataset: 64 MiB of data split into 16 × 4 MiB chunks.
 *     With N_COMP_WORKERS=8, 2 full rounds of 8-way concurrent compression run.
 *   - Compression: LZ4 + quantization (GPUCOMPRESS_PREPROC_QUANTIZE), error_bound=0.01.
 *   - Data: two distinct patterns (ramp and sine-like) interleaved across chunks
 *     so concurrent quantize calls operate on data with DIFFERENT min/max values.
 *     Before the fix, workers mixed each other's range → wrong scale → errors > eb.
 *   - Verification: decompress each chunk and check |original - restored| ≤ error_bound.
 *
 * Pass criterion: 0 error-bound violations across all 16 million elements.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/test_bug7_concurrent_quantize
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Test parameters
 * ============================================================ */
#define DATASET_MB    64
#define CHUNK_MB       4
#define ERROR_BOUND    0.01   /* quantization error bound */

#define OUT_PATH "/tmp/test_bug7_quant.h5"

/* ============================================================
 * Filter wiring
 * ============================================================ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t dcpl_set_gpucompress_quant(hid_t dcpl, double error_bound)
{
    /* LZ4 (algo=1) + quantize (preproc=0x10) + shuffle-4 (shuf_sz=4) */
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 1;    /* ALGO_LZ4   */
    cd[1] = 0x10; /* PREPROC_QUANTIZE */
    cd[2] = 0;    /* shuf_sz=0 (no shuffle, just quantize) */
    pack_double_cd(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* ============================================================
 * GPU data fill kernels — two patterns with DIFFERENT ranges.
 *
 * Even chunks: ramp      buf[i] = (float)i / (float)n   range [0, 1)
 * Odd  chunks: scaled    buf[i] = 100.0f * sinf(...)     range [-100, 100]
 *
 * The range difference is the key: before the fix, concurrent workers
 * would mix the wide (-100..100) range into the narrow (0..1) range,
 * producing a scale_factor that's 100× too small → quantization error >> eb.
 * ============================================================ */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = (float)i / (float)n;
}

__global__ static void sine_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = 100.0f * sinf(2.0f * 3.14159265f * (float)i / (float)n);
}

/* ============================================================
 * Element-wise error check on host
 * ============================================================ */
static int check_errors(const float *orig, const float *rest, size_t n,
                        double eb, int chunk_idx, size_t *out_violations)
{
    size_t viol = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)orig[i] - (double)rest[i]);
        if (err > eb) viol++;
        if (err > max_err) max_err = err;
    }
    *out_violations = viol;
    if (viol > 0) {
        fprintf(stderr, "  [FAIL] chunk %d: %zu violations, max_err=%.6e (eb=%.6e)\n",
                chunk_idx, viol, max_err, eb);
        return -1;
    }
    printf("  [PASS] chunk %2d: max_err=%.6e <= eb=%.6e\n", chunk_idx, max_err, eb);
    return 0;
}

/* ============================================================
 * VOL FAPL helper
 * ============================================================ */
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    printf("=== BUG-7 Regression: Concurrent Quantization Range Corruption ===\n\n");
    printf("  Dataset : %d MiB  (%d chunks × %d MiB)\n", DATASET_MB, DATASET_MB/CHUNK_MB, CHUNK_MB);
    printf("  Algo    : LZ4 + quantize (error_bound=%.4f)\n", ERROR_BOUND);
    printf("  Patterns: even chunks=ramp [0,1), odd chunks=sine*100 [-100,100]\n");
    printf("  Workers : N_COMP_WORKERS=8 → 2 full concurrent rounds\n");
    printf("  Fix     : each CompContext slot owns d_range_min/d_range_max\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    size_t chunk_floats = (size_t)CHUNK_MB  * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)DATASET_MB * 1024 * 1024 / sizeof(float);
    size_t n_chunks     = n_floats / chunk_floats;
    size_t total_bytes  = n_floats * sizeof(float);
    size_t chunk_bytes  = chunk_floats * sizeof(float);

    /* Allocate GPU buffer for one dataset */
    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); gpucompress_cleanup(); return 1;
    }

    /* Fill entire GPU buffer — alternate patterns per chunk */
    printf("  Filling GPU buffer (alternating ramp / sine patterns per chunk)...\n");
    for (size_t c = 0; c < n_chunks; c++) {
        float *chunk_ptr = d_full + c * chunk_floats;
        if (c % 2 == 0)
            ramp_kernel<<<1024, 256>>>(chunk_ptr, chunk_floats);
        else
            sine_kernel<<<1024, 256>>>(chunk_ptr, chunk_floats);
    }
    cudaDeviceSynchronize();

    /* Store a host copy of original for verification */
    float *h_orig = (float*)malloc(total_bytes);
    if (!h_orig) {
        fprintf(stderr, "malloc h_orig failed\n");
        cudaFree(d_full); gpucompress_cleanup(); return 1;
    }
    cudaMemcpy(h_orig, d_full, total_bytes, cudaMemcpyDeviceToHost);

    /* ---- Register VOL ---- */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "H5VL_gpucompress_register failed\n");
        free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    /* ---- Warmup (JIT-compiles nvcomp) ---- */
    printf("  Warmup run...\n");
    {
        remove("/tmp/test_bug7_warmup.h5");
        hsize_t dims[1]  = { (hsize_t)n_floats };
        hsize_t cdims[1] = { (hsize_t)chunk_floats };
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate("/tmp/test_bug7_warmup.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file >= 0) {
            hid_t fsp  = H5Screate_simple(1, dims, NULL);
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, cdims);
            dcpl_set_gpucompress_quant(dcpl, ERROR_BOUND);
            hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Pclose(dcpl); H5Sclose(fsp);
            if (dset >= 0) { H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full); H5Dclose(dset); }
            H5Fclose(file);
        }
        remove("/tmp/test_bug7_warmup.h5");
    }

    /* ---- Write with concurrent quantization ---- */
    printf("\n  Writing dataset with concurrent quantization...\n");
    remove(OUT_PATH);
    {
        hsize_t dims[1]  = { (hsize_t)n_floats };
        hsize_t cdims[1] = { (hsize_t)chunk_floats };
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(OUT_PATH, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file < 0) {
            fprintf(stderr, "H5Fcreate failed\n");
            H5VLclose(vol_id); free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
        }
        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        dcpl_set_gpucompress_quant(dcpl, ERROR_BOUND);
        hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(fsp);
        if (dset < 0) {
            fprintf(stderr, "H5Dcreate2 failed\n");
            H5Fclose(file); H5VLclose(vol_id); free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
        }
        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
        H5Dclose(dset);
        H5Fclose(file);
        if (ret < 0) {
            fprintf(stderr, "H5Dwrite failed\n");
            H5VLclose(vol_id); free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
        }
    }
    printf("  Write complete.\n\n");

    /* ---- Read back and verify chunk by chunk ---- */
    printf("  Reading back and verifying error bound per chunk...\n\n");
    float *d_readback = NULL;
    if (cudaMalloc(&d_readback, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_readback failed\n");
        H5VLclose(vol_id); free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fopen(OUT_PATH, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
        H5Dclose(dset);
        H5Fclose(file);
    }
    cudaDeviceSynchronize();

    /* D→H the readback */
    float *h_readback = (float*)malloc(total_bytes);
    if (!h_readback) {
        fprintf(stderr, "malloc h_readback failed\n");
        cudaFree(d_readback); H5VLclose(vol_id); free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }
    cudaMemcpy(h_readback, d_readback, total_bytes, cudaMemcpyDeviceToHost);

    /* Verify per chunk */
    size_t total_viol = 0;
    int pass = 1;
    for (size_t c = 0; c < n_chunks; c++) {
        size_t viol = 0;
        const float *orig = h_orig + c * chunk_floats;
        const float *rest = h_readback + c * chunk_floats;
        if (check_errors(orig, rest, chunk_floats, ERROR_BOUND, (int)c, &viol) != 0)
            pass = 0;
        total_viol += viol;
    }

    /* Cleanup */
    H5VLclose(vol_id);
    cudaFree(d_readback);
    cudaFree(d_full);
    free(h_readback);
    free(h_orig);
    gpucompress_cleanup();
    remove(OUT_PATH);

    printf("\n  Total violations: %zu / %zu elements\n", total_viol, n_floats);
    printf("\n  Overall: %s\n\n",
           pass ? "ALL CHUNKS PASS — BUG-7 fix verified" : "FAIL — concurrent quantization corrupted data");

    return pass ? 0 : 1;
}
