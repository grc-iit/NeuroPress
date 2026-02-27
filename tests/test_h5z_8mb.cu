/**
 * test_h5z_8mb.cu
 *
 * Generates 8 MB of float32 data on the GPU, writes it to HDF5 using the
 * regular H5Z filter pipeline (CPU path), reads it back, and verifies
 * bitwise round-trip correctness.
 *
 * Dataset : 2 097 152 float32 = 8 MB
 * Chunks  : 1 048 576 float32 = 4 MB each (2 chunks)
 * Filter  : LZ4 + 4-byte shuffle, lossless
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ------------------------------------------------------------------ */
#define N_ELEM  (2097152)       /* 8 MB as float32   */
#define CHUNK   (1048576)       /* 4 MB per chunk    */
#define N_CHUNK (N_ELEM / CHUNK)
#define FNAME   "/tmp/test_h5z_8mb.h5"
#define DSET    "data"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Damped multi-frequency wave — compressible but not trivially so */
__global__ void gen_kernel(float *d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / (float)n;
    d[i] = (__sinf(t * 628.318f) + 0.4f * __cosf(t * 232.478f)
            + 0.1f * __sinf(t * 1884.96f)) * __expf(-t * 3.0f);
}

/* ------------------------------------------------------------------ */
int main(void)
{
    int rc = 0;

    /* ---- Init gpucompress ---- */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "H5Z_gpucompress_register failed\n");
        return 1;
    }

    /* ---- Generate data on GPU, copy to host ---- */
    float *d_src = NULL;
    cudaMalloc(&d_src, N_ELEM * sizeof(float));
    gen_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, N_ELEM);
    cudaDeviceSynchronize();

    float *h_src = (float *)malloc(N_ELEM * sizeof(float));
    float *h_dst = (float *)malloc(N_ELEM * sizeof(float));
    cudaMemcpy(h_src, d_src, N_ELEM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src);

    printf("Dataset : %d floats = %.1f MB\n", N_ELEM, N_ELEM * 4.0 / (1 << 20));
    printf("Chunks  : %d x %d floats = %.1f MB each\n\n", N_CHUNK, CHUNK, CHUNK * 4.0 / (1 << 20));

    /* ---- Create HDF5 file with chunked + gpucompress filter ---- */
    hsize_t dims[1]  = { N_ELEM };
    hsize_t cdims[1] = { CHUNK  };

    hid_t fid   = H5Fcreate(FNAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 4, 0.0);

    hid_t dset  = H5Dcreate2(fid, DSET, H5T_NATIVE_FLOAT, space,
                              H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* ---- Write ---- */
    double   t0         = 0.0;
    double   write_s    = 0.0;
    double   read_s     = 0.0;
    hsize_t  total_comp = 0;
    int      mismatches = 0;
    herr_t   wret, rret;

    t0     = now_sec();
    wret   = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, h_src);
    write_s = now_sec() - t0;

    if (wret < 0) { fprintf(stderr, "H5Dwrite failed\n"); rc = 1; goto cleanup; }

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);

    /* ---- Report compressed chunk sizes ---- */
    fid  = H5Fopen(FNAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(fid, DSET, H5P_DEFAULT);

    printf("Write time : %.3f s  (%.1f MB/s)\n",
           write_s, (N_ELEM * 4.0 / (1 << 20)) / write_s);

    for (int c = 0; c < N_CHUNK; c++) {
        hsize_t offset[1] = { (hsize_t)c * CHUNK };
        hsize_t csz = 0;
        H5Dget_chunk_storage_size(dset, offset, &csz);
        printf("  chunk[%d] offset=%-10llu  raw=%5.1f MB  compressed=%5.2f MB"
               "  ratio=%.2fx\n",
               c, (unsigned long long)offset[0],
               CHUNK * 4.0 / (1 << 20), csz / (1.0 * (1 << 20)),
               (CHUNK * 4.0) / csz);
        total_comp += csz;
    }
    printf("  total compressed : %.2f MB / %.1f MB  (%.2fx overall)\n\n",
           total_comp / (1.0 * (1 << 20)), N_ELEM * 4.0 / (1 << 20),
           (N_ELEM * 4.0) / total_comp);

    /* ---- Read back ---- */
    t0     = now_sec();
    rret   = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, h_dst);
    read_s = now_sec() - t0;

    if (rret < 0) { fprintf(stderr, "H5Dread failed\n"); rc = 1; goto cleanup; }

    printf("Read time  : %.3f s  (%.1f MB/s)\n\n",
           read_s, (N_ELEM * 4.0 / (1 << 20)) / read_s);

    /* ---- Verify bitwise match ---- */
    for (int i = 0; i < N_ELEM; i++)
        if (h_src[i] != h_dst[i]) mismatches++;

    if (mismatches == 0)
        printf("PASS  bitwise match: all %d elements identical\n", N_ELEM);
    else {
        printf("FAIL  %d / %d elements differ\n", mismatches, N_ELEM);
        rc = 1;
    }

cleanup:
    H5Dclose(dset);
    H5Fclose(fid);
    free(h_src);
    free(h_dst);
    gpucompress_cleanup();
    return rc;
}
