/**
 * test_vol_8mb.cu
 *
 * Same 8 MB / 4 MB chunk test as test_h5z_8mb, but using the GPUCompress
 * VOL connector.  Data is generated on the GPU and passed as device pointers
 * directly to H5Dwrite / H5Dread — the CPU never touches the raw floats.
 *
 * Dataset : 2 097 152 float32 = 8 MB
 * Chunks  : 1 048 576 float32 = 4 MB each (2 chunks)
 * Filter  : LZ4 + 4-byte shuffle via VOL GPU path
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"

/* ------------------------------------------------------------------ */
#define N_ELEM  (2097152)
#define CHUNK   (1048576)
#define N_CHUNK (N_ELEM / CHUNK)
#define FNAME   "/tmp/test_vol_8mb.h5"
#define DSET    "data"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void gen_kernel(float *d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / (float)n;
    d[i] = (__sinf(t * 628.318f) + 0.4f * __cosf(t * 232.478f)
            + 0.1f * __sinf(t * 1884.96f)) * __expf(-t * 3.0f);
}

__global__ void verify_kernel(const float *ref, const float *out,
                               int n, int *mismatches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ref[i] != out[i]) atomicAdd(mismatches, 1);
}

/* ------------------------------------------------------------------ */
int main(void)
{
    int rc = 0;

    /* ---- Init gpucompress + register filter (needed for DCPL params) ---- */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n"); return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "H5Z_gpucompress_register failed\n"); return 1;
    }

    /* ---- Register VOL connector ---- */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) { fprintf(stderr, "H5VL_gpucompress_register failed\n"); return 1; }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t native_id = H5VLget_connector_id_by_name("native");
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    /* ---- Generate data on GPU ---- */
    float *d_src = NULL, *d_dst = NULL;
    cudaMalloc(&d_src, N_ELEM * sizeof(float));
    cudaMalloc(&d_dst, N_ELEM * sizeof(float));
    gen_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, N_ELEM);
    cudaDeviceSynchronize();

    printf("Dataset : %d floats = %.1f MB\n", N_ELEM, N_ELEM * 4.0 / (1 << 20));
    printf("Chunks  : %d x %d floats = %.1f MB each\n", N_CHUNK, CHUNK, CHUNK * 4.0 / (1 << 20));
    printf("Path    : VOL connector (GPU device pointers)\n\n");

    /* ---- Create file + dataset with gpucompress filter ---- */
    hsize_t dims[1]  = { N_ELEM };
    hsize_t cdims[1] = { CHUNK  };

    hid_t fid   = H5Fcreate(FNAME, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 4, 0.0);

    hid_t dset = H5Dcreate2(fid, DSET, H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* ---- Write — GPU device pointer, VOL intercepts ---- */
    H5VL_gpucompress_reset_stats();
    double t0    = now_sec();
    herr_t wret  = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_src);
    double write_s = now_sec() - t0;

    if (wret < 0) { fprintf(stderr, "H5Dwrite failed\n"); rc = 1; goto cleanup; }

    {
        int writes, comp;
        H5VL_gpucompress_get_stats(&writes, NULL, &comp, NULL);
        printf("Write time : %.3f s  (%.1f MB/s)  [VOL gpu_writes=%d  chunks_comp=%d]\n",
               write_s, (N_ELEM * 4.0 / (1 << 20)) / write_s, writes, comp);
    }

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);

    /* ---- Report compressed chunk sizes ---- */
    {
        hid_t fid2  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
        hid_t dset2 = H5Dopen2(fid2, DSET, H5P_DEFAULT);
        hsize_t total_comp = 0;
        for (int c = 0; c < N_CHUNK; c++) {
            hsize_t offset[1] = { (hsize_t)c * CHUNK };
            hsize_t csz = 0;
            H5Dget_chunk_storage_size(dset2, offset, &csz);
            printf("  chunk[%d]  raw=%5.1f MB  compressed=%5.2f MB  ratio=%.2fx\n",
                   c, CHUNK * 4.0 / (1 << 20), csz / (1.0 * (1 << 20)),
                   (CHUNK * 4.0) / csz);
            total_comp += csz;
        }
        printf("  total compressed : %.2f MB / %.1f MB  (%.2fx overall)\n\n",
               total_comp / (1.0 * (1 << 20)), N_ELEM * 4.0 / (1 << 20),
               (N_ELEM * 4.0) / total_comp);
        H5Dclose(dset2);
        H5Fclose(fid2);
    }

    /* ---- Read back — GPU device pointer ---- */
    {
        hid_t fid2  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
        hid_t dset2 = H5Dopen2(fid2, DSET, H5P_DEFAULT);

        H5VL_gpucompress_reset_stats();
        double t1   = now_sec();
        herr_t rret = H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, d_dst);
        double read_s = now_sec() - t1;

        if (rret < 0) { fprintf(stderr, "H5Dread failed\n"); rc = 1;
                        H5Dclose(dset2); H5Fclose(fid2); goto cleanup; }

        int reads, decomp;
        H5VL_gpucompress_get_stats(NULL, &reads, NULL, &decomp);
        printf("Read time  : %.3f s  (%.1f MB/s)  [VOL gpu_reads=%d  chunks_decomp=%d]\n\n",
               read_s, (N_ELEM * 4.0 / (1 << 20)) / read_s, reads, decomp);

        H5Dclose(dset2);
        H5Fclose(fid2);
    }

    /* ---- Verify on GPU ---- */
    {
        int *d_mm = NULL;
        cudaMalloc(&d_mm, sizeof(int));
        cudaMemset(d_mm, 0, sizeof(int));
        verify_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, d_dst, N_ELEM, d_mm);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_mm, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_mm);

        if (mismatches == 0)
            printf("PASS  bitwise match: all %d elements identical\n", N_ELEM);
        else {
            printf("FAIL  %d / %d elements differ\n", mismatches, N_ELEM);
            rc = 1;
        }
    }

cleanup:
    cudaFree(d_src);
    cudaFree(d_dst);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return rc;
}
