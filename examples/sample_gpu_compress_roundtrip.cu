/**
 * sample_gpu_compress_roundtrip.cu
 *
 * Minimal end-to-end example:
 *   1. Allocate float32 data on the GPU
 *   2. Write to HDF5 using the gpucompress VOL (GPU compression path)
 *   3. Close the dataset and file
 *   4. Reopen and read back into a fresh GPU buffer (GPU decompression path)
 *   5. Validate: every byte matches the original exactly (lossless)
 *
 * The VOL connector is what makes GPU pointers work with H5Dwrite / H5Dread.
 * Without it, passing a device pointer to HDF5 would result in garbage data.
 *
 * Build:
 *   cmake --build build --target sample_gpu_compress_roundtrip
 *
 * Run:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *     ./build/sample_gpu_compress_roundtrip
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

#define N_ELEMENTS   (1024 * 1024)          /* 4 MB as float32         */
#define CHUNK_ELEMS  (256 * 1024)           /* 1 MB per chunk (4 total)*/
#define OUTPUT_FILE  "/tmp/sample_gpucompress.h5"
#define DATASET_NAME "sensor_data"

/* gpucompress filter ID and cd_values layout */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ------------------------------------------------------------------ */
/* GPU kernels                                                         */
/* ------------------------------------------------------------------ */

/* Fill buffer with a sine wave so there is something compressible */
__global__ void fill_kernel(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / (float)n;
    d[i] = __sinf(t * 6.283185f * 16.0f)   /* 16 cycles across buffer */
           + 0.5f * __cosf(t * 6.283185f * 37.0f);
}

/* Count element-wise mismatches between two device buffers */
__global__ void count_mismatches_kernel(const float *a, const float *b,
                                        int n, int *out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i])
        atomicAdd(out_count, 1);
}

/* ------------------------------------------------------------------ */
/* HDF5 VOL helpers                                                    */
/* ------------------------------------------------------------------ */

static void pack_double(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/*
 * Open (or create) an HDF5 file with the gpucompress VOL registered.
 * flags: H5F_ACC_TRUNC (create/overwrite) or H5F_ACC_RDONLY (read-only).
 */
static hid_t open_vol_file(const char *path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    assert(vol_id >= 0 && "Failed to register gpucompress VOL connector");

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl >= 0);

    herr_t rc = H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    assert(rc >= 0 && "H5Pset_fapl_gpucompress failed");

    hid_t fid;
    if (flags & H5F_ACC_TRUNC)
        fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    else
        fid = H5Fopen(path, H5F_ACC_RDONLY, fapl);

    H5Pclose(fapl);
    H5VLclose(vol_id);   /* decrement ref — file keeps the connector alive */
    return fid;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("=== gpucompress lossless round-trip example ===\n\n");

    /* ----------------------------------------------------------------
     * Step 0: Initialise gpucompress
     * -------------------------------------------------------------- */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init() failed\n");
        return 1;
    }
    printf("[0] gpucompress initialised\n");

    /* ----------------------------------------------------------------
     * Step 1: Allocate and fill data on the GPU
     * -------------------------------------------------------------- */
    const size_t bytes = (size_t)N_ELEMENTS * sizeof(float);

    float *d_original = NULL;
    assert(cudaMalloc(&d_original, bytes) == cudaSuccess);

    fill_kernel<<<(N_ELEMENTS + 255) / 256, 256>>>(d_original, N_ELEMENTS);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    printf("[1] Allocated and filled %zu MB on GPU (%d elements)\n",
           bytes >> 20, N_ELEMENTS);

    /* ----------------------------------------------------------------
     * Step 2: Write to HDF5 using the gpucompress VOL
     *
     * Key points:
     *   - H5Pset_chunk()   tells HDF5 (and the VOL) the chunk shape
     *   - H5Pset_filter()  sets the gpucompress filter on the DCPL
     *   - H5Dwrite()       passes d_original (device ptr) to the VOL
     *   - VOL intercepts:  GPU path active because:
     *       (a) buffer is a device pointer
     *       (b) DCPL has the gpucompress filter
     *   - Data flow: d_original → GPU compress → compressed bytes D→H
     *                → H5Dwrite_chunk() to disk
     * -------------------------------------------------------------- */
    {
        hid_t fid = open_vol_file(OUTPUT_FILE, H5F_ACC_TRUNC);
        assert(fid >= 0 && "Failed to create HDF5 file");

        /* Dataset shape: 1D, N_ELEMENTS floats */
        hsize_t dims[1]  = { N_ELEMENTS };
        hid_t   space    = H5Screate_simple(1, dims, NULL);
        assert(space >= 0);

        /* Dataset creation property list: chunked layout */
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        assert(dcpl >= 0);

        hsize_t chunk[1] = { CHUNK_ELEMS };
        H5Pset_chunk(dcpl, 1, chunk);

        /* Set gpucompress filter:
         *   cd[0] = algorithm   (1 = LZ4, lossless)
         *   cd[1] = preprocessing (2 = byte-shuffle only, no quantization)
         *   cd[2] = shuffle_size  (4 = float32)
         *   cd[3..4] = error_bound (0.0 = lossless, packed as two uint32s) */
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = 1;   /* LZ4 */
        cd[1] = 2;   /* byte-shuffle4, no quantization */
        cd[2] = 4;   /* float32 element size */
        pack_double(0.0, &cd[3], &cd[4]);   /* error_bound = 0.0 → lossless */

        herr_t rc = H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                                  H5Z_FLAG_OPTIONAL,
                                  H5Z_GPUCOMPRESS_CD_NELMTS, cd);
        assert(rc >= 0 && "H5Pset_filter failed");

        /* Create dataset */
        hid_t dset = H5Dcreate2(fid, DATASET_NAME, H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dcreate2 failed");

        /* Write — d_original is a device pointer.
         * The VOL detects this and routes through the GPU compression path. */
        rc = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                      H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      d_original);
        assert(rc >= 0 && "H5Dwrite failed");

        H5Dclose(dset);
        H5Sclose(space);
        H5Pclose(dcpl);
        H5Fclose(fid);

        printf("[2] Written to %s (GPU compression path active)\n", OUTPUT_FILE);
    }

    /* ----------------------------------------------------------------
     * Step 3: Read back into a fresh GPU buffer
     *
     * Data flow: disk → H5Dread_chunk() → compressed bytes H→D
     *            → GPU decompress → d_recovered (device)
     * The VOL routes through the GPU decompression path because
     * d_recovered is a device pointer AND the DCPL has the gpucompress filter.
     * -------------------------------------------------------------- */
    float *d_recovered = NULL;
    assert(cudaMalloc(&d_recovered, bytes) == cudaSuccess);
    assert(cudaMemset(d_recovered, 0, bytes) == cudaSuccess);

    {
        hid_t fid = open_vol_file(OUTPUT_FILE, H5F_ACC_RDONLY);
        assert(fid >= 0 && "Failed to open HDF5 file for reading");

        hid_t dset = H5Dopen2(fid, DATASET_NAME, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        /* Read — d_recovered is a device pointer.
         * The VOL detects this and routes through GPU decompression. */
        herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL, H5P_DEFAULT,
                             d_recovered);
        assert(rc >= 0 && "H5Dread failed");

        H5Dclose(dset);
        H5Fclose(fid);

        printf("[3] Read back from disk (GPU decompression path active)\n");
    }

    /* ----------------------------------------------------------------
     * Step 4: Validate — every element must match exactly (lossless)
     *
     * Comparison happens on the GPU to avoid a full D→H copy.
     * -------------------------------------------------------------- */
    int *d_mismatch_count = NULL;
    assert(cudaMalloc(&d_mismatch_count, sizeof(int)) == cudaSuccess);
    assert(cudaMemset(d_mismatch_count, 0, sizeof(int)) == cudaSuccess);

    count_mismatches_kernel<<<(N_ELEMENTS + 255) / 256, 256>>>(
        d_original, d_recovered, N_ELEMENTS, d_mismatch_count);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    int h_mismatch_count = 0;
    assert(cudaMemcpy(&h_mismatch_count, d_mismatch_count,
                      sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    printf("[4] Validation: %d / %d elements mismatch\n",
           h_mismatch_count, N_ELEMENTS);

    if (h_mismatch_count == 0) {
        printf("\n    RESULT: PASS — lossless round-trip verified (100%% match)\n");
    } else {
        printf("\n    RESULT: FAIL — %d elements differ\n", h_mismatch_count);
    }

    /* ----------------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------------- */
    cudaFree(d_original);
    cudaFree(d_recovered);
    cudaFree(d_mismatch_count);
    gpucompress_cleanup();

    return (h_mismatch_count == 0) ? 0 : 1;
}
