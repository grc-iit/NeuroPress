/**
 * test_perf16_gather_stream.cu
 *
 * PERF-16: gather_chunk_kernel / scatter_chunk_kernel were launched on the
 * null (default) CUDA stream, causing implicit serialization against all
 * 8 CompContext worker streams: Stage-1 would stall the entire worker pool
 * while gathering each non-contiguous chunk.
 *
 * Fix: dedicated gather_stream / scatter_stream per write/read call.
 *
 * Test strategy:
 *   - 2D dataset (N_ROWS × NCOLS) chunked as (CROW × CCOL) where CCOL <
 *     NCOLS, forcing every chunk through the gather_chunk_kernel path.
 *   - N_ROWS/CROW × NCOLS/CCOL = N_CHUNKS (≥ 8) ensures all worker slots
 *     are exercised concurrently.
 *   - Round-trip: byte-exact verify.
 *   - If the fix is wrong (null stream serialization restored), the test
 *     still passes but would deadlock or be very slow; correctness is the
 *     primary guard here.
 *
 * Run: ./test_perf16_gather_stream
 *   Expected: all checks PASS, OVERALL: PASS
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Config: 4×4 grid of (4-row × 256-col) chunks = 16 non-contiguous  */
/* ------------------------------------------------------------------ */
#define N_ROWS  16
#define N_COLS  1024
#define CROW     4    /* chunk rows    */
#define CCOL   256    /* chunk cols < N_COLS → always non-contiguous */
#define N_CHUNKS ((N_ROWS/CROW) * (N_COLS/CCOL))   /* == 16 */

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    hid_t fid = (flags & H5F_ACC_TRUNC)
        ? H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        : H5Fopen(path, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

int main(void) {
    printf("=== PERF-16: gather/scatter kernel on dedicated stream ===\n");
    printf("Dataset %d×%d float32, chunk %d×%d → %d non-contiguous chunks\n\n",
           N_ROWS, N_COLS, CROW, CCOL, N_CHUNKS);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init() failed\n");
        return 1;
    }

    const char* path = "/tmp/test_perf16.h5";
    size_t total_elems = (size_t)N_ROWS * N_COLS;
    size_t nbytes      = total_elems * sizeof(float);

    float *h_data = (float*)malloc(nbytes);
    float *h_back = (float*)malloc(nbytes);
    float *d_data = nullptr, *d_out = nullptr;
    if (!h_data || !h_back ||
        cudaMalloc(&d_data, nbytes) != cudaSuccess ||
        cudaMalloc(&d_out,  nbytes) != cudaSuccess) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < total_elems; i++)
        h_data[i] = (float)i * 0.001f + 1.0f;
    cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice);

    /* --- WRITE --- */
    hid_t fid = open_vol_file(path, H5F_ACC_TRUNC);
    if (fid < 0) { FAIL("open_vol_file(write)"); goto cleanup; }

    {
        hsize_t dims[2]  = { N_ROWS, N_COLS };
        hsize_t cdims[2] = { CROW,   CCOL   };
        hid_t space = H5Screate_simple(2, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, cdims);
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
        cd[0] = 5; cd[1] = 2; cd[2] = 4; /* ZSTD, shuffle4 */
        pack_double(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL,
                      H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(space); H5Pclose(dcpl);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Fclose(fid); goto cleanup; }

        herr_t we = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_data);
        H5Dclose(dset);
        H5Fclose(fid);

        if (we < 0) { FAIL("H5Dwrite"); goto cleanup; }
        PASS("H5Dwrite with 16 non-contiguous chunks");
    }

    /* --- READ BACK --- */
    {
        hid_t fid2 = open_vol_file(path, H5F_ACC_RDONLY);
        if (fid2 < 0) { FAIL("open_vol_file(read)"); goto cleanup; }

        hid_t dset2 = H5Dopen2(fid2, "data", H5P_DEFAULT);
        herr_t re = H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_out);
        H5Dclose(dset2);
        H5Fclose(fid2);

        if (re < 0) { FAIL("H5Dread"); goto cleanup; }
        PASS("H5Dread with 16 non-contiguous chunks (scatter kernel)");
    }

    cudaMemcpy(h_back, d_out, nbytes, cudaMemcpyDeviceToHost);

    {
        int ok = 1;
        for (size_t i = 0; i < total_elems && ok; i++)
            if (h_data[i] != h_back[i]) { ok = 0;
                printf("  mismatch at [%zu]: wrote %f, read %f\n", i, h_data[i], h_back[i]); }
        if (ok) PASS("round-trip byte-exact (2D non-contiguous chunks)");
        else    FAIL("round-trip data mismatch");
    }

    /* Verify stats: all 16 chunks must have been compressed and decompressed */
    {
        int writes = 0, reads = 0, comp = 0, decomp = 0;
        H5VL_gpucompress_get_stats(&writes, &reads, &comp, &decomp);
        printf("  Stats: writes=%d reads=%d comp=%d decomp=%d\n",
               writes, reads, comp, decomp);
        if (comp >= N_CHUNKS)   PASS("all non-contiguous chunks compressed");
        else { printf("  FAIL: comp=%d expected>=%d\n", comp, N_CHUNKS); g_fail++; }
        if (decomp >= N_CHUNKS) PASS("all non-contiguous chunks decompressed");
        else { printf("  FAIL: decomp=%d expected>=%d\n", decomp, N_CHUNKS); g_fail++; }
    }

cleanup:
    free(h_data); free(h_back);
    cudaFree(d_data); cudaFree(d_out);
    remove(path);

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    gpucompress_cleanup();
    return g_fail == 0 ? 0 : 1;
}
