/**
 * test_qw4_atomic_counters.cu
 *
 * QW-4: s_gpu_writes / s_chunks_comp / s_gpu_reads / s_chunks_decomp
 *       were plain int — a C++ data race.
 *
 * The real race for s_chunks_comp is INSIDE the VOL connector:
 *   Stage-2 has up to 8 worker threads, each incrementing s_chunks_comp
 *   upon successful compression of its chunk.  With N_CHUNKS > 1, multiple
 *   workers fire concurrently → concurrent s_chunks_comp++ without atomics
 *   → lost increments → counter < N_CHUNKS.
 *
 * Fix: changed all four counters to std::atomic<int>.
 *
 * Test strategy:
 *   1. Reset all counters.
 *   2. Write a dataset with N_CHUNKS chunks (triggers N_CHUNKS concurrent
 *      Stage-2 worker increments of s_chunks_comp).
 *   3. After H5Dwrite returns, assert s_chunks_comp == N_CHUNKS.
 *   4. Read it back; assert s_chunks_decomp == N_CHUNKS.
 *   5. Repeat ROUNDS times and assert cumulative totals.
 *   6. Verify reset() zeros all counters.
 *
 * Run: ./test_qw4_atomic_counters
 *   Expected: all checks PASS, OVERALL: PASS
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Config                                                               */
/* ------------------------------------------------------------------ */
#define N_CHUNKS      16         /* 16 concurrent Stage-2 worker increments */
#define CHUNK_ELEMS   (32*1024)  /* 128 KB per chunk (float32)               */
#define ROUNDS        3          /* repeat write+read ROUNDS times            */

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

/* ------------------------------------------------------------------ */
/* Filter helpers                                                       */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== QW-4: Atomic VOL statistics counters ===\n");
    printf("N_CHUNKS=%d × CHUNK_ELEMS=%d floats, ROUNDS=%d\n\n",
           N_CHUNKS, CHUNK_ELEMS, ROUNDS);

    /* VOL connector does not call gpucompress_init; do it explicitly */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init() failed\n");
        return 1;
    }

    const char* path = "/tmp/test_qw4.h5";
    size_t total_elems = (size_t)N_CHUNKS * CHUNK_ELEMS;
    size_t nbytes      = total_elems * sizeof(float);

    /* Prepare host + device data */
    float* h_data = (float*)malloc(nbytes);
    float* h_back = (float*)malloc(nbytes);
    float* d_data = nullptr;
    float* d_out  = nullptr;
    if (!h_data || !h_back) { fprintf(stderr, "malloc failed\n"); return 1; }
    if (cudaMalloc(&d_data, nbytes) != cudaSuccess ||
        cudaMalloc(&d_out,  nbytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); return 1;
    }

    for (size_t i = 0; i < total_elems; i++)
        h_data[i] = (float)i * 0.001f + 1.0f;
    cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice);

    /* ---------------------------------------------------------------- */
    /* Round loop                                                         */
    /* ---------------------------------------------------------------- */
    H5VL_gpucompress_reset_stats();

    for (int round = 0; round < ROUNDS; round++) {
        printf("--- Round %d/%d ---\n", round+1, ROUNDS);

        /* CREATE + WRITE */
        hid_t fid = open_vol_file(path, H5F_ACC_TRUNC);
        if (fid < 0) { FAIL("open_vol_file(write)"); continue; }

        hsize_t dims[1]  = { (hsize_t)total_elems };
        hsize_t cdims[1] = { (hsize_t)CHUNK_ELEMS };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
        cd[0] = 5; /* ZSTD — reliable, no oversized-buffer issues */
        cd[1] = 2; /* shuffle4 */
        cd[2] = 4; /* shuffle element size */
        pack_double(0.0, &cd[3], &cd[4]); /* lossless */
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL,
                      H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(space); H5Pclose(dcpl);
        if (dset < 0) { FAIL("H5Dcreate2"); H5Fclose(fid); continue; }

        herr_t we = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_data);
        H5Dclose(dset);
        H5Fclose(fid);

        if (we < 0) { FAIL("H5Dwrite"); continue; }

        /* READ BACK */
        hid_t fid2 = open_vol_file(path, H5F_ACC_RDONLY);
        if (fid2 < 0) { FAIL("open_vol_file(read)"); continue; }

        hid_t dset2  = H5Dopen2(fid2, "data", H5P_DEFAULT);
        herr_t re = H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_out);
        H5Dclose(dset2);
        H5Fclose(fid2);

        if (re < 0) { FAIL("H5Dread"); continue; }

        cudaMemcpy(h_back, d_out, nbytes, cudaMemcpyDeviceToHost);

        /* Verify data */
        int ok = 1;
        for (size_t i = 0; i < total_elems && ok; i++)
            if (h_data[i] != h_back[i]) ok = 0;
        if (ok) PASS("round-trip byte-exact");
        else    FAIL("round-trip data mismatch");

        /* Check per-round counters */
        int writes = 0, reads = 0, comp = 0, decomp = 0;
        H5VL_gpucompress_get_stats(&writes, &reads, &comp, &decomp);

        printf("  Cumulative after round %d: writes=%d reads=%d comp=%d decomp=%d\n",
               round+1, writes, reads, comp, decomp);

        int expected_writes = round + 1;
        int expected_reads  = round + 1;
        int expected_comp   = (round + 1) * N_CHUNKS;
        int expected_decomp = (round + 1) * N_CHUNKS;

        if (writes == expected_writes) PASS("s_gpu_writes correct");
        else { printf("  FAIL: s_gpu_writes=%d expected=%d\n", writes, expected_writes); g_fail++; }

        if (reads == expected_reads)   PASS("s_gpu_reads correct");
        else { printf("  FAIL: s_gpu_reads=%d expected=%d\n", reads, expected_reads); g_fail++; }

        if (comp == expected_comp)     PASS("s_chunks_comp correct (Stage-2 atomic increments)");
        else { printf("  FAIL: s_chunks_comp=%d expected=%d (lost increments!)\n", comp, expected_comp); g_fail++; }

        if (decomp == expected_decomp) PASS("s_chunks_decomp correct");
        else { printf("  FAIL: s_chunks_decomp=%d expected=%d\n", decomp, expected_decomp); g_fail++; }
    }

    /* ---------------------------------------------------------------- */
    /* Test reset                                                         */
    /* ---------------------------------------------------------------- */
    printf("--- Test reset() ---\n");
    H5VL_gpucompress_reset_stats();
    int w=1, r=1, c=1, d=1;
    H5VL_gpucompress_get_stats(&w, &r, &c, &d);
    if (w == 0 && r == 0 && c == 0 && d == 0) PASS("reset() zeros all counters");
    else { printf("  FAIL: after reset: w=%d r=%d c=%d d=%d\n", w, r, c, d); g_fail++; }

    /* ---------------------------------------------------------------- */
    /* Summary                                                            */
    /* ---------------------------------------------------------------- */
    free(h_data); free(h_back);
    cudaFree(d_data); cudaFree(d_out);
    remove(path);

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    gpucompress_cleanup();
    return g_fail == 0 ? 0 : 1;
}
