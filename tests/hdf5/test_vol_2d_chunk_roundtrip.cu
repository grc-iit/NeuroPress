/**
 * @file test_vol_2d_chunk_roundtrip.cu
 * @brief 2D float32 round-trip test: consecutive values → chunked HDF5 → NN-inference
 *        compress → read back → verify chunk ordering and data integrity.
 *
 * Creates a 2D array filled with consecutive floats (0.0, 1.0, 2.0, ...),
 * writes it through the gpucompress VOL with ALGO_AUTO (NN inference-only),
 * reads it back to a fresh GPU buffer, and verifies:
 *   1. The NN selected a compression algorithm for each chunk
 *   2. Compression actually reduced file size
 *   3. Every float matches bitwise (lossless round-trip)
 *   4. Chunk boundaries are correct (no cross-chunk data corruption)
 *
 * Usage:
 *   ./test_vol_2d_chunk_roundtrip [weights.nnwt]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#include <hdf5.h>
#include <cuda_runtime.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */
#define ROWS          2048
#define COLS          2048
#define TOTAL_FLOATS  (ROWS * COLS)
#define DATA_BYTES    (TOTAL_FLOATS * sizeof(float))

/* Chunk: 2 rows x 2048 cols = 4K floats = 16 KB per chunk → 1024 chunks */
#define CHUNK_ROWS    2
#define CHUNK_COLS    COLS
#define NUM_CHUNKS    (ROWS / CHUNK_ROWS)

#define HDF5_FILE     "/tmp/test_vol_2d_chunk_roundtrip.h5"
#define DSET_NAME     "data"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) do {                                     \
    if (cond) {                                                        \
        printf("  [PASS] " fmt "\n", ##__VA_ARGS__);                   \
        g_pass++;                                                      \
    } else {                                                           \
        printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);                   \
        g_fail++;                                                      \
    }                                                                  \
} while (0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static size_t file_size(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0) ? (size_t)st.st_size : 0;
}

/* ------------------------------------------------------------------ */
/* GPU kernels                                                          */
/* ------------------------------------------------------------------ */

/* Fill 2D array with consecutive floats: data[r][c] = r * COLS + c */
__global__ void fill_consecutive(float* out, int rows, int cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)rows * cols;
    if (idx >= total) return;
    out[idx] = (float)idx;
}

/* Bitwise compare on GPU, returns mismatch count */
__global__ void compare_kernel(const float* a, const float* b, size_t n,
                                unsigned long long* mismatch_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int va, vb;
    memcpy(&va, &a[idx], sizeof(unsigned int));
    memcpy(&vb, &b[idx], sizeof(unsigned int));
    if (va != vb) atomicAdd(mismatch_count, 1ULL);
}

/* Per-chunk boundary check: verify each chunk starts with expected value */
__global__ void check_chunk_boundaries(const float* data, int rows, int cols,
                                        int chunk_rows, unsigned long long* errors) {
    int chunk_id = blockIdx.x;
    int n_chunks = (rows + chunk_rows - 1) / chunk_rows;
    if (chunk_id >= n_chunks) return;

    /* First element of this chunk should be chunk_id * chunk_rows * cols */
    size_t chunk_start = (size_t)chunk_id * chunk_rows * cols;
    float expected_first = (float)chunk_start;

    /* Last element of this chunk */
    int actual_rows = (chunk_id == n_chunks - 1)
                      ? (rows - chunk_id * chunk_rows) : chunk_rows;
    size_t chunk_end = chunk_start + (size_t)actual_rows * cols - 1;
    float expected_last = (float)chunk_end;

    /* Check with threadIdx.x == 0 only */
    if (threadIdx.x == 0) {
        unsigned int va, vb;
        memcpy(&va, &data[chunk_start], sizeof(unsigned int));
        memcpy(&vb, (const void*)&expected_first, sizeof(unsigned int));
        if (va != vb) atomicAdd(errors, 1ULL);

        memcpy(&va, &data[chunk_end], sizeof(unsigned int));
        memcpy(&vb, (const void*)&expected_last, sizeof(unsigned int));
        if (va != vb) atomicAdd(errors, 1ULL);
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    const char* weights = (argc > 1) ? argv[1] : NULL;
    if (!weights) weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) {
        /* Try default path */
        weights = "neural_net/weights/model.nnwt";
    }

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  2D Chunk Round-Trip Test (NN Inference, ALGO_AUTO)      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    printf("  Array     : %d x %d float32 (%.1f MB)\n", ROWS, COLS,
           (double)DATA_BYTES / (1024.0 * 1024.0));
    printf("  Chunks    : %d x %d (%d chunks, %.1f KB each)\n",
           CHUNK_ROWS, CHUNK_COLS, NUM_CHUNKS,
           (double)CHUNK_ROWS * CHUNK_COLS * sizeof(float) / 1024.0);
    printf("  Weights   : %s\n", weights);
    printf("  HDF5 file : %s\n\n", HDF5_FILE);

    /* ── Initialize gpucompress ── */
    gpucompress_init(weights);
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* ── Register VOL ── */
    hid_t vol_id = H5VL_gpucompress_register();
    CHECK(vol_id >= 0, "VOL connector registered (id=%lld)", (long long)vol_id);
    if (vol_id < 0) { gpucompress_cleanup(); return 1; }
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* ── Allocate GPU buffers ── */
    float *d_orig = NULL, *d_read = NULL;
    unsigned long long *d_count = NULL;
    CUDA_CHECK(cudaMalloc(&d_orig, DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_read, DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_read, 0, DATA_BYTES));

    /* ── Fill with consecutive values on GPU ── */
    printf("── Experiment 1: Fill 2D array with consecutive floats ──\n");
    int threads = 256;
    int blocks = ((int)TOTAL_FLOATS + threads - 1) / threads;
    fill_consecutive<<<blocks, threads>>>(d_orig, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Quick host-side sanity: check first and last values */
    float h_first, h_last;
    CUDA_CHECK(cudaMemcpy(&h_first, d_orig, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_last, d_orig + TOTAL_FLOATS - 1, sizeof(float),
                           cudaMemcpyDeviceToHost));
    CHECK(h_first == 0.0f, "First element = %.1f (expected 0.0)", h_first);
    CHECK(h_last == (float)(TOTAL_FLOATS - 1),
          "Last element = %.1f (expected %.1f)", h_last, (float)(TOTAL_FLOATS - 1));

    /* ── Write through VOL with ALGO_AUTO (NN inference) ── */
    printf("\n── Experiment 2: Write via VOL (ALGO_AUTO, NN inference) ──\n");

    gpucompress_flush_manager_cache();
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);

    hid_t fid = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    CHECK(fid >= 0, "H5Fcreate succeeded");

    /* 2D dataspace + 2D chunking */
    hsize_t dims[2]  = { ROWS, COLS };
    hsize_t chunk[2] = { CHUNK_ROWS, CHUNK_COLS };
    hid_t space = H5Screate_simple(2, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, chunk);

    /* ALGO_AUTO = NN selects algorithm per chunk */
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)GPUCOMPRESS_ALGO_AUTO;
    cd[1] = 0;  /* no preprocessing */
    cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);  /* lossless (error_bound = 0) */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, DSET_NAME, H5T_NATIVE_FLOAT,
                             space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CHECK(dset >= 0, "H5Dcreate2 succeeded (2D chunked, ALGO_AUTO)");

    herr_t wrc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_orig);
    CHECK(wrc >= 0, "H5Dwrite succeeded (GPU pointer → VOL pipeline)");

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);

    /* ── Check compression happened ── */
    printf("\n── Experiment 3: Verify compression ──\n");

    size_t fsz = file_size(HDF5_FILE);
    double ratio = (fsz > 0) ? (double)DATA_BYTES / (double)fsz : 0;
    CHECK(fsz > 0, "File created: %zu bytes", fsz);
    CHECK(fsz < DATA_BYTES, "File smaller than raw data (ratio=%.2fx)", ratio);

    /* ── Check per-chunk NN diagnostics ── */
    printf("\n── Experiment 4: Per-chunk NN diagnostics ──\n");

    int n_diag = gpucompress_get_chunk_history_count();
    CHECK(n_diag == NUM_CHUNKS,
          "Chunk history count = %d (expected %d)", n_diag, NUM_CHUNKS);

    static const char* algo_names[] = {
        "AUTO","LZ4","Snappy","Deflate","GDeflate","Zstd","ANS","Cascaded","Bitcomp"
    };

    /* Verify all chunks got valid NN selections; only print first/last/middle */
    int algo_counts[9] = {0};
    int all_valid = 1;
    for (int ci = 0; ci < n_diag && ci < NUM_CHUNKS; ci++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(ci, &diag) != 0) { all_valid = 0; continue; }
        int a = diag.nn_action;
        if (a < 1 || a > 8) { all_valid = 0; continue; }
        algo_counts[a]++;

        /* Print first, middle, and last chunk */
        if (ci == 0 || ci == NUM_CHUNKS / 2 || ci == NUM_CHUNKS - 1) {
            const char* aname = (a >= 0 && a < 9) ? algo_names[a] : "???";
            printf("    chunk[%4d]: %s, ratio=%.2fx, comp=%.1fms\n",
                   ci, aname, diag.actual_ratio, diag.compression_ms_raw);
        }
    }
    CHECK(all_valid, "All %d chunks got valid NN selections (action 1-8)", NUM_CHUNKS);

    /* Print algorithm distribution */
    printf("  Algorithm distribution across %d chunks:\n", NUM_CHUNKS);
    for (int a = 1; a <= 8; a++) {
        if (algo_counts[a] > 0)
            printf("    %-10s: %d chunks (%.0f%%)\n",
                   algo_names[a], algo_counts[a],
                   100.0 * algo_counts[a] / NUM_CHUNKS);
    }

    /* ── Read back and verify ── */
    printf("\n── Experiment 5: Read back + bitwise verification ──\n");

    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    fid = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    CHECK(fid >= 0, "H5Fopen succeeded for read-back");

    dset = H5Dopen2(fid, DSET_NAME, H5P_DEFAULT);
    CHECK(dset >= 0, "H5Dopen2 succeeded");

    herr_t rrc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_read);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(rrc >= 0, "H5Dread succeeded (VOL → GPU decompress)");

    H5Dclose(dset);
    H5Fclose(fid);

    /* Global bitwise compare */
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    compare_kernel<<<blocks, threads>>>(d_orig, d_read, TOTAL_FLOATS, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long mm = 0;
    CUDA_CHECK(cudaMemcpy(&mm, d_count, sizeof(mm), cudaMemcpyDeviceToHost));
    CHECK(mm == 0, "Bitwise lossless round-trip: %llu mismatches out of %d floats",
          mm, TOTAL_FLOATS);

    /* ── Chunk boundary verification ── */
    printf("\n── Experiment 6: Chunk boundary integrity ──\n");

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    check_chunk_boundaries<<<NUM_CHUNKS, 1>>>(d_read, ROWS, COLS, CHUNK_ROWS, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long boundary_errors = 0;
    CUDA_CHECK(cudaMemcpy(&boundary_errors, d_count, sizeof(boundary_errors),
                           cudaMemcpyDeviceToHost));
    CHECK(boundary_errors == 0,
          "Chunk boundaries correct: %llu errors across %d chunks",
          boundary_errors, NUM_CHUNKS);

    /* Spot-check: print first value of a few chunks from read-back */
    printf("\n  Chunk start values (spot-check):\n");
    int spot_checks[] = { 0, 1, NUM_CHUNKS / 4, NUM_CHUNKS / 2,
                          3 * NUM_CHUNKS / 4, NUM_CHUNKS - 2, NUM_CHUNKS - 1 };
    int n_spots = sizeof(spot_checks) / sizeof(spot_checks[0]);
    for (int si = 0; si < n_spots; si++) {
        int ci = spot_checks[si];
        if (ci < 0 || ci >= NUM_CHUNKS) continue;
        size_t offset = (size_t)ci * CHUNK_ROWS * COLS;
        float h_val;
        CUDA_CHECK(cudaMemcpy(&h_val, d_read + offset, sizeof(float),
                               cudaMemcpyDeviceToHost));
        float expected = (float)offset;
        const char* ok = (h_val == expected) ? "OK" : "MISMATCH";
        printf("    chunk[%4d]: data[%7zu] = %10.1f (expected %10.1f) %s\n",
               ci, offset, h_val, expected, ok);
    }

    /* ── Cleanup ── */
    cudaFree(d_orig);
    cudaFree(d_read);
    cudaFree(d_count);
    remove(HDF5_FILE);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("════════════════════════════════════════════════════════════\n");
    return (g_fail > 0) ? 1 : 0;
}
