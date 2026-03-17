/**
 * tests/hdf5/test_lru1_manager_cache.cu
 *
 * LRU-1 nvcomp Manager Cache Test
 *
 * Validates that the nvcomp compression manager is cached per CompContext slot
 * and reused across chunks, eliminating per-call workspace alloc/free overhead.
 *
 * All data flows through the HDF5 VOL — same path as real I/O workloads:
 *   GPU kernel → H5Dwrite(GPU ptr) → VOL → 8 workers → compress → I/O → HDF5
 *
 * ─── Test Cases ─────────────────────────────────────────────────────────────
 *
 * A. Fixed algorithm, same-algo cache reuse
 *    16 chunks × 4 MiB, all LZ4. After 8 workers process the first batch
 *    (8 misses), the second batch of 8 chunks should hit the cached LZ4
 *    manager in each slot → at least 8 cache hits.
 *
 * B. Multiple rounds, sustained cache benefit
 *    32 chunks × 4 MiB, all ZSTD. After the first 8 cold misses, the
 *    remaining 24 chunks should all be cache hits → hits >= 24.
 *
 * C. Algorithm switch detection
 *    Write 16 chunks with LZ4 (batch 1+2), then 16 chunks with ZSTD
 *    (batch 3+4). The switch from LZ4→ZSTD in batch 3 triggers 8 new
 *    misses. Batch 4 reuses the ZSTD managers → 8 hits.
 *    Total: hits >= 16 (8 from LZ4 batch 2 + 8 from ZSTD batch 4).
 *
 * D. Data correctness after caching
 *    After all cache tests, verify that compressed data round-trips
 *    bit-exactly. Caching must not corrupt compression output.
 *
 * ─── Expected Results ───────────────────────────────────────────────────────
 *
 * Pre-fix  (no caching):  hits=0 for all tests        → FAIL
 * Post-fix (LRU-1 cache): hits >= thresholds per test  → PASS
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/test_lru1_manager_cache
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Constants
 * ============================================================ */
#define CHUNK_MB       4
#define CHUNK_FLOATS   ((size_t)CHUNK_MB * 1024 * 1024 / sizeof(float))
#define TMP_FILE       "/tmp/test_lru1_cache.h5"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * GPU kernel: ramp pattern (deterministic, formula-verifiable)
 * ============================================================ */
__global__ static void ramp_kernel(float *b, size_t n, size_t total)
{
    for (size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x; i < n;
         i += gridDim.x*(size_t)blockDim.x)
        b[i] = (float)i / (float)total;
}

static inline float expected(size_t i, size_t total) {
    return (float)i / (float)total;
}

/* ============================================================
 * Helpers
 * ============================================================ */
static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg, ...) do { \
    if (cond) { printf("    [PASS] " msg "\n", ##__VA_ARGS__); g_pass++; } \
    else      { printf("    [FAIL] " msg "\n", ##__VA_ARGS__); g_fail++; } } while(0)

static hid_t make_vol_fapl(void) {
    hid_t native = H5VLget_connector_id_by_name("native");
    hid_t fapl   = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native, NULL);
    H5VLclose(native);
    return fapl;
}

/* DCPL for a specific algorithm (cd[0] = algo_id 1-8) */
static hid_t make_dcpl(int algo_id, size_t chunk_floats) {
    hsize_t cdims[1] = { (hsize_t)chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = (unsigned int)algo_id;  /* 1=LZ4, 5=ZSTD, etc. */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

/* Write n_chunks through HDF5 VOL (GPU pointer), return 0 on success */
static int vol_write(float *d_data, size_t n_floats, size_t chunk_floats,
                     int algo_id)
{
    hsize_t dims[1]  = { (hsize_t)n_floats };

    remove(TMP_FILE);
    hid_t fapl = make_vol_fapl();
    hid_t f    = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (f < 0) return -1;

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = make_dcpl(algo_id, chunk_floats);
    hid_t ds   = H5Dcreate2(f, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fsp);

    herr_t w = H5Dwrite(ds, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(ds); H5Fclose(f);
    return (w < 0) ? -1 : 0;
}

/* Read back through HDF5 VOL into GPU buffer, return 0 on success */
static int vol_read(float *d_read, size_t n_floats)
{
    hid_t fapl = make_vol_fapl();
    hid_t f    = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (f < 0) return -1;

    hid_t ds = H5Dopen2(f, "data", H5P_DEFAULT);
    herr_t r = H5Dread(ds, H5T_NATIVE_FLOAT,
                        H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(f);
    return (r < 0) ? -1 : 0;
}

/* ============================================================
 * TEST A: Fixed algorithm, same-algo cache reuse
 *
 * 16 chunks × LZ4. Workers=8 → 2 batches.
 * Batch 1: 8 cold misses. Batch 2: 8 hits (cached LZ4).
 * ============================================================ */
static void test_A(float *d_data)
{
    printf("\n── A: Fixed LZ4, 16 chunks (2 batches of 8) ─────────────\n");

    const int N = 16;
    const size_t n_floats = (size_t)N * CHUNK_FLOATS;

    ramp_kernel<<<512, 256>>>(d_data, n_floats, n_floats);
    cudaDeviceSynchronize();

    gpucompress_reset_cache_stats();
    int rc = vol_write(d_data, n_floats, CHUNK_FLOATS, 1 /* LZ4 */);
    CHECK(rc == 0, "VOL write succeeded");

    int hits = 0, misses = 0;
    gpucompress_get_cache_stats(&hits, &misses);
    printf("    Cache stats: hits=%d  misses=%d  total=%d\n",
           hits, misses, hits + misses);

    CHECK(hits + misses == N,
          "Total manager operations = %d (expected %d)", hits + misses, N);
    CHECK(hits >= 8,
          "Cache hits >= 8 (got %d) — second batch reused cached managers", hits);

    remove(TMP_FILE);
}

/* ============================================================
 * TEST B: Multiple rounds, sustained cache
 *
 * 32 chunks × ZSTD. Workers=8 → 4 batches.
 * Batch 1: 8 misses. Batches 2-4: 24 hits.
 * ============================================================ */
static void test_B(float *d_data)
{
    printf("\n── B: Fixed ZSTD, 32 chunks (4 batches of 8) ────────────\n");

    const int N = 32;
    const size_t n_floats = (size_t)N * CHUNK_FLOATS;

    ramp_kernel<<<512, 256>>>(d_data, n_floats, n_floats);
    cudaDeviceSynchronize();

    gpucompress_reset_cache_stats();
    int rc = vol_write(d_data, n_floats, CHUNK_FLOATS, 5 /* ZSTD */);
    CHECK(rc == 0, "VOL write succeeded");

    int hits = 0, misses = 0;
    gpucompress_get_cache_stats(&hits, &misses);
    printf("    Cache stats: hits=%d  misses=%d  total=%d\n",
           hits, misses, hits + misses);

    CHECK(hits + misses == N,
          "Total manager operations = %d (expected %d)", hits + misses, N);
    CHECK(hits >= 24,
          "Cache hits >= 24 (got %d) — batches 2-4 reused cached managers", hits);

    remove(TMP_FILE);
}

/* ============================================================
 * TEST C: Algorithm switch detection
 *
 * Write 16 chunks LZ4 → then 16 chunks ZSTD (separate H5Dwrite calls).
 * LZ4 pass: 8 misses + 8 hits.
 * ZSTD pass: 8 misses (switch) + 8 hits.
 * Total across both: hits >= 16.
 * ============================================================ */
static void test_C(float *d_data)
{
    printf("\n── C: Algorithm switch LZ4 → ZSTD (16+16 chunks) ────────\n");

    const int N = 16;
    const size_t n_floats = (size_t)N * CHUNK_FLOATS;

    ramp_kernel<<<512, 256>>>(d_data, n_floats, n_floats);
    cudaDeviceSynchronize();

    gpucompress_reset_cache_stats();

    /* Pass 1: LZ4 */
    int rc1 = vol_write(d_data, n_floats, CHUNK_FLOATS, 1 /* LZ4 */);
    CHECK(rc1 == 0, "LZ4 pass write succeeded");

    int h1 = 0, m1 = 0;
    gpucompress_get_cache_stats(&h1, &m1);
    printf("    After LZ4  : hits=%d  misses=%d\n", h1, m1);

    /* Pass 2: ZSTD (same GPU data, different algo) */
    int rc2 = vol_write(d_data, n_floats, CHUNK_FLOATS, 5 /* ZSTD */);
    CHECK(rc2 == 0, "ZSTD pass write succeeded");

    int h2 = 0, m2 = 0;
    gpucompress_get_cache_stats(&h2, &m2);
    printf("    After ZSTD : hits=%d  misses=%d  (cumulative)\n", h2, m2);

    int total_hits = h2;
    CHECK(total_hits >= 16,
          "Total cache hits >= 16 (got %d) — 8 from LZ4 batch 2 + 8 from ZSTD batch 2",
          total_hits);

    remove(TMP_FILE);
}

/* ============================================================
 * TEST D: Data correctness after caching
 *
 * Verify that cached manager produces bit-exact compression.
 * 16 chunks × LZ4, read back and compare against formula.
 * ============================================================ */
static void test_D(float *d_data, float *d_read, float *h_read)
{
    printf("\n── D: Bit-exact correctness after cache reuse ────────────\n");

    const int N = 16;
    const size_t n_floats = (size_t)N * CHUNK_FLOATS;

    ramp_kernel<<<512, 256>>>(d_data, n_floats, n_floats);
    cudaDeviceSynchronize();

    gpucompress_reset_cache_stats();
    int rc = vol_write(d_data, n_floats, CHUNK_FLOATS, 1 /* LZ4 */);
    CHECK(rc == 0, "Write succeeded");

    cudaMemset(d_read, 0, n_floats * sizeof(float));
    int rrc = vol_read(d_read, n_floats);
    CHECK(rrc == 0, "Read succeeded");

    cudaMemcpy(h_read, d_read, n_floats * sizeof(float), cudaMemcpyDeviceToHost);

    size_t errs = 0;
    for (size_t i = 0; i < n_floats; i++) {
        if (h_read[i] != expected(i, n_floats)) {
            if (errs == 0)
                printf("    first mismatch @ %zu: got %.8g exp %.8g\n",
                       i, (double)h_read[i], (double)expected(i, n_floats));
            errs++;
        }
    }
    CHECK(errs == 0, "Bit-exact round-trip (%zu errors)", errs);

    int hits = 0, misses = 0;
    gpucompress_get_cache_stats(&hits, &misses);
    printf("    Cache stats (write only): hits=%d  misses=%d\n", hits, misses);

    remove(TMP_FILE);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  LRU-1 nvcomp Manager Cache Test\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  Chunk: %d MiB  |  Workers: 8  |  Path: HDF5 VOL\n", CHUNK_MB);
    printf("══════════════════════════════════════════════════════════\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Init without NN (explicit algos only) */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: VOL register failed\n");
        gpucompress_cleanup(); return 1;
    }

    /* Allocate max-sized buffers (32 chunks for test B) */
    const size_t max_floats = 32 * CHUNK_FLOATS;
    const size_t max_bytes  = max_floats * sizeof(float);

    float *d_data = NULL, *d_read = NULL, *h_read = NULL;
    if (cudaMalloc(&d_data, max_bytes) != cudaSuccess ||
        cudaMalloc(&d_read, max_bytes) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed\n");
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }
    h_read = (float *)malloc(max_bytes);

    /* Warmup: one LZ4 write to prime nvcomp */
    {
        size_t wf = CHUNK_FLOATS;
        ramp_kernel<<<512, 256>>>(d_data, wf, wf);
        cudaDeviceSynchronize();
        vol_write(d_data, wf, wf, 1);
        remove(TMP_FILE);
    }

    /* Run tests */
    test_A(d_data);
    test_B(d_data);
    test_C(d_data);
    test_D(d_data, d_read, h_read);

    /* Summary */
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  %d / %d checks passed", g_pass, g_pass + g_fail);
    if (g_fail) printf("  (%d FAILED)", g_fail);
    printf("\n");

    if (g_fail == 0)
        printf("  LRU-1 manager caching: OPERATIONAL\n");
    else
        printf("  Cache hits below threshold — caching not implemented or broken\n");

    printf("══════════════════════════════════════════════════════════\n\n");

    free(h_read);
    cudaFree(d_data);
    cudaFree(d_read);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return (g_fail == 0) ? 0 : 1;
}
