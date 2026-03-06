/**
 * tests/test_correctness_vol.cu
 *
 * Correctness test: GPU-resident VOL compression round-trip
 *
 * Validates that compressing and decompressing 64 MB (16 × 4 MB chunks)
 * via the VOL connector produces bit-exact output for every combination of:
 *   - 8 compression algorithms (LZ4, SNAPPY, DEFLATE, GDEFLATE, ZSTD,
 *                                ANS, CASCADED, BITCOMP)
 *   - 2 shuffle modes (no shuffle, 4-byte float shuffle)
 *   - 2 data patterns (ramp: low entropy / xorshift: high entropy)
 *
 * Total: 8 × 2 × 2 = 32 test cases.
 *
 * Concurrency note:
 *   16 chunks with N_COMP_WORKERS=8 → each write exercises 2 full rounds
 *   of concurrent GPU compression (8 workers simultaneously holding
 *   CompContext slots, each on its own CUDA stream with independent
 *   stats/NN/SGD buffers). No NN is loaded; the algorithm is fixed by
 *   the DCPL filter parameter.
 *
 * Verification:
 *   Both patterns are deterministic from a formula or fixed seed, so we
 *   regenerate expected values on-the-fly and never store a 64 MB host copy.
 *   After each read-back (GPU ptr → D→H), we compare every float element.
 *
 * No quantization (lossless only, error_bound = 0.0).
 * No NN weights required.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/test_correctness_vol
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
 * Test parameters
 * ============================================================ */
#define DATASET_MB   64
#define CHUNK_MB      4
#define TMP_FILE     "/tmp/test_correctness_vol.h5"

/* ============================================================
 * Filter wiring (mirrors H5Zgpucompress.h inline)
 * ============================================================ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static herr_t dcpl_set_gpucompress(hid_t dcpl, unsigned int algo,
                                    unsigned int preproc, unsigned int shuf_sz)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = algo;
    cd[1] = preproc;
    cd[2] = shuf_sz;
    /* cd[3], cd[4] = error_bound double packed as two uint32 — 0.0 = all zeros */
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
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
 * GPU data generation kernels
 *
 * Pattern RAMP:   buf[i] = (float)i / (float)n
 * Pattern XSHIFT: buf[i] = xorshift32(seed ^ i) mapped to [-1, 1]
 *                 (covers all byte patterns; very low compressibility)
 * ============================================================ */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = (float)i / (float)n;
}

__global__ static void xorshift_kernel(float *buf, size_t n, uint32_t seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s) {
        /* deterministic per-element xorshift32 with index mixing */
        uint32_t x = seed ^ (uint32_t)(i * 2654435761UL + 1);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        /* reinterpret as float: avoid NaN/Inf by masking exponent bits */
        x = (x & 0x007FFFFFu) | 0x3F800000u; /* exponent = 127 → range [1,2) */
        float f;
        memcpy(&f, &x, sizeof(f));
        buf[i] = f - 1.5f; /* shift to [-0.5, 0.5) */
    }
}

/* ============================================================
 * Expected-value generators (CPU, for verification)
 * Must mirror the GPU kernels exactly.
 * ============================================================ */
static float expected_ramp(size_t i, size_t n)
{
    return (float)i / (float)n;
}

static float expected_xorshift(size_t i, uint32_t seed)
{
    uint32_t x = seed ^ (uint32_t)(i * 2654435761UL + 1);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    x = (x & 0x007FFFFFu) | 0x3F800000u;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f - 1.5f;
}

/* ============================================================
 * Test case descriptor
 * ============================================================ */
typedef enum { PAT_RAMP = 0, PAT_XORSHIFT = 1 } Pattern;

typedef struct {
    unsigned int    algo_id;    /* GPUCOMPRESS_ALGO_* (1-8) */
    const char     *algo_name;
    unsigned int    preproc;    /* 0 or GPUCOMPRESS_PREPROC_SHUFFLE_4 */
    unsigned int    shuf_sz;    /* 0 or 4 */
    const char     *shuffle_label;
} TestConfig;

static const TestConfig CONFIGS[] = {
    /* ---- no shuffle ---- */
    { 1, "LZ4",      0,                              0, "no-shuffle" },
    { 2, "SNAPPY",   0,                              0, "no-shuffle" },
    { 3, "DEFLATE",  0,                              0, "no-shuffle" },
    { 4, "GDEFLATE", 0,                              0, "no-shuffle" },
    { 5, "ZSTD",     0,                              0, "no-shuffle" },
    { 6, "ANS",      0,                              0, "no-shuffle" },
    { 7, "CASCADED", 0,                              0, "no-shuffle" },
    { 8, "BITCOMP",  0,                              0, "no-shuffle" },
    /* ---- byte shuffle (float32, 4 bytes) ---- */
    { 1, "LZ4",      GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 2, "SNAPPY",   GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 3, "DEFLATE",  GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 4, "GDEFLATE", GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 5, "ZSTD",     GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 6, "ANS",      GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 7, "CASCADED", GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
    { 8, "BITCOMP",  GPUCOMPRESS_PREPROC_SHUFFLE_4,  4, "shuffle-4"  },
};
#define N_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static const char *PATTERN_NAMES[] = { "ramp", "xorshift" };
#define N_PATTERNS 2

/* ============================================================
 * Run one round-trip test
 *
 * Returns 0 on PASS, -1 on FAIL.
 * Prints a one-line result.
 * ============================================================ */
static int run_one_test(
    const TestConfig *cfg, Pattern pat, uint32_t xor_seed,
    size_t n_floats, size_t chunk_floats,
    float *d_data,      /* pre-allocated GPU write buffer */
    float *d_read,      /* pre-allocated GPU read  buffer */
    float *h_readback   /* pre-allocated host verify buffer */
) {
    size_t total_bytes = n_floats * sizeof(float);

    /* 1. Fill GPU buffer with chosen pattern */
    if (pat == PAT_RAMP) {
        ramp_kernel<<<65535, 256>>>(d_data, n_floats);
    } else {
        xorshift_kernel<<<65535, 256>>>(d_data, n_floats, xor_seed);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("  [FAIL] %s / %s / %s — GPU fill failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    /* 2. Create dataset and write (GPU pointer → VOL) */
    remove(TMP_FILE);

    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    hid_t fapl  = make_vol_fapl();
    hid_t file  = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        printf("  [FAIL] %s / %s / %s — H5Fcreate failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    dcpl_set_gpucompress(dcpl, cfg->algo_id, cfg->preproc, cfg->shuf_sz);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fspace);
    if (dset < 0) {
        H5Fclose(file); remove(TMP_FILE);
        printf("  [FAIL] %s / %s / %s — H5Dcreate failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_data);   /* GPU pointer */
    H5Dclose(dset);
    H5Fclose(file);
    if (wret < 0) {
        remove(TMP_FILE);
        printf("  [FAIL] %s / %s / %s — H5Dwrite failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    /* 3. Read back (GPU pointer → VOL decompresses into d_read) */
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        remove(TMP_FILE);
        printf("  [FAIL] %s / %s / %s — H5Fopen failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, d_read);    /* GPU pointer */
    cudaDeviceSynchronize();  /* ensure GPU decompression kernels finish */
    H5Dclose(dset);
    H5Fclose(file);
    remove(TMP_FILE);

    if (rret < 0) {
        printf("  [FAIL] %s / %s / %s — H5Dread failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    /* 4. D→H the decompressed result */
    if (cudaMemcpy(h_readback, d_read, total_bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) {
        printf("  [FAIL] %s / %s / %s — D→H readback failed\n",
               cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
        return -1;
    }

    /* 5. Bit-exact comparison against expected values (no host copy of original) */
    for (size_t i = 0; i < n_floats; i++) {
        float expected = (pat == PAT_RAMP)
                       ? expected_ramp(i, n_floats)
                       : expected_xorshift(i, xor_seed);

        if (h_readback[i] != expected) {
            /* Print first mismatch then abort */
            printf("  [FAIL] %s / %s / %s — mismatch at element %zu: "
                   "got %.8g expected %.8g\n",
                   cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat],
                   i, (double)h_readback[i], (double)expected);
            return -1;
        }
    }

    printf("  [PASS] %-9s / %-11s / %s\n",
           cfg->algo_name, cfg->shuffle_label, PATTERN_NAMES[pat]);
    return 0;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    const size_t chunk_mb    = CHUNK_MB;
    const size_t dataset_mb  = DATASET_MB;
    const size_t n_floats    = dataset_mb  * 1024 * 1024 / sizeof(float);
    const size_t chunk_floats= chunk_mb    * 1024 * 1024 / sizeof(float);
    const size_t total_bytes = n_floats * sizeof(float);
    const size_t n_chunks    = n_floats / chunk_floats;
    const uint32_t XOR_SEED  = 0xDEADBEEFu;

    printf("=== VOL Correctness Test: %zu MB, %zu MB chunks, %zu chunks ===\n",
           dataset_mb, chunk_mb, n_chunks);
    printf("    Algorithms: 8  |  Shuffle modes: 2  |  Patterns: 2\n");
    printf("    Total test cases: %d\n", N_CONFIGS * N_PATTERNS);
    printf("    Concurrency: N_COMP_WORKERS=8 (2 full rounds of 8-way parallel compression)\n");
    printf("    No NN loaded; algorithm fixed via DCPL filter parameter.\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Init library (no NN weights needed for explicit algorithm) */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    /* Register VOL connector once */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup(); return 1;
    }

    /* Allocate GPU buffers (write + read) and one host verify buffer */
    float *d_data   = NULL;
    float *d_read   = NULL;
    float *h_readback = NULL;

    if (cudaMalloc(&d_data,    total_bytes) != cudaSuccess ||
        cudaMalloc(&d_read,    total_bytes) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed (need 2 × %zu MiB GPU)\n",
                total_bytes >> 20);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }
    h_readback = (float *)malloc(total_bytes);
    if (!h_readback) {
        fprintf(stderr, "FATAL: malloc %zu MiB host buffer failed\n",
                total_bytes >> 20);
        cudaFree(d_data); cudaFree(d_read);
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    printf("  %-9s   %-11s   %-8s   %s\n",
           "Algorithm", "Shuffle", "Pattern", "Result");
    printf("  ---------   -----------   --------   ------\n");

    /* --- Warmup: one untimed round-trip with LZ4/ramp to prime nvcomp --- */
    {
        ramp_kernel<<<65535, 256>>>(d_data, n_floats);
        cudaDeviceSynchronize();
        remove(TMP_FILE);
        hsize_t dims[1]  = { (hsize_t)n_floats };
        hsize_t cdims[1] = { (hsize_t)chunk_floats };
        hid_t fapl = make_vol_fapl();
        hid_t f = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (f >= 0) {
            hid_t fs   = H5Screate_simple(1, dims, NULL);
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, cdims);
            dcpl_set_gpucompress(dcpl, 1 /*LZ4*/, 0, 0);
            hid_t ds = H5Dcreate2(f, "data", H5T_NATIVE_FLOAT,
                                   fs, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Pclose(dcpl); H5Sclose(fs);
            if (ds >= 0) {
                H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_data);
                H5Dclose(ds);
            }
            H5Fclose(f);
        }
        remove(TMP_FILE);
    }

    /* --- Main test loop --- */
    int pass = 0, fail = 0;
    int results[N_CONFIGS][N_PATTERNS]; /* 0=pass, -1=fail */

    for (int c = 0; c < N_CONFIGS; c++) {
        for (int p = 0; p < N_PATTERNS; p++) {
            int r = run_one_test(
                &CONFIGS[c], (Pattern)p, XOR_SEED,
                n_floats, chunk_floats,
                d_data, d_read, h_readback
            );
            results[c][p] = r;
            if (r == 0) pass++; else fail++;
        }
    }

    /* --- Summary --- */
    printf("\n=== Summary: %d MiB dataset, %d MiB chunks ===\n\n",
           (int)dataset_mb, (int)chunk_mb);
    printf("  %-9s | %-11s | %-8s | %-8s | %s\n",
           "Algorithm", "Shuffle", "ramp", "xorshift", "Status");
    printf("  ----------+-----------+---------+---------+-------\n");

    for (int c = 0; c < N_CONFIGS; c++) {
        const char *r0 = (results[c][0] == 0) ? "PASS" : "FAIL";
        const char *r1 = (results[c][1] == 0) ? "PASS" : "FAIL";
        const char *status = (results[c][0] == 0 && results[c][1] == 0)
                           ? "OK" : "FAIL";
        printf("  %-9s | %-11s | %-8s | %-8s | %s\n",
               CONFIGS[c].algo_name, CONFIGS[c].shuffle_label,
               r0, r1, status);
    }

    printf("\n  Total: %d/%d PASS\n", pass, N_CONFIGS * N_PATTERNS);

    if (fail == 0)
        printf("\n  ALL TESTS PASSED — concurrent round-trip correctness verified.\n\n");
    else
        printf("\n  %d TEST(S) FAILED — data corruption detected.\n\n", fail);

    free(h_readback);
    cudaFree(d_data);
    cudaFree(d_read);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return (fail == 0) ? 0 : 1;
}
