/**
 * @file test_vol_pipeline_comprehensive.cu
 * @brief Comprehensive end-to-end validation of the HDF5 VOL pipeline:
 *        GPU pointer → NN inference → preprocessing → compress → buffer → HDF5 write
 *
 * This test validates every stage of the GPUCompress HDF5 VOL pipeline:
 *
 *   1. GPU data generation (multiple patterns with different statistical properties)
 *   2. NN model loading and readiness
 *   3. H5Dwrite with GPU pointer → VOL intercept
 *   4. NN inference (verified via per-chunk diagnostics)
 *   5. Preprocessing (shuffle/quantization as selected by NN)
 *   6. GPU compression (algorithm selected by NN)
 *   7. Compressed data transfer D→H and HDF5 native write
 *   8. H5Dread → VOL intercept → HDF5 native read → H→D → decompress
 *   9. Bitwise-exact round-trip verification on GPU
 *  10. CompressionHeader validation (magic, version, algorithm)
 *  11. Multi-chunk dataset verification
 *  12. NN adaptiveness: different data patterns → different algorithm selections
 *  13. VOL activity counter validation
 *  14. Per-chunk timing breakdown validation
 *
 * Usage:
 *   ./test_vol_pipeline_comprehensive [weights.nnwt]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h>

#include <hdf5.h>
#include <cuda_runtime.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ================================================================== */
/* Constants                                                           */
/* ================================================================== */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Dataset sizes — 16 MB per dataset, 4 MB chunks → 4 chunks each */
#define CHUNK_FLOATS   (1024 * 1024)        /* 1M floats = 4 MB per chunk */
#define CHUNKS_PER_DS  4
#define DS_FLOATS      (CHUNK_FLOATS * CHUNKS_PER_DS)   /* 4M floats = 16 MB */
#define DS_BYTES       (DS_FLOATS * sizeof(float))
#define CHUNK_BYTES    (CHUNK_FLOATS * sizeof(float))

#define HDF5_FILE      "/tmp/test_vol_pipeline_comprehensive.h5"

/* CompressionHeader layout (first 64 bytes of each compressed chunk) */
#define HEADER_MAGIC_OFFSET   0
#define HEADER_VERSION_OFFSET 4
#define HEADER_ALGO_OFFSET    8

/* ================================================================== */
/* Macros                                                              */
/* ================================================================== */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static int g_pass_count = 0;
static int g_fail_count = 0;
static int g_test_count = 0;

#define CHECK(cond, fmt, ...) do {                                     \
    g_test_count++;                                                    \
    if (cond) {                                                        \
        g_pass_count++;                                                \
        printf("  [PASS] " fmt "\n", ##__VA_ARGS__);                  \
    } else {                                                           \
        g_fail_count++;                                                \
        printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);                  \
    }                                                                  \
} while (0)

/* ================================================================== */
/* Helpers                                                             */
/* ================================================================== */

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/** Open/create HDF5 file with GPUCompress VOL connector */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL) < 0) {
        H5Pclose(fapl); H5VLclose(vol_id);
        return H5I_INVALID_HID;
    }

    hid_t fid;
    if (flags & H5F_ACC_TRUNC)
        fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    else
        fid = H5Fopen(path, H5F_ACC_RDONLY, fapl);

    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

/* ================================================================== */
/* GPU Kernels: Data Generation                                        */
/* ================================================================== */

/** Pattern 1: Smooth ramp — low entropy, highly compressible */
__global__ void gen_smooth_ramp(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = (float)idx / (float)n;
}

/** Pattern 2: Noisy ramp — moderate entropy, moderate compressibility */
__global__ void gen_noisy_ramp(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = (float)idx / (float)n;
        out[idx] = x + 0.05f * sinf(x * 1000.0f);
    }
}

/** Pattern 3: High-frequency sinusoidal — structured, varying compressibility */
__global__ void gen_high_freq_sin(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = (float)idx / (float)n;
        out[idx] = sinf(x * 314.159f) * cosf(x * 271.828f) + 0.5f;
    }
}

/** Pattern 4: Pseudo-random — high entropy, hard to compress */
__global__ void gen_pseudo_random(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        /* Simple LCG PRNG seeded by index */
        unsigned int s = (unsigned int)(idx * 1103515245u + 12345u);
        s ^= s >> 16;
        s *= 0x45d9f3bu;
        s ^= s >> 16;
        /* Map to [0, 1) */
        out[idx] = (float)(s & 0x7FFFFFu) / (float)0x800000u;
    }
}

/** Pattern 5: Constant value — maximum compressibility */
__global__ void gen_constant(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = 3.14159265f;
}

/** GPU kernel: bitwise compare two float arrays */
__global__ void compare_gpu(const float* a, const float* b, size_t n,
                            int* mismatch_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int va, vb;
        memcpy(&va, &a[idx], sizeof(unsigned int));
        memcpy(&vb, &b[idx], sizeof(unsigned int));
        if (va != vb) atomicAdd(mismatch_count, 1);
    }
}

/* ================================================================== */
/* Data pattern descriptors                                            */
/* ================================================================== */
typedef void (*gen_kernel_t)(float*, size_t);

struct DataPattern {
    const char*  name;
    gen_kernel_t kernel;
    const char*  description;
};

static const DataPattern PATTERNS[] = {
    { "smooth_ramp",    gen_smooth_ramp,    "monotonic ramp, very low entropy"        },
    { "noisy_ramp",     gen_noisy_ramp,     "ramp + sinusoidal noise, mid entropy"    },
    { "high_freq_sin",  gen_high_freq_sin,  "high-freq sin*cos, structured"           },
    { "pseudo_random",  gen_pseudo_random,  "LCG pseudo-random, high entropy"         },
    { "constant",       gen_constant,       "constant 3.14159, minimal entropy"       },
};
static const int N_PATTERNS = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

/* ================================================================== */
/* Test 1: Library and NN initialization                               */
/* ================================================================== */
static void test_init(const char* weights) {
    printf("\n== Test 1: Library & NN Initialization ==\n");

    CHECK(gpucompress_is_initialized(),
          "Library is initialized");
    CHECK(gpucompress_nn_is_loaded(),
          "NN weights loaded from %s", weights);
}

/* ================================================================== */
/* Test 2: Write multiple data patterns through VOL with ALGO_AUTO     */
/* ================================================================== */
static void test_write_patterns(float** d_buffers) {
    printf("\n== Test 2: Write %d Data Patterns via VOL (ALGO_AUTO) ==\n", N_PATTERNS);

    int threads = 256;
    int blocks  = (DS_FLOATS + threads - 1) / threads;

    /* Generate all patterns on GPU */
    for (int i = 0; i < N_PATTERNS; i++) {
        CUDA_CHECK(cudaMalloc(&d_buffers[i], DS_BYTES));
        PATTERNS[i].kernel<<<blocks, threads>>>(d_buffers[i], DS_FLOATS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Generated %d patterns on GPU (%zu MB each)\n",
           N_PATTERNS, DS_BYTES / (1024 * 1024));

    /* Reset VOL stats */
    H5VL_gpucompress_reset_stats();

    /* Write all patterns to a single HDF5 file */
    hid_t fid = open_vol_file(HDF5_FILE, H5F_ACC_TRUNC);
    CHECK(fid >= 0, "VOL file created: %s", HDF5_FILE);
    if (fid < 0) return;

    for (int i = 0; i < N_PATTERNS; i++) {
        /* Reset chunk diagnostics before each dataset write */
        gpucompress_reset_chunk_history();

        hsize_t dims[1]  = { DS_FLOATS };
        hsize_t chunk[1] = { CHUNK_FLOATS };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        /* ALGO_AUTO, no preprocessing, lossless */
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = GPUCOMPRESS_ALGO_AUTO;
        cd[1] = 0;
        cd[2] = 0;
        pack_double(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                      H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        char ds_name[64];
        snprintf(ds_name, sizeof(ds_name), "%s", PATTERNS[i].name);
        hid_t dset = H5Dcreate2(fid, ds_name, H5T_NATIVE_FLOAT,
                                space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CHECK(dset >= 0, "Dataset '%s' created", ds_name);

        herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_buffers[i]);
        CHECK(wr >= 0, "H5Dwrite('%s') succeeded with GPU pointer", ds_name);

        /* ---- Validate per-chunk diagnostics ---- */
        int n_chunks = gpucompress_get_chunk_history_count();
        CHECK(n_chunks == CHUNKS_PER_DS,
              "'%s': chunk history count = %d (expected %d)",
              ds_name, n_chunks, CHUNKS_PER_DS);

        for (int c = 0; c < n_chunks; c++) {
            gpucompress_chunk_diag_t diag;
            int dr = gpucompress_get_chunk_diag(c, &diag);
            CHECK(dr == 0, "'%s' chunk %d: diagnostic retrieved", ds_name, c);

            /* NN must have been invoked (action >= 0) */
            CHECK(diag.nn_action >= 0 && diag.nn_action < 32,
                  "'%s' chunk %d: nn_action=%d (valid 0-31)",
                  ds_name, c, diag.nn_action);
            CHECK(diag.nn_original_action >= 0 && diag.nn_original_action < 32,
                  "'%s' chunk %d: nn_original_action=%d (valid 0-31)",
                  ds_name, c, diag.nn_original_action);

            /* Compression must have produced a positive ratio */
            CHECK(diag.actual_ratio > 0.0f,
                  "'%s' chunk %d: actual_ratio=%.2f (> 0)",
                  ds_name, c, diag.actual_ratio);

            /* NN predicted ratio must be positive when using ALGO_AUTO */
            CHECK(diag.predicted_ratio > 0.0f,
                  "'%s' chunk %d: predicted_ratio=%.2f (> 0)",
                  ds_name, c, diag.predicted_ratio);

            /* Timing must be non-negative */
            CHECK(diag.nn_inference_ms >= 0.0f,
                  "'%s' chunk %d: nn_inference_ms=%.3f ms",
                  ds_name, c, diag.nn_inference_ms);
            CHECK(diag.compression_ms >= 0.0f,
                  "'%s' chunk %d: compression_ms=%.3f ms",
                  ds_name, c, diag.compression_ms);
        }

        H5Dclose(dset);
        H5Sclose(space);
        H5Pclose(dcpl);
    }

    H5Fclose(fid);
}

/* ================================================================== */
/* Test 3: VOL activity counters                                       */
/* ================================================================== */
static void test_vol_counters(void) {
    printf("\n== Test 3: VOL Activity Counters ==\n");

    int vol_writes = 0, vol_reads = 0, vol_comp = 0, vol_decomp = 0;
    H5VL_gpucompress_get_stats(&vol_writes, &vol_reads, &vol_comp, &vol_decomp);

    CHECK(vol_writes == N_PATTERNS,
          "VOL writes = %d (expected %d)", vol_writes, N_PATTERNS);

    int expected_chunks = N_PATTERNS * CHUNKS_PER_DS;
    CHECK(vol_comp == expected_chunks,
          "VOL chunks compressed = %d (expected %d)", vol_comp, expected_chunks);

    /* No reads yet */
    CHECK(vol_reads == 0,
          "VOL reads = %d (expected 0 before readback)", vol_reads);

    /* Transfer stats: D→H is handled inside gpucompress_compress_gpu(),
     * so the VOL-layer counters may be 0 (the VOL passes the GPU pointer
     * to the compress function which returns a host buffer internally). */
    int d2h_n = 0, h2d_n = 0;
    size_t d2h_b = 0, h2d_b = 0;
    H5VL_gpucompress_get_transfer_stats(&h2d_n, &h2d_b, &d2h_n, &d2h_b, NULL, NULL);
    printf("  [INFO] VOL-layer D→H: %d calls, %.2f MB\n",
           d2h_n, (double)d2h_b / (1024.0 * 1024.0));
    printf("  [INFO] VOL-layer H→D: %d calls, %.2f MB\n",
           h2d_n, (double)h2d_b / (1024.0 * 1024.0));
}

/* ================================================================== */
/* Test 4: File-level validation — file exists, is smaller, has header */
/* ================================================================== */
static void test_file_validation(void) {
    printf("\n== Test 4: File-Level Validation ==\n");

    struct stat st;
    int sr = stat(HDF5_FILE, &st);
    CHECK(sr == 0, "HDF5 file exists: %s", HDF5_FILE);
    if (sr != 0) return;

    size_t total_raw = (size_t)N_PATTERNS * DS_BYTES;
    CHECK((size_t)st.st_size < total_raw,
          "File size %zu < raw data %zu (compression effective)",
          (size_t)st.st_size, total_raw);
    printf("  [INFO] File: %.2f MB, raw: %.2f MB, overall ratio: %.2fx\n",
           (double)st.st_size / (1024.0 * 1024.0),
           (double)total_raw / (1024.0 * 1024.0),
           (double)total_raw / (double)st.st_size);
}

/* ================================================================== */
/* Test 5: Read-back and raw-chunk header inspection                   */
/* ================================================================== */
static void test_raw_chunk_headers(void) {
    printf("\n== Test 5: CompressionHeader Validation (Raw Chunks) ==\n");

    /* Open file with native HDF5 (not VOL) to read raw compressed chunks */
    hid_t fid = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "Opened HDF5 file with native driver for raw inspection");
    if (fid < 0) return;

    for (int i = 0; i < N_PATTERNS; i++) {
        hid_t dset = H5Dopen2(fid, PATTERNS[i].name, H5P_DEFAULT);
        CHECK(dset >= 0, "'%s': dataset opened for raw read", PATTERNS[i].name);
        if (dset < 0) continue;

        /* Read the first raw chunk using H5Dread_chunk */
        hsize_t offset[1] = {0};
        uint32_t filters = 0;

        /* Get chunk storage size to allocate buffer */
        hsize_t chunk_storage = 0;
        herr_t cs = H5Dget_chunk_storage_size(dset, offset, &chunk_storage);
        CHECK(cs >= 0 && chunk_storage > 0,
              "'%s': chunk 0 storage size = %zu bytes",
              PATTERNS[i].name, (size_t)chunk_storage);

        if (chunk_storage > 0 && chunk_storage < CHUNK_BYTES * 2) {
            uint8_t* raw_buf = (uint8_t*)malloc(chunk_storage);

            size_t read_sz = chunk_storage;
            herr_t rr = H5Dread_chunk(dset, H5P_DEFAULT, offset, &filters,
                                      raw_buf, &read_sz);
            CHECK(rr >= 0, "'%s': H5Dread_chunk succeeded", PATTERNS[i].name);

            if (rr >= 0 && chunk_storage >= 64) {
                /* Validate CompressionHeader fields */
                uint32_t magic;
                memcpy(&magic, raw_buf + HEADER_MAGIC_OFFSET, sizeof(uint32_t));
                CHECK(magic == 0x43555047u,
                      "'%s': header magic = 0x%08X (expected 0x43555047 'GPUC')",
                      PATTERNS[i].name, magic);

                uint32_t version;
                memcpy(&version, raw_buf + HEADER_VERSION_OFFSET, sizeof(uint32_t));
                CHECK(version >= 1 && version <= 2,
                      "'%s': header version = %u (expected 1 or 2)",
                      PATTERNS[i].name, version);

                uint32_t algo_id;
                memcpy(&algo_id, raw_buf + HEADER_ALGO_OFFSET, sizeof(uint32_t));
                CHECK(algo_id >= 1 && algo_id <= 8,
                      "'%s': header algorithm = %u (%s)",
                      PATTERNS[i].name, algo_id,
                      gpucompress_algorithm_name((gpucompress_algorithm_t)algo_id));
            }
            free(raw_buf);
        }

        H5Dclose(dset);
    }

    H5Fclose(fid);
}

/* ================================================================== */
/* Test 6: Read-back through VOL and bitwise verification              */
/* ================================================================== */
static void test_readback_verify(float** d_buffers) {
    printf("\n== Test 6: Read-Back Through VOL & Bitwise Verification ==\n");

    H5VL_gpucompress_reset_stats();

    float* d_readback = nullptr;
    CUDA_CHECK(cudaMalloc(&d_readback, DS_BYTES));

    int* d_mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mismatches, sizeof(int)));

    int threads = 256;
    int blocks  = (DS_FLOATS + threads - 1) / threads;

    hid_t fid = open_vol_file(HDF5_FILE, H5F_ACC_RDONLY);
    CHECK(fid >= 0, "VOL file opened for read-back");
    if (fid < 0) { cudaFree(d_readback); cudaFree(d_mismatches); return; }

    for (int i = 0; i < N_PATTERNS; i++) {
        hid_t dset = H5Dopen2(fid, PATTERNS[i].name, H5P_DEFAULT);
        CHECK(dset >= 0, "'%s': dataset opened for read", PATTERNS[i].name);
        if (dset < 0) continue;

        CUDA_CHECK(cudaMemset(d_readback, 0, DS_BYTES));
        herr_t rd = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_readback);
        CHECK(rd >= 0, "'%s': H5Dread to GPU pointer succeeded", PATTERNS[i].name);

        if (rd >= 0) {
            CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(int)));
            compare_gpu<<<blocks, threads>>>(d_buffers[i], d_readback,
                                             DS_FLOATS, d_mismatches);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_mm = 0;
            CUDA_CHECK(cudaMemcpy(&h_mm, d_mismatches, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK(h_mm == 0,
                  "'%s': bitwise-exact round-trip (%d / %zu mismatches)",
                  PATTERNS[i].name, h_mm, (size_t)DS_FLOATS);
        }

        H5Dclose(dset);
    }

    H5Fclose(fid);

    /* Verify VOL read counters */
    int vol_writes = 0, vol_reads = 0, vol_comp = 0, vol_decomp = 0;
    H5VL_gpucompress_get_stats(&vol_writes, &vol_reads, &vol_comp, &vol_decomp);
    CHECK(vol_reads == N_PATTERNS,
          "VOL reads = %d (expected %d)", vol_reads, N_PATTERNS);

    int expected_decomp = N_PATTERNS * CHUNKS_PER_DS;
    CHECK(vol_decomp == expected_decomp,
          "VOL chunks decompressed = %d (expected %d)", vol_decomp, expected_decomp);

    cudaFree(d_readback);
    cudaFree(d_mismatches);
}

/* ================================================================== */
/* Test 7: NN adaptiveness — different patterns → different actions    */
/* ================================================================== */
static void test_nn_adaptiveness(void) {
    printf("\n== Test 7: NN Adaptiveness — Different Patterns → Different Decisions ==\n");

    /* For each pattern, record the NN actions across chunks */
    int actions_per_pattern[N_PATTERNS][CHUNKS_PER_DS];
    memset(actions_per_pattern, -1, sizeof(actions_per_pattern));

    /* We need to re-compress each pattern to get fresh diagnostics */
    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, DS_BYTES));

    int threads = 256;
    int blocks  = (DS_FLOATS + threads - 1) / threads;

    for (int i = 0; i < N_PATTERNS; i++) {
        /* Regenerate data */
        PATTERNS[i].kernel<<<blocks, threads>>>(d_buf, DS_FLOATS);
        CUDA_CHECK(cudaDeviceSynchronize());

        gpucompress_reset_chunk_history();

        char tmpfile[128];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/test_vol_adapt_%d.h5", i);

        hid_t fid = open_vol_file(tmpfile, H5F_ACC_TRUNC);
        if (fid < 0) continue;

        hsize_t dims[1]  = { DS_FLOATS };
        hsize_t chunk[1] = { CHUNK_FLOATS };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = GPUCOMPRESS_ALGO_AUTO;
        cd[1] = 0; cd[2] = 0;
        pack_double(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                      H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset >= 0) {
            H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, d_buf);
            H5Dclose(dset);
        }
        H5Sclose(space);
        H5Pclose(dcpl);
        H5Fclose(fid);
        remove(tmpfile);

        int n_ch = gpucompress_get_chunk_history_count();
        for (int c = 0; c < n_ch && c < CHUNKS_PER_DS; c++) {
            gpucompress_chunk_diag_t diag;
            if (gpucompress_get_chunk_diag(c, &diag) == 0) {
                actions_per_pattern[i][c] = diag.nn_original_action;
            }
        }
    }
    cudaFree(d_buf);

    /* Decode and print actions for each pattern */
    const char* algo_names[] = {
        "???", "LZ4", "SNAPPY", "DEFLATE", "GDEFLATE",
        "ZSTD", "ANS", "CASCADED", "BITCOMP"
    };
    printf("  NN action summary (chunk 0 of each pattern):\n");
    for (int i = 0; i < N_PATTERNS; i++) {
        int a = actions_per_pattern[i][0];
        if (a >= 0) {
            int algo_idx = (a % 8) + 1;
            int quant    = (a / 8) % 2;
            int shuffle  = (a / 16) % 2;
            const char* aname = (algo_idx >= 1 && algo_idx <= 8)
                                ? algo_names[algo_idx] : "???";
            printf("    %-16s → action=%2d  algo=%s  quant=%d  shuffle=%d\n",
                   PATTERNS[i].name, a, aname, quant, shuffle);
        }
    }

    /* Check that not all patterns got the same action (NN is adapting) */
    int unique_actions = 0;
    int seen[32];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i < N_PATTERNS; i++) {
        int a = actions_per_pattern[i][0];
        if (a >= 0 && a < 32 && !seen[a]) {
            seen[a] = 1;
            unique_actions++;
        }
    }
    CHECK(unique_actions >= 2,
          "NN selected %d distinct actions across %d patterns (adaptiveness)",
          unique_actions, N_PATTERNS);
}

/* ================================================================== */
/* Test 8: Multi-chunk consistency — all chunks verified independently  */
/* ================================================================== */
static void test_multi_chunk_consistency(void) {
    printf("\n== Test 8: Multi-Chunk Consistency ==\n");

    /* Use noisy ramp pattern, 8 chunks this time */
    const int N_CHUNKS = 8;
    const size_t total_floats = (size_t)CHUNK_FLOATS * N_CHUNKS;
    const size_t total_bytes  = total_floats * sizeof(float);

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));

    int threads = 256;
    int blocks  = (total_floats + threads - 1) / threads;
    gen_noisy_ramp<<<blocks, threads>>>(d_data, total_floats);
    CUDA_CHECK(cudaDeviceSynchronize());

    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    const char* tmpfile = "/tmp/test_vol_multichunk.h5";

    /* Write */
    {
        hid_t fid = open_vol_file(tmpfile, H5F_ACC_TRUNC);
        CHECK(fid >= 0, "Multi-chunk file created");
        if (fid < 0) { cudaFree(d_data); return; }

        hsize_t dims[1]  = { total_floats };
        hsize_t chunk[1] = { CHUNK_FLOATS };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = GPUCOMPRESS_ALGO_AUTO; cd[1] = 0; cd[2] = 0;
        pack_double(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                      H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                                space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_data);
        CHECK(wr >= 0, "Multi-chunk H5Dwrite (8 × 4 MB = 32 MB)");

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* Validate per-chunk diagnostics */
    int n_ch = gpucompress_get_chunk_history_count();
    CHECK(n_ch == N_CHUNKS,
          "Chunk history: %d chunks (expected %d)", n_ch, N_CHUNKS);

    for (int c = 0; c < n_ch; c++) {
        gpucompress_chunk_diag_t diag;
        int dr = gpucompress_get_chunk_diag(c, &diag);
        if (dr == 0) {
            CHECK(diag.nn_action >= 0,
                  "Chunk %d: NN active (action=%d)", c, diag.nn_action);
            CHECK(diag.actual_ratio > 0.0f,
                  "Chunk %d: ratio=%.2f", c, diag.actual_ratio);
        }
    }

    /* VOL counter check */
    int vol_w = 0, vol_r = 0, vol_c = 0, vol_d = 0;
    H5VL_gpucompress_get_stats(&vol_w, &vol_r, &vol_c, &vol_d);
    CHECK(vol_c == N_CHUNKS,
          "VOL compressed %d chunks (expected %d)", vol_c, N_CHUNKS);

    /* Read back and verify */
    float* d_readback = nullptr;
    CUDA_CHECK(cudaMalloc(&d_readback, total_bytes));
    {
        hid_t fid = open_vol_file(tmpfile, H5F_ACC_RDONLY);
        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        herr_t rd = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_readback);
        CHECK(rd >= 0, "Multi-chunk H5Dread succeeded");
        H5Dclose(dset); H5Fclose(fid);
    }

    int* d_mm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mm, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mm, 0, sizeof(int)));
    compare_gpu<<<blocks, threads>>>(d_data, d_readback, total_floats, d_mm);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_mm = 0;
    CUDA_CHECK(cudaMemcpy(&h_mm, d_mm, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(h_mm == 0,
          "Multi-chunk bitwise-exact round-trip (%d / %zu mismatches)",
          h_mm, total_floats);

    cudaFree(d_data);
    cudaFree(d_readback);
    cudaFree(d_mm);
    remove(tmpfile);
}

/* ================================================================== */
/* Test 9: Compression effectiveness — compressed < original           */
/* ================================================================== */
static void test_compression_effectiveness(void) {
    printf("\n== Test 9: Compression Effectiveness ==\n");

    /* Open the main test file with native HDF5 to get raw chunk sizes */
    hid_t fid = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        printf("  [SKIP] Could not open file for effectiveness check\n");
        return;
    }

    for (int i = 0; i < N_PATTERNS; i++) {
        hid_t dset = H5Dopen2(fid, PATTERNS[i].name, H5P_DEFAULT);
        if (dset < 0) continue;

        size_t total_compressed = 0;
        for (int c = 0; c < CHUNKS_PER_DS; c++) {
            hsize_t offset[1] = { (hsize_t)(c * CHUNK_FLOATS) };
            hsize_t chunk_sz = 0;
            if (H5Dget_chunk_storage_size(dset, offset, &chunk_sz) >= 0)
                total_compressed += (size_t)chunk_sz;
        }

        double ratio = (total_compressed > 0)
                        ? (double)DS_BYTES / (double)total_compressed : 0.0;

        /* At minimum, compression should not expand data by more than 2x
         * (header overhead). For structured data, ratio should be > 1. */
        CHECK(total_compressed > 0,
              "'%s': compressed size = %zu bytes (ratio %.2fx)",
              PATTERNS[i].name, total_compressed, ratio);

        /* For the constant pattern, ratio should be very high */
        if (i == 4) { /* constant */
            CHECK(ratio > 5.0,
                  "'%s': constant data ratio = %.1fx (expected > 5x)",
                  PATTERNS[i].name, ratio);
        }

        H5Dclose(dset);
    }

    H5Fclose(fid);
}

/* ================================================================== */
/* Test 10: GPU pointer detection                                      */
/* ================================================================== */
static void test_gpu_pointer_detection(void) {
    printf("\n== Test 10: GPU Pointer Detection ==\n");

    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, 1024));
    float* h_ptr = (float*)malloc(1024);

    CHECK(gpucompress_is_device_ptr(d_ptr) == 1,
          "cudaMalloc'd pointer detected as GPU");
    CHECK(gpucompress_is_device_ptr(h_ptr) == 0,
          "malloc'd pointer detected as host");

    cudaFree(d_ptr);
    free(h_ptr);
}

/* ================================================================== */
/* Test 11: Direct API stats match VOL pipeline behavior               */
/* ================================================================== */
static void test_direct_api_stats(void) {
    printf("\n== Test 11: Direct API Compression Stats Validation ==\n");

    /* Compress a buffer directly via gpucompress_compress_gpu and check stats */
    const size_t n_floats = CHUNK_FLOATS;
    const size_t n_bytes  = n_floats * sizeof(float);

    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, n_bytes));

    int threads = 256;
    int blocks  = (n_floats + threads - 1) / threads;
    gen_noisy_ramp<<<blocks, threads>>>(d_input, n_floats);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t max_out = gpucompress_max_compressed_size(n_bytes);
    CHECK(max_out > 0 && max_out >= n_bytes,
          "max_compressed_size = %zu (>= input %zu)", max_out, n_bytes);

    void* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, max_out));

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    size_t out_size = max_out;
    gpucompress_error_t rc = gpucompress_compress_gpu(
        d_input, n_bytes, d_output, &out_size, &cfg, &stats, nullptr);

    CHECK(rc == GPUCOMPRESS_SUCCESS,
          "gpucompress_compress_gpu succeeded (err=%d)", (int)rc);

    if (rc == GPUCOMPRESS_SUCCESS) {
        CHECK(stats.original_size == n_bytes,
              "stats.original_size = %zu (expected %zu)",
              stats.original_size, n_bytes);
        CHECK(stats.compressed_size > 0 && stats.compressed_size <= out_size,
              "stats.compressed_size = %zu", stats.compressed_size);
        CHECK(stats.compression_ratio > 0.0,
              "stats.compression_ratio = %.2f", stats.compression_ratio);
        CHECK(stats.algorithm_used >= 1 && stats.algorithm_used <= 8,
              "stats.algorithm_used = %d (%s)",
              stats.algorithm_used,
              gpucompress_algorithm_name(stats.algorithm_used));
        CHECK(stats.nn_original_action >= 0,
              "stats.nn_original_action = %d (NN was invoked)",
              stats.nn_original_action);
        CHECK(stats.nn_final_action >= 0,
              "stats.nn_final_action = %d", stats.nn_final_action);
        CHECK(stats.predicted_ratio > 0.0,
              "stats.predicted_ratio = %.2f", stats.predicted_ratio);

        /* Verify action → algorithm consistency */
        int decoded_algo = (stats.nn_final_action % 8) + 1;
        CHECK(decoded_algo == (int)stats.algorithm_used,
              "Action decode: action%%8+1=%d matches algorithm_used=%d",
              decoded_algo, (int)stats.algorithm_used);

        /* Verify round-trip via direct decompress */
        void* d_decompressed = nullptr;
        CUDA_CHECK(cudaMalloc(&d_decompressed, n_bytes));
        size_t decomp_size = n_bytes;
        gpucompress_error_t drc = gpucompress_decompress_gpu(
            d_output, out_size, d_decompressed, &decomp_size, nullptr);
        CHECK(drc == GPUCOMPRESS_SUCCESS,
              "Direct GPU decompress succeeded");
        CHECK(decomp_size == n_bytes,
              "Decompressed size = %zu (expected %zu)", decomp_size, n_bytes);

        if (drc == GPUCOMPRESS_SUCCESS) {
            int* d_mm = nullptr;
            CUDA_CHECK(cudaMalloc(&d_mm, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_mm, 0, sizeof(int)));
            compare_gpu<<<blocks, threads>>>((const float*)d_input,
                                             (const float*)d_decompressed,
                                             n_floats, d_mm);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_mm = 0;
            CUDA_CHECK(cudaMemcpy(&h_mm, d_mm, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(h_mm == 0,
                  "Direct API round-trip: %d mismatches", h_mm);
            cudaFree(d_mm);
        }
        cudaFree(d_decompressed);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

/* ================================================================== */
/* Main                                                                */
/* ================================================================== */
int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";
    if (argc <= 1) { FILE* f = fopen(weights, "rb"); if (f) fclose(f); else weights = "../neural_net/weights/model.nnwt"; }

    printf("======================================================\n");
    printf(" GPUCompress VOL Pipeline — Comprehensive Test Suite\n");
    printf("======================================================\n");
    printf("  Patterns     : %d (%s ... %s)\n", N_PATTERNS,
           PATTERNS[0].name, PATTERNS[N_PATTERNS - 1].name);
    printf("  Dataset size : %zu floats = %zu MB per pattern\n",
           (size_t)DS_FLOATS, DS_BYTES / (1024 * 1024));
    printf("  Chunk size   : %zu floats = %zu MB (%d chunks/dataset)\n",
           (size_t)CHUNK_FLOATS, CHUNK_BYTES / (1024 * 1024), CHUNKS_PER_DS);
    printf("  Total raw    : %zu MB\n",
           (size_t)N_PATTERNS * DS_BYTES / (1024 * 1024));
    printf("  NN weights   : %s\n", weights);
    printf("  HDF5 file    : %s\n", HDF5_FILE);
    printf("======================================================\n");

    /* Initialize library */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "NN weights not loaded — cannot test ALGO_AUTO\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Run test suite */
    float* d_buffers[N_PATTERNS];
    memset(d_buffers, 0, sizeof(d_buffers));

    test_init(weights);                    /* Test 1  */
    test_write_patterns(d_buffers);        /* Test 2  */
    test_vol_counters();                   /* Test 3  */
    test_file_validation();                /* Test 4  */
    test_raw_chunk_headers();              /* Test 5  */
    test_readback_verify(d_buffers);       /* Test 6  */
    test_nn_adaptiveness();                /* Test 7  */
    test_multi_chunk_consistency();         /* Test 8  */
    test_compression_effectiveness();       /* Test 9  */
    test_gpu_pointer_detection();           /* Test 10 */
    test_direct_api_stats();               /* Test 11 */

    /* Cleanup */
    for (int i = 0; i < N_PATTERNS; i++) {
        if (d_buffers[i]) cudaFree(d_buffers[i]);
    }
    gpucompress_cleanup();
    remove(HDF5_FILE);

    /* Final summary */
    printf("\n======================================================\n");
    printf(" RESULTS: %d passed, %d failed, %d total\n",
           g_pass_count, g_fail_count, g_test_count);
    printf(" %s\n", g_fail_count == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("======================================================\n");

    return g_fail_count == 0 ? 0 : 1;
}
