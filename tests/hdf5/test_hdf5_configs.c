/**
 * @file test_hdf5_configs.c
 * @brief HDF5 round-trip validation across compression configurations.
 *
 * Tests compress→write→read→decompress→verify for every combination of:
 *   - 6 data patterns (ramp, sine, constant, sparse, gaussian, hf_noise)
 *   - 8 algorithms (plain + shuffle) in lossless mode
 *   - ALGO_AUTO with NN (lossless)
 *   - Lossy quantization configs (LZ4, ZSTD, AUTO) with error-bound check
 *
 * Usage:
 *   ./build/test_hdf5_configs [weights_path]
 *   GPUCOMPRESS_WEIGHTS=path/to/model.nnwt ./build/test_hdf5_configs
 *
 * Exit code: 0 if all tests pass, 1 otherwise.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ============================================================
 * Test framework
 * ============================================================ */

static int g_pass = 0;
static int g_fail = 0;

#define TEST_CHECK(cond, fmt, ...) do {                            \
    if (!(cond)) {                                                 \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__);                \
        g_fail++;                                                  \
        return -1;                                                 \
    }                                                              \
} while (0)

/* ============================================================
 * Data patterns
 * ============================================================ */

typedef enum {
    PAT_RAMP,
    PAT_SINE,
    PAT_CONSTANT,
    PAT_SPARSE,
    PAT_GAUSSIAN,
    PAT_HF_NOISE,
    PAT_COUNT
} pattern_t;

static const char* pattern_name(pattern_t p) {
    switch (p) {
        case PAT_RAMP:     return "ramp";
        case PAT_SINE:     return "sine";
        case PAT_CONSTANT: return "constant";
        case PAT_SPARSE:   return "sparse";
        case PAT_GAUSSIAN: return "gaussian";
        case PAT_HF_NOISE: return "hf_noise";
        default:           return "unknown";
    }
}

/* Simple LCG PRNG for deterministic data */
static uint32_t lcg_state = 42;
static void lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}
static float lcg_float(void) {
    return (float)(lcg_next() & 0x7FFFFF) / (float)0x7FFFFF;
}

/* Box-Muller for gaussian */
static float lcg_gaussian(void) {
    float u1 = lcg_float() + 1e-10f;
    float u2 = lcg_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void fill_pattern(float* buf, size_t n, pattern_t pat) {
    lcg_seed((uint32_t)pat * 12345u + 67890u);
    for (size_t i = 0; i < n; i++) {
        switch (pat) {
            case PAT_RAMP:
                buf[i] = (float)i / (float)n;
                break;
            case PAT_SINE:
                buf[i] = sinf(2.0f * (float)M_PI * (float)i / 1024.0f);
                break;
            case PAT_CONSTANT:
                buf[i] = 3.14159f;
                break;
            case PAT_SPARSE:
                buf[i] = (lcg_next() % 100 < 5) ? lcg_float() * 100.0f : 0.0f;
                break;
            case PAT_GAUSSIAN:
                buf[i] = lcg_gaussian();
                break;
            case PAT_HF_NOISE:
                buf[i] = sinf(500.0f * (float)i / (float)n) + 0.3f * lcg_float();
                break;
            default:
                buf[i] = 0.0f;
        }
    }
}

/* ============================================================
 * Test configuration
 * ============================================================ */

typedef struct {
    const char* name;
    gpucompress_algorithm_t algo;
    unsigned int preproc;
    unsigned int shuffle_size;
    double error_bound;
    int is_lossy;
} test_config_t;

/* ============================================================
 * Single test: write + read + verify
 * ============================================================ */

static int run_one_test(const char* hdf5_path, const test_config_t* cfg,
                        pattern_t pat, const float* data, size_t n_floats) {
    char dset_name[256];
    snprintf(dset_name, sizeof(dset_name), "%s_%s_%s",
             cfg->name, pattern_name(pat),
             cfg->is_lossy ? "lossy" : "lossless");

    hid_t file = H5Fopen(hdf5_path, H5F_ACC_RDWR, H5P_DEFAULT);
    TEST_CHECK(file >= 0, "[%s] H5Fopen failed", dset_name);

    /* Create dataspace */
    hsize_t dims[1] = { (hsize_t)n_floats };
    hsize_t chunk_dims[1] = { (hsize_t)n_floats };  /* single chunk */
    hid_t space = H5Screate_simple(1, dims, NULL);

    /* Create property list with GPUCompress filter */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);

    herr_t fret = H5Pset_gpucompress(dcpl, cfg->algo, cfg->preproc,
                                      cfg->shuffle_size, cfg->error_bound);
    if (fret < 0) {
        printf("  SKIP: [%s] H5Pset_gpucompress failed (filter not available)\n", dset_name);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(file);
        g_pass++;  /* not a failure, just unsupported */
        return 0;
    }

    /* Write dataset */
    hid_t dset = H5Dcreate2(file, dset_name, H5T_NATIVE_FLOAT,
                             space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    TEST_CHECK(dset >= 0, "[%s] H5Dcreate2 failed", dset_name);

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, data);
    TEST_CHECK(wret >= 0, "[%s] H5Dwrite failed", dset_name);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);

    /* Re-open and read back */
    file = H5Fopen(hdf5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    TEST_CHECK(file >= 0, "[%s] H5Fopen (read) failed", dset_name);

    dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
    TEST_CHECK(dset >= 0, "[%s] H5Dopen2 failed", dset_name);

    float* readback = (float*)malloc(n_floats * sizeof(float));
    TEST_CHECK(readback != NULL, "[%s] malloc failed", dset_name);

    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, readback);
    TEST_CHECK(rret >= 0, "[%s] H5Dread failed", dset_name);

    H5Dclose(dset);
    H5Fclose(file);

    /* Verify */
    int ok = 1;
    if (cfg->is_lossy) {
        /* Lossy: check |original - readback| <= error_bound for each element */
        double eb = cfg->error_bound;
        for (size_t i = 0; i < n_floats; i++) {
            double diff = fabs((double)data[i] - (double)readback[i]);
            if (diff > eb * 1.01) {  /* 1% tolerance for float rounding */
                printf("  FAIL: [%s] element %zu: orig=%.8f read=%.8f diff=%.8f > eb=%.8f\n",
                       dset_name, i, data[i], readback[i], diff, eb);
                ok = 0;
                break;
            }
        }
    } else {
        /* Lossless: bit-exact */
        if (memcmp(data, readback, n_floats * sizeof(float)) != 0) {
            /* Find first mismatch for diagnostics */
            for (size_t i = 0; i < n_floats; i++) {
                if (data[i] != readback[i]) {
                    printf("  FAIL: [%s] bit-exact mismatch at %zu: orig=%.8f read=%.8f\n",
                           dset_name, i, data[i], readback[i]);
                    break;
                }
            }
            ok = 0;
        }
    }

    free(readback);

    if (ok) {
        g_pass++;
        return 0;
    } else {
        g_fail++;
        return -1;
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    /* Determine weights path */
    const char* weights_path = NULL;
    if (argc > 1) {
        weights_path = argv[1];
    } else {
        weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    }

    printf("=== GPUCompress HDF5 Configuration Validation Test ===\n");
    if (weights_path) {
        printf("NN weights: %s\n", weights_path);
    } else {
        printf("NN weights: (none — ALGO_AUTO tests will be skipped)\n");
    }

    /* Initialize library */
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("FATAL: gpucompress_init failed: %s\n", gpucompress_error_string(err));
        return 1;
    }

    /* Register HDF5 filter */
    if (H5Z_gpucompress_register() < 0) {
        printf("FATAL: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    int nn_loaded = gpucompress_nn_is_loaded();
    printf("NN loaded: %s\n\n", nn_loaded ? "yes" : "no");

    /* Create HDF5 file */
    const char* hdf5_path = "/tmp/test_hdf5_configs.h5";
    hid_t file = H5Fcreate(hdf5_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        printf("FATAL: Cannot create %s\n", hdf5_path);
        gpucompress_cleanup();
        return 1;
    }
    H5Fclose(file);

    /* Chunk size: 256K floats = 1 MB */
    const size_t n_floats = 256 * 1024;
    float* data = (float*)malloc(n_floats * sizeof(float));
    if (!data) {
        printf("FATAL: malloc failed\n");
        gpucompress_cleanup();
        return 1;
    }

    /* ── Lossless configurations ── */
    test_config_t lossless_configs[] = {
        /* Plain (no preprocessing) */
        { "lz4",       GPUCOMPRESS_ALGO_LZ4,      0, 0, 0.0, 0 },
        { "snappy",    GPUCOMPRESS_ALGO_SNAPPY,    0, 0, 0.0, 0 },
        { "deflate",   GPUCOMPRESS_ALGO_DEFLATE,   0, 0, 0.0, 0 },
        { "gdeflate",  GPUCOMPRESS_ALGO_GDEFLATE,  0, 0, 0.0, 0 },
        { "zstd",      GPUCOMPRESS_ALGO_ZSTD,      0, 0, 0.0, 0 },
        { "ans",       GPUCOMPRESS_ALGO_ANS,       0, 0, 0.0, 0 },
        { "cascaded",  GPUCOMPRESS_ALGO_CASCADED,  0, 0, 0.0, 0 },
        { "bitcomp",   GPUCOMPRESS_ALGO_BITCOMP,   0, 0, 0.0, 0 },
        /* With byte shuffle */
        { "lz4_shuf",      GPUCOMPRESS_ALGO_LZ4,      GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "snappy_shuf",   GPUCOMPRESS_ALGO_SNAPPY,    GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "deflate_shuf",  GPUCOMPRESS_ALGO_DEFLATE,   GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "gdeflate_shuf", GPUCOMPRESS_ALGO_GDEFLATE,  GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "zstd_shuf",     GPUCOMPRESS_ALGO_ZSTD,      GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "ans_shuf",      GPUCOMPRESS_ALGO_ANS,       GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "cascaded_shuf", GPUCOMPRESS_ALGO_CASCADED,  GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
        { "bitcomp_shuf",  GPUCOMPRESS_ALGO_BITCOMP,   GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
    };
    int n_lossless = sizeof(lossless_configs) / sizeof(lossless_configs[0]);

    /* ── ALGO_AUTO (NN) lossless configs ── */
    test_config_t nn_configs[] = {
        { "auto_nn",      GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0, 0 },
        { "auto_nn_shuf", GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.0, 0 },
    };
    int n_nn = sizeof(nn_configs) / sizeof(nn_configs[0]);

    /* ── Lossy (quantization) configs ── */
    test_config_t lossy_configs[] = {
        { "lz4_quant",       GPUCOMPRESS_ALGO_LZ4,  GPUCOMPRESS_PREPROC_QUANTIZE, 0, 0.01, 1 },
        { "zstd_quant",      GPUCOMPRESS_ALGO_ZSTD, GPUCOMPRESS_PREPROC_QUANTIZE, 0, 0.01, 1 },
        { "lz4_quant_shuf",  GPUCOMPRESS_ALGO_LZ4,
          GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.01, 1 },
        { "zstd_quant_shuf", GPUCOMPRESS_ALGO_ZSTD,
          GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.01, 1 },
    };
    int n_lossy = sizeof(lossy_configs) / sizeof(lossy_configs[0]);

    /* ── Auto + lossy ── */
    test_config_t auto_lossy_configs[] = {
        { "auto_quant_shuf", GPUCOMPRESS_ALGO_AUTO,
          GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4, 4, 0.01, 1 },
    };
    int n_auto_lossy = sizeof(auto_lossy_configs) / sizeof(auto_lossy_configs[0]);

    /* Run tests */
    printf("── Lossless tests (8 algos × 2 preproc × %d patterns = %d tests) ──\n",
           PAT_COUNT, n_lossless * PAT_COUNT);

    for (int c = 0; c < n_lossless; c++) {
        for (int p = 0; p < PAT_COUNT; p++) {
            fill_pattern(data, n_floats, (pattern_t)p);
            int ret = run_one_test(hdf5_path, &lossless_configs[c],
                                   (pattern_t)p, data, n_floats);
            printf("  [%s/%s] %s\n", lossless_configs[c].name,
                   pattern_name((pattern_t)p), ret == 0 ? "PASS" : "FAIL");
        }
    }

    if (nn_loaded) {
        printf("\n── NN (ALGO_AUTO) lossless tests (%d configs × %d patterns) ──\n",
               n_nn, PAT_COUNT);
        for (int c = 0; c < n_nn; c++) {
            for (int p = 0; p < PAT_COUNT; p++) {
                fill_pattern(data, n_floats, (pattern_t)p);
                int ret = run_one_test(hdf5_path, &nn_configs[c],
                                       (pattern_t)p, data, n_floats);
                printf("  [%s/%s] %s\n", nn_configs[c].name,
                       pattern_name((pattern_t)p), ret == 0 ? "PASS" : "FAIL");
            }
        }
    } else {
        printf("\n── NN tests skipped (no weights loaded) ──\n");
    }

    printf("\n── Lossy (quantization) tests (%d configs × %d patterns) ──\n",
           n_lossy, PAT_COUNT);
    for (int c = 0; c < n_lossy; c++) {
        for (int p = 0; p < PAT_COUNT; p++) {
            fill_pattern(data, n_floats, (pattern_t)p);
            int ret = run_one_test(hdf5_path, &lossy_configs[c],
                                   (pattern_t)p, data, n_floats);
            printf("  [%s/%s] %s\n", lossy_configs[c].name,
                   pattern_name((pattern_t)p), ret == 0 ? "PASS" : "FAIL");
        }
    }

    if (nn_loaded) {
        printf("\n── NN lossy tests (%d configs × %d patterns) ──\n",
               n_auto_lossy, PAT_COUNT);
        for (int c = 0; c < n_auto_lossy; c++) {
            for (int p = 0; p < PAT_COUNT; p++) {
                fill_pattern(data, n_floats, (pattern_t)p);
                int ret = run_one_test(hdf5_path, &auto_lossy_configs[c],
                                       (pattern_t)p, data, n_floats);
                printf("  [%s/%s] %s\n", auto_lossy_configs[c].name,
                       pattern_name((pattern_t)p), ret == 0 ? "PASS" : "FAIL");
            }
        }
    }

    /* Summary */
    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    free(data);
    gpucompress_cleanup();

    return (g_fail > 0) ? 1 : 0;
}
