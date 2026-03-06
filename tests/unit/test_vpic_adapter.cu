/**
 * @file test_vpic_adapter.cu
 * @brief Correctness tests for VPIC adapter
 *
 * Tests:
 *   1. Lifecycle — create/attach/destroy for each data type
 *   2. get_nbytes correctness for fields (16 vars), hydro (14), particles (7+1)
 *   3. Compression round-trip for fields
 *   4. Compression round-trip for hydro
 *   5. Compression round-trip for particles (float portion)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "gpucompress_vpic.h"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "  FAIL: %s\n", msg);                           \
            g_fail++;                                                       \
            return;                                                         \
        }                                                                   \
    } while (0)

/* ============================================================
 * Synthetic data fill kernel
 * ============================================================ */

__global__ void fill_sinusoidal(float* data, int n_elements, int n_vars)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_elements * n_vars;
    if (idx >= total) return;
    int elem = idx / n_vars;
    int var  = idx % n_vars;
    float x = (float)elem / (float)n_elements;
    data[idx] = sinf(x * 6.2831853f * (var + 1)) * 0.5f + 0.5f;
}

__global__ void fill_int_data(int* data, int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    data[idx] = idx % 4;  /* 4 species */
}

/* ============================================================
 * Test 1: Lifecycle (create/attach/destroy)
 * ============================================================ */

static void test_lifecycle()
{
    printf("Test 1: Lifecycle ...\n");

    vpic_data_type_t types[] = { VPIC_DATA_FIELDS, VPIC_DATA_HYDRO, VPIC_DATA_PARTICLES };
    const char* names[] = { "fields", "hydro", "particles" };

    for (int t = 0; t < 3; t++) {
        VpicSettings s = vpic_default_settings();
        s.data_type = types[t];

        gpucompress_vpic_t handle = NULL;
        gpucompress_error_t err;

        err = gpucompress_vpic_create(&handle, &s);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "create failed");
        TEST_ASSERT(handle != NULL, "handle is NULL after create");

        /* Attach a small synthetic buffer */
        size_t n_elem = 1024;
        int n_vars = (t == 0) ? 16 : (t == 1) ? 14 : 7;
        float* d_data = NULL;
        CHECK_CUDA(cudaMalloc(&d_data, n_elem * n_vars * sizeof(float)));

        int* d_data_i = NULL;
        if (t == 2) {
            CHECK_CUDA(cudaMalloc(&d_data_i, n_elem * sizeof(int)));
        }

        err = gpucompress_vpic_attach(handle, d_data, d_data_i, n_elem);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "attach failed");

        err = gpucompress_vpic_destroy(handle);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "destroy failed");

        CHECK_CUDA(cudaFree(d_data));
        if (d_data_i) CHECK_CUDA(cudaFree(d_data_i));

        printf("  %s: OK\n", names[t]);
    }

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: get_nbytes correctness
 * ============================================================ */

static void test_nbytes()
{
    printf("Test 2: get_nbytes correctness ...\n");

    struct { vpic_data_type_t type; size_t n_elem; size_t expect; const char* name; } cases[] = {
        { VPIC_DATA_FIELDS,    1000, 1000 * 16 * sizeof(float),                        "fields"    },
        { VPIC_DATA_HYDRO,     1000, 1000 * 14 * sizeof(float),                        "hydro"     },
        { VPIC_DATA_PARTICLES, 500,  500 * 7 * sizeof(float) + 500 * sizeof(int),      "particles" },
    };

    for (int c = 0; c < 3; c++) {
        VpicSettings s = vpic_default_settings();
        s.data_type = cases[c].type;

        gpucompress_vpic_t handle = NULL;
        gpucompress_vpic_create(&handle, &s);

        /* Attach with NULL pointers just to set element count */
        gpucompress_vpic_attach(handle, NULL, NULL, 0);

        /* Now attach with proper count (NULL data is OK for n_elements=0 test,
           but let's use dummy to test with actual count) */
        float* d_dummy = NULL;
        CHECK_CUDA(cudaMalloc(&d_dummy, cases[c].n_elem * 16 * sizeof(float)));
        int* d_dummy_i = NULL;
        if (cases[c].type == VPIC_DATA_PARTICLES) {
            CHECK_CUDA(cudaMalloc(&d_dummy_i, cases[c].n_elem * sizeof(int)));
        }

        gpucompress_vpic_attach(handle, d_dummy, d_dummy_i, cases[c].n_elem);

        size_t nbytes = 0;
        gpucompress_vpic_get_nbytes(handle, &nbytes);

        printf("  %s: nbytes=%zu  expected=%zu\n", cases[c].name, nbytes, cases[c].expect);
        TEST_ASSERT(nbytes == cases[c].expect, "nbytes mismatch");

        gpucompress_vpic_destroy(handle);
        CHECK_CUDA(cudaFree(d_dummy));
        if (d_dummy_i) CHECK_CUDA(cudaFree(d_dummy_i));
    }

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Helper: compression round-trip for a given data type
 * ============================================================ */

static void roundtrip_test(vpic_data_type_t dtype, size_t n_elem,
                           const char* label, int test_num)
{
    printf("Test %d: Compression round-trip (%s) ...\n", test_num, label);

    int n_vars = (dtype == VPIC_DATA_FIELDS) ? 16 :
                 (dtype == VPIC_DATA_HYDRO)  ? 14 : 7;
    size_t total_floats = n_elem * n_vars;
    size_t nbytes = total_floats * sizeof(float);

    /* Allocate and fill synthetic data on GPU */
    float* d_data = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, nbytes));
    int grid = ((int)total_floats + 255) / 256;
    fill_sinusoidal<<<grid, 256>>>(d_data, (int)n_elem, n_vars);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Create adapter and attach */
    VpicSettings s = vpic_default_settings();
    s.data_type = dtype;
    gpucompress_vpic_t handle = NULL;
    gpucompress_vpic_create(&handle, &s);
    gpucompress_vpic_attach(handle, d_data, NULL, n_elem);

    /* Get pointers back */
    float* got_data = NULL;
    size_t got_nbytes = 0;
    gpucompress_vpic_get_device_ptrs(handle, &got_data, NULL, &got_nbytes, NULL);
    TEST_ASSERT(got_data == d_data, "get_device_ptrs returned wrong pointer");
    TEST_ASSERT(got_nbytes == nbytes, "get_device_ptrs returned wrong nbytes");

    /* GPU compress via convenience API */
    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    void* d_compressed = NULL;
    CHECK_CUDA(cudaMalloc(&d_compressed, max_comp));
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats;
    err = gpucompress_compress_vpic(d_data, nbytes,
                                    d_compressed, &comp_size,
                                    &cfg, &stats);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "compress_vpic failed");

    printf("  compressed: %zu -> %zu bytes (ratio %.2fx)\n",
           nbytes, comp_size, stats.compression_ratio);

    /* GPU decompress */
    float* d_decomp = NULL;
    CHECK_CUDA(cudaMalloc(&d_decomp, nbytes));
    size_t decomp_size = nbytes;
    err = gpucompress_decompress_gpu(d_compressed, comp_size,
                                     d_decomp, &decomp_size, NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "decompress_gpu failed");
    TEST_ASSERT(decomp_size == nbytes, "decompress size mismatch");

    /* Verify lossless: copy both to host and compare */
    std::vector<float> h_orig(total_floats), h_decomp(total_floats);
    CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, nbytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_decomp.data(), d_decomp, nbytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t i = 0; i < total_floats; i++) {
        if (h_orig[i] != h_decomp[i]) mismatches++;
    }
    TEST_ASSERT(mismatches == 0, "lossless round-trip mismatch");

    gpucompress_vpic_destroy(handle);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_decomp));
    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Tests 3-5: Compression round-trips
 * ============================================================ */

static void test_roundtrip_fields()
{
    roundtrip_test(VPIC_DATA_FIELDS, 4096, "fields/16-var", 3);
}

static void test_roundtrip_hydro()
{
    roundtrip_test(VPIC_DATA_HYDRO, 4096, "hydro/14-var", 4);
}

static void test_roundtrip_particles()
{
    roundtrip_test(VPIC_DATA_PARTICLES, 4096, "particles/7-var", 5);
}

/* ============================================================
 * Main
 * ============================================================ */

int main()
{
    printf("=== VPIC Adapter Test Suite ===\n\n");

    test_lifecycle();
    test_nbytes();
    test_roundtrip_fields();
    test_roundtrip_hydro();
    test_roundtrip_particles();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
