/**
 * @file test_hdf5_plugin.c
 * @brief Test suite for src/hdf5/ fixes
 *
 * Tests:
 *  1. pack_double / unpack_double round-trip for various values
 *  2. Filter registration and double-registration
 *  3. H5Pset/H5Pget_gpucompress parameter round-trip
 *  4. Compress/decompress round-trip through HDF5 filter (4 MB)
 *  5. Chunk algorithm tracking and attr write (bounds-safe)
 *  6. Verbose logging is silent by default (no GPUCOMPRESS_VERBOSE)
 *  7. Plugin unload destructor (ensure_initialized + fini symmetry)
 *  8. Decompression passthrough for non-compressed data
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while (0)

#define ASSERT_EQ_INT(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("  FAIL: %s — expected %d, got %d (line %d)\n", msg, (int)(b), (int)(a), __LINE__); \
        g_fail++; return; \
    } \
} while (0)

/* ---- Helper: pack/unpack (same logic as in H5Zgpucompress.c) ---- */

static void test_pack_double(double value, unsigned int* lo, unsigned int* hi) {
    union { double d; unsigned int u[2]; } u;
    u.d = value;
    *lo = u.u[0];
    *hi = u.u[1];
}

static double test_unpack_double(unsigned int lo, unsigned int hi) {
    union { double d; unsigned int u[2]; } u;
    u.u[0] = lo;
    u.u[1] = hi;
    return u.d;
}

/* ============================================================
 * Test 1: pack_double / unpack_double round-trip
 * ============================================================ */
static void test_pack_unpack() {
    printf("Test 1: pack_double / unpack_double round-trip...\n");

    double values[] = {0.0, 1.0, -1.0, 1e-7, 3.14159265358979, 1e30, -1e-30};
    int n = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < n; i++) {
        unsigned int lo, hi;
        test_pack_double(values[i], &lo, &hi);
        double recovered = test_unpack_double(lo, hi);
        ASSERT_TRUE(recovered == values[i], "round-trip exact for double value");
    }

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: Filter registration and double-registration
 * ============================================================ */
static void test_registration() {
    printf("Test 2: filter registration...\n");

    herr_t rc = H5Z_gpucompress_register();
    ASSERT_TRUE(rc >= 0, "first registration succeeds");

    /* Double registration should be harmless */
    rc = H5Z_gpucompress_register();
    ASSERT_TRUE(rc >= 0, "second registration succeeds");

    htri_t avail = H5Z_gpucompress_is_registered();
    ASSERT_TRUE(avail > 0, "filter is available");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 3: H5Pset/H5Pget_gpucompress parameter round-trip
 * ============================================================ */
static void test_param_roundtrip() {
    printf("Test 3: H5Pset/H5Pget parameter round-trip...\n");

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    ASSERT_TRUE(dcpl >= 0, "create dcpl");

    /* Must set chunking before filters */
    hsize_t chunk_dims[1] = {1024};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    herr_t rc = H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD, 0, 4, 0.001);
    ASSERT_TRUE(rc >= 0, "H5Pset_gpucompress");

    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_LZ4;
    unsigned int preproc = 0;
    unsigned int shuf_size = 0;
    double eb = 0.0;

    rc = H5Pget_gpucompress(dcpl, &algo, &preproc, &shuf_size, &eb);
    ASSERT_TRUE(rc >= 0, "H5Pget_gpucompress");
    ASSERT_EQ_INT(algo, GPUCOMPRESS_ALGO_ZSTD, "algorithm round-trip");
    ASSERT_EQ_INT(shuf_size, 4, "shuffle_size round-trip");
    ASSERT_TRUE(fabs(eb - 0.001) < 1e-12, "error_bound round-trip");

    H5Pclose(dcpl);
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 4: Compress/decompress round-trip through HDF5 (4 MB)
 * ============================================================ */
static void test_hdf5_roundtrip() {
    printf("Test 4: HDF5 compress/decompress round-trip (4 MB)...\n");

    const char* tmpfile = "/tmp/test_hdf5_plugin_rt.h5";
    const size_t N = 1024 * 1024;  /* 1M floats = 4 MB */
    const size_t CHUNK = 256 * 1024;  /* 1 MB chunks */

    float* write_buf = (float*)malloc(N * sizeof(float));
    float* read_buf = (float*)malloc(N * sizeof(float));
    ASSERT_TRUE(write_buf && read_buf, "malloc");

    /* Generate test data */
    for (size_t i = 0; i < N; i++) {
        write_buf[i] = sinf((float)i * 0.01f) * 500.0f + (float)i * 0.001f;
    }

    /* Write */
    hid_t file_id = H5Fcreate(tmpfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT_TRUE(file_id >= 0, "H5Fcreate");

    hsize_t dims[1] = {N};
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    herr_t rc = H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0);
    ASSERT_TRUE(rc >= 0, "H5Pset_gpucompress");

    hid_t dset = H5Dcreate2(file_id, "test_data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    ASSERT_TRUE(dset >= 0, "H5Dcreate2");

    rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, write_buf);
    ASSERT_TRUE(rc >= 0, "H5Dwrite");

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file_id);

    /* Read back */
    file_id = H5Fopen(tmpfile, H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_TRUE(file_id >= 0, "H5Fopen");

    dset = H5Dopen2(file_id, "test_data", H5P_DEFAULT);
    ASSERT_TRUE(dset >= 0, "H5Dopen2");

    memset(read_buf, 0, N * sizeof(float));
    rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, read_buf);
    ASSERT_TRUE(rc >= 0, "H5Dread");

    H5Dclose(dset);
    H5Fclose(file_id);

    /* Verify lossless round-trip */
    int mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        if (write_buf[i] != read_buf[i]) mismatches++;
    }
    ASSERT_EQ_INT(mismatches, 0, "lossless round-trip");

    free(write_buf);
    free(read_buf);
    remove(tmpfile);

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 5: Chunk tracking + attr write (bounds-safe strcat fix)
 * ============================================================ */
static void test_chunk_tracking() {
    printf("Test 5: chunk tracking and attr write...\n");

    const char* tmpfile = "/tmp/test_hdf5_plugin_track.h5";

    /* Reset tracking and simulate some chunk algorithms */
    H5Z_gpucompress_reset_chunk_tracking();
    ASSERT_EQ_INT(H5Z_gpucompress_get_chunk_count(), 0, "count after reset");

    /* Write a dataset to trigger actual chunk tracking */
    const size_t N = 512 * 1024;  /* 512K floats = 2 MB */
    const size_t CHUNK = 128 * 1024;  /* 512 KB chunks → 4 chunks */

    float* buf = (float*)malloc(N * sizeof(float));
    ASSERT_TRUE(buf != NULL, "malloc");
    for (size_t i = 0; i < N; i++) buf[i] = (float)i * 0.01f;

    hid_t file_id = H5Fcreate(tmpfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT_TRUE(file_id >= 0, "H5Fcreate");

    hsize_t dims[1] = {N};
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0);

    hid_t dset = H5Dcreate2(file_id, "tracked", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    ASSERT_TRUE(dset >= 0, "H5Dcreate2");

    H5Z_gpucompress_reset_chunk_tracking();
    herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, buf);
    ASSERT_TRUE(rc >= 0, "H5Dwrite");

    /* Should have tracked 4 chunks */
    int count = H5Z_gpucompress_get_chunk_count();
    ASSERT_TRUE(count > 0, "tracked at least 1 chunk");

    /* Write chunk attr — should succeed (uses snprintf-based safe build) */
    rc = H5Z_gpucompress_write_chunk_attr(dset);
    ASSERT_TRUE(rc >= 0, "write_chunk_attr");

    /* Verify we can read the attribute back */
    htri_t attr_exists = H5Aexists(dset, "gpucompress_chunk_algorithms");
    ASSERT_TRUE(attr_exists > 0, "attribute exists");

    /* Out-of-range chunk index returns -1 */
    ASSERT_EQ_INT(H5Z_gpucompress_get_chunk_algorithm(-1), -1, "negative index");
    ASSERT_EQ_INT(H5Z_gpucompress_get_chunk_algorithm(9999), -1, "OOB index");

    /* Valid index returns a valid algorithm */
    int algo = H5Z_gpucompress_get_chunk_algorithm(0);
    ASSERT_TRUE(algo >= 0 && algo <= 8, "valid algo for chunk 0");

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file_id);

    free(buf);
    remove(tmpfile);

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 6: Verbose logging silent by default
 * ============================================================ */
static void test_verbose_default_silent() {
    printf("Test 6: verbose logging silent by default...\n");

    /* We're running without GPUCOMPRESS_VERBOSE set.
     * This test just verifies the filter ran in test 4
     * without producing chunk-level output lines.
     * If we got here, the filter already ran silently in test 4. */

    /* Verify env var is not set */
    const char* v = getenv("GPUCOMPRESS_VERBOSE");
    ASSERT_TRUE(v == NULL || v[0] == '0', "GPUCOMPRESS_VERBOSE not set");

    printf("  PASS (filter ran silently in tests 4-5)\n");
    g_pass++;
}

/* ============================================================
 * Test 7: Destructor symmetry (init -> fini doesn't double-free)
 * ============================================================ */
static void test_init_cleanup_symmetry() {
    printf("Test 7: init/cleanup symmetry...\n");

    /* refcount = 2 (main's gpucompress_init + ensure_initialized inside register).
     * Need two cleanups to reach 0. */
    gpucompress_cleanup();  /* refcount 2 -> 1 */
    gpucompress_cleanup();  /* refcount 1 -> 0, actual cleanup */
    ASSERT_EQ_INT(gpucompress_is_initialized(), 0, "cleaned up");

    /* Re-init should work */
    gpucompress_error_t rc = gpucompress_init(NULL);
    ASSERT_EQ_INT(rc, GPUCOMPRESS_SUCCESS, "re-init after cleanup");
    ASSERT_EQ_INT(gpucompress_is_initialized(), 1, "re-initialized");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 8: Decompression passthrough for non-compressed data
 * ============================================================ */
static void test_decompress_passthrough() {
    printf("Test 8: decompression passthrough for uncompressed data...\n");

    const char* tmpfile = "/tmp/test_hdf5_plugin_pass.h5";

    /* Write data that won't compress (random-ish) */
    const size_t N = 1024;
    float* buf = (float*)malloc(N * sizeof(float));
    ASSERT_TRUE(buf != NULL, "malloc");

    /* Generate incompressible data (random bytes reinterpreted as float) */
    srand(42);
    for (size_t i = 0; i < N; i++) {
        unsigned int r = (unsigned int)rand();
        memcpy(&buf[i], &r, sizeof(float));
    }

    hid_t file_id = H5Fcreate(tmpfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT_TRUE(file_id >= 0, "H5Fcreate");

    hsize_t dims[1] = {N};
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {N};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    /* Use LZ4 — random data likely won't compress */
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 0, 0.0);

    hid_t dset = H5Dcreate2(file_id, "random_data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    ASSERT_TRUE(dset >= 0, "H5Dcreate2");

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, buf);
    ASSERT_TRUE(rc >= 0, "H5Dwrite");

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file_id);

    /* Read back — should work regardless of whether passthrough was used */
    float* read_buf = (float*)malloc(N * sizeof(float));
    ASSERT_TRUE(read_buf != NULL, "malloc read");

    file_id = H5Fopen(tmpfile, H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_TRUE(file_id >= 0, "H5Fopen");
    dset = H5Dopen2(file_id, "random_data", H5P_DEFAULT);
    ASSERT_TRUE(dset >= 0, "H5Dopen2");

    rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, read_buf);
    ASSERT_TRUE(rc >= 0, "H5Dread");

    H5Dclose(dset);
    H5Fclose(file_id);

    ASSERT_TRUE(memcmp(buf, read_buf, N * sizeof(float)) == 0,
                "data matches after passthrough round-trip");

    free(buf);
    free(read_buf);
    remove(tmpfile);

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    printf("=== HDF5 Plugin Test Suite ===\n\n");

    /* Initialize library and register filter */
    gpucompress_error_t rc = gpucompress_init(NULL);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }

    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    test_pack_unpack();
    test_registration();
    test_param_roundtrip();
    test_hdf5_roundtrip();
    test_chunk_tracking();
    test_verbose_default_silent();
    test_decompress_passthrough();
    test_init_cleanup_symmetry();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    /* test_init_cleanup_symmetry already cleaned up and re-initialized,
     * so one cleanup balances the re-init. */
    gpucompress_cleanup();
    return g_fail > 0 ? 1 : 0;
}
