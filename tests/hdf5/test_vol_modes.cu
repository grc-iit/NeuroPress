/**
 * tests/hdf5/test_vol_modes.cu
 *
 * VOL mode integration test — Bypass / Release / Trace
 *
 * Writes 1 GB of GPU data through the VOL connector in each of the three
 * modes and validates the expected side-effects:
 *
 *   Bypass  — timing file created with a positive total_io_ms value
 *   Release — write succeeds; chunk history populated; no timing/trace files
 *   Trace   — trace CSV created with correct header and n_chunks * 32 data rows
 *
 * Mode is selected by setting GPUCOMPRESS_VOL_MODE before each
 * H5VL_gpucompress_register() call.  Because H5VL_gpucompress_init reads the
 * env var on registration, and H5VL_gpucompress_term is a no-op, we can
 * cycle through all three modes in a single process without re-initialising
 * the GPU compression pool.
 *
 * Dataset: 1 GB = 256 M floats
 * Chunk:   64 MB = 16 M floats  → 16 chunks per write
 * Trace:   16 chunks × 64 configs = 1024 CSV data rows expected
 *
 * Timeout: 600 s (Trace mode exhaustive profiling is expensive)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Parameters
 * ============================================================ */
#define DATASET_GB    1
#define CHUNK_MB      64

#define DATASET_FLOATS ((size_t)(DATASET_GB)  * 1024 * 1024 * 1024 / sizeof(float))
#define CHUNK_FLOATS   ((size_t)(CHUNK_MB)    * 1024 * 1024         / sizeof(float))
#define N_CHUNKS       (DATASET_FLOATS / CHUNK_FLOATS)

#define TMP_H5                "/tmp/test_vol_modes.h5"
#define BYPASS_TIMING_FILE    "/tmp/test_vol_bypass_timing.txt"
#define RELEASE_TIMING_FILE   "/tmp/test_vol_release_timing.txt"
#define TRACE_TIMING_FILE     "/tmp/test_vol_trace_timing.txt"
#define TRACE_CSV_FILE        "/tmp/test_vol_trace.csv"

/* Filter wiring */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * GPU fill kernel — simple ramp
 * ============================================================ */
__global__ static void fill_ramp(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = (float)i * (1.0f / (float)n);
}

/* ============================================================
 * Helpers
 * ============================================================ */

static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

/**
 * Write d_data (1 GB GPU buffer) to HDF5 file using the VOL connector.
 * Algo: LZ4 (1), no preprocessing — reliable, fast, lossless.
 * Returns 0 on success, -1 on failure.
 */
static int write_1gb(const float *d_data)
{
    hsize_t dims[1]  = { (hsize_t)DATASET_FLOATS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS   };

    remove(TMP_H5);

    hid_t fapl  = make_vol_fapl();
    hid_t file  = H5Fcreate(TMP_H5, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "  H5Fcreate failed\n"); return -1; }

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    /* LZ4, no preprocessing */
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {1, 0, 0, 0, 0};
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL,
                  H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(fspace);
    if (dset < 0) {
        H5Fclose(file); remove(TMP_H5);
        fprintf(stderr, "  H5Dcreate2 failed\n"); return -1;
    }

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);   /* ← triggers dumpIoTiming / flushTrace */
    remove(TMP_H5);

    if (wret < 0) { fprintf(stderr, "  H5Dwrite failed\n"); return -1; }
    return 0;
}

/* Parse the canonical io-timing CSV emitted by DiagnosticsStore::dumpIoTiming:
 *   line 1:  e2e_ms,vol_ms
 *   line 2:  <e2e>,<vol>
 * We return vol_ms (sum of H5Dwrite/H5Dread callback wall-clock) — that is
 * the value the test calls "total_io_ms" (total time spent in the VOL). */
static int parse_timing_file(const char *path, double *out_ms)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  Cannot open timing file: %s\n", path); return -1; }
    char header[256], data[256];
    if (!fgets(header, sizeof(header), f) || !fgets(data, sizeof(data), f)) {
        fclose(f);
        fprintf(stderr, "  timing file %s missing header or data row\n", path);
        return -1;
    }
    fclose(f);
    double e2e = 0.0, vol = 0.0;
    if (sscanf(data, "%lf,%lf", &e2e, &vol) != 2) {
        fprintf(stderr, "  timing file %s data row unparseable: %s", path, data);
        return -1;
    }
    *out_ms = vol;
    return 0;
}

/* ── Count lines in a file (0-based header excluded = data rows) ── */
static int count_csv_data_rows(const char *path, int *header_ok)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  Cannot open CSV: %s\n", path); return -1; }

    char line[1024];
    int  first = 1, rows = 0;
    *header_ok = 0;

    while (fgets(line, sizeof(line), f)) {
        /* strip trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (first) {
            /* Verify expected header — must match the canonical writer in
             * src/api/diagnostics_store.hpp (dumpTraceHeader). Updated from
             * the original 8-column format when the trace schema was extended
             * with psnr/ssim/max_error/mape_* and explore_mode columns. */
            const char *expected =
                "chunk_id,action_id,comp_lib,chosen,chunk_bytes,"
                "pred_cost,pred_ratio,pred_comp_ms,pred_decomp_ms,"
                "pred_psnr,pred_ssim,pred_max_error,"
                "real_cost,real_ratio,real_comp_ms,real_decomp_ms,"
                "real_psnr,real_ssim,real_max_error,"
                "mape_cost,mape_ratio,mape_comp_ms,mape_decomp_ms,"
                "explore_mode";
            *header_ok = (strcmp(line, expected) == 0) ? 1 : 0;
            if (!*header_ok)
                fprintf(stderr, "  Unexpected header: '%s'\n", line);
            first = 0;
        } else {
            rows++;
        }
    }
    fclose(f);
    return rows;
}

/* ============================================================
 * Subtest 1: Bypass mode
 * ============================================================ */
static int test_bypass(float *d_data)
{
    printf("\n--- Subtest 1: Bypass mode ---\n");

    setenv("GPUCOMPRESS_VOL_MODE",    "bypass",             1);
    setenv("GPUCOMPRESS_TIMING_OUTPUT", BYPASS_TIMING_FILE, 1);
    remove(BYPASS_TIMING_FILE);

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "  H5VL_gpucompress_register failed\n"); return -1;
    }

    int wret = write_1gb(d_data);   /* file_close dumps timing */
    H5VLclose(vol_id);

    if (wret != 0) { fprintf(stderr, "  write_1gb failed\n"); return -1; }

    /* Assert timing file was created */
    double total_ms = 0.0;
    if (parse_timing_file(BYPASS_TIMING_FILE, &total_ms) != 0) {
        fprintf(stderr, "  [FAIL] timing file not created or unreadable\n");
        return -1;
    }
    if (total_ms <= 0.0) {
        fprintf(stderr, "  [FAIL] total_io_ms=%.6f is not positive\n", total_ms);
        return -1;
    }
    printf("  timing file: %s  total_io_ms=%.3f\n", BYPASS_TIMING_FILE, total_ms);

    /* Assert trace CSV was NOT created in bypass mode */
    FILE *csv_check = fopen(TRACE_CSV_FILE, "r");
    if (csv_check) {
        fclose(csv_check);
        fprintf(stderr, "  [FAIL] trace CSV unexpectedly created in bypass mode\n");
        return -1;
    }

    printf("  [PASS] Bypass: timing file present (%.3f ms), no spurious CSV\n", total_ms);
    return 0;
}

/* ============================================================
 * Subtest 2: Release mode
 * ============================================================ */
static int test_release(float *d_data)
{
    printf("\n--- Subtest 2: Release mode ---\n");

    setenv("GPUCOMPRESS_VOL_MODE",     "release",           1);
    /* Set a path for the timing output — release mode must NOT write it */
    setenv("GPUCOMPRESS_TIMING_OUTPUT", RELEASE_TIMING_FILE, 1);
    remove(RELEASE_TIMING_FILE);
    remove(TRACE_CSV_FILE);

    /* Reset chunk history so we get a clean count for this write */
    gpucompress_reset_chunk_history();

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "  H5VL_gpucompress_register failed\n"); return -1;
    }

    int wret = write_1gb(d_data);
    H5VLclose(vol_id);

    if (wret != 0) { fprintf(stderr, "  write_1gb failed\n"); return -1; }

    /* Release mode MUST write a timing file */
    double rel_ms = 0.0;
    if (parse_timing_file(RELEASE_TIMING_FILE, &rel_ms) != 0) {
        fprintf(stderr, "  [FAIL] timing file not created in release mode\n");
        return -1;
    }
    if (rel_ms <= 0.0) {
        fprintf(stderr, "  [FAIL] total_io_ms=%.6f is not positive\n", rel_ms);
        return -1;
    }
    printf("  timing file: %s  total_io_ms=%.3f\n", RELEASE_TIMING_FILE, rel_ms);

    /* Release mode must NOT write a trace CSV */
    FILE *cf = fopen(TRACE_CSV_FILE, "r");
    if (cf) {
        fclose(cf);
        fprintf(stderr, "  [FAIL] trace CSV created in release mode (unexpected)\n");
        return -1;
    }

    /* Chunk history should be populated (N_CHUNKS entries) */
    int chunk_count = gpucompress_get_chunk_history_count();
    if (chunk_count <= 0) {
        fprintf(stderr, "  [FAIL] chunk history empty after write (got %d)\n",
                chunk_count);
        return -1;
    }
    printf("  chunk_count=%d  (expected ~%zu)\n", chunk_count, N_CHUNKS);

    printf("  [PASS] Release: timing file present (%.3f ms), no CSV, chunk history populated\n",
           rel_ms);
    return 0;
}

/* ============================================================
 * Subtest 3: Trace mode
 * ============================================================ */
static int test_trace(float *d_data)
{
    printf("\n--- Subtest 3: Trace mode ---\n");
    printf("  Dataset: %zu GB, %zu MB chunks → %zu chunks\n",
           (size_t)DATASET_GB, (size_t)CHUNK_MB, N_CHUNKS);
    printf("  Expected CSV rows: %zu × 64 = %zu\n", N_CHUNKS, N_CHUNKS * 64);
    printf("  (Exhaustive profiling — this may take several minutes)\n");

    setenv("GPUCOMPRESS_VOL_MODE",      "trace",           1);
    setenv("GPUCOMPRESS_TRACE_OUTPUT",  TRACE_CSV_FILE,    1);
    setenv("GPUCOMPRESS_TIMING_OUTPUT", TRACE_TIMING_FILE, 1);
    remove(TRACE_CSV_FILE);
    remove(TRACE_TIMING_FILE);

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "  H5VL_gpucompress_register failed\n"); return -1;
    }

    int wret = write_1gb(d_data);   /* file_close flushes trace CSV */
    H5VLclose(vol_id);

    if (wret != 0) { fprintf(stderr, "  write_1gb failed\n"); return -1; }

    /* Assert trace CSV was created */
    FILE *cf = fopen(TRACE_CSV_FILE, "r");
    if (!cf) {
        fprintf(stderr, "  [FAIL] trace CSV not found: %s\n", TRACE_CSV_FILE);
        return -1;
    }
    fclose(cf);

    /* Validate header and count data rows */
    int header_ok = 0;
    int rows = count_csv_data_rows(TRACE_CSV_FILE, &header_ok);
    if (rows < 0) return -1;

    if (!header_ok) {
        fprintf(stderr, "  [FAIL] trace CSV header does not match expected\n");
        return -1;
    }

    int expected_rows = (int)(N_CHUNKS * 64);
    if (rows != expected_rows) {
        fprintf(stderr, "  [FAIL] trace CSV has %d data rows, expected %d "
                "(%zu chunks × 64 configs)\n",
                rows, expected_rows, N_CHUNKS);
        return -1;
    }
    printf("  trace CSV: %s  rows=%d  header=OK\n", TRACE_CSV_FILE, rows);

    /* Spot-check first data row: must have 24 comma-separated fields
     * (23 commas). Matches the canonical header emitted by
     * DiagnosticsStore::dumpTraceHeader — see src/api/diagnostics_store.hpp. */
    {
        FILE *f2 = fopen(TRACE_CSV_FILE, "r");
        char line[1024];
        char *_hdr = fgets(line, sizeof(line), f2); (void)_hdr; /* skip header */
        if (fgets(line, sizeof(line), f2)) {
            int commas = 0;
            for (size_t k = 0; line[k]; k++)
                commas += (line[k] == ',');
            if (commas != 23) {
                fprintf(stderr, "  [FAIL] first data row has %d commas (expected 23): %s\n",
                        commas, line);
                fclose(f2); return -1;
            }
            printf("  first data row: %s", line);
        }
        fclose(f2);
    }

    /* Trace mode MUST also dump the timing file */
    double trace_ms = 0.0;
    if (parse_timing_file(TRACE_TIMING_FILE, &trace_ms) != 0) {
        fprintf(stderr, "  [FAIL] timing file not created in trace mode\n");
        return -1;
    }
    if (trace_ms <= 0.0) {
        fprintf(stderr, "  [FAIL] total_io_ms=%.6f is not positive in trace mode\n",
                trace_ms);
        return -1;
    }
    printf("  timing file: %s  total_io_ms=%.3f\n", TRACE_TIMING_FILE, trace_ms);

    printf("  [PASS] Trace: CSV present, header correct, %d rows = %zu×64, "
           "timing file present (%.3f ms)\n",
           rows, N_CHUNKS, trace_ms);
    return 0;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    printf("=== VOL Mode Integration Test ===\n");
    printf("    Dataset: %d GB (%zu floats)   Chunk: %d MB (%zu floats)   "
           "Chunks: %zu\n",
           DATASET_GB, DATASET_FLOATS, CHUNK_MB, CHUNK_FLOATS, N_CHUNKS);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Initialise library once — VOL can be re-registered without re-init */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    /* Allocate 1 GB GPU buffer and fill with ramp pattern */
    size_t total_bytes = DATASET_FLOATS * sizeof(float);
    float *d_data = NULL;
    if (cudaMalloc(&d_data, total_bytes) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc %zu MB failed\n", total_bytes >> 20);
        gpucompress_cleanup(); return 1;
    }
    fill_ramp<<<65535, 256>>>(d_data, DATASET_FLOATS);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "FATAL: GPU fill kernel failed\n");
        cudaFree(d_data); gpucompress_cleanup(); return 1;
    }

    int fail = 0;
    fail += (test_bypass (d_data) != 0) ? 1 : 0;
    fail += (test_release(d_data) != 0) ? 1 : 0;
    fail += (test_trace  (d_data) != 0) ? 1 : 0;

    cudaFree(d_data);
    gpucompress_cleanup();

    printf("\n=== Summary ===\n");
    if (fail == 0)
        printf("  ALL VOL MODE TESTS PASSED\n\n");
    else
        printf("  %d SUBTEST(S) FAILED\n\n", fail);

    /* Cleanup output files */
    remove(BYPASS_TIMING_FILE);
    remove(TRACE_CSV_FILE);
    remove(TRACE_TIMING_FILE);

    return (fail == 0) ? 0 : 1;
}
