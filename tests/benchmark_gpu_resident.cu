/**
 * tests/benchmark_gpu_resident.cu
 *
 * GPU-resident data HDF5 benchmark: No Compression vs NN Compression
 *
 * Both phases generate data on the GPU with a ramp kernel.
 *
 * Phase 1 — No compression (VOL, host pointer):
 *   GPU data → D→H (full dataset, included in write time) → H5Dwrite(H5S_ALL).
 *   VOL sees host pointer, no gpucompress filter → native HDF5 chunked write.
 *   Baseline: what it costs to get GPU-resident data onto disk without compression.
 *
 * Phase 2 — NN compression (VOL, GPU pointer):
 *   GPU data → H5Dwrite(H5S_ALL, d_full) via VOL.
 *   VOL detects GPU pointer + gpucompress filter → gpu_aware_chunked_write:
 *   per chunk: stats+NN inference+NVCOMP all on GPU, D→H compressed bytes only,
 *   H5Dwrite_chunk to file. No full-dataset D→H.
 *
 * Timing methodology:
 *   - N_TRIALS independent timed runs per phase; median reported.
 *   - fdatasync() inside write timing: ensures data reaches storage, not just
 *     OS page cache.  Eliminates "fast first write to RAM" artifact.
 *   - posix_fadvise(DONTNEED) before each read: drops the OS page cache for
 *     the file so reads are measured from storage, not from RAM.
 *   - Bandwidth in MiB/s (binary 1<<20), not decimal MB/s.
 *   - Speedup computed from medians, not single runs.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/benchmark_gpu_resident neural_net/weights/model.nnwt \
 *       [--dataset-mb N] [--chunk-mb N] [--trials N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Defaults
 * ============================================================ */
#define DEFAULT_DATASET_MB 8
#define DEFAULT_CHUNK_MB   4
#define DEFAULT_TRIALS     3

#define TMP_NOCOMP "/tmp/bm_gpu_nocomp.h5"
#define TMP_NN     "/tmp/bm_gpu_nn.h5"

/* ============================================================
 * Filter wiring (mirrors H5Zgpucompress.h — no link dependency)
 * ============================================================ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t dcpl_set_gpucompress(hid_t dcpl, unsigned int algo,
                                    unsigned int preproc, unsigned int shuf_sz,
                                    double error_bound)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algo; cd[1] = preproc; cd[2] = shuf_sz;
    pack_double_cd(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* ============================================================
 * Timing
 * ============================================================ */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Disk helpers
 * ============================================================ */

/* Force dirty pages for 'path' to reach the storage device.
 * Called inside the write timing window so the reported time
 * includes the actual physical write, not just the OS buffer flush. */
static void fdatasync_path(const char *path)
{
    int fd = open(path, O_WRONLY);
    if (fd >= 0) { fdatasync(fd); close(fd); }
}

/* Drop OS page-cache for 'path' so the subsequent read measures
 * actual storage throughput, not a RAM-resident copy. */
static void drop_pagecache(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

/* ============================================================
 * Statistics helpers (median of N doubles)
 * ============================================================ */
static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n)
{
    double tmp[32];
    if (n > 32) n = 32;
    memcpy(tmp, arr, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    return tmp[n / 2];
}

static double dmin(double *arr, int n)
{
    double m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] < m) m = arr[i];
    return m;
}

static double dmax(double *arr, int n)
{
    double m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
}

/* ============================================================
 * GPU ramp fill kernel: buf[i] = (float)i / (float)n
 * ============================================================ */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += stride)
        buf[i] = (float)i / (float)n;
}

static int gpu_fill_ramp(float *d_buf, size_t n_floats)
{
    ramp_kernel<<<65535, 256>>>(d_buf, n_floats);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
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
 * Create a chunked dataset via VOL
 * ============================================================ */
static int create_vol_dataset(const char *path, size_t n_floats, size_t chunk_floats,
                               int use_gpucompress_filter,
                               hid_t *file_out, hid_t *dset_out)
{
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) return -1;

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    if (use_gpucompress_filter)
        dcpl_set_gpucompress(dcpl, 0 /*ALGO_AUTO*/, 0, 0, 0.0 /*lossless*/);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(fspace);
    if (dset < 0) { H5Fclose(file); return -1; }

    *file_out = file;
    *dset_out = dset;
    return 0;
}

/* ============================================================
 * MiB/s helper  (binary 1 MiB = 1<<20 bytes)
 * ============================================================ */
static double to_mibps(size_t bytes, double ms)
{
    return (double)bytes / (double)(1 << 20) / (ms / 1000.0);
}

/* ============================================================
 * Phase 1: No compression — N_TRIALS timed runs
 *
 * Write timing scope: cudaMemcpy D→H  +  H5Dwrite  +  H5Dclose
 *                     + H5Fclose  +  fdatasync
 * Read  timing scope: posix_fadvise(DONTNEED)  +  H5Dread  +  H5Dclose
 *                     + H5Fclose
 * ============================================================ */
static int bench_nocomp(size_t n_floats, size_t chunk_floats, int n_trials,
                         double *write_ms, double *read_ms, size_t *file_bytes)
{
    size_t total_bytes = n_floats * sizeof(float);

    /* Allocate GPU + host buffers once (allocation not timed) */
    float *d_full = NULL, *h_full = NULL, *h_read = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "nocomp: cudaMalloc(%.1f GiB) failed\n",
                (double)total_bytes / (1u << 30)); return -1;
    }
    h_full = (float *)malloc(total_bytes);
    h_read = (float *)malloc(total_bytes);
    if (!h_full || !h_read) {
        fprintf(stderr, "nocomp: malloc failed\n");
        cudaFree(d_full); free(h_full); free(h_read); return -1;
    }

    printf("  Generating %.1f GiB ramp on GPU...\n",
           (double)total_bytes / (double)(1u << 30));
    if (gpu_fill_ramp(d_full, n_floats) != 0) {
        cudaFree(d_full); free(h_full); free(h_read); return -1;
    }

    int ok = 1;
    for (int t = 0; t < n_trials && ok; t++) {
        remove(TMP_NOCOMP);

        /* ---- WRITE ---- */
        hid_t file, dset;
        if (create_vol_dataset(TMP_NOCOMP, n_floats, chunk_floats,
                               0, &file, &dset) != 0) {
            fprintf(stderr, "nocomp: dataset create failed (trial %d)\n", t);
            ok = 0; break;
        }

        double t0 = now_ms();
        cudaMemcpy(h_full, d_full, total_bytes, cudaMemcpyDeviceToHost);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_full);
        H5Dclose(dset);
        H5Fclose(file);
        fdatasync_path(TMP_NOCOMP);   /* flush to storage — inside timing */
        double t1 = now_ms();
        write_ms[t] = t1 - t0;

        /* ---- READ ---- */
        drop_pagecache(TMP_NOCOMP);   /* evict from RAM — measures storage read */

        hid_t fapl = make_vol_fapl();
        file = H5Fopen(TMP_NOCOMP, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        dset = H5Dopen2(file, "data", H5P_DEFAULT);
        if (t == 0) *file_bytes = (size_t)H5Dget_storage_size(dset);

        double t2 = now_ms();
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_read);
        H5Dclose(dset);
        H5Fclose(file);
        double t3 = now_ms();
        read_ms[t] = t3 - t2;

        printf("  [trial %d/%d] write=%.0f ms (%.1f MiB/s)  "
               "read=%.0f ms (%.1f MiB/s)\n",
               t + 1, n_trials,
               write_ms[t], to_mibps(total_bytes, write_ms[t]),
               read_ms[t],  to_mibps(total_bytes, read_ms[t]));
    }

    remove(TMP_NOCOMP);

    /* Verify last read against the analytical ramp formula */
    if (ok) {
        for (size_t i = 0; i < n_floats && ok; i++) {
            float expected = (float)i / (float)n_floats;
            if (h_read[i] != expected) {
                fprintf(stderr, "nocomp: verify FAIL at [%zu]: %.8g != %.8g\n",
                        i, (double)h_read[i], (double)expected);
                ok = 0;
            }
        }
        if (ok) printf("  Lossless verify: PASS (%zu elements bit-exact)\n", n_floats);
    }

    cudaFree(d_full); free(h_full); free(h_read);
    return ok ? 0 : -1;
}

/* ============================================================
 * Phase 2: NN compression — 1 warmup + N_TRIALS timed runs
 *
 * Write timing scope: H5Dwrite (GPU ptr)  +  H5Dclose  +  H5Fclose
 *                     +  fdatasync
 * Read  timing scope: posix_fadvise(DONTNEED)  +  H5Dread  +  H5Dclose
 *                     + H5Fclose  +  cudaDeviceSynchronize
 * ============================================================ */
static int bench_nn(size_t n_floats, size_t chunk_floats, int n_trials,
                     double *write_ms, double *read_ms, size_t *file_bytes)
{
    size_t total_bytes = n_floats * sizeof(float);

    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "nn: cudaMalloc(%.1f GiB) failed\n",
                (double)total_bytes / (1u << 30)); return -1;
    }

    printf("  Generating %.1f GiB ramp on GPU...\n",
           (double)total_bytes / (double)(1u << 30));
    if (gpu_fill_ramp(d_full, n_floats) != 0) {
        cudaFree(d_full); return -1;
    }

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        cudaFree(d_full);
        fprintf(stderr, "nn: H5VL_gpucompress_register failed\n"); return -1;
    }

    /* ---- Warmup run (untimed): warms GPU kernels and nvcomp state ---- */
    printf("  Warmup run (untimed)...\n");
    {
        remove(TMP_NN);
        hid_t file, dset;
        if (create_vol_dataset(TMP_NN, n_floats, chunk_floats, 1, &file, &dset) == 0) {
            H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
            H5Dclose(dset); H5Fclose(file);
        }
        remove(TMP_NN);
    }

    int ok = 1;
    int first_chunks_compressed = 0;

    for (int t = 0; t < n_trials && ok; t++) {
        remove(TMP_NN);

        /* ---- WRITE ---- */
        hid_t file, dset;
        if (create_vol_dataset(TMP_NN, n_floats, chunk_floats,
                               1, &file, &dset) != 0) {
            fprintf(stderr, "nn: dataset create failed (trial %d)\n", t);
            ok = 0; break;
        }

        H5VL_gpucompress_reset_stats();
        double t0 = now_ms();
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
        H5Dclose(dset);
        H5Fclose(file);
        fdatasync_path(TMP_NN);       /* flush to storage — inside timing */
        double t1 = now_ms();
        write_ms[t] = t1 - t0;

        if (t == 0) {
            int w, r, c, d;
            H5VL_gpucompress_get_stats(&w, &r, &c, &d);
            first_chunks_compressed = c;
        }

        /* ---- READ ---- */
        drop_pagecache(TMP_NN);       /* evict from RAM — measures storage read */

        hid_t fapl = make_vol_fapl();
        file = H5Fopen(TMP_NN, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        dset = H5Dopen2(file, "data", H5P_DEFAULT);
        if (t == 0) *file_bytes = (size_t)H5Dget_storage_size(dset);

        float *d_read = NULL;
        if (cudaMalloc(&d_read, total_bytes) != cudaSuccess) {
            fprintf(stderr, "nn: cudaMalloc read buf failed\n");
            H5Dclose(dset); H5Fclose(file); ok = 0; break;
        }

        double t2 = now_ms();
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();      /* ensure all GPU decompression is done */
        H5Dclose(dset);
        H5Fclose(file);
        double t3 = now_ms();
        read_ms[t] = t3 - t2;

        printf("  [trial %d/%d] write=%.0f ms (%.1f MiB/s)  "
               "read=%.0f ms (%.1f MiB/s)\n",
               t + 1, n_trials,
               write_ms[t], to_mibps(total_bytes, write_ms[t]),
               read_ms[t],  to_mibps(total_bytes, read_ms[t]));

        /* Verify on every trial: D→H + compare against ramp formula */
        if (ok) {
            float *h_verify = (float *)malloc(total_bytes);
            if (!h_verify) {
                fprintf(stderr, "nn: verify malloc failed\n");
                cudaFree(d_read); ok = 0; break;
            }
            cudaMemcpy(h_verify, d_read, total_bytes, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < n_floats && ok; i++) {
                float expected = (float)i / (float)n_floats;
                if (h_verify[i] != expected) {
                    fprintf(stderr, "nn: verify FAIL at [%zu]: %.8g != %.8g\n",
                            i, (double)h_verify[i], (double)expected);
                    ok = 0;
                }
            }
            free(h_verify);
            if (ok) printf("  [trial %d/%d] Lossless verify: PASS\n", t + 1, n_trials);
        }

        cudaFree(d_read);
        remove(TMP_NN);
    }

    printf("  chunks_compressed per write: %d\n", first_chunks_compressed);

    H5VLclose(vol_id);
    cudaFree(d_full);
    return ok ? 0 : -1;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    int dataset_mb = DEFAULT_DATASET_MB;
    int chunk_mb   = DEFAULT_CHUNK_MB;
    int n_trials   = DEFAULT_TRIALS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset-mb") == 0 && i + 1 < argc)
            dataset_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc)
            chunk_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
            n_trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s weights.nnwt [--dataset-mb N] [--chunk-mb N] [--trials N]\n",
                   argv[0]);
            return 0;
        } else if (!weights_path) {
            weights_path = argv[i];
        }
    }

    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s weights.nnwt [--dataset-mb N] [--chunk-mb N] [--trials N]\n",
                argv[0]);
        return 1;
    }
    if (dataset_mb < chunk_mb) {
        fprintf(stderr, "ERROR: --dataset-mb (%d) must be >= --chunk-mb (%d)\n",
                dataset_mb, chunk_mb);
        return 1;
    }
    if (n_trials < 1 || n_trials > 32) {
        fprintf(stderr, "ERROR: --trials must be 1..32\n"); return 1;
    }

    size_t chunk_floats = (size_t)chunk_mb  * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)dataset_mb * 1024 * 1024 / sizeof(float);
    size_t n_chunks     = n_floats / chunk_floats;
    size_t total_bytes  = n_floats * sizeof(float);

    printf("=== GPU-Resident HDF5 Benchmark: No Compression vs NN ===\n\n");
    printf("Weights:  %s\n", weights_path);
    printf("Pattern:  ramp  (buf[i] = i / N)\n");
    printf("Chunk:    %d MiB  (%zu floats)\n", chunk_mb, chunk_floats);
    printf("Dataset:  %d MiB  (%zu floats, %zu chunks)\n",
           dataset_mb, n_floats, n_chunks);
    printf("Trials:   %d timed runs per phase (median reported)\n", n_trials);
    printf("Disk:     fdatasync inside write timing, pagecache drop before read\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded\n");
        gpucompress_cleanup(); return 1;
    }

    double nc_write_ms[32], nc_read_ms[32];
    double nn_write_ms[32], nn_read_ms[32];
    size_t nc_file_bytes = 0, nn_file_bytes = 0;

    /* ---- Phase 1: No compression ---- */
    printf("--- Phase 1: No Compression ---\n");
    printf("    Timing: cudaMemcpy(D→H) + H5Dwrite + H5Dclose + H5Fclose + fdatasync\n\n");
    if (bench_nocomp(n_floats, chunk_floats, n_trials,
                     nc_write_ms, nc_read_ms, &nc_file_bytes) != 0) {
        fprintf(stderr, "Phase 1 FAILED\n");
        gpucompress_cleanup(); return 1;
    }

    double nc_write_med = median(nc_write_ms, n_trials);
    double nc_read_med  = median(nc_read_ms,  n_trials);

    printf("\n  Phase 1 summary (MiB/s, binary 1 MiB = 1<<20 bytes):\n");
    printf("    Write median: %.1f MiB/s  [min %.1f  max %.1f]\n",
           to_mibps(total_bytes, nc_write_med),
           to_mibps(total_bytes, dmax(nc_write_ms, n_trials)),  /* max time = min BW */
           to_mibps(total_bytes, dmin(nc_write_ms, n_trials))); /* min time = max BW */
    printf("    Read  median: %.1f MiB/s  [min %.1f  max %.1f]\n\n",
           to_mibps(total_bytes, nc_read_med),
           to_mibps(total_bytes, dmax(nc_read_ms, n_trials)),
           to_mibps(total_bytes, dmin(nc_read_ms, n_trials)));

    /* ---- Phase 2: NN compression ---- */
    printf("--- Phase 2: NN Compression (ALGO_AUTO, lossless) ---\n");
    printf("    Timing: H5Dwrite(GPU ptr) + H5Dclose + H5Fclose + fdatasync\n\n");
    if (bench_nn(n_floats, chunk_floats, n_trials,
                 nn_write_ms, nn_read_ms, &nn_file_bytes) != 0) {
        fprintf(stderr, "Phase 2 FAILED\n");
        gpucompress_cleanup(); return 1;
    }

    double nn_write_med = median(nn_write_ms, n_trials);
    double nn_read_med  = median(nn_read_ms,  n_trials);
    double nn_ratio     = (nn_file_bytes > 0)
                        ? (double)total_bytes / (double)nn_file_bytes : 0.0;

    printf("\n  Phase 2 summary (MiB/s, binary):\n");
    printf("    Write median: %.1f MiB/s  [min %.1f  max %.1f]\n",
           to_mibps(total_bytes, nn_write_med),
           to_mibps(total_bytes, dmax(nn_write_ms, n_trials)),
           to_mibps(total_bytes, dmin(nn_write_ms, n_trials)));
    printf("    Read  median: %.1f MiB/s  [min %.1f  max %.1f]\n\n",
           to_mibps(total_bytes, nn_read_med),
           to_mibps(total_bytes, dmax(nn_read_ms, n_trials)),
           to_mibps(total_bytes, dmin(nn_read_ms, n_trials)));

    /* ---- Summary ---- */
    printf("=== Summary: %d MiB dataset, %d MiB chunks, ramp pattern ===\n\n",
           dataset_mb, chunk_mb);
    printf("  %-26s | %12s | %12s | %10s | %8s\n",
           "Mode", "Write MiB/s", "Read MiB/s", "Ratio", "File MiB");
    printf("  %-26s-+-%12s-+-%12s-+-%10s-+-%8s\n",
           "--------------------------", "------------", "------------",
           "----------", "--------");
    printf("  %-26s | %12.1f | %12.1f | %10.2fx | %8zu\n",
           "No comp (D→H→disk)",
           to_mibps(total_bytes, nc_write_med),
           to_mibps(total_bytes, nc_read_med),
           (double)total_bytes / (double)nc_file_bytes,
           nc_file_bytes >> 20);
    printf("  %-26s | %12.1f | %12.1f | %10.2fx | %8zu\n",
           "NN  (GPU compress→disk)",
           to_mibps(total_bytes, nn_write_med),
           to_mibps(total_bytes, nn_read_med),
           nn_ratio,
           nn_file_bytes >> 20);
    printf("\n");
    printf("  Write speedup (NN vs no-comp, median): %.2fx\n",
           nc_write_med / nn_write_med);
    printf("  Read  speedup (NN vs no-comp, median): %.2fx\n",
           nc_read_med  / nn_read_med);
    printf("  Space saving  (NN):                    %.1f%%\n",
           100.0 * (1.0 - 1.0 / nn_ratio));
    printf("\n");
    printf("  NOTE: ramp data (i/N) is maximally compressible; ratio will be\n");
    printf("        much lower for real scientific datasets (typically 2-10x).\n");
    printf("\n");

    gpucompress_cleanup();
    return 0;
}
