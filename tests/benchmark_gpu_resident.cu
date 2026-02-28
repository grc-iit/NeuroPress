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
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/benchmark_gpu_resident neural_net/weights/model.nnwt \
 *       [--dataset-mb N] [--chunk-mb N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Defaults
 * ============================================================ */
#define DEFAULT_DATASET_MB 8
#define DEFAULT_CHUNK_MB   4

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
 * GPU ramp fill kernel: buf[i] = (float)i / (float)n
 * ============================================================ */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += stride)
        buf[i] = (float)i / (float)n;
}

/* Fill and sync a GPU buffer with the ramp pattern */
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
 * Create a chunked dataset via VOL, optionally with gpucompress filter.
 * Returns (dset, file) or (H5I_INVALID_HID, H5I_INVALID_HID) on error.
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
 * Phase 1: No compression
 *   Data generated on GPU. D→H transfer is included in write timing
 *   since it is an unavoidable cost for uncompressed GPU-resident writes.
 * ============================================================ */
static int bench_nocomp(size_t n_floats, size_t chunk_floats,
                         double *write_ms, double *read_ms, size_t *file_bytes)
{
    remove(TMP_NOCOMP);
    size_t total_bytes = n_floats * sizeof(float);

    /* --- GPU data generation (not timed) --- */
    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "nocomp: cudaMalloc(%.1f GB) failed\n",
                (double)total_bytes / (1u << 30));
        return -1;
    }
    printf("  Generating %.1f GB ramp on GPU...\n", (double)total_bytes / (1u << 30));
    if (gpu_fill_ramp(d_full, n_floats) != 0) {
        cudaFree(d_full); return -1;
    }

    /* Host staging buffer for D→H (allocation not timed) */
    float *h_full = (float *)malloc(total_bytes);
    if (!h_full) {
        cudaFree(d_full);
        fprintf(stderr, "nocomp: malloc(%.1f GB) failed\n",
                (double)total_bytes / (1u << 30));
        return -1;
    }

    /* --- WRITE: D→H + HDF5 write (both timed together) --- */
    hid_t file, dset;
    if (create_vol_dataset(TMP_NOCOMP, n_floats, chunk_floats,
                           0 /*no filter*/, &file, &dset) != 0) {
        free(h_full); cudaFree(d_full);
        fprintf(stderr, "nocomp: dataset create failed\n"); return -1;
    }

    double t0 = now_ms();
    /* D→H: mandatory full-dataset transfer since no GPU compression path */
    cudaMemcpy(h_full, d_full, total_bytes, cudaMemcpyDeviceToHost);
    /* VOL sees host pointer, no gpucompress filter → native chunked write */
    herr_t hs = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, h_full);
    H5Dclose(dset);
    H5Fclose(file);
    double t1 = now_ms();

    cudaFree(d_full);
    free(h_full);

    if (hs < 0) { fprintf(stderr, "nocomp: H5Dwrite failed\n"); return -1; }
    *write_ms = t1 - t0;

    /* --- READ --- */
    float *h_read = (float *)malloc(total_bytes);
    if (!h_read) { fprintf(stderr, "nocomp: read malloc failed\n"); return -1; }

    hid_t fapl = make_vol_fapl();
    file = H5Fopen(TMP_NOCOMP, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) { free(h_read); fprintf(stderr, "nocomp: reopen failed\n"); return -1; }

    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    *file_bytes = (size_t)H5Dget_storage_size(dset);

    double t2 = now_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_read);
    H5Dclose(dset);
    H5Fclose(file);
    double t3 = now_ms();

    if (hs < 0) { free(h_read); fprintf(stderr, "nocomp: H5Dread failed\n"); return -1; }
    *read_ms = t3 - t2;

    /* Full bit-exact verification against ramp formula */
    int ok = 1;
    for (size_t i = 0; i < n_floats && ok; i++) {
        float expected = (float)i / (float)n_floats;
        if (h_read[i] != expected) {
            fprintf(stderr, "nocomp: verify FAIL at [%zu]: %.8g != %.8g\n",
                    i, (double)h_read[i], (double)expected);
            ok = 0;
        }
    }
    if (!ok) { free(h_read); remove(TMP_NOCOMP); return -1; }
    printf("  Lossless verify: PASS (%zu elements bit-exact)\n", n_floats);

    free(h_read);
    remove(TMP_NOCOMP);
    return 0;
}

/* ============================================================
 * Phase 2: NN compression via VOL (GPU pointer path)
 *   Data stays on GPU through entire compression pipeline.
 *   Only compressed bytes cross PCIe to disk.
 * ============================================================ */
static int bench_nn(size_t n_floats, size_t chunk_floats,
                     double *write_ms, double *read_ms, size_t *file_bytes)
{
    remove(TMP_NN);
    size_t total_bytes = n_floats * sizeof(float);

    /* --- GPU data generation (not timed) --- */
    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "nn: cudaMalloc(%.1f GB) failed — not enough VRAM\n",
                (double)total_bytes / (1u << 30));
        return -1;
    }
    printf("  Generating %.1f GB ramp on GPU...\n", (double)total_bytes / (1u << 30));
    if (gpu_fill_ramp(d_full, n_floats) != 0) {
        cudaFree(d_full); return -1;
    }

    /* Register VOL (idempotent) */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        cudaFree(d_full);
        fprintf(stderr, "nn: H5VL_gpucompress_register failed\n"); return -1;
    }

    /* --- WRITE: GPU pointer → VOL → compress each chunk on GPU → disk --- */
    {
        hid_t file, dset;
        if (create_vol_dataset(TMP_NN, n_floats, chunk_floats,
                               1 /*gpucompress filter*/, &file, &dset) != 0) {
            cudaFree(d_full); H5VLclose(vol_id);
            fprintf(stderr, "nn: dataset create failed\n"); return -1;
        }

        H5VL_gpucompress_reset_stats();

        double t0 = now_ms();
        /* d_full is a GPU pointer: VOL gpu_aware_chunked_write handles each chunk */
        herr_t hs = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
        H5Dclose(dset);
        H5Fclose(file);
        double t1 = now_ms();

        if (hs < 0) {
            cudaFree(d_full); H5VLclose(vol_id);
            fprintf(stderr, "nn: H5Dwrite failed\n"); return -1;
        }
        *write_ms = t1 - t0;

        int w, r, c, d;
        H5VL_gpucompress_get_stats(&w, &r, &c, &d);
        printf("  VOL: write_calls=%d  chunks_compressed=%d\n", w, c);
    }
    cudaFree(d_full);
    d_full = NULL;

    /* --- READ: VOL + GPU destination → decompress each chunk on GPU --- */
    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fopen(TMP_NN, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (file < 0) {
            H5VLclose(vol_id);
            fprintf(stderr, "nn: reopen failed\n"); return -1;
        }

        hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
        *file_bytes = (size_t)H5Dget_storage_size(dset);

        float *d_read = NULL;
        if (cudaMalloc(&d_read, total_bytes) != cudaSuccess) {
            H5Dclose(dset); H5Fclose(file); H5VLclose(vol_id);
            fprintf(stderr, "nn: cudaMalloc read buf failed\n"); return -1;
        }

        double t2 = now_ms();
        herr_t hs = H5Dread(dset, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();
        H5Dclose(dset);
        H5Fclose(file);
        double t3 = now_ms();

        if (hs < 0) {
            cudaFree(d_read); H5VLclose(vol_id);
            fprintf(stderr, "nn: H5Dread failed\n"); return -1;
        }
        *read_ms = t3 - t2;

        /* Full bit-exact verification: D→H entire dataset, compare against ramp formula */
        float *h_verify = (float *)malloc(total_bytes);
        if (!h_verify) {
            cudaFree(d_read); H5VLclose(vol_id);
            fprintf(stderr, "nn: verify malloc failed\n"); return -1;
        }
        cudaMemcpy(h_verify, d_read, total_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_read);

        int ok = 1;
        for (size_t i = 0; i < n_floats && ok; i++) {
            float expected = (float)i / (float)n_floats;
            if (h_verify[i] != expected) {
                fprintf(stderr, "nn: verify FAIL at [%zu]: %.8g != %.8g\n",
                        i, (double)h_verify[i], (double)expected);
                ok = 0;
            }
        }
        free(h_verify);

        if (!ok) { H5VLclose(vol_id); remove(TMP_NN); return -1; }
        printf("  Lossless verify: PASS (%zu elements bit-exact)\n", n_floats);
    }

    H5VLclose(vol_id);
    remove(TMP_NN);
    return 0;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    int dataset_mb = DEFAULT_DATASET_MB;
    int chunk_mb   = DEFAULT_CHUNK_MB;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset-mb") == 0 && i + 1 < argc)
            dataset_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc)
            chunk_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s weights.nnwt [--dataset-mb N] [--chunk-mb N]\n", argv[0]);
            return 0;
        } else if (!weights_path) {
            weights_path = argv[i];
        }
    }

    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s weights.nnwt [--dataset-mb N] [--chunk-mb N]\n", argv[0]);
        return 1;
    }
    if (dataset_mb < chunk_mb) {
        fprintf(stderr, "ERROR: --dataset-mb (%d) must be >= --chunk-mb (%d)\n",
                dataset_mb, chunk_mb);
        return 1;
    }

    size_t chunk_floats = (size_t)chunk_mb  * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)dataset_mb * 1024 * 1024 / sizeof(float);
    size_t n_chunks     = n_floats / chunk_floats;
    size_t total_bytes  = n_floats * sizeof(float);

    printf("=== GPU-Resident HDF5 Benchmark: No Compression vs NN ===\n\n");
    printf("Weights:  %s\n", weights_path);
    printf("Pattern:  ramp  (buf[i] = i / N, linear 0..1)\n");
    printf("Chunk:    %d MB  (%zu floats)\n", chunk_mb, chunk_floats);
    printf("Dataset:  %d MB  (%zu floats, %zu chunks)\n\n",
           dataset_mb, n_floats, n_chunks);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded\n");
        gpucompress_cleanup(); return 1;
    }

    /* ---- Phase 1: No compression ---- */
    printf("--- Phase 1: No Compression ---\n");
    printf("    GPU ramp → D→H (timed) → VOL → native HDF5 chunked write\n\n");
    double nc_write_ms = 0, nc_read_ms = 0;
    size_t nc_file_bytes = 0;
    if (bench_nocomp(n_floats, chunk_floats,
                     &nc_write_ms, &nc_read_ms, &nc_file_bytes) != 0) {
        fprintf(stderr, "Phase 1 FAILED\n");
        gpucompress_cleanup(); return 1;
    }
    double nc_write_mbps = (double)total_bytes / 1e6 / (nc_write_ms / 1000.0);
    double nc_read_mbps  = (double)total_bytes / 1e6 / (nc_read_ms  / 1000.0);
    printf("  Write: %.1f ms  (%.1f MB/s)\n", nc_write_ms, nc_write_mbps);
    printf("  Read:  %.1f ms  (%.1f MB/s)\n", nc_read_ms,  nc_read_mbps);
    printf("  File:  %.2fx ratio (uncompressed)\n\n",
           (double)total_bytes / (double)nc_file_bytes);

    /* ---- Phase 2: NN compression ---- */
    printf("--- Phase 2: NN Compression (ALGO_AUTO, lossless) ---\n");
    printf("    GPU ramp → VOL → GPU compress → D→H compressed bytes → disk\n\n");
    double nn_write_ms = 0, nn_read_ms = 0;
    size_t nn_file_bytes = 0;
    if (bench_nn(n_floats, chunk_floats,
                 &nn_write_ms, &nn_read_ms, &nn_file_bytes) != 0) {
        fprintf(stderr, "Phase 2 FAILED\n");
        gpucompress_cleanup(); return 1;
    }
    double nn_ratio      = (nn_file_bytes > 0)
                         ? (double)total_bytes / (double)nn_file_bytes : 0.0;
    double nn_write_mbps = (double)total_bytes / 1e6 / (nn_write_ms / 1000.0);
    double nn_read_mbps  = (double)total_bytes / 1e6 / (nn_read_ms  / 1000.0);
    printf("  Write: %.1f ms  (%.1f MB/s)\n", nn_write_ms, nn_write_mbps);
    printf("  Read:  %.1f ms  (%.1f MB/s)\n", nn_read_ms,  nn_read_mbps);
    printf("  File:  %.2fx compression ratio\n\n", nn_ratio);

    /* ---- Summary ---- */
    printf("=== Summary: %d MB, %d MB chunks, ramp pattern ===\n\n",
           dataset_mb, chunk_mb);
    printf("  %-24s | %10s | %10s | %8s | %8s\n",
           "Mode", "Write MB/s", "Read MB/s", "Ratio", "File MB");
    printf("  %-24s-+-%10s-+-%10s-+-%8s-+-%8s\n",
           "------------------------", "----------", "----------", "--------", "--------");
    printf("  %-24s | %10.1f | %10.1f | %8.2fx | %8zu\n",
           "No comp (GPU→D→H→disk)",
           nc_write_mbps, nc_read_mbps,
           (double)total_bytes / (double)nc_file_bytes,
           nc_file_bytes >> 20);
    printf("  %-24s | %10.1f | %10.1f | %8.2fx | %8zu\n",
           "NN  (GPU→comp→disk)",
           nn_write_mbps, nn_read_mbps,
           nn_ratio,
           nn_file_bytes >> 20);
    printf("\n");
    printf("  Write speedup (NN vs no-comp):           %.2fx\n",
           nc_write_ms / nn_write_ms);
    printf("  Space saving (NN):                       %.1f%%\n",
           100.0 * (1.0 - 1.0 / nn_ratio));
    printf("  Effective write BW (NN, uncomp-equiv):   %.1f MB/s\n",
           nn_write_mbps * nn_ratio);
    printf("\n");

    gpucompress_cleanup();
    return 0;
}
