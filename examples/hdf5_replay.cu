/**
 * hdf5_replay.cu
 *
 * Reads every dataset from an existing HDF5 file, uploads each to the GPU,
 * then writes it through the GPUCompress VOL connector step_size bytes at a
 * time — simulating how a live simulation would stream data out.
 *
 * Each write step corresponds to exactly one HDF5 chunk, so step_size
 * controls both the write granularity and the compression unit.
 *
 * Usage:
 *   ./hdf5_replay <input.h5> [options]
 *   --step-mb N     Step size in MB per write (default: 4)
 *   --weights PATH  NN weights file (default: neural_net/weights/model.nnwt)
 *   --out PATH      Output HDF5 file (default: /tmp/replay_out.h5)
 *   --eb FLOAT      Error bound for lossy compression (default: 0.0 = lossless)
 *
 * Example:
 *   ./hdf5_replay simulation.h5 --step-mb 8 --weights model.nnwt --eb 1e-4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Defaults
 * ============================================================ */
#define DEFAULT_STEP_MB  4
#define DEFAULT_WEIGHTS  "neural_net/weights/model.nnwt"
#define DEFAULT_OUT      "/tmp/replay_out.h5"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ============================================================
 * Globals set from command line
 * ============================================================ */
static double g_error_bound = 0.0;

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
 * HDF5 helpers
 * ============================================================ */
static void pack_double(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t set_gpucompress_filter(hid_t dcpl, double error_bound)
{
    unsigned cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0;    /* ALGO_AUTO — let NN choose */
    cd[1] = 0x02; /* byte-shuffle preprocessing */
    cd[2] = 4;    /* shuffle element size */
    pack_double(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* Open a file through the GPUCompress VOL connector. */
static hid_t open_vol_file(const char *path, unsigned flags)
{
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

/* ============================================================
 * Per-dataset replay
 * ============================================================ */

/**
 * Read `name` from `in_file`, upload to GPU, then write to `out_file`
 * via the GPUCompress VOL in steps of `step_bytes` bytes.
 *
 * Returns the number of write steps performed, or -1 on error.
 */
static int replay_dataset(hid_t in_file, hid_t out_file,
                           const char *name, size_t step_bytes,
                           double *out_write_ms, size_t *out_bytes)
{
    /* --- Open source dataset --- */
    hid_t in_dset = H5Dopen2(in_file, name, H5P_DEFAULT);
    if (in_dset < 0) {
        fprintf(stderr, "  ERROR: cannot open dataset '%s'\n", name);
        return -1;
    }

    hid_t space = H5Dget_space(in_dset);
    hid_t dtype = H5Dget_type(in_dset);

    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[H5S_MAX_RANK];
    H5Sget_simple_extent_dims(space, dims, NULL);

    hsize_t total_elems = 1;
    for (int i = 0; i < ndims; i++) total_elems *= dims[i];

    size_t elem_size  = H5Tget_size(dtype);
    size_t total_bytes = total_elems * elem_size;

    /* Clamp step to dataset size */
    if (step_bytes > total_bytes) step_bytes = total_bytes;

    hsize_t step_elems  = step_bytes / elem_size;
    if (step_elems == 0) step_elems = 1;
    /* Re-align step_bytes to a whole number of elements */
    step_bytes = step_elems * elem_size;

    /* Compute number of steps */
    int n_steps = (int)((total_elems + step_elems - 1) / step_elems);

    printf("  '%s'  [", name);
    for (int i = 0; i < ndims; i++) {
        printf("%llu", (unsigned long long)dims[i]);
        if (i + 1 < ndims) printf("×");
    }
    printf("]  %.2f MB  →  %d step(s) of %.2f MB\n",
           total_bytes / (1024.0 * 1024.0),
           n_steps,
           step_bytes  / (1024.0 * 1024.0));

    /* --- Read all data to host --- */
    void *h_data = malloc(total_bytes);
    if (!h_data) {
        fprintf(stderr, "  ERROR: out of host memory (%.2f MB) for '%s'\n",
                total_bytes / (1024.0 * 1024.0), name);
        H5Tclose(dtype); H5Sclose(space); H5Dclose(in_dset);
        return -1;
    }

    herr_t rc = H5Dread(in_dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_data);
    H5Dclose(in_dset);
    if (rc < 0) {
        fprintf(stderr, "  ERROR: H5Dread failed for '%s'\n", name);
        free(h_data);
        H5Tclose(dtype); H5Sclose(space);
        return -1;
    }

    /* --- Upload to GPU --- */
    void *d_data = NULL;
    cudaError_t ce = cudaMalloc(&d_data, total_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "  ERROR: cudaMalloc failed (%.2f MB): %s\n",
                total_bytes / (1024.0 * 1024.0), cudaGetErrorString(ce));
        free(h_data);
        H5Tclose(dtype); H5Sclose(space);
        return -1;
    }
    cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice);
    free(h_data); /* host copy no longer needed */

    /* --- Create output dataset (1D, chunked at step_elems) --- */
    hsize_t out_dims[1]   = { total_elems };
    hsize_t chunk_dims[1] = { step_elems  };

    hid_t out_space = H5Screate_simple(1, out_dims, NULL);
    hid_t dcpl      = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    set_gpucompress_filter(dcpl, g_error_bound);

    hid_t out_dset = H5Dcreate2(out_file, name, dtype, out_space,
                                  H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(out_space);

    if (out_dset < 0) {
        fprintf(stderr, "  ERROR: H5Dcreate2 failed for '%s'\n", name);
        cudaFree(d_data);
        H5Tclose(dtype); H5Sclose(space);
        return -1;
    }

    /* --- Replay: write step by step --- */
    hsize_t offset    = 0;
    double  write_ms  = 0.0;
    int     steps_ok  = 0;

    while (offset < total_elems) {
        hsize_t count = step_elems;
        if (offset + count > total_elems)
            count = total_elems - offset;

        /* Memory dataspace: a contiguous slice of `count` elements */
        hid_t mem_space  = H5Screate_simple(1, &count, NULL);

        /* File dataspace: select the corresponding hyperslab */
        hid_t file_space = H5Dget_space(out_dset);
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET,
                             &offset, NULL, &count, NULL);

        /* GPU pointer for this slice */
        char *d_slice = (char *)d_data + offset * elem_size;

        double t0 = now_ms();
        rc = H5Dwrite(out_dset, dtype, mem_space, file_space,
                       H5P_DEFAULT, d_slice);
        write_ms += now_ms() - t0;

        H5Sclose(mem_space);
        H5Sclose(file_space);

        if (rc < 0) {
            fprintf(stderr, "  ERROR: H5Dwrite failed at step %d (offset %llu)\n",
                    steps_ok, (unsigned long long)offset);
            break;
        }

        offset += count;
        steps_ok++;
    }

    H5Dclose(out_dset);
    cudaFree(d_data);
    H5Tclose(dtype);
    H5Sclose(space);

    printf("    → %d step(s) written in %.1f ms  (%.0f MB/s)\n",
           steps_ok, write_ms,
           (total_bytes / (1024.0 * 1024.0)) / (write_ms / 1000.0));

    if (out_write_ms) *out_write_ms += write_ms;
    if (out_bytes)    *out_bytes    += total_bytes;

    return steps_ok;
}

/* ============================================================
 * H5Ovisit callback — invoked for every object in the source file
 * ============================================================ */
typedef struct {
    hid_t   in_file;
    hid_t   out_file;
    size_t  step_bytes;
    int     n_datasets;
    int     n_steps;
    double  total_write_ms;
    size_t  total_bytes;
} VisitCtx;

static herr_t visit_cb(hid_t          loc_id,
                        const char    *name,
                        const H5O_info2_t *info,
                        void          *op_data)
{
    if (info->type != H5O_TYPE_DATASET)
        return 0; /* skip groups and named types */

    VisitCtx *ctx = (VisitCtx *)op_data;

    double  write_ms = 0.0;
    size_t  nbytes   = 0;
    int     steps = replay_dataset(ctx->in_file, ctx->out_file, name,
                                    ctx->step_bytes, &write_ms, &nbytes);
    if (steps > 0) {
        ctx->n_datasets++;
        ctx->n_steps        += steps;
        ctx->total_write_ms += write_ms;
        ctx->total_bytes    += nbytes;
    }

    return 0;
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <input.h5> [options]\n"
                "  --step-mb N     Step size in MB per write (default: %d)\n"
                "  --weights PATH  NN weights file (default: %s)\n"
                "  --out PATH      Output HDF5 file  (default: %s)\n"
                "  --eb FLOAT      Error bound, 0 = lossless (default: 0.0)\n",
                argv[0], DEFAULT_STEP_MB, DEFAULT_WEIGHTS, DEFAULT_OUT);
        return 1;
    }

    const char *input_path = argv[1];
    const char *weights    = DEFAULT_WEIGHTS;
    const char *out_path   = DEFAULT_OUT;
    int         step_mb    = DEFAULT_STEP_MB;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--step-mb") && i + 1 < argc)
            step_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc)
            weights = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else if (!strcmp(argv[i], "--eb") && i + 1 < argc)
            g_error_bound = atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    size_t step_bytes = (size_t)step_mb * 1024 * 1024;

    printf("=== HDF5 Replay ===\n");
    printf("  Input:       %s\n", input_path);
    printf("  Output:      %s\n", out_path);
    printf("  Step size:   %d MB\n", step_mb);
    printf("  Weights:     %s\n", weights);
    printf("  Error bound: %g%s\n\n", g_error_bound,
           g_error_bound == 0.0 ? " (lossless)" : "");

    /* Check CUDA */
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "FATAL: no CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s (%.0f MB)\n\n", prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    /* Init gpucompress */
    gpucompress_error_t gerr = gpucompress_init(weights);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "WARNING: could not load NN weights (%s) — using LZ4 fallback\n",
                gpucompress_error_string(gerr));
        gerr = gpucompress_init(NULL);
        if (gerr != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "FATAL: gpucompress_init failed\n");
            return 1;
        }
    }
    printf("  NN loaded: %s\n\n", gpucompress_nn_is_loaded() ? "yes" : "no");

    /* Suppress HDF5 error stack (we handle errors manually) */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    /* Open input file with native HDF5 (no VOL — just read raw data) */
    hid_t in_file = H5Fopen(input_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_file < 0) {
        fprintf(stderr, "FATAL: cannot open '%s'\n", input_path);
        gpucompress_cleanup();
        return 1;
    }

    /* Create output file through the GPUCompress VOL connector */
    hid_t out_file = open_vol_file(out_path, H5F_ACC_TRUNC);
    if (out_file < 0) {
        fprintf(stderr, "FATAL: cannot create output file '%s'\n", out_path);
        H5Fclose(in_file);
        gpucompress_cleanup();
        return 1;
    }

    gpucompress_reset_chunk_history();

    /* Walk every dataset in the input file and replay it */
    VisitCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.in_file    = in_file;
    ctx.out_file   = out_file;
    ctx.step_bytes = step_bytes;

    double t_wall_start = now_ms();
    H5Ovisit3(in_file, H5_INDEX_NAME, H5_ITER_NATIVE,
               visit_cb, &ctx, H5O_INFO_BASIC);
    double t_wall = now_ms() - t_wall_start;

    H5Fclose(out_file);
    H5Fclose(in_file);

    /* ---- Summary ---- */
    printf("\n=== Summary ===\n");
    printf("  Datasets replayed: %d\n", ctx.n_datasets);
    printf("  Total steps:       %d\n", ctx.n_steps);
    printf("  Data written:      %.2f MB\n",
           ctx.total_bytes / (1024.0 * 1024.0));
    if (ctx.total_write_ms > 0.0)
        printf("  Write throughput:  %.0f MB/s\n",
               (ctx.total_bytes / (1024.0 * 1024.0)) / (ctx.total_write_ms / 1000.0));
    printf("  Wall time:         %.1f ms\n", t_wall);

    /* ---- Per-chunk NN diagnostics ---- */
    int n_chunks = gpucompress_get_chunk_history_count();
    if (n_chunks > 0) {
        int show = n_chunks < 20 ? n_chunks : 20;
        printf("\n  Chunk diagnostics (%d total, showing first %d):\n",
               n_chunks, show);
        for (int i = 0; i < show; i++) {
            gpucompress_chunk_diag_t diag;
            if (gpucompress_get_chunk_diag(i, &diag) == 0) {
                printf("    [%2d] action=%-2d  ratio=%5.2f  comp=%5.1f ms%s%s\n",
                       i, diag.nn_action, diag.actual_ratio,
                       diag.compression_ms,
                       diag.exploration_triggered ? "  EXPLORE" : "",
                       diag.sgd_fired             ? "  SGD"     : "");
            }
        }
        if (n_chunks > 20)
            printf("    ... and %d more\n", n_chunks - 20);
    }

    gpucompress_cleanup();
    printf("\nDone.\n");
    return 0;
}
