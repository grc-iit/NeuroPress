/**
 * nn_vol_demo.cu
 *
 * Demonstrates the GPUCompress VOL connector with neural-network algorithm
 * selection (ALGO_AUTO, lossless only) across five data patterns.
 *
 * Patterns tested:
 *   random    — uniform floats in [0,1)      (high entropy, no structure)
 *   gaussian  — Box-Muller normal, σ≈1       (different byte distribution)
 *   smooth    — ramp + low-freq sine         (scientific-data-like)
 *   sparse    — 5% nonzero, rest zeros       (BITCOMP target)
 *   repeating — 256-element tile, tiled out  (LZ4/ZSTD friendly)
 *
 * For each pattern:
 *   1. Generate 64 MB directly on GPU
 *   2. Write via VOL (ALGO_AUTO, lossless, 4 MB chunks)
 *   3. Inspect per-chunk header: algo / shuffle / ratio
 *   4. Read back into poisoned buffer + bitwise verify
 *
 * Usage:
 *   ./nn_vol_demo [path/to/model.nnwt]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"
#include "compression/compression_header.h"

/* ---- Dataset geometry -------------------------------------------- */
#define N_ELEM   1073741824ULL   /* 4 GB as float32: 1G × 4B          */
#define CHUNK     1048576   /*  4 MB per chunk:   1M × 4B          */
#define N_CHUNK  (N_ELEM / CHUNK)  /* = 16 chunks               */

/* ---- Timing -------------------------------------------------------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- GPU data-generation kernels ---------------------------------- */

/* Pattern 1: uniform random floats in [0,1) via xorshift32 */
__global__ void gen_random(float *d, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = seed ^ (i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    x ^= x >> 11; x *= 0xac4d1d39u; x ^= x >> 15;
    d[i] = (float)(x >> 8) * (1.0f / 16777216.0f);
}

/* Pattern 2: Box-Muller Gaussian, mean=0 σ≈1 */
__global__ void gen_gaussian(float *d, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = seed ^ (i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    float u1 = fmaxf((float)(x >> 8) * (1.0f / 16777216.0f), 1e-7f);
    x ^= x >> 11; x *= 0xac4d1d39u; x ^= x >> 15;
    float u2 = (float)(x >> 8) * (1.0f / 16777216.0f);
    d[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* Pattern 3: smooth ramp + low-frequency sine (scientific-data-like) */
__global__ void gen_smooth(float *d, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / (float)n;
    d[i] = t * 1000.0f + 50.0f * sinf(t * 32.0f * 3.14159265f);
}

/* Pattern 4: sparse — 5% nonzero uniform [0,1), rest 0 */
__global__ void gen_sparse(float *d, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = seed ^ (i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    x ^= x >> 11; x *= 0xac4d1d39u; x ^= x >> 15;
    d[i] = ((x % 20) == 0) ? (float)(x >> 8) * (1.0f / 16777216.0f) : 0.0f;
}

/* Pattern 5: 256-element tile tiled across the buffer (LZ4/ZSTD repetition) */
__global__ void gen_repeating(float *d, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tile_i = i % 256;
    unsigned int x = seed ^ (tile_i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    x ^= x >> 11; x *= 0xac4d1d39u; x ^= x >> 15;
    d[i] = (float)(x >> 8) * (1.0f / 16777216.0f);
}

/* ---- Bitwise verify kernel ---------------------------------------- */
__global__ void verify_kernel(const float *ref, const float *got,
                               unsigned int n, int *err) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ref[i] != got[i]) atomicAdd(err, 1);
}

/* ---- Algorithm name table ----------------------------------------- */
static const char *algo_name(uint8_t id) {
    static const char *names[] = {
        "AUTO", "LZ4", "SNAPPY", "DEFLATE", "GDEFLATE",
        "ZSTD", "ANS", "CASCADED", "BITCOMP"
    };
    return (id < 9) ? names[id] : "?";
}

/* ---- Single-pattern run ------------------------------------------- */
static int run_pattern(const char *label, float *d_src, float *d_dst,
                       hid_t fapl)
{
    char fname[128];
    snprintf(fname, sizeof(fname), "/tmp/nn_vol_demo_%s.h5", label);

    int failures = 0;
    printf("\n╔══ Pattern: %-10s ═══════════════════════════════════════╗\n", label);

    /* --- Write --- */
    {
        hsize_t dims[1]  = { N_ELEM };
        hsize_t cdims[1] = { CHUNK  };

        hid_t fid  = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5VL_gpucompress_reset_stats();
        double t0 = now_sec();
        herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL, H5P_DEFAULT, d_src);
        double dt = now_sec() - t0;

        int n_comp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
        if (rc < 0 || n_comp != N_CHUNK) failures++;
        printf("  Write : %s  %.3f s  %.1f MB/s\n",
               (rc >= 0 && n_comp == N_CHUNK) ? "OK  " : "FAIL",
               dt, (N_ELEM * 4.0 / (1 << 20)) / dt);

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* --- Per-chunk algorithm report --- */
    {
        hid_t fid  = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);

        /* tally how many chunks used each (algo, shuffle) combo */
        int tally[9][2] = {};   /* [algo_id][shuffle] */
        size_t total_comp = 0;
        int all_ok = 1;

        for (int c = 0; c < N_CHUNK; c++) {
            hsize_t off[1] = { (hsize_t)c * CHUNK };
            hsize_t csz = 0;
            H5Dget_chunk_storage_size(dset, off, &csz);
            total_comp += csz;

            void    *raw   = malloc(csz);
            uint32_t filt  = 0;
            size_t   bufsz = csz;
            H5Dread_chunk(dset, H5P_DEFAULT, off, &filt, raw, &bufsz);

            CompressionHeader *hdr = (CompressionHeader *)raw;
            if (!hdr->isValid()) { all_ok = 0; failures++; free(raw); continue; }
            if (hdr->hasQuantizationApplied()) {
                printf("  *** FAIL chunk[%d]: quant set on lossless run!\n", c);
                failures++;
            }
            uint8_t aid = hdr->getAlgorithmId();
            int shuf = hdr->hasShuffleApplied() ? 1 : 0;
            if (aid < 9) tally[aid][shuf]++;

            free(raw);
        }

        double raw_mb  = N_ELEM * 4.0 / (1 << 20);
        double comp_mb = total_comp / (1.0 * (1 << 20));

        printf("  Ratio : %.2fx  (%.1f MB → %.3f MB)  hdr=%s\n",
               raw_mb / comp_mb, raw_mb, comp_mb, all_ok ? "OK" : "FAIL");
        printf("  Algo  :");
        for (int a = 1; a < 9; a++) {
            int n0 = tally[a][0], n1 = tally[a][1];
            if (n0 + n1 == 0) continue;
            if (n1) printf("  %s+shuf×%d", algo_name(a), n1);
            if (n0) printf("  %s×%d",       algo_name(a), n0);
        }
        printf("\n");

        H5Dclose(dset); H5Fclose(fid);
    }

    /* --- Read + bitwise verify --- */
    {
        cudaMemset(d_dst, 0xCD, N_ELEM * sizeof(float));

        hid_t fid  = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);

        H5VL_gpucompress_reset_stats();
        double t0 = now_sec();
        herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, d_dst);
        double dt = now_sec() - t0;

        int n_decomp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, NULL, &n_decomp);
        if (rc < 0 || n_decomp != N_CHUNK) failures++;
        printf("  Read  : %s  %.3f s  %.1f MB/s\n",
               (rc >= 0 && n_decomp == N_CHUNK) ? "OK  " : "FAIL",
               dt, (N_ELEM * 4.0 / (1 << 20)) / dt);

        H5Dclose(dset); H5Fclose(fid);

        int *d_err;
        cudaMalloc(&d_err, sizeof(int));
        cudaMemset(d_err, 0, sizeof(int));
        verify_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, d_dst, N_ELEM, d_err);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_err, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_err);

        printf("  Verify: %s  (%d / %d)\n",
               mismatches == 0 ? "PASS" : "FAIL",
               N_ELEM - mismatches, N_ELEM);
        if (mismatches) failures++;
    }

    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    return failures;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *weights = (argc > 1) ? argv[1] : NULL;

    printf("=== nn_vol_demo: NN algorithm selection across data patterns ===\n");
    printf("    Dataset : %d floats = %.0f MB per pattern\n",
           N_ELEM, N_ELEM * 4.0 / (1 << 20));
    printf("    Chunks  : %d × %.0f MB\n", N_CHUNK, CHUNK * 4.0 / (1 << 20));
    printf("    Model   : %s\n\n",
           weights ? weights : "(none — NN will fall back to LZ4)");

    gpucompress_error_t gerr = gpucompress_init(weights);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %d\n", gerr);
        return 1;
    }
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    printf("NN model loaded: %s\n",
           gpucompress_nn_is_loaded() ? "yes" : "no (fallback to default)");

    H5Z_gpucompress_register();
    hid_t vol_id    = H5VL_gpucompress_register();
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    H5VL_gpucompress_set_trace(0);   /* no per-chunk trace noise */

    float *d_src = NULL, *d_dst = NULL;
    cudaMalloc(&d_src, N_ELEM * sizeof(float));
    cudaMalloc(&d_dst, N_ELEM * sizeof(float));

    unsigned int seed = (unsigned int)time(NULL);
    int total_failures = 0;
    int blocks = (N_ELEM + 255) / 256;

    /* ---- Pattern 1: random uniform ---- */
    gen_random<<<blocks, 256>>>(d_src, N_ELEM, seed);
    cudaDeviceSynchronize();
    total_failures += run_pattern("random", d_src, d_dst, fapl);

    /* ---- Pattern 2: Gaussian ---- */
    gen_gaussian<<<blocks, 256>>>(d_src, N_ELEM, seed + 1);
    cudaDeviceSynchronize();
    total_failures += run_pattern("gaussian", d_src, d_dst, fapl);

    /* ---- Pattern 3: smooth ramp + sine ---- */
    gen_smooth<<<blocks, 256>>>(d_src, N_ELEM);
    cudaDeviceSynchronize();
    total_failures += run_pattern("smooth", d_src, d_dst, fapl);

    /* ---- Pattern 4: sparse (5% nonzero) ---- */
    gen_sparse<<<blocks, 256>>>(d_src, N_ELEM, seed + 2);
    cudaDeviceSynchronize();
    total_failures += run_pattern("sparse", d_src, d_dst, fapl);

    /* ---- Pattern 5: repeating 256-element tile ---- */
    gen_repeating<<<blocks, 256>>>(d_src, N_ELEM, seed + 3);
    cudaDeviceSynchronize();
    total_failures += run_pattern("repeating", d_src, d_dst, fapl);

    printf("\n=== Overall: %d failure(s) ===\n", total_failures);

    cudaFree(d_src);
    cudaFree(d_dst);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return total_failures ? 1 : 0;
}
