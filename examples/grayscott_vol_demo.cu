/**
 * grayscott_vol_demo.cu
 *
 * Demonstrates the full GPUCompress pipeline with Gray-Scott reaction-diffusion:
 *   GPU simulation → NN compression → HDF5 via VOL → read-back & verify
 *
 * The V field is snapshotted at regular intervals and written through the
 * GPUCompress VOL connector.  The NN selects compression algorithm per chunk.
 * Each snapshot is read back and verified bitwise on GPU.
 *
 * Usage:
 *   ./grayscott_vol_demo [model.nnwt] [--L 128] [--steps 5000] [--plotgap 1000]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_grayscott.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"
#include "compression/compression_header.h"

/* ---- Timing -------------------------------------------------------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

/* ---- CLI parsing -------------------------------------------------- */
static void parse_args(int argc, char **argv,
                       const char **weights, int *L, int *steps, int *plotgap)
{
    *weights = NULL;
    *L       = 128;
    *steps   = 5000;
    *plotgap = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            *L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plotgap") == 0 && i + 1 < argc) {
            *plotgap = atoi(argv[++i]);
        } else if (argv[i][0] != '-' && *weights == NULL) {
            *weights = argv[i];
        }
    }
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *weights;
    int L, total_steps, plotgap;
    parse_args(argc, argv, &weights, &L, &total_steps, &plotgap);

    int n_snapshots = total_steps / plotgap;
    if (n_snapshots < 1) n_snapshots = 1;
    size_t n_elem  = (size_t)L * L * L;
    size_t nbytes  = n_elem * sizeof(float);
    int chunk_z    = L / 4;
    if (chunk_z < 1) chunk_z = 1;
    int n_chunks   = (L + chunk_z - 1) / chunk_z;  /* chunks along Z */

    printf("=== grayscott_vol_demo: Gray-Scott → NN Compression → HDF5 ===\n");
    printf("    Grid    : %d³ = %zu floats (%.1f MB)\n",
           L, n_elem, nbytes / (1024.0 * 1024.0));
    printf("    Chunks  : %d × %d × %d  (%d chunks along Z)\n",
           L, L, chunk_z, n_chunks);
    printf("    Steps   : %d  plotgap=%d  snapshots=%d\n",
           total_steps, plotgap, n_snapshots);
    printf("    Model   : %s\n\n",
           weights ? weights : "(none — NN will fall back to LZ4)");

    /* ---- 1. Init GPUCompress + VOL ---- */
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
    H5VL_gpucompress_set_trace(0);

    /* ---- 2. Create Gray-Scott simulation ---- */
    GrayScottSettings s = gray_scott_default_settings();
    s.L = L;
    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);

    /* ---- 3. Create HDF5 file with 3D chunked dataset for V ---- */
    const char *fname = "/tmp/grayscott_vol_demo.h5";
    hsize_t dims[3]  = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };

    hid_t fid   = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(3, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

    hid_t dset_v = H5Dcreate2(fid, "V", H5T_NATIVE_FLOAT, space,
                               H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* ---- Readback buffer ---- */
    float *d_readback = NULL;
    cudaMalloc(&d_readback, nbytes);

    int *d_err = NULL;
    cudaMalloc(&d_err, sizeof(int));

    int total_failures = 0;

    /* ---- 4. Snapshot loop ---- */
    for (int snap = 0; snap < n_snapshots; snap++) {
        gpucompress_grayscott_run(sim, plotgap);
        cudaDeviceSynchronize();

        int cur_step = (snap + 1) * plotgap;
        float *d_u = NULL, *d_v = NULL;
        gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

        printf("\n╔══ Snapshot %d/%d  (step %d) ═══════════════════════════════╗\n",
               snap + 1, n_snapshots, cur_step);

        /* --- Write V field --- */
        H5VL_gpucompress_reset_stats();
        double t0 = now_sec();
        herr_t rc = H5Dwrite(dset_v, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
        double dt = now_sec() - t0;

        int n_comp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
        if (rc < 0 || n_comp != n_chunks) total_failures++;
        printf("  Write : %s  %.3f s  %.1f MB/s  (%d chunks compressed)\n",
               (rc >= 0 && n_comp == n_chunks) ? "OK  " : "FAIL",
               dt, (nbytes / (1024.0 * 1024.0)) / dt, n_comp);

        /* --- Per-chunk algorithm report --- */
        {
            int tally[9][2] = {};   /* [algo_id][shuffle] */
            size_t total_comp = 0;
            int all_ok = 1;

            for (int c = 0; c < n_chunks; c++) {
                hsize_t off[3] = { 0, 0, (hsize_t)c * chunk_z };
                hsize_t csz = 0;
                H5Dget_chunk_storage_size(dset_v, off, &csz);
                total_comp += csz;

                void    *raw   = malloc(csz);
                uint32_t filt  = 0;
                size_t   bufsz = csz;
                H5Dread_chunk(dset_v, H5P_DEFAULT, off, &filt, raw, &bufsz);

                CompressionHeader *hdr = (CompressionHeader *)raw;
                if (!hdr->isValid()) { all_ok = 0; total_failures++; free(raw); continue; }
                if (hdr->hasQuantizationApplied()) {
                    printf("  *** FAIL chunk[%d]: quant set on lossless run!\n", c);
                    total_failures++;
                }
                uint8_t aid = hdr->getAlgorithmId();
                int shuf = hdr->hasShuffleApplied() ? 1 : 0;
                if (aid < 9) tally[aid][shuf]++;

                free(raw);
            }

            double raw_mb  = nbytes / (1024.0 * 1024.0);
            double comp_mb = total_comp / (1024.0 * 1024.0);
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
        }

        /* --- Read-back + bitwise verify --- */
        cudaMemset(d_readback, 0xCD, nbytes);

        H5VL_gpucompress_reset_stats();
        double t1 = now_sec();
        herr_t rrc = H5Dread(dset_v, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
        double rdt = now_sec() - t1;

        int n_decomp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, NULL, &n_decomp);
        if (rrc < 0 || n_decomp != n_chunks) total_failures++;
        printf("  Read  : %s  %.3f s  %.1f MB/s\n",
               (rrc >= 0 && n_decomp == n_chunks) ? "OK  " : "FAIL",
               rdt, (nbytes / (1024.0 * 1024.0)) / rdt);

        cudaMemset(d_err, 0, sizeof(int));
        unsigned int grid = ((unsigned int)n_elem + 255) / 256;
        verify_kernel<<<grid, 256>>>(d_v, d_readback, (unsigned int)n_elem, d_err);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_err, sizeof(int), cudaMemcpyDeviceToHost);

        printf("  Verify: %s  (%zu / %zu)\n",
               mismatches == 0 ? "PASS" : "FAIL",
               n_elem - mismatches, n_elem);
        if (mismatches) total_failures++;

        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }

    /* ---- 5. Cleanup ---- */
    printf("\n=== Overall: %d failure(s) ===\n", total_failures);
    printf("Output file: %s\n", fname);

    cudaFree(d_err);
    cudaFree(d_readback);
    H5Dclose(dset_v);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_grayscott_destroy(sim);
    gpucompress_cleanup();

    return total_failures ? 1 : 0;
}
