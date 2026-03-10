/**
 * @file test_vol_xfer_audit.cu
 * @brief Transfer audit sample: GPU-origin data → HDF5 VOL → NN compress
 *
 * Allocates data directly on the GPU, writes it through the GPUCompress
 * HDF5 VOL connector with ALGO_AUTO (NN-based algorithm selection),
 * reads it back to GPU memory, verifies correctness, and prints the
 * full transfer tracker summary.
 *
 * Usage:
 *   ./test_vol_xfer_audit [weights.nnwt]
 *
 * If no weights path is given, uses neural_net/weights/model.nnwt.
 * The transfer tracker prints per-call detail + summary to stderr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <hdf5.h>
#include <cuda_runtime.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "xfer_tracker.h"

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define NUM_FLOATS  (1024 * 1024)   /* 1M floats = 4 MB */
#define DATA_BYTES  (NUM_FLOATS * sizeof(float))
#define HDF5_FILE   "/tmp/test_vol_xfer_audit.h5"

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* GPU kernel: fill buffer with a noisy-ramp pattern (no host copy needed) */
__global__ void fill_gpu_data(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = (float)idx / (float)n;
        /* Deterministic "noisy" ramp: base ramp + high-frequency ripple */
        out[idx] = x + 0.05f * sinf(x * 1000.0f);
    }
}

/* GPU kernel: bitwise compare two buffers, write mismatch count */
__global__ void compare_gpu(const float* a, const float* b, size_t n,
                            int* mismatch_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        /* Bitwise comparison — lossless round-trip expected */
        unsigned int va, vb;
        memcpy(&va, &a[idx], sizeof(unsigned int));
        memcpy(&vb, &b[idx], sizeof(unsigned int));
        if (va != vb) atomicAdd(mismatch_count, 1);
    }
}

/* Open/create a VOL-enabled HDF5 file */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) {
        fprintf(stderr, "H5VL_gpucompress_register() failed\n");
        return H5I_INVALID_HID;
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL) < 0) {
        fprintf(stderr, "H5Pset_fapl_gpucompress() failed\n");
        H5Pclose(fapl);
        H5VLclose(vol_id);
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

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";
    if (argc <= 1) { FILE* f = fopen(weights, "rb"); if (f) fclose(f); else weights = "../neural_net/weights/model.nnwt"; }

    printf("==============================================\n");
    printf(" GPUCompress VOL Transfer Audit\n");
    printf("==============================================\n");
    printf("  Data size : %zu floats (%zu bytes)\n", (size_t)NUM_FLOATS, (size_t)DATA_BYTES);
    printf("  Weights   : %s\n", weights);
    printf("  HDF5 file : %s\n", HDF5_FILE);
    printf("==============================================\n\n");

    /* ---- 1. Init library with NN weights ---- */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n",
                gpucompress_error_string(rc));
        return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "NN weights not loaded — ALGO_AUTO won't use NN\n");
        gpucompress_cleanup();
        return 1;
    }
    printf("[OK] Library initialized, NN weights loaded\n");

    /* ---- 2. Allocate and fill data directly on GPU ---- */
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_BYTES));

    int threads = 256;
    int blocks  = (NUM_FLOATS + threads - 1) / threads;
    fill_gpu_data<<<blocks, threads>>>(d_data, NUM_FLOATS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[OK] Data generated on GPU (%d blocks x %d threads)\n", blocks, threads);

    /* ---- 3. Reset transfer tracker (ignore setup copies) ---- */
    xfer_tracker_reset();
    H5VL_gpucompress_reset_stats();
    printf("\n--- Transfer tracker reset ---\n\n");

    /* ---- 4. Write GPU data through VOL with ALGO_AUTO ---- */
    printf("[WRITE] GPU pointer → HDF5 via VOL (ALGO_AUTO, lossless) ...\n");
    {
        hid_t fid = open_vol_file(HDF5_FILE, H5F_ACC_TRUNC);
        if (fid < 0) { fprintf(stderr, "open_vol_file(write) failed\n"); return 1; }

        hsize_t dims[1]  = { NUM_FLOATS };
        hsize_t chunk[1] = { NUM_FLOATS };  /* single chunk = full dataset */
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        /* Set gpucompress filter: ALGO_AUTO, no preprocessing, lossless */
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = GPUCOMPRESS_ALGO_AUTO;
        cd[1] = 0;  /* no preprocessing */
        cd[2] = 0;
        pack_double(0.0, &cd[3], &cd[4]);  /* error_bound = 0 (lossless) */
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                      H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t dset = H5Dcreate2(fid, "gpu_data", H5T_NATIVE_FLOAT,
                                space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "H5Dcreate2 failed\n");
            H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            return 1;
        }

        /* H5Dwrite with a GPU pointer — VOL intercepts and compresses on GPU */
        herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_data);
        if (wr < 0) {
            fprintf(stderr, "H5Dwrite(GPU ptr) failed\n");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            return 1;
        }

        H5Dclose(dset);
        H5Sclose(space);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    printf("[OK] Write complete\n");

    /* Print transfer snapshot after write */
    printf("\n--- Transfers after WRITE ---\n");
    xfer_tracker_dump();

    /* ---- 5. Read back to a fresh GPU buffer through VOL ---- */
    float* d_readback = nullptr;
    CUDA_CHECK(cudaMalloc(&d_readback, DATA_BYTES));

    printf("\n[READ] HDF5 → GPU pointer via VOL (decompress on GPU) ...\n");
    {
        hid_t fid = open_vol_file(HDF5_FILE, H5F_ACC_RDONLY);
        if (fid < 0) { fprintf(stderr, "open_vol_file(read) failed\n"); return 1; }

        hid_t dset = H5Dopen2(fid, "gpu_data", H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "H5Dopen2 failed\n");
            H5Fclose(fid);
            return 1;
        }

        herr_t rd = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, d_readback);
        if (rd < 0) {
            fprintf(stderr, "H5Dread(GPU ptr) failed\n");
            H5Dclose(dset); H5Fclose(fid);
            return 1;
        }

        H5Dclose(dset);
        H5Fclose(fid);
    }
    printf("[OK] Read complete\n");

    /* ---- 6. Verify on GPU (no D→H for verification!) ---- */
    int* d_mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mismatches, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(int)));

    compare_gpu<<<blocks, threads>>>(d_data, d_readback, NUM_FLOATS, d_mismatches);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_mismatches = 0;
    cudaMemcpy(&h_mismatches, d_mismatches, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_mismatches == 0) {
        printf("[OK] Lossless round-trip verified on GPU (0 mismatches)\n");
    } else {
        printf("[FAIL] %d / %zu float mismatches!\n", h_mismatches, (size_t)NUM_FLOATS);
    }

    /* ---- 7. Print full transfer summary ---- */
    printf("\n==============================================\n");
    printf(" FULL TRANSFER AUDIT (write + read)\n");
    printf("==============================================\n");
    xfer_tracker_dump();

    /* VOL-level stats */
    int vol_writes = 0, vol_reads = 0, vol_comp = 0, vol_decomp = 0;
    H5VL_gpucompress_get_stats(&vol_writes, &vol_reads, &vol_comp, &vol_decomp);
    printf("\nVOL activity:\n");
    printf("  GPU writes entered : %d\n", vol_writes);
    printf("  GPU reads entered  : %d\n", vol_reads);
    printf("  Chunks compressed  : %d\n", vol_comp);
    printf("  Chunks decompressed: %d\n", vol_decomp);

    int h2d_n = 0, d2h_n = 0, d2d_n = 0;
    size_t h2d_b = 0, d2h_b = 0, d2d_b = 0;
    H5VL_gpucompress_get_transfer_stats(&h2d_n, &h2d_b, &d2h_n, &d2h_b,
                                         &d2d_n, &d2d_b);
    printf("\nVOL-layer transfers (outside compress/decompress):\n");
    printf("  H->D: %d calls, %zu bytes\n", h2d_n, h2d_b);
    printf("  D->H: %d calls, %zu bytes\n", d2h_n, d2h_b);
    printf("  D->D: %d calls, %zu bytes\n", d2d_n, d2d_b);

    /* ---- 8. Cleanup ---- */
    cudaFree(d_data);
    cudaFree(d_readback);
    cudaFree(d_mismatches);
    gpucompress_cleanup();
    remove(HDF5_FILE);

    printf("\n==============================================\n");
    printf(" %s\n", h_mismatches == 0 ? "PASSED" : "FAILED");
    printf("==============================================\n");
    return h_mismatches == 0 ? 0 : 1;
}
