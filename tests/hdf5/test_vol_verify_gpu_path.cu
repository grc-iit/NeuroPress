/**
 * @file test_vol_verify_gpu_path.cu
 * @brief Verification experiment: GPU-resident data → VOL → gpucompress → disk → readback
 *
 * This test verifies the complete HDF5 VOL pipeline:
 *
 *   Experiment 1: GPU pointer detection
 *     - Confirms cudaMalloc'd pointers are recognized as device pointers
 *     - Confirms host pointers are NOT recognized as device pointers
 *
 *   Experiment 2: GPU-aware write path activation
 *     - Writes GPU data through VOL, checks VOL activity counters prove
 *       the GPU-aware path was taken (not the D→H fallback)
 *
 *   Experiment 3: Compression actually happened
 *     - Reads raw chunk bytes from HDF5 via native driver
 *     - Validates CompressionHeader magic (0x43555047), version, sizes
 *     - Confirms compressed size < original size
 *
 *   Experiment 4: Per-chunk NN diagnostics
 *     - Uses ALGO_AUTO with multiple data patterns across chunks
 *     - Verifies per-chunk diagnostics are recorded (algo, ratio, timing)
 *
 *   Experiment 5: Bitwise lossless round-trip
 *     - Reads back to a fresh GPU buffer, compares on-GPU (no host copy)
 *     - Zero mismatches = lossless path is correct
 *
 *   Experiment 6: Transfer accounting
 *     - Verifies D→H transfers are only compressed bytes (not raw data)
 *     - No unexpected H→D transfers during write
 *
 *   Experiment 7: Multi-algorithm correctness
 *     - Repeats write/read with LZ4, Zstd, Snappy explicitly
 *     - Bitwise verification for each
 *
 * Usage:
 *   ./test_vol_verify_gpu_path [weights.nnwt]
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
#include "compression/compression_header.h"

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* 8 chunks × 256K floats = 8 MB total — small enough for any GPU */
#define CHUNK_FLOATS  (256 * 1024)        /* 256K floats = 1 MB per chunk */
#define NUM_CHUNKS    8
#define TOTAL_FLOATS  (CHUNK_FLOATS * NUM_CHUNKS)
#define DATA_BYTES    (TOTAL_FLOATS * sizeof(float))
#define HDF5_FILE     "/tmp/test_vol_verify_gpu_path.h5"

static const char* ALGO_NAMES[] = {
    "AUTO", "LZ4", "Snappy", "Deflate", "GDeflate", "Zstd", "ANS", "Cascaded", "Bitcomp"
};

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

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) do {                                     \
    if (cond) {                                                        \
        printf("  [PASS] " fmt "\n", ##__VA_ARGS__);                   \
        g_pass++;                                                      \
    } else {                                                           \
        printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);                   \
        g_fail++;                                                      \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/* GPU kernels                                                          */
/* ------------------------------------------------------------------ */

/* Fill each chunk with a distinct pattern so NN picks different algos */
__global__ void fill_multi_pattern(float* out, size_t chunk_floats, int n_chunks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = chunk_floats * n_chunks;
    if (idx >= total) return;

    int chunk_id = (int)(idx / chunk_floats);
    size_t local  = idx % chunk_floats;
    float x = (float)local / (float)chunk_floats;

    switch (chunk_id % 4) {
        case 0: /* smooth ramp — highly compressible */
            out[idx] = x;
            break;
        case 1: /* high-frequency sine — moderate compression */
            out[idx] = sinf(x * 500.0f) * 0.5f + 0.5f;
            break;
        case 2: /* pseudo-random via xorshift — hard to compress */
        {
            unsigned int s = (unsigned int)(idx * 2654435761u);
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float r;
            unsigned int masked = s & 0x007FFFFFu;
            unsigned int bits   = 0x3F800000u | masked; /* [1.0, 2.0) */
            memcpy(&r, &bits, sizeof(float));
            out[idx] = r - 1.0f; /* [0.0, 1.0) */
            break;
        }
        case 3: /* constant — maximally compressible */
            out[idx] = 42.0f;
            break;
    }
}

/* Bitwise compare on GPU */
__global__ void compare_kernel(const float* a, const float* b, size_t n,
                                unsigned long long* mismatch_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int va, vb;
    memcpy(&va, &a[idx], sizeof(unsigned int));
    memcpy(&vb, &b[idx], sizeof(unsigned int));
    if (va != vb) atomicAdd(mismatch_count, 1ULL);
}

/* ------------------------------------------------------------------ */
/* VOL file helpers                                                     */
/* ------------------------------------------------------------------ */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);

    hid_t fid;
    if (flags & H5F_ACC_TRUNC)
        fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    else
        fid = H5Fopen(path, H5F_ACC_RDONLY, fapl);

    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

/* Write GPU data to HDF5 with a given algorithm */
static int write_gpu_data(const char* filepath, const char* dset_name,
                           float* d_data, size_t n_floats, size_t chunk_floats,
                           int algo) {
    hid_t fid = open_vol_file(filepath, H5F_ACC_TRUNC);
    if (fid < 0) return -1;

    hsize_t dims[1]  = { n_floats };
    hsize_t chunk[1] = { chunk_floats };
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo;
    cd[1] = 0;  /* no preprocessing */
    cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);  /* lossless */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, dset_name, H5T_NATIVE_FLOAT,
                            space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
        return -1;
    }

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_data);

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);
    return (rc < 0) ? -1 : 0;
}

/* Read back to GPU buffer */
static int read_gpu_data(const char* filepath, const char* dset_name,
                          float* d_out, size_t n_floats) {
    hid_t fid = open_vol_file(filepath, H5F_ACC_RDONLY);
    if (fid < 0) return -1;

    hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_out);

    H5Dclose(dset);
    H5Fclose(fid);
    return (rc < 0) ? -1 : 0;
}

/* GPU bitwise verification — returns mismatch count */
static unsigned long long verify_gpu(float* d_a, float* d_b, size_t n) {
    unsigned long long* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));

    int threads = 256;
    int blocks  = ((int)n + threads - 1) / threads;
    compare_kernel<<<blocks, threads>>>(d_a, d_b, n, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    return h_count;
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";
    if (argc <= 1) {
        FILE* f = fopen(weights, "rb");
        if (f) fclose(f);
        else weights = "../neural_net/weights/model.nnwt";
    }

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    printf("================================================================\n");
    printf("  VOL GPU-Path Verification Experiment\n");
    printf("================================================================\n");
    printf("  Data     : %d chunks x %d floats = %zu bytes (%.1f MB)\n",
           NUM_CHUNKS, CHUNK_FLOATS, (size_t)DATA_BYTES,
           (double)DATA_BYTES / (1 << 20));
    printf("  Weights  : %s\n", weights);
    printf("  HDF5     : %s\n", HDF5_FILE);
    printf("================================================================\n\n");

    /* ---- Init library ---- */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(rc));
        return 1;
    }

    /* ---- Allocate GPU buffers ---- */
    float *d_data = nullptr, *d_readback = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data,     DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_readback, DATA_BYTES));

    /* Fill with multi-pattern data */
    int threads = 256;
    int blocks  = ((int)TOTAL_FLOATS + threads - 1) / threads;
    fill_multi_pattern<<<blocks, threads>>>(d_data, CHUNK_FLOATS, NUM_CHUNKS);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ============================================================
     * Experiment 1: GPU pointer detection
     * ============================================================ */
    printf("── Experiment 1: GPU Pointer Detection ──────────────────────\n");
    {
        float* h_buf = (float*)malloc(1024);
        int gpu_ok  = gpucompress_is_device_ptr(d_data);
        int host_ok = !gpucompress_is_device_ptr(h_buf);
        int null_ok = !gpucompress_is_device_ptr(NULL);
        CHECK(gpu_ok,  "cudaMalloc'd pointer detected as device");
        CHECK(host_ok, "malloc'd pointer NOT detected as device");
        CHECK(null_ok, "NULL pointer NOT detected as device");
        free(h_buf);
    }
    printf("\n");

    /* ============================================================
     * Experiment 2: GPU-aware write path activation
     * ============================================================ */
    printf("── Experiment 2: GPU-Aware Write Path ───────────────────────\n");
    {
        H5VL_gpucompress_reset_stats();
        gpucompress_reset_chunk_history();
        gpucompress_disable_online_learning();
        gpucompress_set_exploration(0);

        int wr = write_gpu_data(HDF5_FILE, "data", d_data,
                                TOTAL_FLOATS, CHUNK_FLOATS,
                                GPUCOMPRESS_ALGO_AUTO);
        CHECK(wr == 0, "H5Dwrite with GPU pointer succeeded");

        int vol_writes = 0, vol_reads = 0, vol_comp = 0, vol_decomp = 0;
        H5VL_gpucompress_get_stats(&vol_writes, &vol_reads, &vol_comp, &vol_decomp);

        CHECK(vol_writes == 1,    "VOL GPU write path entered (writes=%d)", vol_writes);
        CHECK(vol_comp == NUM_CHUNKS,
              "All %d chunks compressed on GPU (comp=%d)", NUM_CHUNKS, vol_comp);
        CHECK(vol_decomp == 0,    "No decompression during write (decomp=%d)", vol_decomp);

        printf("  [INFO] VOL counters: writes=%d, reads=%d, comp=%d, decomp=%d\n",
               vol_writes, vol_reads, vol_comp, vol_decomp);
    }
    printf("\n");

    /* ============================================================
     * Experiment 3: Compression header validation
     * ============================================================ */
    printf("── Experiment 3: Raw Chunk Inspection ───────────────────────\n");
    {
        /* Open file with native driver to read raw chunks */
        hid_t fid = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
        CHECK(fid >= 0, "Opened HDF5 file with native driver");

        if (fid >= 0) {
            hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            CHECK(dset >= 0, "Opened dataset 'data'");

            if (dset >= 0) {
                /* Read raw chunk 0 */
                hsize_t offset[1] = {0};
                uint32_t filters = 0;
                hsize_t chunk_nbytes = 0;

                /* Query chunk storage size */
                herr_t qrc = H5Dget_chunk_storage_size(dset, offset, &chunk_nbytes);
                CHECK(qrc >= 0 && chunk_nbytes > 0,
                      "Chunk 0 stored size: %zu bytes", (size_t)chunk_nbytes);

                if (chunk_nbytes > 0) {
                    uint8_t* raw = (uint8_t*)malloc(chunk_nbytes);
                    size_t read_sz = chunk_nbytes;
                    herr_t rrc = H5Dread_chunk(dset, H5P_DEFAULT, offset, &filters,
                                               raw, &read_sz);
                    CHECK(rrc >= 0, "H5Dread_chunk succeeded");

                    if (rrc >= 0) {
                        /* Parse CompressionHeader */
                        CompressionHeader hdr;
                        memcpy(&hdr, raw, sizeof(CompressionHeader));

                        CHECK(hdr.magic == COMPRESSION_MAGIC,
                              "Magic = 0x%08X (expected 0x%08X)",
                              hdr.magic, COMPRESSION_MAGIC);
                        CHECK(hdr.version == COMPRESSION_HEADER_VERSION,
                              "Version = %u (expected %u)",
                              hdr.version, COMPRESSION_HEADER_VERSION);
                        CHECK(hdr.original_size == CHUNK_FLOATS * sizeof(float),
                              "Original size = %zu (expected %zu)",
                              (size_t)hdr.original_size,
                              (size_t)(CHUNK_FLOATS * sizeof(float)));

                        size_t total_stored = sizeof(CompressionHeader) + hdr.compressed_size;
                        CHECK(total_stored == chunk_nbytes,
                              "Header + compressed = %zu == stored %zu",
                              total_stored, (size_t)chunk_nbytes);
                        CHECK(chunk_nbytes < CHUNK_FLOATS * sizeof(float),
                              "Compressed (%zu) < original (%zu) — compression works",
                              (size_t)chunk_nbytes,
                              (size_t)(CHUNK_FLOATS * sizeof(float)));

                        double ratio = (double)(CHUNK_FLOATS * sizeof(float)) / (double)chunk_nbytes;
                        printf("  [INFO] Chunk 0: %zu → %zu bytes (%.2fx ratio)\n",
                               (size_t)(CHUNK_FLOATS * sizeof(float)),
                               (size_t)chunk_nbytes, ratio);
                    }
                    free(raw);
                }

                /* Check all chunks have valid headers */
                int all_valid = 1;
                for (int c = 0; c < NUM_CHUNKS; c++) {
                    hsize_t off[1] = { (hsize_t)c * CHUNK_FLOATS };
                    hsize_t csz = 0;
                    H5Dget_chunk_storage_size(dset, off, &csz);
                    if (csz < sizeof(CompressionHeader)) { all_valid = 0; continue; }

                    uint8_t* chunk_buf = (uint8_t*)malloc(csz);
                    uint32_t flt = 0;
                    size_t rd_sz = csz;
                    H5Dread_chunk(dset, H5P_DEFAULT, off, &flt,
                                  chunk_buf, &rd_sz);
                    CompressionHeader h;
                    memcpy(&h, chunk_buf, sizeof(CompressionHeader));
                    free(chunk_buf);
                    if (h.magic != COMPRESSION_MAGIC) all_valid = 0;

                    double r = (double)(CHUNK_FLOATS * sizeof(float)) / (double)csz;
                    printf("  [INFO] Chunk %d: stored=%6zu bytes, ratio=%.2fx, "
                           "shuffle_elem=%u\n",
                           c, (size_t)csz, r, h.shuffle_element_size);
                }
                CHECK(all_valid, "All %d chunks have valid CompressionHeader magic",
                      NUM_CHUNKS);

                H5Dclose(dset);
            }
            H5Fclose(fid);
        }
    }
    printf("\n");

    /* ============================================================
     * Experiment 4: Per-chunk NN diagnostics
     * ============================================================ */
    printf("── Experiment 4: Per-Chunk NN Diagnostics ───────────────────\n");
    {
        int n_diag = gpucompress_get_chunk_history_count();
        CHECK(n_diag == NUM_CHUNKS,
              "Chunk history count = %d (expected %d)", n_diag, NUM_CHUNKS);

        int any_action_set = 0;
        for (int i = 0; i < n_diag && i < NUM_CHUNKS; i++) {
            gpucompress_chunk_diag_t diag;
            int drc = gpucompress_get_chunk_diag(i, &diag);
            if (drc == 0) {
                int algo_id = (diag.nn_action % 8) + 1;
                const char* aname = (algo_id >= 1 && algo_id <= 8)
                                    ? ALGO_NAMES[algo_id] : "???";
                printf("  [INFO] Chunk %d: action=%d → %s, ratio=%.2f"
                       " (predicted=%.2f), comp=%.3fms\n",
                       i, diag.nn_action, aname,
                       diag.actual_ratio, diag.predicted_ratio,
                       diag.compression_ms);
                if (diag.nn_action >= 0) any_action_set = 1;
            }
        }
        CHECK(any_action_set, "NN assigned actions to chunks");
    }
    printf("\n");

    /* ============================================================
     * Experiment 5: Bitwise lossless round-trip
     * ============================================================ */
    printf("── Experiment 5: Lossless Round-Trip Verification ───────────\n");
    {
        H5VL_gpucompress_reset_stats();

        int rr = read_gpu_data(HDF5_FILE, "data", d_readback, TOTAL_FLOATS);
        CHECK(rr == 0, "H5Dread with GPU pointer succeeded");

        int vol_writes = 0, vol_reads = 0, vol_comp = 0, vol_decomp = 0;
        H5VL_gpucompress_get_stats(&vol_writes, &vol_reads, &vol_comp, &vol_decomp);
        CHECK(vol_reads == 1,    "VOL GPU read path entered (reads=%d)", vol_reads);
        CHECK(vol_decomp == NUM_CHUNKS,
              "All %d chunks decompressed on GPU (decomp=%d)", NUM_CHUNKS, vol_decomp);

        unsigned long long mismatches = verify_gpu(d_data, d_readback, TOTAL_FLOATS);
        CHECK(mismatches == 0,
              "Bitwise lossless: %llu / %d mismatches", mismatches, TOTAL_FLOATS);
    }
    printf("\n");

    /* ============================================================
     * Experiment 6: Transfer accounting
     * ============================================================ */
    printf("── Experiment 6: Transfer Accounting ────────────────────────\n");
    {
        /* Do a fresh write with stats reset */
        H5VL_gpucompress_reset_stats();

        int h2d_n = 0, d2h_n = 0, d2d_n = 0;
        size_t h2d_b = 0, d2h_b = 0, d2d_b = 0;

        /* Write */
        gpucompress_reset_chunk_history();
        write_gpu_data(HDF5_FILE, "data", d_data,
                       TOTAL_FLOATS, CHUNK_FLOATS, GPUCOMPRESS_ALGO_AUTO);

        H5VL_gpucompress_get_transfer_stats(&h2d_n, &h2d_b,
                                             &d2h_n, &d2h_b,
                                             &d2d_n, &d2d_b);

        printf("  [INFO] VOL-layer transfers during write:\n");
        printf("         H->D: %d calls, %zu bytes\n", h2d_n, h2d_b);
        printf("         D->H: %d calls, %zu bytes\n", d2h_n, d2h_b);
        printf("         D->D: %d calls, %zu bytes\n", d2d_n, d2d_b);

        /* D→H should be compressed data (< raw data size) */
        CHECK(d2h_b < DATA_BYTES,
              "D->H bytes (%zu) < raw data (%zu) — only compressed bytes transferred",
              d2h_b, (size_t)DATA_BYTES);

        /* No H→D during write (data is already on GPU) */
        /* Note: h2d_n may include small metadata transfers, but not bulk data */
        CHECK(h2d_b < DATA_BYTES / 10,
              "H->D bytes (%zu) < 10%% of data (%zu) — no bulk H->D during GPU write",
              h2d_b, (size_t)DATA_BYTES);
    }
    printf("\n");

    /* ============================================================
     * Experiment 7: Multi-algorithm correctness
     * ============================================================ */
    printf("── Experiment 7: Multi-Algorithm Lossless Verification ──────\n");
    {
        int algos[] = {
            GPUCOMPRESS_ALGO_LZ4,
            GPUCOMPRESS_ALGO_ZSTD,
            GPUCOMPRESS_ALGO_SNAPPY,
        };
        const char* names[] = { "LZ4", "Zstd", "Snappy" };
        int n_algos = 3;

        for (int a = 0; a < n_algos; a++) {
            char fname[256];
            snprintf(fname, sizeof(fname), "/tmp/test_vol_verify_%s.h5", names[a]);

            H5VL_gpucompress_reset_stats();
            int wr = write_gpu_data(fname, "data", d_data,
                                     TOTAL_FLOATS, CHUNK_FLOATS, algos[a]);
            CHECK(wr == 0, "%s: write succeeded", names[a]);

            if (wr == 0) {
                CUDA_CHECK(cudaMemset(d_readback, 0, DATA_BYTES));
                int rr = read_gpu_data(fname, "data", d_readback, TOTAL_FLOATS);
                CHECK(rr == 0, "%s: read succeeded", names[a]);

                if (rr == 0) {
                    unsigned long long mm = verify_gpu(d_data, d_readback, TOTAL_FLOATS);
                    CHECK(mm == 0, "%s: bitwise lossless (%llu mismatches)", names[a], mm);
                }
            }
            remove(fname);
        }
    }
    printf("\n");

    /* ============================================================
     * Experiment 8: VOL does not break native HDF5 operations
     *
     * Core question: does our VOL connector interfere with standard
     * HDF5 file structure, metadata, dataspaces, or type handling?
     * ============================================================ */
    printf("── Experiment 8: HDF5 Structural Integrity ──────────────────\n");
    {
        /* Write data via VOL */
        gpucompress_reset_chunk_history();
        write_gpu_data(HDF5_FILE, "data", d_data,
                       TOTAL_FLOATS, CHUNK_FLOATS, GPUCOMPRESS_ALGO_LZ4);

        /* Now open the file with the NATIVE driver (no VOL) and inspect */
        hid_t fid = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
        CHECK(fid >= 0, "File opens with native HDF5 driver (no VOL)");

        if (fid >= 0) {
            /* Dataset exists and is accessible */
            hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            CHECK(dset >= 0, "Dataset 'data' accessible via native driver");

            if (dset >= 0) {
                /* Dataspace matches what we wrote */
                hid_t space = H5Dget_space(dset);
                int ndims = H5Sget_simple_extent_ndims(space);
                CHECK(ndims == 1, "Dataspace is 1D (ndims=%d)", ndims);

                hsize_t dims[1];
                H5Sget_simple_extent_dims(space, dims, NULL);
                CHECK(dims[0] == TOTAL_FLOATS,
                      "Dataspace size = %zu (expected %d)",
                      (size_t)dims[0], TOTAL_FLOATS);

                /* Datatype is correct */
                hid_t dtype = H5Dget_type(dset);
                CHECK(H5Tget_class(dtype) == H5T_FLOAT,
                      "Datatype class is H5T_FLOAT");
                CHECK(H5Tget_size(dtype) == sizeof(float),
                      "Datatype size = %zu (expected %zu)",
                      H5Tget_size(dtype), sizeof(float));

                /* Chunk layout is correct */
                hid_t dcpl = H5Dget_create_plist(dset);
                CHECK(H5Pget_layout(dcpl) == H5D_CHUNKED,
                      "Dataset layout is H5D_CHUNKED");

                hsize_t chunk_dims[1];
                H5Pget_chunk(dcpl, 1, chunk_dims);
                CHECK(chunk_dims[0] == CHUNK_FLOATS,
                      "Chunk size = %zu (expected %d)",
                      (size_t)chunk_dims[0], CHUNK_FLOATS);

                /* Filter is registered in DCPL */
                int nfilters = H5Pget_nfilters(dcpl);
                CHECK(nfilters >= 1, "DCPL has %d filter(s)", nfilters);

                if (nfilters >= 1) {
                    unsigned int flags_out = 0;
                    size_t cd_nelmts = 0;
                    unsigned int cd_values[8] = {0};
                    char fname[64] = {0};
                    cd_nelmts = 8;
                    H5Z_filter_t filt = H5Pget_filter2(dcpl, 0, &flags_out,
                                                        &cd_nelmts, cd_values,
                                                        sizeof(fname), fname, NULL);
                    CHECK(filt == H5Z_FILTER_GPUCOMPRESS,
                          "Filter ID = %d (expected %d gpucompress)",
                          (int)filt, H5Z_FILTER_GPUCOMPRESS);
                }

                /* All chunks are allocated */
                int all_stored = 1;
                for (int c = 0; c < NUM_CHUNKS; c++) {
                    hsize_t off[1] = { (hsize_t)c * CHUNK_FLOATS };
                    hsize_t csz = 0;
                    herr_t crc = H5Dget_chunk_storage_size(dset, off, &csz);
                    if (crc < 0 || csz == 0) { all_stored = 0; break; }
                }
                CHECK(all_stored, "All %d chunks are allocated in HDF5 storage",
                      NUM_CHUNKS);

                H5Pclose(dcpl);
                H5Tclose(dtype);
                H5Sclose(space);
                H5Dclose(dset);
            }

            /* File-level integrity: superblock, root group */
            hid_t root = H5Gopen2(fid, "/", H5P_DEFAULT);
            CHECK(root >= 0, "Root group '/' accessible");
            if (root >= 0) {
                H5G_info_t ginfo;
                H5Gget_info(root, &ginfo);
                CHECK(ginfo.nlinks >= 1,
                      "Root group has %zu link(s)", (size_t)ginfo.nlinks);
                H5Gclose(root);
            }

            H5Fclose(fid);
        }
    }
    printf("\n");

    /* Experiment 9 (Host-Pointer Passthrough) was removed: the gpucompress
     * VOL is GPU-pointer-only by design — dataset_read/write abort() on a
     * host buffer. Callers needing host I/O should use the native VOL.
     * See the file header of src/hdf5/H5VLgpucompress.cu. */

    /* ============================================================
     * Experiment 10: Multiple datasets in one file
     *
     * Verify VOL handles multi-dataset files correctly — writing
     * and reading multiple datasets doesn't cause cross-contamination.
     * ============================================================ */
    printf("── Experiment 10: Multi-Dataset Isolation ───────────────────\n");
    {
        const char* multi_file = "/tmp/test_vol_verify_multi.h5";

        /* Generate a second distinct GPU buffer */
        float* d_data2 = nullptr;
        size_t n2 = CHUNK_FLOATS * 2;  /* 2 chunks worth */
        CUDA_CHECK(cudaMalloc(&d_data2, n2 * sizeof(float)));
        /* Fill with a different pattern: descending ramp */
        fill_multi_pattern<<<(n2 + 255) / 256, 256>>>(d_data2, n2, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Write two datasets to same file */
        hid_t fid = open_vol_file(multi_file, H5F_ACC_TRUNC);
        CHECK(fid >= 0, "Created multi-dataset file");

        if (fid >= 0) {
            /* Dataset A: full data, LZ4 */
            {
                hsize_t dims[1]  = { TOTAL_FLOATS };
                hsize_t chunk[1] = { CHUNK_FLOATS };
                hid_t space = H5Screate_simple(1, dims, NULL);
                hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
                H5Pset_chunk(dcpl, 1, chunk);
                unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
                cd[0] = GPUCOMPRESS_ALGO_LZ4; cd[1] = 0; cd[2] = 0;
                pack_double(0.0, &cd[3], &cd[4]);
                H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                              H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
                hid_t dset = H5Dcreate2(fid, "dset_A", H5T_NATIVE_FLOAT,
                                        space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_data);
                H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl);
            }
            /* Dataset B: smaller data, Zstd */
            {
                hsize_t dims[1]  = { n2 };
                hsize_t chunk[1] = { CHUNK_FLOATS };
                hid_t space = H5Screate_simple(1, dims, NULL);
                hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
                H5Pset_chunk(dcpl, 1, chunk);
                unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
                cd[0] = GPUCOMPRESS_ALGO_ZSTD; cd[1] = 0; cd[2] = 0;
                pack_double(0.0, &cd[3], &cd[4]);
                H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                              H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
                hid_t dset = H5Dcreate2(fid, "dset_B", H5T_NATIVE_FLOAT,
                                        space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_data2);
                H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl);
            }
            H5Fclose(fid);
        }

        /* Read back each dataset independently and verify */
        fid = open_vol_file(multi_file, H5F_ACC_RDONLY);
        if (fid >= 0) {
            /* Read A */
            {
                hid_t dset = H5Dopen2(fid, "dset_A", H5P_DEFAULT);
                CUDA_CHECK(cudaMemset(d_readback, 0, DATA_BYTES));
                H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_readback);
                unsigned long long mm = verify_gpu(d_data, d_readback, TOTAL_FLOATS);
                CHECK(mm == 0, "dset_A: bitwise match (%llu mismatches)", mm);
                H5Dclose(dset);
            }
            /* Read B */
            {
                hid_t dset = H5Dopen2(fid, "dset_B", H5P_DEFAULT);
                float* d_read2 = nullptr;
                CUDA_CHECK(cudaMalloc(&d_read2, n2 * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_read2, 0, n2 * sizeof(float)));
                H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_read2);
                unsigned long long mm = verify_gpu(d_data2, d_read2, n2);
                CHECK(mm == 0, "dset_B: bitwise match (%llu mismatches)", mm);
                cudaFree(d_read2);
                H5Dclose(dset);
            }
            H5Fclose(fid);
        }

        cudaFree(d_data2);
        remove(multi_file);
    }
    printf("\n");

    /* ============================================================
     * Experiment 11: File re-open persistence
     *
     * Write via VOL, close file, reopen via VOL, read back.
     * Ensures VOL flush/close path works and data persists on disk.
     * ============================================================ */
    printf("── Experiment 11: Write-Close-Reopen-Read ───────────────────\n");
    {
        const char* reopen_file = "/tmp/test_vol_verify_reopen.h5";

        /* Write and close */
        write_gpu_data(reopen_file, "persistent", d_data,
                       TOTAL_FLOATS, CHUNK_FLOATS, GPUCOMPRESS_ALGO_LZ4);

        /* Reopen and read */
        CUDA_CHECK(cudaMemset(d_readback, 0, DATA_BYTES));
        int rr = read_gpu_data(reopen_file, "persistent", d_readback, TOTAL_FLOATS);
        CHECK(rr == 0, "Reopen + read succeeded");

        if (rr == 0) {
            unsigned long long mm = verify_gpu(d_data, d_readback, TOTAL_FLOATS);
            CHECK(mm == 0, "Data persisted correctly after close/reopen (%llu mismatches)", mm);
        }

        remove(reopen_file);
    }
    printf("\n");

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("================================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", g_pass, g_fail);
    printf("================================================================\n");

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_readback);
    gpucompress_cleanup();
    remove(HDF5_FILE);

    return g_fail > 0 ? 1 : 0;
}
