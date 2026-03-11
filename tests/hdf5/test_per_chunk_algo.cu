/**
 * test_per_chunk_algo.cu
 *
 * Simple test: write GPU data through HDF5 VOL with ALGO_AUTO,
 * then read back raw chunk headers to see which algorithm was
 * used per chunk. Also reads back through VOL to verify correctness.
 *
 * Uses 4 chunks of 8 MB with different data patterns to encourage
 * the NN to pick different algorithms.
 */

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "compression_header.h"
#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5
#define TMP_FILE "/tmp/test_per_chunk_algo.h5"

static const char* ALGO_NAMES[] = {
    "auto(0)", "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_vol_fapl()
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

int main()
{
    printf("=== Test: Per-Chunk Algorithm Selection via HDF5 VOL ===\n\n");

    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "../../neural_net/weights/model.nnwt";
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed (err=%d)\n", err);
        return 1;
    }
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    /* gpucompress_set_verbose(1); — enable for decompress header debug */

    /* 4 chunks × 8 MB = 32 MB total */
    const size_t CHUNK_FLOATS = 2 * 1024 * 1024;  /* 8 MB */
    const int N_CHUNKS = 4;
    const size_t N_FLOATS = CHUNK_FLOATS * N_CHUNKS;
    const size_t TOTAL_BYTES = N_FLOATS * sizeof(float);

    printf("  Data: %d chunks × %zu MB = %zu MB\n\n",
           N_CHUNKS, CHUNK_FLOATS * sizeof(float) / (1 << 20),
           TOTAL_BYTES / (1 << 20));

    /* Fill each chunk with a different pattern */
    float* h_data = (float*)malloc(TOTAL_BYTES);
    srand(42);
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t off = (size_t)c * CHUNK_FLOATS;
        switch (c) {
            case 0: /* Constant */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = 3.14159f;
                break;
            case 1: /* Random */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = (float)rand() / RAND_MAX;
                break;
            case 2: /* Smooth sine */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = sinf((float)i * 0.0001f) * 1000.0f;
                break;
            case 3: /* Sparse */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = (rand() % 200 == 0) ? (float)(rand() % 10000) : 0.0f;
                break;
        }
    }

    float* d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);
    cudaMemcpy(d_data, h_data, TOTAL_BYTES, cudaMemcpyHostToDevice);

    /* DCPL with ALGO_AUTO */
    hsize_t chunk_dims[1] = { (hsize_t)CHUNK_FLOATS };
    hsize_t dims[1] = { (hsize_t)N_FLOATS };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0; cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    /* ── Write through VOL ── */
    printf("  WRITE: H5Dwrite with ALGO_AUTO through GPUCompress VOL\n");
    remove(TMP_FILE);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    H5Pclose(dcpl);

    /* ── Inspect chunk headers (compression path) ── */
    printf("\n  COMPRESSION — chunk headers on disk:\n");
    printf("    chunk | algo_id | algorithm    | shuffle | compressed_sz | pattern\n");
    printf("    ------+---------+--------------+---------+---------------+---------\n");

    const char* patterns[] = {"constant", "random", "sine", "sparse"};

    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    int seen[9] = {};
    int distinct = 0;

    for (int c = 0; c < N_CHUNKS; c++) {
        hsize_t offset[1] = { (hsize_t)c * CHUNK_FLOATS };
        hsize_t chunk_nbytes = 0;
        H5Dget_chunk_storage_size(dset, offset, &chunk_nbytes);

        uint8_t* buf = (uint8_t*)malloc(chunk_nbytes);
        uint32_t fm = 0;
        size_t rs = chunk_nbytes;
        H5Dread_chunk(dset, H5P_DEFAULT, offset, &fm, buf, &rs);

        CompressionHeader hdr;
        memcpy(&hdr, buf, sizeof(hdr));
        free(buf);

        uint8_t aid = hdr.getAlgorithmId();
        const char* name = (aid <= 8) ? ALGO_NAMES[aid] : "unknown";
        if (aid <= 8 && !seen[aid]) { seen[aid] = 1; distinct++; }

        printf("    %5d | %7d | %-12s | %7s | %10zu B  | %s\n",
               c + 1, aid, name,
               hdr.shuffle_element_size > 0 ? "yes" : "no",
               (size_t)hdr.compressed_size, patterns[c]);
    }

    H5Dclose(dset);
    H5Fclose(file);

    printf("\n    Distinct algorithms: %d\n", distinct);

    /* ── Read back through VOL (decompression path) ── */
    printf("\n  DECOMPRESSION — reading back through GPUCompress VOL:\n");

    float* d_read = NULL;
    cudaMalloc(&d_read, TOTAL_BYTES);

    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);

    /* Bitwise verify */
    float* h_read = (float*)malloc(TOTAL_BYTES);
    cudaMemcpy(h_read, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);

    printf("    Per-chunk verification:\n");
    int all_ok = 1;
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t off = (size_t)c * CHUNK_FLOATS;
        unsigned long long mm = 0;
        for (size_t i = 0; i < CHUNK_FLOATS; i++) {
            unsigned int a, b;
            memcpy(&a, &h_data[off + i], sizeof(unsigned int));
            memcpy(&b, &h_read[off + i], sizeof(unsigned int));
            if (a != b) mm++;
        }
        printf("      chunk %d (%s): mismatches=%llu — %s\n",
               c + 1, patterns[c], mm, mm == 0 ? "PASS" : "FAIL");
        if (mm != 0) all_ok = 0;
    }

    /* Summary */
    printf("\n  SUMMARY:\n");
    printf("    Per-chunk algo selection: %s (%d distinct algorithms)\n",
           distinct > 1 ? "YES" : "NO (all same)", distinct);
    printf("    Bitwise correctness:      %s\n", all_ok ? "PASS" : "FAIL");

    remove(TMP_FILE);
    free(h_data);
    free(h_read);
    cudaFree(d_data);
    cudaFree(d_read);
    gpucompress_cleanup();

    int ok = all_ok && (distinct > 0);
    printf("\n  Result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
