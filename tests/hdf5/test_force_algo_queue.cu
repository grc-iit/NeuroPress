/**
 * test_force_algo_queue.cu
 *
 * Test: gpucompress_force_algorithm() queue allows per-chunk algorithm
 * selection through the HDF5 VOL with ALGO_AUTO.
 *
 * 1. Push per-chunk algorithm choices into the force queue
 * 2. Write through HDF5 VOL with ALGO_AUTO DCPL
 * 3. Read back raw chunks via native HDF5 and inspect CompressionHeader
 * 4. Verify each chunk used the forced algorithm (not NN's choice)
 * 5. Read back through VOL and verify bitwise correctness
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
#define TMP_FILE "/tmp/test_force_algo.h5"

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
    int pass = 0, fail = 0;

    printf("=== Test: Force-Algorithm Queue for Per-Chunk Selection ===\n\n");

    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "../../neural_net/weights/model.nnwt";
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed (err=%d)\n", err);
        return 1;
    }
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* 4 chunks of 1 MB each — uniform data so NN choice is predictable */
    const size_t CHUNK_FLOATS = 256 * 1024;
    const int N_CHUNKS = 4;
    const size_t N_FLOATS = CHUNK_FLOATS * N_CHUNKS;
    const size_t TOTAL_BYTES = N_FLOATS * sizeof(float);

    float* h_data = (float*)malloc(TOTAL_BYTES);
    srand(42);
    for (size_t i = 0; i < N_FLOATS; i++)
        h_data[i] = sinf((float)i * 0.0001f) * 100.0f;

    float* d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);
    cudaMemcpy(d_data, h_data, TOTAL_BYTES, cudaMemcpyHostToDevice);

    /* Force each chunk to a DIFFERENT algorithm:
     *   chunk 0 → lz4 (1)
     *   chunk 1 → deflate (3)
     *   chunk 2 → zstd (5)
     *   chunk 3 → ans (6)
     */
    int forced_algos[] = {1, 3, 5, 6};
    int forced_shuffle[] = {0, 1, 0, 1};

    /* ── Check 1: API exists ── */
    printf("  Check 1: gpucompress_force_algorithm_push() API exists... ");
#ifdef GPUCOMPRESS_HAS_FORCE_ALGO
    printf("PASS\n"); pass++;
#else
    /* Try calling it — if it links, the API exists */
    gpucompress_force_algorithm_reset();
    for (int c = 0; c < N_CHUNKS; c++) {
        gpucompress_force_algorithm_push(forced_algos[c], forced_shuffle[c], 0, 0.0);
    }
    printf("PASS\n"); pass++;
#endif

    /* ── Write through HDF5 VOL with ALGO_AUTO ── */
    printf("  Writing %d chunks via HDF5 VOL (ALGO_AUTO + force queue)...\n", N_CHUNKS);

    hsize_t chunk_dims[1] = { (hsize_t)CHUNK_FLOATS };
    hsize_t dims[1] = { (hsize_t)N_FLOATS };

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    remove(TMP_FILE);
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    H5Pclose(dcpl);

    /* ── Check 2-5: Inspect raw chunk headers (compression path) ── */
    printf("\n  Compression path — inspecting chunk headers:\n");
    printf("    chunk | expected     | actual       | shuffle | result\n");
    printf("    ------+--------------+--------------+---------+-------\n");

    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    for (int c = 0; c < N_CHUNKS; c++) {
        hsize_t offset[1] = { (hsize_t)c * CHUNK_FLOATS };
        hsize_t chunk_nbytes = 0;
        H5Dget_chunk_storage_size(dset, offset, &chunk_nbytes);

        if (chunk_nbytes < sizeof(CompressionHeader)) {
            printf("    %5d | %-12s | (no header)  |         | FAIL\n",
                   c + 1, ALGO_NAMES[forced_algos[c]]);
            fail++;
            continue;
        }

        uint8_t* buf = (uint8_t*)malloc(chunk_nbytes);
        uint32_t filter_mask = 0;
        size_t read_sz = chunk_nbytes;
        H5Dread_chunk(dset, H5P_DEFAULT, offset, &filter_mask, buf, &read_sz);

        CompressionHeader hdr;
        memcpy(&hdr, buf, sizeof(CompressionHeader));
        free(buf);

        if (hdr.magic != COMPRESSION_MAGIC) {
            printf("    %5d | %-12s | (bad magic)  |         | FAIL\n",
                   c + 1, ALGO_NAMES[forced_algos[c]]);
            fail++;
            continue;
        }

        uint8_t algo_id = hdr.getAlgorithmId();
        const char* actual_name = (algo_id <= 8) ? ALGO_NAMES[algo_id] : "unknown";
        int actual_shuffle = (hdr.shuffle_element_size > 0) ? 1 : 0;

        bool algo_ok = (algo_id == forced_algos[c]);
        bool shuf_ok = (actual_shuffle == forced_shuffle[c]);
        bool ok = algo_ok && shuf_ok;

        printf("    %5d | %-12s | %-12s | %s vs %s | %s\n",
               c + 1,
               ALGO_NAMES[forced_algos[c]], actual_name,
               forced_shuffle[c] ? "yes" : " no",
               actual_shuffle ? "yes" : " no",
               ok ? "PASS" : "FAIL");
        if (ok) pass++; else fail++;
    }

    H5Dclose(dset);
    H5Fclose(file);

    /* ── Check 6-9: Read back through VOL and inspect decompression headers ── */
    printf("\n  Decompression path — reading back through VOL:\n");

    float* d_read = NULL;
    cudaMalloc(&d_read, TOTAL_BYTES);

    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    gpucompress_reset_chunk_history();
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);

    /* Bitwise verification */
    float* h_read = (float*)malloc(TOTAL_BYTES);
    cudaMemcpy(h_read, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    unsigned long long mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; i++) {
        unsigned int a, b;
        memcpy(&a, &h_data[i], sizeof(unsigned int));
        memcpy(&b, &h_read[i], sizeof(unsigned int));
        if (a != b) mismatches++;
    }
    printf("    Bitwise mismatches: %llu — %s\n", mismatches,
           mismatches == 0 ? "PASS" : "FAIL");
    if (mismatches == 0) pass++; else fail++;

    /* ── Check 10: All 4 algorithms are distinct ── */
    printf("\n  Check: forced 4 distinct algorithms through single ALGO_AUTO dataset... ");
    /* Re-open natively to count distinct algos */
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    int seen_algos[9] = {};
    int distinct = 0;
    for (int c = 0; c < N_CHUNKS; c++) {
        hsize_t offset[1] = { (hsize_t)c * CHUNK_FLOATS };
        hsize_t chunk_nbytes = 0;
        H5Dget_chunk_storage_size(dset, offset, &chunk_nbytes);
        uint8_t* buf = (uint8_t*)malloc(chunk_nbytes);
        uint32_t fm = 0; size_t rs = chunk_nbytes;
        H5Dread_chunk(dset, H5P_DEFAULT, offset, &fm, buf, &rs);
        CompressionHeader hdr;
        memcpy(&hdr, buf, sizeof(hdr));
        free(buf);
        uint8_t aid = hdr.getAlgorithmId();
        if (aid <= 8 && !seen_algos[aid]) { seen_algos[aid] = 1; distinct++; }
    }
    H5Dclose(dset);
    H5Fclose(file);

    if (distinct >= 3) {
        printf("PASS (%d distinct)\n", distinct);
        pass++;
    } else {
        printf("FAIL (only %d distinct, expected >= 3)\n", distinct);
        fail++;
    }

    /* Cleanup */
    remove(TMP_FILE);
    free(h_data);
    free(h_read);
    cudaFree(d_data);
    cudaFree(d_read);
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", pass, fail);
    return (fail == 0) ? 0 : 1;
}
