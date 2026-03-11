/**
 * inspect_chunk_headers.cu
 *
 * Write data through HDF5 VOL with a fixed algorithm, then read back
 * raw chunk bytes and inspect the CompressionHeader to verify which
 * algorithm was actually used per chunk.
 *
 * Also tests ALGO_AUTO to compare.
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

static void inspect_chunks(const char* filepath, const char* label, int n_chunks)
{
    printf("\n  Inspecting chunk headers in '%s':\n", filepath);

    /* Open with native VOL (not gpucompress VOL) to read raw chunk bytes */
    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { printf("    ERROR: Cannot open file\n"); return; }
    hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) { printf("    ERROR: Cannot open dataset\n"); H5Fclose(file); return; }

    /* Read raw chunk data directly */
    printf("    chunk | algo_id | algorithm    | shuffle | compressed_sz\n");
    printf("    ------+---------+--------------+---------+--------------\n");

    for (int c = 0; c < n_chunks; c++) {
        hsize_t offset[1] = { 0 };
        /* For 1D chunked data, chunk offset = c * chunk_size_in_elements */
        /* We need to figure out the chunk dimensions first */
        hid_t dcpl = H5Dget_create_plist(dset);
        hsize_t chunk_dims[1];
        H5Pget_chunk(dcpl, 1, chunk_dims);
        H5Pclose(dcpl);

        offset[0] = (hsize_t)c * chunk_dims[0];

        /* Allocate buffer for raw chunk */
        uint32_t filter_mask = 0;
        hsize_t chunk_nbytes = 0;

        /* Get chunk storage size */
        herr_t ret = H5Dget_chunk_storage_size(dset, offset, &chunk_nbytes);
        if (ret < 0 || chunk_nbytes == 0) {
            printf("    %5d | (no chunk data at this offset)\n", c + 1);
            continue;
        }

        uint8_t* buf = (uint8_t*)malloc(chunk_nbytes);
        size_t read_sz = chunk_nbytes;
        ret = H5Dread_chunk(dset, H5P_DEFAULT, offset, &filter_mask, buf, &read_sz);
        (void)filter_mask;
        if (ret < 0) {
            printf("    %5d | ERROR reading chunk\n", c + 1);
            free(buf);
            continue;
        }

        /* Parse CompressionHeader from the first 64 bytes */
        if (chunk_nbytes >= sizeof(CompressionHeader)) {
            CompressionHeader hdr;
            memcpy(&hdr, buf, sizeof(CompressionHeader));

            if (hdr.magic == COMPRESSION_MAGIC) {
                uint8_t algo_id = hdr.getAlgorithmId();
                const char* algo_name = (algo_id <= 8) ? ALGO_NAMES[algo_id] : "unknown";
                printf("    %5d | %7d | %-12s | %7s | %zu bytes\n",
                       c + 1, algo_id, algo_name,
                       hdr.shuffle_element_size > 0 ? "yes" : "no",
                       (size_t)hdr.compressed_size);
            } else {
                printf("    %5d | (no GPUCompress header, magic=0x%08x)\n",
                       c + 1, hdr.magic);
            }
        } else {
            printf("    %5d | (chunk too small for header: %zu bytes)\n",
                   c + 1, (size_t)chunk_nbytes);
        }

        free(buf);
    }

    H5Dclose(dset);
    H5Fclose(file);
}

int main()
{
    printf("=== Inspect Chunk Headers: Fixed vs ALGO_AUTO ===\n");

    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "../../neural_net/weights/model.nnwt";
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* Create varied test data: 8 chunks of 1 MB each */
    const size_t CHUNK_FLOATS = 256 * 1024;
    const int N_CHUNKS = 8;
    const size_t N_FLOATS = CHUNK_FLOATS * N_CHUNKS;
    const size_t TOTAL_BYTES = N_FLOATS * sizeof(float);

    float* h_data = (float*)malloc(TOTAL_BYTES);
    srand(42);
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t off = (size_t)c * CHUNK_FLOATS;
        switch (c % 4) {
            case 0: for (size_t i = 0; i < CHUNK_FLOATS; i++) h_data[off+i] = 3.14159f; break;
            case 1: for (size_t i = 0; i < CHUNK_FLOATS; i++) h_data[off+i] = (float)rand()/RAND_MAX; break;
            case 2: for (size_t i = 0; i < CHUNK_FLOATS; i++) h_data[off+i] = sinf((float)i*0.001f)*100.0f; break;
            case 3: for (size_t i = 0; i < CHUNK_FLOATS; i++) h_data[off+i] = (rand()%100==0) ? (float)(rand()%1000) : 0.0f; break;
        }
    }

    float* d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);
    cudaMemcpy(d_data, h_data, TOTAL_BYTES, cudaMemcpyHostToDevice);

    hsize_t chunk_dims[1] = { (hsize_t)CHUNK_FLOATS };
    hsize_t dims[1] = { (hsize_t)N_FLOATS };

    /* ── Test 1: Fixed algorithm (zstd, algo_enum=5) ── */
    printf("\n── Test 1: make_dcpl_fixed(zstd) — expect ALL chunks = zstd ──\n");
    {
        const char* f = "/tmp/test_fixed_algo.h5";
        remove(f);

        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk_dims);
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = 5; /* zstd */
        cd[1] = 0;
        cd[2] = 0;
        pack_double_cd(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(f, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
        H5Fclose(file);
        H5Pclose(dcpl);

        inspect_chunks(f, "fixed(zstd)", N_CHUNKS);
        remove(f);
    }

    /* ── Test 2: ALGO_AUTO — expect per-chunk algorithm variation ── */
    printf("\n── Test 2: ALGO_AUTO — expect DIFFERENT algorithms per chunk ──\n");
    {
        const char* f = "/tmp/test_auto_algo.h5";
        remove(f);

        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk_dims);
        unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
        cd[0] = 0; /* ALGO_AUTO */
        cd[1] = 0;
        cd[2] = 0;
        pack_double_cd(0.0, &cd[3], &cd[4]);
        H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(f, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
        H5Fclose(file);
        H5Pclose(dcpl);

        inspect_chunks(f, "auto", N_CHUNKS);
        remove(f);
    }

    free(h_data);
    cudaFree(d_data);
    gpucompress_cleanup();

    printf("\n=== Done ===\n");
    return 0;
}
