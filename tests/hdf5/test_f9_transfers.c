/**
 * @file test_f9_transfers.c
 * @brief Verify F9 fused pipeline: minimal CUDA transfers during H5Dwrite.
 *
 * 64MB dataset, 32MB chunks (2 chunks). No exploration, no SGD.
 * Expected: only 2 H→D transfers (one per chunk) for the data itself,
 * plus stats + NN inference happen entirely on GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

#define DATASET_MB  64
#define CHUNK_MB    32
#define TMP_FILE    "/tmp/test_f9_transfers.h5"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void fill_sine(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        buf[i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
}

int main(int argc, char **argv) {
    const char *weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";

    size_t chunk_floats  = (CHUNK_MB * 1024UL * 1024UL) / sizeof(float);
    size_t dataset_floats = (DATASET_MB * 1024UL * 1024UL) / sizeof(float);
    size_t n_chunks = dataset_floats / chunk_floats;

    printf("=== F9 Transfer Test ===\n");
    printf("Dataset: %d MB (%zu floats), Chunk: %d MB, Chunks: %zu\n",
           DATASET_MB, dataset_floats, CHUNK_MB, n_chunks);
    printf("Exploration: OFF, SGD: OFF\n\n");

    /* Init library + NN */
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %d\n", err);
        return 1;
    }

    /* Keep online learning OFF — no exploration, no SGD, no stats D→H */

    /* Register HDF5 filter */
    H5Z_gpucompress_register();

    /* Create file */
    hid_t fid = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Dataspace */
    hsize_t dims[1] = { dataset_floats };
    hid_t space = H5Screate_simple(1, dims, NULL);

    /* Dataset creation property: chunked + gpucompress ALGO_AUTO */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[1] = { chunk_floats };
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd_values[3] = { GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_NONE, 0 };
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_MANDATORY, 3, cd_values);

    hid_t dset = H5Dcreate2(fid, "data", H5T_IEEE_F32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* Fill data */
    float *buf = (float *)malloc(dataset_floats * sizeof(float));
    fill_sine(buf, dataset_floats);

    printf("Writing %zu chunks via H5Dwrite...\n", n_chunks);
    printf("(Profile with: nsys profile --stats=true ./build/test_f9_transfers %s)\n\n", weights);

    herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    printf("H5Dwrite %s\n\n", rc >= 0 ? "OK" : "FAILED");

    /* Cleanup */
    free(buf);
    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
    gpucompress_cleanup();
    remove(TMP_FILE);

    printf("Done. Use nsys to count cudaMemcpy calls.\n");
    return 0;
}
