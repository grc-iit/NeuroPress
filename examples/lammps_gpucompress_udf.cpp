/**
 * @file lammps_gpucompress_udf.cpp
 * @brief Implementation of GPUCompress bridge for LAMMPS fixes
 *
 * Compiled separately as a shared library, then linked into a custom
 * LAMMPS fix or loaded via plugin system.
 */

#include "lammps_gpucompress_udf.h"
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"

#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sys/stat.h>

/* Static state */
static hid_t g_fapl = H5I_INVALID_HID;
static int   g_initialized = 0;

extern "C" {

int gpucompress_lammps_init(const char* weights_path, const char* policy)
{
    if (g_initialized) return 0;

    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "[GPUCompress-LAMMPS] gpucompress_init failed: %d\n", err);
        return -1;
    }

    H5Z_gpucompress_register();
    H5VL_gpucompress_register();

    hid_t native_id = H5VLget_connector_id_by_name("native");
    g_fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(g_fapl, native_id, NULL);
    H5VLclose(native_id);

    /* Set ranking policy */
    if (policy && strcmp(policy, "speed") == 0) {
        gpucompress_set_ranking_weights(1.0f, 0.0f, 0.0f);
    } else if (policy && strcmp(policy, "ratio") == 0) {
        gpucompress_set_ranking_weights(0.0f, 0.0f, 1.0f);
    } else {
        gpucompress_set_ranking_weights(1.0f, 1.0f, 0.5f);
    }

    g_initialized = 1;
    return 0;
}

static gpucompress_algorithm_t algo_from_name(const char* name)
{
    if (!name) return GPUCOMPRESS_ALGO_AUTO;
    if (strcmp(name, "lz4") == 0)       return GPUCOMPRESS_ALGO_LZ4;
    if (strcmp(name, "snappy") == 0)    return GPUCOMPRESS_ALGO_SNAPPY;
    if (strcmp(name, "deflate") == 0)   return GPUCOMPRESS_ALGO_DEFLATE;
    if (strcmp(name, "gdeflate") == 0)  return GPUCOMPRESS_ALGO_GDEFLATE;
    if (strcmp(name, "zstd") == 0)      return GPUCOMPRESS_ALGO_ZSTD;
    if (strcmp(name, "ans") == 0)       return GPUCOMPRESS_ALGO_ANS;
    if (strcmp(name, "cascaded") == 0)  return GPUCOMPRESS_ALGO_CASCADED;
    if (strcmp(name, "bitcomp") == 0)   return GPUCOMPRESS_ALGO_BITCOMP;
    return GPUCOMPRESS_ALGO_AUTO;
}

int gpucompress_lammps_write_field(const char* filename,
                                    const char* dset_name,
                                    const void* d_ptr,
                                    size_t n_elements,
                                    int elem_bytes,
                                    const char* algo_name,
                                    double error_bound,
                                    int verify)
{
    if (!g_initialized) {
        fprintf(stderr, "[GPUCompress-LAMMPS] Not initialized!\n");
        return -1;
    }
    if (!d_ptr || n_elements == 0) return -1;

    gpucompress_algorithm_t algo = algo_from_name(algo_name);
    hid_t h5type = (elem_bytes == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    size_t chunk_elems = (4 * 1024 * 1024) / elem_bytes;
    if (chunk_elems > n_elements) chunk_elems = n_elements;

    hsize_t dims[1]  = { (hsize_t)n_elements };
    hsize_t cdims[1] = { (hsize_t)chunk_elems };

    H5VL_gpucompress_reset_stats();

    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, g_fapl);
    if (fid < 0) return -1;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int shuffle_size = (unsigned int)elem_bytes;
    unsigned int preproc = 0;
    if (error_bound > 0.0) preproc = GPUCOMPRESS_PREPROC_QUANTIZE;
    H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);

    hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int rc = 0;
    if (dset >= 0) {
        H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_ptr);
        H5Dclose(dset);
    } else {
        rc = -1;
    }

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
    /* NOTE: cudaDeviceSynchronize() was removed here. It was called inside a
     * KOKKOS END_OF_STEP fix, which corrupts KOKKOS::Cuda's internal stream-
     * ordering bookkeeping (event fences). The crash manifests one timestep later
     * as a silent SIGABRT from KOKKOS's error handler. H5Fclose() already flushes
     * the VOL pipeline; no additional global sync is needed. */

    /* Verification */
    if (verify && rc == 0) {
        hid_t vfid = H5Fopen(filename, H5F_ACC_RDONLY, g_fapl);
        if (vfid < 0) return -1;

        hid_t vdset = H5Dopen2(vfid, dset_name, H5P_DEFAULT);
        if (vdset < 0) { H5Fclose(vfid); return -1; }

        size_t total_bytes = n_elements * elem_bytes;
        void* d_readback = nullptr;
        cudaMalloc(&d_readback, total_bytes);
        H5Dread(vdset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
        cudaDeviceSynchronize();

        H5Dclose(vdset);
        H5Fclose(vfid);

        std::vector<char> h_orig(total_bytes);
        std::vector<char> h_read(total_bytes);
        cudaMemcpy(h_orig.data(), d_ptr, total_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_read.data(), d_readback, total_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_readback);

        if (memcmp(h_orig.data(), h_read.data(), total_bytes) != 0) {
            fprintf(stderr, "[GPUCompress-LAMMPS] VERIFY FAILED: %s/%s mismatch!\n",
                    filename, dset_name);
            rc = -1;
        }
    }

    return rc;
}

void gpucompress_lammps_finalize(void)
{
    if (g_fapl >= 0) {
        H5Pclose(g_fapl);
        g_fapl = H5I_INVALID_HID;
    }
    gpucompress_cleanup();
    g_initialized = 0;
}

} /* extern "C" */
