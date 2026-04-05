/**
 * @file nekrs_gpucompress_bridge.hpp
 * @brief Header-only bridge between nekRS OCCA fields and GPUCompress
 *
 * Writes compressed HDF5 using GPUCompress VOL connector — follows the
 * exact same pattern as VPIC's write_gpu_to_hdf5() and the Nyx bridge.
 * HDF5 handles chunking, VOL intercepts H5Dwrite with GPU pointers.
 *
 * nekRS uses OCCA for GPU abstraction. With the CUDA backend,
 * occa::memory::ptr<void>() returns a raw CUDA device pointer.
 * This bridge extracts those pointers and passes them through
 * the GPUCompress HDF5 VOL connector for transparent compression.
 *
 * Usage in a nekRS .udf file:
 *   1. Call gpucompress_nekrs_bridge::init() once in UDF_Setup()
 *   2. Call gpucompress_nekrs_bridge::compress_fields() in UDF_ExecuteStep()
 *      when nrs->checkpointStep is true
 */

#ifndef NEKRS_GPUCOMPRESS_BRIDGE_HPP
#define NEKRS_GPUCOMPRESS_BRIDGE_HPP

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"

#include <hdf5.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdio>
#include <mpi.h>

namespace gpucompress_nekrs_bridge {

/**
 * Initialize GPUCompress + HDF5 VOL connector. Call once.
 * Returns the FAPL with VOL configured, or H5I_INVALID_HID on failure.
 */
inline hid_t init(const char* weights_path)
{
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) return H5I_INVALID_HID;

    H5Z_gpucompress_register();
    hid_t vol_id = H5VL_gpucompress_register();
    (void)vol_id;

    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    return fapl;
}

/**
 * Write GPU-resident data to compressed HDF5 — same pattern as VPIC deck.
 * HDF5 chunking + VOL connector handles compression transparently.
 */
inline void write_gpu_to_hdf5(const char* filename, const char* dset_name,
                               const void* d_data, size_t n_elements,
                               hid_t h5type, size_t chunk_elements,
                               hid_t fapl,
                               gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
                               double error_bound = 0.0)
{
    if (chunk_elements > n_elements) chunk_elements = n_elements;
    hsize_t dims[1]  = { (hsize_t)n_elements };
    hsize_t cdims[1] = { (hsize_t)chunk_elements };

    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    unsigned int preproc = 0;
    if (error_bound > 0.0) preproc = GPUCOMPRESS_PREPROC_QUANTIZE;
    H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);

    hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset >= 0) {
        /* d_data is a CUDA device pointer — VOL detects this and
         * compresses each chunk on GPU, writes pre-compressed bytes */
        H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset);
    }

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
}

/**
 * Read back a compressed HDF5 file via VOL and verify bitwise against original GPU data.
 * Returns 0 on success (bitwise match), -1 on mismatch or error.
 */
inline int verify_gpu_hdf5(const char* filename, const char* dset_name,
                            const void* d_original, size_t n_elements,
                            hid_t h5type, hid_t fapl)
{
    hid_t fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    if (fid < 0) return -1;

    hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
    if (dset < 0) { H5Fclose(fid); return -1; }

    size_t elem_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    size_t total_bytes = n_elements * elem_size;

    void* d_readback = nullptr;
    cudaMalloc(&d_readback, total_bytes);

    H5Dread(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    cudaDeviceSynchronize();

    H5Dclose(dset);
    H5Fclose(fid);

    /* Bitwise comparison on host */
    std::vector<char> h_orig(total_bytes);
    std::vector<char> h_read(total_bytes);
    cudaMemcpy(h_orig.data(), d_original, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read.data(), d_readback, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_readback);

    int result = memcmp(h_orig.data(), h_read.data(), total_bytes);
    return (result == 0) ? 0 : -1;
}

/* Forward declarations */
inline int verify_field(hid_t fid, const char* dset_name,
                        const void* d_original, size_t n_elements,
                        hid_t h5type);

/**
 * Compress and write nekRS fields to HDF5 via VOL connector.
 *
 * Each field (velocity components, pressure, scalars) is written as a
 * separate dataset in a single HDF5 file per checkpoint step.
 *
 * @param dir           Output directory
 * @param step          Time step number
 * @param field_ptrs    Device pointers from occa::memory::ptr<void>()
 * @param field_names   Names for each field
 * @param n_points      Number of mesh points per field
 * @param fapl          File access property list from init()
 * @param h5type        HDF5 data type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE)
 * @param chunk_bytes   HDF5 chunk size in bytes (default 4 MiB)
 * @param algo          Compression algorithm
 * @param error_bound   0.0 = lossless
 * @param verify        If true, read back and verify bitwise
 * @return Total original bytes written
 */
inline long compress_fields(
    const std::string& dir,
    int step,
    const std::vector<const void*>& field_ptrs,
    const std::vector<std::string>& field_names,
    size_t n_points,
    hid_t fapl,
    hid_t h5type,
    size_t chunk_bytes = 4 * 1024 * 1024,
    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
    double error_bound = 0.0,
    bool verify = false)
{
    size_t elem_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    size_t chunk_elems = chunk_bytes / elem_size;
    long total_original = 0;

    /* Create output directory (rank 0 only) */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        mkdir(dir.c_str(), 0755);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    char fname[512];
    snprintf(fname, sizeof(fname), "%s/step_%06d_rank_%04d.h5",
             dir.c_str(), step, rank);

    /* Create one HDF5 file per rank per step */
    hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return 0;

    for (size_t i = 0; i < field_ptrs.size(); ++i) {
        const void* d_ptr = field_ptrs[i];
        const char* dset_name = field_names[i].c_str();
        size_t field_bytes = n_points * elem_size;

        hsize_t dims[1]  = { (hsize_t)n_points };
        hsize_t ce = (chunk_elems > n_points) ? n_points : chunk_elems;
        hsize_t cdims[1] = { ce };

        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);

        unsigned int shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
        unsigned int preproc = 0;
        if (error_bound > 0.0) preproc = GPUCOMPRESS_PREPROC_QUANTIZE;
        H5Pset_gpucompress(dcpl, algo, preproc, shuffle_size, error_bound);

        hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset >= 0) {
            H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_ptr);
            H5Dclose(dset);
        }

        H5Pclose(dcpl);
        H5Sclose(space);

        cudaDeviceSynchronize();
        total_original += field_bytes;
    }

    H5Fclose(fid);

    /* Verification: read back each dataset and bitwise compare */
    if (verify) {
        hid_t vfid = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
        if (vfid >= 0) {
            for (size_t i = 0; i < field_ptrs.size(); ++i) {
                int vrc = verify_field(vfid, field_names[i].c_str(),
                                       field_ptrs[i], n_points, h5type);
                if (vrc != 0 && rank == 0) {
                    fprintf(stderr, "[GPUCompress] VERIFY FAILED: %s bitwise mismatch!\n",
                            field_names[i].c_str());
                }
            }
            H5Fclose(vfid);
        }
    }

    return total_original;
}

/**
 * Verify a single dataset within an open HDF5 file.
 * Returns 0 on bitwise match, -1 on mismatch.
 */
inline int verify_field(hid_t fid, const char* dset_name,
                        const void* d_original, size_t n_elements,
                        hid_t h5type)
{
    hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
    if (dset < 0) return -1;

    size_t elem_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4;
    size_t total_bytes = n_elements * elem_size;

    void* d_readback = nullptr;
    cudaMalloc(&d_readback, total_bytes);

    H5Dread(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    cudaDeviceSynchronize();

    H5Dclose(dset);

    std::vector<char> h_orig(total_bytes);
    std::vector<char> h_read(total_bytes);
    cudaMemcpy(h_orig.data(), d_original, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read.data(), d_readback, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_readback);

    return (memcmp(h_orig.data(), h_read.data(), total_bytes) == 0) ? 0 : -1;
}

/**
 * Print compression statistics.
 */
inline void print_stats(const std::string& label,
                        long original_bytes, long compressed_bytes,
                        double elapsed_ms = 0.0)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    double ratio = (compressed_bytes > 0)
                 ? (double)original_bytes / compressed_bytes : 0.0;
    double orig_mb = original_bytes / (1024.0 * 1024.0);
    double comp_mb = compressed_bytes / (1024.0 * 1024.0);

    printf("[GPUCompress] %s: %.1f MB -> %.1f MB (ratio %.1fx)",
           label.c_str(), orig_mb, comp_mb, ratio);
    if (elapsed_ms > 0.0)
        printf(" in %.1f ms (%.1f MB/s)", elapsed_ms, orig_mb / (elapsed_ms / 1000.0));
    printf("\n");
}

} /* namespace gpucompress_nekrs_bridge */

#endif /* NEKRS_GPUCOMPRESS_BRIDGE_HPP */
