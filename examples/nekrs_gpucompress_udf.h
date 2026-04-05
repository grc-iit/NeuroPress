/**
 * @file nekrs_gpucompress_udf.h
 * @brief Simple C interface for calling GPUCompress from a nekRS UDF
 *
 * This header provides a minimal C API that a nekRS .udf file can call.
 * The implementation lives in nekrs_gpucompress_udf.cpp, which is compiled
 * separately and linked via NEKRS_UDF_LIBS or udf.cmake.
 *
 * The UDF extracts raw CUDA device pointers from OCCA memory objects
 * via occa::memory::ptr<void>() and passes them through this API.
 * GPUCompress's HDF5 VOL connector handles compression transparently.
 */

#ifndef NEKRS_GPUCOMPRESS_UDF_H
#define NEKRS_GPUCOMPRESS_UDF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize GPUCompress + HDF5 VOL connector.
 * Call once in UDF_Setup().
 *
 * @param weights_path  Path to NN weights file (e.g. "best_model.pt")
 * @param policy        Ranking policy: "speed", "balanced", or "ratio"
 * @return 0 on success, -1 on failure
 */
int gpucompress_nekrs_init(const char* weights_path, const char* policy);

/**
 * Write a single GPU-resident field to compressed HDF5.
 *
 * @param filename      Output HDF5 file path
 * @param dset_name     Dataset name within the file
 * @param d_ptr         CUDA device pointer (from occa::memory::ptr<void>())
 * @param n_elements    Number of elements
 * @param elem_bytes    Bytes per element (4 for float, 8 for double)
 * @param algo_name     Algorithm: "lz4","snappy","deflate","gdeflate",
 *                      "zstd","ans","cascaded","bitcomp","auto"
 * @param error_bound   0.0 for lossless
 * @param verify        If nonzero, read back and verify bitwise
 * @return 0 on success, -1 on failure
 */
int gpucompress_nekrs_write_field(const char* filename,
                                   const char* dset_name,
                                   const void* d_ptr,
                                   size_t n_elements,
                                   int elem_bytes,
                                   const char* algo_name,
                                   double error_bound,
                                   int verify);

/**
 * Finalize GPUCompress. Call at program end (optional).
 */
void gpucompress_nekrs_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* NEKRS_GPUCOMPRESS_UDF_H */
