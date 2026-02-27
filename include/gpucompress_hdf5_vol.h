/**
 * @file gpucompress_hdf5_vol.h
 * @brief Public API for the GPUCompress HDF5 VOL connector
 *
 * The GPUCompress VOL connector intercepts H5Dwrite()/H5Dread() calls with
 * GPU device pointers, compresses/decompresses chunks directly on the GPU,
 * and writes/reads pre-compressed bytes via H5Dwrite_chunk() — bypassing
 * the HDF5 filter pipeline and eliminating GPU→CPU→GPU round-trips.
 *
 * Usage:
 *   // Initialize gpucompress first
 *   gpucompress_init(weights_path);
 *
 *   // Create a file access property list with the VOL connector
 *   hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
 *   H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
 *
 *   // Create or open file with that FAPL
 *   hid_t file = H5Fcreate("out.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
 *
 *   // Create a chunked dataset with the gpucompress filter in its DCPL
 *   hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
 *   hsize_t chunk_dims[1] = {1024};
 *   H5Pset_chunk(dcpl, 1, chunk_dims);
 *   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 4, 0.0);
 *
 *   // Write from GPU pointer — VOL handles compression
 *   float* d_data;  // cudaMalloc'd GPU pointer
 *   H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
 */

#ifndef GPUCOMPRESS_HDF5_VOL_H
#define GPUCOMPRESS_HDF5_VOL_H

#include <hdf5.h>
#include "gpucompress.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * VOL Connector Identity
 * ============================================================ */

/** VOL connector value — user range starts at 512 */
#define H5VL_GPUCOMPRESS_VALUE   512

/** VOL connector name string */
#define H5VL_GPUCOMPRESS_NAME    "gpucompress"

/** VOL connector version */
#define H5VL_GPUCOMPRESS_VERSION 0

/* ============================================================
 * Registration and FAPL helpers
 * ============================================================ */

/**
 * Register the GPUCompress VOL connector with HDF5.
 *
 * Idempotent — safe to call multiple times.
 *
 * @return Connector ID (hid_t) on success, H5I_INVALID_HID on failure.
 *         Caller must call H5VLclose() on the returned ID when done.
 */
hid_t H5VL_gpucompress_register(void);

/**
 * Configure a file access property list to use the GPUCompress VOL connector.
 *
 * @param fapl_id         File access property list to configure
 * @param under_vol_id    Underlying VOL connector ID (H5VL_NATIVE for native HDF5)
 * @param under_vol_info  Info for the underlying connector (NULL for native)
 * @return Non-negative on success, negative on failure
 */
herr_t H5Pset_fapl_gpucompress(hid_t fapl_id, hid_t under_vol_id,
                                const void *under_vol_info);

/* ============================================================
 * Activity counters — for testing / verification
 * ============================================================ */

/**
 * Reset all VOL activity counters to zero.
 * Call before an operation to get a clean count.
 */
void H5VL_gpucompress_reset_stats(void);

/**
 * Enable (on=1) or disable (on=0) call-sequence tracing.
 * When enabled, every VOL callback and every underlying H5VL* native call
 * is printed to stdout with a [VOL] prefix, indented to show call depth.
 */
void H5VL_gpucompress_set_trace(int on);

/**
 * Read VOL activity counters.
 * Any pointer may be NULL if that value is not needed.
 *
 * @param writes  Number of times the GPU write path was entered
 * @param reads   Number of times the GPU read path was entered
 * @param comp    Number of chunks successfully compressed on GPU
 * @param decomp  Number of chunks successfully decompressed on GPU
 */
void H5VL_gpucompress_get_stats(int *writes, int *reads,
                                 int *comp,   int *decomp);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_HDF5_VOL_H */
