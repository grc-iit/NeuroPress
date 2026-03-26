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
 * Get additive pipeline timing from the last H5Dwrite call (ms).
 * stage1_ms:   Stats + NN inference (main thread, sequential per chunk)
 * drain_ms:    Worker drain (S1 end → all compression workers joined)
 * io_drain_ms: I/O drain (workers joined → I/O thread joined, includes cleanup)
 * total_ms:    End-to-end pipeline wall clock
 * Additive: stage1_ms + drain_ms + io_drain_ms = total_ms
 */
void H5VL_gpucompress_get_stage_timing(double *stage1_ms, double *drain_ms,
                                        double *io_drain_ms, double *total_ms);

/**
 * Get actual busy time for overlapping stages (ms).
 * s2_busy_ms: Bottleneck worker wall time (compress + cudaFree + pool + D→H + I/O post)
 * s3_busy_ms: Total I/O write time (serial writes, accumulated across all chunks)
 * These overlap with stage1 and drain — NOT additive with them.
 * Use for bottleneck analysis: if s2_busy > stage1, compression is slower than
 * inference; if io_drain is large, I/O couldn't keep up.
 */
void H5VL_gpucompress_get_busy_timing(double *s2_busy_ms, double *s3_busy_ms);

/**
 * Get VOL function-level timing: setup (VolWriteCtx alloc + thread creation)
 * and total gpu_aware_chunked_write wall clock.
 */
void H5VL_gpucompress_get_vol_func_timing(double *setup_ms, double *vol_func_ms);

/** Release cached VOL pipeline buffers (device + pinned host).
 *  Called automatically by gpucompress_cleanup(). Can also be called
 *  explicitly to free ~1.7GB of cached memory between benchmark phases. */
void H5VL_gpucompress_release_buf_cache(void);

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

/**
 * Read VOL memory transfer counters (tracked across all cudaMemcpy calls
 * issued by the VOL connector itself — does not include transfers inside
 * gpucompress_compress_gpu / gpucompress_decompress_gpu).
 * Any pointer may be NULL.
 */
void H5VL_gpucompress_get_transfer_stats(
    int    *h2d_count, size_t *h2d_bytes,
    int    *d2h_count, size_t *d2h_bytes,
    int    *d2d_count, size_t *d2d_bytes);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_HDF5_VOL_H */
