/**
 * @file gpucompress_hdf5.h
 * @brief HDF5 Filter Plugin for GPUCompress
 *
 * This header provides the HDF5 filter interface for GPUCompress.
 * The filter can be used with H5Pset_filter() or loaded as a plugin.
 *
 * Usage:
 *   // Register filter (automatic if using plugin)
 *   H5Z_gpucompress_register();
 *
 *   // Set filter on dataset creation property list
 *   hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
 *   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);
 *
 *   // Or use H5Pset_filter directly
 *   unsigned int cd_values[5] = {0, 0, 4, 0, 0};  // algo, preproc, shuffle, eb_lo, eb_hi
 *   H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, 5, cd_values);
 */

#ifndef GPUCOMPRESS_HDF5_H
#define GPUCOMPRESS_HDF5_H

#include "gpucompress.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Only include HDF5 types if HDF5 is available */
#ifdef H5_VERS_MAJOR
#include <hdf5.h>
#else
/* Forward declarations for when HDF5 headers not included */
typedef int hid_t;
typedef int herr_t;
typedef int htri_t;
#endif

/* ============================================================
 * Filter Constants
 * ============================================================ */

/**
 * GPUCompress filter ID.
 *
 * Uses testing range (256-511) for development.
 * For production, request official ID from HDF Group.
 */
#define H5Z_FILTER_GPUCOMPRESS 305

/**
 * Number of filter parameters (cd_values elements).
 *
 * Layout:
 *   cd_values[0]: algorithm (gpucompress_algorithm_t)
 *   cd_values[1]: preprocessing flags
 *   cd_values[2]: shuffle element size (0, 2, 4, or 8)
 *   cd_values[3]: error_bound low bits (uint32_t)
 *   cd_values[4]: error_bound high bits (uint32_t)
 */
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/**
 * Filter name for debugging/display.
 */
#define H5Z_GPUCOMPRESS_NAME "gpucompress"

/* ============================================================
 * Filter Registration
 * ============================================================ */

/**
 * Register the GPUCompress filter with HDF5.
 *
 * Call once before using the filter. Automatically initializes
 * the GPUCompress library if not already initialized.
 *
 * If using the plugin mechanism (libH5Zgpucompress.so), registration
 * is automatic and this function does not need to be called.
 *
 * @return >= 0 on success, < 0 on error
 */
herr_t H5Z_gpucompress_register(void);

/**
 * Check if GPUCompress filter is registered.
 *
 * @return 1 if registered, 0 if not, < 0 on error
 */
htri_t H5Z_gpucompress_is_registered(void);

/* ============================================================
 * Convenience Functions
 * ============================================================ */

/**
 * Set GPUCompress filter on a dataset creation property list.
 *
 * This is a convenience function that wraps H5Pset_filter() with
 * proper parameter encoding.
 *
 * @param plist_id      Dataset creation property list
 * @param algorithm     Compression algorithm (GPUCOMPRESS_ALGO_*)
 * @param preprocessing Preprocessing flags (GPUCOMPRESS_PREPROC_*)
 * @param shuffle_size  Shuffle element size (0, 2, 4, or 8)
 * @param error_bound   Quantization error bound (0.0 for lossless)
 * @return >= 0 on success, < 0 on error
 *
 * Example:
 *   // Auto algorithm selection, 4-byte shuffle, lossless
 *   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);
 *
 *   // ZSTD with quantization (lossy)
 *   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_ZSTD,
 *                      GPUCOMPRESS_PREPROC_QUANTIZE, 4, 0.001);
 */
herr_t H5Pset_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t algorithm,
    unsigned int preprocessing,
    unsigned int shuffle_size,
    double error_bound
);

/**
 * Get GPUCompress filter parameters from a property list.
 *
 * @param plist_id      Dataset creation property list
 * @param algorithm     Output: compression algorithm
 * @param preprocessing Output: preprocessing flags
 * @param shuffle_size  Output: shuffle element size
 * @param error_bound   Output: error bound
 * @return >= 0 on success, < 0 on error or filter not set
 */
herr_t H5Pget_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t* algorithm,
    unsigned int* preprocessing,
    unsigned int* shuffle_size,
    double* error_bound
);

/* ============================================================
 * Filter Information
 * ============================================================ */

/**
 * Get filter information for registration.
 *
 * This is primarily used internally or by the plugin mechanism.
 * Returns pointer to the H5Z_class2_t structure.
 *
 * @return Pointer to filter class structure
 */
const void* H5Z_gpucompress_get_filter_info(void);

/* ============================================================
 * Plugin Entry Points (for dynamic loading)
 * ============================================================ */

#ifdef GPUCOMPRESS_BUILD_HDF5_PLUGIN

/**
 * HDF5 plugin type (required export for plugins).
 *
 * @return H5PL_TYPE_FILTER
 */
H5PL_type_t H5PLget_plugin_type(void);

/**
 * HDF5 plugin info (required export for plugins).
 *
 * @return Pointer to H5Z_class2_t structure
 */
const void* H5PLget_plugin_info(void);

#endif /* GPUCOMPRESS_BUILD_HDF5_PLUGIN */

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_HDF5_H */
