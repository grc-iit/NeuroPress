/**
 * @file H5Zgpucompress.h
 * @brief HDF5 Filter Plugin Header for GPUCompress
 *
 * This header is the internal version used when building the plugin.
 * For external use, include gpucompress_hdf5.h from the include directory.
 */

#ifndef H5ZGPUCOMPRESS_H
#define H5ZGPUCOMPRESS_H

#include <hdf5.h>
#include "gpucompress.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Filter ID */
#define H5Z_FILTER_GPUCOMPRESS 305

/* Number of cd_values */
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Filter name */
#define H5Z_GPUCOMPRESS_NAME "gpucompress"

/* Registration */
herr_t H5Z_gpucompress_register(void);
htri_t H5Z_gpucompress_is_registered(void);

/* Convenience functions */
herr_t H5Pset_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t algorithm,
    unsigned int preprocessing,
    unsigned int shuffle_size,
    double error_bound
);

herr_t H5Pget_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t* algorithm,
    unsigned int* preprocessing,
    unsigned int* shuffle_size,
    double* error_bound
);

/* Filter info */
const void* H5Z_gpucompress_get_filter_info(void);

#ifdef __cplusplus
}
#endif

#endif /* H5ZGPUCOMPRESS_H */
