/**
 * @file nekrs_adapter.cu
 * @brief Host driver and C API for nekRS/OCCA data adapter
 *
 * Borrows device pointers from nekRS OCCA memory objects and passes
 * them to gpucompress_compress_gpu(). No simulation logic lives here.
 *
 * Follows the same pattern as vpic_adapter.cu and nyx_adapter.cu.
 */

#include <cstdio>
#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

#include "gpucompress_nekrs.h"

/* ============================================================
 * Internal struct
 * ============================================================ */

struct gpucompress_nekrs {
    NekrsSettings settings;
    const void*   d_data;      /* borrowed device pointer (from occa::memory::ptr()) */
    size_t        n_points;    /* mesh points in attached field */
    size_t        nbytes;      /* n_points * n_components * element_size */
};

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_nekrs_t gpucompress_nekrs_create(const NekrsSettings* settings)
{
    if (!settings) return NULL;
    if (settings->n_components <= 0) return NULL;
    if (settings->element_size != 4 && settings->element_size != 8)
        return NULL;

    gpucompress_nekrs* adapter = new (std::nothrow) gpucompress_nekrs;
    if (!adapter) return NULL;

    adapter->settings = *settings;
    adapter->d_data   = NULL;
    adapter->n_points = 0;
    adapter->nbytes   = 0;

    return adapter;
}

void gpucompress_nekrs_destroy(gpucompress_nekrs_t handle)
{
    /* We don't own the device memory — just delete the handle */
    delete handle;
}

void gpucompress_nekrs_attach(gpucompress_nekrs_t handle,
                              const void* d_ptr,
                              size_t n_points)
{
    if (!handle) return;

    handle->d_data   = d_ptr;
    handle->n_points = n_points;
    handle->nbytes   = n_points
                     * (size_t)handle->settings.n_components
                     * (size_t)handle->settings.element_size;
}

const void* gpucompress_nekrs_get_device_ptr(gpucompress_nekrs_t handle)
{
    return handle ? handle->d_data : NULL;
}

size_t gpucompress_nekrs_get_nbytes(gpucompress_nekrs_t handle)
{
    return handle ? handle->nbytes : 0;
}

gpucompress_error_t gpucompress_compress_nekrs(
    gpucompress_nekrs_t handle,
    void** d_compressed,
    size_t* comp_bytes,
    gpucompress_algorithm_t algo,
    double error_bound)
{
    if (!handle || !d_compressed || !comp_bytes)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (!handle->d_data || handle->nbytes == 0)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm     = algo;
    config.error_bound   = error_bound;
    config.preprocessing = GPUCOMPRESS_PREPROC_NONE;
    if (handle->settings.element_size == 4)
        config.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

    gpucompress_stats_t stats;

    return gpucompress_compress_gpu(
        handle->d_data, handle->nbytes,
        *d_compressed, comp_bytes,
        &config, &stats, NULL);
}

} /* extern "C" */
