/**
 * @file nyx_adapter.cu
 * @brief Host driver and C API for Nyx/AMReX data adapter
 *
 * Borrows device pointers from AMReX MultiFab FArrayBoxes and passes
 * them to gpucompress_compress_gpu(). No simulation logic lives here.
 *
 * Follows the same pattern as vpic_adapter.cu.
 */

#include <cstdio>
#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

#include "gpucompress_nyx.h"

/* ============================================================
 * Internal struct
 * ============================================================ */

struct gpucompress_nyx {
    NyxSettings settings;
    void*  d_data;       /* borrowed device pointer */
    size_t n_cells;      /* cells in attached FArrayBox */
    size_t nbytes;       /* n_cells * n_components * element_size */
};

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_error_t gpucompress_nyx_create(
    gpucompress_nyx_t* handle,
    const NyxSettings* settings)
{
    if (!handle || !settings) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (settings->n_components <= 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (settings->element_size != 4 && settings->element_size != 8)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_nyx* adapter = new (std::nothrow) gpucompress_nyx;
    if (!adapter) return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;

    adapter->settings = *settings;
    adapter->d_data   = NULL;
    adapter->n_cells  = 0;
    adapter->nbytes   = 0;

    *handle = adapter;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_nyx_destroy(gpucompress_nyx_t handle)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    /* We don't own the device memory — just delete the handle */
    delete handle;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_nyx_attach(
    gpucompress_nyx_t handle,
    void* d_data,
    size_t n_cells)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (!d_data && n_cells > 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    handle->d_data  = d_data;
    handle->n_cells = n_cells;
    handle->nbytes  = n_cells
                    * (size_t)handle->settings.n_components
                    * (size_t)handle->settings.element_size;

    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_nyx_get_device_ptr(
    gpucompress_nyx_t handle,
    void** d_data,
    size_t* nbytes)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    if (d_data) *d_data = handle->d_data;
    if (nbytes) *nbytes = handle->nbytes;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_nyx_get_nbytes(
    gpucompress_nyx_t handle,
    size_t* nbytes)
{
    if (!handle || !nbytes) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *nbytes = handle->nbytes;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_compress_nyx(
    const void* d_data,
    size_t nbytes,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats)
{
    if (!d_data || !d_output || !output_size)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    return gpucompress_compress_gpu(
        d_data, nbytes,
        d_output, output_size,
        config, stats, NULL);
}

} /* extern "C" */
