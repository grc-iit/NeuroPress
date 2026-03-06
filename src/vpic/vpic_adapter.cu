/**
 * @file vpic_adapter.cu
 * @brief Host driver and C API for VPIC-Kokkos data adapter
 *
 * Borrows device pointers from VPIC-Kokkos Kokkos::Views and passes
 * them to gpucompress_compress_gpu(). No simulation logic lives here.
 */

#include <cstdio>
#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

#include "gpucompress_vpic.h"

/* Variable counts per data type */
static const int VPIC_FIELDS_NVARS    = 16;
static const int VPIC_HYDRO_NVARS     = 14;
static const int VPIC_PARTICLES_NVARS =  7;  /* float vars; +1 int (species) */

/* ============================================================
 * Internal struct
 * ============================================================ */

struct gpucompress_vpic {
    VpicSettings settings;
    float* d_data;        /* borrowed float device pointer */
    int*   d_data_i;      /* borrowed int device pointer (particles only) */
    size_t n_elements;    /* cells or particles */
    int    n_vars;        /* float variables per element */
    size_t nbytes_float;  /* n_elements * n_vars * sizeof(float) */
    size_t nbytes_int;    /* n_elements * sizeof(int) for particles, else 0 */
};

static int vpic_nvars_for_type(vpic_data_type_t dt)
{
    switch (dt) {
        case VPIC_DATA_FIELDS:    return VPIC_FIELDS_NVARS;
        case VPIC_DATA_HYDRO:     return VPIC_HYDRO_NVARS;
        case VPIC_DATA_PARTICLES: return VPIC_PARTICLES_NVARS;
        default:                  return -1;
    }
}

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_error_t gpucompress_vpic_create(
    gpucompress_vpic_t* handle,
    const VpicSettings* settings)
{
    if (!handle || !settings) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    int nv = vpic_nvars_for_type(settings->data_type);
    if (nv < 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_vpic* adapter = new (std::nothrow) gpucompress_vpic;
    if (!adapter) return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;

    adapter->settings     = *settings;
    adapter->d_data       = NULL;
    adapter->d_data_i     = NULL;
    adapter->n_elements   = 0;
    adapter->n_vars       = nv;
    adapter->nbytes_float = 0;
    adapter->nbytes_int   = 0;

    *handle = adapter;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_vpic_destroy(gpucompress_vpic_t handle)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    /* We don't own the device memory — just delete the handle */
    delete handle;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_vpic_attach(
    gpucompress_vpic_t handle,
    float* d_data,
    int*   d_data_i,
    size_t n_elements)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (!d_data && n_elements > 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    handle->d_data       = d_data;
    handle->d_data_i     = d_data_i;
    handle->n_elements   = n_elements;
    handle->nbytes_float = n_elements * (size_t)handle->n_vars * sizeof(float);

    if (handle->settings.data_type == VPIC_DATA_PARTICLES)
        handle->nbytes_int = n_elements * sizeof(int);
    else
        handle->nbytes_int = 0;

    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_vpic_get_device_ptrs(
    gpucompress_vpic_t handle,
    float** d_data,
    int**   d_data_i,
    size_t* nbytes_float,
    size_t* nbytes_int)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    if (d_data)       *d_data       = handle->d_data;
    if (d_data_i)     *d_data_i     = handle->d_data_i;
    if (nbytes_float) *nbytes_float = handle->nbytes_float;
    if (nbytes_int)   *nbytes_int   = handle->nbytes_int;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_vpic_get_nbytes(
    gpucompress_vpic_t handle,
    size_t* nbytes)
{
    if (!handle || !nbytes) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *nbytes = handle->nbytes_float + handle->nbytes_int;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_compress_vpic(
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
