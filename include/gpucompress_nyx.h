/**
 * @file gpucompress_nyx.h
 * @brief Nyx/AMReX data adapter C API
 *
 * Thin data-access layer that borrows GPU device pointers from AMReX
 * MultiFab FArrayBoxes and passes them to gpucompress_compress_gpu().
 * The adapter does NOT own GPU memory — it borrows pointers via attach().
 *
 * Follows the same pattern as gpucompress_vpic.h.
 */

#ifndef GPUCOMPRESS_NYX_H
#define GPUCOMPRESS_NYX_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Data types
 * ============================================================ */

/** Nyx data categories. */
typedef enum {
    NYX_DATA_HYDRO_STATE = 0,  /**< Hydro state: density, momenta, energy, species */
    NYX_DATA_DIAG_STATE  = 1,  /**< Diagnostic: temperature, Ne, etc. */
    NYX_DATA_GRAV_PHI    = 2,  /**< Gravitational potential */
    NYX_DATA_PARTICLES   = 3   /**< DM/AGN/neutrino particles */
} nyx_data_type_t;

/* ============================================================
 * Settings
 * ============================================================ */

/** Nyx adapter parameters. */
typedef struct {
    nyx_data_type_t data_type;    /**< Which Nyx data category */
    int             n_components; /**< Number of variables (e.g. 8 for hydro state) */
    int             element_size; /**< Bytes per element: 8 for double, 4 for float */
} NyxSettings;

/** Return default settings (hydro state, 0 components, double precision). */
static inline NyxSettings nyx_default_settings(void)
{
    NyxSettings s;
    s.data_type    = NYX_DATA_HYDRO_STATE;
    s.n_components = 0;
    s.element_size = 8;  /* AMReX Real = double by default */
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle to a Nyx adapter instance. */
typedef struct gpucompress_nyx* gpucompress_nyx_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Create a Nyx adapter instance.
 *
 * @param handle   Output: opaque handle
 * @param settings Adapter parameters
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_nyx_create(
    gpucompress_nyx_t* handle,
    const NyxSettings* settings);

/**
 * Destroy a Nyx adapter instance.
 */
gpucompress_error_t gpucompress_nyx_destroy(
    gpucompress_nyx_t handle);

/**
 * Attach a borrowed device pointer from an AMReX FArrayBox.
 *
 * @param handle     Adapter handle
 * @param d_data     Device pointer to FArrayBox data (fab.dataPtr())
 * @param n_cells    Number of cells in this FArrayBox
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_nyx_attach(
    gpucompress_nyx_t handle,
    void* d_data,
    size_t n_cells);

/* ============================================================
 * Data access
 * ============================================================ */

/**
 * Get the currently attached device pointer and byte size.
 *
 * @param handle  Adapter handle
 * @param d_data  Output: device pointer (can be NULL)
 * @param nbytes  Output: total data size in bytes (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_nyx_get_device_ptr(
    gpucompress_nyx_t handle,
    void** d_data,
    size_t* nbytes);

/**
 * Return total data size in bytes.
 */
gpucompress_error_t gpucompress_nyx_get_nbytes(
    gpucompress_nyx_t handle,
    size_t* nbytes);

/* ============================================================
 * Convenience: compress in one call
 * ============================================================ */

/**
 * Compress Nyx data directly from GPU device pointer.
 *
 * @param d_data      Device pointer to input data
 * @param nbytes      Size of input data in bytes
 * @param d_output    Pre-allocated GPU output buffer for compressed data
 * @param output_size [in] max size, [out] actual compressed size
 * @param config      Compression configuration
 * @param stats       Optional compression stats (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_nyx(
    const void* d_data,
    size_t nbytes,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_NYX_H */
