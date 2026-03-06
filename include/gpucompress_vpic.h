/**
 * @file gpucompress_vpic.h
 * @brief VPIC-Kokkos data adapter C API
 *
 * Thin data-access layer that borrows GPU device pointers from VPIC-Kokkos
 * Kokkos::Views and passes them to gpucompress_compress_gpu().
 * The adapter does NOT own GPU memory — it borrows pointers via attach().
 */

#ifndef GPUCOMPRESS_VPIC_H
#define GPUCOMPRESS_VPIC_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Data types
 * ============================================================ */

/** VPIC data categories and their variable counts. */
typedef enum {
    VPIC_DATA_FIELDS    = 0,  /**< EM fields: 16 float variables */
    VPIC_DATA_HYDRO     = 1,  /**< Hydro moments: 14 float variables */
    VPIC_DATA_PARTICLES = 2   /**< Particles: 7 floats + 1 int per particle */
} vpic_data_type_t;

/* ============================================================
 * Settings
 * ============================================================ */

/** VPIC adapter parameters. */
typedef struct {
    vpic_data_type_t data_type;   /**< Which VPIC data category */
    size_t           n_cells;     /**< Number of grid cells (fields/hydro) */
    size_t           n_particles; /**< Number of particles (particles only) */
} VpicSettings;

/** Return default settings (fields, 0 cells, 0 particles). */
static inline VpicSettings vpic_default_settings(void)
{
    VpicSettings s;
    s.data_type   = VPIC_DATA_FIELDS;
    s.n_cells     = 0;
    s.n_particles = 0;
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle to a VPIC adapter instance. */
typedef struct gpucompress_vpic* gpucompress_vpic_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Create a VPIC adapter instance.
 *
 * @param handle   Output: opaque handle
 * @param settings Adapter parameters
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_vpic_create(
    gpucompress_vpic_t* handle,
    const VpicSettings* settings);

/**
 * Destroy a VPIC adapter instance.
 */
gpucompress_error_t gpucompress_vpic_destroy(
    gpucompress_vpic_t handle);

/**
 * Attach borrowed device pointers to the adapter (no ownership transfer).
 *
 * @param handle     Adapter handle
 * @param d_data     Device pointer to float data (fields/hydro/particle floats)
 * @param d_data_i   Device pointer to int data (particle species; NULL for fields/hydro)
 * @param n_elements Number of elements (cells for fields/hydro, particles for particles)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_vpic_attach(
    gpucompress_vpic_t handle,
    float* d_data,
    int*   d_data_i,
    size_t n_elements);

/* ============================================================
 * Data access
 * ============================================================ */

/**
 * Get the currently attached device pointers and byte sizes.
 *
 * @param handle      Adapter handle
 * @param d_data      Output: float device pointer (can be NULL)
 * @param d_data_i    Output: int device pointer (can be NULL)
 * @param nbytes_float Output: float data size in bytes (can be NULL)
 * @param nbytes_int  Output: int data size in bytes (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_vpic_get_device_ptrs(
    gpucompress_vpic_t handle,
    float** d_data,
    int**   d_data_i,
    size_t* nbytes_float,
    size_t* nbytes_int);

/**
 * Return total data size in bytes (float + int combined).
 */
gpucompress_error_t gpucompress_vpic_get_nbytes(
    gpucompress_vpic_t handle,
    size_t* nbytes);

/* ============================================================
 * Convenience: compress in one call
 * ============================================================ */

/**
 * Compress VPIC data directly from GPU device pointer.
 *
 * @param d_data      Device pointer to input data
 * @param nbytes      Size of input data in bytes
 * @param d_output    Pre-allocated GPU output buffer for compressed data
 * @param output_size [in] max size, [out] actual compressed size
 * @param config      Compression configuration
 * @param stats       Optional compression stats (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_vpic(
    const void* d_data,
    size_t nbytes,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_VPIC_H */
