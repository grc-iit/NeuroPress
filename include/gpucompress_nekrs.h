/**
 * @file gpucompress_nekrs.h
 * @brief nekRS/OCCA data adapter C API
 *
 * Thin data-access layer that borrows GPU device pointers from nekRS
 * OCCA memory objects and passes them to gpucompress_compress_gpu().
 * The adapter does NOT own GPU memory — it borrows pointers via attach().
 *
 * Follows the same pattern as gpucompress_vpic.h and gpucompress_nyx.h.
 */

#ifndef GPUCOMPRESS_NEKRS_H
#define GPUCOMPRESS_NEKRS_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Data types
 * ============================================================ */

/** nekRS field categories. */
typedef enum {
    NEKRS_DATA_VELOCITY = 0,  /**< Velocity field (u, v, w components) */
    NEKRS_DATA_PRESSURE = 1,  /**< Pressure field */
    NEKRS_DATA_SCALAR   = 2,  /**< Scalar fields (temperature, species, etc.) */
    NEKRS_DATA_CUSTOM   = 3   /**< User-defined fields (Q-criterion, etc.) */
} nekrs_data_type_t;

/* ============================================================
 * Settings
 * ============================================================ */

/** nekRS adapter parameters. */
typedef struct {
    nekrs_data_type_t data_type;    /**< Which nekRS data category */
    int               n_components; /**< Number of field components (3 for velocity) */
    int               element_size; /**< Bytes per element: 4 for float, 8 for double */
} NekrsSettings;

/** Return default settings (velocity, 3 components, float32). */
static inline NekrsSettings nekrs_default_settings(void)
{
    NekrsSettings s;
    s.data_type    = NEKRS_DATA_VELOCITY;
    s.n_components = 3;
    s.element_size = 4;  /* dfloat = float by default in nekRS */
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle returned by create(). */
typedef struct gpucompress_nekrs* gpucompress_nekrs_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Create a nekRS adapter.
 * @param settings  Adapter configuration
 * @return Handle, or NULL on failure
 */
gpucompress_nekrs_t gpucompress_nekrs_create(const NekrsSettings* settings);

/**
 * Destroy the adapter and free internal bookkeeping (NOT the borrowed buffer).
 */
void gpucompress_nekrs_destroy(gpucompress_nekrs_t handle);

/* ============================================================
 * Data attachment (borrow, not own)
 * ============================================================ */

/**
 * Attach an OCCA device pointer (obtained via occa::memory::ptr<void>()).
 * The adapter borrows this pointer — caller is responsible for lifetime.
 *
 * @param handle    Adapter handle
 * @param d_ptr     GPU device pointer from occa::memory::ptr()
 * @param n_points  Number of mesh points (e.g. mesh->Nlocal)
 */
void gpucompress_nekrs_attach(gpucompress_nekrs_t handle,
                              const void* d_ptr,
                              size_t n_points);

/**
 * Get the borrowed device pointer.
 */
const void* gpucompress_nekrs_get_device_ptr(gpucompress_nekrs_t handle);

/**
 * Get the total number of bytes for the attached field.
 */
size_t gpucompress_nekrs_get_nbytes(gpucompress_nekrs_t handle);

/* ============================================================
 * Compression (forwards to gpucompress_compress_gpu)
 * ============================================================ */

/**
 * Compress the attached nekRS field data.
 * Forwards to gpucompress_compress_gpu() with the borrowed device pointer.
 *
 * @param handle        Adapter handle
 * @param d_compressed  Output: device pointer for compressed data
 * @param comp_bytes    Output: size of compressed data
 * @param algo          Compression algorithm (GPUCOMPRESS_ALGO_AUTO for NN selection)
 * @param error_bound   0.0 for lossless, >0 for lossy
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_nekrs(
    gpucompress_nekrs_t handle,
    void** d_compressed,
    size_t* comp_bytes,
    gpucompress_algorithm_t algo,
    double error_bound);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_NEKRS_H */
