/**
 * @file gpucompress_grayscott.h
 * @brief GPU Gray-Scott reaction-diffusion simulation C API
 *
 * Generates 3D scientific data (species U and V on an L^3 grid) entirely
 * on the GPU.  Data stays GPU-resident so it can feed directly into
 * gpucompress_compress_gpu() with zero D2H transfers.
 */

#ifndef GPUCOMPRESS_GRAYSCOTT_H
#define GPUCOMPRESS_GRAYSCOTT_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Settings
 * ============================================================ */

/** Gray-Scott simulation parameters. */
typedef struct {
    int   L;       /**< Grid dimension (L x L x L).  Default 128. */
    float Du;      /**< Diffusion coefficient for U.  Default 0.05 */
    float Dv;      /**< Diffusion coefficient for V.  Default 0.1  */
    float F;       /**< Feed rate.                    Default 0.04 */
    float k;       /**< Kill rate.                    Default 0.06075 */
    float dt;      /**< Time-step size.               Default 0.2  */
    float noise;   /**< Noise amplitude on U.         Default 0.0  */
    int   steps;   /**< Total simulation steps.       Default 10000 */
    int   plotgap; /**< Output every N steps.          Default 10   */
    int   seed;    /**< RNG seed.                     Default 42   */
} GrayScottSettings;

/** Return default settings matching the reference implementation. */
static inline GrayScottSettings gray_scott_default_settings(void)
{
    GrayScottSettings s;
    s.L       = 128;
    s.Du      = 0.05f;
    s.Dv      = 0.1f;
    s.F       = 0.04f;
    s.k       = 0.06075f;
    s.dt      = 0.2f;
    s.noise   = 0.0f;
    s.steps   = 10000;
    s.plotgap = 10;
    s.seed    = 42;
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle to a Gray-Scott simulation instance. */
typedef struct gpucompress_grayscott* gpucompress_grayscott_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Create a Gray-Scott simulation instance.
 *
 * @param handle   Output: opaque handle
 * @param settings Simulation parameters
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_grayscott_create(
    gpucompress_grayscott_t* handle,
    const GrayScottSettings* settings);

/**
 * Destroy a simulation instance and free GPU memory.
 */
gpucompress_error_t gpucompress_grayscott_destroy(
    gpucompress_grayscott_t handle);

/**
 * Initialize fields (U=1, V=0 with center perturbation).
 */
gpucompress_error_t gpucompress_grayscott_init(
    gpucompress_grayscott_t handle);

/**
 * Run the simulation for @p steps time-steps.
 */
gpucompress_error_t gpucompress_grayscott_run(
    gpucompress_grayscott_t handle,
    int steps);

/* ============================================================
 * Data access
 * ============================================================ */

/**
 * Get raw device pointers to the current U and V fields.
 * Pointers remain valid until destroy().
 */
gpucompress_error_t gpucompress_grayscott_get_device_ptrs(
    gpucompress_grayscott_t handle,
    float** d_u,
    float** d_v);

/**
 * Copy current fields to pre-allocated host buffers.
 * Each buffer must hold at least get_nbytes() bytes.
 */
gpucompress_error_t gpucompress_grayscott_copy_to_host(
    gpucompress_grayscott_t handle,
    float* h_u,
    float* h_v);

/** Return the current simulation step counter. */
gpucompress_error_t gpucompress_grayscott_get_step(
    gpucompress_grayscott_t handle,
    int* step);

/** Return L (grid side length). */
gpucompress_error_t gpucompress_grayscott_get_L(
    gpucompress_grayscott_t handle,
    int* L);

/** Return size of one field in bytes (L^3 * sizeof(float)). */
gpucompress_error_t gpucompress_grayscott_get_nbytes(
    gpucompress_grayscott_t handle,
    size_t* nbytes);

/* ============================================================
 * Convenience: simulate + compress in one call
 * ============================================================ */

/**
 * Generate Gray-Scott V field and compress it, all GPU-resident.
 *
 * Creates sim -> init -> run(settings->steps) -> gets d_v ->
 * gpucompress_compress_gpu() -> destroys sim.
 *
 * @param settings       Simulation parameters
 * @param d_output       Pre-allocated GPU output buffer for compressed data
 * @param output_size    [in] max size, [out] actual compressed size
 * @param config         Compression configuration
 * @param stats          Optional compression stats (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_generate_grayscott(
    const GrayScottSettings* settings,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_GRAYSCOTT_H */
