/**
 * @file H5Zgpucompress.c
 * @brief HDF5 Filter Plugin for GPUCompress
 *
 * Implements the HDF5 filter interface (H5Z_class2_t) for GPU-accelerated
 * compression using the GPUCompress library.
 *
 * This file can be compiled as:
 *   1. A plugin library (libH5Zgpucompress.so) loaded dynamically by HDF5
 *   2. Linked directly into an application
 *
 * Usage:
 *   // As plugin: set HDF5_PLUGIN_PATH environment variable
 *   export HDF5_PLUGIN_PATH=/path/to/plugins
 *
 *   // In application:
 *   H5Z_gpucompress_register();
 *   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <hdf5.h>
#include <H5PLextern.h>

#include "gpucompress.h"

/* ============================================================
 * Filter Constants
 * ============================================================ */

/** GPUCompress filter ID (testing range) */
#define H5Z_FILTER_GPUCOMPRESS 305

/** Number of filter parameters */
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/** Filter name */
#define H5Z_GPUCOMPRESS_NAME "gpucompress"

/* ============================================================
 * Forward Declarations
 * ============================================================ */

static htri_t H5Z_gpucompress_can_apply(hid_t dcpl_id, hid_t type_id, hid_t space_id);
static herr_t H5Z_gpucompress_set_local(hid_t dcpl_id, hid_t type_id, hid_t space_id);
static size_t H5Z_filter_gpucompress(unsigned int flags, size_t cd_nelmts,
                                      const unsigned int cd_values[],
                                      size_t nbytes, size_t *buf_size, void **buf);

/* ============================================================
 * Filter Class Definition
 * ============================================================ */

/**
 * Filter class structure for registration with HDF5.
 */
static const H5Z_class2_t H5Z_GPUCOMPRESS[1] = {{
    H5Z_CLASS_T_VERS,           /* H5Z_class_t version (must be H5Z_CLASS_T_VERS) */
    H5Z_FILTER_GPUCOMPRESS,     /* Filter ID number */
    1,                          /* encoder_present flag (can compress) */
    1,                          /* decoder_present flag (can decompress) */
    H5Z_GPUCOMPRESS_NAME,       /* Filter name for debugging */
    H5Z_gpucompress_can_apply,  /* can_apply callback (validates usage) */
    H5Z_gpucompress_set_local,  /* set_local callback (per-dataset setup) */
    H5Z_filter_gpucompress      /* filter function (compress/decompress) */
}};

/* ============================================================
 * Library State
 * ============================================================ */

/** Flag to track if GPUCompress library is initialized */
static int g_gpucompress_initialized = 0;

/** Flag to track if filter is registered */
static int g_filter_registered = 0;

/* ============================================================
 * Helper Functions
 * ============================================================ */

/**
 * Pack double into two unsigned int values.
 */
static void pack_double(double value, unsigned int* lo, unsigned int* hi) {
    union {
        double d;
        unsigned int u[2];
    } u;
    u.d = value;
    *lo = u.u[0];
    *hi = u.u[1];
}

/**
 * Unpack double from two unsigned int values.
 */
static double unpack_double(unsigned int lo, unsigned int hi) {
    union {
        double d;
        unsigned int u[2];
    } u;
    u.u[0] = lo;
    u.u[1] = hi;
    return u.d;
}

/**
 * Initialize GPUCompress library if not already done.
 */
static int ensure_initialized(void) {
    if (!g_gpucompress_initialized) {
        gpucompress_error_t err = gpucompress_init(NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            return -1;
        }
        g_gpucompress_initialized = 1;
    }
    return 0;
}

/* ============================================================
 * Filter Callbacks
 * ============================================================ */

/**
 * can_apply callback - validate filter can be used with this dataset.
 *
 * Called during H5Dcreate() to check if filter is applicable.
 *
 * @param dcpl_id  Dataset creation property list
 * @param type_id  Dataset datatype
 * @param space_id Dataset dataspace
 * @return Positive if applicable, 0 if not, negative on error
 */
static htri_t H5Z_gpucompress_can_apply(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
    (void)dcpl_id;
    (void)space_id;

    /* Check that we can initialize the library */
    if (ensure_initialized() < 0) {
        return -1;
    }

    /* Get datatype size for validation */
    size_t dtype_size = H5Tget_size(type_id);
    if (dtype_size == 0) {
        return -1;
    }

    /* GPUCompress works with any datatype */
    /* Could add size validation here if needed */

    return 1;  /* Filter can be applied */
}

/**
 * set_local callback - set filter parameters based on dataset.
 *
 * Called after can_apply to allow filter to adjust parameters
 * based on the specific dataset being created.
 *
 * @param dcpl_id  Dataset creation property list
 * @param type_id  Dataset datatype
 * @param space_id Dataset dataspace
 * @return Non-negative on success
 */
static herr_t H5Z_gpucompress_set_local(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
    (void)dcpl_id;
    (void)type_id;
    (void)space_id;

    /* Currently no dataset-specific parameter adjustment needed */
    /* Could potentially auto-detect shuffle size from dtype_size */

    return 0;
}

/**
 * Main filter function - compress or decompress data.
 *
 * @param flags      H5Z_FLAG_REVERSE set for decompression
 * @param cd_nelmts  Number of filter parameters
 * @param cd_values  Filter parameters [algo, preproc, shuffle, eb_lo, eb_hi]
 * @param nbytes     Number of valid bytes in input buffer
 * @param buf_size   Size of allocated buffer (may be updated)
 * @param buf        Data buffer pointer (may be reallocated)
 * @return Number of valid bytes in output, 0 on error
 */
static size_t H5Z_filter_gpucompress(
    unsigned int flags,
    size_t cd_nelmts,
    const unsigned int cd_values[],
    size_t nbytes,
    size_t *buf_size,
    void **buf
) {
    size_t result = 0;
    void* new_buf = NULL;
    size_t new_size;
    gpucompress_error_t err;

    /* Ensure library is initialized */
    if (ensure_initialized() < 0) {
        return 0;
    }

    /* Parse configuration from cd_values */
    gpucompress_config_t config = gpucompress_default_config();

    if (cd_nelmts >= 1) {
        config.algorithm = (gpucompress_algorithm_t)cd_values[0];
    }
    if (cd_nelmts >= 2) {
        config.preprocessing = cd_values[1];
    }
    if (cd_nelmts >= 3) {
        /* Override preprocessing with explicit shuffle size */
        unsigned int shuffle_size = cd_values[2];
        if (shuffle_size > 0) {
            config.preprocessing &= ~(GPUCOMPRESS_PREPROC_SHUFFLE_2 |
                                      GPUCOMPRESS_PREPROC_SHUFFLE_4 |
                                      GPUCOMPRESS_PREPROC_SHUFFLE_8);
            switch (shuffle_size) {
                case 2:
                    config.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_2;
                    break;
                case 4:
                    config.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                    break;
                case 8:
                    config.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_8;
                    break;
            }
        }
    }
    if (cd_nelmts >= 5) {
        config.error_bound = unpack_double(cd_values[3], cd_values[4]);
        if (config.error_bound > 0.0) {
            config.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
        }
    }

    if (flags & H5Z_FLAG_REVERSE) {
        /* ==================== DECOMPRESSION ==================== */

        /* Get original size from compressed header */
        size_t original_size;
        err = gpucompress_get_original_size(*buf, &original_size);
        if (err != GPUCOMPRESS_SUCCESS) {
            return 0;
        }

        /* Allocate output buffer */
        new_buf = H5allocate_memory(original_size, 0);
        if (new_buf == NULL) {
            return 0;
        }

        /* Decompress */
        new_size = original_size;
        err = gpucompress_decompress(*buf, nbytes, new_buf, &new_size);

        if (err != GPUCOMPRESS_SUCCESS) {
            H5free_memory(new_buf);
            return 0;
        }

        /* Replace buffer */
        H5free_memory(*buf);
        *buf = new_buf;
        *buf_size = original_size;
        result = new_size;

    } else {
        /* ==================== COMPRESSION ==================== */

        /* Calculate max output size */
        size_t max_out_size = gpucompress_max_compressed_size(nbytes);

        /* Allocate output buffer */
        new_buf = H5allocate_memory(max_out_size, 0);
        if (new_buf == NULL) {
            return 0;
        }

        /* Compress */
        new_size = max_out_size;
        err = gpucompress_compress(*buf, nbytes, new_buf, &new_size, &config, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            H5free_memory(new_buf);
            return 0;
        }

        /* Only use compressed if it's smaller */
        if (new_size >= nbytes) {
            /* Compression didn't help, return original */
            H5free_memory(new_buf);
            return nbytes;
        }

        /* Replace buffer with compressed data */
        H5free_memory(*buf);
        *buf = new_buf;
        *buf_size = max_out_size;
        result = new_size;
    }

    return result;
}

/* ============================================================
 * Filter Registration
 * ============================================================ */

/**
 * Register the GPUCompress filter with HDF5.
 */
herr_t H5Z_gpucompress_register(void) {
    herr_t ret;

    if (g_filter_registered) {
        return 0;  /* Already registered */
    }

    /* Initialize library */
    if (ensure_initialized() < 0) {
        return -1;
    }

    /* Register filter */
    ret = H5Zregister(H5Z_GPUCOMPRESS);
    if (ret >= 0) {
        g_filter_registered = 1;
    }

    return ret;
}

/**
 * Check if filter is registered.
 */
htri_t H5Z_gpucompress_is_registered(void) {
    return H5Zfilter_avail(H5Z_FILTER_GPUCOMPRESS);
}

/* ============================================================
 * Convenience Functions
 * ============================================================ */

/**
 * Set GPUCompress filter on a dataset creation property list.
 */
herr_t H5Pset_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t algorithm,
    unsigned int preprocessing,
    unsigned int shuffle_size,
    double error_bound
) {
    unsigned int cd_values[H5Z_GPUCOMPRESS_CD_NELMTS];

    /* Ensure filter is registered */
    if (H5Z_gpucompress_register() < 0) {
        return -1;
    }

    /* Pack parameters */
    cd_values[0] = (unsigned int)algorithm;
    cd_values[1] = preprocessing;
    cd_values[2] = shuffle_size;
    pack_double(error_bound, &cd_values[3], &cd_values[4]);

    /* Set filter */
    return H5Pset_filter(
        plist_id,
        H5Z_FILTER_GPUCOMPRESS,
        H5Z_FLAG_OPTIONAL,
        H5Z_GPUCOMPRESS_CD_NELMTS,
        cd_values
    );
}

/**
 * Get GPUCompress filter parameters from a property list.
 */
herr_t H5Pget_gpucompress(
    hid_t plist_id,
    gpucompress_algorithm_t* algorithm,
    unsigned int* preprocessing,
    unsigned int* shuffle_size,
    double* error_bound
) {
    unsigned int cd_values[H5Z_GPUCOMPRESS_CD_NELMTS];
    size_t cd_nelmts = H5Z_GPUCOMPRESS_CD_NELMTS;
    unsigned int flags;
    unsigned int filter_config;
    herr_t ret;

    /* Get filter info */
    ret = H5Pget_filter_by_id2(
        plist_id,
        H5Z_FILTER_GPUCOMPRESS,
        &flags,
        &cd_nelmts,
        cd_values,
        0,
        NULL,
        &filter_config
    );

    if (ret < 0) {
        return ret;
    }

    /* Unpack parameters */
    if (algorithm && cd_nelmts >= 1) {
        *algorithm = (gpucompress_algorithm_t)cd_values[0];
    }
    if (preprocessing && cd_nelmts >= 2) {
        *preprocessing = cd_values[1];
    }
    if (shuffle_size && cd_nelmts >= 3) {
        *shuffle_size = cd_values[2];
    }
    if (error_bound && cd_nelmts >= 5) {
        *error_bound = unpack_double(cd_values[3], cd_values[4]);
    }

    return 0;
}

/**
 * Get filter information pointer.
 */
const void* H5Z_gpucompress_get_filter_info(void) {
    return H5Z_GPUCOMPRESS;
}

/* ============================================================
 * Plugin Entry Points
 * ============================================================ */

/**
 * Plugin type identifier (required for HDF5 plugin mechanism).
 */
H5PL_type_t H5PLget_plugin_type(void) {
    return H5PL_TYPE_FILTER;
}

/**
 * Plugin info (required for HDF5 plugin mechanism).
 */
const void* H5PLget_plugin_info(void) {
    return H5Z_GPUCOMPRESS;
}
