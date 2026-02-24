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
#include <stdio.h>

#include <hdf5.h>
#include <H5PLextern.h>

#include "gpucompress.h"

/* pack_double / unpack_double require unsigned int to be 4 bytes */
_Static_assert(sizeof(unsigned int) == 4,
               "pack_double assumes sizeof(unsigned int) == 4");

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

/** Per-chunk algorithm tracking (max 256 chunks) */
#define MAX_TRACKED_CHUNKS 256
static int g_chunk_algorithms[MAX_TRACKED_CHUNKS];
static int g_chunk_count = 0;

/** Verbose logging flag (set by GPUCOMPRESS_VERBOSE env var) */
static int g_verbose = -1;  /* -1 = not checked yet */

static int is_verbose(void) {
    if (g_verbose < 0) {
        const char* v = getenv("GPUCOMPRESS_VERBOSE");
        g_verbose = (v != NULL && v[0] != '0') ? 1 : 0;
    }
    return g_verbose;
}

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
 *
 * Checks GPUCOMPRESS_WEIGHTS environment variable for NN weights path.
 * When set, ALGO_AUTO will use the neural network for per-chunk algorithm
 * selection. Without it, ALGO_AUTO falls back to LZ4.
 */
static int ensure_initialized(void) {
    if (!g_gpucompress_initialized) {
        const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
        gpucompress_error_t err = gpucompress_init(weights);
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
        /* Override preprocessing with explicit shuffle size (float32 only) */
        unsigned int shuffle_size = cd_values[2];
        if (shuffle_size == 4) {
            config.preprocessing &= ~GPUCOMPRESS_PREPROC_SHUFFLE_4;
            config.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
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

        /* Check if data was actually compressed (has GPUCompress header).
         * When compression didn't reduce size, the filter returns nbytes
         * on write and HDF5 stores uncompressed data. On read, HDF5 still
         * calls the filter, so we must detect this passthrough case. */
        size_t original_size;
        err = gpucompress_get_original_size(*buf, &original_size);
        if (err != GPUCOMPRESS_SUCCESS) {
            /* No valid header — data was stored uncompressed (passthrough) */
            return nbytes;
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
        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));
        err = gpucompress_compress(*buf, nbytes, new_buf, &new_size, &config, &stats);

        if (err != GPUCOMPRESS_SUCCESS) {
            H5free_memory(new_buf);
            return 0;
        }

        if (is_verbose()) {
            printf("[H5Zgpucompress] chunk %zu bytes -> %zu bytes (%.1f:1) algo=%s\n",
                   nbytes, new_size, (double)nbytes / (new_size > 0 ? new_size : 1),
                   gpucompress_algorithm_name(stats.algorithm_used));
        }

        /* Track algorithm for attribute writing */
        if (g_chunk_count < MAX_TRACKED_CHUNKS) {
            g_chunk_algorithms[g_chunk_count++] = (int)stats.algorithm_used;
        }

        /* Only use compressed if it's smaller */
        if (new_size >= nbytes) {
            /* Compression didn't help, return original */
            if (is_verbose()) {
                printf("[H5Zgpucompress]   -> skipped (no size reduction), storing uncompressed\n");
            }
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
 * Per-Chunk Algorithm Tracking
 * ============================================================ */

/**
 * Reset chunk algorithm tracking. Call before H5Dwrite.
 */
void H5Z_gpucompress_reset_chunk_tracking(void) {
    g_chunk_count = 0;
}

/**
 * Get the number of tracked chunks since last reset.
 */
int H5Z_gpucompress_get_chunk_count(void) {
    return g_chunk_count;
}

/**
 * Get the algorithm used for a specific chunk.
 *
 * @param chunk_idx  Chunk index (0-based)
 * @return Algorithm enum value, or -1 if out of range
 */
int H5Z_gpucompress_get_chunk_algorithm(int chunk_idx) {
    if (chunk_idx < 0 || chunk_idx >= g_chunk_count)
        return -1;
    return g_chunk_algorithms[chunk_idx];
}

/**
 * Write per-chunk algorithm names as a string attribute on a dataset.
 *
 * Creates attribute "gpucompress_chunk_algorithms" with a comma-separated
 * list of algorithm names, e.g. "zstd,zstd,ans,zstd".
 *
 * @param dset_id  Open dataset handle
 * @return 0 on success, -1 on error
 */
herr_t H5Z_gpucompress_write_chunk_attr(hid_t dset_id) {
    if (g_chunk_count <= 0)
        return -1;

    /* Build comma-separated algorithm string with bounds checking */
    char attr_buf[4096];
    size_t offset = 0;
    size_t remaining = sizeof(attr_buf);
    for (int i = 0; i < g_chunk_count && remaining > 1; i++) {
        int written = snprintf(attr_buf + offset, remaining, "%s%s",
            (i > 0) ? "," : "",
            gpucompress_algorithm_name(
                (gpucompress_algorithm_t)g_chunk_algorithms[i]));
        if (written < 0 || (size_t)written >= remaining) break;
        offset += (size_t)written;
        remaining -= (size_t)written;
    }

    /* Create variable-length string attribute */
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, strlen(attr_buf) + 1);

    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(dset_id, "gpucompress_chunk_algorithms",
                             atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(aspace);
        H5Tclose(atype);
        return -1;
    }

    herr_t write_rc = H5Awrite(attr, atype, attr_buf);
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);

    return (write_rc < 0) ? -1 : 0;
}

/* ============================================================
 * Automatic Cleanup on Plugin Unload
 * ============================================================ */

/**
 * Called automatically when the shared library is unloaded.
 * Ensures gpucompress_cleanup() is called to free CUDA resources.
 */
__attribute__((destructor))
static void H5Z_gpucompress_fini(void) {
    if (g_gpucompress_initialized) {
        gpucompress_cleanup();
        g_gpucompress_initialized = 0;
    }
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
