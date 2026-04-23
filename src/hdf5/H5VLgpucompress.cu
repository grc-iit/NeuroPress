/**
 * @file H5VLgpucompress.cu
 * @brief HDF5 VOL connector for GPU-native compression
 *
 * Intercepts H5Dwrite()/H5Dread() with GPU device pointers.
 * Compresses/decompresses chunks on the GPU, then uses H5Dwrite_chunk()
 * / H5Dread_chunk() (via H5VL_NATIVE_DATASET_CHUNK_WRITE/READ) to bypass
 * the HDF5 filter pipeline.
 *
 * For host pointers the call is forwarded unchanged to the underlying VOL.
 *
 * Compiled as CUDA C++ so that cudaPointerGetAttributes() can be called
 * directly.  All public symbols are declared extern "C".
 */

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

/* Public HDF5 headers */
#include <hdf5.h>
#include <H5VLnative.h>

/* GPUCompress public API */
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* Internal API: split-phase inference + CompContext pool */
#include "api/internal.hpp"
#include "api/gpucompress_state.hpp"
#include "api/diagnostics_store.hpp"
#include "selection/heuristic.h"

/* gpucompress HDF5 filter ID (must match H5Zgpucompress.h) */
#define H5Z_FILTER_GPUCOMPRESS 305

/* ============================================================
 * VOL Operating Mode
 *
 * Controlled by GPUCOMPRESS_VOL_MODE environment variable:
 *   bypass   — D→H passthrough to native HDF5, no GPU compression.
 *              Measures I/O baseline; writes timing file on file_close.
 *   release  — Full NN autoselection + online learning (default).
 *   trace    — Everything in release, plus exhaustive per-chunk profiling
 *              of all 32 compression configs. Writes a per-row CSV.
 *
 * All three modes accumulate total I/O wall time in DiagnosticsStore.
 * ============================================================ */
enum class VOLMode { BYPASS, RELEASE, TRACE };
static VOLMode s_vol_mode = VOLMode::RELEASE;

static VOLMode detect_vol_mode() {
    const char* env = getenv("GPUCOMPRESS_VOL_MODE");
    if (!env) return VOLMode::RELEASE;
    if (strcasecmp(env, "bypass") == 0) return VOLMode::BYPASS;
    if (strcasecmp(env, "trace")  == 0) return VOLMode::TRACE;
    return VOLMode::RELEASE;
}

/* ============================================================
 * Typedefs / object structs
 * ============================================================ */

/** VOL info object — identifies the underlying connector */
typedef struct H5VL_gpucompress_info_t {
    hid_t  under_vol_id;    /**< ID of the underlying VOL connector       */
    void  *under_vol_info;  /**< Info for the underlying VOL connector     */
} H5VL_gpucompress_info_t;

/* M4 fix: persistent write state — allocated on first write, reused, freed at close */
#define M4_N_COMP_WORKERS 8
#define M4_N_IO_BUFS      (M4_N_COMP_WORKERS * 2)

struct VolWriteCtx {
    uint8_t* d_comp_w[M4_N_COMP_WORKERS]; /**< Per-worker device output buffers */
    void*    io_pool_bufs[M4_N_IO_BUFS];  /**< Pinned host staging buffers      */
    size_t   max_comp;                     /**< Size each buffer was allocated for */
    bool     initialized;
};

/* P0 fix: global buffer cache — survives H5Dclose → H5Dcreate cycles.
 * H5Dclose donates buffers here instead of freeing; next H5Dwrite reclaims.
 * Single-slot cache. Mutex ordering: acquirable independently or under g_init_mutex. */
struct VolBufCache {
    uint8_t* d_comp_w[M4_N_COMP_WORKERS];
    void*    io_pool_bufs[M4_N_IO_BUFS];
    size_t   max_comp;
    bool     valid;
};
static std::mutex   s_buf_cache_mtx;
static VolBufCache  s_buf_cache = {};

/** Wrapped HDF5 object.  For datasets, dcpl_id holds a copy of the DCPL. */
typedef struct H5VL_gpucompress_t {
    hid_t  under_vol_id;    /**< ID of the underlying VOL connector       */
    void  *under_object;    /**< Underlying VOL object pointer             */
    hid_t  dcpl_id;         /**< DCPL copy (datasets only; else INVALID)  */
    VolWriteCtx* write_ctx; /**< M4: persistent write buffers (NULL until first write) */
} H5VL_gpucompress_t;

/** Wrap-context object used by HDF5 when wrapping sub-objects */
typedef struct H5VL_gpucompress_wrap_ctx_t {
    hid_t  under_vol_id;    /**< Underlying VOL ID                        */
    void  *under_wrap_ctx;  /**< Underlying wrap context                  */
} H5VL_gpucompress_wrap_ctx_t;

/* ============================================================
 * Forward declarations — all callbacks
 * ============================================================ */

struct TraceConfigResult {
    int   action_id;       /* 0-63 in 8×2×4 trace space */
    int   nn_idx;          /* 0-31 in NN 8×2×2 space */
    float real_ratio;
    float real_comp_ms;
    float real_decomp_ms;
    float real_psnr;
    float real_cost;
    float pred_cost;       /* from per_config[nn_idx], 0 if per_config==nullptr */
    bool  valid;           /* false if compression failed */
};

/* Helper functions */
static void run_trace_exhaustive_chunk(const void*, size_t, int, double,
                                        const NNDebugPerConfig*,
                                        TraceConfigResult* /* [64], may be nullptr */);
static herr_t gpu_trace_chunked_write(H5VL_gpucompress_t*, hid_t, hid_t, hid_t, const void*);
static herr_t gpu_bypass_dh_write(H5VL_gpucompress_t*, hid_t, hid_t, hid_t, hid_t, const void*, void**);
static herr_t gpu_bypass_hd_read(H5VL_gpucompress_t*, hid_t, hid_t, hid_t, hid_t, void*, void**);

/* Management */
static herr_t H5VL_gpucompress_init(hid_t vipl_id);
static herr_t H5VL_gpucompress_term(void);

/* Info callbacks */
static void  *H5VL_gpucompress_info_copy(const void *info);
static herr_t H5VL_gpucompress_info_cmp(int *cmp, const void *i1, const void *i2);
static herr_t H5VL_gpucompress_info_free(void *info);
static herr_t H5VL_gpucompress_info_to_str(const void *info, char **str);
static herr_t H5VL_gpucompress_str_to_info(const char *str, void **info);

/* Wrap callbacks */
static void  *H5VL_gpucompress_get_object(const void *obj);
static herr_t H5VL_gpucompress_get_wrap_ctx(const void *obj, void **wrap_ctx);
static void  *H5VL_gpucompress_wrap_object(void *obj, H5I_type_t obj_type, void *wctx);
static void  *H5VL_gpucompress_unwrap_object(void *obj);
static herr_t H5VL_gpucompress_free_wrap_ctx(void *wctx);

/* Attribute callbacks (pure pass-through) */
static void  *H5VL_gpucompress_attr_create(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t type_id, hid_t space_id,
                hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_gpucompress_attr_open(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_read(void *attr, hid_t mem_type_id, void *buf,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_write(void *attr, hid_t mem_type_id, const void *buf,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_get(void *obj, H5VL_attr_get_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_specific(void *obj, const H5VL_loc_params_t *lp,
                H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_optional(void *obj, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Dataset callbacks (custom) */
static void  *H5VL_gpucompress_dataset_create(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id,
                hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_gpucompress_dataset_open(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_dataset_read(size_t count, void *dset[],
                hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
                hid_t plist_id, void *buf[], void **req);
static herr_t H5VL_gpucompress_dataset_write(size_t count, void *dset[],
                hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
                hid_t plist_id, const void *buf[], void **req);
static herr_t H5VL_gpucompress_dataset_get(void *dset, H5VL_dataset_get_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_dataset_specific(void *obj,
                H5VL_dataset_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_dataset_optional(void *obj, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Datatype callbacks (pure pass-through) */
static void  *H5VL_gpucompress_datatype_commit(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id,
                hid_t tapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_gpucompress_datatype_open(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t tapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_datatype_get(void *dt, H5VL_datatype_get_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_datatype_specific(void *obj,
                H5VL_datatype_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_datatype_optional(void *obj, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_datatype_close(void *dt, hid_t dxpl_id, void **req);

/* File callbacks (pure pass-through) */
static void  *H5VL_gpucompress_file_create(const char *name, unsigned flags,
                hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req);
static void  *H5VL_gpucompress_file_open(const char *name, unsigned flags,
                hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_file_get(void *file, H5VL_file_get_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_file_specific(void *file,
                H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_file_optional(void *file, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_file_close(void *file, hid_t dxpl_id, void **req);

/* Group callbacks (pure pass-through) */
static void  *H5VL_gpucompress_group_create(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
                hid_t dxpl_id, void **req);
static void  *H5VL_gpucompress_group_open(void *obj, const H5VL_loc_params_t *lp,
                const char *name, hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_group_get(void *obj, H5VL_group_get_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_group_specific(void *obj,
                H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_group_optional(void *obj, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_group_close(void *grp, hid_t dxpl_id, void **req);

/* Link callbacks (pure pass-through) */
static herr_t H5VL_gpucompress_link_create(H5VL_link_create_args_t *args, void *obj,
                const H5VL_loc_params_t *lp, hid_t lcpl_id, hid_t lapl_id,
                hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_link_copy(void *src_obj, const H5VL_loc_params_t *lp1,
                void *dst_obj, const H5VL_loc_params_t *lp2,
                hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_link_move(void *src_obj, const H5VL_loc_params_t *lp1,
                void *dst_obj, const H5VL_loc_params_t *lp2,
                hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_link_get(void *obj, const H5VL_loc_params_t *lp,
                H5VL_link_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_link_specific(void *obj, const H5VL_loc_params_t *lp,
                H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_link_optional(void *obj, const H5VL_loc_params_t *lp,
                H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Object callbacks (pure pass-through) */
static void  *H5VL_gpucompress_object_open(void *obj, const H5VL_loc_params_t *lp,
                H5I_type_t *opened_type, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_object_copy(void *src_obj,
                const H5VL_loc_params_t *src_lp, const char *src_name,
                void *dst_obj, const H5VL_loc_params_t *dst_lp, const char *dst_name,
                hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_object_get(void *obj, const H5VL_loc_params_t *lp,
                H5VL_object_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_object_specific(void *obj, const H5VL_loc_params_t *lp,
                H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_gpucompress_object_optional(void *obj, const H5VL_loc_params_t *lp,
                H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Introspect callbacks */
static herr_t H5VL_gpucompress_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
                const H5VL_class_t **conn_cls);
static herr_t H5VL_gpucompress_introspect_get_cap_flags(const void *info,
                uint64_t *cap_flags);
static herr_t H5VL_gpucompress_introspect_opt_query(void *obj, H5VL_subclass_t cls,
                int opt_type, uint64_t *flags);

/* Request callbacks */
static herr_t H5VL_gpucompress_request_wait(void *req, uint64_t timeout,
                H5VL_request_status_t *status);
static herr_t H5VL_gpucompress_request_notify(void *obj,
                H5VL_request_notify_t cb, void *ctx);
static herr_t H5VL_gpucompress_request_cancel(void *req,
                H5VL_request_status_t *status);
static herr_t H5VL_gpucompress_request_specific(void *req,
                H5VL_request_specific_args_t *args);
static herr_t H5VL_gpucompress_request_optional(void *req, H5VL_optional_args_t *args);
static herr_t H5VL_gpucompress_request_free(void *req);

/* Blob callbacks */
static herr_t H5VL_gpucompress_blob_put(void *obj, const void *buf, size_t size,
                void *blob_id, void *ctx);
static herr_t H5VL_gpucompress_blob_get(void *obj, const void *blob_id, void *buf,
                size_t size, void *ctx);
static herr_t H5VL_gpucompress_blob_specific(void *obj, void *blob_id,
                H5VL_blob_specific_args_t *args);
static herr_t H5VL_gpucompress_blob_optional(void *obj, void *blob_id,
                H5VL_optional_args_t *args);

/* Token callbacks */
static herr_t H5VL_gpucompress_token_cmp(void *obj, const H5O_token_t *t1,
                const H5O_token_t *t2, int *cmp_value);
static herr_t H5VL_gpucompress_token_to_str(void *obj, H5I_type_t obj_type,
                const H5O_token_t *token, char **token_str);
static herr_t H5VL_gpucompress_token_from_str(void *obj, H5I_type_t obj_type,
                const char *token_str, H5O_token_t *token);

/* Generic optional */
static herr_t H5VL_gpucompress_optional(void *obj, H5VL_optional_args_t *args,
                hid_t dxpl_id, void **req);

/* ============================================================
 * Connector class struct
 * ============================================================ */

static const H5VL_class_t H5VL_gpucompress_g = {
    H5VL_VERSION,
    (H5VL_class_value_t)H5VL_GPUCOMPRESS_VALUE,
    H5VL_GPUCOMPRESS_NAME,
    H5VL_GPUCOMPRESS_VERSION,
    0,                                           /* cap_flags */
    H5VL_gpucompress_init,
    H5VL_gpucompress_term,
    {   /* info_cls */
        sizeof(H5VL_gpucompress_info_t),
        H5VL_gpucompress_info_copy,
        H5VL_gpucompress_info_cmp,
        H5VL_gpucompress_info_free,
        H5VL_gpucompress_info_to_str,
        H5VL_gpucompress_str_to_info
    },
    {   /* wrap_cls */
        H5VL_gpucompress_get_object,
        H5VL_gpucompress_get_wrap_ctx,
        H5VL_gpucompress_wrap_object,
        H5VL_gpucompress_unwrap_object,
        H5VL_gpucompress_free_wrap_ctx
    },
    {   /* attr_cls */
        H5VL_gpucompress_attr_create,
        H5VL_gpucompress_attr_open,
        H5VL_gpucompress_attr_read,
        H5VL_gpucompress_attr_write,
        H5VL_gpucompress_attr_get,
        H5VL_gpucompress_attr_specific,
        H5VL_gpucompress_attr_optional,
        H5VL_gpucompress_attr_close
    },
    {   /* dataset_cls */
        H5VL_gpucompress_dataset_create,
        H5VL_gpucompress_dataset_open,
        H5VL_gpucompress_dataset_read,
        H5VL_gpucompress_dataset_write,
        H5VL_gpucompress_dataset_get,
        H5VL_gpucompress_dataset_specific,
        H5VL_gpucompress_dataset_optional,
        H5VL_gpucompress_dataset_close
    },
    {   /* datatype_cls */
        H5VL_gpucompress_datatype_commit,
        H5VL_gpucompress_datatype_open,
        H5VL_gpucompress_datatype_get,
        H5VL_gpucompress_datatype_specific,
        H5VL_gpucompress_datatype_optional,
        H5VL_gpucompress_datatype_close
    },
    {   /* file_cls */
        H5VL_gpucompress_file_create,
        H5VL_gpucompress_file_open,
        H5VL_gpucompress_file_get,
        H5VL_gpucompress_file_specific,
        H5VL_gpucompress_file_optional,
        H5VL_gpucompress_file_close
    },
    {   /* group_cls */
        H5VL_gpucompress_group_create,
        H5VL_gpucompress_group_open,
        H5VL_gpucompress_group_get,
        H5VL_gpucompress_group_specific,
        H5VL_gpucompress_group_optional,
        H5VL_gpucompress_group_close
    },
    {   /* link_cls */
        H5VL_gpucompress_link_create,
        H5VL_gpucompress_link_copy,
        H5VL_gpucompress_link_move,
        H5VL_gpucompress_link_get,
        H5VL_gpucompress_link_specific,
        H5VL_gpucompress_link_optional
    },
    {   /* object_cls */
        H5VL_gpucompress_object_open,
        H5VL_gpucompress_object_copy,
        H5VL_gpucompress_object_get,
        H5VL_gpucompress_object_specific,
        H5VL_gpucompress_object_optional
    },
    {   /* introspect_cls */
        H5VL_gpucompress_introspect_get_conn_cls,
        H5VL_gpucompress_introspect_get_cap_flags,
        H5VL_gpucompress_introspect_opt_query
    },
    {   /* request_cls */
        H5VL_gpucompress_request_wait,
        H5VL_gpucompress_request_notify,
        H5VL_gpucompress_request_cancel,
        H5VL_gpucompress_request_specific,
        H5VL_gpucompress_request_optional,
        H5VL_gpucompress_request_free
    },
    {   /* blob_cls */
        H5VL_gpucompress_blob_put,
        H5VL_gpucompress_blob_get,
        H5VL_gpucompress_blob_specific,
        H5VL_gpucompress_blob_optional
    },
    {   /* token_cls */
        H5VL_gpucompress_token_cmp,
        H5VL_gpucompress_token_to_str,
        H5VL_gpucompress_token_from_str
    },
    H5VL_gpucompress_optional
};

/* ============================================================
 * Activity counters — queryable from tests / demos
 * ============================================================ */

static std::atomic<int> s_gpu_writes      {0};  /* GPU compression path (gpu_aware_chunked_write) */
static std::atomic<int> s_gpu_reads       {0};  /* GPU decompression path (gpu_aware_chunked_read)*/
static std::atomic<int> s_chunks_comp     {0};  /* chunks successfully compressed on GPU          */
static std::atomic<int> s_chunks_decomp   {0};  /* chunks successfully decompressed on GPU        */
/* Transfer byte counters (atomic: written from 8 concurrent worker threads via vol_memcpy) */
static std::atomic<size_t> s_h2d_bytes{0};  /* total bytes copied host→device */
static std::atomic<size_t> s_d2h_bytes{0};  /* total bytes copied device→host */
static std::atomic<size_t> s_d2d_bytes{0};  /* total bytes copied device→device */
static std::atomic<int>    s_h2d_count{0};  /* number of H→D cudaMemcpy calls  */
static std::atomic<int>    s_d2h_count{0};  /* number of D→H cudaMemcpy calls  */
static std::atomic<int>    s_d2d_count{0};  /* number of D→D cudaMemcpy calls  */
/* Wall-clock stage timing (ms) for the last H5Dwrite call */
static double s_stage1_ms = 0;         /* Stage 1: stats + NN inference (main thread, sequential) */
static double s_drain_ms = 0;          /* Worker drain: S1 end → all workers joined (sentinel + tail) */
static double s_io_drain_ms = 0;       /* I/O drain: workers joined → I/O thread joined */
static double s_s2_busy_ms = 0;        /* Stage 2 actual: max per-worker wall time (bottleneck) */
static double s_s3_busy_ms = 0;        /* Stage 3 actual: I/O thread write time (serial writes) */
static double s_total_ms  = 0;         /* Total pipeline wall clock (stage1 + drain + io_drain) */
static double s_vol_func_ms = 0;       /* Total gpu_aware_chunked_write wall clock */
static double s_setup_ms = 0;          /* Setup before pipeline (VolWriteCtx, threads, etc) */

/* ── Lifetime accumulators (never reset, printed/written at process exit) ── */
static double g_life_vol_ms   = 0;  /* wall-clock inside gpu_aware_chunked_write */
static double g_life_io_ms    = 0;  /* actual disk write time (s3_busy) */
static double g_life_s1_ms    = 0;  /* Stage 1: stats + NN inference (main thread) */
static double g_life_s2_ms    = 0;  /* Stage 2: max worker wall-clock (compress + D2H) */
static double g_life_drain_ms = 0;  /* worker drain (S1 end → workers joined) */
static double g_life_iodrain_ms = 0; /* I/O drain (workers joined → I/O thread joined) */
static double g_life_setup_ms = 0;  /* VOL setup (alloc, thread launch) */
static size_t g_life_bytes_in = 0;  /* total uncompressed bytes written */
static size_t g_life_bytes_out = 0; /* total compressed bytes written */
static int    g_life_writes   = 0;  /* number of H5Dwrite calls */
static int    g_life_chunks   = 0;  /* total chunks processed */
static std::once_flag g_atexit_flag;
static char   g_life_output_dir[512] = "";

static double _now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Raw bypass mode (GPUCOMPRESS_VOL_BYPASS env var) ──
 * When enabled, gpu_aware_chunked_write() skips stats+NN inference AND
 * the nvCOMP compression call. Each chunk still flows through the same
 * worker+IO pipeline (D→H copy into the pinned pool, push to I/O thread,
 * native H5Dwrite_chunk) so timings are directly comparable to a
 * normal compression run: the only thing removed is the NN + nvCOMP
 * work. Produces a raw-I/O baseline for subtractive "pipeline overhead"
 * measurement.
 *
 * The env var is read exactly once via std::call_once on first access
 * so subsequent writes see the same mode. */
static bool s_vol_bypass = false;
static std::once_flag s_vol_bypass_once;

static inline bool vol_bypass_enabled(void) {
    std::call_once(s_vol_bypass_once, [](){
        const char* e = getenv("GPUCOMPRESS_VOL_BYPASS");
        s_vol_bypass = (e && *e && atoi(e) != 0);
        if (s_vol_bypass) {
            fprintf(stderr,
                "[gpucompress VOL] BYPASS MODE ENABLED — "
                "NN + nvCOMP skipped; raw D->H copy + native write only.\n");
        }
    });
    return s_vol_bypass;
}

/* ── Program wall-clock marker ──
 * Captured at the very first H5VL_gpucompress_register() call — the
 * earliest app-visible VOL entry point. Subtract the accumulated
 * g_life_vol_ms at process exit to derive a "compute time" that
 * represents everything NOT spent inside the VOL pipeline (simulator
 * physics, AMReX bookkeeping, etc.). Invariant: compute_ms should be
 * approximately the same across compression and bypass modes for the
 * same workload config. */
static double g_prog_start_ms = 0;
static std::once_flag g_prog_start_once;

static inline void vol_prog_start_mark(void) {
    std::call_once(g_prog_start_once, [](){
        g_prog_start_ms = _now_ms();
    });
}

static void gpucompress_vol_atexit() {
    if (g_life_writes == 0) return;

    double vol_s  = g_life_vol_ms / 1000.0;
    double io_s   = g_life_io_ms / 1000.0;
    double s1_s   = g_life_s1_ms / 1000.0;
    double s2_s   = g_life_s2_ms / 1000.0;
    double drain_s = g_life_drain_ms / 1000.0;
    double iodrain_s = g_life_iodrain_ms / 1000.0;
    double setup_s = g_life_setup_ms / 1000.0;
    double ratio  = (g_life_bytes_out > 0) ? (double)g_life_bytes_in / g_life_bytes_out : 1.0;
    double in_mb  = g_life_bytes_in / (1024.0 * 1024.0);
    double out_mb = g_life_bytes_out / (1024.0 * 1024.0);
    double wr_bw  = (vol_s > 0) ? in_mb / vol_s : 0;

    const bool bypass_mode = vol_bypass_enabled();
    const char *mode_line = bypass_mode
        ? "  MODE:               BYPASS (NN + nvCOMP skipped)\n"
        : "  MODE:               COMPRESSION (NN + nvCOMP on)\n";

    double prog_wall_ms = (g_prog_start_ms > 0) ? (_now_ms() - g_prog_start_ms) : 0.0;
    double compute_ms   = prog_wall_ms - g_life_vol_ms;
    if (compute_ms < 0) compute_ms = 0;
    double prog_wall_s  = prog_wall_ms / 1000.0;
    double compute_s    = compute_ms / 1000.0;

    /* Print to stderr */
    fprintf(stderr,
        "\n========================================\n"
        " GPUCompress VOL — Lifetime Summary\n"
        "========================================\n"
        "%s"
        "  H5Dwrite calls:    %d (%d chunks)\n"
        "  Data in:            %.1f MiB\n"
        "  Data out:           %.1f MiB (%.2fx ratio)\n"
        "  Write throughput:   %.1f MiB/s\n"
        "----------------------------------------\n"
        "  Program wall:       %.3f s  (VOL register → process exit)\n"
        "  Compute (non-VOL):  %.3f s  (program − VOL; mode-invariant)\n"
        "  Total VOL time:     %.3f s  (wall-clock, app blocked)\n"
        "  Disk I/O time:      %.3f s  (actual writes to storage)\n"
        "  Stage 1 (stats+NN): %.3f s  (sequential, main thread)\n"
        "  Stage 2 (compress): %.3f s  (max worker wall-clock)\n"
        "  Worker drain:       %.3f s  (S1 done → workers joined)\n"
        "  I/O drain:          %.3f s  (workers done → I/O done)\n"
        "  Setup:              %.3f s\n"
        "========================================\n",
        mode_line,
        g_life_writes, g_life_chunks,
        in_mb, out_mb, ratio, wr_bw,
        prog_wall_s, compute_s,
        vol_s, io_s, s1_s, s2_s, drain_s, iodrain_s, setup_s);

    /* Write to file */
    char path[600];
    const char *dir = g_life_output_dir[0] ? g_life_output_dir : ".";
    snprintf(path, sizeof(path), "%s/gpucompress_vol_summary.txt", dir);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp,
            "GPUCompress VOL — Lifetime Summary\n"
            "==================================\n"
            "bypass_mode,%d\n"
            "h5dwrite_calls,%d\n"
            "total_chunks,%d\n"
            "bytes_in,%zu\n"
            "bytes_out,%zu\n"
            "ratio,%.4f\n"
            "program_wall_ms,%.2f\n"
            "compute_ms,%.2f\n"
            "vol_time_ms,%.2f\n"
            "disk_io_ms,%.2f\n"
            "stage1_ms,%.2f\n"
            "stage2_ms,%.2f\n"
            "worker_drain_ms,%.2f\n"
            "io_drain_ms,%.2f\n"
            "setup_ms,%.2f\n"
            "write_mibps,%.1f\n",
            bypass_mode ? 1 : 0,
            g_life_writes, g_life_chunks,
            g_life_bytes_in, g_life_bytes_out, ratio,
            prog_wall_ms, compute_ms,
            g_life_vol_ms, g_life_io_ms,
            g_life_s1_ms, g_life_s2_ms,
            g_life_drain_ms, g_life_iodrain_ms,
            g_life_setup_ms, wr_bw);
        fclose(fp);
        fprintf(stderr, "  Written to: %s\n\n", path);
    }

    /* Also dump IO timing CSV (e2e_ms, vol_ms) */
    {
        auto& ds = gpucompress::DiagnosticsStore::instance();
        ds.recordProcessEnd();
        const char* timing_path = getenv("GPUCOMPRESS_TIMING_OUTPUT");
        ds.dumpIoTiming(timing_path ? timing_path : "gpucompress_io_timing.csv");
    }
}

/* Wrapper: track and execute cudaMemcpy */
static inline cudaError_t vol_memcpy(void *dst, const void *src,
                                      size_t bytes, cudaMemcpyKind kind)
{
    switch (kind) {
        case cudaMemcpyHostToDevice:   s_h2d_bytes.fetch_add(bytes, std::memory_order_relaxed); s_h2d_count.fetch_add(1, std::memory_order_relaxed); break;
        case cudaMemcpyDeviceToHost:   s_d2h_bytes.fetch_add(bytes, std::memory_order_relaxed); s_d2h_count.fetch_add(1, std::memory_order_relaxed); break;
        case cudaMemcpyDeviceToDevice: s_d2d_bytes.fetch_add(bytes, std::memory_order_relaxed); s_d2d_count.fetch_add(1, std::memory_order_relaxed); break;
        default: break;
    }
    return cudaMemcpy(dst, src, bytes, kind);
}

/* ── Timing helpers callable from Python via ctypes (libH5VLgpucompress.so) ──
 * These operate on the DiagnosticsStore instance inside THIS library, which
 * is the correct singleton for vol_ms accumulation and e2e tracking.
 * Do NOT call the equivalents in libgpucompress.so — that is a separate DSO
 * with its own singleton instance. */

/* Reset the e2e start timer to now.  Call just before the training/simulation
 * loop begins so that e2e_ms excludes startup overhead (CUDA init, data load,
 * model loading). */
extern "C" void
H5VL_gpucompress_record_process_start(void)
{
    gpucompress::DiagnosticsStore::instance().resetProcessStart();
}

/* Dump e2e + vol timing CSV.  Call explicitly from Python after all writes
 * complete; do not rely on C atexit ordering through ctypes.
 * path: file path, or NULL to use GPUCOMPRESS_TIMING_OUTPUT env var. */
extern "C" void
H5VL_gpucompress_dump_timing(const char *path)
{
    auto& s = gpucompress::DiagnosticsStore::instance();
    s.recordProcessEnd();
    if (!path) path = getenv("GPUCOMPRESS_TIMING_OUTPUT");
    if (!path) path = "gpucompress_io_timing.csv";
    s.dumpIoTiming(path);
}

extern "C" void
H5VL_gpucompress_reset_stats(void)
{
    s_gpu_writes.store(0);      s_gpu_reads.store(0);
    s_chunks_comp.store(0);     s_chunks_decomp.store(0);
    s_h2d_bytes.store(0); s_d2h_bytes.store(0); s_d2d_bytes.store(0);
    s_h2d_count.store(0); s_d2h_count.store(0); s_d2d_count.store(0);
    s_stage1_ms = s_drain_ms = s_io_drain_ms = s_total_ms = 0;
    s_s2_busy_ms = s_s3_busy_ms = 0;
    s_vol_func_ms = s_setup_ms = 0;
}

extern "C" void
H5VL_gpucompress_get_stage_timing(double *stage1_ms, double *drain_ms,
                                   double *io_drain_ms, double *total_ms)
{
    if (stage1_ms)   *stage1_ms   = s_stage1_ms;
    if (drain_ms)    *drain_ms    = s_drain_ms;
    if (io_drain_ms) *io_drain_ms = s_io_drain_ms;
    if (total_ms)    *total_ms    = s_total_ms;
}

extern "C" void
H5VL_gpucompress_get_busy_timing(double *s2_busy_ms, double *s3_busy_ms)
{
    if (s2_busy_ms) *s2_busy_ms = s_s2_busy_ms;
    if (s3_busy_ms) *s3_busy_ms = s_s3_busy_ms;
}

extern "C" void
H5VL_gpucompress_get_vol_func_timing(double *setup_ms, double *vol_func_ms)
{
    if (setup_ms)    *setup_ms    = s_setup_ms;
    if (vol_func_ms) *vol_func_ms = s_vol_func_ms;
}

extern "C" int
H5VL_gpucompress_is_bypass_mode(void)
{
    return vol_bypass_enabled() ? 1 : 0;
}

extern "C" void
H5VL_gpucompress_get_program_wall(double *program_ms, double *compute_ms)
{
    double now = _now_ms();
    double pm  = (g_prog_start_ms > 0) ? (now - g_prog_start_ms) : 0.0;
    double cm  = pm - g_life_vol_ms;
    if (cm < 0) cm = 0;
    if (program_ms) *program_ms = pm;
    if (compute_ms) *compute_ms = cm;
}

extern "C" void
H5VL_gpucompress_release_buf_cache(void)
{
    std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
    if (s_buf_cache.valid) {
        for (int w = 0; w < M4_N_COMP_WORKERS; w++) {
            if (s_buf_cache.d_comp_w[w]) { cudaFree(s_buf_cache.d_comp_w[w]); s_buf_cache.d_comp_w[w] = nullptr; }
        }
        for (int i = 0; i < M4_N_IO_BUFS; i++) {
            if (s_buf_cache.io_pool_bufs[i]) { cudaFreeHost(s_buf_cache.io_pool_bufs[i]); s_buf_cache.io_pool_bufs[i] = nullptr; }
        }
        s_buf_cache.valid = false;
        s_buf_cache.max_comp = 0;
    }
}

extern "C" void
H5VL_gpucompress_get_stats(int *writes, int *reads, int *comp, int *decomp)
{
    if (writes) *writes = s_gpu_writes.load();
    if (reads)  *reads  = s_gpu_reads.load();
    if (comp)   *comp   = s_chunks_comp.load();
    if (decomp) *decomp = s_chunks_decomp.load();
}

extern "C" void
H5VL_gpucompress_get_transfer_stats(
    int *h2d_count, size_t *h2d_bytes,
    int *d2h_count, size_t *d2h_bytes,
    int *d2d_count, size_t *d2d_bytes)
{
    if (h2d_count) *h2d_count = s_h2d_count.load();
    if (h2d_bytes) *h2d_bytes = s_h2d_bytes.load();
    if (d2h_count) *d2h_count = s_d2h_count.load();
    if (d2h_bytes) *d2h_bytes = s_d2h_bytes.load();
    if (d2d_count) *d2d_count = s_d2d_count.load();
    if (d2d_bytes) *d2d_bytes = s_d2d_bytes.load();
}

/* ============================================================
 * Call-sequence trace — toggled by H5VL_gpucompress_set_trace()
 * ============================================================ */
static int s_trace = 0;

#define VOL_TRACE(fmt, ...) \
    do { if (s_trace) printf("[VOL] " fmt "\n", ##__VA_ARGS__); } while(0)

extern "C" void H5VL_gpucompress_set_trace(int on) { s_trace = on; }

/* Decode H5VL_dataset_get_t op codes */
static const char *dset_get_name(int op)
{
    switch (op) {
        case H5VL_DATASET_GET_DAPL:         return "GET_DAPL";
        case H5VL_DATASET_GET_DCPL:         return "GET_DCPL";
        case H5VL_DATASET_GET_SPACE:        return "GET_SPACE";
        case H5VL_DATASET_GET_SPACE_STATUS: return "GET_SPACE_STATUS";
        case H5VL_DATASET_GET_STORAGE_SIZE: return "GET_STORAGE_SIZE";
        case H5VL_DATASET_GET_TYPE:         return "GET_TYPE";
        default:                            return "GET_?";
    }
}

/* Decode H5VL_NATIVE_DATASET_* optional op codes */
static const char *dset_opt_name(int op)
{
    switch (op) {
        case H5VL_NATIVE_DATASET_CHUNK_READ:              return "CHUNK_READ";
        case H5VL_NATIVE_DATASET_CHUNK_WRITE:             return "CHUNK_WRITE";
        case H5VL_NATIVE_DATASET_GET_CHUNK_STORAGE_SIZE:  return "GET_CHUNK_STORAGE_SIZE";
        case H5VL_NATIVE_DATASET_GET_NUM_CHUNKS:          return "GET_NUM_CHUNKS";
        case H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_IDX:   return "GET_CHUNK_INFO_BY_IDX";
        case H5VL_NATIVE_DATASET_GET_CHUNK_INFO_BY_COORD: return "GET_CHUNK_INFO_BY_COORD";
        case H5VL_NATIVE_DATASET_CHUNK_ITER:              return "CHUNK_ITER";
        default:                                           return "OPT_?";
    }
}

/* ============================================================
 * Helper macros / functions
 * ============================================================ */

#define UNDER_OBJ(o)    (((H5VL_gpucompress_t*)(o))->under_object)
#define UNDER_VOL(o)    (((H5VL_gpucompress_t*)(o))->under_vol_id)

static H5VL_gpucompress_t *
new_obj(void *under_obj, hid_t under_vol_id)
{
    H5VL_gpucompress_t *o =
        (H5VL_gpucompress_t*)calloc(1, sizeof(H5VL_gpucompress_t));
    if (!o) return NULL;
    o->under_object = under_obj;
    o->under_vol_id = under_vol_id;
    o->dcpl_id      = H5I_INVALID_HID;
    H5Iinc_ref(under_vol_id);
    return o;
}

static herr_t
free_obj(H5VL_gpucompress_t *o)
{
    hid_t err_id = H5Eget_current_stack();
    if (o->dcpl_id != H5I_INVALID_HID)
        H5Pclose(o->dcpl_id);
    H5Idec_ref(o->under_vol_id);
    H5Eset_current_stack(err_id);
    free(o);
    return 0;
}

/* ============================================================
 * Management callbacks
 * ============================================================ */

static herr_t H5VL_gpucompress_init(hid_t /*vipl_id*/) {
    s_vol_mode = detect_vol_mode();
    const char* mode_str =
        (s_vol_mode == VOLMode::BYPASS)  ? "bypass"  :
        (s_vol_mode == VOLMode::TRACE)   ? "trace"   : "release";
    fprintf(stderr, "gpucompress VOL: mode=%s\n", mode_str);

    if (s_vol_mode == VOLMode::TRACE) {
        const char* path = getenv("GPUCOMPRESS_TRACE_OUTPUT");
        gpucompress::DiagnosticsStore::instance().openTraceFile(
            path ? path : "gpucompress_trace.csv");
    }

    /* Start process-level e2e timer; atexit dumps timing after all writes. */
    gpucompress::DiagnosticsStore::instance().recordProcessStart();
    std::atexit([]() {
        auto& s = gpucompress::DiagnosticsStore::instance();
        s.recordProcessEnd();
        const char* path = getenv("GPUCOMPRESS_TIMING_OUTPUT");
        s.dumpIoTiming(path ? path : "gpucompress_io_timing.csv");
    });

    return 0;
}

static herr_t H5VL_gpucompress_term(void) { return 0; }

/* ============================================================
 * Info callbacks (adapted from H5VLpassthru.c)
 * ============================================================ */

static void *
H5VL_gpucompress_info_copy(const void *_info)
{
    const H5VL_gpucompress_info_t *info = (const H5VL_gpucompress_info_t*)_info;
    if (!info || H5Iis_valid(info->under_vol_id) <= 0) return NULL;

    H5VL_gpucompress_info_t *ni =
        (H5VL_gpucompress_info_t*)calloc(1, sizeof(*ni));
    ni->under_vol_id = info->under_vol_id;
    H5Iinc_ref(ni->under_vol_id);
    if (info->under_vol_info)
        H5VLcopy_connector_info(ni->under_vol_id,
                                &ni->under_vol_info, info->under_vol_info);
    return ni;
}

static herr_t
H5VL_gpucompress_info_cmp(int *cmp, const void *_i1, const void *_i2)
{
    const H5VL_gpucompress_info_t *i1 = (const H5VL_gpucompress_info_t*)_i1;
    const H5VL_gpucompress_info_t *i2 = (const H5VL_gpucompress_info_t*)_i2;
    assert(i1); assert(i2);
    *cmp = 0;
    H5VLcmp_connector_cls(cmp, i1->under_vol_id, i2->under_vol_id);
    if (*cmp) return 0;
    H5VLcmp_connector_info(cmp, i1->under_vol_id,
                           i1->under_vol_info, i2->under_vol_info);
    return 0;
}

static herr_t
H5VL_gpucompress_info_free(void *_info)
{
    H5VL_gpucompress_info_t *info = (H5VL_gpucompress_info_t*)_info;
    hid_t err_id = H5Eget_current_stack();
    if (info->under_vol_info)
        H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
    H5Idec_ref(info->under_vol_id);
    H5Eset_current_stack(err_id);
    free(info);
    return 0;
}

static herr_t
H5VL_gpucompress_info_to_str(const void *_info, char **str)
{
    const H5VL_gpucompress_info_t *info = (const H5VL_gpucompress_info_t*)_info;
    H5VL_class_value_t uval = (H5VL_class_value_t)-1;
    char *us = NULL;
    size_t ulen = 0;
    H5VLget_value(info->under_vol_id, &uval);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &us);
    if (us) ulen = strlen(us);
    size_t sz = 64 + ulen;
    *str = (char*)H5allocate_memory(sz, (bool)0);
    snprintf(*str, sz, "under_vol=%u;under_info={%s}",
             (unsigned)uval, us ? us : "");
    if (us) H5free_memory(us);
    return 0;
}

static herr_t
H5VL_gpucompress_str_to_info(const char *str, void **_info)
{
    unsigned uval = 0;
    sscanf(str, "under_vol=%u;", &uval);
    hid_t uid = H5VLregister_connector_by_value((H5VL_class_value_t)uval, H5P_DEFAULT);

    const char *s = strchr(str, '{');
    const char *e = strrchr(str, '}');
    void *uinfo = NULL;
    if (s && e && e > s + 1) {
        char *sub = (char*)malloc((size_t)(e - s));
        memcpy(sub, s + 1, (size_t)(e - s - 1));
        sub[e - s - 1] = '\0';
        H5VLconnector_str_to_info(sub, uid, &uinfo);
        free(sub);
    }

    H5VL_gpucompress_info_t *info =
        (H5VL_gpucompress_info_t*)calloc(1, sizeof(*info));
    info->under_vol_id   = uid;
    info->under_vol_info = uinfo;
    *_info = info;
    return 0;
}

/* ============================================================
 * Wrap callbacks
 * ============================================================ */

static void *
H5VL_gpucompress_get_object(const void *obj)
{
    const H5VL_gpucompress_t *o = (const H5VL_gpucompress_t*)obj;
    return H5VLget_object(o->under_object, o->under_vol_id);
}

static herr_t
H5VL_gpucompress_get_wrap_ctx(const void *obj, void **wrap_ctx)
{
    const H5VL_gpucompress_t *o = (const H5VL_gpucompress_t*)obj;
    H5VL_gpucompress_wrap_ctx_t *wc =
        (H5VL_gpucompress_wrap_ctx_t*)calloc(1, sizeof(*wc));
    wc->under_vol_id = o->under_vol_id;
    H5Iinc_ref(wc->under_vol_id);
    H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &wc->under_wrap_ctx);
    *wrap_ctx = wc;
    return 0;
}

static void *
H5VL_gpucompress_wrap_object(void *obj, H5I_type_t obj_type, void *_wctx)
{
    H5VL_gpucompress_wrap_ctx_t *wc = (H5VL_gpucompress_wrap_ctx_t*)_wctx;
    void *under = H5VLwrap_object(obj, obj_type, wc->under_vol_id, wc->under_wrap_ctx);
    return under ? (void*)new_obj(under, wc->under_vol_id) : NULL;
}

static void *
H5VL_gpucompress_unwrap_object(void *obj)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLunwrap_object(o->under_object, o->under_vol_id);
    if (under) free_obj(o);
    return under;
}

static herr_t
H5VL_gpucompress_free_wrap_ctx(void *_wctx)
{
    H5VL_gpucompress_wrap_ctx_t *wc = (H5VL_gpucompress_wrap_ctx_t*)_wctx;
    hid_t err_id = H5Eget_current_stack();
    if (wc->under_wrap_ctx)
        H5VLfree_wrap_ctx(wc->under_wrap_ctx, wc->under_vol_id);
    H5Idec_ref(wc->under_vol_id);
    H5Eset_current_stack(err_id);
    free(wc);
    return 0;
}

/* ============================================================
 * Attribute callbacks (pure pass-through)
 * ============================================================ */

static void *
H5VL_gpucompress_attr_create(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id,
    hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLattr_create(o->under_object, lp, o->under_vol_id,
                                  name, type_id, space_id, acpl_id, aapl_id,
                                  dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *attr = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return attr;
}

static void *
H5VL_gpucompress_attr_open(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLattr_open(o->under_object, lp, o->under_vol_id,
                                name, aapl_id, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *attr = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return attr;
}

static herr_t
H5VL_gpucompress_attr_read(void *attr, hid_t mem_type_id, void *buf,
                            hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)attr;
    herr_t rv = H5VLattr_read(o->under_object, o->under_vol_id,
                               mem_type_id, buf, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_attr_write(void *attr, hid_t mem_type_id, const void *buf,
                             hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)attr;
    herr_t rv = H5VLattr_write(o->under_object, o->under_vol_id,
                                mem_type_id, buf, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_attr_get(void *obj, H5VL_attr_get_args_t *args,
                           hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_attr_specific(void *obj, const H5VL_loc_params_t *lp,
    H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLattr_specific(o->under_object, lp, o->under_vol_id,
                                   args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_attr_optional(void *obj, H5VL_optional_args_t *args,
                                hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLattr_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_attr_close(void *attr, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)attr;
    herr_t rv = H5VLattr_close(o->under_object, o->under_vol_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    if (rv >= 0) free_obj(o);
    return rv;
}

/* ============================================================
 * GPU chunk-iteration helpers
 * ============================================================ */

/**
 * CUDA kernel: gather one chunk (arbitrary N-D, C-order) from a source buffer.
 *
 * Each thread handles one element (of elem_size bytes).
 * Dimension information is passed inline; max H5S_MAX_RANK = 32 dims.
 */
__global__ static void gather_chunk_kernel(
    const uint8_t * __restrict__ d_src,
    uint8_t       * __restrict__ d_dst,
    int            ndims,
    size_t         elem_size,
    size_t         chunk_nelems,
    /* src_dims, chunk_dims, chunk_start — encoded as 3 × 32 hsize_t arrays */
    const hsize_t * __restrict__ src_dims,
    const hsize_t * __restrict__ chunk_dims,
    const hsize_t * __restrict__ chunk_start)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_nelems) return;

    /* Compute strides for source buffer (C-order, elem as unit) */
    hsize_t strides[32];
    strides[ndims - 1] = 1;
    for (int d = ndims - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * src_dims[d + 1];

    /* Convert linear chunk index → N-D source offset */
    size_t remaining = idx;
    size_t src_elem  = 0;
    for (int d = ndims - 1; d >= 0; d--) {
        size_t local = remaining % chunk_dims[d];
        remaining   /= chunk_dims[d];
        src_elem    += (chunk_start[d] + local) * (size_t)strides[d];
    }

    const uint8_t *src = d_src + src_elem * elem_size;
    uint8_t       *dst = d_dst + idx      * elem_size;
    for (size_t b = 0; b < elem_size; b++)
        dst[b] = src[b];
}

/**
 * CUDA kernel: scatter one chunk back to a destination buffer (inverse of gather).
 */
__global__ static void scatter_chunk_kernel(
    const uint8_t * __restrict__ d_src,
    uint8_t       * __restrict__ d_dst,
    int            ndims,
    size_t         elem_size,
    size_t         chunk_nelems,
    const hsize_t * __restrict__ dst_dims,
    const hsize_t * __restrict__ chunk_dims,
    const hsize_t * __restrict__ chunk_start)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_nelems) return;

    hsize_t strides[32];
    strides[ndims - 1] = 1;
    for (int d = ndims - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * dst_dims[d + 1];

    size_t remaining = idx;
    size_t dst_elem  = 0;
    for (int d = ndims - 1; d >= 0; d--) {
        size_t local = remaining % chunk_dims[d];
        remaining   /= chunk_dims[d];
        dst_elem    += (chunk_start[d] + local) * (size_t)strides[d];
    }

    const uint8_t *src = d_src + idx      * elem_size;
    uint8_t       *dst = d_dst + dst_elem  * elem_size;
    for (size_t b = 0; b < elem_size; b++)
        dst[b] = src[b];
}

/** Allocate device copies of dimension arrays used by the kernels */
static int
alloc_dim_arrays(int ndims, const hsize_t *src_dims,
                 const hsize_t *chunk_dims, const hsize_t *chunk_start,
                 hsize_t **d_src_dims, hsize_t **d_chunk_dims,
                 hsize_t **d_chunk_start)
{
    size_t bytes = (size_t)ndims * sizeof(hsize_t);
    if (cudaMalloc(d_src_dims,    bytes) != cudaSuccess) return -1;
    if (cudaMalloc(d_chunk_dims,  bytes) != cudaSuccess) { cudaFree(*d_src_dims);   return -1; }
    if (cudaMalloc(d_chunk_start, bytes) != cudaSuccess) { cudaFree(*d_src_dims); cudaFree(*d_chunk_dims); return -1; }
    vol_memcpy(*d_src_dims,    src_dims,   bytes, cudaMemcpyHostToDevice);
    vol_memcpy(*d_chunk_dims,  chunk_dims, bytes, cudaMemcpyHostToDevice);
    vol_memcpy(*d_chunk_start, chunk_start, bytes, cudaMemcpyHostToDevice);
    return 0;
}

static void free_dim_arrays(hsize_t *d_src, hsize_t *d_chk, hsize_t *d_start)
{
    cudaFree(d_src); cudaFree(d_chk); cudaFree(d_start);
}

/* ============================================================
 * get_gpucompress_config_from_dcpl
 * ============================================================ */

/**
 * Parse the GPUCompress filter parameters from a DCPL.
 * Returns GPUCOMPRESS_SUCCESS and fills *cfg on success.
 * Returns non-zero if the filter is absent or parsing fails.
 */
static int
get_gpucompress_config_from_dcpl(hid_t dcpl_id, gpucompress_config_t *cfg)
{
    if (dcpl_id == H5I_INVALID_HID || !cfg) return -1;

    unsigned int  flags     = 0;
    size_t        cd_nelmts = 5;
    unsigned int  cd_values[5] = {0, 0, 0, 0, 0};
    char          filter_name[64] = {0};
    size_t        name_len = sizeof(filter_name);

    herr_t rv = H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_GPUCOMPRESS,
                                      &flags, &cd_nelmts, cd_values,
                                      name_len, filter_name, NULL);
    if (rv < 0) return -1;   /* filter not found */

    *cfg = gpucompress_default_config();

    if (cd_nelmts >= 1) cfg->algorithm     = (gpucompress_algorithm_t)cd_values[0];
    if (cd_nelmts >= 2) cfg->preprocessing = cd_values[1];
    if (cd_nelmts >= 3 && cd_values[2] == 4)
        cfg->preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
    if (cd_nelmts >= 5) {
        /* unpack_double: lo = cd_values[3], hi = cd_values[4] */
        union { double d; unsigned int u[2]; } u;
        u.u[0] = cd_values[3];
        u.u[1] = cd_values[4];
        cfg->error_bound = u.d;
        if (cfg->error_bound > 0.0)
            cfg->preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;
    }
    return 0;
}

/* ============================================================
 * write_chunk_to_native / read_chunk_from_native
 * ============================================================ */

/**
 * Write one pre-compressed chunk to the underlying native VOL.
 * filters = 0: all pipeline filters have been applied (gpucompress data).
 */
static herr_t
write_chunk_to_native(void *under_object, hid_t under_vol_id,
                      const hsize_t *offset,
                      const void    *buf,
                      size_t         compressed_size,
                      hid_t          dxpl_id)
{
    VOL_TRACE("    → H5VLdataset_optional(native, CHUNK_WRITE) off=%llu size=%zu B",
              (unsigned long long)offset[0], compressed_size);
    H5VL_native_dataset_optional_args_t native_args;
    H5VL_optional_args_t                opt_args;

    memset(&native_args, 0, sizeof(native_args));
    native_args.chunk_write.offset  = offset;
    native_args.chunk_write.filters = 0;        /* filters applied = all */
    native_args.chunk_write.size    = compressed_size;
    native_args.chunk_write.buf     = buf;

    opt_args.op_type = H5VL_NATIVE_DATASET_CHUNK_WRITE;
    opt_args.args    = &native_args;

    return H5VLdataset_optional(under_object, under_vol_id, &opt_args, dxpl_id, NULL);
}

/**
 * Read one compressed chunk from the underlying native VOL.
 * On entry, *buf_size is the allocation capacity of buf.
 * On exit,  *buf_size holds the actual number of bytes written.
 */
static herr_t
read_chunk_from_native(void *under_object, hid_t under_vol_id,
                       const hsize_t *offset,
                       void          *buf,
                       size_t        *buf_size,
                       uint32_t      *filter_mask_out,
                       hid_t          dxpl_id)
{
    VOL_TRACE("    → H5VLdataset_optional(native, CHUNK_READ)  off=%llu buf_size=%zu B",
              (unsigned long long)offset[0], buf_size ? *buf_size : 0);
    H5VL_native_dataset_optional_args_t native_args;
    H5VL_optional_args_t                opt_args;

    memset(&native_args, 0, sizeof(native_args));
    native_args.chunk_read.offset   = offset;
    native_args.chunk_read.filters  = filter_mask_out ? *filter_mask_out : 0;
    native_args.chunk_read.buf      = buf;
    native_args.chunk_read.buf_size = buf_size;

    opt_args.op_type = H5VL_NATIVE_DATASET_CHUNK_READ;
    opt_args.args    = &native_args;

    return H5VLdataset_optional(under_object, under_vol_id, &opt_args, dxpl_id, NULL);
}

/* ============================================================
 * gpu_trace_chunked_write
 *
 * Trace mode write path: sequential (no worker threads, no I/O thread).
 * For every chunk, runs exhaustive profiling across all 64 configs, then
 * picks the cheapest by predicted cost and writes that to file.
 * Also fires SGD + exploration from the measured results.
 * ============================================================ */

static herr_t
gpu_trace_chunked_write(H5VL_gpucompress_t *o,
                        hid_t               mem_type_id,
                        hid_t               file_space_id,
                        hid_t               dxpl_id,
                        const void         *d_buf)
{
    herr_t ret = -1;
    hid_t  actual_space_id = H5I_INVALID_HID;
    int    need_close_space = 0;

    /* ---- Resolve H5S_ALL → actual dataset space ---- */
    if (file_space_id == H5S_ALL) {
        H5VL_dataset_get_args_t get_args;
        memset(&get_args, 0, sizeof(get_args));
        get_args.op_type                 = H5VL_DATASET_GET_SPACE;
        get_args.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id,
                            &get_args, dxpl_id, NULL) < 0) {
            fprintf(stderr, "gpu_trace_chunked_write: H5VLdataset_get(GET_SPACE) failed\n");
            goto done_trace;
        }
        actual_space_id  = get_args.args.get_space.space_id;
        need_close_space = 1;
    } else {
        actual_space_id = file_space_id;
    }

    {
        int ndims = H5Sget_simple_extent_ndims(actual_space_id);
        if (ndims <= 0 || ndims > 32) {
            fprintf(stderr, "gpu_trace_chunked_write: invalid ndims=%d\n", ndims);
            goto done_trace;
        }

        hsize_t dset_dims[32];
        H5Sget_simple_extent_dims(actual_space_id, dset_dims, NULL);

        hsize_t chunk_dims[32] = {0};
        if (H5Pget_chunk(o->dcpl_id, ndims, chunk_dims) < 0) {
            fprintf(stderr,
                    "gpu_trace_chunked_write: H5Pget_chunk failed "
                    "(dcpl layout not H5D_CHUNKED or dcpl invalid) ndims=%d\n",
                    ndims);
            goto done_trace;
        }

        size_t elem_size = H5Tget_size(mem_type_id);
        if (elem_size == 0) {
            fprintf(stderr, "gpu_trace_chunked_write: H5Tget_size returned 0\n");
            goto done_trace;
        }

        gpucompress_config_t cfg;
        if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) != 0) {
            fprintf(stderr, "gpu_trace_chunked_write: get_gpucompress_config_from_dcpl failed\n");
            goto done_trace;
        }

        size_t chunk_elems = 1;
        for (int d = 0; d < ndims; d++) chunk_elems *= (size_t)chunk_dims[d];
        size_t chunk_bytes = chunk_elems * elem_size;
        size_t max_comp    = gpucompress_max_compressed_size(chunk_bytes);

        /* Single device compression buffer */
        void* d_comp = nullptr;
        {
            cudaError_t ce = cudaMalloc(&d_comp, max_comp);
            if (ce != cudaSuccess) {
                fprintf(stderr,
                        "gpu_trace_chunked_write: cudaMalloc(d_comp, %zu bytes) "
                        "failed: %s (chunk_bytes=%zu)\n",
                        max_comp, cudaGetErrorString(ce), chunk_bytes);
                goto done_trace;
            }
        }

        /* Single pinned host buffer for D→H */
        void* h_comp = nullptr;
        {
            cudaError_t ce = cudaMallocHost(&h_comp, max_comp);
            if (ce != cudaSuccess) {
                fprintf(stderr,
                        "gpu_trace_chunked_write: cudaMallocHost(h_comp, %zu bytes) "
                        "failed: %s\n",
                        max_comp, cudaGetErrorString(ce));
                cudaFree(d_comp);
                goto done_trace;
            }
        }

        /* CompContext for NN inference */
        CompContext* infer_ctx = nullptr;
        bool use_nn = (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO);
        if (use_nn) {
            infer_ctx = gpucompress::acquireCompContext();
            if (!infer_ctx) {
                fprintf(stderr, "gpucompress VOL trace: failed to acquire inference CompContext\n");
                cudaFree(d_comp); cudaFreeHost(h_comp); goto done_trace;
            }
        }

        /* Chunk iteration setup */
        hsize_t num_chunks[32];
        for (int d = 0; d < ndims; d++)
            num_chunks[d] = (dset_dims[d] + chunk_dims[d] - 1) / chunk_dims[d];
        size_t total_chunks = 1;
        for (int d = 0; d < ndims; d++) total_chunks *= (size_t)num_chunks[d];

        bool contiguous = true;
        for (int d = 1; d < ndims; d++) {
            if (chunk_dims[d] != dset_dims[d]) { contiguous = false; break; }
        }

        hsize_t *d_dset_dims = NULL, *d_chunk_dims_d = NULL, *d_chunk_start_d = NULL;
        cudaStream_t gather_stream = nullptr;
        cudaStreamCreate(&gather_stream);

        ret = 0;

        for (size_t ci = 0; ci < total_chunks && ret == 0; ci++) {
            /* ---- Compute N-D chunk coordinates ---- */
            hsize_t chunk_idx[32];
            size_t  remaining = ci;
            for (int d = ndims - 1; d >= 0; d--) {
                chunk_idx[d] = (hsize_t)(remaining % (size_t)num_chunks[d]);
                remaining   /= (size_t)num_chunks[d];
            }
            hsize_t chunk_start[32];
            for (int d = 0; d < ndims; d++)
                chunk_start[d] = chunk_idx[d] * chunk_dims[d];

            hsize_t actual_chunk[32];
            size_t  actual_elems = 1;
            for (int d = 0; d < ndims; d++) {
                actual_chunk[d] = chunk_dims[d];
                if (chunk_start[d] + actual_chunk[d] > dset_dims[d])
                    actual_chunk[d] = dset_dims[d] - chunk_start[d];
                actual_elems *= (size_t)actual_chunk[d];
            }
            size_t actual_bytes = actual_elems * elem_size;

            /* ---- Gather chunk data to a device pointer (src) ---- */
            const uint8_t* src = nullptr;
            uint8_t* d_owned = nullptr;

            if (contiguous) {
                size_t off = 0, stride = elem_size;
                for (int d = ndims - 1; d >= 0; d--) {
                    off    += (size_t)chunk_start[d] * stride;
                    stride *= (size_t)dset_dims[d];
                }
                const uint8_t *raw = static_cast<const uint8_t*>(d_buf) + off;
                if (actual_bytes == chunk_bytes) {
                    src     = raw;
                    d_owned = NULL;
                } else {
                    cudaError_t ce = cudaMalloc(&d_owned, actual_bytes);
                    if (ce != cudaSuccess) {
                        fprintf(stderr,
                                "gpu_trace_chunked_write: chunk %zu cudaMalloc(d_owned, "
                                "%zu bytes) contiguous-partial failed: %s\n",
                                ci, actual_bytes, cudaGetErrorString(ce));
                        ret = -1; break;
                    }
                    vol_memcpy(d_owned, raw, actual_bytes, cudaMemcpyDeviceToDevice);
                    src     = d_owned;
                }
            } else {
                cudaError_t ce = cudaMalloc(&d_owned, chunk_bytes);
                if (ce != cudaSuccess) {
                    fprintf(stderr,
                            "gpu_trace_chunked_write: chunk %zu cudaMalloc(d_owned, "
                            "%zu bytes) non-contiguous failed: %s\n",
                            ci, chunk_bytes, cudaGetErrorString(ce));
                    ret = -1; break;
                }
                if (!d_dset_dims) {
                    if (alloc_dim_arrays(ndims, dset_dims, actual_chunk, chunk_start,
                                        &d_dset_dims, &d_chunk_dims_d, &d_chunk_start_d) < 0) {
                        fprintf(stderr,
                                "gpu_trace_chunked_write: chunk %zu alloc_dim_arrays failed\n",
                                ci);
                        cudaFree(d_owned); ret = -1; break;
                    }
                } else {
                    vol_memcpy(d_chunk_dims_d,  actual_chunk, (size_t)ndims * sizeof(hsize_t),
                               cudaMemcpyHostToDevice);
                    vol_memcpy(d_chunk_start_d, chunk_start,  (size_t)ndims * sizeof(hsize_t),
                               cudaMemcpyHostToDevice);
                }
                int threads = 256;
                int blocks  = (int)((actual_elems + threads - 1) / threads);
                gather_chunk_kernel<<<blocks, threads, 0, gather_stream>>>(
                    static_cast<const uint8_t*>(d_buf), d_owned,
                    ndims, elem_size, actual_elems,
                    d_dset_dims, d_chunk_dims_d, d_chunk_start_d);
                cudaError_t k_ce   = cudaGetLastError();
                cudaError_t syn_ce = cudaStreamSynchronize(gather_stream);
                if (k_ce != cudaSuccess || syn_ce != cudaSuccess) {
                    fprintf(stderr,
                            "gpu_trace_chunked_write: chunk %zu gather_chunk_kernel "
                            "failed: launch=%s sync=%s\n",
                            ci, cudaGetErrorString(k_ce), cudaGetErrorString(syn_ce));
                    cudaFree(d_owned); ret = -1; break;
                }
                src = d_owned;
            }

            if (!src) {
                fprintf(stderr,
                        "gpu_trace_chunked_write: chunk %zu src==nullptr after gather\n",
                        ci);
                ret = -1; break;
            }

            /* ---- NN inference: get action + per_config[32] ---- */
            int chosen_nn_action = 0;
            float infer_ratio = 0, infer_ct = 0, infer_dt = 0, infer_psnr = 0;
            float infer_rmse = 0, infer_max_err = 0, infer_mae = 0, infer_ssim = 0;
            int top_actions[32] = {};
            float pred_costs[32] = {};
            NNDebugPerConfig per_config[32] = {};
            bool has_inference = false;
            float infer_nn_ms = 0, infer_stats_ms = 0;

            if (use_nn && infer_ctx) {
                gpucompress_error_t ie = gpucompress_infer_gpu(
                    src, actual_bytes, &cfg, nullptr, infer_ctx,
                    &chosen_nn_action, &infer_ratio, &infer_ct, &infer_dt, &infer_psnr,
                    top_actions, pred_costs,
                    &infer_rmse, &infer_max_err, &infer_mae, &infer_ssim,
                    per_config);
                if (ie == GPUCOMPRESS_SUCCESS && chosen_nn_action >= 0) {
                    has_inference = true;
                    cudaEventElapsedTime(&infer_nn_ms, infer_ctx->nn_start, infer_ctx->nn_stop);
                    cudaEventElapsedTime(&infer_stats_ms, infer_ctx->stats_start, infer_ctx->stats_stop);
                }
            }

            /* ---- Exhaustive trace: all 64 configs ---- */
            TraceConfigResult results[64] = {};
            run_trace_exhaustive_chunk(
                src, actual_bytes,
                has_inference ? chosen_nn_action : -1,
                cfg.error_bound,
                has_inference ? per_config : nullptr,
                results);

            /* ---- Choose action by lowest predicted cost among valid results ---- */
            int chosen_trace_action = has_inference ? chosen_nn_action : 0;
            {
                float best_pred = 1e30f;
                for (int i = 0; i < 64; i++) {
                    if (results[i].valid && results[i].pred_cost > 0.0f
                            && results[i].pred_cost < best_pred) {
                        best_pred = results[i].pred_cost;
                        chosen_trace_action = results[i].action_id;
                    }
                }
            }

            /* ---- Find the chosen result ---- */
            const TraceConfigResult* chosen_result = nullptr;
            for (int i = 0; i < 64; i++) {
                if (results[i].valid && results[i].action_id == chosen_trace_action) {
                    chosen_result = &results[i]; break;
                }
            }

            /* ---- Compress the chosen config and write to file ---- */
            {
                static const float s_quant_ebs[4] = { 0.0f, 0.1f, 0.01f, 0.001f };
                int algo_idx  = chosen_trace_action / 8;
                int shuf_idx  = (chosen_trace_action % 8) / 4;
                int quant_idx = chosen_trace_action % 4;

                gpucompress_config_t chosen_cfg = cfg;
                chosen_cfg.algorithm   = static_cast<gpucompress_algorithm_t>(algo_idx + 1);
                chosen_cfg.error_bound = (double)s_quant_ebs[quant_idx];
                chosen_cfg.preprocessing = 0;
                if (shuf_idx > 0)  chosen_cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
                if (quant_idx > 0) chosen_cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;

                size_t comp_sz = max_comp;
                gpucompress_error_t ce = gpucompress_compress_gpu(
                    src, actual_bytes, d_comp, &comp_sz, &chosen_cfg, nullptr, nullptr);

                if (ce == GPUCOMPRESS_SUCCESS && comp_sz > 0) {
                    if (cudaMemcpy(h_comp, d_comp, comp_sz,
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        write_chunk_to_native(o->under_object, o->under_vol_id,
                                              chunk_start, h_comp, comp_sz, dxpl_id);
                    }
                }
            }

            /* ---- SGD / Exploration using trace measurements ---- */
            if (has_inference && infer_ctx && infer_ctx->d_stats && chosen_result
                    && !g_best_mode.load() && g_online_learning_enabled) {

                float w0 = g_rank_w0, w1 = g_rank_w1, w2 = g_rank_w2;
                float bw = fmaxf(1.0f, g_measured_bw_bytes_per_ms);
                static constexpr float TIME_FLOOR = 5.0f, RATIO_CAP = 100.0f;
                float r_ct   = fmaxf(TIME_FLOOR, chosen_result->real_comp_ms);
                float r_dt   = fmaxf(TIME_FLOOR, chosen_result->real_decomp_ms);
                float r_r    = fminf(RATIO_CAP,  chosen_result->real_ratio);
                float r_cost = w0*r_ct + w1*r_dt + w2*(float)actual_bytes/(r_r*bw);
                float p_cost = chosen_result->pred_cost;
                float mape   = (r_cost > 1e-6f) ? fabsf(p_cost - r_cost) / r_cost : 0.0f;

                /* SGD: proportional update */
                if (mape > g_reinforce_mape_threshold) {
                    SGDSample sample{};
                    sample.action             = chosen_result->nn_idx;
                    sample.actual_ratio       = chosen_result->real_ratio;
                    sample.actual_comp_time   = chosen_result->real_comp_ms;
                    sample.actual_decomp_time = chosen_result->real_decomp_ms;
                    sample.actual_psnr        = chosen_result->real_psnr;
                    std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                    gpucompress::runNNSGDCtx(infer_ctx->d_stats, &sample, 1,
                        actual_bytes, cfg.error_bound, g_reinforce_lr, infer_ctx);
                }

                /* Exploration: randomly pick N from the 64 measured configs */
                bool do_explore = (mape > (float)g_exploration_threshold) || g_best_mode.load();
                if (do_explore) {
                    int n_explore = (g_exploration_k_override > 0) ? g_exploration_k_override : 3;
                    /* Collect valid non-chosen results */
                    int valid_idxs[64]; int n_valid = 0;
                    for (int i = 0; i < 64; i++) {
                        if (results[i].valid && results[i].action_id != chosen_trace_action)
                            valid_idxs[n_valid++] = i;
                    }
                    /* Fisher-Yates shuffle */
                    for (int i = n_valid - 1; i > 0; i--) {
                        int j = rand() % (i + 1);
                        int tmp = valid_idxs[i]; valid_idxs[i] = valid_idxs[j]; valid_idxs[j] = tmp;
                    }
                    int n = (n_explore < n_valid) ? n_explore : n_valid;
                    if (n > 0) {
                        SGDSample explore_samples[64];
                        for (int k = 0; k < n; k++) {
                            const TraceConfigResult& er = results[valid_idxs[k]];
                            SGDSample& s = explore_samples[k];
                            s.action             = er.nn_idx;
                            s.actual_ratio       = er.real_ratio;
                            s.actual_comp_time   = er.real_comp_ms;
                            s.actual_decomp_time = er.real_decomp_ms;
                            s.actual_psnr        = er.real_psnr;
                        }
                        std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);
                        gpucompress::runNNSGDCtx(infer_ctx->d_stats, explore_samples, n,
                            actual_bytes, cfg.error_bound, g_reinforce_lr, infer_ctx);
                    }
                }
            }

            /* Free owned gather buffer */
            if (d_owned) { cudaFree(d_owned); d_owned = nullptr; }
        } /* chunk loop */

        if (d_dset_dims) free_dim_arrays(d_dset_dims, d_chunk_dims_d, d_chunk_start_d);
        if (infer_ctx) gpucompress::releaseCompContext(infer_ctx);
        cudaStreamDestroy(gather_stream);
        gpucompress::DiagnosticsStore::instance().flushTrace();

        cudaFree(d_comp);
        cudaFreeHost(h_comp);
    }

done_trace:
    if (need_close_space) H5Sclose(actual_space_id);
    return ret;
}

/* ============================================================
 * gpu_aware_chunked_write
 * ============================================================ */

static herr_t
gpu_aware_chunked_write(H5VL_gpucompress_t *o,
                        hid_t               mem_type_id,
                        hid_t               file_space_id,
                        hid_t               dxpl_id,
                        const void         *d_buf)
{
    herr_t ret = -1;
    hid_t  actual_space_id = H5I_INVALID_HID;
    int    need_close_space = 0;

    /* ---- Resolve H5S_ALL → actual dataset space ---- */
    if (file_space_id == H5S_ALL) {
        VOL_TRACE("  → H5VLdataset_get(native, GET_SPACE)  [H5S_ALL → resolve actual dims]");
        H5VL_dataset_get_args_t get_args;
        memset(&get_args, 0, sizeof(get_args));
        get_args.op_type             = H5VL_DATASET_GET_SPACE;
        get_args.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id,
                            &get_args, dxpl_id, NULL) < 0) goto done;
        actual_space_id  = get_args.args.get_space.space_id;
        need_close_space = 1;
    } else {
        actual_space_id = file_space_id;
    }

    {
        int ndims = H5Sget_simple_extent_ndims(actual_space_id);
        if (ndims <= 0 || ndims > 32) goto done;

        hsize_t dset_dims[32];
        H5Sget_simple_extent_dims(actual_space_id, dset_dims, NULL);

        /* ---- Get chunk dimensions from DCPL ---- */
        hsize_t chunk_dims[32] = {0};
        if (H5Pget_chunk(o->dcpl_id, ndims, chunk_dims) < 0) goto done;

        /* ---- Get element size ---- */
        size_t elem_size = H5Tget_size(mem_type_id);
        if (elem_size == 0) goto done;

        /* ---- Get gpucompress config from DCPL ---- */
        /* NOTE: caller (H5VL_gpucompress_dataset_write) already verified the
         * filter is present and dispatches to gpu_fallback_dh_write() if not.
         * This check is kept as a defensive safety net only. */
        gpucompress_config_t cfg;
        if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) != 0) {
            fprintf(stderr, "gpucompress VOL [internal]: gpu_aware_chunked_write "
                            "called without gpucompress filter — should not happen.\n");
            goto done;
        }

        /* ── Raw bypass mode (GPUCOMPRESS_VOL_BYPASS=1) ──
         * When enabled, skip stats+NN inference AND nvCOMP entirely. Each
         * chunk is still iterated, D→H copied via the pinned pool, and
         * written through the same I/O thread, but the compression kernel
         * is replaced with a no-op pass-through. Yields a raw-I/O baseline
         * whose timings are directly comparable to the compression path. */
        const bool bypass = vol_bypass_enabled();

        size_t chunk_elems = 1;
        for (int d = 0; d < ndims; d++) chunk_elems *= (size_t)chunk_dims[d];
        size_t chunk_bytes = chunk_elems * elem_size;
        size_t max_comp    = gpucompress_max_compressed_size(chunk_bytes);

        /* ---- 3-Stage concurrent pipeline: M4_N_COMP_WORKERS + I/O thread ---- */
#define N_COMP_WORKERS M4_N_COMP_WORKERS
        struct WorkItem {
            const uint8_t* src;      /* source for compression */
            uint8_t*       d_owned;  /* non-NULL: worker cudaFrees after compression */
            size_t         sz;
            hsize_t        cs[32];
            int            nd;
            bool           valid;
            /* Pre-computed inference results (sequential-inference pipeline) */
            bool           has_inference;
            int            action;
            float          predicted_ratio;
            float          predicted_comp_time;
            float          predicted_decomp_time;
            float          predicted_psnr;
            float          predicted_rmse;
            float          predicted_max_error;
            float          predicted_mae;
            float          predicted_ssim;
            int            top_actions[32];
            float          predicted_costs[32];
            NNDebugPerConfig per_config[32]; /* per-NN-action predictions (populated in TRACE mode) */
            /* Timing from Stage 1 inference (ms) */
            float          infer_nn_ms;
            float          infer_stats_ms;
            /* Pre-computed stats from Stage 1 (D→D copy from infer_ctx).
             * Avoids redundant stats recomputation in Stage 2 for SGD. */
            AutoStatsGPU*  d_stats_copy;   /* stats for SGD; ring-managed or worker-owned */
            bool           d_stats_ring_managed; /* true: from ring buffer (don't cudaFree) */
            /* Detailed timing: VOL Stage 1 per-chunk */
            float          vol_stats_malloc_ms;
            float          vol_stats_copy_ms;
            float          vol_wq_post_wait_ms;
        };

        /* M4 fix: use session-level buffers from write_ctx (persistent across writes).
         * Allocate on first write, reuse on subsequent writes, free at dataset close. */
#define N_IO_BUFS M4_N_IO_BUFS

        /* All pipeline-local variables declared up-front (before any goto,
         * nvcc requires no init-bypass for non-trivial types). */
        double _pipeline_start = 0, _s1_start = 0, _t_drain_end = 0;
        double io_write_total_ms = 0;  /* direct S3 measurement from I/O thread */
        size_t io_bytes_written  = 0;  /* total compressed bytes written to disk */
        size_t life_n_chunks     = 0;  /* chunks this call (for lifetime tracking) */
        size_t life_chunk_bytes  = 0;  /* uncompressed chunk size (for lifetime tracking) */
        double worker_comp_ms[M4_N_COMP_WORKERS];
        memset(worker_comp_ms, 0, sizeof(worker_comp_ms));
        hsize_t *d_dset_dims = NULL, *d_chunk_dims = NULL, *d_chunk_start = NULL;
        uint8_t* d_comp_w[N_COMP_WORKERS] = {};
        AutoStatsGPU** d_stats_ring = nullptr;
        size_t d_stats_ring_count = 0;
        std::atomic<herr_t>     worker_err{0};
        std::mutex              pool_mtx;
        std::condition_variable pool_cv;
        std::queue<void*>       pool_free;
        auto pool_acquire = [&]() -> void* {
            std::unique_lock<std::mutex> lk(pool_mtx);
            pool_cv.wait(lk, [&]{ return !pool_free.empty() || worker_err.load() != 0; });
            if (pool_free.empty()) return nullptr;
            void* b = pool_free.front(); pool_free.pop();
            return b;
        };
        auto pool_release = [&](void* b) {
            { std::lock_guard<std::mutex> lk(pool_mtx); pool_free.push(b); }
            pool_cv.notify_one();
        };
        std::mutex              wq_mtx;
        std::condition_variable wq_full_cv, wq_ready_cv;
        std::queue<WorkItem>    wq;
        struct IOItem { void *data; size_t sz; hsize_t cs[32]; int nd; };
        std::mutex              io_mtx;
        std::condition_variable io_cv;
        std::queue<IOItem>      io_q;
        std::atomic<bool>       io_done_flag{false};
        herr_t                  io_err       = 0;
        bool                    io_started   = false;
        std::thread             io_thr;

        double _vol_func_start = _now_ms();

        if (!o->write_ctx) {
            o->write_ctx = new VolWriteCtx{};  /* value-init zeros all members */
            /* P0: try reclaim buffers from global cache (always reclaim if valid;
             * the realloc path below handles size growth if cached < needed). */
            {
                std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
                if (s_buf_cache.valid) {
                    memcpy(o->write_ctx->d_comp_w, s_buf_cache.d_comp_w, sizeof(s_buf_cache.d_comp_w));
                    memcpy(o->write_ctx->io_pool_bufs, s_buf_cache.io_pool_bufs, sizeof(s_buf_cache.io_pool_bufs));
                    o->write_ctx->max_comp = s_buf_cache.max_comp;
                    o->write_ctx->initialized = true;
                    s_buf_cache.valid = false;
                    memset(s_buf_cache.d_comp_w, 0, sizeof(s_buf_cache.d_comp_w));
                    memset(s_buf_cache.io_pool_bufs, 0, sizeof(s_buf_cache.io_pool_bufs));
                }
            }
        }
        VolWriteCtx* wctx = o->write_ctx;

        /* (Re)allocate if first use or chunk size grew */
        if (!wctx->initialized || wctx->max_comp < max_comp) {
            if (wctx->initialized) {
                for (int w = 0; w < N_COMP_WORKERS; w++) {
                    if (wctx->d_comp_w[w]) { cudaFree(wctx->d_comp_w[w]); wctx->d_comp_w[w] = nullptr; }
                }
                for (int i = 0; i < N_IO_BUFS; i++) {
                    if (wctx->io_pool_bufs[i]) { cudaFreeHost(wctx->io_pool_bufs[i]); wctx->io_pool_bufs[i] = nullptr; }
                }
            }
            bool ok = true;
            for (int w = 0; w < N_COMP_WORKERS && ok; w++) {
                if (cudaMalloc(&wctx->d_comp_w[w], max_comp) != cudaSuccess) ok = false;
            }
            for (int i = 0; i < N_IO_BUFS && ok; i++) {
                if (cudaMallocHost(&wctx->io_pool_bufs[i], max_comp) != cudaSuccess) ok = false;
            }
            if (!ok) {
                /* Clean up partial allocations to avoid leaks on retry */
                for (int w = 0; w < N_COMP_WORKERS; w++) {
                    if (wctx->d_comp_w[w]) { cudaFree(wctx->d_comp_w[w]); wctx->d_comp_w[w] = nullptr; }
                }
                for (int i = 0; i < N_IO_BUFS; i++) {
                    if (wctx->io_pool_bufs[i]) { cudaFreeHost(wctx->io_pool_bufs[i]); wctx->io_pool_bufs[i] = nullptr; }
                }
                wctx->initialized = false;
                wctx->max_comp = 0;
                ret = -1;
                goto done_write;
            }
            wctx->max_comp = max_comp;
            wctx->initialized = true;
        }

        /* Alias persistent buffers + populate free pool */
        for (int w = 0; w < N_COMP_WORKERS; w++) d_comp_w[w] = wctx->d_comp_w[w];
        for (int i = 0; i < N_IO_BUFS; i++) pool_free.push(wctx->io_pool_bufs[i]);

        /* ---- Launch I/O writer thread (Stage 3) ---- */
        io_thr = std::thread([&]() {
            while (true) {
                IOItem item;
                { std::unique_lock<std::mutex> lk(io_mtx);
                  io_cv.wait(lk, [&]{ return !io_q.empty() || io_done_flag; });
                  if (io_q.empty()) break;
                  item = io_q.front(); io_q.pop(); }
                io_cv.notify_one();
                double _t_io_w = _now_ms();
                herr_t r = write_chunk_to_native(o->under_object, o->under_vol_id,
                                                  item.cs, item.data, item.sz, dxpl_id);
                io_write_total_ms += _now_ms() - _t_io_w;
                io_bytes_written  += item.sz;
                pool_release(item.data);
                if (r < 0) { std::lock_guard<std::mutex> lk(io_mtx); io_err = r; }
            }
        });
        io_started = true;

        /* ---- Launch worker threads (Stage 2) + Stage 1 ---- */
        {
            std::vector<std::thread> workers;
            workers.reserve(N_COMP_WORKERS);
            for (int w = 0; w < N_COMP_WORKERS; w++) {
                workers.emplace_back([&, w]() {
                    double w_total = 0;
                    while (true) {
                        WorkItem wi;
                        { std::unique_lock<std::mutex> lk(wq_mtx);
                          wq_ready_cv.wait(lk, [&]{ return !wq.empty() || worker_err.load() != 0; });
                          if (wq.empty()) break;
                          wi = wq.front(); wq.pop(); }
                        wq_full_cv.notify_one();
                        if (!wi.valid) break;

                        double _w_start = _now_ms();
                        size_t comp_sz = max_comp;
                        gpucompress_stats_t wstats = {};
                        gpucompress_error_t ce = GPUCOMPRESS_SUCCESS;
                        int diag_slot = -1;
                        if (bypass) {
                            /* Raw pass-through: skip nvCOMP entirely. The D→H
                             * copy below will move wi.src bytes directly into
                             * the pinned pool; the I/O thread then writes them
                             * uncompressed. comp_sz == wi.sz so downstream
                             * accounting treats bytes_out == bytes_in. */
                            comp_sz = wi.sz;
                        } else if (wi.has_inference) {
                            ce = gpucompress_compress_with_action_gpu(
                                wi.src, wi.sz, d_comp_w[w], &comp_sz, &cfg, &wstats, NULL,
                                wi.action, wi.predicted_ratio, wi.predicted_comp_time,
                                wi.predicted_decomp_time, wi.predicted_psnr,
                                wi.top_actions, wi.predicted_costs,
                                wi.infer_nn_ms, wi.infer_stats_ms,
                                wi.d_stats_copy, &diag_slot,
                                wi.predicted_rmse, wi.predicted_max_error,
                                wi.predicted_mae, wi.predicted_ssim);
                        } else {
                            ce = gpucompress_compress_gpu(
                                wi.src, wi.sz, d_comp_w[w], &comp_sz, &cfg, &wstats, NULL);
                        }

                        /* Free per-chunk owned buffers after compression completes.
                         * In bypass mode, defer freeing wi.d_owned until after the
                         * D→H copy below (since wi.src is the copy source). */
                        if (wi.d_owned && !bypass) { cudaFree(wi.d_owned); wi.d_owned = NULL; }
                        if (wi.d_stats_copy && !wi.d_stats_ring_managed) { cudaFree(wi.d_stats_copy); wi.d_stats_copy = NULL; }

                        if (ce != GPUCOMPRESS_SUCCESS) {
                            fprintf(stderr, "gpucompress VOL: worker %d compress failed "
                                    "(err=%d, chunk_sz=%zu)\n", w, (int)ce, wi.sz);
                            worker_err.store((herr_t)-1);
                            wq_ready_cv.notify_all();
                            pool_cv.notify_all();
                            break;
                        }

                        s_chunks_comp++;

                        /* Acquire a pool buffer; D→H directly into it (no extra memcpy) */
                        double _t_pa = _now_ms();
                        void *hbuf = pool_acquire();
                        float vol_pool_ms = (float)(_now_ms() - _t_pa);
                        if (!hbuf) {
                            worker_err.store((herr_t)-1);
                            wq_ready_cv.notify_all();
                            pool_cv.notify_all();
                            break;
                        }
                        /* Bypass: copy the raw input chunk (wi.src) straight to
                         * host. Compression: copy the compressor output buffer
                         * (d_comp_w[w]). Same pinned-pool destination, same
                         * timing slot, so the D→H latency is directly
                         * comparable between modes. */
                        const void* _d2h_src = bypass ? (const void*)wi.src
                                                      : (const void*)d_comp_w[w];
                        double _t_d2h = _now_ms();
                        if (cudaMemcpy(hbuf, _d2h_src, comp_sz,
                                       cudaMemcpyDeviceToHost) != cudaSuccess) {
                            pool_release(hbuf);
                            worker_err.store((herr_t)-1);
                            wq_ready_cv.notify_all();
                            pool_cv.notify_all();
                            break;
                        }
                        float vol_d2h_ms = (float)(_now_ms() - _t_d2h);

                        /* In bypass mode, wi.d_owned was the source of the D→H
                         * copy above. It's safe to free now. */
                        if (bypass && wi.d_owned) { cudaFree(wi.d_owned); wi.d_owned = NULL; }

                        IOItem item;
                        item.data = hbuf;
                        item.sz   = comp_sz;
                        item.nd   = wi.nd;
                        memcpy(item.cs, wi.cs, (size_t)wi.nd * sizeof(hsize_t));

                        /* E3 fix: raise I/O queue cap to match pinned buffer pool.
                         * Workers no longer block while pool buffers are available. */
                        double _t_io = _now_ms();
                        { std::unique_lock<std::mutex> lk(io_mtx);
                          io_cv.wait(lk, [&]{ return io_q.size() < (size_t)N_IO_BUFS || io_done_flag; });
                          io_q.push(item); }
                        float vol_io_wait_ms = (float)(_now_ms() - _t_io);
                        io_cv.notify_one();

                        /* Write VOL timing back to chunk diagnostic */
                        if (diag_slot >= 0) {
                            auto& store = gpucompress::DiagnosticsStore::instance();
                            store.recordVolTiming(diag_slot,
                                vol_pool_ms, vol_d2h_ms, vol_io_wait_ms);
                            store.recordS1Timing(diag_slot,
                                wi.vol_stats_malloc_ms, wi.vol_stats_copy_ms, wi.vol_wq_post_wait_ms);
                        }

                        w_total += _now_ms() - _w_start;
                    }
                    /* Store per-worker total wall time (full iteration: compress + cudaFree + pool + D2H + I/O post) */
                    worker_comp_ms[w] = w_total;
                });
            }

            /* ---- Stage 1: iterate chunks, sequential inference + post WorkItems ---- */
            s_setup_ms = _now_ms() - _vol_func_start;
            _pipeline_start = _now_ms();
            _s1_start = _pipeline_start;
            {
                /* Non-default stream for gather kernel: avoids null-stream serialization
                 * against the 8 CompContext worker streams. */
                cudaStream_t gather_stream = nullptr;
                if (cudaStreamCreate(&gather_stream) != cudaSuccess) {
                    fprintf(stderr, "gpucompress VOL: cudaStreamCreate failed for gather_stream\n");
                    ret = -1;
                    goto done_write;
                }
                hsize_t num_chunks[32];
                for (int d = 0; d < ndims; d++)
                    num_chunks[d] = (dset_dims[d] + chunk_dims[d] - 1) / chunk_dims[d];
                size_t total_chunks = 1;
                for (int d = 0; d < ndims; d++) total_chunks *= (size_t)num_chunks[d];
                life_n_chunks = total_chunks;
                life_chunk_bytes = chunk_bytes;

                ret = 0;
                s_gpu_writes++;

                /* I6 fix: contiguity depends only on chunk_dims vs dset_dims,
                 * which are constant across chunks — compute once before loop. */
                bool contiguous = true;
                for (int d = 1; d < ndims; d++) {
                    if (chunk_dims[d] != dset_dims[d]) { contiguous = false; break; }
                }

                /* Acquire dedicated CompContext for sequential inference (slot 8).
                 * This lets the main thread run stats+inference without racing
                 * with workers, and each chunk sees the latest SGD update. */
                bool use_seq_inference = (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO) && !bypass;
                CompContext* infer_ctx = nullptr;
                if (use_seq_inference) {
                    infer_ctx = gpucompress::acquireCompContext();
                    if (!infer_ctx) {
                        fprintf(stderr, "gpucompress VOL: failed to acquire inference CompContext\n");
                        use_seq_inference = false;
                    }
                }

                /* Pre-allocate d_stats_copy ring buffer (one per chunk).
                 * Eliminates per-chunk cudaMalloc (~0.31ms each) from the hot path.
                 * Each entry holds a copy of infer_ctx->d_stats for SGD in Stage 2.
                 * Safe: each chunk index is used exactly once, freed after worker join. */
                if (use_seq_inference && total_chunks > 0) {
                    d_stats_ring = (AutoStatsGPU**)calloc(total_chunks, sizeof(AutoStatsGPU*));
                    if (d_stats_ring) {
                        bool ring_ok = true;
                        for (size_t i = 0; i < total_chunks && ring_ok; i++) {
                            if (cudaMalloc(&d_stats_ring[i], sizeof(AutoStatsGPU)) != cudaSuccess)
                                ring_ok = false;
                        }
                        if (ring_ok) {
                            d_stats_ring_count = total_chunks;
                        } else {
                            /* Partial alloc cleanup */
                            for (size_t i = 0; i < total_chunks; i++) {
                                if (d_stats_ring[i]) cudaFree(d_stats_ring[i]);
                            }
                            free(d_stats_ring);
                            d_stats_ring = nullptr;
                        }
                    }
                }

                for (size_t ci = 0; ci < total_chunks && ret == 0; ci++) {
                    /* ---- Compute N-D chunk coordinates ---- */
                    hsize_t chunk_idx[32];
                    size_t  remaining = ci;
                    for (int d = ndims - 1; d >= 0; d--) {
                        chunk_idx[d] = (hsize_t)(remaining % (size_t)num_chunks[d]);
                        remaining   /= (size_t)num_chunks[d];
                    }
                    hsize_t chunk_start[32];
                    for (int d = 0; d < ndims; d++)
                        chunk_start[d] = chunk_idx[d] * chunk_dims[d];

                    hsize_t actual_chunk[32];
                    size_t  actual_elems = 1;
                    for (int d = 0; d < ndims; d++) {
                        actual_chunk[d] = chunk_dims[d];
                        if (chunk_start[d] + actual_chunk[d] > dset_dims[d])
                            actual_chunk[d] = dset_dims[d] - chunk_start[d];
                        actual_elems *= (size_t)actual_chunk[d];
                    }
                    size_t actual_bytes = actual_elems * elem_size;

                    WorkItem wi = {};
                    wi.nd    = ndims;
                    wi.valid = true;
                    wi.has_inference = false;
                    memcpy(wi.cs, chunk_start, (size_t)ndims * sizeof(hsize_t));

                    VOL_TRACE("  chunk[%zu] off=%llu actual=%zuB  path=%s",
                              ci, (unsigned long long)chunk_start[0], actual_bytes,
                              contiguous ? "contiguous" : "gather-kernel");

                    if (contiguous) {
                        size_t off = 0, stride = elem_size;
                        for (int d = ndims - 1; d >= 0; d--) {
                            off    += (size_t)chunk_start[d] * stride;
                            stride *= (size_t)dset_dims[d];
                        }
                        const uint8_t *raw = static_cast<const uint8_t*>(d_buf) + off;
                        if (actual_bytes == chunk_bytes) {
                            /* Full-size contiguous chunk: no copy needed */
                            wi.src     = raw;
                            wi.sz      = actual_bytes;
                            wi.d_owned = NULL;
                        } else {
                            /* Partial boundary: copy only actual_bytes (no zero-padding) */
                            uint8_t *d_owned = NULL;
                            if (cudaMalloc(&d_owned, actual_bytes) != cudaSuccess)
                                { ret = -1; break; }
                            vol_memcpy(d_owned, raw, actual_bytes, cudaMemcpyDeviceToDevice);
                            wi.src     = d_owned;
                            wi.sz      = actual_bytes;
                            wi.d_owned = d_owned;
                        }
                    } else {
                        /* Non-contiguous: gather into fresh device buffer */
                        uint8_t *d_owned = NULL;
                        if (cudaMalloc(&d_owned, chunk_bytes) != cudaSuccess)
                            { ret = -1; break; }
                        if (!d_dset_dims) {
                            if (alloc_dim_arrays(ndims, dset_dims, actual_chunk, chunk_start,
                                                &d_dset_dims, &d_chunk_dims, &d_chunk_start) < 0)
                                { cudaFree(d_owned); ret = -1; break; }
                        } else {
                            vol_memcpy(d_chunk_dims,  actual_chunk, (size_t)ndims * sizeof(hsize_t),
                                       cudaMemcpyHostToDevice);
                            vol_memcpy(d_chunk_start, chunk_start,  (size_t)ndims * sizeof(hsize_t),
                                       cudaMemcpyHostToDevice);
                        }
                        int threads = 256;
                        int blocks  = (int)((actual_elems + threads - 1) / threads);
                        gather_chunk_kernel<<<blocks, threads, 0, gather_stream>>>(
                            static_cast<const uint8_t*>(d_buf), d_owned,
                            ndims, elem_size, actual_elems,
                            d_dset_dims, d_chunk_dims, d_chunk_start);
                        if (cudaGetLastError() != cudaSuccess ||
                            cudaStreamSynchronize(gather_stream) != cudaSuccess)
                            { cudaFree(d_owned); ret = -1; break; }
                        wi.src     = d_owned;
                        wi.sz      = actual_bytes;
                        wi.d_owned = d_owned;
                    }

                    /* ---- Sequential inference: run stats+NN on main thread ----
                     *
                     * Design intent: if chunk N has high MAPE prediction error,
                     * its worker fires SGD to update NN weights in-place on
                     * g_sgd_stream (via nnSGDKernel). Chunk N+K's inference
                     * then sees corrected weights through the GPU-level barrier:
                     *   runNNFusedInferenceCtx → cudaStreamWaitEvent(g_sgd_done)
                     * which prevents the inference kernel from reading d_nn_weights
                     * until the SGD kernel on g_sgd_stream completes.
                     *
                     * The SGD visibility depends on timing. cudaStreamWaitEvent
                     * captures the event's most recent recording at HOST call time.
                     * For chunk N+1 to see chunk N's SGD, the worker must finish
                     * compression (~20ms) + fire SGD + cudaEventRecord(g_sgd_done)
                     * BEFORE the main thread reaches chunk N+1's inference call.
                     * Since S1 processes ~7ms per chunk, chunk N+1's WaitEvent
                     * typically captures a PRIOR SGD recording (already completed),
                     * so chunk N+1 reads old weights. The update becomes visible
                     * at chunk N+K where K ≈ ceil(compress_ms / per_chunk_s1_ms),
                     * typically 3-4 chunks later.
                     *
                     * With many chunks (64+), this converges quickly: early SGD
                     * corrections propagate to later chunks within the same write.
                     * With few chunks (2-4), most chunks run with stale weights
                     * and the SGD benefit is deferred to the next H5Dwrite call.
                     *
                     * In heuristic mode, we skip the NN and use entropy-based
                     * rules instead — stats kernel still runs for entropy. */
                    if (use_seq_inference && wi.src && wi.sz > 0) {
                        bool use_heuristic = (g_selection_mode.load() == GPUCOMPRESS_SELECT_HEURISTIC);

                        if (use_heuristic) {
                            /* Heuristic path: run stats only, pick action from entropy.
                             * Use runStatsKernelsNoSync + manual D→H so stats_stop
                             * records before the sync (pure GPU time, not wall-clock). */
                            double h_entropy = 0, h_mad = 0, h_deriv = 0;
                            cudaEventRecord(infer_ctx->stats_start, infer_ctx->stream);
                            AutoStatsGPU* d_heur = gpucompress::runStatsKernelsNoSync(
                                wi.src, wi.sz, infer_ctx->stream, infer_ctx);
                            cudaEventRecord(infer_ctx->stats_stop, infer_ctx->stream);
                            int src = -1;
                            if (d_heur) {
                                struct { double e, m, d; } h_r;
                                cudaError_t ce = cudaMemcpyAsync(&h_r, &d_heur->entropy,
                                    sizeof(h_r), cudaMemcpyDeviceToHost, infer_ctx->stream);
                                if (ce == cudaSuccess)
                                    ce = cudaStreamSynchronize(infer_ctx->stream);
                                if (ce == cudaSuccess) {
                                    h_entropy = h_r.e; h_mad = h_r.m; h_deriv = h_r.d;
                                    src = 0;
                                }
                            }

                            if (src != 0) {
                                fprintf(stderr, "gpucompress VOL: heuristic stats pipeline failed\n");
                            } else if (src == 0) {
                                wi.has_inference      = true;
                                wi.action             = heuristic_select_action(h_entropy);
                                wi.predicted_ratio    = 0.0f;  /* no prediction */
                                wi.predicted_comp_time = 0.0f;
                                wi.predicted_decomp_time = 0.0f;
                                wi.predicted_psnr     = 0.0f;
                                wi.predicted_rmse     = 0.0f;
                                wi.predicted_max_error = 0.0f;
                                wi.predicted_mae      = 0.0f;
                                wi.predicted_ssim     = 0.0f;
                                memset(wi.top_actions, 0, sizeof(wi.top_actions));
                                memset(wi.predicted_costs, 0, sizeof(wi.predicted_costs));
                                wi.infer_nn_ms    = 0.0f;
                                cudaEventElapsedTime(&wi.infer_stats_ms,
                                    infer_ctx->stats_start, infer_ctx->stats_stop);
                                wi.d_stats_copy   = nullptr;  /* no SGD in heuristic mode */
                                wi.d_stats_ring_managed = false;
                            }
                        } else {
                            /* NN path: full stats + inference */
                            int infer_action = -1;
                            float infer_ratio = 0, infer_ct = 0, infer_dt = 0, infer_psnr = 0;
                            float infer_rmse = 0, infer_max_err = 0, infer_mae = 0, infer_ssim = 0;
                            int infer_top[32] = {0};
                            float infer_costs[32] = {0};
                            gpucompress_error_t ie = gpucompress_infer_gpu(
                                wi.src, wi.sz, &cfg, nullptr, infer_ctx,
                                &infer_action, &infer_ratio, &infer_ct, &infer_dt, &infer_psnr,
                                infer_top, infer_costs,
                                &infer_rmse, &infer_max_err, &infer_mae, &infer_ssim,
                                nullptr);
                            if (ie == GPUCOMPRESS_SUCCESS && infer_action >= 0) {
                                wi.has_inference       = true;
                                wi.action              = infer_action;
                                wi.predicted_ratio     = infer_ratio;
                                wi.predicted_comp_time = infer_ct;
                                wi.predicted_decomp_time = infer_dt;
                                wi.predicted_psnr      = infer_psnr;
                                wi.predicted_rmse      = infer_rmse;
                                wi.predicted_max_error = infer_max_err;
                                wi.predicted_mae       = infer_mae;
                                wi.predicted_ssim      = infer_ssim;
                                memcpy(wi.top_actions, infer_top, sizeof(infer_top));
                                memcpy(wi.predicted_costs, infer_costs, sizeof(infer_costs));
                                /* Capture Stage 1 timing from infer_ctx CUDA events */
                                cudaEventElapsedTime(&wi.infer_nn_ms, infer_ctx->nn_start, infer_ctx->nn_stop);
                                cudaEventElapsedTime(&wi.infer_stats_ms, infer_ctx->stats_start, infer_ctx->stats_stop);
                                /* Copy stats from infer_ctx to a pre-allocated per-chunk buffer
                                 * so Stage 2 worker can reuse them for SGD without recomputing.
                                 * infer_ctx->d_stats will be overwritten by the next chunk. */
                                double _t_sm = _now_ms();
                                AutoStatsGPU* d_sc = nullptr;
                                if (d_stats_ring && ci < d_stats_ring_count) {
                                    d_sc = d_stats_ring[ci];  /* pre-allocated, no cudaMalloc */
                                } else {
                                    /* Fallback: per-chunk alloc (ring not available) */
                                    if (cudaMalloc(&d_sc, sizeof(AutoStatsGPU)) != cudaSuccess) {
                                        fprintf(stderr, "gpucompress VOL: stats copy alloc failed\n");
                                        ret = -1; break;
                                    }
                                }
                                wi.vol_stats_malloc_ms = (float)(_now_ms() - _t_sm);

                                double _t_sc = _now_ms();
                                cudaError_t sc_err = cudaMemcpyAsync(d_sc, infer_ctx->d_stats, sizeof(AutoStatsGPU),
                                                                      cudaMemcpyDeviceToDevice, infer_ctx->stream);
                                if (sc_err != cudaSuccess) {
                                    fprintf(stderr, "gpucompress VOL: stats copy memcpy failed\n");
                                    if (!d_stats_ring) cudaFree(d_sc);
                                    ret = -1; break;
                                }
                                sc_err = cudaStreamSynchronize(infer_ctx->stream);
                                if (sc_err != cudaSuccess) {
                                    fprintf(stderr, "gpucompress VOL: stats copy sync failed\n");
                                    if (!d_stats_ring) cudaFree(d_sc);
                                    ret = -1; break;
                                }
                                wi.vol_stats_copy_ms = (float)(_now_ms() - _t_sc);
                                wi.d_stats_copy = d_sc;
                                wi.d_stats_ring_managed = (d_stats_ring != nullptr && ci < d_stats_ring_count);
                            }
                        }
                    }

                    /* ---- Post WorkItem to bounded work queue ---- */
                    /* E3 fix: allow 2x workers in work queue so Stage 1 can
                     * pre-queue chunks while workers are posting to I/O. */
                    double _t_wq = _now_ms();
                    { std::unique_lock<std::mutex> lk(wq_mtx);
                      wq_full_cv.wait(lk, [&]{ return wq.size() < (size_t)(N_COMP_WORKERS * 2)
                                                      || ret != 0; });
                      wi.vol_wq_post_wait_ms = (float)(_now_ms() - _t_wq);
                      if (ret == 0)
                          wq.push(wi);
                      else {
                          if (wi.d_owned) { cudaFree(wi.d_owned); wi.d_owned = NULL; }
                          if (wi.d_stats_copy && !wi.d_stats_ring_managed) { cudaFree(wi.d_stats_copy); wi.d_stats_copy = NULL; }
                      } }
                    wq_ready_cv.notify_one();
                } /* chunk loop */
                if (infer_ctx) {
                    gpucompress::releaseCompContext(infer_ctx);
                    infer_ctx = nullptr;
                }
                cudaStreamDestroy(gather_stream);
            } /* Stage 1 scope */
            s_stage1_ms = _now_ms() - _s1_start;

            /* ---- Send sentinel WorkItems to stop all workers ---- */
            for (int w = 0; w < N_COMP_WORKERS; w++) {
                WorkItem sentinel = {};
                sentinel.valid = false;
                { std::unique_lock<std::mutex> lk(wq_mtx);
                  wq_full_cv.wait(lk, [&]{ return wq.size() < (size_t)(N_COMP_WORKERS * 2)
                                                  || worker_err.load() != 0; });
                  wq.push(sentinel); }
                wq_ready_cv.notify_one();
            }

            /* ---- Join workers and propagate errors ---- */
            for (auto &t : workers) t.join();
            _t_drain_end = _now_ms();
            s_drain_ms = _t_drain_end - _s1_start - s_stage1_ms; /* S1 end → workers joined */
            /* Actual Stage 2: max per-worker wall time (bottleneck) */
            for (int w = 0; w < N_COMP_WORKERS; w++) {
                if (worker_comp_ms[w] > s_s2_busy_ms) s_s2_busy_ms = worker_comp_ms[w];
            }
            if (worker_err.load() != 0 && ret == 0) ret = (herr_t)-1;

        } /* workers + Stage 1 scope */

        /* Free pre-allocated d_stats ring buffer (all workers joined, safe) */
        if (d_stats_ring) {
            for (size_t i = 0; i < d_stats_ring_count; i++) {
                if (d_stats_ring[i]) cudaFree(d_stats_ring[i]);
            }
            free(d_stats_ring);
            d_stats_ring = nullptr;
        }

done_write:
        /* ---- Signal I/O thread and join ---- */
        if (io_started) {
            /* Use _t_drain_end (captured after worker join) so that ring buffer
             * cleanup time is absorbed into io_drain — preserving the additive
             * invariant: stage1 + drain + io_drain = total. */
            { std::lock_guard<std::mutex> lk(io_mtx); io_done_flag = true; }
            io_cv.notify_all();
            io_thr.join();
            double _t_end = _now_ms();
            s_io_drain_ms = (_t_drain_end > 0) ? (_t_end - _t_drain_end) : 0;
            s_s3_busy_ms = io_write_total_ms;      /* actual I/O write time */
            s_total_ms = (_pipeline_start > 0) ? (_t_end - _pipeline_start) : 0;
            if (io_err < 0 && ret == 0) ret = -1;
        }

#ifndef NDEBUG
        if (s_total_ms > 0) {
            double _sum = s_stage1_ms + s_drain_ms + s_io_drain_ms;
            double _diff = _sum - s_total_ms;
            if (_diff < 0) _diff = -_diff;
            assert(_diff < 0.01); /* additive invariant (exact by construction) */
        }
#endif

        s_vol_func_ms = _now_ms() - _vol_func_start;

        /* ---- Accumulate lifetime totals ---- */
        g_life_vol_ms     += s_vol_func_ms;
        g_life_io_ms      += s_s3_busy_ms;
        g_life_s1_ms      += s_stage1_ms;
        g_life_s2_ms      += s_s2_busy_ms;
        g_life_drain_ms   += s_drain_ms;
        g_life_iodrain_ms += s_io_drain_ms;
        g_life_setup_ms   += s_setup_ms;
        g_life_bytes_in   += life_n_chunks * life_chunk_bytes;
        g_life_bytes_out  += io_bytes_written;
        g_life_chunks     += (int)life_n_chunks;
        g_life_writes++;
        std::call_once(g_atexit_flag, []{
            const char *dir = getenv("GPUCOMPRESS_RESULTS_DIR");
            if (!dir) dir = getenv("VPIC_RESULTS_DIR");
            if (dir) snprintf(g_life_output_dir, sizeof(g_life_output_dir), "%s", dir);
            std::atexit(gpucompress_vol_atexit);
        });

        /* ---- Cleanup (per-write only; buffers persist in write_ctx) ---- */
        if (d_dset_dims) free_dim_arrays(d_dset_dims, d_chunk_dims, d_chunk_start);
        /* M4 fix: d_comp_w and io_pool_bufs are NOT freed here — they persist
         * in wctx for reuse on the next H5Dwrite. Freed in dataset_close. */
#undef N_IO_BUFS
#undef N_COMP_WORKERS
    }

done:
    if (need_close_space) H5Sclose(actual_space_id);
    return ret;
}

/* ============================================================
 * Trace Mode: exhaustive per-chunk profiling
 *
 * Called in Stage 1 after NN inference when mode == TRACE.
 * Iterates all 32 action IDs, compresses with each config, decompresses,
 * and writes one CSV row per config per chunk to DiagnosticsStore.
 *
 * This is intentionally sequential and slow — Trace mode is for offline
 * analysis only.
 * ============================================================ */
static const char* s_algo_names_trace[] = {
    "lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"};

/* ── Quality metrics kernel (trace mode only) ──────────────────────────
 * One-pass GPU reduction over d_orig vs d_decomp (both float arrays).
 * Computes sums needed for RMSE, max_error, SSIM in one kernel launch.
 *
 * Output layout (9 floats, caller must initialize before launch):
 *   [0] sum of (x-y)^2        → RMSE = sqrt([0]/n)
 *   [1] max |x-y|             → max_error
 *   [2] data_min(x)           → data range for PSNR/SSIM C constants
 *   [3] data_max(x)
 *   [4] sum(x)                → mu_x
 *   [5] sum(x^2)              → var_x = [5]/n - mu_x^2
 *   [6] sum(y)                → mu_y
 *   [7] sum(y^2)              → var_y
 *   [8] sum(x*y)              → cov_xy = [8]/n - mu_x*mu_y
 * ──────────────────────────────────────────────────────────────────── */
__device__ static void trace_atomicMaxF(float* addr, float val) {
    int* ai = (int*)addr, assumed, old = *ai;
    do { assumed = old;
         if (__int_as_float(assumed) >= val) return;
         old = atomicCAS(ai, assumed, __float_as_int(val));
    } while (assumed != old);
}
__device__ static void trace_atomicMinF(float* addr, float val) {
    int* ai = (int*)addr, assumed, old = *ai;
    do { assumed = old;
         if (__int_as_float(assumed) <= val) return;
         old = atomicCAS(ai, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ static void traceQualityKernel(
    const float* __restrict__ d_orig,
    const float* __restrict__ d_decomp,
    int n, float* __restrict__ out)
{
    __shared__ float s[9][256];
    int t = threadIdx.x;
    float v[9] = {0,0, 3.4e38f,-3.4e38f, 0,0,0,0,0};

    for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
        float x = d_orig[i], y = d_decomp[i], d = x - y;
        v[0] += d * d;
        v[1]  = fmaxf(v[1], fabsf(d));
        v[2]  = fminf(v[2], x);
        v[3]  = fmaxf(v[3], x);
        v[4] += x;
        v[5] += x * x;
        v[6] += y;
        v[7] += y * y;
        v[8] += x * y;
    }
    for (int k = 0; k < 9; k++) s[k][t] = v[k];
    __syncthreads();
    for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) {
        if (t < s2) {
            s[0][t] += s[0][t+s2];
            s[1][t]  = fmaxf(s[1][t], s[1][t+s2]);
            s[2][t]  = fminf(s[2][t], s[2][t+s2]);
            s[3][t]  = fmaxf(s[3][t], s[3][t+s2]);
            for (int k = 4; k < 9; k++) s[k][t] += s[k][t+s2];
        }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(&out[0], s[0][0]);
        trace_atomicMaxF(&out[1], s[1][0]);
        trace_atomicMinF(&out[2], s[2][0]);
        trace_atomicMaxF(&out[3], s[3][0]);
        for (int k = 4; k < 9; k++) atomicAdd(&out[k], s[k][0]);
    }
}

/* Compute quality metrics on host from traceQualityKernel output. */
static void computeTraceQuality(const float* h, int n, float data_range,
                                 float* out_psnr, float* out_ssim, float* out_max_error)
{
    float rmse      = (n > 0) ? sqrtf(h[0] / (float)n) : 0.0f;
    *out_max_error  = h[1];
    float dr        = (data_range > 0.0f) ? data_range : 1.0f;
    *out_psnr       = (rmse < 1e-10f) ? 120.0f
                                      : fminf(120.0f, 20.0f * log10f(dr / rmse));
    /* Global SSIM using standard C1/C2 constants scaled by data range */
    float mu_x  = h[4] / n,  mu_y  = h[6] / n;
    float var_x = h[5] / n - mu_x*mu_x;
    float var_y = h[7] / n - mu_y*mu_y;
    float cov   = h[8] / n - mu_x*mu_y;
    float C1 = (0.01f * dr) * (0.01f * dr);
    float C2 = (0.03f * dr) * (0.03f * dr);
    float num = (2.0f*mu_x*mu_y + C1) * (2.0f*cov + C2);
    float den = (mu_x*mu_x + mu_y*mu_y + C1) * (var_x + var_y + C2);
    *out_ssim = (den > 0.0f) ? fmaxf(-1.0f, fminf(1.0f, num / den)) : 1.0f;
}

static void run_trace_exhaustive_chunk(
    const void*               d_src,
    size_t                    chunk_bytes,
    int                       nn_action,
    double                    error_bound,
    const NNDebugPerConfig*   per_config,   /* [32] NN predictions per action; may be nullptr */
    TraceConfigResult*        out_results)  /* [64], may be nullptr */
{
    size_t max_comp  = gpucompress_max_compressed_size(chunk_bytes);
    int    n_elems   = (int)(chunk_bytes / sizeof(float));
    void*  d_comp    = nullptr;
    void*  d_decomp  = nullptr;
    float* d_quality = nullptr;   /* 9-float reduction output */

    if (cudaMalloc(&d_comp,    max_comp)         != cudaSuccess) return;
    if (cudaMalloc(&d_decomp,  chunk_bytes)      != cudaSuccess) { cudaFree(d_comp); return; }
    if (cudaMalloc(&d_quality, 9 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_comp); cudaFree(d_decomp); return;
    }

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int chunk_id = gpucompress::DiagnosticsStore::instance().nextChunkId();

    /* Cost-ranking weights from globals (same formula as NN kernel) */
    float w0 = g_rank_w0, w1 = g_rank_w1, w2 = g_rank_w2;
    float bw = fmaxf(1.0f, g_measured_bw_bytes_per_ms);

    /* Full 64-config training space: 8 algos × 2 shuffle × 4 quant levels.
     * action_id = algo_idx * 8 + shuf_idx * 4 + quant_idx  → [0, 63] */
    static const float s_quant_ebs[4]   = { 0.0f, 0.1f, 0.01f, 0.001f };
    static const bool  s_quant_lossy[4] = { false, true, true, true    };

    for (int algo_idx = 0; algo_idx < 8; algo_idx++) {
      for (int shuf_idx = 0; shuf_idx < 2; shuf_idx++) {
        for (int quant_idx = 0; quant_idx < 4; quant_idx++) {

        int  action    = algo_idx * 8 + shuf_idx * 4 + quant_idx;
        bool use_shuf  = (shuf_idx > 0);
        bool use_quant = s_quant_lossy[quant_idx];
        float eb       = s_quant_ebs[quant_idx];

        char comp_lib[64];
        snprintf(comp_lib, sizeof(comp_lib), "%s%s%s",
                 s_algo_names_trace[algo_idx],
                 use_shuf  ? "+shuf"  : "",
                 use_quant ? "+quant" : "");

        gpucompress_config_t cfg{};
        cfg.algorithm   = static_cast<gpucompress_algorithm_t>(algo_idx + 1);
        cfg.error_bound = (double)eb;
        cfg.preprocessing = 0;
        if (use_shuf)  cfg.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
        if (use_quant) cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;

        /* ── Compress ── */
        size_t comp_sz = max_comp;
        cudaEventRecord(ev_start, stream);
        gpucompress_error_t ce = gpucompress_compress_gpu(
            d_src, chunk_bytes, d_comp, &comp_sz, &cfg, nullptr, stream);
        cudaEventRecord(ev_stop, stream);
        cudaStreamSynchronize(stream);
        if (ce != GPUCOMPRESS_SUCCESS || comp_sz == 0) continue;

        float comp_ms = 0.0f;
        cudaEventElapsedTime(&comp_ms, ev_start, ev_stop);
        float real_ratio = static_cast<float>(chunk_bytes) / static_cast<float>(comp_sz);

        /* ── Decompress ── */
        size_t decomp_sz = chunk_bytes;
        float  decomp_ms = 0.0f;
        cudaEventRecord(ev_start, stream);
        ce = gpucompress_decompress_gpu(d_comp, comp_sz, d_decomp, &decomp_sz, stream);
        cudaEventRecord(ev_stop, stream);
        cudaStreamSynchronize(stream);
        if (ce == GPUCOMPRESS_SUCCESS)
            cudaEventElapsedTime(&decomp_ms, ev_start, ev_stop);

        /* ── Real quality metrics ── */
        float real_psnr = 120.0f, real_ssim = 1.0f, real_max_error = 0.0f;
        if (use_quant && ce == GPUCOMPRESS_SUCCESS && n_elems > 0) {
            /* Initialize reduction buffer: sums=0, min=+inf, max=-inf */
            float init[9] = {0,0, 3.4e38f,-3.4e38f, 0,0,0,0,0};
            cudaMemcpy(d_quality, init, sizeof(init), cudaMemcpyHostToDevice);

            int threads = 256, blocks = min(256, (n_elems + threads-1)/threads);
            traceQualityKernel<<<blocks, threads, 0, stream>>>(
                (const float*)d_src, (const float*)d_decomp, n_elems, d_quality);
            cudaStreamSynchronize(stream);

            float h[9];
            cudaMemcpy(h, d_quality, sizeof(h), cudaMemcpyDeviceToHost);
            float data_range = h[3] - h[2];
            computeTraceQuality(h, n_elems, data_range,
                                &real_psnr, &real_ssim, &real_max_error);
        }
        /* Lossless: psnr=120, ssim=1, max_error=0 (bit-perfect) */

        /* ── Real cost ── */
        float real_cost = w0 * comp_ms + w1 * decomp_ms
                        + w2 * static_cast<float>(chunk_bytes) / (real_ratio * bw);

        /* ── Predicted values: map trace action → NN action ──
         * NN encoding: nn_idx = algo_idx + (quant_binary)*8 + shuf_idx*16
         * All 3 lossy levels share one NN action (binary lossless/lossy). */
        int nn_idx = algo_idx + (quant_idx > 0 ? 1 : 0) * 8 + shuf_idx * 16;
        float pred_cost = 0.0f, pred_ratio = 0.0f, pred_ct = 0.0f, pred_dt = 0.0f;
        float pred_psnr = 0.0f, pred_ssim = 0.0f, pred_max_error = 0.0f;
        if (per_config) {
            const NNDebugPerConfig& pc = per_config[nn_idx];
            pred_ratio     = pc.ratio;
            pred_ct        = pc.comp_time;
            pred_dt        = pc.decomp_time;
            pred_cost      = pc.cost;
            pred_psnr      = pc.psnr;
            pred_max_error = pc.max_error;
            pred_ssim      = pc.ssim;
        }

        gpucompress::DiagnosticsStore::instance().writeTraceRow(
            chunk_id, action, comp_lib,
            (action == nn_action),
            chunk_bytes,
            pred_cost, pred_ratio, pred_ct, pred_dt, pred_psnr, pred_ssim, pred_max_error,
            real_cost, real_ratio, comp_ms, decomp_ms, real_psnr, real_ssim, real_max_error,
            g_reinforce_mape_threshold,
            static_cast<float>(g_exploration_threshold),
            bw, w0, w1, w2);

        if (out_results) {
            int ridx = algo_idx * 8 + shuf_idx * 4 + quant_idx;
            TraceConfigResult& r  = out_results[ridx];
            r.action_id           = action;
            r.nn_idx              = nn_idx;
            r.real_ratio          = real_ratio;
            r.real_comp_ms        = comp_ms;
            r.real_decomp_ms      = decomp_ms;
            r.real_psnr           = real_psnr;
            r.real_cost           = real_cost;
            r.pred_cost           = pred_cost;
            r.valid               = true;
        }

        } /* quant_idx */
      } /* shuf_idx */
    } /* algo_idx */

    cudaStreamDestroy(stream);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_comp);
    cudaFree(d_decomp);
    cudaFree(d_quality);
}

/* ============================================================
 * Bypass Mode: silent D→H write / H→D read
 *
 * Like gpu_fallback_dh_write / gpu_fallback_hd_read but without
 * the "no gpucompress filter" warning — in Bypass mode the filter
 * is intentionally skipped by VOL policy, not missing from the DCPL.
 * ============================================================ */
static herr_t
gpu_bypass_dh_write(H5VL_gpucompress_t *o,
                    hid_t mem_type_id, hid_t mem_space_id,
                    hid_t file_space_id, hid_t dxpl_id,
                    const void *d_buf, void **req)
{
    hid_t actual_space_id = H5I_INVALID_HID;
    int   need_close = 0;
    if (file_space_id == H5S_ALL) {
        H5VL_dataset_get_args_t ga{};
        ga.op_type                 = H5VL_DATASET_GET_SPACE;
        ga.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id, &ga, dxpl_id, NULL) < 0)
            return -1;
        actual_space_id = ga.args.get_space.space_id;
        need_close = 1;
    } else {
        actual_space_id = file_space_id;
    }

    hsize_t n_elems = (hsize_t)H5Sget_simple_extent_npoints(actual_space_id);
    size_t  elem_sz = H5Tget_size(mem_type_id);
    if (need_close) H5Sclose(actual_space_id);

    size_t total_bytes = (size_t)n_elems * elem_sz;

    double t0 = _now_ms();  /* time the full bypass: D→H memcpy + disk write */

    void *h_tmp = nullptr;
    if (cudaMallocHost(&h_tmp, total_bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(h_tmp, d_buf, total_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFreeHost(h_tmp); return -1;
    }

    void       *under_obj = o->under_object;
    const void *h_ptr     = h_tmp;
    herr_t ret = H5VLdataset_write(1, &under_obj, o->under_vol_id,
                                   &mem_type_id, &mem_space_id, &file_space_id,
                                   dxpl_id, &h_ptr, req);
    cudaFreeHost(h_tmp);

    double elapsed = _now_ms() - t0;
    gpucompress::DiagnosticsStore::instance().accumulateIoMs(elapsed);
    g_life_writes++;
    std::call_once(g_atexit_flag, []{
        const char *dir = getenv("GPUCOMPRESS_RESULTS_DIR");
        if (!dir) dir = getenv("VPIC_RESULTS_DIR");
        if (dir) snprintf(g_life_output_dir, sizeof(g_life_output_dir), "%s", dir);
        std::atexit(gpucompress_vol_atexit);
    });

    return ret;
}

static herr_t
gpu_bypass_hd_read(H5VL_gpucompress_t *o,
                   hid_t mem_type_id, hid_t mem_space_id,
                   hid_t file_space_id, hid_t dxpl_id,
                   void *d_buf, void **req)
{
    hid_t actual_space_id = H5I_INVALID_HID;
    int   need_close = 0;
    if (file_space_id == H5S_ALL) {
        H5VL_dataset_get_args_t ga{};
        ga.op_type                 = H5VL_DATASET_GET_SPACE;
        ga.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id, &ga, dxpl_id, NULL) < 0)
            return -1;
        actual_space_id = ga.args.get_space.space_id;
        need_close = 1;
    } else {
        actual_space_id = file_space_id;
    }

    hsize_t n_elems = (hsize_t)H5Sget_simple_extent_npoints(actual_space_id);
    size_t  elem_sz = H5Tget_size(mem_type_id);
    if (need_close) H5Sclose(actual_space_id);

    size_t total_bytes = (size_t)n_elems * elem_sz;
    void *h_tmp = nullptr;
    if (cudaMallocHost(&h_tmp, total_bytes) != cudaSuccess) return -1;

    void  *under_obj = o->under_object;
    herr_t ret = H5VLdataset_read(1, &under_obj, o->under_vol_id,
                                  &mem_type_id, &mem_space_id, &file_space_id,
                                  dxpl_id, &h_tmp, req);
    if (ret >= 0)
        cudaMemcpy(d_buf, h_tmp, total_bytes, cudaMemcpyHostToDevice);
    cudaFreeHost(h_tmp);
    return ret;
}

/* ============================================================
 * gpu_fallback_dh_write
 *
 * Called when a GPU (device) pointer is passed to a dataset that has NO
 * gpucompress filter in its DCPL.  Instead of failing, we:
 *   1. Print a one-time warning so the caller can see what happened.
 *   2. Allocate a pinned host buffer (falls back to regular malloc on OOM).
 *   3. cudaMemcpy D→H.
 *   4. Write the entire dataset via the underlying native VOL (one-shot,
 *      no chunk-by-chunk loop — native HDF5 handles the chunked layout).
 * ============================================================ */
static herr_t
gpu_fallback_dh_write(H5VL_gpucompress_t *o,
                      hid_t mem_type_id, hid_t mem_space_id,
                      hid_t file_space_id, hid_t dxpl_id,
                      const void *d_buf, void **req)
{
    /* Resolve H5S_ALL → actual dataset space to compute total_bytes */
    hid_t actual_space_id = H5I_INVALID_HID;
    int   need_close = 0;
    if (file_space_id == H5S_ALL) {
        H5VL_dataset_get_args_t ga;
        memset(&ga, 0, sizeof(ga));
        ga.op_type                 = H5VL_DATASET_GET_SPACE;
        ga.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id,
                            &ga, dxpl_id, NULL) < 0) {
            fprintf(stderr, "gpucompress VOL fallback-write: failed to resolve "
                            "dataset space (H5S_ALL)\n");
            return -1;
        }
        actual_space_id = ga.args.get_space.space_id;
        need_close = 1;
    } else {
        actual_space_id = file_space_id;
    }

    hsize_t n_elems  = (hsize_t)H5Sget_simple_extent_npoints(actual_space_id);
    size_t  elem_sz  = H5Tget_size(mem_type_id);
    if (need_close) H5Sclose(actual_space_id);

    assert(n_elems  > 0 && "VOL fallback-write: dataset reports zero elements");
    assert(elem_sz  > 0 && "VOL fallback-write: H5Tget_size returned zero");

    size_t total_bytes = (size_t)n_elems * elem_sz;

    fprintf(stderr, "gpucompress VOL: WARNING — GPU buffer passed to dataset "
                    "without gpucompress filter; performing D->H copy "
                    "(%zu MiB) and writing via native path (no compression).\n",
                    total_bytes >> 20);

    /* Allocate host staging buffer — pinned memory required */
    void *h_tmp  = NULL;
    if (cudaMallocHost(&h_tmp, total_bytes) != cudaSuccess) {
        fprintf(stderr, "gpucompress VOL: pinned host alloc failed (%zu bytes)\n", total_bytes);
        return -1;
    }

    cudaError_t ce = cudaMemcpy(h_tmp, d_buf, total_bytes, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        fprintf(stderr, "gpucompress VOL: D→H copy failed: %s\n", cudaGetErrorString(ce));
        cudaFreeHost(h_tmp);
        return -1;
    }

    /* One-shot native write — HDF5 handles the chunked layout internally */
    void       *under_obj = o->under_object;
    const void *h_ptr     = h_tmp;
    herr_t ret = H5VLdataset_write(1, &under_obj, o->under_vol_id,
                                   &mem_type_id, &mem_space_id, &file_space_id,
                                   dxpl_id, &h_ptr, req);
    if (ret < 0)
        fprintf(stderr, "gpucompress VOL: H5VLdataset_write failed in fallback path\n");

    cudaFreeHost(h_tmp);
    return ret;
}

/* ============================================================
 * gpu_fallback_hd_read
 *
 * Symmetric counterpart to gpu_fallback_dh_write for the read path.
 * Called when a GPU buffer is passed to H5Dread on a dataset with no
 * gpucompress filter.  Reads into a host buffer via native VOL, then
 * H→D copies the result into the caller's GPU buffer.
 * ============================================================ */
static herr_t
gpu_fallback_hd_read(H5VL_gpucompress_t *o,
                     hid_t mem_type_id, hid_t mem_space_id,
                     hid_t file_space_id, hid_t dxpl_id,
                     void *d_buf, void **req)
{
    /* Resolve H5S_ALL → actual dataset space */
    hid_t actual_space_id = H5I_INVALID_HID;
    int   need_close = 0;
    if (file_space_id == H5S_ALL) {
        H5VL_dataset_get_args_t ga;
        memset(&ga, 0, sizeof(ga));
        ga.op_type                 = H5VL_DATASET_GET_SPACE;
        ga.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id,
                            &ga, dxpl_id, NULL) < 0) {
            fprintf(stderr, "gpucompress VOL fallback-read: failed to resolve "
                            "dataset space (H5S_ALL)\n");
            return -1;
        }
        actual_space_id = ga.args.get_space.space_id;
        need_close = 1;
    } else {
        actual_space_id = file_space_id;
    }

    hsize_t n_elems  = (hsize_t)H5Sget_simple_extent_npoints(actual_space_id);
    size_t  elem_sz  = H5Tget_size(mem_type_id);
    if (need_close) H5Sclose(actual_space_id);

    assert(n_elems  > 0 && "VOL fallback-read: dataset reports zero elements");
    assert(elem_sz  > 0 && "VOL fallback-read: H5Tget_size returned zero");

    size_t total_bytes = (size_t)n_elems * elem_sz;

    fprintf(stderr, "gpucompress VOL: WARNING — GPU buffer passed to dataset "
                    "without gpucompress filter; reading via native path then "
                    "H->D copy (%zu MiB).\n",
                    total_bytes >> 20);

    /* Allocate host staging buffer — pinned memory required */
    void *h_tmp  = NULL;
    if (cudaMallocHost(&h_tmp, total_bytes) != cudaSuccess) {
        fprintf(stderr, "gpucompress VOL: pinned host alloc failed (%zu bytes)\n", total_bytes);
        return -1;
    }

    /* One-shot native read */
    void  *under_obj = o->under_object;
    herr_t ret = H5VLdataset_read(1, &under_obj, o->under_vol_id,
                                  &mem_type_id, &mem_space_id, &file_space_id,
                                  dxpl_id, &h_tmp, req);
    if (ret < 0) {
        fprintf(stderr, "gpucompress VOL: H5VLdataset_read failed in fallback path\n");
        cudaFreeHost(h_tmp);
        return ret;
    }

    cudaError_t ce = cudaMemcpy(d_buf, h_tmp, total_bytes, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        fprintf(stderr, "gpucompress VOL: H→D copy failed: %s\n", cudaGetErrorString(ce));
        cudaFreeHost(h_tmp);
        return -1;
    }

    cudaFreeHost(h_tmp);
    return ret;
}

/* ============================================================
 * gpu_aware_chunked_read
 * ============================================================ */

static herr_t
gpu_aware_chunked_read(H5VL_gpucompress_t *o,
                       hid_t               mem_type_id,
                       hid_t               file_space_id,
                       hid_t               dxpl_id,
                       void               *d_buf)
{
    herr_t ret = -1;
    hid_t  actual_space_id = H5I_INVALID_HID;
    int    need_close_space = 0;

    if (file_space_id == H5S_ALL) {
        VOL_TRACE("  → H5VLdataset_get(native, GET_SPACE)  [H5S_ALL → resolve actual dims]");
        H5VL_dataset_get_args_t get_args;
        memset(&get_args, 0, sizeof(get_args));
        get_args.op_type                 = H5VL_DATASET_GET_SPACE;
        get_args.args.get_space.space_id = H5I_INVALID_HID;
        if (H5VLdataset_get(o->under_object, o->under_vol_id,
                            &get_args, dxpl_id, NULL) < 0) goto done;
        actual_space_id  = get_args.args.get_space.space_id;
        need_close_space = 1;
    } else {
        actual_space_id = file_space_id;
    }

    {
        int ndims = H5Sget_simple_extent_ndims(actual_space_id);
        if (ndims <= 0 || ndims > 32) goto done;

        hsize_t dset_dims[32];
        H5Sget_simple_extent_dims(actual_space_id, dset_dims, NULL);

        hsize_t chunk_dims[32] = {0};
        if (H5Pget_chunk(o->dcpl_id, ndims, chunk_dims) < 0) goto done;

        size_t elem_size = H5Tget_size(mem_type_id);
        if (elem_size == 0) goto done;

        size_t chunk_elems = 1;
        for (int d = 0; d < ndims; d++) chunk_elems *= (size_t)chunk_dims[d];
        size_t chunk_bytes = chunk_elems * elem_size;
        size_t max_comp    = gpucompress_max_compressed_size(chunk_bytes);

        /* ---- Prefetch pipeline: disk reader thread + GPU decompress main thread ---- */
/* I1 fix: increased from 2 to 4 prefetch slots to overlap I/O and decompress */
#define N_SLOTS_R 4
        void    *h_comp_r[N_SLOTS_R] = {};   /* pinned host staging for compressed data */
        uint8_t *d_compressed        = NULL;
        uint8_t *d_decompressed      = NULL; /* lazy: only if non-contiguous chunk seen */
        hsize_t *d_dset_dims = NULL, *d_chunk_dims = NULL, *d_chunk_start = NULL;
        cudaStream_t scatter_stream = nullptr;
        if (cudaStreamCreate(&scatter_stream) != cudaSuccess) {
            scatter_stream = nullptr;
            goto done;
        }

        /* CUDA events for precise decomp timing (matches comp-side event timing) */
        cudaEvent_t decomp_ev_start = nullptr, decomp_ev_stop = nullptr;
        cudaEventCreate(&decomp_ev_start);
        cudaEventCreate(&decomp_ev_stop);

        hsize_t num_chunks[32];
        for (int d = 0; d < ndims; d++)
            num_chunks[d] = (dset_dims[d] + chunk_dims[d] - 1) / chunk_dims[d];
        size_t total_chunks = 1;
        for (int d = 0; d < ndims; d++) total_chunks *= (size_t)num_chunks[d];

        /* Semaphore (mutex+condvar): tracks free prefetch slots */
        int                     free_slots_count = N_SLOTS_R;
        std::mutex              free_mtx;
        std::condition_variable free_cv;

        /* Queue from prefetch thread to main thread */
        struct PrefetchItem { int slot; size_t comp_sz; uint32_t filter_mask;
                              hsize_t cs[32]; herr_t status; };
        std::queue<PrefetchItem> ready_q;
        std::mutex               ready_mtx;
        std::condition_variable  ready_cv;

        bool        pre_started = false;
        std::thread pre_thr;

        /* I6 fix: hoist contiguity check before resource alloc (avoids goto issue) */
        bool contiguous_r = true;
        for (int d = 1; d < ndims; d++) {
            if (chunk_dims[d] != dset_dims[d]) { contiguous_r = false; break; }
        }

        /* ---- Allocate resources ---- */
        bool ok = true;
        for (int s = 0; s < N_SLOTS_R && ok; s++) {
            if (cudaMallocHost(&h_comp_r[s], max_comp) != cudaSuccess) ok = false;
        }
        if (ok && cudaMalloc(&d_compressed, max_comp) != cudaSuccess) ok = false;
        if (!ok) goto done_read;

        /* ---- Launch prefetch (disk reader) thread ---- */
        pre_thr = std::thread([&]() {
            for (size_t ci = 0; ci < total_chunks; ci++) {
                /* Acquire a free slot */
                { std::unique_lock<std::mutex> lk(free_mtx);
                  free_cv.wait(lk, [&]{ return free_slots_count > 0; });
                  free_slots_count--; }
                int slot = (int)(ci % N_SLOTS_R);

                /* Compute chunk_start for this ci */
                hsize_t chunk_idx[32];
                size_t  remaining = ci;
                for (int d = ndims - 1; d >= 0; d--) {
                    chunk_idx[d] = (hsize_t)(remaining % (size_t)num_chunks[d]);
                    remaining   /= (size_t)num_chunks[d];
                }
                hsize_t chunk_start[32];
                for (int d = 0; d < ndims; d++)
                    chunk_start[d] = chunk_idx[d] * chunk_dims[d];

                size_t   comp_sz     = max_comp;
                uint32_t filter_mask = 0;
                herr_t r = read_chunk_from_native(o->under_object, o->under_vol_id,
                                                   chunk_start, h_comp_r[slot],
                                                   &comp_sz, &filter_mask, dxpl_id);
                PrefetchItem item;
                item.slot        = slot;
                item.comp_sz     = comp_sz;
                item.filter_mask = filter_mask;
                item.status      = r;
                memcpy(item.cs, chunk_start, (size_t)ndims * sizeof(hsize_t));
                { std::lock_guard<std::mutex> lk(ready_mtx); ready_q.push(item); }
                ready_cv.notify_one();
            }
        });
        pre_started = true;

        ret = 0;
        s_gpu_reads++;

        for (size_t ci = 0; ci < total_chunks; ci++) {
            /* Wait for the prefetch thread to deliver the next chunk */
            PrefetchItem item;
            { std::unique_lock<std::mutex> lk(ready_mtx);
              ready_cv.wait(lk, [&]{ return !ready_q.empty(); });
              item = ready_q.front(); ready_q.pop(); }

            if (item.status < 0) { ret = -1; break; }

            VOL_TRACE("  chunk[%zu] off=%llu → read %zuB compressed from file",
                      ci, (unsigned long long)item.cs[0], item.comp_sz);

            /* Recompute actual_chunk / actual_bytes from chunk_start */
            hsize_t *chunk_start = item.cs;
            hsize_t  actual_chunk[32];
            size_t   actual_elems = 1;
            for (int d = 0; d < ndims; d++) {
                actual_chunk[d] = chunk_dims[d];
                if (chunk_start[d] + actual_chunk[d] > dset_dims[d])
                    actual_chunk[d] = dset_dims[d] - chunk_start[d];
                actual_elems *= (size_t)actual_chunk[d];
            }
            size_t actual_bytes = actual_elems * elem_size;

            bool    direct_decomp = contiguous_r && (actual_bytes == chunk_bytes);
            uint8_t *dst_ptr;

            if (direct_decomp) {
                size_t off = 0, s = elem_size;
                for (int d = ndims - 1; d >= 0; d--) {
                    off += (size_t)chunk_start[d] * s;
                    s   *= (size_t)dset_dims[d];
                }
                dst_ptr = static_cast<uint8_t*>(d_buf) + off;
            } else {
                if (!d_decompressed &&
                    cudaMalloc(&d_decompressed, chunk_bytes) != cudaSuccess) {
                    ret = -1; break;
                }
                dst_ptr = d_decompressed;
            }

            /* H→D compressed bytes (synchronous — fast for compressed data) */
            VOL_TRACE("    cudaMemcpy H→D %zuB (compressed)", item.comp_sz);
            if (vol_memcpy(d_compressed, h_comp_r[item.slot], item.comp_sz,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                ret = -1; break;
            }

            /* Release this prefetch slot now that H→D is done */
            { std::lock_guard<std::mutex> lk(free_mtx); free_slots_count++; }
            free_cv.notify_one();

            /* Decompress on GPU — measure with CUDA events for precision
             * consistent with compression-side timing. */
            size_t decomp_size = chunk_bytes;
            cudaEventRecord(decomp_ev_start, scatter_stream);
            gpucompress_error_t ce = gpucompress_decompress_gpu(
                d_compressed, item.comp_sz, dst_ptr, &decomp_size,
                scatter_stream);
            cudaEventRecord(decomp_ev_stop, scatter_stream);
            cudaEventSynchronize(decomp_ev_stop);
            if (ce != GPUCOMPRESS_SUCCESS) { ret = -1; break; }
            float _decomp_ms = 0.0f;
            cudaEventElapsedTime(&_decomp_ms, decomp_ev_start, decomp_ev_stop);
            gpucompress::DiagnosticsStore::instance().recordDecompMs((int)ci, _decomp_ms);
            VOL_TRACE("    gpucompress_decompress_gpu() → %zuB (%.2f ms)", decomp_size, _decomp_ms);

            if (direct_decomp) {
                s_chunks_decomp++;
            } else if (contiguous_r) {
                size_t off = 0, s = elem_size;
                for (int d = ndims - 1; d >= 0; d--) {
                    off += (size_t)chunk_start[d] * s;
                    s   *= (size_t)dset_dims[d];
                }
                if (vol_memcpy(static_cast<uint8_t*>(d_buf) + off,
                               d_decompressed, actual_bytes,
                               cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    ret = -1; break;
                }
                s_chunks_decomp++;
            } else {
                if (!d_dset_dims) {
                    if (alloc_dim_arrays(ndims, dset_dims, actual_chunk, chunk_start,
                                        &d_dset_dims, &d_chunk_dims, &d_chunk_start) < 0) {
                        ret = -1; break;
                    }
                } else {
                    vol_memcpy(d_chunk_dims,  actual_chunk, (size_t)ndims * sizeof(hsize_t),
                               cudaMemcpyHostToDevice);
                    vol_memcpy(d_chunk_start, chunk_start,  (size_t)ndims * sizeof(hsize_t),
                               cudaMemcpyHostToDevice);
                }
                int threads = 256;
                int blocks  = (int)((actual_elems + threads - 1) / threads);
                scatter_chunk_kernel<<<blocks, threads, 0, scatter_stream>>>(
                    d_decompressed, static_cast<uint8_t*>(d_buf),
                    ndims, elem_size, actual_elems,
                    d_dset_dims, d_chunk_dims, d_chunk_start);
                if (cudaGetLastError() != cudaSuccess ||
                    cudaStreamSynchronize(scatter_stream) != cudaSuccess) {
                    ret = -1; break;
                }
                s_chunks_decomp++;
            }
        } /* chunk loop */

        /* Batched deferred decomp SGD: one update from all chunks' decomp times. */
        if (ret == 0) gpucompress_batched_decomp_sgd();

done_read:
        /* Unblock prefetch thread (in case main exited early with ret==-1) */
        if (pre_started) {
            { std::lock_guard<std::mutex> lk(free_mtx);
              free_slots_count = N_SLOTS_R; }
            free_cv.notify_all();
            pre_thr.join();
        }

        if (decomp_ev_start) cudaEventDestroy(decomp_ev_start);
        if (decomp_ev_stop)  cudaEventDestroy(decomp_ev_stop);
        if (scatter_stream) cudaStreamDestroy(scatter_stream);
        if (d_dset_dims) free_dim_arrays(d_dset_dims, d_chunk_dims, d_chunk_start);
        cudaFree(d_decompressed);
        cudaFree(d_compressed);
        for (int s = 0; s < N_SLOTS_R; s++) {
            if (h_comp_r[s]) cudaFreeHost(h_comp_r[s]);
        }
#undef N_SLOTS_R
    }

done:
    if (need_close_space) H5Sclose(actual_space_id);
    return ret;
}

/* ============================================================
 * Dataset callbacks
 * ============================================================ */

static void *
H5VL_gpucompress_dataset_create(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id,
    hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    VOL_TRACE("dataset_create(\"%s\")", name ? name : "?");
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;

    /* Bypass mode: strip the gpucompress filter so native HDF5 stores raw data.
     * Without this, the native connector would invoke the CPU filter stub. */
    hid_t dcpl_use = dcpl_id;
    bool  dcpl_copied = false;
    if (s_vol_mode == VOLMode::BYPASS && dcpl_id != H5P_DEFAULT) {
        int nfilters = H5Pget_nfilters(dcpl_id);
        for (int fi = 0; fi < nfilters; fi++) {
            unsigned int flags, cd[32]; size_t cd_n = 32; char name_buf[64]; unsigned int ff;
            H5Z_filter_t fid = H5Pget_filter2(dcpl_id, (unsigned)fi, &flags, &cd_n,
                                               cd, sizeof(name_buf), name_buf, &ff);
            if (fid == H5Z_FILTER_GPUCOMPRESS) {
                if (!dcpl_copied) { dcpl_use = H5Pcopy(dcpl_id); dcpl_copied = true; }
                H5Premove_filter(dcpl_use, H5Z_FILTER_GPUCOMPRESS);
                break;
            }
        }
    }

    VOL_TRACE("  → H5VLdataset_create(native, \"%s\")", name ? name : "?");
    void *under = H5VLdataset_create(o->under_object, lp, o->under_vol_id,
                                     name, lcpl_id, type_id, space_id,
                                     dcpl_use, dapl_id, dxpl_id, req);
    if (dcpl_copied) H5Pclose(dcpl_use);
    if (!under) return NULL;

    H5VL_gpucompress_t *dset = new_obj(under, o->under_vol_id);
    /* Store a copy of the DCPL so we can access chunk dims later */
    dset->dcpl_id = (dcpl_id != H5P_DEFAULT) ? H5Pcopy(dcpl_id) : H5I_INVALID_HID;

    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return dset;
}

static void *
H5VL_gpucompress_dataset_open(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    VOL_TRACE("dataset_open(\"%s\")", name ? name : "?");
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    VOL_TRACE("  → H5VLdataset_open(native, \"%s\")", name ? name : "?");
    void *under = H5VLdataset_open(o->under_object, lp, o->under_vol_id,
                                   name, dapl_id, dxpl_id, req);
    if (!under) return NULL;

    H5VL_gpucompress_t *dset = new_obj(under, o->under_vol_id);

    /* Retrieve the DCPL from the opened dataset so we know the chunk dims */
    H5VL_dataset_get_args_t get_args;
    memset(&get_args, 0, sizeof(get_args));
    get_args.op_type                = H5VL_DATASET_GET_DCPL;
    get_args.args.get_dcpl.dcpl_id  = H5I_INVALID_HID;
    VOL_TRACE("  → H5VLdataset_get(native, GET_DCPL)  [retrieve chunk/filter config]");
    if (H5VLdataset_get(under, o->under_vol_id, &get_args, dxpl_id, NULL) >= 0)
        dset->dcpl_id = get_args.args.get_dcpl.dcpl_id;

    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return dset;
}

static herr_t
H5VL_gpucompress_dataset_read(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
    hid_t plist_id, void *buf[], void **req)
{
    double _t0 = _now_ms();
    herr_t ret;

    /* For each dataset, check if buf is a GPU pointer */
    for (size_t i = 0; i < count; i++) {
        H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset[i];

        if (o->dcpl_id != H5I_INVALID_HID && buf[i] &&
            gpucompress_is_device_ptr(buf[i])) {

            gpucompress_config_t cfg;
            if (s_vol_mode == VOLMode::BYPASS) {
                VOL_TRACE("dataset_read[%zu]: buf=%p → bypass (native + H→D)", i, buf[i]);
                ret = gpu_bypass_hd_read(o, mem_type_id[i], mem_space_id[i],
                                         file_space_id[i], plist_id,
                                         buf[i], req);
            } else if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) == 0) {
                VOL_TRACE("dataset_read[%zu]: buf=%p → GPU path (gpucompress filter)", i, buf[i]);
                /* RELEASE / TRACE: decompress chunk by chunk */
                ret = gpu_aware_chunked_read(o, mem_type_id[i],
                                             file_space_id[i], plist_id,
                                             buf[i]);
            } else {
                VOL_TRACE("dataset_read[%zu]: buf=%p → fallback native+H→D (no gpucompress filter)", i, buf[i]);
                ret = gpu_fallback_hd_read(o, mem_type_id[i], mem_space_id[i],
                                           file_space_id[i], plist_id,
                                           buf[i], req);
            }
            if (ret < 0) return ret;
        } else {
            /* Host pointer rejected — this VOL requires device pointers */
            fprintf(stderr,
                    "gpucompress VOL FATAL: dataset_read[%zu] received a host "
                    "pointer (buf=%p). The gpucompress VOL connector only "
                    "accepts CUDA device pointers.\n", i, buf[i]);
            abort();
        }
    }

    gpucompress::DiagnosticsStore::instance().accumulateIoMs(_now_ms() - _t0);
    if (req && *req && count > 0)
        *req = new_obj(*req, ((H5VL_gpucompress_t*)dset[0])->under_vol_id);
    return 0;
}

static herr_t
H5VL_gpucompress_dataset_write(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
    hid_t plist_id, const void *buf[], void **req)
{
    double _t0 = _now_ms();
    herr_t ret;

    for (size_t i = 0; i < count; i++) {
        H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset[i];

        if (o->dcpl_id != H5I_INVALID_HID && buf[i] &&
            gpucompress_is_device_ptr(buf[i])) {

            gpucompress_config_t cfg;
            if (s_vol_mode == VOLMode::BYPASS) {
                VOL_TRACE("dataset_write[%zu]: buf=%p → bypass (D→H + native)", i, buf[i]);
                ret = gpu_bypass_dh_write(o, mem_type_id[i], mem_space_id[i],
                                          file_space_id[i], plist_id,
                                          buf[i], req);
            } else if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) == 0) {
                if (s_vol_mode == VOLMode::TRACE) {
                    VOL_TRACE("dataset_write[%zu]: buf=%p → trace path (exhaustive)", i, buf[i]);
                    ret = gpu_trace_chunked_write(o, mem_type_id[i],
                                                  file_space_id[i], plist_id,
                                                  buf[i]);
                } else {
                    VOL_TRACE("dataset_write[%zu]: buf=%p → release path (gpucompress filter)", i, buf[i]);
                    ret = gpu_aware_chunked_write(o, mem_type_id[i],
                                                  file_space_id[i], plist_id,
                                                  buf[i]);
                }
            } else {
                VOL_TRACE("dataset_write[%zu]: buf=%p → fallback D→H (no gpucompress filter)", i, buf[i]);
                ret = gpu_fallback_dh_write(o, mem_type_id[i], mem_space_id[i],
                                            file_space_id[i], plist_id,
                                            buf[i], req);
            }
            if (ret < 0) return ret;
        } else {
            /* Host pointer rejected — this VOL requires device pointers */
            fprintf(stderr,
                    "gpucompress VOL FATAL: dataset_write[%zu] received a host "
                    "pointer (buf=%p). The gpucompress VOL connector only "
                    "accepts CUDA device pointers.\n", i, buf[i]);
            abort();
        }
    }

    gpucompress::DiagnosticsStore::instance().accumulateIoMs(_now_ms() - _t0);
    if (req && *req && count > 0)
        *req = new_obj(*req, ((H5VL_gpucompress_t*)dset[0])->under_vol_id);
    return 0;
}

static herr_t
H5VL_gpucompress_dataset_get(void *dset, H5VL_dataset_get_args_t *args,
                              hid_t dxpl_id, void **req)
{
    VOL_TRACE("dataset_get(op=%s)", dset_get_name(args->op_type));
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset;
    herr_t rv = H5VLdataset_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args,
                                   hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o          = (H5VL_gpucompress_t*)obj;
    hid_t               under_vol  = o->under_vol_id;
    herr_t rv = H5VLdataset_specific(o->under_object, under_vol, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_dataset_optional(void *obj, H5VL_optional_args_t *args,
                                   hid_t dxpl_id, void **req)
{
    VOL_TRACE("dataset_optional(op=%s)", dset_opt_name(args->op_type));
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLdataset_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_dataset_close(void *dset, hid_t dxpl_id, void **req)
{
    VOL_TRACE("dataset_close");
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset;
    VOL_TRACE("  → H5VLdataset_close(native)");
    herr_t rv = H5VLdataset_close(o->under_object, o->under_vol_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    if (rv >= 0) {
        /* P0 fix: donate write buffers to global cache instead of freeing.
         * Next H5Dwrite reclaims them, avoiding ~750ms of cudaMalloc/cudaMallocHost. */
        if (o->write_ctx) {
            VolWriteCtx* wctx = o->write_ctx;
            if (wctx->initialized) {
                std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
                if (!s_buf_cache.valid || wctx->max_comp > s_buf_cache.max_comp) {
                    /* Evict smaller cached buffers if present */
                    if (s_buf_cache.valid) {
                        for (int w = 0; w < M4_N_COMP_WORKERS; w++)
                            if (s_buf_cache.d_comp_w[w]) cudaFree(s_buf_cache.d_comp_w[w]);
                        for (int i = 0; i < M4_N_IO_BUFS; i++)
                            if (s_buf_cache.io_pool_bufs[i]) cudaFreeHost(s_buf_cache.io_pool_bufs[i]);
                    }
                    /* Donate to cache — no cudaFree, pointers transferred */
                    memcpy(s_buf_cache.d_comp_w, wctx->d_comp_w, sizeof(wctx->d_comp_w));
                    memcpy(s_buf_cache.io_pool_bufs, wctx->io_pool_bufs, sizeof(wctx->io_pool_bufs));
                    s_buf_cache.max_comp = wctx->max_comp;
                    s_buf_cache.valid = true;
                } else {
                    /* Cache already has equal/larger buffers — free these normally */
                    for (int w = 0; w < M4_N_COMP_WORKERS; w++)
                        if (wctx->d_comp_w[w]) cudaFree(wctx->d_comp_w[w]);
                    for (int i = 0; i < M4_N_IO_BUFS; i++)
                        if (wctx->io_pool_bufs[i]) cudaFreeHost(wctx->io_pool_bufs[i]);
                }
            }
            delete wctx;
            o->write_ctx = nullptr;
        }
        if (o->dcpl_id != H5I_INVALID_HID) { H5Pclose(o->dcpl_id); o->dcpl_id = H5I_INVALID_HID; }
        free_obj(o);
    }
    return rv;
}

/* ============================================================
 * Datatype callbacks (pure pass-through)
 * ============================================================ */

static void *
H5VL_gpucompress_datatype_commit(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id,
    hid_t tapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLdatatype_commit(o->under_object, lp, o->under_vol_id,
                                      name, type_id, lcpl_id, tcpl_id,
                                      tapl_id, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *dt = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return dt;
}

static void *
H5VL_gpucompress_datatype_open(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t tapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLdatatype_open(o->under_object, lp, o->under_vol_id,
                                    name, tapl_id, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *dt = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return dt;
}

static herr_t
H5VL_gpucompress_datatype_get(void *dt, H5VL_datatype_get_args_t *args,
                               hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dt;
    herr_t rv = H5VLdatatype_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args,
                                    hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o         = (H5VL_gpucompress_t*)obj;
    hid_t               under_vol = o->under_vol_id;
    herr_t rv = H5VLdatatype_specific(o->under_object, under_vol, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_datatype_optional(void *obj, H5VL_optional_args_t *args,
                                    hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLdatatype_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_datatype_close(void *dt, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dt;
    herr_t rv = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    if (rv >= 0) free_obj(o);
    return rv;
}

/* ============================================================
 * File callbacks (pure pass-through)
 * ============================================================ */

static void *
H5VL_gpucompress_file_create(const char *name, unsigned flags,
    hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req)
{
    VOL_TRACE("file_create(\"%s\", flags=0x%x)", name ? name : "?", flags);
    H5VL_gpucompress_info_t *info = NULL;
    H5Pget_vol_info(fapl_id, (void**)&info);
    if (!info) return NULL;

    hid_t under_fapl = H5Pcopy(fapl_id);
    if (under_fapl < 0) {
        H5VL_gpucompress_info_free(info);
        return NULL;
    }
    H5Pset_vol(under_fapl, info->under_vol_id, info->under_vol_info);
    VOL_TRACE("  → H5VLfile_create(native, \"%s\")", name ? name : "?");
    void *under = H5VLfile_create(name, flags, fcpl_id, under_fapl, dxpl_id, req);
    H5VL_gpucompress_t *file = under ? new_obj(under, info->under_vol_id) : NULL;
    if (req && *req) *req = new_obj(*req, info->under_vol_id);
    H5Pclose(under_fapl);
    H5VL_gpucompress_info_free(info);
    return file;
}

static void *
H5VL_gpucompress_file_open(const char *name, unsigned flags,
    hid_t fapl_id, hid_t dxpl_id, void **req)
{
    VOL_TRACE("file_open(\"%s\", flags=0x%x)", name ? name : "?", flags);
    H5VL_gpucompress_info_t *info = NULL;
    H5Pget_vol_info(fapl_id, (void**)&info);
    if (!info) return NULL;

    hid_t under_fapl = H5Pcopy(fapl_id);
    if (under_fapl < 0) {
        H5VL_gpucompress_info_free(info);
        return NULL;
    }
    H5Pset_vol(under_fapl, info->under_vol_id, info->under_vol_info);
    VOL_TRACE("  → H5VLfile_open(native, \"%s\")", name ? name : "?");
    void *under = H5VLfile_open(name, flags, under_fapl, dxpl_id, req);
    H5VL_gpucompress_t *file = under ? new_obj(under, info->under_vol_id) : NULL;
    if (req && *req) *req = new_obj(*req, info->under_vol_id);
    H5Pclose(under_fapl);
    H5VL_gpucompress_info_free(info);
    return file;
}

static herr_t
H5VL_gpucompress_file_get(void *file, H5VL_file_get_args_t *args,
                           hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)file;
    herr_t rv = H5VLfile_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_file_specific(void *file, H5VL_file_specific_args_t *args,
                                hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t       *o          = (H5VL_gpucompress_t*)file;
    H5VL_file_specific_args_t my_args;
    H5VL_file_specific_args_t *new_args  = args;
    H5VL_gpucompress_info_t  *info       = NULL;
    hid_t                     under_vol  = o ? o->under_vol_id : -1;
    void                     *under_obj  = o ? o->under_object : NULL;

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        memcpy(&my_args, args, sizeof(my_args));
        H5Pget_vol_info(args->args.is_accessible.fapl_id, (void**)&info);
        if (!info) return -1;
        under_vol = info->under_vol_id;
        my_args.args.is_accessible.fapl_id = H5Pcopy(args->args.is_accessible.fapl_id);
        if (my_args.args.is_accessible.fapl_id < 0) {
            H5VL_gpucompress_info_free(info);
            return -1;
        }
        H5Pset_vol(my_args.args.is_accessible.fapl_id, info->under_vol_id, info->under_vol_info);
        new_args  = &my_args;
        under_obj = NULL;
    } else if (args->op_type == H5VL_FILE_DELETE) {
        memcpy(&my_args, args, sizeof(my_args));
        H5Pget_vol_info(args->args.del.fapl_id, (void**)&info);
        if (!info) return -1;
        under_vol = info->under_vol_id;
        my_args.args.del.fapl_id = H5Pcopy(args->args.del.fapl_id);
        if (my_args.args.del.fapl_id < 0) {
            H5VL_gpucompress_info_free(info);
            return -1;
        }
        H5Pset_vol(my_args.args.del.fapl_id, info->under_vol_id, info->under_vol_info);
        new_args  = &my_args;
        under_obj = NULL;
    }

    herr_t rv = H5VLfile_specific(under_obj, under_vol, new_args, dxpl_id, req);

    if (req && *req) *req = new_obj(*req, under_vol);

    if (args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        H5Pclose(my_args.args.is_accessible.fapl_id);
        H5VL_gpucompress_info_free(info);
    } else if (args->op_type == H5VL_FILE_DELETE) {
        H5Pclose(my_args.args.del.fapl_id);
        H5VL_gpucompress_info_free(info);
    } else if (args->op_type == H5VL_FILE_REOPEN) {
        if (rv >= 0 && *args->args.reopen.file)
            *args->args.reopen.file = new_obj(*args->args.reopen.file, under_vol);
    }

    return rv;
}

static herr_t
H5VL_gpucompress_file_optional(void *file, H5VL_optional_args_t *args,
                                hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)file;
    herr_t rv = H5VLfile_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_file_close(void *file, hid_t dxpl_id, void **req)
{
    VOL_TRACE("file_close");
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)file;
    VOL_TRACE("  → H5VLfile_close(native)");
    herr_t rv = H5VLfile_close(o->under_object, o->under_vol_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    if (rv >= 0) free_obj(o);

    /* Trace: flush the CSV on each file close so data is not lost on crash */
    if (s_vol_mode == VOLMode::TRACE) {
        gpucompress::DiagnosticsStore::instance().flushTrace();
    }

    return rv;
}

/* ============================================================
 * Group callbacks (pure pass-through)
 * ============================================================ */

static void *
H5VL_gpucompress_group_create(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLgroup_create(o->under_object, lp, o->under_vol_id,
                                   name, lcpl_id, gcpl_id, gapl_id, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *grp = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return grp;
}

static void *
H5VL_gpucompress_group_open(void *obj, const H5VL_loc_params_t *lp,
    const char *name, hid_t gapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLgroup_open(o->under_object, lp, o->under_vol_id,
                                 name, gapl_id, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *grp = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return grp;
}

static herr_t
H5VL_gpucompress_group_get(void *obj, H5VL_group_get_args_t *args,
                            hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_group_specific(void *obj, H5VL_group_specific_args_t *args,
                                 hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o         = (H5VL_gpucompress_t*)obj;
    hid_t               under_vol = o->under_vol_id;

    if (args->op_type == H5VL_GROUP_MOUNT) {
        H5VL_group_specific_args_t vol_args;
        vol_args.op_type              = H5VL_GROUP_MOUNT;
        vol_args.args.mount.name      = args->args.mount.name;
        vol_args.args.mount.child_file =
            ((H5VL_gpucompress_t*)args->args.mount.child_file)->under_object;
        vol_args.args.mount.fmpl_id   = args->args.mount.fmpl_id;
        herr_t rv = H5VLgroup_specific(o->under_object, under_vol, &vol_args, dxpl_id, req);
        if (req && *req) *req = new_obj(*req, under_vol);
        return rv;
    }

    herr_t rv = H5VLgroup_specific(o->under_object, under_vol, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_group_optional(void *obj, H5VL_optional_args_t *args,
                                 hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLgroup_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_group_close(void *grp, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)grp;
    herr_t rv = H5VLgroup_close(o->under_object, o->under_vol_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    if (rv >= 0) free_obj(o);
    return rv;
}

/* ============================================================
 * Link callbacks (pure pass-through)
 * ============================================================ */

static herr_t
H5VL_gpucompress_link_create(H5VL_link_create_args_t *args, void *obj,
    const H5VL_loc_params_t *lp, hid_t lcpl_id, hid_t lapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    hid_t under_vol = o ? o->under_vol_id : -1;

    if (H5VL_LINK_CREATE_HARD == args->op_type && args->args.hard.curr_obj) {
        H5VL_gpucompress_t *co = (H5VL_gpucompress_t*)args->args.hard.curr_obj;
        if (under_vol < 0) under_vol = co->under_vol_id;
        args->args.hard.curr_obj = co->under_object;
    }

    herr_t rv = H5VLlink_create(args, o ? o->under_object : NULL, lp,
                                 under_vol, lcpl_id, lapl_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_link_copy(void *src_obj, const H5VL_loc_params_t *lp1,
    void *dst_obj, const H5VL_loc_params_t *lp2,
    hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *s = (H5VL_gpucompress_t*)src_obj;
    H5VL_gpucompress_t *d = (H5VL_gpucompress_t*)dst_obj;
    hid_t under_vol = s ? s->under_vol_id : d->under_vol_id;
    herr_t rv = H5VLlink_copy(s ? s->under_object : NULL, lp1,
                               d ? d->under_object : NULL, lp2,
                               under_vol, lcpl_id, lapl_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_link_move(void *src_obj, const H5VL_loc_params_t *lp1,
    void *dst_obj, const H5VL_loc_params_t *lp2,
    hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *s = (H5VL_gpucompress_t*)src_obj;
    H5VL_gpucompress_t *d = (H5VL_gpucompress_t*)dst_obj;
    hid_t under_vol = s ? s->under_vol_id : d->under_vol_id;
    herr_t rv = H5VLlink_move(s ? s->under_object : NULL, lp1,
                               d ? d->under_object : NULL, lp2,
                               under_vol, lcpl_id, lapl_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_link_get(void *obj, const H5VL_loc_params_t *lp,
    H5VL_link_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLlink_get(o->under_object, lp, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_link_specific(void *obj, const H5VL_loc_params_t *lp,
    H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLlink_specific(o->under_object, lp, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_link_optional(void *obj, const H5VL_loc_params_t *lp,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLlink_optional(o->under_object, lp, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

/* ============================================================
 * Object callbacks (pure pass-through)
 * ============================================================ */

static void *
H5VL_gpucompress_object_open(void *obj, const H5VL_loc_params_t *lp,
    H5I_type_t *opened_type, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    void *under = H5VLobject_open(o->under_object, lp, o->under_vol_id,
                                  opened_type, dxpl_id, req);
    if (!under) return NULL;
    H5VL_gpucompress_t *no = new_obj(under, o->under_vol_id);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return no;
}

static herr_t
H5VL_gpucompress_object_copy(void *src_obj, const H5VL_loc_params_t *slp,
    const char *src_name, void *dst_obj, const H5VL_loc_params_t *dlp,
    const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *s = (H5VL_gpucompress_t*)src_obj;
    H5VL_gpucompress_t *d = (H5VL_gpucompress_t*)dst_obj;
    herr_t rv = H5VLobject_copy(s->under_object, slp, src_name,
                                 d->under_object, dlp, dst_name,
                                 s->under_vol_id, ocpypl_id, lcpl_id, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, s->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_object_get(void *obj, const H5VL_loc_params_t *lp,
    H5VL_object_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLobject_get(o->under_object, lp, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

static herr_t
H5VL_gpucompress_object_specific(void *obj, const H5VL_loc_params_t *lp,
    H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o         = (H5VL_gpucompress_t*)obj;
    hid_t               under_vol = o->under_vol_id;
    herr_t rv = H5VLobject_specific(o->under_object, lp, under_vol, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, under_vol);
    return rv;
}

static herr_t
H5VL_gpucompress_object_optional(void *obj, const H5VL_loc_params_t *lp,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLobject_optional(o->under_object, lp, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

/* ============================================================
 * Introspect callbacks
 * ============================================================ */

static herr_t
H5VL_gpucompress_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
                                          const H5VL_class_t **conn_cls)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    if (H5VL_GET_CONN_LVL_CURR == lvl) {
        *conn_cls = &H5VL_gpucompress_g;
        return 0;
    }
    return H5VLintrospect_get_conn_cls(o->under_object, o->under_vol_id, lvl, conn_cls);
}

static herr_t
H5VL_gpucompress_introspect_get_cap_flags(const void *_info, uint64_t *cap_flags)
{
    const H5VL_gpucompress_info_t *info = (const H5VL_gpucompress_info_t*)_info;
    if (!info || H5Iis_valid(info->under_vol_id) <= 0) return -1;
    herr_t rv = H5VLintrospect_get_cap_flags(info->under_vol_info, info->under_vol_id, cap_flags);
    if (rv >= 0) *cap_flags |= H5VL_gpucompress_g.cap_flags;
    return rv;
}

static herr_t
H5VL_gpucompress_introspect_opt_query(void *obj, H5VL_subclass_t cls,
                                       int opt_type, uint64_t *flags)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLintrospect_opt_query(o->under_object, o->under_vol_id, cls, opt_type, flags);
}

/* ============================================================
 * Request callbacks
 * ============================================================ */

static herr_t
H5VL_gpucompress_request_wait(void *req, uint64_t timeout,
                               H5VL_request_status_t *status)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    herr_t rv = H5VLrequest_wait(o->under_object, o->under_vol_id, timeout, status);
    if (rv >= 0 && *status != H5VL_REQUEST_STATUS_IN_PROGRESS) free_obj(o);
    return rv;
}

static herr_t
H5VL_gpucompress_request_notify(void *req, H5VL_request_notify_t cb, void *ctx)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    herr_t rv = H5VLrequest_notify(o->under_object, o->under_vol_id, cb, ctx);
    if (rv >= 0) free_obj(o);
    return rv;
}

static herr_t
H5VL_gpucompress_request_cancel(void *req, H5VL_request_status_t *status)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    herr_t rv = H5VLrequest_cancel(o->under_object, o->under_vol_id, status);
    if (rv >= 0) free_obj(o);
    return rv;
}

static herr_t
H5VL_gpucompress_request_specific(void *req, H5VL_request_specific_args_t *args)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    return H5VLrequest_specific(o->under_object, o->under_vol_id, args);
}

static herr_t
H5VL_gpucompress_request_optional(void *req, H5VL_optional_args_t *args)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    return H5VLrequest_optional(o->under_object, o->under_vol_id, args);
}

static herr_t
H5VL_gpucompress_request_free(void *req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)req;
    herr_t rv = H5VLrequest_free(o->under_object, o->under_vol_id);
    if (rv >= 0) free_obj(o);
    return rv;
}

/* ============================================================
 * Blob callbacks
 * ============================================================ */

static herr_t
H5VL_gpucompress_blob_put(void *obj, const void *buf, size_t size,
                           void *blob_id, void *ctx)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLblob_put(o->under_object, o->under_vol_id, buf, size, blob_id, ctx);
}

static herr_t
H5VL_gpucompress_blob_get(void *obj, const void *blob_id, void *buf,
                           size_t size, void *ctx)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLblob_get(o->under_object, o->under_vol_id, blob_id, buf, size, ctx);
}

static herr_t
H5VL_gpucompress_blob_specific(void *obj, void *blob_id,
                                H5VL_blob_specific_args_t *args)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLblob_specific(o->under_object, o->under_vol_id, blob_id, args);
}

static herr_t
H5VL_gpucompress_blob_optional(void *obj, void *blob_id,
                                H5VL_optional_args_t *args)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLblob_optional(o->under_object, o->under_vol_id, blob_id, args);
}

/* ============================================================
 * Token callbacks
 * ============================================================ */

static herr_t
H5VL_gpucompress_token_cmp(void *obj, const H5O_token_t *t1,
                            const H5O_token_t *t2, int *cmp_value)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLtoken_cmp(o->under_object, o->under_vol_id, t1, t2, cmp_value);
}

static herr_t
H5VL_gpucompress_token_to_str(void *obj, H5I_type_t obj_type,
                               const H5O_token_t *token, char **token_str)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLtoken_to_str(o->under_object, obj_type, o->under_vol_id, token, token_str);
}

static herr_t
H5VL_gpucompress_token_from_str(void *obj, H5I_type_t obj_type,
                                 const char *token_str, H5O_token_t *token)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    return H5VLtoken_from_str(o->under_object, obj_type, o->under_vol_id, token_str, token);
}

/* ============================================================
 * Generic optional callback
 * ============================================================ */

static herr_t
H5VL_gpucompress_optional(void *obj, H5VL_optional_args_t *args,
                           hid_t dxpl_id, void **req)
{
    H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)obj;
    herr_t rv = H5VLoptional(o->under_object, o->under_vol_id, args, dxpl_id, req);
    if (req && *req) *req = new_obj(*req, o->under_vol_id);
    return rv;
}

/* ============================================================
 * Public registration API
 * ============================================================ */

extern "C" hid_t
H5VL_gpucompress_register(void)
{
    /* Capture program start wall-clock on first call — used by the
     * compute_ms derivation in the lifetime summary and the
     * H5VL_gpucompress_get_program_wall() getter. */
    vol_prog_start_mark();

    /* H5VLregister_connector is idempotent — multiple calls return the same ID */
    return H5VLregister_connector(&H5VL_gpucompress_g, H5P_DEFAULT);
}

extern "C" herr_t
H5Pset_fapl_gpucompress(hid_t fapl_id, hid_t under_vol_id,
                         const void *under_vol_info)
{
    H5VL_gpucompress_info_t info;
    hid_t vol_id;
    herr_t rv;

    vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return -1;

    /* Resolve under_vol_id: H5VL_NATIVE is a class marker, not a regular hid_t.
     * Always use H5VLget_connector_id_by_* to get a proper registered ID. */
    if (!H5Iis_valid(under_vol_id)) {
        /* Caller passed H5VL_NATIVE or an unresolved ID — resolve to native */
        under_vol_id = H5VLget_connector_id_by_name("native");
        if (under_vol_id < 0)
            under_vol_id = H5VLget_connector_id_by_value(H5VL_NATIVE_VALUE);
    }

    info.under_vol_id   = under_vol_id;
    info.under_vol_info = (void*)under_vol_info;

    rv = H5Pset_vol(fapl_id, vol_id, &info);
    H5VLclose(vol_id);
    return rv;
}
