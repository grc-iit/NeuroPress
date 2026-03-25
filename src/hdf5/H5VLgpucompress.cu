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
#include "selection/heuristic.h"

/* gpucompress HDF5 filter ID (must match H5Zgpucompress.h) */
#define H5Z_FILTER_GPUCOMPRESS 305

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
/* Transfer byte counters */
static size_t s_h2d_bytes = 0;   /* total bytes copied host→device */
static size_t s_d2h_bytes = 0;   /* total bytes copied device→host */
static size_t s_d2d_bytes = 0;   /* total bytes copied device→device */
static int    s_h2d_count = 0;   /* number of H→D cudaMemcpy calls  */
static int    s_d2h_count = 0;   /* number of D→H cudaMemcpy calls  */
static int    s_d2d_count = 0;   /* number of D→D cudaMemcpy calls  */
/* Wall-clock stage timing (ms) for the last H5Dwrite call */
static double s_stage1_ms = 0;   /* Stage 1: stats + NN inference (main thread) */
static double s_stage2_ms = 0;   /* Stage 2: worker join - stage1 end (compression + D→H wall clock) */
static double s_stage3_ms = 0;   /* Stage 3: HDF5 chunk writes (I/O thread wall clock) */
static double s_total_ms  = 0;   /* Total wall clock for the pipeline */
static double s_vol_func_ms = 0; /* Total wall clock for gpu_aware_chunked_write */
static double s_setup_ms = 0;   /* Setup before pipeline (VolWriteCtx, threads, etc) */
static double s_join_ms_g = 0;  /* Thread join time (signal I/O done + join) */
/* Max per-worker wall-clock time (bottleneck worker = actual Stage 2 wall time) */
static double s_max_worker_comp_ms = 0;

static double _now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Wrapper: track and execute cudaMemcpy */
static inline cudaError_t vol_memcpy(void *dst, const void *src,
                                      size_t bytes, cudaMemcpyKind kind)
{
    switch (kind) {
        case cudaMemcpyHostToDevice:   s_h2d_bytes += bytes; s_h2d_count++; break;
        case cudaMemcpyDeviceToHost:   s_d2h_bytes += bytes; s_d2h_count++; break;
        case cudaMemcpyDeviceToDevice: s_d2d_bytes += bytes; s_d2d_count++; break;
        default: break;
    }
    return cudaMemcpy(dst, src, bytes, kind);
}

extern "C" void
H5VL_gpucompress_reset_stats(void)
{
    s_gpu_writes.store(0);      s_gpu_reads.store(0);
    s_chunks_comp.store(0);     s_chunks_decomp.store(0);
    s_h2d_bytes = s_d2h_bytes = s_d2d_bytes = 0;
    s_h2d_count = s_d2h_count = s_d2d_count = 0;
    s_stage1_ms = s_stage2_ms = s_stage3_ms = s_total_ms = 0;
    s_vol_func_ms = s_setup_ms = s_join_ms_g = 0;
    s_max_worker_comp_ms = 0;
}

extern "C" void
H5VL_gpucompress_get_stage_timing(double *stage1_ms, double *stage2_ms,
                                   double *stage3_ms, double *total_ms)
{
    if (stage1_ms) *stage1_ms = s_stage1_ms;
    if (stage2_ms) *stage2_ms = s_stage2_ms;
    if (stage3_ms) *stage3_ms = s_stage3_ms;
    if (total_ms)  *total_ms  = s_total_ms;
}

extern "C" void
H5VL_gpucompress_get_vol_func_timing(double *setup_ms, double *vol_func_ms,
                                      double *join_ms)
{
    if (setup_ms)    *setup_ms    = s_setup_ms;
    if (vol_func_ms) *vol_func_ms = s_vol_func_ms;
    if (join_ms)     *join_ms     = s_join_ms_g;
}

extern "C" void
H5VL_gpucompress_get_worker_timing(double *max_worker_ms)
{
    if (max_worker_ms) *max_worker_ms = s_max_worker_comp_ms;
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
    if (h2d_count) *h2d_count = s_h2d_count;
    if (h2d_bytes) *h2d_bytes = s_h2d_bytes;
    if (d2h_count) *d2h_count = s_d2h_count;
    if (d2h_bytes) *d2h_bytes = s_d2h_bytes;
    if (d2d_count) *d2d_count = s_d2d_count;
    if (d2d_bytes) *d2d_bytes = s_d2d_bytes;
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

static herr_t H5VL_gpucompress_init(hid_t /*vipl_id*/) { return 0; }
static herr_t H5VL_gpucompress_term(void)               { return 0; }

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
            int            top_actions[32];
            float          predicted_costs[32];
            /* Timing from Stage 1 inference (ms) */
            float          infer_nn_ms;
            float          infer_stats_ms;
            /* Pre-computed stats from Stage 1 (D→D copy from infer_ctx).
             * Avoids redundant stats recomputation in Stage 2 for SGD. */
            AutoStatsGPU*  d_stats_copy;   /* owned: worker cudaFrees after use */
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
        double _pipeline_start = 0, _s1_start = 0;
        double io_write_total_ms = 0;  /* direct S3 measurement from I/O thread */
        double worker_comp_ms[M4_N_COMP_WORKERS];
        memset(worker_comp_ms, 0, sizeof(worker_comp_ms));
        hsize_t *d_dset_dims = NULL, *d_chunk_dims = NULL, *d_chunk_start = NULL;
        uint8_t* d_comp_w[N_COMP_WORKERS] = {};
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
            o->write_ctx = new VolWriteCtx{};
            memset(o->write_ctx, 0, sizeof(VolWriteCtx));
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
            if (!ok) goto done_write;
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
                        gpucompress_error_t ce;
                        int diag_slot = -1;
                        if (wi.has_inference) {
                            ce = gpucompress_compress_with_action_gpu(
                                wi.src, wi.sz, d_comp_w[w], &comp_sz, &cfg, &wstats, NULL,
                                wi.action, wi.predicted_ratio, wi.predicted_comp_time,
                                wi.predicted_decomp_time, wi.predicted_psnr,
                                wi.top_actions, wi.predicted_costs,
                                wi.infer_nn_ms, wi.infer_stats_ms,
                                wi.d_stats_copy, &diag_slot);
                        } else {
                            ce = gpucompress_compress_gpu(
                                wi.src, wi.sz, d_comp_w[w], &comp_sz, &cfg, &wstats, NULL);
                        }

                        w_total += _now_ms() - _w_start;

                        /* Free per-chunk owned buffers after compression completes */
                        if (wi.d_owned) { cudaFree(wi.d_owned); wi.d_owned = NULL; }
                        if (wi.d_stats_copy) { cudaFree(wi.d_stats_copy); wi.d_stats_copy = NULL; }

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
                        double _t_d2h = _now_ms();
                        if (cudaMemcpy(hbuf, d_comp_w[w], comp_sz,
                                       cudaMemcpyDeviceToHost) != cudaSuccess) {
                            pool_release(hbuf);
                            worker_err.store((herr_t)-1);
                            wq_ready_cv.notify_all();
                            pool_cv.notify_all();
                            break;
                        }
                        float vol_d2h_ms = (float)(_now_ms() - _t_d2h);

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
                            gpucompress_record_chunk_vol_timing(diag_slot,
                                vol_pool_ms, vol_d2h_ms, vol_io_wait_ms);
                            gpucompress_record_chunk_s1_timing(diag_slot,
                                wi.vol_stats_malloc_ms, wi.vol_stats_copy_ms, wi.vol_wq_post_wait_ms);
                        }
                    }
                    /* Store per-worker total wall time */
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
                bool use_seq_inference = (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO);
                CompContext* infer_ctx = nullptr;
                if (use_seq_inference) {
                    infer_ctx = gpucompress::acquireCompContext();
                    if (!infer_ctx) {
                        fprintf(stderr, "gpucompress VOL: failed to acquire inference CompContext\n");
                        use_seq_inference = false;
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
                     * Each chunk sees the latest SGD-updated weights because
                     * runNNFusedInferenceCtx waits on g_sgd_done before reading.
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
                                memset(wi.top_actions, 0, sizeof(wi.top_actions));
                                memset(wi.predicted_costs, 0, sizeof(wi.predicted_costs));
                                wi.infer_nn_ms    = 0.0f;
                                cudaEventElapsedTime(&wi.infer_stats_ms,
                                    infer_ctx->stats_start, infer_ctx->stats_stop);
                                wi.d_stats_copy   = nullptr;  /* no SGD in heuristic mode */
                            }
                        } else {
                            /* NN path: full stats + inference */
                            int infer_action = -1;
                            float infer_ratio = 0, infer_ct = 0, infer_dt = 0, infer_psnr = 0;
                            int infer_top[32] = {0};
                            float infer_costs[32] = {0};
                            gpucompress_error_t ie = gpucompress_infer_gpu(
                                wi.src, wi.sz, &cfg, nullptr, infer_ctx,
                                &infer_action, &infer_ratio, &infer_ct, &infer_dt, &infer_psnr,
                                infer_top, infer_costs);
                            if (ie == GPUCOMPRESS_SUCCESS && infer_action >= 0) {
                                wi.has_inference      = true;
                                wi.action             = infer_action;
                                wi.predicted_ratio    = infer_ratio;
                                wi.predicted_comp_time = infer_ct;
                                wi.predicted_decomp_time = infer_dt;
                                wi.predicted_psnr     = infer_psnr;
                                memcpy(wi.top_actions, infer_top, sizeof(infer_top));
                                memcpy(wi.predicted_costs, infer_costs, sizeof(infer_costs));
                                /* Capture Stage 1 timing from infer_ctx CUDA events */
                                cudaEventElapsedTime(&wi.infer_nn_ms, infer_ctx->nn_start, infer_ctx->nn_stop);
                                cudaEventElapsedTime(&wi.infer_stats_ms, infer_ctx->stats_start, infer_ctx->stats_stop);
                                /* Copy stats from infer_ctx to a per-chunk buffer so
                                 * Stage 2 worker can reuse them for SGD without recomputing.
                                 * infer_ctx->d_stats will be overwritten by the next chunk. */
                                double _t_sm = _now_ms();
                                AutoStatsGPU* d_sc = nullptr;
                                if (cudaMalloc(&d_sc, sizeof(AutoStatsGPU)) != cudaSuccess) {
                                    fprintf(stderr, "gpucompress VOL: stats copy alloc failed\n");
                                    ret = -1; break;
                                }
                                wi.vol_stats_malloc_ms = (float)(_now_ms() - _t_sm);

                                double _t_sc = _now_ms();
                                cudaError_t sc_err = cudaMemcpyAsync(d_sc, infer_ctx->d_stats, sizeof(AutoStatsGPU),
                                                                      cudaMemcpyDeviceToDevice, infer_ctx->stream);
                                if (sc_err != cudaSuccess) {
                                    fprintf(stderr, "gpucompress VOL: stats copy memcpy failed\n");
                                    cudaFree(d_sc); ret = -1; break;
                                }
                                sc_err = cudaStreamSynchronize(infer_ctx->stream);
                                if (sc_err != cudaSuccess) {
                                    fprintf(stderr, "gpucompress VOL: stats copy sync failed\n");
                                    cudaFree(d_sc); ret = -1; break;
                                }
                                wi.vol_stats_copy_ms = (float)(_now_ms() - _t_sc);
                                wi.d_stats_copy = d_sc;
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
                          if (wi.d_stats_copy) { cudaFree(wi.d_stats_copy); wi.d_stats_copy = NULL; }
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
            s_stage2_ms = _now_ms() - _s1_start - s_stage1_ms; /* exclude S1 overlap */
            /* Compute max per-worker total wall time (= bottleneck worker) */
            for (int w = 0; w < N_COMP_WORKERS; w++) {
                if (worker_comp_ms[w] > s_max_worker_comp_ms) s_max_worker_comp_ms = worker_comp_ms[w];
            }
            if (worker_err.load() != 0 && ret == 0) ret = (herr_t)-1;

        } /* workers + Stage 1 scope */

done_write:
        /* ---- Signal I/O thread and join ---- */
        double s_join_ms = 0;
        if (io_started) {
            double _t_join = _now_ms();
            { std::lock_guard<std::mutex> lk(io_mtx); io_done_flag = true; }
            io_cv.notify_all();
            io_thr.join();
            s_join_ms = _now_ms() - _t_join;
            s_join_ms_g = s_join_ms;
            s_total_ms = _now_ms() - _pipeline_start;  /* total wall clock */
            /* Stage 3: direct measurement from I/O thread */
            s_stage3_ms = io_write_total_ms;
            if (io_err < 0 && ret == 0) ret = -1;
        }

        s_vol_func_ms = _now_ms() - _vol_func_start;

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
            gpucompress_record_chunk_decomp_ms((int)ci, _decomp_ms);
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

        /* Batched deferred decomp SGD: one update from all chunks' decomp times */
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
    VOL_TRACE("  → H5VLdataset_create(native, \"%s\")", name ? name : "?");
    void *under = H5VLdataset_create(o->under_object, lp, o->under_vol_id,
                                     name, lcpl_id, type_id, space_id,
                                     dcpl_id, dapl_id, dxpl_id, req);
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
    herr_t ret;

    /* For each dataset, check if buf is a GPU pointer */
    for (size_t i = 0; i < count; i++) {
        H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset[i];

        if (o->dcpl_id != H5I_INVALID_HID && buf[i] &&
            gpucompress_is_device_ptr(buf[i])) {

            gpucompress_config_t cfg;
            if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) == 0) {
                VOL_TRACE("dataset_read[%zu]: buf=%p → GPU path (gpucompress filter)", i, buf[i]);
                /* GPU path: read+decompress chunk by chunk */
                ret = gpu_aware_chunked_read(o, mem_type_id[i],
                                             file_space_id[i], plist_id,
                                             buf[i]);
            } else {
                VOL_TRACE("dataset_read[%zu]: buf=%p → fallback native+H→D (no gpucompress filter)", i, buf[i]);
                /* No gpucompress filter — native one-shot read then H→D copy */
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

    if (req && *req && count > 0)
        *req = new_obj(*req, ((H5VL_gpucompress_t*)dset[0])->under_vol_id);
    return 0;
}

static herr_t
H5VL_gpucompress_dataset_write(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
    hid_t plist_id, const void *buf[], void **req)
{
    herr_t ret;

    for (size_t i = 0; i < count; i++) {
        H5VL_gpucompress_t *o = (H5VL_gpucompress_t*)dset[i];

        if (o->dcpl_id != H5I_INVALID_HID && buf[i] &&
            gpucompress_is_device_ptr(buf[i])) {

            gpucompress_config_t cfg;
            if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) == 0) {
                VOL_TRACE("dataset_write[%zu]: buf=%p → GPU path (gpucompress filter)", i, buf[i]);
                /* GPU path: compress+write chunk by chunk */
                ret = gpu_aware_chunked_write(o, mem_type_id[i],
                                              file_space_id[i], plist_id,
                                              buf[i]);
            } else {
                VOL_TRACE("dataset_write[%zu]: buf=%p → fallback D→H (no gpucompress filter)", i, buf[i]);
                /* No gpucompress filter — D→H copy then native one-shot write */
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
        /* M4 fix: free persistent write buffers */
        if (o->write_ctx) {
            VolWriteCtx* wctx = o->write_ctx;
            if (wctx->initialized) {
                for (int w = 0; w < M4_N_COMP_WORKERS; w++) {
                    if (wctx->d_comp_w[w]) cudaFree(wctx->d_comp_w[w]);
                }
                for (int i = 0; i < M4_N_IO_BUFS; i++) {
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
