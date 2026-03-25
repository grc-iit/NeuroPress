# P0: Global VolWriteCtx Buffer Cache

**Status**: Designed & reviewed — ready to implement
**Priority**: Highest — eliminates 71% of write overhead
**Estimated savings**: ~680ms/timestep (setup) + ~290ms/timestep (close) = ~970ms/timestep

## Problem

The HDF5 lifecycle `H5Fcreate → H5Dcreate → H5Dwrite → H5Dclose → H5Fclose` destroys
`VolWriteCtx` at `H5Dclose` (`H5VLgpucompress.cu:2195-2206`), freeing:

- 8 × `cudaMalloc(72MB)` = 576MB device memory
- 16 × `cudaMallocHost(72MB)` = 1.15GB pinned host memory

Every subsequent `H5Dwrite` re-allocates all of this. Cost: **~750ms setup + ~290ms close
= ~1040ms/timestep** of pure allocation overhead.

Affects all 11 compressed phases equally (~735-745ms each). `no-comp` bypasses the VOL
pipeline so is unaffected.

## Measured Data (Gray-Scott 640³, 64MB chunks, balanced)

```
phase    T  setup_ms  pipeline_ms  h5dclose_ms  write_ms
nn       0  743       62           290          1153
nn       1  745       14           288          1064
lz4      0  737       810          286          1097
lz4      1  731       811          289          1100
```

Actual pipeline work is only 14-62ms. Setup dominates at 70-91% of h5dwrite_ms.

## Solution: File-Static Single-Slot Buffer Cache

### Concept

```
T=0:  H5Dwrite → allocate 1.7GB (cold)  → compress → H5Dclose → DONATE to cache
T=1:  H5Dwrite → RECLAIM from cache (0ms) → compress → H5Dclose → DONATE to cache
T=2:  H5Dwrite → RECLAIM from cache (0ms) → compress → H5Dclose → DONATE to cache
cleanup: FREE cached 1.7GB
```

### Implementation

**New struct** (file-static, `H5VLgpucompress.cu` near line 61):

```cpp
struct VolBufCache {
    uint8_t* d_comp_w[M4_N_COMP_WORKERS];
    void*    io_pool_bufs[M4_N_IO_BUFS];
    size_t   max_comp;
    bool     valid;
};
static std::mutex   s_buf_cache_mtx;
static VolBufCache  s_buf_cache = {};
```

**H5Dwrite setup** (replace lines 1131-1157):

```cpp
if (!o->write_ctx) {
    o->write_ctx = new VolWriteCtx{};
    // Try reclaim from global cache
    std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
    if (s_buf_cache.valid && s_buf_cache.max_comp >= max_comp) {
        memcpy(o->write_ctx->d_comp_w, s_buf_cache.d_comp_w, sizeof(s_buf_cache.d_comp_w));
        memcpy(o->write_ctx->io_pool_bufs, s_buf_cache.io_pool_bufs, sizeof(s_buf_cache.io_pool_bufs));
        o->write_ctx->max_comp = s_buf_cache.max_comp;
        o->write_ctx->initialized = true;
        s_buf_cache.valid = false;
    }
}
VolWriteCtx* wctx = o->write_ctx;
// Fall through to existing alloc code if cache missed
```

**H5Dclose** (replace lines 2195-2206):

```cpp
if (o->write_ctx) {
    VolWriteCtx* wctx = o->write_ctx;
    if (wctx->initialized) {
        std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
        if (!s_buf_cache.valid || wctx->max_comp > s_buf_cache.max_comp) {
            // Evict smaller cache if present
            if (s_buf_cache.valid) { /* free old cache buffers */ }
            // Donate to cache
            memcpy(s_buf_cache.d_comp_w, wctx->d_comp_w, ...);
            memcpy(s_buf_cache.io_pool_bufs, wctx->io_pool_bufs, ...);
            s_buf_cache.max_comp = wctx->max_comp;
            s_buf_cache.valid = true;
        } else {
            // Cache has larger buffers; free these normally
            for (...) cudaFree / cudaFreeHost;
        }
    }
    delete wctx;
    o->write_ctx = nullptr;
}
```

**Cleanup** (new function, called from `gpucompress_cleanup()`):

```cpp
void H5VL_gpucompress_release_buf_cache(void) {
    std::lock_guard<std::mutex> lk(s_buf_cache_mtx);
    if (s_buf_cache.valid) {
        for (int w = 0; w < M4_N_COMP_WORKERS; w++)
            if (s_buf_cache.d_comp_w[w]) cudaFree(s_buf_cache.d_comp_w[w]);
        for (int i = 0; i < M4_N_IO_BUFS; i++)
            if (s_buf_cache.io_pool_bufs[i]) cudaFreeHost(s_buf_cache.io_pool_bufs[i]);
        s_buf_cache.valid = false;
    }
}
```

## Expected Impact

| Metric         | Before    | After     | Savings          |
|----------------|-----------|-----------|------------------|
| T=0 setup      | 750ms     | 750ms     | 0 (cold start)   |
| T≥1 setup      | 750ms     | ~3ms      | **747ms/ts**      |
| T≥1 h5dclose   | 290ms     | ~0.1ms    | **290ms/ts**      |
| 15-ts run total | 15.6s overhead | 1.0s overhead | **~14.6s saved** |

## Required Changes (from reviewer)

1. **C1**: Fix pre-existing partial-allocation leak — if `cudaMalloc` fails midway after
   cache reclaim, clean up already-allocated buffers before `goto done_write`
2. **W1**: Call `H5VL_gpucompress_release_buf_cache()` from `gpucompress_cleanup()`, NOT
   `atexit` — ensures CUDA context is still alive
3. **W2**: Remove redundant `memset` after `new VolWriteCtx{}` (value-init already zeros)

## Design Decisions

- **Single cache slot**: Covers the common case (same chunk size across timesteps).
  Larger cached buffers serve smaller requests. No LRU needed.
- **Thread pool NOT included**: Thread creation is ~1-2ms after cache fix. Not worth
  the 200-line complexity for that savings.
- **Mutex-protected**: HDF5 serializes VOL calls, so the mutex is never contended.
  Kept for defensive correctness.
- **Memory**: Same peak allocation as today. Buffers just survive across H5Dclose/H5Dcreate.

## Files to Modify

- `src/hdf5/H5VLgpucompress.cu` — cache struct, reclaim in setup, donate in close
- `include/gpucompress_hdf5_vol.h` — declare `H5VL_gpucompress_release_buf_cache()`
- `src/api/gpucompress_api.cpp` — call release from `gpucompress_cleanup()`
