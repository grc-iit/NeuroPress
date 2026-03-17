# Plan: LRU-1 nvcomp Manager Cache per CompContext Slot

## Context

The original N1 optimization (commit `d77d906`) cached one nvcomp manager per
(slot x algorithm) — 8 slots x 8 algos = 64 managers. Deflate/Zstd each hold
~1 GB workspace, so the full cache permanently consumed ~8 GB of GPU memory and
caused OOM on large workloads (VPIC 4 GB, Gray-Scott 8 GB). It was reverted in
commit `3108544`, paying a ~30-40% write throughput regression on small datasets.

The correct fix is **LRU-1**: one cached manager per slot (8 total), replaced
only when the algorithm changes for the next chunk on that slot.
Memory bound = 8 x workspace_of_selected_algo. For workloads where the NN
consistently picks ANS/Cascaded/Bitcomp (~50 MB workspaces), total overhead
is ~400 MB — far below the OOM threshold — and cache hit rate approaches 100%.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cache the exploration winner? | **Yes** | After exploration, `algo_to_use` is updated to the winner. Cache this so the next chunk on this slot benefits. |
| Delete on destroy vs intentional leak? | **Safe delete** | `destroyCompContextSlot()` runs during `gpucompress_cleanup()` while CUDA is still alive. Add `cudaStreamSynchronize(ctx.stream)` before deleting to drain in-flight kernels. |
| Slot allocation strategy? | **No change needed** | `acquireCompContext()` uses first-available scan. With a fixed algo, all slots cache the same algo → hit regardless of which slot a worker gets. With ALGO_AUTO, the NN tends to converge → same effect. |

---

## TDD Approach

### Pre-fix state (DONE)

1. Added cache stats counters (`g_mgr_cache_hits`, `g_mgr_cache_misses`) to
   `gpucompress_api.cpp` — instrumentation only, no caching logic.
2. Added public API: `gpucompress_reset_cache_stats()`, `gpucompress_get_cache_stats()`.
3. `createCompManagerForCtx()` increments `g_mgr_cache_misses` on every call.
4. Created `tests/hdf5/test_lru1_manager_cache.cu` — 4 test cases through HDF5 VOL.

**Pre-fix expected results**: All 4 tests FAIL (hits=0, all misses).

### Post-fix state (TO DO)

Implement LRU-1 caching. Same tests should PASS.

---

## Critical Files

| File | Role | Status |
|------|------|--------|
| `include/gpucompress.h` | Public API: cache stats | DONE |
| `src/api/gpucompress_api.cpp` | Counters + API implementation | DONE (instrumentation) |
| `tests/hdf5/test_lru1_manager_cache.cu` | Pre/post-fix test | DONE |
| `cmake/HDF5Vol.cmake` | Build wiring | DONE |
| `src/api/internal.hpp` | CompContext struct — add 2 fields | TO DO |
| `src/api/gpucompress_api.cpp` | Caching logic + call site updates | TO DO |

---

## What to Change (Post-fix Implementation)

### 1. `src/api/internal.hpp` — Add 2 fields to CompContext

After the existing `d_range_max` field (line ~56), add:

```cpp
/* LRU-1 manager cache — one nvcomp manager per slot.
 * Raw pointer (not unique_ptr): CompContext is memset-zeroed in
 * initCompContextPool(), which would corrupt a non-trivial dtor.
 * Deleted safely in destroyCompContextSlot() after stream sync. */
nvcomp::nvcompManagerBase* comp_mgr;      /* nullptr = empty cache */
int                        comp_mgr_algo; /* -1 = empty, 0-7 = CompressionAlgorithm */
```

Also add a forward declaration for the new function below `createCompManagerForCtx`:

```cpp
/** LRU-1: return cached manager if algo matches, else delete old + create new.
 *  Thread-safe because each CompContext is exclusively owned by one thread. */
nvcomp::nvcompManagerBase* getOrCreateCompManager(CompContext* ctx,
                                                   CompressionAlgorithm algo);
```

Keep `createCompManagerForCtx` — still used for exploration alternatives.

---

### 2. `src/api/gpucompress_api.cpp` — 5 targeted changes

#### 2a. `initCompContextPool()` — after the `memset` call

The pool does `memset(g_comp_pool, 0, sizeof(g_comp_pool))` which zeros
`comp_mgr` (nullptr OK) but sets `comp_mgr_algo = 0` (= LZ4, wrong sentinel).
Fix by explicitly setting the sentinel after memset:

```cpp
// after: memset(g_comp_pool, 0, sizeof(g_comp_pool));
for (int i = 0; i < N_COMP_CTX; i++)
    g_comp_pool[i].comp_mgr_algo = -1;   /* -1 = no cached manager */
```

#### 2b. `destroyCompContextSlot()` — safe delete with stream sync

Add BEFORE the `cudaStreamDestroy` call:

```cpp
/* LRU-1: sync stream to drain in-flight kernels that may reference
 * the manager's workspace, then safely delete the manager. */
if (ctx.stream)
    cudaStreamSynchronize(ctx.stream);
if (ctx.comp_mgr) {
    delete ctx.comp_mgr;
    ctx.comp_mgr = nullptr;
}
ctx.comp_mgr_algo = -1;
```

This is safe because `destroyCompContextSlot()` is called from
`gpucompress_cleanup()` while the CUDA primary context is still alive.

#### 2c. New `getOrCreateCompManager()` — add near `createCompManagerForCtx`

```cpp
nvcomp::nvcompManagerBase* gpucompress::getOrCreateCompManager(
    CompContext* ctx, CompressionAlgorithm algo)
{
    int idx = static_cast<int>(algo);
    if (idx < 0 || idx >= CompContext::N_COMP_ALGOS) return nullptr;

    if (ctx->comp_mgr && ctx->comp_mgr_algo == idx) {
        g_mgr_cache_hits.fetch_add(1);
        return ctx->comp_mgr;          /* cache hit — zero alloc */
    }

    /* Cache miss: CUDA is alive here, safe to delete old manager. */
    delete ctx->comp_mgr;
    auto mgr = createCompressionManager(
        algo, gpucompress::DEFAULT_CHUNK_SIZE, ctx->stream, nullptr);
    ctx->comp_mgr      = mgr.release();
    ctx->comp_mgr_algo = idx;
    g_mgr_cache_misses.fetch_add(1);
    return ctx->comp_mgr;
}
```

#### 2d. Update the 2 PRIMARY compression call sites

**Host path** (~line 748 in `gpucompress_compress`):
```cpp
// BEFORE:
auto compressor_uptr = gpucompress::createCompManagerForCtx(ctx, internal_algo);
auto* compressor = compressor_uptr.get();

// AFTER:
auto* compressor = gpucompress::getOrCreateCompManager(ctx, internal_algo);
```
Remove the `compressor_uptr` variable — no longer needed.

**GPU path** (~line 2133 in `gpucompress_compress_gpu`):
Same substitution.

#### 2e. Cache the exploration winner

After the exploration phase picks the winner (variable `algo_to_use` is updated),
the cached manager should reflect the winner — not the NN's initial pick.
This happens naturally: the primary call at line 748/2133 caches the NN's pick,
and if exploration overrides `algo_to_use`, the NEXT chunk on this slot will
call `getOrCreateCompManager` with the winner's algo. If it matches, it's a hit.

To make the CURRENT chunk benefit (not just the next one): after exploration
updates `algo_to_use`, explicitly update the cache:

```cpp
// After exploration winner is known (GPU path ~line 2522, host path ~line 1214):
// Update cache to the winner so next chunk benefits even if NN's initial
// pick differs from what exploration ultimately chose.
ctx->comp_mgr_algo = static_cast<int>(toInternalAlgorithm(algo_to_use));
// Note: the winning compressed data is already in the output buffer.
// We don't re-compress — we just update what algo the cache "thinks" it holds.
// On next chunk, if the NN picks the same algo as the exploration winner → hit.
```

Wait — this would be inconsistent: `comp_mgr_algo` says "ZSTD" but `comp_mgr`
was created for "LZ4". The manager workspace is algo-specific.

**Correct approach**: after exploration, if the winner differs from the primary,
replace the cached manager with a fresh one for the winner:

```cpp
// After exploration picks winner, if algo changed:
if (algo_to_use != primary_algo) {
    auto winner_internal = toInternalAlgorithm(algo_to_use);
    // This replaces the cache with the winner's manager.
    // The winner's compressed data is already in the output buffer,
    // so we don't use this manager for the current chunk — we create
    // it for the NEXT chunk's benefit.
    getOrCreateCompManager(ctx, winner_internal);
}
```

This is a **miss** (creates a new manager) but ensures the cache holds the
winning algo for the next chunk. Cost: 1 extra creation per exploration switch.
Benefit: next chunk on same slot with same winner algo → hit.

**Exploration alternatives** (~line 1022): **DO NOT CHANGE.**
These stay as `createCompManagerForCtx` unique_ptrs (temporary, freed per-alternative).

---

## Memory Analysis

| Config | Managers | Peak GPU memory |
|--------|----------|-----------------|
| Old N1 (reverted) | 8 slots x 8 algos = 64 | ~8-64 GB permanent |
| Current (no cache) | 0 idle; 1 per active call | ~0 idle |
| **LRU-1 (this plan)** | **8 max (1 per slot)** | **~400 MB typical; 8 GB worst case** |

Worst case (8 GB) requires all 8 slots to simultaneously hold a cached Zstd/Deflate
manager. This cannot happen alongside large-dataset benchmarks because the NN
empirically selects ANS/Cascaded/Bitcomp for Gray-Scott and VPIC data
(verified in `test_nn_vol_correctness` output: ANS 8x, Cascaded 18x, Bitcomp 6x
across 32 chunks).

---

## Constraints Preserved

- **Raw pointer in CompContext**: required because `memset(g_comp_pool, ...)` is
  used in `initCompContextPool()`. A `unique_ptr` member with non-trivial dtor
  would be corrupted by memset.
- **Safe delete on destroy**: `cudaStreamSynchronize` before `delete` ensures no
  in-flight kernels reference the manager's workspace. Runs while CUDA is alive.
- **Exploration alternatives unchanged**: K alternative managers stay as temporary
  `createCompManagerForCtx` unique_ptrs. Only the primary/winner is cached.
- **Stream binding**: each manager is bound to `ctx->stream` at creation time.
  Safe because each CompContext is exclusively owned by one thread at a time.
- **`N_COMP_ALGOS = 8` constant**: already present in CompContext; reused as the
  bounds check in `getOrCreateCompManager`.

---

## Test Cases (through HDF5 VOL)

| Test | Chunks | Algo | Expected pre-fix | Expected post-fix |
|------|--------|------|-----------------|-------------------|
| A | 16 LZ4 | Fixed | hits=0, FAIL | hits>=8, PASS |
| B | 32 ZSTD | Fixed | hits=0, FAIL | hits>=24, PASS |
| C | 16 LZ4 + 16 ZSTD | Switch | hits=0, FAIL | hits>=16, PASS |
| D | 16 LZ4 | Correctness | hits=0, FAIL (cache check) | hits>=8, PASS + bit-exact |

File: `tests/hdf5/test_lru1_manager_cache.cu`

---

## Verification

```bash
# Step 1: Build and run PRE-FIX test (should FAIL on cache hit assertions)
cmake --build build --target test_lru1_manager_cache -j$(nproc)
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
./build/test_lru1_manager_cache

# Step 2: Implement LRU-1 changes (internal.hpp + gpucompress_api.cpp)

# Step 3: Rebuild and run POST-FIX test (should PASS)
cmake --build build --target test_lru1_manager_cache -j$(nproc)
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
./build/test_lru1_manager_cache

# Step 4: Run existing correctness tests (regression check)
export GPUCOMPRESS_WEIGHTS=/u/imuradli/GPUCompress/neural_net/weights/model.nnwt
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
./build/test_nn_vol_correctness

LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
HDF5_PLUGIN_PATH=$(pwd)/build \
./build/test_hdf5_compat

# Step 5: Optional large-scale OOM check
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
./build/grayscott_benchmark --steps 10 --chunk-mb 64
```
