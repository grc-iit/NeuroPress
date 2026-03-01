# GPUCompress — Bug Analysis Report

**Date:** 2026-03-01
**Scope:** VOL-2, VOL-1, API-3, PREP-1 — detailed correctness analysis
**Source:** Cross-referenced against source code; findings from ISSUES_AND_FINDINGS.md verified.

---

## Table of Contents

1. [VOL-2 — Missing gpucompress filter returns failure instead of fall-through](#vol-2)
2. [VOL-1 — GPU path ignores mem_space_id (hyperslab writes unsupported)](#vol-1)
3. [API-3 — Pool init partial failure leaks GPU memory](#api-3)
4. [PREP-1 — TOCTOU race in ensure_range_bufs()](#prep-1)
5. [Priority Summary](#priority-summary)

---

## VOL-2 ✅ FIXED (2026-03-01)

**Fix:** Option B — D↔H copy fallback implemented in `gpu_fallback_dh_write()` / `gpu_fallback_hd_read()`.
**Test:** `tests/test_vol2_gpu_fallback.cu` — 6/6 pass.

### Missing gpucompress filter returns failure instead of fall-through

| Field | Detail |
|---|---|
| **File** | `src/hdf5/H5VLgpucompress.cu` |
| **Function** | `gpu_aware_chunked_write()` |
| **Lines** | 946, 982–984, 1280–1282 |
| **Severity** | **High — real correctness bug, always triggers on affected datasets** |
| **Type** | Wrong return value / misleading comment |

### What the code does

`gpu_aware_chunked_write()` opens with `ret = -1` (failure) and uses a `goto done` pattern for all early exits:

```cpp
static herr_t gpu_aware_chunked_write(...) {
    herr_t ret = -1;          // line 946 — initialized to failure
    ...

    if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) != 0) {
        /* No gpucompress filter found — fall through to host path */
        goto done;             // line 984 — jumps with ret still -1
    }
    ...
    ret = 0;                   // only set on full success deep in the function
done:
    if (need_close_space) H5Sclose(actual_space_id);
    return ret;                // line 1282 — returns -1
}
```

The comment on line 983 says **"fall through to host path"** but the code returns `-1` (HDF5 error). These are opposite behaviors.

### When does this trigger?

Any time a user passes a GPU (device) pointer to `H5Dwrite()` on a dataset whose creation property list (DCPL) does **not** contain the gpucompress filter. Concrete scenarios:

- Writing to a dataset created by external code with a different filter (zlib, zstd, etc.)
- Writing to a dataset with no compression filter at all
- During testing / development when the filter was accidentally omitted from the DCPL

### Impact

`H5Dwrite()` returns a negative `herr_t`. The write silently fails from the application's perspective. If the application does not check the return value (common in research code), the dataset is left in an undefined / empty state with no error message beyond HDF5's internal error stack.

The same bug exists symmetrically in `gpu_aware_chunked_read()` (line 1534) for the read path.

### Intended behavior (from the comment)

When no gpucompress filter is present, the GPU path should fall back — either:
- **Option A (silent success, no write):** set `ret = 0` before `goto done` — the GPU write does nothing; behavior mirrors host pass-through for unrecognized datasets.
- **Option B (D→H copy + native write):** copy `d_buf` to host, then call `H5VLdataset_write()` on the underlying VOL — data is written without GPU compression.

Option A is the minimal fix. Option B preserves data integrity for the user.

### Fix (Option A — minimal)

```cpp
if (get_gpucompress_config_from_dcpl(o->dcpl_id, &cfg) != 0) {
    /* No gpucompress filter — dataset not managed by gpucompress; succeed silently */
    ret = 0;
    goto done;
}
```

---

## VOL-1

### GPU path ignores mem_space_id (hyperslab writes unsupported)

| Field | Detail |
|---|---|
| **File** | `src/hdf5/H5VLgpucompress.cu` |
| **Functions** | `H5VL_gpucompress_dataset_write()`, `gpu_aware_chunked_write()` |
| **Lines** | 1628, 1641–1643 |
| **Severity** | **Medium — silent wrong data for partial/hyperslab GPU writes** |
| **Type** | Dropped parameter / unsupported HDF5 I/O mode |

### What the code does

`H5VL_gpucompress_dataset_write()` receives both `mem_space_id` (how the buffer is laid out in memory) and `file_space_id` (which dataset elements are selected). It silently drops `mem_space_id` when dispatching to `gpu_aware_chunked_write()`:

```cpp
// Outer function receives mem_space_id:
H5VL_gpucompress_dataset_write(... hid_t mem_space_id[] ...)

// But only passes file_space_id to the GPU path:
ret = gpu_aware_chunked_write(o, mem_type_id[i],
                              file_space_id[i],   // ← file space
                              plist_id,
                              buf[i]);
//  mem_space_id[i] is never passed ← dropped here
```

`gpu_aware_chunked_write()` then iterates over all chunk indices in the file dataspace, computing buffer offsets assuming `d_buf` is a contiguous C-order array of the **entire** dataset.

### Full-dataset write — works correctly

```
H5Dwrite(dset, type, H5S_ALL, H5S_ALL, plist, d_buf)

d_buf:  [e0][e1][e2]...[eN]   ← all N elements of the dataset
file:   all chunks, whole dataset extent

VOL: iterates all chunks, offsets = chunk_index * chunk_size → correct ✅
```

### Hyperslab write — produces wrong data

```
// Application writes only rows 10–19 of a 100-row dataset:
H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, start={10}, count={10})
H5Dwrite(dset, type, mem_space, file_space, plist, d_buf)

d_buf:  [e10][e11]...[e19]    ← only 10 elements in the buffer
file:   rows 10–19 of dataset

VOL: iterates ALL chunk indices (0..99), computes offsets as if d_buf
     has 100 elements → reads past buffer end, writes wrong data ❌
```

### Impact

Any code that uses `H5Sselect_hyperslab()` or other selection functions with a GPU buffer will silently write incorrect data. Reads have the symmetric problem — wrong buffer regions are filled. This is a data corruption scenario with no error returned to the caller.

**Most GPU users are unaffected** because simulation workflows typically write full in-memory arrays with `H5S_ALL`. Partial hyperslab writes with GPU buffers are an advanced HDF5 usage pattern that happens to be completely broken today.

### Recommended resolution

Two options:

**Option A — Document as unsupported (minimal change):**
Add a check at the start of `gpu_aware_chunked_write()` that verifies `file_space_id == H5S_ALL` or that the full dataset extent is selected. If a partial selection is detected, either return an error with a clear message or fall back to the host path.

**Option B — Implement hyperslab support (full fix):**
Pass `mem_space_id` into `gpu_aware_chunked_write()`, use `H5Sget_select_hyper_blocklist()` to enumerate selected file regions, and compute correct buffer offsets per chunk using both the file selection and the memory layout. Significantly more complex.

Option A is the correct near-term action — make the limitation explicit rather than silently producing wrong data.

---

## API-3

### Pool init partial failure leaks GPU memory

| Field | Detail |
|---|---|
| **File** | `src/api/gpucompress_api.cpp` |
| **Function** | `initCompContextPool()` |
| **Lines** | 167–201 |
| **Severity** | **Low — one-time GPU memory leak on OOM at startup** |
| **Type** | Missing cleanup on error path |

### What the code does

`initCompContextPool()` allocates resources for 8 `CompContext` slots in a loop. Each slot requires ~76.5 KB of GPU memory across 9 allocations (stream, 2 events, stats workspace, inference buffers, SGD buffers, range buffers). Any failed allocation causes an immediate `return -1`:

```cpp
for (int i = 0; i < N_COMP_CTX; i++) {   // 8 iterations
    if (cudaStreamCreate(&ctx.stream)            != cudaSuccess) return -1;
    if (cudaEventCreate (&ctx.t_start)           != cudaSuccess) return -1;
    if (cudaEventCreate (&ctx.t_stop)            != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_stats_workspace, ...)  != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_fused_infer_output,..) != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_fused_top_actions, ..) != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_sgd_grad_buffer, ...)  != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_sgd_output, ...)       != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_sgd_samples, ...)      != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_range_min, ...)        != cudaSuccess) return -1;
    if (cudaMalloc(&ctx.d_range_max, ...)        != cudaSuccess) return -1;
    // ↑ any of these failing returns without freeing slots 0..i-1
}
```

**Example failure scenario:** Slot 5's `d_sgd_grad_buffer` (`NN_SGD_GRAD_SIZE * sizeof(float)` = ~76 KB) fails. At this point slots 0–4 are fully allocated (~380 KB GPU) and slot 5 is partially allocated. `return -1` fires — none of these resources are freed.

### Impact in practice

Very limited. This requires `cudaMalloc` to fail during library initialization, which only happens if the GPU is already severely out of memory at process startup. In that case the process is almost certainly going to abort anyway, making the leak inconsequential. There is no runtime correctness risk — this is a one-time startup event.

### Fix

Call `destroyCompContextPool()` before returning on failure. Since `destroyCompContextPool()` guards every free with a null check, it safely handles partially-initialized pools:

```cpp
int initCompContextPool() {
    ...
    for (int i = 0; i < N_COMP_CTX; i++) {
        ...
        if (cudaMalloc(&ctx.d_range_max, sizeof(double)) != cudaSuccess) {
            destroyCompContextPool();   // clean up already-allocated slots
            return -1;
        }
        ...
    }
    return 0;
}
```

---

## PREP-1

### TOCTOU race in ensure_range_bufs()

| Field | Detail |
|---|---|
| **File** | `src/preprocessing/quantization_kernels.cu` |
| **Function** | `ensure_range_bufs()`, no-buffer overload of `quantize_simple()` |
| **Lines** | 28–39, 438–447 |
| **Severity** | **Low — race is unreachable on the production path** |
| **Type** | Check-then-act race on shared mutable state |

### What the code does

The no-buffer overload of `quantize_simple()` uses two shared static device pointers for min/max reduction:

```cpp
static void* d_range_min = nullptr;   // shared across all callers
static void* d_range_max = nullptr;

static int ensure_range_bufs() {
    if (d_range_min == nullptr) {          // CHECK
        if (cudaMalloc(&d_range_min, ...) // ACT — not atomic with CHECK
            != cudaSuccess) return -1;
    }
    if (d_range_max == nullptr) {
        if (cudaMalloc(&d_range_max, ...)
            != cudaSuccess) return -1;
    }
    return 0;
}
```

### The race

If two threads call the no-buffer `quantize_simple()` concurrently:

```
Thread A                          Thread B
─────────────────────────────     ────────────────────────────
reads d_range_min → nullptr       reads d_range_min → nullptr
cudaMalloc → ptr_A                cudaMalloc → ptr_B
d_range_min = ptr_A               d_range_min = ptr_B   ← overwrites A
                                  ptr_A is leaked forever
```

**Two consequences:**
1. `ptr_A` is leaked (GPU memory, 8 bytes — negligible).
2. Both threads now share `d_range_min = ptr_B` and run concurrent GPU kernels writing to the same 8-byte device pointer. The min/max reduction results corrupt each other → incorrect quantization scale → wrong compressed data.

### Is the production path affected?

**No.** The production path (VOL + `gpucompress_compress_gpu()`) always uses the thread-safe overload:

```cpp
// Thread-safe: each CompContext slot has its own d_range_min/d_range_max
quantize_simple(d_input, n, elem_sz, cfg, ctx->d_range_min, ctx->d_range_max, stream);
```

The no-buffer (racing) overload is only called from `gpucompress_compress()` (the CPU host compress path), which is single-threaded by design. The race can only be triggered by:
- Calling the public `gpucompress_compress()` concurrently from multiple threads, which is not a documented or tested usage pattern.

The code even documents this limitation with a comment:
```
// NOTE: Not safe for concurrent calls — use the overload with explicit
// d_range_min_buf/d_range_max_buf (e.g. from CompContext) for that.
```

### Fix

Replace the check-then-act with `std::call_once`:

```cpp
#include <mutex>

static void*     d_range_min = nullptr;
static void*     d_range_max = nullptr;
static std::once_flag s_range_buf_once;

static int ensure_range_bufs() {
    int rc = 0;
    std::call_once(s_range_buf_once, [&rc]() {
        if (cudaMalloc(&d_range_min, sizeof(double)) != cudaSuccess) { rc = -1; return; }
        if (cudaMalloc(&d_range_max, sizeof(double)) != cudaSuccess) { rc = -1; return; }
    });
    return rc;
}
```

---

## Priority Summary

| ID | Severity | Triggers in normal use? | Data corruption? | Fix complexity |
|---|---|---|---|---|
| **VOL-2** | High | **Yes** — any GPU write to non-gpucompress dataset | Write silently dropped | ✅ Fixed (2026-03-01) |
| **VOL-1** | Medium | Only with hyperslab + GPU buffer | **Yes** — wrong data written | Medium (design decision needed) |
| **API-3** | Low | Only on GPU OOM at startup | No (process aborts) | Easy |
| **PREP-1** | Low | Only with concurrent host-path callers | Yes, if triggered | Easy |

**Recommended fix order:** ~~VOL-2~~ (done) → API-3 → PREP-1 → VOL-1 (requires design decision first).
