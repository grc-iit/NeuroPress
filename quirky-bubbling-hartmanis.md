# Parallelization Plan: GPU Compression Pipeline

## Context

The current `gpu_aware_chunked_write` and `gpu_aware_chunked_read` loops in the VOL connector
are fully serial: every chunk goes through **compress → D→H → HDF5 write** with no overlap.
The GPU sits idle during disk I/O. The CPU blocks on 4–5 `cudaStreamSynchronize` calls per
chunk inside `gpucompress_compress_gpu`, several of which are unconditional even when no
preprocessing or stats output is needed. The benchmark (8MB, 2 chunks, lossless) shows
116ms wall-clock for what should be ~50ms of GPU work.

There are **four independent improvements**, ordered by ROI and risk:

---

## Change 1 — Remove Superfluous Sync Barriers in `gpucompress_compress_gpu`

**File**: `src/api/gpucompress_api.cpp`

### 1a. Unconditional preprocessing sync (line 1585)

```cpp
// CURRENT — always fires, even when preproc_to_use == 0
cudaError_t cuda_err = cudaStreamSynchronize(stream);
```

All subsequent operations (createCompressionManager, configure_compression, compress) are
submitted to the same `stream`, so GPU ordering is already guaranteed by stream semantics.
The sync only serves to block the CPU — it is not needed for correctness.

**Fix**: Guard with a preprocessing-was-applied check:
```cpp
// Only sync when preprocessing actually ran (shuffle/quant allocated GPU buffers)
cudaError_t cuda_err = (d_quantized || d_shuffled)
    ? cudaStreamSynchronize(stream)
    : cudaSuccess;
```

### 1b. Unconditional timing event sync (line 1643)

```cpp
// CURRENT — always records and syncs even when nobody reads the result
if (timing_ok) {
    cudaEventRecord(g_t_stop, stream);
    cudaEventSynchronize(g_t_stop);          // blocks CPU ← remove when unneeded
    cudaEventElapsedTime(&primary_comp_time_ms, g_t_start, g_t_stop);
}
```

`primary_comp_time_ms` is only used inside the exploration block which is gated on
`g_online_learning_enabled` (line 1699). When both `stats == NULL` and online learning is
off, the timing result is discarded.

**Fix**:
```cpp
bool need_timing = timing_ok && (stats != nullptr || g_online_learning_enabled);
if (need_timing) cudaEventRecord(g_t_start, stream);
// ... compress ...
if (need_timing) {
    cudaEventRecord(g_t_stop, stream);
    cudaEventSynchronize(g_t_stop);
    cudaEventElapsedTime(&primary_comp_time_ms, g_t_start, g_t_stop);
}
```

**Net effect**: Removes 2 of the 5 per-chunk sync barriers in the normal (no-stats, no-OL) path.
No API changes. Safe for the same-stream explore path (gated).

---

## Change 2 — VOL Write Loop: Double-Buffer + I/O Thread Pipeline

**File**: `src/hdf5/H5VLgpucompress.cu`, function `gpu_aware_chunked_write` (~line 930)

### Current serial critical path per chunk (ms budget)

```
[GPU: stats+NN+nvcomp compress]  ← ~50–100ms (blocks CPU on internal syncs)
[PCIe: D→H compressed bytes]     ← ~0.2ms (2MB compressed at 10 GB/s, currently blocking)
[CPU: write_chunk_to_native]      ← ~30–80ms (HDF5 file I/O, blocking)
─────────────────────────────────────────────────────
Total per chunk ≈ sum of all three = ~130–260ms
```

### Target pipeline (double-buffer, N_SLOTS=2)

```
Time →
Main thread: [compress N]       [compress N+1]     [compress N+2]
xfer stream:         [D→H N]           [D→H N+1]          [D→H N+2]
I/O thread:                   [write N]      [write N+1]       [write N+2]
─────────────────────────────────────────────────────────────────────────
Steady-state per chunk ≈ max(compress_time, write_time)   ← GPU and disk overlap
```

### New data structures to add (inside `gpu_aware_chunked_write`)

```c
#define N_SLOTS 2

// Per-slot buffers (allocated once before loop, freed after)
uint8_t  *d_compressed[N_SLOTS];
void     *h_compressed[N_SLOTS];   // pinned
size_t    comp_sizes[N_SLOTS];
hsize_t   slot_chunk_start[N_SLOTS][32];  // save N-D chunk coordinates per slot

// New CUDA stream for D→H transfers (separate from comp stream)
cudaStream_t xfer_stream;
cudaStreamCreate(&xfer_stream);

// Per-slot event to signal "D→H complete for this slot"
cudaEvent_t  xfer_done[N_SLOTS];
for (int s = 0; s < N_SLOTS; s++) cudaEventCreate(&xfer_done[s]);

// I/O worker state
struct IOItem { void *h_buf; size_t comp_sz; hsize_t chunk_start[32]; int ndims; };
std::mutex              io_mutex;
std::condition_variable io_cv;
std::queue<IOItem>      io_queue;
bool                    io_done = false;
herr_t                  io_ret  = 0;    // I/O thread error capture
std::thread io_thread([&]() {
    while (true) {
        IOItem item;
        { std::unique_lock<std::mutex> lk(io_mutex);
          io_cv.wait(lk, [&]{ return !io_queue.empty() || io_done; });
          if (io_queue.empty()) break;
          item = io_queue.front(); io_queue.pop(); }
        herr_t r = write_chunk_to_native(o->under_object, o->under_vol_id,
                                          item.chunk_start, item.h_buf, item.comp_sz, dxpl_id);
        if (r < 0) { std::lock_guard<std::mutex> lk(io_mutex); io_ret = r; }
    }
});
```

### Rewritten chunk loop

```c
ret = 0;
s_gpu_writes++;

for (size_t ci = 0; ci < total_chunks; ci++) {
    int slot = (int)(ci % N_SLOTS);

    // --- Compute chunk coords (same logic as before) ---
    hsize_t chunk_start[32], actual_chunk[32];
    size_t actual_bytes;
    /* ... identical index math ... */

    // --- GPU compression: stats + NN + nvcomp → d_compressed[slot] ---
    // (uses g_default_stream internally; blocks CPU via internal syncs on return)
    size_t comp_size = max_comp;
    gpucompress_error_t ce = gpucompress_compress_gpu(
        src_ptr, actual_bytes, d_compressed[slot],
        &comp_size, &cfg, NULL, NULL);
    if (ce != GPUCOMPRESS_SUCCESS) { ret = -1; break; }
    comp_sizes[slot] = comp_size;
    memcpy(slot_chunk_start[slot], chunk_start, (size_t)ndims * sizeof(hsize_t));

    // --- Async D→H: compressed bytes only (on xfer_stream) ---
    // xfer_stream is independent of g_default_stream → overlaps with next compress
    cudaMemcpyAsync(h_compressed[slot], d_compressed[slot],
                    comp_size, cudaMemcpyDeviceToHost, xfer_stream);
    cudaEventRecord(xfer_done[slot], xfer_stream);

    // --- Hand off previous slot to I/O thread once its D→H is confirmed done ---
    if (ci >= N_SLOTS) {
        int done_slot = (int)((ci - N_SLOTS + 1) % N_SLOTS);  // oldest in-flight slot
        cudaEventSynchronize(xfer_done[done_slot]);  // fast: D→H of tiny compressed bytes
        { std::lock_guard<std::mutex> lk(io_mutex);
          io_queue.push({h_compressed[done_slot], comp_sizes[done_slot],
                         {}, ndims});
          memcpy(io_queue.back().chunk_start, slot_chunk_start[done_slot],
                 (size_t)ndims * sizeof(hsize_t));
        }
        io_cv.notify_one();
    }
}

// Flush remaining in-flight slots (up to N_SLOTS-1)
for (int lag = (int)std::min((size_t)(N_SLOTS-1), total_chunks-1); lag >= 0 && ret == 0; lag--) {
    size_t ci_done = total_chunks - 1 - lag;
    int slot = (int)(ci_done % N_SLOTS);
    cudaEventSynchronize(xfer_done[slot]);
    { std::lock_guard<std::mutex> lk(io_mutex);
      io_queue.push({h_compressed[slot], comp_sizes[slot], {}, ndims});
      memcpy(io_queue.back().chunk_start, slot_chunk_start[slot],
             (size_t)ndims * sizeof(hsize_t));
    }
    io_cv.notify_one();
}

// Signal I/O thread to drain and join
{ std::lock_guard<std::mutex> lk(io_mutex); io_done = true; }
io_cv.notify_all();
io_thread.join();

if (io_ret < 0) ret = -1;
```

### Buffer allocation/cleanup changes

Replace the current single-buffer allocs (lines 985–993) with per-slot:
```c
// Allocate
for (int s = 0; s < N_SLOTS; s++) {
    if (cudaMallocHost(&h_compressed[s], max_comp) != cudaSuccess) goto done;
    if (cudaMalloc(&d_compressed[s], max_comp) != cudaSuccess) goto done;
}
// Cleanup
for (int s = 0; s < N_SLOTS; s++) {
    cudaFree(d_compressed[s]);
    cudaFreeHost(h_compressed[s]);
    cudaEventDestroy(xfer_done[s]);
}
cudaStreamDestroy(xfer_stream);
```

### Safety considerations

- `gpucompress_compress_gpu` uses `g_default_stream` (or a passed stream). The xfer_stream
  is a DIFFERENT stream — CUDA guarantees they execute independently and can overlap on
  hardware with copy engines.
- `g_d_stats_workspace` is global but only one compression runs at a time (main thread is
  serial). No race condition.
- `d_fused_infer_output` is global but same reasoning — single-threaded main loop.
- The I/O thread only calls `write_chunk_to_native` (pure HDF5/CPU, no CUDA) so no CUDA
  context issues.
- Error from I/O thread is captured in `io_ret` and propagated after join.
- If ret==-1 in the compress loop, the io_done flag still gets set and thread still joins
  cleanly (queue drain).

---

## Change 3 — VOL Read Loop: I/O Prefetch Thread Pipeline

**File**: `src/hdf5/H5VLgpucompress.cu`, function `gpu_aware_chunked_read` (~line 1180)

### Current serial critical path per chunk

```
[CPU: read_chunk_from_native]   ← disk I/O blocking (HDF5 chunk read to h_compressed)
[PCIe: H→D compressed bytes]   ← blocking cudaMemcpy
[GPU: gpucompress_decompress]   ← GPU work + internal syncs
```

### Target pipeline (prefetch reader thread)

```
Prefetch thread: [read chunk N] [read chunk N+1] [read chunk N+2]
Main thread:               [H→D+decomp N]  [H→D+decomp N+1]  [H→D+decomp N+2]
```

### Design

```c
// Prefetch thread fills h_comp[slot] with compressed bytes from disk
// Main thread consumes h_comp[slot], does H→D + decompress
struct PrefetchItem { size_t comp_sz; uint32_t filter_mask; int slot; herr_t status; };
std::queue<PrefetchItem> prefetch_queue;  // ready slots
std::mutex               pre_mutex;
std::condition_variable  pre_cv;
// Semaphore-style: allow at most N_SLOTS outstanding reads
std::counting_semaphore<N_SLOTS> free_slots(N_SLOTS);  // or use simple int + condvar

std::thread prefetch_thread([&]() {
    for (size_t ci = 0; ci < total_chunks; ci++) {
        free_slots.acquire();          // wait for a free h_comp slot
        int slot = (int)(ci % N_SLOTS);
        size_t comp_size = max_comp;
        uint32_t filter_mask = 0;
        /* compute chunk_start for ci ... */
        herr_t r = read_chunk_from_native(o->under_object, o->under_vol_id,
                                           chunk_start, h_compressed[slot],
                                           &comp_size, &filter_mask, dxpl_id);
        { std::lock_guard<std::mutex> lk(pre_mutex);
          prefetch_queue.push({comp_size, filter_mask, slot, r}); }
        pre_cv.notify_one();
    }
});

// Main thread: consume prefetched chunks
for (size_t ci = 0; ci < total_chunks; ci++) {
    PrefetchItem item;
    { std::unique_lock<std::mutex> lk(pre_mutex);
      pre_cv.wait(lk, [&]{ return !prefetch_queue.empty(); });
      item = prefetch_queue.front(); prefetch_queue.pop(); }
    if (item.status < 0) { ret = -1; break; }

    // H→D compressed bytes (can be async if xfer_stream used)
    vol_memcpy(d_compressed, h_compressed[item.slot], item.comp_sz, cudaMemcpyHostToDevice);
    // Decompress on GPU
    gpucompress_decompress_gpu(d_compressed, item.comp_sz, dst_ptr, &actual_bytes, NULL, NULL);
    // Release slot back to prefetch thread
    free_slots.release();
}
prefetch_thread.join();
```

Note: `std::counting_semaphore` requires C++20. Fallback: use `int free_count + mutex + condvar`.

---

## Change 4 — Replace `cudaDeviceSynchronize` with Stream Sync

**Files**: `src/hdf5/H5VLgpucompress.cu`

In the non-contiguous chunk paths, both write and read loops use `cudaDeviceSynchronize()`
after `gather_chunk_kernel` / `scatter_chunk_kernel`. This blocks ALL CUDA streams including
any background work.

**Write path (line 1138)**:
```cpp
// CURRENT:
if (cudaGetLastError() != cudaSuccess ||
    cudaDeviceSynchronize() != cudaSuccess) { ret = -1; break; }
// FIX:
if (cudaGetLastError() != cudaSuccess ||
    cudaStreamSynchronize(cudaStreamDefault) != cudaSuccess) { ret = -1; break; }
```

**Read path (line 1362)**:
```cpp
// CURRENT:
cudaDeviceSynchronize();
// FIX:
cudaStreamSynchronize(cudaStreamDefault);
```

This ensures xfer_stream / prefetch D→H traffic is not interrupted.

---

## Non-Goals (Out of Scope)

- **Multi-threaded concurrent compression**: Requires per-thread stats workspace, per-thread
  NN inference buffers, and either per-thread weight copies or locked SGD — too invasive.
- **SGD parallelism**: SGD is currently disabled in the benchmark; not relevant.
- **nvcomp manager reuse**: nvcomp managers are not designed for reuse across different
  input data; do not attempt.
- **CUDA graph capture**: Would require major API refactoring; deferred.

---

## File Modification Summary

| File | Change |
|------|--------|
| `src/api/gpucompress_api.cpp` | Change 1a: guard preprocessing sync; 1b: guard timing events |
| `src/hdf5/H5VLgpucompress.cu` | Change 2: double-buffer + I/O thread write loop |
| `src/hdf5/H5VLgpucompress.cu` | Change 3: prefetch thread read loop |
| `src/hdf5/H5VLgpucompress.cu` | Change 4: replace cudaDeviceSynchronize x2 |

Add `#include <thread>`, `#include <mutex>`, `#include <condition_variable>`, `#include <queue>`
to `H5VLgpucompress.cu`.

---

## Verification & Performance Tracking

### Baseline capture (before any changes)

Before implementing any change, run the benchmark and record the output in
`tests/benchmark_gpu_resident_results/baseline.txt`:

```bash
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
  ./build/benchmark_gpu_resident neural_net/weights/model.nnwt \
  --dataset-mb 8 --chunk-mb 4 2>/dev/null | tee tests/benchmark_gpu_resident_results/baseline.txt
```

### After each change

Rebuild, re-run benchmark, save results to a numbered file, compare:

```bash
cmake --build build -j$(nproc) --target benchmark_gpu_resident 2>&1 | tail -3

LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
  ./build/benchmark_gpu_resident neural_net/weights/model.nnwt \
  --dataset-mb 8 --chunk-mb 4 2>/dev/null \
  | tee tests/benchmark_gpu_resident_results/after_changeN.txt

diff tests/benchmark_gpu_resident_results/baseline.txt \
     tests/benchmark_gpu_resident_results/after_changeN.txt
```

Changes to track per run:
- Write MB/s (Phase 2 NN) — primary metric
- Read MB/s (Phase 2 NN)
- Write ms / Read ms
- **Lossless verify: PASS** — correctness gate; if this fails, rollback the change

### Larger dataset test (pipeline steady-state)

After each change also run with 64 MB / 4 MB chunks (16 chunks) to see whether the
pipeline overhead amortizes to a real speedup:

```bash
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
  ./build/benchmark_gpu_resident neural_net/weights/model.nnwt \
  --dataset-mb 64 --chunk-mb 4 2>/dev/null
```

### Full regression suite

```bash
./build/test_hdf5_configs            # 120 tests must pass
./build/test_quantization_roundtrip  # 6 tests must pass
```

These must stay green after every change.
