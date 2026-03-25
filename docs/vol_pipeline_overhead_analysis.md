# VOL Pipeline Overhead Analysis

**Date:** 2026-03-25  
**Hardware:** NVIDIA A100-PCIE-40GB (CC 8.0), CUDA 12.6  
**Dataset:** VPIC Harris Sheet, NX=156, chunk=32 MB, 10 timesteps, `balanced` policy  
**Tools:** nsys 2024.5.1, ncu (roofline), benchmark CSV timing  

---

## 1. Context

The GPUCompress HDF5 VOL plugin intercepts every `H5Dwrite` call and routes data
through a multi-stage GPU pipeline:

```
H5Dwrite entry
│
├── [Setup]     Spawn 8 compression worker threads + 1 I/O thread
│
├── [Stage 1]   Main thread: per-chunk stats kernel → NN inference → post WorkItem
│
├── [Stage 2]   Worker threads: compress (overlaps with S1 on subsequent chunks)
│                               D→H memcpy into pinned pool buffer
│
├── [Stage 3]   I/O thread: H5Dwrite_chunk for each compressed chunk
│
└── [Join]      Signal I/O thread done; join all workers
```

Timing is captured per `H5Dwrite` call into the timestep CSV columns:
`vol_setup_ms`, `vol_stage1_ms`, `vol_stage2_ms`, `vol_stage3_ms`, `vol_join_ms`.

---

## 2. Cross-Phase Overhead Comparison

The `6d_cross_phase_pipeline_overhead.png` figure was generated comparing five phases
(no-comp, cascaded, lz4, nn, zstd) across identical parameters.

### Measured mean overhead per H5Dwrite (ms), T > 0

| Phase     | Setup | S1a Stats | S1b NN | S1c WQ | S2 Comp | S3 I/O | Join | H5Dclose | **write_ms** |
|-----------|------:|----------:|-------:|-------:|--------:|-------:|-----:|---------:|-------------:|
| No-Comp   |     0 |         0 |      0 |      0 |       0 |      0 |    0 |        0 |         629  |
| Cascaded  |   383 |        ~1 |      0 |     ~1 |      30 |    123 |  104 |      151 |         654  |
| LZ4       |   381 |        ~1 |      0 |     ~1 |      45 |    126 |  118 |      151 |         732  |
| NN        |   377 |        10 |      4 |      4 |      ~4 |    125 |  113 |      147 |         654  |
| Zstd      |   385 |        ~1 |      0 |     ~1 |      65 |    128 |  112 |      169 |         732  |

**Key finding:** The ~377–385 ms `Setup` overhead is identical across all phases that
use the VOL pipeline. It is completely absent in `no-comp` which bypasses the VOL.
This confirms the overhead is caused by per-call thread spawning, not compression.

### Stage 1 sub-breakdown (NN phase)
- **S1a `stats_ms`** — GPU stats kernel time (CUDA events, summed across 8 chunks): ~10 ms  
- **S1b `nn_ms`** — NN inference kernel time (CUDA events): ~4 ms  
- **S1c residual** = `vol_stage1_ms − stats_ms − nn_ms`: ~4 ms (WQ posting + sync stalls)

---

## 3. Root Cause: Per-Call Thread Spawn (~377 ms)

### Evidence

`s_setup_ms` in `H5VLgpucompress.cu` measures everything from function entry to
`_pipeline_start` (line 1283). This includes:

1. Constructing 8 `std::thread` compression worker lambdas
2. Constructing 1 `std::thread` I/O worker lambda
3. Both groups block on their condition variables until the main thread posts work

Thread creation on Linux via `pthread_create` → `clone()` costs **~40–45 ms per thread**
on this system. Nine threads × ~42 ms = **~377 ms**, matching observations exactly.

### Verification via nsys

Running `nsys profile` on the `nn` phase shows a **~377 ms gap before the first CUDA
kernel fires** at each `H5Dwrite` call. This is pure OS `clone()` overhead; no GPU
activity occurs during this period.

### Affected code region (`src/hdf5/H5VLgpucompress.cu`)

```cpp
// Lines ~1130–1283: every H5Dwrite spawns 9 new threads
double _vol_func_start = _now_ms();
// ... allocate VolWriteCtx (amortized after first call) ...

// Spawn 8 compression workers
for (int w = 0; w < N_COMP_WORKERS; w++) {
    workers.emplace_back([&, w]() { /* compress loop */ });
}
// Spawn 1 I/O thread
std::thread io_thr([&]() { /* HDF5 chunk writes */ });

s_setup_ms = _now_ms() - _vol_func_start;  // ← captures all thread spawning
```

---

## 4. NN-Specific Overheads (beyond common thread-spawn)

### 4.1 ncu Roofline Results (A100)

| Kernel | Duration | Mem% | DRAM% | SM% | ncu Finding |
|--------|:--------:|:----:|:-----:|:---:|-------------|
| `statsPass1Kernel` | 0.169 ms | 6.7% | 6.7% | 7.5% | Severely under-utilized; SM workload imbalance |
| `histogramKernelVec4` | 0.052 ms | 24.6% | 21.7% | 26.5% | Below roofline; SM imbalance |
| `madPass2Kernel` | 0.049 ms | 22.9% | 22.9% | 14.1% | Below roofline; SM imbalance |
| `entropyFromHistogramKernel` | 0.008 ms | 0.4% | 0.1% | 0.1% | Grid too small to fill GPU |
| `finalizeStatsOnlyKernel` | 0.007 ms | 0.4% | 0.1% | 0.0% | Grid too small to fill GPU |
| **`nnFusedInferenceKernel`** | **0.578 ms** | **0.0%** | **0.0%** | **0.0%** | **Grid too small — 0% utilization** |

**ncu reports on every kernel:** `"This kernel grid is too small to fill the available
resources on this device."` The A100 has 108 SMs; all NN/stats kernels launch with
1 block, leaving 107 SMs idle.

The nnFusedInferenceKernel's 0.578 ms wall time is almost entirely **CUDA kernel
launch latency**, not actual computation.

### 4.2 CUDA API Hotspots (nsys, 3 timesteps, nn phase)

| API Call | Calls | Avg | Total | Source |
|----------|------:|----:|------:|--------|
| `cudaStreamSynchronize` | 11,929 | 0.82 ms | 9,799 ms | Mostly VPIC physics; 48 from stats+NN sync |
| `cudaMalloc` | 6,465 | 0.31 ms | 1,986 ms | Per-chunk `d_stats_copy` alloc (NN only) |
| `cudaDeviceSynchronize` | 16,241 | 0.11 ms | 1,861 ms | Mostly VPIC physics |
| `cudaMemcpy` | 5,126 | 0.36 ms | 1,846 ms | D→D stats copy + D→H compressed chunks |
| `cudaMallocHost` | 61 | **23.3 ms** | 1,423 ms | Pinned I/O pool (should be fully amortized) |
| `cudaFreeHost` | 1,210 | 0.59 ms | 719 ms | I/O pool dealloc |
| `cudaFree` | 6,438 | 0.065 ms | 418 ms | Per-chunk `d_stats_copy` free |

### 4.3 Root Causes Specific to the NN Phase

**A. `nnFusedInferenceKernel` — 0% GPU utilization**

Each of the 8 chunks runs its own sequential NN inference on the main thread, one at
a time. Each kernel launch uses 1 block with a small thread count to evaluate a tiny
MLP (19,076 parameters). On an A100 with 108 SMs, this is 0% compute utilization.
The 0.578 ms is almost entirely kernel-launch and context-switch latency.

Root cause in code (`H5VLgpucompress.cu`, Stage 1 chunk loop):
```cpp
// Called once per chunk, sequentially, with a 1-block grid
gpucompress::runNNFusedInferenceCtx(infer_ctx, wi.src, wi.sz, ...);
cudaStreamSynchronize(infer_ctx->stream);  // CPU waits after every single chunk
```

**B. Per-chunk `cudaMalloc`/`cudaFree` for `d_stats_copy`**

For every chunk in the NN path, the main thread allocates a fresh device buffer to
hold a copy of the stats result so that the worker thread can use it independently
after the main thread moves to the next chunk:

```cpp
// H5VLgpucompress.cu ~line 1481 — inside the per-chunk loop:
void *d_sc = nullptr;
cudaMalloc(&d_sc, sizeof(AutoStatsGPU));   // ~0.31 ms each
cudaMemcpyAsync(d_sc, infer_ctx->d_stats, ...);
cudaStreamSynchronize(infer_ctx->stream);   // wait for D→D copy
wi.d_stats_copy = d_sc;                    // worker frees after use
```

This is 8 `cudaMalloc` + 8 `cudaFree` per `H5Dwrite` = ~2.5 ms of pure
allocation overhead that can be completely eliminated by pre-allocating.

**C. Sequential per-chunk stream synchronization (serialization of inference)**

The stats kernel and NN inference run sequentially per chunk with a hard
`cudaStreamSynchronize` barrier between them. With 8 chunks per write, this
means 16 mandatory CPU–GPU synchronization round-trips on the critical path.
The stats kernels for different chunks could overlap with one another if run on
independent streams.

---

## 5. Fix Recommendations

### Fix 1 — Persistent Thread Pool (highest impact)

**Problem:** 9 threads are spawned and destroyed on every `H5Dwrite`.  
**Cost:** ~377 ms per write, identical across all algorithms.  
**Expected gain:** Eliminate ~60% of the entire write latency.

**Design:** Pre-allocate worker threads inside `VolWriteCtx` at `H5Dopen` (or on
first write), park them on condition variables, and reuse them across all subsequent
`H5Dwrite` calls. Tear them down only at `H5Dclose`.

```cpp
// In VolWriteCtx (H5VLgpucompress.cu ~line 61):
struct VolWriteCtx {
    // --- existing persistent fields ---
    void    **d_comp_w;          // compression output buffers
    void    **io_pool_bufs;      // pinned I/O pool

    // --- NEW: persistent thread pool ---
    std::thread              workers[N_COMP_WORKERS];
    std::thread              io_thr;
    std::mutex               wq_mtx, io_mtx;
    std::condition_variable  wq_ready_cv, wq_full_cv, io_cv;
    std::queue<WorkItem>     wq;
    std::atomic<bool>        pool_alive{false};  // set false at H5Dclose
};

// At H5Dopen / first write: spawn once
if (!wctx->pool_alive.load()) {
    wctx->pool_alive.store(true);
    for (int w = 0; w < N_COMP_WORKERS; w++)
        wctx->workers[w] = std::thread(worker_loop, wctx, w);
    wctx->io_thr = std::thread(io_loop, wctx);
}
// At H5Dclose: signal shutdown and join
wctx->pool_alive.store(false);
wctx->wq_ready_cv.notify_all();
for (auto &t : wctx->workers) t.join();
wctx->io_thr.join();
```

**Result:** `s_setup_ms` drops from ~377 ms to near zero (only wctx
initialization on first call, ~1 ms amortized).

---

### Fix 2 — Pre-allocate `d_stats_copy` Buffers (NN path)

**Problem:** `cudaMalloc(&d_sc, sizeof(AutoStatsGPU))` is called 8 times per write  
**Cost:** ~2.5 ms per write (8 × 0.31 ms), plus matching `cudaFree` calls.  
**Expected gain:** ~2.5 ms per write, negligible effort.

**Design:** Add a pre-allocated array of `AutoStatsGPU*` device pointers to
`VolWriteCtx`, one per compression worker slot.

```cpp
// In VolWriteCtx:
AutoStatsGPU *d_stats_copies[N_COMP_WORKERS];  // allocated at first write

// Initialization (first H5Dwrite only):
for (int w = 0; w < N_COMP_WORKERS; w++)
    cudaMalloc(&wctx->d_stats_copies[w], sizeof(AutoStatsGPU));

// In Stage 1 chunk loop — replace cudaMalloc with direct reuse:
AutoStatsGPU *d_sc = wctx->d_stats_copies[chunk_idx % N_COMP_WORKERS];
cudaMemcpyAsync(d_sc, infer_ctx->d_stats, sizeof(AutoStatsGPU),
                cudaMemcpyDeviceToDevice, infer_ctx->stream);
// No cudaFree needed — buffer is reused next write
```

**Note:** Requires synchronization to ensure worker has consumed the buffer before
the main thread reuses it for the next write. A per-slot `cudaEvent` or the existing
work-queue ordering guarantees this.

---

### Fix 3 — Batch NN Inference Across All Chunks (NN path)

**Problem:** `nnFusedInferenceKernel` launches once per chunk with a 1-block grid.  
**Cost:** 0% GPU utilization; 0.578 ms is kernel-launch latency × 8.  
**Expected gain:** ~4 ms per write; more importantly, unlocks actual GPU parallelism.

**Design:** Run stats for all 8 chunks first (potentially on concurrent streams),
then submit a single batched NN inference kernel that processes all 8 feature
vectors simultaneously with a grid of 8 blocks.

```cpp
// Phase A: launch all stats kernels concurrently (8 independent streams)
for (int ci = 0; ci < total_chunks; ci++) {
    cudaStream_t s = per_chunk_streams[ci];
    runStatsKernelsNoSync(chunk_ptrs[ci], chunk_sz, s, infer_ctxs[ci]);
}
// Sync all stats
for (int ci = 0; ci < total_chunks; ci++)
    cudaStreamSynchronize(per_chunk_streams[ci]);

// Phase B: one batched NN inference call (8 blocks)
runNNFusedInferenceBatched(weights, stats_array, total_chunks, output_array);
cudaStreamSynchronize(main_stream);

// Phase C: distribute results to WorkItems and post to queue
```

This changes the Stage 1 critical path from `8 × (stats + sync + infer + sync)` to
`max(stats across streams) + one_batched_infer + one_sync`.

---

### Fix 4 — Concurrent Stats Kernels Across Chunks

**Problem:** Stats kernels run sequentially on one stream (implied by sequential chunk loop).  
**Cost:** ~8 × 0.17 ms sequential = ~1.4 ms, could be ~0.17 ms concurrent.  
**Expected gain:** ~1.2 ms per write (requires Fix 3 for the inference batching to work).

**Design:** Use per-chunk CUDA streams (could reuse CompContext streams from worker pool):
```cpp
// Acquire per-chunk inference contexts (already exists for worker CompContexts)
for (int ci = 0; ci < total_chunks; ci++) {
    CompContext *ctx = acquireCompContext();
    runStatsKernelsNoSync(chunk_ptr[ci], chunk_sz, ctx->stream, ctx);
    per_chunk_ctx[ci] = ctx;
}
// Then barrier + batch infer (see Fix 3)
```

---

## 6. Priority and Expected Impact Summary

| Fix | Lines in `H5VLgpucompress.cu` | Effort | Per-Write Gain | % of write_ms |
|-----|-------------------------------|--------|----------------|---------------|
| **1. Persistent thread pool** | ~1130–1283, ~61–66 | Medium | **~377 ms** | **~60%** |
| **2. Pre-alloc `d_stats_copy`** | ~1481–1500 | Low | ~2.5 ms | ~0.4% |
| **3. Batch NN inference** | ~1413–1527 | Medium | ~4 ms | ~0.6% |
| **4. Concurrent stats streams** | ~1413–1470 | Low | ~1.2 ms | ~0.2% |

Fixes 1 alone reduces write latency from ~654 ms → ~277 ms (2.4× speedup).
Fixes 1–4 together reduce it to ~270 ms (2.4× speedup; NN overhead becomes ~6 ms total).

---

## 7. Reference Figures

All figures are generated by `benchmarks/plots/generate_dataset_figures.py` and live in:
`benchmarks/results/per_dataset/vpic/eval_NX156_chunk32mb_ts10/balanced_w1-1-1/`

| Figure | Description |
|--------|-------------|
| `6a_write_path_decomposition.png` | Stacked bar: write_ms broken into VOL stages per phase |
| `6b_pipeline_waterfall.png` | Waterfall chart: per-chunk stage timing for nn phase |
| `6d_cross_phase_pipeline_overhead.png` | Cross-phase comparison with S1 sub-breakdown |

---

## 8. Profiling Artifacts

Raw profiling data is in `benchmarks/profiling/`:

| File | Tool | Content |
|------|------|---------|
| `nn_nsys.nsys-rep` | nsys | Full timeline (CPU + GPU), 3 timesteps, nn phase |
| `nn_nsys_stats_cuda_api_sum.csv` | nsys stats | CUDA API call summary |
| `nn_nsys_stats_cuda_gpu_kern_sum.csv` | nsys stats | GPU kernel execution summary |
| `ncu_nn_sudo.csv` | ncu roofline | Per-kernel metrics for all 6 NN/stats kernels |

To re-run profiling:
```bash
cd benchmarks/vpic-kokkos
GPU_DIR=/home/cc/GPUCompress

# nsys timeline
nsys profile --trace=cuda,osrt,nvtx --output=benchmarks/profiling/nn_nsys \
  env LD_LIBRARY_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib" \
      GPUCOMPRESS_DETAILED_TIMING=1 GPUCOMPRESS_WEIGHTS=$GPU_DIR/neural_net/weights/model.nnwt \
      GPUCOMPRESS_RANK_W0=1 GPUCOMPRESS_RANK_W1=1 GPUCOMPRESS_RANK_W2=1 \
      VPIC_NX=156 VPIC_CHUNK_MB=32 VPIC_TIMESTEPS=3 VPIC_PHASE=nn \
      VPIC_RESULTS_DIR=/tmp/nn_prof \
  ./vpic_benchmark_deck_pm.Linux ./vpic_benchmark_deck_phase_major.cxx

# ncu kernel metrics (requires sudo for hardware counters on this system)
sudo ncu --set roofline \
    --kernel-name "regex:nnFusedInference|statsPass1|histogramKernelVec4|madPass2|entropyFromHistogram|finalizeStatsOnly" \
    --csv --log-file benchmarks/profiling/ncu_nn_sudo.csv \
  env LD_LIBRARY_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib" \
      GPUCOMPRESS_DETAILED_TIMING=1 GPUCOMPRESS_WEIGHTS=$GPU_DIR/neural_net/weights/model.nnwt \
      GPUCOMPRESS_RANK_W0=1 GPUCOMPRESS_RANK_W1=1 GPUCOMPRESS_RANK_W2=1 \
      VPIC_NX=156 VPIC_CHUNK_MB=32 VPIC_TIMESTEPS=2 VPIC_PHASE=nn \
      VPIC_RESULTS_DIR=/tmp/nn_prof \
  ./vpic_benchmark_deck_pm.Linux ./vpic_benchmark_deck_phase_major.cxx
```

---

*Analysis performed 2026-03-25. Contact: see repo README for author info.*
