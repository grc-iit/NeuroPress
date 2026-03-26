# Pipeline Timing Analysis

**Date:** 2026-03-26
**Hardware:** NVIDIA A100-PCIE-40GB (CC 8.0), CUDA 12.6
**Reference dataset:** Gray-Scott L=406, 128 MB chunks, 10 timesteps, `ratio_only` policy, NN Inference phase
**Grid:** 406^3 = 255.3 MB, 2 chunks x ~128 MB each

---

## 1. Timing Model Overview

The write pipeline is instrumented at three levels, each measuring different things:

### Tier A: Benchmark wall clock (external)

Consecutive `now_ms()` slices around HDF5 API calls in the benchmark driver.
These are **strictly additive** by construction: `write_ms = h5dwrite_ms + cuda_sync_ms + h5dclose_ms + h5fclose_ms`.

| Column | What it measures |
|--------|-----------------|
| `write_ms` | Full write path: H5Dwrite through H5Fclose |
| `h5dwrite_ms` | Time inside H5Dwrite (triggers entire VOL pipeline) |
| `cuda_sync_ms` | cudaDeviceSynchronize after H5Dwrite (redundant — pipeline already synced) |
| `h5dclose_ms` | H5Dclose (metadata flush) |
| `h5fclose_ms` | H5Fclose (file close) |

### Tier B: VOL pipeline wall clock (internal)

Wall-clock measurements inside `gpu_aware_chunked_write`. The pipeline total decomposes into
three **strictly additive** sequential phases, enforced by a debug assertion:

```
vol_pipeline_ms = vol_stage1_ms + vol_drain_ms + vol_io_drain_ms   (exact)
h5dwrite_ms ≈ vol_setup_ms + vol_pipeline_ms                       (0.1% residual from HDF5 framework)
```

| Column | What it measures |
|--------|-----------------|
| `vol_setup_ms` | Function entry to pipeline start (VolWriteCtx alloc, thread spawn on first call, buffer reuse check) |
| `vol_stage1_ms` | S1: sequential per-chunk stats + NN inference + work queue posting (main thread) |
| `vol_drain_ms` | Worker drain: S1 end to all 8 compression workers joined |
| `vol_io_drain_ms` | I/O drain: workers joined to I/O thread joined (includes ring buffer cleanup) |
| `vol_pipeline_ms` | Pipeline start to I/O thread joined = stage1 + drain + io_drain |

### Tier C: GPU component times (CUDA events)

Per-chunk CUDA event times (`cudaEventElapsedTime`), **summed across all chunks** in the timestep.
These measure pure GPU kernel execution and are NOT wall-clock pipeline times.

| Column | What it measures |
|--------|-----------------|
| `stats_ms` | Stats kernel execution (sum across chunks) |
| `nn_ms` | NN inference kernel execution (sum across chunks) |
| `comp_ms` | Compression kernel execution, unclamped (sum across chunks) |
| `decomp_ms` | Decompression kernel execution, unclamped (sum across chunks, from read path) |

### Tier D: Worker busy times (bottleneck analysis)

Non-additive wall-clock metrics for identifying pipeline bottlenecks.

| Column | What it measures |
|--------|-----------------|
| `vol_s2_busy_ms` | Max single-worker cumulative wall time (compress + cudaFree + pool acquire + D2H + I/O post) |
| `vol_s3_busy_ms` | I/O thread total write time (serial H5Dwrite_chunk calls, accumulated) |

---

## 2. Pipeline Architecture

```
H5Dwrite entry
│
├── [Setup]  VolWriteCtx reuse / alloc, thread spawn (first call only)
│            vol_setup_ms: ~0.6ms steady state, ~1500ms first call
│
├── [Stage 1]  Main thread — SEQUENTIAL per chunk:
│              for each chunk:
│                1. Compute chunk coordinates
│                2. Run stats kernel         → cudaStreamSynchronize
│                3. Run NN inference kernel   → cudaStreamSynchronize
│                4. Copy d_stats to ring buf  → cudaStreamSynchronize
│                5. Post WorkItem to worker queue
│              vol_stage1_ms: ~7ms per chunk (1.1ms GPU + 5.9ms sync overhead)
│
├── [Workers]  8 compression worker threads (PARALLEL, start during S1):
│              dequeue WorkItem → compress → cudaFree → pool acquire → D2H → post to I/O queue
│              vol_s2_busy_ms: max worker wall time
│
├── [I/O]      1 I/O thread (PARALLEL, starts during S1):
│              dequeue compressed chunk → H5Dwrite_chunk to native HDF5
│              vol_s3_busy_ms: total write time
│
├── [Drain]    S1 end → all workers joined
│              vol_drain_ms: wall clock of remaining worker tail after S1 completes
│
└── [I/O Drain]  Workers joined → I/O thread joined
                 vol_io_drain_ms: final I/O flush + ring buffer cleanup
```

### Why S1 is sequential

The main thread processes chunks one-at-a-time because:

1. **Single shared `infer_ctx`**: One CUDA stream, one set of NN weight buffers. The `d_stats` output
   buffer is reused for the next chunk, so it must be copied to a ring buffer before advancing.
2. **Host readback required**: The NN output (action, predicted ratio/comp/decomp times) must be
   read back to host memory to populate the WorkItem before posting to the worker queue.
3. **SGD weight consistency**: When online learning is enabled, each chunk must see the latest
   SGD-updated weights from workers that finished previous chunks.

Each chunk requires 3 `cudaStreamSynchronize` calls on the critical path, adding ~6ms of
host-GPU round-trip latency per chunk despite only ~1.1ms of actual GPU kernel time.

---

## 3. Steady-State Breakdown (T>=1, NN Inference, 2 chunks)

| Stage | Time (ms) | % of Pipeline | Notes |
|-------|----------|---------------|-------|
| Setup | 0.6 | 1.5% | Buffer reuse check only (no allocation) |
| **S1: Inference** | **14.2** | **35.9%** | 2 chunks x 7.1ms each (sequential) |
| -- stats kernel | 1.4 | | CUDA event, summed across 2 chunks |
| -- NN kernel | 0.9 | | CUDA event, summed across 2 chunks |
| -- sync overhead | 11.9 | | 84% of S1 is cudaStreamSynchronize |
| **Worker Drain** | **22.7** | **57.3%** | Tail of parallel worker execution after S1 ends |
| I/O Drain | 2.7 | 6.8% | Final I/O flush + ring cleanup |
| **Pipeline Total** | **~40 ms** | 100% | |
| cuda_sync + h5dclose + h5fclose | 0.1 | | Negligible post-pipeline overhead |

### Cold start (T=0)

| Component | T=0 (ms) | T>=1 mean (ms) |
|-----------|----------|----------------|
| vol_setup_ms | 1497 | 0.6 |
| vol_pipeline_ms | 42.6 | 39.6 |
| **write_ms** | **1540** | **40.3** |

T=0 pays a one-time cost for thread spawning (9 threads x ~42ms each via `pthread_create`/`clone()`)
and initial buffer allocation (8 device compression buffers, 16 pinned host staging buffers).
This amortizes to negligible over 100+ timesteps.

---

## 4. Understanding the Apparent comp_ms vs drain_ms Discrepancy

The GPU breakdown plot shows ~40ms of compression time, but the pipeline waterfall shows only
~23ms of worker drain. This is expected — they measure fundamentally different things.

### The three related metrics

| Metric | Value (ms) | What it measures |
|--------|-----------|-----------------|
| `comp_ms` | 38.4 | Sum of CUDA event kernel times across 2 chunks |
| `s2_busy_ms` | 32.4 | Max single-worker wall clock (bottleneck worker) |
| `drain_ms` | 22.7 | Wall clock from S1 end to workers joined (tail only) |

### Why they differ

```
Timeline (2 chunks, 8 workers):

           S1 (14.2ms)              Drain (22.7ms)     IO Drain
    |════════════════════════|═══════════════════════════|════|
    |                        |                           |    |
    | chunk0: stats+NN (7ms) |                           |    |
    |   ↓ post to WQ         |                           |    |
    | chunk1: stats+NN (7ms) |                           |    |
    |                        |                           |    |
    |   Worker0: ←── s2_busy (32.4ms) ──────────────→|  |    |
    |        [compress chunk0 (~20ms) + D2H + overhead]  |    |
    |                                                    |    |
    |          Worker1: [compress chunk1 + D2H + overhead]|   |
    |                                                         |
    |←── overlap (9.7ms) ──→|                                 |
```

- **`comp_ms` (38.4ms)**: Sum of per-chunk CUDA compression kernel times. Two chunks each ~20ms
  = ~40ms aggregate. This is serialized GPU time even though workers overlap in wall clock.
- **`s2_busy_ms` (32.4ms)**: The bottleneck worker's total lifetime including compression kernel
  + cudaFree + pool acquire + D2H copy + I/O queue posting. Exceeds pure kernel time by ~10ms.
- **`drain_ms` (22.7ms)**: Only the **tail** after S1 ends. Workers started ~10ms earlier (during S1),
  so `drain_ms ≈ s2_busy_ms - overlap_with_S1 ≈ 32.4 - 9.7 = 22.7ms`.

### Expected relationships

- `comp_ms >= drain_ms` — sum of parallel work exceeds wall clock of its tail
- `s2_busy_ms > drain_ms` — worker starts during S1, drain only captures the tail
- `s2_busy_ms > comp_ms / n_chunks` — worker wall time includes non-kernel overhead
- `s2_busy_ms ≈ drain_ms + overlap_with_S1`

---

## 5. Parallelism Assessment (2 chunks, 8 workers)

With only 2 chunks and 8 workers, 6 workers are always idle. However, the pipeline benefits
from concurrency between S1 and the workers:

| Scenario | Total time |
|----------|-----------|
| No pipeline overlap: S1 + max_worker + IO | 14.2 + 32.4 + 2.7 = **49.3 ms** |
| Actual (pipeline overlap) | **39.6 ms** |
| Savings from overlap | **9.7 ms (20%)** |

The worker is the bottleneck: `s2_busy_ms` (32.4ms) >> S1 per-chunk time (7.1ms).
S1 finishes posting chunk 1 long before worker 0 finishes chunk 0.

With more chunks (smaller chunk size), the S1-worker overlap would increase and more workers
would be utilized, improving the parallelism benefit.

---

## 6. Timing Column Reference

### benchmark_*_timesteps.csv

| Column | Units | Clock type | Scope | Additive group |
|--------|-------|-----------|-------|----------------|
| `write_ms` | ms | Wall | Full write path | = h5dwrite + cuda_sync + h5dclose + h5fclose |
| `h5dwrite_ms` | ms | Wall | H5Dwrite call | ≈ vol_setup + vol_pipeline |
| `cuda_sync_ms` | ms | Wall | Post-write sync | Typically ~0 (redundant) |
| `h5dclose_ms` | ms | Wall | Metadata flush | |
| `h5fclose_ms` | ms | Wall | File close | |
| `vol_setup_ms` | ms | Wall | VOL setup | Outside pipeline total |
| `vol_stage1_ms` | ms | Wall | S1 inference loop | } |
| `vol_drain_ms` | ms | Wall | Worker drain tail | } = vol_pipeline_ms |
| `vol_io_drain_ms` | ms | Wall | I/O drain + cleanup | } |
| `vol_pipeline_ms` | ms | Wall | Pipeline total | stage1 + drain + io_drain |
| `vol_s2_busy_ms` | ms | Wall | Bottleneck worker | NOT additive with above |
| `vol_s3_busy_ms` | ms | Wall | I/O write total | NOT additive with above |
| `stats_ms` | ms | CUDA event | Stats kernels | } |
| `nn_ms` | ms | CUDA event | NN inference kernels | } Sum across chunks, |
| `comp_ms` | ms | CUDA event | Compression kernels | } NOT wall-clock |
| `decomp_ms` | ms | CUDA event | Decompression kernels | } (from read path) |
| `explore_ms` | ms | Wall | Exploration probes | Sum across chunks |
| `sgd_ms` | ms | Wall | SGD weight updates | Sum across chunks |

### benchmark_*_timestep_chunks.csv (per-chunk detail)

| Column | Units | Clock type | Notes |
|--------|-------|-----------|-------|
| `predicted_ratio` | x | — | NN output (write path) |
| `actual_ratio` | x | — | file_bytes / orig_bytes (write path) |
| `predicted_comp_ms` | ms | — | NN output (write path) |
| `actual_comp_ms_raw` | ms | CUDA event | Unclamped compression kernel time (write path) |
| `predicted_decomp_ms` | ms | — | NN output (write path) |
| `actual_decomp_ms_raw` | ms | CUDA event | Unclamped decompression kernel time (read path) |

---

## 7. Bottleneck Summary and Optimization Targets

### Priority 1: S1 synchronization overhead (~12ms, 30% of pipeline)

84% of S1 time is `cudaStreamSynchronize` calls, not GPU work. Three syncs per chunk for tiny
data transfers (~72 bytes each). The stats-copy sync at line 1580 of `H5VLgpucompress.cu` is
avoidable — the D2D copy could remain async since the worker will sync on its own stream.

### Priority 2: Worker compression overhead (~10ms gap per chunk)

The bottleneck worker spends 32.4ms wall clock but only 20ms in the CUDA compression kernel.
The ~10ms gap includes nvcomp manager setup, cudaFree of per-chunk buffers, pool acquire
contention, D2H memcpy, and I/O queue posting under mutex.

### Priority 3: Cold start thread spawning (~1500ms at T=0)

Nine threads are spawned and destroyed on every H5Dwrite via `pthread_create`/`clone()`.
A persistent thread pool would eliminate this entirely.

### Non-bottlenecks

- I/O drain (2.7ms): I/O thread keeps up with worker output
- Post-write ops (0.1ms): cudaDeviceSynchronize is redundant, h5dclose/h5fclose are fast
- Setup at steady state (0.6ms): Buffer cache eliminates reallocation

---

*Analysis performed 2026-03-26. Based on automated agent review by gpucompress-reviewer,
gpucompress-perf-optimizer, and benchmark-timing-auditor.*
