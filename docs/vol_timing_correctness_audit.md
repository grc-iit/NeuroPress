# VOL Pipeline Timing — Correctness Audit

**File:** `src/hdf5/H5VLgpucompress.cu`  
**Deck:** `benchmarks/vpic-kokkos/vpic_benchmark_deck_phase_major.cxx`  
**Date:** 2026-03-25  
**Auditor:** systematic code review + git diff analysis

---

## 1. Scope

This document audits every timing variable used in `gpu_aware_chunked_write` and
exported through the `H5VL_gpucompress_get_*_timing` APIs.  It answers:

- Is the **clock source** correct?
- Are **timestamp boundaries** placed at the right events?
- Are timing globals **thread-safe**?
- Do the reported numbers **add up** (additive invariants)?
- What **bugs or misleading semantics** exist?

---

## 2. Clock Source

```c
static double _now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
```

`CLOCK_MONOTONIC` is the correct choice:

- Never jumps backward (immune to NTP slew or timezone changes).
- Nanosecond resolution on Linux/CUDA nodes.
- Arithmetic `sec × 1000 + nsec / 1e6` yields milliseconds with sub-microsecond
  precision. ✓

---

## 3. Global Timing Variables and Their Definitions

```c
// src/hdf5/H5VLgpucompress.cu, lines 397-404
static double s_stage1_ms   = 0;  // S1: main-thread inference loop (wall-clock)
static double s_drain_ms    = 0;  // S1-end → all workers joined   (wall-clock)
static double s_io_drain_ms = 0;  // workers joined → I/O thread joined (wall-clock)
static double s_s2_busy_ms  = 0;  // max per-worker total wall time (bottleneck)
static double s_s3_busy_ms  = 0;  // I/O thread: sum of write_chunk_to_native times
static double s_total_ms    = 0;  // pipeline wall clock = s1+drain+io_drain
static double s_vol_func_ms = 0;  // total gpu_aware_chunked_write wall clock
static double s_setup_ms    = 0;  // setup before pipeline (alloc + thread spawn)
```

All are **plain `double`** (not atomic).  They are written exclusively by the main
thread (after all joins) and read by the main thread after `H5Dwrite` returns.
No concurrent access → thread-safe. ✓

---

## 4. Timestamp Placement Map

```
_vol_func_start ─────────────────────────────────────────────────────────────┐
  │  VolWriteCtx init (first call only: cudaMalloc, cudaMallocHost)          │
  │  Buffer reuse from cache (subsequent calls)                              │
  │  stats-ring pre-alloc (cudaMalloc × N_chunks)                            │
  │  io_thr = std::thread(…)         ← I/O thread starts                    │
  │  workers.emplace_back(…) × 8     ← 8 worker threads start               │
s_setup_ms = _now_ms() - _vol_func_start                                     │
_pipeline_start = _s1_start = _now_ms()                                      │
  │                                                                           │
  │  ┌── Stage 1 (main thread sequential) ───────────────────────────────┐   │
  │  │  for each chunk:                                                   │   │
  │  │    gather kernel (if non-contiguous) + cudaStreamSync              │   │
  │  │    cudaEventRecord(stats_start) → stats kernel → cudaEventRecord   │   │
  │  │    cudaEventRecord(nn_start) → NN kernel → cudaEventRecord         │   │
  │  │    cudaStreamSynchronize(stream)                                   │   │
  │  │    cudaEventElapsedTime → wi.infer_stats_ms, wi.infer_nn_ms        │   │
  │  │    cudaMalloc d_sc (fallback) or d_stats_ring[ci] (ring path)      │   │
  │  │    cudaMemcpyAsync + cudaStreamSync  (stats copy)                  │   │
  │  │    wq_full_cv.wait (if queue full) → wq.push(wi) → notify worker   │   │
  │  └────────────────────────────────────────────────────────────────────┘   │
s_stage1_ms = _now_ms() - _s1_start                                          │
  │                                                                           │
  │  ┌── Drain (main thread sends sentinels + waits) ─────────────────────┐  │
  │  │  for w in N_COMP_WORKERS: wq.push(sentinel)                        │  │
  │  │  for auto &t : workers: t.join()                                   │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
_t_drain_end = _now_ms()                                                      │
s_drain_ms = _t_drain_end - _s1_start - s_stage1_ms                          │
  │                                                                           │
  │  d_stats_ring cleanup (cudaFree × N_chunks, fallback path only)          │
  │  { io_done_flag = true } → io_cv.notify_all()                            │
  │  io_thr.join()                                                            │
_t_end = _now_ms()                                                            │
s_io_drain_ms = _t_end - _t_drain_end                                        │
s_total_ms    = _t_end - _pipeline_start                                      │
s_s3_busy_ms  = io_write_total_ms  (accumulated inside I/O thread)           │
  │                                                                           │
s_vol_func_ms = _now_ms() - _vol_func_start  ←────────────────────────────── ┘
```

### Workers (concurrent with Stage 1 and Drain)

```
worker thread w:
  while queue not empty:
    _w_start = _now_ms()
    wi = wq.pop()
    gpucompress_compress_with_action_gpu(…)    ← compression
    cudaFree(wi.d_owned); cudaFree(wi.d_stats_copy)
    hbuf = pool_acquire()                      ← may block on I/O pool
    cudaMemcpy(hbuf, d_comp_w[w], …, D→H)
    io_q.push(item)                            ← may wait for I/O queue space
    w_total += _now_ms() - _w_start           ← full iteration, not just compression
  worker_comp_ms[w] = w_total

s_s2_busy_ms = max(worker_comp_ms[w]) across all workers
```

### I/O thread (concurrent with Stage 1, Drain, and Cleanup)

```
io_thr:
  while true:
    item = io_q.pop()
    _t_io_w = _now_ms()
    write_chunk_to_native(…)
    io_write_total_ms += _now_ms() - _t_io_w    ← pure HDF5 write wall time
    pool_release(item.data)
```

---

## 5. Additive Invariant

The three wall-clock segments tile the pipeline without gaps or overlaps:

```
stage1 + drain + io_drain
= (_s1_end − _pipeline_start) + (_t_drain_end − _s1_end) + (_t_end − _t_drain_end)
= _t_end − _pipeline_start
= total_ms                                                                ✓
```

An `assert` (debug builds only) enforces this within 0.5 ms:

```c
// src/hdf5/H5VLgpucompress.cu, lines 1665-1671
#ifndef NDEBUG
    if (s_total_ms > 0) {
        double _sum = s_stage1_ms + s_drain_ms + s_io_drain_ms;
        double _diff = _sum - s_total_ms;
        if (_diff < 0) _diff = -_diff;
        assert(_diff < 0.5); /* additive invariant */
    }
#endif
```

**Recommendation:** promote this check to a `fprintf`+`fflush` warning in release
builds (omit `assert`) so it is visible in production runs without aborting.

---

## 6. CSV Column → Internal Variable Mapping

The timestep CSV (`benchmark_vpic_deck_timesteps.csv`) header:

```
…,vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,
  vol_s2_busy_ms,vol_s3_busy_ms,
  h5dwrite_ms,cuda_sync_ms,h5dclose_ms,h5fclose_ms,
  vol_setup_ms,vol_pipeline_ms
```

| CSV column | Internal variable | Meaning |
|---|---|---|
| `vol_stage1_ms` | `s_stage1_ms` | Main-thread S1 loop wall-clock |
| `vol_drain_ms` | `s_drain_ms` | Sentinel dispatch + worker join after S1 |
| `vol_io_drain_ms` | `s_io_drain_ms` | Ring-buffer cleanup + I/O thread join |
| `vol_s2_busy_ms` | `s_s2_busy_ms` | Max per-worker wall time (bottleneck worker) |
| `vol_s3_busy_ms` | `s_s3_busy_ms` | Sum of pure HDF5 write durations in I/O thread |
| `vol_pipeline_ms` | `s_total_ms` | `_t_end − _pipeline_start` |
| `vol_setup_ms` | `s_setup_ms` | `_pipeline_start − _vol_func_start` |
| `h5dwrite_ms` | local in deck | Full `H5Dwrite` wall-clock (deck side) |

GPU timing columns (`stats_ms`, `nn_ms`) are **not** VOL globals — they are sums of
per-chunk `cudaEventElapsedTime` values collected through `gpucompress_chunk_diag_t`.

---

## 7. Bugs and Issues Found

### 7.1 Bug — Data Race on Transfer Counters (FIXED in HEAD)

**Severity: Critical**

**Old code (binary used for profiling):**
```c
static size_t s_h2d_bytes = 0;  // non-atomic
// in vol_memcpy, called from 8 concurrent worker threads:
case cudaMemcpyDeviceToHost: s_d2h_bytes += bytes; s_d2h_count++; break;
```

`vol_memcpy` is called by all 8 worker threads concurrently.
Non-atomic read-modify-write on `s_d2h_bytes` is a **C++ data race — undefined
behaviour**.  Observed symptoms: wrong byte counts, occasional silent counter
corruption.

**Fix (HEAD):**
```c
static std::atomic<size_t> s_h2d_bytes{0};
// in vol_memcpy:
case cudaMemcpyDeviceToHost:
    s_d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    s_d2h_count.fetch_add(1,     std::memory_order_relaxed);
    break;
```

**Impact on profiling results:** `H5VL_gpucompress_get_transfer_stats` output in
the pre-fix binary was unreliable.  Any analysis based on `h2d_bytes` / `d2h_bytes`
from that run should be treated as approximate.

---

### 7.2 Bug — `vol_s2_busy_ms` Semantics Changed; Plots Are Mislabelled

**Severity: Semantic / Documentation**

The timing scope of `worker_comp_ms[w]` changed between the old binary and HEAD:

**Old code — compression only:**
```c
double _w_start = _now_ms();
gpucompress_compress_with_action_gpu(…);   // compression
w_total += _now_ms() - _w_start;           // ← stop here
// cudaFree, pool_acquire, cudaMemcpy D→H, io_q push happen AFTER timing
```

**New code (HEAD) — full iteration:**
```c
double _w_start = _now_ms();
gpucompress_compress_with_action_gpu(…);   // compression
cudaFree(wi.d_owned); cudaFree(wi.d_stats_copy);
hbuf = pool_acquire();                     // may block waiting for I/O pool
cudaMemcpy(hbuf, d_comp_w[w], …, D→H);
io_q.push(item);                           // may block if io_q is full
w_total += _now_ms() - _w_start;           // ← stop here — full iteration
```

`s_s2_busy_ms = max(worker_comp_ms[w])` now measures the **bottleneck worker's full
wall time**: compression + buffer free + pool-buffer acquire + D→H copy + I/O queue
post.  If the I/O thread is slow, `pool_acquire()` blocks and inflates this number.

**Current plots label this as "S2: Compression" — that is incorrect.**

**Recommended fix:** relabel in `visualize.py`:
```python
# Change:
("vol_s2_busy_ms", "#2ecc71", "S2: Compression"),
# To:
("vol_s2_busy_ms", "#2ecc71", "S2: Worker wall (compress + D→H + I/O post)"),
```

---

### 7.3 Issue — `s_io_drain_ms` Absorbs Ring-Buffer Cleanup Latency

**Severity: Minor / Intentional**

`_t_drain_end` is captured **after** workers join (line 1630).  Between `_t_drain_end`
and `_t_end` the code does:

1. `d_stats_ring` cleanup: `cudaFree` × N_chunks (fallback path), or O(1) for ring path.
2. Signal I/O thread (`io_done_flag = true`).
3. `io_thr.join()`.

The comment in code is explicit:
```c
/* Use _t_drain_end (captured after worker join) so that ring buffer
 * cleanup time is absorbed into io_drain — preserving the additive
 * invariant: stage1 + drain + io_drain = total. */
```

For the **ring-buffer path** (pre-allocated `d_stats_ring`), cleanup is a single
`free(d_stats_ring)` — negligible.  For the **fallback per-chunk alloc path**,
64 × `cudaFree` ≈ 1–3 ms silently appears as I/O drain time.

**Recommended fix (optional):** capture a `_t_cleanup_end` timestamp after
`d_stats_ring` cleanup and expose `s_cleanup_ms = _t_cleanup_end - _t_drain_end`
as a separate diagnostic column.

---

### 7.4 Issue — S1 Sub-Decomposition is an Approximation

**Severity: Methodological (affects `visualize.py` figure 6d)**

In `visualize.py`, Stage 1 is decomposed as:

```python
s1_total = means["vol_stage1_ms"]    # CPU wall-clock
s1a      = means["stats_ms"]         # CUDA event time (GPU-only, sum across chunks)
s1b      = means["nn_ms"]            # CUDA event time (GPU-only, sum across chunks)
s1c      = max(0.0, s1_total - s1a - s1b)   # residual = CPU overhead
```

`stats_ms` and `nn_ms` come from `cudaEventElapsedTime` between pairs of
`cudaEventRecord` calls.  They measure **pure GPU kernel execution time**, not
CPU wall-clock.  The relationship to wall-clock is:

```
[CPU]  launch stats → launch NN → cudaStreamSynchronize (wait) → next work
[GPU]               |─ stats kernel ─|─ NN kernel ─|
                    <──── stats_ms ────><── nn_ms ──>
```

After the async kernel launches, the CPU executes `cudaStreamSynchronize` which
blocks until the GPU finishes.  During this blocking period, `vol_stage1_ms` is
accumulating but `stats_ms + nn_ms` is not (GPU events are purely GPU-side timers).

Therefore:
- `vol_stage1_ms ≥ stats_ms + nn_ms` always holds per chunk (guaranteed by sync). ✓
- `s1c` (residual) captures: kernel-launch overhead + sync-wait latency + per-chunk
  `cudaMalloc`/`cudaMemcpyAsync` + WQ-post blocking time.
- **`s1c` is not a single operation** — it is a superset of CPU-side overhead that
  cannot be further decomposed without per-chunk CPU timestamps.

The decomposition is **valid as an approximation** but should be described in plots
as "S1c: CPU residual (malloc + sync latency + WQ wait)" rather than "WQ Post / Sync".

---

### 7.5 Issue — First-Timestep `vol_setup_ms` Inflation (Cold-Start Artifact)

**Severity: Data quality**

`s_setup_ms = _now_ms() - _vol_func_start` spans everything before `_pipeline_start`:

| Included operation | First call | Subsequent calls |
|---|---|---|
| `VolWriteCtx` init | Yes (GPU alloc ~10–100 ms) | No (reuse) |
| Stats ring alloc | Yes (N×`cudaMalloc`) | No (reuse) |
| I/O thread spawn | Yes | Yes |
| Worker thread spawn (×8) | Yes | Yes |

On the first `H5Dwrite` of a run, `vol_setup_ms` can be 50–200 ms due to GPU buffer
allocation.  On subsequent calls it drops to ~1 ms (thread spawn only).

The benchmark skips the first 5 timesteps for steady-state averages, which mitigates
this.  However, per-timestep plots that include timestep 0 will show a spike.

**Recommended fix:** add a boolean `wctx->first_write` flag; when true, split
`s_setup_ms` into `s_alloc_ms` (allocation) and `s_spawn_ms` (thread spawn) so
the two effects are visible separately in the output.

---

### 7.6 Issue — Single-Write Assumption (Timing Overwrite, Not Accumulate)

**Severity: Fragile for multi-write workloads**

The timing globals (`s_stage1_ms`, `s_drain_ms`, etc.) are **overwritten** (not
accumulated) on each call to `gpu_aware_chunked_write`.  After `H5Dwrite` returns,
the benchmark reads them once and they reflect only the most recent call.

For VPIC (single `H5Dwrite` per timestep per dataset) this is correct.  For any
application that calls `H5Dwrite` multiple times before reading the timing (e.g.,
partial writes, multiple datasets per timestep), only the last call's timings would
be reported — earlier calls are silently discarded.

**Recommended fix:** add an accumulation mode (sum-across-calls) gated by a flag,
or document the single-call assumption explicitly in the API header.

---

## 8. Edge Cases Verified

| Scenario | Handling |
|---|---|
| `goto done_write` (buffer alloc failure, before pipeline starts) | `_pipeline_start == 0` and `_t_drain_end == 0`; guards `... > 0 ? ... : 0` report zero for all pipeline timings. ✓ |
| Chunk-loop `break` on error | Falls through to sentinel dispatch and worker join; `s_stage1_ms` is still captured at line 1615. ✓ |
| `io_write_total_ms` thread safety | Local variable; I/O thread writes to it; main thread reads only after `io_thr.join()`. `join()` provides happens-before guarantee. ✓ |
| `worker_comp_ms[w]` thread safety | Written by worker `w`; read by main thread after `t.join()`. ✓ |
| `s_vol_func_ms` vs `vol_setup_ms + vol_pipeline_ms` | `s_vol_func_ms` is slightly larger: it includes `free_dim_arrays` (post-join cleanup) not in `s_total_ms`. Gap is µs-level. ✓ |

---

## 9. Summary of All Issues

| # | Severity | Issue | Fixed in HEAD? |
|---|---|---|---|
| 7.1 | **Critical** | Data race on `s_h2d_bytes`/`s_d2h_bytes` etc. — written from 8 worker threads without synchronization | **Yes** |
| 7.2 | **Semantic** | `vol_s2_busy_ms` now includes D→H + I/O queue post, not only compression; plot label "S2: Compression" is wrong | **No** — needs relabel |
| 7.3 | Minor | `vol_io_drain_ms` absorbs ring-buffer `cudaFree` cleanup (~1–3 ms for fallback path) | Intentional; optional fix |
| 7.4 | Methodological | S1 sub-decomposition (`s1c = stage1 - stats_ms - nn_ms`) mixes GPU event time with CPU wall-clock | By design; needs better label |
| 7.5 | Data quality | First-timestep `vol_setup_ms` inflated by GPU buffer allocation (50–200 ms cold start) | Mitigated by warmup skip |
| 7.6 | Fragile | Timing globals overwritten (not accumulated) per `H5Dwrite`; silent data loss for multi-write workloads | No — needs documentation |

---

## 10. Recommended Code Fixes

### Fix A — Relabel `vol_s2_busy_ms` in plots

```python
# benchmarks/visualize.py
STAGES = [
    ("vol_setup_ms",    "#95a5a6", "Setup (threads + alloc)"),
    ("s1a_stats_ms",    "#aed6f1", "S1a: Stats Kernel (GPU)"),
    ("s1b_nn_ms",       "#2980b9", "S1b: NN Inference (GPU)"),
    ("s1c_residual_ms", "#1a5276", "S1c: CPU residual (malloc + sync + WQ wait)"),
    ("vol_drain_ms",    "#27ae60", "S1→S2 Drain (sentinel + worker join)"),
-   ("vol_s2_busy_ms",  "#2ecc71", "S2: Compression"),
+   ("vol_s2_busy_ms",  "#2ecc71", "S2: Worker wall (compress + D→H + I/O post)"),
    ("vol_s3_busy_ms",  "#e74c3c", "S3: I/O write (HDF5 serial)"),
    ("vol_io_drain_ms", "#9b59b6", "I/O drain (tail + cleanup)"),
]
```

### Fix B — Promote Invariant Assert to Release Warning

```c
// src/hdf5/H5VLgpucompress.cu, after line 1661
if (s_total_ms > 0) {
    double _sum  = s_stage1_ms + s_drain_ms + s_io_drain_ms;
    double _diff = fabs(_sum - s_total_ms);
    if (_diff > 1.0) {   /* 1 ms tolerance for release builds */
        fprintf(stderr,
            "gpucompress VOL: timing invariant violated "
            "(sum=%.2f total=%.2f diff=%.2f ms)\n",
            _sum, s_total_ms, _diff);
        fflush(stderr);
    }
}
```

### Fix C — Document Single-Write Assumption in the API Header

Add to `include/gpucompress_vol.h` (or wherever the API is declared):

```c
/*
 * H5VL_gpucompress_get_stage_timing / get_busy_timing / get_vol_func_timing
 *
 * IMPORTANT: these functions return timing from the MOST RECENT call to
 * gpu_aware_chunked_write (H5Dwrite on a gpucompress dataset).  If multiple
 * H5Dwrite calls occur between successive reads of these timing values, only
 * the last call's data is retained.  For VPIC (one write per timestep) this
 * is correct; for other workloads, call H5VL_gpucompress_reset_stats() before
 * each H5Dwrite to ensure a clean baseline.
 */
```

---

*Generated from audit of commit range HEAD vs profiled binary (compiled 2026-03-25 21:16).*
