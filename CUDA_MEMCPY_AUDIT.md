# CUDA Memory Transfer Audit

An audit of every `cudaMemcpy`, `cudaMalloc`, and `cudaFree` across the compress, decompress, stats, and NN inference hot paths. Findings are ranked by performance impact.

---

## Why this matters

`cudaMalloc` and `cudaFree` are **implicit synchronization points** — they block the CPU until any pending GPU work completes, then perform an OS-level allocation. Each call can cost **10–100 μs**. For tiny buffers (4–128 bytes), the allocation overhead dwarfs the actual work. Repeated per-call alloc/free patterns create a serialization bottleneck that prevents full CPU–GPU overlap.

`cudaMemcpy` (synchronous) is even worse — it blocks the CPU until the transfer completes, destroying any pipelining. `cudaMemcpyAsync` is preferred, but even async transfers of tiny buffers incur per-call driver overhead.

---

## Finding 1 (HIGH): Per-call cudaMalloc/cudaFree in NN inference

**File:** `src/nn/nn_gpu.cu:574–640`

Every call to `runNNInference()` allocates and frees 3 tiny device buffers:

```
cudaMalloc(&d_action, 4 bytes)           // line 574
cudaMalloc(&d_predicted_ratio, 4 bytes)  // line 578
cudaMalloc(&d_top_actions, 128 bytes)    // line 586
// ... kernel launch + 3x cudaMemcpyAsync D2H ...
cudaFree(d_action)                       // line 638
cudaFree(d_predicted_ratio)              // line 639
cudaFree(d_top_actions)                  // line 640
```

That is **3 malloc + 3 free = 6 synchronization points** for 136 total bytes. Estimated overhead: **60–600 μs per inference call** — likely more than the kernel itself (~150 μs).

**Fix:** Pre-allocate `d_action`, `d_predicted_ratio`, and `d_top_actions` inside `loadNNFromBinary()` alongside `d_nn_weights`. Reuse across all calls. Free in `unloadNN()`.

---

## Finding 2 (HIGH): Per-call cudaMalloc/cudaFree in stats pipeline

**File:** `src/stats/stats_kernel.cu:438, 516` (duplicated at lines 550, 642 for Q-table path)

Every call to `runStatsKernels()` allocates the workspace:

```
cudaMalloc(&d_workspace, sizeof(AutoStatsGPU) + 256 * sizeof(uint))  // line 438
cudaMemcpyAsync(d_stats, &h_init, sizeof(AutoStatsGPU), H2D)        // line 462
// ... 5 kernel launches ...
cudaMemcpyAsync x2 (results D2H)                                     // lines 495, 502
cudaFree(d_workspace)                                                 // line 516
```

Same problem: malloc/free per call for a workspace that is always the same size.

**Fix:** Pre-allocate the stats workspace once during `gpucompress_init()` or on first use, reuse across all calls, free during `gpucompress_cleanup()`.

---

## Finding 3 (HIGH when triggered): Active learning loop — sync memcpy + per-iteration malloc/free

**File:** `src/api/gpucompress_api.cpp:529–671`

The Level 2 exploration loop runs up to **K=31 iterations**, and each iteration does:

```
cudaStreamSynchronize(stream)                                     // line 580
cudaMalloc(&d_alt_out, alt_max)                                   // line 597
cudaMemcpy(output + header_sz, d_alt_out, alt_comp_size, D2H)    // line 650 (SYNCHRONOUS!)
cudaFree(d_alt_out)                                               // line 661
cudaFree(d_alt_quant)                                             // line 668
cudaFree(d_alt_shuf)                                              // line 669
```

The `cudaMemcpy` at line 650 is **synchronous** (not Async), and there are up to 5 malloc/free calls per iteration. Worst case (K=31): **31 × (sync + malloc + sync_memcpy + 3×free)** — devastating for latency.

**Fix:**
1. Pre-allocate a single `d_alt_out` buffer large enough for the worst case, reuse across iterations
2. Switch `cudaMemcpy` at line 650 to `cudaMemcpyAsync`
3. Consider batching: preprocess all K alternatives, then compress all, to amortize sync overhead

---

## Finding 4 (MEDIUM): Header round-trips host → GPU → host

**File:** `src/api/gpucompress_api.cpp:463, 466`

The compression header is built entirely on the CPU (lines 437–460), then:

```cpp
writeHeaderToDevice(d_output, header, stream);                     // H2D: header to GPU
cudaMemcpyAsync(output, d_output, total_size, D2H, stream);       // D2H: header+compressed back
```

The header makes a pointless round-trip: host → GPU → host. It was never needed on the device.

**Fix:**

```cpp
// Before (unnecessary round-trip):
writeHeaderToDevice(d_output, header, stream);
cudaMemcpyAsync(output, d_output, total_size, D2H, stream);

// After (header stays on host, only compressed data crosses PCIe):
memcpy(output, &header, sizeof(CompressionHeader));               // host-to-host, instant
cudaMemcpyAsync((uint8_t*)output + header_size,
                d_compressed, compressed_size, D2H, stream);      // D2H: compressed only
```

Savings: eliminates 1 H2D transfer, reduces D2H by `GPUCOMPRESS_HEADER_SIZE` bytes. The `d_output` allocation can also shrink by `header_size` bytes since it no longer needs space for the header.

---

## Finding 5 (MEDIUM): Decompress copies header bytes to GPU unnecessarily

**File:** `src/api/gpucompress_api.cpp:734, 740`

```cpp
cudaMalloc(&d_compressed, input_size);                              // allocates header+payload
cudaMemcpyAsync(d_compressed, input, input_size, H2D, stream);     // copies header+payload
uint8_t* d_compressed_data = d_compressed + header_size;            // then skips the header
```

The header was already read and parsed on the host (line 716: `memcpy(&header, input, ...)`). Those `header_size` bytes are dead weight on the GPU — they are uploaded and never used.

**Fix:**

```cpp
// Before:
cudaMalloc(&d_compressed, input_size);
cudaMemcpyAsync(d_compressed, input, input_size, H2D, stream);
uint8_t* d_compressed_data = d_compressed + header_size;

// After (only copy the payload):
size_t payload_size = input_size - header_size;
cudaMalloc(&d_compressed, payload_size);
cudaMemcpyAsync(d_compressed, (uint8_t*)input + header_size, payload_size, H2D, stream);
// d_compressed now points directly to compressed data, no offset needed
```

Savings: `GPUCOMPRESS_HEADER_SIZE` (64 bytes) less H2D transfer and device allocation.

---

## Finding 6 (LOW-MEDIUM): Per-call cudaMalloc/cudaFree in Q-table lookup

**File:** `src/lib/qtable_gpu.cu:402, 417`

```cpp
cudaMalloc(&d_best_action, sizeof(int));     // 4 bytes
// ... kernel ...
cudaFree(d_best_action);
```

Same pattern as NN inference. Only used in the fallback path (when NN isn't loaded), but still 2 unnecessary sync points.

**Fix:** Pre-allocate during `loadQTable()`.

---

## Finding 7 (LOW): Stats results copied in 2 separate small transfers

**File:** `src/stats/stats_kernel.cu:495–504`

```cpp
cudaMemcpyAsync(&h_result.entropy, &d_stats->entropy, 8 bytes, D2H);
cudaMemcpyAsync(&h_result.mad_normalized, &d_stats->mad_normalized, 16 bytes, D2H);
```

Two async transfers for 24 total bytes. These could be combined into a single 24-byte transfer if `entropy`, `mad_normalized`, and `deriv_normalized` were placed contiguously in the `AutoStatsGPU` struct.

---

## Finding 8 (LOW): Quantization min/max uses 4 separate tiny transfers

**File:** `src/preprocessing/quantization_kernels.cu:353–361`

```cpp
cudaMemcpyAsync(d_min, &init_min, 4B, H2D);      // init min
cudaMemcpyAsync(d_max, &init_max, 4B, H2D);      // init max
// ... kernel ...
cudaMemcpyAsync(&h_min, d_min, 4B, D2H);          // read min
cudaMemcpyAsync(&h_max, d_max, 4B, D2H);          // read max
```

4 separate transfers for 16 total bytes. Could pack into `struct { float min, max; }` and use 1 transfer each way instead of 2. Same pattern repeats at lines 378–385 for the `double` variant.

---

## Summary table

| # | Finding | Impact | Location | Fix Effort |
|---|---------|--------|----------|------------|
| 1 | NN inference malloc/free per call | **High** | `nn_gpu.cu:574–640` | Low — pre-allocate in load |
| 2 | Stats workspace malloc/free per call | **High** | `stats_kernel.cu:438, 516` | Low — pre-allocate in init |
| 3 | Active learning loop sync + malloc | **High** (triggered) | `gpucompress_api.cpp:529–671` | Medium — pre-allocate + batch |
| 4 | Header round-trip H→D→H | **Medium** | `gpucompress_api.cpp:463, 466` | Low — memcpy on host |
| 5 | Decompress copies header to GPU | **Medium** | `gpucompress_api.cpp:734, 740` | Low — offset source pointer |
| 6 | Q-table malloc/free per call | **Low-Med** | `qtable_gpu.cu:402, 417` | Low — pre-allocate in load |
| 7 | Stats 2 separate tiny D2H | **Low** | `stats_kernel.cu:495–504` | Low — struct reorder |
| 8 | Quantization 4 tiny transfers | **Low** | `quantization_kernels.cu:353–361` | Low — pack into struct |

---

## Complete transfer map for ALGO_AUTO compress path

```
Host                              GPU
─────                             ────
input data ──── H2D (big) ──────> d_input               gpucompress_api.cpp:245
h_init     ──── H2D (small) ───> d_stats                stats_kernel.cu:462
                                  5 kernels (stats)
entropy    <─── D2H (8B) ──────  d_stats->entropy       stats_kernel.cu:495
mad+deriv  <─── D2H (16B) ─────  d_stats->mad_norm      stats_kernel.cu:502
                                  [cudaFree d_workspace]
                                  [cudaMalloc d_action, d_predicted_ratio, d_top_actions]
                                  nnInferenceKernel<<<1,32>>>
h_action   <─── D2H (4B) ──────  d_action               nn_gpu.cu:604
pred_ratio <─── D2H (4B) ──────  d_predicted_ratio      nn_gpu.cu:614
top_acts   <─── D2H (128B) ────  d_top_actions           nn_gpu.cu:625
                                  [cudaFree d_action, d_predicted_ratio, d_top_actions]
                                  preprocess (quant/shuffle)
                                  nvcomp compress
header     ──── H2D (64B) ─────> d_output               compression_header.h:203
output     <─── D2H (big) ──────  d_output               gpucompress_api.cpp:466
```

The small transfers (marked with byte sizes) are dominated by their setup cost, not data volume. The `cudaMalloc`/`cudaFree` brackets around the NN inference and stats pipelines are the primary optimization targets.
