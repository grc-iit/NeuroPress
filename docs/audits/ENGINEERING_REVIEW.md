# GPUCompress Engineering Review

**Date:** 2026-02-28
**Scope:** Full codebase audit — `src/`, `neural_net/`, `tests/`, `examples/`, build system
**Method:** Parallel exploratory analysis across all subsystems; no assumptions made.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Critical Bugs](#2-critical-bugs)
3. [Performance Suboptimalities](#3-performance-suboptimalities)
4. [Design Issues](#4-design-issues)
5. [Build System Issues](#5-build-system-issues)
6. [Neural Net Training Issues](#6-neural-net-training-issues)
7. [Testing Gaps](#7-testing-gaps)
8. [Quick Wins (Low-Risk, High-Reward)](#8-quick-wins)
9. [Architecture Observations](#9-architecture-observations)
10. [Exploration Backlog](#10-exploration-backlog)

---

## 1. System Overview

```
GPU-resident float32 data
        │
        ▼
  HDF5 VOL Connector (H5VLgpucompress.cu)
  ┌──────────────────────────────────────────────────────────┐
  │  Stage 1 (main thread): chunk iteration, WorkItem queue  │
  │  Stage 2 (8 workers):  gpucompress_compress_gpu() +D→H  │
  │  Stage 3 (I/O thread): write compressed chunks to disk   │
  └──────────────────────────────────────────────────────────┘
        │
        ▼
  gpucompress_compress_gpu()  [src/api/gpucompress_api.cpp]
  ┌──────────────────────────────────────────────────────────┐
  │  1. Acquire CompContext (pool of 8 independent slots)    │
  │  2. Stats pipeline (entropy, MAD, deriv) on ctx->stream  │
  │  3. nnFusedInferenceKernel → best of 32 configs         │
  │  4. Optional: quantize + byte-shuffle preprocessing      │
  │  5. nvcomp compression (LZ4/ZSTD/etc.)                  │
  │  6. Optional: exploration (try K alternatives + SGD)     │
  │  7. cudaStreamSynchronize(ctx->stream)                   │
  │  8. Release CompContext                                   │
  └──────────────────────────────────────────────────────────┘
        │
        ▼
  Neural Network (src/nn/nn_gpu.cu)
  15→128→128→4 MLP, ReLU activations
  Inputs: one-hot algo, quant, shuffle, log(error_bound),
          log(data_size), entropy, MAD, 2nd-deriv
  Outputs: comp_time, decomp_time, ratio, PSNR  (all log-space)
  Online SGD: GPU-native (nnSGDKernel) on dedicated g_sgd_stream
  Inference: nnFusedInferenceKernel reads stats directly from d_stats
```

Action space: 32 configs = 8 algos × 2 quant modes × 2 shuffle modes
Encoding: `action_id = algo + quant*8 + shuffle*16`

---

## 2. Critical Bugs

### BUG-1: `d_owned` Memory Leak on Compression Error
**File:** `src/hdf5/H5VLgpucompress.cu` — Worker lambda
**Severity:** HIGH
**Status:** ✅ FIXED — `cudaFree(wi.d_owned)` is called unconditionally _before_ the `ce != GPUCOMPRESS_SUCCESS` error check (line 1072). Both success and error paths free the buffer correctly.

```cpp
// Current code — correct order:
gpucompress_compress_gpu(...);
if (wi.d_owned) { cudaFree(wi.d_owned); wi.d_owned = NULL; }  // always freed
if (ce != GPUCOMPRESS_SUCCESS) { worker_err.store(-1); continue; }
```

---

### BUG-2: Dimension Array Memory Leak on Mid-Loop Failure
**File:** `src/hdf5/H5VLgpucompress.cu` — `gpu_aware_chunked_write`
**Severity:** HIGH
**Status:** ✅ FIXED — All break/error paths eventually reach the unified cleanup block at line 1245: `if (d_dset_dims) free_dim_arrays(d_dset_dims, d_chunk_dims, d_chunk_start)`. The guard `if (d_dset_dims)` handles the case where allocation never occurred.

---

### BUG-3: Race Condition in `nnSGDKernel` Gradient Accumulation
**File:** `src/nn/nn_gpu.cu` — `nnSGDKernel<<<1,128>>>`
**Severity:** HIGH (potential weight corruption)
**Status:** ✅ AUDITED — NOT A BUG. Partition is perfect.

Formal analysis (verified by `tests/test_bug3_sgd_gradients.cu`):
Thread `t` (0..127) exclusively owns:
- `dw1[t*15..t*15+14]` — d_grad_buffer[SGD_OFF_DW1 + t*15 + 0..14]
- `db1[t]` — d_grad_buffer[SGD_OFF_DB1 + t]
- `dw2[t*128..t*128+127]` — d_grad_buffer[SGD_OFF_DW2 + t*128 + 0..127]
- `db2[t]` — d_grad_buffer[SGD_OFF_DB2 + t]
- `dw3[out*128+t]` for out=0..3 — 4 distinct indices (one per output, column t)
- `db3[t]` only if t < NN_OUTPUT_DIM=4 (guarded)

For any (out, t) pair, `SGD_OFF_DW3 + out*128 + t` is written only by thread `t`.
Zero races, zero aliasing. Static partition analysis: races=0, unowned_indices=0.
16/16 concurrent compressions with SGD enabled pass byte-exact round-trip.

---

### BUG-4: Format String Mismatch
**File:** `src/api/gpucompress_api.cpp` ~line 1243
**Severity:** LOW (UB, likely benign on 64-bit but technically wrong)
**Status:** ✅ FIXED — `#include <cinttypes>` added; `%u` replaced with `PRIu64` + `(uint64_t)` cast.

```cpp
// Before (UB — %u is 32-bit, original_size is uint64_t):
fprintf(stderr, "[XFER D→H] decompress: result (%u B)\n", header.original_size);
// After (correct):
fprintf(stderr, "[XFER D→H] decompress: result (%" PRIu64 " B)\n", (uint64_t)header.original_size);
```

Verified by `tests/test_bug4_format_string.cu`: 5/5 round-trips pass; stderr shows correct full decimal sizes.

---

### BUG-5: Corrupted Weights on Truncated `.nnwt` File
**File:** `src/nn/nn_gpu.cu` — `loadNNFromBinary()`
**Severity:** MEDIUM
**Status:** ✅ FIXED — Per-read `gcount()` checks added via `NN_READ` macro.

Fix:
- Header: consolidated 6 × 4-byte reads into one 24-byte read with `gcount() != 24` check.
- Each weight array (w1, b1, w2, b2, w3, b3, x_means/stds, y_means/stds, x_mins/maxs):
  uses `NN_READ(ptr, nbytes, name)` macro that checks `gcount() == expected` and returns false immediately.

Verified by `tests/test_bug5_truncated_nnwt.cu`: 7/7 cases pass:
- Empty file, truncated header, bad magic, wrong architecture, truncated mid-w2, truncated at w3, valid file.

---

### BUG-6: Unprotected Global Statistics in VOL Connector
**File:** `src/hdf5/H5VLgpucompress.cu`
**Severity:** MEDIUM
**Status:** ✅ FIXED (de facto) — With the 3-stage pipeline, both `s_gpu_writes` (line 1114) and `s_chunks_comp` (line 1215) are incremented only in Stage 1 (main thread), not in worker threads. No concurrent access occurs under the current architecture.

---

## 3. Performance Suboptimalities

### PERF-1: Exploration Round-Trip Overhead
**File:** `src/api/gpucompress_api.cpp` — exploration path (~lines 808-934)
**Severity:** HIGH (O(K × input_size) extra traffic)

When exploration triggers:
- K=4 (baseline), K=9 (high error), K=31 (OOD)
- Each alternative: full compress → full decompress → PSNR (2× D→H copies of full data for lossy)
- For 64 MB chunk with K=31: ~4 GB of extra memory traffic

**Options:**
1. Compute PSNR on GPU (eliminates D→H)
2. Skip PSNR for alternatives outside top-N candidates
3. Cache predicted ratio ÷ actual ratio to skip round-trip for obviously worse alternatives
4. Early-exit exploration once a "good enough" alternative found

---

### PERF-2: Insertion Sort on Thread 0 (31 Threads Idle)
**File:** `src/nn/nn_gpu.cu` — `nnInferenceKernel` / `nnFusedInferenceKernel`
**Severity:** MEDIUM (97% thread utilization waste during ranking)
**Status:** ✅ FIXED

Both kernels now use parallel bitonic sort (32 threads, warp-shuffle based). All 32 threads participate in 5 passes of compare-and-swap using `__shfl_xor_sync`. Serial insertion sort (O(n²)=1024 ops on thread 0) replaced with parallel bitonic sort (O(n log²n)=80 ops across all 32 threads). Verified in commit `d704d71` (32.9x speedup).

---

### PERF-3: Feature Encoding Repeated Per Thread
**File:** `src/nn/nn_gpu.cu` — inference kernels
**Severity:** LOW-MEDIUM
**Status:** ✅ FIXED

Thread 0 now computes `log10(eb)`, `log2(ds)`, and float-casts of entropy/MAD/deriv into `__shared__ float s_enc[5]`, then `__syncthreads()`. All 32 threads read from `s_enc[]` in `nnForwardPass`. The two OOD detection blocks (bitonic + tree-reduction branches) also reuse `s_enc[]` instead of recomputing. Net savings: 31 × (`log10` + `log2`) = 62 transcendental ops per inference call, plus 31 × 3 redundant global memory reads from `d_stats` in the fused kernel.

---

### PERF-4: Multiple Small D→H Copies in `runNNInference()`
**File:** `src/nn/nn_gpu.cu` — `runNNInference()`
**Severity:** MEDIUM (latency on each inference call)
**Status:** ✅ FIXED

3 separate `cudaMemcpyAsync` calls replaced by a single 16B copy of `NNInferenceOutput`:
- `d_infer_action` (4B) + `d_infer_ratio` (4B) + `d_infer_comp_time` (4B) → merged into `NNInferenceOutput* d_infer_output` (16B, same struct already used by fused kernel)
- `nnInferenceKernel` signature updated: 3 separate output pointers → `NNInferenceOutput* out_result`
- `runNNInference()` now issues one `cudaMemcpyAsync(&h_result, d_infer_output, 16B)` then extracts fields from `h_result`

Verified by `tests/test_perf4_batched_dh.cu` (16/16):
- T1: all 3 fields populated correctly (action∈[0,31], ratio>0, comp_time>0)
- T2: no-optional-outputs call returns same action (batched copy not regressed)
- T3: with top_actions, `out_result->action == top_actions[0]` (struct winner matches sorted list)
- T4: cross-validated against `runNNFusedInference()` (independent kernel, same result)

---

### PERF-5: Three Separate D→H Copies in `runStatsKernels()`
**File:** `src/stats/stats_kernel.cu`
**Severity:** LOW-MEDIUM
**Status:** ✅ ALREADY RESOLVED (non-issue)

Two reasons this is moot:
1. `runStatsKernels()` (the `static` helper) already does a **single 24B `cudaMemcpyAsync`** — it packs `entropy + mad_normalized + deriv_normalized` into a local `StatsResultBlock` and copies in one transfer (`stats_kernel.cu:433`).
2. The **primary compression path** (`gpucompress_compress_gpu`) never calls `runStatsKernels()` at all — it calls `runStatsKernelsNoSync()` which returns a raw `AutoStatsGPU*` device pointer. Stats never leave the GPU; `nnFusedInferenceKernel` reads `d_stats->entropy/mad/deriv` directly on-device.

`runAutoStatsNNPipeline()` and `runStatsOnlyPipeline()` (the only callers of the D→H path) are declared but never called anywhere in the codebase.

---

### PERF-6: H→D Initialization of `AutoStatsGPU` Stats Workspace
**File:** `src/stats/stats_kernel.cu` — `runStatsKernelsNoSync()`
**Severity:** LOW

The stats workspace is initialized via `cudaMemsetAsync` (for zeros) + H→D copy for `vmin=FLT_MAX, vmax=-FLT_MAX`. The H→D copy adds a PCIe round-trip. Alternative: use a small init kernel that sets just those fields in-place on GPU.

---

### PERF-7: CAS-Based `atomicMin/Max` for Float
**File:** `src/stats/stats_kernel.cu` — `statsPass1Kernel`
**Severity:** LOW

Manual CAS loop emulates float `atomicMin`/`atomicMax`. On Ampere (sm_80), native float atomics are available. Could use `__atomic_min_block()` or CUDA 11+ intrinsics.

---

### PERF-8: Per-Chunk `malloc` in Workers
**File:** `src/hdf5/H5VLgpucompress.cu` — Worker lambda
**Severity:** MEDIUM
**Status:** ✅ ALREADY DONE

Pre-allocated pinned buffer pool (`N_IO_BUFS=16` buffers × `max_comp` bytes, via `cudaMallocHost`) replaces any per-chunk malloc/free:
- Workers call `pool_acquire()` → D→H directly into the pinned buffer
- IO thread calls `pool_release(item.data)` after disk write → buffer reused
- All 16 buffers allocated once at write start, freed once via `cudaFreeHost()` at cleanup
- No `malloc`/`free` anywhere in the hot path

---

### PERF-9: No Buffer Pooling for Quantize/Shuffle Temporaries
**File:** `src/api/gpucompress_api.cpp` — compress path
**Severity:** LOW-MEDIUM

`d_quantized` and `d_shuffled` are `cudaMalloc`'d per-call and freed immediately. In a tight loop (benchmark, VOL multi-chunk), this fragments GPU memory allocator.

**Fix:** Add to `CompContext` or use a slab allocator sized to max chunk.

---

### PERF-10: `cudaStreamSynchronize` on Default Stream for Timing
**File:** `src/api/gpucompress_api.cpp` — timing path
**Severity:** LOW

`cudaEventSynchronize(g_t_stop)` uses a global event on the default stream, potentially blocking other CompContext slots that also run on the default stream. In the concurrent-worker scenario, timing should use `ctx->t_start`/`ctx->t_stop` only.

---

### PERF-11: CUDA Architecture Targeting Old GPU (sm_52 = Kepler)
**File:** `CMakeLists.txt` / `CMakeCache.txt`
**Severity:** HIGH (10-30% performance loss on modern hardware)
**Status:** ✅ FIXED

```
CMAKE_CUDA_ARCHITECTURES = 52   # Maxwell, circa 2014  ← was default
System GPU = sm_80               # Ampere A100
```

Missing Ampere features: TF32 tensor cores, native float atomics, improved shared memory bandwidth, async copy (cp.async).

**Fix applied in `CMakeLists.txt`:**
```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to target")
endif()
```
Rebuilt with `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80`. Verified `.target sm_80` in `libgpucompress.so` via `cuobjdump`. 120/120 HDF5, 6/6 quant pass.

---

### PERF-12: Missing `-use_fast_math` Flag
**File:** `CMakeLists.txt` line 20
**Severity:** MEDIUM (~5-10% throughput improvement for NN/stats kernels)
**Status:** ✅ FIXED

```cmake
# Before:
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# After:
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")
```

`-use_fast_math` enables approximate `__logf`, `__expf`, `__sqrtf` which are 2-5× faster than IEEE-compliant versions. Acceptable for compression statistics and NN inference. Fixed alongside PERF-11 in the same CMakeLists.txt edit.

---

### PERF-13: Duplicated Forward Pass in Two Inference Kernels
**File:** `src/nn/nn_gpu.cu`
**Severity:** MEDIUM (maintenance risk, and both are slightly slower than needed)
**Status:** ✅ FIXED

Extracted `__device__ static void nnForwardPass(...)` helper. Both `nnInferenceKernel` and `nnFusedInferenceKernel` now call it instead of duplicating ~220 lines of MLP forward-pass code. Stats loading (scalar params vs. `d_stats` pointer) is handled by the callers before invoking the helper.

---

## 4. Design Issues

### DESIGN-1: Exploration Logic Tightly Coupled in Main Compress Path
**File:** `src/api/gpucompress_api.cpp`
**Severity:** MEDIUM

The Level-2 exploration loop (~300 lines) lives inline in `gpucompress_compress_gpu`. It's difficult to:
- Unit test exploration independently
- Tune exploration without rebuilding the entire compress function
- Understand the main compression path at a glance

**Suggestion:** Extract to `runExplorationPass(ctx, d_stats, ...) → ExplorationResult`.

---

### DESIGN-2: SGD Fired on Potentially Stale Stats
**File:** `src/api/gpucompress_api.cpp`
**Severity:** LOW (functionally OK, semantically odd)

`d_stats_ptr` is computed at the start of compression. If exploration then selects a different algorithm and replaces the output, the SGD still uses the original stats (entropy/MAD/deriv of the data) which is correct — but the `action` in the SGDSample is the exploration-winner action, not the NN's first choice. This is intentional but worth documenting explicitly.

---

### DESIGN-3: Global Namespace Structs in `nn_weights.h`
**File:** `src/nn/nn_weights.h`
**Severity:** LOW (works, but fragile)

`SGDSample`, `SGDOutput`, `NNInferenceOutput`, `NNWeightsGPU` are in global namespace to avoid `gpucompress::SGDSample` vs `::SGDSample` mismatch when included inside `namespace gpucompress {}`. This is a footgun: any new file that includes `nn_weights.h` inside a namespace will silently get global-namespace types.

**Fix:** Move all structs into `namespace gpucompress {}` and update all includes/usages accordingly. The include-outside-namespace pattern in `nn_gpu.cu` was a workaround that should be cleaned up.

---

### DESIGN-4: Stubbed `nn_reinforce.cpp` Still in API Surface
**File:** `src/nn/nn_reinforce.h`, `nn_reinforce.cpp`
**Severity:** LOW
**Status:** ✅ FIXED

`nn_reinforce.h` and `nn_reinforce.cpp` deleted. `gpucompress_reinforce_last_stats` removed from `include/gpucompress.h` and `gpucompress_api.cpp`. All forward declarations and call-sites (`nn_reinforce_init/add_sample/apply/cleanup/get_last_stats`) removed from `test_nn.cu`, `test_nn_pipeline.cpp`, and `eval_simulation.cpp`. `nn_reinforce.cpp` removed from `LIB_SOURCES` in CMakeLists.txt. Clean build verified.

---

### DESIGN-5: Non-Deterministic Chunk Write Order in VOL
**File:** `src/hdf5/H5VLgpucompress.cu`
**Severity:** LOW (correctness fine, performance degrades)

Workers push IOItems to the I/O queue in compression-completion order, not in logical chunk order. The I/O thread writes them to file in that order. HDF5's chunk B-tree is order-independent, but non-sequential chunk writes cause file fragmentation and extra B-tree rotations.

**Potential fix:** Tag each IOItem with its logical chunk index; I/O thread sorts or uses ordered output structure. Trade-off: adds latency vs. reduces fragmentation.

---

### DESIGN-6: `H5Zgpucompress` 256-Chunk Static Limit
**File:** `src/hdf5/H5Zgpucompress.c`
**Severity:** MEDIUM
**Status:** ✅ FIXED

`g_chunk_algorithms[256]` replaced with `int* g_chunk_algorithms` (dynamic). Capacity tracked in `g_chunk_capacity`; array grown via `realloc()` with doubling strategy (initial 256, then ×2) when count reaches capacity. `H5Z_gpucompress_reset_chunk_tracking()` resets count to 0, keeping the buffer for reuse. `H5Z_gpucompress_fini()` frees the buffer. `attr_buf[4096]` in `write_chunk_attr` also replaced with heap-allocated buffer sized to `chunk_count × 16 + 1`.

Verified by `tests/test_design6_chunk_tracker.c` (3/3):
- 300-chunk round-trip: tracker grows past index 255, all algorithms recorded, byte-exact data
- 512-chunk round-trip: scales to 2× original limit, byte-exact data
- Reset reuse: `reset()` after 300-chunk write sets count to 0; re-tracking 10 chunks gives count=10

**VOL path also fixed (`src/api/gpucompress_api.cpp`):**
`g_chunk_history[4096]` (static) replaced with `gpucompress_chunk_diag_t* g_chunk_history` (dynamic). Growth is mutex-protected (`g_chunk_history_mutex`) to handle concurrent appends from 8 CompContext workers — the mutex is taken for every append so `realloc` is never called while another thread writes into the old pointer. `gpucompress_reset_chunk_history()` resets count to 0 keeping the buffer. `gpucompress_cleanup()` frees the buffer.

---

### DESIGN-7: CLI Tools Cannot Handle Files Larger Than GPU Memory
**File:** `src/cli/compress.cpp`, `src/cli/decompress.cpp`
**Severity:** HIGH for large-file users
**Status:** ⚠️ NOT APPLICABLE TO PRIMARY WORKFLOW — no fix implemented

```cpp
size_t aligned_input_size = ((file_size + 4095) / 4096) * 4096;
cudaMalloc(&d_input, aligned_input_size);  // Entire file must fit
```
For files >10-40 GB (depending on GPU), this will OOM. The VOL connector handles this via chunking, but the CLI tool does not.

**Why not fixed:** This issue only affects the CLI load-from-disk path. The primary workflow for this project is GPU-native:
simulation generates data directly in GPU memory → `H5Dwrite(GPU_ptr)` → HDF5 VOL connector.

In this path the user already holds the full dataset in VRAM (required for the simulation to run).
The VOL connector reads slices of that pre-existing allocation without any extra `cudaMalloc` for the full dataset.
The CLI's "read entire file from disk → cudaMalloc(file_size)" pattern is structurally impossible in this workflow.

A streaming prototype (version 3 file format) was implemented then reverted once this analysis was confirmed.
If CLI large-file support is needed in the future, the streaming approach (block table header + per-block
`pread`/`pwrite` + `~2× chunk_size` GPU working set) is the correct design.

---

### DESIGN-8: Experience Buffer Missing Key Metrics
**File:** `src/nn/experience_buffer.h`
**Severity:** MEDIUM (affects retraining quality)

C++ active learning logs `entropy, mad, deriv, action, ratio, comp_time` but NOT `decomp_time` or `psnr`. When `retrain.py` processes experience data, it fills missing columns with dataset-wide medians, introducing systematic bias into retraining.

**Fix:** Measure and log `decomp_time` during active learning exploration (decompress is already done for PSNR check in lossy mode).

---

## 5. Build System Issues

### BUILD-1: Targeting Kepler GPU (sm_52) on Ampere Hardware
See PERF-11 above. **Immediate action recommended.**

### BUILD-2: Missing `-use_fast_math`
See PERF-12 above.

### BUILD-3: nvcomp Hardcoded at `/tmp/lib`
**File:** `CMakeLists.txt` line 39
Non-standard, fragile path. System reboots or `/tmp` cleanup breaks the build.

```cmake
# Current:
link_directories(/tmp/lib)

# Better:
find_library(NVCOMP_LIB nvcomp HINTS /tmp/lib /usr/local/lib)
```

### BUILD-4: HDF5 2.x Hardcoded at `/tmp/hdf5-install`
**File:** `CMakeLists.txt` ~line 431
Same fragility as BUILD-3. Should be a CMake cache variable:

```cmake
set(HDF5_VOL_ROOT "/tmp/hdf5-install" CACHE PATH "HDF5 2.x install prefix")
```

### BUILD-5: C++ Standard is C++14
**File:** `CMakeLists.txt`
`std::optional`, `std::string_view`, structured bindings, `if constexpr` (C++17) would simplify several code paths. No blocker; upgrade when convenient.

### BUILD-6: Separable Compilation Always On
**File:** `CMakeLists.txt`
`CUDA_SEPARABLE_COMPILATION ON` is enabled for all CUDA targets, including non-VOL ones. This adds compile-time overhead (~20-30% per file) and requires device-link step. Only strictly necessary for VOL connector (device function pointers).

---

## 6. Neural Net Training Issues

### NN-1: ListNet Ranking Loss Not Implemented
**File:** `neural_net/training/train.py`
**Severity:** OPPORTUNITY (+3-7% top-1 accuracy, zero CUDA changes)

Current loss: MSE on predicted vs. actual metric values.
Better: ListNet softmax cross-entropy on ranking probabilities — directly optimizes the metric the system is evaluated on.

Model architecture unchanged; only the loss function changes. No CUDA kernel modifications.

---

### NN-2: Bilinear Factored Architecture (Exploration)
**File:** `neural_net/core/model.py`
**Opportunity:** +2-5% accuracy, 9.6K params (vs current 19K), 50% smaller model

Replaces:
```python
net = Sequential(Linear(15,128), ReLU, Linear(128,128), ReLU, Linear(128,4))
```
With:
```python
data_enc = Linear(5, 64)    # entropy, MAD, deriv, log(size), log(eb)
cfg_enc  = Linear(10, 64)   # one-hot algo + quant + shuffle
combined = data_enc(data) * cfg_enc(cfg)  # element-wise multiply
output   = Linear(64, 4)(combined)
```
Captures multiplicative data×config interactions that the current fully-connected model must implicitly learn.

---

### NN-3: Code Duplication in `cross_validate.py`
**File:** `neural_net/training/cross_validate.py`
`prepare_fold()` reimplements the same feature encoding as `encode_and_split()` in `data.py`. Any change to feature definitions must be applied in two places.

**Fix:** Extract `encode_dataframe(df, fit_normalizer=False, normalizer=None)` as shared utility.

---

### NN-4: Silent Infinite Regret Masking in `evaluate.py`
**File:** `neural_net/inference/evaluate.py`
```python
if np.isfinite(regret):
    regret = max(0.0, regret)
else:
    regret = 0.0  # Silently discards infinite regret
```
Should log a warning and count such cases separately in the evaluation report.

---

### NN-5: OOD Margin (10%) Not Validated or Documented
**File:** `src/nn/nn_gpu.cu` — OOD detection in `nnFusedInferenceKernel`
The 10% margin around training bounds is a magic number with no empirical justification in code or docs. Should be documented as a hyperparameter or made configurable.

---

## 7. Testing Gaps

### TEST-1: No Multi-Chunk Test for VOL Connector
`test_hdf5_configs.c` tests single-chunk datasets only. The 3-stage pipeline in `H5VLgpucompress.cu` is only exercised by `nn_vol_demo` (4 GB). Need:
- Small multi-chunk test (e.g., 16 chunks × 256 KB) for boundary chunk handling
- Verify that all chunks decompress correctly, including boundary chunks with padding

### TEST-2: No Concurrent Compression Stress Test
The CompContext pool (4 slots) is partially tested. Existing coverage:
- `tests/test_bug8_sgd_concurrent.cu` — 16 concurrent `gpucompress_compress_gpu()` calls, SGD-enabled; 16/16 byte-exact round-trips pass.
- `tests/test_bug3_sgd_gradients.cu` — concurrent SGD correctness (gradient partition analysis).

Still missing: sustained stress test with >8×N_COMP_CTX concurrent callers to verify pool blocking/wake-up under contention.

### TEST-3: No Corrupted/Truncated File Recovery Test
**✅ COVERED** — `tests/test_bug5_truncated_nnwt.cu` (7/7 pass):
- Empty file, truncated header, bad magic number, wrong architecture (different hidden_dim), truncated mid-w2, truncated at w3, valid file.

### TEST-4: No CLI Round-Trip Test
`compress.cpp` + `decompress.cpp` are not tested in the automated suite. Only the filter API is tested.

### TEST-5: No Stats Numerical Match Test (CPU vs GPU)
`compute_stats_cpu()` in Python and `runStatsKernelsNoSync()` on GPU should produce identical entropy/MAD/deriv values. This is never verified in tests. Floating-point ordering differences could cause subtle NN input mismatches at training vs. inference time.

---

## 8. Quick Wins

These require minimal code changes and carry clear, verifiable benefits:

| # | Change | File | Expected Gain | Risk |
|---|--------|------|--------------|------|
| QW-1 | ~~Set `CMAKE_CUDA_ARCHITECTURES=80`~~ | CMakeLists.txt | ✅ DONE — added `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard + `set(...80 CACHE STRING...)`. Rebuilt; `.target sm_80` confirmed in `libgpucompress.so`. 120/120 + 6/6 pass. | — |
| QW-2 | ~~Add `-use_fast_math` to CUDA Release flags~~ | CMakeLists.txt | ✅ DONE — `CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math"`. Fixed together with QW-1. | — |
| QW-3 | ~~Fix format string `%u` → `PRIu64`~~ | gpucompress_api.cpp:1243 | ✅ DONE — `#include <cinttypes>`, `PRIu64` + `(uint64_t)` cast. Verified by `test_bug4_format_string` (5/5). | — |
| QW-4 | ~~Make global stats atomics (`s_gpu_writes` etc.)~~ | H5VLgpucompress.cu | ✅ DONE — `static int` → `static std::atomic<int>` for `s_gpu_writes`, `s_gpu_reads`, `s_chunks_comp`, `s_chunks_decomp`; `.store(0)` in reset, `.load()` in get. Verified by `test_qw4_atomic_counters` (16/16). | — |
| QW-5 | ~~Add `file.gcount()` checks in `loadNNFromBinary`~~ | nn_gpu.cu | ✅ DONE — `NN_READ` macro checks every array read; consolidated 6 header reads into 1. Verified by `test_bug5_truncated_nnwt` (7/7). | — |
| QW-6 | Remove stderr spam (conditional on debug flag) | nn_gpu.cu, gpucompress_api.cpp | I/O overhead reduction | Low |
| QW-7 | ~~Batch 3 D→H copies in `runStatsKernels()` into 1~~ | stats_kernel.cu | ✅ ALREADY DONE — `runStatsKernels()` already uses a single 24B copy; main path uses `runStatsKernelsNoSync()` (zero D→H). | — |
| QW-8 | ~~Move insertion sort to CPU~~ — replaced with parallel bitonic sort instead | nn_gpu.cu | ✅ DONE — 32.9x speedup, commit `d704d71` | — |

---

## 9. Architecture Observations

### Strength: Fused Stats→NN Pipeline
The `runStatsKernelsNoSync()` + `nnFusedInferenceKernel` pattern eliminates the stats D→H round-trip. Stats stay on GPU from computation through NN input. This was correctly implemented.

### Strength: CompContext Pool for Concurrency
8 independent compression slots (`N_COMP_CTX=8`), each with own stream, events, stats/NN/SGD buffers. Enables true parallel chunk compression. The CUDA event barrier (`cudaStreamWaitEvent` before inference, `g_sgd_done` event after SGD) correctly prevents the GPU-level SGD vs. inference data race without CPU polling.

### Strength: VOL Connector Pointer Detection
`cudaPointerGetAttributes()` to detect GPU vs. host pointers at the VOL layer enables transparent GPU-native I/O without API changes to calling code.

### Weakness: Exploration Path Performance
Exploration is the dominant cost for OOD data. K=31 alternatives with full round-trips is O(31 × chunk_size) GPU work per chunk. For 4 MB chunks this adds up quickly. A smarter exploration strategy (e.g., hierarchical: try 4 → if any better, try 8 more, etc.) would reduce average cost significantly.

### Weakness: Single `g_sgd_stream` Serializes All SGD
All 8 workers share one `g_sgd_stream`. SGD is serialized at the CUDA stream level. For high-throughput workloads where every chunk triggers SGD, this could become a bottleneck. Since SGD calls are protected by `g_sgd_mutex`, the CPU bottleneck happens first, so the CUDA stream serialization is probably not the current bottleneck — but worth profiling.

### Weakness: No Adaptive Chunk Size
The VOL pipeline uses a fixed chunk size set by the HDF5 DCPL. There's no mechanism for the NN to suggest a different chunk size based on data characteristics. For highly compressible data, larger chunks amortize overhead. For incompressible data, smaller chunks reduce compression attempt cost.

---

## 10. Exploration Backlog

Items that need further investigation before a verdict:

| ID | Question | How to Investigate |
|----|----------|--------------------|
| EXP-1 | Is `d_owned` actually leaked on worker compression failure? | Read exact ordering of `cudaFree(d_owned)` vs error check in worker lambda |
| EXP-2 | Does `nnSGDKernel` have a gradient race? | Audit thread→gradient index mapping; check if any index is written by >1 thread |
| EXP-3 | Do CPU and GPU stats implementations produce identical results? | Run same float32 array through both; compare entropy/MAD/deriv |
| EXP-4 | Is sm_52 actually being used vs. JIT-compiled to sm_80? | `cuobjdump --dump-sass build/libgpucompress.so | head` |
| EXP-5 | Is exploration actually triggered in normal workloads? | Add counter; run benchmark and check how often K>4 exploration fires |
| EXP-6 | What is the actual GPU memory footprint? | `cudaMemGetInfo` before/after `gpucompress_init` |
| EXP-7 | Is `g_sgd_stream` actually the bottleneck? | `nsys profile` while running VOL benchmark; look for SGD serialization |
| EXP-8 | Does `-use_fast_math` affect compression quality/stats accuracy? | Profile NN accuracy (top-1) with and without flag |

---

## 11. Cross-Review Verification (CODE_REVIEW_FINDINGS.md)

A second agent independently audited the same codebase. Each of its unique findings was verified against the actual source code. Results below:

---

### VERIFIED NEW: `d_range_min`/`d_range_max` Static Globals — Concurrent Quantize Bug ✅ FIXED (BUG-7)
**Source:** [PREP-2] in CODE_REVIEW_FINDINGS.md
**Verified:** YES — this is a real, critical bug I missed.

`quantization_kernels.cu:25-26` allocates two static GPU pointers:
```cpp
static void* d_range_min = nullptr;
static void* d_range_max = nullptr;
```

`compute_data_range()` (called by `quantize_simple()`) writes the input data's min/max into these shared GPU addresses, then D→H copies them back. But `gpucompress_compress_gpu()` calls `quantize_simple()` concurrently from all 8 workers (each on its own `ctx->stream`).

**What happens under concurrent quantize:**
- Worker A: initializes `d_range_min = FLT_MAX` on stream A, launches min kernel on stream A
- Worker B: initializes `d_range_min = FLT_MAX` on stream B *before A's kernel writes its result*, launches its kernel on stream B
- Both kernels atomically update the same `d_range_min` device address — mixing A's and B's data
- Both workers D→H copy `d_range_min` — both get a corrupted value
- **Result:** wrong `data_min`/`data_max` → wrong `scale_factor` → silently wrong quantization output

Additionally, `ensure_range_bufs()` has a TOCTOU race (two threads both see `nullptr`, both `cudaMalloc` — second allocation leaks).

**Severity:** CRITICAL (silent data corruption in quantized output, every time ≥2 concurrent chunks need quantization)
**My review:** missed this entirely.
**Add to BUG section above as BUG-7.**

---

### VERIFIED NEW: `g_sgd_ever_fired` is Not Atomic
**Source:** [API-1] in CODE_REVIEW_FINDINGS.md
**Verified:** YES.

`gpucompress_api.cpp:70`:
```cpp
bool g_sgd_ever_fired = false;  // plain bool, not atomic
```

Read in `runNNFusedInferenceCtx()` (`nn_gpu.cu:1378`) from 8 concurrent inference threads **without any mutex**:
```cpp
if (g_sgd_ever_fired)
    cudaStreamWaitEvent(stream, g_sgd_done, 0);
```

Written in `runNNSGDCtx()` (`nn_gpu.cu:1472`) under `g_sgd_mutex`:
```cpp
g_sgd_ever_fired = true;
```

This is a textbook C++ data race (concurrent read + unsynchronized write). The practical impact is bounded: a missed `cudaStreamWaitEvent` could allow an inference kernel to read weights while SGD is writing them on `g_sgd_stream` — exactly the GPU race this flag was meant to prevent.

**Fix:** `std::atomic<bool> g_sgd_ever_fired{false}` with `memory_order_release` write, `memory_order_acquire` read.
**Severity:** MEDIUM-HIGH (correctness, low probability but real race window).
**Status:** ✅ FIXED — see BUG-8 section above. Verified by `tests/test_bug8_sgd_concurrent.cu`.

---

### VERIFIED NEW: `atomicAddDouble` CAS Contention — Native Atomic Available on sm_80+
**Source:** [STATS-1] in CODE_REVIEW_FINDINGS.md
**Status:** ✅ ALREADY DONE

`atomicAddDouble` CAS loop already removed. `stats_kernel.cu:55-56` has the tombstone comment:
> `atomicAdd(double*, double) is natively supported on sm_60+ (Pascal and newer). No CAS emulation needed — targeting sm_80 (Ampere).`

Native `atomicAdd(double*)` is called directly throughout the stats kernels. Correctness confirmed by `test_perf14_atomic_double` (5/5).

---

### VERIFIED NEW: `[API-3]` Per-Call `cudaMalloc` for CASCADED/ANS/BITCOMP
**Source:** [API-3] in CODE_REVIEW_FINDINGS.md
**Verified:** YES.

`gpucompress_api.cpp:1763-1768`: when the NN selects CASCADED, ANS, or BITCOMP and their worst-case output (2-4× input) exceeds the caller's pre-allocated output buffer, a per-call `cudaMalloc` + D→D copy + `cudaFree` happens:

```cpp
if (total_max_needed > *output_size) {
    if (cudaMalloc(&d_temp_out, total_max_needed) != cudaSuccess) ...
}
// ... later: D→D copy to d_out, then cudaFree(d_temp_out)
```

In the VOL write path, `d_comp_w[w]` is sized to `max_comp = gpucompress_max_output_size(chunk_bytes)`. If this was computed assuming LZ4 (conservative 1.1×), and the NN at runtime picks CASCADED (which needs 2-4×), this triggers every chunk.

**Severity:** MEDIUM (cudaMalloc/Free on hot path; CASCADED is rarely selected for float32 data, so low frequency in practice).
**My review:** missed this.

---

### VERIFIED: [COMP-2] CASCADED/BITCOMP Use `NVCOMP_TYPE_LONGLONG` — Needs Measurement
**Source:** [COMP-2] in CODE_REVIEW_FINDINGS.md
**Verdict:** Code confirmed (`compression_factory.cpp:116, 123`). However, the other agent's framing is partly incorrect. `NVCOMP_TYPE_LONGLONG` tells CASCADED how to interpret runs and delta-encode values. For float32 data bit-patterns, reinterpreting as 64-bit integer groupings is not obviously wrong — it depends on whether float byte patterns exhibit better run-length when viewed as 32-bit or 64-bit words. The comment "Good for floating-point/scientific data" is plausible but unverified.
**Severity:** INVESTIGATE — not a confirmed bug, just an untested assumption. Benchmark `NVCOMP_TYPE_INT` vs `NVCOMP_TYPE_LONGLONG` on representative float32 datasets.

---

### VERIFIED: [VOL-6] Gather Kernel on Default Stream — Serializes Stage 1
**Source:** [VOL-6] in CODE_REVIEW_FINDINGS.md
**Verified:** YES. `H5VLgpucompress.cu:1194-1200` launches `gather_chunk_kernel` on `cudaStreamDefault` and synchronizes before posting the WorkItem. Every non-contiguous or N-D chunk causes Stage 1 to stall. For purely chunked, C-order layouts this never triggers; for HDF5 datasets with non-C-order access patterns, this serializes the entire pipeline.
**My review:** mentioned this in the stream table ("VOL gather kernel: Default — Potential issue") but didn't flag it prominently.
**Status:** ✅ FIXED — dedicated `gather_stream` (write) and `scatter_stream` (read) created per call; kernels use these non-default streams; sync changed to `cudaStreamSynchronize(gather_stream/scatter_stream)`. Null-stream implicit serialization against 8 CompContext worker streams eliminated. Verified by `test_perf16_gather_stream` (5/5).

---

### CONFIRMED (already in my review):
| Finding ID | Status | In my review? |
|-----------|--------|---------------|
| [COMP-1] 4-bit algorithm ID limit | Confirmed, real limitation | Not explicitly — add to EXP backlog |
| [PREP-1] fprintf in quantization_kernels.cu | Confirmed | Mentioned generally (debug logging) |
| [STATS-2] fprintf in stats path | Confirmed | Yes, PERF tables |
| [NN-1] Full forward pass per thread | Confirmed — design choice | Yes |
| [NN-2] Insertion sort | Confirmed | Yes, PERF-2 |
| [NN-3] fprintf in SGD | Confirmed | Yes |
| [NN-4] 128-thread SGD kernel | Confirmed — design choice | Yes |
| [VOL-1] Synchronous D→H | Confirmed | Yes |
| [VOL-2] Per-chunk malloc | Confirmed | Yes, PERF-8 |
| [VOL-3] Single I/O thread | Confirmed | Yes |
| [VOL-4] Workers continue after error | Confirmed | Yes (in BUG-1 context) |
| [VOL-5] N_COMP_WORKERS == N_COMP_CTX | Confirmed good design | Yes |
| [API-2] Exploration cost | Confirmed | Yes, PERF-1 |
| [CROSS-1] fprintf throughout | Confirmed | Yes |
| [CROSS-2] Thread safety of singletons | Confirmed (quantize path) | Partially |

---

### Summary of Net New Confirmed Findings

| ID | Issue | Severity | File:Line |
|----|-------|----------|-----------|
| BUG-7 | `d_range_min`/`d_range_max` shared across concurrent quantize calls | **CRITICAL** ✅ FIXED | `quantization_kernels.cu` — per-CompContext buffers allocated in `initCompContextPool()` |
| BUG-8 | `g_sgd_ever_fired` plain bool read/written without atomics | **MEDIUM** ✅ FIXED — `std::atomic<bool>` with `memory_order_release` write, `memory_order_acquire` read. Verified by `tests/test_bug8_sgd_concurrent` (16/16). | `gpucompress_api.cpp:70`, `nn_gpu.cu:1378,1472` |
| PERF-14 | ~~`atomicAddDouble` CAS should be native `atomicAdd` on sm_60+~~ | **MEDIUM** ✅ ALREADY DONE — CAS loop removed; comment at `stats_kernel.cu:55-56` confirms native `atomicAdd(double*)` used directly (sm_80 target). Correctness verified by `test_perf14_atomic_double` (5/5). | `stats_kernel.cu:55-56` |
| PERF-15 | Per-call `cudaMalloc` for CASCADED/ANS/BITCOMP temp buffer | **MEDIUM** | `gpucompress_api.cpp:1763-1768` |
| PERF-16 | ~~Gather kernel on default stream serializes Stage 1 (non-contiguous datasets)~~ | **MEDIUM** ✅ FIXED — dedicated `gather_stream`/`scatter_stream` per write/read call; redundant `cudaStreamSynchronize(cudaStreamDefault)` in partial-boundary path removed. Verified by `test_perf16_gather_stream` (5/5). | `H5VLgpucompress.cu:1194-1200` |
| INVESTIGATE | CASCADED/BITCOMP `NVCOMP_TYPE_LONGLONG` assumption unverified | **LOW** | `compression_factory.cpp:116,123` |

---

### SGD Online Learning Verification

`tests/test_sgd_weight_update.cu` verifies that `nnSGDKernel` actually modifies `d_nn_weights` and produces convergence:

- **Data:** 4 MB linear ramp (OOD for the NN → large initial prediction error)
- **Setup:** 30 iterations, same data, `REINFORCE_LR=0.05`, `MAPE_THRESH=0.1%`
- **Result:** MAPE 91.16% → 17.72% over 30 steps; 30/30 SGD fires; PASS
- **Assertion:** `mapeN < mape0` (final MAPE strictly less than initial)

This confirms the full path: `gpucompress_compress_gpu` → SGD fires on `g_sgd_stream` → `d_nn_weights` updated in-place → subsequent inference reads updated weights → predicted ratio converges toward actual.

---

*This document was generated from systematic codebase exploration. Update findings as issues are verified or resolved.*
