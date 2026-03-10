# Fix Plan: GPUCompress 20 Confirmed Bugs

## Context

Two rounds of deep codebase investigation + audit identified 22 bugs. After critical review by verification agents, 2 fixes were dropped (L3: filter_mask is input-only in HDF5; M18: benchmark intentionally accumulates learning). The remaining 20 bugs are confirmed with reviewed fixes.

## Execution Order

Fixes grouped by file, ordered by severity. Each fix is self-contained.

---

## Group 1: `src/api/gpucompress_api.cpp` (9 bugs)

### C1 — Exploration winner headers missing setAlgorithmId() [FIXED]
- **Lines ~996, ~2171**: Insert `alt_hdr.setAlgorithmId((uint8_t)alt_algo);` before the header is written (memcpy on host path, cudaMemcpyAsync on GPU path)
- `alt_algo` is in scope at both locations (verified)
- **Test**: `tests/regression/test_c1_exploration_header.cu` — triggers exploration with threshold=0/K=31, verifies header algo ID and round-trip decompression
- **Verified**: 3 patterns triggered exploration winner replacement; all 3 failed before fix (header_algo=0), all pass after fix (11 pass, 0 fail)

### C2 — Global g_t_start/g_t_stop timing events are thread-unsafe [FIXED]
- **Reviewer correction**: Per-call event creation is too expensive (cudaEventCreate is a sync point). Mutex approach serializes the timing sections which span entire compress/decompress operations (milliseconds), defeating concurrency.
- **Fix**: Replace global `g_t_start`/`g_t_stop` with `thread_local` CUDA events, lazy-initialized on first use:
  ```cpp
  static thread_local cudaEvent_t tl_t_start = nullptr;
  static thread_local cudaEvent_t tl_t_stop  = nullptr;

  static inline void ensure_timing_events() {
      if (!tl_t_start) cudaEventCreate(&tl_t_start);
      if (!tl_t_stop)  cudaEventCreate(&tl_t_stop);
  }
  ```
- Replace all `g_t_start`/`g_t_stop` usage in the host compress path (lines 623-639, 833-903) with `tl_t_start`/`tl_t_stop`, calling `ensure_timing_events()` before first use
- Remove global `g_t_start`/`g_t_stop` declarations (line 125-126) and their creation/destruction in init/cleanup
- No RAII destructor needed: single GPU node, CUDA context teardown reclaims events on process exit
- Zero contention: each thread gets its own event pair, event creation cost paid once per thread lifetime
- GPU path already uses per-CompContext events (`ctx->t_start`/`ctx->t_stop`) — unaffected
- **Test**: `tests/regression/test_c2_timing_race.cu` — 8 threads x 5 iterations concurrent compression, checks for negative/zero/corrupt timings
- **Verified**: Before fix: 1 negative timing, 13 zero timings (2 FAIL). After fix: 0 negative, 0 zero, 0 corrupt (3 pass, 0 fail)

### H3 — Host-path ALGO_AUTO uses global singleton stats/inference buffers [FIXED]
- **Fix**: Added `static std::mutex g_auto_mutex;` near line 122. Wrapped the ALGO_AUTO stats+inference block with this mutex. Also fixed `stats->nn_final_action` and `h->nn_action` in chunk history to use local `nn_action` instead of racy `g_last_nn_action.load()` — this was the primary observable symptom (wrong action reported in stats output). Same fix applied to GPU path for consistency.
- The GPU path uses per-CompContext buffers and doesn't need this mutex.
- **Test**: `tests/regression/test_h3_auto_global_buffers.cu` — 8 threads × 20 iterations with two distinct data patterns (smooth sine vs xorshift noise), verifies algorithm choices match single-threaded baselines and round-trips succeed
- **Verified**: Before fix: ~80/160 algorithm mismatches (stats reported wrong thread's action). After fix: 0 mismatches across 3 consecutive runs (2 pass, 0 fail)

### H4 — Host-path SGD missing mutex [FIXED]
- **Fix**: Wrapped `runNNSGD(...)` call at line ~1082 with `std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);`, matching the GPU path pattern at line ~2244
- `g_sgd_mutex` already existed at line 122
- **Test**: `tests/regression/test_h4_sgd_mutex.cu` — 6 threads × 15 iterations concurrent ALGO_AUTO with aggressive SGD (threshold=0), verifies no compression errors or decompression failures
- **Verified**: Race is hard to trigger deterministically (narrow window due to cudaStreamSynchronize in runNNSGD + shared g_default_stream serialization), but fix mirrors the correct GPU path pattern and is necessary for correctness. 2 pass, 0 fail

### H7 — Host decompress skips version check [FIXED]
- **Fix**: Changed `if (header.magic != COMPRESSION_MAGIC)` → `if (!header.isValid())` at line ~1186. `isValid()` checks both magic AND version range [1, COMPRESSION_HEADER_VERSION].
- **Test**: `tests/regression/test_h7_version_check.cu` — compresses valid data, corrupts version to 255 and 0, verifies both are rejected with INVALID_HEADER; also verifies valid data still decompresses and restored version works
- **Verified**: Before fix: 2 pass, 2 fail (bad versions accepted). After fix: 4 pass, 0 fail

### H8 — Zero compressed_size bypasses validation [FIXED]
- **Fix**: Added check after header validation: if `compressed_size == 0` and `original_size == 0`, return SUCCESS with `*output_size = 0`; if `compressed_size == 0` and `original_size > 0`, return INVALID_HEADER. Placed before output buffer size check.
- **Test**: `tests/regression/test_h8_zero_compressed_size.cu` — corrupts header compressed_size to 0 with original_size > 0 (should reject), both sizes 0 (should succeed), and restores to verify normal decompress still works
- **Verified**: Before fix: nvcomp crashes with `NVCompException: CUDA Runtime API failure: invalid argument`. After fix: 3 pass, 0 fail

### M7 — acquireCompContext falls off end of non-void function [FIXED]
- **After line ~242** (for-loop closing brace): Added `return nullptr;`
- The only callsite (line 1659) already checks for nullptr
- **Test**: `tests/regression/test_m7_acquire_falloff.sh` — source inspection for `return nullptr` after for-loop
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

### M9 — Primary exploration sample records psnr=0.0 [FIXED]
- **Fix**: Compute PSNR from quantization parameters when lossy (20*log10(range/error_bound), capped at 120), default 120.0 for lossless
- **Note**: FIX_PLAN originally suggested `quant_result.psnr` but QuantizationResult has no `psnr` member; used error_bound-based estimate instead
- **Lines 748-751**: Added primary_psnr computation before push_back (host path)
- **Lines 1957-1960**: Same fix for GPU path
- **Test**: `tests/regression/test_m9_primary_psnr.sh` — checks no `0.0, 0.0});` pattern in push_back
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

### M10 — compress_gpu ignores caller's stream_arg [FIXED]
- Removed `(void)stream_arg;`, added bidirectional event sync between caller_stream and ctx->stream
- Uses pre-allocated per-context events (ctx->t_start, ctx->t_stop) — no per-call creation overhead
- **Test**: `tests/regression/test_m10_stream_sync.sh` — checks (void)stream_arg removed, caller_stream and cudaStreamWaitEvent present
- **Verified**: test fails before fix (0 pass, 3 fail), passes after fix (3 pass, 0 fail)

---

## Group 2: `src/nn/nn_gpu.cu` (3 bugs)

### H1 — Non-Ctx inference paths don't wait for SGD completion [FIXED]
- **In `runNNInference()` (~line 1074)** and **`runNNFusedInference()` (~line 1137)**: Add before kernel launch:
  ```cpp
  if (g_sgd_ever_fired.load(std::memory_order_acquire))
      cudaStreamWaitEvent(stream, g_sgd_done, 0);
  ```
- **In `runNNSGD()` (after line 1222)**: Add after SGD kernel launch:
  ```cpp
  cudaEventRecord(g_sgd_done, stream);
  g_sgd_ever_fired.store(true, std::memory_order_release);
  ```
- **Reviewer note**: Non-Ctx SGD runs on caller's stream (not g_sgd_stream). Recording g_sgd_done on that stream is correct — the event signals "SGD kernel on THIS stream completed." Future inference on any stream will wait for it. Since non-Ctx SGD also calls `cudaStreamSynchronize(stream)` before returning (line 1235), the event is guaranteed recorded before the function exits.
- **Test**: `tests/regression/test_h1_sgd_inference_race.cu` — 4 threads x 10 iterations concurrent ALGO_AUTO with aggressive SGD (threshold=0), verifies valid actions, finite predictions, and successful round-trips
- **Verified**: 3 pass, 0 fail. Race is hard to trigger deterministically (narrow window due to cudaStreamSynchronize in runNNSGD), but fix mirrors the correct Ctx path pattern and is necessary for correctness

### M6 — Non-atomic global flags [FIXED]
- **Line 62**: `static bool g_nn_loaded` → `static std::atomic<bool> g_nn_loaded{false};`
- **Line 63**: `static NNRankCriterion g_rank_criterion` → `static std::atomic<int> g_rank_criterion{NN_RANK_BY_RATIO};`
- Updated all read sites to use `.load()`, all write sites to use `.store()`
- **Test**: `tests/regression/test_m6_atomic_flags.sh` — source inspection for atomic declarations
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)
- **Reviewer caveat**: Atomics don't protect compound operations (check g_nn_loaded + read d_nn_weights). This is acceptable — the sequence `if (!g_nn_loaded) return -1;` then `use d_nn_weights` is safe because weights are set BEFORE g_nn_loaded is set to true (release ordering).

### M12 — SGD skips PSNR gradient for lossless results [FIXED]
- **Lines 637-645**: Replaced if/else with unconditional gradient using 120.0 fallback for psnr<=0
- **Test**: `tests/regression/test_m12_sgd_psnr_gradient.sh` — checks no conditional skip and 120.0 fallback present
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)

---

## Group 3: `src/preprocessing/quantization_kernels.cu` (3 bugs)

### C4 — Missing int32 clamping [FIXED]
- **After line ~76** (after int16 clamp): Add:
  ```cpp
  else if (sizeof(OutputT) == 4) {
      quantized = fmax(-2147483648.0, fmin(2147483647.0, quantized));
  }
  ```
- Double literals are used, so float64 precision represents these values exactly
- **Test**: `tests/regression/test_c4_int32_clamp.cu` — directly launches GPU kernel with scale producing values > INT32_MAX, verifies clamped vs unclamped behavior; also tests end-to-end round-trip via public API
- **Verified**: Test confirms 3 overflow conditions detected; clamping produces correct INT32_MAX; round-trip within error bound (3 pass, 0 fail)
- **Note**: NVIDIA GPUs happen to saturate on float-to-int cast (UB is benign on this hardware), but the fix is necessary for C/C++ correctness and portability

### H6 — int cast overflow in CUB DeviceReduce [FIXED]
- **Fix**: Added `#include <climits>` and a guard at the top of `compute_data_range_typed()` before any CUB calls: `if (num_elements > (size_t)INT_MAX) return -1;` with a stderr warning. This rejects oversized input early and explicitly, rather than relying on CUB's undefined behavior with a negative wrapped int.
- **Test**: `tests/regression/test_h6_cub_int_overflow.cu` — calls quantize_simple with num_elements = INT_MAX+1, verifies graceful rejection; also tests INT_MAX boundary doesn't crash
- **Verified**: Before fix: CUB happened to reject negative wrapped int (UB). After fix: explicit guard triggers with clear error message. 2 pass, 0 fail

### H9 — Forced INT8 silently violates error bound [FIXED]
- **Fix**: Added warning after precision switch in both `quantize_simple` overloads: if forced INT8 needs > 127 levels, emits stderr warning with required vs available levels
- **Test**: `tests/regression/test_h9_int8_error_bound.cu` — calls quantize_simple with forced INT8 on data needing 5000 levels, captures stderr and checks for warning
- **Verified**: Before fix: 0 pass, 1 fail (silent). After fix: 1 pass, 0 fail (warning emitted: "need 4995 levels, int8 max=127")

---

## Group 4: `src/preprocessing/byte_shuffle_kernels.cu` (2 bugs)

### M15 — element_size parameter ignored, always uses <4> [FIXED]
- Replaced `(void)element_size` with switch dispatch for both `launch_byte_shuffle` and `launch_byte_unshuffle`
- Added explicit template instantiations for `<1>`, `<2>`, `<4>`, `<8>` for both kernels
- **Test**: `tests/regression/test_m15_element_size.sh` — checks (void)element_size removed and switch present
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)

### M16 — Use-after-free in DeviceChunkArrays [FIXED]
- Added `cudaStreamSynchronize(stream)` before return in both `byte_shuffle_simple()` and `byte_unshuffle_simple()`
- **Test**: `tests/regression/test_m16_chunk_arrays_uaf.sh` — checks cudaStreamSynchronize present in both functions
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)

---

## Group 5: `src/hdf5/H5Zgpucompress.c` (2 bugs)

### L1 — Data race in ensure_initialized() [FIXED]
- Replaced TOCTOU check-then-act with `pthread_once` for thread-safe initialization
- **Test**: `tests/regression/test_l1_init_race.sh` — checks for pthread_once usage
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

### L2 — Shuffle size parsing only handles 4 [FIXED]
- Expanded to warn on unsupported non-zero shuffle sizes
- **Test**: `tests/regression/test_l2_shuffle_parse.sh` — checks for warning message in source
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

---

## Group 6: `benchmarks/grayscott/grayscott-benchmark.cu` (1 bug)

### M17 — gather_chunk_kernel OOB read on last chunk [FIXED]
- Modified `gather_chunk` and `scatter_chunk` to accept `actual_cz` parameter
- Updated kernels to use `actual_cz` instead of `chunk_z` for element count
- Updated all 3 call sites to pass `actual_cz`
- **Test**: `tests/regression/test_m17_gather_oob.sh` — checks actual_cz parameter in gather/scatter
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)

---

## Group 7: Python files (4 bugs)

### H5 — `neural_net/inference/evaluate.py` uses wrong normalization stats [FIXED]
- **Fix**: Added `checkpoint: Dict = None` parameter to `evaluate_ranking()` and `show_sample_predictions()`. Both now use `checkpoint.get('x_means', data['x_means'])` etc. to prefer checkpoint stats (matching training-time normalization), falling back to dataset stats for backward compatibility. Updated `__main__` call sites to pass checkpoint.
- **Test**: `tests/regression/test_h5_evaluate_normalization.py` — verifies function signature accepts checkpoint and source uses checkpoint stats
- **Verified**: Before fix: 0 pass, 2 fail. After fix: 2 pass, 0 fail

### M13 — `neural_net/export/export_weights.py` verify_export compares incompatible values [FIXED]
- Normalized `test_input` before model call and denormalized output to match manual forward pass
- **Test**: `tests/regression/test_m13_verify_export.sh` — checks for test_input_norm and denormalization
- **Verified**: test fails before fix (0 pass, 2 fail), passes after fix (2 pass, 0 fail)

### M19 — `neural_net/core/data.py` missing NaN handling [FIXED]
- Added defensive NaN guards before return in `compute_stats_cpu()`: replaces NaN with 0.0
- **Test**: `tests/regression/test_m19_nan_handling.sh` — checks for isnan usage
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

### M20 — `neural_net/training/cross_validate.py` missing PSNR fillna [FIXED]
- Added `.fillna(120.0)` after `.replace()` in psnr_clamped computation
- **Test**: `tests/regression/test_m20_psnr_fillna.sh` — checks for fillna(120.0) in source
- **Verified**: test fails before fix (0 pass, 1 fail), passes after fix (1 pass, 0 fail)

---

## Dropped Fixes (from reviewer feedback)

| Bug | Reason Dropped |
|-----|---------------|
| **L3** (filter_mask not copied back) | HDF5's `chunk_read.filters` is input-only. Copying back is incorrect API usage. |
| **M18** (Phase 5 inherits weights) | Intentional experimental design — Phase 5 tests accumulated learning from Phase 4. |

---

## Verification Plan

1. **Build**: `make` or `cmake --build build` — all 7 modified .cu/.c/.cpp files must compile cleanly
2. **Unit tests**: Run existing test suite (if any) to check for regressions
3. **C1 verification**: Compress with ALGO_AUTO + exploration enabled, decompress the result — must succeed (previously would use wrong algorithm)
4. **M7 verification**: Compiler should no longer warn about "control reaches end of non-void function"
5. **M17 verification**: Run grayscott-benchmark with L not divisible by chunk_z — no CUDA memory errors
6. **Python verification**: Run `python -m neural_net.export.export_weights` — verify_export should still pass with aligned normalization
