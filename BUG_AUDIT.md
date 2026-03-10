# GPUCompress Bug Audit Report

**Date:** 2026-03-10
**Scope:** Full codebase audit across 6 areas — compression/decompression, NN/SGD, preprocessing, HDF5 VOL, benchmark methodology, and Python training pipeline.
**Method:** Initial automated sweep followed by deep-dive investigation with 6 parallel agents confirming each finding against source code.

---

## Confirmed Bugs (Fixed)

### 1. Memory leak on NN failure — `src/api/gpucompress_api.cpp:540`
- **Severity:** High
- **Issue:** `d_input` GPU buffer allocated at line 421 but not freed when NN inference fails and returns `GPUCOMPRESS_ERROR_NN_NOT_LOADED`.
- **Fix:** Added `cudaFree(d_input)` before the early return.

### 2. TOCTOU race in `gpucompress_get_chunk_diag` — `src/api/gpucompress_api.cpp:1567`
- **Severity:** High
- **Issue:** `g_chunk_history_count` checked *outside* the mutex, then struct copied *inside* the mutex. A concurrent writer could be mid-write to that index between the check and the copy.
- **Fix:** Moved mutex acquisition before the count check so the entire read is atomic.

### 3. `allocSGDBuffers` returns void — `src/nn/nn_gpu.cu:797`
- **Severity:** High
- **Issue:** Function returns `void`, so callers can't detect `cudaMalloc` failure. The caller at line 1226 already checks `d_sgd_grad_buffer == nullptr` after the call, but the init-time caller at line 966 does not.
- **Fix:** Changed return type to `bool` (returns `false` on any allocation failure).

### 4. `effective_eb` vs `error_bound` mismatch — `src/preprocessing/quantization_kernels.cu:477`
- **Severity:** Medium
- **Issue:** `QuantizationResult.error_bound` stored the original `config.error_bound`, but quantization actually used `effective_eb` (adjusted for float precision, int32 overflow limits). Dequantization works correctly (uses `scale_factor`), but any code using `error_bound` for validation/PSNR gets the wrong value.
- **Fix:** Changed `result.error_bound = config.error_bound` to `result.error_bound = effective_eb`.

### 5. `io_done_flag` data race — `src/hdf5/H5VLgpucompress.cu:1047`
- **Severity:** Low-Medium
- **Issue:** Declared as plain `bool` but accessed across threads. Currently protected by `io_mtx` in all access sites, but fragile — any future access outside the mutex would be undefined behavior.
- **Fix:** Changed to `std::atomic<bool>`.

---

## False Positives (Investigated and Dismissed)

### Compression/Decompression
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Division by zero when `compressed_size == 0` | False positive | nvcomp never returns 0 for valid compression; minor defensive concern at most |
| Missing `cudaStreamSynchronize` before reading compressed size | False positive | `cudaEventSynchronize(tl_t_stop)` provides implicit sync; explicit sync at line 729 before D->H copy |
| Integer overflow in `n_chunks * chunk_size` | False positive | All operations use `size_t` on 64-bit; no actual overflow path exists |

### NN Inference & SGD
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| `log1pf()` domain violation producing NaN | False positive | All inputs (ratio, times) are physically positive; `log1pf(x)` valid for x > -1 |
| Race condition in SGD weight updates | False positive | Perfect thread partitioning — each of 128 threads owns distinct weight indices (verified by test) |
| Gradient clipping divide-by-zero | False positive | Guard `norm > 1.0f` prevents division; norm=0 uses identity scaling `clip_scale = 1.0f` |

### Preprocessing
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Static buffer race in `quantize()` | False positive | Thread-safe overload exists with per-slot buffers (`ctx->d_range_min/max`); API correctly uses it |

### HDF5 VOL Plugin
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Worker pool deadlock on error | False positive | Errors caught via `worker_err`; sentinels guarantee orderly shutdown |
| Prefetch thread deadlock on early exit | False positive | Explicit cleanup path at `done_read` resets semaphore and unblocks thread |
| Sentinel shutdown deadlock | False positive | Proper condition variable signaling after each sentinel push |

### Benchmark Methodology
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Missing `cudaDeviceSynchronize` before write timer stop | False positive | VOL's `H5Dwrite` joins all worker threads before returning (synchronous) |
| GPU cache warmup bias across phases | Minor concern | Negligible for I/O-bound storage benchmarking; phases process different compression configs |
| Throughput calculation ambiguity | False positive | Intentional design: wall-clock for throughput, GPU-time for breakdown; correctly labeled |

### Python Training Pipeline
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| `std()` uses ddof=0 instead of ddof=1 | False positive | Large N (100K+ rows) makes difference negligible (<0.05%); consistent across train/inference |

---

## Additional Findings (No Issues)

The Python training pipeline deep-dive also verified:
- No data leakage between train/val splits (correctly splits by file)
- Loss function (MSELoss) correctly applied to normalized targets
- Weight export format matches C++ loader exactly (verified field-by-field)
- Feature normalization consistent between Python training and CUDA inference
- SGD target normalization in active learning matches training normalization

---

## Summary

**5 confirmed bugs fixed, 14 false positives dismissed.**

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | Memory leak on NN failure path | High | Fixed |
| 2 | TOCTOU race in chunk_diag reader | High | Fixed |
| 3 | allocSGDBuffers silent failure | High | Fixed |
| 4 | effective_eb vs error_bound mismatch | Medium | Fixed |
| 5 | io_done_flag not atomic | Low-Med | Fixed |
