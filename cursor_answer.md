# GPUCompress: verified findings (HDF5 metrics + NN / snapshots)

This file records **code-backed** conclusions from independent audits. Claims below were re-checked against the repository; line numbers refer to the current tree.

---

## Part A — HDF5 file size vs per-chunk “compressed sum”

### Symptom

**HDF5 output file sizes** from `get_file_size` can look **much smaller** than **sums inferred from per-chunk diagnostics**, producing extreme compression ratios (e.g. hundreds×) for float32 scientific data.

### High-level conclusion

On the **successful VOL write path**, compression workers and the **I/O thread are fully joined** before `H5Dwrite` returns; **`H5Fclose` is not returning “early”** relative to that pipeline. The mismatch is **primarily diagnostic / accounting**, not silent truncation of chunks on disk.

When **online learning** is on and **exploration** replaces the primary compression with a **smaller winner**, chunk history **`actual_ratio`** is still derived from the **primary** payload (`primary_compressed_size`) for MAPE semantics, **not** from the **final** payload written. Inverting `actual_ratio` to estimate bytes on disk **overstates** compressed size vs the real file.

**Follow-up in code:** `gpucompress_chunk_diag_t` now includes **`final_ratio`**, computed from **`d.compressed_size`** (post-exploration) in `recordChunkDiagnostic` — use it when accounting must match **bytes written**. `actual_ratio` remains **primary-oriented** when `primary_compressed_size > 0`.

### Checks performed

| Question | Verdict | Where |
|----------|---------|--------|
| Same temp path per field? | Yes; `remove` + `H5F_ACC_TRUNC` each iteration | `benchmarks/sdrbench/generic-benchmark.cu` **108–109**, **1756–1760** |
| `H5Fclose` before I/O done? | No on success path; workers joined, then I/O thread joined | `src/hdf5/H5VLgpucompress.cu` **1649–1653**, **1671–1684** |
| `get_file_size` races I/O? | Not in benchmark order: read/close then stat | `generic-benchmark.cu` **225–231**, **1799–1817** |
| Field index pattern 5,7,11,…? | No index bug; data-dependent exploration + wrong metric | — |
| Winner copy vs I/O? | Stream sync before freeing exploration slots | `gpucompress_compress.cpp` **763–823** |
| `write_chunk_to_native` size | Uses post-compress `comp_sz` from worker | `H5VLgpucompress.cu` **1252–1318** |

### Root mechanism (`actual_ratio` vs disk)

- **`gpucompress_compress.cpp` 506–514:** saves **`primary_compressed_size`** before exploration overwrites output (for MAPE vs NN primary).
- **`gpucompress_diagnostics.cpp` 133–141:** `primary_sz` drives **`actual_ratio`** (primary when `primary_compressed_size > 0`); **`final_ratio`** uses **`d.compressed_size`** (post-exploration, aligned with bytes written). Both ratios use **`min(100, input_size / …)`**.
- Public struct comment in `include/gpucompress.h` may still read like generic `input_size / compressed_size` for **`actual_ratio`**; **`final_ratio`** is the field meant for **on-disk** ratio accounting.
- **`std::min(100, …)`** still caps stored ratios; very large true ratios distort naive inversion.

### Verification hint

If **`do_verify`** and **`gpu_compare`** report **no mismatches**, the small file matches what was written and read back; inflated “chunk sums” from **`actual_ratio`** are then a **metric** issue, not missing HDF5 payload.

### Separate: VOL error-path UB

`H5VLgpucompress.cu` **1347** (`goto done_write` after I/O + workers start, on `cudaStreamCreate` failure) can skip worker join → **undefined behavior** on failure paths only.

---

## Part B — NN load, weight snapshots, VPIC (skeptically verified)

### B1 — `loadNNFromBinary()` partial publish after swap

**Confirmed (with correct line refs).** The other audit cited **1449**; that line is a **comment**. The real sequence is:

1. Allocate **`d_new`**, H→D copy (failures **return before** swapping — OK).
2. Under **`g_nn_ptr_mutex`:** `d_nn_weights = d_new` (**1472–1475**).
3. **`g_nn_loaded.store(true)`** (**1476**), outside the mutex.
4. Optionally sync comp/SGD streams and **`cudaFree(d_old)`** (**1478–1482**).
5. **`allocInferenceBuffers() || allocFusedInferenceBuffers()`** (**1485–1488**). On failure: **`return false`** with **no** rollback of `d_nn_weights`, **no** `g_nn_loaded = false`.

**Effects:** `isNNLoaded()` can be **true** while fused/non-fused inference buffers are incomplete; library is **inconsistent**.

**“Leaks GPU memory” (nuance):** The **weight** buffer `d_new` is not orphaned — it **is** `d_nn_weights`. True **leaks** can still come from **`allocInferenceBuffers` / `allocFusedInferenceBuffers`** (**`nn_gpu.cu` 74–87, 486–498**): either helper can **fail after one successful `cudaMalloc`** in that function **without freeing** the first allocation, leaving a persistent leak until exit.

### B2 — Snapshot APIs vs GPU streams

**Confirmed.** `gpucompress_nn_save_snapshot`, `restore`, and `*_device` (**`nn_gpu.cu` 1541–1587**) only hold **`g_nn_ptr_mutex`** and call **`cudaMemcpy`** on **`d_nn_weights`**. That mutex does **not** serialize **in-flight kernels** on **`g_sgd_stream`** or compression streams. **Callers** must **`cudaDeviceSynchronize`**, **`syncAllCompContextStreams`**, and/or **`cudaStreamSynchronize(g_sgd_stream)`** if they need a consistent copy.

### B3 — VPIC restore/save (**1366**, **1828**) and “active race”

**Partially overstated.** VPIC runs **`cudaDeviceSynchronize()`** immediately after **`H5Dwrite`** (**`vpic_benchmark_deck.cxx` ~1486**) and again after **`H5Dread`** (**~1556**). **`cudaDeviceSynchronize`** waits for **all** device streams, including **`g_sgd_stream`**, for work already queued. So for the **single-threaded main loop**, weight updates from the **timed compress/write** path are usually **finished** before later host work; **explicit** `cudaStreamSynchronize(g_sgd_stream)` in the deck is often redundant **if** no later async weight use is queued without another sync.

**Still valid:** (1) snapshot **APIs** are unsafe **by themselves**; (2) any future path that touches **`d_nn_weights`** asynchronously **after** the last sync and **before** save could reintroduce a race; (3) blanket claims that VPIC **proves** nondeterministic snapshots **without** citing the device syncs are **too strong**.

The Kendall ranking profiler uses **`gpucompress_compress_gpu`** with **fixed** `algorithm`, not **AUTO** (`kendall_tau_profiler.cuh`), so it is an unlikely source of extra online-learning SGD on **`g_sgd_stream`** after the read sync.

### B4 — `releaseCompContext()` and stream quiescence

**Confirmed as fragile contract.** **`releaseCompContext`** (`gpucompress_pool.cpp` **181–187**) only marks the pool slot free; it does **not** **`cudaStreamSynchronize(ctx->stream)`**. **`syncAllCompContextStreams`** is separate (**189–197**). Risk is **future** misuse, not a demonstrated bug in every current caller.

### B5 — Misleading comment in `gpucompress_compress_gpu` (~115)

**Confirmed false.** The text at **115–118** says the inference context is **released before delegating**. **`ContextGuard infer_guard`** (**85–87**) lives until **`gpucompress_compress_gpu` returns**, so the destructor runs **after** **`gpucompress_compress_with_action_gpu`** completes (**120–123**). The **infer slot stays acquired for the entire delegation** — correctly described at **80–83**, not at **115**.

### B6 — Test coverage

**Confirmed gap.** **`tests/nn/test_nn.cu`** **`test_load_reload_failure`** covers **failed reload** (missing file), **not** success through the **weight swap** followed by failure at **1485–1488**.

### B7 — Items verified sound

- **`runNNFusedInferenceCtx`:** **`cudaStreamWaitEvent(stream, g_sgd_done, …)`** when **`g_sgd_ever_fired`** (**`nn_gpu.cu` ~1856–1858**) — coherent SGD→inference ordering on that stream.
- **`runNNSGDCtx`:** launch on **`g_sgd_stream`**, **`cudaEventRecord(g_sgd_done, …)`**, documented fire-and-forget (**~1983–1990**) — behavior matches design; issue is **callers** assuming host mutex equals GPU done.
- **VOL `s_s2_busy_ms`:** per-worker wall time accumulated, then **max** across workers — consistent with bottleneck semantics.

---

## Reference index

| Topic | Location |
|--------|-----------|
| Temp HDF5 path / per-field recreate | `benchmarks/sdrbench/generic-benchmark.cu` **108–109**, **1756–1760** |
| VOL worker join + I/O join | `src/hdf5/H5VLgpucompress.cu` **1649–1653**, **1671–1684** |
| Diagnostic `actual_ratio` / `final_ratio` | `src/api/gpucompress_diagnostics.cpp` **133–141**, swapped-branch debug **227–231** |
| Primary vs final compressed size | `gpucompress_compress.cpp` **506–514**, **803**, **930–932** |
| NN load swap / loaded flag / alloc failure | `src/nn/nn_gpu.cu` **1472–1488** |
| NN snapshots | `src/nn/nn_gpu.cu` **1541–1587** |
| VPIC write/read sync | `benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx` **~1486**, **~1556**, save **~1828** |
| Comp context release | `src/api/gpucompress_pool.cpp` **181–187** |
| Two-context AUTO compress note vs stale comment | `src/api/gpucompress_compress.cpp` **80–83**, **115–118** |

---

*Merged from HDF5/VOL/diagnostics analysis and a skeptical pass on NN-load / snapshot / VPIC claims. Line numbers may drift with edits; re-grep if unsure.*
