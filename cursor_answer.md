# GPUCompress benchmark: HDF5 file size vs per-chunk “compressed sum” discrepancy

This document summarizes an investigation into reports that **HDF5 output file sizes** (from `get_file_size`) look **much smaller** than **sums inferred from per-chunk diagnostics**, yielding “impossible” compression ratios (e.g. 346×, 1673×) for scientific float32 data.

**Conclusion (high level):** On the **successful write path**, the VOL connector **fully drains** compression workers and the I/O thread before `H5Dwrite` returns; **`H5Fclose` does not return early** relative to that pipeline. The mismatch is primarily a **diagnostic / accounting bug**: chunk history **`actual_ratio`** is computed from the **primary (NN) compressed payload** when online learning is enabled, **not** from the **final bytes written** after **exploration** may replace the output with a smaller winner. Reconstructing “total compressed bytes” by inverting `actual_ratio` therefore **overstates** on-disk payload vs the real file when exploration wins big.

---

## Context

- **Benchmark:** `benchmarks/sdrbench/generic-benchmark.cu` — multi-field (timestep-major) loop, same temp HDF5 path per field.
- **VOL:** `src/hdf5/H5VLgpucompress.cu` — async pipeline: Stage 1 inference, Stage 2 worker compression, Stage 3 I/O thread → native `CHUNK_WRITE`.
- **Exploration / winner replacement:** `src/api/gpucompress_compress.cpp` (~lines 584–823).

---

## 1. Temp file path: same for all fields?

**Yes.** Path is fixed per MPI rank, e.g. `/tmp/bm_generic_nn_rl_rank%d.h5`.

- **Lines:** `benchmarks/sdrbench/generic-benchmark.cu` **108–109** (`snprintf` for `TMP_NN_RL`).
- Each field: **`remove(TMP_NN_RL)`** then **`H5Fcreate(..., H5F_ACC_TRUNC, ...)`** — **1756–1760**.

So the file is recreated each iteration; `H5F_ACC_TRUNC` replaces content. (Return value of `remove` is not checked; unlikely to explain a **repeating field index** pattern.)

---

## 2. Does `H5Fclose` return before the I/O thread finishes?

**No (success path).** `gpu_aware_chunked_write` **joins all compression workers**, then sets **`io_done_flag`**, **`notify_all`s**, and **`io_thr.join()`** before returning.

- **Worker join + timing:** `H5VLgpucompress.cu` **1649–1653**.
- **I/O thread drain:** **1671–1684** (`io_done_flag`, `io_cv.notify_all()`, `io_thr.join()`).

`H5VL_gpucompress_file_close` (**2561–2567**) only closes the underlying file object after writes have already completed through the dataset write path.

---

## 3. Can `get_file_size()` race async I/O?

**Not in the benchmark sequence.** After write/close, the code opens the file, **`H5Dread`**, closes, **then** calls **`get_file_size(TMP_NN_RL)`**.

- **Read + close + size:** `generic-benchmark.cu` **1799–1803**, **1817**.
- **`get_file_size`:** **225–231** (`open`, `lseek(SEEK_END)`).

By then the VOL I/O thread has already been joined inside `H5Dwrite`.

---

## 4. Pattern on fields 5, 7, 11, 13, 15

No evidence of an **index-based** bug in the loop. Those fields can stand out **data-dependently** when **exploration** runs and picks a **much smaller** winner than the **primary** NN compression, **and** when metrics are inferred from diagnostics that still reflect the **primary** size (see below).

---

## 5. Pipeline drain and `write_chunk_to_native`

Workers push **`IOItem{ data, sz }`** where **`sz` is `comp_sz`** returned from **`gpucompress_compress_with_action_gpu`** (after full compress + exploration + sync on that path). The I/O thread calls **`write_chunk_to_native`** with that size.

- **Worker `comp_sz` + D2H + queue:** `H5VLgpucompress.cu` **1252–1318**.

---

## 6. Exploration winner replacement vs I/O (race?)

**No race** between “winner overwrites output” and “I/O reads size”: exploration uses **`cudaMemcpyAsync`** for the winner payload, then **`cudaStreamSynchronize(stream)`** before freeing slot buffers.

- **Winner write + `*output_size` update:** `gpucompress_compress.cpp` **763–803**.
- **Drain before cleanup:** **821–823**.

---

## Root cause: chunk diagnostics use **primary** compressed size for `actual_ratio`

### What gets written (ground truth for bytes on the wire)

When **online learning** is enabled and algorithm is **AUTO**, the compressor saves **`primary_compressed_size`** = **primary** nvcomp **payload** (before exploration may change the output):

- **`gpucompress_compress.cpp` 506–514** — comment states this is for **MAPE** so prediction accuracy reflects the **NN’s chosen algorithm**, not the exploration winner.

After exploration, **`compressed_size`**, **`total_size`**, and **`*output_size`** reflect the **final** (possibly winning) result.

### What chunk history records

`recordChunkDiagnostic` builds **`actual_ratio`** using **`primary_compressed_size` if it is non-zero**, else **`compressed_size`**:

- **`src/api/gpucompress_diagnostics.cpp` 133–136:**
  - `comp_sz = (d.primary_compressed_size > 0) ? d.primary_compressed_size : d.compressed_size`
  - `actual_ratio = min(100.f, input_size / comp_sz)`

So when **exploration replaces** the buffer with a **smaller** winner:

- **On-disk / VOL:** writes **`comp_sz`** matching the **final** layout (**`H5VLgpucompress.cu` worker path**).
- **Diagnostics:** **`actual_ratio`** still behaves like **primary** compression (larger payload → **smaller** ratio number). Reconstructing compressed bytes as **`input / actual_ratio`** (or similar) **overestimates** stored data vs **`get_file_size`**.

Additionally, **`std::min(100.0f, …)`** caps the stored ratio; for **very** high true ratios, inversion **overstates** implied compressed size further.

### Header vs payload note

`*output_size` / VOL `comp_sz` include the **gpucompress header + compressed payload** (`total_size` in compress path). `primary_compressed_size` / diagnostic `comp_sz` in the ratio numerator/denominator logic use **payload** for the primary branch (see compress path around **434–436**, **514**). That is a second-order inconsistency vs “full bytes per chunk”; the **dominant** issue when exploration wins is **primary vs winner payload**.

### Public API comment mismatch

`include/gpucompress.h` documents **`actual_ratio`** as `input_size / compressed_size` (**~605**), but implementation prefers **`primary_compressed_size`** when set — so the **documented meaning** does not match **on-disk bytes** after exploration.

---

## Verification hint

If **`do_verify`** is enabled and **`gpu_compare`** shows **no mismatches**, the **small file** is consistent with **what was written and read back**; the **inflated “chunk sum”** is then very likely **wrong inference from `actual_ratio`**, not file truncation.

---

## Separate issue: error-path `goto` (not steady-state)

`H5VLgpucompress.cu` **1347** (`goto done_write` on `cudaStreamCreate` failure after **I/O thread** and **workers** start) can skip **sentinel/join** logic and lead to **UB** (`std::thread` destructor on joinable threads). This affects **failure** paths, not successful multi-field runs.

---

## Key file / line reference summary

| Topic | Location |
|--------|-----------|
| Temp path, per-field `remove` / `H5Fcreate` | `benchmarks/sdrbench/generic-benchmark.cu` **108–109**, **1756–1760** |
| `get_file_size`, read-then-stat order | **225–231**, **1799–1817** |
| Worker join + I/O `join` | `src/hdf5/H5VLgpucompress.cu` **1649–1653**, **1671–1684** |
| Worker `comp_sz` → I/O | **1252–1318** |
| Save `primary_compressed_size` (MAPE vs NN primary) | `src/api/gpucompress_compress.cpp` **506–514**, **930–932** |
| Diagnostic `actual_ratio` uses primary when > 0 | `src/api/gpucompress_diagnostics.cpp` **133–136** |
| Exploration winner + stream sync | `gpucompress_compress.cpp` **763–823** |

---

*Written from codebase analysis; no requirement to change behavior unless product owners want diagnostics to track “bytes written” separately from “primary MAPE.”*
