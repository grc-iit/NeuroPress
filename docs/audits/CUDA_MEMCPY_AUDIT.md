# CUDA GPU Operations Latency Audit

An audit of every `cudaMemcpy`, `cudaMalloc`, `cudaFree`, `cudaStreamSynchronize`,
kernel launch, and `cudaEventCreate` across the compress, decompress, stats, and NN
inference hot paths. Findings are ranked by performance impact.

> **Status legend:** ✅ FIXED &nbsp;|&nbsp; 🔴 OPEN (Critical) &nbsp;|&nbsp; 🟠 OPEN (High) &nbsp;|&nbsp; 🟡 OPEN (Medium) &nbsp;|&nbsp; ⚪ OBSOLETE

---

## Why This Matters

`cudaMalloc` and `cudaFree` are **implicit synchronization points** — they block the
CPU until all pending GPU work completes, then perform an OS-level allocation. Each
call costs **10–100 µs**. For tiny buffers (4–128 bytes), allocation overhead dwarfs
the actual work.

`cudaMemcpy` (synchronous) blocks the CPU until the transfer finishes, destroying any
CPU–GPU overlap. `cudaMemcpyAsync` is preferred but even async tiny transfers carry
per-call driver overhead (~2–5 µs each).

`cudaEventCreate` acquires a CUDA driver resource and implicitly synchronizes the GPU.
Using it inside a compress call adds 10–50 µs per invocation.

Kernel launches with too few threads (< 1 warp = 32 threads per SM) waste GPU
occupancy. The ~5–10 µs launch overhead then dominates over the actual computation.

---

## Finding 1 — ✅ FIXED: Per-call cudaMalloc/cudaFree in NN inference

**File:** `src/nn/nn_gpu.cu`

**Was:** Every `runNNInference()` call allocated and freed 3 tiny device buffers
(`d_action` 4B, `d_predicted_ratio` 4B, `d_top_actions` 128B) — 6 synchronization
points for 136 total bytes.

**Fix applied:** `allocInferenceBuffers()` is called inside `loadNNFromBinary()` to
pre-allocate `d_infer_action`, `d_infer_ratio`, `d_infer_comp_time`, and
`d_infer_top_actions` once at load time. They are freed only in `freeInferenceBuffers()`
during cleanup. ✅

---

## Finding 2 — 🔴 OPEN: Per-call cudaMalloc/cudaFree in stats pipeline

**File:** `src/stats/stats_kernel.cu:332–411`
**Estimated overhead:** 20–200 µs per compress call

Every call to `runStatsKernels()` (invoked on every `ALGO_AUTO` compress) allocates
and immediately frees a fixed-size workspace:

```cpp
// stats_kernel.cu:332–336
size_t workspace_size = sizeof(AutoStatsGPU) + 256 * sizeof(unsigned int);
void* d_workspace = nullptr;
err = cudaMalloc(&d_workspace, workspace_size);  // ← implicit GPU sync + OS alloc
// ... 5 kernels + 2 D→H copies ...
cudaFree(d_workspace);                           // ← implicit GPU sync
```

The workspace is always the same size (`sizeof(AutoStatsGPU) + 1024` bytes ≈ 1.1 KB)
and is independent of data size. There is no reason to allocate it per call.

**Fix:** Pre-allocate the workspace once in `nn_gpu.cu`'s `loadNNFromBinary()` (or in
a new `initStatsPipeline()` called from there) alongside the inference buffers. Reuse
across all calls. Free in `cleanupNN()`.

```cpp
// In nn_gpu.cu — add to static state:
static void*          d_stats_workspace  = nullptr;
static AutoStatsGPU*  d_stats            = nullptr;
static unsigned int*  d_stats_histogram  = nullptr;

// In allocInferenceBuffers():
constexpr size_t kStatsWorkspaceSize =
    sizeof(AutoStatsGPU) + 256 * sizeof(unsigned int);
if (d_stats_workspace == nullptr) {
    cudaMalloc(&d_stats_workspace, kStatsWorkspaceSize);
    d_stats       = static_cast<AutoStatsGPU*>(d_stats_workspace);
    d_stats_histogram = reinterpret_cast<unsigned int*>(
        static_cast<uint8_t*>(d_stats_workspace) + sizeof(AutoStatsGPU));
}

// In runStatsKernels() — remove cudaMalloc/cudaFree, use d_stats/d_stats_histogram:
// (replace the dynamic allocation with the pre-allocated pointers)
// cudaMemsetAsync still needed to zero them before each pipeline run.
```

---

## Finding 3 — 🔴 OPEN: Active learning loop — per-iteration sync + malloc + synchronous memcpy

**File:** `src/api/gpucompress_api.cpp:615–929`
**Estimated overhead:** up to ~8 ms when K=31 (OOD + high MAPE)

The Level-2 exploration loop runs up to K=31 alternative compressions. Each iteration
performs:

```cpp
// gpucompress_api.cpp (per iteration)
cudaStreamSynchronize(stream);                          // CPU stall
cudaMalloc(&d_alt_out, alt_max);                        // implicit sync + OS alloc
//  ... compress ...
cudaMemcpy(output + header_sz, d_alt_out, ...);         // SYNCHRONOUS D→H (blocking!)
cudaFree(d_alt_out);                                    // implicit sync
cudaFree(d_alt_quant); cudaFree(d_alt_shuf);            // 2 more implicit syncs
```

Worst case (K=31): `31 × (sync + malloc + blocking_memcpy + 3×free)` = up to 5
synchronization events per iteration × 31 × ~50 µs each = **~7.75 ms** in pure
synchronization overhead alone, before any actual compression work.

Additionally, the PSNR path for lossy configs copies the entire input dataset to CPU
(`cudaMemcpy` blocking, full data size) for MSE computation, incurring a full PCIe
round-trip per explored lossy config.

**Fix (three independent parts, apply in order of impact):**

**Part A — Pre-allocate `d_alt_out`** (eliminates the per-iteration malloc/free):

```cpp
// In gpucompress_api.cpp — before the exploration loop, allocate once:
size_t worst_case_size = gpucompress_max_compressed_size(input_size);
uint8_t* d_alt_out = nullptr;
cudaMalloc(&d_alt_out, worst_case_size);  // one allocation before the loop

// Inside the loop — remove cudaMalloc/cudaFree for d_alt_out, reuse the pointer.
// After the loop:
cudaFree(d_alt_out);
```

**Part B — Switch the winner-copy memcpy to Async** (eliminates the blocking copy):

```cpp
// Before (blocks CPU until transfer done):
cudaMemcpy(static_cast<uint8_t*>(output) + header_sz,
           d_alt_out, alt_comp_size, cudaMemcpyDeviceToHost);

// After (queued on stream, overlaps with next iteration's preprocessing):
cudaMemcpyAsync(static_cast<uint8_t*>(output) + header_sz,
                d_alt_out, alt_comp_size,
                cudaMemcpyDeviceToHost, stream);
// Add one cudaStreamSynchronize(stream) AFTER the loop to ensure last copy lands.
```

**Part C — GPU PSNR instead of CPU MSE** (eliminates the full-data D→H copy for lossy configs):

Replace the CPU MSE loop (lines 805–826) with a small GPU reduction kernel that
computes MSE in-place on `d_input` vs `d_rt_result`, copying only the final `double`
result (8 bytes) to CPU. This is a standard parallel-reduction pattern already used
elsewhere in the codebase (`statsPass1Kernel`, `madPass2Kernel`).

---

## Finding 4 — 🟠 OPEN: Header round-trip Host → GPU → Host

**File:** `src/api/gpucompress_api.cpp:522–526`
**Estimated overhead:** ~1 µs + 64 bytes wasted PCIe bandwidth + extra GPU allocation

The 64-byte `CompressionHeader` is built entirely on the CPU, written to GPU
(`writeHeaderToDevice`), and then read back in the D→H copy of the full output. The
header never needs to be on the GPU.

```cpp
// Current (unnecessary round-trip):
writeHeaderToDevice(d_output, header, stream);   // H→D: 64 bytes
cudaMemcpyAsync(output, d_output, total_size,    // D→H: header + compressed data
                cudaMemcpyDeviceToHost, stream);
```

**Fix:** Write the header directly to the host output buffer; D→H copy only the
compressed payload from `d_compressed` (not `d_output`). The `d_output` allocation
can then shrink by `header_size` bytes since the header no longer lives on GPU.

```cpp
// After:
memcpy(output, &header, sizeof(CompressionHeader));        // host-to-host, ~0 cost
cudaMemcpyAsync(
    static_cast<uint8_t*>(output) + header_size,           // after the header
    d_compressed,                                          // d_compressed = d_output (no +offset needed)
    compressed_size,
    cudaMemcpyDeviceToHost, stream);

// d_output allocation shrinks from (header_size + max_compressed_size)
// to just max_compressed_size.
```

---

## Finding 5 — ✅ FIXED: Decompress copies header bytes to GPU unnecessarily

**File:** `src/api/gpucompress_api.cpp`

**Was:** The audit described allocating `input_size` (header + payload) and uploading
all of it including the already-parsed header.

**Fix applied:** Current code correctly allocates `compressed_size` (payload only) and
sets the source pointer to `input + header_size`, skipping the header. ✅

```cpp
// Current (correct):
uint8_t* d_compressed_data = nullptr;
cuda_err = cudaMalloc(&d_compressed_data, compressed_size);  // payload only
const uint8_t* host_payload = static_cast<const uint8_t*>(input) + header_size;
cuda_err = cudaMemcpyAsync(d_compressed_data, host_payload, compressed_size, ...);
```

---

## Finding 6 — ⚪ OBSOLETE: Q-table malloc/free per call

**File:** `src/lib/qtable_gpu.cu` — **this file no longer exists in the codebase.**

The Q-Table code path has been replaced by the neural-network inference path. This
finding is no longer applicable.

---

## Finding 7 — 🟡 OPEN: Stats results copied in 2 separate small D→H transfers

**File:** `src/stats/stats_kernel.cu:390–403`
**Estimated overhead:** 1 extra D→H call overhead (~2–5 µs)

```cpp
// Two separate calls for 24 total bytes:
cudaMemcpyAsync(&h_result.entropy,        &d_stats->entropy,        8B,  D→H);
cudaMemcpyAsync(&h_result.mad_normalized, &d_stats->mad_normalized, 16B, D→H);
```

Examination of the `AutoStatsGPU` struct layout confirms the three fields are
**already contiguous in device memory**:
- `entropy`         at offset 40 (8 bytes)
- `mad_normalized`  at offset 48 (8 bytes)
- `deriv_normalized`at offset 56 (8 bytes)

**Fix:** Replace with a single 24-byte copy:

```cpp
// Single 24-byte copy instead of two separate calls:
err = cudaMemcpyAsync(&h_result.entropy, &d_stats->entropy,
                      3 * sizeof(double),
                      cudaMemcpyDeviceToHost, stream);
```

No struct reorder needed — the fields are already contiguous.

---

## Finding 8 — 🟠 OPEN: Quantization min/max uses per-call cudaMalloc/cudaFree + extra sync

**File:** `src/preprocessing/quantization_kernels.cu:308–365`
**Estimated overhead:** ~40–200 µs per quantized compress call (4 malloc/free + 1 sync)

`compute_data_range()` is called from `quantize_simple()` on every quantized
compression. It allocates and frees two tiny device scalars per call:

```cpp
// quantization_kernels.cu:312–337 (float path):
cudaMalloc(&d_min, sizeof(float));     // ← 4 bytes, implicit sync
cudaMalloc(&d_max, sizeof(float));     // ← 4 bytes, implicit sync
// ...kernel...
cudaMemcpyAsync(&h_min, d_min, ...);
cudaMemcpyAsync(&h_max, d_max, ...);
cudaStreamSynchronize(stream);         // ← extra sync (quantize_simple also syncs)
cudaFree(d_min);                       // ← implicit sync
cudaFree(d_max);                       // ← implicit sync
```

Additionally, `quantize_simple()` calls `cudaStreamSynchronize(stream)` at line 556
after the quantize kernel, even though `gpucompress_compress()` already issues its own
`cudaStreamSynchronize` at line 415 immediately after preprocessing. This creates a
**redundant sync**.

**Fix (two parts):**

**Part A — Pre-allocate the min/max scalars** alongside the stats workspace
(see Finding 2 fix). A pair of `float` pointers (8 bytes) at module level costs nothing:

```cpp
// In nn_gpu.cu static state (or a dedicated quant_prealloc.cu):
static float*  d_quant_min_f = nullptr;
static float*  d_quant_max_f = nullptr;
static double* d_quant_min_d = nullptr;
static double* d_quant_max_d = nullptr;

// Allocate in allocInferenceBuffers():
cudaMalloc(&d_quant_min_f, sizeof(float));
cudaMalloc(&d_quant_max_f, sizeof(float));
cudaMalloc(&d_quant_min_d, sizeof(double));
cudaMalloc(&d_quant_max_d, sizeof(double));
// Free in freeInferenceBuffers().
```

**Part B — Remove redundant sync in `quantize_simple()`** (line 556):

```cpp
// Remove this line from quantize_simple() — the caller already synchronizes:
// cudaStreamSynchronize(stream);   ← delete
```

**Bonus — Reuse stats min/max:** When quantization is called after `runStatsKernels()`,
the data min/max are already known from `AutoStatsGPU::vmin`/`vmax`. Passing them as
parameters to `quantize_simple()` would skip `compute_data_range()` entirely.

---

## Finding 9 — 🔴 OPEN: NN inference kernel launches only 32 threads

**File:** `src/nn/nn_gpu.cu:596–605`
**Estimated overhead:** 5–10 µs kernel launch with near-zero GPU utilization

```cpp
nnInferenceKernel<<<1, NN_NUM_CONFIGS, 0, stream>>>(...);
//                   ^  ^32 threads
```

A modern GPU SM has 2048 active thread slots and requires a minimum of 32 threads
(1 warp) to do any useful work. Launching 32 threads means **one warp occupies one
SM** while all remaining SMs are idle. The ~5–10 µs kernel launch overhead alone
dominates the actual 18K multiply-accumulate operations per config.

Inside the kernel, each of the 32 threads runs the full 3-layer forward pass **with
serial inner loops** over `NN_HIDDEN_DIM=128`:

```cpp
// Per thread — fully sequential:
for (int j = 0; j < 128; j++) {           // outer: 128 output neurons
    float sum = weights->b1[j];
    for (int i = 0; i < 15; i++) {         // inner: 15 input features
        sum += weights->w1[j*15 + i] * input[i];
    }
    hidden1[j] = (sum > 0.0f) ? sum : 0.0f;
}
```

**Root cause:** The NN inputs (entropy, MAD, deriv) arrive from the stats pipeline
as **host-side values** after a D→H copy and `cudaStreamSynchronize`. The kernel
then passes them back to the GPU as scalar arguments just to get 1 integer (action)
and 2 floats (ratio, comp_time) back. The GPU is used as a slow calculator for work
that is faster on CPU.

**Fix: Run NN inference on CPU using a host-side weight copy.**

The total computation is 32 configs × 18,944 MACs ≈ 600K float ops. On a modern CPU
(~4–16 GFLOPS scalar) this is **< 1 µs** with basic loops, far cheaper than the
GPU kernel launch overhead alone. This also eliminates the kernel launch, the 4 D→H
output copies, and the `cudaStreamSynchronize` after the kernel.

Implementation steps (all in `src/nn/nn_gpu.cu`):

```cpp
// Step 1 — Keep a host-side copy of weights:
static NNWeightsGPU h_nn_weights;   // add to static state
static bool g_host_weights_valid = false;

// Step 2 — Copy to host in loadNNFromBinary(), right after the file read
// (h_weights local is already populated before cudaMemcpy to device):
memcpy(&h_nn_weights, &h_weights, sizeof(NNWeightsGPU));
g_host_weights_valid = true;

// Step 3 — Add CPU forward-pass function:
static int runNNInferenceCPU(
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound,
    float* out_predicted_ratio,
    float* out_predicted_comp_time,
    int*   out_top_actions
) {
    float rank_vals[NN_NUM_CONFIGS];
    float ratios[NN_NUM_CONFIGS];
    float comp_times[NN_NUM_CONFIGS];

    for (int tid = 0; tid < NN_NUM_CONFIGS; ++tid) {
        int algo_idx = tid % 8;
        int quant    = (tid / 8) % 2;
        int shuffle  = (tid / 16) % 2;

        // Build input (identical logic to the GPU kernel)
        float input_raw[NN_INPUT_DIM];
        for (int i = 0; i < 8; i++) input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
        input_raw[8]  = static_cast<float>(quant);
        input_raw[9]  = static_cast<float>(shuffle);
        double eb_c   = (error_bound < 1e-7) ? 1e-7 : error_bound;
        input_raw[10] = static_cast<float>(log10(eb_c));
        double ds     = (data_size < 1) ? 1.0 : static_cast<double>(data_size);
        input_raw[11] = static_cast<float>(log2(ds));
        input_raw[12] = static_cast<float>(entropy);
        input_raw[13] = static_cast<float>(mad_norm);
        input_raw[14] = static_cast<float>(deriv_norm);

        // Standardize
        float input[NN_INPUT_DIM];
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            float std_val = h_nn_weights.x_stds[i];
            if (std_val < 1e-8f) std_val = 1e-8f;
            input[i] = (input_raw[i] - h_nn_weights.x_means[i]) / std_val;
        }

        // Layer 1: 15 → 128 + ReLU
        float h1[NN_HIDDEN_DIM];
        for (int j = 0; j < NN_HIDDEN_DIM; j++) {
            float s = h_nn_weights.b1[j];
            for (int i = 0; i < NN_INPUT_DIM; i++)
                s += h_nn_weights.w1[j * NN_INPUT_DIM + i] * input[i];
            h1[j] = (s > 0.0f) ? s : 0.0f;
        }

        // Layer 2: 128 → 128 + ReLU
        float h2[NN_HIDDEN_DIM];
        for (int j = 0; j < NN_HIDDEN_DIM; j++) {
            float s = h_nn_weights.b2[j];
            for (int i = 0; i < NN_HIDDEN_DIM; i++)
                s += h_nn_weights.w2[j * NN_HIDDEN_DIM + i] * h1[i];
            h2[j] = (s > 0.0f) ? s : 0.0f;
        }

        // Layer 3: 128 → 4
        float out_norm[NN_OUTPUT_DIM];
        for (int j = 0; j < NN_OUTPUT_DIM; j++) {
            float s = h_nn_weights.b3[j];
            for (int i = 0; i < NN_HIDDEN_DIM; i++)
                s += h_nn_weights.w3[j * NN_HIDDEN_DIM + i] * h2[i];
            out_norm[j] = s;
        }

        // De-normalize
        float comp_time  = expm1f(out_norm[0] * h_nn_weights.y_stds[0] + h_nn_weights.y_means[0]);
        float decomp_time= expm1f(out_norm[1] * h_nn_weights.y_stds[1] + h_nn_weights.y_means[1]);
        float ratio      = expm1f(out_norm[2] * h_nn_weights.y_stds[2] + h_nn_weights.y_means[2]);
        float psnr       =        out_norm[3] * h_nn_weights.y_stds[3] + h_nn_weights.y_means[3];
        ratios[tid]      = ratio;
        comp_times[tid]  = comp_time;

        // Rank value
        float rv;
        switch (g_rank_criterion) {
            case NN_RANK_BY_RATIO:      rv =  ratio;      break;
            case NN_RANK_BY_COMP_TIME:  rv = -comp_time;  break;
            case NN_RANK_BY_DECOMP_TIME:rv = -decomp_time;break;
            case NN_RANK_BY_PSNR:       rv =  psnr;       break;
            default:                    rv =  ratio;       break;
        }
        if (quant == 1 && error_bound <= 0.0) rv = -INFINITY;
        rank_vals[tid] = rv;
    }

    // Find best action
    int best = 0;
    for (int i = 1; i < NN_NUM_CONFIGS; i++)
        if (rank_vals[i] > rank_vals[best]) best = i;

    if (out_predicted_ratio)    *out_predicted_ratio    = ratios[best];
    if (out_predicted_comp_time)*out_predicted_comp_time = comp_times[best];

    if (out_top_actions) {
        // Simple insertion sort (32 elements)
        int sorted[NN_NUM_CONFIGS];
        for (int i = 0; i < NN_NUM_CONFIGS; i++) sorted[i] = i;
        for (int i = 1; i < NN_NUM_CONFIGS; i++) {
            int key = sorted[i]; int j = i - 1;
            while (j >= 0 && rank_vals[sorted[j]] < rank_vals[key]) {
                sorted[j + 1] = sorted[j]; j--;
            }
            sorted[j + 1] = key;
        }
        memcpy(out_top_actions, sorted, NN_NUM_CONFIGS * sizeof(int));
    }
    return best;
}

// Step 4 — In runNNInference(), call CPU path when host weights are available:
int runNNInference(...) {
    if (!g_nn_loaded) return -1;
    if (g_host_weights_valid) {
        return runNNInferenceCPU(entropy, mad_norm, deriv_norm, data_size,
                                 error_bound, out_predicted_ratio,
                                 out_predicted_comp_time, out_top_actions);
    }
    // GPU fallback (existing code) for cases where host weights are unavailable.
    ...
}
```

**Important:** When `nn_reinforce_apply()` updates the GPU weights, also update
`h_nn_weights` by calling `cudaMemcpy(&h_nn_weights, d_nn_weights, sizeof(NNWeightsGPU), D→H)`
so the CPU inference uses the latest weights.

---

## Finding 10 — 🟠 OPEN: `finalizeStatsOnlyKernel` is a single-thread kernel

**File:** `src/stats/stats_kernel.cu:380`
**Estimated overhead:** ~5 µs kernel launch overhead for 6 arithmetic operations

```cpp
finalizeStatsOnlyKernel<<<1, 1, 0, stream>>>(d_stats);
//                         ^  ^1 thread — single-thread kernel
```

The kernel performs 6 divisions/multiplications and writes 2 doubles. Kernel launch
overhead (~5 µs) is 1000× the actual computation time. The values it computes
(`mad_normalized`, `deriv_normalized`) are only needed on the CPU anyway — they are
immediately copied back from GPU after this kernel completes.

**Fix: Move normalization to CPU after copying the raw accumulator values.**

Replace the `finalizeStatsOnlyKernel` launch and the two fragmented D→H copies with a
single D→H copy of the whole `AutoStatsGPU` struct (80 bytes), then compute on CPU:

```cpp
// In runStatsKernels() — remove the finalizeStatsOnlyKernel launch and the two
// separate cudaMemcpyAsync calls. Replace with:

// One D→H copy of the whole struct (80 bytes instead of 24 bytes across 2 calls):
AutoStatsGPU h_stats;
err = cudaMemcpyAsync(&h_stats, d_stats, sizeof(AutoStatsGPU),
                      cudaMemcpyDeviceToHost, stream);
if (err != cudaSuccess) { cudaFree(d_workspace); return -1; }

err = cudaStreamSynchronize(stream);
if (err != cudaSuccess) { cudaFree(d_workspace); return -1; }
cudaFree(d_workspace);

// Normalization on CPU (replaces the GPU kernel):
size_t n   = h_stats.num_elements;
double range = static_cast<double>(h_stats.vmax) - static_cast<double>(h_stats.vmin);
*out_entropy = h_stats.entropy;
*out_mad     = (range > 0.0 && n > 0)
                   ? (h_stats.mad_sum / static_cast<double>(n)) / range
                   : 0.0;
*out_deriv   = (range > 0.0 && n > 2)
                   ? (h_stats.abs_diff_sum / static_cast<double>(n - 2)) / range
                   : 0.0;
return 0;
```

This eliminates the `<<<1,1>>>` kernel launch (saves ~5 µs), reduces the copy from
2 calls for 24 bytes to 1 call for 80 bytes (net savings: 1 call overhead ~3 µs), and
removes the dependency on `finalizeStatsOnlyKernel` entirely (the function can be
deleted).

---

## Finding 11 — 🟠 OPEN: Redundant `cudaStreamSynchronize` before stats pipeline

**File:** `src/api/gpucompress_api.cpp:315`
**Estimated overhead:** 1–5 µs unnecessary CPU stall

```cpp
if (num_elements > 0 && gpucompress_nn_is_loaded_impl()) {
    // GPU stats from device pointer (data already copied to d_input)
    cudaStreamSynchronize(stream);   // ← WHY?
    int stats_rc = gpucompress::runStatsOnlyPipeline(
        d_input, input_size, stream, &entropy, &mad, &second_derivative);
```

The `d_input` buffer is populated by `cudaMemcpyAsync` on the same stream (line 284).
`runStatsOnlyPipeline` enqueues its kernels on the same stream. CUDA stream ordering
**guarantees** that kernels will not execute until all prior commands on the same stream
(including the H→D memcpy) have completed. The `cudaStreamSynchronize` here stalls the
CPU thread but provides no correctness guarantee that is not already provided by stream
ordering.

**Fix:** Delete the `cudaStreamSynchronize(stream)` at line 315. No other changes needed.

```cpp
// Before:
cudaStreamSynchronize(stream);          // ← remove this line
int stats_rc = gpucompress::runStatsOnlyPipeline(...);

// After:
int stats_rc = gpucompress::runStatsOnlyPipeline(...);
```

---

## Finding 12 — 🟠 OPEN: `cudaEventCreate`/`Destroy` on every compress call

**File:** `src/api/gpucompress_api.cpp:456–479` (primary path)
          `src/api/gpucompress_api.cpp:686–699` (exploration loop, per iteration)
**Estimated overhead:** 10–50 µs per compress call; up to 31× that in exploration

```cpp
// gpucompress_api.cpp:456–458 — called on EVERY compress:
cudaEvent_t t_start, t_stop;
bool timing_ok = (cudaEventCreate(&t_start) == cudaSuccess &&
                  cudaEventCreate(&t_stop)  == cudaSuccess);
// ...
cudaEventDestroy(t_start);
cudaEventDestroy(t_stop);
```

`cudaEventCreate` acquires a CUDA driver resource and serializes with pending GPU
operations. Creating and destroying a pair of events on every compress call wastes
10–50 µs. The exploration loop repeats this up to 31 times (a new pair per alternative
config), multiplying the overhead.

**Fix:** Pre-allocate two timing events as static globals during `gpucompress_init()`,
and a second pre-allocated pair for exploration (or reuse the same pair sequentially
since the exploration loop runs synchronously):

```cpp
// In gpucompress_api.cpp — add to anonymous namespace state:
static cudaEvent_t g_t_start = nullptr;
static cudaEvent_t g_t_stop  = nullptr;

// In gpucompress_init() — after cudaStreamCreate:
cudaEventCreate(&g_t_start);
cudaEventCreate(&g_t_stop);

// In gpucompress_cleanup():
if (g_t_start) { cudaEventDestroy(g_t_start); g_t_start = nullptr; }
if (g_t_stop)  { cudaEventDestroy(g_t_stop);  g_t_stop  = nullptr; }

// In gpucompress_compress() — replace local creation with:
bool timing_ok = (g_t_start != nullptr && g_t_stop != nullptr);
if (timing_ok) cudaEventRecord(g_t_start, stream);
// ... (same logic, no Destroy calls) ...
if (timing_ok) {
    cudaEventRecord(g_t_stop, stream);
    cudaEventSynchronize(g_t_stop);
    cudaEventElapsedTime(&primary_comp_time_ms, g_t_start, g_t_stop);
}
// The exploration loop can reuse g_t_start / g_t_stop for each iteration's timing.
```

---

## Finding 13 — 🟡 OPEN: Four sequential GPU synchronization barriers per compress call

**File:** `src/api/gpucompress_api.cpp` + `src/stats/stats_kernel.cu` + `src/nn/nn_gpu.cu`
**Estimated overhead:** cumulative ~10–20 µs in unnecessary stalls

Tracing a single `ALGO_AUTO` compress call reveals four `cudaStreamSynchronize` or
equivalent blocking operations:

| # | Location | Code | Necessary? |
|---|----------|------|------------|
| 1 | `gpucompress_api.cpp:315` | `cudaStreamSynchronize` before stats | ❌ No — see Finding 11 |
| 2 | `stats_kernel.cu:405` | `cudaStreamSynchronize` after stats kernels | ✅ Yes — need values on CPU |
| 3 | `nn_gpu.cu:633` | `cudaStreamSynchronize` after NN kernel | ✅ Yes (GPU path) / ❌ Not needed (CPU path, Finding 9) |
| 4 | `gpucompress_api.cpp:415` | `cudaStreamSynchronize` after preprocessing | ✅ Yes — nvCOMP requires sync |

**With fixes from Finding 9 (CPU NN), 10 (CPU finalize), and 11 (remove pre-sync):**

- Barrier 1 is eliminated (Finding 11).
- Barriers 2 + 3 collapse into a single sync: the stats D→H copy (Finding 10)
  already delivers all values to CPU; no separate NN kernel sync is needed because
  the CPU path (Finding 9) never touches the GPU.
- Barrier 4 remains (required by nvCOMP).

Net result: **4 sync points → 2 sync points** per compress call.

**No code changes beyond those in Findings 9, 10, and 11.**

---

## Summary Table

| # | Finding | Status | Impact | Location | Fix Effort |
|---|---------|--------|--------|----------|------------|
| 1 | NN inference malloc/free per call | ✅ Fixed | High | `nn_gpu.cu` | — |
| 2 | Stats workspace malloc/free per call | 🔴 Open | **Critical** | `stats_kernel.cu:332` | Low — pre-alloc in `allocInferenceBuffers` |
| 3 | Exploration loop: sync + malloc + blocking memcpy | 🔴 Open | **Critical** (conditional) | `gpucompress_api.cpp:615` | Medium — 3 independent sub-fixes |
| 4 | Header round-trip H→D→H | 🟠 Open | Medium | `gpucompress_api.cpp:522` | Low — `memcpy` header on host |
| 5 | Decompress copies header to GPU | ✅ Fixed | Medium | `gpucompress_api.cpp` | — |
| 6 | Q-table malloc/free per call | ⚪ Obsolete | — | `qtable_gpu.cu` (deleted) | — |
| 7 | Stats results in 2 separate D→H copies | 🟡 Open | Low | `stats_kernel.cu:390` | Trivial — change `1*sizeof` to `3*sizeof` |
| 8 | Quantization min/max: malloc/free + extra sync | 🟠 Open | High | `quantization_kernels.cu:312` | Low — pre-alloc scalars + remove redundant sync |
| 9 | NN kernel: 1 block × 32 threads, GPU for CPU work | 🔴 Open | **Critical** | `nn_gpu.cu:596` | Medium — add CPU forward-pass function |
| 10 | `finalizeStatsOnlyKernel<<<1,1>>>`: single-thread | 🟠 Open | High | `stats_kernel.cu:380` | Low — remove kernel, compute on CPU |
| 11 | Redundant `cudaStreamSynchronize` before stats | 🟠 Open | High | `gpucompress_api.cpp:315` | Trivial — delete 1 line |
| 12 | `cudaEventCreate`/`Destroy` on every call | 🟠 Open | High | `gpucompress_api.cpp:456` | Low — pre-alloc 2 events in init |
| 13 | 4 sync barriers (2 removable) | 🟡 Open | Medium | pipeline-wide | None extra — falls out of fixes 9/10/11 |

---

## Recommended Fix Priority

Apply in this order for maximum latency reduction per effort:

1. **Finding 11** (1 line delete) — instant win, no risk.
2. **Finding 7** (1 number change) — instant win, no risk.
3. **Finding 10** (remove kernel, add 10-line CPU normalization) — eliminates a
   kernel launch and simplifies the copy path.
4. **Finding 9** (add CPU forward-pass function, ~80 lines) — eliminates the entire
   GPU NN kernel, its D→H copies, and a sync barrier. Highest single-finding impact
   after Fix 2/3.
5. **Finding 2** (pre-alloc stats workspace) — eliminates the biggest per-call malloc.
6. **Finding 12** (pre-alloc events) — eliminate event-create overhead.
7. **Finding 4** (header host-side memcpy) — small PCIe and allocation savings.
8. **Finding 8** (pre-alloc quant scalars + remove redundant sync) — helps quantized paths.
9. **Finding 3** (exploration pre-alloc + async memcpy + GPU PSNR) — only matters when
   online learning is enabled and MAPE is high; still important for production use.

---

## Complete Transfer Map — ALGO_AUTO Compress Path (Current)

```
Host                                    GPU
────                                    ────
input data ──── H→D big ─────────────> d_input               api.cpp:284
h_init     ──── H→D 80B ─────────────> d_stats               stats_kernel.cu:357
                                        [cudaMalloc workspace]   ← Finding 2
                                        statsPass1Kernel
                                        histogramKernel
                                        entropyFromHistogramKernel
                                        madPass2Kernel
                                        finalizeStatsOnlyKernel<<<1,1>>>  ← Finding 10
entropy    <──── D→H 8B  ──────────── d_stats->entropy        stats_kernel.cu:390
mad+deriv  <──── D→H 16B ──────────── d_stats->mad_normalized stats_kernel.cu:397
                                        [cudaFree workspace]    ← Finding 2
                                        [cudaStreamSync]        ← barrier 2 (required)
                                        [cudaStreamSync]        ← barrier 1 (Finding 11: remove)
                                        nnInferenceKernel<<<1,32>>>  ← Finding 9
h_action   <──── D→H 4B  ──────────── d_infer_action          nn_gpu.cu:610
pred_ratio <──── D→H 4B  ──────────── d_infer_ratio           nn_gpu.cu:615
pred_ct    <──── D→H 4B  ──────────── d_infer_comp_time       nn_gpu.cu:621
top_acts   <──── D→H 128B ─────────── d_infer_top_actions     nn_gpu.cu:627
                                        [cudaStreamSync]        ← barrier 3 (Finding 9: remove)
                                        quantize_simple()       ← Finding 8
                                        byte_shuffle_simple()
                                        [cudaStreamSync]        ← barrier 4 (required)
                                        nvCOMP compress
header     ──── H→D 64B ─────────────> d_output               api.cpp:522  ← Finding 4
output     <──── D→H big ────────────  d_output               api.cpp:525
```

## Complete Transfer Map — ALGO_AUTO Compress Path (After All Fixes)

```
Host                                    GPU
────                                    ────
input data ──── H→D big ─────────────> d_input               (unchanged)
h_init     ──── H→D 80B ─────────────> d_stats (pre-alloc)   Finding 2
                                        statsPass1Kernel
                                        histogramKernel
                                        entropyFromHistogramKernel
                                        madPass2Kernel
                                        [NO finalizeKernel]   Finding 10
h_stats    <──── D→H 80B ──────────── d_stats (one copy)     Findings 7+10
                                        [cudaStreamSync]      ← barrier 2 (only sync needed pre-nvCOMP)
                                        [CPU: normalize MAD/deriv]    Finding 10
                                        [CPU: runNNInferenceCPU()]    Finding 9
                                        quantize_simple() (pre-alloc scalars) Finding 8
                                        byte_shuffle_simple()
                                        [cudaStreamSync]      ← barrier 4 (required by nvCOMP)
                                        nvCOMP compress
[CPU: memcpy header to output]          (no H→D for header)  Finding 4
output     <──── D→H compressed_only ─ d_compressed          Finding 4
```

**Net change:** 7 D→H/H→D transfers → 4 transfers. 4 synchronization barriers → 2.
1 kernel launch (NN) + 1 kernel launch (finalize) eliminated. 2 per-call `cudaMalloc`/
`cudaFree` pairs eliminated. `cudaEventCreate`/`Destroy` eliminated from hot path.
