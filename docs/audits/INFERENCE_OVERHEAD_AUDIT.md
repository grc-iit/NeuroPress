# Inference Overhead Audit — Issues Beyond CUDA_MEMCPY_AUDIT.md

This document audits every overhead source found in `src/` and `neural_net/` that is
**not already covered** by `CUDA_MEMCPY_AUDIT.md`. Findings span unnecessary GPU
synchronizations, per-call allocations outside the NN/stats path, double-precision
kernel arithmetic, hidden memory leaks, redundant pipeline stages, and I/O stalls.

> **Status legend:** 🔴 Critical &nbsp;|&nbsp; 🟠 High &nbsp;|&nbsp; 🟡 Medium &nbsp;|&nbsp; 🔵 Low / Correctness

---

## Finding A — 🔴 byte_shuffle_simple: 4 cudaMalloc + 1 hidden sync per shuffle call

**File:** `src/preprocessing/byte_shuffle_kernels.cu:165–217` (`createDeviceChunkArrays`)
**Also:** `byte_shuffle_simple` (line 282) and `byte_unshuffle_simple` (line 313)
**Estimated overhead:** ~60–400 µs per shuffle call (4 allocs + 1 sync)

`byte_shuffle_simple` is called on every compress that uses shuffle preprocessing.
Internally it calls `createDeviceChunkArrays`, which performs **4 cudaMalloc calls**
every time:

```cpp
// createDeviceChunkArrays — runs on every byte_shuffle_simple / byte_unshuffle_simple call:
cudaMalloc(&result.d_input_ptrs,  num_chunks * sizeof(uint8_t*));   // ← sync + OS alloc
cudaMalloc(&result.d_output_ptrs, num_chunks * sizeof(uint8_t*));   // ← sync + OS alloc
cudaMalloc(&result.d_sizes,       num_chunks * sizeof(size_t));      // ← sync + OS alloc
// ... plus byte_shuffle_simple itself:
cudaMalloc(&device_output, total_bytes);                              // ← sync + OS alloc
```

After launching `populateChunkArraysKernel` to fill the pointer arrays,
`createDeviceChunkArrays` then calls **`cudaStreamSynchronize(stream)`** (line 212).
This sync is completely unnecessary: the shuffle kernel that follows is enqueued on the
same stream and will naturally wait for `populateChunkArraysKernel` to finish via CUDA
stream ordering. The CPU stall serves no correctness purpose.

For a typical HDF5 workload compressing many same-size chunks, the `d_input_ptrs`,
`d_output_ptrs`, and `d_sizes` arrays are always the same length and the output buffer
is always the same size. None of these need to be re-allocated per call.

**Fix:**

Pre-allocate the chunk pointer arrays and the shuffle output buffer alongside the other
inference buffers (in `allocInferenceBuffers()`). Use the maximum chunk count derived
from the worst-case input size and `SHUFFLE_CHUNK_SIZE`:

```cpp
// In nn_gpu.cu static state:
static uint8_t**  d_shuf_input_ptrs  = nullptr;
static uint8_t**  d_shuf_output_ptrs = nullptr;
static size_t*    d_shuf_sizes       = nullptr;
static uint8_t*   d_shuf_output      = nullptr;
static size_t     g_shuf_buf_bytes   = 0;

// In allocInferenceBuffers() (or a dedicated allocShuffleBufs(max_input_size)):
constexpr size_t kMaxInputSize = /* e.g., 64MB */;
constexpr size_t kMaxChunks = (kMaxInputSize + SHUFFLE_CHUNK_SIZE - 1) / SHUFFLE_CHUNK_SIZE;
cudaMalloc(&d_shuf_input_ptrs,  kMaxChunks * sizeof(uint8_t*));
cudaMalloc(&d_shuf_output_ptrs, kMaxChunks * sizeof(uint8_t*));
cudaMalloc(&d_shuf_sizes,       kMaxChunks * sizeof(size_t));
cudaMalloc(&d_shuf_output,      kMaxInputSize);
g_shuf_buf_bytes = kMaxInputSize;

// Remove the cudaStreamSynchronize from createDeviceChunkArrays:
// populateChunkArraysKernel + the shuffle kernel are on the same stream —
// no explicit sync needed between them.
```

---

## Finding B — 🔴 byte_shuffle_simple: Memory leak of chunk pointer arrays

**File:** `src/preprocessing/byte_shuffle_kernels.cu:270–353`
**Estimated impact:** 3 device buffers (24–768 bytes each) leaked per compress + per decompress

`createDeviceChunkArrays` allocates `d_input_ptrs`, `d_output_ptrs`, and `d_sizes` on
the device and returns a `DeviceChunkArrays` struct by value. Neither `byte_shuffle_simple`
nor `byte_unshuffle_simple` ever calls `cudaFree` on these three device pointers after use:

```cpp
// byte_shuffle_simple — simplified:
DeviceChunkArrays arrays = createDeviceChunkArrays(...);  // 3 cudaMallocs
launch_byte_shuffle(..., arrays.d_input_ptrs, arrays.d_output_ptrs, arrays.d_sizes, ...);
return device_output;
// ← arrays goes out of scope here; d_input_ptrs, d_output_ptrs, d_sizes are NEVER freed
```

`DeviceChunkArrays` has no destructor. Every compress call that uses shuffle leaks these
three buffers. Every decompress call that uses unshuffle leaks three more. Over a run
compressing N chunked datasets, this is `6N` leaked device allocations.

**Fix:**

Either add a destructor to `DeviceChunkArrays`:

```cpp
struct DeviceChunkArrays {
    uint8_t** d_input_ptrs  = nullptr;
    uint8_t** d_output_ptrs = nullptr;
    size_t*   d_sizes       = nullptr;
    size_t    num_chunks    = 0;

    ~DeviceChunkArrays() {
        if (d_input_ptrs)  cudaFree(d_input_ptrs);
        if (d_output_ptrs) cudaFree(d_output_ptrs);
        if (d_sizes)       cudaFree(d_sizes);
    }
    // Delete copy, allow move
    DeviceChunkArrays(const DeviceChunkArrays&) = delete;
    DeviceChunkArrays& operator=(const DeviceChunkArrays&) = delete;
    DeviceChunkArrays(DeviceChunkArrays&&) = default;
    DeviceChunkArrays& operator=(DeviceChunkArrays&&) = default;
};
```

Or apply the pre-allocation fix from Finding A, which eliminates these per-call
allocations entirely.

---

## Finding C — 🟠 dequantize_simple: Per-call cudaMalloc + redundant cudaStreamSynchronize

**File:** `src/preprocessing/quantization_kernels.cu:573–629`
**Estimated overhead:** ~30–200 µs per lossy decompress (1 cudaMalloc + 1 sync)

`dequantize_simple()` is called on every decompress of a quantized stream
(`gpucompress_api.cpp:1155`). It always allocates the output buffer and synchronizes:

```cpp
// quantization_kernels.cu:583–592
void* d_output;
cudaError_t err = cudaMalloc(&d_output, output_bytes);   // ← implicit sync + OS alloc
// ... launch dequantize kernel ...
cudaStreamSynchronize(stream);                            // ← explicit sync, redundant
return d_output;
```

The output buffer has a fixed maximum size (`header.original_size` bytes) that is known
at library init time. The `cudaStreamSynchronize` at line 628 is also redundant: the
caller (`gpucompress_decompress`) already issues its own `cudaStreamSynchronize` at
line 1186 after the memcpy. The dequantize kernel and the D→H memcpy are on the same
stream; the memcpy will naturally wait for the kernel.

**Fix:**

Pre-allocate a reusable dequantize output buffer (same pre-alloc pool used for shuffle
in Finding A):

```cpp
// In nn_gpu.cu static state:
static uint8_t* d_dequant_output = nullptr;
static size_t   g_dequant_output_bytes = 0;

// In allocInferenceBuffers():
cudaMalloc(&d_dequant_output, kMaxInputSize);
g_dequant_output_bytes = kMaxInputSize;
```

Pass `d_dequant_output` as a parameter to `dequantize_simple` instead of allocating
internally. Remove the `cudaStreamSynchronize` at the end of `dequantize_simple`;
the caller already synchronizes.

---

## Finding D — 🟠 quantize_linear_kernel / dequantize_linear_kernel: Double-precision arithmetic on float data

**File:** `src/preprocessing/quantization_kernels.cu:172–215`
**Estimated overhead:** 2×–32× slower GPU throughput for float32 inputs

The quantization and dequantization kernels promote all arithmetic to `double` regardless
of input type:

```cpp
// quantize_linear_kernel<float, OutputT> — float input, all math in double:
double val      = static_cast<double>(input[i]);     // float → double widening
double centered = val - offset;                       // double subtract
double quantized = round(centered * scale);           // double multiply + round
```

```cpp
// dequantize_linear_kernel<float, OutputT> — double output for float data:
double quantized = static_cast<double>(input[i]);
double restored  = quantized * inv_scale + offset;    // double multiply + add
output[i] = static_cast<InputT>(restored);            // double → float narrowing
```

On NVIDIA consumer GPUs (GeForce), the double-to-single throughput ratio is typically
**1:32** (e.g., RTX 3090: 35.6 TFLOPS float32 vs 0.556 TFLOPS float64). Even on data-
center GPUs (A100: 19.5 vs 9.7 TFLOPS) the ratio is 2:1. For float32 input data —
which is the only type currently used at the API level — all intermediate calculations
can be done in float32 without any loss of correctness, since the error bound and scale
factor are ultimately limited by float32 range anyway.

**Fix:**

Specialize the templates for float input to use float arithmetic:

```cpp
// float-specialized quantize (replaces the double path for float inputs):
template<typename OutputT>
__global__ void quantize_linear_kernel_f32(
    const float* input, OutputT* output, size_t num_elements,
    float scale, float offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < num_elements; i += stride) {
        float centered  = input[i] - offset;
        float quantized = rintf(centered * scale);   // single-precision round
        if constexpr (sizeof(OutputT) == 1)
            quantized = fmaxf(-128.f, fminf(127.f, quantized));
        else if constexpr (sizeof(OutputT) == 2)
            quantized = fmaxf(-32768.f, fminf(32767.f, quantized));
        output[i] = static_cast<OutputT>(quantized);
    }
}
```

Keep the double-precision path for `element_size == 8` (double input). Dispatch on
`element_size` in the host-side `quantize_simple()`.

---

## Finding E — 🟠 printf / fprintf in the hot compress path

**File:** `src/api/gpucompress_api.cpp:362–370`
**Estimated overhead:** 1–20 µs per compress call (libc I/O + possible syscall)

Two `printf` calls fire on every `ALGO_AUTO` compress:

```cpp
// gpucompress_api.cpp:362–370 — runs on EVERY ALGO_AUTO compress:
printf("NN: chose %s%s%s (action=%d, predicted_ratio=%.2f)\n",
       gpucompress_algorithm_name(algo_to_use), ...);
// or on failure:
printf("NN: fallback to lz4 (inference failed)\n");
```

In a typical HDF5 workload that compresses thousands of chunks, each `printf` acquires
the libc FILE lock, formats the string, and calls `write(1, ...)` (a syscall) once the
buffer fills. Even with line-buffered stdout, 1000 chunks × ~2 µs ≈ 2 ms of pure I/O
overhead. Under heavy parallelism or if stdout is unbuffered (e.g. piped to a log),
this is a visible bottleneck.

Similarly, when exploration is active, multiple `fprintf(stderr, ...)` calls fire
per explored config (lines 605, 849, 942).

**Fix:**

Guard all diagnostic output behind a verbosity flag, similar to the existing
`is_verbose()` pattern in `H5Zgpucompress.c`:

```cpp
// Add a flag in the anonymous namespace:
static bool g_verbose_compress = false;

// In gpucompress_api.cpp — replace unconditional printfs:
if (g_verbose_compress) {
    printf("NN: chose %s%s%s (action=%d, predicted_ratio=%.2f)\n", ...);
}
```

Expose `gpucompress_set_verbose(int)` in the public API and check the
`GPUCOMPRESS_VERBOSE` env var in `gpucompress_init`, consistent with the HDF5 filter.

---

## Finding F — 🟡 Stats workspace: double initialization (memset + H2D overwrite)

**File:** `src/stats/stats_kernel.cu:343–363`
**Estimated overhead:** ~1–3 µs wasted work (redundant zero-write of 80 bytes)

`runStatsKernels()` initializes the stats workspace in two sequential operations on the
same stream:

```cpp
// Step 1: zero the ENTIRE workspace (AutoStatsGPU=80B + histogram=1024B):
cudaMemsetAsync(d_workspace, 0, workspace_size, stream);   // writes 1104 bytes

// Step 2: overwrite the AutoStatsGPU portion (80 bytes) with h_init:
//   h_init has all-zeros EXCEPT vmin=FLT_MAX, vmax=-FLT_MAX, num_elements=n
cudaMemcpyAsync(d_stats, &h_init, sizeof(AutoStatsGPU), H2D, stream);
```

Step 1 writes zeros to `d_stats` (the AutoStatsGPU struct). Step 2 immediately
overwrites the same 80 bytes. The struct is initialized twice; only the second write
produces correct initial values. The first 80 bytes of step 1 are wasted writes.

**Fix:**

Separate the two initializations: only `cudaMemsetAsync` the histogram portion; use
the H2D copy exclusively for the struct:

```cpp
// Only zero the histogram (1024 bytes at the end of the workspace):
cudaMemsetAsync(d_histogram, 0, 256 * sizeof(unsigned int), stream);

// Initialize the struct (vmin=FLT_MAX, vmax=-FLT_MAX, others=0):
cudaMemcpyAsync(d_stats, &h_init, sizeof(AutoStatsGPU), cudaMemcpyHostToDevice, stream);
```

This replaces two overlapping writes with two distinct non-overlapping writes and
removes a dependency between step 1 and step 2.

---

## Finding G — 🟠 Decompress path: redundant cudaStreamSynchronize before post-processing

**File:** `src/api/gpucompress_api.cpp:1166`
**Estimated overhead:** ~1–5 µs unnecessary CPU stall

The decompress path synchronizes the stream **before** launching unshuffle and
dequantize kernels:

```cpp
// gpucompress_decompress() — simplified:
decompressor->decompress(d_decompressed, d_compressed_data, decomp_config);

cuda_err = cudaStreamSynchronize(stream);   // ← WHY? (line 1166)
if (cuda_err != cudaSuccess) { ... return ...; }

// Then post-processing kernels are launched:
if (header.hasShuffleApplied()) {
    d_unshuffled = byte_unshuffle_simple(d_decompressed, ...);  // new kernel on same stream
}
if (header.hasQuantizationApplied()) {
    d_dequantized = dequantize_simple(d_result, quant_result, stream);
}
```

The decompressor, unshuffle kernel, and dequantize kernel are all enqueued on the
same CUDA stream. Stream ordering guarantees that each will wait for the prior command
to complete before executing. The explicit `cudaStreamSynchronize` at line 1166 stalls
the CPU thread without providing any correctness benefit — it only delays issuing the
post-processing kernels to the GPU.

**Fix:**

Remove the `cudaStreamSynchronize(stream)` at line 1166 entirely. Keep only the
existing sync at line 1186 (after the final D→H memcpy), which is required.

```cpp
// Before:
cuda_err = cudaStreamSynchronize(stream);    // line 1166 — DELETE THIS
if (cuda_err != cudaSuccess) { ... }

// After: no sync here. The unshuffle/dequant kernels and the final memcpy
// are all on the same stream and will execute in order.
```

---

## Finding H — 🟠 nn_reinforce_apply: synchronous cudaMemcpy for ~270 KB H2D weight update

**File:** `src/nn/nn_reinforce.cpp:278`
**Estimated overhead:** ~50–150 µs when online SGD fires

After computing the SGD update, `nn_reinforce_apply` pushes the new weights to GPU
using a **blocking** `cudaMemcpy`:

```cpp
// nn_reinforce.cpp:278
cudaError_t err = cudaMemcpy(d_weights, &updated,
                              sizeof(NNWeightsGPU),   // ~270 KB
                              cudaMemcpyHostToDevice); // SYNCHRONOUS — blocks CPU
```

`NNWeightsGPU` is approximately:
- w1: 128×15×4 = 7,680 B
- w2: 128×128×4 = 65,536 B
- w3: 4×128×4 = 2,048 B
- biases + norms ≈ 2,500 B
- **Total ≈ 78 KB**

The synchronous copy stalls the CPU thread for the full PCIe transfer duration.
This fires whenever the ratio MAPE exceeds `g_reinforce_mape_threshold` (default 20%),
which can be frequent during warm-up or novel data.

**Fix:**

Switch to `cudaMemcpyAsync` on the default stream, and add a single sync only in
the error-checking path:

```cpp
// Replace the synchronous copy:
cudaError_t err = cudaMemcpyAsync(d_weights, &updated,
                                   sizeof(NNWeightsGPU),
                                   cudaMemcpyHostToDevice,
                                   0  // default stream, or pass stream as parameter
                                  );
// The updated h_weights commit still happens below; no blocking sync needed
// unless the caller immediately needs to use the GPU weights.
```

Note: if the CPU inference path (CUDA_MEMCPY_AUDIT.md Finding 9) is implemented,
`d_nn_weights` is no longer read during inference and the GPU copy can be deferred
until the next explicit `loadNNFromBinary` or cleanup.

---

## Finding I — 🟠 Per-call nvCOMP manager construction

**File:** `src/compression/compression_factory.cpp:68–131`  
**Also:** `src/api/gpucompress_api.cpp:430–431` (compress) and `:1093` (decompress)
**Estimated overhead:** unknown, likely 10–100 µs per call (nvcomp internal state setup)

A brand-new nvCOMP manager is instantiated on every compress and every decompress call:

```cpp
// gpucompress_api.cpp:430–431 — every compress:
auto compressor = createCompressionManager(
    internal_algo, gpucompress::DEFAULT_CHUNK_SIZE, stream, d_compress_input);

// gpucompress_api.cpp:1093 — every decompress:
auto decompressor = createDecompressionManager(d_compressed_data, stream);
```

`createCompressionManager` returns a `std::unique_ptr<nvcompManagerBase>` created via
`std::make_unique<LZ4Manager>(...)` etc. The nvCOMP manager constructors may allocate
internal CUDA device state (scratch buffers, metadata tables) and involve driver-level
setup. Creating and immediately destroying a manager on every call wastes this setup cost.

For a fixed-algorithm, fixed-chunk-size workload (the common HDF5 case), the same
manager instance can be safely reused across calls by calling
`configure_compression(new_size)` each time with the new data size.

**Fix:**

Cache one manager per algorithm in static state:

```cpp
// In nn_gpu.cu or a new compressor_cache.cu:
static std::unique_ptr<nvcomp::nvcompManagerBase> g_compressor_cache[8]; // one per algo
static CompressionAlgorithm g_compressor_algo[8] = { ... };

// In gpucompress_compress():
int algo_idx = static_cast<int>(internal_algo);
if (g_compressor_cache[algo_idx] == nullptr) {
    g_compressor_cache[algo_idx] = createCompressionManager(internal_algo, ...);
}
auto& compressor = g_compressor_cache[algo_idx];
CompressionConfig comp_config = compressor->configure_compression(compress_input_size);
// ... (rest unchanged)
```

Free all cached managers in `gpucompress_cleanup()`.

---

## Finding J — 🔴 Per-call cudaMalloc for d_input and d_output in compress

**File:** `src/api/gpucompress_api.cpp:283, 449`
**Estimated overhead:** ~40–200 µs per compress call (2 large allocs + 2 large frees)

Every call to `gpucompress_compress` performs two large device allocations:

```cpp
// api.cpp:283 — input buffer (full input size, possibly MBs):
cuda_err = cudaMalloc(&d_input, aligned_input_size);

// api.cpp:449 — output buffer (header + max compressed size, similar size):
cuda_err = cudaMalloc(&d_output, total_max_size);
```

And corresponding `cudaFree` calls at lines 542–543. These two allocations dwarf all
the smaller ones (min/max scalars, stats workspace) in raw bytes. For a 1 MB HDF5
chunk: two ~1 MB mallocs + two ~1 MB frees per compress = ~40–200 µs of pure
allocation overhead at typical CUDA driver speeds.

In the HDF5 filter path (`H5Zgpucompress.c`), the same chunk size is used for every
call in a dataset write, so both buffers have an identical size on every iteration.

**Fix:**

Pre-allocate a pair of pinned/device buffers sized for the maximum expected chunk:

```cpp
// In gpucompress_api.cpp anonymous namespace:
static uint8_t* g_d_input  = nullptr;
static uint8_t* g_d_output = nullptr;
static size_t   g_d_input_bytes  = 0;
static size_t   g_d_output_bytes = 0;

// Helper — grow-or-reuse:
static cudaError_t ensureDeviceBuf(uint8_t** buf, size_t* cap, size_t needed) {
    if (needed <= *cap) return cudaSuccess;
    if (*buf) cudaFree(*buf);
    cudaError_t e = cudaMalloc(buf, needed);
    if (e == cudaSuccess) *cap = needed;
    return e;
}

// In gpucompress_compress():
cuda_err = ensureDeviceBuf(&g_d_input, &g_d_input_bytes, aligned_input_size);
// (replace cudaMalloc(&d_input, ...) and cudaFree(d_input))
```

Free in `gpucompress_cleanup()`. This converts O(N) mallocs to O(1) amortized,
with realloc only when a larger chunk is seen.

---

## Finding K — 🟡 gpucompress_compute_stats: synchronous cudaMemcpy H2D

**File:** `src/api/gpucompress_api.cpp:1327`
**Estimated overhead:** ~5–20 µs unnecessary CPU stall (blocks until H2D transfer done)

The public `gpucompress_compute_stats()` API function uses **synchronous** `cudaMemcpy`
for the H2D data upload:

```cpp
// api.cpp:1327:
cuda_err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);  // BLOCKING
```

This blocks the calling CPU thread until the entire input is on GPU. All subsequent
kernel launches on the same stream would wait for this anyway via stream ordering.
Using the synchronous variant here adds unnecessary latency versus `cudaMemcpyAsync`.

**Fix:**

```cpp
// Replace:
cuda_err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

// With:
cuda_err = cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, g_default_stream);
// (The subsequent runStatsOnlyPipeline already issues its own cudaStreamSynchronize)
```

---

## Finding L — 🟡 experience_buffer_append: fflush on every sample

**File:** `src/nn/experience_buffer.cpp:111`
**Estimated overhead:** ~5–50 µs per sample (OS write syscall) when experience logging is enabled

```cpp
// experience_buffer.cpp:111 — called for EVERY explored alternative when logging enabled:
fflush(g_file);   // forces kernel-level write; blocks until OS accepts data
```

For an exploration run with K=31 alternatives, this triggers 32 `fflush` calls per
compress (1 primary + 31 alternatives). With the default libc FILE buffer of ~4 KB,
most `fprintf` calls do not immediately write to disk — but `fflush` forces a
`write(2)` syscall regardless. At 50 µs per flush × 32 flushes × 1000 chunks ≈
**1.6 seconds** of pure flush overhead per file.

**Fix:**

Remove the per-append `fflush` and flush only on close or at user-controlled intervals:

```cpp
// experience_buffer.cpp — remove fflush from append:
// fflush(g_file);   ← DELETE

// Add a periodic flush (every 100 samples) or flush only in cleanup:
extern "C" void experience_buffer_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file != nullptr) {
        fflush(g_file);  // flush once on close
        fclose(g_file);
        g_file = nullptr;
    }
    g_count.store(0);
}
```

Alternatively, use `setvbuf(g_file, nullptr, _IOFBF, 64*1024)` after `fopen` to set a
larger full-buffering block, reducing syscall frequency naturally.

---

## Finding M — 🟡 atomicAddDouble: CAS loop instead of native double atomics (SM 6.0+)

**File:** `src/stats/stats_kernel.cu:54–63`
**Estimated overhead:** 2×–10× slower global-memory reduction under contention

The custom `atomicAddDouble` function uses a compare-and-swap loop:

```cpp
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
```

This pattern is the pre-Pascal workaround. Since CUDA Compute Capability **6.0** (Pascal
architecture, 2016), `atomicAdd` natively supports `double*`, making the CAS loop
unnecessary. Under heavy contention (many blocks all updating `stats->sum`), the CAS
loop causes warp serialization that native atomics avoid.

The project already targets GPUs capable of running nvCOMP (Pascal+), so SM 6.0+ is
a safe assumption.

**Fix:**

```cpp
// Replace atomicAddDouble entirely:
#if __CUDA_ARCH__ >= 600
    // Native double atomicAdd available since Pascal (SM 6.0)
    #define atomicAddDouble(addr, val) atomicAdd((addr), (val))
#else
    __device__ double atomicAddDouble(double* address, double val) { /* CAS loop */ }
#endif
```

Or unconditionally use `atomicAdd(double*, double)` since the minimum SM for nvCOMP
is already well above 6.0.

---

## Finding N — 🟡 populateChunkArraysKernel: GPU kernel for trivial CPU-computable pointer arithmetic

**File:** `src/preprocessing/byte_shuffle_kernels.cu:143–217`
**Estimated overhead:** ~5–10 µs kernel launch + hidden sync (see Finding A) for a trivial loop

`populateChunkArraysKernel` fills `d_input_ptrs[i]`, `d_output_ptrs[i]`, and
`d_sizes[i]` for each chunk by computing simple pointer offsets:

```cpp
__global__ void populateChunkArraysKernel(
    uint8_t** d_input_ptrs, uint8_t** d_output_ptrs, size_t* d_sizes,
    uint8_t* base_input, uint8_t* base_output,
    size_t total_bytes, size_t chunk_bytes, size_t num_chunks)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;
    const size_t offset = idx * chunk_bytes;
    const size_t remaining = total_bytes - offset;
    d_input_ptrs[idx]  = base_input  + offset;
    d_output_ptrs[idx] = base_output + offset;
    d_sizes[idx]       = (remaining < chunk_bytes) ? remaining : chunk_bytes;
}
```

For `SHUFFLE_CHUNK_SIZE = 256 KB` and typical input sizes of 64 KB–1 MB, `num_chunks`
is at most 4–8. Launching a kernel, paying the ~5–10 µs launch overhead, and then
syncing (Finding A's hidden sync) to fill 4–8 pointer values is dramatically more
expensive than a CPU loop over the same 4–8 elements.

**Fix:**

Replace the kernel launch with a host-side loop that writes directly to pinned or device
memory using `cudaMemcpyAsync`:

```cpp
// Replace populateChunkArraysKernel with a CPU-side fill of a pinned staging buffer:
static uint8_t** h_input_ptrs_stage  = nullptr;  // pinned host staging
static uint8_t** h_output_ptrs_stage = nullptr;
static size_t*   h_sizes_stage       = nullptr;
// (Allocate with cudaMallocHost in init for zero-copy access)

for (size_t i = 0; i < num_chunks; i++) {
    size_t offset    = i * chunk_bytes;
    size_t remaining = total_bytes - offset;
    h_input_ptrs_stage[i]  = base_input  + offset;
    h_output_ptrs_stage[i] = base_output + offset;
    h_sizes_stage[i]       = (remaining < chunk_bytes) ? remaining : chunk_bytes;
}
cudaMemcpyAsync(d_input_ptrs,  h_input_ptrs_stage,  num_chunks * sizeof(uint8_t*), H2D, stream);
cudaMemcpyAsync(d_output_ptrs, h_output_ptrs_stage, num_chunks * sizeof(uint8_t*), H2D, stream);
cudaMemcpyAsync(d_sizes,       h_sizes_stage,       num_chunks * sizeof(size_t),   H2D, stream);
// No explicit sync needed — shuffle kernel is on same stream.
```

Or, better: redesign `byte_shuffle_kernel_specialized` to compute chunk offsets inline
from `base_input`, `total_bytes`, and `chunk_bytes` passed as scalars, eliminating the
pointer arrays entirely.

---

## Finding O — 🟡 runStatsKernels: cudaFree on every error path is an implicit GPU sync

**File:** `src/stats/stats_kernel.cu:334–411`
**Estimated overhead:** ~10–100 µs on error paths (each cudaFree on d_workspace syncs GPU)

Inside `runStatsKernels`, every error path calls `cudaFree(d_workspace)`:

```cpp
// Multiple occurrences, e.g.:
err = cudaMemsetAsync(d_workspace, 0, workspace_size, stream);
if (err != cudaSuccess) {
    cudaFree(d_workspace);   // ← implicit GPU sync
    return -1;
}
err = cudaMemcpyAsync(d_stats, &h_init, sizeof(AutoStatsGPU), H2D, stream);
if (err != cudaSuccess) {
    cudaFree(d_workspace);   // ← another implicit GPU sync
    return -1;
}
```

While these are error paths and do not affect the happy path, the workspace allocation
and these frees are the targets of Finding 2 in `CUDA_MEMCPY_AUDIT.md`. Once the
workspace is pre-allocated (per that fix), these error paths must also be updated to
not call `cudaFree` on the pre-allocated buffer. Currently this is a latent bug: after
applying Finding 2's fix, any of these early `return -1` paths would free the
permanently-allocated workspace, causing a use-after-free on the next call.

**Fix:**

When the pre-alloc fix from `CUDA_MEMCPY_AUDIT.md` Finding 2 is applied, remove all
`cudaFree(d_workspace)` calls from error paths in `runStatsKernels`. The workspace
is owned by `allocInferenceBuffers` / `freeInferenceBuffers` exclusively.

---

## Finding P — 🔵 NNWeightsGPU loaded to GPU even when CPU inference path is used

**File:** `src/nn/nn_gpu.cu:443–456`
**Impact:** ~78 KB wasted GPU memory allocation (one-time, at load)

`loadNNFromBinary` always allocates and uploads weights to GPU:

```cpp
cudaMalloc(&d_nn_weights, sizeof(NNWeightsGPU));   // ~78 KB on GPU
cudaMemcpy(d_nn_weights, &h_weights, sizeof(NNWeightsGPU), cudaMemcpyHostToDevice);
```

When the CPU inference path from `CUDA_MEMCPY_AUDIT.md` Finding 9 is implemented,
`d_nn_weights` is only needed as a fallback (or for `nn_reinforce_apply`'s D2H copy
in `nn_reinforce_init`). The GPU allocation could be made conditional:

```cpp
// Only allocate GPU weights if GPU kernel fallback is needed:
if (!g_host_weights_valid) {   // if CPU path is available, skip GPU alloc
    cudaMalloc(&d_nn_weights, sizeof(NNWeightsGPU));
    cudaMemcpy(d_nn_weights, &h_weights, ...);
}
```

For `nn_reinforce_apply`, the H2D copy can be deferred until the GPU kernel is
actually invoked.

---

## Finding Q — 🔵 gpucompress_stats_t Python mirror is structurally incomplete

**File:** `neural_net/core/gpu_library.py:60–71`
**Impact:** Silent struct misalignment — Python reads wrong field values from C

The Python ctypes mirror of `gpucompress_stats_t` only maps 9 fields:

```python
class gpucompress_stats_t(ctypes.Structure):
    _fields_ = [
        ('original_size',      ctypes.c_size_t),
        ('compressed_size',    ctypes.c_size_t),
        ('compression_ratio',  ctypes.c_double),
        ('entropy_bits',       ctypes.c_double),
        ('mad',                ctypes.c_double),
        ('second_derivative',  ctypes.c_double),
        ('algorithm_used',     ctypes.c_int),
        ('preprocessing_used', ctypes.c_uint),
        ('throughput_mbps',    ctypes.c_double),
    ]
```

The C struct `gpucompress_stats_t` (in `gpucompress.h`) contains additional fields
including `predicted_ratio`, `predicted_comp_time_ms`, `actual_comp_time_ms`,
`sgd_fired`, `exploration_triggered`, `nn_original_action`, and `nn_final_action`.
Since ctypes reads field values by byte offset, any field after `throughput_mbps` is
misread, and the benchmark scripts silently receive garbage values for training data.

**Fix:** Extend the Python struct to match all fields of the C definition exactly,
in the same order and with matching `ctypes` types. Add an `assert ctypes.sizeof(...)`
check against the known C struct size after any future struct changes.

---

## Summary Table

| # | Finding | Severity | Location | Category |
|---|---------|----------|----------|----------|
| A | `byte_shuffle_simple`: 4 cudaMalloc + hidden sync per call | 🔴 Critical | `byte_shuffle_kernels.cu:165` | Redundant alloc + sync |
| B | `byte_shuffle_simple`: chunk pointer array memory leak | 🔴 Critical | `byte_shuffle_kernels.cu:270` | Memory leak |
| C | `dequantize_simple`: per-call cudaMalloc + redundant sync | 🟠 High | `quantization_kernels.cu:573` | Redundant alloc + sync |
| D | Quant/dequant kernels: double arithmetic on float data | 🟠 High | `quantization_kernels.cu:173` | GPU throughput |
| E | `printf`/`fprintf` on every ALGO_AUTO compress | 🟠 High | `gpucompress_api.cpp:362` | I/O latency |
| F | Stats workspace: double initialization (memset + H2D) | 🟡 Medium | `stats_kernel.cu:343` | Redundant write |
| G | Decompress: redundant sync before post-processing | 🟠 High | `gpucompress_api.cpp:1166` | Redundant sync |
| H | `nn_reinforce_apply`: synchronous ~78 KB H2D memcpy | 🟠 High | `nn_reinforce.cpp:278` | Blocking transfer |
| I | Per-call nvCOMP manager construction | 🟠 High | `compression_factory.cpp:68` | Redundant object init |
| J | Per-call cudaMalloc for d_input and d_output | 🔴 Critical | `gpucompress_api.cpp:283,449` | Redundant large alloc |
| K | `gpucompress_compute_stats`: synchronous H2D memcpy | 🟡 Medium | `gpucompress_api.cpp:1327` | Blocking transfer |
| L | `experience_buffer_append`: fflush on every sample | 🟡 Medium | `experience_buffer.cpp:111` | I/O latency |
| M | `atomicAddDouble`: CAS loop instead of native SM6+ atomics | 🟡 Medium | `stats_kernel.cu:54` | GPU throughput |
| N | `populateChunkArraysKernel`: GPU kernel for trivial CPU loop | 🟡 Medium | `byte_shuffle_kernels.cu:143` | Unnecessary kernel launch |
| O | `cudaFree(d_workspace)` in error paths (latent post-fix bug) | 🟡 Medium | `stats_kernel.cu:334` | Latent use-after-free |
| P | GPU weights allocated even when CPU inference path used | 🔵 Low | `nn_gpu.cu:443` | Wasted GPU memory |
| Q | Python `gpucompress_stats_t` struct incomplete | 🔵 Low / Bug | `gpu_library.py:60` | Silent misread |

---

## Recommended Fix Priority

Apply in order of latency/correctness impact per effort:

1. **Finding B** (add destructor, ~5 lines) — stops a memory leak on every compress call with shuffle.
2. **Finding G** (delete 1 line at api.cpp:1166) — free sync in decompress hot path.
3. **Finding E** (guard printfs behind verbosity flag) — eliminates I/O syscalls on every compress.
4. **Finding D** (specialize quantize/dequant for float32) — 2×–32× faster quant kernels on consumer GPUs.
5. **Finding J** (grow-or-reuse d_input / d_output) — eliminates the two largest per-call allocs.
6. **Finding A** (pre-alloc chunk ptr arrays, remove hidden sync) — eliminates 4 mallocs + 1 sync per shuffle.
7. **Finding C** (pre-alloc dequant output + remove redundant sync) — mirrors Finding A for decompress.
8. **Finding I** (cache nvCOMP manager per algorithm) — eliminates manager construction overhead.
9. **Finding M** (use native `atomicAdd` for double on SM6+) — faster stats reduction.
10. **Finding N** (replace `populateChunkArraysKernel` with CPU loop) — eliminates extra kernel launch + sync.
11. **Finding H** (async weight copy in reinforce) — reduces SGD-path latency.
12. **Finding F** (split memset from H2D in stats init) — minor, easy.
13. **Finding L** (remove per-append fflush) — I/O improvement for experience logging.
14. **Finding K** (async H2D in `compute_stats`) — trivial one-liner fix.
15. **Finding Q** (fix Python struct) — correctness fix for benchmark/training pipeline.
16. **Finding O** (audit error-path frees after pre-alloc fix) — must be done alongside CUDA_MEMCPY_AUDIT Finding 2.
17. **Finding P** (conditional GPU weight alloc) — memory efficiency cleanup, deferred.
