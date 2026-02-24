# GPUCompress HDF5 Integration

## Historical Note

This document describes the original commit (`6c4909a`) that introduced the shared library C API and HDF5 filter plugin. The Q-Table RL system described in some sections has since been replaced by a neural network-based approach (see `neural_net/`).

## Commit Overview

This commit transforms GPUCompress from a standalone CLI tool into a reusable shared library (`libgpucompress.so`) with a public C API and an HDF5 filter plugin so any HDF5 application can transparently use GPU compression.

---

## Full Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION LAYER                          │
│                                                                        │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
│  │ gpu_compress  │   │  gpu_decompress  │   │  Any HDF5 App (h5py,  │  │
│  │ (CLI binary)  │   │  (CLI binary)    │   │  C/C++, Fortran...)   │  │
│  └──────┬───────┘   └───────┬──────────┘   └──────────┬─────────────┘  │
│         │                   │                         │                │
└─────────┼───────────────────┼─────────────────────────┼────────────────┘
          │                   │                         │
          ▼                   ▼                         ▼
┌──────────────────────────────────┐   ┌─────────────────────────────────┐
│    libgpucompress.so (NEW)       │   │  libH5Zgpucompress.so (NEW)    │
│    ─────────────────────         │   │  ─────────────────────────     │
│    Public C API:                 │   │  HDF5 Filter Plugin:           │
│    gpucompress.h                 │◄──┤  H5Zgpucompress.c              │
│                                  │   │  Filter ID: 305                │
│  ┌─────────────────────────────┐ │   │                                │
│  │ gpucompress_api.cpp         │ │   │  Callbacks:                    │
│  │ > gpucompress_init()        │ │   │  > can_apply()                 │
│  │ > gpucompress_compress()    │ │   │  > set_local()                 │
│  │ > gpucompress_decompress()  │ │   │  > filter() (compress/decomp)  │
│  │ > gpucompress_load_qtable() │ │   │                                │
│  │ > gpucompress_recommend()   │ │   │  Plugin entry points:          │
│  └─────────┬───────────────────┘ │   │  > H5PLget_plugin_type()       │
│            │                     │   │  > H5PLget_plugin_info()        │
│            ▼                     │   └─────────────────────────────────┘
│  ┌─────────────────────────────┐ │
│  │ internal.hpp    │ │
│  │ > State/Action encoding     │ │
│  │ > Algorithm mapping         │ │
│  │ > LibraryState singleton    │ │
│  └─────────┬───────────────────┘ │
│            │                     │
│       ┌────┴─────┐               │
│       ▼          ▼               │
│ ┌──────────┐ ┌──────────────┐    │
│ │entropy_  │ │qtable_gpu.cu │    │
│ │kernel.cu │ │              │    │
│ │          │ │> __constant__│    │
│ │> GPU     │ │  d_qtable[]  │    │
│ │  byte    │ │> JSON/Binary │    │
│ │  histogram│ │  loading    │    │
│ │> Shannon │ │> Argmax      │    │
│ │  entropy │ │  kernel      │    │
│ │  H=-Sp*  │ │  (inference) │    │
│ │  log2(p) │ │              │    │
│ └──────────┘ └──────────────┘    │
│            │                     │
│            ▼                     │
│  ┌─────────────────────────────┐ │
│  │   EXISTING COMPONENTS              │ │
│  │  > core/compression_factory.cpp   │ │
│  │  > preprocessing/byte_shuffle_kernels.cu  │ │
│  │  > preprocessing/quantization_kernels.cu  │ │
│  │  > core/compression_header.h      │ │
│  └─────────┬───────────────────┘ │
│            ▼                     │
│      ┌──────────┐               │
│      │  nvcomp  │               │
│      │ (8 algos)│               │
│      └──────────┘               │
└──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              OFFLINE RL TRAINING SYSTEM (Python, NEW)                   │
│                                                                        │
│  scripts/run_rl_training.sh                                            │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  rl/trainer.py  -  QTableTrainer                             │      │
│  │  Main training loop:                                         │      │
│  │    for each epoch:                                           │      │
│  │      for each file:                                          │      │
│  │        1. Calculate entropy  -->  rl/executor.py             │      │
│  │        2. Encode state       -->  rl/qtable.py               │      │
│  │        3. Select action      -->  rl/policy.py               │      │
│  │        4. Execute compress   -->  rl/executor.py (subprocess)│      │
│  │        5. Compute reward     -->  rl/reward.py               │      │
│  │        6. Update Q(s,a)      -->  rl/qtable.py               │      │
│  │      Decay epsilon                                           │      │
│  │      Save checkpoint                                         │      │
│  └──────────────────────────────────────────────────────────────┘      │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────┐    ┌─────────────────┐                               │
│  │ qtable.json  │--->│ export_qtable.py│---> qtable.bin --> GPU         │
│  │ (30x32 table)│    │                 │     (magic "QTAB")            │
│  └──────────────┘    └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Shared Library with C API (`libgpucompress.so`)

### Goal

Make GPUCompress callable from any language (C, Python, Fortran, HDF5), not just the CLI binaries.

### Files Created

| File | Purpose |
|------|---------|
| `include/gpucompress.h` | Public C header defining the entire API (~400 lines) |
| `src/api/gpucompress_api.cpp` | Implementation of that API (~835 lines) |
| `src/api/internal.hpp` | Internal C++ utilities, not exposed (~239 lines) |

### How It Works

**Initialization** (`gpucompress_init`): Sets up the CUDA device, creates a default stream, optionally loads a Q-Table. Uses reference counting + mutex for thread safety so multiple callers can init/cleanup independently.

**Compression pipeline** (`gpucompress_compress`):

```
Host input
  --> cudaMemcpy to GPU
  --> [Auto-select algo via Q-Table if ALGO_AUTO]
  --> [Optional: Quantize (error-bounded lossy)]
  --> [Optional: Byte Shuffle]
  --> nvcomp compress
  --> Prepend 64-byte header
  --> cudaMemcpy back to host
```

**Decompression pipeline** (`gpucompress_decompress`):

```
Host input
  --> Read 64-byte header (magic, version, algo, preprocessing metadata)
  --> cudaMemcpy to GPU
  --> nvcomp decompress (auto-detects algorithm)
  --> [Reverse: Unshuffle]
  --> [Reverse: Dequantize]
  --> cudaMemcpy back to host
```

**Auto algorithm selection** (`GPUCOMPRESS_ALGO_AUTO`): When the user passes `ALGO_AUTO`, the compress function:

1. Calculates entropy on GPU
2. Encodes a state from entropy + error bound
3. Looks up the Q-Table for the best action
4. Decodes the action into (algorithm, preprocessing)

### Key Design Decisions

- `extern "C"` wrapper so the `.so` has C linkage (callable from C, Python ctypes, HDF5 plugins)
- Config struct passed by pointer; `NULL` means "use defaults" (LZ4, no preprocessing)
- Stats struct is optional output; pass `NULL` to skip
- Thread-safe initialization with atomic reference counting

### API Summary

```c
// Lifecycle
gpucompress_error_t gpucompress_init(const char* qtable_path);
void                gpucompress_cleanup(void);
int                 gpucompress_is_initialized(void);

// Core operations
gpucompress_error_t gpucompress_compress(input, size, output, &out_size, &config, &stats);
gpucompress_error_t gpucompress_decompress(input, size, output, &out_size);

// Utilities
size_t              gpucompress_max_compressed_size(size_t input_size);
gpucompress_error_t gpucompress_get_original_size(compressed, &original_size);
gpucompress_error_t gpucompress_calculate_entropy(data, size, &entropy);
gpucompress_error_t gpucompress_load_qtable(filepath);
gpucompress_error_t gpucompress_recommend_config(entropy, error_bound, &algo, &preproc);
```

### Error Codes

| Code | Meaning |
|------|---------|
| `GPUCOMPRESS_SUCCESS` (0) | Operation succeeded |
| `GPUCOMPRESS_ERROR_INVALID_INPUT` (-1) | Invalid parameter |
| `GPUCOMPRESS_ERROR_CUDA_FAILED` (-2) | CUDA operation failed |
| `GPUCOMPRESS_ERROR_COMPRESSION` (-3) | Compression failed |
| `GPUCOMPRESS_ERROR_DECOMPRESSION` (-4) | Decompression failed |
| `GPUCOMPRESS_ERROR_OUT_OF_MEMORY` (-5) | Allocation failed |
| `GPUCOMPRESS_ERROR_QTABLE_NOT_LOADED` (-6) | Q-Table not available for AUTO |
| `GPUCOMPRESS_ERROR_INVALID_HEADER` (-7) | Bad compression header |
| `GPUCOMPRESS_ERROR_NOT_INITIALIZED` (-8) | Library not initialized |
| `GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL` (-9) | Output buffer too small |

---

## Step 2: GPU Entropy Calculation (`entropy_kernel.cu`)

### Goal

Measure data entropy on GPU to classify data for the RL agent. The entropy value determines which "state" the Q-Table looks up, driving algorithm selection.

### Algorithm (3-Kernel Pipeline)

```
Input bytes (GPU memory)
        |
        v
+-----------------------------------+
| Kernel 1: histogramKernelVec4     |
| - 256-bin shared-memory histogram |
| - atomicAdd per byte value        |
| - Reads 4 bytes at a time (uint32)|
| - Block histograms merged to      |
|   global histogram                |
+-----------------------------------+
        |
        v
+-----------------------------------+
| Kernel 2: entropyFromHistogram    |
| - 1 block, 256 threads            |
| - Each thread: -p_i * log2(p_i)  |
| - Parallel block reduction        |
|   (power-of-2 tree sum)          |
+-----------------------------------+
        |
        v
   Entropy: 0.0 to 8.0 bits
   (0 = all identical, 8 = uniform random)
```

### Key Implementation Details

- **Vectorized reads**: `histogramKernelVec4` processes 4 bytes at a time via `uint32_t` reads, extracting each byte with bit shifts. This improves memory throughput over byte-by-byte reads.
- **Shared memory atomics**: Each block maintains a local histogram in shared memory (fast), then merges into the global histogram (one atomic per non-zero bin).
- **Grid sizing**: Capped at 1024 blocks to avoid oversubscribing the GPU.
- **Pre-allocated variant**: `calculateEntropyGPUWithBuffers` avoids `cudaMalloc`/`cudaFree` per call when entropy is calculated repeatedly.

---

## Step 3: GPU Q-Table for Inference (`qtable_gpu.cu`)

### Goal

Store the trained RL policy on GPU for fast lookup during compression. The Q-Table maps data characteristics to the best compression configuration.

### Q-Table Structure

```
30 states = 10 entropy bins x 3 error levels
32 actions = 8 algorithms x 2 quantization options x 2 shuffle options
960 floats total (~3.84 KB)
```

Stored in CUDA `__constant__` memory:

```cuda
__constant__ float d_qtable[960];  // Broadcast to all threads, cached in L1
```

### State Encoding

```
state = entropy_bin * 3 + error_level

  entropy_bin: int(entropy), clamped to [0, 9]
  error_level:
    0 = aggressive  (error_bound >= 0.01)
    1 = balanced     (error_bound >= 0.001)
    2 = precise      (error_bound < 0.001 or lossless)
```

### Action Encoding

```
action = algorithm_idx + quantization * 8 + shuffle * 16

  algorithm_idx: 0=LZ4, 1=Snappy, 2=Deflate, 3=Gdeflate,
                 4=Zstd, 5=ANS, 6=Cascaded, 7=Bitcomp
  quantization:  0=none, 1=linear
  shuffle:       0=none, 1=4-byte shuffle
```

### Loading

The loader auto-detects format by file extension:

- **Binary** (`.bin`): Magic `"QTAB"` (0x51544142), version, dimensions, flat float array
- **JSON** (`.json`): `{ "version": 1, "n_states": 30, "n_actions": 32, "q_values": [[...], ...] }`

Values are copied to constant memory via `cudaMemcpyToSymbol` and a host-side copy is kept for CPU access.

### Inference

**GPU path** (`qtableArgmaxKernel`): 1 block of 32 threads performs parallel reduction to find the argmax action for a given state.

**CPU path** (`getBestActionCPU`): Simple linear scan over 32 values. Actually used in practice since single-lookup latency doesn't justify a kernel launch.

---

## Step 4: HDF5 Filter Plugin (`H5Zgpucompress.c`)

### Goal

Make GPU compression transparent to any HDF5 application. When a dataset is read/written, HDF5 automatically calls GPUCompress.

### How HDF5 Filters Work

```
Application --> H5Dwrite(dataset, data) --> HDF5 calls filter --> compressed chunk stored
Application --> H5Dread(dataset, buf)   --> HDF5 calls filter --> decompressed chunk returned
```

### Implementation

Written in pure C for maximum HDF5 compatibility.

**Filter class** (`H5Z_class2_t`): Registered with filter ID 305 (testing range 256-511). Three callbacks:

| Callback | Purpose |
|----------|---------|
| `can_apply` | Validates GPUCompress can initialize (CUDA available) |
| `set_local` | Per-dataset setup (currently no-op, could auto-detect shuffle size) |
| `filter` | Main compress/decompress dispatch based on `H5Z_FLAG_REVERSE` |

**Filter parameters** (`cd_values[5]`): Packed into HDF5's unsigned int array:

```
cd_values[0]: algorithm (0=AUTO, 1=LZ4, 2=Snappy, ..., 8=Bitcomp)
cd_values[1]: preprocessing flags
cd_values[2]: shuffle element size (0, 2, 4, or 8)
cd_values[3]: error_bound low bits  (double packed via union)
cd_values[4]: error_bound high bits (double packed via union)
```

**Plugin mechanism**: Exports `H5PLget_plugin_type()` and `H5PLget_plugin_info()` so HDF5 can auto-discover the filter when `HDF5_PLUGIN_PATH` is set.

**Smart fallback**: If compression makes data larger, the filter returns the original uncompressed data (line 300-303 in `H5Zgpucompress.c`).

### Usage Example

```c
#include <hdf5.h>
#include "gpucompress_hdf5.h"

// Register filter (automatic if using plugin)
H5Z_gpucompress_register();

// Create dataset with GPU compression
hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
H5Pset_chunk(dcpl, 2, chunk_dims);
H5Pset_gpucompress(dcpl,
    GPUCOMPRESS_ALGO_AUTO,  // RL-based selection
    0,                       // no extra preprocessing
    4,                       // 4-byte shuffle
    0.0);                    // lossless

hid_t dataset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, space,
                           H5P_DEFAULT, dcpl, H5P_DEFAULT);
H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
```

### Plugin Setup

```bash
# Set plugin path so HDF5 auto-discovers the filter
export HDF5_PLUGIN_PATH=/path/to/GPUCompress/build

# Now any HDF5 application can use the filter
h5repack -f "gpucompress=0,0,4,0,0" input.h5 output.h5
```

---

## HDF5 + AUTO Algorithm (RL Q-Table): Full Execution Flow

This section traces every function call, GPU allocation, and kernel launch when a user writes a chunked HDF5 dataset with the GPUCompress filter configured for **AUTO** algorithm selection. The RL-trained Q-Table dynamically picks the best compression algorithm and preprocessing based on data characteristics (entropy and error tolerance). The example uses `error_bound=0.001`, which the RL agent uses as input to its state encoding.

### User Code

```c
// Load the trained RL Q-Table into GPU constant memory
gpucompress_init("rl/models/qtable.bin");

hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
hsize_t chunk_dims[2] = {1000, 1000};
H5Pset_chunk(dcpl, 2, chunk_dims);

// AUTO = let the RL Q-Table choose algorithm + preprocessing
// No explicit preprocessing flags -- the Q-Table decides everything
H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO,
                   GPUCOMPRESS_PREPROC_NONE, 0, 0.001);

hid_t dset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, space,
                        H5P_DEFAULT, dcpl, H5P_DEFAULT);
H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
// ...later...
H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
```

### Compression Flow (H5Dwrite)

```
USER APPLICATION
================
  float data[1000][1000];
  H5Dwrite(dset, ..., data)
       |
       | HDF5 splits data into chunks (1000x1000 floats = 4 MB per chunk)
       | For EACH chunk, HDF5 calls the filter pipeline:
       |
       v
+============================================================================+
|  HDF5 FILTER PIPELINE                                                      |
|  File: src/hdf5/H5Zgpucompress.c                                          |
|  Function: H5Z_filter_gpucompress()         flags = 0 (compression)        |
+============================================================================+
       |
       |  (1) Parse cd_values[5] into gpucompress_config_t
       |      cd_values[0] = 0           --> config.algorithm = GPUCOMPRESS_ALGO_AUTO
       |      cd_values[1] = 0x00        --> config.preprocessing = NONE
       |      cd_values[2] = 0           --> (no shuffle override)
       |      cd_values[3,4] = 0.001     --> config.error_bound = 0.001
       |
       |  (2) Allocate HDF5 output buffer
       |      max_out = gpucompress_max_compressed_size(4MB) = ~4.5MB
       |      new_buf = H5allocate_memory(max_out)
       |
       |  (3) Call C API
       |      gpucompress_compress(*buf, 4MB, new_buf, &new_size, &config, NULL)
       |
       v
+============================================================================+
|  C API: gpucompress_compress()                                             |
|  File: src/api/gpucompress_api.cpp:173                                     |
+============================================================================+
       |
       |  STAGE 1: HOST --> GPU TRANSFER
       |  ==============================
       |
       |  (a) cudaMalloc(&d_input, 4MB)              [GPU: allocate input buffer]
       |  (b) cudaMemcpyAsync(d_input, host_buf,     [DMA: host RAM --> GPU VRAM]
       |                      4MB, H2D, stream)
       |
       |      GPU VRAM state:
       |      +---------------------------+
       |      | d_input: 1M float32 values|  4,000,000 bytes
       |      | [0.173, -2.41, 0.007, ...]|
       |      +---------------------------+
       |
       |  STAGE 2: RL Q-TABLE ALGORITHM SELECTION  (ALGO_AUTO triggers this)
       |  ===================================================================
       |  File: src/api/gpucompress_api.cpp:225-259
       |
       |  cfg.algorithm == GPUCOMPRESS_ALGO_AUTO --> enter RL selection path
       |
       |  (a) CALCULATE ENTROPY ON GPU  (entropy_kernel.cu)
       |      ================================================
       |      gpucompress_entropy_gpu_impl(d_input, 4MB, stream)
       |
       |      KERNEL 1: histogramKernelVec4
       |      ==============================
       |      - Grid: min(4MB/256, 1024) = 1024 blocks x 256 threads
       |      - 256-bin shared-memory histogram per block
       |      - Each thread reads 4 bytes at a time via uint32_t
       |      - Extracts individual bytes with bit-shift: (word >> 0/8/16/24) & 0xFF
       |      - atomicAdd to shared memory histogram (fast, no bank conflicts)
       |      - __syncthreads(), then merge non-zero bins to global histogram
       |
       |      GPU VRAM state (temporary):
       |      +---------------------------+     +---------------------------+
       |      | d_input (4MB)             |     | d_histogram: 256 x uint  |
       |      | (unchanged)               |     | (byte frequency counts)  |
       |      +---------------------------+     +---------------------------+
       |
       |      KERNEL 2: entropyFromHistogramKernel
       |      =====================================
       |      - Grid: 1 block x 256 threads
       |      - Thread i: p_i = histogram[i] / total_count
       |      -            partial = -p_i * log2(p_i)  (0 if p_i == 0)
       |      - Parallel block reduction (power-of-2 tree sum):
       |        stride = 128, 64, 32, 16, 8, 4, 2, 1
       |        each step: shared[tid] += shared[tid + stride]
       |      - Thread 0 writes final sum to d_entropy
       |      - Result: entropy = 5.3 bits  (example: moderately structured data)
       |
       |      cudaMemcpy(&entropy, d_entropy, sizeof(double), D2H)
       |      cudaFree(d_histogram), cudaFree(d_entropy)
       |
       |  (b) MAP ERROR BOUND TO ERROR LEVEL  (internal.hpp:129)
       |      ==============================================================
       |      errorBoundToLevel(0.001):
       |        error_bound = 0.001
       |        0.001 >= 0.01?  NO   (not aggressive)
       |        0.001 >= 0.001? YES  --> error_level = 1 (balanced)
       |
       |      Error level mapping:
       |        Level 0: error_bound >= 0.01  (aggressive lossy)
       |        Level 1: error_bound >= 0.001 (balanced)
       |        Level 2: error_bound < 0.001 or 0.0  (precise / lossless)
       |
       |  (c) ENCODE STATE  (internal.hpp:102)
       |      =============================================
       |      encodeState(entropy=5.3, error_level=1):
       |        entropy_bin = int(5.3) = 5  (clamped to [0, 9])
       |        state = entropy_bin * NUM_ERROR_LEVELS + error_level
       |        state = 5 * 3 + 1 = 16
       |
       |      State space (30 states = 10 entropy bins x 3 error levels):
       |      +------+------+------+------+------+------+------+------+------+------+
       |      | Ent  |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   | 9  |
       |      +------+------+------+------+------+------+------+------+------+------+
       |      | Aggr | s=0  | s=3  | s=6  | s=9  | s=12 |*s=15*| s=18 | s=21 | s=24 |s=27|
       |      | Bal  | s=1  | s=4  | s=7  | s=10 | s=13 |*s=16*| s=19 | s=22 | s=25 |s=28|
       |      | Prec | s=2  | s=5  | s=8  | s=11 | s=14 |*s=17*| s=20 | s=23 | s=26 |s=29|
       |      +------+------+------+------+------+------+------+------+------+------+
       |                                                   ^ our state = 16
       |
       |  (d) Q-TABLE LOOKUP: GET BEST ACTION  (qtable_gpu.cu)
       |      ==================================================
       |      gpucompress_qtable_get_best_action_impl(state=16)
       |        --> getBestActionCPU(state=16)
       |
       |      Scans host-side Q-Table copy (g_qtable_host[]):
       |        Row 16: 32 Q-values (one per action)
       |        +--------+--------+--------+--------+-----+--------+
       |        | a=0    | a=1    | a=2    | ...    |     | a=31   |
       |        | Q=0.42 | Q=0.38 | Q=0.61 | ...    |     | Q=0.29 |
       |        +--------+--------+--------+--------+-----+--------+
       |
       |      Linear scan: best_action = argmax(Q[16, 0..31])
       |      Example result: best_action = 20
       |
       |      Why CPU path (not GPU kernel)?
       |        - Single lookup: 32 comparisons on CPU < kernel launch overhead
       |        - The GPU argmax kernel exists (qtableArgmaxKernel) but is not
       |          used here; it's designed for batched inference scenarios
       |
       |  (e) DECODE ACTION  (internal.hpp:114)
       |      ==============================================
       |      decodeAction(action_id=20):
       |        algorithm       = 20 % 8  = 4   --> ZSTD (index 4)
       |        use_quantization = (20 / 8) % 2 = 0  --> no quantization
       |        shuffle_size    = (20 / 16) % 2 = 1  --> 4-byte shuffle
       |
       |      Action encoding scheme (5-bit):
       |      +-------+-------+-------+
       |      | Bit 4 | Bit 3 |Bits 0-2|
       |      |shuffle | quant | algo  |
       |      +-------+-------+-------+
       |      |   1   |   0   |  100  |  = 10100 = 20
       |      +-------+-------+-------+
       |
       |      Algorithm index to enum mapping (+1 offset):
       |        Q-Table idx 0=LZ4, 1=Snappy, 2=Deflate, 3=Gdeflate,
       |                    4=ZSTD, 5=ANS, 6=Cascaded, 7=Bitcomp
       |        algo_to_use = (gpucompress_algorithm_t)(4 + 1) = GPUCOMPRESS_ALGO_ZSTD
       |
       |  (f) SET PREPROCESSING FLAGS  (gpucompress_api.cpp:246-254)
       |      =======================================================
       |      decoded.shuffle_size = 4 (> 0):
       |        preproc_to_use |= GPUCOMPRESS_PREPROC_SHUFFLE_4  (0x02)
       |      decoded.use_quantization = false:
       |        (skip quantization flag)
       |
       |      Final RL decision:
       |      +--------------------------------------------------+
       |      | Algorithm:      ZSTD  (strong ratio, mid-speed) |
       |      | Quantization:   OFF   (keep lossless path)       |
       |      | Byte Shuffle:   4-byte (group similar bytes)     |
       |      +--------------------------------------------------+
       |
       |      Why this decision makes sense for entropy=5.3, balanced error:
       |        - Mid-entropy data has structure but isn't trivially compressible
       |        - ZSTD's LZ77+Huffman handles this well (better than fast LZ4)
       |        - Byte shuffle exposes byte-level patterns for better LZ77 matching
       |        - No quantization: error_bound=0.001 is balanced, but Q-Table
       |          learned that shuffle alone gives sufficient ratio improvement
       |
       |  STAGE 3: QUANTIZATION  (SKIPPED by RL decision)
       |  =================================================
       |
       |  The Q-Table decided use_quantization=false, so no quantization is applied.
       |  d_compress_input = d_input  (original float32 data)
       |  compress_input_size = 4MB   (no size reduction from quantization)
       |
       |  NOTE: If the RL agent had chosen an action with quantization enabled
       |  (e.g., action=28 = ZSTD + quantize + shuffle), this stage would execute
       |  the quantization pipeline as described in the quantization documentation.
       |
       |  STAGE 4: BYTE SHUFFLE  (preprocessing/byte_shuffle.cuh / preprocessing/byte_shuffle_kernels.cu)
       |  =====================================================================
       |
       |  (a) shuffle_size = 4 (chosen by RL Q-Table via GPUCOMPRESS_PREPROC_SHUFFLE_4)
       |      Applied to raw float32 data, using 4-byte element grouping (sizeof(float))
       |
       |  (b) byte_shuffle_simple(d_input, 4MB, 4, 256KB, AUTO, stream)
       |      |
       |      |  Internally:
       |      |  - cudaMalloc(&d_shuffled, 4MB)
       |      |  - Splits into chunks: 4MB / 256KB = 16 chunks
       |      |  - createDeviceChunkArrays (GPU kernel populates pointers)
       |      |
       |      |  KERNEL 3: populateChunkArraysKernel
       |      |  ====================================
       |      |  - 1 block x 16 threads
       |      |  - Each thread computes offset/size for one chunk
       |      |  - Writes chunk pointers directly in GPU memory
       |      |
       |      |  AUTO kernel selection:
       |      |  element_size=4, aligned --> selects VECTORIZED_4B
       |      |
       |      |  KERNEL 4: byte_shuffle_kernel_vectorized_4byte
       |      |  ===============================================
       |      |  - 1 warp (32 threads) per chunk = 16 warps total
       |      |  - Uses float4 for 128-bit reads (4 elements at once)
       |      |  - Reorders bytes by position within each 4-byte float:
       |      |    Input:  [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]...
       |      |    Output: [A0 B0 C0...][A1 B1 C1...][A2 B2 C2...][A3 B3 C3...]
       |      |  - Exponent bytes grouped, mantissa bytes grouped --> better LZ77
       |      |
       |      v
       |      GPU VRAM state:
       |      +---------------------------+     +---------------------------+
       |      | d_input (4MB)             |     | d_shuffled (4MB)          |
       |      | original floats           |     | byte-reordered floats    |
       |      +---------------------------+     +---------------------------+
       |
       |      d_compress_input = d_shuffled  (pointer swap)
       |      compress_input_size = 4MB      (unchanged, no quantization)
       |
       |  cudaStreamSynchronize(stream)  -- ensure preprocessing done
       |
       |  STAGE 5: NVCOMP COMPRESSION  (core/compression_factory.cpp)
       |  ======================================================
       |
       |  The RL agent chose ZSTD. CompressionFactory creates the right manager:
       |
       |  (a) createCompressionManager(ZSTD, 64KB chunk, stream, d_shuffled)
       |      --> nvcomp::ZstdManager with default settings
       |
       |  (b) comp_config = compressor->configure_compression(4MB)
       |      --> max_compressed_buffer_size = ~4.5MB
       |
       |  (c) cudaMalloc(&d_output, 64 + 4.5MB)   [header + compressed]
       |      d_compressed = d_output + 64          [skip header area]
       |
       |  (d) KERNEL 5+: compressor->compress(d_shuffled, d_compressed, config)
       |      ===================================================================
       |      nvcomp ZSTD internally launches multiple kernels:
       |      - Splits 4MB into 64KB chunks (64 chunks)
       |      - LZ77 matching on GPU (finds repeated byte sequences)
       |      - Huffman encoding on GPU (entropy coding)
       |      - Bitstream packing
       |      Output: compressed_size (e.g., ~600KB for mid-entropy shuffled data)
       |
       |      Note: Without quantization, input is 4MB (not 2MB), so compressed
       |      output is larger than the quantized case, but data is fully lossless.
       |
       |  (e) compressed_size = compressor->get_compressed_output_size(d_compressed)
       |      total_size = 64 + 600KB = ~600KB
       |
       |      GPU VRAM state:
       |      +------------------------------------------------------------------+
       |      | d_output:                                                        |
       |      | [64-byte header][~600KB ZSTD compressed shuffled float32 data]  |
       |      +------------------------------------------------------------------+
       |
       |  STAGE 6: BUILD HEADER  (core/compression_header.h)
       |  ===============================================
       |
       |  CompressionHeader (64 bytes, packed):
       |  +-----+-----+-------+--------+----------+----------+
       |  |magic|ver  |shuf=4 |qflags  |orig=4MB  |comp=600KB|
       |  |GPUC |  2  |       |NONE    |          |          |
       |  |     |     |       |disabled|          |          |
       |  |     |     |       |        |          |          |
       |  +-----+-----+-------+--------+----------+----------+
       |  |quant_error_bound  |quant_scale         |
       |  |0.0 (unused)       |0.0 (unused)        |
       |  +-------------------+--------------------+
       |  |data_min           |data_max            |
       |  |0.0 (unused)       |0.0 (unused)        |
       |  +-------------------+--------------------+
       |
       |  NOTE: Quantization fields are zeroed because the RL agent chose no
       |  quantization. The shuffle_element_size=4 is recorded so decompression
       |  knows to reverse the byte shuffle.
       |
       |  writeHeaderToDevice(d_output, header, stream)
       |    --> cudaMemcpyAsync header to first 64 bytes of d_output
       |
       |  STAGE 7: GPU --> HOST TRANSFER
       |  ===============================
       |
       |  cudaMemcpyAsync(host_output, d_output, ~600KB, D2H, stream)
       |  cudaStreamSynchronize(stream)
       |
       |  STAGE 8: CLEANUP
       |  =================
       |
       |  cudaFree(d_output)
       |  cudaFree(d_shuffled)
       |  cudaFree(d_input)
       |
       |  *output_size = ~600KB
       |  return GPUCOMPRESS_SUCCESS
       |
       v
+============================================================================+
|  Back in H5Z_filter_gpucompress()                                          |
+============================================================================+
       |
       |  Check: new_size (600KB) < nbytes (4MB)?  YES --> compression helped
       |
       |  H5free_memory(*buf)         [free original chunk buffer]
       |  *buf = new_buf              [replace with compressed data]
       |  *buf_size = max_out_size
       |  return new_size (600KB)
       |
       v
+============================================================================+
|  HDF5 Internal                                                             |
+============================================================================+
       |
       |  Stores 600KB compressed chunk to .h5 file on disk
       |  Chunk metadata records: filter ID 305, cd_values[5]
       |  Compression ratio: 4MB / 600KB = ~6.7x (lossless, shuffle-only)
       |
       v
    [data.h5 on disk]
```

### Decompression Flow (H5Dread)

Decompression is fully automatic and does **not** consult the Q-Table. All information needed to reverse the pipeline is stored in the 64-byte header that was written during compression.

```
USER APPLICATION
================
  float buf[1000][1000];
  H5Dread(dset, H5T_NATIVE_FLOAT, ..., buf)
       |
       | HDF5 reads compressed chunk from disk (~600KB)
       | Calls filter pipeline with H5Z_FLAG_REVERSE:
       |
       v
+============================================================================+
|  HDF5 FILTER PIPELINE                                                      |
|  Function: H5Z_filter_gpucompress()         flags = H5Z_FLAG_REVERSE       |
+============================================================================+
       |
       |  (1) gpucompress_get_original_size(*buf, &original_size)
       |      --> reads header.original_size = 4MB from first 64 bytes
       |
       |  (2) new_buf = H5allocate_memory(4MB)
       |
       |  (3) gpucompress_decompress(*buf, 600KB, new_buf, &new_size)
       |
       v
+============================================================================+
|  C API: gpucompress_decompress()                                           |
|  File: src/api/gpucompress_api.cpp:442                                     |
+============================================================================+
       |
       |  STAGE 1: READ HEADER (CPU, from host buffer)
       |  ==============================================
       |
       |  memcpy(&header, input, 64)
       |  Validate: header.magic == 0x43555047 ("GPUC")  --> OK
       |
       |  Extracted metadata:
       |    header.shuffle_element_size = 4   --> shuffle was applied
       |    header.quant_flags & 0x100 = 0    --> NO quantization was applied
       |    header.original_size = 4MB
       |    header.compressed_size = ~600KB
       |
       |  NOTE: The header is self-describing. Decompression does not need to
       |  know that AUTO was used -- it simply reads what was actually applied.
       |
       |  STAGE 2: HOST --> GPU TRANSFER
       |  ===============================
       |
       |  cudaMalloc(&d_compressed, 600KB)
       |  cudaMemcpyAsync(d_compressed, host_input, 600KB, H2D, stream)
       |  d_compressed_data = d_compressed + 64  [skip header]
       |
       |  STAGE 3: NVCOMP DECOMPRESSION  (core/compression_factory.cpp)
       |  =========================================================
       |
       |  (a) createDecompressionManager(d_compressed_data, stream)
       |      --> nvcomp auto-detects ZSTD from compressed data header
       |      --> returns ZstdManager
       |
       |  (b) decomp_config = decompressor->configure_decompression(d_compressed_data)
       |      decompressed_size = 4MB  (the shuffled float32 data)
       |
       |  (c) cudaMalloc(&d_decompressed, 4MB)
       |
       |  (d) KERNELS: decompressor->decompress(d_decompressed, d_compressed_data, ...)
       |      nvcomp ZSTD decompresses: 600KB --> 4MB
       |
       |      GPU VRAM state:
       |      +-------------------------------+
       |      | d_decompressed: 4MB           |
       |      | (shuffled float32 values)     |
       |      +-------------------------------+
       |
       |  STAGE 4: REVERSE BYTE SHUFFLE  (preprocessing/byte_shuffle_kernels.cu)
       |  ==========================================================
       |
       |  header.hasShuffleApplied() --> true (shuffle_element_size = 4)
       |
       |  byte_unshuffle_simple(d_decompressed, 4MB, 4, 256KB, AUTO, stream)
       |      |
       |      |  cudaMalloc(&d_unshuffled, 4MB)
       |      |
       |      |  KERNEL: byte_unshuffle_kernel (reverses the reordering)
       |      |  =======================================================
       |      |  Input:  [A0 B0 C0...][A1 B1 C1...][A2 B2 C2...][A3 B3 C3...]
       |      |  Output: [A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]...
       |      |
       |      v
       |      GPU VRAM state:
       |      +-------------------------------+
       |      | d_unshuffled: 4MB             |
       |      | (original float32 values,     |
       |      |  original byte order)         |
       |      +-------------------------------+
       |
       |      d_result = d_unshuffled
       |
       |  STAGE 5: DEQUANTIZATION  (SKIPPED)
       |  ====================================
       |
       |  header.hasQuantizationApplied() --> false (quant_flags bit 8 not set)
       |  No dequantization needed -- data is already in original float32 format.
       |
       |  STAGE 6: GPU --> HOST TRANSFER
       |  ===============================
       |
       |  cudaMemcpyAsync(host_output, d_unshuffled, 4MB, D2H, stream)
       |  cudaStreamSynchronize(stream)
       |
       |  STAGE 7: CLEANUP
       |  =================
       |
       |  cudaFree(d_unshuffled)
       |  cudaFree(d_decompressed)
       |  cudaFree(d_compressed)
       |
       |  *output_size = 4MB
       |  return GPUCOMPRESS_SUCCESS
       |
       v
+============================================================================+
|  Back in H5Z_filter_gpucompress()                                          |
+============================================================================+
       |
       |  H5free_memory(*buf)         [free compressed chunk buffer]
       |  *buf = new_buf              [replace with decompressed data]
       |  *buf_size = 4MB
       |  return 4MB
       |
       v
+============================================================================+
|  HDF5 Internal                                                             |
+============================================================================+
       |
       |  Returns 4MB float chunk to application via H5Dread()
       |  Data is EXACT (lossless -- no quantization was applied)
       |
       v
  float buf[1000][1000];  // Restored data available to user
```

### RL Decision Varies by Data: Example Scenarios

The same AUTO code path produces different decisions depending on data characteristics:

```
SCENARIO A: Low-entropy scientific data (entropy=1.2, error_bound=0.001)
  State: entropy_bin=1, error_level=1 --> state=4
  RL decision: action=0 --> LZ4, no quantization, no shuffle
  Rationale: Highly compressible data, LZ4 gives excellent ratio at maximum speed

SCENARIO B: Mid-entropy sensor data (entropy=5.3, error_bound=0.001)
  State: entropy_bin=5, error_level=1 --> state=16
  RL decision: action=20 --> ZSTD + 4-byte shuffle, no quantization
  Rationale: Structured but non-trivial data benefits from ZSTD + shuffle

SCENARIO C: High-entropy noisy data (entropy=7.8, error_bound=0.01)
  State: entropy_bin=7, error_level=0 --> state=21
  RL decision: action=29 --> ANS + quantization + 4-byte shuffle
  Rationale: Near-random data needs aggressive preprocessing to become compressible;
             quantization reduces precision (acceptable with aggressive error bound),
             shuffle groups byte patterns, ANS is the strongest entropy coder

SCENARIO D: High-entropy data, lossless required (entropy=7.8, error_bound=0.0)
  State: entropy_bin=7, error_level=2 --> state=23
  RL decision: action=21 --> ANS + 4-byte shuffle, no quantization
  Rationale: Can't quantize (lossless), but shuffle + ANS squeezes maximum ratio
```

### Data Size Through the Pipeline (AUTO: ZSTD + Shuffle, No Quantization)

```
COMPRESSION:                                DECOMPRESSION (reverse):

 float32 input     4,000,000 bytes          compressed      ~600,000 bytes
       |                                          |
       |  RL Q-Table lookup (CPU):                |  (no Q-Table needed --
       |  entropy=5.3, state=16                   |   header is self-describing)
       |  --> action=20: ZSTD + shuffle           |
       |                                          |
       v  byte shuffle (reorder)                  v  nvcomp ZSTD decompress
 shuffled float32  4,000,000 bytes          shuffled float32 4,000,000 bytes
       |                                          |
       v  nvcomp ZSTD compress                    v  byte unshuffle (restore order)
 compressed        ~600,000 bytes           float32 output  4,000,000 bytes
       |                                          |
       v  prepend 64-byte header                  |  (EXACT -- lossless path)
 total output      ~600,064 bytes                 |

 Ratio: 4,000,000 / 600,064 = ~6.7x (lossless)
```

### GPU Memory Allocations Timeline

```
Time -->

d_histogram (1KB) |██| free                                         (entropy calc)
d_entropy (8B)    |██| free                                         (entropy calc)
d_input (4MB)     |████████████████████████████████████████████████| cudaFree
d_shuffled (4MB)         |██████████████████████████████████████████| cudaFree
d_output (4.5MB)                |████████████████████████████████████| cudaFree

Peak GPU usage: ~12.5 MB for a 4MB input chunk
(Higher than explicit-algo case due to entropy calculation buffers + no quantization)
```

### CUDA Kernel Launch Sequence

| # | Kernel | Grid | What It Does |
|---|--------|------|-------------|
| 1 | `histogramKernelVec4` | 1024 blocks x 256 threads | Build 256-bin byte histogram for entropy |
| 2 | `entropyFromHistogramKernel` | 1 block x 256 threads | Compute H = -sum(p*log2(p)) via parallel reduction |
| - | *CPU: Q-Table lookup* | *(no kernel)* | *getBestActionCPU: scan 32 Q-values for state 16* |
| 3 | `populateChunkArraysKernel` | 1 block x 16 threads | Build chunk pointer arrays on GPU |
| 4 | `byte_shuffle_kernel_vectorized_4byte` | 4 blocks x 128 threads | Reorder bytes for compression |
| 5+ | nvcomp ZSTD internal kernels | (internal) | LZ77 + Huffman compression |
| N | `writeHeaderToDevice` (cudaMemcpyAsync) | - | Copy 64-byte header to output |

### Figure 1: AUTO Compression Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HDF5 APPLICATION                                   │
│                                                                             │
│  H5Pset_gpucompress(dcpl, ALGO_AUTO, PREPROC_NONE, 0, 0.001)              │
│  H5Dwrite(dataset, data)                                                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  HDF5 Filter Pipeline  (H5Zgpucompress.c)                                   │
│                                                                             │
│  cd_values[0]=0 (AUTO) ─── cd_values[3,4]=0.001 (error_bound)              │
│                     │                                                       │
│                     ▼                                                       │
│           gpucompress_compress(buf, 4MB, &config{AUTO, 0.001})             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────┐    ┌───────────────────────┐    ┌──────────────┐
   │  STAGE 1    │    │      STAGE 2          │    │  STAGES 3-8  │
   │  Host→GPU   │───▶│  RL Q-Table Selection │───▶│  Preprocess  │
   │  Transfer   │    │  (AUTO only)          │    │  + Compress  │
   │  4MB DMA    │    │                       │    │  + Transfer  │
   └─────────────┘    └───────────────────────┘    └──────────────┘
                              │
                   ┌──────────┴──────────┐
                   │  DETAILED BELOW     │
                   │  (Figure 2)         │
                   └─────────────────────┘
```

### Figure 2: RL Q-Table Algorithm Selection (Stage 2 Detail)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: d_input (4MB float32 on GPU) + error_bound (0.001 from config)     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: GPU ENTROPY CALCULATION            (entropy_kernel.cu)             │
│  ═══════════════════════════════                                            │
│                                                                             │
│  ┌─────────────────────────┐     ┌─────────────────────────┐               │
│  │ KERNEL: histogramVec4   │     │ KERNEL: entropyFromHist │               │
│  │ 1024 blocks x 256 thds │────▶│ 1 block x 256 threads   │               │
│  │                         │     │                         │               │
│  │ 4 bytes/thread via      │     │ p_i = count[i] / N     │               │
│  │ uint32 + bit shifts     │     │ H = -Σ p_i·log₂(p_i)  │               │
│  │                         │     │                         │               │
│  │ Output: histogram[256]  │     │ Output: entropy = 5.3   │               │
│  └─────────────────────────┘     └────────────┬────────────┘               │
└───────────────────────────────────────────────┼─────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: STATE ENCODING                     (internal.hpp)      │
│  ══════════════════════                                                     │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │  Entropy Binning     │    │  Error Level Mapping  │                      │
│  │                      │    │                       │                      │
│  │  entropy = 5.3       │    │  error_bound = 0.001  │                      │
│  │  bin = int(5.3) = 5  │    │  >= 0.01? NO          │                      │
│  │  (range: 0-9)        │    │  >= 0.001? YES → L1   │                      │
│  └──────────┬───────────┘    └──────────┬────────────┘                      │
│             │                           │                                   │
│             └──────────┬────────────────┘                                   │
│                        ▼                                                    │
│              state = bin × 3 + level                                        │
│              state = 5 × 3 + 1 = 16                                         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Q-TABLE LOOKUP (CPU)               (qtable_gpu.cu)                │
│  ════════════════════════════                                               │
│                                                                             │
│  Q-Table in GPU __constant__ memory (loaded at init from qtable.bin):      │
│                                                                             │
│         action: 0    1    2    3    4   ...  20   ...  31                   │
│                LZ4  SNP  DEF  GDF  ZST      ZST       BTC                 │
│                                              +shf      +q+s               │
│  state  ┌─────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐              │
│    0    │0.89 │0.72│0.45│0.51│0.63│    │    │    │    │    │              │
│    1    │0.85 │0.68│0.50│0.48│0.60│    │    │    │    │    │              │
│   ...   │     │    │    │    │    │    │    │    │    │    │              │
│  ►16◄   │0.42 │0.38│0.61│0.55│0.70│    │►0.82◄│   │    │0.29│  ◄── scan  │
│   ...   │     │    │    │    │    │    │    │    │    │    │              │
│   29    │0.15 │0.12│0.30│0.28│0.35│    │    │    │    │    │              │
│         └─────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘              │
│                                                                             │
│  getBestActionCPU(16): argmax over row 16 → best_action = 20               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ACTION DECODING                    (internal.hpp)      │
│  ═══════════════════════                                                    │
│                                                                             │
│  action_id = 20 = 0b10100                                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────┐                      │
│  │  Bit 4    │  Bit 3       │  Bits 2-1-0           │                      │
│  │  shuffle  │  quantize    │  algorithm index      │                      │
│  ├───────────┼──────────────┼───────────────────────┤                      │
│  │    1      │     0        │    1  0  0  (= 4)     │                      │
│  │  4-byte   │   none       │    ZSTD               │                      │
│  └───────────┴──────────────┴───────────────────────┘                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────┐                      │
│  │  FINAL RL DECISION:                               │                      │
│  │  Algorithm:     ZSTD  (algo_idx 4 + 1 = enum 5)  │                      │
│  │  Quantization:  OFF                               │                      │
│  │  Byte Shuffle:  4-byte                            │                      │
│  └──────────────────────────────────────────────────┘                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
                         Continue to Stage 3+
```

### Figure 3: Full Compression Pipeline (Stages 1-8)

```
┌───────────┐
│ Host RAM  │     STAGE 1: DMA Transfer
│ float32   │     ══════════════════════
│ 4MB chunk │
└─────┬─────┘
      │ cudaMemcpyAsync (H2D)
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GPU VRAM                                  │
│                                                                  │
│  ┌─────────────┐   STAGE 2: RL Selection                        │
│  │  d_input     │   ═════════════════════                        │
│  │  4MB float32 │                                                │
│  │  [0.17, -2.4,│──┐                                            │
│  │   0.007, ...]│  │  histogramKernelVec4 ──▶ ┌──────────────┐  │
│  └─────────────┘  │  entropyFromHistogram ──▶ │ entropy=5.3  │  │
│        │          │                           └──────┬───────┘  │
│        │          │  CPU: state=16, action=20        │          │
│        │          │  Result: ZSTD + shuffle           │          │
│        │          └──────────────────────────────────┘          │
│        │                                                        │
│        │          STAGE 3: Quantization (SKIPPED by RL)         │
│        │          ═════════════════════════════════════          │
│        │          RL chose no quantization → pass through       │
│        │                                                        │
│        ▼          STAGE 4: Byte Shuffle                         │
│  ┌─────────────┐  ═════════════════════                         │
│  │  d_shuffled  │                                                │
│  │  4MB bytes   │  populateChunkArraysKernel (16 chunks)        │
│  │  [A0 B0 C0..]│  byte_shuffle_vectorized_4byte                │
│  │  [A1 B1 C1..]│                                                │
│  │  [A2 B2 C2..]│  Float bytes regrouped:                       │
│  │  [A3 B3 C3..]│  exponents together, mantissa together        │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼         STAGE 5: nvcomp ZSTD Compression              │
│  ┌─────────────┐  ════════════════════════════════              │
│  │  d_output    │                                                │
│  │  64B header  │  nvcomp splits 4MB into 64KB chunks           │
│  │  + ~600KB    │  LZ77 matching → Huffman encoding             │
│  │  compressed  │  4MB → ~600KB                                  │
│  └──────┬──────┘                                                │
│         │         STAGE 6: Write Header                         │
│         │         ═════════════════════                          │
│         │         magic=GPUC, ver=2, shuffle=4,                 │
│         │         quant=disabled, orig=4MB, comp=600KB          │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │ cudaMemcpyAsync (D2H)    STAGE 7: DMA Transfer
          ▼
┌───────────────┐
│  Host RAM     │    STAGE 8: cudaFree all GPU buffers
│  ~600KB       │
│  [hdr][comp]  │──▶ HDF5 writes to .h5 file (6.7x ratio)
└───────────────┘
```

### Figure 4: Decompression Pipeline (Header-Driven, No Q-Table)

```
┌───────────────┐
│  Host RAM     │    Read from .h5 file by HDF5
│  ~600KB       │
│  [hdr][comp]  │
└───────┬───────┘
        │  STAGE 1: Read Header (CPU)
        │  header.magic = "GPUC" ✓
        │  header.shuffle = 4 → unshuffle needed
        │  header.quant = disabled → skip dequantization
        │
        │  cudaMemcpyAsync (H2D)       STAGE 2: DMA Transfer
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GPU VRAM                                  │
│                                                                  │
│  ┌──────────────┐  STAGE 3: nvcomp ZSTD Decompression           │
│  │ d_compressed  │  ═════════════════════════════════            │
│  │ ~600KB        │  nvcomp auto-detects ZSTD from data header   │
│  └──────┬───────┘  ~600KB → 4MB                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐  STAGE 4: Reverse Byte Shuffle                │
│  │d_decompressed│  ═════════════════════════════                 │
│  │ 4MB shuffled │  byte_unshuffle_kernel (element_size=4)       │
│  └──────┬───────┘  Restore original byte order within floats    │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐  STAGE 5: Dequantization (SKIPPED)            │
│  │ d_unshuffled │  ═════════════════════════════════             │
│  │ 4MB float32  │  header says no quantization was applied      │
│  │ (exact data) │  → data is already in original float32 format │
│  └──────┬───────┘                                                │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │  cudaMemcpyAsync (D2H)       STAGE 6: DMA Transfer
          ▼
┌───────────────┐
│  Host RAM     │    STAGE 7: cudaFree all GPU buffers
│  4MB float32  │
│  (EXACT copy) │──▶ HDF5 returns to application via H5Dread()
└───────────────┘
```

### Figure 5: How Different Data Triggers Different RL Decisions

```
                    ┌────────────────────────┐
                    │  SAME CODE PATH:       │
                    │  GPUCOMPRESS_ALGO_AUTO  │
                    └───────────┬────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ Low Entropy     │ │ Mid Entropy     │ │ High Entropy    │
   │ (e.g. 1.2 bits) │ │ (e.g. 5.3 bits) │ │ (e.g. 7.8 bits) │
   │                 │ │                 │ │                 │
   │ Repetitive data │ │ Structured data │ │ Noisy/random    │
   │ (simulations,   │ │ (sensor arrays, │ │ (encrypted,     │
   │  sparse arrays) │ │  images, CFD)   │ │  high-freq)     │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ State Encoding  │ │ State Encoding  │ │ State Encoding  │
   │ bin=1, lvl=1    │ │ bin=5, lvl=1    │ │ bin=7, lvl=0    │
   │ state = 4       │ │ state = 16      │ │ state = 21      │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ Q-Table Row 4   │ │ Q-Table Row 16  │ │ Q-Table Row 21  │
   │ argmax → a=0    │ │ argmax → a=20   │ │ argmax → a=29   │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ DECISION:       │ │ DECISION:       │ │ DECISION:       │
   │                 │ │                 │ │                 │
   │ ✦ LZ4           │ │ ✦ ZSTD          │ │ ✦ ANS            │
   │ ✦ No quant      │ │ ✦ No quant      │ │ ✦ Quantization   │
   │ ✦ No shuffle    │ │ ✦ 4-byte shuffle│ │ ✦ 4-byte shuffle │
   │                 │ │                 │ │                 │
   │ WHY: Data is    │ │ WHY: Structured │ │ WHY: Near-random │
   │ already very    │ │ data benefits   │ │ data needs       │
   │ compressible,   │ │ from byte       │ │ aggressive       │
   │ LZ4 gives max   │ │ reordering +    │ │ preprocessing +  │
   │ speed at good   │ │ strong ZSTD     │ │ best entropy     │
   │ ratio           │ │ compression     │ │ coder (ANS)      │
   │                 │ │                 │ │                 │
   │ Ratio: ~20x     │ │ Ratio: ~6.7x    │ │ Ratio: ~3.2x     │
   │ Speed: ████████ │ │ Speed: █████    │ │ Speed: ███       │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Figure 6: Q-Table State Space Visualization (30 States)

```
  Error Level →    0 (Aggressive)         1 (Balanced)          2 (Precise/Lossless)
                   err ≥ 0.01             0.001 ≤ err < 0.01   err < 0.001 or 0.0
                   ┌───────────────┐      ┌───────────────┐    ┌───────────────┐
  Entropy   0 bits │ s=0  LZ4      │      │ s=1  LZ4      │    │ s=2  LZ4      │
  (all same)       │ +quant +shuf  │      │ (no preproc)  │    │ (no preproc)  │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            1 bit  │ s=3  LZ4      │      │ s=4  LZ4      │    │ s=5  Snappy   │
                   │ +quant        │      │ (no preproc)  │    │ (no preproc)  │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            2 bits │ s=6  Snappy   │      │ s=7  LZ4      │    │ s=8  Deflate  │
                   │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            3 bits │ s=9  Deflate  │      │ s=10 Deflate  │    │ s=11 ZSTD     │
                   │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            4 bits │ s=12 ZSTD     │      │ s=13 ZSTD     │    │ s=14 ZSTD     │
                   │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
  ▶        5 bits  │ s=15 ZSTD     │      │ s=16 ZSTD ◀◀◀│    │ s=17 ZSTD     │
  example          │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
  data             ├───────────────┤      ├───────────────┤    ├───────────────┤
            6 bits │ s=18 ANS      │      │ s=19 ZSTD     │    │ s=20 ANS      │
                   │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            7 bits │ s=21 ANS      │      │ s=22 ANS      │    │ s=23 ANS      │
                   │ +quant +shuf  │      │ +quant +shuf  │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            8 bits │ s=24 Bitcomp  │      │ s=25 ANS      │    │ s=26 ANS      │
                   │ +quant +shuf  │      │ +shuffle      │    │ +shuffle      │
                   ├───────────────┤      ├───────────────┤    ├───────────────┤
            9 bits │ s=27 Bitcomp  │      │ s=28 ANS      │    │ s=29 ANS      │
  (uniform random) │ +quant +shuf  │      │ +shuffle      │    │ (no preproc)  │
                   └───────────────┘      └───────────────┘    └───────────────┘

  Key patterns learned by RL:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ ● Low entropy  → fast algorithms (LZ4, Snappy) -- data compresses easy │
  │ ● Mid entropy  → strong algorithms (ZSTD, Deflate) + byte shuffle     │
  │ ● High entropy → entropy coders (ANS) + aggressive preprocessing      │
  │ ● Aggressive error → quantization enabled (acceptable quality loss)    │
  │ ● Precise/lossless → quantization disabled (preserve fidelity)         │
  └──────────────────────────────────────────────────────────────────────────┘
```

### Figure 7: Action Space Encoding (32 Actions)

```
  5-bit action ID decomposition:

  ┌─────────┬─────────┬─────────────────┐
  │  Bit 4  │  Bit 3  │  Bits 2-1-0     │
  │ shuffle │  quant  │  algorithm      │
  ├─────────┼─────────┼─────────────────┤
  │  0=off  │  0=off  │  0=LZ4          │
  │  1=4B   │  1=on   │  1=Snappy       │
  │         │         │  2=Deflate      │
  │         │         │  3=Gdeflate     │
  │         │         │  4=ZSTD         │
  │         │         │  5=ANS          │
  │         │         │  6=Cascaded     │
  │         │         │  7=Bitcomp      │
  └─────────┴─────────┴─────────────────┘

  All 32 actions:

  action │ algo     │ quant │ shuffle │ binary
  ───────┼──────────┼───────┼─────────┼────────
    0    │ LZ4      │  off  │  off    │ 00000
    1    │ Snappy   │  off  │  off    │ 00001
    2    │ Deflate  │  off  │  off    │ 00010
    3    │ Gdeflate │  off  │  off    │ 00011
    4    │ ZSTD     │  off  │  off    │ 00100
    5    │ ANS      │  off  │  off    │ 00101
    6    │ Cascaded │  off  │  off    │ 00110
    7    │ Bitcomp  │  off  │  off    │ 00111
  ───────┼──────────┼───────┼─────────┼────────
    8    │ LZ4      │  ON   │  off    │ 01000
    9    │ Snappy   │  ON   │  off    │ 01001
   ...   │ ...      │  ON   │  off    │ ...
   15    │ Bitcomp  │  ON   │  off    │ 01111
  ───────┼──────────┼───────┼─────────┼────────
   16    │ LZ4      │  off  │  ON     │ 10000
   17    │ Snappy   │  off  │  ON     │ 10001
   18    │ Deflate  │  off  │  ON     │ 10010
   19    │ Gdeflate │  off  │  ON     │ 10011
  ►20◄   │ ZSTD     │  off  │  ON     │ 10100  ◄── example decision
   21    │ ANS      │  off  │  ON     │ 10101
   22    │ Cascaded │  off  │  ON     │ 10110
   23    │ Bitcomp  │  off  │  ON     │ 10111
  ───────┼──────────┼───────┼─────────┼────────
   24    │ LZ4      │  ON   │  ON     │ 11000
   25    │ Snappy   │  ON   │  ON     │ 11001
   26    │ Deflate  │  ON   │  ON     │ 11010
   27    │ Gdeflate │  ON   │  ON     │ 11011
   28    │ ZSTD     │  ON   │  ON     │ 11100
   29    │ ANS      │  ON   │  ON     │ 11101
   30    │ Cascaded │  ON   │  ON     │ 11110
   31    │ Bitcomp  │  ON   │  ON     │ 11111
```

---

## Step 5: Offline RL Training System (Python `rl/` package)

### Goal

Learn which compression configuration works best for each type of data, producing a Q-Table that the C library loads at runtime.

### Why Offline Training?

```
OFFLINE TRAINING (Python)              ONLINE INFERENCE (C++/CUDA)
===========================            ============================

- Runs ONCE before deployment          - Runs EVERY compression call
- Takes hours (acceptable)             - Takes microseconds (required)
- Easy experimentation                 - Zero overhead
- Hyperparameter tuning                - Fixed logic
- No speed requirement                 - Speed critical

Output: qtable.bin (3.8 KB)  -------> Loaded to GPU constant memory
```

### Architecture (6 Modules)

| Module | Role |
|--------|------|
| `config.py` | Constants: 30 states, 32 actions, hyperparameters (alpha=0.1, gamma=0, epsilon: 1.0 to 0.01) |
| `qtable.py` | Core Q-Table: 30x32 numpy array, Q-learning update rule, save/load JSON+binary |
| `policy.py` | Epsilon-greedy with exponential decay + pure greedy for inference |
| `executor.py` | Runs `gpu_compress` binary via subprocess, collects ratio/throughput/PSNR |
| `reward.py` | Weighted reward: `w1*ratio + w2*throughput + w3*PSNR` with 6 presets |
| `trainer.py` | Main loop: epoch -> shuffle files -> episode per file -> update Q-table |

### Training Loop Per Episode

```
1. Load binary file
       |
       v
2. Calculate byte-level Shannon entropy (NumPy or GPU via ctypes)
       |
       v
3. Encode state: state = entropy_bin * 3 + error_level
       |
       v
4. Epsilon-greedy: random action with probability epsilon, else argmax Q(s,.)
       |
       v
5. Decode action --> (algorithm_name, quantization?, shuffle_size)
       |
       v
6. Run gpu_compress subprocess with those settings
       |
       v
7. Measure: compression ratio, throughput MB/s, PSNR (if lossy)
       |
       v
8. Compute reward = weighted sum (configurable presets)
       |
       v
9. Q-learning update: Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
```

### Q-Learning Update Rule

```python
Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))
```

No discount factor (gamma=0) because this is a single-step decision: there is no "next state". Each data chunk is compressed independently.

### Reward Presets

| Preset | Ratio Weight | Speed Weight | Quality Weight | Use Case |
|--------|-------------|-------------|----------------|----------|
| `balanced` | 0.4 | 0.3 | 0.3 | General purpose |
| `max_ratio` | 0.8 | 0.1 | 0.1 | Minimize storage |
| `max_speed` | 0.1 | 0.8 | 0.1 | Real-time streaming |
| `max_quality` | 0.1 | 0.1 | 0.8 | Scientific accuracy |
| `storage` | 0.6 | 0.2 | 0.2 | Archival storage |
| `streaming` | 0.3 | 0.5 | 0.2 | Network transfer |

### Exploration Policy

Epsilon-greedy with exponential decay:

```
epsilon_start = 1.0    (100% exploration at beginning)
epsilon_end   = 0.01   (1% exploration at convergence)
epsilon_decay = 0.995  (multiply per epoch)
```

This ensures broad exploration early in training, converging to exploitation of learned values.

### Bootstrap: Heuristic Initial Q-Table

`generate_initial_qtable.py` creates a heuristic Q-Table based on known algorithm characteristics so the system works before any training:

- **Low entropy data**: LZ4, Snappy score high (fast, good ratios on compressible data)
- **High entropy data**: ZSTD, ANS score high (strong algorithms for hard-to-compress data)
- **Mid-range entropy with structure**: Shuffle preprocessing gets a bonus
- **Aggressive error level**: Quantization gets a bonus
- **Precise/lossless error level**: Quantization gets a penalty

### Export Pipeline

```
Training
    |
    v
qtable.json (human-readable, for debugging)
    |
    v
export_qtable.py
    |
    v
qtable.bin (binary, 3.8 KB)
    Header: magic "QTAB" (0x51544142), version=1, n_states=30, n_actions=32
    Data:   960 x float32
    |
    v
gpucompress_init("qtable.bin")
    |
    v
cudaMemcpyToSymbol --> __constant__ d_qtable[960]
```

### How to Train

```bash
# Step 1: Build the project
mkdir -p build && cd build && cmake .. && make

# Step 2: Generate training data
python syntheticGeneration/generator.py batch \
    -c syntheticGeneration/datasets/config.yaml \
    -o datasets/training/

# Step 3: Run training
./scripts/run_rl_training.sh \
    --data-dir datasets/training \
    --epochs 100 \
    --preset balanced

# Step 4: Use trained model
# The trained qtable.bin is loaded at gpucompress_init()
```

---

## Step 6: Build System Overhaul (`CMakeLists.txt`)

### Changes Made

- Added **C language** support (for the HDF5 plugin, which is pure C)
- Default build type changed from **Debug to Release** (`-O3`)
- Deduplicated sources into reusable CMake variables:
  - `PREPROCESSING_SOURCES` = preprocessing/byte_shuffle + preprocessing/quantization kernels
  - `LIB_SOURCES` = API + entropy kernel + Q-Table GPU
  - `FACTORY_SOURCES` = core/compression_factory

### New Build Targets

| Target | Type | Description |
|--------|------|-------------|
| `libgpucompress.so` | Shared library | Core C API with `-fPIC`, version 1.0.0 |
| `libH5Zgpucompress.so` | Shared library | HDF5 filter plugin (only if HDF5 found) |
| `gpu_compress` | Executable | CLI compression tool (existing, updated) |
| `gpu_decompress` | Executable | CLI decompression tool (existing, updated) |
| `test_quantization` | Executable | Quantization test suite (existing) |

### Install Rules

```cmake
install(TARGETS gpucompress      LIBRARY DESTINATION lib)
install(DIRECTORY include/       DESTINATION include FILES_MATCHING PATTERN "*.h")
install(TARGETS gpu_compress gpu_decompress RUNTIME DESTINATION bin)
install(TARGETS H5Zgpucompress   LIBRARY DESTINATION lib/hdf5/plugin)  # if HDF5 found
```

---

## End-to-End Flow

### Training Phase (Offline, Python)

```
Synthetic data files (.bin)
       |
       v
run_rl_training.sh --> trainer.py
       |
       |  For each file:
       |    calculate entropy --> pick action --> compress --> measure --> learn
       |
       v
qtable.json --> export --> qtable.bin (3.8 KB)
```

### Runtime Phase (C/C++/HDF5)

```
Application
       |
       +-- Direct C API:
       |     gpucompress_init("qtable.bin")
       |     gpucompress_compress(data, ..., ALGO_AUTO, ...)
       |
       +-- HDF5 Plugin:
             H5Pset_gpucompress(dcpl, ALGO_AUTO, 0, 4, 0.0)
             H5Dwrite(dataset, data)  <-- filter auto-calls gpucompress

       |
       v
[Inside gpucompress_compress with ALGO_AUTO]:
  1. GPU entropy calculation (histogram --> Shannon entropy)
  2. Encode state = f(entropy, error_bound)
  3. Q-Table lookup --> best action (from constant memory)
  4. Decode action --> algorithm + preprocessing flags
  5. Apply quantization/shuffle --> nvcomp compress --> write 64-byte header
```

---

## Complete File Reference

Every source file in the project, what it does, and how it connects to the rest.

### `include/` - Public Headers (Installed with the Library)

| File | Lines | What It Does |
|------|-------|-------------|
| `gpucompress.h` | 402 | **The public C API.** Defines all types (`gpucompress_algorithm_t`, `gpucompress_config_t`, `gpucompress_stats_t`, `gpucompress_error_t`), all functions (`gpucompress_init`, `gpucompress_compress`, `gpucompress_decompress`, etc.), and constants. Wrapped in `extern "C"` so it is callable from C, Python ctypes, Fortran, or any FFI. This is the only header an external user needs for the core library. |
| `gpucompress_hdf5.h` | 184 | **HDF5 filter public interface.** Defines `H5Z_FILTER_GPUCOMPRESS` (filter ID 305), the `cd_values` layout, and convenience functions `H5Pset_gpucompress` / `H5Pget_gpucompress`. Includes `gpucompress.h` so users get both APIs with one include. Forward-declares HDF5 types when HDF5 headers are not available. |

### `src/` - Core Engine (Existing Files)

These files existed before this commit and form the foundation the new library wraps.

| File | Lines | What It Does |
|------|-------|-------------|
| `cli/compress.cpp` | ~350 | **CLI compression executable.** The `main()` for `gpu_compress`. Reads a file using GPUDirect Storage (cuFile) directly into GPU memory, applies optional preprocessing (quantization, shuffle), compresses via nvcomp through `CompressionFactory`, prepends the 64-byte `CompressionHeader`, and writes the result back via GDS. Accepts command-line arguments for algorithm selection, shuffle size, error bound, and quantization type. |
| `cli/decompress.cpp` | ~300 | **CLI decompression executable.** The `main()` for `gpu_decompress`. Reads compressed data via GDS, reads the 64-byte header to auto-detect algorithm and preprocessing, decompresses via nvcomp, reverses shuffle and dequantization if needed, writes restored data via GDS. Fully automatic - no arguments needed beyond input/output paths. |
| `core/compression_factory.hpp` | 54 | **Factory pattern header.** Declares `CompressionAlgorithm` enum (LZ4 through Bitcomp), `createCompressionManager()`, and `createDecompressionManager()`. The decompression factory auto-detects the algorithm from the nvcomp header in the compressed data. |
| `core/compression_factory.cpp` | ~130 | **Factory pattern implementation.** Includes all 8 nvcomp algorithm headers (`nvcomp/lz4.hpp`, `nvcomp/zstd.hpp`, etc.) and implements the switch-case factory. Each algorithm gets a default configuration (e.g., LZ4 uses type `nvcomp_native`, Zstd uses default chunk size). The decompression manager is created via `nvcomp::create_manager()` which inspects the compressed buffer header. |
| `core/compression_header.h` | 244 | **Self-describing 64-byte binary header.** `struct CompressionHeader` with `__attribute__((packed))` for exact layout. Contains: magic number `"GPUC"` (0x43555047), version, shuffle element size, quantization flags (bitfield: type + precision + enabled), original/compressed sizes, and quantization metadata (error bound, scale, data min/max). Includes helper functions to write/read header to/from GPU memory, get compressed data pointer after header. `static_assert` ensures exactly 64 bytes. |
| `preprocessing/byte_shuffle.cuh` | 332 | **Byte shuffle header.** Declares 7 kernel variants (baseline, shared-memory, vectorized 4-byte, vectorized 8-byte, template-specialized) plus the high-level API (`byte_shuffle_simple` / `byte_unshuffle_simple`). Defines `ShuffleKernelType` enum with `AUTO` for adaptive kernel selection. The shuffle algorithm reorders bytes by position across elements: `[A0 A1 A2 A3][B0 B1 B2 B3]` becomes `[A0 B0][A1 B1][A2 B2][A3 B3]`, grouping similar bytes for better compression. |
| `preprocessing/byte_shuffle_kernels.cu` | ~600 | **Byte shuffle CUDA kernels.** Implements all kernel variants declared in `byte_shuffle.cuh`. Architecture: one warp per chunk, threads handle different byte positions. Strategy A (outer loop parallelization) gives coalesced output writes with strided input reads. Shared-memory variant stages data for coalesced reads too (30-50% faster for large elements). Vectorized variants use `float4`/`double2` for 128-bit transactions. The `AUTO` selector picks the best kernel based on element size and alignment. Includes `byte_shuffle_simple` which handles allocation, chunking (via `createDeviceChunkArrays`), kernel launch, and cleanup. |
| `preprocessing/quantization.cuh` | 386 | **Quantization header.** Defines `QuantizationType` (NONE, LINEAR), `QuantizationPrecision` (AUTO, INT8, INT16, INT32), `QuantizationConfig` struct, and `QuantizationResult` struct (carries all metadata needed for dequantization: scale, min, max, precision, error bound). Declares `quantize_simple()` / `dequantize_simple()` high-level API and `verify_error_bound()` for validation. Includes precision selection logic: computes required bits from `data_range / (2 * error_bound)` to choose int8/int16/int32. Also provides bitfield pack/unpack helpers for storing quantization flags in the compression header. |
| `preprocessing/quantization_kernels.cu` | ~450 | **Quantization CUDA kernels.** Implements `quantize_simple()`: first computes data range via CUB block reduction (`compute_min_max_kernel`), determines precision, then applies `round(value / (2 * error_bound))` via the quantization kernel. Supports float32 and float64 input with int8/int16/int32 output. `dequantize_simple()` reverses: `value = quantized * (2 * error_bound)`. The `verify_error_bound()` function computes max absolute error across all elements to confirm the guarantee holds. |
| `core/util.h` | 230 | **GPU chunk array utilities.** Defines `DeviceChunkArrays` (RAII wrapper for device-allocated pointer arrays) and `createDeviceChunkArrays()`. The key optimization: a GPU kernel (`populateChunkArraysKernel`) computes chunk offsets and sizes directly on device, avoiding host-side array construction and `cudaMemcpy`. Each thread computes metadata for one chunk. Used by the byte shuffle system to set up per-chunk pointers with zero CPU-GPU transfer overhead. Move-only semantics prevent accidental double-free of device pointers. |

### `src/api/`, `src/nn/`, `src/stats/` - Shared Library Source (NEW in This Commit)

These files implement `libgpucompress.so` and are the bridge between the public C API and the existing engine.

| File | Lines | What It Does |
|------|-------|-------------|
| `gpucompress_api.cpp` | 835 | **C API implementation.** The core of `libgpucompress.so`. Compiled as CUDA (via `set_source_files_properties`). Implements every function from `gpucompress.h`. Manages global state with `std::atomic<bool>` for initialization flag, `std::atomic<int>` for reference counting, and `std::mutex` for thread-safe init/cleanup. The `gpucompress_compress()` function orchestrates the full pipeline: allocate GPU memory, copy input, optionally run Q-Table auto-selection (entropy -> state -> action -> decode), apply quantization, apply shuffle, create nvcomp manager via `CompressionFactory`, compress, build `CompressionHeader`, copy result to host, free GPU memory. The `gpucompress_decompress()` function reads the header, auto-detects everything, and reverses the pipeline. Also implements `gpucompress_calculate_entropy()` (host-to-GPU wrapper) and `gpucompress_recommend_config()` (Q-Table lookup without compressing). GPU-to-GPU API (`compress_gpu`/`decompress_gpu`) declared but stubbed as TODO. |
| `internal.hpp` | 239 | **Internal C++ utilities.** Not part of the public API. Defines: `gpucompress` namespace constants (chunk sizes, alignment, Q-Table dimensions), algorithm mapping functions (`toInternalAlgorithm` / `fromInternalAlgorithm` between C enum and C++ enum), Q-Table state/action encoding/decoding (`encodeState`, `decodeAction`, `errorBoundToLevel`, `QTableAction` struct), CUDA helpers (`alignSize`, `checkCuda`), and `LibraryState` singleton class (thread-safe initialization, Q-Table access, CUDA resource management). |
| `entropy_kernel.cu` | 339 | **GPU Shannon entropy calculation.** Two histogram kernels: `histogramKernel` (byte-at-a-time) and `histogramKernelVec4` (4 bytes via `uint32_t` reads with bit-shift extraction). Both use shared-memory per-block histograms with `atomicAdd`, merging to global histogram. `entropyFromHistogramKernel` computes `H = -sum(p_i * log2(p_i))` using parallel block reduction. `calculateEntropyGPU()` allocates, launches, and returns the result. `calculateEntropyGPUWithBuffers()` skips allocation for repeated calls. C wrapper `gpucompress_entropy_gpu_impl()` bridges to the `extern "C"` API. |
| `qtable_gpu.cu` | 457 | **GPU Q-Table storage and inference.** `__constant__ float d_qtable[960]` stores the 30x32 Q-Table in constant memory (~4KB, broadcast-cached). Supports loading from binary (magic `"QTAB"` / 0x51544142 header) or JSON (simple parser scanning for `"q_values"` array). `loadQTableToGPU()` copies to constant memory via `cudaMemcpyToSymbol` and keeps a host mirror. `qtableArgmaxKernel` runs a 32-thread parallel reduction to find the best action. `getBestActionCPU()` does a linear scan on the host copy (used in practice since single-query latency favors CPU). `initializeDefaultQTable()` fills with zeros for when no trained model exists. C wrappers `gpucompress_qtable_load_impl()`, `gpucompress_qtable_get_best_action_impl()`, etc. bridge to the `extern "C"` API. |

### `src/hdf5/` - HDF5 Filter Plugin (NEW in This Commit)

| File | Lines | What It Does |
|------|-------|-------------|
| `H5Zgpucompress.c` | 460 | **HDF5 filter implementation.** Pure C for maximum HDF5 compatibility. Defines the `H5Z_class2_t` filter class (ID 305) with three callbacks: `can_apply` (checks CUDA availability), `set_local` (no-op, placeholder for future auto-shuffle-size detection), and `H5Z_filter_gpucompress` (the main function). On write: parses `cd_values[5]` for config, calls `gpucompress_compress()`, replaces HDF5 buffer. On read: calls `gpucompress_get_original_size()` from header, allocates, calls `gpucompress_decompress()`, replaces buffer. Uses `H5allocate_memory`/`H5free_memory` for HDF5-managed buffers. Double-to-uint32 packing via union for `cd_values[3:4]` error bound. Smart fallback: returns original data if compression makes it larger. Plugin exports: `H5PLget_plugin_type()` returns `H5PL_TYPE_FILTER`, `H5PLget_plugin_info()` returns the class struct. `H5Pset_gpucompress()` is the convenience wrapper around `H5Pset_filter()`. Tracks initialization state to auto-init the library on first use. |
| `H5Zgpucompress.h` | 56 | **Internal filter header.** Used when building the plugin directly. Declares `H5Z_gpucompress_register()`, `H5Pset_gpucompress()`, `H5Pget_gpucompress()`, and the filter info getter. Includes `<hdf5.h>` and `gpucompress.h`. |

### `rl/` - Offline RL Training System (Python, NEW in This Commit)

| File | Lines | What It Does |
|------|-------|-------------|
| `__init__.py` | 32 | **Package exports.** Makes `rl/` a Python package. Exports `QTable`, `EpsilonGreedyPolicy`, `GreedyPolicy`, `compute_reward`, `CompressionExecutor`, `QTableTrainer`, and configuration constants. |
| `config.py` | 91 | **All constants and hyperparameters in one place.** State space: `NUM_ENTROPY_BINS=10`, `NUM_ERROR_LEVELS=3`, `NUM_STATES=30`. Action space: `NUM_ALGORITHMS=8`, `NUM_ACTIONS=32`. Hyperparameters: `LEARNING_RATE=0.1`, `DISCOUNT_FACTOR=0.0` (single-step), epsilon schedule (1.0 -> 0.01, decay 0.995). Training settings: 100 epochs, checkpoint every 10. Reward presets dictionary (6 presets with ratio/throughput/psnr weights). Normalization constants for reward computation. |
| `qtable.py` | 242 | **Core Q-Table data structure.** 30x32 numpy array (`q_values`) + visit count tracking. `encode_state()`: discretizes entropy to bin 0-9, maps error bound to level 0-2, combines as `bin * 3 + level`. `decode_action()`: extracts algorithm (mod 8), quantization (div 8 mod 2), shuffle (div 16 mod 2). `update()`: single-step Q-learning `Q(s,a) += alpha * (reward - Q(s,a))`. `save()`/`load()`: JSON format with version, dimensions, q_values, visit_counts. `export_binary()`: writes `"QTAB"` magic + version + dimensions + flat float32 array (matches `qtable_gpu.cu` loader format exactly). `print_best_actions()`: pretty-prints the policy table. `get_state_coverage()`: training diagnostics. |
| `policy.py` | 106 | **Exploration strategies.** `EpsilonGreedyPolicy`: with probability epsilon selects random action (explore), otherwise selects argmax (exploit). Epsilon decays multiplicatively per epoch (`epsilon *= 0.995`). `GreedyPolicy`: always exploits (epsilon=0), used for inference after training. Both return `(action, was_exploration)` tuple for tracking statistics. |
| `reward.py` | 86 | **Reward function.** `compute_reward()` takes ratio, throughput (MB/s), and optional PSNR (dB). Normalizes each to [0, 1] against maximums (ratio/10, throughput/5000, psnr/100). Applies preset weights as weighted sum. Lossless data gets perfect quality score (psnr_norm=1.0). `get_preset_description()` returns human-readable strings. |
| `executor.py` | 261 | **Compression execution bridge.** Two modes: (1) Subprocess mode: builds a command line for the `gpu_compress` binary, runs it with `subprocess.run()`, parses stdout for ratio/PSNR via regex, measures elapsed time for throughput. (2) C API mode: loads `libgpucompress.so` via `ctypes.CDLL`, calls `gpucompress_init()` and `gpucompress_calculate_entropy()` directly. `calculate_entropy()` has a NumPy fallback: builds 256-bin histogram, computes `-sum(p * log2(p))`. `execute_action()` takes a decoded action dict and dispatches to `execute()`. Uses temp files for compressed output, cleaned up in `finally` block. |
| `trainer.py` | 332 | **Main training loop.** `QTableTrainer` class orchestrates everything. `__init__`: collects `.bin`/`.raw`/`.dat` files from data directory, initializes QTable, EpsilonGreedyPolicy, CompressionExecutor. `train()`: for each epoch, shuffles files, runs `_train_episode()` per file, decays epsilon, saves checkpoints. `_train_episode()`: loads file as float32 numpy, calculates entropy, encodes state, selects action via policy, executes compression, computes reward, updates Q-table. `_save_final()`: writes both JSON and binary formats. `_print_summary()`: state coverage, exploration/exploitation ratio, best actions table. CLI via `argparse` with `--data-dir`, `--epochs`, `--preset`, `--error-bound`, `--gpu-compress`, `--use-c-api` flags. |
| `export_qtable.py` | 187 | **JSON-to-binary converter.** `export_json_to_binary()`: reads JSON, validates shape is 30x32, writes binary with `"QTAB"` header, verifies file size matches expected (16 + 960*4 = 3856 bytes). `verify_binary()`: reads back and validates magic, version, dimensions, data size. Also provides `export_numpy_to_binary()` for direct numpy array export. CLI: `python -m rl.export_qtable input.json output.bin --verify`. |
| `generate_initial_qtable.py` | 207 | **Heuristic bootstrap.** Creates a Q-Table with reasonable starting values so the system works before training. Each algorithm has characteristic scores: LZ4 (fast, good for low entropy: 0.9/0.3/1.0), Zstd (strong ratio, slow: 0.6/0.8/0.4), ANS (best entropy coder: 0.5/0.85/0.3). Scores interpolate between low-entropy and high-entropy characteristics. Quantization gets +0.15 bonus for aggressive error level, -0.3 penalty for precise/lossless. Shuffle gets +0.1 bonus for mid-range entropy (structured data). Outputs both JSON and binary to `rl/models/`. |
| `models/qtable.json` | 1031 | **Pre-generated heuristic Q-Table.** The 30x32 table of float values produced by `generate_initial_qtable.py`. Ships with the repository so `gpucompress_init(NULL)` has a working default policy without requiring training. |

### `tests/` - Test Suite

| File | Lines | What It Does |
|------|-------|-------------|
| `quantization/test_quantization_suite.cu` | ~400 | **Quantization correctness tests.** Validates error bounds are respected across float32/float64, various data distributions, and int8/int16/int32 precision levels. Tests round-trip: quantize -> compress -> decompress -> dequantize -> verify max error <= bound. |

### `scripts/` - Tooling and Utilities

| File | What It Does |
|------|-------------|
| `run_rl_training.sh` | Shell wrapper for `python3 -m rl.trainer`. Parses `--data-dir`, `--epochs`, `--preset`, `--error-bound`, validates inputs, prints config, runs training. |
| `run_simple_tests.sh` | End-to-end test runner: generates data, compresses with each algorithm, decompresses, verifies output matches original. |
| `test_quantization_errors.sh` | Runs quantization tests across different error bounds and verifies correctness. |
| `synthetic_data_generator.cc` | C++ program that generates synthetic binary data files with configurable entropy levels, data sizes, and patterns (smooth, turbulent, periodic, noisy). |
| `install_dependencies.sh` | Downloads and installs nvcomp library dependencies to `/tmp/include` and `/tmp/lib`. |
| `setup_env.sh` | Sets `LD_LIBRARY_PATH` and other environment variables for running the tools. |
| `visualize_results.py` | Python script using matplotlib to plot compression ratios, throughput, and other metrics from test results. |

### `CMakeLists.txt` - Build System

| Section | What It Does |
|---------|-------------|
| Project config | Enables C, C++, CUDA languages. Sets C++14/CUDA14 standard. Release build by default (`-O3`). |
| Source variables | `PREPROCESSING_SOURCES` (preprocessing/ shuffle + quantization kernels), `LIB_SOURCES` (API + entropy + qtable), `FACTORY_SOURCES` (core/compression_factory). Reused across targets to avoid duplication. |
| `libgpucompress.so` | Shared library target. Links `LIB_SOURCES` + `PREPROCESSING_SOURCES` + `FACTORY_SOURCES`. Links against nvcomp and CUDA runtime. `-fPIC`, version 1.0.0. Public include directory at `include/`. |
| `libH5Zgpucompress.so` | HDF5 filter plugin. Only built if `find_package(HDF5)` succeeds. Links against `libgpucompress.so` and HDF5 C library. Defines `GPUCOMPRESS_BUILD_HDF5_PLUGIN` for plugin exports. |
| `gpu_compress` | CLI compression executable. Links `cli/compress.cpp` + preprocessing + factory against nvcomp, CUDA, cuFile (GDS). |
| `gpu_decompress` | CLI decompression executable. Same structure as compress. |
| `test_quantization` | Test executable. Links test suite against nvcomp and CUDA. |
| Install rules | Library to `lib/`, headers to `include/`, executables to `bin/`, HDF5 plugin to `lib/hdf5/plugin/`. |

---

## How the Files Connect: Dependency Graph

```
                        gpucompress.h (PUBLIC API)
                              |
                    gpucompress_api.cpp
                     /        |         \
                    /         |          \
   entropy_kernel.cu  qtable_gpu.cu  internal.hpp
                    \         |          /
                     \        |         /
                      core/compression_factory.hpp/.cpp
                       /           \
                      /             \
       preprocessing/byte_shuffle.cuh         preprocessing/quantization.cuh
       preprocessing/byte_shuffle_kernels.cu  preprocessing/quantization_kernels.cu
              |                                      |
           core/util.h                    core/compression_header.h
        (chunk arrays)                     (64-byte packed header)

   cli/compress.cpp ─────┐
   cli/decompress.cpp ───┤─── All use core/ + preprocessing/ + header
                         │
   H5Zgpucompress.c ─────┘─── Calls gpucompress.h C API (not internal components)

   rl/trainer.py ──> rl/executor.py ──> gpu_compress (subprocess)
        |                  |
   rl/qtable.py      rl/policy.py
        |
   rl/reward.py      rl/config.py (constants shared by all rl/ modules)
        |
   rl/export_qtable.py ──> qtable.bin ──> qtable_gpu.cu (loaded at runtime)
```

---

## Summary Table

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| `libgpucompress.so` | C API + CUDA | Core compression library |
| `libH5Zgpucompress.so` | HDF5 filter (C) | Transparent HDF5 integration |
| GPU Entropy Kernel | CUDA (histogram + reduction) | Real-time entropy calculation |
| GPU Q-Table | CUDA constant memory | Fast algorithm selection |
| CompressionFactory | C++ factory pattern | nvcomp algorithm instantiation |
| Byte Shuffle | CUDA (7 kernel variants) | Preprocessing for better ratios |
| Quantization | CUDA (CUB reductions) | Error-bounded lossy preprocessing |
| Compression Header | 64-byte packed struct | Self-describing compressed format |
| Python RL Training | Offline Q-learning | Learn optimal configurations |
| Q-Table Binary | 3.8 KB file | Portable trained model |
| CLI Tools | C++/CUDA + GDS | Direct file compression/decompression |

### Key Design Decisions

1. **Offline training, online inference** - Training in Python for flexibility; inference in CUDA for speed
2. **Q-Table in GPU constant memory** - Microsecond lookup time, no allocation overhead, broadcast to all threads
3. **Host memory interface for HDF5** - Matches HDF5 filter API requirements; GPU transfer handled internally
4. **Self-describing compressed format** - 64-byte packed header enables fully automatic decompression
5. **Reference-counted initialization** - Thread-safe, multiple callers can init/cleanup independently
6. **Smart compression fallback** - HDF5 filter returns original data if compression makes it larger
7. **Layered architecture** - CLI tools and HDF5 plugin both build on the same `libgpucompress.so`, avoiding code duplication
8. **Adaptive kernel selection** - Byte shuffle auto-picks the best CUDA kernel variant based on element size and alignment
