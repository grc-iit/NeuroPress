# Neural Network Algorithm Selection: Complete Implementation Tutorial

This document walks through the entire NN-based automatic algorithm selection system in GPUCompress, from data collection through training, GPU inference, and active learning.

---

## Table of Contents

1. [System Overview](#1-system-overview)
   - [1.1 Inference Pipeline Architecture](#11-complete-architecture-inference-pipeline)
   - [1.2 Training Pipeline Architecture](#12-complete-architecture-training-pipeline)
   - [1.3 Key Data Structures](#13-key-data-structures)
2. [The Problem: 32 Compression Configs](#2-the-problem-32-compression-configs)
3. [Data Pipeline (Python)](#3-data-pipeline-python)
4. [Model Architecture (Python)](#4-model-architecture-python)
5. [Training (Python)](#5-training-python)
6. [Weight Export: PyTorch to Binary (.nnwt)](#6-weight-export-pytorch-to-binary-nnwt)
7. [GPU Inference Kernel (CUDA)](#7-gpu-inference-kernel-cuda)
8. [Stats Pipeline Integration (CUDA)](#8-stats-pipeline-integration-cuda)
   - [8.1 Stats-Only Pipeline](#81-stats-only-pipeline-for-training-data-generation)
   - [8.2 Public C API: gpucompress_compute_stats()](#82-public-c-api-gpucompress_compute_stats)
9. [API Integration: gpucompress_compress()](#9-api-integration-gpucompress_compress)
10. [Active Learning System](#10-active-learning-system)
11. [Retraining with Experience Data](#11-retraining-with-experience-data)
12. [File Map](#12-file-map)
13. [End-to-End Walkthrough](#13-end-to-end-walkthrough)

---

## 1. System Overview

When a user calls `gpucompress_compress()` with `ALGO_AUTO`, the library must choose the best combination of:

- **Algorithm** (8 options): LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp
- **Quantization** (2 options): none, linear
- **Shuffle** (2 options): none, 4-byte

That's **32 possible configurations**. The NN predicts the performance of all 32 in a single GPU kernel launch and picks the best one.

```
User data ──> GPU Stats Pipeline ──> NN Inference Kernel ──> Best config ──> Compress
                  (entropy,             (32 threads,           (action
                   MAD,                  parallel               0-31)
                   derivative)           forward pass)
```

The system has two operational modes:
- **Static mode** (default): NN trained once on benchmark data, frozen at deploy
- **Active learning mode** (opt-in): collects experience during use, enables manual retraining

### 1.1 Complete Architecture: Inference Pipeline

The diagram below traces every function call when `gpucompress_compress()` is called with `ALGO_AUTO`.

```
gpucompress_compress(data, size, output, &out_size, &config, &stats)
│                                                [src/lib/gpucompress_api.cpp]
│
├── 1. cudaMemcpy(d_input ← data)                      Copy input to GPU
│
├── 2. runAutoStatsNNPipeline(d_input, size, ...)       [src/lib/stats_kernel.cu]
│      │
│      ├── 2a. statsPass1Kernel<<<N, 256>>>             [stats_kernel.cu]
│      │        Per-block: sum, min, max, |x[i]-x[i-1]|
│      │        Atomic reduction to AutoStatsGPU struct
│      │
│      ├── 2b. launchEntropyKernelsAsync()              [entropy_kernel.cu]
│      │        256-bin byte histogram → Shannon entropy
│      │
│      ├── 2c. madPass2Kernel<<<N, 256>>>               [stats_kernel.cu]
│      │        mean = sum/n → MAD = sum(|x[i] - mean|)
│      │
│      ├── 2d. finalizeStatsOnlyKernel<<<1, 1>>>        [stats_kernel.cu]
│      │        mad_norm = (mad_sum/n) / range
│      │        deriv_norm = (abs_diff_sum/(n-1)) / range
│      │
│      ├── 2e. cudaMemcpy(host ← entropy, mad, deriv)  Copy stats to host
│      │
│      └── 2f. runNNInference(entropy, mad, deriv, ...) [src/lib/nn_gpu.cu]
│              │
│              └── nnInferenceKernel<<<1, 32>>>
│                  │
│                  │  Each of 32 threads (one per config):
│                  │
│                  ├── Build 15-feature input vector
│                  │   [0-7]  one-hot algorithm
│                  │   [8]    quantization binary
│                  │   [9]    shuffle binary
│                  │   [10]   log10(error_bound)
│                  │   [11]   log2(data_size)
│                  │   [12-14] entropy, mad, deriv
│                  │
│                  ├── Standardize: (x - x_means) / x_stds
│                  │
│                  ├── Layer 1: Linear(15→128) + ReLU
│                  ├── Layer 2: Linear(128→128) + ReLU
│                  ├── Layer 3: Linear(128→4)
│                  │
│                  ├── De-normalize: y * y_stds + y_means
│                  │   → comp_time, decomp_time, ratio, psnr
│                  │
│                  └── Parallel reduction: pick best action (0-31)
│
├── 3. decodeAction(action)                             [src/lib/internal.hpp]
│      algo = action % 8
│      quant = (action / 8) % 2
│      shuffle = (action / 16) % 2
│
├── 4. Preprocessing (if needed)
│      ├── quantize_simple()                            [preprocessing/quantization.cuh]
│      └── byte_shuffle_simple()                        [preprocessing/byte_shuffle.cuh]
│
├── 5. createCompressionManager() → compress()          [core/compression_factory.hpp]
│      nvcomp compression with chosen algorithm
│
├── 6. Build CompressionHeader + cudaMemcpy(host ← output)
│
├── 7. [If active learning enabled]
│      ├── experience_buffer_append(sample)             [src/lib/experience_buffer.cpp]
│      ├── Check |predicted - actual| / actual > threshold?
│      └── If yes: explore top-K alternatives
│          └── For each: preprocess → compress → record → keep best
│
└── 8. Return GPUCOMPRESS_SUCCESS
```

### 1.2 Complete Architecture: Training Pipeline

```
python neural_net/train.py --data-dir data/ --lib-path build/libgpucompress.so
│
├── 1. load_and_prepare_from_binary(data_dir)           [neural_net/binary_data.py]
│      │
│      ├── GPUCompressLib(lib_path).__enter__()          [neural_net/gpucompress_ctypes.py]
│      │   └── gpucompress_init()                       ctypes → libgpucompress.so
│      │
│      ├── For each .bin file (1,568 files):
│      │   │
│      │   ├── lib.compute_stats(raw_bytes)             ctypes call
│      │   │   └── gpucompress_compute_stats()          [src/lib/gpucompress_api.cpp]
│      │   │       ├── cudaMemcpy(d_input ← data)
│      │   │       ├── runStatsOnlyPipeline()           [src/lib/stats_kernel.cu]
│      │   │       │   ├── statsPass1Kernel
│      │   │       │   ├── launchEntropyKernelsAsync
│      │   │       │   ├── madPass2Kernel
│      │   │       │   └── finalizeStatsOnlyKernel
│      │   │       └── return (entropy, mad, deriv)
│      │   │
│      │   └── For each of 64 configs (8 algo × 2 shuffle × 4 quant):
│      │       ├── lib.compress(raw_bytes, config)      time it → comp_time_ms
│      │       ├── lib.decompress(compressed, size)     time it → decomp_time_ms
│      │       ├── ratio = original_size / compressed_size
│      │       ├── _compute_psnr() for lossy configs (120 dB for lossless)
│      │       └── Record: (file, algo, quant, shuffle, eb, stats, ratio, times, psnr)
│      │
│      └── _encode_and_split(df)
│          ├── One-hot encode algorithm (8 features)
│          ├── Binary encode quant, shuffle (2 features)
│          ├── Log-transform error_bound, data_size (2 features)
│          ├── Raw stats: entropy, mad, deriv (3 features)
│          ├── Log-transform outputs: comp_time, decomp_time, ratio
│          ├── Clamp PSNR to 120 dB
│          ├── Z-score normalize continuous features
│          ├── Split by file: 80% train, 20% val
│          └── Return {train_X, train_Y, val_X, val_Y, normalization params}
│
├── 2. train_model_with_data(data)                      [neural_net/train.py]
│      │
│      ├── CompressionPredictor(15, 128, 4)             [neural_net/model.py]
│      │   Linear(15→128) + ReLU → Linear(128→128) + ReLU → Linear(128→4)
│      │   19,076 parameters
│      │
│      ├── Training loop (up to 200 epochs):
│      │   ├── Forward: model(batch_X) → predictions
│      │   ├── Loss: MSE(predictions, batch_Y)
│      │   ├── Backward: loss.backward()
│      │   ├── Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
│      │   ├── Scheduler: ReduceLROnPlateau (factor=0.5)
│      │   └── Early stopping: patience=20
│      │
│      └── Save neural_net/weights/model.pt
│          {model_state_dict, x_means, x_stds, x_mins, x_maxs, y_means, y_stds}
│
└── 3. export_weights.py (run separately)               [neural_net/export_weights.py]
       │
       ├── Load model.pt
       ├── Extract: w1[128×15], b1[128], w2[128×128], b2[128], w3[4×128], b3[4]
       └── Write model.nnwt (binary, ~76.6 KB)
           ├── Header: magic, version=2, dims
           ├── x_means[15], x_stds[15], y_means[4], y_stds[4]
           ├── Layer 1-3 weights + biases
           └── x_mins[15], x_maxs[15] (v2: OOD bounds)
```

### 1.3 Key Data Structures

```
NNWeightsGPU (nn_gpu.cu)          AutoStatsGPU (stats_kernel.cu)
├── x_means[15]                   ├── sum, abs_diff_sum
├── x_stds[15]                    ├── vmin, vmax
├── y_means[4]                    ├── num_elements
├── y_stds[4]                     ├── mad_sum
├── w1[128*15], b1[128]           ├── entropy
├── w2[128*128], b2[128]          ├── mad_normalized
├── w3[4*128], b3[4]              └── deriv_normalized
├── x_mins[15]  (v2)
└── x_maxs[15]  (v2)

QTableAction (internal.hpp)       ExperienceSample (experience_buffer.h)
├── algorithm      (0-7)          ├── entropy, mad, first_derivative
├── use_quantization               ├── data_size, error_bound
└── shuffle_size   (0 or 4)       ├── action (0-31)
                                   ├── actual_ratio
                                   └── actual_comp_time_ms
```

---

## 2. The Problem: 32 Compression Configs

Each configuration is encoded as a single integer called an **action** (0-31):

```
action = algorithm + quantization * 8 + shuffle * 16
```

Decoding (defined in `internal.hpp:154`):
```cpp
struct QTableAction {
    int algorithm;          // 0-7
    bool use_quantization;  // (action / 8) % 2
    int shuffle_size;       // (action / 16) % 2 ? 4 : 0
};
```

| Action | Algorithm | Quant | Shuffle |
|--------|-----------|-------|---------|
| 0      | LZ4       | no    | no      |
| 1      | Snappy    | no    | no      |
| ...    | ...       | ...   | ...     |
| 8      | LZ4       | yes   | no      |
| 16     | LZ4       | no    | yes     |
| 24     | LZ4       | yes   | yes     |
| 31     | Bitcomp   | yes   | yes     |

The NN must predict which of these 32 gives the best compression ratio (or speed, or PSNR) for any given input data.

---

## 3. Data Pipeline (Python)

There are two data pipelines:

| Pipeline | File | Input | When to use |
|----------|------|-------|-------------|
| CSV-based | `neural_net/data.py` | Pre-generated `benchmark_results.csv` | When you already have benchmark results |
| Binary (on-the-fly) | `neural_net/binary_data.py` | Raw `.bin` files + `libgpucompress.so` | Train directly from data files, no CSV needed |

Both produce the same dict format consumed by `train.py`.

### 3.0 Binary Data Pipeline (recommended)

**File: `neural_net/binary_data.py`**

Reads raw `.bin` files (float32 arrays), computes stats on GPU via `libgpucompress.so` (through the ctypes wrapper in `neural_net/gpucompress_ctypes.py`), and benchmarks all 64 compression configs per file on-the-fly.

**64 configs per file:**
- 8 algorithms: LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp
- 2 shuffle options: none, 4-byte
- 4 quantization options: none, linear(eb=0.1), linear(eb=0.01), linear(eb=0.001)

```python
from binary_data import load_and_prepare_from_binary

data = load_and_prepare_from_binary(
    data_dir='syntheticGeneration/training_data/',
    lib_path='build/libgpucompress.so',
    max_files=None,     # None = all files
    val_fraction=0.2,
    seed=42,
)
```

For each file, it:
1. Reads raw bytes
2. Calls `gpucompress_compute_stats()` on GPU → entropy, MAD, first_derivative
3. For each of 64 configs: compress, decompress, time both, compute PSNR for lossy configs
4. Builds a DataFrame, applies the same feature encoding as `data.py`, splits by file

**Python ctypes wrapper:** `neural_net/gpucompress_ctypes.py` wraps `libgpucompress.so`:
- Mirrors C structs (`gpucompress_config_t`, `gpucompress_stats_t`)
- Context-managed `GPUCompressLib` class (init/cleanup via `__enter__`/`__exit__`)
- Methods: `compute_stats()`, `compress()`, `decompress()`, `make_config()`

### 3.1 CSV Data Pipeline (legacy)

**File: `neural_net/data.py`**

The benchmark CSV contains ~100K rows. Each row records one (file, config) pair with measured results.

### 3.2 Input Feature Encoding (15 features)

The NN receives 15 input features per config:

| Index | Feature | Encoding | Range |
|-------|---------|----------|-------|
| 0-7   | `alg_lz4` ... `alg_bitcomp` | One-hot (8 binary) | {0, 1} |
| 8     | `quant_enc` | Binary: 1 if linear quantization | {0, 1} |
| 9     | `shuffle_enc` | Binary: 1 if shuffle > 0 | {0, 1} |
| 10    | `error_bound_enc` | `log10(clip(error_bound, 1e-7))` | ~[-7, 0] |
| 11    | `data_size_enc` | `log2(original_size)` | ~[10, 25] |
| 12    | `entropy` | Raw Shannon entropy | [0, 8] |
| 13    | `mad` | Normalized MAD (divided by range) | [0, 1] |
| 14    | `first_derivative` | Normalized mean abs derivative | [0, 1] |

Key design decisions:
- **One-hot for algorithm**: The NN sees all 8 algorithm flags. Each of the 32 threads sets a different one-hot pattern.
- **Log transforms**: Error bound and data size span many orders of magnitude, so log-scale compresses the range.
- **Normalization**: Only the 5 continuous features (indices 10-14) are standardized (zero-mean, unit-variance). Binary/one-hot features keep mean=0, std=1 in the normalization arrays so `(x - 0) / 1 = x`.

### 3.3 Output Encoding (4 targets)

| Index | Output | Encoding |
|-------|--------|----------|
| 0     | `comp_time_log` | `log1p(compression_time_ms)` |
| 1     | `decomp_time_log` | `log1p(decompression_time_ms)` |
| 2     | `ratio_log` | `log1p(compression_ratio)` |
| 3     | `psnr_clamped` | `clip(psnr_db, max=120)` (inf → 120) |

`log1p` is used because times and ratios are positive and right-skewed. The inverse is `expm1`.

### 3.4 Normalization

Both inputs and outputs are standardized using training set statistics:

```python
# Inputs (only continuous features)
x_standardized = (x_raw - x_means) / x_stds

# Outputs (all 4)
y_standardized = (y_raw - y_means) / y_stds
```

These `x_means`, `x_stds`, `y_means`, `y_stds` arrays are saved in the `.nnwt` file so the CUDA kernel can apply the same transformation at inference time.

### 3.5 Feature Bounds (for OOD detection)

The data pipeline also computes per-feature min/max from the training set:

```python
x_mins = train_X_raw.min(axis=0)  # shape: [15]
x_maxs = train_X_raw.max(axis=0)  # shape: [15]
```

These are stored in the `.nnwt` v2 file and used by the C++ `isInputOOD()` function to detect when live data falls outside the training distribution.

### 3.6 Train/Validation Split

Data is split **by file** (not by row) to prevent data leakage:

```python
files = sorted(df['file'].unique())
rng.shuffle(files)
train_files = files[:80%]
val_files = files[20%:]
```

This ensures the model is evaluated on data distributions it hasn't seen during training.

---

## 4. Model Architecture (Python)

**File: `neural_net/model.py`**

```python
class CompressionPredictor(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, output_dim=4):
        self.net = nn.Sequential(
            nn.Linear(15, 128),   # Layer 1: 15 → 128
            nn.ReLU(),
            nn.Linear(128, 128),  # Layer 2: 128 → 128
            nn.ReLU(),
            nn.Linear(128, 4),    # Layer 3: 128 → 4
        )
```

- **Architecture**: 15 → 128 → 128 → 4 (fully connected, ReLU activations)
- **Parameters**: 19,076 total (~75 KB on GPU)
- **Multi-output regression**: Predicts all 4 metrics simultaneously
- No dropout or batch norm — the model is small enough not to need regularization beyond weight decay

---

## 5. Training (Python)

**File: `neural_net/train.py`**

```bash
# Train from binary files (recommended — no CSV step needed)
python neural_net/train.py --data-dir syntheticGeneration/training_data/ --lib-path build/libgpucompress.so

# Train from CSV (legacy)
python neural_net/train.py --csv benchmark_results.csv

# With options
python neural_net/train.py --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so \
    --max-files 100 \
    --epochs 200 --batch-size 512 --lr 1e-3 --patience 20 --hidden-dim 128
```

The `--data-dir` path calls `binary_data.load_and_prepare_from_binary()` which benchmarks all 64 configs per file on GPU. The `--csv` path uses the legacy `data.load_and_prepare()`.

### Training configuration:
- **Loss**: MSE on standardized outputs
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **LR scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Early stopping**: patience=20 epochs
- **Batch size**: 512

### What gets saved to `model.pt`:

```python
torch.save({
    'model_state_dict': best_state,       # Neural net weights
    'x_means': data['x_means'],           # Input normalization means [15]
    'x_stds': data['x_stds'],             # Input normalization stds [15]
    'x_mins': data['x_mins'],             # Feature bounds for OOD [15]
    'x_maxs': data['x_maxs'],             # Feature bounds for OOD [15]
    'y_means': data['y_means'],           # Output normalization means [4]
    'y_stds': data['y_stds'],             # Output normalization stds [4]
    'feature_names': [...],               # Column names
    'output_names': [...],                # Output column names
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
})
```

---

## 6. Weight Export: PyTorch to Binary (.nnwt)

**File: `neural_net/export_weights.py`**

```bash
cd neural_net && python export_weights.py
```

The `.nnwt` binary format is designed for direct `cudaMemcpy` into a single GPU struct:

### Binary layout (v2, little-endian):

```
Offset  Size    Contents
──────  ──────  ──────────────────────────────────
0       24      Header: magic(4) version(4) n_layers(4) input_dim(4) hidden_dim(4) output_dim(4)
24      60      x_means[15] (float32)
84      60      x_stds[15]  (float32)
144     16      y_means[4]  (float32)
160     16      y_stds[4]   (float32)
176     7680    Layer 1 weights [128×15] (float32)
7856    512     Layer 1 biases [128]
8368    65536   Layer 2 weights [128×128]
73904   512     Layer 2 biases [128]
74416   2048    Layer 3 weights [4×128]
76464   16      Layer 3 biases [4]
76480   60      x_mins[15] (float32)  ← v2 only
76540   60      x_maxs[15] (float32)  ← v2 only
```

Total: ~76.6 KB. The entire struct maps directly to `NNWeightsGPU` in CUDA.

### Version compatibility:
- **v1**: No feature bounds. The CUDA loader sets `x_mins = -1e30`, `x_maxs = +1e30` (no OOD detection).
- **v2**: Includes feature bounds. OOD detection is active.

The loader (`loadNNFromBinary()`) handles both:

```cpp
if (version >= 2) {
    file.read(h_weights.x_mins, ...);
    file.read(h_weights.x_maxs, ...);
    g_has_bounds = true;
} else {
    // Set x_mins = -inf, x_maxs = +inf
    g_has_bounds = false;
}
```

---

## 7. GPU Inference Kernel (CUDA)

**File: `src/lib/nn_gpu.cu`**

This is the core of the system. A single kernel launch with 32 threads evaluates all 32 configurations in parallel.

### 7.1 Kernel signature

```cuda
__global__ void nnInferenceKernel(
    const NNWeightsGPU* weights,    // NN weights in GPU global memory
    double entropy,                  // Data statistics (computed by stats pipeline)
    double mad_norm,
    double deriv_norm,
    size_t data_size,
    double error_bound,
    int criterion,                   // 0=ratio, 1=comp_time, 2=decomp_time, 3=psnr
    int* out_action,                 // Best action ID (0-31)
    float* out_predicted_ratio,      // Predicted ratio for winner (nullable)
    int* out_top_actions             // All 32 actions sorted by rank (nullable)
);
```

### 7.2 Thread-to-config mapping

Each of the 32 threads represents one configuration:

```cuda
int tid = threadIdx.x;           // 0-31
int algo_idx = tid % 8;          // 0-7 (which algorithm)
int quant    = (tid / 8) % 2;    // 0 or 1
int shuffle  = (tid / 16) % 2;   // 0 or 1
```

### 7.3 Forward pass per thread

Each thread independently:

**Step 1: Build the 15-feature input vector**

```cuda
float input_raw[15];
// One-hot algorithm (features 0-7)
for (int i = 0; i < 8; i++)
    input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
// Binary features (8-9)
input_raw[8] = float(quant);
input_raw[9] = float(shuffle);
// Continuous features (10-14) - apply same transforms as Python
input_raw[10] = log10(clip(error_bound, 1e-7));
input_raw[11] = log2(data_size);
input_raw[12] = entropy;
input_raw[13] = mad_norm;
input_raw[14] = deriv_norm;
```

**Step 2: Standardize**

```cuda
float input[15];
for (int i = 0; i < 15; i++)
    input[i] = (input_raw[i] - weights->x_means[i]) / weights->x_stds[i];
```

**Step 3: Layer 1 (15 → 128, ReLU)**

```cuda
float hidden1[128];
for (int j = 0; j < 128; j++) {
    float sum = weights->b1[j];
    for (int i = 0; i < 15; i++)
        sum += weights->w1[j * 15 + i] * input[i];
    hidden1[j] = max(sum, 0.0f);  // ReLU
}
```

**Step 4: Layer 2 (128 → 128, ReLU)**

Same pattern with `w2`, `b2`, `hidden1` → `hidden2`.

**Step 5: Layer 3 (128 → 4, linear)**

```cuda
float output_norm[4];
for (int j = 0; j < 4; j++) {
    float sum = weights->b3[j];
    for (int i = 0; i < 128; i++)
        sum += weights->w3[j * 128 + i] * hidden2[i];
    output_norm[j] = sum;
}
```

**Step 6: De-normalize and convert**

```cuda
float output_raw[4];
for (int j = 0; j < 4; j++)
    output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];

float ratio = expm1f(output_raw[2]);  // inverse of log1p
```

### 7.4 Finding the best config

After all 32 threads compute their predictions, the kernel selects the winner:

**Normal path** (no active learning): Tree reduction over shared memory

```cuda
__shared__ float s_vals[32];
__shared__ int   s_idxs[32];
s_vals[tid] = rank_val;
s_idxs[tid] = tid;
__syncthreads();

// Tree reduction: 32 → 16 → 8 → 4 → 2 → 1
for (int s = 16; s > 0; s >>= 1) {
    if (tid < s && s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
    }
    __syncthreads();
}
```

**Active learning path** (when `out_top_actions != nullptr`): Full sort

Thread 0 does an insertion sort on all 32 `(value, action_id)` pairs to produce a ranked list. This is needed for Level 2 exploration, which tries the top-K alternatives.

### 7.5 Host-side wrapper: `runNNInference()`

```cpp
int runNNInference(
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio = nullptr,  // optional
    int* out_top_actions = nullptr          // optional
);
```

This function:
1. Allocates device memory for outputs (`d_action`, optionally `d_predicted_ratio`, `d_top_actions`)
2. Launches the kernel: `nnInferenceKernel<<<1, 32, 0, stream>>>(...)`
3. Copies results back to host
4. Frees device memory
5. Returns the best action ID (0-31)

---

## 8. Stats Pipeline Integration (CUDA)

**File: `src/lib/stats_kernel.cu`**

Before the NN can run, we need to compute the data statistics (entropy, MAD, derivative). This is done entirely on GPU in multiple kernel passes:

```
d_input (float array on GPU)
    │
    ├── statsPass1Kernel ──> sum, min, max, derivative_sum
    │
    ├── Entropy kernels ───> histogram → Shannon entropy
    │
    ├── madPass2Kernel ────> sum(|x - mean|)  (needs mean from pass 1)
    │
    └── finalizeStatsOnlyKernel ──> normalize MAD and derivative by range
```

The full pipeline is orchestrated by `runAutoStatsNNPipeline()`:

```cpp
int runAutoStatsNNPipeline(
    const void* d_input, size_t input_size,
    double error_bound, cudaStream_t stream,
    int* out_action,
    double* out_entropy, double* out_mad, double* out_deriv,
    float* out_predicted_ratio,    // forwarded to NN
    int* out_top_actions           // forwarded to NN
);
```

After the stats kernels complete and results are copied to host, this function calls `runNNInference()` with the computed stats.

### 8.1 Stats-Only Pipeline (for training data generation)

**File: `src/lib/stats_kernel.cu`**, function `runStatsOnlyPipeline()`

A standalone stats pipeline that computes entropy, MAD, and first derivative without running NN inference. Used by the Python ctypes wrapper during training data generation.

```cpp
int runStatsOnlyPipeline(
    const void* d_input,      // Data already on GPU
    size_t input_size,
    cudaStream_t stream,
    double* out_entropy,      // Output: Shannon entropy (bits)
    double* out_mad,          // Output: normalized MAD
    double* out_deriv         // Output: normalized first derivative
);
```

Runs the same GPU kernels as `runAutoStatsNNPipeline()` (statsPass1, entropy, madPass2, finalizeStatsOnly) but stops before NN inference.

### 8.2 Public C API: `gpucompress_compute_stats()`

**File: `include/gpucompress.h`**, **`src/lib/gpucompress_api.cpp`**

Exposes the stats-only pipeline as a public API function for use from Python or other languages:

```c
gpucompress_error_t gpucompress_compute_stats(
    const void* data,              // Host pointer to raw data
    size_t size,                   // Size in bytes
    double* entropy,               // Out: Shannon entropy
    double* mad,                   // Out: normalized MAD
    double* first_derivative       // Out: normalized first derivative
);
```

Copies data to GPU, calls `runStatsOnlyPipeline()`, returns results. This is used by `gpucompress_ctypes.py` to compute stats during on-the-fly benchmarking.

---

## 9. API Integration: gpucompress_compress()

**File: `src/lib/gpucompress_api.cpp`**

Here's the flow when a user calls `gpucompress_compress()` with `ALGO_AUTO`:

```
gpucompress_compress(input, size, output, &out_size, &config, &stats)
    │
    ├── 1. Copy input to GPU (d_input)
    │
    ├── 2. ALGO_AUTO decision:
    │   ├── NN loaded? → runAutoStatsNNPipeline() → action (0-31)
    │   ├── Q-Table loaded? → runAutoStatsPipeline() → action (0-31)
    │   └── Neither? → default to LZ4
    │
    ├── 3. Decode action → algorithm + preprocessing
    │
    ├── 4. Apply preprocessing (quantization, shuffle)
    │
    ├── 5. Compress with chosen algorithm (nvcomp)
    │
    ├── 6. Build header + copy output to host
    │
    ├── 7. [If active learning enabled] Level 1 + Level 2
    │
    └── 8. Fill stats struct, return
```

The NN is preferred over Q-Table. The fallback chain is:
```
NN loaded? → use NN
    └── Q-Table loaded? → use Q-Table
            └── use LZ4 (default)
```

---

## 10. Active Learning System

Active learning is **off by default**. The user opts in:

```c
gpucompress_enable_active_learning("/path/to/experiences.csv");
```

### 10.1 Level 1: Passive Collection

Every `ALGO_AUTO` compression call (when active learning is on) records one experience sample:

```cpp
ExperienceSample sample = {
    entropy, mad, first_derivative,   // data statistics
    input_size, error_bound,          // config inputs
    nn_action,                        // which config was chosen (0-31)
    actual_ratio,                     // actual compression ratio achieved
    0.0                               // compression time (not timed here)
};
experience_buffer_append(&sample);
```

This is written to CSV with columns matching `benchmark_results.csv`, so the retrain script can concatenate them directly.

### 10.2 Prediction Error Check

After compression, the actual ratio is compared to the NN's predicted ratio:

```cpp
double error_pct = |predicted_ratio - actual_ratio| / actual_ratio;
```

### 10.3 Level 2: Exploration

If `error_pct > threshold` (default 20%) or the input is out-of-distribution, the system tries alternative configs:

```
error_pct > 50% or OOD  →  K = 31 (try all configs)
error_pct > 50%          →  K = 9
error_pct > 20%          →  K = 4
```

For each alternative:
1. Decode the action into algorithm + preprocessing
2. Apply preprocessing (quantization, shuffle) to `d_input` (still on GPU)
3. Compress with the alternative algorithm
4. Record the result as another experience sample
5. If it beats the original, replace the output

This means a single compression call may produce up to 32 experience rows when the model is wrong.

### 10.4 OOD Detection

**File: `src/lib/nn_gpu.cu`, function `isInputOOD()`**

Before trusting the NN prediction, the system checks if the 5 continuous features fall within the training data range (with 10% margin):

```cpp
bool isInputOOD(double entropy, double mad, double deriv,
                size_t data_size, double error_bound) {
    // Encode features same way as the kernel
    float error_bound_enc = log10(clip(error_bound, 1e-7));
    float data_size_enc = log2(data_size);

    // Check each against stored training bounds
    for each feature:
        if (value < x_min - 10% * range || value > x_max + 10% * range)
            return true;  // OOD!
    return false;
}
```

If OOD is detected, Level 2 exploration uses K=31 (try everything).

### 10.5 Experience Buffer

**Files: `src/lib/experience_buffer.h`, `src/lib/experience_buffer.cpp`**

Thread-safe CSV writer with `std::mutex`:

```c
int  experience_buffer_init(const char* csv_path);    // Open/create CSV
int  experience_buffer_append(const ExperienceSample*); // Write one row
size_t experience_buffer_count(void);                 // Rows this session
void experience_buffer_cleanup(void);                 // Close file
```

CSV format (matches benchmark_results.csv):
```
entropy,mad,first_derivative,original_size,error_bound,algorithm,quantization,shuffle,compression_ratio,compression_time_ms
```

The action integer is decoded back to human-readable strings (e.g., action=13 → algorithm=`ans`, quantization=`linear`, shuffle=0).

### 10.6 Public API

```c
// Enable with CSV path
gpucompress_enable_active_learning("/path/to/experience.csv");

// Disable
gpucompress_disable_active_learning();

// Query state
int gpucompress_active_learning_enabled(void);

// Set exploration sensitivity (default 0.20 = 20%)
gpucompress_set_exploration_threshold(0.10);  // more sensitive

// How many samples collected
size_t gpucompress_experience_count(void);

// Hot-reload new NN weights without restarting
gpucompress_reload_nn("/path/to/new_model.nnwt");
```

---

## 11. Retraining with Experience Data

**File: `neural_net/retrain.py`**

```bash
python neural_net/retrain.py \
    --original benchmark_results.csv \
    --experience experiences.csv \
    --output neural_net/weights/model.nnwt
```

The retrain script:
1. Loads the original benchmark CSV
2. Loads experience CSV(s) and converts them to benchmark format
3. Fills missing columns (`decompression_time_ms`, `psnr_db`) with medians from original data
4. Combines into one dataset and trains from scratch
5. Exports new `.nnwt` weights

After retraining, hot-load the new model:

```c
gpucompress_reload_nn("neural_net/weights/model.nnwt");
```

---

## 12. File Map

### Python (training side)

| File | Purpose |
|------|---------|
| `neural_net/data.py` | Load CSV, encode features, normalize, split, compute bounds (legacy) |
| `neural_net/binary_data.py` | Load `.bin` files, benchmark on GPU, encode features (recommended) |
| `neural_net/gpucompress_ctypes.py` | Python ctypes wrapper for `libgpucompress.so` |
| `neural_net/model.py` | PyTorch model definition (15→128→128→4) |
| `neural_net/train.py` | Training loop with `--data-dir` and `--csv` paths, save `model.pt` |
| `neural_net/export_weights.py` | Convert `model.pt` → `model.nnwt` binary |
| `neural_net/retrain.py` | Combine original + experience data, retrain, export |

### C/CUDA (inference side)

| File | Purpose |
|------|---------|
| `src/lib/nn_gpu.cu` | `NNWeightsGPU` struct, `nnInferenceKernel`, `loadNNFromBinary()`, `isInputOOD()`, `runNNInference()` |
| `src/lib/stats_kernel.cu` | GPU stats pipeline (entropy, MAD, derivative), `runAutoStatsNNPipeline()`, `runStatsOnlyPipeline()` |
| `src/lib/internal.hpp` | Internal declarations for NN functions |
| `src/lib/gpucompress_api.cpp` | Public API, ALGO_AUTO logic, `gpucompress_compute_stats()`, active learning integration |
| `src/lib/experience_buffer.h` | Experience buffer C API |
| `src/lib/experience_buffer.cpp` | Thread-safe CSV writer |
| `include/gpucompress.h` | Public C API declarations |

### Weights

| File | Format |
|------|--------|
| `neural_net/weights/model.pt` | PyTorch checkpoint (training artifacts) |
| `neural_net/weights/model.nnwt` | Binary weights for CUDA (deployed) |

---

## 13. End-to-End Walkthrough

### Initial training

```bash
# Option A: Train directly from binary data files (recommended)
# No CSV step needed — benchmarks all 64 configs on GPU automatically
cd GPUCompress
cmake --build build                    # Build libgpucompress.so first
python neural_net/train.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so
#    Benchmarks 1,568 files × 64 configs on GPU
#    Produces: neural_net/weights/model.pt

# Option B: Train from pre-generated CSV (legacy)
python neural_net/train.py --csv benchmark_results.csv
#    Produces: neural_net/weights/model.pt

# Export to binary (same for both options)
cd neural_net
python export_weights.py
#    Produces: neural_net/weights/model.nnwt

# Rebuild the library
cd .. && cmake --build build
```

### Using in application code

```c
#include "gpucompress.h"

// Initialize with NN weights
gpucompress_init("neural_net/weights/model.nnwt");

// Compress with automatic algorithm selection
gpucompress_config_t cfg = gpucompress_default_config();
cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
cfg.error_bound = 0.01;  // lossy, or 0.0 for lossless

gpucompress_stats_t stats;
gpucompress_compress(data, size, output, &out_size, &cfg, &stats);
// stats.algorithm_used tells you what the NN chose
```

### With active learning

```c
// Enable experience collection
gpucompress_enable_active_learning("/tmp/experiences.csv");

// Use the library normally - experiences are collected automatically
for (each data buffer) {
    gpucompress_compress(data, size, output, &out_size, &cfg, &stats);
}

// Check how many samples collected
printf("Collected %zu samples\n", gpucompress_experience_count());

// Disable when done
gpucompress_disable_active_learning();
```

### Retrain and reload

```bash
# Retrain with accumulated experience
python neural_net/retrain.py \
    --original benchmark_results.csv \
    --experience /tmp/experiences.csv \
    --output neural_net/weights/model.nnwt
```

```c
// Hot-reload without restarting
gpucompress_reload_nn("neural_net/weights/model.nnwt");
```

### Running tests

```bash
cd build

# Existing NN test (backward compat with v1 .nnwt)
LD_LIBRARY_PATH=. ../neural_net/test_nn_cuda ../neural_net/weights/model.nnwt

# Active learning integration test
LD_LIBRARY_PATH=. ./test_active_learning ../neural_net/weights/model.nnwt
```
