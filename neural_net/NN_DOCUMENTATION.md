# Neural Network Algorithm Selection

This document covers the neural network system that automatically selects the best compression configuration in GPUCompress. It describes the training pipeline, model architecture, GPU inference, active learning, and the complete API.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Configuration Space](#2-configuration-space)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training](#5-training)
6. [Weight Export: PyTorch to Binary (.nnwt)](#6-weight-export-pytorch-to-binary-nnwt)
7. [GPU Inference](#7-gpu-inference)
8. [API Integration](#8-api-integration)
9. [Active Learning](#9-active-learning)
10. [Retraining with Experience Data](#10-retraining-with-experience-data)
11. [Key Data Structures](#11-key-data-structures)
12. [Accuracy](#12-accuracy)
13. [File Map](#13-file-map)
14. [End-to-End Walkthrough](#14-end-to-end-walkthrough)

---

## 1. System Overview

When a user calls `gpucompress_compress()` with `ALGO_AUTO`, the library chooses the best combination of algorithm, quantization, and shuffle for the input data. The NN predicts the performance of all configurations in a single GPU kernel launch and picks the best one.

```
User data ──> GPU Stats Pipeline ──> NN Inference Kernel ──> Best config ──> Compress
                  (entropy,             (32 threads,           (action
                   MAD,                  parallel               0-31)
                   derivative)           forward pass)
```

The system has two operational modes:
- **Static mode** (default): NN trained once on benchmark data, frozen at deploy
- **Active learning mode** (opt-in): collects experience during use, enables manual retraining

---

## 2. Configuration Space

GPUCompress supports **8 compression algorithms** with optional preprocessing:

```
8 Algorithms          2 Shuffle Options       Quantization Options
─────────────         ──────────────────      ──────────────────────
LZ4                   No shuffle              No quantization (lossless)
Snappy                4-byte shuffle          Linear quantization (lossy)
Deflate
Gdeflate
Zstd
ANS
Cascaded
Bitcomp
```

### Training vs Inference Configuration Count

The number of configurations differs between training and inference:

| Context | Configs per file | Breakdown |
|---------|-----------------|-----------|
| **Training** (`binary_data.py`) | **64** | 8 algo × 2 shuffle × 4 quant levels (none, eb=0.1, eb=0.01, eb=0.001) |
| **CUDA inference** (`nn_gpu.cu`) | **32** | 8 algo × 2 shuffle × 2 quant states (on/off, user-specified error_bound) |
| **CPU inference** (`predict.py`) | **64** | 8 algo × 2 shuffle × 4 quant levels (matches training) |

Training benchmarks multiple error bounds so the model learns how error_bound affects compression quality. At inference time the user specifies a single error_bound, so the kernel only needs to evaluate quantization on vs off.

### Action Encoding (Inference)

Each of the 32 CUDA inference configurations is encoded as an integer (0-31):

```
action = algorithm + quantization * 8 + shuffle * 16
```

Decoding (defined in `internal.hpp`):
```cpp
struct QTableAction {
    int algorithm;          // action % 8        (0-7)
    bool use_quantization;  // (action / 8) % 2  (0 or 1)
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

---

## 3. Data Pipeline

### 3.1 Binary Data Benchmarking

**File: `neural_net/binary_data.py`**

The primary data source reads raw `.bin` files (float32 arrays), computes stats on GPU via `libgpucompress.so` (through `gpucompress_ctypes.py`), and benchmarks all 64 compression configs per file.

```python
from binary_data import load_and_prepare_from_binary

data = load_and_prepare_from_binary(
    data_dir='syntheticGeneration/training_data/',
    lib_path='build/libgpucompress.so',
    max_files=None,
    val_fraction=0.2,
    seed=42,
)
```

For each `.bin` file:
1. Reads raw bytes
2. Calls `gpucompress_compute_stats()` on GPU → entropy, MAD, second_derivative
3. For each of 64 configs: compress, decompress, time both, compute PSNR for lossy configs
4. Builds a DataFrame, passes it to `encode_and_split()` from `data.py`

The 64 configs per file are: 8 algorithms × 2 shuffle options (0, 4) × 4 quantization options (none, linear with eb=0.1, 0.01, 0.001).

### 3.2 Feature Encoding (15 inputs)

**File: `neural_net/data.py`**, function `encode_and_split()`

| Index | Feature | Encoding | Range |
|-------|---------|----------|-------|
| 0-7   | `alg_lz4` ... `alg_bitcomp` | One-hot (8 binary) | {0, 1} |
| 8     | `quant_enc` | Binary: 1 if linear quantization | {0, 1} |
| 9     | `shuffle_enc` | Binary: 1 if shuffle > 0 | {0, 1} |
| 10    | `error_bound_enc` | `log10(clip(error_bound, 1e-7))` | ~[-7, 0] |
| 11    | `data_size_enc` | `log2(original_size)` | ~[10, 25] |
| 12    | `entropy` | Raw Shannon entropy (byte-level, 256-bin) | [0, 8] |
| 13    | `mad` | Normalized MAD (divided by range) | [0, 1] |
| 14    | `second_derivative` | Normalized mean abs derivative | [0, 1] |

Design decisions:
- **One-hot for algorithm**: No ordinal relationship between algorithms.
- **Log transforms**: Error bound and data size span many orders of magnitude.
- **Only 5 continuous features (indices 10-14) are standardized** (zero-mean, unit-variance). Binary/one-hot features keep mean=0, std=1 in the normalization arrays so `(x - 0) / 1 = x`.

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
x_standardized = (x_raw - x_means) / x_stds    # inputs (continuous features only)
y_standardized = (y_raw - y_means) / y_stds      # outputs (all 4)
```

These `x_means`, `x_stds`, `y_means`, `y_stds` arrays are saved in the `.nnwt` file so the CUDA kernel can apply the same transformation at inference time.

### 3.5 Feature Bounds (for OOD detection)

Per-feature min/max computed from the training set (before normalization):

```python
x_mins = train_X_raw.min(axis=0)  # shape: [15]
x_maxs = train_X_raw.max(axis=0)  # shape: [15]
```

Stored in `.nnwt` v2 and used by `isInputOOD()` in `nn_gpu.cu` to detect when live data falls outside the training distribution.

### 3.6 Train/Validation Split

Data is split **by file** (not by row) to prevent data leakage:

```python
files = sorted(df['file'].unique())
rng.shuffle(files)
train_files = files[:80%]
val_files = files[20%:]
```

### 3.7 CPU Stats Fallback

**File: `neural_net/data.py`**, function `compute_stats_cpu()`

A pure-numpy fallback that mirrors the GPU implementation. Used by `predict.py` for CPU-only inference:
- **Entropy**: byte-level Shannon entropy (256-bin histogram, log2)
- **MAD**: mean absolute deviation from mean, normalized by data range
- **Second derivative**: mean |x[i+1] - 2*x[i] + x[i-1]|, normalized by data range

### 3.8 Python ctypes Wrapper

**File: `neural_net/gpucompress_ctypes.py`**

Wraps `libgpucompress.so` via ctypes. Context-managed `GPUCompressLib` class with methods:
- `compute_stats(data)` → (entropy, mad, second_derivative)
- `compress(data, config)` → compressed bytes
- `decompress(compressed, original_size)` → decompressed bytes
- `make_config(algo, shuffle, quantize, error_bound)` → `gpucompress_config_t`

Mirrors C structs: `gpucompress_config_t`, `gpucompress_stats_t`.

---

## 4. Model Architecture

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

```
                    15 inputs                 4 outputs
                 ┌───────────┐            ┌───────────┐
                 │ algorithm │            │ comp time │
                 │ quant     │            │ decomp    │
                 │ shuffle   │            │ ratio     │
                 │ err bound │            │ psnr      │
                 │ data size │            └───────────┘
                 │ entropy   │                  ▲
                 │ mad       │                  │
                 │ deriv     │                  │
                 └─────┬─────┘                  │
                       │                        │
                       ▼                        │
               ┌──────────────┐         ┌──────────────┐
               │              │         │              │
               │   Layer 1    │         │   Layer 3    │
               │  15 → 128    │         │  128 → 4     │
               │  + ReLU      │         │  (linear)    │
               │              │         │              │
               │  1,920 weights│         │  512 weights │
               │  128 biases  │         │  4 biases    │
               │              │         │              │
               └──────┬───────┘         └──────▲───────┘
                      │                        │
                      ▼                        │
               ┌──────────────┐         ┌──────────────┐
               │              │         │              │
               │   Layer 2    │────────>│   128 values │
               │  128 → 128   │         │   passed to  │
               │  + ReLU      │         │   Layer 3    │
               │              │         │              │
               │ 16,384 weights│         └──────────────┘
               │  128 biases  │
               │              │
               └──────────────┘

               Total: 19,076 parameters (~75 KB)
```

- **Multi-output regression**: Predicts all 4 metrics simultaneously
- **ReLU**: If positive, keep it; if negative, set to zero
- **Last layer has no activation** — outputs raw values for de-normalization
- No dropout or batch norm — the model is small enough not to need regularization beyond weight decay

---

## 5. Training

**File: `neural_net/train.py`**

```bash
# Train from binary files (benchmarks all 64 configs on GPU automatically)
python neural_net/train.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so

# Train from CSV
python neural_net/train.py --csv benchmark_results.csv

# With options
python neural_net/train.py --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so \
    --max-files 100 --epochs 200 --batch-size 512 --lr 1e-3 \
    --patience 20 --hidden-dim 128
```

### Training configuration

| Parameter | Value |
|-----------|-------|
| Loss function | MSE on standardized outputs |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Early stopping | patience=20 epochs |
| Batch size | 512 |
| Max epochs | 200 |

### What gets saved to `model.pt`

```python
torch.save({
    'model_state_dict': best_state,       # Neural net weights
    'input_dim': 15,                      # Architecture dimensions
    'hidden_dim': 128,
    'output_dim': 4,
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

### Validation metrics

After training, per-output MAE, R², and MAPE are reported on the validation set using `inverse_transform_outputs()` to convert predictions back to original scale.

---

## 6. Weight Export: PyTorch to Binary (.nnwt)

**File: `neural_net/export_weights.py`**

```bash
cd neural_net && python export_weights.py
```

The `.nnwt` binary format is designed for direct `cudaMemcpy` into a single GPU struct (`NNWeightsGPU`).

### Binary layout (v2, little-endian)

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

Total: ~76.6 KB. Magic number: `0x4E4E5754` ("NNWT").

### Version compatibility

- **v1**: No feature bounds. The CUDA loader sets `x_mins = -1e30`, `x_maxs = +1e30` (no OOD detection).
- **v2**: Includes feature bounds. OOD detection is active.

The loader (`loadNNFromBinary()` in `nn_gpu.cu`) handles both versions.

The exporter runs a verification pass that reads back the binary, asserts every field matches the PyTorch model, and runs a sample input through both paths.

---

## 7. GPU Inference

### 7.1 Stats Pipeline

**File: `src/lib/stats_kernel.cu`**

Before the NN runs, three data statistics are computed on GPU in multiple kernel passes:

```
d_input (float array on GPU)
    │
    ├── statsPass1Kernel ──> sum, min, max, |x[i+1]-2*x[i]+x[i-1]| sum
    │
    ├── Entropy kernels ───> 256-bin byte histogram → Shannon entropy
    │
    ├── madPass2Kernel ────> sum(|x[i] - mean|)  (needs mean from pass 1)
    │
    └── finalizeStatsOnlyKernel ──> normalize MAD and derivative by range
```

Two pipeline variants exist:

| Function | Purpose |
|----------|---------|
| `runAutoStatsNNPipeline()` | Computes stats then calls `runNNInference()` — used during compression |
| `runStatsOnlyPipeline()` | Computes stats only, no inference — used by Python ctypes during training data generation |

### 7.2 NN Inference Kernel

**File: `src/lib/nn_gpu.cu`**

A single kernel launch with 32 threads evaluates all 32 configurations in parallel.

```cuda
__global__ void nnInferenceKernel(
    const NNWeightsGPU* weights,
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound,
    int criterion,              // 0=ratio, 1=comp_time, 2=decomp_time, 3=psnr
    int* out_action,            // Best action ID (0-31)
    float* out_predicted_ratio, // nullable
    int* out_top_actions        // nullable: all 32 actions sorted by rank
);
```

#### Thread-to-config mapping

```cuda
int tid = threadIdx.x;           // 0-31
int algo_idx = tid % 8;          // 0-7 (which algorithm)
int quant    = (tid / 8) % 2;    // 0 or 1
int shuffle  = (tid / 16) % 2;   // 0 or 1
```

#### Forward pass per thread

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
// Continuous features (10-14)
input_raw[10] = log10(clip(error_bound, 1e-7));
input_raw[11] = log2(data_size);
input_raw[12] = entropy;
input_raw[13] = mad_norm;
input_raw[14] = deriv_norm;
```

All 32 threads share the same data statistics (features 10-14) but differ in configuration features (0-9). For example:

```
                            Thread 4               Thread 20
                            (zstd, no_q, no_s)     (zstd, no_q, shuf)
                            ───────────────────    ───────────────────
Feature 0:  alg_lz4         0.0                    0.0
Feature 4:  alg_zstd        1.0  ◄── one-hot      1.0  ◄── same algo
Feature 8:  quant           0.0                    0.0
Feature 9:  shuffle         0.0  ◄── different     1.0  ◄── different
Feature 12: entropy         5.23                   5.23  ◄── same stats
Feature 13: mad             0.142                  0.142 ◄── same stats
Feature 14: deriv           0.087                  0.087 ◄── same stats
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

**Steps 4-5**: Same pattern for Layer 2 and Layer 3 (Layer 3 has no ReLU).

**Step 6: De-normalize and convert**

```cuda
float output_raw[4];
for (int j = 0; j < 4; j++)
    output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];

float ratio = expm1f(output_raw[2]);  // inverse of log1p
```

#### Finding the best config

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

**Active learning path** (when `out_top_actions != nullptr`): Thread 0 does an insertion sort on all 32 `(value, action_id)` pairs to produce a ranked list for Level 2 exploration.

### 7.3 Host-side Wrapper

```cpp
int runNNInference(
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio,      // nullable
    int* out_top_actions             // nullable
);
// Returns best action ID (0-31), or -1 on error
```

### 7.4 Complete Inference Flow

```
gpucompress_compress(data, size, output, &out_size, &config, &stats)
│                                                [src/lib/gpucompress_api.cpp]
│
├── 1. cudaMemcpy(d_input ← data)
│
├── 2. runAutoStatsNNPipeline(d_input, size, ...)       [src/lib/stats_kernel.cu]
│      │
│      ├── 2a. statsPass1Kernel<<<N, 256>>>
│      │        Per-block: sum, min, max, |x[i+1]-2*x[i]+x[i-1]|
│      │
│      ├── 2b. launchEntropyKernelsAsync()
│      │        256-bin byte histogram → Shannon entropy
│      │
│      ├── 2c. madPass2Kernel<<<N, 256>>>
│      │        mean = sum/n → MAD = sum(|x[i] - mean|)
│      │
│      ├── 2d. finalizeStatsOnlyKernel<<<1, 1>>>
│      │        mad_norm = (mad_sum/n) / range
│      │        deriv_norm = (abs_diff_sum/(n-1)) / range
│      │
│      ├── 2e. cudaMemcpy(host ← entropy, mad, deriv)
│      │
│      └── 2f. runNNInference(entropy, mad, deriv, ...) [src/lib/nn_gpu.cu]
│              │
│              └── nnInferenceKernel<<<1, 32>>>
│                  ├── Each thread: build input → forward pass → predict metrics
│                  └── Parallel reduction: pick best action (0-31)
│
├── 3. decodeAction(action)                             [src/lib/internal.hpp]
│      algo = action % 8
│      quant = (action / 8) % 2
│      shuffle = (action / 16) % 2
│
├── 4. Preprocessing (if needed)
│      ├── quantize_simple()
│      └── byte_shuffle_simple()
│
├── 5. createCompressionManager() → compress()
│      nvcomp compression with chosen algorithm
│
├── 6. Build CompressionHeader + cudaMemcpy(host ← output)
│
├── 7. [If active learning enabled]
│      ├── experience_buffer_append(sample)
│      ├── Check |predicted - actual| / actual > threshold?
│      └── If yes: explore top-K alternatives
│
└── 8. Return GPUCOMPRESS_SUCCESS
```

---

## 8. API Integration

**File: `src/lib/gpucompress_api.cpp`**

### Fallback chain

```
compress(data, ALGO_AUTO)
    │
    ├── NN loaded?
    │     YES ──> Use NN prediction
    │     NO ──┐
    │          │
    │          ├── Q-Table loaded?
    │          │     YES ──> Use Q-Table lookup
    │          │     NO ──┐
    │          │          │
    │          │          └──> Default to LZ4
    │          │
    └──────────┘
```

If the NN fails during inference (GPU error, corrupted weights), it returns -1 and the system falls back to LZ4.

### Stats-only API

```c
gpucompress_error_t gpucompress_compute_stats(
    const void* data,        // Host pointer to raw data
    size_t size,             // Size in bytes
    double* entropy,         // Out: Shannon entropy (0-8 bits)
    double* mad,             // Out: normalized MAD (0-1)
    double* second_derivative // Out: normalized second derivative (0-1)
);
```

Copies data to GPU, calls `runStatsOnlyPipeline()`, returns results. Used by `gpucompress_ctypes.py` during training data generation.

---

## 9. Active Learning

Active learning is **off by default**. The NN runs with frozen weights. Active learning is an optional mode that collects data from real-world usage so the model can be retrained later.

**Active learning does NOT update weights at runtime.** It only logs results to a file.

### 9.1 Enabling Active Learning

```c
gpucompress_enable_active_learning("/path/to/experiences.csv");

// ... compress many files with ALGO_AUTO ...

gpucompress_disable_active_learning();
```

### 9.2 Level 1: Passive Collection (every call, zero cost)

After each `ALGO_AUTO` compression, the actual result is recorded for free:

```
Compress call:
  Input stats:  entropy=5.23, mad=0.142, deriv=0.087, size=4MB, error_bound=0
  NN selected:  zstd + shuffle (action=20)
  NN predicted: ratio=35.8x
  Actual:       ratio=33.1x

Store: (entropy, mad, deriv, size, error_bound, action=20, actual_ratio=33.1, time=26ms)
```

**Limitation**: We only learn about the config we selected, not about alternatives.

### 9.3 Level 2: Active Exploration (triggered by high error)

After compression, the actual ratio is compared to the NN's predicted ratio:

```cpp
double error_pct = |predicted_ratio - actual_ratio| / actual_ratio;
```

If the error exceeds the threshold (default 20%) or the input is OOD, the system tries alternative configs. The data is still on the GPU from the original call — no re-upload needed.

#### How many alternatives to explore

```
if features_out_of_distribution:
    K = 31          # Full exploration — try all 32 configs
elif prediction_error > 50%:
    K = 9           # Heavy exploration
elif prediction_error > 20%:
    K = 4           # Light exploration
else:
    K = 0           # No exploration — model was accurate enough
```

#### What happens during exploration

```
Step 1: NN already predicted all 32 configs with a ranked list:

  Rank  Config               NN Predicted Ratio
  ─────────────────────────────────────────────
  #1    zstd+shuffle         35.8x    ◄── already ran, actual was 22.4x
  #2    deflate+shuffle      31.2x
  #3    gdeflate+shuffle     29.7x
  #4    bitcomp              27.1x
  #5    zstd                 25.9x

Step 2: Run the next top-K alternatives (e.g. K=4):

  Config               NN Predicted    Actual (measured now)
  ─────────────────────────────────────────────────────────
  zstd+shuffle         35.8x           22.4x    (already done)
  deflate+shuffle      31.2x           38.7x    ◄── actually BETTER
  gdeflate+shuffle     29.7x           35.1x    ◄── also better
  bitcomp              27.1x           15.9x
  zstd                 25.9x           19.8x
```

Each measurement becomes an experience row. The user gets the **actual best** result found during exploration — if exploration discovers a better config, the output is swapped.

#### Cost summary

```
                        Compression calls    Extra cost per trigger
                        ─────────────────    ─────────────────────
Level 1 (passive):      1 (normal)           0 (free)
Level 2 (light, K=4):   5 total              4x extra
Level 2 (heavy, K=9):   10 total             9x extra
Level 2 (full, K=31):   32 total             31x extra
```

### 9.4 OOD Detection

**File: `src/lib/nn_gpu.cu`**, function `isInputOOD()`

Before trusting the NN prediction, the system checks if the 5 continuous features (indices 10-14) fall within the training data range (with 10% margin):

```cpp
bool isInputOOD(double entropy, double mad, double deriv,
                size_t data_size, double error_bound) {
    // Encode features same way as the kernel
    float error_bound_enc = log10(clip(error_bound, 1e-7));
    float data_size_enc = log2(data_size);

    // Check each against stored training bounds (x_mins, x_maxs)
    for each continuous feature:
        float range = x_max - x_min;
        if (value < x_min - 0.1 * range || value > x_max + 0.1 * range)
            return true;  // OOD!
    return false;
}
```

If OOD is detected, Level 2 exploration uses K=31 (try everything).

### 9.5 Experience Buffer

**Files: `src/lib/experience_buffer.h`, `src/lib/experience_buffer.cpp`**

Thread-safe CSV writer with `std::mutex`:

```c
int  experience_buffer_init(const char* csv_path);
int  experience_buffer_append(const ExperienceSample* sample);
size_t experience_buffer_count(void);
void experience_buffer_cleanup(void);
```

CSV format:
```
entropy,mad,second_derivative,original_size,error_bound,algorithm,quantization,shuffle,compression_ratio,compression_time_ms
```

The action integer is decoded back to human-readable strings (e.g., action=13 → algorithm=`ans`, quantization=`linear`, shuffle=0).

### 9.6 Cold Start Behavior

When the model encounters a completely new workload (data that looks nothing like the training data):

```
Chunk 1-50:   Features OOD → full exploration (32 configs each)
              50 × 32 = 1,600 new training samples collected
              Trigger retrain → model now knows this workload

Chunk 51+:    Features now in-distribution → NN predicts → within threshold
              No exploration needed
```

The model pays upfront to learn, then amortizes that cost. After warmup, overhead is near-zero.

### 9.7 Exploration Threshold

Configurable at runtime:

```c
gpucompress_set_exploration_threshold(0.10);  // 10% (more aggressive)
gpucompress_set_exploration_threshold(0.30);  // 30% (less aggressive)
// Default: 0.20 (20%)
```

### 9.8 Public API

```c
gpucompress_enable_active_learning("/path/to/experience.csv");
gpucompress_disable_active_learning();
int gpucompress_active_learning_enabled(void);
void gpucompress_set_exploration_threshold(double threshold);
size_t gpucompress_experience_count(void);
gpucompress_error_t gpucompress_reload_nn("/path/to/new_model.nnwt");
```

---

## 10. Retraining with Experience Data

**File: `neural_net/retrain.py`**

```bash
python neural_net/retrain.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so \
    --experience experiences.csv \
    --output neural_net/weights/model.nnwt
```

The retrain script:
1. Benchmarks original `.bin` files on GPU via `benchmark_binary_files()`
2. Loads experience CSV(s) from C++ active learning and converts them to benchmark format
3. Fills missing columns (`decompression_time_ms`, `psnr_db`) with medians from original data
4. Combines into one dataset, encodes via `encode_and_split()`, and trains from scratch
5. Exports new `.nnwt` weights

After retraining, hot-load the new model without restarting:

```c
gpucompress_reload_nn("neural_net/weights/model.nnwt");
```

This swaps the GPU weights atomically. Compression calls in progress complete with the old weights; subsequent calls use the new weights.

---

## 11. Key Data Structures

```
NNWeightsGPU (nn_gpu.cu)          AutoStatsGPU (stats_kernel.cu)
├── x_means[15]                   ├── sum, abs_diff_sum
├── x_stds[15]                    ├── vmin, vmax
├── y_means[4]                    ├── num_elements
├── y_stds[4]                     ├── mad_sum
├── w1[128*15], b1[128]           ├── entropy
├── w2[128*128], b2[128]          ├── mad_normalized
├── w3[4*128], b3[4]              ├── deriv_normalized
├── x_mins[15]  (v2)              ├── state
└── x_maxs[15]  (v2)              └── action

QTableAction (internal.hpp)       ExperienceSample (experience_buffer.h)
├── algorithm      (0-7)          ├── entropy, mad, second_derivative
├── use_quantization               ├── data_size, error_bound
└── shuffle_size   (0 or 4)       ├── action (0-31)
                                   ├── actual_ratio
                                   └── actual_comp_time_ms
```

### CUDA Constants

| Constant | Value |
|----------|-------|
| `NN_INPUT_DIM` | 15 |
| `NN_HIDDEN_DIM` | 128 |
| `NN_OUTPUT_DIM` | 4 |
| `NN_NUM_CONFIGS` | 32 |

### Ranking Criteria Enum

| Value | Name | Behavior |
|-------|------|----------|
| 0 | `NN_RANK_BY_RATIO` | Higher is better (default) |
| 1 | `NN_RANK_BY_COMP_TIME` | Lower is better |
| 2 | `NN_RANK_BY_DECOMP_TIME` | Lower is better |
| 3 | `NN_RANK_BY_PSNR` | Higher is better |

---

## 12. Accuracy

Evaluation is performed by `neural_net/evaluate.py`, which predicts all configs for each (file, error_bound) group in the validation set, ranks them, and compares to actual ranking.

| Property | Value |
|----------|-------|
| Top-1 accuracy (NN picks the actual best) | ~75-80% |
| Top-3 accuracy (actual best is in NN's top 3) | ~88-100% |
| Mean regret (ratio lost vs optimal) | ~0-5 |
| Median regret | 0.00 |

Exact numbers depend on the training data and evaluation split. The NN reliably picks the best or near-best configuration. When it is wrong, the actual best is within its top 3 predictions.

### Cross-validation

`neural_net/cv_eval.py` runs 5-fold cross-validation (split by file), training a fresh model per fold and reporting MAE, R², and MAPE per output.

---

## 13. File Map

### Python (training side)

| File | Purpose |
|------|---------|
| `neural_net/data.py` | Shared feature encoding (`encode_and_split()`), normalization, CPU stats fallback (`compute_stats_cpu()`), inverse transforms |
| `neural_net/binary_data.py` | Load `.bin` files, benchmark all 64 configs on GPU (`benchmark_binary_files()`), return raw DataFrame |
| `neural_net/gpucompress_ctypes.py` | Python ctypes wrapper for `libgpucompress.so` (`GPUCompressLib` class) |
| `neural_net/model.py` | PyTorch model definition (`CompressionPredictor`: 15→128→128→4) |
| `neural_net/train.py` | Training loop with `--data-dir` or `--csv`, saves `model.pt` |
| `neural_net/export_weights.py` | Convert `model.pt` → `model.nnwt` binary with verification |
| `neural_net/evaluate.py` | Ranking accuracy evaluation (top-1, top-3, regret) |
| `neural_net/predict.py` | CPU-only inference: reads `.bin` file, predicts all 64 configs, ranks them |
| `neural_net/retrain.py` | Combine original binary benchmarks + experience data, retrain, export |
| `neural_net/cv_eval.py` | 5-fold cross-validation, split by file |
| `neural_net/inspect_weights.py` | Pretty-print contents of a `model.pt` checkpoint |
| `neural_net/xgb_train.py` | XGBoost alternative model (one model per output) with SHAP analysis |
| `neural_net/xgb_predict.py` | XGBoost CPU-only prediction |

### C/CUDA (inference side)

| File | Purpose |
|------|---------|
| `src/lib/nn_gpu.cu` | `NNWeightsGPU` struct, `nnInferenceKernel`, `loadNNFromBinary()`, `isInputOOD()`, `runNNInference()` |
| `src/lib/stats_kernel.cu` | GPU stats pipeline (entropy, MAD, derivative), `runAutoStatsNNPipeline()`, `runStatsOnlyPipeline()` |
| `src/lib/internal.hpp` | Internal declarations for NN functions, `QTableAction` struct |
| `src/lib/gpucompress_api.cpp` | Public API, ALGO_AUTO logic, `gpucompress_compute_stats()`, active learning integration |
| `src/lib/experience_buffer.h` | `ExperienceSample` struct, experience buffer C API |
| `src/lib/experience_buffer.cpp` | Thread-safe CSV writer |
| `include/gpucompress.h` | Public C API declarations (NN, active learning, stats, compression) |

### Weights

| File | Format |
|------|--------|
| `neural_net/weights/model.pt` | PyTorch checkpoint (training artifacts) |
| `neural_net/weights/model.nnwt` | Binary weights for CUDA (deployed, ~76.6 KB) |
| `neural_net/weights/xgb_model.pkl` | XGBoost models (alternative, pickle) |

---

## 14. End-to-End Walkthrough

### Initial training

```bash
# Build the library first
cd GPUCompress
cmake --build build

# Train from binary data files (benchmarks all 64 configs on GPU)
python neural_net/train.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so
#    Produces: neural_net/weights/model.pt

# Export to binary
cd neural_net && python export_weights.py
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

### CPU-only prediction (no GPU needed)

```bash
python neural_net/predict.py --bin-file some_data.bin
python neural_net/predict.py --bin-file some_data.bin --rank-by psnr_db
```

### With active learning

```c
gpucompress_enable_active_learning("/tmp/experiences.csv");

for (each data buffer) {
    gpucompress_compress(data, size, output, &out_size, &cfg, &stats);
}

printf("Collected %zu samples\n", gpucompress_experience_count());
gpucompress_disable_active_learning();
```

### Retrain and reload

```bash
python neural_net/retrain.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so \
    --experience /tmp/experiences.csv \
    --output neural_net/weights/model.nnwt
```

```c
// Hot-reload without restarting
gpucompress_reload_nn("neural_net/weights/model.nnwt");
```

### Inspect a checkpoint

```bash
python neural_net/inspect_weights.py neural_net/weights/model.pt
```

### Evaluate ranking accuracy

```bash
python neural_net/evaluate.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so
```

### Cross-validate

```bash
python neural_net/cv_eval.py --csv benchmark_results.csv --folds 5
```

---

## 15. Key Numbers

| Property | Value |
|----------|-------|
| Input features | 15 (8 algorithm + 2 binary + 5 continuous) |
| Output predictions | 4 (ratio, comp time, decomp time, PSNR) |
| Architecture | 15 → 128 → 128 → 4 |
| Total parameters | 19,076 |
| Model size on GPU | ~76.6 KB |
| Training configs per file | 64 (8 algo × 2 shuffle × 4 quant levels) |
| Inference configs per call | 32 (8 algo × 2 shuffle × 2 quant on/off) |
| Inference overhead | ~150 μs (stats + NN combined) |
| Training time | ~15 seconds on GPU |
| Binary format | `.nnwt` v2, magic=0x4E4E5754 |
| Header size | 64 bytes (`GPUCOMPRESS_HEADER_SIZE`) |
