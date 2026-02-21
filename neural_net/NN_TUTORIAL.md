# Neural Network Tutorial — GPUCompress

A step-by-step beginner guide to every part of the neural network system in GPUCompress.

---

## Table of Contents

0. [The Big Picture](#step-0-the-big-picture)
1. [The Configuration Space](#step-1-the-configuration-space)
2. [Data Collection](#step-2-data-collection)
3. [Feature Engineering](#step-3-feature-engineering)
4. [Output Encoding](#step-4-output-encoding)
5. [Normalization](#step-5-normalization)
6. [Train/Validation Split](#step-6-trainvalidation-split)
7. [The Model Architecture](#step-7-the-model-architecture)
8. [Training](#step-8-training)
9. [Weight Export — PyTorch to Binary (.nnwt)](#step-9-weight-export)
10. [GPU Inference — The CUDA Kernel](#step-10-gpu-inference)
11. [The Full Pipeline — End to End](#step-11-the-full-pipeline)
12. [CPU-Only Prediction](#step-12-cpu-only-prediction)
13. [Evaluation](#step-13-evaluation)
14. [Active Learning](#step-14-active-learning)
15. [Retraining](#step-15-retraining)
16. [The ctypes Bridge](#step-16-the-ctypes-bridge)
17. [XGBoost Alternative](#step-17-xgboost-alternative)
18. [Inspection Tool](#step-18-inspection-tool)
19. [Summary — File Map](#step-19-summary)

---

## Step 0: The Big Picture

GPUCompress is a GPU compression library that supports **8 different compression algorithms** (LZ4, Snappy, Deflate, GDeflate, Zstd, ANS, Cascaded, Bitcomp), each with optional **shuffle** and **quantization** preprocessing. That gives dozens of possible configurations.

**The problem**: Which configuration is best for *this specific data*? Trying all of them is too slow. The neural network solves this by **predicting the performance of every configuration in ~150 microseconds** without actually compressing anything.

```
User's data ──> Compute 3 statistics ──> NN predicts all configs ──> Pick best ──> Compress once
                (entropy, MAD, deriv)    (~150μs, 32 threads)       (argmax)
```

### Two operational modes

- **Static mode** (default): The NN is trained once on benchmark data and frozen at deploy time. Weights never change. Zero overhead beyond the ~150us inference.
- **Active learning mode** (opt-in): The NN still runs with frozen weights, but real-world compression results are logged to a CSV file. The model can be retrained offline and hot-reloaded without restarting.

---

## Step 1: The Configuration Space

There are 3 "knobs" to tune:

| Knob | Options | Count |
|------|---------|-------|
| Algorithm | LZ4, Snappy, Deflate, GDeflate, Zstd, ANS, Cascaded, Bitcomp | 8 |
| Shuffle | Off (0) or 4-byte shuffle | 2 |
| Quantization | Off, or Linear with error_bound 0.1 / 0.01 / 0.001 | 4 |

**During training**: 8 x 2 x 4 = **64 configs** (multiple error bounds so the NN learns how error_bound affects quality).

**During GPU inference**: 8 x 2 x 2 = **32 configs** (quantization is just on/off — the user provides a single error_bound).

Each config is encoded as a single integer:

```
action = algorithm + quantization*8 + shuffle*16
```

So action=0 means "LZ4, no quant, no shuffle", action=31 means "Bitcomp, quant, shuffle".

### Action decoding (C++ side)

On the C++ side, actions are decoded back into config parameters using `QTableAction` (defined in `internal.hpp`):

```cpp
struct QTableAction {
    int algorithm;          // action % 8        (0-7)
    bool use_quantization;  // (action / 8) % 2  (0 or 1)
    int shuffle_size;       // (action / 16) % 2 ? 4 : 0
};
```

Example action table:

| Action | Algorithm | Quant | Shuffle |
|--------|-----------|-------|---------|
| 0 | LZ4 | no | no |
| 1 | Snappy | no | no |
| 7 | Bitcomp | no | no |
| 8 | LZ4 | yes | no |
| 16 | LZ4 | no | yes |
| 24 | LZ4 | yes | yes |
| 31 | Bitcomp | yes | yes |

---

## Step 2: Data Collection

The training script (`train.py`) accepts data from **two sources**:

### Option A: From raw binary files (GPU required)

**File: `binary_data.py`**

```bash
python neural_net/train.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so
```

This benchmarks real data files **on the fly**. For each `.bin` file (raw float32 arrays):

1. **Compute 3 statistics** on the GPU (via `gpucompress_ctypes.py` calling the C library):
   - **Entropy**: How random the data is (0 = perfectly uniform, 8 = maximum randomness)
   - **MAD** (Mean Absolute Deviation): How spread out the values are, normalized by range
   - **Second Derivative**: How much the rate of change varies (curvature), normalized by range

2. **Try all 64 configs**: For each one, compress, measure time, decompress, measure time, compute PSNR (quality metric for lossy compression — see Step 4 for what PSNR means)

3. **Record a row** with: file name, algorithm, quant, shuffle, error_bound, data size, entropy, MAD, derivative, compression ratio, compression time, decompression time, PSNR, success/fail

This produces a big table — if you have 100 files, you get 100 x 64 = 6,400 rows.

If a config fails (e.g., an algorithm doesn't support the data), the row is recorded with `success=False` and zeroed metrics. These rows are filtered out during encoding.

You can limit the number of files processed with `--max-files N` (useful for testing).

### Option B: From CSV benchmark results (no GPU needed)

**File: `data.py`**, function `load_from_csv()`

```bash
python neural_net/train.py --csv benchmark_results.csv
```

If you already have benchmark results saved as CSV files, you can skip the GPU benchmarking step entirely. The CSV must contain the same columns: `file`, `algorithm`, `quantization`, `shuffle`, `error_bound`, `original_size`, `entropy`, `mad`, `second_derivative`, `compression_time_ms`, `decompression_time_ms`, `compression_ratio`, `psnr_db`, `success`. You can even pass multiple CSV files: `--csv file1.csv file2.csv`.

This path exists so you can benchmark once, save results, and retrain later without a GPU.

### Both paths converge

Regardless of which option you use, the result is a pandas DataFrame that gets passed to `encode_and_split()` in `data.py`. From that point on, the pipeline is identical.

---

## Step 3: Feature Engineering

**File: `data.py`**, function `encode_and_split()`

The NN needs numerical inputs. The raw columns are transformed into **15 features**:

### Features 0-7: Algorithm (one-hot encoding)

The algorithm name like "lz4" or "zstd" has no numerical meaning. So we use one-hot encoding, where exactly one position is 1 and the rest are 0:

```
lz4     → [1, 0, 0, 0, 0, 0, 0, 0]
snappy  → [0, 1, 0, 0, 0, 0, 0, 0]
zstd    → [0, 0, 0, 0, 1, 0, 0, 0]
bitcomp → [0, 0, 0, 0, 0, 0, 0, 1]
```

This gives each algorithm its own "channel" — the NN can learn that LZ4 is fast but low ratio, Zstd is slow but high ratio, etc.

### Feature 8: Quantization (binary)

```
none   → 0.0
linear → 1.0
```

### Feature 9: Shuffle (binary)

```
shuffle=0 → 0.0
shuffle=4 → 1.0
```

### Feature 10: Error Bound (log-transformed)

Error bounds span many orders of magnitude (0.001 to 0.1). Log makes them comparable:

```
error_bound = 0.01  → log10(0.01)  = -2.0
error_bound = 0.001 → log10(0.001) = -3.0
error_bound = 0 (lossless) → log10(1e-7) = -7.0  (clipped)
```

### Feature 11: Data Size (log2-transformed)

Same idea — data sizes range from KB to GB:

```
1 MB = 1,048,576 bytes → log2(1048576) = 20.0
4 KB = 4,096 bytes     → log2(4096)    = 12.0
```

### Features 12-14: Data Statistics

- **Feature 12**: Entropy (raw, range 0-8)
- **Feature 13**: MAD normalized (range 0-1)
- **Feature 14**: Second derivative normalized (range 0-1)

These 3 features describe *what the data looks like* — the NN uses them to figure out which algorithm will work best.

### Summary table

| Index | Feature | Encoding | Range |
|-------|---------|----------|-------|
| 0-7 | `alg_lz4` ... `alg_bitcomp` | One-hot (8 binary) | {0, 1} |
| 8 | `quant_enc` | Binary: 1 if linear quantization | {0, 1} |
| 9 | `shuffle_enc` | Binary: 1 if shuffle > 0 | {0, 1} |
| 10 | `error_bound_enc` | `log10(clip(error_bound, 1e-7))` | ~[-7, 0] |
| 11 | `data_size_enc` | `log2(original_size)` | ~[10, 25] |
| 12 | `entropy` | Raw Shannon entropy (byte-level, 256-bin) | [0, 8] |
| 13 | `mad` | Normalized MAD (divided by range) | [0, 1] |
| 14 | `second_derivative` | Normalized mean abs derivative | [0, 1] |

---

## Step 4: Output Encoding

The NN predicts **4 values** simultaneously:

| Output | Encoding | Why |
|--------|----------|-----|
| Compression time | `log1p(ms)` | Times are positive and right-skewed; log makes them more normally distributed |
| Decompression time | `log1p(ms)` | Same reason |
| Compression ratio | `log1p(ratio)` | Same reason |
| PSNR | `clip(psnr, max=120)` | Infinity (lossless) is clamped to 120 |

### What is PSNR?

**PSNR (Peak Signal-to-Noise Ratio)** measures the quality of lossy compression in decibels (dB). Higher = better. It is computed as:

```
PSNR = 10 * log10(data_range^2 / MSE)
```

Where `MSE` is the mean squared error between original and decompressed data, and `data_range` is `max(data) - min(data)`. If the data is perfectly reconstructed (lossless), MSE = 0, so PSNR = infinity — which we clamp to 120 dB.

### What is log1p / expm1?

`log1p(x)` means `log(1 + x)`. The inverse is `expm1(x)` which means `e^x - 1`. We use `log1p` instead of plain `log` because it handles zero gracefully (`log1p(0) = 0`, whereas `log(0) = -infinity`).

### Inverse transform

To convert model predictions back to real-world units, we reverse the encoding. This is done by `inverse_transform_outputs()` in `data.py`:

```python
compression_time_ms    = expm1(predicted[0] * y_std[0] + y_mean[0])   # de-normalize, then expm1
decompression_time_ms  = expm1(predicted[1] * y_std[1] + y_mean[1])
compression_ratio      = expm1(predicted[2] * y_std[2] + y_mean[2])
psnr_db                = predicted[3] * y_std[3] + y_mean[3]           # de-normalize only (no log)
```

This two-step reversal (de-standardize, then undo the log transform) is used everywhere predictions are displayed or compared.

---

## Step 5: Normalization

**File: `data.py`**, inside `encode_and_split()`

Features have very different scales (entropy is 0-8, data_size_enc is 10-25). Neural networks learn faster when all features are on the same scale. So we standardize:

```python
x_normalized = (x_raw - mean) / std
```

This transforms each feature to have mean=0 and standard deviation=1.

Important details:
- **Only the 5 continuous features** (indices 10-14) are standardized
- **Binary/one-hot features** (indices 0-9) already have natural values (0 and 1), so their mean is set to 0 and std to 1 — the formula becomes `(x - 0) / 1 = x`, which is a no-op
- **Outputs are also standardized** using training set statistics
- **Normalization statistics come from the training set only** — never from validation data. Using validation data would leak future information and inflate accuracy

The mean and std arrays are saved and shipped with the model — the CUDA kernel applies the exact same normalization at inference time.

### Feature bounds (for OOD detection)

In addition to mean/std, the training set's per-feature **min and max** are computed and saved:

```python
x_mins = train_X_raw.min(axis=0)  # shape: [15]
x_maxs = train_X_raw.max(axis=0)  # shape: [15]
```

These are stored in the `.nnwt` file (v2) and used at inference time to detect when live data falls outside the training distribution (see Step 14: Active Learning, OOD Detection).

---

## Step 6: Train/Validation Split

Data is split **by file**, not by row:

```python
files = sorted(df['file'].unique())
rng.shuffle(files)
train_files = files[:80%]
val_files = files[20%:]
```

**Why by file?** Each file generates 64 rows (one per config). If rows from the same file appear in both train and validation, the NN could "cheat" by memorizing the data statistics. Splitting by file ensures the validation set contains entirely unseen data.

---

## Step 7: The Model Architecture

**File: `model.py`**

```python
class CompressionPredictor(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, output_dim=4):
        self.net = nn.Sequential(
            nn.Linear(15, 128),   # Layer 1: 15 inputs → 128 neurons
            nn.ReLU(),            # Activation: max(x, 0)
            nn.Linear(128, 128),  # Layer 2: 128 → 128 neurons
            nn.ReLU(),            # Activation
            nn.Linear(128, 4),    # Layer 3: 128 → 4 outputs (no activation)
        )
```

### What each part does

**`nn.Linear(in, out)`** — A fully-connected layer. Each output neuron computes:

```
output[j] = bias[j] + sum(weight[j][i] * input[i]  for all i)
```

It is a matrix multiplication plus a bias vector. Layer 1 has `15 * 128 = 1,920` weights plus 128 biases.

**`nn.ReLU()`** — The activation function. It introduces non-linearity:

```
ReLU(x) = max(x, 0)
```

Without this, stacking linear layers would just be one big linear layer (a matrix times a matrix is still a matrix). ReLU lets the network learn curved, complex relationships. It works by "turning off" neurons whose output is negative and keeping positive outputs unchanged.

**No activation on the last layer** — Because we are doing regression (predicting continuous numbers), not classification. The outputs can be any real number.

**Multi-output regression** — The model predicts all 4 outputs (ratio, comp time, decomp time, PSNR) simultaneously from a single forward pass. This is more efficient than training 4 separate models (which is what the XGBoost alternative in Step 17 does), and it allows the hidden layers to share learned features across all outputs.

**No dropout or batch normalization** — The model is small enough (19K parameters) that it doesn't need extra regularization. Weight decay in the optimizer (Step 8) is sufficient to prevent overfitting.

### Visual diagram

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
               │   Layer 1    │         │   Layer 3    │
               │  15 → 128    │         │  128 → 4     │
               │  + ReLU      │         │  (linear)    │
               │              │         │              │
               │ 1,920 weights│         │  512 weights │
               │   128 biases │         │    4 biases  │
               └──────┬───────┘         └──────▲───────┘
                      │                        │
                      ▼                        │
               ┌──────────────┐                │
               │   Layer 2    │────────────────┘
               │  128 → 128   │
               │  + ReLU      │
               │              │
               │16,384 weights│
               │   128 biases │
               └──────────────┘
```

### Parameter count

```
Layer 1:  15 × 128 + 128 =  2,048
Layer 2: 128 × 128 + 128 = 16,512
Layer 3: 128 ×   4 +   4 =    516
                            ──────
Total:                      19,076 parameters (~75 KB)
```

This is intentionally tiny — it needs to run in a single GPU kernel launch in ~150 microseconds.

---

## Step 8: Training

**File: `train.py`**

### The training loop

```
For each epoch (up to 200):
    For each mini-batch (512 samples):
        1. Forward pass:  input → predictions
        2. Compute loss:  MSE between predictions and actual values
        3. Backward pass: compute gradients (how to adjust each weight)
        4. Update weights: Adam optimizer adjusts weights

    Compute validation loss
    If best so far → save this model state
    If no improvement for 20 epochs → stop early
```

### Key training components

**Loss function: MSE (Mean Squared Error)**

```
loss = mean((predicted - actual)^2)
```

This penalizes large errors more than small ones. It is computed on the *standardized* outputs.

**Optimizer: Adam**

- Learning rate: 0.001 — controls how big each weight update step is
- Weight decay: 0.00001 — very light regularization that prevents weights from growing too large
- Adam adapts the learning rate per-parameter based on past gradients, making training faster and more stable than basic gradient descent

**Learning rate scheduler: ReduceLROnPlateau**

- If validation loss hasn't improved for 10 epochs, halve the learning rate
- This helps fine-tune in later training stages when big steps would overshoot

**Early stopping: patience=20**

- If validation loss hasn't improved for 20 consecutive epochs, stop training
- Prevents overfitting — the model with the best validation loss is kept, not the last one

### What gets saved to `model.pt`

```python
{
    'model_state_dict': weights,      # The trained neural network weights
    'input_dim': 15,                  # Architecture dimensions
    'hidden_dim': 128,
    'output_dim': 4,
    'x_means': [...],                 # Input normalization means (15 values)
    'x_stds': [...],                  # Input normalization stds  (15 values)
    'x_mins': [...],                  # Per-feature min for OOD detection (15 values)
    'x_maxs': [...],                  # Per-feature max for OOD detection (15 values)
    'y_means': [...],                 # Output normalization means (4 values)
    'y_stds': [...],                  # Output normalization stds  (4 values)
    'feature_names': [...],           # Column names
    'output_names': [...],            # Output names
    'best_epoch': 45,                 # Which epoch had the best model
    'best_val_loss': 0.032,           # That epoch's validation loss
}
```

### Validation metrics

After training, four per-output metrics are reported on the validation set:
- **MAE** (Mean Absolute Error): average absolute prediction error
- **R-squared**: how much variance the model explains (1.0 = perfect)
- **MAPE** (Mean Absolute Percentage Error): average % error

These are computed in *original scale* (not standardized) using `inverse_transform_outputs()`.

---

## Step 9: Weight Export

**File: `export_weights.py`**

PyTorch `.pt` files are Python-specific. The CUDA kernel needs a raw binary blob it can load with a single `cudaMemcpy`. The export script creates a `.nnwt` file:

### Binary layout (v2, little-endian)

```
Offset  Size     Contents
──────  ───────  ──────────────────────────────────
0       24       Header: magic(4) version(4) n_layers(4)
                         input_dim(4) hidden_dim(4) output_dim(4)
24      60       x_means[15] (float32)
84      60       x_stds[15]  (float32)
144     16       y_means[4]  (float32)
160     16       y_stds[4]   (float32)
176     7680     Layer 1 weights [128x15] (float32)
7856    512      Layer 1 biases [128]
8368    65536    Layer 2 weights [128x128]
73904   512      Layer 2 biases [128]
74416   2048     Layer 3 weights [4x128]
76464   16       Layer 3 biases [4]
76480   60       x_mins[15] (float32)  — v2 only
76540   60       x_maxs[15] (float32)  — v2 only
```

Total: ~76.6 KB. Magic number: `0x4E4E5754` (ASCII for "NNWT").

### Version compatibility

- **v1**: No feature bounds. The CUDA loader disables OOD detection.
- **v2**: Includes feature bounds. OOD detection is active.

### Weight storage layout

PyTorch stores `nn.Linear` weights as `[out_features, in_features]` (row-major). The binary file keeps this same layout. The CUDA kernel reads them as `weight[out_idx * in_dim + in_idx]` — each thread iterates over all input dimensions for each output neuron.

### Usage

```bash
cd neural_net && python export_weights.py
# or with explicit paths:
python export_weights.py --input weights/model.pt --output weights/model.nnwt
```

### Verification

The exporter runs a verification pass that reads back the binary, asserts every field matches the PyTorch model, and runs a sample input through both paths. If anything mismatches, it raises an assertion error.

---

## Step 10: GPU Inference

This is where it all comes together. A **single kernel launch with 32 threads** evaluates all 32 inference configurations in parallel.

### Thread-to-config mapping

```
Thread 0  → LZ4,      no_quant, no_shuffle
Thread 1  → Snappy,   no_quant, no_shuffle
...
Thread 7  → Bitcomp,  no_quant, no_shuffle
Thread 8  → LZ4,      quant,    no_shuffle
...
Thread 16 → LZ4,      no_quant, shuffle
...
Thread 31 → Bitcomp,  quant,    shuffle
```

Computed as:
```cuda
int tid     = threadIdx.x;       // 0-31
int algo    = tid % 8;           // 0-7 (which algorithm)
int quant   = (tid / 8) % 2;    // 0 or 1
int shuffle = (tid / 16) % 2;   // 0 or 1
```

### What each thread does

**1. Build the 15-feature input vector**

All 32 threads share the same data statistics (entropy, MAD, derivative, size, error_bound) but each has different configuration features (algorithm one-hot, quant, shuffle):

```cuda
float input_raw[15];
// One-hot algorithm (features 0-7)
for (int i = 0; i < 8; i++)
    input_raw[i] = (i == algo_idx) ? 1.0f : 0.0f;
// Binary features (8-9)
input_raw[8]  = float(quant);
input_raw[9]  = float(shuffle);
// Continuous features (10-14) — same for all threads
input_raw[10] = log10(clip(error_bound, 1e-7));
input_raw[11] = log2(data_size);
input_raw[12] = entropy;
input_raw[13] = mad_norm;
input_raw[14] = deriv_norm;
```

Example: Thread 4 (zstd, no quant, no shuffle) vs Thread 20 (zstd, no quant, shuffle):

```
                            Thread 4               Thread 20
                            (zstd, no_q, no_s)     (zstd, no_q, shuf)
                            ───────────────────    ───────────────────
Feature 0:  alg_lz4         0.0                    0.0
Feature 4:  alg_zstd        1.0  ← one-hot         1.0  ← same algo
Feature 8:  quant           0.0                    0.0
Feature 9:  shuffle         0.0  ← different        1.0  ← different
Feature 12: entropy         5.23                   5.23 ← same stats
Feature 13: mad             0.142                  0.142
Feature 14: deriv           0.087                  0.087
```

**2. Standardize**

```cuda
float input[15];
for (int i = 0; i < 15; i++)
    input[i] = (input_raw[i] - weights->x_means[i]) / weights->x_stds[i];
```

**3. Forward pass — Layer 1 (15 → 128 with ReLU)**

```cuda
float hidden1[128];
for (int j = 0; j < 128; j++) {
    float sum = weights->b1[j];
    for (int i = 0; i < 15; i++)
        sum += weights->w1[j * 15 + i] * input[i];
    hidden1[j] = max(sum, 0.0f);  // ReLU
}
```

**4. Layer 2 (128 → 128 with ReLU)** — Same pattern.

**5. Layer 3 (128 → 4, no activation)** — Produces raw normalized outputs.

**6. De-normalize and convert**

```cuda
float output_raw[4];
for (int j = 0; j < 4; j++)
    output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];

float ratio = expm1f(output_raw[2]);  // inverse of log1p
```

### Finding the best config

All 32 threads write their ranking metric to **shared memory**, then a **tree reduction** finds the maximum:

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
// Thread 0 now has the best action ID
```

This is a standard GPU parallel reduction pattern — 5 comparison rounds to find the best out of 32.

### Ranking criteria

The NN predicts 4 metrics, but only one is used for ranking. The `criterion` parameter controls which one:

| Value | Name | Behavior |
|-------|------|----------|
| 0 | `NN_RANK_BY_RATIO` | Higher is better (default) |
| 1 | `NN_RANK_BY_COMP_TIME` | Lower is better |
| 2 | `NN_RANK_BY_DECOMP_TIME` | Lower is better |
| 3 | `NN_RANK_BY_PSNR` | Higher is better |

For time-based criteria, the reduction finds the minimum instead of maximum.

### Active learning path (ranked list)

When active learning is enabled and `out_top_actions` is not null, the kernel produces a **full ranked list** of all 32 configs (not just the best one). Thread 0 does an insertion sort on all 32 `(value, action_id)` pairs. This ranked list is used by the Level 2 exploration system (Step 14) to decide which alternative configs to try next.

### Host-side wrapper

The CUDA kernel is called through a host function:

```cpp
int runNNInference(
    double entropy, double mad_norm, double deriv_norm,
    size_t data_size, double error_bound,
    cudaStream_t stream,
    float* out_predicted_ratio,      // nullable
    int* out_top_actions             // nullable: all 32 actions sorted
);
// Returns best action ID (0-31), or -1 on error
```

---

## Step 11: The Full Pipeline

When a user calls `gpucompress_compress(data, config=ALGO_AUTO)`:

```
1. cudaMemcpy(d_input ← data)

2. runAutoStatsNNPipeline(d_input, size, ...)
   a. statsPass1Kernel      → per-block: sum, min, max, |x[i+1] - 2*x[i] + x[i-1]| sum
   b. launchEntropyKernelsAsync → 256-bin byte histogram → Shannon entropy
   c. madPass2Kernel        → sum(|x[i] - mean|)  (needs mean from pass 1)
   d. finalizeStatsOnlyKernel → normalize MAD and derivative by range
   e. cudaMemcpy(host ← entropy, mad, deriv)
   f. runNNInference(entropy, mad, deriv, ...)
      └── nnInferenceKernel<<<1, 32>>>
          ├── Each thread: build input → forward pass → predict metrics
          └── Parallel reduction: pick best action (0-31)

3. decodeAction(action)
   algo  = action % 8
   quant = (action/8) % 2
   shuf  = (action/16) % 2

4. Preprocess if needed (quantize, shuffle)

5. Compress with the chosen algorithm (nvcomp)

6. Build CompressionHeader + cudaMemcpy(host ← output)

7. [If active learning enabled]
   ├── experience_buffer_append(sample)
   ├── Check |predicted - actual| / actual > threshold?
   └── If yes: explore top-K alternatives

8. Return GPUCOMPRESS_SUCCESS
```

### Stats pipeline variants

Two pipeline functions exist in `stats_kernel.cu`:

| Function | Purpose |
|----------|---------|
| `runAutoStatsNNPipeline()` | Computes stats **then** calls `runNNInference()` — used during compression |
| `runStatsOnlyPipeline()` | Computes stats only, no inference — used by Python ctypes during training data generation |

### Stats-only public API

The stats computation is also exposed as a standalone C function:

```c
gpucompress_error_t gpucompress_compute_stats(
    const void* data,        // Host pointer to raw data
    size_t size,             // Size in bytes
    double* entropy,         // Out: Shannon entropy (0-8 bits)
    double* mad,             // Out: normalized MAD (0-1)
    double* second_derivative // Out: normalized second derivative (0-1)
);
```

This copies data to GPU, calls `runStatsOnlyPipeline()`, and returns the results. It is used by `gpucompress_ctypes.py` during training data generation.

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

---

## Step 12: CPU-Only Prediction

**File: `predict.py`**

For users without a GPU, this script:

1. Reads a `.bin` file
2. Computes statistics **on CPU** using `compute_stats_cpu()` from `data.py` (pure numpy implementation that mirrors the GPU kernels):
   - Entropy: byte-level 256-bin histogram → Shannon entropy
   - MAD: mean absolute deviation from mean, normalized by data range
   - Second derivative: mean |x[i+1] - 2*x[i] + x[i-1]|, normalized by data range
3. Builds all 64 config input vectors
4. Runs them through the PyTorch model on CPU
5. Ranks by the chosen metric and prints the results

This evaluates 64 configs (not 32) because it includes multiple error bounds.

Usage:

```bash
python neural_net/predict.py --bin-file some_data.bin
python neural_net/predict.py --bin-file some_data.bin --rank-by psnr_db
```

---

## Step 13: Evaluation

### Ranking accuracy

**File: `evaluate.py`**

For each (file, error_bound) group in the validation set:

1. Predict all configs through the model
2. Rank by predicted metric (e.g., compression ratio)
3. Compare to actual ranking from real benchmarks

The script evaluates **multiple ranking criteria** automatically: `compression_ratio`, `compression_time_ms`, and `psnr_db`.

Key metrics:

| Metric | Meaning | Typical Value |
|--------|---------|---------------|
| **Top-1 accuracy** | Did the NN pick the actual best config? | ~75-80% |
| **Top-3 accuracy** | Is the actual best in the NN's top 3 predictions? | ~88-100% |
| **Mean regret** | Compression ratio lost vs the true optimal choice | ~0-5 |
| **Median regret** | Half the predictions have less regret than this | 0.00 |

The NN reliably picks the best or near-best configuration. When it is wrong, the actual best is almost always within its top 3 predictions.

The script also shows **sample predictions** for a few random validation files, printing the top-8 configs with both actual and predicted values side by side — useful for visually checking how close the NN's estimates are.

### Cross-validation

**File: `cv_eval.py`**

5-fold cross-validation (split by file): divides files into 5 groups, trains on 4 groups, validates on 1, rotates through all 5 combinations. Reports per-output MAE, R-squared, and MAPE across all folds. This gives a more robust estimate of model quality than a single train/val split.

---

## Step 14: Active Learning

Active learning is **off by default**. The NN runs with frozen weights. Active learning is an optional mode that collects data from real-world usage so the model can be retrained later.

**Active learning does NOT update weights at runtime.** It only logs results to a CSV file.

### Enabling active learning

```c
gpucompress_enable_active_learning("/path/to/experiences.csv");
// ... compress many files with ALGO_AUTO ...
gpucompress_disable_active_learning();
```

### Level 1: Passive Collection (every call, zero cost)

After each ALGO_AUTO compression, the actual result is recorded for free:

```
Compress call:
  Input stats:  entropy=5.23, mad=0.142, deriv=0.087, size=4MB, error_bound=0
  NN selected:  zstd + shuffle (action=20)
  NN predicted: ratio=35.8x
  Actual:       ratio=33.1x

Store: (entropy, mad, deriv, size, error_bound, action=20, actual_ratio=33.1, time=26ms)
```

**Limitation**: We only learn about the config we selected, not about alternatives.

### Level 2: Active Exploration (triggered by high error)

After compression, the actual ratio is compared to the NN's predicted ratio:

```cpp
double error_pct = |predicted_ratio - actual_ratio| / actual_ratio;
```

If the error exceeds a threshold, the system tries alternative configs:

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

The data is still on the GPU from the original call — no re-upload needed. Each measurement becomes a new training sample. If exploration discovers a better config, the output is swapped — the user gets the actual best result found.

### Cost summary

```
                        Compression calls    Extra cost per trigger
                        ─────────────────    ─────────────────────
Level 1 (passive):      1 (normal)           0 (free)
Level 2 (light, K=4):   5 total              4x extra
Level 2 (heavy, K=9):   10 total             9x extra
Level 2 (full, K=31):   32 total             31x extra
```

### OOD Detection

**File: `src/lib/nn_gpu.cu`**, function `isInputOOD()`

Before trusting the NN prediction, the system checks if the 5 continuous input features (error_bound, data_size, entropy, MAD, derivative) fall within the training data range (with 10% margin):

```cpp
for each continuous feature:
    float range = x_max - x_min;
    if (value < x_min - 0.1 * range || value > x_max + 0.1 * range)
        return true;  // OOD!
```

If any feature is outside that range, the input is considered Out-Of-Distribution (OOD) and full exploration (K=31) is triggered. The `x_mins` and `x_maxs` arrays come from the `.nnwt` v2 file (see Step 5 and Step 9).

### Cold Start Behavior

When the model encounters a completely new workload:

```
Chunk 1-50:   Features OOD → full exploration (32 configs each)
              50 × 32 = 1,600 new training samples collected
              Trigger retrain → model now knows this workload

Chunk 51+:    Features now in-distribution → NN predicts → within threshold
              No exploration needed
```

The model pays upfront to learn, then amortizes that cost. After warmup, overhead is near-zero.

### Experience buffer

**Files: `src/lib/experience_buffer.h`, `src/lib/experience_buffer.cpp`**

The experience buffer is a thread-safe CSV writer protected by `std::mutex`. Its C API:

```c
int    experience_buffer_init(const char* csv_path);
int    experience_buffer_append(const ExperienceSample* sample);
size_t experience_buffer_count(void);
void   experience_buffer_cleanup(void);
```

Each row in the CSV contains:

```
entropy,mad,second_derivative,original_size,error_bound,algorithm,quantization,shuffle,compression_ratio,compression_time_ms
```

The action integer is decoded back to human-readable strings before writing (e.g., action=13 becomes algorithm=`ans`, quantization=`linear`, shuffle=0).

### Exploration threshold

Configurable at runtime:

```c
gpucompress_set_exploration_threshold(0.10);  // 10% (more aggressive)
gpucompress_set_exploration_threshold(0.30);  // 30% (less aggressive)
// Default: 0.20 (20%)
```

### Full active learning API

```c
gpucompress_enable_active_learning("/path/to/experience.csv");
gpucompress_disable_active_learning();
int    gpucompress_active_learning_enabled(void);   // 1 if enabled, 0 if not
void   gpucompress_set_exploration_threshold(double threshold);
size_t gpucompress_experience_count(void);           // how many samples collected
gpucompress_error_t gpucompress_reload_nn("/path/to/new_model.nnwt");  // hot-reload
```

---

## Step 15: Retraining

**File: `retrain.py`**

When enough experience data has been collected, retrain the model:

```bash
python neural_net/retrain.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so \
    --experience experiences.csv \
    --output neural_net/weights/model.nnwt
```

The retrain script:

1. Benchmarks original `.bin` files on GPU (same as initial training)
2. Loads experience CSV(s) from C++ active learning and converts them to benchmark format
3. Fills missing columns — the experience buffer only records `compression_ratio` and `compression_time_ms`, so `decompression_time_ms` and `psnr_db` are filled with medians from the original benchmark data
4. Combines original + experience data into one dataset, encodes via `encode_and_split()`, and **trains from scratch** (not fine-tuning — a completely fresh model)
5. Exports new `.nnwt` weights directly

After retraining, hot-load the new model without restarting:

```c
gpucompress_reload_nn("neural_net/weights/model.nnwt");
```

This swaps the GPU weights atomically. Compression calls in progress complete with the old weights; subsequent calls use the new weights.

---

## Step 16: The ctypes Bridge

**File: `gpucompress_ctypes.py`**

This is the glue between Python and the C/CUDA library. It uses Python's `ctypes` module to call `libgpucompress.so` directly:

- Mirrors C structs (`gpucompress_config_t`, `gpucompress_stats_t`) as Python classes
- Wraps `gpucompress_compress()`, `gpucompress_decompress()`, `gpucompress_compute_stats()`
- Context-managed: `with GPUCompressLib() as lib:` handles init/cleanup automatically

### Preprocessing constants

The wrapper defines bitmask constants that mirror the C enum:

```python
PREPROC_NONE       = 0x00
PREPROC_SHUFFLE_2  = 0x01
PREPROC_SHUFFLE_4  = 0x02
PREPROC_SHUFFLE_8  = 0x04
PREPROC_QUANTIZE   = 0x10
HEADER_SIZE        = 64     # bytes prepended to every compressed output
```

### Key methods

| Method | Purpose |
|--------|---------|
| `init(model_path)` | Initialize the library (called automatically by `__enter__`) |
| `cleanup()` | Free GPU resources (called automatically by `__exit__`) |
| `compute_stats(data)` | Returns (entropy, mad, second_derivative) computed on GPU |
| `compress(data, config)` | Compress bytes with a specific config |
| `decompress(compressed, original_size)` | Decompress bytes |
| `make_config(algo, shuffle, quantize, error_bound)` | Build a config struct |
| `load_nn(path)` | Load `.nnwt` weights for ALGO_AUTO |

### Algorithm constants

```python
ALGO_AUTO = 0    # NN selects the best config
ALGO_LZ4 = 1, ALGO_SNAPPY = 2, ALGO_DEFLATE = 3, ALGO_GDEFLATE = 4
ALGO_ZSTD = 5, ALGO_ANS = 6, ALGO_CASCADED = 7, ALGO_BITCOMP = 8
```

---

## Step 17: XGBoost Alternative

**Files: `xgb_train.py`, `xgb_predict.py`**

An alternative model using XGBoost (gradient-boosted decision trees) instead of a neural network:

- Trains **one XGBoost model per output** (4 separate models) — unlike the NN which predicts all 4 simultaneously
- Uses **SHAP values** for feature importance analysis — produces a heatmap showing which features matter most for each output. The 8 algorithm one-hot features are aggregated into a single "Compression Library" importance value for cleaner visualization
- Uses the same data pipeline, same feature encoding as the NN
- Saved as `xgb_model.pkl` (pickle format)
- Cannot run on GPU in real-time like the NN — this is for offline analysis and comparison

Usage:

```bash
# Train
python neural_net/xgb_train.py --csv benchmark_results.csv

# Cross-validate
python neural_net/xgb_train.py --csv benchmark_results.csv --cv 5

# Predict
python neural_net/xgb_predict.py --bin-file some_data.bin
```

---

## Step 18: Inspection Tool

**File: `inspect_weights.py`**

A utility to pretty-print a `model.pt` checkpoint:

- Architecture dimensions (input/hidden/output)
- Layer shapes and parameter counts
- Per-feature normalization statistics (mean, std, min, max)
- Per-output normalization statistics
- Training metadata (best epoch, best validation loss)

Usage:

```bash
python neural_net/inspect_weights.py neural_net/weights/model.pt
```

---

## Step 19: Summary

### File map

| File | Role |
|------|------|
| `model.py` | NN architecture definition (15 → 128 → 128 → 4) |
| `data.py` | Feature encoding, normalization, CPU stats, inverse transforms |
| `binary_data.py` | GPU benchmarking of .bin files (64 configs each) |
| `gpucompress_ctypes.py` | Python-to-C/CUDA bridge via ctypes |
| `train.py` | Training loop with early stopping |
| `export_weights.py` | PyTorch `.pt` → binary `.nnwt` with verification |
| `predict.py` | CPU-only inference for all 64 configs |
| `evaluate.py` | Ranking accuracy evaluation (top-1, top-3, regret) |
| `cv_eval.py` | 5-fold cross-validation by file |
| `retrain.py` | Retrain with original + experience data |
| `inspect_weights.py` | Checkpoint inspector |
| `xgb_train.py` | XGBoost alternative with SHAP analysis |
| `xgb_predict.py` | XGBoost CPU-only prediction |

### The core flow

```
  Option A: binary_data.py ─┐
  (benchmark .bin on GPU)    │
                             ├──> data.py ──> train.py ──> export_weights.py ──> CUDA kernel
  Option B: --csv files ─────┘   (encode,     (train,      (export .nnwt,        (32 threads,
                                  normalize)   early stop)   76.6 KB)              ~150μs)
```

### Key data structures

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

### CUDA constants

| Constant | Value |
|----------|-------|
| `NN_INPUT_DIM` | 15 |
| `NN_HIDDEN_DIM` | 128 |
| `NN_OUTPUT_DIM` | 4 |
| `NN_NUM_CONFIGS` | 32 |
| `GPUCOMPRESS_HEADER_SIZE` | 64 |

### Key numbers

| Property | Value |
|----------|-------|
| Input features | 15 (8 algorithm + 2 binary + 5 continuous) |
| Output predictions | 4 (ratio, comp time, decomp time, PSNR) |
| Architecture | 15 → 128 → 128 → 4 |
| Total parameters | 19,076 |
| Model size on GPU | ~76.6 KB |
| Training configs per file | 64 (8 algo x 2 shuffle x 4 quant levels) |
| Inference configs per call | 32 (8 algo x 2 shuffle x 2 quant on/off) |
| Inference overhead | ~150 us (stats + NN combined) |
| Training time | ~15 seconds on GPU |
| Binary format | `.nnwt` v2, magic = 0x4E4E5754 |
