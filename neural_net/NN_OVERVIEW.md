# Neural Network Algorithm Selection: Architecture Overview

This document introduces the neural network system that automatically selects the best compression algorithm for any given data. It covers the training process, model architecture, and how inference works at runtime.

---

## 1. The Problem

GPUCompress supports **8 compression algorithms**, each with **2 shuffle options** and **2 quantization options**, giving **32 possible configurations** for any piece of data.

```
8 Algorithms          2 Shuffle Options       2 Quantization Options
─────────────         ──────────────────      ──────────────────────
LZ4                   No shuffle              No quantization (lossless)
Snappy                4-byte shuffle          Linear quantization (lossy)
Deflate
Gdeflate                    8 × 2 × 2 = 32 configurations
Zstd
ANS
Cascaded
Bitcomp
```

Different data benefits from different configurations. Highly structured data compresses best with Zstd. Random data barely compresses with anything. The NN learns which configuration works best for which type of data.

---

## 2. Training Pipeline

Training happens **offline** before deployment. It is a three-stage process.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA COLLECTION                        │
│                                                                        │
│   1000 binary files (float32 arrays)                                   │
│   ─────────────────────────────────                                    │
│                                                                        │
│   For EACH file:                                                       │
│                                                                        │
│     ┌──────────┐      GPU Stats Pipeline      ┌──────────────────┐    │
│     │          │  ─────────────────────────>   │ entropy = 2.69   │    │
│     │ raw data │  compute on GPU               │ MAD     = 0.13   │    │
│     │ (floats) │                               │ deriv   = 0.003  │    │
│     │          │                               └──────────────────┘    │
│     └──────────┘                                                       │
│          │                                                             │
│          │         Try ALL 32 configurations                           │
│          ▼                                                             │
│     ┌──────────────────────────────────────────────────────────┐      │
│     │  Config 1:  lz4,        no shuffle, no quant  → ratio=40 │      │
│     │  Config 2:  lz4,        no shuffle, quant     → ratio=73 │      │
│     │  Config 3:  lz4,        shuffle,    no quant  → ratio=18 │      │
│     │  ...                                                      │      │
│     │  Config 32: bitcomp,    shuffle,    quant     → ratio=12 │      │
│     └──────────────────────────────────────────────────────────┘      │
│                                                                        │
│   Result: 1000 files × 32 configs = 32,000 benchmark rows             │
│                                                                        │
│   Each row records:                                                    │
│     INPUT:  data stats + config choice                                 │
│     OUTPUT: actual compression ratio, speed, quality                   │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: FEATURE ENCODING                         │
│                                                                        │
│   Raw benchmark data is transformed into numbers the NN can learn:     │
│                                                                        │
│   INPUTS (15 features per row):                                        │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                                                              │     │
│   │  [0-7]  Algorithm        one-hot encoded (8 binary flags)    │     │
│   │                          e.g. zstd = [0,0,0,0,1,0,0,0]      │     │
│   │                                                              │     │
│   │  [8]    Quantization     0 = off, 1 = on                    │     │
│   │                                                              │     │
│   │  [9]    Shuffle          0 = off, 1 = on                    │     │
│   │                                                              │     │
│   │  [10]   Error bound      log10 scale (large range → small)   │     │
│   │                                                              │     │
│   │  [11]   Data size        log2 scale  (large range → small)   │     │
│   │                                                              │     │
│   │  [12]   Entropy          raw value (0 to 8 bits)             │     │
│   │                                                              │     │
│   │  [13]   MAD              normalized (0 to 1)                 │     │
│   │                                                              │     │
│   │  [14]   First derivative normalized (0 to 1)                 │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│   OUTPUTS (4 targets per row):                                         │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                                                              │     │
│   │  [0]  Compression time      log1p scale (ms)                 │     │
│   │  [1]  Decompression time    log1p scale (ms)                 │     │
│   │  [2]  Compression ratio     log1p scale                      │     │
│   │  [3]  PSNR                  clamped to 120 dB for lossless   │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│   NORMALIZATION:                                                       │
│                                                                        │
│   All continuous features are standardized using training set stats:    │
│                                                                        │
│       normalized = (value - mean) / std_deviation                      │
│                                                                        │
│   These means and standard deviations (x_means, x_stds, y_means,      │
│   y_stds) are saved and shipped with the model — the GPU needs them    │
│   to apply the same transformation at inference time.                  │
│                                                                        │
│   DATA SPLIT:                                                          │
│                                                                        │
│       800 files (80%) → Training set                                   │
│       200 files (20%) → Validation set (never trained on)              │
│                                                                        │
│   Split is by FILE, not by row, to prevent data leakage.              │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3: TRAINING                              │
│                                                                        │
│          Training data                                                 │
│          (25,600 rows)                                                 │
│               │                                                        │
│               ▼                                                        │
│     ┌───────────────────┐                                              │
│     │   Shuffle rows    │                                              │
│     │   into batches    │                                              │
│     │   of 512          │                                              │
│     └────────┬──────────┘                                              │
│              │                                                         │
│              ▼                                                         │
│     ┌───────────────────┐      ┌──────────────────────┐               │
│     │                   │      │                      │               │
│     │   Neural Network  │─────>│  Predicted outputs   │               │
│     │   (15 → 128 →    │      │  (ratio, time, etc)  │               │
│     │    128 → 4)       │      │                      │               │
│     │                   │      └──────────┬───────────┘               │
│     └───────────────────┘                 │                            │
│                                           │  Compare predictions      │
│                                           │  to actual results        │
│                                           ▼                            │
│                                  ┌────────────────┐                    │
│                                  │   Loss (MSE)   │                    │
│                                  │   How wrong    │                    │
│                                  │   was the NN?  │                    │
│                                  └───────┬────────┘                    │
│                                          │                             │
│                                          │  Backpropagation:           │
│                                          │  adjust weights to          │
│                                          │  reduce the error           │
│                                          ▼                             │
│                                  ┌────────────────┐                    │
│                                  │ Update weights │                    │
│                                  │ (Adam optimizer)│                   │
│                                  └────────────────┘                    │
│                                                                        │
│   Repeat for up to 200 epochs (full passes over all training data).    │
│   Stop early if validation loss stops improving for 20 epochs.         │
│                                                                        │
│   Final output: model.pt (PyTorch checkpoint)                          │
│     Contains: trained weights + normalization stats                    │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture

The neural network is a simple 3-layer fully connected network.

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

**ReLU** (Rectified Linear Unit) is a simple function: if the value is positive, keep it; if negative, set it to zero. This gives the network the ability to learn non-linear patterns.

The last layer has **no** ReLU — it outputs raw numbers that can be any value (positive or negative), which are then transformed back into real-world units.

---

## 4. Weight Export

The trained model must be converted from PyTorch format to a binary format the GPU can load directly.

```
    model.pt                              model.nnwt
 (PyTorch format)                      (GPU binary format)
 ─────────────                         ──────────────────

 Trained weights              export_weights.py           Flat binary file
 Normalization stats    ────────────────────────>    loaded directly into
 Training metadata                                   GPU memory via cudaMemcpy

                                                     ┌──────────────────┐
                                                     │ Header (24 B)    │
                                                     │  magic, version  │
                                                     │  dimensions      │
                                                     ├──────────────────┤
                                                     │ x_means  (15)   │
                                                     │ x_stds   (15)   │
                                                     │ y_means  (4)    │
                                                     │ y_stds   (4)    │
                                                     ├──────────────────┤
                                                     │ Layer 1 weights  │
                                                     │ Layer 1 biases   │
                                                     │ Layer 2 weights  │
                                                     │ Layer 2 biases   │
                                                     │ Layer 3 weights  │
                                                     │ Layer 3 biases   │
                                                     ├──────────────────┤
                                                     │ x_mins   (15)   │
                                                     │ x_maxs   (15)   │
                                                     │ (OOD detection)  │
                                                     └──────────────────┘
                                                          ~75 KB total
```

---

## 5. Online Inference

When a user compresses data with `ALGO_AUTO`, the entire decision happens on the GPU in microseconds.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME INFERENCE PIPELINE                         │
│                                                                        │
│   User calls: compress(data, ALGO_AUTO)                                │
│                                                                        │
│   ┌──────────┐                                                         │
│   │          │                                                         │
│   │ New data │──── Copy to GPU ────┐                                   │
│   │ (floats) │                     │                                   │
│   │          │                     ▼                                   │
│   └──────────┘          ┌─────────────────────┐                        │
│                         │   GPU Stats Kernels  │                       │
│                         │                     │                        │
│                         │  Pass 1: sum, min,  │                        │
│                         │    max, derivative  │                        │
│                         │                     │                        │
│                         │  Pass 2: entropy    │                        │
│                         │    (byte histogram) │                        │
│                         │                     │                        │
│                         │  Pass 3: MAD        │                        │
│                         │    (mean abs dev)   │                        │
│                         │                     │                        │
│                         │  Finalize:          │                        │
│                         │    normalize by     │                        │
│                         │    data range       │                        │
│                         └──────────┬──────────┘                        │
│                                    │                                   │
│                                    ▼                                   │
│                         ┌──────────────────────┐                       │
│                         │  3 statistics:       │                       │
│                         │  entropy  = 2.69     │                       │
│                         │  MAD      = 0.13     │                       │
│                         │  deriv    = 0.003    │                       │
│                         └──────────┬───────────┘                       │
│                                    │                                   │
│                                    ▼                                   │
│              ┌─────────────────────────────────────────┐               │
│              │        NN INFERENCE KERNEL               │               │
│              │        (1 block, 32 threads)             │               │
│              │                                         │               │
│              │  Thread 0:  lz4,    no quant, no shuf   │               │
│              │  Thread 1:  snappy, no quant, no shuf   │               │
│              │  Thread 2:  deflate, no quant, no shuf  │               │
│              │  ...                                    │               │
│              │  Thread 8:  lz4,    quant,    no shuf   │               │
│              │  ...                                    │               │
│              │  Thread 16: lz4,    no quant, shuf      │               │
│              │  ...                                    │               │
│              │  Thread 31: bitcomp, quant,   shuf      │               │
│              │                                         │               │
│              │  EACH thread independently:             │               │
│              │                                         │               │
│              │  ┌───────────────────────────────────┐  │               │
│              │  │ 1. Build 15-feature input vector  │  │               │
│              │  │    (my config + data stats)       │  │               │
│              │  │                                   │  │               │
│              │  │ 2. Normalize: (x - means) / stds │  │               │
│              │  │                                   │  │               │
│              │  │ 3. Layer 1: multiply + ReLU       │  │               │
│              │  │ 4. Layer 2: multiply + ReLU       │  │               │
│              │  │ 5. Layer 3: multiply              │  │               │
│              │  │                                   │  │               │
│              │  │ 6. De-normalize output            │  │               │
│              │  │    → predicted ratio, time, etc   │  │               │
│              │  └───────────────────────────────────┘  │               │
│              │                                         │               │
│              │  After all 32 threads finish:           │               │
│              │                                         │               │
│              │  ┌───────────────────────────────────┐  │               │
│              │  │  Parallel reduction:              │  │               │
│              │  │  Compare all 32 predicted ratios  │  │               │
│              │  │  Pick the highest one             │  │               │
│              │  │                                   │  │               │
│              │  │  32 → 16 → 8 → 4 → 2 → 1       │  │               │
│              │  │  Winner: action = 4 (zstd)       │  │               │
│              │  └───────────────────────────────────┘  │               │
│              └────────────────────┬────────────────────┘               │
│                                   │                                    │
│                                   ▼                                    │
│                        ┌─────────────────────┐                         │
│                        │ Decode action = 4   │                         │
│                        │                     │                         │
│                        │ algo  = 4 % 8  = 4  │ → Zstd                 │
│                        │ quant = 4 / 8  = 0  │ → No quantization      │
│                        │ shuf  = 4 / 16 = 0  │ → No shuffle           │
│                        └──────────┬──────────┘                         │
│                                   │                                    │
│                                   ▼                                    │
│                        ┌─────────────────────┐                         │
│                        │                     │                         │
│                        │  Compress with Zstd │                         │
│                        │  (nvcomp library)   │                         │
│                        │                     │                         │
│                        └──────────┬──────────┘                         │
│                                   │                                    │
│                                   ▼                                    │
│                        ┌─────────────────────┐                         │
│                        │  Return compressed  │                         │
│                        │  data to user       │                         │
│                        └─────────────────────┘                         │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. What the NN Actually Predicts

The NN does **not** output a single "best algorithm" label. Instead, it predicts **4 performance metrics** for each of the 32 configurations, then picks the one with the best predicted metric.

```
    For one configuration (e.g. zstd, no shuffle, no quant):

    ┌─────────────────┐         ┌──────────────────┐        ┌────────────────┐
    │ NN raw output   │         │ De-normalized    │        │ Real-world     │
    │ (normalized)    │─── ×std+mean ──│ (log-space)      │── expm1 ──│ values          │
    │                 │         │                  │        │                │
    │ [-1.36]         │         │ [1.14]           │        │ ratio = 2.14   │
    │ [ 1.77]         │         │ [2.85]           │        │ comp  = 16 ms  │
    │ [ 2.79]         │         │ [2.34]           │        │ decomp = 9 ms  │
    │ [ 1.60]         │         │ [120.2]          │ direct │ psnr = 120 dB  │
    └─────────────────┘         └──────────────────┘        └────────────────┘
```

The system then compares the predicted **compression ratio** across all 32 threads and picks the highest one as the winner.

---

## 7. Complete Lifecycle

```
                         OFFLINE (one-time)
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │   .bin files ──> GPU Benchmark ──> Training ──> model.pt
    │   (1000 files)   (32K rows)       (200 epochs)    │
    │                                                    │
    │                                          export_weights.py
    │                                                    │
    │                                                    ▼
    │                                               model.nnwt
    │                                               (75 KB)
    └───────────────────────────────────────────────────┬───┘
                                                        │
                        ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─
                                                        │
                         ONLINE (every compression)     │
    ┌───────────────────────────────────────────────────┼───┐
    │                                                   │   │
    │   Application starts                              │   │
    │       │                                           │   │
    │       ├── gpucompress_init()                      │   │
    │       │       Load model.nnwt into GPU memory ◄───┘   │
    │       │                                               │
    │       ├── gpucompress_compress(data, ALGO_AUTO)        │
    │       │       │                                       │
    │       │       ├── Compute stats on GPU (~100 μs)      │
    │       │       ├── NN inference on GPU (~50 μs)        │
    │       │       ├── Pick best config                    │
    │       │       └── Compress with chosen algorithm      │
    │       │                                               │
    │       ├── gpucompress_compress(more_data, ALGO_AUTO)   │
    │       │       └── (same process, different result      │
    │       │            based on data characteristics)      │
    │       │                                               │
    │       └── gpucompress_cleanup()                        │
    │               Free GPU memory                         │
    │                                                       │
    └───────────────────────────────────────────────────────┘
```

---

## 8. Active Learning (Optional)

By default, the NN runs with **frozen weights** — it predicts and compresses, nothing more. Active learning is an optional mode that collects data from real-world usage so the model can be retrained later.

**Important: Active learning does NOT update weights at runtime.** It only logs results to a file.

### Enabling Active Learning

```
gpucompress_enable_active_learning("/path/to/experiences.csv");

// ... compress many files with ALGO_AUTO ...

gpucompress_disable_active_learning();
```

### What Happens During Each ALGO_AUTO Call

```
    compress(data, ALGO_AUTO)
        │
        ├── NN picks config, compresses → gets actual_ratio
        │
        ├── LEVEL 1: Passive Collection (always)
        │       │
        │       └── Log one row to experience file:
        │           [entropy, MAD, deriv, size, config, actual_ratio]
        │
        ├── Check: |predicted_ratio - actual_ratio| / actual_ratio
        │
        │   If error > 20% threshold, OR data is out-of-distribution:
        │
        └── LEVEL 2: Exploration
                │
                ├── Error > 50% → try 9 alternative configs
                ├── Error > 20% → try 4 alternative configs
                └── OOD data    → try ALL 31 alternative configs

                Each alternative:
                  1. Actually compress with that config
                  2. Log the result to experience file
                  3. If better than NN's pick → swap the output
                     (user gets the better result automatically)
```

### Exploration Threshold

The threshold that triggers Level 2 exploration is configurable:

```
gpucompress_set_exploration_threshold(0.10);  // 10% (more aggressive)
gpucompress_set_exploration_threshold(0.30);  // 30% (less aggressive)
// Default: 0.20 (20%)
```

### Out-of-Distribution (OOD) Detection

The model stores the min/max of each continuous feature from training data (saved in the `.nnwt` file as `x_mins` and `x_maxs`). At inference time, if any input feature falls outside the training range, the data is flagged as OOD.

```
    Training data had:  entropy ∈ [0.1, 6.8]
    New data has:       entropy = 7.9
                                       → OOD detected
                                       → Explore all 31 alternatives
```

### Retraining Loop

The experience file is just raw data. To actually improve the model, you retrain offline:

```
    RUNTIME                              OFFLINE
    ───────                              ───────

    compress(data1, ALGO_AUTO) ──┐
    compress(data2, ALGO_AUTO) ──┤
    compress(data3, ALGO_AUTO) ──┤
    ...                          │
                                 ▼
                          experiences.csv
                          (accumulates over time)
                                 │
                                 ▼
                         python retrain.py \
                           --data-dir training_data/ \
                           --experience experiences.csv \
                           --output model_v2.nnwt
                                 │
                                 ▼
                          model_v2.nnwt
                                 │
                                 ▼
                    gpucompress_reload_nn("model_v2.nnwt")
                    (hot-reload, no restart needed)
```

### Hot-Reload

New weights can be loaded into a running system without restarting:

```
gpucompress_reload_nn("/path/to/model_v2.nnwt");
```

This swaps the GPU weights atomically. Compression calls in progress complete with the old weights; subsequent calls use the new weights.

---

## 9. Key Numbers

| Property | Value |
|----------|-------|
| Input features | 15 (8 algorithm + 2 binary + 5 continuous) |
| Output predictions | 4 (ratio, comp time, decomp time, PSNR) |
| Architecture | 15 → 128 → 128 → 4 |
| Total parameters | 19,076 |
| Model size on GPU | ~75 KB |
| Configurations evaluated | 32 (in parallel, one per GPU thread) |
| Inference overhead | ~150 μs (stats + NN combined) |
| Training data | 1000 files × 32 configs = 32,000 rows |
| Training time | ~15 seconds on GPU |

---

## 10. Fallback Behavior

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

If the NN fails during inference (GPU error, corrupted weights), it returns -1 and the system falls back to LZ4 as a safe default.

---

## 11. Accuracy (tested on 200 training files, 52 held-out test files)

| Metric | Value |
|--------|-------|
| Top-1 accuracy (NN picks the actual best) | 79.5% |
| Top-3 accuracy (actual best is in NN's top 3) | 100.0% |
| Mean regret (ratio lost vs optimal) | 4.89 |
| Median regret | 0.00 |

The NN reliably picks the best or near-best configuration. When it is wrong, the actual best is always within its top 3 predictions.

---

## File References

| File | Role |
|------|------|
| `neural_net/data.py` | Feature encoding and normalization |
| `neural_net/binary_data.py` | GPU benchmarking of training files |
| `neural_net/model.py` | PyTorch model definition |
| `neural_net/train.py` | Training loop |
| `neural_net/export_weights.py` | Convert model.pt → model.nnwt |
| `neural_net/evaluate.py` | Ranking accuracy evaluation |
| `src/lib/nn_gpu.cu` | CUDA inference kernel + OOD detection |
| `src/lib/stats_kernel.cu` | GPU statistics computation |
| `src/lib/experience_buffer.cpp` | Thread-safe experience logger |
| `src/lib/gpucompress_api.cpp` | Public API + active learning logic |
| `neural_net/retrain.py` | Retrain model with experience data |
