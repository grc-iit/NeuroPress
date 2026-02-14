# Neural Network for Compression Performance Prediction & Ranking

## 1. Problem Statement

GPUCompress supports 8 compression algorithms with optional preprocessing (quantization, byte shuffle). For a given data buffer, which configuration will produce the best compression ratio? The fastest time? The best quality?

Instead of a classification model that picks "the best" action, we build a **regression model** that predicts the actual performance metrics for any given (data + configuration) combination. This lets us **rank all algorithms** for any data — and the user chooses what "best" means.

---

## 2. Model Design

### 2.1 Inputs (15 features after encoding)

| # | Input | Type | Encoding | CSV Column |
|---|-------|------|----------|------------|
| 1 | Compress Library | Categorical (8) | One-hot → 8 features | `algorithm` |
| 2 | Quantization | Binary | 0=none, 1=linear → 1 feature | `quantization` |
| 3 | Shuffle | Binary | 0=off, 1=on → 1 feature | `shuffle` |
| 4 | Error Bound | Continuous | Log-scaled → 1 feature | `error_bound` |
| 5 | Data Size | Continuous | Log2-scaled → 1 feature | `original_size` |
| 6 | Entropy | Continuous | Raw → 1 feature | `entropy` |
| 7 | MAD | Continuous | Raw → 1 feature | `mad` |
| 8 | 1st Derivative | Continuous | Raw → 1 feature | `first_derivative` |

**Total: 15 input features** (8 from one-hot algorithm + 7 scalar)

### 2.2 Outputs (4 regression targets)

| # | Output | Unit | CSV Column |
|---|--------|------|------------|
| 1 | Compression Time | milliseconds | `compression_time_ms` |
| 2 | Decompression Time | milliseconds | `decompression_time_ms` |
| 3 | Compression Ratio | ratio (higher=better) | `compression_ratio` |
| 4 | PSNR | dB (inf for lossless) | `psnr_db` |

### 2.3 How Ranking Works

For a new data buffer, the system:
1. Computes data statistics (entropy, MAD, 1st derivative) — already done by GPU kernels
2. Constructs **64 input vectors** — one for each (algorithm × quantization × shuffle) combo — all sharing the same data stats
3. Runs all 64 through the model → gets 64 sets of predicted (comp_time, decomp_time, ratio, psnr)
4. **Ranks** the 64 configurations by any user-chosen criterion:
   - Best ratio: sort by predicted `compression_ratio` descending
   - Fastest: sort by predicted `compression_time` ascending
   - Balanced: weighted score of all metrics
   - Best quality: sort by predicted `psnr` descending

```
Data buffer → compute stats → [entropy, mad, deriv, size]
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
    [lz4, no_quant, no_shuf]   [lz4, quant, no_shuf]   ... [bitcomp, quant, shuf]
            │                           │                           │
            ▼                           ▼                           ▼
         Model()                     Model()                    Model()
            │                           │                           │
    [time, ratio, psnr]         [time, ratio, psnr]        [time, ratio, psnr]
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        │
                                   RANK & SORT
                                        │
                                  #1: zstd+quant+shuf  (ratio=45.2x)
                                  #2: deflate+quant    (ratio=42.1x)
                                  #3: lz4+quant+shuf   (ratio=38.7x)
                                  ...
```

### 2.4 Why Regression Instead of Classification

| Approach | Pros | Cons |
|----------|------|------|
| **Classification** (predict best action) | Simple, one forward pass | Locked to one reward preset; no runner-up info; can't explain WHY |
| **Regression** (predict metrics) | Flexible ranking criteria; explainable predictions; all algorithms scored | 64 forward passes per query (still fast) |

The regression approach is **strictly more powerful**: you can always derive the classification answer from it (rank then pick #1), but you can't go the other way.

---

## 3. Network Architecture

```
Input [15 features]
    │
    ▼
Linear(15, 128) + ReLU
    │
    ▼
Linear(128, 128) + ReLU
    │
    ▼
Linear(128, 4)
    │
    ▼
Output: [comp_time, decomp_time, ratio, psnr]
```

**Parameter count:**
```
Layer 1: 15×128 + 128  =  2,048
Layer 2: 128×128 + 128 = 16,512
Layer 3: 128×4 + 4     =    516
                          ──────
Total:                   19,076 parameters (~75 KB)
```

**Why 128 hidden units?**
- We have 100K training rows — plenty for a ~19K parameter model (5:1 sample-to-parameter ratio)
- The model must learn interactions between 8 algorithm types and 4 data characteristics across 4 output dimensions — this needs capacity
- Still tiny: 64 forward passes of this model take microseconds

---

## 4. Training Data

### Source: `benchmark_results.csv` (100,352 rows)

Each row is one experiment: a specific file compressed with a specific configuration.

**Data split strategy:** Split by **file** (not by row) to avoid data leakage.
- Train: 80% of files (~1,254 files × 64 configs = ~80,256 rows)
- Validation: 20% of files (~314 files × 64 configs = ~20,096 rows)

### Preprocessing

| Feature | Transform | Rationale |
|---------|-----------|-----------|
| `algorithm` | One-hot (8 dims) | Categorical — no ordinal relationship |
| `quantization` | Binary (0/1) | "none"=0, "linear"=1 |
| `shuffle` | Binary (0/1) | 0→0, 4→1 |
| `error_bound` | log10(x + 1e-6) | Spans 4 orders of magnitude (0 to 0.1) |
| `original_size` | log2(x) | Size in bytes, typically 2^20 to 2^30 |
| `entropy` | Standardize (zero mean, unit var) | Range [0, 8] |
| `mad` | Standardize | Range [0, 0.5] |
| `first_derivative` | Standardize | Range [0, 0.5] |

### Output handling

| Output | Transform | Notes |
|--------|-----------|-------|
| `compression_time_ms` | log1p(x) | Skewed distribution, log stabilizes |
| `decompression_time_ms` | log1p(x) | Same |
| `compression_ratio` | log1p(x) | Ratios from 0.9 to 780 — log compresses range |
| `psnr_db` | Clamp inf→120, then standardize | Lossless gives inf; cap at 120 dB |

### Filtering

- Remove rows where `success=False` (~127 rows)
- Remove rows with missing metrics

---

## 5. Training Details

| Parameter | Value |
|-----------|-------|
| Loss function | MSE (mean squared error) on log-transformed outputs |
| Optimizer | Adam |
| Learning rate | 1e-3, with ReduceLROnPlateau |
| Batch size | 512 |
| Epochs | 200 (early stopping patience=20) |
| Weight decay | 1e-5 (light regularization) |

---

## 6. Evaluation Metrics

### Per-Output Regression Quality
- **MAE** (Mean Absolute Error) on each output (in original scale)
- **R²** (coefficient of determination) on each output
- **MAPE** (Mean Absolute Percentage Error) where meaningful

### Ranking Quality (the real test)
For each (file, error_bound) group in the validation set:
1. Predict metrics for all 64 configs
2. Rank by a criterion (e.g. compression_ratio)
3. Compare predicted rank #1 to actual rank #1
4. Report:
   - **Top-1 accuracy**: how often the predicted best = actual best
   - **Top-3 accuracy**: how often the actual best is in predicted top-3
   - **Regret**: how much worse the predicted best is vs actual best (in ratio, time, etc.)

---

## 7. Data Gaps & Limitations

| Issue | Status | Impact |
|-------|--------|--------|
| CSV has `first_derivative`, not 2nd | **Confirmed: use 1st** | Minimal — 1st derivative captures smoothness well |
| All data is synthetic (from generator) | Known | Model may not generalize perfectly to real-world data; validate with real files in Phase 3 |
| All files are 4MB (`original_size=4194304`) | **Check needed** | If all files are same size, `data_size` feature is useless for training (but still needed at runtime) |
| `psnr_db=inf` for lossless | Handled | Clamp to 120 dB during training |
| Throughput measured includes subprocess overhead | Known | `compression_time_ms` from the CSV measures actual GPU time, not Python subprocess time |

---

## 8. Implementation Plan

### Phase 1: Data Pipeline (`neural_net/data.py`)
- Load CSV
- Encode features
- Split by file
- Normalize

### Phase 2: Model + Training (`neural_net/model.py`, `neural_net/train.py`)
- Define architecture
- Train with early stopping
- Report regression metrics

### Phase 3: Ranking Evaluation (`neural_net/evaluate.py`)
- Predict all 64 configs per file
- Rank and compare to ground truth
- Report top-1/top-3 accuracy and regret

### Phase 4: CUDA Inference Kernel (DONE)
- `export_weights.py` — export PyTorch model to binary `.nnwt` format
- `src/lib/nn_gpu.cu` — GPU weight loading + 32-thread inference kernel
- `src/lib/stats_kernel.cu` — `runAutoStatsNNPipeline()` uses NN instead of Q-Table
- `gpucompress_api.cpp` — auto-detects `.nnwt` vs Q-Table, prefers NN when loaded
- `gpucompress.h` — public API: `gpucompress_load_nn()`, `gpucompress_nn_is_loaded()`
- Integration test verified: loads weights, selects algorithm, compresses, decompresses

---

## 9. File Structure

```
neural_net/
├── PROPOSAL.md          ← this document
├── simple_nn.cu         ← XOR learning example (educational)
├── data.py              ← CSV loading, feature encoding, splits
├── model.py             ← NN architecture definition
├── train.py             ← Training loop with early stopping
├── evaluate.py          ← Ranking evaluation and comparison
├── export_weights.py    ← PyTorch → binary weight exporter
├── test_nn_cuda.cu      ← End-to-end CUDA integration test
└── weights/
    ├── model.pt         ← PyTorch checkpoint (training)
    └── model.nnwt       ← Binary weights for CUDA inference (76 KB)

src/lib/
├── nn_gpu.cu            ← GPU neural net inference kernel
├── stats_kernel.cu      ← Stats pipeline (Q-Table + NN modes)
├── ...
```
