# Reproducibility Guide

This document describes how to reproduce the GPUCompress neural network training
pipeline from scratch: synthetic data generation, hyperparameter sweep, and final
model evaluation with figures.

All commands are run from the repository root (`/workspaces/GPUCompress`).

---

## Neural Network Architecture

The VOL uses a 5-layer fully-connected neural network:

```
8 → 64 (ReLU) → 64 (ReLU) → 64 (ReLU) → 64 (ReLU) → 8
```

**Inputs (8 features):**
| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | `alg_id` | Integer 0–7 (one of 8 algorithms) |
| 1 | `quant` | Binary (0 or 1) |
| 2 | `shuffle` | Binary (0 or 1) |
| 3 | `error_bound` | Raw float, z-score normalized |
| 4 | `original_size` (bytes) | Raw float, z-score normalized |
| 5 | `entropy` | Shannon entropy in bits |
| 6 | `mad` | Raw mean absolute deviation |
| 7 | `second_derivative` | Raw, not normalized by range |

**Outputs (8 total; GPU inference uses only the first 4):**
| Index | Output | Post-processing |
|-------|--------|-----------------|
| 0 | `comp_time_log` | `expm1(·)` → compression time (ms) |
| 1 | `decomp_time_log` | `expm1(·)` → decompression time (ms) |
| 2 | `ratio_log` | `expm1(·)` → compression ratio |
| 3 | `psnr_clamped` | identity → PSNR (dB) |
| 4–7 | Additional quality metrics | Ignored by GPU inference |

**Parameter count:** 64×8 + 64×64×3 + 8×64 = 512 + 12288 + 512 = 13312 weights + biases.

---

## Prerequisites

Build GPUCompress and the synthetic benchmark:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Install Python dependencies:

```bash
pip install torch torchvision pyyaml xgboost shap matplotlib seaborn scikit-learn pandas numpy
```

---

## Step 1 — Generate Synthetic Training Data (128 K entries)

The synthetic benchmark runs every compression configuration (8 algorithms ×
2 shuffle modes × 2 quantization levels = 32 configs) against randomly generated
float32 chunks and records timing, ratio, and quality metrics for each
(chunk, config) pair.

128 000 rows = **4 000 chunks × 32 configs**:

```bash
build/bench_tests/synthetic_benchmark \
    --num-chunks 4000 \
    --output results/synthetic_results.csv \
    --seed 42
```

The output CSV has one row per (chunk × config) trial with columns:
`chunk_id`, `algorithm`, `shuffle_bytes`, `quantized`, `error_bound`,
`entropy_bits`, `mad`, `second_derivative`, `comp_time_ms`, `decomp_time_ms`,
`compression_ratio`, `psnr_db`, `ssim`, `rmse`, `max_abs_error`, `success`, …

To generate a larger dataset (e.g. 640 K rows = 20 000 chunks):

```bash
build/bench_tests/synthetic_benchmark --num-chunks 20000 --output results/synthetic_20k.csv
```

---

## Step 2 — Hyperparameter Sweep

The sweep runs 5-fold cross-validation over a grid of hidden layer width,
depth, learning rate, and batch size.  It reports mean validation loss and R²
for every combination and saves a ranked CSV summary.

```bash
python3 neural_net/training/hyperparam_study.py \
    --csv results/synthetic_results.csv \
    --folds 5 \
    --epochs 150 \
    --patience 15
```

Output written to `results/hyperparam_study.csv`.

Each row contains:
`hidden_dim`, `num_hidden_layers`, `lr`, `batch_size`,
`val_loss_mean`, `val_loss_std`, `r2_mean`, `r2_std`,
and per-output R² / MAPE columns
(`r2_comp_time`, `r2_decomp_time`, `r2_ratio`, `r2_psnr`, …).

The best configuration identified by the sweep:

```
Best hyperparameters:
  hidden_dim           = 64
  num_hidden_layers    = 4
  lr                   = 3e-4
  batch_size           = 512
```

---

## Step 3 — Cross-Validation on the Best Model (Neural Network)

Run 5-fold CV with the architecture selected in Step 2 to get final
per-output MAE, R², and MAPE numbers for the paper:

```bash
python3 neural_net/training/cross_validate.py \
    --csv results/synthetic_results.csv \
    --folds 5 \
    --hidden-dim 64 \
    --num-hidden-layers 4 \
    --lr 3e-4 \
    --epochs 300
```

Per-fold and summary statistics are printed to stdout.

**Cross-validation results (5-fold):**
| Output | MAPE |
|--------|------|
| comp_time | 12.8% |
| decomp_time | 5.7% |
| ratio | 22.9% |

---

## Step 4 — Export Weights to Binary Format

After training the final model, export to the `.nnwt` binary format:

```bash
python3 neural_net/export/export_weights.py \
    --model neural_net/weights/model.pt \
    --output neural_net/weights/model.nnwt
```

The binary format layout:
```
Header:       magic(4) version(4) n_layers(4) input_dim(4) hidden_dim(4) output_dim(4)
Normalization: x_means(8×4) x_stds(8×4) y_means(8×4) y_stds(8×4)
Layer 1:      W(64×8×4)  b(64×4)
Layer 2:      W(64×64×4) b(64×4)
Layer 3:      W(64×64×4) b(64×4)
Layer 4:      W(64×64×4) b(64×4)
Layer 5:      W(8×64×4)  b(8×4)
Feature bounds: x_mins(8×4) x_maxs(8×4)
```

---

## Step 5 — Loading the Model in the VOL

The model is passed to `gpucompress_init()` before registering the VOL connector:

```cpp
// Initialize with NN weights (enables ALGO_AUTO inference)
gpucompress_init("neural_net/weights/model.nnwt");

// Then register the VOL as usual
hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
```

On startup, `loadNNFromBinary()` in `src/nn/nn_gpu.cu`:
1. Reads and validates the binary header (magic, version, architecture)
2. Copies normalization parameters (`x_means`, `x_stds`, `y_means`, `y_stds`)
3. Reads all 5 layers of weights (W1/b1 through W5/b5)
4. Reads feature bounds for OOD detection (`x_mins`, `x_maxs`)
5. Initializes `log_var` (uncertainty weighting) and `sgd_call_count` to 0
6. Uploads the entire `NNWeightsGPU` struct to device memory

During inference, the 8-input feature vector is constructed with:
- Integer algorithm encoding (`alg_id` as float 0–7)
- Raw `error_bound` and `data_size` (z-score normalized via `x_means`/`x_stds`)

Only outputs 0–3 are used for action selection (comp_time, decomp_time, ratio, psnr).

---

## Summary of Outputs

| Step | Command | Output |
|---|---|---|
| 1 | `synthetic_benchmark` | `results/synthetic_results.csv` |
| 2 | `hyperparam_study.py` | `results/hyperparam_study.csv` |
| 3 | `cross_validate.py` | stdout (MAE / R² / MAPE per fold) |
| 4 | `export_weights.py` | `neural_net/weights/model.nnwt` |
| 5 | `gpucompress_init("model.nnwt")` | VOL loads model at runtime |

---

## VOL Configuration Reference

### VOL Operating Mode

Set `GPUCOMPRESS_VOL_MODE` before opening any HDF5 file.

| Value | Behaviour |
|-------|-----------|
| *(unset)* or `release` | Full NN inference + online SGD. Default for production runs. |
| `bypass` | GPU→CPU passthrough, no compression. Use for baseline I/O throughput measurements. |
| `trace` | Everything in `release` plus exhaustive per-chunk profiling of all 32 algorithm configs. Writes a per-row CSV. ~32× slower. |

```bash
export GPUCOMPRESS_VOL_MODE=release   # default
export GPUCOMPRESS_VOL_MODE=bypass    # baseline
export GPUCOMPRESS_VOL_MODE=trace     # full profiling
```

---

### Environment Variables

All variables are read at `gpucompress_init()` time unless noted otherwise.

#### Core

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUCOMPRESS_DEVICE` | int | `0` | CUDA device index. **Mutually exclusive** with `CUDA_VISIBLE_DEVICES` — if both are set, this is ignored with a warning. |
| `GPUCOMPRESS_WEIGHTS` | path | *(none)* | Path to `.nnwt` weights file. Used by the **H5Z filter** path only; the VOL uses the `gpucompress_init()` argument instead. |

#### Cost Model

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUCOMPRESS_BW_GBPS` | float | `5.0` | Storage bandwidth in GB/s fed into the cost model: `cost = w0·comp_ms + w1·decomp_ms + w2·bytes/(ratio·bw)`. Set to your filesystem's measured write bandwidth. |

#### Online Learning (SGD + Exploration)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUCOMPRESS_MAPE_LOW_THRESH` | float 0–1 | `0.30` | **X1** — SGD trigger. When cost MAPE exceeds this, a proportional SGD gradient step is applied to the NN weights. |
| `GPUCOMPRESS_MAPE_HIGH_THRESH` | float 0–1 | `0.50` | **X2** — Exploration trigger. When cost MAPE exceeds this, the chunk is re-compressed with K alternative configs and the best result is used for the SGD update. |

#### Output Paths

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUCOMPRESS_TIMING_OUTPUT` | path | `gpucompress_io_timing.csv` | Two-column CSV written on every `H5Fclose`: `e2e_ms` (H5Fopen→H5Fclose wall time) and `vol_ms` (sum of all H5Dwrite/H5Dread callback times). |
| `GPUCOMPRESS_TRACE_OUTPUT` | path | `gpucompress_trace.csv` | Per-chunk × per-config profiling CSV. Written only when `GPUCOMPRESS_VOL_MODE=trace`. |
| `GPUCOMPRESS_RESULTS_DIR` | path | *(cwd)* | Directory for benchmark lifetime summary files (written at process exit by the VOL atexit handler). Falls back to `VPIC_RESULTS_DIR` if unset. |

#### Debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPUCOMPRESS_DEBUG_NN` | `1`/`y` | off | Print per-chunk NN inference inputs and outputs to stderr. |
| `GPUCOMPRESS_DETAILED_TIMING` | `1`/`y` | off | Print per-chunk stage timing breakdown (stats, NN, compression, SGD) to stderr. |
| `GPUCOMPRESS_VERBOSE` | any non-`0` | off | Enable verbose logging in the **H5Z filter** path (not the VOL). |

---

### C API Configuration

These can be called at any point after `gpucompress_init()` and take effect on the next `H5Dwrite`.

#### Algorithm Selection

```c
// Switch between NN inference (default) and entropy-based heuristic rules
gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_NN);        // default
gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_HEURISTIC);
```

#### Online Learning

```c
// Master switch
gpucompress_enable_online_learning();   // enable SGD
gpucompress_disable_online_learning();  // disable SGD + exploration

// SGD parameters (X1 threshold, learning rate)
gpucompress_set_reinforcement(
    /*enable (ignored)*/ 1,
    /*learning_rate*/    0.01f,   // default
    /*mape_threshold*/   0.30f,   // X1: default 30%
    /*ct_mape (ignored)*/ 0.0f
);

// Exploration (X2 threshold)
gpucompress_set_exploration(1);                     // enable
gpucompress_set_exploration_threshold(0.50);        // X2: default 50%
gpucompress_set_exploration_k(8);                   // fix K; -1 = dynamic default

// Exhaustive best-config search (ceiling benchmark, ~32× slower, SGD disabled)
gpucompress_set_best_mode(1);
```

#### Cost Model

```c
// Ranking formula: cost = w0*comp_ms + w1*decomp_ms + w2*bytes/(ratio*bw)
gpucompress_set_ranking_weights(1.0f, 1.0f, 1.0f);   // default

// Override auto-detected storage bandwidth
gpucompress_set_bandwidth(3.0f);   // 3 GB/s (e.g. NVMe)

// Reject configs with predicted PSNR below floor (lossy only; 0 = disabled)
gpucompress_set_min_psnr(30.0f);
```

#### NN Weights

```c
// Hot-reload weights without restarting
gpucompress_load_nn("neural_net/weights/model_v2.nnwt");

// Check if NN is active
int loaded = gpucompress_nn_is_loaded();  // 1 = yes, 0 = LZ4 fallback
```

---

### Typical Experiment Configurations

```bash
# Baseline (no compression, pure I/O throughput)
export GPUCOMPRESS_VOL_MODE=bypass
export GPUCOMPRESS_TIMING_OUTPUT=results/bypass_timing.csv

# NN inference only (no online learning)
export GPUCOMPRESS_VOL_MODE=release
export GPUCOMPRESS_TIMING_OUTPUT=results/nn_timing.csv

# NN + SGD online learning
export GPUCOMPRESS_VOL_MODE=release
export GPUCOMPRESS_MAPE_LOW_THRESH=0.30
export GPUCOMPRESS_TIMING_OUTPUT=results/nn_rl_timing.csv

# NN + SGD + exploration
export GPUCOMPRESS_VOL_MODE=release
export GPUCOMPRESS_MAPE_LOW_THRESH=0.30
export GPUCOMPRESS_MAPE_HIGH_THRESH=0.50
export GPUCOMPRESS_TIMING_OUTPUT=results/nn_rl_exp_timing.csv

# Full trace (generates per-chunk CSV for analysis)
export GPUCOMPRESS_VOL_MODE=trace
export GPUCOMPRESS_TRACE_OUTPUT=results/trace.csv
export GPUCOMPRESS_TIMING_OUTPUT=results/trace_timing.csv
```
