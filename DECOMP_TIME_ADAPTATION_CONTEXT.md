# Decomp Time Adaptation — Full Context for Solution Design

## 1. Problem Statement

A 15→128→128→4 MLP predicts compression metrics for GPU-based scientific data compression. The 4 outputs are:

| Output | Name | What It Predicts |
|--------|------|-----------------|
| 0 | comp_time | Compression kernel time (ms) |
| 1 | decomp_time | Decompression kernel time (ms) |
| 2 | ratio | Compression ratio (input_size / compressed_size) |
| 3 | psnr | Peak signal-to-noise ratio (dB) |

Online SGD runs during the **write path** (H5Dwrite) to adapt predictions to the current GPU and data. **Three outputs adapt well; decomp_time does not:**

```
T=0:  sMAPE_R=23.6%  sMAPE_C=80.7%  sMAPE_D=159.4%  SGD=15
T=1:  sMAPE_R=6.0%   sMAPE_C=33.9%  sMAPE_D=149.7%  SGD=2
T=3:  sMAPE_R=5.1%   sMAPE_C=11.0%  sMAPE_D=143.0%  SGD=0
T=10: sMAPE_R=5.9%   sMAPE_C=12.1%  sMAPE_D=145.0%  SGD=0
T=50: sMAPE_R=17.3%  sMAPE_C=18.3%  sMAPE_D=148.0%  SGD=0
T=99: sMAPE_R=40.2%  sMAPE_C=49.5%  sMAPE_D=154.9%  SGD=0
```

**Ratio:** 23.6% → 5.1% by T=3. Adapts fast.
**Comp time:** 80.7% → 11.0% by T=3. Adapts fast.
**Decomp time:** 159% → 143% → 155%. **Never drops below ~140%.** Stuck permanently.

The pretrained model predicts decomp_time ~6-10ms, but actual decomp on this GPU is ~0.4-0.5ms.

---

## 2. Why Decomp Time Cannot Adapt

### The Root Cause: actual_decomp_time = 0 at write time

The SGD kernel receives an `SGDSample` struct per chunk:

```cpp
struct SGDSample {
    int action;
    float actual_ratio;          // ✅ Known at write time
    float actual_comp_time;      // ✅ Known at write time
    float actual_decomp_time;    // ❌ ALWAYS 0.0 at write time
    float actual_psnr;           // ✅ Known at write time
};
```

At write time, compression has just completed — we know the compressed size (ratio), we timed the compression kernel (comp_time), we computed PSNR from quantization. But **decompression hasn't happened yet**, so `actual_decomp_time = 0.0`.

### What the SGD kernel does with decomp_time = 0

In the error computation (inside `nnSGDKernel`):

```cpp
// Output 1: decomp_time
if (sample.actual_decomp_time > 0.0f) {
    float y_std1 = weights->y_stds[1];
    if (y_std1 < 1e-8f) y_std1 = 1e-8f;
    s_d3[1] = s_y[1] - (log1pf(clamped_decomp) - weights->y_means[1]) / y_std1;
} else {
    s_d3[1] = 0.0f;  // ← GRADIENT IS ZEROED
}
```

**When `actual_decomp_time = 0`, the error for output 1 is set to zero.** This means:
- No gradient flows back through W3[1] (decomp output head)
- No gradient flows back through W1/W2 for the decomp output
- The uncertainty weighting sees `raw_mse[1] = 0`, which makes it think decomp is "perfectly predicted"
- The decomp head's weights are frozen at pretrained values

### When is actual decomp time available?

Only during the **read path** (H5Dread), measured in the HDF5 VOL connector:

```cpp
// src/hdf5/H5VLgpucompress.cu, line 1730-1739
struct timespec _ts0, _ts1;
clock_gettime(CLOCK_MONOTONIC, &_ts0);
gpucompress_decompress_gpu(d_compressed, item.comp_sz, dst_ptr, &decomp_size, scatter_stream);
cudaStreamSynchronize(scatter_stream);
clock_gettime(CLOCK_MONOTONIC, &_ts1);
float _decomp_ms = (float)((_ts1.tv_sec - _ts0.tv_sec) * 1000.0
                  + (_ts1.tv_nsec - _ts0.tv_nsec) / 1e6);
gpucompress_record_chunk_decomp_ms((int)ci, _decomp_ms);
```

This stores the measured decomp time in the chunk diagnostic history, but **no SGD runs during the read path** — the value is recorded but never used for learning.

---

## 3. Data Flow Timeline

```
Timestep T:
  WRITE (H5Dwrite):
    For each chunk (64 chunks):
      1. Compute stats (entropy, MAD, deriv) on GPU
      2. NN inference → pick best action (algo+quant+shuffle)
      3. Compress → measure comp_time, compressed_size → compute ratio
      4. Compute PSNR from quantization
      5. Build SGDSample:
         - actual_ratio = input_size / compressed_size  ✅
         - actual_comp_time = measured                  ✅
         - actual_decomp_time = 0.0                     ❌ (not yet known)
         - actual_psnr = computed                       ✅
      6. If cost_error > threshold → run SGD on shared W1/W2 + all W3 heads
         → decomp gradient = 0 because actual_decomp_time = 0
      7. Store chunk diagnostics (predictions, actuals, features)

  READ (H5Dread):
    For each chunk:
      1. Decompress → measure decomp_time ✅
      2. Store decomp_time in chunk_diag_t
      3. (Currently: nothing else — no SGD, no learning)

Timestep T+1:
  WRITE: NN uses same weights (decomp head unchanged)
         → decomp predictions still ~6-10ms vs actual ~0.4ms
         → sMAPE_D stays at ~150%
```

---

## 4. Why Shared-Layer Updates Don't Help Decomp

Even though the main SGD updates W1/W2 (shared layers) for ratio and comp_time, this doesn't meaningfully improve decomp predictions because:

1. **The decomp head (W3[1] + b3[1]) is frozen** — it was trained offline on different hardware with different decomp times
2. **Shared layer changes are optimized for ratio/comp_time** — they might incidentally shift the decomp prediction, but in random directions
3. **The uncertainty weighting treats decomp as "low error"** since `raw_mse[1] = 0`, so decomp gets high precision weight but zero gradient — it amplifies nothing

The ~5% sMAPE_D improvement (159% → 143%) seen in multi-timestep runs is purely coincidental from shared layer drift, not targeted learning.

---

## 5. Complete NN Architecture & SGD Details

### Network
```
Input:  [algo_onehot(8), quant(1), shuffle(1), log10(eb)(1), log2(size)(1), entropy(1), MAD(1), deriv(1)] = 15
Layer1: 15 → 128, ReLU
Layer2: 128 → 128, ReLU
Layer3: 128 → 4 (linear)
Output: [comp_time_normalized, decomp_time_normalized, ratio_normalized, psnr_normalized]
```

### Normalization
Inputs are standardized: `x_std = (x_raw - x_mean) / x_std`

Outputs are in normalized space: `y_normalized = (log1p(actual) - y_mean) / y_std`
(Exception: PSNR uses raw value, not log1p)

To get real predictions: `actual = exp(y_normalized * y_std + y_mean) - 1`

### Weight Layout (NNWeightsGPU on GPU)
```
x_means[15], x_stds[15]           — input normalization
y_means[4], y_stds[4]             — output denormalization
w1[128×15], b1[128]               — layer 1: 2048 params
w2[128×128], b2[128]              — layer 2: 16512 params
w3[4×128], b3[4]                  — layer 3: 516 params (decomp head = w3[128..255] + b3[1])
x_mins[15], x_maxs[15]            — OOD detection bounds
log_var[4]                         — uncertainty weighting (learned)
prev_grad_dot, sgd_call_count     — anti-flip state
Total: 19076 trainable parameters
```

### SGD Kernel (nnSGDKernel)
- **Launch:** `<<<1, 128>>>` — 128 threads, thread t owns hidden neuron t
- **Per-sample loop:** builds input → forward L1 → L2 → L3 → compute errors
- **Error processing:**
  - Noise gate: output 0 (comp_time) only, threshold 0.10
  - Huber clamp: all outputs, ±0.5 in normalized space
  - Cache raw (unclamped) errors for uncertainty weighting
- **Uncertainty weighting (Kendall 2018):**
  - Each output has `log_var[o]` (learned)
  - Precision: `exp(-log_var[o])`, clamped exp argument to [-20, 20]
  - UW update: `grad_lv = 0.5 * (1 - precision * raw_mse)`, LR=0.01
  - `log_var` clamped to [-2.0, 4.0]
  - Clamped errors scaled by `exp(-0.5 * log_var[o])` before backprop
- **Per-output backward (PCGrad):**
  - Each output's gradient through W2→W1 computed independently
  - Gradients normalized to unit vectors
  - Projected to remove conflicts (cosine threshold -0.1)
  - Rescaled by original error magnitude
  - Per-output gradient norm clipped at 0.1
  - Accumulated into region 0
- **Trust-region + EMA:**
  - Gradient normalized to unit vector
  - Step = clamp(0.08 × avg_error, 1e-4, 0.02)
  - Anti-flip: halve step if gradient direction reverses vs EMA
  - EMA: `g_ema = 0.85 * g_ema + 0.15 * g_current`
  - Apply: `W -= step * g_ema`
  - **Weight clamp ±10** on all W1/W2/W3/b (recently added for NaN stability)
  - **NaN guard:** skip update if gradient norm is NaN/Inf

### SGD Trigger (Cost-Based)
```cpp
actual_cost = w0 * comp_time + w1 * decomp_time + w2 * data_size / (ratio * bandwidth)
predicted_cost = w0 * pred_ct + w1 * pred_dt + w2 * data_size / (pred_ratio * bandwidth)
error_pct = |actual_cost - predicted_cost| / actual_cost

if (error_pct > 10%) → fire SGD
```

When `actual_decomp_time = 0`, `act_dt = pred_dt` (fallback to prediction), so decomp contributes equally to both costs and partially cancels in error_pct.

---

## 6. What We Previously Tried for Decomp Adaptation

### Approach: Deferred Head-Only SGD at Read Time

**Idea:** Save NN input features at write time, fire a lightweight SGD kernel during read that updates only W3[1] + b3[1] (129 parameters, no shared layers).

**What worked:**
- On 20-timestep runs with 64 chunks: sMAPE_D dropped from 164% → 12% at T=1, stable 5-15%
- The head-only kernel correctly recomputed the forward pass with current W1/W2 weights
- Log-space error with 1/||h2||² scaling gave stable per-step convergence

**What broke on longer runs (100+ timesteps):**
- At T=22, the **main SGD** (not the deferred one) exploded, destroying all outputs
- Root cause: the deferred SGD changed decomp predictions, which changed the cost function, which changed which chunks fired the main SGD, creating a coupled-optimizer feedback loop
- Even with: W3[1] skip in main SGD, cost decoupling (exclude decomp from error_pct), harmonic LR scaling, weight clamps — the T=22 phase transition still triggered the blowup

**We reverted all decomp changes** and discovered the T=22 blowup was a pre-existing NaN bug in the main SGD kernel (unrelated to our changes). We fixed that with NaN guards + weight clamps + safe exp().

### Current State
- All decomp-specific code is reverted
- The main SGD NaN fix is in place (weight clamp ±10, NaN guard, safe exp)
- The benchmark runs 100 timesteps without blowing up
- Ratio and comp_time adapt normally
- Decomp_time is stuck at ~145% sMAPE permanently

---

## 7. Benchmark Results (Current Baseline, 640^3, 32 chunks, 100 timesteps)

### Adaptation Patterns Over Time

```
T=0:  sMAPE_R=23.6%  sMAPE_C=80.7%  sMAPE_D=159.4%  SGD=15  (pretrained)
T=1:  sMAPE_R=6.0%   sMAPE_C=33.9%  sMAPE_D=149.7%  SGD=2   (fast adaptation)
T=3:  sMAPE_R=5.1%   sMAPE_C=11.0%  sMAPE_D=143.0%  SGD=0   (converged)
T=10: sMAPE_R=5.9%   sMAPE_C=12.1%  sMAPE_D=145.0%  SGD=0
T=22: sMAPE_R=14.9%  sMAPE_C=11.0%  sMAPE_D=140.2%  SGD=2   (phase transition — survived)
T=50: sMAPE_R=17.3%  sMAPE_C=18.3%  sMAPE_D=148.0%  SGD=0
T=99: sMAPE_R=40.2%  sMAPE_C=49.5%  sMAPE_D=154.9%  SGD=0   (drift, no SGD to correct)
```

### Why Ratio/CompTime Adapt But DecompTime Doesn't

| Metric | actual available at write? | SGD gradient | Adapts? |
|--------|--------------------------|-------------|---------|
| ratio | ✅ Yes (compressed_size measured) | ✅ Non-zero | ✅ Yes, fast |
| comp_time | ✅ Yes (kernel timed) | ✅ Non-zero (noise-gated) | ✅ Yes, fast |
| decomp_time | ❌ No (deferred to read) | ❌ Always 0.0 | ❌ Never |
| psnr | ✅ Yes (from quantization) | ✅ Non-zero | ✅ Yes |

### Per-Chunk Decomp Predictions (T=0)
```
Chunk  1: pred_d=9.649ms  actual_d=0.506ms  sMAPE_D=180%
Chunk 13: pred_d=4.508ms  actual_d=0.465ms  sMAPE_D=163%  (shared layer drift)
Chunk 37: pred_d=3.076ms  actual_d=0.457ms  sMAPE_D=148%
Chunk 64: pred_d=3.447ms  actual_d=0.453ms  sMAPE_D=154%
```

The prediction drops from 9.6ms→3.4ms over 64 chunks — but this is the main SGD's shared-layer side-effect (updating W1/W2 for ratio/comp_time), not targeted decomp learning. Actual is ~0.46ms, so even 3.4ms is still 150% off.

---

## 8. Constraints for Any Solution

1. **Must not destabilize the main SGD** — the coupled-optimizer problem killed our first attempt
2. **Must survive 100+ timesteps** including data distribution shifts (Gray-Scott phase transition at T=22)
3. **Actual decomp_time is only measured during H5Dread** — this is a hard architectural constraint of HDF5 (bulk write then bulk read)
4. **Per-chunk diagnostic history exists** — `gpucompress_chunk_diag_t` stores all predictions and actuals per chunk, survives across write→read
5. **g_sgd_stream + g_sgd_done + g_sgd_mutex** synchronization is already in place for the main SGD
6. **Weight clamp ±10 is active** on all weights (recently added NaN fix)
7. **The SGD threshold is 10%** — most chunks don't trigger SGD after T=3 (the NN becomes "good enough" that cost error drops below 10%)
8. **The benchmark resets chunk_history between timesteps** (`gpucompress_reset_chunk_history()` before each write)

---

## 9. Questions for the Solution

1. How should actual decomp_time measurements from the read path be fed back to update the NN?
2. Should it update only the decomp head (W3[1] + b3[1]) or also shared layers?
3. How to prevent the feedback loop where decomp prediction changes → cost function changes → main SGD behavior changes → instability?
4. Should updates be per-chunk (during read) or batched (accumulate across all chunks in a read, apply once)?
5. What learning rate and stabilization approach works for long runs (1000+ timesteps)?
6. Should the main SGD's W3[1] be frozen (only deferred SGD updates it)?
