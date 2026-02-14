# Active Learning System for Compression Algorithm Selection

## 1. The Problem: Static Model in a Dynamic World

The current neural network is **frozen** — trained once on `benchmark_results.csv` (100K synthetic rows) and never updated. It doesn't know what it doesn't know.

| Factor | Training Data | Real World | Impact |
|--------|--------------|------------|--------|
| File size | All 4MB | KB to GB | `data_size` feature was constant — model learned nothing from it |
| Data patterns | Synthetic (uniform, bimodal, etc.) | Arbitrary real signals | May hit distributions never seen |
| GPU timing | One specific GPU, specific load | Different GPUs, varying load | Timing predictions unreliable |
| Compression ratio | Stable (deterministic per algo) | Same | **Generalizes well** |
| PSNR | Deterministic | Same | **Generalizes well** |

The ratio and PSNR predictions should transfer reasonably because the relationship between data statistics and compression ratio is physics-like — it depends on data entropy and algorithm design, not on which GPU runs it. But timing is fundamentally system-dependent.

---

## 2. Key Insight: Every Compression is Free Ground Truth

When the library runs `gpucompress_compress()`, it gets the actual compression ratio, actual time, and actual size as a byproduct. We are currently **throwing away this free supervised signal**.

The active learning system captures it and uses it to improve over time.

---

## 3. How `selected_config` is Decided (NN Inference Detail)

### Step 1: GPU computes data statistics

The existing kernels (`statsPass1Kernel`, entropy kernels, `madPass2Kernel`) run over the raw data and produce three numbers:

```
Input: 4MB of float32 data (1,048,576 elements)

statsPass1Kernel  → sum, min, max, abs_diff_sum
entropyKernels    → byte histogram → Shannon entropy
madPass2Kernel    → sum(|x[i] - mean|)
finalizeStatsOnly → normalize by range

Output:
  entropy    = 5.23        (bits, range 0-8)
  mad_norm   = 0.142       (normalized by data range, 0-1)
  deriv_norm = 0.087       (normalized by data range, 0-1)
```

These three numbers describe **what the data looks like**, independent of which algorithm we'll use.

### Step 2: NN inference kernel launches 32 threads

Each thread IS one configuration:

```
Thread 0:   lz4,      no quant, no shuffle    (action=0)
Thread 1:   snappy,   no quant, no shuffle    (action=1)
Thread 2:   deflate,  no quant, no shuffle    (action=2)
Thread 3:   gdeflate, no quant, no shuffle    (action=3)
Thread 4:   zstd,     no quant, no shuffle    (action=4)
Thread 5:   ans,      no quant, no shuffle    (action=5)
Thread 6:   cascaded, no quant, no shuffle    (action=6)
Thread 7:   bitcomp,  no quant, no shuffle    (action=7)
Thread 8:   lz4,      quant,    no shuffle    (action=8)
Thread 9:   snappy,   quant,    no shuffle    (action=9)
...
Thread 16:  lz4,      no quant, shuffle       (action=16)
...
Thread 24:  lz4,      quant,    shuffle       (action=24)
...
Thread 31:  bitcomp,  quant,    shuffle       (action=31)
```

Encoding: `action = algo_idx + quant*8 + shuffle*16`

### Step 3: Each thread builds its own 15-feature input vector

Every thread shares the same data statistics but has different configuration features:

```
                            Thread 4               Thread 20
                            (zstd, no_q, no_s)     (zstd, no_q, shuf)
                            ───────────────────    ───────────────────
Feature 0:  alg_lz4         0.0                    0.0
Feature 1:  alg_snappy      0.0                    0.0
Feature 2:  alg_deflate     0.0                    0.0
Feature 3:  alg_gdeflate    0.0                    0.0
Feature 4:  alg_zstd        1.0  ◄── one-hot      1.0  ◄── same algo
Feature 5:  alg_ans         0.0                    0.0
Feature 6:  alg_cascaded    0.0                    0.0
Feature 7:  alg_bitcomp     0.0                    0.0
Feature 8:  quant           0.0                    0.0
Feature 9:  shuffle         0.0  ◄── different     1.0  ◄── different
Feature 10: error_bound     -6.0 (log10(1e-7))     -6.0
Feature 11: data_size       22.0 (log2(4MB))       22.0
Feature 12: entropy         5.23                   5.23  ◄── same stats
Feature 13: mad             0.142                  0.142 ◄── same stats
Feature 14: deriv           0.087                  0.087 ◄── same stats
```

Features 10-14 are then standardized using stored training means/stds:
```c
input[i] = (input_raw[i] - weights->x_means[i]) / weights->x_stds[i];
```

### Step 4: Each thread runs the full forward pass

```
Layer 1:  15 inputs × 128 neurons + bias → ReLU → 128 values
Layer 2:  128 inputs × 128 neurons + bias → ReLU → 128 values
Layer 3:  128 inputs × 4 neurons + bias → 4 output values
```

Each thread does `15×128 + 128×128 + 128×4 = 18,816` multiply-adds. Tiny for a GPU thread.

### Step 5: Each thread decodes its predicted metrics

The NN outputs are in normalized/log space. Each thread converts back:

```c
// De-normalize
output_raw[j] = output_norm[j] * weights->y_stds[j] + weights->y_means[j];

// Convert from log-space to original scale
comp_time   = expm1f(output_raw[0]);   // predicted compression time (ms)
decomp_time = expm1f(output_raw[1]);   // predicted decompression time (ms)
ratio       = expm1f(output_raw[2]);   // predicted compression ratio
psnr        = output_raw[3];           // predicted PSNR (dB)
```

Each thread now holds its predicted metrics:

```
Thread 4  (zstd, no_q, no_s):    ratio=31.2x  time=24ms  psnr=120
Thread 20 (zstd, no_q, shuf):    ratio=35.8x  time=28ms  psnr=120
Thread 7  (bitcomp, no_q, no_s): ratio=12.1x  time=15ms  psnr=120
Thread 2  (deflate, no_q, no_s): ratio=29.4x  time=38ms  psnr=120
...all 32 threads have their predictions...
```

### Step 6: Parallel reduction picks the winner

Ranking by `compression_ratio` (higher is better):

```c
__shared__ float s_vals[32];   // Thread 4 writes 31.2, Thread 20 writes 35.8, ...
__shared__ int   s_idxs[32];   // Thread 4 writes 4, Thread 20 writes 20, ...

// Tree reduction: 32 → 16 → 8 → 4 → 2 → 1
// At each step, keep the higher value and its index

// Thread 0 has the winner
*out_action = s_idxs[0];  // → 20 (zstd + shuffle, ratio=35.8x)
```

### Step 7: Action decoded to config

```c
int action = 20;

decoded.algorithm         = 20 % 8      = 4   → zstd
decoded.use_quantization  = (20 / 8) % 2 = 0  → no quant
decoded.shuffle_size      = (20 / 16) % 2 = 1 → 4-byte shuffle
```

**This is `selected_config`**: the configuration whose thread predicted the highest compression ratio.

---

## 4. Reinforcement: Three Levels

### Level 1: Passive Collection (every call, zero cost)

After the library compresses with the selected config, we **already know** the actual result. We just need to store it.

```
Compress call:
  Input stats:  entropy=5.23, mad=0.142, deriv=0.087, size=4MB, error_bound=0
  NN selected:  zstd + shuffle (action=20)
  NN predicted: ratio=35.8x, time=28ms
  Actual:       ratio=33.1x, time=26ms

Store: (entropy=5.23, mad=0.142, deriv=0.087, size=4MB,
        error_bound=0, config=zstd+shuffle,
        actual_ratio=33.1, actual_time=26ms)
```

That stored row is one new training sample collected for free.

**Level 1 limitation:** We only learn about the config we selected. We know zstd+shuffle gave 33.1x but we do NOT know what deflate or lz4 would have given. We can correct the model's prediction accuracy for zstd+shuffle on this data type, but we can't fix the ranking if it was wrong. That's what Level 2 is for.

### Level 2: Active Exploration (triggered by high error)

Level 1 gave us **one** training sample for free. Level 2 gives us **many** — but at a cost.

#### When does Level 2 trigger?

After the NN-selected config finishes compressing, we compare:

```
NN predicted ratio for zstd+shuffle:  35.8x
Actual ratio after compression:       22.4x

Error = |35.8 - 22.4| / 22.4 = 59.8%  →  ABOVE 20% threshold

The model was significantly wrong. Two things are now true:
  1. The prediction for this config was bad
  2. We have NO idea if this was even the right config to pick
     (maybe deflate would have given 40x and the model predicted 18x)
```

We know the model is unreliable for this data. But we only know it was wrong about **one** config. We don't know the full picture. That's the problem Level 2 solves.

#### Running alternative configs

The data is still on the GPU from the original compression call. We exploit this — no need to re-upload.

```
Step 1: NN already predicted all 32 configs. We have the ranked list:

  Rank  Config               NN Predicted Ratio
  ─────────────────────────────────────────────
  #1    zstd+shuffle         35.8x    ◄── we ran this, actual was 22.4x (WRONG)
  #2    deflate+shuffle      31.2x
  #3    gdeflate+shuffle     29.7x
  #4    bitcomp              27.1x
  #5    zstd                 25.9x
  ...
  #32   snappy               8.2x

Step 2: Run the next top-K alternatives (say K=4):

  Config               NN Predicted    Actual (measured now)
  ─────────────────────────────────────────────────────────
  zstd+shuffle         35.8x           22.4x    (already done)
  deflate+shuffle      31.2x           38.7x    ◄── actually BETTER
  gdeflate+shuffle     29.7x           35.1x    ◄── also better
  bitcomp              27.1x           15.9x
  zstd                 25.9x           19.8x
```

Now we have 5 actual measurements instead of 1.

#### Two types of mistakes revealed

```
The model made two mistakes:

1. PREDICTION ERROR:  zstd+shuffle predicted 35.8x, actual 22.4x
                      deflate+shuffle predicted 31.2x, actual 38.7x
                      (the model was wrong about individual values)

2. RANKING ERROR:     Model ranked zstd+shuffle #1
                      Actually deflate+shuffle was best
                      (the model picked the wrong config entirely)
```

These are different problems. Level 1 (passive) can only catch prediction errors for the selected config. Level 2 catches **ranking errors** by actually testing alternatives.

#### The training samples collected

Each measurement becomes a row in the experience buffer:

```
entropy  mad    deriv  size  err_bound  algo      quant  shuffle  actual_ratio  actual_time
─────────────────────────────────────────────────────────────────────────────────────────────
5.23     0.142  0.087  4MB   0          zstd      0      1        22.4          26ms
5.23     0.142  0.087  4MB   0          deflate   0      1        38.7          42ms
5.23     0.142  0.087  4MB   0          gdeflate  0      1        35.1          38ms
5.23     0.142  0.087  4MB   0          bitcomp   0      0        15.9          18ms
5.23     0.142  0.087  4MB   0          zstd      0      0        19.8          21ms
```

Same data stats, different configs, different actual results. This is extremely valuable training data because it shows the model **how configs compare to each other** for this specific data profile.

#### Should we return the actual best or the predicted best?

Two options:

```
Option A: Return the originally selected result (zstd+shuffle, 22.4x)
  - Simpler, no wasted work
  - But we KNOW a better option exists (deflate+shuffle, 38.7x)
  - The user gets suboptimal compression

Option B: Return the best result from exploration (deflate+shuffle, 38.7x)
  - User gets optimal compression
  - But we ran 5 compressions instead of 1 (5x slower this call)
  - The extra compression results are thrown away (only the best is kept)
```

**Option B is better** — if we already paid the cost of exploration, we should give the user the best result we found. The extra latency is the cost of the model being wrong, and it only happens when the model IS wrong.

```
Caller's perspective:

  Compress call #1:   Model confident → returns in 25ms (normal)
  Compress call #2:   Model confident → returns in 23ms (normal)
  Compress call #3:   Model wrong → explores → returns in 120ms (slower, but better result)
  Compress call #4:   Model confident → returns in 24ms (normal)
  ...
  After retrain:      Model now correct for this data type → always 25ms
```

#### Choosing K: how many alternatives to explore

```
K=0:   No exploration at all (Level 1 only)
       Cheapest, but we only learn about one config

K=4:   Run top-5 total (1 original + 4 alternatives)
       Good balance: covers the configs the model thought were best
       If the ranking is off by a few positions, we'll catch it
       Cost: 5x one compression

K=31:  Run ALL 32 configs
       Maximum information: we get the complete ground truth
       Can't miss the actual best
       Cost: 32x one compression
       When to use: features are completely out of distribution (OOD)
```

The decision adapts to how wrong the model was:

```
if features_out_of_distribution:
    K = 31          # Full exploration — we know nothing about this data
elif prediction_error > 50%:
    K = 9           # Heavy exploration — model was very wrong
elif prediction_error > 20%:
    K = 4           # Light exploration — model was somewhat wrong
else:
    K = 0           # No exploration — model was accurate enough
```

#### Experience buffer growth over time

```
                    Passive (Level 1)                    Active (Level 2)
                    1 sample per call                    5-32 samples per trigger
                    ─────────────────                    ───────────────────────

Call 1:             [ent=5.2, zstd+s, ratio=22.4]
Call 2:             [ent=3.1, lz4, ratio=45.2]
Call 3 (trigger!):                                       [ent=5.2, zstd+s, ratio=22.4]
                                                         [ent=5.2, deflate+s, ratio=38.7]
                                                         [ent=5.2, gdeflate+s, ratio=35.1]
                                                         [ent=5.2, bitcomp, ratio=15.9]
                                                         [ent=5.2, zstd, ratio=19.8]
Call 4:             [ent=3.0, lz4, ratio=44.8]
Call 5:             [ent=7.1, bitcomp+s, ratio=1.4]
...

After 1000 calls:
  Level 1 samples: ~1000 (one per call, single config each)
  Level 2 samples: ~200  (from ~40 trigger events, ~5 configs each)
  Total buffer:    ~1200 new training samples
```

Level 2 samples are **much more valuable** than Level 1 samples because they show relative performance across configs for the same data. Level 1 only tells you about one config in isolation.

#### The full Level 2 flow in pseudocode

```python
def compress_with_active_learning(data, config):

    # 1. Compute data stats (existing GPU kernels)
    stats = compute_stats_gpu(data)  # entropy, mad, deriv

    # 2. Check if features are in distribution
    if is_out_of_distribution(stats):
        # Full exploration — run all 32 configs
        results = {}
        for config_id in range(32):
            results[config_id] = actually_compress(data, config_id)

        best = pick_best(results, criterion="ratio")
        experience_buffer.store_all(stats, results)
        return best.compressed_data

    # 3. NN predicts all 32 configs
    predictions = nn_predict_all_32(stats, data_size, error_bound)
    best_config = predictions.rank_by("ratio")[0]

    # 4. Compress with predicted best
    result = actually_compress(data, best_config)

    # 5. Check prediction accuracy
    error = abs(predictions[best_config].ratio - result.actual_ratio) / result.actual_ratio

    if error > THRESHOLD:
        # LEVEL 2: Explore alternatives
        all_results = {best_config: result}

        for alt_config in predictions.rank_by("ratio")[1:5]:  # top 5
            alt_result = actually_compress(data, alt_config)
            all_results[alt_config] = alt_result

        # Store all results as training data
        experience_buffer.store_all(stats, all_results)

        # Return the ACTUAL best, not the predicted best
        actual_best = pick_best(all_results, criterion="ratio")
        return actual_best.compressed_data

    else:
        # LEVEL 1: Just store passively
        experience_buffer.store_one(stats, best_config, result)
        return result.compressed_data
```

#### Cost summary

```
                        Compression calls    Extra cost per trigger
                        ─────────────────    ─────────────────────
Level 1 (passive):      1 (normal)           0 (free)
Level 2 (light, K=4):   5 total              4x extra
Level 2 (heavy, K=9):   10 total             9x extra
Level 2 (full, K=31):   32 total             31x extra

Trigger frequency (depends on model quality):
  New workload:         ~50% of calls trigger    → high overhead, fast learning
  After 500 samples:    ~10% of calls trigger    → moderate overhead
  After retrain:        ~2% of calls trigger     → near-zero overhead
  Mature model:         ~0.1% of calls trigger   → negligible
```

The model pays upfront to learn, then amortizes that cost over thousands of future calls where it predicts correctly.

### Level 3: Batch Retrain (periodic)

When the experience buffer accumulates enough new samples:

```
Experience buffer reaches 500 new samples
  → Retrain model: original 100K rows + 500 new rows
  → Export new weights/model.nnwt
  → Hot-reload into GPU memory
  → Model now knows about the new workloads
```

The model is 19K parameters — retraining takes seconds, not hours.

---

## 5. How Do We Know When to Reinforce?

Two detection moments:

### Moment A: BEFORE compression (proactive — out-of-distribution detection)

Check if input features fall within the range seen during training:

```
Training data saw:  entropy ∈ [0.8, 7.9], mad ∈ [0.0001, 0.48], deriv ∈ [0.00001, 0.45]

New data arrives:   entropy = 4.2, mad = 0.62, deriv = 0.03
                                       ^^^^
                              MAD is OUTSIDE training range
                              → Model is extrapolating → don't trust it
                              → Skip NN, go straight to full exploration
```

Feature bounds can be stored in the `.nnwt` file and checked in the CUDA kernel.

### Moment B: AFTER compression (reactive — threshold check)

Compare prediction to actual result:

```c
float predicted_ratio = nn_predicted_ratio_for_selected_config;
float actual_ratio = (float)original_size / (float)compressed_size;

float error = fabsf(predicted_ratio - actual_ratio) / actual_ratio;

if (error > REINFORCEMENT_THRESHOLD) {
    // Model was wrong — trigger Level 2 exploration
    trigger_exploration(data_stats, actual_ratio);
}
```

### Threshold Selection

| Threshold | Exploration Rate | Learning Speed | Overhead |
|-----------|-----------------|----------------|----------|
| 5% | High | Fast | High initially |
| 20% | Moderate | Moderate | Balanced |
| 50% | Low | Slow | Low |

Practical starting point: **20% MAPE for ratio, 30% for timing**.

The threshold should also **adapt**: when the model is new, use a tighter threshold (explore more). As accuracy improves, relax the threshold (explore less).

### Combined Decision Flow

```
if features_out_of_distribution(data_stats):
    → Full exploration (Level 2): run all 32 configs
    → Reason: we KNOW the model can't handle this

elif prediction_error > threshold (checked after compression):
    → Partial exploration: run top-5 alternative configs
    → Reason: model tried but was significantly wrong

else:
    → No exploration: trust the model
    → Store (stats, config, actual) passively (Level 1)
```

---

## 6. What If the Model Is Initially Bad for a New Workload?

This is the **cold start problem**. A completely new workload (e.g., genomic data) that looks nothing like the synthetic training data.

### What Happens With Active Learning

```
Chunk 1:     Features OOD → full exploration (32 configs) → ~15ms extra
Chunk 2:     Features still OOD → full exploration → ~15ms extra
...
Chunk 50:    50 × 32 = 1,600 new training samples collected
             Trigger retrain → model now has real data for this workload
Chunk 51:    Features now in-distribution → NN predicts → error within threshold
             → No exploration needed
...
Chunk 1000:  Model runs at full speed, near-zero overhead
```

### Amortized Cost Analysis

Say full exploration (running all 32 configs) costs 32x the compression time:

```
Cost of 50 exploration rounds:    50 × 32 × T  = 1,600T
Cost of 950 exploitation rounds:  950 × 1 × T  = 950T
Total for 1000 chunks:            2,550T

Compare:
  Naive (always LZ4):              1,000T  (but suboptimal compression)
  Full benchmark every time:       32,000T (optimal but way too slow)
  Active learning:                 2,550T  (optimal after warmup)

Active learning overhead for first 1000 chunks: 2.55x
After warmup: ~1.0x (model is accurate, no exploration)
```

But the model-selected config compresses BETTER than naive LZ4, so actual throughput exceeds the naive approach even during warmup.

### Workload Adaptation Over Time

```
Day 1:  Model knows synthetic data only
        New workload (genomic data) → predictions are bad
        High error rate → triggers heavy exploration

Day 1 (after ~200 chunks): Model retrained with genomic experiences
        Predictions for genomic data improve dramatically
        Exploration rate drops

Day 7:  Another new workload (climate simulation)
        Predictions bad for climate data, good for genomic
        Exploration triggered again for climate

Day 7 (after ~200 chunks): Model knows synthetic + genomic + climate
        Good predictions across all three
```

The model **specializes to the user's actual workload mix**. A genomics lab's model becomes expert at genomic data. A climate research group's model becomes expert at simulation outputs.

---

## 7. Handling the Silent Ranking Error

There's a subtle failure mode: the model picks a config that seems reasonable (prediction error is low) but a different config would have been significantly better.

Example:
```
Model predicts:  zstd+shuffle → ratio=25x    (actual: 22x, error=12% → below threshold)
Reality:         deflate+shuffle → ratio=35x  (model predicted 20x, never checked)

The model picked the wrong config but we didn't catch it because
the per-config prediction error was within threshold.
```

Solutions:

**Periodic random exploration:** With probability epsilon (say 5%), ignore the model and run a random config. If the random config significantly outperforms the predicted best, flag a ranking error.

**Rotating top-K verification:** Every Nth call, run the top-3 predicted configs instead of just the top-1. If #2 or #3 actually beats #1, log a ranking error.

**Decay trust:** Track a running accuracy score. If the model hasn't been verified in a while (N compressions without any exploration), force a verification round.

---

## 8. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     INFERENCE PATH                           │
│                                                              │
│  Data → Compute Stats → OOD Check ──→ Features known?       │
│                              │              │                │
│                              NO            YES               │
│                              │              │                │
│                    Full Exploration    NN Predict             │
│                    (32 configs)      (select best)           │
│                              │              │                │
│                              │         Compress              │
│                              │              │                │
│                              │     Check Prediction Error    │
│                              │         │           │         │
│                              │     Error > θ    Error ≤ θ    │
│                              │         │           │         │
│                              │    Partial         Store      │
│                              │    Explore        passively   │
│                              │    (top-5)              │     │
│                              │         │               │     │
│                              └────┬────┘               │     │
│                                   │                    │     │
│                          Experience Buffer              │     │
│                                   │                    │     │
│                          Buffer full?                  │     │
│                            │       │                   │     │
│                           YES     NO                   │     │
│                            │       │                   │     │
│                        Retrain   Wait                  │     │
│                            │                           │     │
│                     New .nnwt                          │     │
│                     Hot-reload                         │     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Summary: Static vs Active Learning

| Question | Static Model | Active Learning |
|----------|-------------|----------------|
| How does it improve? | It doesn't | Every compression is training data; retrain periodically |
| How does it generalize? | Must hope synthetic data was representative | Adapts to whatever workloads it actually sees |
| When to reinforce? | Never | When prediction error > threshold OR features are OOD |
| What if initially bad? | Stays bad forever | Explores heavily at first, learns within ~50-200 samples, then exploits |
| Cost model | Zero overhead, fixed accuracy | Small overhead during warmup, improving accuracy over time |
| Long-term behavior | Same accuracy forever | Converges to near-perfect accuracy for user's workload |

The threshold-based approach is the right engineering answer — it's simple to implement, has a clear cost model, and degrades gracefully. When confident, the system is fast (just NN inference). When uncertain, it pays the cost of exploration but gains knowledge. Over time, it converges to near-zero exploration overhead.

---

## 10. Why Regression Makes This Possible

The regression model (predict metrics) is critical for active learning. With a classification model (predict best action), we'd only know "right" or "wrong" — not by how much. With regression:

- We can compute prediction error after the fact (`|predicted_ratio - actual_ratio|`)
- We can estimate confidence before the fact (how spread out are the top-K predictions?)
- We can retrain on continuous targets (actual_ratio=33.1) rather than binary labels (correct/incorrect)
- We can change the ranking criterion without retraining (sort by ratio today, by time tomorrow)

This is also why `selected_config` comes from running all 32 forward passes and ranking, rather than a single classification output. The regression approach is **strictly more powerful**: you can always derive the classification answer from it (rank then pick #1), but you can't go the other way.

---

## 11. Complete Architecture: From Empty State to Fully Trained NN

### Phase 0: No Model Exists

```
┌─────────────────────────────────────────────────────┐
│                    EMPTY STATE                       │
│                                                      │
│  GPUCompress library exists with 8 algorithms        │
│  No model, no Q-table, no intelligence               │
│  ALGO_AUTO defaults to LZ4 every time                │
│                                                      │
│  Question: Which algorithm is best for which data?   │
│  Answer:   We don't know yet.                        │
└─────────────────────────────────────────────────────┘
```

### Phase 1: Generate Benchmark Data

We need ground truth. Run every algorithm on every data type and measure what happens.

There are two ways to generate and consume benchmark data:

**Option A: Direct from binary files (recommended)**

The training script benchmarks all configs on-the-fly via `libgpucompress.so`, eliminating the CSV intermediary:

```bash
python neural_net/train.py \
    --data-dir syntheticGeneration/training_data/ \
    --lib-path build/libgpucompress.so
```

This uses `binary_data.py` which calls `gpucompress_compute_stats()` (GPU) for stats and `gpucompress_compress()`/`gpucompress_decompress()` for benchmarking, all through `gpucompress_ctypes.py`.

**Option B: Pre-generated CSV (legacy)**

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION                                   │
│                                                                       │
│  Data Generator (scripts/generate_data.py)                           │
│  ├── float32_uniform_w12_p0_const.bin                                │
│  ├── float32_bimodal_w8_p100_rand.bin                                │
│  ├── float32_normal_w16_p325_rand.bin                                │
│  └── ... (1,568 synthetic files, different distributions)            │
│           │                                                           │
│           ▼                                                           │
│  Benchmark Runner (scripts/benchmark.cpp)                            │
│  For EACH file:                                                       │
│    For EACH of 8 algorithms:                                         │
│      For EACH of 2 quantization options:                             │
│        For EACH of 2 shuffle options:                                │
│          For EACH of 4 error bounds:                                 │
│            → Actually compress                                        │
│            → Measure: ratio, comp_time, decomp_time, psnr            │
│            → Record: entropy, mad, first_derivative                  │
│            → Write one row to CSV                                    │
│                                                                       │
│  Output: benchmark_results.csv                                       │
│          1,568 files × 64 configs = 100,352 rows                    │
│          26 columns per row                                          │
│                                                                       │
│  Each row says: "For THIS data with THESE stats,                     │
│                  using THIS algorithm with THIS config,               │
│                  the result was THIS ratio, THIS time, THIS psnr"     │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Train the Neural Network

Turn the benchmark data into a model that can predict outcomes without running compression.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                                │
│                                                                       │
│  Input (one of):                                                     │
│    Option A: .bin files → binary_data.py (on-the-fly GPU benchmark)  │
│    Option B: benchmark_results.csv → data.py (legacy CSV loader)     │
│           │                                                           │
│           ▼                                                           │
│  Data Loading & Preparation (data.py or binary_data.py)                                             │
│  ├── Filter failures (100,352 → 100,225 rows)                       │
│  ├── Encode inputs:                                                  │
│  │   ├── algorithm → one-hot (8 features)                            │
│  │   ├── quantization → binary (1 feature)                           │
│  │   ├── shuffle → binary (1 feature)                                │
│  │   ├── error_bound → log10 (1 feature)                             │
│  │   ├── data_size → log2 (1 feature)                                │
│  │   └── entropy, mad, deriv → raw (3 features)                     │
│  │   Total: 15 input features                                        │
│  ├── Encode outputs:                                                 │
│  │   ├── comp_time → log1p                                           │
│  │   ├── decomp_time → log1p                                        │
│  │   ├── ratio → log1p                                               │
│  │   └── psnr → clamp inf to 120                                    │
│  │   Total: 4 output targets                                        │
│  ├── Split by FILE (not by row, avoids data leakage):               │
│  │   ├── Train: 1,254 files → 80,156 rows (80%)                    │
│  │   └── Val:   314 files   → 20,069 rows (20%)                    │
│  ├── Standardize continuous features (mean=0, std=1):               │
│  │   Compute x_means, x_stds from training set only                 │
│  │   Apply to both train and val                                     │
│  └── Standardize outputs:                                            │
│      Compute y_means, y_stds from training set only                  │
│           │                                                           │
│           ▼                                                           │
│  model.py: CompressionPredictor                                      │
│  ┌─────────────────────────────────┐                                 │
│  │  Input [15 features]            │                                 │
│  │       │                         │                                 │
│  │  Linear(15, 128) + ReLU         │  2,048 parameters               │
│  │       │                         │                                 │
│  │  Linear(128, 128) + ReLU        │  16,512 parameters              │
│  │       │                         │                                 │
│  │  Linear(128, 4)                 │  516 parameters                 │
│  │       │                         │                                 │
│  │  Output [4 predictions]         │  Total: 19,076 parameters       │
│  └─────────────────────────────────┘                                 │
│           │                                                           │
│           ▼                                                           │
│  train.py: Training Loop                                             │
│  ├── Optimizer: Adam (lr=1e-3)                                       │
│  ├── Loss: MSE on standardized outputs                               │
│  ├── Scheduler: ReduceLROnPlateau                                    │
│  ├── Early stopping: patience=20 epochs                              │
│  ├── Batch size: 512                                                 │
│  ├── Max epochs: 200                                                 │
│  └── Best model saved at epoch 143                                   │
│           │                                                           │
│           ▼                                                           │
│  Output: weights/model.pt                                            │
│          Contains: model weights + x_means + x_stds + y_means + y_stds│
│                                                                       │
│  Validation results:                                                 │
│  ┌─────────────────────────────────────────────────┐                 │
│  │  Output              R²      MAE     MAPE       │                 │
│  │  ─────────────────────────────────────────────── │                 │
│  │  Compression Ratio   0.877   47.5    19.6%      │                 │
│  │  PSNR                0.999   0.72    1.7%       │                 │
│  │  Compression Time    0.10    -       34.8%      │                 │
│  │  Decompression Time  0.07    -       38.6%      │                 │
│  └─────────────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Evaluate Ranking Quality

The model predicts metrics, but the real question is: does it pick the right config?

```
┌──────────────────────────────────────────────────────────────────────┐
│                     RANKING EVALUATION                                │
│                                                                       │
│  evaluate.py                                                         │
│  For each (file, error_bound) group in validation set:               │
│    1. Get all configs for this group (up to 32 rows)                 │
│    2. Run all configs through model → predicted metrics              │
│    3. Rank by predicted compression_ratio (descending)               │
│    4. Compare predicted rank #1 to actual rank #1                    │
│                                                                       │
│  Results (1,256 validation groups):                                  │
│  ┌──────────────────────────────────────────────────────┐            │
│  │  Criterion           Top-1    Top-3    Zero-Regret   │            │
│  │  ──────────────────────────────────────────────────── │            │
│  │  Compression Ratio   75.1%    87.7%    75.2%         │            │
│  │  Compression Time    23.9%    61.4%    23.9%         │            │
│  │  PSNR                5.5%     13.5%    100.0%        │            │
│  └──────────────────────────────────────────────────────┘            │
│                                                                       │
│  Reading: For compression ratio ranking, the model picks the         │
│  actual best config 75% of the time. The actual best is in the       │
│  model's top 3 nearly 88% of the time.                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Export to CUDA and Deploy

Convert the Python model into a GPU-native binary that runs inside the library.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     EXPORT & DEPLOYMENT                               │
│                                                                       │
│  export_weights.py                                                   │
│  weights/model.pt (PyTorch)                                          │
│           │                                                           │
│           ▼                                                           │
│  weights/model.nnwt (binary, 76 KB)                                  │
│  ┌──────────────────────────────────────────┐                        │
│  │  Header (24 bytes):                      │                        │
│  │    magic=0x4E4E5754, version=1           │                        │
│  │    layers=3, in=15, hidden=128, out=4    │                        │
│  │  Normalization (152 bytes):              │                        │
│  │    x_means[15], x_stds[15]              │                        │
│  │    y_means[4], y_stds[4]                │                        │
│  │  Layer 1 (8,192 bytes):                  │                        │
│  │    W1[128×15], b1[128]                  │                        │
│  │  Layer 2 (66,048 bytes):                 │                        │
│  │    W2[128×128], b2[128]                 │                        │
│  │  Layer 3 (2,064 bytes):                  │                        │
│  │    W3[4×128], b3[4]                     │                        │
│  └──────────────────────────────────────────┘                        │
│           │                                                           │
│           ▼                                                           │
│  gpucompress_init("model.nnwt")                                      │
│           │                                                           │
│           ▼                                                           │
│  nn_gpu.cu: loadNNFromBinary()                                       │
│  ├── Read binary file                                                │
│  ├── Validate header (magic, version, dimensions)                    │
│  ├── cudaMalloc(sizeof(NNWeightsGPU)) → 76 KB on GPU                │
│  ├── cudaMemcpy(host → device)                                       │
│  └── g_nn_loaded = true                                              │
│           │                                                           │
│           ▼                                                           │
│  Model is now live on GPU, ready for inference                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Inference (Single Compression Call)

What happens when a user calls `gpucompress_compress()` with `ALGO_AUTO`:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SINGLE INFERENCE CALL                             │
│                                                                       │
│  User calls: gpucompress_compress(data, 4MB, ..., ALGO_AUTO)         │
│           │                                                           │
│           ▼                                                           │
│  cudaMemcpy: data → GPU (d_input)                                    │
│           │                                                           │
│           ▼                                                           │
│  ┌─── GPU Stats Pipeline ────────────────────────────────────┐       │
│  │                                                            │       │
│  │  statsPass1Kernel ──→ sum, min, max, abs_diff_sum          │       │
│  │  entropyKernels   ──→ histogram → Shannon entropy          │       │
│  │  madPass2Kernel   ──→ sum(|x - mean|)                      │       │
│  │  finalizeStatsOnly ──→ mad_norm, deriv_norm                │       │
│  │                                                            │       │
│  │  Output: entropy=5.23, mad=0.142, deriv=0.087             │       │
│  └────────────────────────────────────────────────────────────┘       │
│           │                                                           │
│           ▼                                                           │
│  ┌─── NN Inference Kernel (32 threads) ──────────────────────┐       │
│  │                                                            │       │
│  │  Thread 0 (lz4,no_q,no_s):                                │       │
│  │    Build input → forward pass → ratio=12.1x               │       │
│  │                                                            │       │
│  │  Thread 4 (zstd,no_q,no_s):                               │       │
│  │    Build input → forward pass → ratio=31.2x               │       │
│  │                                                            │       │
│  │  Thread 20 (zstd,no_q,shuf):                              │       │
│  │    Build input → forward pass → ratio=35.8x  ◄── WINNER   │       │
│  │                                                            │       │
│  │  ... (all 32 threads run in parallel) ...                  │       │
│  │                                                            │       │
│  │  Parallel reduction: max(all ratios) → Thread 20 wins     │       │
│  │  Output: action = 20                                       │       │
│  └────────────────────────────────────────────────────────────┘       │
│           │                                                           │
│           ▼                                                           │
│  Decode action 20:                                                   │
│    algorithm = zstd, quant = no, shuffle = 4-byte                    │
│           │                                                           │
│           ▼                                                           │
│  Apply shuffle → Run zstd compression → Get result                   │
│    actual_ratio = 33.1x, actual_time = 26ms                          │
│           │                                                           │
│           ▼                                                           │
│  Return compressed data to user                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Active Learning Loop (Continuous Improvement)

The model improves itself through deployment:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ACTIVE LEARNING LIFECYCLE                         │
│                                                                       │
│                         ┌─────────────┐                              │
│                         │  Compress    │                              │
│                         │  Call        │                              │
│                         └──────┬──────┘                              │
│                                │                                      │
│                    ┌───────────┴───────────┐                         │
│                    │   OOD Check           │                         │
│                    │   Features in range?  │                         │
│                    └───┬─────────────┬─────┘                         │
│                        │             │                                │
│                       NO            YES                               │
│                        │             │                                │
│              ┌─────────┴──────┐  ┌──┴───────────────┐               │
│              │ Full explore   │  │ NN predicts best  │               │
│              │ (32 configs)   │  │ Compress with it  │               │
│              │ Pick actual    │  └──┬───────────────┘               │
│              │ best result    │     │                                 │
│              └─────────┬──────┘     │                                │
│                        │      ┌─────┴──────────────┐                │
│                        │      │ Check prediction   │                │
│                        │      │ error vs actual     │                │
│                        │      └──┬──────────┬──────┘                │
│                        │         │          │                        │
│                        │    Error > θ    Error ≤ θ                   │
│                        │         │          │                        │
│                        │   ┌─────┴────┐  ┌──┴────────────┐         │
│                        │   │ Explore   │  │ Level 1:      │         │
│                        │   │ top-K     │  │ Store 1       │         │
│                        │   │ configs   │  │ sample        │         │
│                        │   │ Return    │  │ passively     │         │
│                        │   │ actual    │  └──┬────────────┘         │
│                        │   │ best      │     │                       │
│                        │   └─────┬────┘     │                       │
│                        │         │          │                        │
│                        └────┬────┘          │                        │
│                             │               │                        │
│                   ┌─────────┴───────┐       │                        │
│                   │ Experience       │◄──────┘                       │
│                   │ Buffer           │                                │
│                   │ (accumulating)   │                                │
│                   └─────────┬───────┘                                │
│                             │                                        │
│                    Buffer full? (N samples)                          │
│                        │          │                                   │
│                       YES        NO                                  │
│                        │          │                                   │
│              ┌─────────┴──────┐   └──→ Wait for more                │
│              │  LEVEL 3:      │                                      │
│              │  RETRAIN       │                                      │
│              └─────────┬──────┘                                      │
│                        │                                             │
│              ┌─────────┴───────────────────────────────────┐        │
│              │                                              │        │
│              │  1. Combine: original CSV + experience buffer │        │
│              │     (100K rows + 1K new rows)                │        │
│              │                                              │        │
│              │  2. Retrain model from scratch                │        │
│              │     Adam optimizer, MSE loss                  │        │
│              │     ~10 seconds for 19K parameters           │        │
│              │                                              │        │
│              │  3. Export new model.nnwt                     │        │
│              │                                              │        │
│              │  4. Hot-reload:                               │        │
│              │     cudaMemcpy(new weights → GPU)            │        │
│              │     Update OOD feature bounds                │        │
│              │     Clear experience buffer                  │        │
│              │     Reset error tracking                     │        │
│              │                                              │        │
│              └──────────────────────────────────────────────┘        │
│                        │                                             │
│                        │  Model is now smarter                       │
│                        │  Fewer predictions wrong                    │
│                        │  Less exploration needed                    │
│                        │  Lower overhead                             │
│                        │                                             │
│                        └──→ Back to top (next compress call)         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Complete Timeline: From Nothing to Expert

```
TIME ──────────────────────────────────────────────────────────────────►

Phase 0          Phase 1           Phase 2          Phase 3
NO MODEL         BENCHMARK         TRAIN            EVALUATE

  ?              Run 1,568 files   100K rows        Top-1: 75%
  │              × 64 configs      ──→ model.pt     Top-3: 88%
  │              = 100,352 rows    19K params        │
  │                  │             ~10 sec            │
  │                  │                │               │
  ▼                  ▼                ▼               ▼

Phase 4          Phase 5           Phase 5+          Phase 5++
DEPLOY           INFERENCE         ACTIVE LEARN      MATURE MODEL

model.nnwt       32 threads        Collect data     Exploration
──→ GPU          predict &         Explore when     rate < 1%
76 KB            rank              wrong            Near-optimal
  │              <0.1ms            Retrain           for user's
  │                │               periodically     workload
  │                │                   │               │
  ▼                ▼                   ▼               ▼

                 ┌─────────── Model Quality Over Time ───────────┐
                 │                                                │
      Accuracy   │                    ···············────────────  │
                 │              ·····                              │
                 │          ····                                   │
                 │       ···              ▲ Retrain               │
                 │     ··                 │ events                │
                 │   ··                   │                        │
                 │  · ▲ New workload      │                        │
                 │ ·  │ (accuracy drops,  │                        │
                 │·   │  exploration      │                        │
                 │    │  kicks in)        │                        │
                 └────┴──────────────────┴────────────────────────┘
                 Day 1              Day 7            Day 30
```
