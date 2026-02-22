# NN System Overview — 5 Timestep Example

Imagine a simulation compressing 5 different scientific datasets in sequence using `ALGO_AUTO`.

## The NN

```
Input [15 features]                    Output [4 values]
┌─────────────────────┐                ┌──────────────────────┐
│ algo one-hot [0-7]  │   128    128   │ [0] comp_time (log)  │
│ quantization [8]    │──►ReLU──►ReLU──│ [1] decomp_time(log) │
│ shuffle [9]         │                │ [2] ratio (log)      │ ← primary target
│ error_bound [10]    │                │ [3] psnr (clamped)   │
│ data_size [11]      │                └──────────────────────┘
│ entropy [12]        │
│ mad [13]            │
│ 2nd_derivative [14] │
└─────────────────────┘
```

32 configs exist: `8 algos × 2 quant × 2 shuffle`. Each gets an action ID: `algo + quant*8 + shuffle*16`.

---

## Timestep 1 — Good Prediction (no exploration, no SGD)

```
Data arrives: temperature field, 1MB, entropy=4.2, mad=0.31

  ┌──────────────────────────────────────────────────────────┐
  │ GPU: nnInferenceKernel (32 threads in parallel)          │
  │                                                          │
  │  Thread 0 (lz4, no quant, no shuf):  ratio=2.1          │
  │  Thread 1 (snappy, no quant, no shuf): ratio=1.8        │
  │  ...                                                     │
  │  Thread 5 (ans, no quant, no shuf):   ratio=3.4  ← BEST │
  │  ...                                                     │
  │  Thread 31:                           ratio=1.2          │
  │                                                          │
  │  Output: action=5 (ANS), predicted_ratio=3.4             │
  │          top_actions=[5, 21, 13, 3, ...]  (sorted)       │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ GPU: Compress with ANS                                   │
  │  actual_ratio = 3.2    comp_time = 0.8ms                 │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  MAPE = |3.4 - 3.2| / 3.2 = 6.25%  < threshold (20%)

  ✓ Level 1: Write to experience CSV  (action=5, ratio=3.2, time=0.8)
  ✗ Level 2: No exploration needed
  ✗ SGD: explored_samples has only 1 entry, reinforcement still fires
         but with just the primary sample

  → Return compressed data (ANS, ratio 3.2)
```

**Weights unchanged or barely nudged** — prediction was close.

---

## Timestep 2 — Bad Prediction (exploration + SGD)

```
Data arrives: velocity field, 512KB, entropy=7.1, mad=0.02 (noisy, unfamiliar)

  ┌──────────────────────────────────────────────────────────┐
  │ GPU: nnInferenceKernel                                   │
  │  Winner: action=13 (ANS+quant), predicted_ratio=4.1     │
  │  top_actions=[13, 5, 29, 21, 3, ...]                    │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ GPU: Compress with ANS+quant                             │
  │  actual_ratio = 2.5    comp_time = 1.2ms                 │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  MAPE = |4.1 - 2.5| / 2.5 = 64%  > threshold (20%)
                                     > 50% → K=9

  ✓ Level 1: Write primary (action=13, ratio=2.5, time=1.2)

  ┌──────────────────────────────────────────────────────────┐
  │ Level 2 EXPLORATION: Try 9 alternatives from top_actions │
  │                                                          │
  │  alt action=5  (ANS)         → ratio=2.8, time=0.9  ✓   │
  │  alt action=29 (ANS+shuf+q) → ratio=3.1, time=1.5  ✓✓  │
  │  alt action=21 (ANS+shuf)   → ratio=2.9, time=0.7  ✓   │
  │  alt action=3  (gdeflate)   → ratio=2.1, time=2.0      │
  │  ...5 more alternatives...                               │
  │                                                          │
  │  Best found: action=29, ratio=3.1 > original 2.5        │
  │  → SWAP output to ANS+shuffle+quant result               │
  └──────────────────────────────────────────────────────────┘

  Each alternative → experience_buffer_append() (Level 1 for each)

  ┌──────────────────────────────────────────────────────────┐
  │ ONLINE REINFORCEMENT (SGD)                               │
  │                                                          │
  │  explored_samples = [                                    │
  │    {action=13, ratio=2.5, time=1.2},  ← primary         │
  │    {action=5,  ratio=2.8, time=0.9},  ← explored        │
  │    {action=29, ratio=3.1, time=1.5},  ← explored        │
  │    {action=21, ratio=2.9, time=0.7},  ← explored        │
  │    ...6 more...                                          │
  │  ]                                                       │
  │                                                          │
  │  For EACH sample:                                        │
  │    nn_reinforce_add_sample(features, ratio, time)        │
  │    ├─ CPU forward pass (same math as GPU kernel)         │
  │    ├─ Loss = 0.5*(pred_ratio - actual_ratio)²           │
  │    │       + 0.5*(pred_time  - actual_time)²            │
  │    ├─ Backprop through 3 layers                          │
  │    └─ Accumulate gradients: dw1+=, db1+=, dw2+=, ...    │
  │                                                          │
  │  nn_reinforce_apply():                                   │
  │    ├─ Average gradients by sample_count (10)             │
  │    ├─ Clip gradient L2 norm to 1.0                       │
  │    ├─ SGD: w -= lr * grad  (lr=1e-4)                    │
  │    └─ cudaMemcpy updated weights → GPU                   │
  └──────────────────────────────────────────────────────────┘

  → Return compressed data (ANS+shuf+quant, ratio 3.1) — the better one
```

**Weights updated.** Next inference uses improved weights.

---

## Timestep 3 — Similar Data Benefits from T2's Update

```
Data arrives: another velocity field, 512KB, entropy=6.9, mad=0.03

  GPU inference now uses UPDATED weights from timestep 2.
  Model learned that high-entropy noisy data → ANS+shuf+quant is good.

  Winner: ANS+shuf+quant (action=29), predicted_ratio=3.0
  Actual: ratio=3.05

  MAPE = 1.6% < 20%  → no exploration, minimal SGD nudge

  → Fast path, good prediction thanks to T2's reinforcement
```

---

## Timestep 4 — OOD Data (aggressive exploration)

```
Data arrives: brand new data type, 4MB, entropy=1.1, mad=0.95

  GPU: nnInferenceKernel runs, but also checks OOD:
    isInputOOD() → entropy 1.1 is below training min → OOD=true

  Winner: action=7 (bitcomp), predicted_ratio=5.0
  Actual: ratio=8.2

  MAPE = 39%, but OOD=true → K=31 (try ALL alternatives)

  Exploration: runs all 31 other configs on GPU
    → discovers action=23 (bitcomp+shuf) gets ratio=9.1

  SGD: 32 samples (primary + 31 explored) all fed to backprop
       Massive learning opportunity for this new data regime

  → Return bitcomp+shuf, ratio 9.1
  → 32 rows written to experience CSV
```

---

## Timestep 5 — Steady State

```
Data arrives: pressure field, 1MB, entropy=3.8, mad=0.28

  Weights now reflect learning from T1-T4.
  Prediction: action=5 (ANS), ratio=3.5
  Actual: ratio=3.4

  MAPE = 2.9% — good, no exploration

  → Fast path, 1 row to CSV, minimal SGD
```

---

## Summary: What Lives Where

```
                    Process Memory              Disk
                   ┌──────────────┐     ┌─────────────────┐
  GPU weights      │ ■ Updated by │     │                 │
  (live inference) │   SGD each   │     │ model.nnwt      │
                   │   timestep   │     │ (stale until    │
                   │              │     │  offline retrain │
                   │ Host shadow  │     │  or manual save) │
                   │ (for backprop)│     │                 │
                   └──────────────┘     │ experiences.csv  │
                                        │  T1: 1 row      │
                                        │  T2: 10 rows    │
                                        │  T3: 1 row      │
                                        │  T4: 32 rows    │
                                        │  T5: 1 row      │
                                        │  (= 45 rows)    │
                                        └─────────────────┘
                                               │
                                    offline retrain.py reads this
                                    + original benchmark data
                                    → produces new model.nnwt
                                    → gpucompress_reload_nn()
```

The key insight: **online SGD makes the model better within a session** (T2's learning helps T3), but those improvements vanish on restart. The experience CSV preserves the ground-truth data so offline retraining can permanently incorporate it.
