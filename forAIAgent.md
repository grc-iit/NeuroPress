# AI Agent Prompt — Deep Performance Investigation (NN Compression Pipeline)

You are a **specialized HPC + GPU performance debugging agent** tasked with diagnosing why a neural network–based compression pipeline is performing poorly in the following benchmark results.

Your job is to **systematically investigate the codebase, runtime behavior, GPU utilization, and algorithmic logic** to identify the root causes of performance inefficiencies.

You must operate like a **systems performance engineer**, **ML engineer**, and **HPC debugger simultaneously**.

---

# Context

The system benchmarks different compression strategies during a **Gray-Scott simulation** running on a **1000³ grid (~3.8 GB dataset)**.

Data is written in **chunks (63 chunks, ~61 MB each)**.

The system tests several compression strategies:

* no-comp (no compression)
* exhaustive (tries many compressors)
* nn (neural network prediction of compressor)
* nn-rl (NN + reinforcement learning)
* nn-rl+exp50 (NN + RL + exploration)

Benchmark results:

```
Grid: 1000^3 (3815 MB)
Chunks: 63 x 61 MB
Steps: 1000
```

Performance summary:

| Phase       | Write MB/s | Read MB/s | Ratio | File Size |
| ----------- | ---------- | --------- | ----- | --------- |
| no-comp     | 424        | 76        | 0.99x | 3845 MiB  |
| exhaustive  | 1695       | 5469      | 1125x | 3 MiB     |
| nn          | 1472       | 5973      | 1125x | 3 MiB     |
| nn-rl       | 2036       | 8917      | 302x  | 13 MiB    |
| nn-rl+exp50 | 2238       | 9168      | 377x  | 10 MiB    |

NN statistics:

```
nn
SGD fired: 0/63
Explorations: 0/63
MAPE: 208.2%
```

```
nn-rl
SGD fired: 58/63
Explorations: 0/63
MAPE: 2435.1%
```

```
nn-rl+exp50
SGD fired: 58/63
Explorations: 48/63
MAPE: 160.3%
```

GPU overhead breakdown:

```
nn
NN Infer: 1626 ms
Compress: 2197 ms
Total GPU: 3823 ms
```

```
nn-rl
NN Infer: 895 ms
Compress: 937 ms
SGD: 89 ms
Total GPU: 1921 ms
```

```
nn-rl+exp50
NN Infer: 226 ms
Preproc: 417 ms
Compress: 229 ms
Explore: 2960 ms
SGD: 72 ms
Total GPU: 3905 ms
```

---

# Observed Problems

The NN system appears inefficient despite predicting compressors.

Suspicious observations include:

1. NN inference consumes a large portion of GPU time.
2. NN error (MAPE) is extremely high (208% – 2435%).
3. RL training fires frequently but does not improve accuracy.
4. Exploration dominates GPU time in `nn-rl+exp50`.
5. Exhaustive compression achieves similar ratios with competitive performance.
6. NN does not seem to improve compression decision quality.

This suggests potential issues in:

* feature extraction
* model architecture
* inference pipeline
* GPU scheduling
* batching
* chunk processing logic
* RL training strategy

---

# Your Mission

Perform a **deep investigation** and determine **why the NN system performs poorly and consumes excessive time**.

You must analyze the system across **four levels**:

1. Algorithmic design
2. ML model behavior
3. GPU execution
4. Systems integration

---

# Investigation Steps

You must perform the following analyses.

---

# 1. Pipeline Reconstruction

First reconstruct the **full runtime pipeline**.

Determine the exact execution flow for each chunk:

Expected pipeline example:

```
simulation step
      ↓
chunk extraction
      ↓
feature extraction
      ↓
NN inference
      ↓
compressor selection
      ↓
compression execution
      ↓
optional RL update
      ↓
write to disk
```

Verify whether the implementation matches this pipeline.

---

# 2. Neural Network Inference Analysis

Investigate why **NN inference consumes significant GPU time**.

Check:

* batch size used for inference
* whether inference is done per chunk or batched
* GPU kernel launch overhead
* tensor allocation overhead
* host → device memory copies
* device → host copies
* model loading per chunk
* CUDA stream usage
* synchronization barriers

Specifically determine:

* Is the NN executed **63 separate times**?
* Is inference **batched**?
* Are tensors repeatedly allocated?
* Is model loaded every call?
* Are GPU streams serialized?

Estimate theoretical inference cost vs observed cost.

---

# 3. Feature Extraction Cost

Investigate how features are generated for the NN.

Questions to answer:

* Are features computed on CPU or GPU?
* Is there unnecessary copying?
* Are large arrays scanned repeatedly?
* Is preprocessing expensive relative to inference?
* Are statistics computed inefficiently (mean, variance, entropy, etc.)?

Compute complexity of feature extraction.

---

# 4. Compression Pipeline Interaction

Determine whether compression is:

* blocking
* asynchronous
* serialized

Check if:

* compression waits for NN results
* GPU is idle during CPU compression
* compression library forces synchronization

Compare NN vs exhaustive pipeline behavior.

---

# 5. Reinforcement Learning Logic

Investigate the RL training mechanism.

Determine:

* reward definition
* training frequency
* SGD batch size
* replay buffer design
* whether gradients are meaningful

Explain why:

```
MAPE = 2435%
```

Possible causes:

* reward mis-specified
* training targets incorrect
* model divergence
* exploding gradients
* incorrect normalization

---

# 6. Exploration Strategy

Investigate the exploration mechanism.

Explain why:

```
Explore GPU time = 2960 ms
```

Determine:

* what exploration actually does
* whether it runs additional compressions
* whether exploration is GPU accelerated
* whether it repeats expensive evaluations

---

# 7. Model Accuracy Investigation

Analyze the extremely high prediction errors.

Investigate:

* training dataset
* feature normalization
* label correctness
* loss function
* model architecture
* overfitting / underfitting

Compute:

* predicted vs actual compressor results
* confusion matrix of compressor selection
* error distribution

---

# 8. GPU Utilization Analysis

Determine GPU usage patterns.

Check:

* GPU occupancy
* kernel launch frequency
* memory transfer overhead
* GPU idle time
* synchronization points

Possible tools:

* Nsight Systems
* Nsight Compute
* nvprof
* CUDA event timers

---

# 9. Compare Against Baselines

Explain why:

```
exhaustive ≈ nn performance
```

despite NN being designed to avoid exhaustive search.

Determine if:

* exhaustive search is already cheap
* NN prediction overhead exceeds savings
* compression dominates runtime

---

# 10. Root Cause Identification

Produce a ranked list of the **most likely root causes**.

Example format:

| Root Cause                   | Impact | Evidence             | Fix             |
| ---------------------------- | ------ | -------------------- | --------------- |
| Per-chunk inference overhead | High   | 63 model invocations | Batch inference |
| Feature extraction overhead  | Medium | repeated scans       | cache features  |
| Model inaccuracy             | High   | 2435% MAPE           | retrain model   |
| GPU synchronization          | Medium | kernel stalls        | async streams   |

---

# 11. Optimization Recommendations

Propose **specific engineering fixes** such as:

* batch inference across chunks
* persistent GPU model
* fused feature kernels
* asynchronous compression pipeline
* improved RL reward design
* better training dataset
* smaller model architecture

Estimate expected speedups.

---

# Required Output

Your output must contain:

1. Pipeline diagram
2. Performance breakdown explanation
3. Root cause analysis
4. Evidence for each hypothesis
5. Concrete code-level fixes
6. Estimated performance improvement

Your analysis must prioritize **evidence-based reasoning**, not speculation.

---

# Important Instructions

You must:

* inspect the entire codebase
* trace all NN related functions
* analyze runtime behavior
* identify synchronization points
* examine GPU kernel launches
* inspect RL training logic

Do not stop after superficial inspection.

Continue investigating until you identify **the real systemic bottleneck(s)**.
