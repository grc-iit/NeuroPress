# GPUCompress — Roadmap to SC Submission

> **Current Assessment: Weak Reject (fixable)**
>
> The core contribution — online-adaptive algorithm selection with GPU-native NN inference
> and HDF5 VOL integration — is genuinely novel and well-engineered.
> The evaluation does not yet support the generality claims the work implicitly makes.
>
> **Path to Weak Accept:** ~2-3 weeks of benchmark runs.
> **Path to Solid Accept:** + oracle baseline + heuristic comparison + policy evaluation.

---

## Table of Contents

1. [What Reviewers Will Like](#1-what-reviewers-will-like)
2. [Critical Gaps — P0 (Desk-Reject Risk)](#2-critical-gaps--p0-desk-reject-risk)
3. [Important Gaps — P1 (Reviewers Will Raise)](#3-important-gaps--p1-reviewers-will-raise)
4. [Strengthening — P2 (Competitive Edge)](#4-strengthening--p2-competitive-edge)
5. [Aspirational — P3 (Exceptional)](#5-aspirational--p3-exceptional)
6. [Resolved Issues](#6-resolved-issues)
7. [Prioritized Action Plan](#7-prioritized-action-plan)
8. [Framing Strategy](#8-framing-strategy)
9. [Existing Infrastructure to Leverage](#9-existing-infrastructure-to-leverage)
10. [Questions the Paper Must Answer](#10-questions-the-paper-must-answer)

---

## 1. What Reviewers Will Like

These are genuine strengths — lead with them in the paper.

- **Online SGD convergence.** MAPE drops from ~700% to ~10-20% within 20 timesteps on VPIC.
  This is the paper's strongest empirical result and should be the lead figure.

- **4-phase ablation structure.** `no-comp` / `nn` / `nn-rl` / `nn-rl+exp50` is exactly the
  ablation SC expects for a learned system.

- **HDF5 VOL connector.** 3,500+ lines of production-quality integration. Closes the gap between
  standalone benchmark and real in-situ coupling. Genuine differentiator vs. prior work.

- **Log-space cost model.** 4 named policy modes (Speed, Balanced, Ratio-First, Throughput),
  scale-invariant, multi-backend I/O aware. Principled and theoretically motivated.

- **Measurement discipline.** Per-chunk diagnostics, 5-fold cross-validation infrastructure,
  timing breakdown CSVs, publication-quality visualization pipeline. Ahead of typical SC submissions.

- **Test suite.** 90+ executables covering unit tests, regression, concurrency, VOL integration.
  Sign of a mature system, not a research prototype.

---

## 2. Critical Gaps — P0 (Desk-Reject Risk)

### 2.1 Dataset Diversity

| Problem | Only 2 workloads (Gray-Scott + VPIC), both float32 structured grids |
|---------|---------------------------------------------------------------------|
| **Risk** | Reviewers will ask: *"Does this generalize beyond float32 reaction-diffusion?"* |

**What to add (minimum 3-4 more):**

| Dataset | Domain | Type | Why It Matters |
|---------|--------|------|----------------|
| Hurricane Isabel | Climate | float32, 3D fields | Smooth fields, 2-5x compressible, widely used SC benchmark |
| Nyx / HACC | Cosmology | float32, particles | Highly compressible, spatial locality patterns |
| Miranda or LAMMPS | Turbulence / MD | **float64** | Covers the float64 gap — currently zero float64 results |
| CESM-ATM | Atmospheric | float32, mixed | Multi-variable, different entropy profiles per field |

**Fastest path:** [SDRBench](https://sdrbench.github.io) provides standardized HDF5/binary files
for all of these. Loading a binary file into GPU memory requires ~50 lines of C.

**Effort:** 2-3 days

---

### 2.2 Comparison Against State-of-the-Art Compressors

| Problem | Zero comparison vs SZ3, ZFP, cuSZ, MGARD, or LibPressio |
|---------|----------------------------------------------------------|
| **Risk** | The 4-phase comparison is entirely self-referential. Reviewer question: *"For VPIC at 1.73x ratio, how does this compare to ZFP at the same throughput?"* — the paper has no answer. |

**Required baselines:**

```
Lossless:
  - fixed-zstd   (always use zstd via nvCOMP — is the NN better than this?)
  - fixed-lz4    (always use lz4 via nvCOMP — speed baseline)
  - CPU zstd      (the "just use the CPU" sanity check)

Lossy (at matching error bounds):
  - cuSZ          (GPU-native SZ)
  - ZFP           (GPU-native ZFP)
  - SZ3           (CPU baseline for lossy)
```

**Key comparison table the paper needs:**

| Method | Ratio | Comp GB/s | Decomp GB/s | NN Overhead |
|--------|-------|-----------|-------------|-------------|
| fixed-lz4 | — | — | — | N/A |
| fixed-zstd | — | — | — | N/A |
| GPUCompress (nn) | — | — | — | ~0.22 ms |
| GPUCompress (nn-rl) | — | — | — | ~0.22 ms |
| cuSZ | — | — | — | N/A |
| ZFP | — | — | — | N/A |

**Effort:** 2-3 days (build system + ~100 lines benchmark driver code)

---

### 2.3 VPIC Benchmark Completeness

| Problem | `benchmark_vpic_deck.csv` only has 2 of 4 phases (`no-comp`, `nn`), all `n_runs=1` |
|---------|-----------------------------------------------------------------------------------|
| **Risk** | Summary figure for the most realistic workload will be missing `nn-rl` and `nn-rl+exp50` bars |

**Fix:** Run the full 4-phase VPIC benchmark with `--runs 5` and regenerate the CSV.
The multi-timestep data (`benchmark_vpic_timesteps.csv`) already exists with 50 timesteps
and is the paper's strongest result — but the aggregate summary that `visualize.py` reads is incomplete.

**Effort:** Overnight benchmark run (no code changes)

---

### 2.4 Statistical Rigor

| Problem | Everything is single-run. GPU timing fluctuates 5-15% between runs. |
|---------|---------------------------------------------------------------------|
| **Risk** | Single-run numbers are not publishable at SC. |

**Fix:**
- Run every phase with `--runs 5` minimum (infrastructure already implemented via `--runs N`)
- Report mean +/- standard deviation for all timing measurements
- Consider median + IQR for skewed distributions (e.g., first-chunk warmup)
- Ensure `n_runs`, `*_std` columns are populated in all CSVs

**Effort:** Overnight runs per dataset

---

## 3. Important Gaps — P1 (Reviewers Will Raise)

### 3.1 Exhaustive Oracle Baseline

| Problem | `visualize.py` references an `exhaustive` phase but no results exist |
|---------|----------------------------------------------------------------------|
| **Risk** | Cannot answer: *"How close is the NN to the optimal selection?"* |

The oracle tries all 32 configurations per chunk and picks the best *measured* (not predicted) one.
This bounds the NN's maximum possible improvement and is the most fundamental comparison.

**Implementation:** Set exploration `K=31` to try all configs. Add as a named phase.
Compute **NN regret:** `(oracle_ratio - nn_ratio) / oracle_ratio`.

**Target headline:** *"Median regret <5% with NN overhead of <0.5 ms per chunk"*

**Effort:** 1 day

---

### 3.2 Heuristic Baseline

| Problem | No comparison against a simple rule-based selector |
|---------|---------------------------------------------------|
| **Risk** | Reviewer question: *"Does the NN do anything a 5-line if-statement can't?"* |

**Implementation:** A selector using only entropy and data size (both already computed by the stats kernel):

```
entropy < 3.5  →  zstd
3.5 - 5.5      →  deflate
> 5.5          →  lz4
```

Run as an `entropy-heuristic` phase. The NN should beat this by selecting shuffle+preprocessing
when appropriate and by adapting to chunk-specific patterns entropy alone cannot distinguish.

**Effort:** 2 hours

---

### 3.3 Lossy Compression Evaluation

| Problem | NN predicts PSNR, quantization path exists, but benchmark defaults to lossless (`error_bound=0.0`) |
|---------|---------------------------------------------------------------------------------------------------|
| **Risk** | Lossy is where algorithm selection matters most (10-100x more effective). Leaving the most interesting use case unevaluated is a missed opportunity. |

**Required:**
- Error-bound sweep: lossless, 1e-2, 1e-3, 1e-4, 1e-5
- Rate-distortion plots: ratio vs PSNR at each error bound
- Show NN correctly selects different algorithms at different error bounds
- Compare against ZFP/SZ3 at matching error bounds

**Effort:** Medium (2-3 days)

---

### 3.4 Policy Mode Evaluation

| Problem | 4 cost model modes defined in `CostModel.md` but never benchmarked |
|---------|-------------------------------------------------------------------|
| **Risk** | Untested claims are weak claims. Easy to produce, hard to justify omitting. |

**Required table:**

| Policy Mode | alpha | beta | delta | % LZ4 | % Zstd | % GDeflate | Avg Ratio | Avg Comp GB/s |
|-------------|-------|------|-------|--------|--------|------------|-----------|---------------|
| Speed | 1 | 0 | 0 | ? | ? | ? | ? | ? |
| Balanced | 1 | 1 | 0.5 | ? | ? | ? | ? | ? |
| Ratio-First | 0.3 | 1 | 1 | ? | ? | ? | ? | ? |
| Throughput | 1 | 0 | 1 | ? | ? | ? | ? | ? |

**Effort:** Half day (pure benchmark run, no code changes)

---

### 3.5 Training Data Disclosure & Model Characterization

| Problem | `model.nnwt` is shipped but undocumented — what data was it trained on? |
|---------|-------------------------------------------------------------------------|
| **Risk** | Cold-start MAPE of 3730% on VPIC suggests the model never saw similar data. Reproducibility concern. |

**Required:**
- Document training dataset composition (sources, sample count, algorithm distribution)
- Report 5-fold CV metrics (MAE, R2, MAPE per output) from `cross_validate.py`
- Clarify: is the online learning improvement (700% → 10%) simply adapting from the
  training distribution to the test distribution, or does it demonstrate genuine online learning?

**Effort:** Medium (1-2 days)

---

### 3.6 Cold-Start Reliability

| Problem | First 3 chunks on VPIC show `predicted_ratio=100.0`, actuals are 2.6-10x (MAPE: 946-3730%) |
|---------|------------------------------------------------------------------------------------------|
| **Risk** | Algorithm selection for the first N chunks of any new dataset is essentially random. |

**Mitigation strategies (pick one or more):**
1. Frame as a feature: *"The system starts naive but adapts within 20 timesteps"*
2. Add a lightweight calibration pass: compress 1 chunk with 3 candidate algorithms, then start NN
3. Ship multiple domain-specific pretrained models (climate, plasma, cosmology)
4. Show the convergence curve prominently — the recovery *is* the result

---

### 3.7 Ablation Study (Component Contributions)

| Component | What to Measure |
|-----------|-----------------|
| Stats kernel | Overhead: ~0.22 ms/chunk |
| NN inference | Overhead: ~0.22 ms/chunk vs compression: ~6.7 ms (3.3% overhead) |
| Byte shuffle | Ratio improvement when enabled vs disabled |
| Quantization | Ratio improvement at each error bound |
| SGD online learning | MAPE reduction curve over timesteps |
| Exploration | % of chunks where exploration finds a better action than NN predicted |
| Cost model | Algorithm selection distribution under each policy mode |

**Key headline:** *"NN adds 0.22 ms overhead but saves Y ms in I/O due to better algorithm choice"*

**Effort:** Medium (most data already in per-chunk CSVs, needs analysis + visualization)

---

### 3.8 Algorithm Diversity in Practice

| Problem | Chunks cycle through zstd, deflate+shuf, gdeflate+shuf. Bitcomp/ANS/Cascaded/Snappy never appear. |
|---------|---------------------------------------------------------------------------------------------------|
| **Risk** | If the NN only selects 3-4 of 8 algorithms, why have 8? |

**Required:**
- Report algorithm selection frequency histogram across all datasets
- Show at least one dataset where Bitcomp/ANS/Cascaded is selected
- If some algorithms are never competitive, acknowledge it honestly
- The `find_bitcomp_data.cu` and `find_snappy_data.cu` tests suggest this was a known concern

---

### 3.9 Decompression MAPE

| Problem | Persistent ~154% decomp MAPE despite SGD being wired in |
|---------|----------------------------------------------------------|
| **Risk** | Leaving a 154% MAPE as a top-level metric without explanation attracts questions |

**Resolution options:**
1. Show convergence (if possible with tuned noise gate threshold)
2. Explain why decomp time is not decision-critical (cost model uses ct, not dt, for ranking at write time)
3. Remove decomp time as an evaluated accuracy metric and report only ratio + compress time MAPE
4. De-emphasize: *"Decompression times are sub-millisecond; sMAPE is inherently unstable at this scale"*

---

## 4. Strengthening — P2 (Competitive Edge)

### 4.1 Latency Breakdown Figure

For a representative chunk, show a stacked bar:

```
stats=0.22ms | nn=0.22ms | preproc=0ms | compress=6.7ms | total=7.1ms
```

Data is already in per-chunk CSVs. Answers *"Is the overhead worth it?"* at a glance.

### 4.2 Chunk-Size Scaling Study

Fix total data size (512 MB), vary chunk size from 1 MB to 64 MB.
Show how ratio and throughput vary. Exercises the per-chunk adaptation logic.

### 4.3 Data-Size Scaling Study

Use existing Gray-Scott sizes: L=128 (8 MB) through L=1260 (8 GB).
Frame as strong scaling with proper axis labels. Reuse existing data.

### 4.4 End-to-End Application Coupling

Show total simulation time with and without GPUCompress I/O compression.
Compute I/O time reduction fraction: *"compression reduces I/O from 40% to 3% of total runtime"*

### 4.5 Roofline / Performance Model

Measure achieved fraction of peak GPU memory bandwidth.
Identify bottleneck per data size: NN inference? nvCOMP kernel? Memory transfer?

### 4.6 Gray-Scott Pattern Injection Disclosure

`inject_patterns_kernel` in `grayscott-benchmark.cu` (lines 143-189) inserts synthetic patterns
into the middle third of chunks. Clarify in the paper or provide a separate pure-simulation result.

---

## 5. Aspirational — P3 (Exceptional)

| Item | Effort | Impact |
|------|--------|--------|
| Multi-GPU evaluation | Very High | Would be exceptional for a systems paper |
| Multi-node MPI-parallel HDF5 | Very High | Addresses scalability questions |
| Second GPU architecture (H100) | Medium | Currently only A100 (sm_80) |
| Energy/power measurement | Medium | Increasingly expected at SC |
| Cross-domain transfer study | Medium | Train on Gray-Scott, test on VPIC (or vice versa) |

---

## 6. Resolved Issues

| Issue | Status | Description |
|-------|--------|-------------|
| MAPE/sMAPE labeling | **DONE** | Both metrics computed, CSV columns correctly labeled |
| Single-shot warmup | **DONE** | Discard warmup write cycle added before timed runs |
| Isolated compression throughput | **DONE** | `comp_gbps` and `decomp_gbps` fields added with full timing breakdown |
| `--runs N` infrastructure | **DONE** | Multi-run with mean +/- std, NN weights reloaded per run |
| no-comp baseline caveat | **DONE** | Documented as "uncompressed through same VOL stack" |

---

## 7. Prioritized Action Plan

### Week 1 — Foundation (P0)

| # | Action | Effort | Deliverable |
|---|--------|--------|-------------|
| 1 | Download SDRBench datasets (Hurricane, Nyx, Miranda, CESM) | 1 day | Raw data on disk |
| 2 | Write generic binary-to-GPU loader (~50 LOC) | 1 day | `generic-benchmark.cu` |
| 3 | Run full 4-phase VPIC benchmark with `--runs 5` | Overnight | Complete `benchmark_vpic_deck.csv` |
| 4 | Run Gray-Scott benchmark with `--runs 5` | Overnight | Updated CSVs with error bars |
| 5 | Run all SDRBench datasets through 4-phase benchmark | 2 days | 3-4 new dataset CSVs |

### Week 2 — Baselines & Comparisons (P0 + P1)

| # | Action | Effort | Deliverable |
|---|--------|--------|-------------|
| 6 | Add `fixed-zstd` and `fixed-lz4` phases to benchmark | Half day | Baseline comparison |
| 7 | Install cuSZ, add as baseline phase | 1-2 days | GPU compressor comparison |
| 8 | Implement exhaustive oracle (K=31) | 1 day | Oracle + NN regret numbers |
| 9 | Implement entropy-heuristic baseline | 2 hours | Rule-based comparison |
| 10 | Run 4-policy comparison on all datasets | Half day | Policy mode table |

### Week 3 — Analysis & Polish (P1 + P2)

| # | Action | Effort | Deliverable |
|---|--------|--------|-------------|
| 11 | Run 5-fold CV, document model training data | 1 day | Model characterization |
| 12 | Lossy error-bound sweep on 1-2 datasets | 2 days | Rate-distortion curves |
| 13 | Generate ablation figures (latency breakdown, convergence) | 1 day | Publication figures |
| 14 | Algorithm frequency histogram across all datasets | Half day | Diversity analysis |
| 15 | Update `visualize.py` for new baselines + datasets | 1 day | Final multi-panel figures |

---

## 8. Framing Strategy

### DO frame as:

> **"An online-adaptive algorithm selection system for in-situ HPC I/O"**
>
> - Lead result: VPIC convergence curve (MAPE 700% → 10% in 20 timesteps)
> - The cold-start problem becomes a *feature* ("adapts to any data distribution")
> - The VOL connector integration makes this *deployable*, not just a benchmark
> - The cost model is *policy-controlled* — users declare intent, not parameters

### DO NOT frame as:

> ~~"A compression library that uses a neural network"~~
>
> - This framing requires defeating SZ3 and ZFP on ratio (much harder bar)
> - The NN is a *means*, not the *end* — the adaptive selection is the contribution
> - Avoid the "we added ML therefore it's better" trap that reviewers are weary of

### Lead figures (in order):

1. **VPIC multi-timestep convergence** — MAPE dropping over 20 timesteps (the money plot)
2. **Algorithm selection diversity** — heatmap of which algorithm was chosen per chunk over time
3. **Comparison table** — GPUCompress vs fixed-algo vs cuSZ vs ZFP across datasets
4. **Latency breakdown** — stacked bar showing NN overhead is <5% of total compression time
5. **Policy mode comparison** — same data, 4 policies, different algorithm distributions

---

## 9. Existing Infrastructure to Leverage

| Asset | Location | Status |
|-------|----------|--------|
| VPIC benchmark skeleton | `benchmarks/vpic-kokkos/` | Exists, needs full-phase run |
| VPIC data adapter | `src/vpic/vpic_adapter.cu` | Implemented |
| Per-chunk diagnostics | `gpucompress_diagnostics.cpp` | Full timing + prediction data |
| Multi-panel visualization | `benchmarks/visualize.py` | 45 KB, publication-quality |
| `--runs N` multi-run | `grayscott-benchmark.cu` | Implemented, produces `*_std` columns |
| LR sweep automation | `sweep_lr.sh` | Shows how to automate parameter sweeps |
| 5-fold cross-validation | `neural_net/training/cross_validate.py` | Implemented |
| Weight export/inspect | `neural_net/export/` | PyTorch → binary `.nnwt` pipeline |
| Synthetic data generation | `syntheticGeneration/generator.py` | Available for controlled experiments |

---

## 10. Questions the Paper Must Answer

These are the questions SC reviewers *will* ask. Each must have a clear answer in the paper.

| # | Question | Where to Answer |
|---|----------|-----------------|
| 1 | *"Does this generalize beyond Gray-Scott?"* | Multi-dataset evaluation (P0) |
| 2 | *"Is it better than just always using zstd?"* | Fixed-algo baseline comparison (P0) |
| 3 | *"Is it better than SZ3/ZFP?"* | External compressor comparison (P0) |
| 4 | *"How close to optimal is the NN?"* | Exhaustive oracle + regret analysis (P1) |
| 5 | *"Does the NN beat a simple heuristic?"* | Entropy-heuristic baseline (P1) |
| 6 | *"What happens on completely new data?"* | Cold-start analysis + convergence curve (P1) |
| 7 | *"What was the model trained on?"* | Training data disclosure (P1) |
| 8 | *"Do the policy modes actually work?"* | Policy comparison table (P1) |
| 9 | *"What is the NN overhead?"* | Latency breakdown figure (P2) |
| 10 | *"Does it scale?"* | Data-size scaling + chunk-size study (P2) |
