# GPUCompress SC Paper: Complete Figure-by-Figure Evaluation Plan

> **Date:** 2026-03-22
> **Status:** Definitive evaluation section design for SC submission
> **Estimated total benchmark time:** ~72 GPU-hours (across all figures)

---

## Dataset Inventory

| ID | Dataset | Domain | Type | Dims | Per-field Size | Fields | Total Data |
|----|---------|--------|------|------|----------------|--------|------------|
| D1 | Gray-Scott (L=400) | Reaction-diffusion | float32 | 400^3 | 244 MB | 1 (V-field) | 244 MB |
| D2 | Gray-Scott (L=640) | Reaction-diffusion | float32 | 640^3 | 1 GB | 1 (V-field) | 1 GB |
| D3 | VPIC (NX=156) | Plasma physics | float32 | 158^3 x 16 | 253 MB | 16 (particles) | 253 MB |
| D4 | Hurricane Isabel | Climate/weather | float32 | 100x500x500 | 96 MB | 20 fields | 1.9 GB |
| D5 | Nyx | Cosmology | float32 | 512^3 | 512 MB | 6 fields (+template) | 3.1 GB |
| D6 | CESM-ATM | Atmospheric science | float32 | 1800x3600 | 25 MB | 78 fields | 1.9 GB |

**Rationale:** These 6 datasets span 4 scientific domains, 3 dimensionalities (2D, 3D, 3D+species), and entropy profiles from smooth (Hurricane) to near-incompressible (Nyx) to high-cardinality multi-variable (CESM-ATM). This exceeds the SC minimum of 5 diverse workloads.

---

## Phase Reference

| Phase ID | Phase Name | Description | Single-shot? | Multi-timestep? |
|----------|-----------|-------------|--------------|-----------------|
| P1 | `no-comp` | Uncompressed through VOL (D->H + native HDF5) | Yes | No |
| P2 | `cpu-zstd` | CPU zstd level 3, chunked to match GPU chunk size | Yes (script) | No |
| P3 | `fixed-lz4` | Always LZ4 via nvCOMP (speed extreme) | Yes | No |
| P4 | `fixed-gdeflate` | Always GDeflate via nvCOMP (GPU-optimized) | Yes | No |
| P5 | `fixed-zstd` | Always Zstd via nvCOMP (ratio extreme) | Yes | No |
| P6 | `entropy-heuristic` | Rule-based: entropy<3.5->zstd, 3.5-5.5->gdeflate, >5.5->lz4 | Yes | No |
| P7 | `best` | Exhaustive search: tries all 32 configs per chunk (oracle) | Yes | No |
| P8 | `nn` | NN inference only, no learning | Yes | Yes (static) |
| P9 | `nn-rl` | NN + online SGD | No | Yes |
| P10 | `nn-rl+exp50` | NN + online SGD + exploration | No | Yes |

---

## Global Parameters

Unless overridden per-figure:
- **Chunk size:** 4 MB (`--chunk-mb 4`)
- **Runs:** 5 (`--runs 5`) for all single-shot phases
- **Timesteps:** 100 (`--timesteps 100`) for multi-timestep phases
- **Cost model (balanced):** `--w0 1.0 --w1 1.0 --w2 1.0`
- **Learning rate:** 0.1 (default `REINFORCE_LR`)
- **Error bound:** 0.0 (lossless) unless stated otherwise
- **Warmup:** Built-in (discard warmup write before timed run)

---

## Figure 1: Multi-Dataset Phase Comparison (THE OVERVIEW FIGURE)

**Objective:** Answer "Does GPUCompress work across diverse scientific datasets?" This is the SC reviewer's first question. Establishes generality across domains with a single glance.

**Phases:** P1 (no-comp), P2 (cpu-zstd), P3 (fixed-lz4), P4 (fixed-gdeflate), P5 (fixed-zstd), P6 (entropy-heuristic), P7 (best/oracle), P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D1 (Gray-Scott L=400), D3 (VPIC NX=156), D4 (Hurricane Isabel), D5 (Nyx), D6 (CESM-ATM)

**Parameters:**
- `--chunk-mb 4 --runs 5 --timesteps 100`
- Balanced cost model: `--w0 1.0 --w1 1.0 --w2 1.0`

**Format:** Two stacked grouped bar charts.
- **Top panel:** X-axis = datasets, grouped bars = phases, Y-axis = compression ratio. Error bars from 5 runs.
- **Bottom panel:** Same layout, Y-axis = write throughput (MiB/s).

**Key insight:** NN-guided selection (nn-rl+exp50) matches or exceeds the best fixed algorithm on every dataset, while no single fixed algorithm wins everywhere. The oracle (best) provides the upper bound; nn-rl+exp50 should be within 5% of it.

**Exact commands to generate data:**

```bash
# Gray-Scott (D1)
L=400 TIMESTEPS=100 RUNS=5 bash benchmarks/grayscott/run_gs_eval.sh

# VPIC (D3)
NX=156 TIMESTEPS=100 RUNS=5 bash benchmarks/vpic-kokkos/run_vpic_eval.sh

# SDRBench datasets (D4, D5, D6)
RUNS=5 CHUNK_MB=4 bash benchmarks/sdrbench/run_sdrbench_eval.sh

# CPU zstd baseline
RUNS=5 CHUNK_MB=4 bash benchmarks/sdrbench/run_cpu_zstd_baseline.sh

# Visualize
python3 benchmarks/visualize.py --view summary
```

**Existing visualizer function:** `make_multi_dataset_figure()` and `make_per_dataset_phase_comparison()` in `visualize.py`.

---

## Figure 2: Online Learning Convergence (THE MONEY PLOT)

**Objective:** Answer "How quickly does the NN adapt to unseen data?" This demonstrates the core contribution: online SGD reduces prediction error from cold-start (hundreds of percent MAPE) to operational accuracy (<20%) within 20 timesteps. The convergence curve is the paper's signature result.

**Phases:** P8 (nn, static baseline), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D3 (VPIC NX=156) -- best dataset for this because VPIC data is furthest from training distribution, producing the most dramatic cold-start-to-convergence arc.

**Parameters:**
- `--timesteps 100 --chunk-mb 4`
- Balanced cost model: `--w0 1.0 --w1 1.0 --w2 1.0`

**Format:** 3-row subplot, one per predicted metric:
- Row 1: Compression ratio MAPE (%) over timesteps
- Row 2: Compression time MAPE (%) over timesteps
- Row 3: Decompression time MAPE (%) over timesteps
- Each row has 3 lines: nn (flat, no learning), nn-rl (drops), nn-rl+exp50 (drops faster or finds better solutions)
- Horizontal reference line at 20% MAPE

**Key insight:** Ratio MAPE drops from ~700% to ~10-20% within 20 timesteps on VPIC. This means the system self-corrects even when the pretrained model has never seen similar data. The exploration variant (nn-rl+exp50) converges faster.

**Exact commands:**

```bash
NX=156 TIMESTEPS=100 RUNS=1 bash benchmarks/vpic-kokkos/run_vpic_eval.sh
python3 benchmarks/visualize.py --vpic-dir benchmarks/vpic-kokkos/results/eval_NX156_chunk4mb_ts100/balanced_w1-1-1 --view timesteps
```

**Existing visualizer function:** `make_timestep_figure()` in `visualize.py`.

---

## Figure 3: SGD and Exploration Firing Rate

**Objective:** Answer "How active is the online learning system over time?" Shows that SGD fires frequently at first (many chunks have high cost model error) and tapers off as the model converges. Exploration triggers less frequently but discovers better configurations in early timesteps.

**Phases:** P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D3 (VPIC NX=156)

**Parameters:** Same as Figure 2.

**Format:** 2-row subplot:
- Row 1: SGD fires per timestep (Y) vs timestep (X), one line per phase
- Row 2: Exploration triggers per timestep (Y) vs timestep (X)

**Key insight:** SGD fires on 60-80% of chunks at T=1, dropping to <10% by T=20. Exploration fires on ~50% initially, then converges. This proves the system is actively adapting, not just memorizing.

**Exact commands:** Same data as Figure 2, different visualization.

```bash
python3 benchmarks/visualize.py --vpic-dir benchmarks/vpic-kokkos/results/eval_NX156_chunk4mb_ts100/balanced_w1-1-1 --view timesteps
```

**Existing visualizer function:** `make_sgd_exploration_figure()` in `visualize.py`.

---

## Figure 4: Predicted vs Actual at Milestone Timesteps

**Objective:** Answer "How accurate are the NN predictions at different points in adaptation?" Provides visual proof that predicted-vs-actual curves converge over timesteps. Shows per-chunk granularity -- some chunks are harder to predict than others.

**Phases:** P9 (nn-rl)

**Datasets:** D1 (Gray-Scott L=400) -- has injected contrasting patterns, so different chunks have visibly different behavior.

**Parameters:** `--L 400 --timesteps 100 --chunk-mb 4`

**Format:** Grid of subplots: 5 rows (milestone timesteps: T=1, T=25, T=50, T=75, T=100) x 3 columns (ratio, comp time, decomp time). Each cell: X-axis = chunk index, two lines = predicted (blue dashed) and actual (black solid), shaded error region, MAPE stats box.

**Key insight:** At T=1, predicted and actual lines are wildly divergent. By T=50, they track closely. Chunks with injected random noise remain harder to predict than smooth chunks.

**Exact commands:**

```bash
L=400 TIMESTEPS=100 RUNS=1 bash benchmarks/grayscott/run_gs_eval.sh
python3 benchmarks/visualize.py --gs-dir benchmarks/grayscott/results/eval_L400_chunk4mb_ts100/balanced_w1-1-1 --view timesteps
```

**Existing visualizer function:** `make_timestep_chunks_figure()` in `visualize.py`.

---

## Figure 5: Algorithm Selection Heatmap Over Time

**Objective:** Answer "What algorithms does the NN actually select, and how does selection change as it learns?" This is the algorithm diversity figure. Shows that the NN does not degenerate to always picking one algorithm -- different chunks get different algorithms based on their statistics.

**Phases:** P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D1 (Gray-Scott L=400)

**Parameters:** `--L 400 --timesteps 100 --chunk-mb 4`

**Format:** Grid with columns = phases, rows = milestone timesteps. Each cell is a horizontal color strip where each pixel is one chunk, colored by the full 32-config selection (algo + shuffle + quant). Below: frequency histogram of selections per phase.

**Key insight:** Initial NN selection is suboptimal (perhaps always picking the same algorithm). After SGD, selection diversifies -- smooth chunks get zstd+shuf, noisy chunks get lz4, constant chunks get gdeflate+shuf. With exploration, rare algorithms (ans, cascaded) may appear for specific chunk profiles.

**Exact commands:**

```bash
# Data generated from Figure 4 run
python3 benchmarks/visualize.py --gs-dir benchmarks/grayscott/results/eval_L400_chunk4mb_ts100/balanced_w1-1-1 --view actions
```

**Existing visualizer functions:** `make_chunk_actions_figure()` and `make_milestone_actions_figure()` in `visualize.py`.

---

## Figure 6: Latency Breakdown (Overhead Analysis)

**Objective:** Answer "What is the overhead of NN-guided selection?" This is the ablation figure for overhead. Shows a stacked bar of where time is spent: stats kernel, NN inference, preprocessing, compression, exploration, SGD. SC reviewers need to see that NN overhead is small relative to compression time.

**Phases:** P3 (fixed-lz4), P4 (fixed-gdeflate), P5 (fixed-zstd), P6 (entropy-heuristic), P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D1 (Gray-Scott L=400) -- medium size for clear timing separation.

**Parameters:** `--L 400 --chunk-mb 4 --runs 5`

**Format:** Stacked bar chart. X-axis = phases. Y-axis = time (ms). Stacked components: stats_ms (green), nn_ms (orange), preproc_ms (purple), comp_ms (blue), explore_ms (red), sgd_ms (amber). Total annotated on top.

**Key insight:** NN inference adds ~0.22 ms/chunk. Compression takes ~6.7 ms/chunk. NN overhead is ~3.3% of total compression pipeline. Stats kernel is similarly small. The overhead is negligible compared to the I/O savings from better algorithm selection.

**Exact commands:**

```bash
# Data comes from Gray-Scott balanced run (Figure 1 run)
# Timing breakdown fields: nn_ms, stats_ms, preproc_ms, comp_ms, decomp_ms, explore_ms, sgd_ms
# Already in the aggregate CSV from run_gs_eval.sh
python3 benchmarks/visualize.py --gs-dir benchmarks/grayscott/results/eval_L400_chunk4mb_ts100/balanced_w1-1-1 --view summary
```

**Existing visualizer function:** `make_latency_breakdown_figure()` in `visualize.py`.

---

## Figure 7: Algorithm Selection Frequency Across Datasets

**Objective:** Answer "Do different datasets trigger different algorithm selections?" and "Are all 8 algorithms used, or does the NN degenerate to 2-3?" This directly addresses the concern raised in `roadMapToSC.md` Section 3.8.

**Phases:** P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50) -- only phases that perform selection

**Datasets:** D1, D3, D4, D5, D6 (all five)

**Parameters:** `--chunk-mb 4 --runs 5`

**Format:** Grouped bar chart. X-axis = algorithm+preprocessing configs (only configs actually selected). Grouped bars = datasets. Y-axis = selection frequency (% of chunks).

**Key insight:** Different datasets produce different selection distributions. Hurricane Isabel (smooth) favors zstd+shuf. Nyx (incompressible) favors lz4 (fastest). CESM-ATM (mixed) shows the most diversity. If some algorithms are never selected, the paper should acknowledge this honestly and discuss why.

**Exact commands:**

```bash
# Data comes from all benchmark runs (per-chunk CSVs contain action columns)
# Collect chunk CSVs from:
#   benchmarks/grayscott/results/eval_L400_chunk4mb_ts100/balanced_w1-1-1/benchmark_grayscott_vol_chunks.csv
#   benchmarks/vpic-kokkos/results/eval_NX156_chunk4mb_ts100/balanced_w1-1-1/benchmark_vpic_chunks.csv
#   benchmarks/sdrbench/results/eval_chunk4mb/balanced_w1-1-1/benchmark_*_chunks.csv

python3 benchmarks/visualize.py --view actions
# The make_algorithm_histogram() function handles multi-dataset input
```

**Existing visualizer function:** `make_algorithm_histogram()` in `visualize.py`.

---

## Figure 8: Policy Mode Comparison

**Objective:** Answer "Do the 4 cost model policies produce meaningfully different behavior?" This validates the policy-controlled optimization claim from CostModel.md. Shows that changing (alpha, beta, delta) weights shifts the algorithm selection and ratio/throughput tradeoff.

**Phases:** P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D4 (Hurricane Isabel) -- compressible enough to show meaningful differences between speed and ratio policies.

**Parameters:** Three cost model configurations run per dataset:

| Config | w0 (alpha, speed) | w1 (beta, I/O) | w2 (delta, ratio) | Label |
|--------|----|----|-------|-------|
| Speed-only | 1.0 | 1.0 | 0.0 | `speed_only_w1-1-0` |
| Balanced | 1.0 | 1.0 | 1.0 | `balanced_w1-1-1` |
| Ratio-only | 0.0 | 0.0 | 1.0 | `ratio_only_w0-0-1` |

**Format:** 2x2 subplot:
- (0,0): Bar chart of compression ratio per config (grouped by nn/nn-rl/nn-rl+exp50)
- (0,1): Bar chart of write throughput per config
- (1,0): Algorithm selection distribution per config (stacked bar)
- (1,1): Pareto scatter: ratio vs throughput, one point per (phase, config) combination

**Key insight:** Speed-only selects LZ4/Snappy (fast, low ratio). Ratio-only selects zstd+shuf (slow, high ratio). Balanced selects gdeflate+shuf (middle ground). The policies produce 2-5x difference in ratio at corresponding throughput differences. This validates that the cost model is not degenerate.

**Exact commands:**

```bash
# All 3 configs are already run by run_sdrbench_eval.sh:
RUNS=5 CHUNK_MB=4 bash benchmarks/sdrbench/run_sdrbench_eval.sh

# Results in:
#   benchmarks/sdrbench/results/eval_chunk4mb/balanced_w1-1-1/
#   benchmarks/sdrbench/results/eval_chunk4mb/ratio_only_w0-0-1/
#   benchmarks/sdrbench/results/eval_chunk4mb/speed_only_w1-1-0/

# NEW visualizer function needed: make_policy_comparison_figure()
# Input: 3 result directories for the same dataset
python3 benchmarks/visualize.py \
    --sdrbench-dir benchmarks/sdrbench/results/eval_chunk4mb/balanced_w1-1-1 \
    --sdrbench-dir benchmarks/sdrbench/results/eval_chunk4mb/ratio_only_w0-0-1 \
    --sdrbench-dir benchmarks/sdrbench/results/eval_chunk4mb/speed_only_w1-1-0
```

**Visualizer status:** Partially exists (summary per config). Needs a NEW `make_policy_comparison_figure()` that overlays configs for the same dataset.

---

## Figure 9: Oracle Regret Analysis

**Objective:** Answer "How close to optimal is the NN?" The oracle (exhaustive/best phase) tries all 32 configurations per chunk and picks the one with the best measured cost. Regret = (oracle_cost - nn_cost) / oracle_cost. This bounds the maximum possible improvement from a better selector.

**Phases:** P7 (best/oracle), P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D1 (Gray-Scott L=400), D4 (Hurricane Isabel)

**Parameters:**
- `--chunk-mb 4 --runs 1` (oracle is 32x slower, runs=1 is acceptable since oracle itself is an average over configs)
- `--phase best --phase nn --phase nn-rl --phase nn-rl+exp50`

**Format:** 2-panel figure:
- Left: CDF of per-chunk regret (X-axis = regret %, Y-axis = fraction of chunks). One curve per phase (nn, nn-rl, nn-rl+exp50). Target: 90th percentile regret < 10%.
- Right: Bar chart showing mean ratio achieved: oracle vs nn vs nn-rl vs nn-rl+exp50. Error bars.

**Key insight:** Target headline: "Median regret <5% with NN overhead of <0.5 ms per chunk." The oracle is 32x slower but only marginally better, demonstrating that the NN's selection quality justifies its speed.

**Exact commands:**

```bash
# Gray-Scott with oracle phase (slow -- ~30 min for L=400)
./build/grayscott_benchmark neural_net/weights/model.nnwt \
    --L 400 --chunk-mb 4 --timesteps 1 --runs 1 \
    --phase best --phase nn --phase nn-rl --phase nn-rl+exp50 \
    --w0 1.0 --w1 1.0 --w2 1.0 \
    --out-dir benchmarks/grayscott/results/oracle_L400

# Hurricane Isabel with oracle (slow -- ~20 min per field x 20 fields)
./build/generic_benchmark neural_net/weights/model.nnwt \
    --data-dir data/sdrbench/hurricane_isabel/100x500x500 \
    --dims 100,500,500 --ext .bin.f32 \
    --chunk-mb 4 --runs 1 \
    --phase best --phase nn --phase nn-rl --phase nn-rl+exp50 \
    --w0 1.0 --w1 1.0 --w2 1.0 \
    --out-dir benchmarks/sdrbench/results/oracle_hurricane
```

**Visualizer status:** NEEDS NEW `make_regret_figure()`. Must read per-chunk CSVs from both oracle and nn phases, compute per-chunk regret, and plot CDF + bar chart.

---

## Figure 10: Chunk-Size Scaling Study

**Objective:** Answer "How does performance vary with chunk size?" Exercises the per-chunk adaptation logic at different granularities. Larger chunks have better compression but less adaptation opportunity. Smaller chunks have more NN overhead proportionally.

**Phases:** P5 (fixed-zstd), P8 (nn), P9 (nn-rl)

**Datasets:** D1 (Gray-Scott L=640, ~1 GB) -- large enough to have many chunks at small chunk sizes.

**Parameters:**
- Chunk sizes: 1 MB, 2 MB, 4 MB, 8 MB, 16 MB, 32 MB, 64 MB
- `--runs 5 --timesteps 50`
- Balanced cost model

**Format:** 2-panel line plot:
- Left: X-axis = chunk size (MB, log scale), Y-axis = compression ratio. Lines for fixed-zstd, nn, nn-rl.
- Right: X-axis = chunk size, Y-axis = write throughput (MiB/s).
- Secondary Y-axis (right): NN overhead as % of total compression time.

**Key insight:** At very small chunks (1 MB), NN overhead is proportionally larger (~5-10% of comp time). At 8+ MB, overhead drops below 3%. Ratio should be relatively stable across chunk sizes. There is a sweet spot around 4-8 MB.

**Exact commands:**

```bash
for CHUNK in 1 2 4 8 16 32 64; do
    ./build/grayscott_benchmark neural_net/weights/model.nnwt \
        --L 640 --chunk-mb $CHUNK --timesteps 50 --runs 5 \
        --phase fixed-zstd --phase nn --phase nn-rl \
        --w0 1.0 --w1 1.0 --w2 1.0 \
        --out-dir benchmarks/grayscott/results/chunk_scaling_${CHUNK}mb
done
```

**Visualizer status:** NEEDS NEW `make_chunk_scaling_figure()`. Must aggregate across the 7 result directories.

---

## Figure 11: Data-Size Scaling Study

**Objective:** Answer "Does performance scale with dataset size?" Shows throughput and ratio as the 3D grid grows from 8 MB to 8 GB. Demonstrates that the system handles multi-GB datasets without degradation.

**Phases:** P1 (no-comp), P5 (fixed-zstd), P8 (nn), P9 (nn-rl)

**Datasets:** D1 at various L values:

| L | Dataset size | Chunks (4 MB) |
|---|-------------|----------------|
| 128 | 8 MB | 2 |
| 256 | 64 MB | 16 |
| 400 | 244 MB | 61 |
| 512 | 512 MB | 128 |
| 640 | 1 GB | 256 |
| 800 | 2 GB | 512 |
| 1000 | 4 GB | 1024 |

**Parameters:** `--chunk-mb 4 --runs 5 --timesteps 1`

**Format:** 2-panel line plot:
- Left: X-axis = dataset size (MB, log scale), Y-axis = write throughput (MiB/s). Lines for no-comp, fixed-zstd, nn, nn-rl.
- Right: X-axis = dataset size, Y-axis = compression ratio.

**Key insight:** Throughput should increase with dataset size (amortizing fixed overhead) and plateau at GPU memory bandwidth saturation. Ratio should be stable regardless of size for Gray-Scott data.

**Exact commands:**

```bash
for L in 128 256 400 512 640 800 1000; do
    ./build/grayscott_benchmark neural_net/weights/model.nnwt \
        --L $L --chunk-mb 4 --timesteps 1 --runs 5 \
        --phase no-comp --phase fixed-zstd --phase nn --phase nn-rl \
        --w0 1.0 --w1 1.0 --w2 1.0 \
        --out-dir benchmarks/grayscott/results/size_scaling_L${L}
done
```

**Visualizer status:** NEEDS NEW `make_size_scaling_figure()`. Must aggregate across result directories.

---

## Table 1: Comprehensive Comparison (THE MAIN RESULTS TABLE)

**Objective:** The canonical comparison table that every SC reviewer looks for. Shows all methods, all datasets, all key metrics in one place. This is the table that goes in the abstract/introduction as "Table 1".

**Phases:** All 10 phases

**Datasets:** All 5 primary datasets (D1, D3, D4, D5, D6)

**Parameters:** `--chunk-mb 4 --runs 5`, balanced cost model

**Format:**

| Method | Gray-Scott | | VPIC | | Hurricane | | Nyx | | CESM-ATM | |
|--------|-----|-----|------|-----|-----|-----|------|-----|------|------|
| | Ratio | GB/s | Ratio | GB/s | Ratio | GB/s | Ratio | GB/s | Ratio | GB/s |
| No compression | 1.00x | N/A | 1.00x | N/A | ... | | | | | |
| CPU zstd (level 3) | ?x | ? | ?x | ? | ... | | | | | |
| Fixed LZ4 | ?x | ? | ?x | ? | ... | | | | | |
| Fixed GDeflate | ?x | ? | ?x | ? | ... | | | | | |
| Fixed Zstd | ?x | ? | ?x | ? | ... | | | | | |
| Entropy Heuristic | ?x | ? | ?x | ? | ... | | | | | |
| Oracle (Exhaustive) | ?x | ? | ?x | ? | ... | | | | | |
| GPUCompress (NN) | ?x | ? | ?x | ? | ... | | | | | |
| GPUCompress (NN+SGD) | ?x | ? | ?x | ? | ... | | | | | |
| GPUCompress (NN+SGD+Exp) | ?x | ? | ?x | ? | ... | | | | | |

All values: mean +/- std from 5 runs. Bold = best in column (excluding oracle).

**Key insight:** GPUCompress (NN+SGD+Exp) should be within 5% of oracle on ratio, 10-50x faster than CPU zstd on throughput, and competitive with or better than the best fixed algorithm across all datasets.

**Data source:** Same runs as Figure 1.

---

## Table 2: NN Overhead Breakdown

**Objective:** Answer "Exactly how much does the NN add in latency?" Per-component timing breakdown showing that NN inference is a small fraction of total compression pipeline time.

**Phases:** P5 (fixed-zstd, no NN), P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D1 (Gray-Scott L=400)

**Parameters:** `--chunk-mb 4 --runs 5`

**Format:**

| Component | Fixed Zstd | NN (inference) | NN+SGD | NN+SGD+Exp |
|-----------|-----------|----------------|---------|------------|
| Stats kernel (ms) | 0 | ? | ? | ? |
| NN inference (ms) | 0 | ? | ? | ? |
| Preprocessing (ms) | 0 | ? | ? | ? |
| Compression (ms) | ? | ? | ? | ? |
| Exploration (ms) | 0 | 0 | 0 | ? |
| SGD update (ms) | 0 | 0 | ? | ? |
| **Total (ms)** | ? | ? | ? | ? |
| **NN overhead (%)** | 0 | ? | ? | ? |

**Key insight:** Target: "NN adds <5% overhead to the compression pipeline."

**Data source:** Per-component timing fields (nn_ms, stats_ms, preproc_ms, comp_ms, explore_ms, sgd_ms) from aggregate CSV.

---

## Table 3: Dataset Characteristics

**Objective:** Describe the datasets used in the evaluation so reviewers understand the diversity. Standard SC table.

**Format:**

| Dataset | Domain | Dimensions | Type | Per-field | Fields | Entropy (bits) | 2nd Deriv | Compressibility |
|---------|--------|-----------|------|-----------|--------|---------------|-----------|-----------------|
| Gray-Scott | Reaction-diffusion | 400^3 | float32 | 244 MB | 1 | ? | ? | High |
| VPIC | Plasma physics | 158^3 x 16 | float32 | 253 MB | 16 | ? | ? | Medium |
| Hurricane Isabel | Climate | 100x500x500 | float32 | 96 MB | 20 | ? | ? | Very High |
| Nyx | Cosmology | 512^3 | float32 | 512 MB | 6 | ? | ? | Low |
| CESM-ATM | Atmospheric | 1800x3600 | float32 | 25 MB | 78 | ? | ? | Variable |

Entropy and 2nd derivative values come from the stats kernel output (already in per-chunk CSVs).

**Data source:** Per-chunk diagnostics from any run. Extract min/max/mean entropy and second_derivative.

---

## Figure 12: Cross-Dataset Convergence Comparison

**Objective:** Answer "Does the convergence rate depend on the dataset?" Shows that online learning converges at different rates on different data -- VPIC (far from training distribution) converges slower than Hurricane Isabel (more typical). Demonstrates robustness.

**Phases:** P9 (nn-rl)

**Datasets:** D1 (Gray-Scott), D3 (VPIC), D4 (Hurricane Isabel) -- three datasets with different convergence profiles

**Parameters:** `--timesteps 100 --chunk-mb 4`, balanced cost model

**Format:** Single panel, X-axis = timestep, Y-axis = MAPE (ratio) %. Three lines, one per dataset. Horizontal reference at 20%.

**Key insight:** All datasets converge, but at different rates. Gray-Scott converges fastest (~5 timesteps) because it was likely closest to training data. VPIC converges slowest (~20 timesteps) because it is most dissimilar. This demonstrates the generality of online learning.

**Exact commands:**

```bash
# Data already generated from Figure 1 runs
# Extract timestep CSVs from each eval directory and overlay
# NEEDS NEW visualizer: make_cross_dataset_convergence_figure()
```

**Visualizer status:** NEEDS NEW function. Must read timestep CSVs from multiple directories and overlay.

---

## Figure 13: Pareto Frontier -- Ratio vs. Throughput

**Objective:** Answer "Where does GPUCompress sit on the compression ratio vs. throughput tradeoff?" Shows all methods as scatter points on a Pareto plane. The NN-guided methods should be on or near the Pareto frontier.

**Phases:** All phases including cpu-zstd

**Datasets:** D4 (Hurricane Isabel) -- most compressible, widest range of ratio/throughput tradeoffs

**Parameters:** `--chunk-mb 4 --runs 5`

**Format:** Scatter plot. X-axis = compression ratio (log scale). Y-axis = compression throughput (GB/s, log scale). Each method is a labeled point with error bars in both dimensions. Draw the Pareto frontier line connecting non-dominated points.

**Key insight:** CPU zstd is in the lower-left (low throughput, moderate ratio). Fixed LZ4 is upper-left (high throughput, low ratio). Fixed zstd is lower-right (low throughput, high ratio). GPUCompress (nn-rl) should be on the Pareto frontier, offering the best ratio at its throughput level.

**Data source:** Aggregate CSVs from Figure 1 runs.

**Existing visualizer:** Partially exists in the summary figure's bottom-right panel (`plot_bars` creates a Pareto scatter). Needs to be extracted as a standalone publication figure with error bars and frontier line.

---

## Table 4: Cold-Start Behavior

**Objective:** Answer "What happens on the very first write to completely new data?" Quantifies the cold-start penalty and recovery.

**Phases:** P9 (nn-rl), P10 (nn-rl+exp50)

**Datasets:** D3 (VPIC), D5 (Nyx), D6 (CESM-ATM) -- three datasets likely far from training distribution

**Format:**

| Dataset | Phase | T=1 MAPE | T=5 MAPE | T=10 MAPE | T=20 MAPE | T=50 MAPE | T=100 MAPE |
|---------|-------|----------|----------|-----------|-----------|-----------|------------|
| VPIC | nn-rl | ? | ? | ? | ? | ? | ? |
| VPIC | nn-rl+exp50 | ? | ? | ? | ? | ? | ? |
| Nyx | nn-rl | ? | ? | ? | ? | ? | ? |
| ... | | | | | | | |

**Key insight:** Cold-start MAPE is 200-3000% but drops below 20% within 10-20 timesteps on all datasets. The "first 5 timesteps" penalty is the cost of generality.

**Data source:** Timestep CSVs from Figure 1/2 runs.

---

## Figure 14: NN Inference vs. Heuristic vs. Oracle (Ablation)

**Objective:** Answer "Is the NN actually better than a simple rule-based selector?" This is the ablation that isolates the NN's contribution. Compare: (a) entropy heuristic (no NN, just thresholds), (b) NN inference (pretrained, no learning), (c) NN + SGD (adapted), (d) oracle (optimal).

**Phases:** P6 (entropy-heuristic), P8 (nn), P9 (nn-rl), P10 (nn-rl+exp50), P7 (best/oracle)

**Datasets:** D1, D3, D4 (three datasets)

**Parameters:** `--chunk-mb 4 --runs 5`

**Format:** Grouped bar chart. X-axis = datasets. Grouped bars = methods. Y-axis = compression ratio. Horizontal line = oracle ratio.

**Key insight:** The entropy heuristic achieves ~70-80% of oracle ratio. Pretrained NN (cold-start) may be worse than heuristic on unseen data. NN+SGD (adapted) achieves ~95% of oracle. This proves: (1) the NN adds value over simple rules, and (2) online learning is essential for generalization.

**Data source:** Same runs as Figure 1 + oracle runs from Figure 9.

---

## Table 5: Feature Comparison vs. Related Work

**Objective:** Position GPUCompress against the state of the art. Standard SC comparison table.

**Format:**

| Feature | GPUCompress | nvCOMP | cuSZ | ZFP | SZ3 | LibPressio |
|---------|-------------|--------|------|-----|-----|------------|
| GPU-native compression | Yes | Yes | Yes | Yes | No | Wrapper |
| Adaptive algorithm selection | NN-guided | Manual | N/A | N/A | N/A | Autotuning |
| Online learning | SGD | No | No | No | No | Offline |
| HDF5 integration | VOL connector | No | No | No | Yes | Yes |
| # lossless algorithms | 8 | 8 | 0 | 0 | 0 | Many |
| Preprocessing (shuffle/quant) | Auto | Manual | N/A | N/A | N/A | Manual |
| Policy-controlled cost model | Yes | No | No | No | No | No |
| Per-chunk adaptation | Yes | No | No | No | No | No |

This is a manually constructed table (no benchmark run needed). Requires literature review.

---

## Figure 15: Lossy Rate-Distortion Curves (STRETCH GOAL)

**Objective:** Answer "How does GPUCompress perform in the lossy regime?" Shows compression ratio vs. PSNR at different error bounds. The NN should select different algorithms at different error bounds.

**Phases:** P8 (nn), P9 (nn-rl)

**Datasets:** D4 (Hurricane Isabel) -- smooth, meaningful lossy compression

**Parameters:** Error bound sweep: lossless (0.0), 1e-2, 1e-3, 1e-4, 1e-5
- `--chunk-mb 4 --runs 5`
- For each error bound, pass via HDF5 filter cd_values

**Format:** Line plot. X-axis = PSNR (dB). Y-axis = compression ratio (log scale). Each error bound is a point on the curve. Lines for nn, nn-rl. If cuSZ/ZFP baselines available, overlay them.

**Key insight:** At error_bound=1e-3, ratio jumps from ~2x (lossless) to ~10-20x with PSNR > 60 dB. The NN selects different preprocessing (quantization enabled) at different error bounds.

**Exact commands:**

```bash
for EB in 0.0 1e-5 1e-4 1e-3 1e-2; do
    ./build/generic_benchmark neural_net/weights/model.nnwt \
        --data-dir data/sdrbench/hurricane_isabel/100x500x500 \
        --dims 100,500,500 --ext .bin.f32 \
        --chunk-mb 4 --runs 5 \
        --error-bound $EB \
        --phase nn --phase nn-rl \
        --w0 1.0 --w1 1.0 --w2 1.0 \
        --out-dir benchmarks/sdrbench/results/lossy_eb_${EB}
done
```

**NOTE:** Requires `--error-bound` CLI flag to be propagated through DCPL. Verify this exists in `generic-benchmark.cu`. If not, it must be added.

**Visualizer status:** NEEDS NEW `make_rate_distortion_figure()`.

---

## Summary of New Visualizer Functions Needed

| Function | Figure | Priority |
|----------|--------|----------|
| `make_policy_comparison_figure()` | Fig 8 | P1 |
| `make_regret_figure()` | Fig 9 | P1 |
| `make_chunk_scaling_figure()` | Fig 10 | P2 |
| `make_size_scaling_figure()` | Fig 11 | P2 |
| `make_cross_dataset_convergence_figure()` | Fig 12 | P1 |
| `make_rate_distortion_figure()` | Fig 15 | P2 (stretch) |

---

## Execution Schedule

### Phase A: Full-Scale Core Runs (Days 1-3)

These produce data for Figures 1, 2, 3, 4, 5, 6, 7, 12, 13, Tables 1, 2, 3, 4.

```bash
# Day 1: SDRBench + CPU baseline (overnight)
RUNS=5 CHUNK_MB=4 bash benchmarks/sdrbench/run_sdrbench_eval.sh
RUNS=5 CHUNK_MB=4 bash benchmarks/sdrbench/run_cpu_zstd_baseline.sh

# Day 2: Gray-Scott + VPIC (overnight)
L=400 TIMESTEPS=100 RUNS=5 bash benchmarks/grayscott/run_gs_eval.sh
NX=156 TIMESTEPS=100 RUNS=5 bash benchmarks/vpic-kokkos/run_vpic_eval.sh

# Day 3: Oracle runs (slow, daytime)
# Gray-Scott oracle
./build/grayscott_benchmark neural_net/weights/model.nnwt \
    --L 400 --chunk-mb 4 --timesteps 1 --runs 1 \
    --phase best --phase nn \
    --out-dir benchmarks/grayscott/results/oracle_L400

# Hurricane oracle
./build/generic_benchmark neural_net/weights/model.nnwt \
    --data-dir data/sdrbench/hurricane_isabel/100x500x500 \
    --dims 100,500,500 --ext .bin.f32 \
    --chunk-mb 4 --runs 1 --phase best --phase nn \
    --out-dir benchmarks/sdrbench/results/oracle_hurricane
```

### Phase B: Scaling Studies (Days 3-4)

Data for Figures 10, 11.

```bash
# Chunk-size scaling (L=640, 7 chunk sizes)
for CHUNK in 1 2 4 8 16 32 64; do
    ./build/grayscott_benchmark neural_net/weights/model.nnwt \
        --L 640 --chunk-mb $CHUNK --timesteps 50 --runs 5 \
        --phase fixed-zstd --phase nn --phase nn-rl \
        --out-dir benchmarks/grayscott/results/chunk_scaling_${CHUNK}mb
done

# Data-size scaling (7 L values)
for L in 128 256 400 512 640 800 1000; do
    ./build/grayscott_benchmark neural_net/weights/model.nnwt \
        --L $L --chunk-mb 4 --timesteps 1 --runs 5 \
        --phase no-comp --phase fixed-zstd --phase nn --phase nn-rl \
        --out-dir benchmarks/grayscott/results/size_scaling_L${L}
done
```

### Phase C: Policy + Ablation (Day 4)

Data for Figures 8, 14. (Policy data already comes from Phase A since eval scripts run 3 configs.)

### Phase D: Visualization + New Plot Functions (Days 5-6)

Write the 6 new visualizer functions. Generate all figures.

### Phase E: Lossy Sweep (Days 6-7, stretch)

Data for Figure 15.

---

## Checklist of SC Reviewer Questions Answered

| # | Reviewer Question | Answered By |
|---|-------------------|-------------|
| 1 | "Does this generalize beyond one dataset?" | Fig 1, Table 1 (5 datasets) |
| 2 | "Is it better than just always using zstd?" | Fig 1, Table 1 (fixed baselines) |
| 3 | "Is it better than CPU compression?" | Table 1 (cpu-zstd row) |
| 4 | "How close to optimal is the NN?" | Fig 9, Fig 14 (oracle regret) |
| 5 | "Does the NN beat a simple heuristic?" | Fig 14 (entropy-heuristic vs NN) |
| 6 | "What happens on completely new data?" | Fig 2, Table 4 (cold-start + convergence) |
| 7 | "What was the model trained on?" | Table 3 + text section (training data disclosure) |
| 8 | "Do the policy modes actually work?" | Fig 8 (3 configs, different distributions) |
| 9 | "What is the NN overhead?" | Fig 6, Table 2 (latency breakdown) |
| 10 | "Does it scale with data size?" | Fig 11 (8 MB to 4 GB) |
| 11 | "How does chunk size affect performance?" | Fig 10 (1 MB to 64 MB) |
| 12 | "Which algorithms get selected?" | Fig 5, Fig 7 (heatmaps + histograms) |
| 13 | "Where does time go?" | Fig 6, Table 2 (stacked bar + table) |
| 14 | "What is the convergence rate?" | Fig 2, Fig 12 (MAPE curves) |
| 15 | "How does it compare to related work?" | Table 5 (feature matrix) |

---

## What This Plan Does NOT Cover (Known Limitations to Acknowledge)

1. **Multi-GPU / Multi-Node:** Not evaluated. Acknowledge as future work. The 9-slot context pool and HDF5 VOL connector support multi-dataset parallelism but not MPI-parallel I/O.

2. **Second GPU Architecture:** All results are on A100 (sm_80). Acknowledge. The nvCOMP backend supports V100/H100 but no H100 results are available.

3. **Float64 Data:** The stats kernel is float32-only. Miranda (float64 turbulence) was skipped. Acknowledge as a limitation.

4. **External Lossy Compressors (cuSZ, ZFP):** Not yet integrated. Table 5 covers feature comparison but not performance comparison. This is the biggest remaining gap for a strong accept.

5. **Energy Measurement:** Not included. Increasingly expected at SC but not yet a hard requirement.

6. **Real Application Coupling:** The benchmarks use standalone drivers, not actual simulation codes (except Gray-Scott). The HDF5 VOL connector is production-quality but end-to-end application integration benchmarks (e.g., total simulation time with/without GPUCompress) are not included.
