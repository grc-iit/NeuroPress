# SC Paper Benchmark Gaps Analysis

Assessment of what the GPUCompress benchmark suite needs for a Supercomputing (SC) conference paper submission.

## Current Strengths

- NN-guided algorithm selection with GPU-native online learning (novel contribution)
- Zero-copy fused stats-to-NN inference pipeline
- 4-phase comparison: baseline, NN, NN+SGD, NN+SGD+exploration
- Per-chunk diagnostic instrumentation with timing breakdown
- Multi-timestep convergence tracking (SGD learning curve)
- HDF5 VOL integration (relevant to HPC community)
- Bitwise correctness verification
- Configurable CLI with LR sweep infrastructure

## Critical Gaps (Likely Desk-Reject Without These)

### 1. Dataset Diversity

The benchmark only uses Gray-Scott (one reaction-diffusion pattern). SC reviewers will ask: "Does this generalize?"

**Required:**
- Minimum 3-4 diverse scientific datasets with different entropy/smoothness profiles
- Candidates: VPIC (plasma physics), Nyx (cosmology), Hurricane Isabel (climate), HACC (N-body)
- The codebase already has `vpic_adapter` and `vpic-kokkos` benchmark directory but no VPIC benchmark driver comparable to the Gray-Scott one
- Include at least one real experimental dataset (not just simulation)
- Vary data characteristics: smooth fields, turbulent fields, sparse data, high-entropy data

### 2. Comparison Against State-of-the-Art Compressors

No comparison with established GPU/CPU compressors. SC reviewers will reject without this.

**Required:**
- Compare against: SZ3, ZFP, cuSZ, MGARD (GPU-accelerated lossy compressors)
- Also compare against: standalone nvCOMP algorithms with fixed best-algorithm selection
- Same dataset, same error bound, compare:
  - Compression ratio
  - Compression throughput (MB/s)
  - Decompression throughput (MB/s)
  - Reconstruction quality (PSNR, max pointwise error)
- Show that NN-based automatic selection matches or beats hand-tuned algorithm choice
- Show overhead of NN selection vs. just always using the single best algorithm

### 3. Statistical Rigor

Runs appear to be single-shot. No error bars or confidence intervals.

**Required:**
- Minimum 3 runs per configuration (5 preferred)
- Report mean +/- standard deviation for all timing measurements
- GPU timing has inherent variance; single-run numbers are not publishable
- Consider median + IQR for skewed distributions (e.g., first-chunk warmup)

## Important Gaps (Reviewers Will Raise These)

### 4. Lossy Compression Evaluation

The benchmark defaults to lossless (error_bound=0.0). The NN supports lossy quantization but it is not benchmarked systematically.

**Required:**
- Error-bound sweep: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, lossless
- Rate-distortion plots: compression ratio vs PSNR at each error bound
- Show NN correctly selects different algorithms at different error bounds
- Compare lossy results against SZ/ZFP at matching error bounds

### 5. Ablation Study

No isolated measurement of what each component contributes.

**Required:**
- NN inference overhead: how many microseconds does inference add per chunk?
- Comparison: NN selection vs. brute-force trying all 32 configs vs. oracle (best possible)
- SGD convergence benefit: MAPE reduction over timesteps, broken down by metric
- Exploration benefit: how often does exploration find a better action than NN predicted?
- Preprocessing benefit: byte shuffle and quantization contribution to ratio
- Show: "NN adds X microseconds overhead but saves Y ms in I/O due to better algorithm choice"

### 6. Training Data and Generalization Analysis

No discussion of what training data the NN was trained on or how it generalizes.

**Required:**
- Describe training dataset composition (which files, how many configs benchmarked)
- Cross-domain transfer results: train on dataset A, test on dataset B
- Show OOD detection in action: what happens when data is very different from training?
- Report: fraction of chunks where NN's first choice is optimal vs. suboptimal

## Strengthening Additions (Would Improve Chances)

### 7. Scaling Study

Single-GPU only. No formal scaling analysis.

**Recommended:**
- Strong scaling: fix data size, increase chunk parallelism (already have 8 worker slots)
- Data size scaling: L=128 (8 MB) through L=1260 (8 GB), show throughput vs. size
- Frame existing data as a scaling study with proper axis labels and analysis
- Multi-GPU results would be very strong (if feasible with VOL connector)
- Multi-node with MPI-parallel HDF5 would be exceptional

### 8. Roofline / Performance Model

No bandwidth or compute bound analysis.

**Recommended:**
- Measure achieved fraction of peak GPU memory bandwidth
- Identify bottleneck at each data size: NN inference? nvCOMP kernel? I/O?
- Compare: time in stats computation vs. NN inference vs. compression vs. I/O
- The per-chunk timing breakdown already exists; just needs analysis and visualization

### 9. End-to-End Application Integration

Running a standalone benchmark is good, but SC values real simulation coupling.

**Recommended:**
- Show: simulation running with in-situ compression via VOL, total application slowdown
- Time-to-solution comparison: simulation+I/O with vs. without compression
- Compute the I/O time fraction reduction: e.g., "compression reduces I/O from 40% to 3% of total runtime"

### 10. Additional Metrics

**Recommended:**
- Decompression throughput breakdown (currently only aggregate read_ms)
- Energy/power measurement (increasingly expected at SC)
- NN inference cost vs. brute-force: "NN takes 3 microseconds vs. 50 ms to try all 32"
- Memory overhead: per-context GPU allocation (already documented as ~76 KB per slot)

## Benchmark Code Issues (from SC reviewer pass)

### Issue #1: MAPE/sMAPE Labeling Inconsistency

The single-shot path computes **sMAPE** (symmetric) but stores it in fields and CSV columns labeled `mape_*`. The multi-timestep path computes both sMAPE and real MAPE correctly into separate columns (`smape_*` vs `mape_*`), but the single-shot aggregate CSV uses `mape_ratio_pct` for what is actually sMAPE.

**Decision:** Use MAPE for ratio predictions (ratios are always >1, no near-zero instability). Use sMAPE for timing predictions (comp_time, decomp_time) where sub-millisecond actuals cause MAPE to blow up. Label both correctly in CSV headers, code, and paper text. Add a sentence in the paper justifying sMAPE for timing: "compression kernel times range from 0.1–50 ms; standard MAPE is unstable for sub-millisecond actuals."

**Fix applied:** Both sMAPE and MAPE are now computed in the single-shot path, matching multi-timestep behavior. CSV columns clearly labeled `smape_*` and `mape_*`. Console output prints both. Per-chunk CSV also reports both metrics per chunk. Multi-timestep summary overwrite now correctly populates both `smape_*` and `mape_*` result fields.

### Issue #2: High Decomp MAPE Despite Deferred SGD

~~Initially appeared that batched decomp SGD was never called.~~ Investigation shows the VOL connector already calls `gpucompress_record_chunk_decomp_ms()` per-chunk (H5VLgpucompress.cu:1787) and `gpucompress_batched_decomp_sgd()` after each H5Dread (H5VLgpucompress.cu:1831). So the decomp SGD pipeline IS wired in — the persistently high decomp MAPE (~154%) indicates the head-only SGD is not converging, likely because decompression times are very small and noisy (sub-ms), making prediction inherently difficult.

**Action:** Investigate whether the noise gate threshold (0.05 in normalized space) is filtering out too many samples. Consider whether decomp MAPE should be de-emphasized in the paper since decompression time is less decision-relevant than ratio and compression time for algorithm selection.

### Issue #3: Single-Shot Phases Include Cold-Start Costs

First write pays one-time costs (nvCOMP manager creation, first NN inference JIT, VOL buffer allocation). Multi-timestep mode correctly skips 5 warmup iterations. Single-shot phases had no warmup.

**Investigation:** `gpucompress_init()` pre-allocates CompContext infrastructure but not nvCOMP managers (lazy, LRU-3 cache) or VOL write buffers (per-dataset, allocated on first H5Dwrite). Phase ordering does NOT provide cross-phase warmup since each phase creates separate HDF5 files/datasets with independent VOL state. The nvCOMP manager cache in CompContext does persist across phases but starts empty.

**Fix applied:** Added a discard warmup write cycle before the timed run in both `run_phase_nocomp` and `run_phase_vol`. The warmup primes nvCOMP manager cache (persists in CompContext LRU) and NN inference kernel JIT (cached per-process by CUDA). VOL write buffers are per-dataset so not reused, but their allocation cost is negligible (~microseconds) compared to manager creation.

### Issue #4: No Isolated Compression Throughput

`write_mbps` includes VOL overhead, chunking, D→H staging, and disk I/O — not pure compression throughput. The per-chunk `compression_ms` was collected but not surfaced as a separate throughput metric.

**Fix applied:** Added `comp_gbps` and `decomp_gbps` fields (isolated nvCOMP kernel throughput in GB/s) computed from cumulative per-chunk timing. Added full timing breakdown to aggregate CSV (nn_ms, stats_ms, preproc_ms, comp_ms, decomp_ms, explore_ms, sgd_ms). Added Comp (GB/s) column to console summary table. SC paper can now report both end-to-end I/O throughput (write_mibps) and isolated compression throughput (comp_gbps) separately.

### Issue #5: no-comp Baseline Includes D→H Copy

The no-comp phase uses VOL fallback (D→H + native HDF5), so its throughput reflects `cudaMemcpy + fwrite`, not true storage bandwidth.

**Investigation:** Both compressed and uncompressed paths go through the same VOL connector. The compressed path also pays D→H transfer cost, but for smaller (compressed) chunks. The throughput difference is genuine — it reflects the real benefit of GPU compression: reduced D→H volume + reduced disk I/O. This is actually the correct apples-to-apples comparison since both paths use identical HDF5/VOL infrastructure.

**Fix applied:** Added detailed documentation comment in the benchmark explaining what no-comp measures and why it's the correct baseline. No code logic change needed — the comparison is fair as-is. The paper should describe this as "uncompressed baseline through the same VOL connector stack."

### Issue #6: Single Run, No Error Bars

All measurements were single execution. GPU timing fluctuates 5–15% between runs.

**Fix applied:** Added `--runs N` flag. Each single-shot phase runs N times (up to 32). Results are merged: timing fields report mean, `_std` columns report sample standard deviation. For learning phases (nn-rl, nn-rl+exp50), NN weights are reloaded before each run to ensure independence. Aggregate CSV includes `n_runs`, `write_ms_std`, `read_ms_std`, `comp_gbps_std`, `decomp_gbps_std`. Default `--runs 1` preserves existing behavior.

## Priority Roadmap

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Dataset diversity (3+ datasets) | High | Required |
| P0 | SZ/ZFP/cuSZ comparison | High | Required |
| P0 | Error bars (--runs N) | Low | Required — DONE |
| P0 | Fix MAPE/sMAPE labeling (Issue #1) — DONE | Low | Required |
| P1 | Investigate decomp MAPE convergence (Issue #2) | Medium | Expected |
| P1 | Lossy rate-distortion curves | Medium | Expected |
| P1 | Ablation study | Medium | Expected |
| P1 | Training data and generalization | Medium | Expected |
| P1 | Add warmup for single-shot (Issue #3) — DONE | Low | Expected |
| P1 | Report isolated compression throughput (Issue #4) — DONE | Low | Expected |
| P2 | Data size scaling analysis | Low | Strengthening |
| P2 | Roofline analysis | Medium | Strengthening |
| P2 | End-to-end application coupling | High | Strengthening |
| P2 | Document no-comp baseline caveat (Issue #5) — DONE | Low | Strengthening |
| P3 | Multi-GPU / multi-node | Very High | Exceptional |
| P3 | Energy measurement | Medium | Trending |

## Existing Infrastructure to Leverage

- `benchmarks/vpic-kokkos/`: VPIC benchmark skeleton exists
- `benchmarks/synthetic/`: Synthetic data benchmark exists
- `benchmarks/nn_adaptiveness/`: NN adaptiveness tests exist
- `src/vpic/vpic_adapter.cu`: VPIC data adapter already implemented
- Per-chunk diagnostics already capture all timing and prediction data
- `visualize.py` already generates multi-panel SC-style figures
- `sweep_lr.sh` shows how to automate parameter sweeps
