# Benchmark Timing Audit

Date: 2026-04-04

Scope:
- [benchmarks/grayscott/grayscott-benchmark-pm.cu](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu)
- [benchmarks/sdrbench/generic-benchmark.cu](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu)
- [benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx)
- [src/hdf5/H5VLgpucompress.cu](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu)
- [src/api/gpucompress_compress.cpp](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp)
- [src/api/gpucompress_diagnostics.cpp](/home/cc/GPUCompress/src/api/gpucompress_diagnostics.cpp)
- [src/nn/nn_gpu.cu](/home/cc/GPUCompress/src/nn/nn_gpu.cu)

## Timing Audit Summary

The benchmark drivers are mostly capturing the important wall-clock timings correctly. Across Gray-Scott, SDRBench, and VPIC, `write_ms` starts before `H5Dwrite`, stops after `cudaDeviceSynchronize()`, `H5Dclose()`, and `H5Fclose()`, and `read_ms` similarly waits for GPU completion before stopping. That means the top-line write/read timings are flushed end-to-end timings, not premature async snapshots.

The shared raw-vs-clamped split is also implemented correctly in the core library. Compression and decompression diagnostics keep both a clamped 5 ms floor for MAPE / cost-model comparisons and an unclamped raw event time for latency breakdown and throughput. The NN cost model uses the same 5 ms clamp.

The main problems are semantic rather than synchronization bugs:
- Aggregate and timestep CSVs still label raw cumulative chunk sums as plain `comp_ms` and `decomp_ms`.
- `explore_ms` and especially `sgd_ms` can be misread as pipeline wall-clock contributions when they are not.
- VPIC aggregate CSV hardcodes throughput std-dev to zero instead of computing it.
- Hardcoded `/tmp` output paths keep I/O results deployment-dependent; on tmpfs systems they are not meaningful disk-throughput numbers.

## Critical Issues

### P0: CSV headers still blur raw timing and cumulative semantics

Affected files:
- [benchmarks/grayscott/grayscott-benchmark-pm.cu#L785](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L785)
- [benchmarks/grayscott/grayscott-benchmark-pm.cu#L1304](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L1304)
- [benchmarks/sdrbench/generic-benchmark.cu#L1060](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1060)
- [benchmarks/sdrbench/generic-benchmark.cu#L1653](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1653)
- [benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L903](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L903)
- [benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1194](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1194)

What is wrong:
- These CSV headers expose `comp_ms` and `decomp_ms`, but the values written are sums of `compression_ms_raw` and `decompression_ms_raw` across chunks, not clamped timings and not pipeline wall-clock.
- The same files expose `explore_ms` and `sgd_ms` without saying they are cumulative per-chunk totals rather than write-path wall-clock slices.

Evidence:
- Gray sums raw chunk timings at [grayscott-benchmark-pm.cu#L704](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L704) and [grayscott-benchmark-pm.cu#L705](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L705), then writes them to `comp_ms,decomp_ms,explore_ms,sgd_ms` at [grayscott-benchmark-pm.cu#L788](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L788) and [grayscott-benchmark-pm.cu#L815](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L815). The timestep path does the same at [grayscott-benchmark-pm.cu#L1503](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L1503) to [grayscott-benchmark-pm.cu#L1506](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L1506), then writes them at [grayscott-benchmark-pm.cu#L1573](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L1573) and [grayscott-benchmark-pm.cu#L1583](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L1583).
- SDRBench collects raw sums in [generic-benchmark.cu#L564](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L564) and [generic-benchmark.cu#L565](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L565), stores them in `r->comp_ms` and `r->decomp_ms` at [generic-benchmark.cu#L579](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L579) and [generic-benchmark.cu#L580](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L580), and exports them under plain `comp_ms,decomp_ms` at [generic-benchmark.cu#L1063](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1063) and [generic-benchmark.cu#L1094](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1094). The multi-field timestep CSV repeats that at [generic-benchmark.cu#L1658](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1658) and [generic-benchmark.cu#L1906](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L1906).
- VPIC sums raw chunk timings at [vpic_benchmark_deck.cxx#L1581](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1581) and [vpic_benchmark_deck.cxx#L1582](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1582), but labels them as plain `comp_ms,decomp_ms` in the timestep header at [vpic_benchmark_deck.cxx#L1199](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1199) and aggregate header at [vpic_benchmark_deck.cxx#L906](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L906).

Impact:
- No arithmetic bug in the raw values themselves.
- High interpretation risk: readers can incorrectly compare `comp_ms` or `sgd_ms` against `write_ms` as if all were additive wall-clock contributions.
- In pipelined runs this can make the breakdown look impossible or suggest double-counting when the real issue is labeling.

### P1: `sgd_ms` is not GPU SGD cost; it mostly measures host-side dispatch

Affected files:
- [src/api/gpucompress_compress.cpp#L555](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L555)
- [src/api/gpucompress_compress.cpp#L842](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L842)
- [src/api/gpucompress_compress.cpp#L874](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L874)
- [src/nn/nn_gpu.cu#L1930](/home/cc/GPUCompress/src/nn/nn_gpu.cu#L1930)

What is wrong:
- `sgd_ms` is measured with `std::chrono::steady_clock` around calls to `runNNSGDCtx`.
- `runNNSGDCtx` explicitly launches the SGD kernel on `g_sgd_stream` in fire-and-forget mode and returns without synchronizing that stream.

Evidence:
- Phase-1 and phase-2 SGD timing uses host wall-clock in [gpucompress_compress.cpp#L555](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L555) to [gpucompress_compress.cpp#L581](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L581) and [gpucompress_compress.cpp#L847](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L847) to [gpucompress_compress.cpp#L879](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L879).
- `runNNSGDCtx` launches on `g_sgd_stream` and records completion, then documents "fire-and-forget" with no stream sync at [nn_gpu.cu#L1986](/home/cc/GPUCompress/src/nn/nn_gpu.cu#L1986) to [nn_gpu.cu#L1990](/home/cc/GPUCompress/src/nn/nn_gpu.cu#L1990).

Impact:
- `sgd_ms` materially underreports actual GPU work if interpreted as SGD compute time.
- It is still useful as host overhead, but it must not be plotted or discussed as SGD kernel latency.

### P1: VPIC aggregate CSV zeroes throughput std-dev

Affected file:
- [benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1113](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1113)

What is wrong:
- VPIC aggregate CSV header advertises `comp_gbps_std,decomp_gbps_std`, but the writer emits literal `0.0000,0.0000`.

Evidence:
- Header at [vpic_benchmark_deck.cxx#L911](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L911).
- Literal zeros at [vpic_benchmark_deck.cxx#L1113](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1113).

Impact:
- This is a CSV integrity issue, not a timing-capture bug.
- It makes VPIC look more stable than Gray-Scott and SDRBench and prevents cross-driver comparisons of isolated throughput variability.

### P1: `/tmp` still makes I/O conclusions deployment-dependent

Affected files:
- [benchmarks/grayscott/grayscott-benchmark-pm.cu#L119](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L119)
- [benchmarks/sdrbench/generic-benchmark.cu#L96](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L96)
- [benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L86](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L86)

What is wrong:
- All drivers write benchmark files to `/tmp`.
- VPIC explicitly documents that `/tmp` is typically tmpfs and that `drop_pagecache()` is ineffective there.

Evidence:
- The hardcoded `/tmp` paths are obvious in each driver.
- On the current machine, `stat -f -c %T /tmp` returned `ext2/ext3`, so this environment is not obviously tmpfs. But the code remains deployment-sensitive, and the VPIC comment already warns about tmpfs behavior.

Impact:
- On systems where `/tmp` is tmpfs, write/read throughput numbers are not storage throughput; they are effectively memory-path results plus HDF5 overhead.
- On this machine, the risk is lower, but the benchmark remains non-portable as an I/O benchmark.

## Evidence

### What is measured correctly

- Write wall-clock includes post-write sync and close:
  - Gray-Scott: [grayscott-benchmark-pm.cu#L647](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L647) to [grayscott-benchmark-pm.cu#L653](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L653)
  - SDRBench: [generic-benchmark.cu#L754](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L754) to [generic-benchmark.cu#L760](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L760)
  - VPIC: [vpic_benchmark_deck.cxx#L1482](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1482) to [vpic_benchmark_deck.cxx#L1492](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1492)

- Read wall-clock also waits for GPU completion before stopping:
  - Gray-Scott: [grayscott-benchmark-pm.cu#L667](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L667) to [grayscott-benchmark-pm.cu#L673](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L673)
  - SDRBench: [generic-benchmark.cu#L779](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L779) to [generic-benchmark.cu#L784](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L784)
  - VPIC: [vpic_benchmark_deck.cxx#L1531](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1531) to [vpic_benchmark_deck.cxx#L1536](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1536)

- VOL overlap metrics are implemented correctly:
  - Stage timing globals and meanings: [H5VLgpucompress.cu#L397](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L397) to [H5VLgpucompress.cu#L404](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L404)
  - Export API: [H5VLgpucompress.cu#L438](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L438) to [H5VLgpucompress.cu#L451](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L451)
  - `s_stage1_ms`, `s_drain_ms`, `s_s2_busy_ms`, `s_io_drain_ms`, `s_s3_busy_ms`, `s_total_ms`: [H5VLgpucompress.cu#L1637](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L1637) to [H5VLgpucompress.cu#L1692](/home/cc/GPUCompress/src/hdf5/H5VLgpucompress.cu#L1692)

- Raw vs clamped timing split is correct in the library:
  - Compression clamp/raw split: [gpucompress_compress.cpp#L428](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L428) to [gpucompress_compress.cpp#L431](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L431)
  - Decompression clamp/raw split: [gpucompress_diagnostics.cpp#L74](/home/cc/GPUCompress/src/api/gpucompress_diagnostics.cpp#L74) to [gpucompress_diagnostics.cpp#L78](/home/cc/GPUCompress/src/api/gpucompress_diagnostics.cpp#L78)
  - NN policy clamp: [nn_gpu.cu#L185](/home/cc/GPUCompress/src/nn/nn_gpu.cu#L185) to [nn_gpu.cu#L188](/home/cc/GPUCompress/src/nn/nn_gpu.cu#L188)
  - Cost model clamp consistency: [gpucompress_compress.cpp#L521](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L521) to [gpucompress_compress.cpp#L542](/home/cc/GPUCompress/src/api/gpucompress_compress.cpp#L542)

- Diagnostics and quality collection occur after timed write/read regions, not inside them:
  - Gray-Scott: [grayscott-benchmark-pm.cu#L678](/home/cc/GPUCompress/benchmarks/grayscott/grayscott-benchmark-pm.cu#L678) onward
  - SDRBench: [generic-benchmark.cu#L787](/home/cc/GPUCompress/benchmarks/sdrbench/generic-benchmark.cu#L787) onward
  - VPIC: [vpic_benchmark_deck.cxx#L1544](/home/cc/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx#L1544) onward

## Impact Analysis

- The top-line `write_ms`, `read_ms`, `write_mibps`, and `read_mibps` are credible for the path they bracket. I do not see a missing `cudaDeviceSynchronize()` bug in the current drivers.
- The raw component timings are also credible as sums of chunk-local work, but they are not additive with pipeline wall-clock. Interpreting them as stacked contributors to `write_ms` will overstate cost, sometimes by a large margin, because Stage 2 and Stage 3 overlap and because `comp_ms` is a sum across chunks while `vol_s2_busy_ms` is the bottleneck worker wall-clock.
- `sgd_ms` is the most misleading column. In NN phases it can look tiny because it excludes most GPU SGD execution. If someone uses it to claim SGD overhead is negligible, that conclusion is not supported by the current measurement definition.
- VPIC aggregate std-dev being zero does not distort means, but it hides run-to-run or timestep-to-timestep spread entirely.
- `/tmp` hardcoding does not break the internal pipeline audit, but it can invalidate claims about actual storage throughput depending on the node setup.

## Recommended Fixes

1. Rename CSV headers to reflect raw cumulative semantics.
   - Use `comp_ms_raw_sum`, `decomp_ms_raw_sum`, `explore_ms_sum`, `sgd_ms_sum_host`, or similarly explicit names in aggregate and timestep CSVs.
   - Leave per-chunk CSVs as-is; those are already explicit about `_raw`.

2. Keep both wall-clock and cumulative metrics separate in plots and tables.
   - Use `write_ms` or `vol_pipeline_ms` for wall-clock percentages.
   - Use `vol_s2_busy_ms` and `vol_s3_busy_ms` for bottleneck analysis.
   - Use raw chunk sums only for per-component accounting and isolated throughput, never as additive waterfall bars.

3. Either rename `sgd_ms` or add a second metric for actual SGD GPU completion time.
   - Minimal fix: rename current metric to `sgd_dispatch_ms`.
   - Better fix: record a CUDA event pair on `g_sgd_stream` and store `sgd_gpu_ms_raw` separately.

4. Compute VPIC aggregate throughput std-dev instead of writing zeros.
   - The timestep parser already computes `wr_std` and `rd_std`.
   - Add analogous accumulation for per-timestep `comp_gbps` and `decomp_gbps`.

5. Make benchmark output location configurable for real I/O benchmarking.
   - Keep `/tmp` for pipeline-isolation experiments if desired.
   - Add a documented `*_TMP_DIR` or `BENCH_IO_DIR` override and label tmpfs runs clearly.

6. Clarify the existing stage timing semantics in headers and docs.
   - `vol_stage1_ms`: sequential main-thread wall-clock.
   - `vol_drain_ms`: tail after Stage 1 until workers join.
   - `vol_s2_busy_ms`: max worker wall-clock, not a sum.
   - `vol_s3_busy_ms`: serial I/O-thread busy time, not total end-to-end stage 3 wall-clock.

## CSV Column Audit Table

### Aggregate CSVs

| CSV | Column group | Actual source | Clamped or raw | Units | Header accuracy |
|---|---|---|---|---|---|
| Gray aggregate | `write_ms`,`read_ms` | outer `now_ms()` around full API path | wall-clock | ms | Accurate |
| Gray aggregate | `ratio` | `orig_bytes / file_bytes` | n/a | ratio | Accurate |
| Gray aggregate | `write_mibps`,`read_mibps` | `orig_bytes / wall_ms` | n/a | MiB/s | Accurate |
| Gray aggregate | `nn_ms`,`stats_ms`,`preproc_ms` | chunk sums | raw cumulative | ms | Semantically incomplete |
| Gray aggregate | `comp_ms`,`decomp_ms` | sums of `*_ms_raw` | raw cumulative | ms | Misleading name |
| Gray aggregate | `explore_ms` | chunk sum of host exploration wall-clock | cumulative | ms | Semantically incomplete |
| Gray aggregate | `sgd_ms` | chunk sum of host SGD dispatch timing | cumulative host ms | ms | Misleading name |
| Gray aggregate | `comp_gbps`,`decomp_gbps` | `orig_bytes / raw_sum_ms` | raw | GB/s | Accurate |
| Gray aggregate | `mape_*` | clamped diagnostics | clamped | percent | Accurate |
| Gray aggregate | `mae_*` | mixed: ratio raw, comp/decomp raw | raw | ratio or ms | Accurate |
| Gray aggregate | `comp_gbps_std`,`decomp_gbps_std` | sample std | raw | GB/s | Accurate |
| Gray aggregate | `vol_stage1_ms`,`vol_drain_ms`,`vol_io_drain_ms` | VOL internal wall-clock | wall-clock | ms | Accurate |
| Gray aggregate | `vol_s2_busy_ms`,`vol_s3_busy_ms` | bottleneck worker / I/O busy | wall-clock busy | ms | Accurate |
| SDR aggregate | same pattern as Gray | same logic | same | same | same |
| VPIC aggregate | `write_ms`,`read_ms` | parsed timestep wall-clock means | wall-clock | ms | Accurate |
| VPIC aggregate | `comp_ms`,`decomp_ms` | parsed timestep raw sums | raw cumulative | ms | Misleading name |
| VPIC aggregate | `explore_ms`,`sgd_ms` | parsed timestep cumulative sums | cumulative | ms | Semantically incomplete |
| VPIC aggregate | `comp_gbps`,`decomp_gbps` | recomputed from raw average ms | raw | GB/s | Accurate |
| VPIC aggregate | `comp_gbps_std`,`decomp_gbps_std` | hardcoded `0.0000` | n/a | GB/s | Incorrect |

### Per-chunk CSVs

| CSV | Column group | Actual source | Clamped or raw | Units | Header accuracy |
|---|---|---|---|---|---|
| Gray chunks | `actual_comp_ms_raw`,`actual_decomp_ms_raw` | chunk diagnostics raw | raw | ms | Accurate |
| Gray chunks | `mape_comp`,`mape_decomp` | compare prediction vs clamped diag | clamped denominator | percent | Accurate |
| Gray chunks | `explore_comp_ms_*` | exploration sample comp times | raw | ms | Accurate |
| Gray chunks | `stats_ms`,`nn_inference_ms`,`preprocessing_ms` | chunk timings | raw | ms | Accurate |
| Gray chunks | `exploration_ms`,`sgd_update_ms` | host-side timing | wall/cumulative per chunk | ms | Mostly accurate, but easy to overinterpret |
| SDR chunks | `actual_comp_ms_raw`,`actual_decomp_ms_raw` | chunk diagnostics raw | raw | ms | Accurate |
| SDR chunks | `mape_comp`,`mape_decomp` | compare prediction vs clamped diag | clamped denominator | percent | Accurate |
| VPIC timestep_chunks | raw comp/decomp columns | chunk diagnostics raw | raw | ms | Accurate |
| VPIC timestep_chunks | PSNR fields | predicted and actual | n/a | dB | Accurate |
| VPIC timestep_chunks | exploration alternatives | chunk diagnostics | raw | mixed | Accurate |

### Timestep CSVs

| CSV | Column group | Actual source | Clamped or raw | Units | Header accuracy |
|---|---|---|---|---|---|
| Gray timestep | `write_ms`,`read_ms` | full wall-clock | wall-clock | ms | Accurate |
| Gray timestep | `stats_ms`,`nn_ms`,`preproc_ms` | per-chunk sums | raw cumulative | ms | Semantically incomplete |
| Gray timestep | `comp_ms`,`decomp_ms` | raw sums | raw cumulative | ms | Misleading name |
| Gray timestep | `explore_ms`,`sgd_ms` | cumulative | cumulative | ms | Semantically incomplete |
| Gray timestep | `vol_*` | VOL timing APIs | wall-clock/busy | ms | Accurate |
| Gray timestep | `h5dwrite_ms`,`cuda_sync_ms`,`h5dclose_ms`,`h5fclose_ms` | deck-side wall slices | wall-clock | ms | Accurate |
| SDR timestep | same pattern as Gray plus quality fields | same | same | same | same |
| VPIC timestep | same pattern as Gray plus `orig_mib`, PSNR, R² | same | same | same | same |

### Timestep-chunks CSVs

| CSV | Column group | Actual source | Clamped or raw | Units | Header accuracy |
|---|---|---|---|---|---|
| Gray timestep_chunks | per-chunk prediction vs actual | chunk diagnostics | raw actuals, clamped MAPE denominator | mixed | Accurate |
| SDR timestep_chunks | per-chunk prediction vs actual | chunk diagnostics | raw actuals, clamped MAPE denominator | mixed | Accurate |
| VPIC timestep_chunks | per-chunk prediction vs actual incl. PSNR | chunk diagnostics | raw actuals, clamped MAPE denominator | mixed | Accurate |

## Corrected Measurement Strategy

For this pipelined architecture, the clean methodology is:

1. Keep `write_ms` as the primary end-to-end write metric.
   - It already includes the full flushed benchmark-visible cost.

2. Treat VOL timings as a separate internal decomposition.
   - `vol_stage1_ms + vol_drain_ms + vol_io_drain_ms = vol_pipeline_ms`
   - This is the correct additive internal wall-clock breakdown.

3. Treat `vol_s2_busy_ms` and `vol_s3_busy_ms` as bottleneck diagnostics only.
   - They are not additive with each other or with `vol_stage1_ms`.

4. Treat `stats_ms`, `nn_ms`, `comp_ms_raw_sum`, `decomp_ms_raw_sum`, `explore_ms_sum`, and `sgd_dispatch_ms_sum` as cumulative component totals.
   - Good for understanding where work exists.
   - Bad for naive waterfall charts or percentages of end-to-end wall-clock.

5. Keep throughput formulas split by intent.
   - End-to-end write throughput: `orig_bytes / write_ms`
   - End-to-end read throughput: `orig_bytes / read_ms`
   - Isolated compression throughput: `orig_bytes / comp_ms_raw_sum`
   - Isolated decompression throughput: `orig_bytes / decomp_ms_raw_sum`

6. Keep the 5 ms clamp only where it belongs.
   - Use clamped values for MAPE and cost-model consistency.
   - Use raw values for throughput and latency breakdown.

## Bottom Line

The current code is correctly capturing the top-line write and read timings. I do not see a live missing-sync bug in the three benchmark drivers or in the VOL timing implementation.

The main remaining problems are reporting semantics:
- aggregate and timestep CSVs still underspecify raw cumulative timing columns,
- `sgd_ms` is not true SGD GPU time,
- VPIC aggregate throughput std-dev is still wrong,
- `/tmp` keeps I/O conclusions environment-sensitive.
