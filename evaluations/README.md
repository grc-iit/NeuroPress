# Paper evaluation pipelines

Each subdirectory reproduces one paper figure. Every experiment is a
Jarvis-CD pipeline — no shell orchestrators, no `sbatch` wrappers. The
flow is always the same two commands:

```bash
jarvis ppl load yaml <path-to-pipeline.yaml>
jarvis ppl run
```

## Layout

```
evaluations/
├── figure_5/                         — per-chunk cost breakdown (anatomy donuts)
│   ├── pipeline_{vpic,warpx,lammps,nyx}.yaml
│   └── plot_anatomy.py               — two-donut PNG per run
│
├── figure_6/                         — threshold sensitivity heatmap (paper's sweep_fig6.png)
│   └── pipeline_{vpic,warpx,lammps,nyx}.yaml    — all in vol_mode=trace
│                                       Plot with analysis/plot_trace.py
│
├── figure_7/                         — cross-workload regret + MAPE convergence
│   └── run_plot_e2e.py               — wrapper for analysis/plot_e2e.py that
│                                        points at the 3 trace.csvs under figure_6/
│
└── figure_8/                         — cross-workload regret (Section 7.1)
    ├── pipeline_{vpic,nyx,warpx,lammps}_{balanced,ratio}.yaml   — 8 cells
    └── plot.py                       — combined plotter, 3 figures per policy
```

Package source lives under `gpucompress_pkgs/gpucompress_pkgs/<pkg_name>/`
(importable Python packages Jarvis-CD loads). The YAMLs here reference
those packages by dotted name — don't move either tree or the YAMLs will
fail to load.

## What each figure measures

### Figure 5 — per-chunk cost breakdown (anatomy)

Two donut charts showing where time goes inside one `H5Dwrite` (Stats
Kernel / NN Inference / Compress Choice / Compress Factory / Compress
Time / I/O Time) and one `H5Dread` (Compress Factory / Decompress Time /
I/O Time). One workload pipeline writes `benchmark_<w>_timesteps.csv`;
`plot_anatomy.py` averages across timesteps and renders one combined
`anatomy.png`.

### Figure 6 — threshold sensitivity heatmap

Log-spaced 40×40 `(X1, X2−X1)` threshold heatmap with the ★ at
(30%, 20%), rendered by the primary author's `analysis/plot_trace.py`
from one `gpucompress_trace.csv` produced under `vol_mode: trace`. All
four workload pipelines follow the same flow now that the VPIC deck
honors `GPUCOMPRESS_VOL_MODE=trace`. LAMMPS uses its native `fix`;
WarpX/Nyx use Phase-2 `generic_benchmark`; VPIC uses the patched deck
directly.

### Figure 7 — cross-workload regret + MAPE convergence

Line plots showing regret and cost-model MAPE converging over chunk
index, one curve per workload. Uses the primary author's
`analysis/plot_e2e.py` (Fig 7a/b/c). Our `figure_7/run_plot_e2e.py` is a
thin wrapper that points that plotter at the trace CSVs produced by the
four `figure_6/pipeline_<w>.yaml` runs.

### Figure 8 — cross-workload regret (Section 7.1)

Each workload × cost policy (`balanced`, `ratio`) is one single-cell
run with equalized hyperparameters
`PHASE=nn-rl+exp50 ERROR_BOUND=0.001 SGD_LR=0.2 SGD_MAPE=0.10
EXPLORE_K=4 EXPLORE_THRESH=0.20`. 4 workloads × 2 policies = **8
YAMLs**. After all eight have run, invoke
`figure_8/plot.py --sc26-dir <parent-of-8-dirs> --policies balanced ratio`
to produce three per-policy figures:

- `sc26_<policy>_cost_mape.png` — cost-model MAPE convergence
- `sc26_<policy>_regret.png` — top-1 regret convergence
- `sc26_<policy>_metric_breakdown.png` — per-metric MAPE bar chart

**Known gap**: the paper's Fig 8 also includes an AI/ViT-B cell.
`figure_8/plot.py` reads `inline_benchmark_chunks.csv` (training-time
in-situ VOL) for AI, but the existing `gpucompress_ai_training_delta`
writes `benchmark_<name>_timestep_chunks.csv` via `generic_benchmark`
replay. Wiring AI through the in-situ path is follow-up work.

## Build-once-run-many container model

Every paper-figure YAML uses `name: gpucompress_<workload>_delta_single_node`,
matching the cached SIF. Build VPIC / WarpX / LAMMPS / Nyx once — every
pipeline for that workload (figure_5 / figure_6 / figure_7 / figure_8)
reuses the same SIF. The only rebuild triggers are changes to
`cuda_arch`, `gpucompress_ref`, `hdf5_version`, `nvcomp_version`,
`build.sh`, or the deploy Dockerfile.

(VPIC's figure_6 YAML triggered one rebuild after we patched the deck to
honor `GPUCOMPRESS_VOL_MODE=trace` — that SIF is now cached too.)

## Plotter inventory

| Plotter | Inputs | Paper figure |
|---|---|---|
| `figure_5/plot_anatomy.py` | one `benchmark_<w>_timesteps.csv` | **Fig 5** (anatomy donuts) |
| `analysis/plot_trace.py` (primary author) | one `trace.csv` | **Fig 6** (40×40 threshold heatmap + line plots + config heatmaps) |
| `analysis/plot_e2e.py` (primary author) | 3 `trace.csv`s (LAMMPS, VPIC, NYX) | **Fig 7** (cross-workload regret + MAPE convergence) |
| `figure_8/plot.py` | `<sc26_dir>/<w>_<policy>/benchmark_*.csv` | **Fig 8** (regret + MAPE convergence + per-metric MAPE breakdown, 3 figs per policy) |
| `scripts/plot_inline_benchmark.py` | `inline_benchmark.csv` (AI training `--benchmark`) | AI workload figures (follow-up) |
| `scripts/plot_cv_comparison.py` | NN pipeline `.out` log | NN CV MAPE / R² (offline training, not Jarvis-scoped) |

## VOL-contract in brief

All the trace / timing CSVs above come from the **same VOL**
(`docs/reproducability.md` §VOL Configuration Reference). Plumbing:

1. Workload wrappers set their own env vars (`VPIC_MAPE_THRESHOLD`,
   `GPUCOMPRESS_MAPE`, `gpucompress.sgd_mape`, …). Those get translated
   internally into the VOL's `GPUCOMPRESS_MAPE_LOW_THRESH` /
   `_HIGH_THRESH` contract.
2. The VOL's `gpucompress_vol_atexit` handler emits
   `gpucompress_io_timing.csv` (e2e_ms + vol_ms) at process exit.
   Pointed to `GPUCOMPRESS_TIMING_OUTPUT` (or `VPIC_RESULTS_DIR`).
3. Setting `GPUCOMPRESS_VOL_MODE=trace` additionally produces
   `gpucompress_trace.csv` with per-chunk × per-config profiling (~32×
   slower than release).

Every paper-figure package exposes `vol_mode`, `timing_csv_name`,
`trace_csv_name` knobs that wire these env vars per run.
