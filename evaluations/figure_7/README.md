# Figure 7 — cross-workload regret + MAPE convergence

Line plots overlaying regret and cost-model MAPE across all four
workloads, one curve per workload, rendered by the primary author's
`analysis/plot_e2e.py` from the **same trace CSVs that figure_6 produces**.

This directory is post-processing only — no Jarvis pipelines here. The
inputs are the `gpucompress_trace.csv` files already produced by
`evaluations/figure_6/pipeline_{vpic,warpx,lammps,nyx}.yaml` runs.

## Contents

```
evaluations/figure_7/
├── README.md
└── run_plot_e2e.py   — wrapper that invokes analysis/plot_e2e.py with
                        the trace.csvs produced under figure_6/
```

`analysis/plot_e2e.py` has Docker-devcontainer paths hardcoded for
`WORKLOADS = {"LAMMPS": Path(...), "VPIC": Path(...), "NYX": Path(...)}`.
`run_plot_e2e.py` is a thin wrapper that monkey-patches that dict at
import time to point at `$HOME/GPUCompress/tmp/figure_6_<workload>_trace/gpucompress_trace.csv`,
then calls `plot_fig7` + `plot_fig8` from the primary author's script.
The primary author's `analysis/plot_e2e.py` itself stays untouched.

## Prerequisite: figure_6 trace CSVs

Run the figure_6 pipelines for at least three workloads first (LAMMPS,
VPIC, Nyx — what `plot_e2e.py` expects):

```bash
for wl in lammps vpic nyx; do
    jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_6/pipeline_${wl}.yaml
    jarvis ppl run
done
```

That populates `$HOME/GPUCompress/tmp/figure_6_{lammps,vpic,nyx}_trace/gpucompress_trace.csv`.

## Run

```bash
python3 $HOME/GPUCompress/evaluations/figure_7/run_plot_e2e.py
# Default output dirs (override with --fig7-out / --fig8-out):
#   $HOME/GPUCompress/trace_runs/figures/fig7/
#   $HOME/GPUCompress/trace_runs/figures/fig8/
```

Or point explicitly at a different trace root (for example, if you
collected trace CSVs into a shared `<root>/<workload>/trace.csv` layout):

```bash
python3 $HOME/GPUCompress/evaluations/figure_7/run_plot_e2e.py \
        --trace-root /path/to/shared/trace_root \
        --fig7-out  /path/to/fig7_out \
        --fig8-out  /path/to/fig8_out
```

## Output

Five paper-style PNGs (three for Fig 7, two for Fig 8 throughput):

- `fig7a_regret.png` — regret (chosen / oracle − 1) vs chunk index, one line per workload, smoothed over bins of 16 chunks
- `fig7b_mape_convergence.png` — cost-model MAPE (clamped 0–100%) vs chunk index, one line per workload
- `fig7c_mape_per_stat.png` — clustered bar chart: per-stat MAPE (ratio, comp ms, decomp ms, PSNR, SSIM, max error) by workload, steady state
- `fig8_throughput.png` — stacked-bar throughput breakdown: compression compute + disk I/O per config per workload, 8 configs (Baseline, nvCOMP, +Tier, NeuroPr.+Tier, NeuroPr.+Tier+Async, +ε_L/ε_M/ε_H lossy)

## CSV contract

`plot_e2e.py` reads from each workload's `trace.csv`:

- Grouping: `chunk_id`, `chosen`
- Predictions: `pred_cost`, `pred_ratio`, `pred_comp_ms`, `pred_decomp_ms`, `pred_psnr`, `pred_ssim`, `pred_max_error`
- Measurements: `real_cost`, `real_ratio`, `real_comp_ms`, `real_decomp_ms`, `real_psnr`, `real_ssim`, `real_max_error`
- Errors: `mape_cost`

All of these come from the VOL's trace-mode emitter when
`GPUCOMPRESS_VOL_MODE=trace` is set (which figure_6's YAMLs pin via
their `vol_mode: trace` knob).

## Caveat: WarpX and Nyx are replay-based

`analysis/plot_e2e.py` (and `plot_trace.py`) assume each `trace.csv`
comes from **in-situ** runs. LAMMPS and VPIC's figure_6 runs are in-situ
(fix_gpucompress_kokkos / patched VPIC deck → H5Dwrite → VOL). WarpX
and Nyx go through **Phase 2 `generic_benchmark`** in trace mode — a
replay over raw dumped fields, not the live simulator. The NN still
sees the same data so regret + MAPE numbers are valid, but cost numbers
reflect generic_benchmark's end-to-end timings rather than the live
simulator's.

For true in-situ WarpX / Nyx trace, you'd modify
`gpucompress_warpx_delta` / `gpucompress_nyx_delta` to call the VOL
during Phase 1 (the simulator's native output path) rather than only
during Phase 2. Follow-up work.
