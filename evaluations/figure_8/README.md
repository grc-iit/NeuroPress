# Figure 8 — cross-workload plots

The paper's Figure 8 section has **two panels**, produced by two
scripts in this directory:

| Panel | Renderer | Primary data source |
|---|---|---|
| **Throughput stacked bar** (paper `end2end.png` / Fig 9) | `throughput.py` → `analysis/plot_e2e.py:plot_fig8` | `figure_6_<wl>_trace/gpucompress_trace.csv` |
| **Regret + cost-MAPE convergence + per-metric MAPE** (paper `regret.png`) | `plot.py` | `figure_6_<wl>_trace/gpucompress_trace.csv` (regret + cost-MAPE), `figure_8_<wl>_<policy>/benchmark_<wl>_*.csv` (per-metric bar) |

Both panels read per-chunk trace data produced by running the **Figure
6 pipelines in `vol_mode: trace`**. The cross-workload regret+MAPE
panel additionally needs Figure 8's release-mode runs when you want
the per-metric breakdown bar and/or policy comparison.

## Panel 1 — Throughput stacked bar

Prereq: at least one workload has a populated `gpucompress_trace.csv`
under `$HOME/GPUCompress/tmp/figure_6_<workload>_trace/`. Missing
workloads are silently skipped with a warning.

```bash
python3 $HOME/GPUCompress/evaluations/figure_8/throughput.py
# → $HOME/GPUCompress/tmp/figure_8/fig8_throughput.png
```

Override a nested/nonstandard path per workload:

```bash
python3 $HOME/GPUCompress/evaluations/figure_8/throughput.py \
        --warpx-trace $HOME/GPUCompress/tmp/figure_6_warpx_trace/vol_nn-rl+exp50/gpucompress_trace.csv
```

The stacked bar has 8 configurations per workload (Baseline, NVComp,
NVComp+Tier, NP+Tier, NP+Tier+Async, +Lossy L/M/H). Compute is
rendered opaque; I/O is rendered at α=0.55 (a lighter tint of the
same hue) on top. Matches paper Fig 9 exactly.

## Panel 2 — Regret + cost-MAPE + per-metric MAPE

`plot.py` produces three PNGs per policy:

- `sc26_<policy>_regret.png` — per-chunk regret convergence, one line per workload
- `sc26_<policy>_cost_mape.png` — per-chunk cost-MAPE convergence
- `sc26_<policy>_metric_breakdown.png` — per-metric MAPE clustered bar (comp / decomp / ratio / PSNR)

Regret and cost-MAPE use **trace-derived per-chunk data** (matches the
primary author's `analysis/plot_e2e.py` method bit-exactly — see the
comment inside `load_trace_regret()`). The per-metric bar still needs
figure_8 release-mode chunks CSVs.

### Minimum run (just regret + cost-MAPE, no policy axis needed)

If you only want regret + cost-MAPE convergence, figure_6 trace runs
are sufficient:

```bash
# After figure_6 pipelines have populated figure_6_<wl>_trace/:
python3 $HOME/GPUCompress/evaluations/figure_8/plot.py \
        --sc26-dir $HOME/GPUCompress/tmp --policies balanced
```

The `metric_breakdown` panel will render empty for workloads that
don't have figure_8 data — that's expected.

### Full per-policy run (both policies, all panels)

The `balanced` vs `ratio` split lives only in the figure_8 release-mode
runs, not in figure_6 trace runs. To populate both, flip `policy:` in
each figure_8 YAML between runs:

```bash
# On the compute node, for each workload × policy:
for wl in vpic nyx warpx lammps; do
    for pol in balanced ratio; do
        sed -i "s|^    policy: .*\$|    policy: ${pol}|" \
            $HOME/GPUCompress/evaluations/figure_8/pipeline_${wl}.yaml
        jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_8/pipeline_${wl}.yaml
        cp $HOME/hostfile_single.txt \
           $HOME/.ppi-jarvis/shared/gpucompress_${wl}_delta_single_node/hostfile
        jarvis ppl run
    done
    sed -i 's|^    policy: ratio$|    policy: balanced|' \
        $HOME/GPUCompress/evaluations/figure_8/pipeline_${wl}.yaml   # restore
done

# Then render both policies:
python3 $HOME/GPUCompress/evaluations/figure_8/plot.py \
        --sc26-dir $HOME/GPUCompress/tmp --policies balanced ratio
```

End state: 8 directories under `$HOME/GPUCompress/tmp/figure_8_<wl>_<policy>/`
and 6 PNGs in `$HOME/GPUCompress/tmp/figures/` (3 per policy).

## Equalized hyperparameters (baked into every YAML)

```
phase: nn-rl+exp50
error_bound: 0.001
sgd_lr: 0.2  sgd_mape: 0.10
explore_k: 4  explore_thresh: 0.20
results_dir_policy_suffix: true
```

The only knob that varies between the two runs of each workload is
`policy:` — `balanced` weights comp_time + decomp_time + I/O equally
`(1,1,1)`; `ratio` weights compression ratio only `(0,0,1)`.

## Files

| File | Purpose |
|---|---|
| `pipeline_vpic.yaml`   | Figure_8 VPIC release-mode run (NX=147, 8 dumps) |
| `pipeline_nyx.yaml`    | Figure_8 Nyx (88³ Sedov, 500 steps) |
| `pipeline_warpx.yaml`  | Figure_8 WarpX (LWFA, 128×128×384) |
| `pipeline_lammps.yaml` | Figure_8 LAMMPS (fast_test canonical) |
| `throughput.py`        | Renders paper Fig 9 stacked bar via `plot_e2e.py:plot_fig8` |
| `plot.py`              | Renders paper Fig 7a/b + per-metric MAPE panel |

## Data sources — which script reads what

```
figure_6_<wl>_trace/gpucompress_trace.csv   (per-chunk × per-config)
  ├── plot_e2e.py:plot_fig8    via throughput.py     → fig8_throughput.png
  ├── plot.py:load_trace_regret                      → sc26_<policy>_regret.png
  └── plot.py:load_trace_cost_mape                   → sc26_<policy>_cost_mape.png

figure_8_<wl>_<policy>/benchmark_<wl>_timestep_chunks.csv
  └── plot.py:METRIC_DEFS loop                       → sc26_<policy>_metric_breakdown.png
```

## Ground-truth verification

`plot.py:load_trace_regret` matches the primary author's
`analysis/plot_e2e.py:load_workload` regret derivation bit-exactly
(Δ = 0 on all chunks tested). Same `chosen==1` filter, same
`groupby("chunk_id")["real_cost"].min()` oracle, same
`(real / oracle) - 1` formula — the only difference is ×100 for the
paper's percent Y-axis convention.

`plot.py:load_trace_cost_mape` likewise mirrors `plot_e2e.py:plot_mape_convergence`:
reads `mape_cost` column from `chosen==1` rows, clipped 0-100%.

## Known gap

The paper's Figure 8 includes a ViT-B/16 (AI) cell. AI data feeds
`plot.py` via `inline_benchmark_chunks.csv` (training-time in-situ
emitter), but the existing `gpucompress_ai_training_delta` package
writes the generic_benchmark replay schema instead. Wiring AI through
the in-situ `--hdf5-direct` path is follow-up work — once
`figure_6/pipeline_ai_training.yaml` exists and produces a trace.csv,
the regret and cost-MAPE panels pick it up automatically.
