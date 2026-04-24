# Figure 6 — threshold sensitivity heatmap

Reproduces the paper's Figure 6 (`sweep_fig6.png`): a log-spaced 40×40
threshold-sensitivity heatmap for one workload, rendered by
`analysis/plot_trace.py` from **one** `gpucompress_trace.csv` produced by
a VOL-trace-mode run.

**All four workloads use the same flow** — load YAML, run, plot. The
trace CSV lands at `$HOME/GPUCompress/tmp/figure_6_<workload>_trace/`,
and `analysis/plot_trace.py` consumes it unchanged.

## VPIC-specific note: chunk sizing

VPIC writes a 1D flat field dataset (`16 × (nx+2)³ × 4 B` per snapshot).
At `nx=32` the dataset is ~2 MiB; at `chunk_mb=4` it would fit in a
single **partial** chunk, which trips a silent failure in
`gpu_trace_chunked_write` (the trace path's partial-chunk branch). The
checked-in `pipeline_vpic.yaml` sets `chunk_mb: 1` so the dataset
splits into 2 full 1-MiB chunks (matching LAMMPS / WarpX multi-chunk
shape). Don't raise VPIC's `chunk_mb` above `1` unless you also raise
`nx` enough that `dset_bytes % chunk_bytes == 0`.

LAMMPS, WarpX, and Nyx don't hit this — their writes are multi-chunk
by construction through `fix_gpucompress_kokkos` / Phase-2
`generic_benchmark`.

## Run any workload

The four pipelines share the same flow — only the YAML name changes.
Each maps to a cached single-cell SIF (`gpucompress_<wl>_delta_single_node`)
so there's no rebuild on repeat runs.

**Login node — one-time repo register:**

```bash
jarvis repo add $HOME/GPUCompress/gpucompress_pkgs
```

**Login node — load the pipeline** (builds SIF first time, cache hit after):

```bash
# Pick one: vpic | lammps | warpx | nyx
WL=lammps
jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_6/pipeline_${WL}.yaml
```

**Compute (GPU) node — refresh hostfile then run.** The hostfile must
point at the current allocated node; `jarvis ppl load` captures the
hostfile at load time, so refresh it on the compute node:

```bash
WL=lammps
hostname > $HOME/hostfile_single.txt && \
cp $HOME/hostfile_single.txt $HOME/.ppi-jarvis/shared/gpucompress_${WL}_delta_single_node/hostfile && \
jarvis ppl run
```

**Plot** (runs from login or compute node — reads trace.csv only):

```bash
WL=lammps
python3 $HOME/GPUCompress/analysis/plot_trace.py \
        $HOME/GPUCompress/tmp/figure_6_${WL}_trace/gpucompress_trace.csv \
        --out-dir $HOME/GPUCompress/tmp/figure_6_${WL}_trace \
        --title ${WL^^}
```

WarpX is a special case — its trace.csv nests one level deeper under
`vol_nn-rl+exp50/`:

```bash
python3 $HOME/GPUCompress/analysis/plot_trace.py \
        $HOME/GPUCompress/tmp/figure_6_warpx_trace/vol_nn-rl+exp50/gpucompress_trace.csv \
        --out-dir $HOME/GPUCompress/tmp/figure_6_warpx_trace/vol_nn-rl+exp50 \
        --title WarpX
```

## Run all four workloads

Login node — load all four to warm the SIFs:

```bash
for wl in vpic nyx warpx lammps; do
    jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_6/pipeline_${wl}.yaml
done
```

Compute node — run + plot each in a loop:

```bash
hostname > $HOME/hostfile_single.txt
for wl in vpic nyx warpx lammps; do
    jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_6/pipeline_${wl}.yaml
    cp $HOME/hostfile_single.txt $HOME/.ppi-jarvis/shared/gpucompress_${wl}_delta_single_node/hostfile
    jarvis ppl run
    TRACE=$HOME/GPUCompress/tmp/figure_6_${wl}_trace/gpucompress_trace.csv
    [ -f "$TRACE" ] || TRACE=$HOME/GPUCompress/tmp/figure_6_${wl}_trace/vol_nn-rl+exp50/gpucompress_trace.csv
    python3 $HOME/GPUCompress/analysis/plot_trace.py "$TRACE" \
            --out-dir $(dirname "$TRACE") --title ${wl^^}
done
```

Each run produces the paper-style figure set in `figure_6_<workload>_trace/`:

- `cost_mape.png` / `.svg` — cost MAPE line plot over chunk index
- `regret.png` / `.svg` — regret line plot
- `config_optimal.png` / `.svg` — oracle's algorithm choice heatmap
- `config_chosen.png` / `.svg` — NN's algorithm choice heatmap
- **`threshold_regret.png`** / `.svg` — **40×40 log-spaced threshold heatmap with ★ at (30%, 20%) — matches `sweep_fig6.png`**
- `threshold_mape.png`, `threshold_write_bw.png`, `threshold_read_bw.png` — sibling heatmaps with different metrics as colormap

## Files

| File | Workload | Path to VOL | Typical wall |
|---|---|---|---|
| `pipeline_vpic.yaml`   | VPIC   | deck → H5Dwrite → VOL (post-patch) | ~3-5 min |
| `pipeline_lammps.yaml` | LAMMPS | `fix_gpucompress_kokkos` → H5Dwrite → VOL | ~15 s |
| `pipeline_warpx.yaml`  | WarpX  | Phase 2 `generic_benchmark` → H5Dwrite → VOL | ~2-5 min |
| `pipeline_nyx.yaml`    | Nyx    | Phase 2 `generic_benchmark` → H5Dwrite → VOL | ~2-5 min |

The primary author's `analysis/plot_trace.py` is not duplicated here;
invoke it directly on any of the four `gpucompress_trace.csv` outputs.

## CSV-schema contract

`plot_trace.py` needs these columns, all emitted by the VOL in trace mode:

- Grouping / labels: `chunk_id`, `chosen`, `action_id`, `comp_lib`, `chunk_bytes`
- Predictions: `pred_cost`, `pred_ratio`, `pred_comp_ms`, `pred_decomp_ms`, `pred_psnr`, `pred_ssim`, `pred_max_error`
- Measurements: `real_cost`, `real_ratio`, `real_comp_ms`, `real_decomp_ms`, `real_psnr`, `real_ssim`, `real_max_error`
- Errors: `mape_cost`, `mape_ratio`, `mape_comp_ms`, `mape_decomp_ms`
- Explore bookkeeping: `explore_mode`

All come from the VOL's trace emitter when `GPUCOMPRESS_VOL_MODE=trace` is
set — the four pipeline YAMLs pin that via their `vol_mode: trace` knob.

## Tuning the heatmap resolution

More chunks = sharper 40×40 threshold emulation. The smoke defaults
produce ~60-100 chunks per workload; bump these knobs for a denser
figure (trade: ~5-10× wall time):

| Workload | Smoke defaults | For a denser figure |
|---|---|---|
| VPIC   | `nx: 32, timesteps: 3` | `nx: 64, timesteps: 8` (~5x chunks) |
| LAMMPS | `atoms: 40, timesteps: 3` | `atoms: 80, timesteps: 10` (~9x chunks) |
| WarpX  | `ncell: "32 32 128", max_step: 30` | `ncell: "64 64 256", max_step: 100` |
| Nyx    | `ncell: 32, max_step: 30` | `ncell: 64, max_step: 100` |
