# Figure 5 — per-chunk cost breakdown (anatomy pies)

Reproduces the paper's Figure 5: a pair of donut charts showing where
time is spent inside one H5Dwrite (write path) and one H5Dread (read path).

The write donut has 6 wedges: Stats Kernel, NN Inference, Compress
Choice, Compress Factory, Compress Time, I/O Time. The read donut has 3
wedges: Compress Factory, Decompress Time, I/O Time.

## Run — one workload

Four independent Jarvis pipelines, one per workload. Each maps to its
own cached SIF (keyed by the YAML's `name:` field). Pipeline names:

| Workload | `name:` | Output dir |
|---|---|---|
| VPIC   | `gpucompress_figure_5_vpic`   | `$HOME/GPUCompress/tmp/figure_5_vpic`   |
| Nyx    | `gpucompress_figure_5_nyx`    | `$HOME/GPUCompress/tmp/figure_5_nyx`    |
| WarpX  | `gpucompress_figure_5_warpx`  | `$HOME/GPUCompress/tmp/figure_5_warpx`  |
| LAMMPS | `gpucompress_figure_5_lammps` | `$HOME/GPUCompress/tmp/figure_5_lammps` |

**Login node — one-time repo register:**

```bash
jarvis repo add $HOME/GPUCompress/gpucompress_pkgs
```

**Login node — load the pipeline** (builds the SIF on first run; cache
hit on subsequent loads unless you edit GPUCompress source):

```bash
jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_5/pipeline_<workload>.yaml
```

**Compute (GPU) node — run.** The hostfile must point at the current
node; `jarvis ppl load` captures the hostfile at load time, so refresh
it before `run`:

```bash
hostname > $HOME/hostfile_single.txt && \
cp $HOME/hostfile_single.txt $HOME/.ppi-jarvis/shared/gpucompress_figure_5_<workload>/hostfile && \
jarvis ppl run
```

**Plot** (runs from login or compute node — needs only read access to
the CSV):

```bash
python3 $HOME/GPUCompress/evaluations/figure_5/plot_anatomy.py \
        $HOME/GPUCompress/tmp/figure_5_<workload>
```

Output PNG: `$HOME/GPUCompress/tmp/figure_5_<workload>/anatomy.png`.

### Concrete example — Nyx

```bash
# Login node:
jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_5/pipeline_nyx.yaml

# Compute node:
hostname > $HOME/hostfile_single.txt && \
cp $HOME/hostfile_single.txt $HOME/.ppi-jarvis/shared/gpucompress_figure_5_nyx/hostfile && \
jarvis ppl run

# Plot (anywhere):
python3 $HOME/GPUCompress/evaluations/figure_5/plot_anatomy.py \
        $HOME/GPUCompress/tmp/figure_5_nyx
```

## Run — all four workloads

**Login node** (builds any missing SIFs; cache-hits the rest):

```bash
for wl in vpic nyx warpx lammps; do
    jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_5/pipeline_${wl}.yaml
done
```

**Compute node** (runs each pipeline serially + produces the PNG after
each):

```bash
hostname > $HOME/hostfile_single.txt
for wl in vpic nyx warpx lammps; do
    jarvis ppl load yaml $HOME/GPUCompress/evaluations/figure_5/pipeline_${wl}.yaml
    cp $HOME/hostfile_single.txt $HOME/.ppi-jarvis/shared/gpucompress_figure_5_${wl}/hostfile
    jarvis ppl run
    python3 $HOME/GPUCompress/evaluations/figure_5/plot_anatomy.py \
            $HOME/GPUCompress/tmp/figure_5_${wl}
done
```

End state: four `anatomy.png` files, one per workload.

## Files

| File | Purpose |
|---|---|
| `pipeline_vpic.yaml`   | Single-cell VPIC (NX=150, 18 timesteps, chunk_mb=8). Canonical Figure 5 config. |
| `pipeline_warpx.yaml`  | Single-cell WarpX LWFA (dump-then-replay, ncell=64×64×128, 5 diags). |
| `pipeline_lammps.yaml` | Single-cell LAMMPS (atoms=40, 5 dumps, in-situ fix_gpucompress_kokkos). |
| `pipeline_nyx.yaml`    | Single-cell Nyx Sedov (ncell=64, 5 dumps, dump-then-replay). |
| `plot_anatomy.py`      | Reads `benchmark_<workload>_timesteps.csv`, averages per-stage timing columns across timesteps, writes the two-donut PNG. |

## CSV-schema caveat

`plot_anatomy.py` expects these columns in the timesteps CSV:
`stats_ms`, `nn_ms`, `comp_ms`, `vol_setup_ms`, `vol_io_drain_ms`,
`read_ms`, `decomp_ms`.

- **VPIC** emits the full set from the VPIC benchmark deck binary — anatomy
  pie will be complete. This is the canonical Figure 5 workload.
- **WarpX / Nyx** emit the set via `generic_benchmark` (Phase 2) — the VOL
  runs there, so the same per-stage columns populate. Anatomy should work.
- **LAMMPS** writes its per-timestep CSV via the fix's own logging path
  rather than generic_benchmark. The fix's CSV schema may be a subset of
  VPIC-deck's columns. If a column is missing, the plotter silently reads
  zero for that wedge (so the pie may show 0% for Stats Kernel or similar
  rather than failing).

For the paper's Figure 5 stick to VPIC; use the other workloads' anatomy
pies as supplementary / sanity checks.

## Tuning the pie shape

The paper's figure has Compress Time at ~70% and everything else small.
At the default config here Compress Time is typically ~35-40%.
The gap is because the paper's data is a *4 MB periodic random pattern*
(not currently in the repo — see `evaluations/README.md` for the gap
discussion), which compresses differently from VPIC plasma fields.

To push the VPIC pie toward the paper's shape, the highest-leverage knob
is **`chunk_mb`**: larger chunks amortize the fixed per-chunk overhead
(stats, NN forward pass, factory setup) over more compressed bytes, so
the Compress Time wedge grows.

| `chunk_mb` | Compress Time (expected) | Notes |
|---|---|---|
| 4  | ~25-30% | Smallest chunks, maximum relative NN overhead |
| 8  | ~35-40% | Default |
| 16 | ~50-60% | Closer to paper shape |
| 32 | ~65-70% | Paper-match territory |

Second-highest leverage: **`nx`** (cubic in per-dump size). Third:
`timesteps` (linear in total, but doesn't change per-chunk shape much).

## Label → CSV column mapping

`plot_anatomy.py` reads `benchmark_vpic_deck_timesteps.csv` and averages
across all rows. The paper labels map to CSV columns as follows:

| Paper wedge | CSV column | Notes |
|---|---|---|
| Stats Kernel      | `stats_ms`        | data characterization (entropy, MAD, ∂²) |
| NN Inference      | 99% of `nn_ms`    | NN forward pass per chunk |
| Compress Choice   | 1%  of `nn_ms`    | ranking step (paper reports ~0.03%; no dedicated column) |
| Compress Factory  | `vol_setup_ms`    | nvcomp manager init |
| Compress Time     | `comp_ms`         | compression kernel (the big wedge) |
| I/O Time (write)  | `vol_io_drain_ms` | disk write time |
| Compress Factory (read) | 5% of `decomp_ms` | setup estimate |
| Decompress Time   | 95% of `decomp_ms`| decompression kernel |
| I/O Time (read)   | max(0, `read_ms` − `decomp_ms`) | |

All columns come from the VOL's per-timestep emitter, which requires
`GPUCOMPRESS_DETAILED_TIMING=1` (set by the single-cell VPIC package by
default).

## Other workloads

Four sibling pipelines cover the rest: `pipeline_{warpx,lammps,nyx}.yaml`.
Each writes to a distinct `$HOME/GPUCompress/tmp/figure_5_<workload>/`
path; pointing `plot_anatomy.py` at that dir produces the anatomy PNG for
that workload. See the "CSV-schema caveat" section above for what to
expect from each.
