# gpucompress_vpic_delta

Jarvis Path-B package that builds the **VPIC plasma PIC benchmark deck**
(`vpic_benchmark_deck.Linux`) against GPUCompress and runs it inside an
Apptainer SIF on NCSA Delta A100 nodes.

VPIC is a **single-phase** workload: the benchmark deck runs every selected
compression phase inline inside one simulation, driven entirely by
environment variables. The pkg adapts this to Apptainer paths and the
Jarvis lifecycle.

**Pipeline YAML:** `gpucompress_pkgs/pipelines/gpucompress_vpic_single_node.yaml`

---

## Prerequisites

- Delta account with an A100 allocation (set `$SLURM_ACCOUNT` to your own)
- Jarvis-CD installed and on `$PATH` (checked via `which jarvis`)
- ~10 GB of free quota under `/u/$USER/` (SIF ~6-10 GB + build artefacts)
- This repo cloned to `/u/$USER/GPUCompress` (paths in the YAML assume that)

---

## End-to-end, from zero (copy-paste ready)

Four phases: **clean → build → allocate → run+archive**.
`dt-loginNN` = login node (no GPU), `gpuaNNN` = compute node (A100).

### Phase 0 — Clean state (login node)

```bash
# Scan for stray CSVs first — never delete these
find ~/.jarvis-cd/pipelines/gpucompress_vpic_delta_single_node -name '*.csv' 2>/dev/null
# If anything prints, move it aside before continuing.

rm -rf ~/.jarvis-cd/pipelines/gpucompress_vpic_delta_single_node/
rm -f  ~/hostfile_single.txt ~/hostfile_*.txt
rm -f  ~/build_vpic.log ~/run_vpic_*.log
apptainer cache clean --force

# Jarvis validates hostfile: at load time — create a placeholder on login.
# The real compute-node hostname gets written in Phase 3.
echo "$(hostname)" > ~/hostfile_single.txt
```

### Phase 1 — Build the SIF (login node, ~15-20 min cold)

```bash
cd /u/$USER/GPUCompress/gpucompress_pkgs
jarvis repo add /u/$USER/GPUCompress/gpucompress_pkgs 2>/dev/null || true
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_vpic_single_node.yaml
jarvis ppl build 2>&1 | tee ~/build_vpic.log

# Verify
ls -la ~/.jarvis-cd/pipelines/gpucompress_vpic_delta_single_node/shared/*.sif
```

Expected tail of log:
```
INFO:    Build complete: .../gpucompress_vpic_delta_single_node.sif
```

### Phase 2 — Get a GPU node

```bash
salloc --account="$SLURM_ACCOUNT" --partition=gpuA100x4 \
       --nodes=1 --ntasks=1 --gpus=1 --time=01:00:00 --mem=64g
```

Wait until the shell prompt changes from `dt-loginNN` to `gpuaNNN`. If it
still says `dt-login`, the allocation is pending — `squeue -u $USER` to
check. **Do not run the pipeline on a login node — it has no GPU.**

### Phase 3 — Run on the compute node

```bash
# Sanity
hostname          # must start with "gpua"
nvidia-smi        # must list an A100

# Refresh BOTH hostfiles: the global one AND the per-pipeline cached copy
# (jarvis hostfile set does NOT propagate to the per-pipeline copy)
hostname > ~/hostfile_single.txt
jarvis hostfile set ~/hostfile_single.txt
hostname > ~/.ppi-jarvis/shared/gpucompress_vpic_delta_single_node/hostfile

# Enter pipeline
jarvis cd gpucompress_vpic_delta_single_node

# Adaptive run — hdf5_mode=vol, phase=nn-rl, policy=balanced
jarvis ppl run 2>&1 | tee ~/run_vpic_adaptive.log

# Baseline run — reload the sibling baseline YAML (hdf5_mode=default).
# Editing YAML + reloading is the canonical way to switch configs.
# Avoid `jarvis pkg conf` overrides: multi-key calls are not atomic and
# leave the persisted config in inconsistent states.
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_vpic_single_node_baseline.yaml
jarvis cd gpucompress_vpic_delta_single_node
jarvis ppl run 2>&1 | tee ~/run_vpic_baseline.log

# To go back to adaptive, reload the primary YAML:
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_vpic_single_node.yaml
```

### Phase 4 — Archive results BEFORE releasing the allocation

`/tmp` is per-node and vanishes when the allocation ends.

```bash
STAMP=$(date +%Y%m%d_%H%M)
OUT=~/vpic_results_${STAMP}
mkdir -p "$OUT"
cp -r /tmp/gpucompress_vpic_*_vol      "$OUT/" 2>/dev/null
cp -r /tmp/gpucompress_vpic_*_default  "$OUT/" 2>/dev/null
cp    ~/run_vpic_adaptive.log  ~/run_vpic_baseline.log  "$OUT/"

ls -la "$OUT/"
du -sh "$OUT"

exit   # release the allocation
```

---

## Configuration reference

All knobs live in `pkg.py:_configure_menu()`. **Configure them by editing
the pipeline YAML and re-running `jarvis ppl load yaml <path>`.** This is
the canonical workflow.

`jarvis pkg conf gpuc_vpic <k>=<v>` also exists but is not recommended —
multi-key invocations are not atomic, and the persisted config can drift
out of sync with the source YAML, producing confusing validation failures
at the next `ppl run`. Edit the YAML, reload, done.

For repeatable variants (baseline vs adaptive, policy sweeps, etc.),
create sibling YAMLs with descriptive names (e.g.
`gpucompress_vpic_single_node_baseline.yaml`) and switch between them by
reloading.

### HDF5 mode / algorithm / policy

| Key | Default | Meaning |
|---|---|---|
| `hdf5_mode` | `default` | `default` (no-comp baseline) or `vol` (GPUCompress VOL) |
| `phase` | `lz4` | `lz4`/`snappy`/`deflate`/`gdeflate`/`zstd`/`ans`/`cascaded`/`bitcomp` (fixed) or `nn`/`nn-rl`/`nn-rl+exp50` (adaptive) |
| `policy` | `balanced` | NN cost-model weights: `balanced` (w=1,1,1), `ratio` (0,0,1), `speed` (1,1,0) |
| `error_bound` | `0.0` | Lossy tolerance; `0.0` = lossless |

### I/O volume knobs

| Key | Default | Notes |
|---|---|---|
| `nx` | `64` | Grid dim; data/snapshot ≈ (NX+2)³ × 64 B. NX=64 → ~17 MB, NX=200 → ~533 MB. Must be divisible by `nprocs`. |
| `timesteps` | `3` | Benchmark write cycles |
| `sim_interval` | `190` | Physics steps between writes |
| `warmup` | `100` | Physics steps before first snapshot |
| `chunk_mb` | `4` | HDF5 chunk size |
| `verify` | `1` | `1` = bitwise readback verify, `0` = skip |

### Physics / data-variety

| Key | Default | Range | Effect |
|---|---|---|---|
| `mi_me` | `25` | 1-400 | Ion/electron mass ratio |
| `wpe_wce` | `3.0` | 0.5-5 | Plasma/cyclotron frequency ratio |
| `ti_te` | `1.0` | 0.1-10 | Ion/electron temperature ratio |
| `perturbation` | `0.1` | 0-0.5 | Tearing-mode seed fraction of B0 |
| `guide_field` | `0.0` | 0-0.5 | Out-of-plane guide field |
| `nppc` | `2` | | Particles per cell |

### NN online learning

| Key | Default | Purpose |
|---|---|---|
| `sgd_lr` | `0.2` | SGD learning rate |
| `sgd_mape` | `0.10` | MAPE threshold for SGD firing |
| `explore_k` | `4` | Top-K exploration alternatives |
| `explore_thresh` | `0.20` | Exploration error threshold |

### Launch

| Key | Default | Notes |
|---|---|---|
| `nprocs` | `1` | MPI processes (must divide `nx`) |
| `ppn` | `1` | Processes per node |
| `cuda_arch` | `80` | A100 = 80. Must match `gpucompress_base`. |

---

## Expected outputs

Each run writes to `/tmp/gpucompress_vpic_<pkg_id>_NX<N>_ts<T>_<mode>/`:

| File | Contents |
|---|---|
| `vpic_bench.log` | Full stdout+stderr |
| `benchmark_vpic_deck_timesteps.csv` | Per-timestep write_ms / read_ms / ratio for each phase |
| `benchmark_vpic_deck_timestep_chunks.csv` | Per-chunk breakdown (the primary analysis CSV) |
| `benchmark_vpic_deck_ranking.csv` | Kendall τ + regret per timestep |
| `benchmark_vpic_deck_ranking_costs.csv` | NN-predicted vs observed costs (MAPE source) |
| `gpucompress_vol_summary.txt` | Lifetime I/O totals, VOL timing, overall ratio |

### Success indicators

- `normal exit` appears in `vpic_bench.log` near the end (VPIC convention)
- `=== VPIC Multi-Timestep complete (<T> timesteps) ===` printed
- Four CSVs listed above exist and have >1 row past the header
- Adaptive runs: ratios ≈ 1.3-4×, τ ≈ 0.4-0.9, regret ≈ 1.00-1.05×
- Baseline: phase=no-comp, ratio=1.00 across all chunks

### Benign artefacts to ignore

- `Signal 11 (Segmentation fault)` **after** `normal exit` and the lifetime
  summary — library-destructor-order teardown in MPI finalize. The pkg's
  `_log_has_normal_exit()` accepts this as success.
- `environment: line 17: /usr/share/lmod/lmod/libexec/lmod: No such file or
  directory` — harmless lmod call inside the container.
- `Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable
  not set` — OpenMP isn't used for GPU work, the binding warning doesn't
  affect CUDA execution.

---

## Scaling up

For paper-scale runs, override on the YAML or via `jarvis pkg conf`:

| Scale | `nx` | `timesteps` | `sim_interval` | Data/snap | Wall (A100, 1 GPU) |
|---|---|---|---|---|---|
| Smoke (default) | 64 | 3 | 190 | ~17 MB | ~15 s |
| Small | 128 | 10 | 190 | ~140 MB | ~3 min |
| Medium | 200 | 20 | 190 | ~530 MB | ~30 min |
| Paper (SC default) | 200 | 50 | 190 | ~530 MB | ~1.5 h |

Multi-GPU requires `VPIC_NX` divisible by `nprocs`; e.g. `nx=128, nprocs=4`.
Set `nprocs` and `ppn` in the YAML and request more `--gpus=N` in `salloc`.

---

## Troubleshooting

### `Error: Hostfile not found: /u/$USER/hostfile_single.txt`
Jarvis validates `hostfile:` at `ppl load` time. Create a placeholder on
the login node before loading:
```bash
echo "$(hostname)" > ~/hostfile_single.txt
```

### Benchmark aborts with `cudaErrorNoDevice: no CUDA-capable device`
You're running on the login node. Check the prompt — it must say
`gpuaNNN`, not `dt-loginNN`. Re-`salloc` if needed. The MPI line
`exited on signal 6 (Aborted)` with `on node dt-loginNN` is the giveaway.

### Apptainer asks for a SSH password
The per-pipeline cached hostfile still points at the old node.
`jarvis hostfile set` updates the global hostfile but does NOT propagate
to the per-pipeline copy. Overwrite it directly:
```bash
hostname > ~/.ppi-jarvis/shared/gpucompress_vpic_delta_single_node/hostfile
```

### `Pipeline startup failed at package 'gpuc_vpic': VPIC benchmark failed (exit 134)`
Almost always a GPU/hostfile problem. Check:
1. `nvidia-smi` lists an A100 (you're on a compute node)
2. `cat ~/.ppi-jarvis/shared/gpucompress_vpic_delta_single_node/hostfile`
   shows this node's hostname (matches `hostname`)
3. `tail -100 /tmp/gpucompress_vpic_*/vpic_bench.log` for the actual crash
   reason

### `phase 'no-comp' not in [...]`
`no-comp` is not a valid `phase` value — it's an internal pseudo-phase
selected implicitly when `hdf5_mode=default`. Either:
- Use the baseline YAML variant (recommended):
  ```bash
  jarvis ppl load yaml .../gpucompress_vpic_single_node_baseline.yaml
  ```
- Or in your own YAML, set `hdf5_mode: default` and keep `phase` at any
  real value (e.g. `phase: nn-rl`) — it is ignored in default mode.

If the persisted config is already stuck with `phase: no-comp`, patch it:
```bash
sed -i 's/^  phase: no-comp$/  phase: nn-rl/' \
    ~/.ppi-jarvis/config/pipelines/gpucompress_vpic_delta_single_node/pipeline.yaml
```

### `VPIC_NX must be divisible by nprocs`
Change `nx` to a multiple of `nprocs`. Validation happens in
`pkg.py:_validate()` and blocks the run before any compute.

### Rebuild from scratch
If you change source under `/u/$USER/GPUCompress/` or the base image:
```bash
rm -rf ~/.jarvis-cd/pipelines/gpucompress_vpic_delta_single_node/
apptainer cache clean --force
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_vpic_single_node.yaml
jarvis ppl build
```

---

## File map

```
gpucompress_vpic_delta/
├── pkg.py               # Jarvis Application class; start/stop/clean lifecycle
├── build.sh             # apptainer fakeroot build step (patches VPIC, links GPUCompress)
├── Dockerfile.deploy    # deploy-stage template (COPY /opt + /usr/local, LD_LIBRARY_PATH)
├── __init__.py          # empty, required for Python import
└── README.md            # this file
```

Paired pipeline YAMLs:
```
gpucompress_pkgs/pipelines/gpucompress_vpic_single_node.yaml           # adaptive
gpucompress_pkgs/pipelines/gpucompress_vpic_single_node_baseline.yaml  # no-comp
```
