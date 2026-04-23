# gpucompress_warpx_delta

Jarvis Path-B package that builds **WarpX** (AMReX-based laser-wakefield
PIC code) against GPUCompress and runs a Laser Wakefield Acceleration
(LWFA) workload inside an Apptainer SIF on NCSA Delta A100 nodes.

WarpX is a **two-phase** workload (like Nyx, unlike VPIC/LAMMPS):

- **Phase 1** — `warpx.3d` runs the LWFA simulation with
  `WARPX_DUMP_FIELDS=1`, writing raw `.f32` files (one per field
  component per FAB per diagnostic dump) into `<results>/raw_fields/diag*/`.
- **Phase 2** — the pkg flattens those dumps into a single directory via
  symlinks and invokes `generic_benchmark` with the requested HDF5 mode /
  phase / policy, producing ranking + cost CSVs.

**Pipeline YAMLs:**
- `gpucompress_pkgs/pipelines/gpucompress_warpx_single_node.yaml` — adaptive (VOL + NN or fixed algorithm)
- `gpucompress_pkgs/pipelines/gpucompress_warpx_single_node_baseline.yaml` — no-comp

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
find ~/.jarvis-cd/pipelines/gpucompress_warpx_delta_single_node -name '*.csv' 2>/dev/null
# If anything prints, move it aside before continuing.

rm -rf ~/.jarvis-cd/pipelines/gpucompress_warpx_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_warpx_delta_single_node/
rm -f  ~/build_warpx.log ~/run_warpx_*.log

# Jarvis validates hostfile: at load time — create a placeholder on login.
# The real compute-node hostname gets written in Phase 3.
echo "$(hostname)" > ~/hostfile_single.txt
```

Only wipe the apptainer cache (`apptainer cache clean --force`) if you
need to force a cold rebuild of `gpucompress_base`. Otherwise the WarpX
build reuses the base layer from any earlier workload and finishes in
~10-15 min (AMReX + WarpX with all feature modules is the slowest of
the four sim builds).

### Phase 1 — Build the SIF (login node)

`jarvis ppl load yaml <path>` builds the SIF as part of loading when
`install_manager: container` is set in the YAML (see
`jarvis_cd/core/pipeline.py` — `_load_from_file` calls
`_build_pipeline_container()` at line 1024). There is no separate
`jarvis ppl build` step.

```bash
cd /u/$USER/GPUCompress/gpucompress_pkgs
jarvis repo add /u/$USER/GPUCompress/gpucompress_pkgs 2>/dev/null || true
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_warpx_single_node.yaml 2>&1 | tee ~/build_warpx.log

# Verify the SIF was produced
ls -lh ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/*.sif
```

Expected tail of log:
```
Apptainer SIF ready: .../gpucompress_warpx_delta_single_node.sif
Loaded pipeline: gpucompress_warpx_delta_single_node
```

The build produces the feature-tagged binary
`/opt/sims/warpx/build-gpucompress/bin/warpx.3d.MPI.CUDA.SP.PSP.OPMD.EB.QED`
plus a `warpx.3d` symlink to it. pkg.py addresses the feature-tagged
name directly (`pkg.py:41-42`) so the symlink is belt-and-suspenders.

Subsequent loads print `Deploy image '<name>' already exists, skipping
build` and return in seconds.

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
hostname > ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/hostfile

# Enter pipeline
jarvis cd gpucompress_warpx_delta_single_node

# Adaptive run — see the YAML's phase setting. Default ships with a real
# algorithm (edit the YAML to switch between nn-rl, nn-rl+exp50, zstd, lz4, …).
jarvis ppl run 2>&1 | tee ~/run_warpx_adaptive.log

# Baseline run — reload the sibling baseline YAML (hdf5_mode=default).
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_warpx_single_node_baseline.yaml
hostname > ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/hostfile
jarvis cd gpucompress_warpx_delta_single_node
jarvis ppl run 2>&1 | tee ~/run_warpx_baseline.log

# To go back to adaptive, reload the primary YAML:
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_warpx_single_node.yaml
```

### Phase 4 — Archive results BEFORE releasing the allocation

`/tmp` is per-node and vanishes when the allocation ends.

```bash
STAMP=$(date +%Y%m%d_%H%M)
OUT=~/warpx_results_${STAMP}
mkdir -p "$OUT"
cp -r /tmp/gpucompress_warpx_*_vol      "$OUT/" 2>/dev/null
cp -r /tmp/gpucompress_warpx_*_default  "$OUT/" 2>/dev/null
cp    ~/run_warpx_adaptive.log  ~/run_warpx_baseline.log  "$OUT/"

ls -la "$OUT/"
du -sh "$OUT"

exit   # release the allocation
```

---

## Configuration reference

All knobs live in `pkg.py:_configure_menu()`. **Configure them by editing
the pipeline YAML and re-running `jarvis ppl load yaml <path>`.** This is
the canonical workflow.

`jarvis pkg conf gpuc_warpx <k>=<v>` also exists but is not recommended —
multi-key invocations are not atomic, and the persisted config can drift
out of sync with the source YAML, producing confusing validation failures
at the next `ppl run`. Edit the YAML, reload, done.

For repeatable variants (adaptive vs fixed algorithms vs baseline,
policy sweeps, etc.), create sibling YAMLs with descriptive names and
switch between them by reloading.

### HDF5 mode / algorithm / policy

| Key | Default | Meaning |
|---|---|---|
| `hdf5_mode` | `default` | `default` (no-comp baseline) or `vol` (GPUCompress VOL) |
| `phase` | `lz4` | `lz4`/`snappy`/`deflate`/`gdeflate`/`zstd`/`ans`/`cascaded`/`bitcomp` (fixed nvcomp) or `nn`/`nn-rl`/`nn-rl+exp50` (adaptive) |
| `policy` | `ratio` | NN cost-model weights: `balanced` (w=1,1,1), `ratio` (0,0,1), `speed` (1,1,0). Ratio is recommended for LWFA (fields vary from smooth to turbulent) |
| `error_bound` | `0.0` | Lossy tolerance; `0.0` = lossless. LWFA typically tolerates `1e-3` — `1e-2` |

### I/O volume knobs

| Key | Default | Notes |
|---|---|---|
| `ncell` | `"32 32 256"` | Grid cells as space-separated `"nx ny nz"`. Data/dump ≈ `nx×ny×nz × 6 components × 4 B`. `"32 32 256"` → ~6 MB, `"64 64 512"` → ~48 MB, `"128 128 1024"` → ~384 MB |
| `max_step` | `30` | Total WarpX simulation steps |
| `diag_int` | `10` | Steps between `diag1` dumps. Number of dumps ≈ `max_step / diag_int` |
| `max_grid_size` | `""` | AMR max grid size (empty = WarpX default) |
| `blocking_factor` | `""` | AMR blocking factor (empty = WarpX default) |
| `chunk_mb` | `4` | HDF5 chunk size for Phase-2 replay |
| `verify` | `1` | `1` = bitwise readback verify in Phase 2, `0` = skip |

### NN online learning (Phase 2 only, ignored for fixed algorithms)

| Key | Default | Purpose |
|---|---|---|
| `sgd_lr` | `0.2` | SGD learning rate (`--lr`) |
| `sgd_mape` | `0.10` | MAPE threshold for SGD firing (`--mape`) |
| `explore_k` | `4` | Top-K exploration alternatives (`--explore-k`) |
| `explore_thresh` | `0.20` | Exploration error threshold (`--explore-thresh`) |

### Container / launch

| Key | Default | Notes |
|---|---|---|
| `cuda_arch` | `80` | A100 = 80. Must match `gpucompress_base`. |
| `deploy_base` | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | Pinned to CUDA 12.6 to avoid a Kokkos/AMReX NVCC 12.8 ICE. |
| `use_gpu` | `True` | Adds `--nv` to the apptainer invocation |

Phase 1 (the WarpX sim itself) hardcodes `gpucompress.algorithm=auto` and
`gpucompress.policy=ratio` in the command-line override block
(`pkg.py:227-228`). Those values only shape the in-situ raw-field dump,
not the Phase-2 benchmark; the Phase-2 replay uses `phase`/`policy` from
the YAML.

---

## Expected outputs

Each run writes to `/tmp/gpucompress_warpx_<pkg_id>_<ncell>_ms<M>_<mode>/`:

```
<results_dir>/
├── warpx_sim.log                       # Phase-1 stdout+stderr
├── raw_fields/                         # Phase-1 output
│   ├── diag00000000/*.f32  (per-component raw dumps)
│   ├── diag00000010/*.f32
│   └── ...
├── flat_fields/                        # symlinks consumed by Phase 2
│   └── diag<NNNNNNNN>_fab0000_comp<NN>_<name>.f32
└── <mode>_<phase>/                     # e.g. vol_nn-rl or default_no-comp
    ├── warpx_bench.log                 # Phase-2 stdout+stderr
    ├── benchmark_warpx_<ncell>.csv             # per-field summary (primary CSV)
    ├── benchmark_warpx_<ncell>_ranking.csv     # Kendall τ + regret per field
    ├── benchmark_warpx_<ncell>_ranking_costs.csv # NN-predicted vs observed costs
    └── gpucompress_vol_summary.txt     # lifetime I/O totals, VOL timing
```

Note the `dims="1,N"` convention in Phase 2 (WarpX dumps each FAB
component as a 1-D blob with flat-last indexing — opposite of Nyx's
`"N,1"`). This is handled automatically in `pkg.py:278`.

### Success indicators

- Phase 1: `[GPUCompress] Raw field dump enabled → <results>/raw_fields`
  and `Dumped: <D> diag* directories` where `D = max_step / diag_int`
- Phase 2: `Benchmark PASSED` printed near the end of `warpx_bench.log`
- Adaptive runs on the smoke defaults produce lossless ratios ≈ 1.5-4×
  (LWFA fields are smoother than Nyx's Sedov blast, so ratios are
  moderate compared to Nyx's 100×+ sparse regime)
- Baseline: `phase=no-comp`, ratio=1.00 across all fields

### Benign artefacts to ignore

- `environment: line 17: /usr/share/lmod/lmod/libexec/lmod: No such file
  or directory` — harmless lmod call inside the container.
- AMReX `Using host 0 for default device 0` / `GPU global memory (MB)
  spread across MPI` blocks — runtime info.
- **Trailing `exit 139` or `signal 11` (SIGSEGV) AFTER `Benchmark PASSED`
  and the lifetime summary** — this is a known library-destructor-order
  issue between `libH5VLgpucompress.so` and AMReX during MPI finalize.
  All output is already written; the pkg's exit-code check accepts
  non-zero exit when the log contains the success markers. Cosmetic.
- OpenPMD / BOOST warnings during AMReX init — harmless.

---

## Scaling up

| Scale | `ncell` | `max_step` | `diag_int` | Dumps | Data/dump | Wall (A100, 1 GPU) |
|---|---|---|---|---|---|---|
| Smoke (default) | `"32 32 256"` | 30 | 10 | 3 | ~6 MB | ~20-40 s |
| Small | `"64 64 512"` | 50 | 10 | 5 | ~48 MB | ~2-5 min |
| Medium | `"128 128 1024"` | 100 | 25 | 4 | ~384 MB | ~15-30 min |
| Paper | `"128 128 1024"` | 200 | 50 | 4 | ~384 MB | ~30-60 min |

WarpX runs single-rank in this pipeline (the wrapper does not split the
grid across MPI ranks). Multi-rank requires editing `pkg.py:start()` to
pass `nprocs > 1` to `MpiExecInfo` and picking `amr.max_grid_size` so
the domain decomposes cleanly.

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
`gpuaNNN`, not `dt-loginNN`. Re-`salloc` if needed.

### Apptainer asks for a SSH password
The per-pipeline cached hostfile still points at the old node.
`jarvis hostfile set` updates the global hostfile but does NOT propagate
to the per-pipeline copy. Overwrite it directly:
```bash
hostname > ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/hostfile
```

### `Pipeline startup failed at package 'gpuc_warpx'`
1. Open `/tmp/gpucompress_warpx_*/warpx_sim.log` — if Phase 1 crashed,
   the real cause is in the AMReX trace at the top.
2. If Phase 1 says "Dumped: 0 diag* directories", the WarpX binary died
   before writing any dump. Common causes: wrong CUDA host, OOM at large
   `ncell`, or a bad override in the Phase-1 command line.
3. If Phase 2 died, inspect `/tmp/gpucompress_warpx_*/<mode>_<phase>/warpx_bench.log`.
4. If you see `exit 139`/`signal 11` **after** `Benchmark PASSED`,
   that's the known destructor-order teardown — not a failure.

### WarpX binary not found after build
The build produces a feature-tagged binary name
(`warpx.3d.MPI.CUDA.SP.PSP.OPMD.EB.QED`). `build.sh` also creates a
`warpx.3d` symlink. If the symlink is missing or self-referential
(older builds had this bug), the pkg uses the feature-tagged path
directly — check `WARPX_BIN` in `pkg.py:41-42`.

### `phase 'no-comp' not in [...]`
`no-comp` is not a valid `phase` value — it's an internal pseudo-phase
selected implicitly when `hdf5_mode=default`. Either:
- Use the baseline YAML variant (recommended):
  ```bash
  jarvis ppl load yaml .../gpucompress_warpx_single_node_baseline.yaml
  ```
- Or in your own YAML, set `hdf5_mode: default` and keep `phase` at any
  real value (e.g. `phase: nn-rl+exp50`) — it is ignored in default mode.

If the persisted config is stuck with `phase: no-comp`, patch it:
```bash
sed -i 's/^  phase: no-comp$/  phase: nn-rl+exp50/' \
    ~/.ppi-jarvis/config/pipelines/gpucompress_warpx_delta_single_node/pipeline.yaml
```

### Rebuild from scratch
If you change source under `/u/$USER/GPUCompress/` or the base image,
wipe the cached SIF + apptainer cache and reload — the load step
triggers a fresh build:
```bash
rm -rf ~/.jarvis-cd/pipelines/gpucompress_warpx_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_warpx_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_warpx_delta_single_node/
apptainer cache clean --force
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_warpx_single_node.yaml
```

---

## File map

```
gpucompress_warpx_delta/
├── pkg.py               # Jarvis Application class; two-phase start/stop/clean lifecycle
├── build.sh             # apptainer fakeroot build: clones WarpX + AMReX, applies patches, builds warpx.3d
├── Dockerfile.deploy    # deploy-stage template (COPY /opt + /usr/local, LD_LIBRARY_PATH)
├── __init__.py          # empty, required for Python import
└── README.md            # this file
```

Paired pipeline YAMLs:
```
gpucompress_pkgs/pipelines/gpucompress_warpx_single_node.yaml           # adaptive / fixed
gpucompress_pkgs/pipelines/gpucompress_warpx_single_node_baseline.yaml  # no-comp
```
