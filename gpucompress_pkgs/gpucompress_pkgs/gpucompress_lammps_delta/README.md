# gpucompress_lammps_delta

Jarvis Path-B package that builds **LAMMPS** with the GPUCompress Kokkos
fix and runs a Lennard-Jones shock-expansion workload inside an Apptainer
SIF on NCSA Delta A100 nodes.

LAMMPS is a **single-phase** workload (like VPIC, unlike Nyx/WarpX's
two-phase dump-then-replay): the `fix gpucompress` Kokkos fix writes
compressed HDF5 in-situ during the MD run through the GPUCompress VOL +
filter, producing `gpuc_step_*` output directories. The pkg parses those
host-side into a per-timestep CSV at the end of the run — no separate
`generic_benchmark` invocation.

**Pipeline YAMLs:**
- `gpucompress_pkgs/pipelines/gpucompress_lammps_single_node.yaml` — adaptive (VOL + NN)
- `gpucompress_pkgs/pipelines/gpucompress_lammps_single_node_baseline.yaml` — no-comp

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
find ~/.jarvis-cd/pipelines/gpucompress_lammps_delta_single_node -name '*.csv' 2>/dev/null
# If anything prints, move it aside before continuing.

rm -rf ~/.jarvis-cd/pipelines/gpucompress_lammps_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_lammps_delta_single_node/
rm -f  ~/build_lammps.log ~/run_lammps_*.log

# Jarvis validates hostfile: at load time — create a placeholder on login.
# The real compute-node hostname gets written in Phase 3.
echo "$(hostname)" > ~/hostfile_single.txt
```

Only wipe the apptainer cache (`apptainer cache clean --force`) if you
need to force a cold rebuild of `gpucompress_base`. Otherwise the LAMMPS
build reuses the base layer from any earlier workload and finishes in
~5-10 min.

### Phase 1 — Build the SIF (login node)

`jarvis ppl load yaml <path>` builds the SIF as part of loading when
`install_manager: container` is set in the YAML (see
`jarvis_cd/core/pipeline.py` — `_load_from_file` calls
`_build_pipeline_container()` at line 1024). There is no separate
`jarvis ppl build` step.

```bash
cd /u/$USER/GPUCompress/gpucompress_pkgs
jarvis repo add /u/$USER/GPUCompress/gpucompress_pkgs 2>/dev/null || true
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_lammps_single_node.yaml 2>&1 | tee ~/build_lammps.log

# Verify the SIF was produced
ls -la ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/*.sif
```

Expected tail of log:
```
[100%] Linking CXX executable lmp
[100%] Built target lmp
Apptainer SIF ready: .../gpucompress_lammps_delta_single_node.sif
Loaded pipeline: gpucompress_lammps_delta_single_node
```

The build includes:
- `liblammps_gpucompress_udf.so` — bridge UDF that wires VOL + Filter
  into the LAMMPS fix
- `liblammps_ranking_profiler.so` — Kendall-τ profiler, built separately
  because `nvcc_wrapper` + C++20 hits cudafe++ internal errors when it's
  inside `KOKKOS_PKG_SOURCES`
- The `lmp` binary itself, linked against parallel HDF5, nvcomp, and
  GPUCompress

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
hostname > ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/hostfile

# Enter pipeline
jarvis cd gpucompress_lammps_delta_single_node

# Adaptive run — hdf5_mode=vol, phase=nn-rl, policy=balanced, lossless
jarvis ppl run 2>&1 | tee ~/run_lammps_adaptive.log

# Baseline run — reload the sibling baseline YAML (hdf5_mode=default).
# In default mode the fix still runs algo=lz4 internally but CSV rows
# report ratio=1.00 so the baseline measures raw I/O overhead.
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_lammps_single_node_baseline.yaml
hostname > ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/hostfile
jarvis cd gpucompress_lammps_delta_single_node
jarvis ppl run 2>&1 | tee ~/run_lammps_baseline.log

# To go back to adaptive, reload the primary YAML:
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_lammps_single_node.yaml
```

### Phase 4 — Archive results BEFORE releasing the allocation

`/tmp` is per-node and vanishes when the allocation ends.

```bash
STAMP=$(date +%Y%m%d_%H%M)
OUT=~/lammps_results_${STAMP}
mkdir -p "$OUT"
cp -r /tmp/gpucompress_lammps_*_vol      "$OUT/" 2>/dev/null
cp -r /tmp/gpucompress_lammps_*_default  "$OUT/" 2>/dev/null
cp    ~/run_lammps_adaptive.log  ~/run_lammps_baseline.log  "$OUT/"

ls -la "$OUT/"
du -sh "$OUT"

exit   # release the allocation
```

---

## Configuration reference

All knobs live in `pkg.py:_configure_menu()`. **Configure them by editing
the pipeline YAML and re-running `jarvis ppl load yaml <path>`.** This is
the canonical workflow.

`jarvis pkg conf gpuc_lammps <k>=<v>` also exists but is not recommended —
multi-key invocations are not atomic, and the persisted config can drift
out of sync with the source YAML, producing confusing validation failures
at the next `ppl run`. Edit the YAML, reload, done.

For repeatable variants (baseline vs adaptive, policy sweeps, etc.),
create sibling YAMLs with descriptive names and switch between them by
reloading.

### HDF5 mode / algorithm / policy

| Key | Default | Meaning |
|---|---|---|
| `hdf5_mode` | `default` | `default` (no-comp baseline: algo=lz4 internally, ratio reported as 1.00) or `vol` (GPUCompress VOL + adaptive algorithm) |
| `phase` | `lz4` | `lz4`/`snappy`/`deflate`/`gdeflate`/`zstd`/`ans`/`cascaded`/`bitcomp` (fixed) or `nn`/`nn-rl`/`nn-rl+exp50` (adaptive) |
| `policy` | `balanced` | NN cost-model policy: `balanced` / `ratio` / `speed` |
| `error_bound` | `0.0` | Lossy tolerance; `0.0` = lossless |

### I/O volume knobs

| Key | Default | Notes |
|---|---|---|
| `atoms` | `40` | Box edge in FCC unit cells. Total atoms = `4 × atoms³`. 20 → ~3 MB/dump, 40 → ~27 MB, 80 → ~216 MB, 120 → ~730 MB |
| `timesteps` | `3` | Number of field dumps (passed to `fix gpucompress`) |
| `sim_interval` | `50` | MD steps between dumps |
| `warmup_steps` | `20` | MD steps before first dump (the fix still emits a step-0 dump; this also filters `gpuc_step_*` dirs during CSV post-processing, so anything below this threshold is dropped from the CSV) |
| `chunk_mb` | `4` | HDF5 chunk size |
| `verify` | `1` | `1` = bitwise readback verify, `0` = skip |

### Physics / data-variety

| Key | Default | Effect |
|---|---|---|
| `t_hot` | `10.0` | Hot-sphere temperature (LJ units) — creates a shock front |
| `t_cold` | `0.01` | Cold-lattice temperature (LJ units) — near-zero variance |
| `hot_radius_frac` | `0.25` | Hot-sphere radius as fraction of box half-width |

### MPI / GPU launch

| Key | Default | Notes |
|---|---|---|
| `nprocs` | `1` | Number of MPI processes |
| `ppn` | `1` | Processes per node |
| `num_gpus` | `1` | GPUs per rank (passed as `-k on g <N>` to LAMMPS) |

### Container build

| Key | Default | Notes |
|---|---|---|
| `cuda_arch` | `80` | A100 = 80. Must match `gpucompress_base`. |
| `kokkos_arch` | `AMPERE80` | Kokkos arch flag: `AMPERE80` (A100), `AMPERE86` (A40), `HOPPER90` (H100) |
| `deploy_base` | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | Pinned to CUDA 12.6 |
| `use_gpu` | `True` | Adds `--nv` to the apptainer invocation |

---

## Expected outputs

Each run writes to `/tmp/gpucompress_lammps_<pkg_id>_box<N>_ts<T>_<mode>/`:

```
<results_dir>/
├── work_<phase_label>/                # e.g. work_nn-rl or work_no-comp
│   ├── input.lmp                      # generated LAMMPS input deck
│   ├── lammps.log                     # LAMMPS stdout+stderr
│   └── gpuc_step_<NNNNNNNN>/          # per-dump HDF5 output (one dir per fix emit)
│       └── *.h5
├── benchmark_lammps_timesteps.csv     # parsed per-timestep summary
└── gpucompress_vol_summary.txt        # lifetime I/O + VOL timing
```

CSV schema:
```
rank,phase,timestep,write_ms,ratio,orig_mb,comp_mb,verify
```

### Success indicators

- `Created <4*atoms³> atoms` matching your `atoms` config (e.g. 256000 for atoms=40)
- `[GPUCompress-LAMMPS] Initialized: algo=<X> policy=<P> verify=1 error_bound=0`
- `[GPUCompress-LAMMPS] Step <N>: wrote 3 fields (<MB>/rank) algo=<X>` lines (one per dump)
- `Total wall time: 0:00:0X` at the end of `lammps.log`
- `Pipeline started successfully` from Jarvis
- `benchmark_lammps_timesteps.csv` has `timesteps` rows past the header
  (dump at step 0 is filtered out by `warmup_steps > 0`)

Typical numbers on A100 (smoke defaults: atoms=40, timesteps=3, sim_interval=50):
- Adaptive: ~1.4-2× lossless ratio, ~350 MiB/s write, ~1 s wall
- Baseline: ratio=1.00 reported, similar wall time (same code path, just no real compression)

### Benign artefacts to ignore

- `environment: line 17: /usr/share/lmod/lmod/libexec/lmod: No such file
  or directory` — harmless lmod call inside the container
- `Generated 0 of 0 mixed pair_coeff terms from geometric mixing rule`
  — LAMMPS info message, not an error
- `CITE-CITE-CITE ... KOKKOS package: https://doi.org/10.1145/...`
  — Kokkos citation reminder, cosmetic

---

## Scaling up

| Scale | `atoms` | `timesteps` | Atoms | Data/dump | Wall (A100, 1 GPU) |
|---|---|---|---|---|---|
| Smoke (default) | 40 | 3 | 256 k | ~27 MB | ~1-2 s |
| Small | 60 | 10 | 864 k | ~90 MB | ~10-20 s |
| Medium | 80 | 20 | 2.05 M | ~216 MB | ~1-2 min |
| Large | 120 | 20 | 6.91 M | ~730 MB | ~5-10 min |

LAMMPS can scale across multiple GPUs via `nprocs > 1` + matching
`--gpus=N` in `salloc`; Kokkos handles the per-rank GPU partitioning
automatically. Total atoms don't have to be divisible by `nprocs` since
LAMMPS uses its own atom decomposition.

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
hostname > ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/hostfile
```

### `phase 'no-comp' not in [...]`
`no-comp` is not a valid `phase` value — it's an internal pseudo-phase
selected implicitly when `hdf5_mode=default`. Either:
- Use the baseline YAML variant (recommended):
  ```bash
  jarvis ppl load yaml .../gpucompress_lammps_single_node_baseline.yaml
  ```
- Or in your own YAML, set `hdf5_mode: default` and keep `phase` at any
  real value (e.g. `phase: nn-rl`) — it is ignored in default mode.

If the persisted config is stuck with `phase: no-comp`, patch it:
```bash
sed -i 's/^  phase: no-comp$/  phase: nn-rl/' \
    ~/.ppi-jarvis/config/pipelines/gpucompress_lammps_delta_single_node/pipeline.yaml
```

### `No gpuc_step_* directories past warmup_steps=<N> in <work_dir>`
The sim didn't run long enough to produce any dumps past the warmup
threshold. Either `warmup_steps` is larger than `timesteps * sim_interval`,
or `fix gpucompress` never fired. Check `lammps.log` for the
`[GPUCompress-LAMMPS] Step <N>: wrote 3 fields` lines.

### Kokkos build error `cudafe++ internal error` or C++20 issues
The Kokkos build rejects `lammps_ranking_profiler.cu` when it's added to
`KOKKOS_PKG_SOURCES`. Our `build.sh` builds it **separately** as
`liblammps_ranking_profiler.so` and links it via `CMAKE_EXE_LINKER_FLAGS`
to work around this. If you modify the build flow, keep the profiler out
of the Kokkos source list.

### NVCC internal compiler error at `lexical.c:22310`
You're on CUDA 12.8 with gcc-13. The YAML pins CUDA 12.6 specifically
to avoid this. If the error appears, double-check `deploy_base` in the
YAML and the `container_base` line.

### Rebuild from scratch
If you change source under `/u/$USER/GPUCompress/` or the base image,
wipe the cached SIF + apptainer cache and reload — the load step
triggers a fresh build:
```bash
rm -rf ~/.jarvis-cd/pipelines/gpucompress_lammps_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_lammps_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_lammps_delta_single_node/
apptainer cache clean --force
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_lammps_single_node.yaml
```

---

## File map

```
gpucompress_lammps_delta/
├── pkg.py               # Jarvis Application class; start/stop/clean lifecycle, CSV parser
├── build.sh             # apptainer fakeroot build: bridges + LAMMPS Kokkos fix + lmp
├── Dockerfile.deploy    # deploy-stage template (COPY /opt + /usr/local, LD_LIBRARY_PATH)
├── __init__.py          # empty, required for Python import
└── README.md            # this file
```

Paired pipeline YAMLs:
```
gpucompress_pkgs/pipelines/gpucompress_lammps_single_node.yaml           # adaptive
gpucompress_pkgs/pipelines/gpucompress_lammps_single_node_baseline.yaml  # no-comp
```
