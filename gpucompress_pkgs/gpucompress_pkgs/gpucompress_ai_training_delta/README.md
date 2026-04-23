# gpucompress_ai_training_delta

Jarvis Path-B package that trains a small neural network on Delta A100
nodes and benchmarks GPUCompress on the exported per-epoch checkpoints.
Runs entirely inside an Apptainer SIF.

AI-training is a **two-phase** workload:

- **Phase 1** — Python training: `train_and_export_checkpoints.py` loads
  a pretrained model, fine-tunes for a few epochs on a dataset, and at
  selected epochs exports four `.f32` tensor files per checkpoint
  (`weights`, `adam_m`, `adam_v`, `gradients`) into `<results>/checkpoints/`.
- **Phase 2** — the pkg flattens the exported `.f32` files into a single
  directory via symlinks and invokes `generic_benchmark` with the
  requested HDF5 mode / phase / policy. The benchmark treats each
  `.f32` as one "field" and measures per-algorithm ratio + ranking τ +
  regret across every field.

The shipped defaults (ResNet-18 + CIFAR-10, 5 epochs, 20 batches/epoch)
are smoke-scale so the end-to-end run completes in ~1 min on A100. The
YAML comments show how to scale up for paper-canonical runs (40 epochs,
uncapped batches).

**Pipeline YAMLs:**
- `gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node.yaml` — adaptive
- `gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node_baseline.yaml` — no-comp

---

## Prerequisites

- Delta account with an A100 allocation (set `$SLURM_ACCOUNT` to your own)
- Jarvis-CD installed and on `$PATH` (checked via `which jarvis`)
- ~12 GB of free quota under `/u/$USER/` (SIF ~8-12 GB since it ships
  PyTorch + torchvision + HuggingFace + h5py wheels)
- ~500 MB of free `/tmp` on the compute node for the CIFAR-10 cache
  (auto-downloaded on first run) plus checkpoints
- This repo cloned to `/u/$USER/GPUCompress` (paths in the YAML assume that)

---

## End-to-end, from zero (copy-paste ready)

Four phases: **clean → build → allocate → run+archive**.
`dt-loginNN` = login node (no GPU), `gpuaNNN` = compute node (A100).

### Phase 0 — Clean state (login node)

```bash
# Scan for stray CSVs first — never delete these
find ~/.jarvis-cd/pipelines/gpucompress_ai_training_delta_single_node -name '*.csv' 2>/dev/null
# If anything prints, move it aside before continuing.

rm -rf ~/.jarvis-cd/pipelines/gpucompress_ai_training_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_ai_training_delta_single_node/
rm -f  ~/build_ai.log ~/run_ai_*.log

# Jarvis validates hostfile: at load time — create a placeholder on login.
# The real compute-node hostname gets written in Phase 3.
echo "$(hostname)" > ~/hostfile_single.txt
```

Only wipe the apptainer cache (`apptainer cache clean --force`) if you
need to force a cold rebuild of `gpucompress_base`. Otherwise the
AI-training build reuses the base layer from any earlier workload and
finishes in ~5-8 min (mostly PyTorch + torchvision + HuggingFace wheels
downloading from PyPI).

### Phase 1 — Build the SIF (login node)

`jarvis ppl load yaml <path>` builds the SIF as part of loading when
`install_manager: container` is set in the YAML (see
`jarvis_cd/core/pipeline.py` — `_load_from_file` calls
`_build_pipeline_container()` at line 1024). There is no separate
`jarvis ppl build` step.

```bash
cd /u/$USER/GPUCompress/gpucompress_pkgs
jarvis repo add /u/$USER/GPUCompress/gpucompress_pkgs 2>/dev/null || true
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node.yaml 2>&1 | tee ~/build_ai.log

# Verify the SIF was produced
ls -lh ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/*.sif
```

Expected tail of log:
```
Apptainer SIF ready: .../gpucompress_ai_training_delta_single_node.sif
Loaded pipeline: gpucompress_ai_training_delta_single_node
```

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
hostname > ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/hostfile

# Enter pipeline
jarvis cd gpucompress_ai_training_delta_single_node

# Adaptive run — Phase 1 trains + exports .f32 checkpoints,
# Phase 2 replays with nn-rl+exp50 + balanced policy
jarvis ppl run 2>&1 | tee ~/run_ai_adaptive.log

# Baseline run — reload the sibling baseline YAML (hdf5_mode=default).
# Phase 1 re-trains from scratch (set skip_training=1 in the baseline
# YAML if you want to reuse the adaptive run's checkpoints).
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node_baseline.yaml
hostname > ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/hostfile
jarvis cd gpucompress_ai_training_delta_single_node
jarvis ppl run 2>&1 | tee ~/run_ai_baseline.log

# To go back to adaptive, reload the primary YAML:
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node.yaml
```

### Phase 4 — Archive results BEFORE releasing the allocation

`/tmp` is per-node and vanishes when the allocation ends.

```bash
STAMP=$(date +%Y%m%d_%H%M)
OUT=~/ai_training_results_${STAMP}
mkdir -p "$OUT"
cp -r /tmp/gpucompress_ai_training_*_vol      "$OUT/" 2>/dev/null
cp -r /tmp/gpucompress_ai_training_*_default  "$OUT/" 2>/dev/null
cp    ~/run_ai_adaptive.log  ~/run_ai_baseline.log  "$OUT/"

# Don't archive the CIFAR-10 cache (170 MB of non-experimental data)
find "$OUT" -type d -name 'dataset_cache' -prune -exec rm -rf {} \; 2>/dev/null

ls -la "$OUT/"
du -sh "$OUT"

exit   # release the allocation
```

---

## Configuration reference

All knobs live in `pkg.py:_configure_menu()`. **Configure them by editing
the pipeline YAML and re-running `jarvis ppl load yaml <path>`.** This is
the canonical workflow.

`jarvis pkg conf gpuc_ai_training <k>=<v>` also exists but is not
recommended — multi-key invocations are not atomic, and the persisted
config can drift out of sync with the source YAML, producing confusing
validation failures at the next `ppl run`. Edit the YAML, reload, done.

### Model / dataset

| Key | Default | Supported |
|---|---|---|
| `model` | `resnet18` | `resnet18`, `vit_b_16`, `vit_l_16`, `gpt2` |
| `dataset` | `cifar10` | `cifar10` (ResNet / ViT), `wikitext2` (GPT-2 only) |

Script-level constraint enforced in `pkg.py:_validate()`: `gpt2` requires
`wikitext2`, everything else requires `cifar10`.

### Training (Phase 1)

| Key | Default | Notes |
|---|---|---|
| `epochs` | `40` | Total training epochs (smoke YAML overrides to 5) |
| `checkpoint_epochs` | `"1,5,10,15,20,25,30,35,40"` | Comma-separated epochs at which to export `.f32` checkpoints |
| `batch_size` | `64` | Training batch size |
| `max_batches_per_epoch` | `20` | `0` = uncapped (paper default). Smoke YAML caps at 20 so each epoch is a few seconds |
| `skip_validate` | `1` | `1` skips the per-epoch validation pass, avoids a second CIFAR-10 forward pass. Validation accuracy then shows `nan%` in the log — expected |
| `skip_training` | `0` | `1` = skip Phase 1 entirely and reuse `{results_dir}/checkpoints/` from a previous run. Useful for iterating on Phase-2 policy without re-training |

### HDF5 mode / algorithm / policy (Phase 2)

| Key | Default | Meaning |
|---|---|---|
| `hdf5_mode` | `default` | `default` (no-comp baseline) or `vol` (GPUCompress VOL) |
| `phase` | `lz4` | `lz4`/`snappy`/`deflate`/`gdeflate`/`zstd`/`ans`/`cascaded`/`bitcomp` (fixed nvcomp) or `nn`/`nn-rl`/`nn-rl+exp50` (adaptive) |
| `policy` | `balanced` | NN cost-model weights: `balanced` (w=1,1,1), `ratio` (0,0,1), `speed` (1,1,0) |
| `error_bound` | `0.0` | Lossy tolerance; `0.0` = lossless. Inference deployment typically tolerates `1e-4 – 1e-3` |

### Benchmark / NN-online-learning knobs (Phase 2)

| Key | Default | Purpose |
|---|---|---|
| `chunk_mb` | `4` | HDF5 chunk size for Phase-2 replay |
| `verify` | `1` | Bitwise readback verify: `1` = on, `0` = skip |
| `sgd_lr` | `0.2` | SGD learning rate (`--lr`) |
| `sgd_mape` | `0.10` | MAPE threshold for SGD firing (`--mape`) |
| `explore_k` | `4` | Top-K exploration alternatives (`--explore-k`) |
| `explore_thresh` | `0.20` | Exploration error threshold (`--explore-thresh`) |

### Container / launch

| Key | Default | Notes |
|---|---|---|
| `cuda_arch` | `80` | A100 = 80. Must match `gpucompress_base`. |
| `deploy_base` | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | Pinned to CUDA 12.6 |
| `use_gpu` | `True` | Adds `--nv` to the apptainer invocation |
| `num_gpus` | `1` | PyTorch uses `cuda:0` by default regardless; this is a hint for future multi-GPU training |

---

## Expected outputs

Each run writes to `/tmp/gpucompress_ai_training_<pkg_id>_<model>_<dataset>_ep<N>_<mode>/`:

```
<results_dir>/
├── train.log                           # Phase-1 stdout+stderr
├── dataset_cache/                      # CIFAR-10 auto-download (~170 MB; ignore/skip in archive)
├── checkpoints/                        # Phase-1 output: per-epoch .f32 tensors
│   ├── epoch01_weights.f32  (42.7 MB for ResNet-18)
│   ├── epoch01_adam_m.f32
│   ├── epoch01_adam_v.f32
│   ├── epoch01_gradients.f32
│   ├── epoch03_*.f32
│   └── epoch05_*.f32
├── flat_fields/                        # symlinks consumed by Phase 2
│   └── epoch<NN>_<tensor>.f32
└── <mode>_<phase>/                     # e.g. vol_nn-rl+exp50 or default_no-comp
    ├── ai_bench.log                    # Phase-2 stdout+stderr
    ├── benchmark_<model>_<dataset>.csv           # per-field summary (primary CSV)
    ├── benchmark_<model>_<dataset>_ranking.csv   # Kendall τ + regret per field
    ├── benchmark_<model>_<dataset>_ranking_costs.csv  # NN-predicted vs observed costs
    └── gpucompress_vol_summary.txt     # lifetime I/O totals, VOL timing
```

### Success indicators

- Phase 1:
  - `Loading pretrained <model>...` then `Parameters: <N> (... MB)`
  - Per-epoch lines `Epoch N/M loss=... train_acc=... val_acc=nan%`
  - Per-checkpoint export lines: `epoch<NN>_<tensor>.f32 <size>`
  - `Checkpoint Export Complete` banner with total file count
- Phase 2:
  - `Fields: <N> files (*.f32)` (4 tensors × # checkpoint epochs)
  - Per-field ratio/MAPE/SGD/EXP columns
  - `Benchmark PASSED` printed near the end of `ai_bench.log`

### Typical numbers (ResNet-18 + CIFAR-10, smoke defaults)

| Metric | Value |
|---|---|
| Phase-1 wall time | ~15-20 s (5 epochs × 20 batches) |
| Phase-1 VOL ratio (reported in the training-script summary) | ~1.08× |
| Phase-2 per-field lossless ratios | 1.00-1.22× |
| Median τ across fields | ~0.73 |
| Median regret | 1.009× |
| Most-compressible tensor type | `adam_v` (squared gradients, sparse early in training) |

**These ratios are the paper's validation point**, not a bug:
neural-network weights / gradients / optimizer state are near-
incompressible under lossless constraints. Adaptive NN+RL still picks
within 1% of the optimal algorithm per field (regret ≈ 1.01×), so the
framework behaves correctly even on adversarial workloads.

### Benign artefacts to ignore

- `environment: line 17: /usr/share/lmod/lmod/libexec/lmod: No such
  file or directory` — harmless lmod call inside the container.
- `VisibleDeprecationWarning: dtype(): align should be passed...` from
  torchvision — cosmetic numpy/torchvision compatibility notice.
- `val_acc=nan%` per epoch — expected when `skip_validate=1`.
- Phase-1 VOL lifetime summary appearing **before** the training output
  in the log — output buffering artefact; the summary is emitted by the
  atexit hook when the Python script exits, but the training script's
  stdout was buffered and flushes at the same time.
- Post-`Benchmark PASSED` output like the SDRBench run-config hint
  (`Add to run_sdr.sh...`) — advice from the training script for an
  unrelated shell driver; harmless.

Unlike VPIC and WarpX, AI-training **does not** trigger the
library-destructor-order segfault at teardown — the pipeline exits
cleanly.

---

## Scaling up

| Scale | `epochs` | `checkpoint_epochs` | `max_batches_per_epoch` | # `.f32` files | Wall (A100, 1 GPU) |
|---|---|---|---|---|---|
| Smoke (default) | 5 | `"1,3,5"` | 20 | 12 | ~1-2 min |
| Small | 10 | `"1,3,5,7,10"` | 50 | 20 | ~5 min |
| Medium | 20 | `"1,5,10,15,20"` | 100 | 20 | ~15-20 min |
| Paper | 40 | `"1,5,10,15,20,25,30,35,40"` | 0 (uncapped) | 36 | ~1-2 h |

For GPT-2 + wikitext2 runs set `model: gpt2` and `dataset: wikitext2`.
Note that the GPT-2 training script is separate
(`train_gpt2_checkpoints.py`) and is invoked automatically when the
model selector is `gpt2`; it does not accept `--max-batches-per-epoch`
or `--no-validate` so those config values are silently ignored.

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
hostname > ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/hostfile
```

### Training hangs or `CUDA out of memory`
Reduce `batch_size` (default 64 → try 32 or 16) or switch to a smaller
model (`resnet18` is the smallest shipped option).

### `Read-only file system: '/opt/GPUCompress/data'`
The dataset-cache path wasn't propagated. pkg.py sets `--data-root` to
`{results_dir}/dataset_cache`, which is under `/tmp` and writable.
If you see this error, inspect `pkg.py:230-234` — the `data_root`
variable must be created with `Mkdir()` and passed as `--data-root` on
the training command line.

### `phase 'no-comp' not in [...]`
`no-comp` is not a valid `phase` value — it's an internal pseudo-phase
selected implicitly when `hdf5_mode=default`. Either:
- Use the baseline YAML variant (recommended):
  ```bash
  jarvis ppl load yaml .../gpucompress_ai_training_single_node_baseline.yaml
  ```
- Or in your own YAML, set `hdf5_mode: default` and keep `phase` at any
  real value (e.g. `phase: nn-rl+exp50`) — it is ignored in default mode.

If the persisted config is stuck with `phase: no-comp`, patch it:
```bash
sed -i 's/^  phase: no-comp$/  phase: nn-rl+exp50/' \
    ~/.ppi-jarvis/config/pipelines/gpucompress_ai_training_delta_single_node/pipeline.yaml
```

### `No .f32 checkpoint files under <ckpt_dir>`
Phase 1 finished without producing checkpoints. Check `train.log` for
the training progress — if epochs ran but no `Exporting checkpoint`
lines appeared, the `checkpoint_epochs` list may not include any of the
actual epoch numbers (e.g. `checkpoint_epochs: "10"` with `epochs: 5`).

### Rebuild from scratch
If you change source under `/u/$USER/GPUCompress/` or the base image,
wipe the cached SIF + apptainer cache and reload — the load step
triggers a fresh build:
```bash
rm -rf ~/.jarvis-cd/pipelines/gpucompress_ai_training_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_ai_training_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_ai_training_delta_single_node/
apptainer cache clean --force
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node.yaml
```

---

## File map

```
gpucompress_ai_training_delta/
├── pkg.py               # Jarvis Application class; two-phase (train + bench) start/stop/clean lifecycle
├── build.sh             # apptainer fakeroot build: pip-installs PyTorch + torchvision + HuggingFace + h5py
├── Dockerfile.deploy    # deploy-stage template (COPY /opt + /usr/local, LD_LIBRARY_PATH + PYTHONPATH)
├── __init__.py          # empty, required for Python import
└── README.md            # this file
```

Paired pipeline YAMLs:
```
gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node.yaml           # adaptive
gpucompress_pkgs/pipelines/gpucompress_ai_training_single_node_baseline.yaml  # no-comp
```
