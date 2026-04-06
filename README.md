# GPUCompress

A GPU-accelerated compression library with neural network–driven algorithm selection and online reinforcement learning, designed for HPC in-situ I/O.

## Key Innovation

GPUCompress replaces static compression choices with a lightweight neural network that evaluates **all 32 compression configurations** (8 algorithms × 2 preprocessing options) in a single GPU kernel (~0.22 ms), selecting the best one per data chunk based on learned data characteristics.  An online SGD loop adapts the model in real time — demonstrated MAPE drops from ~700% to ~10% within 20 timesteps on unseen VPIC plasma data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application / HDF5                    │
│         (Gray-Scott, VPIC-Kokkos, SDRBench, ...)        │
├─────────────────────────────────────────────────────────┤
│              HDF5 VOL Connector (2,744 LOC)             │
│   Intercepts H5Dwrite/H5Dread, detects GPU pointers,   │
│   routes to GPU-native compress/decompress pipeline     │
├─────────────────────────────────────────────────────────┤
│                   GPUCompress C API                      │
│   gpucompress_compress_gpu() / gpucompress_decompress() │
├────────┬────────┬──────────┬────────────┬───────────────┤
│ Stats  │   NN   │ Cost     │ Compress/  │   Online      │
│Kernels │Infer-  │ Model    │ Decompress │   Learning    │
│entropy,│ence    │log-space │ via nvCOMP │   SGD +       │
│MAD,    │15→128→ │policy-   │ 8 algos    │   Exploration │
│2nd-deriv│128→4  │controlled│            │               │
├────────┴────────┴──────────┴────────────┴───────────────┤
│              NVIDIA nvCOMP 5.1.0 + CUDA 12.8+          │
└─────────────────────────────────────────────────────────┘
```

## Compression Algorithms

All 8 algorithms are GPU-accelerated via nvCOMP:

| Algorithm  | Speed     | Ratio | Notes                              |
|------------|-----------|-------|------------------------------------|
| LZ4        | Fastest   | Low   | General-purpose, always competitive|
| Snappy     | Very Fast | Low   | Byte-oriented, fast decode         |
| Deflate    | Medium    | Med   | CPU-style, slower on GPU           |
| Gdeflate   | Medium    | Med   | GPU-optimized deflate variant      |
| Zstd       | Slow      | High  | Best ratio for low-entropy data    |
| ANS        | Slow      | High  | Entropy coding, structured data    |
| Cascaded   | Slow      | High  | Floating-point specific            |
| Bitcomp    | Slow      | High  | Bit-level compression              |

---

## Prerequisites

- **Linux** (RHEL 9+, Ubuntu 20.04+)
- **NVIDIA GPU** with compute capability >= 7.0
- **CUDA Toolkit** >= 12.0
- **NVIDIA driver** >= 525.60.13
- **cmake** >= 3.18
- **g++** >= 9.0

## Installation on NCSA Delta

Tested on **Delta** (A100-SXM4-40GB, x86_64, CUDA 12.8, Cray MPICH 8.1.32).

> **Delta vs DeltaAI:** The instructions below target Delta (A100, x86_64, `--account=bekn-delta-gpu`). For DeltaAI (GH200, ARM aarch64), use `--account=bekn-dtai-gh`, `--partition=ghx4`, and `CUDA_ARCH=90`. The SLURM scripts in `benchmarks/slurm/` are pre-configured for DeltaAI.

### Step 1: Clone the repository

```bash
cd /u/$USER
git clone <repo-url> GPUCompress
cd GPUCompress
```

### Step 2: Load required modules

```bash
module load cuda
module load gcc-native/13.2 cray-mpich/8.1.32
```

### Step 3: Install dependencies and build (on a compute node)

The install script downloads **nvcomp 5.1.0** and **HDF5 2.0.0**, builds GPUCompress, and downloads SDRBench datasets. It must run on a node with a GPU.

```bash
# Request an interactive compute node and run the full install
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  bash scripts/install_dependencies.sh
```

This does 4 things:
1. Installs nvcomp to `/tmp/include` and `/tmp/lib`
2. Installs HDF5 to `/tmp/hdf5-install`
3. Builds the project in `build/`
4. Downloads SDRBench datasets (Hurricane Isabel, Nyx, CESM-ATM) into `data/sdrbench/`

> **Note:** Dependencies in `/tmp` are node-local and deleted after the job ends.

**For multi-node runs**, install deps to the shared filesystem so every node can access them:

```bash
NVCOMP_INSTALL_DIR=/u/$USER/GPUCompress/.deps \
HDF5_INSTALL_DIR=/u/$USER/GPUCompress/.deps/hdf5 \
  bash scripts/install_dependencies.sh --node-local-only
```

This only installs nvcomp and HDF5 (no build, no dataset download). Run it on each node via `srun --ntasks-per-node=1`.

### Step 4: Set up the environment

```bash
source scripts/setup_env.sh
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
```

For shared-filesystem deps (multi-node), use:

```bash
export LD_LIBRARY_PATH=$PWD/.deps/lib:$PWD/.deps/hdf5/lib:$PWD/build:$LD_LIBRARY_PATH
```

### Step 5: Verify the build

The smoke test also requires a GPU:

```bash
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:10:00 \
  bash scripts/smoke_test.sh
```

### Step 6 (optional): Generate AI training checkpoint data

To generate CIFAR-10 ViT checkpoint data for NN training experiments:

```bash
bash scripts/install_dependencies.sh --with-ai-training
```

This trains a ViT-B/16 model and exports weight checkpoints into `data/ai_training/`.

---

## Build Targets

| Target | Output | Description |
|--------|--------|-------------|
| `gpucompress` | `libgpucompress.so` | Core compression library |
| `gpu_compress` | CLI tool | Command-line compressor (requires cuFile) |
| `gpu_decompress` | CLI tool | Command-line decompressor |
| `H5Zgpucompress` | `libH5Zgpucompress.so` | HDF5 filter plugin |
| `H5VLgpucompress` | `libH5VLgpucompress.so` | HDF5 VOL connector |
| `benchmark` | executable | GPU benchmark harness |

To rebuild manually:

```bash
cmake -B build \
  -DNVCOMP_PREFIX=/tmp \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

---

## Running Applications

### 1. GPU Benchmark (standalone compression evaluation)

Evaluates all compression algorithms and NN selection on synthetic or real data.

```bash
# On a compute node:
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  bash -c 'source scripts/setup_env.sh && ./build/benchmark'
```

### 2. Benchmark Suite (full evaluation pipeline)

The unified benchmark script (`benchmarks/benchmark.sh`) runs a 12-phase evaluation pipeline comparing all system modes across different workloads.

**Phases:**

| Phase | Description |
|-------|-------------|
| `no-comp` | Uncompressed baseline |
| `lz4` | Always LZ4 (speed extreme) |
| `snappy` | Always Snappy |
| `deflate` | Always Deflate |
| `gdeflate` | Always Gdeflate (GPU-optimized) |
| `zstd` | Always Zstd (ratio extreme) |
| `ans` | Always ANS |
| `cascaded` | Always Cascaded |
| `bitcomp` | Always Bitcomp |
| `nn` | NN inference (static, no learning) |
| `nn-rl` | NN + SGD (online learning) |
| `nn-rl+exp50` | NN + SGD + exploration (full system) |

**Available workloads:** `grayscott`, `vpic`, `sdrbench`

#### Run Gray-Scott (reaction-diffusion simulation)

```bash
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=01:00:00 \
  bash -c '
    module load cuda
    source scripts/setup_env.sh
    export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
    BENCHMARKS=grayscott DATA_MB=256 TIMESTEPS=25 POLICIES=balanced \
      bash benchmarks/benchmark.sh
  '
```

Results are written to `benchmarks/grayscott/results/`.

#### Run VPIC (plasma particle-in-cell simulation)

VPIC requires building the simulation binary first:

```bash
# 1. Build VPIC binary (one-time, on login node)
module load gcc-native/13.2 cray-mpich/8.1.32
cd benchmarks/vpic-kokkos && bash build_vpic_pm.sh && cd ../..

# 2. Run VPIC benchmark (on a compute node)
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=01:00:00 \
  bash -c '
    module load cuda
    source scripts/setup_env.sh
    export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
    BENCHMARKS=vpic VPIC_NX=128 DATA_MB=256 TIMESTEPS=25 POLICIES=balanced \
      bash benchmarks/benchmark.sh
  '
```

Results are written to `benchmarks/vpic-kokkos/results/`. See [`deltaRunVPICParameters.md`](deltaRunVPICParameters.md) for recommended production parameters on Delta (NX=320, 4n×4g, physics tuning).

#### Run SDRBench (scientific datasets)

Requires SDRBench datasets in `data/sdrbench/` (downloaded automatically during install).

```bash
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=01:00:00 \
  bash -c '
    module load cuda
    source scripts/setup_env.sh
    export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
    BENCHMARKS=sdrbench DATA_MB=256 TIMESTEPS=25 POLICIES=balanced \
      bash benchmarks/benchmark.sh
  '
```

Results are written to `benchmarks/sdrbench/results/`.

#### Benchmark environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARKS` | `grayscott,vpic,sdrbench` | Which workloads to run |
| `DATA_MB` | `512` | Per-snapshot data size (MB) |
| `CHUNK_MB` | `16` | HDF5 chunk size (MB) |
| `TIMESTEPS` | `50` | Number of write cycles |
| `POLICIES` | `balanced,ratio,speed` | Cost model policies for NN phases |
| `VERIFY` | `1` | Bitwise verification (0 to skip) |
| `MPI_NP` | `1` | Total MPI ranks (for multi-GPU) |
| `GPUS_PER_NODE` | `1` | GPUs per node |

### 3. Multi-GPU / Multi-Node Benchmarks (SLURM batch)

Pre-configured SLURM wrapper scripts are in `benchmarks/slurm/`:

```bash
# 1 node × 4 GPUs
bash benchmarks/slurm/deltaai_1n4g.sh

# 2 nodes × 2 GPUs
bash benchmarks/slurm/deltaai_2n2g.sh

# 4 nodes × 4 GPUs (16 total)
bash benchmarks/slurm/deltaai_4n4g.sh

# Or submit directly with custom config:
BENCHMARKS=vpic DATA_MB=512 TIMESTEPS=50 \
  sbatch -N2 --gpus-per-node=4 --ntasks-per-node=4 \
  benchmarks/slurm/deltaai_benchmark.sbatch
```

For interactive multi-GPU runs:

```bash
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N1 --gpus-per-node=2 --ntasks-per-node=2 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

# On the compute node:
cd /u/$USER/GPUCompress
export LD_LIBRARY_PATH=$PWD/.deps/lib:$PWD/.deps/hdf5/lib:$PWD/build:$LD_LIBRARY_PATH
MPI_NP=2 GPUS_PER_NODE=2 BENCHMARKS=vpic VPIC_NX=160 TIMESTEPS=5 \
  bash benchmarks/benchmark.sh
```

### 4. CLI Tools (compress/decompress individual files)

```bash
# Compress a binary file (auto-selects algorithm via NN)
./build/gpu_compress -i input.bin -o output.bin -algo auto -chunk-mb 4

# Decompress (algorithm auto-detected from header)
./build/gpu_decompress -i output.bin -o recovered.bin
```

### 5. HDF5 Integration

Use GPUCompress as a transparent HDF5 compression layer:

```bash
# As HDF5 filter plugin
export HDF5_PLUGIN_PATH=$PWD/build

# As HDF5 VOL connector (GPU-native I/O)
export HDF5_VOL_CONNECTOR=gpucompress
export HDF5_PLUGIN_PATH=$PWD/build
```

---

## Simulation Integrations

GPUCompress provides zero-copy GPU adapters for 5 HPC simulation codes. Each has detailed deployment instructions (clone, patch, build, run) in its own README:

| Simulation | Description | Integration method | README |
|------------|-------------|-------------------|--------|
| **VPIC-Kokkos** | Plasma particle-in-cell | Pre-built benchmark binary | [`benchmarks/vpic-kokkos/`](benchmarks/vpic-kokkos/) |
| **LAMMPS** | Molecular dynamics (Kokkos GPU) | LAMMPS fix + 2 new source files + cmake patch | [`benchmarks/lammps/README.md`](benchmarks/lammps/README.md) |
| **nekRS** | Spectral-element CFD | Case-level UDF only (zero nekRS source changes) | [`benchmarks/nekrs/README.md`](benchmarks/nekrs/README.md) |
| **Nyx** | AMReX cosmological hydro | 3 patched source files (`#ifdef` guarded) | [`benchmarks/nyx/README.md`](benchmarks/nyx/README.md) |
| **WarpX** | Plasma acceleration (AMReX) | Direct adapter API or AMReX bridge | [`benchmarks/warpx/README.md`](benchmarks/warpx/README.md) |

Each integration follows the same pattern: GPU device pointers from the simulation are passed directly through the HDF5 VOL connector without CPU round-trips. Patches for each simulation are provided in `benchmarks/<sim>/patches/`.

---

## Neural Network Model

### Architecture

```
Input [15 features] → ReLU [128] → ReLU [128] → Output [4 predictions]
```

**15 input features:**
- One-hot algorithm encoding (8 values)
- Quantization flag, shuffle flag
- `log10(error_bound)`, `log2(data_size)`
- Shannon entropy, normalized MAD, normalized 2nd derivative

**4 output predictions:**
- `log1p(compression_time_ms)`
- `log1p(decompression_time_ms)`
- `log1p(compression_ratio)` — primary ranking target
- PSNR (clamped to 120 dB)

**Parameters:** ~19,076 floats (~76 KB), evaluated for all 32 configs in parallel via 32-thread GPU kernel.

### Cost Model (Scale-Invariant Ranking)

```
cost = α · log(ct + γ · dt) + β · log(data_size / (ratio · bw_eff)) − δ · log(ratio)
```

Policy presets:
- **Speed** (α=1, β=0, δ=0): minimize compute time
- **Balanced** (α=1, β=1, δ=0.5): balance compute + I/O + ratio
- **Ratio-First** (α=0.3, β=1, δ=1): maximize compression

### Online Learning System

Three-level adaptation that runs during compression:

1. **Experience Logging** — every chunk records (action, features, actual ratio/time)
2. **Exploration** — when |predicted − actual| / actual > threshold, tries K alternative configurations and picks the best
3. **SGD Reinforcement** — updates output-layer weights using measured ground truth; gradient-clipped, heads-only for stability

Convergence on VPIC data: MAPE ~700% (cold start) → ~10% after 20 timesteps.

### Training the NN (optional)

Pre-trained weights ship in `neural_net/weights/model.nnwt`. To retrain:

```bash
# Generate training data via benchmarks
python3 neural_net/training/benchmark.py

# Train
python3 neural_net/training/train.py

# Export to binary format
python3 neural_net/export/export_weights.py
```

See [`neural_net/docs/TUTORIAL.md`](neural_net/docs/TUTORIAL.md) for the full training walkthrough.

---

## Project Structure

```
GPUCompress/
├── include/                    # Public C API headers
│   ├── gpucompress.h           #   Main API
│   ├── gpucompress_hdf5.h      #   HDF5 filter
│   ├── gpucompress_hdf5_vol.h  #   HDF5 VOL connector
│   ├── gpucompress_vpic.h      #   VPIC adapter
│   ├── gpucompress_lammps.h    #   LAMMPS adapter
│   ├── gpucompress_nekrs.h     #   nekRS adapter
│   ├── gpucompress_nyx.h       #   Nyx adapter
│   ├── gpucompress_warpx.h     #   WarpX adapter
│   └── gpucompress_grayscott.h #   Gray-Scott adapter
│
├── src/
│   ├── api/                    # Core implementation (~7K lines)
│   ├── nn/nn_gpu.cu            # GPU NN inference + SGD kernels
│   ├── stats/                  # Feature extraction (entropy, MAD, 2nd deriv)
│   ├── compression/            # nvCOMP wrapper factory (8 algorithms)
│   ├── preprocessing/          # Byte shuffle + quantization kernels
│   ├── selection/heuristic.cu  # Entropy-threshold baseline selector
│   ├── hdf5/                   # VOL connector + filter plugin
│   └── cli/                    # Command-line compress/decompress tools
│
├── neural_net/
│   ├── core/                   # PyTorch model, data loading, configs
│   ├── training/               # Train, cross-validate, retrain
│   ├── inference/              # CPU-side prediction + evaluation
│   ├── export/                 # PyTorch → binary .nnwt export
│   ├── weights/model.nnwt      # Pre-trained weights (shipped)
│   └── docs/                   # Architecture, tutorial, execution flow
│
├── benchmarks/
│   ├── benchmark.sh            # Unified benchmark entry point
│   ├── grayscott/              # Gray-Scott benchmark driver
│   ├── vpic-kokkos/            # VPIC benchmark driver
│   ├── sdrbench/               # SDRBench (Hurricane, Nyx, CESM) driver
│   ├── lammps/                 # LAMMPS integration + patches
│   ├── nekrs/                  # nekRS integration + patches
│   ├── nyx/                    # Nyx integration + patches
│   ├── warpx/                  # WarpX integration + patches
│   ├── slurm/                  # SLURM job scripts for Delta
│   └── visualize.py            # Publication-quality figure generation
│
├── tests/
│   ├── regression/             # 50+ regression tests
│   └── run_all_tests.sh        # Test runner
│
├── scripts/
│   ├── install_dependencies.sh # Automated dependency installation
│   ├── setup_env.sh            # Environment variable setup
│   ├── smoke_test.sh           # Quick validation test
│   └── run_tests.sh            # Test runner
│
├── cmake/                      # Modular CMake build system
├── docs/                       # Technical deep dives
└── data/                       # Benchmark datasets
```

## Quick API Example

```c
#include "gpucompress.h"

// Initialize with pre-trained NN weights
gpucompress_init("neural_net/weights/model.nnwt");

// Enable online learning
gpucompress_enable_online_learning();
gpucompress_set_exploration(1);

// Configure cost model (balanced policy)
gpucompress_set_ranking_weights(1.0, 1.0, 0.5);

// Compress GPU data — NN selects best algorithm automatically
gpucompress_config_t cfg = {.error_bound = 0.0, .chunk_size = 16*1024*1024};
gpucompress_stats_t stats;
gpucompress_compress_gpu(d_input, n_bytes, d_output, &out_size, &cfg, &stats, stream);

// Decompress — algorithm auto-detected from header
gpucompress_decompress_gpu(d_compressed, comp_size, d_output, &out_size, stream);
```

## Tests

```bash
# Run all tests (on a compute node)
bash tests/run_all_tests.sh

# Run specific regression test
./build/tests/test_nn_ratio_prediction
```

50+ regression tests covering concurrency, memory safety, NN accuracy, SGD convergence, integer overflow, and RAII cleanup.

## Generating Figures

After running benchmarks, generate publication-quality plots:

```bash
python3 benchmarks/visualize.py --view summary --view timesteps
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `libgpucompress.so: cannot open shared object` | Set `LD_LIBRARY_PATH` to include the `build/` directory |
| `libnvcomp.so: cannot open shared object` | Run `source scripts/setup_env.sh` or set `LD_LIBRARY_PATH` to include nvcomp lib dir |
| `nvcc not found` | Run `module load cuda` |
| `No CUDA-capable device` | You are on a login node — use `srun` or `salloc` to get a compute node |
| SDRBench data missing | Re-run `bash scripts/install_dependencies.sh` (Step 4 downloads datasets) |
| VPIC binary not found | Build it first: `cd benchmarks/vpic-kokkos && bash build_vpic_pm.sh` |
| `/tmp` deps gone after job | `/tmp` is node-local and wiped after jobs — use `.deps/` for persistent installs |

## Documentation

- [`deltaRunVPICParameters.md`](deltaRunVPICParameters.md) — Recommended VPIC parameters for Delta (4n×4g production config)
- [`docs/multi_gpu_guide.md`](docs/multi_gpu_guide.md) — Multi-GPU/multi-node execution guide
- [`neural_net/docs/ARCHITECTURE.md`](neural_net/docs/ARCHITECTURE.md) — NN architecture details
- [`neural_net/docs/NN_EXECUTION_FLOW.md`](neural_net/docs/NN_EXECUTION_FLOW.md) — 5-timestep execution walkthrough
- [`neural_net/docs/TUTORIAL.md`](neural_net/docs/TUTORIAL.md) — Training tutorial
