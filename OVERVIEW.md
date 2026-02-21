# GPUCompress: Project Overview

GPUCompress is a GPU-accelerated compression library that uses reinforcement learning to automatically select the best compression algorithm based on data characteristics. The project has three major components:

1. **GPU Compression Library** (`src/`, `include/`) -- A C/CUDA library wrapping 8 nvCOMP algorithms with optional byte-shuffle and quantization preprocessing, exposed through both a C API (`libgpucompress.so`) and CLI tools (`gpu_compress`, `gpu_decompress`).

2. **RL Training System** (`rl/`) -- An offline Q-learning pipeline that trains a Q-table mapping data features (entropy, MAD, smoothness) to optimal compression configurations. The trained table is loaded into GPU constant memory at inference time.

3. **Synthetic Data Generator** (`syntheticGeneration/`) -- A palette-based generator that produces datasets with precisely controlled entropy, locality, and value distributions for training the RL agent.

---

## Architecture

```
 TRAINING (offline, CPU+GPU)                  INFERENCE (runtime, GPU)
 ============================                 =========================

 generator.py                                 Application Code
      |                                            |
      | .bin files with controlled                 | raw data buffer
      | entropy characteristics                    v
      v                                     gpucompress_compress()
 trainer.py                                        |
      |                                            |--- calculate entropy,
      |--- load file                               |    MAD, 2nd derivative
      |--- calculate entropy, MAD, deriv           |
      |--- encode state (1024 states)              |--- encode state
      |--- select action (epsilon-greedy)          |--- Q-table lookup (GPU
      |--- execute gpu_compress                    |    constant memory)
      |--- measure ratio, throughput, PSNR         |--- decode action
      |--- compute reward                          |--- apply preprocessing
      |--- Q(s,a) += alpha * (r - Q(s,a))         |    (quantize, shuffle)
      v                                            |--- compress (nvCOMP)
 qtable.json / qtable.bin                          v
      |                                     compressed buffer +
      +-------- deployed to -------->>      64-byte header
```

---

## GPU Compression Library

### C API (`include/gpucompress.h`)

The public API follows an init/use/cleanup lifecycle:

```c
#include "gpucompress.h"

// 1. Initialize (optionally load Q-table for AUTO mode)
gpucompress_init("/path/to/qtable.bin");

// 2. Configure
gpucompress_config_t cfg = gpucompress_default_config();
cfg.algorithm     = GPUCOMPRESS_ALGO_AUTO;  // RL-based selection
cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4 | GPUCOMPRESS_PREPROC_QUANTIZE;
cfg.error_bound   = 0.001;

// 3. Compress
gpucompress_stats_t stats;
gpucompress_compress(input, input_size, output, &output_size, &cfg, &stats);

// 4. Decompress (auto-detects settings from header)
gpucompress_decompress(compressed, comp_size, output, &output_size);

// 5. Cleanup
gpucompress_cleanup();
```

**Key types and constants:**

| Symbol | Description |
|--------|-------------|
| `GPUCOMPRESS_HEADER_SIZE` | 64 bytes prepended to every compressed buffer |
| `GPUCOMPRESS_MAGIC` | `0x43555047` ("GPUC" in little-endian) |
| `gpucompress_config_t` | Algorithm, preprocessing bitmask, error bound, CUDA device/stream |
| `gpucompress_stats_t` | Original/compressed sizes, ratio, entropy, MAD, 2nd derivative, throughput |

### Algorithms

| Enum | Name | Characteristics |
|------|------|-----------------|
| `GPUCOMPRESS_ALGO_AUTO` (0) | RL auto-select | Uses Q-table to pick best option |
| `GPUCOMPRESS_ALGO_LZ4` (1) | LZ4 | Fast, general purpose |
| `GPUCOMPRESS_ALGO_SNAPPY` (2) | Snappy | Fastest, lower ratio |
| `GPUCOMPRESS_ALGO_DEFLATE` (3) | Deflate | Better ratio, slower |
| `GPUCOMPRESS_ALGO_GDEFLATE` (4) | GDeflate | GPU-optimized deflate |
| `GPUCOMPRESS_ALGO_ZSTD` (5) | Zstd | Best ratio, configurable |
| `GPUCOMPRESS_ALGO_ANS` (6) | ANS | Entropy coding, numerical data |
| `GPUCOMPRESS_ALGO_CASCADED` (7) | Cascaded | High compression, floating-point |
| `GPUCOMPRESS_ALGO_BITCOMP` (8) | Bitcomp | Lossless, scientific data |

### Preprocessing

Preprocessing flags are OR'd together in `gpucompress_config_t.preprocessing`:

| Flag | Value | Effect |
|------|-------|--------|
| `GPUCOMPRESS_PREPROC_NONE` | `0x00` | No preprocessing |
| `GPUCOMPRESS_PREPROC_SHUFFLE_2` | `0x01` | 2-byte element shuffle |
| `GPUCOMPRESS_PREPROC_SHUFFLE_4` | `0x02` | 4-byte element shuffle |
| `GPUCOMPRESS_PREPROC_SHUFFLE_8` | `0x04` | 8-byte element shuffle |
| `GPUCOMPRESS_PREPROC_QUANTIZE` | `0x10` | Linear quantization (lossy) |

**Byte shuffle** reorders bytes so that the most-significant bytes of consecutive elements are grouped together (improves entropy coding). **Quantization** maps float values to integers within an error bound: `scale = 1 / (2 * error_bound)`.

### Compression Header (`src/core/compression_header.h`)

Every compressed buffer is prefixed with a 64-byte header:

```
Offset  Size  Field                    Description
------  ----  -----                    -----------
 0       4    magic                    0x43555047 ("GPUC")
 4       4    version                  Header format version (currently 2)
 8       4    shuffle_element_size     0 = no shuffle, 2/4/8 = shuffled
12       4    quant_flags              Bits [0-3]=type, [4-7]=precision, [8]=enabled
16       8    original_size            Uncompressed data size in bytes
24       8    compressed_size          Compressed payload size (excludes header)
32       8    quant_error_bound        Absolute error bound used
40       8    quant_scale              Scale factor = 1 / (2 * error_bound)
48       8    data_min                 Minimum value in original data
56       8    data_max                 Maximum value in original data
```

Decompression reads this header to auto-detect all settings (algorithm, shuffle, quantization parameters) so no external metadata is needed.

### GPU Memory API

For data already resident on the GPU, use `gpucompress_compress_gpu()` and `gpucompress_decompress_gpu()` to avoid host-device transfers. Both accept a CUDA stream for async operation.

### HDF5 Filter Plugin (`src/hdf5/H5Zgpucompress.c`)

Registers as HDF5 filter ID **305** under the name `"gpucompress"`. Usage:

```bash
export HDF5_PLUGIN_PATH=/path/to/plugins
# Any HDF5 reader will automatically decompress datasets that used this filter
```

Or in application code:

```c
H5Z_gpucompress_register();
H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);
```

### CLI Tools

**`gpu_compress`** -- Reads input via GPUDirect Storage, compresses on GPU, writes output:

```
gpu_compress <input> <output> [algorithm] [options]

Options:
  --shuffle <size>         Byte shuffle element size: 2, 4, 8 (0 = disabled)
  --quant-type <type>      Quantization: linear (default: none)
  --error-bound <value>    Absolute error bound (required with --quant-type)

Examples:
  gpu_compress input.bin out.bin lz4
  gpu_compress input.bin out.bin lz4 --shuffle 4
  gpu_compress input.bin out.bin zstd --quant-type linear --error-bound 0.001 --shuffle 4
```

**`gpu_decompress`** -- Reads header, auto-detects all settings, decompresses:

```
gpu_decompress <compressed_input> <output>
```

### Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `SUCCESS` | Operation completed |
| -1 | `ERROR_INVALID_INPUT` | Bad parameter |
| -2 | `ERROR_CUDA_FAILED` | CUDA operation failed |
| -3 | `ERROR_COMPRESSION` | Compression failed |
| -4 | `ERROR_DECOMPRESSION` | Decompression failed |
| -5 | `ERROR_OUT_OF_MEMORY` | Allocation failed |
| -6 | `ERROR_QTABLE_NOT_LOADED` | Q-table required for AUTO but not loaded |
| -7 | `ERROR_INVALID_HEADER` | Corrupt or missing header |
| -8 | `ERROR_NOT_INITIALIZED` | `gpucompress_init()` not called |
| -9 | `ERROR_BUFFER_TOO_SMALL` | Output buffer insufficient |

---

## RL Training System

### State Space (1024 states)

Each data block is characterized by four features, each discretized into bins:

| Feature | Bins | Thresholds | Meaning |
|---------|------|------------|---------|
| Shannon entropy | 16 | 0.5-bit width: [0.0, 0.5), [0.5, 1.0), ..., [7.5, 8.0) | Byte-level randomness |
| Error level | 4 | `>= 0.1` aggressive, `>= 0.01` moderate, `>= 0.001` precise, `<= 0` lossless | Quantization tolerance |
| MAD | 4 | `< 0.05`, `< 0.15`, `< 0.30`, `>= 0.30` | Data spread |
| 2nd derivative | 4 | `< 0.05`, `< 0.15`, `< 0.35`, `>= 0.35` | Smoothness |

**State encoding** (mixed-radix):

```
state = ((entropy_bin * 4 + error_level) * 4 + mad_bin) * 4 + deriv_bin
```

**State decoding:**

```
deriv_bin    = state % 4;     state /= 4
mad_bin      = state % 4;     state /= 4
error_level  = state % 4
entropy_bin  = state / 4
```

### Action Space (32 actions)

Each action encodes three choices:

| Component | Options | Count |
|-----------|---------|-------|
| Algorithm | lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp | 8 |
| Quantization | None, Linear | 2 |
| Byte shuffle | None, 4-byte | 2 |

**Action encoding:**

```
action = algorithm_idx + (quant_flag * 8) + (shuffle_flag * 16)
```

**Action decoding:**

```
algorithm_idx    = action % 8
use_quantization = (action / 8) % 2 == 1
shuffle_size     = (action / 16) % 2 == 1 ? 4 : 0
```

When operating in lossless mode (`error_bound <= 0`), quantization actions are masked out, reducing the effective action space to 16.

### Q-Learning Update

Single-step Q-learning with no discount (each compression is an independent decision):

```
Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))
```

| Hyperparameter | Value |
|----------------|-------|
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.0 |
| Epsilon start | 1.0 |
| Epsilon end | 0.01 |
| Epsilon decay (per epoch) | 0.995 |
| Default epochs | 100 |
| Checkpoint interval | 10 epochs |

### Reward Computation

Reward is a weighted sum of three normalized metrics:

```
reward = w_ratio * min(ratio / 10, 1)
       + w_throughput * min(throughput_mbps / 5000, 1)
       + w_psnr * min(psnr_db / 100, 1)
```

For lossless compression, PSNR is infinity and `psnr_norm = 1.0`.

**Presets:**

| Preset | Ratio | Throughput | PSNR | Use case |
|--------|-------|------------|------|----------|
| `balanced` | 0.5 | 0.3 | 0.2 | General purpose |
| `max_ratio` | 0.8 | 0.1 | 0.1 | Smallest files |
| `max_speed` | 0.1 | 0.8 | 0.1 | Fastest compression |
| `max_quality` | 0.1 | 0.1 | 0.8 | Minimize distortion |
| `storage` | 0.6 | 0.2 | 0.2 | Archival |
| `streaming` | 0.3 | 0.5 | 0.2 | Real-time streaming |

### Exploration Policy

Epsilon-greedy with exponential decay:

- With probability epsilon: select a random valid action (explore)
- With probability 1 - epsilon: select `argmax Q(s, :)` (exploit)
- After each epoch: `epsilon = max(epsilon_end, epsilon * 0.995)`

A `GreedyPolicy` class (epsilon = 0) is used for inference after training.

### Q-Table Persistence

**JSON format** (`qtable.json`) -- Human-readable, includes visit counts:

```json
{
  "version": 2,
  "n_states": 1024,
  "n_actions": 32,
  "q_values": [[...], ...],
  "visit_counts": [[...], ...]
}
```

**Binary format** (`qtable.bin`) -- For GPU loading:

```
Offset  Size                   Field
------  ----                   -----
 0       4                     Magic: 0x51544142 ("QTAB")
 4       4                     Version: 1
 8       4                     n_states: 1024
12       4                     n_actions: 32
16       1024 * 32 * 4 = 128KB q_values (float32, row-major)
```

### Training CLI

```
python -m rl.trainer [options]

Required:
  -d, --data-dir DIR        Directory with .bin/.raw/.dat training files

Optional:
  -o, --output-dir DIR      Model output directory (default: rl/models)
  -e, --epochs N            Training epochs (default: 100)
  -p, --preset NAME         Reward preset (default: balanced)
  --error-bound VALUE       Single bound or "all" for all 4 levels
  --resume PATH             Resume from previous qtable.json
  --gpu-compress PATH       Path to gpu_compress binary (default: ./build/gpu_compress)
  --use-c-api               Use libgpucompress.so for entropy (faster)
  -q, --quiet               Reduce output
  --clean                   Remove all Q-table files from output directory
```

### Module Structure

| File | Role |
|------|------|
| `rl/config.py` | All constants: state/action dimensions, hyperparameters, reward presets |
| `rl/qtable.py` | Q-table data structure, encode/decode state/action, save/load, export binary |
| `rl/policy.py` | `EpsilonGreedyPolicy` (training) and `GreedyPolicy` (inference) |
| `rl/reward.py` | `compute_reward()` with preset-based weighting |
| `rl/executor.py` | `CompressionExecutor`: runs `gpu_compress` via subprocess or C API, measures metrics |
| `rl/trainer.py` | `QTableTrainer`: main training loop, file loading, episode management |
| `rl/export_qtable.py` | Export trained Q-table to binary format |
| `rl/generate_initial_qtable.py` | Generate initial Q-table with heuristic values |

---

## Synthetic Data Generator

### Overview

The generator (`syntheticGeneration/generator.py`) produces datasets with controllable compression characteristics via four independent parameters:

| Parameter | Controls | Range |
|-----------|----------|-------|
| `palette` | Bin weight distribution (which value ranges dominate) | 7 types |
| `perturbation` | Spatial locality / run lengths | 0.0 (long runs) to 1.0 (random) |
| `fill_mode` | Value pattern within each bin | 5 types |
| `bin_width` | Width of each bin in normalized value space | 0.1 to 64.0 |

### 32-Bin Palette System

Data values are drawn from 32 bins in the normalized [0, 1] range. Each palette assigns different probability weights to the bins:

| Palette | Weight Distribution |
|---------|-------------------|
| `uniform` | Equal weight per bin (1/32 each) |
| `normal` | Gaussian bell curve centered on middle bins |
| `gamma` | Skewed toward lower bins (`x * exp(-x)`, shape=2) |
| `exponential` | Extreme concentration in bin 0 (99.99% of data) |
| `bimodal` | Two peaks: 40% in lowest bins, 40% in highest bins, 20% middle |
| `grayscott` | Simulates reaction-diffusion: 70% background (low bins), 20% spots (high bins), 10% edges (middle) |
| `high_entropy` | Uniform weights but logarithmic bin spacing from 1e-6 to 1e0 (float32 only) |

For each palette, the weights are normalized to sum to 1.0. The number of elements assigned to each bin is proportional to its weight.

### Fill Modes

Each fill mode determines how values are distributed within a single bin's range `[b_lo, b_hi]`:

| Mode | Formula | Entropy Impact |
|------|---------|----------------|
| `constant` | All values = `b_lo` | Minimum (maximum repetition) |
| `linear` | `linspace(b_lo, b_hi, burst)` | Low (smooth gradient) |
| `quadratic` | Parabolic curve: `mid + a * (t - 0.5)^2 - range/4` | Low-medium |
| `sinusoidal` | `mid + range/2 * sin(2*pi*t)` | Medium (smooth oscillation) |
| `random` | `uniform(b_lo, b_hi)` | Maximum within bin |

### Perturbation

Controls how elements from different bins are interleaved:

```
burst_size = max(1, int(remaining_for_bin * (1.0 - perturbation)))
```

| Value | Burst Size | Effect |
|-------|-----------|--------|
| 0.0 | Full remaining | All elements from one bin before next (maximum locality) |
| 0.5 | Half remaining | Moderate interleaving |
| 1.0 | 1 element | Fully random ordering (no locality) |

### Data Types

| Type | Value Range | Scale |
|------|-------------|-------|
| `float32` | [0.0, 1.0] | Normalized, no clamping |
| `uint8` | [0, 255] | `bin_value * 255` |
| `int32` | [-1e6, 1e6] | `bin_value * 2e6 - 1e6` |

### Statistics

Four statistics are computed for each generated dataset:

| Statistic | Formula | Range |
|-----------|---------|-------|
| Shannon entropy | `-sum(p_i * log2(p_i))` over 256-byte histogram | 0.0 to 8.0 bits |
| MAD | `mean(\|data - mean(data)\|)` | >= 0.0 |
| 1st derivative | `mean(\|data[i+1] - data[i]\|)` | >= 0.0 |
| 2nd derivative | `mean(\|data[i+1] - 2*data[i] + data[i-1]\|)` | >= 0.0 |

### CLI Commands

**`generate`** -- Produce a single dataset:

```
python generator.py generate [options]

Required:
  -o, --output PATH          Output file (.bin or .h5)

Optional:
  -p, --palette TYPE         Bin weight distribution (default: uniform)
  --perturbation FLOAT       Spatial locality 0.0-1.0 (default: 0.5)
  -f, --fill-mode TYPE       Value pattern within bins (default: random)
  -w, --bin-width FLOAT      Bin width in value space (default: 1.0)
  -d, --dtype TYPE           float32, uint8, or int32 (default: float32)
  -s, --size SIZE            Dataset size e.g. 4MB, 128KB (default: 4MB)
  --seed INTEGER             Random seed
  -q, --quiet                Suppress output
```

**`batch`** -- Generate a full dataset collection:

```
python generator.py batch [options]

Optional:
  -o, --output-dir DIR       Output directory (default: datasets)
  -s, --size SIZE            Size per file (default: 4MB)
  -m, --mode MODE            training or comprehensive (default: training)
  -d, --dtypes TYPES         Comma-separated types (default: float32)
  -r, --repeats N            Repeats per combo with different seeds (default: 1)
  --format FORMAT            bin or h5 (default: bin)
  -q, --quiet                Suppress output
```

Configuration matrix per mode:

| Parameter | Training | Comprehensive |
|-----------|----------|---------------|
| Palettes | all 7 | all 7 |
| Bin widths | 0.1, 0.12, 0.15, 0.25, 0.5, 1.0, 16.0 | 0.1, 0.25, 0.5, 1.0, 4.0, 16.0, 64.0 |
| Perturbations | 0.0, 0.325, 0.95, 1.0 | 0.0, 0.05, 0.1, 0.2, 0.325, 0.5, 0.75, 0.9 |
| Fill modes | constant, linear, quadratic, random | all 5 |
| **Files per dtype** | **784** | **1960** |

**`stats`** -- Compute statistics without writing files:

```
python generator.py stats [options]

Optional:
  -o, --output PATH          CSV output path (default: compression_stats.csv)
  -t, --threads N            Worker threads (default: 1)
  -m, --mode MODE            comprehensive or training (default: comprehensive)
  -v, --verbose              Print progress
```

Output CSV columns: `data_type, data_size, palette, bin_width, perturbation, fill_mode, shannon_entropy, mad, first_derivative, second_derivative`

**`info`** -- Inspect a generated file:

```
python generator.py info <filename>
```

Prints entropy, MAD, derivatives, mean, std, min/max, and (for HDF5) generation parameters.

### Filename Convention

Generated filenames encode all parameters:

```
{dtype}_{palette}_w{width*100}_p{perturbation*1000}_{fill_short}[_s{seed}].bin

Example: float32_normal_w100_p325_rand.bin
         dtype=float32, palette=normal, bin_width=1.0, perturbation=0.325, fill=random
```

Fill mode short names: `constant` -> `const`, `linear` -> `linear`, `quadratic` -> `quad`, `sinusoidal` -> `sin`, `random` -> `rand`.

---

## Data Flow

End-to-end pipeline from dataset generation through training to GPU inference:

```
1. GENERATE TRAINING DATA
   python generator.py batch -o train_data/ --mode training -s 128KB --format bin
   -> 784 .bin files with diverse entropy/locality characteristics

2. TRAIN Q-TABLE
   python -m rl.trainer -d train_data/ -o rl/models/ -e 100 -p balanced --error-bound all
   -> For each file, each error bound:
      a. Load data, compute entropy + MAD + 2nd derivative
      b. Encode state (1024 states)
      c. Select action via epsilon-greedy (32 actions)
      d. Run gpu_compress with that algorithm + preprocessing
      e. Measure compression ratio, throughput, PSNR
      f. Compute weighted reward
      g. Update Q(s,a)
   -> Output: rl/models/qtable.json + qtable.bin

3. DEPLOY TO GPU
   gpucompress_init("rl/models/qtable.bin");
   // Q-table loaded into GPU constant memory

4. RUNTIME INFERENCE
   gpucompress_config_t cfg = gpucompress_default_config();
   cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
   gpucompress_compress(data, size, out, &out_size, &cfg, &stats);
   // Library internally:
   //   a. Computes entropy, MAD, 2nd derivative on GPU
   //   b. Encodes state
   //   c. Looks up argmax Q(s,:) in constant memory
   //   d. Applies selected preprocessing + algorithm
   //   e. Prepends 64-byte header with all metadata
```

---

## Build & Dependencies

### Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | >= 3.18 | Build system |
| CUDA Toolkit | Required | GPU kernels, runtime |
| nvCOMP | Required | Compression algorithms (installed to `/tmp/lib`, `/tmp/include`) |
| cuFile (GDS) | Required | GPUDirect Storage for CLI tools |
| HDF5 (C) | Optional | HDF5 filter plugin |
| Python 3 | For training | RL pipeline and data generator |
| numpy | Python dep | Array operations, entropy computation |
| click | Python dep | Generator CLI framework |
| h5py >= 3.0 | Python dep | HDF5 file I/O |

### Building

```bash
# Install nvCOMP (run once)
bash scripts/install_nvcomp.sh

# Build all targets
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Targets

| Target | Output | Description |
|--------|--------|-------------|
| `gpucompress` | `libgpucompress.so` | Shared library with C API |
| `gpu_compress` | `gpu_compress` | Compression CLI with GDS |
| `gpu_decompress` | `gpu_decompress` | Decompression CLI with GDS |
| `test_quantization` | `test_quantization` | Quantization test suite |
| `H5Zgpucompress` | `libH5Zgpucompress.so` | HDF5 filter plugin (only if HDF5 found) |

### Installation

```bash
cd build && sudo make install
# Installs:
#   lib/libgpucompress.so      -> /usr/local/lib/
#   include/gpucompress.h      -> /usr/local/include/
#   bin/gpu_compress           -> /usr/local/bin/
#   bin/gpu_decompress         -> /usr/local/bin/
#   lib/hdf5/plugin/libH5Zgpucompress.so  (if HDF5)
```

### Source Layout

```
GPUCompress/
├── include/
│   └── gpucompress.h                  # Public C API header
├── src/
│   ├── lib/
│   │   ├── gpucompress_api.cpp        # C API implementation
│   │   ├── entropy_kernel.cu          # GPU entropy calculation
│   │   └── qtable_gpu.cu             # Q-table GPU constant memory
│   ├── core/
│   │   ├── compression_factory.cpp    # Algorithm factory (8 nvCOMP managers)
│   │   ├── compression_factory.hpp
│   │   └── compression_header.h       # 64-byte header format
│   ├── preprocessing/
│   │   ├── byte_shuffle_kernels.cu    # Byte shuffle CUDA kernels
│   │   ├── byte_shuffle.cuh
│   │   ├── quantization_kernels.cu    # Linear quantization kernels
│   │   └── quantization.cuh
│   ├── cli/
│   │   ├── compress.cpp               # gpu_compress entry point
│   │   └── decompress.cpp             # gpu_decompress entry point
│   └── hdf5/
│       └── H5Zgpucompress.c           # HDF5 filter plugin
├── rl/
│   ├── config.py                      # State/action/reward configuration
│   ├── qtable.py                      # Q-table data structure
│   ├── policy.py                      # Epsilon-greedy / greedy policies
│   ├── reward.py                      # Reward computation with presets
│   ├── executor.py                    # Compression executor (subprocess / C API)
│   ├── trainer.py                     # Main training loop
│   ├── export_qtable.py               # Binary export utility
│   └── generate_initial_qtable.py     # Heuristic initialization
├── syntheticGeneration/
│   ├── generator.py                   # Python data generator
│   └── newScript.cc                   # C++ reference implementation
├── tests/
│   └── quantization/
│       └── test_quantization_suite.cu # Quantization correctness tests
├── CMakeLists.txt                     # Build configuration
└── OVERVIEW.md                        # This document
```
