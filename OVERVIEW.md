# GPUCompress: Project Overview

GPUCompress is a GPU-accelerated compression library that uses a neural network to automatically select the best compression algorithm based on data characteristics. The project has three major components:

1. **GPU Compression Library** (`src/`, `include/`) -- A C/CUDA library wrapping 8 nvCOMP algorithms with optional byte-shuffle and quantization preprocessing, exposed through both a C API (`libgpucompress.so`) and CLI tools (`gpu_compress`, `gpu_decompress`).

2. **Neural Network Training System** (`neural_net/`) -- A training pipeline that produces a neural network mapping data features (entropy, MAD, smoothness) to optimal compression configurations. The trained weights (`.nnwt`) are loaded to GPU at inference time.

3. **Synthetic Data Generator** (`syntheticGeneration/`) -- A palette-based generator that produces datasets with precisely controlled entropy, locality, and value distributions for training the neural network.

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
 neural_net/training/                              |
      |                                            |--- calculate entropy,
      |--- load file                               |    MAD, 2nd derivative
      |--- calculate entropy, MAD, deriv           |
      |--- train NN on all 32 configs              |--- NN inference (GPU)
      |--- predict ratio, comp_time, PSNR         |--- rank all 32 configs
      |--- minimize prediction error               |--- select best action
      v                                            |--- apply preprocessing
 model.nnwt                                        |    (quantize, shuffle)
      |                                            |--- compress (nvCOMP)
      +-------- deployed to -------->>             v
                                            compressed buffer +
                                            64-byte header
```

---

## GPU Compression Library

### C API (`include/gpucompress.h`)

The public API follows an init/use/cleanup lifecycle:

```c
#include "gpucompress.h"

// 1. Initialize (optionally load NN weights for AUTO mode)
gpucompress_init("/path/to/model.nnwt");

// 2. Configure
gpucompress_config_t cfg = gpucompress_default_config();
cfg.algorithm     = GPUCOMPRESS_ALGO_AUTO;  // NN-based selection
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
| `GPUCOMPRESS_ALGO_AUTO` (0) | NN auto-select | Uses neural network to pick best option |
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

### Compression Header (`src/compression/compression_header.h`)

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
| -6 | `ERROR_RESERVED_6` | Reserved |
| -7 | `ERROR_INVALID_HEADER` | Corrupt or missing header |
| -8 | `ERROR_NOT_INITIALIZED` | `gpucompress_init()` not called |
| -9 | `ERROR_BUFFER_TOO_SMALL` | Output buffer insufficient |

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

2. TRAIN NEURAL NETWORK
   python -m neural_net.training.retrain --experience data.csv --output model.nnwt
   -> Trains NN to predict compression ratio, time, and PSNR for all 32 configs
   -> Output: model.nnwt

3. DEPLOY TO GPU
   gpucompress_init("model.nnwt");
   // NN weights loaded to GPU memory

4. RUNTIME INFERENCE
   gpucompress_config_t cfg = gpucompress_default_config();
   cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
   gpucompress_compress(data, size, out, &out_size, &cfg, &stats);
   // Library internally:
   //   a. Computes entropy, MAD, 2nd derivative on GPU
   //   b. Runs NN inference over all 32 configs
   //   c. Selects best config by predicted metrics
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
│   │   ├── nn_gpu.cu                 # NN inference on GPU
│   │   ├── nn_reinforce.cpp           # Online reinforcement
│   │   ├── stats_kernel.cu            # Stats pipeline kernels
│   │   └── experience_buffer.cpp      # Active learning buffer
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
├── neural_net/                            # NN training and inference
│   ├── core/                              # Model definition and export
│   └── training/                          # Training and retraining scripts
├── syntheticGeneration/
│   └── generator.py                   # Python data generator
├── tests/
│   └── quantization/
│       └── test_quantization_suite.cu # Quantization correctness tests
├── CMakeLists.txt                     # Build configuration
└── OVERVIEW.md                        # This document
```
