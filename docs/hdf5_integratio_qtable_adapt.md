# GPUCompress HDF5 Filter + GPU-based RL Implementation

## Executive Summary

This document describes the complete implementation of:
1. **HDF5 Filter Plugin** (`libH5Zgpucompress.so`) - Integrates GPU compression into HDF5 I/O pipeline
2. **Shared Library** (`libgpucompress.so`) - C API for GPU-accelerated compression
3. **GPU-based RL System** - Automatic algorithm selection using Q-Table in GPU constant memory
4. **Offline Training System** - Python-based Q-Table training with configurable reward presets

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      HDF5 APPLICATION                               │    │
│  │                                                                     │    │
│  │   hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);                      │    │
│  │   H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);      │    │
│  │   H5Dcreate(..., dcpl);                                             │    │
│  │   H5Dwrite(dataset, data);  // Compression happens here            │    │
│  │                                                                     │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              HDF5 FILTER PIPELINE (libH5Zgpucompress.so)           │    │
│  │                                                                     │    │
│  │   H5Z_filter_gpucompress() receives host buffer from HDF5          │    │
│  │                                                                     │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  libgpucompress.so (C API)                          │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. cudaMemcpy Host → GPU                                      │  │    │
│  │  │ 2. GPU Entropy Calculation (histogram-based, ~1ms)            │  │    │
│  │  │ 3. Q-Table Lookup (GPU constant memory, ~1μs)                 │  │    │
│  │  │ 4. Apply Preprocessing (quantization + shuffle)               │  │    │
│  │  │ 5. nvcomp Compression (8 algorithms available)                │  │    │
│  │  │ 6. cudaMemcpy GPU → Host                                      │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: HDF5 Filter Integration

### How It Works

HDF5 filters intercept data during I/O operations. Our filter:
1. Receives **host memory buffer** from HDF5 (not GPU memory)
2. Transfers data to GPU, compresses, transfers back
3. Returns compressed buffer to HDF5 for storage

### Filter Registration

```c
// Filter ID 305 (testing range 256-511)
#define H5Z_FILTER_GPUCOMPRESS 305

// Filter class definition
static const H5Z_class2_t H5Z_GPUCOMPRESS[1] = {{
    H5Z_CLASS_T_VERS,           // version = 1
    H5Z_FILTER_GPUCOMPRESS,     // filter id
    1, 1,                       // encoder/decoder present
    "gpucompress",              // name
    H5Z_gpucompress_can_apply,  // validation callback
    H5Z_gpucompress_set_local,  // per-dataset setup
    H5Z_filter_gpucompress      // main filter function
}};
```

### Filter Parameters (cd_values)

```
cd_values[0]: algorithm (0=AUTO, 1=LZ4, 2=Snappy, ..., 8=Bitcomp)
cd_values[1]: preprocessing flags
cd_values[2]: shuffle element size (0, 2, 4, or 8)
cd_values[3]: error_bound low bits (uint32)
cd_values[4]: error_bound high bits (uint32)
```

### Usage Example

```c
#include <hdf5.h>
#include "gpucompress_hdf5.h"

// Register filter (automatic if using plugin)
H5Z_gpucompress_register();

// Create dataset with GPU compression
hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
H5Pset_chunk(dcpl, 2, chunk_dims);
H5Pset_gpucompress(dcpl,
    GPUCOMPRESS_ALGO_AUTO,  // RL-based selection
    0,                       // no extra preprocessing
    4,                       // 4-byte shuffle
    0.0);                    // lossless

hid_t dataset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, space,
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);
H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
```

### Files Implemented

| File | Purpose |
|------|---------|
| `src/hdf5/H5Zgpucompress.c` | Filter implementation |
| `src/hdf5/H5Zgpucompress.h` | Filter header |
| `include/gpucompress_hdf5.h` | Public HDF5 API |

---

## Part 2: Q-Table Reinforcement Learning

### Why Q-Table?

Different data characteristics benefit from different compression strategies:
- **Low entropy data** → Fast algorithms (LZ4, Snappy)
- **High entropy data** → Strong algorithms (ZSTD, ANS) with preprocessing
- **Floating-point data** → Quantization + shuffle improves compression

The Q-Table learns these mappings through offline training.

### Q-Table Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Q-TABLE STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATES (30 total):                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ state = entropy_bin × 3 + error_level                               │    │
│  │                                                                     │    │
│  │ Entropy Bins (10):  0=[0,1), 1=[1,2), ..., 9=[9,10] bits           │    │
│  │ Error Levels (3):   0=aggressive (≥0.01)                            │    │
│  │                     1=balanced (≥0.001)                             │    │
│  │                     2=precise (<0.001 or lossless)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ACTIONS (32 total):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ action = algorithm + quantization×8 + shuffle×16                   │    │
│  │                                                                     │    │
│  │ Algorithms (8):     LZ4, Snappy, Deflate, Gdeflate,                │    │
│  │                     ZSTD, ANS, Cascaded, Bitcomp                   │    │
│  │ Quantization (2):   None, Linear                                    │    │
│  │ Shuffle (2):        None, 4-byte                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Q-TABLE SIZE:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 30 states × 32 actions = 960 float values = 3.84 KB                │    │
│  │                                                                     │    │
│  │ Stored in GPU constant memory for ~1μs lookup time                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Q-Table Location: GPU Constant Memory

```cuda
// From src/lib/qtable_gpu.cu
__constant__ float d_qtable[960];  // GPU constant memory

// Benefits:
// - Broadcast to all threads simultaneously
// - Cached (64KB constant cache per SM)
// - Persistent across kernel launches
// - No allocation overhead
```

### Inference Flow (Real-time, in C++/CUDA)

```
Input Data
    │
    ▼
┌─────────────────────────────────┐
│ GPU Entropy Kernel (~1ms)       │  Histogram-based Shannon entropy
│ entropy = -Σ p_i × log2(p_i)    │  256 bins, shared memory atomics
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ State Encoding                  │  entropy_bin = floor(entropy)
│ state = bin×3 + error_level    │  30 possible states
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ Q-Table Lookup (~1μs)           │  argmax(Q[state, :])
│ best_action = argmax(Q-values)  │  From GPU constant memory
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ Action Decoding                 │  algorithm = action % 8
│ → algorithm, quant, shuffle     │  quant = (action/8) % 2
└───────────────┬─────────────────┘  shuffle = (action/16) % 2
                │
                ▼
Apply Selected Configuration
```

---

## Part 3: Offline Training Process

### Why Offline? Why Python?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING vs ONLINE INFERENCE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OFFLINE TRAINING (Python)              ONLINE INFERENCE (C++/CUDA)        │
│  ══════════════════════════             ═══════════════════════════        │
│                                                                             │
│  • Runs ONCE before deployment          • Runs EVERY compression           │
│  • Takes hours (acceptable)             • Takes microseconds (required)    │
│  • Easy experimentation                 • Zero overhead                    │
│  • Hyperparameter tuning                • Fixed logic                      │
│  • No speed requirement                 • Speed critical                   │
│                                                                             │
│  Output: qtable.bin (3.8 KB)  ────────► Loaded to GPU constant memory      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Algorithm: Q-Learning

```python
# Simplified training loop
for epoch in range(num_epochs):
    for data_file in training_files:
        # 1. Calculate entropy (GPU)
        entropy = calculate_entropy(data)

        # 2. Encode state
        state = entropy_bin * 3 + error_level

        # 3. Select action (ε-greedy)
        if random() < epsilon:
            action = random_action()      # Explore
        else:
            action = argmax(Q[state, :])  # Exploit

        # 4. Execute compression
        metrics = compress(data, action)  # ratio, throughput, PSNR

        # 5. Compute reward
        reward = w1*ratio + w2*throughput + w3*psnr

        # 6. Update Q-Table
        Q[state, action] += α * (reward - Q[state, action])

    # Decay exploration rate
    epsilon *= decay_rate
```

### Reward Presets

| Preset | Ratio | Throughput | PSNR | Use Case |
|--------|-------|------------|------|----------|
| `balanced` | 0.4 | 0.3 | 0.3 | General purpose |
| `max_ratio` | 0.8 | 0.1 | 0.1 | Minimize storage |
| `max_speed` | 0.1 | 0.8 | 0.1 | Real-time streaming |
| `max_quality` | 0.1 | 0.1 | 0.8 | Scientific accuracy |
| `storage` | 0.6 | 0.2 | 0.2 | Archival storage |
| `streaming` | 0.3 | 0.5 | 0.2 | Network transfer |

### Training Output Files

```
rl/models/
├── qtable.json     # Human-readable (for debugging)
│                   # {
│                   #   "version": 1,
│                   #   "n_states": 30,
│                   #   "n_actions": 32,
│                   #   "q_values": [[...], ...]
│                   # }
│
├── qtable.bin      # Binary for GPU loading (3.8 KB)
│                   # Header: magic, version, n_states, n_actions
│                   # Data: 960 × float32
│
└── qtable_epoch*.json  # Checkpoints during training
```

### How to Train

```bash
# Step 1: Build the project
mkdir -p build && cd build && cmake .. && make

# Step 2: Generate training data
python syntheticGeneration/generator.py batch \
    -c syntheticGeneration/datasets/config.yaml \
    -o datasets/training/

# Step 3: Run training
./scripts/run_rl_training.sh \
    --data-dir datasets/training \
    --epochs 100 \
    --preset balanced

# Step 4: Use trained model
# The trained qtable.bin is loaded at gpucompress_init()
```

---

## Part 4: Implementation Files

### Directory Structure

```
GPUCompress/
├── include/                          # Public headers
│   ├── gpucompress.h                 # C API (400+ lines)
│   └── gpucompress_hdf5.h            # HDF5 filter interface
│
├── src/
│   ├── lib/                          # Shared library source
│   │   ├── gpucompress_api.cpp       # C API implementation
│   │   ├── gpucompress_internal.hpp  # Internal C++ utilities
│   │   ├── entropy_kernel.cu         # GPU entropy calculation
│   │   └── qtable_gpu.cu             # GPU Q-Table inference
│   │
│   ├── hdf5/                         # HDF5 filter plugin
│   │   ├── H5Zgpucompress.c          # Filter implementation
│   │   └── H5Zgpucompress.h          # Filter header
│   │
│   └── [existing files...]           # CompressionFactory, quantization, etc.
│
├── rl/                               # RL training system (Python)
│   ├── __init__.py
│   ├── config.py                     # Hyperparameters
│   ├── qtable.py                     # Q-Table class
│   ├── policy.py                     # Epsilon-greedy policy
│   ├── reward.py                     # Reward computation
│   ├── executor.py                   # Compression executor
│   ├── trainer.py                    # Training loop
│   ├── export_qtable.py              # JSON → binary converter
│   └── models/                       # Saved Q-Tables
│       ├── qtable.json
│       └── qtable.bin
│
├── scripts/
│   └── run_rl_training.sh            # Training launcher
│
└── CMakeLists.txt                    # Updated build system
```

### Build Outputs

```
build/
├── libgpucompress.so                 # Core shared library
├── libH5Zgpucompress.so              # HDF5 filter plugin
├── gpu_compress                       # Compression executable
└── gpu_decompress                     # Decompression executable
```

---

## Part 5: C API Reference

### Initialization

```c
// Initialize library with Q-Table
gpucompress_error_t gpucompress_init(const char* qtable_path);

// Cleanup
void gpucompress_cleanup(void);
```

### Compression

```c
// Configuration
gpucompress_config_t config = gpucompress_default_config();
config.algorithm = GPUCOMPRESS_ALGO_AUTO;  // Use RL
config.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
config.error_bound = 0.001;  // For quantization

// Compress
gpucompress_error_t err = gpucompress_compress(
    input, input_size,
    output, &output_size,
    &config,
    &stats  // Optional statistics
);

// Decompress (auto-detects settings from header)
err = gpucompress_decompress(
    compressed, compressed_size,
    decompressed, &decompressed_size
);
```

### Entropy Calculation

```c
// Calculate entropy (GPU-accelerated)
double entropy;
gpucompress_calculate_entropy(data, size, &entropy);
// Returns 0.0 to 8.0 bits
```

### Algorithm Selection

```c
// Get recommendation based on data characteristics
gpucompress_algorithm_t algo;
unsigned int preprocessing;
gpucompress_recommend_config(entropy, error_bound, &algo, &preprocessing);
```

---

## Part 6: Usage Examples

### Example 1: Basic HDF5 Usage

```c
#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5.h"

int main() {
    // Initialize
    gpucompress_init("rl/models/qtable.bin");

    // Create HDF5 file
    hid_t file = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Setup compression
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[2] = {1000, 1000};
    H5Pset_chunk(dcpl, 2, chunk);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

    // Create and write dataset
    hsize_t dims[2] = {10000, 1000};
    hid_t space = H5Screate_simple(2, dims, NULL);
    hid_t dset = H5Dcreate(file, "data", H5T_NATIVE_FLOAT, space,
                           H5P_DEFAULT, dcpl, H5P_DEFAULT);

    float* data = malloc(10000 * 1000 * sizeof(float));
    // ... fill data ...

    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // Cleanup
    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(file);
    gpucompress_cleanup();

    return 0;
}
```

### Example 2: Direct C API Usage

```c
#include "gpucompress.h"

int main() {
    gpucompress_init("rl/models/qtable.bin");

    // Read input data
    size_t size = 1000000;
    float* data = malloc(size * sizeof(float));
    // ... fill data ...

    // Compress
    size_t max_out = gpucompress_max_compressed_size(size * sizeof(float));
    void* compressed = malloc(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_stats_t stats;
    gpucompress_compress(data, size * sizeof(float),
                         compressed, &compressed_size,
                         &config, &stats);

    printf("Ratio: %.2fx\n", stats.compression_ratio);
    printf("Algorithm: %s\n", gpucompress_algorithm_name(stats.algorithm_used));
    printf("Entropy: %.2f bits\n", stats.entropy_bits);

    gpucompress_cleanup();
    return 0;
}
```

---

## Part 7: Build Instructions

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install cmake libhdf5-dev

# CUDA and nvcomp must be installed
# nvcomp headers in /tmp/include, libs in /tmp/lib
```

### Build Commands

```bash
cd GPUCompress
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Outputs:
# - libgpucompress.so
# - libH5Zgpucompress.so (if HDF5 found)
# - gpu_compress
# - gpu_decompress
```

### Using the HDF5 Plugin

```bash
# Set plugin path
export HDF5_PLUGIN_PATH=/path/to/GPUCompress/build

# Now any HDF5 application can use the filter
h5repack -f "gpucompress=0,0,4,0,0" input.h5 output.h5
```

---

## Summary

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| `libgpucompress.so` | C API + CUDA | Core compression library |
| `libH5Zgpucompress.so` | HDF5 filter | Transparent HDF5 integration |
| GPU Entropy Kernel | CUDA | Real-time entropy calculation |
| GPU Q-Table | CUDA constant memory | Fast algorithm selection |
| Python RL Training | Offline | Learn optimal configurations |
| Q-Table Binary | 3.8 KB file | Portable trained model |

**Key Design Decisions:**
1. **Offline training, online inference** - Training in Python for flexibility, inference in CUDA for speed
2. **Q-Table in GPU constant memory** - ~1μs lookup time, no allocation overhead
3. **Host memory interface for HDF5** - Matches HDF5 filter API requirements
4. **Self-describing compressed format** - 64-byte header enables auto-decompression
