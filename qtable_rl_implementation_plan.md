# Q-Table Reinforcement Learning for GPUCompress

## Overview

Build a Q-Table based RL model that learns optimal compression configurations based on data entropy. The model will select preprocessing options (quantization, shuffling) and compression algorithms through trial-and-error learning.

---

## 1. Current Repository Structure (Key Files)

### Core Compression Pipeline
- `src/GPU_Compress.cpp` - Main compression with GDS (766 lines)
- `src/GPU_Decompress.cpp` - Decompression (565 lines)
- `src/CompressionFactory.hpp/.cpp` - Factory for 8 algorithms
- `src/compression_header.h` - 64-byte metadata structure

### Preprocessing
- `src/quantization.cuh` / `quantization_kernels.cu` - Linear quantization with error bounds
- `src/byte_shuffle.cuh` / `byte_shuffle_kernels.cu` - Byte-level shuffling

### Existing Utilities (to reuse)
- `syntheticGeneration/attributes/entropy.py` - Shannon entropy calculation
- `syntheticGeneration/utils/metrics.py` - Data metrics measurement
- `scripts/run_simple_tests.sh` - Benchmarking patterns

### Available Algorithms
LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp (8 total)

---

## 2. RL System Design

### 2.1 State Space

**Primary State: Entropy Bins (10 bins)**
```
Bin 0: entropy [0.0, 1.0)  - Very low entropy (highly compressible)
Bin 1: entropy [1.0, 2.0)
...
Bin 9: entropy [9.0, 10.0] - High entropy (random/incompressible)
```

**Secondary State: Error Bound Level (3 levels)**
- Level 0: error_bound = 0.01 (aggressive)
- Level 1: error_bound = 0.001 (balanced)
- Level 2: error_bound = 0.0001 (precise)

**Total State Space: 10 entropy bins × 3 error bounds = 30 states**

### 2.2 Action Space

```
Action = (Quantization, Shuffle, Algorithm)

Quantization: [None, Linear]     = 2 options
Shuffle:      [None, 4-byte]     = 2 options
Algorithm:    [8 algorithms]     = 8 options

Total Actions = 2 × 2 × 8 = 32 discrete actions
```

**Action Mapping (examples):**
```
Action 0:  {quant: None,   shuffle: 0, algo: 'lz4'}
Action 8:  {quant: None,   shuffle: 4, algo: 'lz4'}
Action 16: {quant: linear, shuffle: 0, algo: 'lz4'}
Action 24: {quant: linear, shuffle: 4, algo: 'lz4'}
...
Action 31: {quant: linear, shuffle: 4, algo: 'bitcomp'}
```

### 2.3 Q-Table Structure

```
Q-Table Shape: (30 states, 32 actions) = 960 entries

State index = entropy_bin * 3 + error_bound_level
```

### 2.4 Reward Function (Configurable Presets)

```python
REWARD_PRESETS = {
    'balanced':    {'ratio': 0.4, 'throughput': 0.3, 'psnr': 0.3},
    'max_ratio':   {'ratio': 0.8, 'throughput': 0.1, 'psnr': 0.1},
    'max_speed':   {'ratio': 0.1, 'throughput': 0.8, 'psnr': 0.1},
    'max_quality': {'ratio': 0.1, 'throughput': 0.1, 'psnr': 0.8},
    'storage':     {'ratio': 0.6, 'throughput': 0.2, 'psnr': 0.2},
    'streaming':   {'ratio': 0.3, 'throughput': 0.5, 'psnr': 0.2},
}

def compute_reward(ratio, throughput_mbps, psnr_db, preset='balanced'):
    weights = REWARD_PRESETS[preset]
    ratio_norm = min(ratio / 10.0, 1.0)
    throughput_norm = min(throughput_mbps / 5000.0, 1.0)
    psnr_norm = 1.0 if psnr_db == inf else min(psnr_db / 100.0, 1.0)

    return (weights['ratio'] * ratio_norm +
            weights['throughput'] * throughput_norm +
            weights['psnr'] * psnr_norm)
```

### 2.5 Metrics Tracked

| Metric | Description | Source |
|--------|-------------|--------|
| Compression Ratio | original_size / compressed_size | Computed from file sizes |
| Throughput (MB/s) | data_size / compression_time | Timed execution |
| PSNR (dB) | 20*log10(max_value / RMSE) | Only when quantization applied |

---

## 3. Proposed File Structure

```
GPUCompress/
├── rl/                                # NEW: RL system
│   ├── __init__.py
│   ├── qtable.py                     # Q-Table class
│   ├── policy.py                     # Epsilon-greedy policy
│   ├── reward.py                     # Reward presets and computation
│   ├── state.py                      # State discretization (entropy + error bound)
│   ├── action.py                     # Action space mapping
│   ├── executor.py                   # Interface to compression (Python/C++)
│   ├── entropy_analyzer.py           # Wrapper for entropy calculation
│   ├── trainer.py                    # Training loop
│   ├── inference.py                  # Production recommendation API
│   ├── config.py                     # Hyperparameters and constants
│   ├── cli.py                        # Command-line interface
│   ├── visualizer.py                 # NEW: Dashboard visualization
│   ├── models/                       # Saved Q-Tables
│   │   └── qtable.json
│   └── tests/
│       ├── test_qtable.py
│       ├── test_policy.py
│       └── test_reward.py
├── scripts/
│   ├── run_rl_training.sh           # NEW: Training launcher
│   └── run_rl_inference.sh          # NEW: Inference launcher
└── visuals/
    └── rl_dashboard.py              # NEW: Q-Table heatmaps, training curves
```

---

## 4. Component Implementation Plan

### Phase 1: Core RL Components

**4.1 Entropy Analyzer** (`rl/entropy_analyzer.py`)
- Wrap existing `syntheticGeneration/attributes/entropy.py`
- Add `discretize_entropy(value, n_bins=10) -> int` function
- Handle edge cases (empty data, single-value data)

**4.2 Q-Table** (`rl/qtable.py`)
- `QTable` class with `q_values` and `visit_counts` arrays
- Methods: `get_q()`, `update_q()`, `get_best_action()`, `save()`, `load()`
- Q-Learning update: `Q(s,a) += α * (r - Q(s,a))` (γ=0 since single-step)

**4.3 Policy** (`rl/policy.py`)
- `EpsilonGreedyPolicy` with decay
- Start ε=1.0, decay to ε=0.01 over training

**4.4 Reward** (`rl/reward.py`)
- `compute_reward()` with configurable presets
- Normalization functions for each metric

**4.5 Action/State** (`rl/action.py`, `rl/state.py`)
- `ACTION_SPACE` dictionary mapping action_id to config
- `encode_state(entropy_bin, error_bound_level) -> state_id`
- `decode_action(action_id) -> config dict`

### Phase 2: Execution Interface

**4.6 Executor** (`rl/executor.py`)
- **Option A (Python)**: Call `gpu_compress` binary via subprocess
- **Option B (C++)**: Direct integration (TBD based on user preference)
- Parse metrics from output or JSON
- Return dict: `{ratio, throughput, psnr, compressed_size}`

### Phase 3: Training System

**4.7 Trainer** (`rl/trainer.py`)
- `run_training_episode(data_file, q_table, policy)`
- `train_on_dataset(data_dir, n_epochs, save_interval)`
- Support for multiple error bound levels
- Logging and checkpointing

### Phase 4: Inference API

**4.8 Inference** (`rl/inference.py`)
- `CompressionAdvisor` class
- `recommend(input_file) -> config`
- `compress_optimal(input_file, output_file) -> metrics`
- Confidence score based on visit counts

### Phase 5: Visualization Dashboard

**4.9 Visualizer** (`rl/visualizer.py`)
- Q-Table heatmap (entropy bins × actions)
- Training progress curves (reward over episodes)
- Entropy distribution of training data
- Best action per state summary table
- Action frequency pie charts

---

## 5. Training Workflow

```
1. Generate/collect training data with various entropy levels
   - Use existing synthetic generator or real datasets
   - Cover entropy range 1-8 bits

2. For each epoch:
   a. Shuffle training files
   b. For each file:
      - Calculate entropy → discretize to state
      - Select action (ε-greedy)
      - Execute compression with selected config
      - Measure metrics (ratio, throughput, PSNR)
      - Compute reward (using selected preset)
      - Update Q-Table
   c. Decay epsilon
   d. Save checkpoint

3. Final Q-Table saved to rl/models/qtable.json
```

---

## 6. Inference Workflow

```
1. Load trained Q-Table from file
2. For input data:
   a. Read data and calculate entropy
   b. Discretize entropy to bin
   c. Determine error bound level (user-specified or default)
   d. Look up state in Q-Table
   e. Select action with highest Q-value
   f. Return recommended config:
      - Algorithm
      - Quantization (on/off)
      - Shuffle (on/off)
      - Confidence score
```

---

## 7. Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_entropy_bins | 10 | Granularity vs Q-Table size |
| n_error_levels | 3 | 0.01, 0.001, 0.0001 |
| alpha (learning rate) | 0.1 | Standard Q-learning |
| gamma (discount) | 0.0 | Single-step decision |
| epsilon_start | 1.0 | Full exploration initially |
| epsilon_end | 0.01 | Small exploration retained |
| epsilon_decay | 0.995 | ~500 episodes to converge |
| n_epochs | 100-500 | Depends on dataset size |

---

## 8. Visualization Dashboard Features

1. **Q-Table Heatmap**
   - X-axis: Actions (32)
   - Y-axis: States (30 = 10 entropy × 3 error bounds)
   - Color: Q-value intensity

2. **Training Progress**
   - Episode vs. Average Reward
   - Episode vs. Epsilon value
   - Cumulative reward curve

3. **State Distribution**
   - Histogram of training data by entropy bin
   - Coverage analysis (which states seen)

4. **Action Analysis**
   - Pie chart: Action selection frequency
   - Best action per entropy bin table
   - Algorithm preference by entropy level

5. **Performance Comparison**
   - RL-selected vs. default (LZ4) configuration
   - Improvement percentage by data type

---

## 9. Testing Strategy

### Unit Tests
- Q-Table initialization, update, save/load
- Reward computation for edge cases
- State/action encoding/decoding

### Integration Tests
- End-to-end training on small dataset
- Inference consistency (same input → same output)

### Performance Validation
- Compare RL-selected vs. default configuration
- Measure improvement in compression ratio
- Verify throughput is acceptable

---

## 10. Open Questions (To Be Decided)

1. **Architecture**: Python module vs. C++ integration vs. Hybrid
   - **TBD** - User will decide later
   - Options:
     - Python module: Calls gpu_compress binary via subprocess
     - C++ integration: Embedded in GPU_Compress.cpp with CUDA entropy kernel
     - Hybrid: Python training, C++ header-only Q-Table for production

2. **Shuffle element sizes**: Currently fixed at 4-byte
   - Could expand to 2, 4, 8 as additional action dimension

3. **Online learning**: Update Q-Table during production?
   - Dashboard feature requested, consider later

---

## 11. Implementation Order

1. Core RL components (qtable, policy, reward, state, action)
2. Entropy analyzer wrapper
3. Executor interface (Python subprocess initially)
4. Training loop
5. Inference API
6. CLI interface
7. Visualization dashboard
8. Unit tests
9. Integration tests
10. Documentation

---

## 12. Verification Plan

1. **Training verification**:
   - Generate synthetic data with known entropy levels (using existing tools)
   - Train for 100 epochs
   - Verify Q-values converge (not all zero)
   - Check that high-entropy data → different optimal action than low-entropy

2. **Inference verification**:
   - Run inference on test files
   - Compare recommended config vs. actual best (exhaustive search on small set)
   - Measure recommendation accuracy

3. **End-to-end verification**:
   - Compress real data using RL recommendations
   - Compare metrics to default LZ4
   - Document improvement percentages
