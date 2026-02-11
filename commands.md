# GPUCompress — RL Training Commands

## Prerequisites

```bash
# 1. Install Python dependencies
pip install numpy click

# 2. Build the GPU compression binaries (requires CUDA + nvCOMP)
cd /path/to/GPUCompress
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..

# Verify binaries exist
ls build/gpu_compress build/gpu_decompress
```

---

## Step 1: Clean Old Q-Table Data

```bash
# Remove all previous Q-table models (JSON, binary, reports)
python3 -m rl.trainer --clean --output-dir rl/models
```

---

## Step 2: Generate Synthetic Training Datasets

### Single file

```bash
python3 syntheticGeneration/generator.py generate \
    -p normal --perturbation 0.5 -f random -w 1.0 \
    -s 128KB --seed 1 -o syntheticGeneration/training_data/sample.bin
```

### Small batch (100 files for quick experiments)

```bash
mkdir -p syntheticGeneration/training_data

palettes=(uniform normal gamma exponential bimodal)
perturbations=(0.0 0.25 0.5 0.75 1.0)
fills=(constant linear quadratic random)
widths=(0.25 0.5 1.0 4.0 16.0)

count=0
for p in "${palettes[@]}"; do
  wi=0
  for pert in "${perturbations[@]}"; do
    w=${widths[$wi]}
    wi=$(( (wi + 1) % 5 ))
    for fm in "${fills[@]}"; do
      count=$((count + 1))
      python3 syntheticGeneration/generator.py generate \
        -p "$p" --perturbation "$pert" -f "$fm" -w "$w" \
        -s 128KB --seed "$count" \
        -o "syntheticGeneration/training_data/float32_${p}_w${w}_p${pert}_${fm}.bin" -q
    done
  done
done
echo "Generated $count files"
```

### Full training set (784 files — all combos)

```bash
python3 syntheticGeneration/generator.py batch \
    -o syntheticGeneration/training_data \
    --mode training -s 128KB --format bin -q
```

### Full comprehensive set (5,880 files — all dtypes)

```bash
python3 syntheticGeneration/generator.py batch \
    -o syntheticGeneration/training_data \
    --mode comprehensive -d float32,uint8,int32 -s 128KB --format bin -q
```

---

## Step 3: (Optional) Generate Heuristic Baseline Q-Table

```bash
# Creates an initial Q-table with hand-tuned heuristic values
# Useful as a starting point before training
python3 -m rl.generate_initial_qtable
```

---

## Step 4: Run RL Training

### Quick test (few files, few epochs)

```bash
python3 -m rl.trainer \
    --data-dir syntheticGeneration/training_data \
    --output-dir rl/models \
    --epochs 5 \
    --preset balanced \
    --error-bound 0.001 \
    --gpu-compress ./build/gpu_compress
```

### Standard training (all error levels)

```bash
python3 -m rl.trainer \
    --data-dir syntheticGeneration/training_data \
    --output-dir rl/models \
    --epochs 100 \
    --preset balanced \
    --error-bound all \
    --gpu-compress ./build/gpu_compress
```

### Reward presets

```bash
# Maximize compression ratio (archival/storage)
python3 -m rl.trainer -d syntheticGeneration/training_data -e 100 -p max_ratio --error-bound all

# Maximize throughput (streaming/real-time)
python3 -m rl.trainer -d syntheticGeneration/training_data -e 100 -p max_speed --error-bound all

# Maximize quality (scientific data)
python3 -m rl.trainer -d syntheticGeneration/training_data -e 100 -p max_quality --error-bound all

# Storage-optimized
python3 -m rl.trainer -d syntheticGeneration/training_data -e 100 -p storage --error-bound all

# Streaming-optimized
python3 -m rl.trainer -d syntheticGeneration/training_data -e 100 -p streaming --error-bound all
```

### Resume from checkpoint

```bash
python3 -m rl.trainer \
    --data-dir syntheticGeneration/training_data \
    --output-dir rl/models \
    --epochs 50 \
    --preset balanced \
    --error-bound all \
    --resume rl/models/qtable.json
```

### Quiet mode

```bash
python3 -m rl.trainer \
    --data-dir syntheticGeneration/training_data \
    --epochs 100 --preset balanced --error-bound all -q
```

---

## Step 5: Export and Inspect Trained Model

### Export JSON to binary (for GPU loading)

```bash
python3 -m rl.export_qtable rl/models/qtable.json rl/models/qtable.bin --verify
```

### Dump human-readable report

```bash
python3 -m rl.export_qtable rl/models/qtable.json --dump rl/models/qtable_report.txt
```

### Export + verify + dump all at once

```bash
python3 -m rl.export_qtable rl/models/qtable.json rl/models/qtable.bin --verify --dump rl/models/qtable_report.txt
```

---

## Step 6: Generate Dataset Statistics (no files saved)

```bash
# Compute entropy/MAD/derivative stats for all parameter combos, write CSV
python3 syntheticGeneration/generator.py stats -o compression_stats.csv --threads 8 -v

# Training-mode subset only
python3 syntheticGeneration/generator.py stats -o stats.csv --threads 4 --mode training -v
```

---

## Step 7: Inspect a Dataset

```bash
# Binary file
python3 syntheticGeneration/generator.py info syntheticGeneration/training_data/some_file.bin

# HDF5 file
python3 syntheticGeneration/generator.py info datasets/some_file.h5
```

---

## Quick Reference

### Generator CLI flags

| Flag | Default | Description |
|---|---|---|
| `-p, --palette` | `uniform` | `uniform`, `normal`, `gamma`, `exponential`, `bimodal`, `grayscott`, `high_entropy` |
| `--perturbation` | `0.5` | `0.0` (long runs) to `1.0` (fully random) |
| `-f, --fill-mode` | `random` | `constant`, `linear`, `quadratic`, `sinusoidal`, `random` |
| `-w, --bin-width` | `1.0` | Bin width in value space |
| `-d, --dtype` | `float32` | `float32`, `uint8`, `int32` |
| `-s, --size` | `4MB` | Size per dataset (e.g., `128KB`, `1MB`, `1GB`) |
| `--seed` | None | Random seed |
| `-o, --output` | *required* | Output file path (`.bin` or `.h5`) |

### Trainer CLI flags

| Flag | Default | Description |
|---|---|---|
| `-d, --data-dir` | *required* | Directory with `.bin` / `.raw` / `.dat` files |
| `-o, --output-dir` | `rl/models` | Save dir for `qtable.json` + `qtable.bin` |
| `-e, --epochs` | `100` | Training epochs |
| `-p, --preset` | `balanced` | `balanced`, `max_ratio`, `max_speed`, `max_quality`, `storage`, `streaming` |
| `--error-bound` | `0.001` | Float or `all` (trains all 4 levels: 0.1, 0.01, 0.001, 0.0) |
| `--resume` | None | Path to previous `qtable.json` |
| `--gpu-compress` | `./build/gpu_compress` | Path to compression binary |
| `--use-c-api` | off | Use `libgpucompress.so` for faster entropy |
| `-q, --quiet` | off | Reduce verbosity |
| `--clean` | — | Delete all qtable files from output dir |
