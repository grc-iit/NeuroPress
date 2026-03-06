# GPUCompress — Commands Reference

## Prerequisites

```bash
# 1. Install Python dependencies
pip install numpy click torch

# 2. Build the GPU compression binaries (requires CUDA + nvCOMP)
cd /path/to/GPUCompress
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..

# Verify binaries exist
ls build/gpu_compress build/gpu_decompress
```

---

## Generate Synthetic Training Datasets

### Single file

```bash
python3 syntheticGeneration/generator.py generate \
    -p normal --perturbation 0.5 -f random -w 1.0 \
    -s 128KB --seed 1 -o syntheticGeneration/training_data/sample.bin
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

## Generate Dataset Statistics (no files saved)

```bash
# Compute entropy/MAD/derivative stats for all parameter combos, write CSV
python3 syntheticGeneration/generator.py stats -o compression_stats.csv --threads 8 -v

# Training-mode subset only
python3 syntheticGeneration/generator.py stats -o stats.csv --threads 4 --mode training -v
```

---

## Inspect a Dataset

```bash
# Binary file
python3 syntheticGeneration/generator.py info syntheticGeneration/training_data/some_file.bin

# HDF5 file
python3 syntheticGeneration/generator.py info datasets/some_file.h5
```

---

## Neural Network Training

### Retrain with experience data

```bash
python3 neural_net/training/retrain.py \
    --experience eval/experience.csv \
    --output model.nnwt
```

---

## Evaluation Pipeline

```bash
bash eval/run_eval_pipeline.sh --timesteps 15 --weights model.nnwt
```

---

## CPU Flame Graph (perf + FlameGraph)

```bash
# 1. Record CPU stacks while running benchmark
sudo perf record -g -F 999 -o /tmp/bench_adapt_perf.data \
  ./build/bench_adaptation neural_net/weights/model.nnwt gaussian

# 2. Collapse stacks into folded format
sudo perf script -i /tmp/bench_adapt_perf.data | \
  /home/cc/FlameGraph/stackcollapse-perf.pl > /tmp/bench_adapt.folded

# 3. Generate interactive SVG flame graph
/home/cc/FlameGraph/flamegraph.pl --title "bench_adaptation" \
  /tmp/bench_adapt.folded > /tmp/bench_adapt_flamegraph.svg

# Open in browser
xdg-open /tmp/bench_adapt_flamegraph.svg
```

---

## VPIC Benchmark (Real Harris Sheet Simulation)

```bash
# Build the deck
cd /u/imuradli/vpic-kokkos/build-compress
./bin/vpic /u/imuradli/GPUCompress/tests/benchmarks/vpic_benchmark_deck.cxx

# Run (single rank)
export LD_LIBRARY_PATH=/u/imuradli/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
export GPUCOMPRESS_WEIGHTS=/u/imuradli/GPUCompress/neural_net/weights/model.nnwt
./vpic_benchmark_deck.Linux
```

> For multi-rank under SLURM: `srun -n <ranks> ./vpic_benchmark_deck.Linux`

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
