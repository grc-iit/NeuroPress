# GPUCompress — Quickstart Guide

Install dependencies, build, and run tests on an NCSA Delta GPU node.

---

## 1. Allocate a GPU Node

Start a tmux session first so you can reconnect if disconnected:

```bash
tmux new -s gpu
```

Request an interactive GPU node:

```bash
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
     --nodes=1 --gpus=1 --tasks=1 --cpus-per-task=16 \
     --mem=64g --time=00:30:00 --pty bash
```

> If disconnected, reconnect with: `tmux attach -t gpu`

---

## 2. Install Dependencies

Dependencies install to node-local `/tmp` and must be reinstalled on each new node.

```bash
cd /u/imuradli/GPUCompress
bash scripts/install_dependencies.sh
```

This installs:
- **nvcomp 5.1.0** → `/tmp/include`, `/tmp/lib`
- **HDF5 2.0.0** → `/tmp/hdf5-install`
- Builds the entire project → `build/`

Override CUDA architecture if needed (default is sm_80 for A100):

```bash
CUDA_ARCH=90 bash scripts/install_dependencies.sh   # H100
```

### Manual clean rebuild

If you need to rebuild from scratch (e.g., stale CMake cache):

```bash
cd /u/imuradli/GPUCompress
rm -rf build
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
make -j$(nproc)
```

> **Important:** Always pass `-DCMAKE_CUDA_ARCHITECTURES=80` (or higher) explicitly.
> A stale cache with `sm_52` will cause `atomicAdd(double*, double)` compile errors.

---

## 3. Set Up Environment

```bash
source scripts/setup_env.sh
```

Or manually:

```bash
export LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
```

---

## 4. Verify the Build

```bash
ls build/gpu_compress build/gpu_decompress
ls build/libgpucompress.so build/libH5Zgpucompress.so build/libH5VLgpucompress.so
```

---

## 5. Run Tests

### Quick smoke test

```bash
./build/test_quantization_roundtrip
./build/test_nn_pipeline
```

### Full test suite (unit + HDF5 + VOL + benchmarks)

```bash
bash scripts/run_tests.sh
```

This runs all tests with timeouts and prints a PASS/FAIL summary.

### Individual test categories

**Unit / Regression:**

```bash
./build/test_quantization_roundtrip
./build/test_nn
./build/test_nn_pipeline
./build/test_nn_reinforce
./build/test_sgd_weight_update
./build/test_bug3_sgd_gradients
./build/test_bug4_format_string
./build/test_bug5_truncated_nnwt
./build/test_bug7_concurrent_quantize
./build/test_bug8_sgd_concurrent
```

**HDF5 filter:**

```bash
./build/test_hdf5_configs
./build/test_h5z_8mb
./build/test_f9_transfers
```

**VOL connector:**

```bash
./build/test_vol_gpu_write
./build/test_vol2_gpu_fallback
./build/test_vol_8mb
./build/test_vol_comprehensive
./build/test_correctness_vol
```

---

## 6. Run Demos

### Gray-Scott simulation + NN compression

```bash
./build/grayscott_vol_demo neural_net/weights/model.nnwt --L 128 --steps 1000 --chunk_mb 4
```

This produces:
- `/tmp/grayscott_compressed.h5` — NN-compressed via VOL
- `/tmp/grayscott_decompressed.h5` — plain HDF5 after round-trip
- Bitwise GPU verification (PASS/FAIL per snapshot)

**Dataset size reference:**

| `--L` | Dataset Size |
|-------|-------------|
| 128   | 8 MB        |
| 256   | 64 MB       |
| 512   | 512 MB      |
| 640   | 1 GB        |
| 1000  | 4 GB        |
| 1260  | 8 GB        |

**More examples:**

```bash
# 64 MB dataset, 4 MB chunks
./build/grayscott_vol_demo neural_net/weights/model.nnwt --L 256 --chunk_mb 4

# 1 GB dataset, 64 MB chunks
./build/grayscott_vol_demo neural_net/weights/model.nnwt --L 640 --chunk_mb 64

# 5 snapshots of 64 MB
./build/grayscott_vol_demo neural_net/weights/model.nnwt --L 256 --steps 5000 --plotgap 1000

# 8 GB dataset (needs ~30 GB GPU memory)
./build/grayscott_vol_demo neural_net/weights/model.nnwt --L 1260 --chunk_mb 64
```

### Algorithm sweep (verify NN's choice)

```bash
./build/algo_sweep_verify neural_net/weights/model.nnwt --L 256 --chunk_mb 4
```

Runs all 8 compression algorithms on the same data and compares against NN's pick.

### Verify compressed vs decompressed files

```bash
HDF5_PLUGIN_PATH=$PWD/build \
h5diff /tmp/grayscott_compressed.h5 /tmp/grayscott_decompressed.h5
```

No output means the files match exactly.

---

## 7. Run Benchmarks

```bash
./build/benchmark neural_net/weights/model.nnwt
./build/benchmark_hdf5 neural_net/weights/model.nnwt
./build/benchmark_gpu_resident neural_net/weights/model.nnwt
./build/benchmark_vol_gpu neural_net/weights/model.nnwt
./build/benchmark_algo_sweep
./build/benchmark_lz4_vs_nocomp
```

---

## 8. VPIC Benchmark (Real Harris Sheet Simulation)

Runs a real VPIC-Kokkos Harris sheet reconnection simulation, then benchmarks
GPU-resident field compression through the GPUCompress VOL connector.

| Parameter | Value |
|-----------|-------|
| Grid cells | 128³ + ghost cells (~2.2M voxels) |
| Vars/cell | 16 floats |
| Dataset size | ~133 MB |
| Chunk size | 4 MB |
| Warmup | 100 simulation steps |

### Build the deck

```bash
cd /u/imuradli/vpic-kokkos/build-compress
./bin/vpic /u/imuradli/GPUCompress/tests/benchmarks/vpic_benchmark_deck.cxx
```

### Run the deck

```bash
export LD_LIBRARY_PATH=/u/imuradli/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
export GPUCOMPRESS_WEIGHTS=/u/imuradli/GPUCompress/neural_net/weights/model.nnwt
./vpic_benchmark_deck.Linux
```

> **Note:** `mpirun` is not available on Cray/MPICH nodes. Run the binary directly
> for single-rank, or use `srun -n 1 ./vpic_benchmark_deck.Linux` under SLURM.

### Output

- Console: per-phase ratio, write/read MB/s, verification, SGD/exploration stats
- CSV: `benchmark_vpic_deck_results/benchmark_vpic_deck.csv`
- Temp files: `/tmp/bm_vpic_*.h5` (removed after each phase)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `libnvcomp.so.5: cannot open shared object file` | `export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH` |
| `libhdf5.so.320: cannot open shared object file` | `export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH` |
| `atomicAdd(double*, double)` compile error | `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80` |
| Dependencies missing on new node | Re-run `bash scripts/install_dependencies.sh` |
| tmux session lost | `tmux attach -t gpu` |
| h5diff can't read compressed file | `export HDF5_PLUGIN_PATH=$PWD/build` |
