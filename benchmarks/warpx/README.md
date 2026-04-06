# WarpX + GPUCompress Integration Benchmark

## Overview

WarpX is a Particle-In-Cell (PIC) plasma simulation built on AMReX.
This integration compresses WarpX's GPU-resident field data (E, B, J, rho)
and particle arrays in-situ using GPUCompress, avoiding device-to-host
round-trips.

## Prerequisites

- CUDA toolkit (11.0+)
- CMake 3.24+
- C++17 compiler
- MPI (optional, for multi-GPU runs)
- HDF5 with parallel support (optional, for VOL connector path)
- nvcomp (already part of GPUCompress build)

## Building WarpX from source

```bash
git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx
cd $HOME/src/warpx

# CPU-only smoke test (no GPU/MPI complexity)
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_MPI=OFF \
  -DWarpX_DIMS=3
cmake --build build-cpu -j $(nproc)

# CUDA build (production)
export CC=$(which gcc)
export CXX=$(which g++)
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=CUDA \
  -DWarpX_MPI=ON \
  -DWarpX_DIMS=3
cmake --build build-cuda -j $(nproc)
```

AMReX and PICSAR are fetched automatically by WarpX's CMake superbuild.

## Building GPUCompress with WarpX adapter

```bash
cd /path/to/GPUCompress
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j $(nproc)
```

The WarpX adapter (`warpx_adapter.cu`) is compiled into `libgpucompress.so`
alongside all other adapters.

## Integration approaches

### Approach A: Direct adapter API

Use `gpucompress_warpx.h` to borrow device pointers and compress directly:

```c
#include "gpucompress_warpx.h"

WarpxSettings settings = warpx_default_settings();
settings.data_type    = WARPX_DATA_EFIELD;
settings.n_components = 1;       /* one component per staggered MultiFab */
settings.element_size = 8;       /* AMReX Real = double */

gpucompress_warpx_t handle;
gpucompress_warpx_create(&handle, &settings);

/* Borrow device pointer from AMReX FArrayBox */
gpucompress_warpx_attach(handle, fab.dataPtr(), ncells);

/* Compress */
gpucompress_config_t config = gpucompress_default_config();
config.algorithm = GPUCOMPRESS_ALGO_AUTO;
gpucompress_stats_t stats;
gpucompress_compress_warpx(fab.dataPtr(), nbytes, d_output, &out_size, &config, &stats);

gpucompress_warpx_destroy(handle);
```

### Approach B: AMReX bridge with HDF5 VOL connector

Use `warpx_amrex_bridge.hpp` to write compressed HDF5 from GPU MultiFabs:

```cpp
#include "warpx_amrex_bridge.hpp"

hid_t fapl = gpucompress_warpx_bridge::init("weights.nnwt");

/* Write each field MultiFab */
gpucompress_warpx_bridge::write_field_compressed(
    "output/step_00100", "Ex", *Ex_mf, fapl,
    4*1024*1024,                        /* 4 MiB chunks */
    GPUCOMPRESS_ALGO_AUTO, 0.0, true);  /* lossless + verify */
```

## Validation

Run a stock WarpX 3D example to verify the build, then attach GPUCompress:

```bash
cd $HOME/src/warpx
# Pick a 3D example input
mpirun -np 4 build-cuda/bin/warpx Examples/Physics_applications/laser_acceleration/inputs_3d
```

Check for output in the configured diagnostic format (plotfiles by default).

## WarpX field data layout

WarpX uses a staggered Yee grid with separate MultiFabs per component:

| Field      | Components | Staggering      | Typical size per box |
|------------|-----------|-----------------|---------------------|
| Ex, Ey, Ez | 1 each    | Edge-centered   | ncells * sizeof(Real) |
| Bx, By, Bz | 1 each    | Face-centered   | ncells * sizeof(Real) |
| jx, jy, jz | 1 each    | Edge-centered   | ncells * sizeof(Real) |
| rho        | 1         | Node-centered   | ncells * sizeof(Real) |

Particle data is stored in AMReX Structure-of-Arrays (SoA) format:
x, y, z, ux, uy, uz, w (7 real components per particle).

## VPIC-Compatible Multi-Phase Benchmark

The `run_warpx_benchmark.sh` script runs a full 12-phase compression benchmark
using a laser wakefield acceleration (LWFA) simulation. Electromagnetic fields
evolve as the laser pulse creates a plasma wake, providing diverse compression
characteristics across timesteps.

### How it works

**Phase 1 — Simulation dump:**
Run WarpX with `WARPX_DUMP_FIELDS=1`. The patched `FlushFormatGPUCompress.cpp`
writes raw `.f32` binary files for each field component (Ex, Ey, Ez, Bx, By,
Bz, jx, jy, jz, rho) × each FArrayBox at each diagnostic interval. If
`amrex::Real` is double, values are downcast to float32. The LWFA physics
creates evolving electromagnetic fields — from smooth initial conditions
through laser-driven oscillations to complex wake structure.

**Phase 2 — Compression sweep:**
The `generic_benchmark` binary loads the `.f32` files in chronological order
(zero-padded `diag00000_`, `diag00020_`, ... ensures alphabetical = temporal).
All 12 phases (no-comp, lz4, snappy, deflate, gdeflate, zstd, ans, cascaded,
bitcomp, nn, nn-rl, nn-rl+exp50) run on each field. NN weights are reloaded
per phase for clean isolation. Runs once per cost-model policy.

### Quick start

```bash
cd /path/to/GPUCompress

# Set paths
export WARPX_BIN=$HOME/src/warpx/build-gpucompress/bin/warpx.3d
export GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt

# Run benchmark (32x32x256 LWFA, 20 timesteps, lossy eb=0.01)
WARPX_MAX_STEP=200 \
WARPX_DIAG_INT=10 \
WARPX_NCELL="32 32 256" \
CHUNK_MB=4 \
ERROR_BOUND=0.01 \
POLICIES="balanced,ratio" \
bash benchmarks/warpx/run_warpx_benchmark.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WARPX_BIN` | auto-detect | Path to `warpx.3d` binary |
| `WARPX_INPUTS` | LWFA example | Path to WarpX inputs file |
| `WARPX_MAX_STEP` | `200` | Total simulation steps |
| `WARPX_DIAG_INT` | `10` | Steps between diagnostics (200/10 = 20 timesteps) |
| `WARPX_NCELL` | `32 32 256` | Grid cells (nx ny nz) |
| `CHUNK_MB` | `4` | HDF5 chunk size in MB |
| `POLICIES` | `balanced,ratio` | Cost-model policies to sweep |
| `ERROR_BOUND` | `0.01` | Lossy error bound (WarpX fields benefit from lossy) |
| `RESULTS_DIR` | auto | Output directory |

### Data evolution

LWFA creates electromagnetic field evolution:

| Stage | Physical state | E-field character | Compression |
|-------|---------------|-------------------|-------------|
| Initial | Uniform plasma + laser seed | Smooth, sinusoidal | High (~77x vacuum FABs) |
| Early | Laser entering plasma | Oscillatory in wake region | Moderate |
| Mid | Bubble forming | Complex wake structure | Lower (~8x active FABs) |
| Late | Electron trapping | Shear layers, fine features | Lowest (~1.2x active FABs) |

The mix of vacuum FABs (high compression) and active-physics FABs (low
compression) within each timestep creates diverse chunk statistics — ideal
for NN algorithm selection benchmarking.

### Raw field dump format

With `WARPX_DUMP_FIELDS=1`, each diagnostic creates:

```
raw_fields/
  diag00000/
    lev0_Ex_fab0000.f32    # Level 0, Ex component, FAB 0
    lev0_Ex_fab0001.f32    # Level 0, Ex component, FAB 1
    ...
    lev0_rho_fab0003.f32   # Level 0, rho component, FAB 3
  diag00020/
    lev0_Ex_fab0000.f32
    ...
```

Each file contains `ncells` float32 values (one FAB × one component).
For a 32×32×256 grid with `max_grid_size=64`, there are 4 FABs.

### WarpX-specific notes

- **Staggered Yee grid**: Different field components (E, B, J, rho) have
  slightly different grid sizes due to edge/face/node centering. The benchmark
  handles this by measuring each file's size independently.
- **Single precision**: WarpX is typically built with `WarpX_PRECISION=SINGLE`
  for GPU performance. The dump writes float32 directly in this case.
- **Lossy compression**: WarpX electromagnetic fields compress very well with
  lossy modes (`ERROR_BOUND=0.01`). The Pareto plot shows the ratio-throughput
  tradeoff across algorithms.

### Manual two-step run

```bash
# Step 1: Run WarpX with field dumping
export WARPX_DUMP_FIELDS=1
export WARPX_DUMP_DIR=/tmp/warpx_fields
$WARPX_BIN inputs_3d_lwfa \
    max_step=200 \
    amr.n_cell="32 32 256" \
    diagnostics.diags_names=diag1 \
    diag1.intervals=10 \
    diag1.diag_type=Full \
    diag1.format=gpucompress \
    gpucompress.weights_path=neural_net/weights/model.nnwt \
    gpucompress.algorithm=auto

# Step 2: Flatten and benchmark
mkdir -p /tmp/warpx_flat
for d in /tmp/warpx_fields/diag*; do
    for f in $d/*.f32; do
        ln -sf "$f" "/tmp/warpx_flat/$(basename $d)_$(basename $f)"
    done
done

N_FLOATS=$(( $(stat -c%s /tmp/warpx_flat/diag00000_lev0_Ex_fab0000.f32) / 4 ))
./build/generic_benchmark neural_net/weights/model.nnwt \
    --data-dir /tmp/warpx_flat \
    --dims ${N_FLOATS},1 \
    --ext .f32 \
    --chunk-mb 4 \
    --out-dir /tmp/warpx_results \
    --name "warpx_lwfa" \
    --w0 1.0 --w1 1.0 --w2 1.0 \
    --error-bound 0.01
```

### Generate figures

```bash
SDR_DIR=/tmp/warpx_results \
python3 benchmarks/plots/generate_dataset_figures.py --dataset warpx_lwfa
```

### Scaling

| Grid | Cells | Components | FABs (msg=64) | Data/timestep |
|------|-------|------------|---------------|---------------|
| 32×32×256 | 262K | 10 | 4 | 40 MB |
| 64×64×512 | 2.1M | 10 | 32 | 320 MB |
| 128×128×1024 | 16.8M | 10 | 256 | 2.5 GB |
