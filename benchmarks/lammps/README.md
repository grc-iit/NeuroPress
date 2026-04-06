# LAMMPS Molecular Dynamics Integration

GPU-accelerated lossless compression for [LAMMPS](https://github.com/lammps/lammps)
molecular dynamics simulations using the KOKKOS GPU backend. Device pointers from
Kokkos Views (`atomKK->k_x.view_device().data()`) are passed directly through the
GPUCompress HDF5 VOL connector without CPU round-trips, following the same zero-copy
pattern as the VPIC, Nyx, and nekRS integrations.

## Architecture

```
LAMMPS GPU Simulation (KOKKOS + CUDA)
    |
    v  (every N timesteps)
fix gpucompress  --- end_of_step()
    |
    |--- atomKK->k_x.view_device().data()  (positions, GPU pointer)
    |--- atomKK->k_v.view_device().data()  (velocities, GPU pointer)
    |--- atomKK->k_f.view_device().data()  (forces, GPU pointer)
    |
    v
gpucompress_lammps_write_field()  (C bridge library)
    |
    v  (GPU device pointer passed to H5Dwrite)
HDF5 VOL Connector (libH5VLgpucompress.so)
    |--- detects CUDA device pointer via cudaPointerGetAttributes()
    |--- splits into 4 MiB chunks (H5Pset_chunk)
    |--- per chunk: stats -> NN inference -> algorithm selection -> nvCOMP compress
    v
H5Dwrite_chunk() --- pre-compressed bytes written to HDF5 file
```

## Files

### In GPUCompress (adapter + bridge, no LAMMPS dependency at build time)

| File | Description |
|------|-------------|
| `include/gpucompress_lammps.h` | LAMMPS adapter C API (opaque handle, borrows GPU pointers) |
| `src/lammps/lammps_adapter.cu` | Adapter implementation (built into libgpucompress.so) |
| `examples/lammps_gpucompress_udf.h` | C bridge API for LAMMPS fixes |
| `examples/lammps_gpucompress_udf.cpp` | Bridge implementation (compiled as liblammps_gpucompress_udf.so) |
| `benchmarks/lammps/README.md` | This file |
| `benchmarks/lammps/patches/` | LAMMPS source patches and modified files |

### In LAMMPS (minimal changes)

| File | Lines changed | Description |
|------|---------------|-------------|
| `src/KOKKOS/fix_gpucompress_kokkos.h` | +53 (new) | Fix header with style registration |
| `src/KOKKOS/fix_gpucompress_kokkos.cpp` | +174 (new) | Fix implementation — accesses KOKKOS device arrays |
| `cmake/Modules/Packages/KOKKOS.cmake` | +4 | Register fix source and style |

The modified and new LAMMPS files are provided in `benchmarks/lammps/patches/`.

## Prerequisites

- CUDA 12.x with an NVIDIA GPU (tested on A100-40GB)
- MPI (OpenMPI or MPICH)
- HDF5 1.14+ with VOL support (built from source or system package)
- nvCOMP 5.x
- GPUCompress (libgpucompress.so, libH5VLgpucompress.so, libH5Zgpucompress.so)

## Step 1: Build GPUCompress

```bash
cd /path/to/GPUCompress
mkdir -p build && cd build
cmake -S .. -B . \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DHDF5_ROOT=/path/to/hdf5-install
cmake --build . -j$(nproc)
```

This produces:
- `libgpucompress.so` (core library, includes LAMMPS adapter)
- `libH5VLgpucompress.so` (HDF5 VOL connector)
- `libH5Zgpucompress.so` (HDF5 filter plugin)

## Step 2: Build the GPUCompress bridge library

```bash
cd /path/to/GPUCompress/examples
g++ -shared -fPIC -o liblammps_gpucompress_udf.so lammps_gpucompress_udf.cpp \
    -I../include -I/path/to/hdf5-install/include -I/usr/local/cuda/include \
    -L../build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -L/path/to/hdf5-install/lib -lhdf5 \
    -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/path/to/GPUCompress/build \
    -Wl,-rpath,/path/to/hdf5-install/lib
```

## Step 3: Clone and Patch LAMMPS

```bash
git clone https://github.com/lammps/lammps.git
cd lammps
```

### Patch files and where to place them

The `benchmarks/lammps/patches/` directory contains all files needed:

| Patch file | Place in LAMMPS repo at | Action |
|-----------|------------------------|--------|
| `fix_gpucompress_kokkos.h` | `src/KOKKOS/fix_gpucompress_kokkos.h` | **New file** — copy into KOKKOS package |
| `fix_gpucompress_kokkos.cpp` | `src/KOKKOS/fix_gpucompress_kokkos.cpp` | **New file** — copy into KOKKOS package |
| `KOKKOS.cmake` | `cmake/Modules/Packages/KOKKOS.cmake` | **Reference** — full modified file for comparison |
| `lammps-gpucompress.patch` | Apply at repo root | **Git patch** — modifies `cmake/Modules/Packages/KOKKOS.cmake` |

### Option A: Apply via git patch (recommended)

```bash
# Copy the 2 new source files
cp /path/to/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.h src/KOKKOS/
cp /path/to/GPUCompress/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp src/KOKKOS/

# Apply the cmake patch (4 lines added to KOKKOS.cmake)
git apply /path/to/GPUCompress/benchmarks/lammps/patches/lammps-gpucompress.patch
```

### Option B: Manual edit

Copy the 2 new source files as above, then manually edit `cmake/Modules/Packages/KOKKOS.cmake`:

**Change 1** — After the `PKG_PHONON` block (around line 254), before `set_property(GLOBAL ...)`, add:
```cmake
# GPUCompress fix (KOKKOS-only, no base style)
list(APPEND KOKKOS_PKG_SOURCES ${KOKKOS_PKG_SOURCES_DIR}/fix_gpucompress_kokkos.cpp)
```

**Change 2** — After `# register kokkos-only styles` (around line 262), add:
```cmake
RegisterFixStyle(${KOKKOS_PKG_SOURCES_DIR}/fix_gpucompress_kokkos.h)
```

A complete reference copy of the modified `KOKKOS.cmake` is provided at `patches/KOKKOS.cmake` for comparison.

## Step 4: Build LAMMPS with KOKKOS + GPUCompress

Set paths:

```bash
export GPUCOMPRESS_DIR=/path/to/GPUCompress
export HDF5_DIR=/path/to/hdf5-install
```

### Float32 build (recommended for compression benchmarks)

```bash
cd lammps
mkdir build && cd build
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DKOKKOS_PREC=SINGLE \
    -DBUILD_MPI=ON \
    -DCMAKE_CXX_FLAGS="-I${GPUCOMPRESS_DIR}/include -I${GPUCOMPRESS_DIR}/examples -I${HDF5_DIR}/include -I/usr/local/cuda/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${GPUCOMPRESS_DIR}/examples -llammps_gpucompress_udf -L${GPUCOMPRESS_DIR}/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L${HDF5_DIR}/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,${GPUCOMPRESS_DIR}/build -Wl,-rpath,${GPUCOMPRESS_DIR}/examples -Wl,-rpath,${HDF5_DIR}/lib"
make -j$(nproc)
```

Change `Kokkos_ARCH_AMPERE80` to match your GPU:
- V100: `Kokkos_ARCH_VOLTA70`
- A100: `Kokkos_ARCH_AMPERE80`
- H100: `Kokkos_ARCH_HOPPER90`

### Float64 build

Replace `-DKOKKOS_PREC=SINGLE` with `-DKOKKOS_PREC=DOUBLE` (or omit it — double is the default).

## Step 5: Run with GPUCompress

Set the library path:

```bash
export LD_LIBRARY_PATH=${GPUCOMPRESS_DIR}/build:${GPUCOMPRESS_DIR}/examples:${HDF5_DIR}/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Recommended: Large-scale LJ melt (2M atoms, fp32)

Create an input script `in.melt_gpuc`:

```lammps
# Large-scale LJ melt — 2M atoms for GPU compression benchmark
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 80 0 80 0 80
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 3.0 87287 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

fix             1 all nve

# GPUCompress: write compressed positions, velocities, forces every 50 steps
fix             gpuc all gpucompress 50 positions velocities forces

thermo          50
run             200
```

Run with KOKKOS on GPU:

```bash
export GPUCOMPRESS_ALGO=auto
export GPUCOMPRESS_POLICY=ratio
export GPUCOMPRESS_VERIFY=1
export GPUCOMPRESS_WEIGHTS=/path/to/GPUCompress/neural_net/weights/model.nnwt

./lmp -k on g 1 -sf kk -in in.melt_gpuc
```

### Fix syntax

```
fix ID group gpucompress N [positions] [velocities] [forces]
```

- `ID` — fix name (any string)
- `group` — atom group (typically `all`)
- `N` — dump every N timesteps
- `positions` / `velocities` / `forces` — which fields to compress (default: all three)

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUCOMPRESS_ALGO` | `auto` | Algorithm: `auto`, `lz4`, `snappy`, `deflate`, `gdeflate`, `zstd`, `ans`, `cascaded`, `bitcomp` |
| `GPUCOMPRESS_POLICY` | `ratio` | NN ranking policy: `speed`, `balanced`, `ratio` |
| `GPUCOMPRESS_VERIFY` | `0` | Set to `1` for lossless round-trip verification |
| `GPUCOMPRESS_WEIGHTS` | (hardcoded path) | Path to NN model weights file (.nnwt) |

### Compare all algorithms

```bash
for ALGO in lz4 snappy deflate gdeflate zstd ans cascaded bitcomp auto; do
    rm -rf gpuc_step_*
    GPUCOMPRESS_ALGO=$ALGO ./lmp -k on g 1 -sf kk -in in.melt_gpuc \
        2>&1 | grep "GPUCompress"
    echo "=== $ALGO ==="
    du -sh gpuc_step_0000000100/
done
```

### Compare NN policies

```bash
for POLICY in speed balanced ratio; do
    rm -rf gpuc_step_*
    GPUCOMPRESS_ALGO=auto GPUCOMPRESS_POLICY=$POLICY \
        ./lmp -k on g 1 -sf kk -in in.melt_gpuc \
        2>&1 | grep "GPUCompress"
    echo "=== $POLICY ==="
    du -sh gpuc_step_0000000100/
done
```

### Scale up for data-intensive benchmarks

```lammps
# ~8.6M atoms (increase box from 80 to 120)
region          box block 0 120 0 120 0 120

# ~27M atoms
region          box block 0 160 0 160 0 160
```

| Box size | Atoms | Data/dump (fp32, 3 fields) | GPU memory |
|----------|-------|---------------------------|------------|
| 40^3 | 256K | 8.8 MB | ~1 GB |
| 80^3 | 2.0M | 70.3 MB | ~2 GB |
| 120^3 | 6.9M | 237 MB | ~6 GB |
| 160^3 | 16.4M | 563 MB | ~12 GB |

## Benchmark Results

LJ melt, 2,048,000 atoms, fp32 (SINGLE_SINGLE), A100-40GB, step 100 (equilibrated):

### Fixed algorithms (lossless, verified)

| Algorithm | Compressed (MB) | Ratio | Notes |
|-----------|----------------|-------|-------|
| Zstd | 56.9 | 1.23x | Best ratio |
| Deflate | 57.2 | 1.22x | |
| GDeflate | 57.4 | 1.22x | |
| Auto (NN) | 59.6 | 1.17x | NN ratio policy |
| ANS | 60.5 | 1.16x | |
| Snappy | 62.2 | 1.12x | |
| LZ4 | 64.4 | 1.09x | Fastest |
| Bitcomp | 66.1 | 1.06x | |
| Cascaded | 67.4 | 1.04x | Worst for MD data |

Original data per dump: 70.3 MB (2M atoms × 3 components × 4 bytes × 3 fields).

### NN policy comparison (auto algorithm, lossless, verified)

| Policy | Compressed (MB) | Ratio |
|--------|----------------|-------|
| ratio | 59.6 | 1.17x |
| speed | 66.5 | 1.05x |
| balanced | 67.4 | 1.04x |

Note: Equilibrated MD particle data (random positions, velocities, forces) is
inherently hard to compress losslessly. Initial timesteps with lattice structure
compress better (up to 1.18x for step 0 positions).

### Data characteristics

LAMMPS MD data differs from grid-based simulations (Nyx, nekRS, VPIC):
- **Particle data** — positions, velocities, forces are per-atom vectors
- **High entropy** — after equilibration, values appear near-random
- **No spatial coherence** — atoms are randomly ordered in memory
- **fp32** — 4 bytes per component, 12 bytes per atom per field

This makes MD data one of the hardest workloads for lossless compression,
making it an excellent stress test for the NN algorithm selector.

## Output Format

Compressed snapshots are written to `gpuc_step_<NNNNNNNNNN>/`:

```
gpuc_step_0000000100/
    x_rank0000.h5    # Positions (compressed HDF5, VOL format)
    v_rank0000.h5    # Velocities
    f_rank0000.h5    # Forces
```

Each `.h5` file contains a single dataset with `n_atoms * 3` elements, chunked
at 4 MiB, compressed by GPUCompress. With MPI, each rank writes its local atoms
as separate files (`rank0000`, `rank0001`, etc.).

## How it works (zero source changes to LAMMPS core)

The integration uses LAMMPS's fix system — the standard extension mechanism:

1. `fix_gpucompress_kokkos.cpp` implements `end_of_step()` hook
2. On each output step, it calls `atomKK->sync(Device, X_MASK | V_MASK | F_MASK)` to ensure GPU data is current
3. Extracts raw CUDA pointers via `atomKK->k_x.view_device().data()`
4. Passes them to the precompiled GPUCompress bridge library
5. Bridge calls `H5Dwrite()` with the GPU pointer — VOL connector detects CUDA memory and compresses on GPU

The fix is a new KOKKOS-only style (`fix gpucompress`). It requires 2 new source files + 4 lines in the KOKKOS cmake. No existing LAMMPS files are modified beyond the cmake registration.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Unrecognized fix style 'gpucompress'` | Fix not registered in cmake | Re-run cmake after adding fix files and cmake changes |
| `libgpucompress.so: cannot open` | LD_LIBRARY_PATH missing | `export LD_LIBRARY_PATH=...` |
| `liblammps_gpucompress_udf.so: cannot open` | Bridge library not found | Build bridge library and set rpath/LD_LIBRARY_PATH |
| `Not running with KOKKOS!` | Missing `-k on g 1 -sf kk` flags | Always use KOKKOS runtime flags |
| Segfault at end of run | GPUCompress finalize after CUDA teardown | Fixed in latest version (skip finalize in destructor) |
| Low compression ratios | MD data is inherently high-entropy | Expected for lossless; try lossy with `GPUCOMPRESS_ERROR_BOUND=0.01` |

## VPIC-Compatible Multi-Phase Benchmark

The `run_lammps_vpic_benchmark.sh` script runs a full 12-phase compression
benchmark that produces the same CSV schema and figure suite as the VPIC
benchmark deck. This enables apple-to-apple comparison across simulations.

### How it works

The benchmark uses a two-phase approach:

**Phase 1 — Simulation dump:**
Run LAMMPS once with `LAMMPS_DUMP_FIELDS=1`. The fix writes raw `.f32` binary
files (positions, velocities, forces) at each dump interval. The simulation
uses a hot-sphere explosion (hot region velocity=10.0, cold=0.01) to create
evolving density fronts across timesteps — data is never static.

**Phase 2 — Compression sweep:**
The `generic_benchmark` binary loads each `.f32` file into GPU memory and
sweeps all 12 compression phases (no-comp, 8 fixed algorithms, nn, nn-rl,
nn-rl+exp50) on the same data. Files are read in chronological order
(zero-padded filenames ensure alphabetical = temporal). NN weights are
reloaded between phases for clean isolation. The sweep runs once per
cost-model policy (balanced, ratio).

### Quick start

```bash
cd /path/to/GPUCompress

# Set paths
export LMP_BIN=$HOME/lammps/build/lmp
export GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt

# Run benchmark (2M atoms, 10 timesteps, balanced+ratio policies)
LMP_ATOMS=80 \
TIMESTEPS=10 \
SIM_INTERVAL=50 \
WARMUP_STEPS=100 \
CHUNK_MB=4 \
POLICIES="balanced,ratio" \
bash benchmarks/lammps/run_lammps_vpic_benchmark.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LMP_BIN` | `$HOME/lammps/build/lmp` | Path to LAMMPS binary |
| `LMP_ATOMS` | `80` | Box size per dimension (80 = ~2M atoms) |
| `TIMESTEPS` | `10` | Number of benchmark write cycles |
| `SIM_INTERVAL` | `50` | Physics steps between each dump (data evolution) |
| `WARMUP_STEPS` | `100` | Physics steps before first benchmark dump |
| `CHUNK_MB` | `4` | HDF5 chunk size in MB |
| `POLICIES` | `balanced,ratio` | Cost-model policies to sweep |
| `ERROR_BOUND` | `0.0` | Lossy error bound (0.0 = lossless) |
| `RESULTS_DIR` | auto | Output directory |

### NN phase control

The fix supports `GPUCOMPRESS_SGD` and `GPUCOMPRESS_EXPLORE` env vars to
distinguish the three NN phases:

| Phase | `GPUCOMPRESS_SGD` | `GPUCOMPRESS_EXPLORE` | Behavior |
|-------|-------------------|-----------------------|----------|
| `nn` | `0` | `0` | Inference only (frozen weights) |
| `nn-rl` | `1` | `0` | Online SGD learning, no exploration |
| `nn-rl+exp50` | `1` | `1` | Online SGD + exploration (default) |

### Manual two-step run

```bash
# Step 1: Dump fields
export LAMMPS_DUMP_FIELDS=1
export LAMMPS_DUMP_DIR=/tmp/lammps_fields
export GPUCOMPRESS_ALGO=lz4
export GPUCOMPRESS_SGD=0
export GPUCOMPRESS_EXPLORE=0
$LMP_BIN -k on g 1 -sf kk -in in.melt_gpuc

# Step 2: Benchmark one field type
N_FLOATS=$(( $(stat -c%s /tmp/lammps_fields/positions_step0000000000.f32) / 4 ))
./build/generic_benchmark neural_net/weights/model.nnwt \
    --data-dir /tmp/lammps_fields \
    --dims ${N_FLOATS},1 \
    --ext .f32 \
    --chunk-mb 4 \
    --out-dir /tmp/lammps_results \
    --name "lammps_positions" \
    --w0 1.0 --w1 1.0 --w2 1.0
```

### Output files

```
results/
  balanced/
    positions/
      benchmark_lammps_positions.csv              # Aggregate (1 row per phase)
      benchmark_lammps_positions_timesteps.csv    # Per-timestep (52 columns)
      benchmark_lammps_positions_timestep_chunks.csv  # Per-chunk at milestones
      benchmark_lammps_positions_ranking.csv      # Kendall tau quality
      benchmark_lammps_positions_ranking_costs.csv
    velocities/
      ...
    forces/
      ...
  ratio/
    ...
```

### Generate figures

```bash
# Point plotter at results directory
SDR_DIR=benchmarks/lammps/results/vpic_eval_box80_chunk4mb_ts10/balanced \
python3 benchmarks/plots/generate_dataset_figures.py --dataset lammps_positions
```

Produces the full figure suite: summary bar charts, SGD convergence, algorithm
evolution heatmap, predicted-vs-actual scatter, ranking quality, pipeline
waterfall — identical to VPIC/Gray-Scott figures.

### Scaling

| Box size | Atoms | Data/field (fp32) | Data/dump (3 fields) |
|----------|-------|-------------------|----------------------|
| 40^3 | 256K | 2.9 MB | 8.8 MB |
| 80^3 | 2.0M | 23.4 MB | 70.3 MB |
| 120^3 | 6.9M | 79 MB | 237 MB |
| 160^3 | 16.4M | 188 MB | 563 MB |
