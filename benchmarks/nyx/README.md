# Nyx Cosmological Simulation Integration

GPU-accelerated lossless compression for the [Nyx](https://github.com/AMReX-Astro/Nyx)
AMReX-based cosmological hydrodynamics code. Data flows directly from GPU-resident
AMReX MultiFab memory through the GPUCompress HDF5 VOL connector without CPU
round-trips, following the same zero-copy pattern as the VPIC-Kokkos integration.

## Architecture

```
Nyx GPU Simulation (AMReX MultiFab on CUDA device)
    |
    v
writePlotFile() --- #ifdef AMREX_USE_GPUCOMPRESS
    |
    v
nyx_amrex_bridge.hpp::write_multifab_compressed()
    |
    v  (per FArrayBox)
write_gpu_to_hdf5()
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

### In GPUCompress (adapter + bridge, no Nyx dependency at build time)

| File | Description |
|------|-------------|
| `include/gpucompress_nyx.h` | Nyx adapter C API (opaque handle, borrows GPU pointers) |
| `src/nyx/nyx_adapter.cu` | Adapter implementation (120 lines, built into libgpucompress.so) |
| `examples/nyx_amrex_bridge.hpp` | Header-only AMReX MultiFab bridge (HDF5 VOL write + verify) |
| `cmake/NyxIntegration.cmake` | Build integration documentation |
| `benchmarks/nyx/README.md` | This file |
| `benchmarks/nyx/patches/` | Nyx source patches and modified files for reference |

### In Nyx (minimal, all behind `#ifdef AMREX_USE_GPUCOMPRESS`)

| File | Lines changed | Description |
|------|---------------|-------------|
| `Source/Driver/Nyx.H` | +3 | `static int use_gpucompress;` member declaration |
| `Source/Driver/Nyx.cpp` | +7 | Static initialization + ParmParse query |
| `Source/IO/Nyx_output.cpp` | +113 | GPUCompress plotfile write path |

When `AMREX_USE_GPUCOMPRESS` is not defined, all changes are compiled out.
The modified Nyx source files are provided in `benchmarks/nyx/patches/` for reference.

## Prerequisites

- CUDA 12.x with an NVIDIA GPU (tested on A100-40GB)
- MPI (OpenMPI or MPICH)
- HDF5 2.x built with parallel MPI support
- nvCOMP 5.x
- GPUCompress (libgpucompress.so, libH5VLgpucompress.so, libH5Zgpucompress.so)

## Step 1: Build GPUCompress

```bash
cd /path/to/GPUCompress
mkdir -p build && cd build
cmake -S .. -B . \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DHDF5_ROOT=/path/to/hdf5-mpi-install
cmake --build . -j$(nproc)
```

This produces:
- `libgpucompress.so` (core library, includes Nyx adapter)
- `libH5VLgpucompress.so` (HDF5 VOL connector)
- `libH5Zgpucompress.so` (HDF5 filter plugin)

## Step 2: Build HDF5 with MPI (if not already available)

```bash
cd /path/to/hdf5-source
mkdir build-mpi && cd build-mpi
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/path/to/hdf5-mpi-install \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_BUILD_CPP_LIB=OFF \
    -DHDF5_BUILD_FORTRAN=OFF \
    -DCMAKE_C_COMPILER=mpicc
cmake --build . -j$(nproc) && cmake --install .
```

## Step 3: Clone and Patch Nyx

```bash
git clone --recursive https://github.com/AMReX-Astro/Nyx.git
cd Nyx

# Update AMReX submodule to latest development (required for checkPointNow API)
cd subprojects/amrex && git checkout development && cd ../..

# Apply GPUCompress patch
git apply /path/to/GPUCompress/benchmarks/nyx/patches/nyx-gpucompress.patch
```

Alternatively, manually edit the 3 source files using the copies in `patches/` as reference.

## Step 4: Build Nyx with GPUCompress

Set paths to your installations:

```bash
export GPUCOMPRESS_DIR=/path/to/GPUCompress
export HDF5_DIR=/path/to/hdf5-mpi-install
export NVCOMP_DIR=/path/to/nvcomp
```

### Float32 build (recommended for compression benchmarks)

```bash
cd Nyx
mkdir build && cd build
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DNyx_MPI=YES \
    -DNyx_OMP=NO \
    -DNyx_HYDRO=YES \
    -DNyx_HEATCOOL=NO \
    -DNyx_GPU_BACKEND=CUDA \
    -DAMReX_CUDA_ARCH=Ampere \
    -DAMReX_PRECISION=SINGLE \
    -DAMReX_PARTICLES_PRECISION=SINGLE \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
    "-DCMAKE_CXX_FLAGS=-DAMREX_USE_GPUCOMPRESS -I${GPUCOMPRESS_DIR}/include -I${GPUCOMPRESS_DIR}/examples -I${HDF5_DIR}/include" \
    "-DCMAKE_CUDA_FLAGS=-DAMREX_USE_GPUCOMPRESS -I${GPUCOMPRESS_DIR}/include -I${GPUCOMPRESS_DIR}/examples -I${HDF5_DIR}/include" \
    "-DCMAKE_EXE_LINKER_FLAGS=-L${GPUCOMPRESS_DIR}/build -L${HDF5_DIR}/lib -L${NVCOMP_DIR}/lib -Wl,--no-as-needed -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -lhdf5 -lnvcomp -Wl,--as-needed" \
    "-DCMAKE_SHARED_LINKER_FLAGS=-L${HDF5_DIR}/lib -lhdf5"
```

### Build targets

```bash
# Sedov blast wave (recommended: fast-evolving, non-uniform data)
cmake --build . --target nyx_HydroTests -j$(nproc)

# Santa Barbara cosmological test (slow evolution, small grid)
cmake --build . --target nyx_MiniSB -j$(nproc)
```

### Float64 build (standard Nyx precision)

Remove `-DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE` from the
cmake command above. All other flags remain the same.

## Step 5: Run with GPUCompress

For all runs below, first set the library path:

```bash
export LD_LIBRARY_PATH=${GPUCOMPRESS_DIR}/build:${NVCOMP_DIR}/lib:${HDF5_DIR}/lib:$LD_LIBRARY_PATH
```

### Recommended: Sedov blast wave (256^3, float32, fast-evolving shock)

The Sedov point-explosion creates an expanding spherical shock front that
evolves from nearly uniform (highly compressible) to complex non-uniform
structure. This produces diverse chunks with varying compressibility,
ideal for exercising the NN algorithm selector. Float32 precision matches
VPIC's data type and produces better compression than float64.

```bash
cd build/Exec/HydroTests

# Full benchmark: 200 steps, write every 50, NN auto, ratio policy, verify on
mpirun -n 1 ./nyx_HydroTests inputs.3d.sph.sedov \
    nyx.use_gpucompress=1 \
    nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
    nyx.gpucompress_algorithm=auto \
    nyx.gpucompress_policy=ratio \
    nyx.gpucompress_verify=1 \
    amr.n_cell="256 256 256" \
    amr.max_grid_size=128 \
    max_step=200 \
    amr.plot_int=50 \
    amr.check_int=0 \
    nyx.v=1 amr.v=1
```

Expected output per plotfile: `832 MB -> 2-5 MB (141x-369x)` depending on
evolution stage. Verification adds ~1.5s per plotfile for read-back + compare.

### Compare all 8 fixed algorithms + NN auto

Run Sedov to step 100 (well-developed shock) and sweep algorithms:

```bash
for ALGO in lz4 snappy deflate gdeflate zstd bitcomp cascaded ans auto; do
    rm -rf plt*_gpuc
    mpirun -n 1 ./nyx_HydroTests inputs.3d.sph.sedov \
        nyx.use_gpucompress=1 \
        nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
        nyx.gpucompress_algorithm=$ALGO \
        nyx.gpucompress_policy=ratio \
        nyx.gpucompress_verify=1 \
        amr.n_cell="256 256 256" \
        amr.max_grid_size=128 \
        max_step=100 \
        amr.plot_int=100 \
        amr.check_int=0 \
        nyx.v=0 amr.v=0 nyx.sum_interval=-1 \
        2>&1 | grep "GPUCompress"
    # Check compressed size
    du -sh plt00100_gpuc/
    echo "---"
done
```

### Compare NN policies (speed vs balanced vs ratio)

```bash
for POLICY in speed balanced ratio; do
    rm -rf plt*_gpuc
    mpirun -n 1 ./nyx_HydroTests inputs.3d.sph.sedov \
        nyx.use_gpucompress=1 \
        nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
        nyx.gpucompress_algorithm=auto \
        nyx.gpucompress_policy=$POLICY \
        amr.n_cell="256 256 256" \
        amr.max_grid_size=128 \
        max_step=100 \
        amr.plot_int=100 \
        amr.check_int=0 \
        nyx.v=0 amr.v=0 nyx.sum_interval=-1 \
        2>&1 | grep "GPUCompress"
    du -sh plt00100_gpuc/
    echo "---"
done
```

### Sod shock tube (1D problem in 3D grid)

A simpler discontinuity problem. Less data diversity than Sedov but
still creates sharp density/pressure jumps:

```bash
cd build/Exec/HydroTests

mpirun -n 1 ./nyx_HydroTests inputs-sod-x \
    nyx.use_gpucompress=1 \
    nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
    nyx.gpucompress_algorithm=auto \
    nyx.gpucompress_policy=ratio \
    amr.n_cell="256 32 32" \
    amr.max_grid_size=128 \
    max_step=200 \
    amr.plot_int=50 \
    amr.check_int=0
```

### Santa Barbara cosmological test (32^3, float32 or float64)

The standard cosmological structure formation test. Uses pre-computed
initial conditions from `ic_sb_32.ascii`. Evolution is gravity-driven
(slower than Sedov). Requires many steps (500+) for structure to develop.
Small 32^3 grid limits compression ratios.

```bash
cd build/Exec/MiniSB

# Float32 build, 500 steps, evolve from z=63 to z~5
mpirun -n 1 ./nyx_MiniSB inputs.32 \
    nyx.use_gpucompress=1 \
    nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
    nyx.gpucompress_algorithm=auto \
    nyx.gpucompress_policy=ratio \
    amr.max_grid_size=32 \
    max_step=500 \
    amr.plot_int=100 \
    amr.check_int=0
```

Note: MiniSB with 32^3 grid produces only 4 MB per snapshot. Compression
ratios will be modest (~1.1x lossless) due to the small data size.
Use Sedov 256^3 for meaningful compression benchmarks.

### Grid size and box size guidelines

| Grid | max_grid_size | FArrayBoxes | Data/snapshot (f32) | GPU memory |
|------|---------------|-------------|---------------------|------------|
| 128^3 | 64 | 8 | 128 MB | ~4 GB |
| 256^3 | 128 | 8 | 832 MB | ~12 GB |
| 512^3 | 128 | 64 | 6.5 GB | ~30 GB |

Larger `max_grid_size` means fewer, larger FArrayBoxes. Each FArrayBox is
written as a single HDF5 file, chunked at 4 MiB by the VOL connector.
Larger boxes give the NN more chunks per file to analyze.

### Multi-GPU / multi-node

```bash
# 4 MPI ranks, 1 GPU per rank (set CUDA_VISIBLE_DEVICES via launcher)
mpirun -n 4 ./nyx_HydroTests inputs.3d.sph.sedov \
    nyx.use_gpucompress=1 \
    nyx.gpucompress_weights=${GPUCOMPRESS_DIR}/neural_net/weights/model.nnwt \
    amr.n_cell="256 256 256" \
    amr.max_grid_size=128 \
    max_step=100 \
    amr.plot_int=100 \
    amr.check_int=0
```

Each MPI rank independently compresses its local FArrayBoxes on its GPU.
No inter-rank communication for compression.

## Runtime Parameters

All parameters are set via Nyx's standard ParmParse mechanism (input file or command line):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nyx.use_gpucompress` | `0` | Enable GPUCompress for plotfile output |
| `nyx.gpucompress_weights` | (none) | Path to NN weights file (.nnwt). Required for `algorithm=auto` |
| `nyx.gpucompress_algorithm` | `auto` | Compression algorithm: `auto`, `lz4`, `snappy`, `deflate`, `gdeflate`, `zstd`, `ans`, `cascaded`, `bitcomp` |
| `nyx.gpucompress_policy` | `ratio` | NN ranking policy: `speed` (minimize compress time), `balanced` (equal weights), `ratio` (maximize compression ratio) |
| `nyx.gpucompress_verify` | `0` | Enable lossless round-trip verification (read back, decompress, compare bitwise) |

## Compression Algorithms

All algorithms are GPU-accelerated via nvCOMP:

| Algorithm | Speed | Ratio | Best for |
|-----------|-------|-------|----------|
| `lz4` | Fast | Moderate | General purpose, low latency |
| `snappy` | Fastest | Low | Streaming, byte-oriented data |
| `deflate` | Slow | Good | Moderate entropy data |
| `gdeflate` | Slow | Moderate | GPU-optimized deflate |
| `zstd` | Slow | Best | Structured scientific data |
| `ans` | Medium | Variable | Entropy coding |
| `cascaded` | Medium | Good | Floating-point specific |
| `bitcomp` | Fastest | Moderate | Bit-level scientific data |
| `auto` | Adaptive | Best tradeoff | **Recommended** -- NN selects per chunk |

## Benchmark Results

Sedov blast wave, 256^3 grid, float32, A100-40GB, step 100 (evolved shock front):

### Fixed algorithms (lossless, verified)

| Algorithm | Original | Compressed | Ratio | Time (ms) |
|-----------|----------|-----------|-------|-----------|
| zstd | 832 MB | 1.4 MB | 610x | 394 |
| auto (NN) | 832 MB | 3.3 MB | 250x | 288 |
| lz4 | 832 MB | 6.1 MB | 137x | 303 |
| deflate | 832 MB | 9.2 MB | 91x | 440 |
| bitcomp | 832 MB | 10.6 MB | 78x | 99 |
| cascaded | 832 MB | 11.2 MB | 75x | 152 |
| gdeflate | 832 MB | 17.8 MB | 47x | 435 |
| snappy | 832 MB | 40.1 MB | 21x | 497 |
| ans | 832 MB | 62.8 MB | 13x | 196 |

### NN policy comparison (auto algorithm, lossless)

| Policy | Compressed | Ratio | Time (ms) |
|--------|-----------|-------|-----------|
| ratio | 3.3 MB | 250x | 292 |
| speed | 352.6 MB | 2.4x | 446 |
| balanced | 352.6 MB | 2.4x | 433 |

### Evolution over time (auto, ratio policy)

| Step | Redshift | Compressed | Ratio | Physics |
|------|----------|-----------|-------|---------|
| 0 | - | 2 MB | 369x | Uniform ambient + point energy |
| 50 | - | 2 MB | 343x | Shock front forming |
| 100 | - | 3 MB | 250x | Shock expanding |
| 150 | - | 4 MB | 188x | Shock covers ~20% of domain |
| 200 | - | 5 MB | 141x | Shock covers ~30% of domain |

## Output Format

Compressed plotfiles are written to `<plotfile_name>_gpuc/`:

```
plt00100_gpuc/
    fab_0000.h5    # Compressed HDF5 per FArrayBox (VOL connector format)
    fab_0001.h5
    ...
    fab_0007.h5    # 8 boxes for 256^3 with max_grid_size=128
```

Each `.h5` file contains a single dataset `"data"` with `n_cells * n_components`
elements, chunked at 4 MiB, compressed by GPUCompress. The VOL connector stores
per-chunk compression metadata (algorithm, preprocessing, header) inside the
HDF5 chunk format. Files can be read back using the same VOL connector.

## Applying the Patch

The patch modifies 3 files in the Nyx source tree:

```bash
cd /path/to/Nyx
git apply /path/to/GPUCompress/benchmarks/nyx/patches/nyx-gpucompress.patch
```

If the patch does not apply cleanly (e.g., different Nyx version), apply the
changes manually. The modifications are minimal and self-contained:

**`Source/Driver/Nyx.H`** -- Add after `static int write_hdf5;` (line ~146):
```cpp
#ifdef AMREX_USE_GPUCOMPRESS
    static int use_gpucompress;
#endif
```

**`Source/Driver/Nyx.cpp`** -- Add after `int Nyx::write_hdf5 = 0;` (line ~225):
```cpp
#ifdef AMREX_USE_GPUCOMPRESS
int Nyx::use_gpucompress = 0;
#endif
```

And in `read_params()` after the `AMREX_USE_HDF5` check:
```cpp
#ifdef AMREX_USE_GPUCOMPRESS
    pp_nyx.query("use_gpucompress", use_gpucompress);
#endif
```

**`Source/IO/Nyx_output.cpp`** -- Add the GPUCompress include and write path
at the top of `writePlotFile()`. See `patches/Nyx_output.cpp` for the complete
modified file.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `libgpucompress.so.1: cannot open` | LD_LIBRARY_PATH missing | `export LD_LIBRARY_PATH=...` |
| `undefined reference to gpucompress_init` | CUDA linker order | Use `-Wl,--no-as-needed -lgpucompress ... -Wl,--as-needed` |
| `undefined reference to H5Dcreate2` | HDF5 not linked to shared libs | Add `-DCMAKE_SHARED_LINKER_FLAGS="-L.../lib -lhdf5"` |
| `AMReX HDF5 CUDA error 700` | AMReX's own HDF5 writer conflicts with CUDA | Use GPUCompress path (`nyx.use_gpucompress=1`) instead of `nyx.write_hdf5=1`. Do not enable both |
| `Forcing with device memory not implemented` | DrivenTurbulence forcing not GPU-ready | Use HydroTests (Sedov) or MiniSB instead |
| AMReX submodule too old | `checkPointNow()` API mismatch | `cd subprojects/amrex && git checkout development` |

## VPIC-Compatible Multi-Phase Benchmark

The `run_nyx_benchmark.sh` script runs a full 12-phase compression benchmark
using the Sedov blast wave test problem. Data evolves dramatically across
timesteps (compression ratio drops from ~369x to ~141x as the shock expands),
making it ideal for evaluating NN algorithm selection.

### How it works

**Phase 1 — Simulation dump:**
Run Nyx HydroTests (Sedov blast) with `NYX_DUMP_FIELDS=1`. The patched
`Nyx_output.cpp` writes raw `.f32` binary files for each FArrayBox component
(density, xmom, ymom, zmom, rho_E, rho_e, Temp, Ne, species, pressure) at
each plot interval. If `amrex::Real` is double, values are downcast to float32
for the dump. Data evolves as the Sedov shock expands from a point explosion
into uniform background.

**Phase 2 — Compression sweep:**
The `generic_benchmark` binary loads the `.f32` files in chronological order
(zero-padded `plt00000_`, `plt00001_`, ... ensures alphabetical = temporal).
All 12 phases run on each field. NN weights are reloaded per phase for clean
isolation. The sweep runs once per cost-model policy.

### Quick start

```bash
cd /path/to/GPUCompress

# Set paths
export NYX_BIN=$HOME/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests
export GPUCOMPRESS_WEIGHTS=neural_net/weights/model.nnwt

# Run benchmark (128^3 Sedov, 20 timesteps, balanced+ratio)
NYX_NCELL=128 \
NYX_MAX_STEP=200 \
NYX_PLOT_INT=10 \
CHUNK_MB=4 \
POLICIES="balanced,ratio" \
bash benchmarks/nyx/run_nyx_benchmark.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NYX_BIN` | auto-detect | Path to `nyx_HydroTests` binary |
| `NYX_NCELL` | `128` | Grid cells per dimension (128 = 2M cells, ~25 MB/component) |
| `NYX_MAX_STEP` | `200` | Total simulation steps |
| `NYX_PLOT_INT` | `10` | Steps between diagnostic dumps (200/10 = 20 timesteps) |
| `CHUNK_MB` | `4` | HDF5 chunk size in MB |
| `POLICIES` | `balanced,ratio` | Cost-model policies to sweep |
| `ERROR_BOUND` | `0.0` | Lossy error bound (0.0 = lossless) |
| `NO_RANKING` | `1` | Disable ranking profiler (recommended for large grids) |
| `RESULTS_DIR` | auto | Output directory |

### Data evolution

The Sedov blast wave creates excellent data diversity:

| Timestep | Physical state | Density ratio | Character |
|----------|---------------|---------------|-----------|
| 0 | Uniform background + point energy | ~370x | Highly compressible |
| 50 | Shock forming | ~340x | Slightly less |
| 100 | Developed shock, ~10% domain | ~250x | Mixed compressibility |
| 150 | Shock propagating, ~20% domain | ~190x | Diverse chunk types |
| 200 | Late time, ~30% domain | ~140x | Least compressible |

Behind the shock: turbulent mixing develops (high entropy, low compression).
Ahead of the shock: still uniform (high compression). This mix of high/low
compressibility chunks within each timestep is ideal for NN algorithm selection.

### Raw field dump format

With `NYX_DUMP_FIELDS=1`, each plot interval creates:

```
raw_fields/
  plt00000/
    fab0000_comp00_density.f32    # FArrayBox 0, component 0
    fab0000_comp01_xmom.f32       # FArrayBox 0, component 1
    ...
    fab0000_comp12_pressure.f32   # FArrayBox 0, component 12
  plt00001/
    fab0000_comp00_density.f32
    ...
```

Each file contains `ncells` float32 values (one FAB × one component).
For a 128^3 grid with `max_grid_size=128`, there is 1 FAB per file
(2,097,152 cells = 8 MB). For 256^3 with `max_grid_size=128`, there
are 8 FABs per timestep.

### Manual two-step run

```bash
# Step 1: Run Nyx with field dumping
export NYX_DUMP_FIELDS=1
export NYX_DUMP_DIR=/tmp/nyx_fields
$NYX_BIN inputs.sedov \
    nyx.use_gpucompress=1 \
    nyx.gpucompress_weights=neural_net/weights/model.nnwt \
    amr.n_cell="128 128 128" \
    amr.max_step=200 \
    amr.plot_int=10

# Step 2: Flatten and benchmark
mkdir -p /tmp/nyx_flat
for d in /tmp/nyx_fields/plt*; do
    for f in $d/*.f32; do
        ln -sf "$f" "/tmp/nyx_flat/$(basename $d)_$(basename $f)"
    done
done

N_FLOATS=$(( $(stat -c%s /tmp/nyx_flat/plt00000_fab0000_comp00_density.f32) / 4 ))
NO_RANKING=1 ./build/generic_benchmark neural_net/weights/model.nnwt \
    --data-dir /tmp/nyx_flat \
    --dims ${N_FLOATS},1 \
    --ext .f32 \
    --chunk-mb 4 \
    --out-dir /tmp/nyx_results \
    --name "nyx_sedov" \
    --w0 1.0 --w1 1.0 --w2 1.0
```

### Generate figures

```bash
SDR_DIR=/tmp/nyx_results \
python3 benchmarks/plots/generate_dataset_figures.py --dataset nyx_sedov
```

### Performance note

For grids >= 256^3, the ranking profiler becomes expensive (~2 min per
milestone at 200 chunks × 16 configs × 3 repeats). Use `NO_RANKING=1`
(default in the script) to disable it, or run ranking separately on a
single timestep.

### Scaling

| Grid | Cells | Components | Data/timestep | Dump files/timestep |
|------|-------|------------|---------------|---------------------|
| 64^3 | 262K | 13 | 13 MB | 13 |
| 128^3 | 2.1M | 13 | 104 MB | 13 |
| 256^3 | 16.8M | 13 | 832 MB | 104 (8 FABs × 13) |
