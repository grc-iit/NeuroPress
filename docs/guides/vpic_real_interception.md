# Real VPIC-Kokkos Data Interception

This guide shows how to intercept **real** GPU-resident simulation data from
VPIC-Kokkos and compress it on-the-fly using GPUCompress. No synthetic data —
the compression operates directly on Kokkos::Views produced by the PIC solver.

## Architecture

```
VPIC timestep loop (advance.cc)
  └─ user_diagnostics()          ← called every step
       └─ begin_diagnostics {}   ← your deck code runs here
            ├─ field_array->k_f_d.data()   → float* on GPU (16 vars × nv cells)
            ├─ hydro_array->k_h_d.data()   → float* on GPU (14 vars × nv cells)
            └─ sp->k_p_d.data()            → float* on GPU (7 vars × np particles)
                  ↓
            vpic_attach_fields(handle, k_f_d)
            H5Dwrite(dset, ..., d_data)     ← VOL intercepts GPU pointer
                  ↓
            VOL: GPU compress (nvcomp) → H5Dwrite_chunk (pre-compressed)
                  ↓
            compressed HDF5 file on disk (zero CPU copies)
```

**Key insight**: VPIC-Kokkos keeps all field/hydro/particle data GPU-resident
in Kokkos::Views. `View::data()` returns a raw `float*` device pointer. The
GPUCompress HDF5 VOL connector detects the GPU pointer in `H5Dwrite()`,
compresses each chunk on the GPU via nvcomp, and writes pre-compressed bytes
directly — the data never touches the CPU.

## Data Layout

| Data Type  | Kokkos View            | Variables | Access                    |
|------------|------------------------|-----------|---------------------------|
| Fields     | `k_field_t` = `float*[16]` | ex,ey,ez,div_e_err,cbx,cby,cbz,div_b_err,tcax,tcay,tcaz,rhob,jfx,jfy,jfz,rhof | `field_array->k_f_d` |
| Hydro      | `k_hydro_d_t` = `float*[14]` | jx,jy,jz,rho,px,py,pz,ke,txx,tyy,tzz,tyz,tzx,txy | `hydro_array->k_h_d` |
| Particles  | `k_particles_t` = `float*[7]` | dx,dy,dz,ux,uy,uz,w | `sp->k_p_d` (per species) |
| Particle i | `k_particles_i_t` = `int*`    | voxel index | `sp->k_p_i_d` |

## Quick Start (Delta/NCSA)

These are copy-paste commands for the NCSA Delta system. Run them on a
**GPU interactive node** (not the login node).

### Step 0: Get on a GPU node

```bash
salloc --account=YOUR_ACCOUNT --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --time=01:00:00
```

### Step 1: Build GPUCompress (if not already built)

```bash
cd /u/imuradli/GPUCompress
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Step 2: Initialize Kokkos submodule (one time only)

```bash
cd /u/imuradli/vpic-kokkos
git submodule update --init kokkos
```

### Step 3: Configure VPIC-Kokkos with GPUCompress deck

```bash
cd /u/imuradli/vpic-kokkos
export CRAYPE_LINK_TYPE=dynamic

rm -rf build-compress

cmake -S . -B build-compress -DENABLE_KOKKOS_CUDA=ON -DBUILD_INTERNAL_KOKKOS=ON -DENABLE_KOKKOS_OPENMP=OFF -DCMAKE_CXX_STANDARD=17 -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DUSER_DECK="/u/imuradli/GPUCompress/examples/vpic_compress_deck.cxx" '-DCMAKE_CXX_FLAGS=-I/u/imuradli/GPUCompress/include -I/u/imuradli/GPUCompress/examples -I/tmp/hdf5-install/include' '-DCMAKE_EXE_LINKER_FLAGS=-L/u/imuradli/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/tmp/hdf5-install/lib -lhdf5 -L/tmp/lib -lnvcomp'
```

**Note**: Use `-DUSER_DECK=` (singular), not `-DUSER_DECKS=`. VPIC's CMakeLists
has a bug where the plural form doesn't iterate correctly.

**Note**: If your GPU is not A100, change `Kokkos_ARCH_AMPERE80`:
- A100 → `Kokkos_ARCH_AMPERE80=ON`
- A40/A30 → `Kokkos_ARCH_AMPERE86=ON`
- H100 → `Kokkos_ARCH_HOPPER90=ON`
- V100 → `Kokkos_ARCH_VOLTA70=ON`

### Step 4: Build

```bash
cmake --build build-compress -j$(nproc)
```

This takes a while (Kokkos builds from source). The deck executable will be at
`build-compress/vpic_compress_deck`.

### Step 5: Verify the binary exists

```bash
ls -la build-compress/vpic_compress_deck
```

If it shows `build-compress/bin/vpic` instead, the `USER_DECK` variable wasn't
picked up. Double-check you used singular `USER_DECK` in Step 3.

### Step 6: Run

```bash
export LD_LIBRARY_PATH=/u/imuradli/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH

./build-compress/vpic_compress_deck
```

No `mpirun` or `srun` needed for single-rank — most MPI implementations
auto-initialize. If it fails with an MPI error, use:

```bash
mpiexec -np 1 ./build-compress/vpic_compress_deck
```

### Step 7: Check output

```bash
ls -la /tmp/vpic_*.h5
```

## Data Sizes

For the default 64x64x1 grid with 100 particles/cell:

| Data | Formula | Size |
|------|---------|------|
| Fields | nv × 16 × 4 bytes | ~0.8 MB |
| Hydro | nv × 14 × 4 bytes | ~0.7 MB |
| Particles (ions) | 204,800 × 7 × 4 bytes | ~5.5 MB |
| Particles (electrons) | 204,800 × 7 × 4 bytes | ~5.5 MB |
| **Total per snapshot** | | **~12.5 MB** |
| **Total (10 snapshots × 100 steps)** | | **~125 MB** |

To increase data sizes, edit lines 67-70 in `vpic_compress_deck.cxx`:

```cpp
double nx   = 64;    // increase for larger runs (e.g., 256)
double ny   = 64;    // increase for larger runs (e.g., 256)
double nz   = 1;     // set to 64+ for 3D
double nppc = 100;   // particles per cell
```

| Grid | Fields per snapshot | Particles (2 species) | Total |
|------|--------------------|-----------------------|-------|
| 64³ | ~17 MB | ~117 MB | ~134 MB |
| 128³ | ~134 MB | ~939 MB | ~1.1 GB |
| 256³ | ~1.1 GB | ~7.5 GB | ~8.6 GB |

## Writing Your Own Deck

The example deck (`vpic_compress_deck.cxx`) is a template. To add compression
to an existing deck, add these changes:

### 1. Include GPUCompress + HDF5 headers at the top

```cpp
#include "gpucompress.h"
#include "gpucompress_vpic.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
#include "vpic_kokkos_bridge.hpp"
#include <hdf5.h>
```

### 2. Add handles to globals

```cpp
begin_globals {
    // ... your existing globals ...
    gpucompress_vpic_t field_compressor;
    hid_t              vol_fapl;   // VOL-enabled file access property list
    hid_t              vol_id;     // VOL connector ID
};
```

### 3. Initialize GPUCompress + VOL in begin_initialization

```cpp
// At the end of begin_initialization:
gpucompress_init(NULL);  // NULL = no NN weights, fallback to LZ4
H5Z_gpucompress_register();
global->vol_id = H5VL_gpucompress_register();
hid_t native_id = H5VLget_connector_id_by_name("native");
global->vol_fapl = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_gpucompress(global->vol_fapl, native_id, NULL);
H5VLclose(native_id);

VpicSettings fs = vpic_default_settings();
fs.data_type = VPIC_DATA_FIELDS;
fs.n_cells   = grid->nv;
gpucompress_vpic_create(&global->field_compressor, &fs);
```

### 4. Write GPU data to compressed HDF5 in begin_diagnostics

```cpp
begin_diagnostics {
    if (step() % 10 != 0) return;

    // Fields — zero-copy from Kokkos::View to HDF5 via VOL
    vpic_attach_fields(global->field_compressor, field_array->k_f_d);

    float*  d_data = NULL;
    size_t  nbytes = 0;
    gpucompress_vpic_get_device_ptrs(global->field_compressor,
                                     &d_data, NULL, &nbytes, NULL);
    size_t n_floats = nbytes / sizeof(float);
    size_t chunk_floats = 4 * 1024 * 1024 / sizeof(float);  // 4 MB chunks

    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/vpic_fields_%06d.h5", step());

    hid_t fid   = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, global->vol_fapl);
    hid_t space = H5Screate_simple(1, (hsize_t[]){n_floats}, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, (hsize_t[]){chunk_floats});
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

    hid_t dset = H5Dcreate2(fid, "fields", H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    // d_data is a CUDA device pointer — VOL detects this, compresses on GPU,
    // writes pre-compressed chunks via H5Dwrite_chunk. Zero CPU copies.
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);

    H5Dclose(dset); H5Pclose(dcpl); H5Sclose(space); H5Fclose(fid);
}
```

## Available Data at Each Timestep

In `begin_diagnostics`, these are directly accessible (all GPU-resident):

| Variable | Type | Description |
|----------|------|-------------|
| `field_array->k_f_d` | `Kokkos::View<float*[16]>` | EM fields, all voxels |
| `hydro_array->k_h_d` | `Kokkos::View<float*[14]>` | Hydro moments, all voxels |
| `sp->k_p_d` | `Kokkos::View<float*[7], LayoutLeft>` | Particle floats (per species) |
| `sp->k_p_i_d` | `Kokkos::View<int*>` | Particle voxel indices |
| `grid->nv` | `int` | Total voxels (nx+2)*(ny+2)*(nz+2) |
| `sp->np` | `int` | Number of particles in species |
| `step()` | `int` | Current timestep |

Iterate species with:
```cpp
species_t* sp;
LIST_FOR_EACH(sp, species_list) {
    // sp->k_p_d, sp->k_p_i_d, sp->np, sp->name
}
```

## Expected Output

```
Step 10: writing GPU-compressed HDF5...
  Fields: 17 MB -> /tmp/vpic_fields_000010.h5 (5 chunks compressed on GPU)
  Hydro : 15 MB -> /tmp/vpic_hydro_000010.h5 (4 chunks compressed on GPU)
  Particles (ion): 204800 particles, 5 MB (2 chunks compressed on GPU)
  Particles (electron): 204800 particles, 5 MB (2 chunks compressed on GPU)
  Done. Files written to /tmp/vpic_*_10.h5
```

Output HDF5 files are standard and readable by any HDF5 tool. The VOL
connector writes pre-compressed chunks — decompression uses the registered
H5Z filter automatically on read.

Compression ratios depend on physics (smooth fields compress better than
turbulent particles). Use `benchmark_vpic_algo_sweep` with synthetic data
to find the best algorithm before deploying in a real simulation.

## Compression Algorithms

Set the algorithm in `H5Pset_gpucompress(dcpl, algo, ...)`:

| Enum Constant | Value | Best For |
|---------------|-------|----------|
| `GPUCOMPRESS_ALGO_AUTO` | 0 | NN-based automatic selection |
| `GPUCOMPRESS_ALGO_LZ4` | 1 | General purpose, fastest |
| `GPUCOMPRESS_ALGO_SNAPPY` | 2 | Low-latency |
| `GPUCOMPRESS_ALGO_DEFLATE` | 3 | Higher ratio |
| `GPUCOMPRESS_ALGO_GDEFLATE` | 4 | GPU-optimized deflate |
| `GPUCOMPRESS_ALGO_ZSTD` | 5 | Best ratio |
| `GPUCOMPRESS_ALGO_ANS` | 6 | Entropy coding |
| `GPUCOMPRESS_ALGO_CASCADED` | 7 | Structured numerical data |
| `GPUCOMPRESS_ALGO_BITCOMP` | 8 | Floating-point specific |

Add byte-shuffle preprocessing with `GPUCOMPRESS_PREPROC_SHUFFLE_4` as the
preprocessing argument.

## Troubleshooting

### Binary not found after build

If `build-compress/vpic_compress_deck` doesn't exist but `build-compress/bin/vpic` does:
- You used `-DUSER_DECKS=` (plural) instead of `-DUSER_DECK=` (singular)
- VPIC's CMakeLists has a bug: `USER_DECKS` triggers the check but iterates `USER_DECK`
- Fix: re-run cmake with `-DUSER_DECK=...` (singular)

### MPI initialization fails

VPIC requires MPI. If running without a launcher fails:
```bash
mpiexec -np 1 ./build-compress/vpic_compress_deck
```

### Kokkos architecture auto-detection fails

Pass the GPU arch explicitly: `-DKokkos_ARCH_AMPERE80=ON`

Check your GPU: `nvidia-smi --query-gpu=name --format=csv,noheader`

### CRAYPE_LINK_TYPE error

On Cray systems, set before configuring:
```bash
export CRAYPE_LINK_TYPE=dynamic
```

## HDF5 VOL Integration

The deck uses the GPUCompress HDF5 VOL connector by default. The full
pipeline is: GPU Kokkos::View → `H5Dwrite(d_ptr)` → VOL intercepts →
GPU compress (nvcomp) → `H5Dwrite_chunk` → HDF5 file. Zero CPU copies.

See `examples/vpic_vol_demo.cu` for a standalone demo with synthetic data
that tests the same VOL write/read/verify pipeline.
