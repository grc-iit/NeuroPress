# Post Neutron Star Merger Dataset

## Overview

The **Post Neutron Star Merger** dataset is part of [The Well](https://polymathic-ai.org/the_well/) collection by Polymathic AI. It contains axisymmetrized snapshots from 3D general relativistic neutrino radiation magnetohydrodynamics (GR-nu-RMHD) simulations of accretion disk evolution after a neutron star merger.

- **Source page**: <https://polymathic-ai.org/the_well/datasets/post_neutron_star_merger/>
- **HuggingFace**: <https://huggingface.co/datasets/polymathic-ai/post_neutron_star_merger>
- **Simulation code**: [nubhlight](https://github.com/lanl/nubhlight) (LANL, open-source, LA-CC C20019)
- **License**: CC-BY-4.0
- **Primary reference**: Miller et al. 2019, ApJS 241:2

---

## Dataset Specifications

| Property | Value |
|----------|-------|
| Total size | ~105 GB (8 HDF5 files) |
| File format | HDF5 (`.hdf5`) |
| Simulations | 8 trajectories |
| Timesteps per simulation | 181 |
| Spatial resolution | 192 x 128 x 66 (radial x polar x azimuthal) |
| Grid type | Uniform in log-spherical (MKS) coordinates |
| Data precision | float32 (fields), float64 (metric) |
| Storage cadence | 50 code units (~0.6 ms physical) |
| Total time range | 10,000 code units (~127 ms physical) |
| Per-file size | ~13.14 GB |
| Per-sample size (1 timestep) | ~167 MB |

### Spatial Domain

- **Radius**: 2 to 1,000 code units (outer boundary ~4,000)
- **Polar angle**: 0 to pi
- **Azimuthal angle**: 0 to 2*pi
- **Coordinate system**: Modified Kerr-Schild (MKS) with logarithmic radial spacing and equatorial focusing
- **Boundary conditions**: Open (radial), Polar (theta), Periodic (phi)

---

## File Layout

### Splits and Files

| Split | File | Size | Samples |
|-------|------|------|---------|
| train | `post_neutron_star_merger_scenario_0.hdf5` | 13.14 GB | 180 |
| train | `post_neutron_star_merger_scenario_3.hdf5` | 13.14 GB | 180 |
| train | `post_neutron_star_merger_scenario_4.hdf5` | 13.14 GB | 180 |
| train | `post_neutron_star_merger_scenario_5.hdf5` | 13.14 GB | 180 |
| train | `post_neutron_star_merger_scenario_6.hdf5` | 13.14 GB | 180 |
| train | `post_neutron_star_merger_scenario_7.hdf5` | 13.14 GB | 180 |
| valid | `post_neutron_star_merger_scenario_1.hdf5` | 13.14 GB | 180 |
| test  | `post_neutron_star_merger_scenario_2.hdf5` | 13.14 GB | 180 |
| **Total** | **8 files** | **~105 GB** | **1,440** |

### Internal HDF5 Structure (per file)

```
scalars/
  a                                  ()                        float32   # black hole spin
  mbh                                ()                        float32   # black hole mass (solar masses)

dimensions/
  log_r                              (192,)                    float32   # radial grid points
  theta                              (128,)                    float32   # polar grid points
  phi                                (66,)                     float32   # azimuthal grid points
  time                               (181,)                    float32   # timestep values

t0_fields/                                                               # scalar fields
  density                            (1, 181, 192, 128, 66)   float32   # fluid density
  internal_energy                    (1, 181, 192, 128, 66)   float32
  electron_fraction                  (1, 181, 192, 128, 66)   float32   # Ye
  temperature                        (1, 181, 192, 128, 66)   float32
  entropy                            (1, 181, 192, 128, 66)   float32
  pressure                           (1, 181, 192, 128, 66)   float32

t1_fields/                                                               # vector fields
  velocity                           (1, 181, 192, 128, 66, 3) float32  # 3-velocity
  magnetic_field                     (1, 181, 192, 128, 66, 3) float32  # B-field

t2_fields/                                                               # (empty)

additional_information/
  g_contravariant                    (1, 192, 128, 4, 4)      float64   # spacetime metric (time-independent)

boundary_conditions/
  log_r_open/mask                    (192,)                    bool
  theta_periodic/mask                (128,)                    bool
  phi_periodic/mask                  (66,)                     bool
```

---

## The 8 Simulations

| Scenario | Name | Description | Black Hole Mass | Spin (a) |
|----------|------|-------------|-----------------|----------|
| 0 | collapsar_hi | Disk from massive rotating star collapse | varies | varies |
| 1 | torus_b10 | GW170817-inspired, strongest B-field | 2.8 M_sun | 0.8 |
| 2 | torus_b30 | GW170817-inspired, intermediate B-field | 2.8 M_sun | 0.8 |
| 3 | torus_gw170817 | GW170817-inspired, weakest B-field | 2.8 M_sun | 0.8 |
| 4 | torus_MBH_10 | BH-NS merger | 10 M_sun | varies |
| 5 | torus_MBH_2p31 | BH-NS merger | 2.31 M_sun | varies |
| 6 | torus_MBH_2p67 | BH-NS merger | 2.67 M_sun | varies |
| 7 | torus_MBH_2p69 | BH-NS merger | 2.79 M_sun | varies |

### Variable Parameters Across Simulations

- Black hole spin (a): 0 to 1
- Black hole mass (M_BH): 2.31 to 10 solar masses
- Torus inner radius (R_in) and max pressure radius (R_max)
- Initial electron fraction (Y_e)
- Initial entropy (k_B per baryon)
- Magnetic field strength (plasma beta)

---

## Governing Equations

The fluid sector solves conservation laws for:

- **Mass**: `d_t(sqrt(g) rho_0 u^t) + d_i(sqrt(g) rho_0 u^i) = 0`
- **Momentum-energy**: `d_t[sqrt(g)(T^t_nu + rho_0 u^t delta^t_nu)] + d_i[sqrt(g)(T^i_nu + rho_0 u^i delta^t_nu)] = sqrt(g)(T^k_lambda Gamma^k_nu_kappa + G_nu)`
- **Magnetic field**: Constrained transport (divergence-free)
- **Electron fraction (Y_e)**: Advection + neutrino source terms

Radiation transport uses the Monte Carlo radiative transfer equation for frequency-dependent neutrino transport.

### Numerical Methods

- **Fluid**: 2nd-order shock-capturing HARM scheme (Gammie, McKinney & Toth 2003), WENO reconstruction
- **Radiation**: 2nd-order Monte Carlo (GRMONTY; Dolence et al. 2009)
- **Coupling**: Explicit 1st-order operator-split
- **EOS**: Tabulated nuclear EOS (SFHo) assuming nuclear statistical equilibrium

---

## Physical Significance

The electron fraction (Y_e) in disk outflows is the core deliverable: it directly controls:

- Heavy element nucleosynthesis (r-process)
- Electromagnetic counterparts (kilonova)
- Observable signatures of neutron star mergers

---

## Computational Cost

| Property | Value |
|----------|-------|
| Hardware | CPUs (Intel Xeon, ARM ThunderX2), hybrid MPI + OpenMP |
| Nodes per simulation | ~33 |
| Cores per simulation | ~300 |
| Time per simulation | ~3 weeks |
| Precision | Double precision |
| Clusters used | Badger, Capulin (LANL) |

---

## How to Access the Data

### Install

```bash
# Requires Python >= 3.10
pip install the-well huggingface_hub h5py
```

### Load via WellDataset (lazy, streamed)

```python
from the_well.data import WellDataset

ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="post_neutron_star_merger",
    well_split_name="train",       # or "test", "valid"
    n_steps_input=1,               # input timesteps per sample
    n_steps_output=1,              # output timesteps per sample
)

sample = ds[0]
# sample['input_fields']   -> torch.Size([1, 192, 128, 66, 12])
# sample['output_fields']  -> torch.Size([1, 192, 128, 66, 12])
# sample['space_grid']     -> torch.Size([192, 128, 66, 3])
```

The 12 channels in flattened order: 6 scalar fields + 3 velocity components + 3 magnetic field components.

### Use with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
for batch in loader:
    x = batch['input_fields']    # shape: [4, 1, 192, 128, 66, 12]
    y = batch['output_fields']   # shape: [4, 1, 192, 128, 66, 12]
    # pass to model
```

### Use WellDataModule (Lightning-compatible)

```python
from the_well.data import WellDataModule

dm = WellDataModule(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="post_neutron_star_merger",
    batch_size=2,
    n_steps_input=1,
    n_steps_output=1,
    data_workers=4,
)

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
```

### Filter to Specific Simulations

```python
ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="post_neutron_star_merger",
    well_split_name="train",
    include_filters=["scenario_3"],   # only GW170817 fiducial
)
```

### Direct HDF5 Access (without the-well)

```python
import h5py
import fsspec

fs, _ = fsspec.url_to_fs("hf://datasets/polymathic-ai/")
path = "post_neutron_star_merger/train/post_neutron_star_merger_scenario_1.hdf5"

with fs.open(path, "rb") as fp:
    with h5py.File(fp, "r") as f:
        density = f["t0_fields/density"][0, :, :, :, :]    # (181, 192, 128, 66)
        velocity = f["t1_fields/velocity"][0, :, :, :, :]  # (181, 192, 128, 66, 3)
        Ye = f["t0_fields/electron_fraction"][0, :, :, :, :]
        metric = f["additional_information/g_contravariant"][0]  # (192, 128, 4, 4)
        spin = f["scalars/a"][()]
        mass = f["scalars/mbh"][()]
```

---

## WellDataset API Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `well_base_path` | str | — | Base path (e.g. `"hf://datasets/polymathic-ai/"`) |
| `well_dataset_name` | str | — | Dataset name (`"post_neutron_star_merger"`) |
| `well_split_name` | str | — | Split: `"train"`, `"valid"`, or `"test"` |
| `n_steps_input` | int | 1 | Number of input timesteps per sample |
| `n_steps_output` | int | 1 | Number of output timesteps per sample |
| `min_dt_stride` | int | 1 | Minimum temporal stride |
| `max_dt_stride` | int | 1 | Maximum temporal stride (random between min/max) |
| `full_trajectory_mode` | bool | False | Return complete trajectory from t0 |
| `use_normalization` | bool | False | Apply field normalization |
| `normalization_type` | str | — | `"ZScoreNormalization"` or `"RMSNormalization"` |
| `flatten_tensors` | bool | True | Flatten vector/tensor fields to channels |
| `return_grid` | bool | True | Include spatial/temporal grid coordinates |
| `include_filters` | list | None | Only load files containing these strings |
| `exclude_filters` | list | None | Skip files containing these strings |
| `transform` | callable | None | Custom augmentation function |
| `cache_small` | bool | True | Cache small tensors in memory |
| `max_cache_size` | float | 1e9 | Max elements to cache |

### Key Methods

- `ds[i]` — Returns a dict with keys: `input_fields`, `output_fields`, `boundary_conditions`, `space_grid`, `input_time_grid`, `output_time_grid`
- `len(ds)` — Total number of samples
- `ds.to_xarray(backend='dask')` — Export to xarray Dataset (lazy with dask, eager with numpy)

---

## Reproducing the Simulations with nubhlight

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt install g++ build-essential gsl-bin libgsl-dev \
    libopenmpi-dev libhdf5-openmpi-dev python3 python3-pip gfortran

# Python packages
pip install numpy matplotlib h5py
```

### Clone and Setup

```bash
git clone https://github.com/lanl/nubhlight.git
cd nubhlight
```

### Download EOS Table

```bash
cd data
python get_stellar_collapse_eos.py -p SFHo
cd ..
```

This downloads `Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5` from stellarcollapse.org.

Manual fallback URL:
```
https://stellarcollapse.org/~evanoc/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2
```

### Get Neutrino Opacity Tables

Required file: `opacity.SFHo.nohoro.juo.brem1.h5`

Generated via the [modified NuLib fork](https://github.com/evanoconnor/NuLib) compatible with nubhlight. Build NuLib with SFHo EOS and place the output in `data/`.

### Build (3D with neutrinos, matching The Well dataset)

```bash
cd prob/torus_cbc
python build.py -3d -nu -hdf \
    -M 2.8 -a 0.8 -Ye 0.1 -ent 4 \
    -rin 3.7 -rmax 9.268 -beta 100 \
    -tfinal 10000 -nph 10 \
    -n1n2n3tot -n1tot 192 -n2tot 128 -n3tot 66 \
    -n1n2n3cpu -n1cpu 1 -n2cpu 2 -n3cpu 11
```

### Build Flags Reference

| Flag | Description |
|------|-------------|
| `-3d` | Full 3D simulation |
| `-nu` | Enable neutrino radiation transport |
| `-hdf` | Use HDF5 opacity tables |
| `-M <val>` | Black hole mass (solar masses) |
| `-a <val>` | Black hole spin (0 to 1) |
| `-Ye <val>` | Initial electron fraction |
| `-ent <val>` | Initial entropy (k_B/baryon) |
| `-rin <val>` | Torus inner radius (code units) |
| `-rmax <val>` | Radius of max pressure (code units) |
| `-beta <val>` | Plasma beta (gas/magnetic pressure) |
| `-tfinal <val>` | Final simulation time (code units) |
| `-nph <val>` | Monte Carlo photons per cell |
| `-small` | Reduced resolution |
| `-nob` | No magnetic fields |

### Run

```bash
mpirun -np 22 ./bhlight -p param_template.dat -o /path/to/output/
```

Output directories:
- `dumps/` — Simulation snapshots (HDF5), every 5 code time units
- `restarts/` — Checkpoint files, every 100 code time units

### Quick Test (2D, no neutrinos)

```bash
python build.py -small -M 2.8 -a 0.8 -tfinal 500
mpirun -np 4 ./bhlight -p param_template.dat
```

---

## Benchmark Results (from The Well)

| Model | VRMSE |
|-------|-------|
| FNO | 0.3866 |
| **TFNO** | **0.3793** |
| UNet / CNextU-net | N/A (dimensional constraints) |

Lower VRMSE is better; score of 1.0 = predicting the mean value.

---

## References

1. Miller, J. M., Ryan, B. R., Dolence, J. C. 2019, ApJS, 241:2 — primary nubhlight paper
2. Ryan, B. R., Dolence, J. C., & Gammie, C. F. 2015, ApJ, 807:31 — ebhlight foundation
3. Miller, J. M. et al. 2019, PRD, 100:2 — GW170817 application
4. Curtis et al. 2023 — nucleosynthesis analysis
5. The Well: arxiv:2412.00568

---

## Links

- Dataset page: <https://polymathic-ai.org/the_well/datasets/post_neutron_star_merger/>
- The Well API docs: <https://polymathic-ai.org/the_well/api/>
- The Well GitHub: <https://github.com/PolymathicAI/the_well>
- The Well PyPI: <https://pypi.org/project/the-well/>
- nubhlight GitHub: <https://github.com/lanl/nubhlight>
- nubhlight Wiki: <https://github.com/lanl/nubhlight/wiki>
- NuLib (opacity tables): <https://github.com/evanoconnor/NuLib>
- stellarcollapse.org EOS tables: <https://stellarcollapse.org/equationofstate.html>
- HuggingFace collection: <https://huggingface.co/collections/polymathic-ai/the-well-67e129f4ca23e0447395d74c>