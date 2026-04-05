# Delta Run VPIC Parameters

## Grid & Particles

| Parameter | Value | Description |
|-----------|-------|-------------|
| NX | 320 | Grid size (320x320x320) |
| nppc | 10 | Particles per cell |
| Data/rank | ~139 MiB | Per-rank per-timestep write size |

## Physics

| Parameter | Value | Description |
|-----------|-------|-------------|
| mi_me | 5 | Ion-to-electron mass ratio |
| wpe_wce | 1 | Plasma-to-cyclotron frequency ratio |
| Ti_Te | 5 | Ion-to-electron temperature ratio |
| pert_amp | 0.05 | Perturbation amplitude (fraction of B0) |
| guide_field | 0.0 | Guide field strength |
| damp | 0.001 | Damping coefficient |

## Numerical Stability

| Parameter | Value | Description |
|-----------|-------|-------------|
| cfl_req | 0.70 | CFL number (controls timestep size) |
| wpedt_max | 0.20 | Plasma frequency x dt constraint |
| clean_div_e_interval | 200 | Divergence cleaning for E field (steps) |
| clean_div_b_interval | 200 | Divergence cleaning for B field (steps) |

## Multi-Node / MPI

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_comm_round | 6 | Particle boundary exchange rounds (if nproc > 1) |
| max_nm | max_np / 5 (20%) | Mover buffer size (if nproc > 4) |

## Simulation Control

| Parameter | Value | Description |
|-----------|-------|-------------|
| warmup_steps | 500 | Physics steps before first benchmark write |
| sim_interval | 190 | Physics steps between benchmark writes |

## Benchmark Configuration (4n4g)

| Parameter | Value |
|-----------|-------|
| Nodes | 4 |
| GPUs/node | 4 |
| Total ranks | 16 |
| Chunk size | 32 MB |
| Timesteps | 50 |
| Phases | all 12 (no-comp, lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp, nn, nn-rl, nn-rl+exp50) |
| Policies | balanced, ratio |
| Lossy error bound | 0 (lossless) and 0.01 (lossy) |
