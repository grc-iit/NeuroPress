#!/usr/bin/env python3
"""Visualize chunk boundaries overlaid on simulation data.

Generates a figure showing the full dataset as a 2D heatmap with
chunk boundaries drawn as a grid overlay. Helps explain how the
data is partitioned for per-chunk NN algorithm selection.

Supports:
  - VPIC Harris sheet (magnetic reconnection) — synthetic generation
  - Gray-Scott reaction-diffusion — synthetic generation
  - HDF5 file input (if available)

Usage:
  python3 visualize_chunks.py --dataset vpic --nx 156 --chunk-mb 16
  python3 visualize_chunks.py --dataset grayscott --L 512 --chunk-mb 16
  python3 visualize_chunks.py --hdf5 /path/to/file.h5 --dataset vpic --nx 156 --chunk-mb 16
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


def generate_vpic_harris_sheet(nx):
    """Generate VPIC Harris current sheet field data.

    VPIC stores 16 field variables per voxel on a (NX+2)^3 grid.
    The dominant field is Bz = B0 * tanh(x / L_sheet).
    Layout: 1D array of [(NX+2)^3 * 16] floats, with fields interleaved
    per-voxel as [ex, ey, ez, div_e_err, bx, by, bz, div_b_err,
                  tcax, tcay, tcaz, rhob, jfx, jfy, jfz, rhof].
    """
    n = nx + 2  # ghost cells
    L_sheet = 0.5  # sheet half-width (normalized)
    B0 = 1.0
    Lx = 1.0

    # Grid coordinates (normalized)
    x = np.linspace(-Lx / 2, Lx / 2, n)
    y = np.linspace(0, Lx * n / n, n)
    z = np.linspace(0, Lx * n / n, n)

    # 16 fields per voxel
    fields = np.zeros((n, n, n, 16), dtype=np.float32)

    # Bz (field index 6) = B0 * tanh(x / L_sheet) — the Harris sheet
    xx = x[:, None, None]
    fields[:, :, :, 6] = B0 * np.tanh(xx / L_sheet)

    # Jfy (field index 13) = current density = -B0 / (L * cosh^2(x/L))
    fields[:, :, :, 13] = -B0 / (L_sheet * np.cosh(xx / L_sheet) ** 2)

    # Add small perturbation to Bx (index 4) — reconnection seed
    yy = y[None, :, None]
    zz = z[None, None, :]
    fields[:, :, :, 4] = 0.01 * B0 * np.sin(2 * np.pi * yy / Lx)

    # Ex, Ey (indices 0, 1) — small noise from particle loading
    rng = np.random.default_rng(42)
    fields[:, :, :, 0] = rng.normal(0, 0.001, (n, n, n)).astype(np.float32)
    fields[:, :, :, 1] = rng.normal(0, 0.001, (n, n, n)).astype(np.float32)

    return fields.reshape(-1), n


def load_grayscott_raw(raw_path, L):
    """Load raw float32 binary dumped from the GPU Gray-Scott simulation."""
    data = np.fromfile(raw_path, dtype=np.float32)
    expected = L * L * L
    if data.size != expected:
        raise ValueError(f"Expected {expected} floats for L={L}, got {data.size}")
    return data.reshape(L, L, L)


def plot_vpic_chunks(nx, chunk_mb, output_path, field_idx=6, field_name="Bz"):
    """Visualize VPIC data with chunk boundaries."""
    flat_data, n = generate_vpic_harris_sheet(nx)
    total_floats = len(flat_data)
    chunk_floats = chunk_mb * 1024 * 1024 // 4
    n_chunks = (total_floats + chunk_floats - 1) // chunk_floats

    # Reshape to (n, n, n, 16) and extract one field
    data_4d = flat_data.reshape(n, n, n, 16)
    # Take middle z-slice of the selected field
    z_mid = n // 2
    slice_2d = data_4d[:, :, z_mid, field_idx]

    # Compute chunk boundaries in the 1D flat layout
    # Each voxel = 16 floats. Chunks are contiguous in the flat array.
    floats_per_zslice = n * n * 16
    floats_per_yrow = n * 16

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # ── Left panel: 2D slice with chunk boundaries ──
    ax = axes[0]
    im = ax.imshow(slice_2d.T, origin='lower', cmap='RdBu_r',
                   aspect='equal', interpolation='bilinear',
                   extent=[0, n, 0, n])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label=field_name)

    # Draw chunk boundaries
    # In flat layout: chunk i covers floats [i*chunk_floats, (i+1)*chunk_floats)
    # Convert to voxel indices: voxel = flat_idx // 16
    # Then to (x, y, z) indices: x = voxel // (n*n), y = (voxel // n) % n, z = voxel % n
    colors_chunks = plt.cm.Set3(np.linspace(0, 1, min(n_chunks, 12)))

    chunk_info = []
    for ci in range(n_chunks):
        start_float = ci * chunk_floats
        end_float = min((ci + 1) * chunk_floats, total_floats)
        start_voxel = start_float // 16
        end_voxel = (end_float - 1) // 16

        # Convert to 3D indices
        sx = start_voxel // (n * n)
        ex = end_voxel // (n * n)

        chunk_info.append({
            'id': ci,
            'start_voxel': start_voxel,
            'end_voxel': end_voxel,
            'x_start': sx,
            'x_end': ex,
            'n_voxels': end_voxel - start_voxel + 1,
            'mb': (end_float - start_float) * 4 / (1024 * 1024),
        })

        # Draw horizontal lines at chunk x-boundaries on the 2D slice
        if sx > 0:
            ax.axhline(y=sx, color=colors_chunks[ci % len(colors_chunks)],
                       linewidth=2, linestyle='--', alpha=0.8)

        # Label chunk in the middle of its x-range
        mid_x = (sx + ex) / 2
        if ex > sx:
            ax.text(n * 0.02, mid_x, f'C{ci}',
                    fontsize=8, fontweight='bold',
                    color='white', backgroundcolor=colors_chunks[ci % len(colors_chunks)],
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', alpha=0.8,
                              facecolor=colors_chunks[ci % len(colors_chunks)]))

    ax.set_xlabel('Y index', fontweight='bold')
    ax.set_ylabel('X index', fontweight='bold')
    ax.set_title(f'VPIC Harris Sheet — {field_name} (z={z_mid})\n'
                 f'Grid: {nx}³ ({n}³ with ghosts), {n_chunks} chunks × {chunk_mb} MB',
                 fontweight='bold')

    # ── Right panel: chunk map showing data range per chunk ──
    ax2 = axes[1]

    # Show compression-relevant statistics per chunk
    chunk_ids = []
    chunk_means = []
    chunk_stds = []
    chunk_ranges = []

    for ci, info in enumerate(chunk_info):
        start_f = ci * chunk_floats
        end_f = min((ci + 1) * chunk_floats, total_floats)
        chunk_data = flat_data[start_f:end_f]

        chunk_ids.append(ci)
        chunk_means.append(np.mean(chunk_data))
        chunk_stds.append(np.std(chunk_data))
        chunk_ranges.append(np.max(chunk_data) - np.min(chunk_data))

    # Horizontal bar chart showing data range per chunk
    bars = ax2.barh(chunk_ids, chunk_ranges,
                    color=[colors_chunks[i % len(colors_chunks)] for i in chunk_ids],
                    edgecolor='black', linewidth=0.5, alpha=0.85)

    # Annotate with std dev
    for ci, (rng_val, std_val, mean_val) in enumerate(
            zip(chunk_ranges, chunk_stds, chunk_means)):
        ax2.text(rng_val + max(chunk_ranges) * 0.02, ci,
                 f'σ={std_val:.3f}  μ={mean_val:.3f}',
                 va='center', fontsize=7, color='#333')

    ax2.set_xlabel('Data Range (max − min)', fontweight='bold')
    ax2.set_ylabel('Chunk Index', fontweight='bold')
    ax2.set_title('Per-Chunk Data Statistics\n(affects compressibility)',
                  fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.2)

    fig.suptitle('VPIC Simulation Data: Chunk Partitioning & Data Characteristics',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def plot_grayscott_chunks(L, chunk_mb, output_path, steps=10000, raw_path=None):
    """Visualize Gray-Scott data with chunk boundaries.

    Uses real GPU simulation data dumped as raw float32 binary.
    Generate raw data with:
      nvcc -o /tmp/gs_dump /tmp/gs_dump.cu -I include -L build -lgpucompress
      LD_LIBRARY_PATH=build /tmp/gs_dump <L> <steps>
    """
    if raw_path is None:
        raw_path = f'/tmp/gs_v_bench_L{L}_{steps // 1000}k.raw'
    if not os.path.exists(raw_path):
        # Try alternate naming
        raw_path = f'/tmp/gs_v_L{L}_s{steps}.raw'
    if not os.path.exists(raw_path):
        print(f'ERROR: Raw data not found. Generate with GPU solver first.')
        print(f'  Tried: /tmp/gs_v_bench_L{L}_{steps // 1000}k.raw')
        return

    print(f'Loading Gray-Scott V field from {raw_path}...')
    v_3d = load_grayscott_raw(raw_path, L)
    v_flat = v_3d.flatten()
    total_floats = L * L * L
    chunk_floats = chunk_mb * 1024 * 1024 // 4
    n_chunks = (total_floats + chunk_floats - 1) // chunk_floats

    # x is fastest-varying in CUDA kernel: i = x + y*L + z*L*L
    # reshape(L,L,L) gives (z,y,x), so midplane z = L//2 is [L//2,:,:]
    z_mid = L // 2
    v_slice = v_3d[z_mid, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    im = ax.imshow(v_slice.T, origin='lower', cmap='viridis',
                   aspect='equal', interpolation='bilinear',
                   extent=[0, L, 0, L])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='v concentration')

    # Chunk boundaries: flat layout i = x + y*L + z*L*L
    # chunk_floats voxels per chunk; boundary at z = start_voxel // (L*L)
    colors_chunks = plt.cm.Set3(np.linspace(0, 1, min(n_chunks, 12)))

    for ci in range(n_chunks):
        start_voxel = ci * chunk_floats
        end_voxel = min((ci + 1) * chunk_floats, total_floats) - 1
        sz = start_voxel // (L * L)
        ez = end_voxel // (L * L)

        # Draw boundary line at z-slab edges (shown as horizontal on the y-x slice)
        if sz > 0:
            ax.axhline(y=sz, color=colors_chunks[ci % len(colors_chunks)],
                       linewidth=2, linestyle='--', alpha=0.8)

        mid_z = (sz + ez) / 2
        if ez > sz:
            ax.text(L * 0.02, mid_z, f'C{ci}',
                    fontsize=8, fontweight='bold', color='white',
                    backgroundcolor=colors_chunks[ci % len(colors_chunks)],
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', alpha=0.8,
                              facecolor=colors_chunks[ci % len(colors_chunks)]))

    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_title(f'Gray-Scott — V field (z={z_mid})\n'
                 f'Grid: {L}³, {n_chunks} chunks × {chunk_mb} MB\n'
                 f'F=0.04, k=0.06075, Du=0.05, Dv=0.1, dt=0.2, {steps} steps',
                 fontweight='bold', fontsize=10)

    # Right panel: per-chunk statistics from the full 3D data
    ax2 = axes[1]
    chunk_ids = []
    chunk_stds = []
    chunk_ranges = []
    chunk_means = []

    for ci in range(n_chunks):
        start = ci * chunk_floats
        end = min((ci + 1) * chunk_floats, total_floats)
        chunk_data = v_flat[start:end]
        chunk_ids.append(ci)
        chunk_stds.append(np.std(chunk_data))
        chunk_ranges.append(np.max(chunk_data) - np.min(chunk_data))
        chunk_means.append(np.mean(chunk_data))

    bars = ax2.barh(chunk_ids, chunk_stds,
                    color=[colors_chunks[i % len(colors_chunks)] for i in chunk_ids],
                    edgecolor='black', linewidth=0.5, alpha=0.85)

    for ci, (std_val, mean_val) in enumerate(zip(chunk_stds, chunk_means)):
        ax2.text(std_val + max(chunk_stds) * 0.02, ci,
                 f'σ={std_val:.4f}  μ={mean_val:.4f}',
                 va='center', fontsize=7, color='#333')

    ax2.set_xlabel('Standard Deviation', fontweight='bold')
    ax2.set_ylabel('Chunk Index', fontweight='bold')
    ax2.set_title('Per-Chunk Data Variability\n(higher σ → harder to compress)',
                  fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.2)

    fig.suptitle('Gray-Scott Simulation: Chunk Partitioning & Data Characteristics',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize chunk boundaries on simulation data')
    parser.add_argument('--dataset', choices=['vpic', 'grayscott'], default='vpic')
    parser.add_argument('--nx', type=int, default=156, help='VPIC grid size (NX)')
    parser.add_argument('--L', type=int, default=400, help='Gray-Scott grid size (default: 400)')
    parser.add_argument('--chunk-mb', type=int, default=16, help='Chunk size in MB')
    parser.add_argument('--steps', type=int, default=10000, help='GS simulation steps (default: 10000)')
    parser.add_argument('--raw', type=str, default=None, help='Path to raw float32 binary (GS V field)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--field', type=int, default=6, help='VPIC field index (default: 6=Bz)')
    parser.add_argument('--field-name', type=str, default='Bz', help='Field name for label')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'chunk_visualization_{args.dataset}.png'

    if args.dataset == 'vpic':
        plot_vpic_chunks(args.nx, args.chunk_mb, args.output,
                         field_idx=args.field, field_name=args.field_name)
    else:
        plot_grayscott_chunks(args.L, args.chunk_mb, args.output,
                              steps=args.steps, raw_path=args.raw)


if __name__ == '__main__':
    main()
