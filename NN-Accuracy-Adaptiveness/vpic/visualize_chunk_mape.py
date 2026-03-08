#!/usr/bin/env python3
"""
Per-Chunk MAPE Heatmap across Simulation Timesteps.

Reads sim_chunk_metrics.csv and produces:
  1. Heatmap: nn_baseline MAPE per chunk (x) vs timestep (y)
  2. Heatmap: nn_sgd MAPE per chunk (x) vs timestep (y)
  3. Side-by-side comparison of baseline vs SGD

Usage:
  python3 benchmarks/vpic/visualize_chunk_mape.py [--output-dir DIR]
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


DEFAULT_DIR = "benchmarks/vpic"


def parse_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def build_mape_grid(rows, phase):
    """Return (timesteps, chunks, mape_2d) for a given phase."""
    phase_rows = [r for r in rows if r["phase"] == phase]
    if not phase_rows:
        return None, None, None

    timesteps = sorted(set(int(r["timestep"]) for r in phase_rows))
    chunks = sorted(set(int(r["chunk"]) for r in phase_rows))

    grid = np.full((len(timesteps), len(chunks)), np.nan)
    ts_idx = {t: i for i, t in enumerate(timesteps)}
    ch_idx = {c: i for i, c in enumerate(chunks)}

    for r in phase_rows:
        ts = int(r["timestep"])
        ch = int(r["chunk"])
        pred = r.get("predicted_ratio", 0)
        actual = r.get("actual_ratio", 0)
        if pred > 0 and actual > 0:
            mape = abs(pred - actual) / actual * 100.0
            grid[ts_idx[ts], ch_idx[ch]] = mape

    return timesteps, chunks, grid


def _add_heatmap(ax, grid, timesteps, chunks, title, vmax=None):
    """Draw a single MAPE heatmap on the given axes."""
    if vmax is None:
        vmax = min(np.nanpercentile(grid, 95), 5000)

    norm = mcolors.LogNorm(vmin=1, vmax=vmax, clip=True)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r",
                   norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels([str(c) for c in chunks], fontsize=7)
    ax.set_xlabel("Chunk Index", fontsize=10)

    step = max(1, len(timesteps) // 20)
    ax.set_yticks(range(0, len(timesteps), step))
    ax.set_yticklabels([str(timesteps[i]) for i in range(0, len(timesteps), step)],
                       fontsize=7)
    ax.set_ylabel("Simulation Timestep", fontsize=10)
    ax.set_title(title, fontsize=12)

    return im


# -- Plot 1 & 2: Individual heatmaps ------------------------------------------

def plot_single_heatmap(rows, phase, label, output_dir):
    timesteps, chunks, grid = build_mape_grid(rows, phase)
    if grid is None:
        print(f"  Skipping {phase}: no data")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(chunks) * 0.7), 8))
    im = _add_heatmap(ax, grid, timesteps, chunks,
                      f"Per-Chunk MAPE (%): {label}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MAPE (%)", fontsize=10)

    # Annotate cells if grid is small enough
    if len(timesteps) * len(chunks) <= 600:
        for i in range(len(timesteps)):
            for j in range(len(chunks)):
                val = grid[i, j]
                if not np.isnan(val):
                    txt = f"{val:.0f}" if val < 10000 else f"{val/1000:.0f}K"
                    color = "white" if val > 500 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=5, color=color)

    fig.tight_layout()
    fname = f"chunk_mape_{phase}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 3: Side-by-side comparison ------------------------------------------

def plot_side_by_side(rows, output_dir):
    ts_bl, ch_bl, grid_bl = build_mape_grid(rows, "nn_baseline")
    ts_sgd, ch_sgd, grid_sgd = build_mape_grid(rows, "nn_sgd")

    if grid_bl is None or grid_sgd is None:
        print("  Skipping side-by-side: missing data")
        return

    vmax = min(max(np.nanpercentile(grid_bl, 95),
                   np.nanpercentile(grid_sgd, 95)), 5000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    im1 = _add_heatmap(axes[0], grid_bl, ts_bl, ch_bl,
                       "NN Baseline (no learning)", vmax=vmax)
    fig.colorbar(im1, ax=axes[0], orientation="horizontal",
                 pad=0.08, shrink=0.6, label="MAPE (%)")

    im2 = _add_heatmap(axes[1], grid_sgd, ts_sgd, ch_sgd,
                       "NN + SGD (online learning)", vmax=vmax)
    fig.colorbar(im2, ax=axes[1], orientation="horizontal",
                 pad=0.08, shrink=0.6, label="MAPE (%)")

    fig.suptitle("Per-Chunk Prediction Error: Baseline vs SGD", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "chunk_mape_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary stats
    bl_mean = np.nanmean(grid_bl)
    sgd_mean = np.nanmean(grid_sgd)
    bl_median = np.nanmedian(grid_bl)
    sgd_median = np.nanmedian(grid_sgd)
    print(f"    Baseline  mean={bl_mean:.0f}%  median={bl_median:.0f}%")
    print(f"    SGD       mean={sgd_mean:.0f}%  median={sgd_median:.0f}%")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-Chunk MAPE Heatmap Visualizer")
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    chunk_csv = os.path.join(args.output_dir, "sim_chunk_metrics.csv")
    if not os.path.exists(chunk_csv):
        print(f"ERROR: {chunk_csv} not found")
        sys.exit(1)

    rows = parse_csv(chunk_csv)
    print(f"Loaded {len(rows)} rows from {chunk_csv}")

    print("\nGenerating plots...")
    plot_single_heatmap(rows, "nn_baseline", "NN Baseline (no learning)",
                        args.output_dir)
    plot_single_heatmap(rows, "nn_sgd", "NN + SGD (online learning)",
                        args.output_dir)
    plot_side_by_side(rows, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
