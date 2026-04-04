#!/usr/bin/env python3
"""
3D bar charts for exploration threshold sweep evaluation.

Reads benchmark CSVs from the threshold sweep output directory and
produces 4 charts:
  1. Average Regret vs X1 vs X2-X1
  2. Average MAPE vs X1 vs X2-X1
  3. Write Bandwidth vs X1 vs X2-X1
  4. Read Bandwidth vs X1 vs X2-X1

Usage:
    python3 benchmarks/plots/plot_threshold_sweep.py [SWEEP_DIR]

    SWEEP_DIR defaults to benchmarks/sdrbench/results/threshold_sweep/
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm

# ── Threshold values (must match eval_exploration_threshold.sh) ──
X1_VALUES = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00]
DELTA_VALUES = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00]

# Display labels (percentages)
X1_LABELS = ["5%", "10%", "20%", "30%", "50%", "100%", "1000%"]
DELTA_LABELS = ["5%", "10%", "20%", "30%", "50%", "100%", "1000%"]

DATASET_NAME = "hurricane_isabel"


def read_summary_csv(path):
    """Read summary CSV and return the nn-rl+exp50 row as a dict."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "").strip()
            if "exp" in phase or "nn-rl" in phase:
                return row
        # If no nn-rl phase found, return last row
        f.seek(0)
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None


def read_ranking_csv(path):
    """Read ranking CSV and return mean top1_regret."""
    if not os.path.isfile(path):
        return None
    regrets = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = float(row["top1_regret"])
                if np.isfinite(r) and r > 0:
                    regrets.append(r)
            except (KeyError, ValueError):
                continue
    return np.mean(regrets) if regrets else None


def collect_data(sweep_dir):
    """Collect metrics for all (X1, delta) pairs."""
    n_x1 = len(X1_VALUES)
    n_delta = len(DELTA_VALUES)

    regret = np.full((n_x1, n_delta), np.nan)
    mape = np.full((n_x1, n_delta), np.nan)
    write_bw = np.full((n_x1, n_delta), np.nan)
    read_bw = np.full((n_x1, n_delta), np.nan)

    found = 0
    for i, x1 in enumerate(X1_VALUES):
        for j, delta in enumerate(DELTA_VALUES):
            run_dir = os.path.join(sweep_dir, f"x1_{x1:.2f}_delta_{delta:.2f}")
            summary_path = os.path.join(run_dir, f"benchmark_{DATASET_NAME}.csv")
            ranking_path = os.path.join(run_dir, f"benchmark_{DATASET_NAME}_ranking.csv")

            row = read_summary_csv(summary_path)
            if row:
                found += 1
                # MAPE: average of ratio, comp, decomp
                try:
                    mape_r = float(row.get("mape_ratio_pct", 0))
                    mape_c = float(row.get("mape_comp_pct", 0))
                    mape_d = float(row.get("mape_decomp_pct", 0))
                    mape[i, j] = (mape_r + mape_c + mape_d) / 3.0
                except (ValueError, TypeError):
                    pass

                # Bandwidth
                try:
                    write_bw[i, j] = float(row.get("write_mibps", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    read_bw[i, j] = float(row.get("read_mibps", 0))
                except (ValueError, TypeError):
                    pass

            # Regret from ranking CSV
            r = read_ranking_csv(ranking_path)
            if r is not None:
                regret[i, j] = r

    print(f"Collected data from {found}/{n_x1 * n_delta} runs")
    return regret, mape, write_bw, read_bw


def plot_3d_bars(data, title, zlabel, filename, cmap_name="viridis",
                 invert_cmap=False):
    """Create a 3D bar chart for a 2D grid of values."""
    n_x1, n_delta = data.shape

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Bar positions
    xpos, ypos = np.meshgrid(np.arange(n_x1), np.arange(n_delta), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 0.6
    dz = data.flatten()

    # Handle NaN: replace with 0 for plotting, mark as missing
    valid = np.isfinite(dz)
    dz_plot = np.where(valid, dz, 0)

    # Color by value
    cmap = plt.colormaps.get_cmap(cmap_name)
    valid_vals = dz[valid]
    if len(valid_vals) > 0:
        vmin, vmax = valid_vals.min(), valid_vals.max()
        if vmax == vmin:
            vmax = vmin + 1
        norm = plt.Normalize(vmin, vmax)
        if invert_cmap:
            colors = [cmap(1.0 - norm(v)) if f else (0.8, 0.8, 0.8, 0.3)
                      for v, f in zip(dz, valid)]
        else:
            colors = [cmap(norm(v)) if f else (0.8, 0.8, 0.8, 0.3)
                      for v, f in zip(dz, valid)]
    else:
        colors = [(0.8, 0.8, 0.8, 0.3)] * len(dz)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz_plot, color=colors, alpha=0.85,
             edgecolor="grey", linewidth=0.3)

    ax.set_xticks(np.arange(n_x1) + dx / 2)
    ax.set_xticklabels(X1_LABELS, fontsize=8, rotation=15)
    ax.set_yticks(np.arange(n_delta) + dy / 2)
    ax.set_yticklabels(DELTA_LABELS, fontsize=8, rotation=-15)

    ax.set_xlabel("X1 (SGD MAPE threshold)", fontsize=10, labelpad=10)
    ax.set_ylabel("X2 - X1 (exploration delta)", fontsize=10, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=13, pad=15)

    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def main():
    if len(sys.argv) > 1:
        sweep_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_dir = os.path.join(script_dir, "results", "threshold_sweep")

    if not os.path.isdir(sweep_dir):
        print(f"ERROR: sweep directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Reading results from: {sweep_dir}")
    regret, mape, write_bw, read_bw = collect_data(sweep_dir)

    out_dir = sweep_dir
    print(f"\nGenerating plots...")

    plot_3d_bars(
        regret,
        "Average Selection Regret vs Thresholds",
        "Regret (1.0 = optimal)",
        os.path.join(out_dir, "threshold_sweep_regret.png"),
        cmap_name="RdYlGn_r",
    )

    plot_3d_bars(
        mape,
        "Average MAPE vs Thresholds",
        "MAPE (%)",
        os.path.join(out_dir, "threshold_sweep_mape.png"),
        cmap_name="RdYlGn_r",
    )

    plot_3d_bars(
        write_bw,
        "Write Bandwidth vs Thresholds",
        "Write BW (MiB/s)",
        os.path.join(out_dir, "threshold_sweep_write_bw.png"),
        cmap_name="viridis",
    )

    plot_3d_bars(
        read_bw,
        "Read Bandwidth vs Thresholds",
        "Read BW (MiB/s)",
        os.path.join(out_dir, "threshold_sweep_read_bw.png"),
        cmap_name="viridis",
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
