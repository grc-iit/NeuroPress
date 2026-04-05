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

DATASET_NAME = None  # auto-detected from first run directory


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


def detect_dataset_name(sweep_dir):
    """Auto-detect dataset name from the first run directory."""
    import glob as _glob
    for x1 in X1_VALUES:
        for delta in DELTA_VALUES:
            run_dir = os.path.join(sweep_dir, f"x1_{x1:.2f}_delta_{delta:.2f}")
            csvs = _glob.glob(os.path.join(run_dir, "benchmark_*.csv"))
            for c in csvs:
                bn = os.path.basename(c)
                if "_ranking" not in bn and "_timestep" not in bn and "_chunks" not in bn:
                    name = bn.replace("benchmark_", "").replace(".csv", "")
                    return name
    return "hurricane_isabel"  # fallback


def collect_data(sweep_dir):
    """Collect metrics for all (X1, delta) pairs."""
    global DATASET_NAME
    if DATASET_NAME is None:
        DATASET_NAME = detect_dataset_name(sweep_dir)
        print(f"  Auto-detected dataset: {DATASET_NAME}")

    n_x1 = len(X1_VALUES)
    n_delta = len(DELTA_VALUES)

    regret = np.full((n_x1, n_delta), np.nan)
    mape_ratio = np.full((n_x1, n_delta), np.nan)
    mape_comp = np.full((n_x1, n_delta), np.nan)
    mape_decomp = np.full((n_x1, n_delta), np.nan)
    mape_psnr = np.full((n_x1, n_delta), np.nan)
    write_bw = np.full((n_x1, n_delta), np.nan)
    read_bw = np.full((n_x1, n_delta), np.nan)
    explorations = np.full((n_x1, n_delta), np.nan)
    sgd_fires = np.full((n_x1, n_delta), np.nan)

    found = 0
    for i, x1 in enumerate(X1_VALUES):
        for j, delta in enumerate(DELTA_VALUES):
            run_dir = os.path.join(sweep_dir, f"x1_{x1:.2f}_delta_{delta:.2f}")
            summary_path = os.path.join(run_dir, f"benchmark_{DATASET_NAME}.csv")
            ranking_path = os.path.join(run_dir, f"benchmark_{DATASET_NAME}_ranking.csv")

            row = read_summary_csv(summary_path)
            if row:
                found += 1
                # Per-statistic MAPE
                try:
                    mape_ratio[i, j] = float(row.get("mape_ratio_pct", 0))
                    mape_comp[i, j] = float(row.get("mape_comp_pct", 0))
                    mape_decomp[i, j] = float(row.get("mape_decomp_pct", 0))
                    mape_psnr[i, j] = float(row.get("mape_psnr_pct", 0))
                except (ValueError, TypeError):
                    pass

                # End-to-end write/read bandwidth (write_ms already includes
                # exploration and SGD — it's the wall-clock H5Dwrite→H5Fclose time)
                try:
                    write_bw[i, j] = float(row.get("write_mibps", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    read_bw[i, j] = float(row.get("read_mibps", 0))
                except (ValueError, TypeError):
                    pass

                # Exploration and SGD counts — sum from per-timestep CSV for totals
                # (aggregate CSV reports averages which hide early-timestep activity)
                ts_path = os.path.join(run_dir, f"benchmark_{DATASET_NAME}_timesteps.csv")
                if os.path.isfile(ts_path):
                    tot_expl, tot_sgd = 0, 0
                    with open(ts_path, "r") as tf:
                        for tr in csv.DictReader(tf):
                            tot_expl += int(tr.get("explorations", 0))
                            tot_sgd += int(tr.get("sgd_fires", 0))
                    explorations[i, j] = tot_expl
                    sgd_fires[i, j] = tot_sgd
                else:
                    try:
                        explorations[i, j] = float(row.get("explorations", 0))
                    except (ValueError, TypeError):
                        pass
                    try:
                        sgd_fires[i, j] = float(row.get("sgd_fires", 0))
                    except (ValueError, TypeError):
                        pass

            # Regret from ranking CSV
            r = read_ranking_csv(ranking_path)
            if r is not None:
                regret[i, j] = r

    print(f"Collected data from {found}/{n_x1 * n_delta} runs")
    return regret, mape_ratio, mape_comp, mape_decomp, mape_psnr, write_bw, read_bw, explorations, sgd_fires


def plot_heatmap(data, cbar_label, filename, cmap_name="viridis",
                 fmt="%.2f", vmin=None, vmax=None):
    """Create a 2D heatmap with annotated cell values."""
    n_x1, n_delta = data.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    valid = np.isfinite(data)
    plot_data = np.where(valid, data, np.nan)

    if vmin is None:
        vmin = np.nanmin(plot_data)
    if vmax is None:
        vmax = np.nanmax(plot_data)

    im = ax.imshow(plot_data, cmap=cmap_name, aspect="equal",
                   vmin=vmin, vmax=vmax, origin="lower")

    # Annotate each cell
    for i in range(n_x1):
        for j in range(n_delta):
            if valid[i, j]:
                v = data[i, j]
                # Pick text color for contrast
                mid = (vmin + vmax) / 2
                color = "white" if v > mid else "black"
                ax.text(j, i, fmt % v, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(np.arange(n_delta))
    ax.set_xticklabels(DELTA_LABELS, fontsize=9)
    ax.set_yticks(np.arange(n_x1))
    ax.set_yticklabels(X1_LABELS, fontsize=9)

    ax.set_xlabel("X2 - X1 (exploration delta)", fontsize=11)
    ax.set_ylabel("X1 (SGD MAPE threshold)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)

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
    regret, mape_ratio, mape_comp, mape_decomp, mape_psnr, write_bw, read_bw, explorations, sgd_fires = collect_data(sweep_dir)

    out_dir = sweep_dir
    print(f"\nGenerating plots...")

    plot_heatmap(
        regret,
        "Regret (1.0x = optimal)",
        os.path.join(out_dir, "threshold_sweep_regret.png"),
        cmap_name="RdYlGn_r",
        fmt="%.4f",
    )

    plot_heatmap(
        mape_ratio,
        "Ratio MAPE (%)",
        os.path.join(out_dir, "threshold_sweep_mape_ratio.png"),
        cmap_name="RdYlGn_r",
    )

    plot_heatmap(
        mape_comp,
        "Comp Time MAPE (%)",
        os.path.join(out_dir, "threshold_sweep_mape_comp.png"),
        cmap_name="RdYlGn_r",
    )

    plot_heatmap(
        mape_decomp,
        "Decomp Time MAPE (%)",
        os.path.join(out_dir, "threshold_sweep_mape_decomp.png"),
        cmap_name="RdYlGn_r",
    )

    plot_heatmap(
        mape_psnr,
        "PSNR MAPE (%)",
        os.path.join(out_dir, "threshold_sweep_mape_psnr.png"),
        cmap_name="RdYlGn_r",
    )

    plot_heatmap(
        write_bw,
        "Write BW (MiB/s)",
        os.path.join(out_dir, "threshold_sweep_write_bw.png"),
        cmap_name="viridis",
        fmt="%.0f",
    )

    plot_heatmap(
        read_bw,
        "Read BW (MiB/s)",
        os.path.join(out_dir, "threshold_sweep_read_bw.png"),
        cmap_name="viridis",
        fmt="%.0f",
    )

    plot_heatmap(
        explorations,
        "Exploration Count",
        os.path.join(out_dir, "threshold_sweep_explorations.png"),
        cmap_name="YlOrRd",
        fmt="%.0f",
    )

    # Total SGD samples = Phase 1 (1 sample per fire) + Phase 2 (K samples per exploration)
    # K=4 (EXPLORE_K default in the sweep script)
    EXPLORE_K = 4
    total_sgd_samples = np.where(np.isfinite(sgd_fires) & np.isfinite(explorations),
                                 sgd_fires + explorations * EXPLORE_K, np.nan)
    plot_heatmap(
        total_sgd_samples,
        "Total SGD Training Samples",
        os.path.join(out_dir, "threshold_sweep_total_sgd.png"),
        cmap_name="YlOrRd",
        fmt="%.0f",
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
