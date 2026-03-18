#!/usr/bin/env python3
"""
Plot LR sweep results for per-output SGD evaluation.

Reads results from benchmarks/grayscott/results_sweep/ and generates:
  1. sMAPE_R convergence over timesteps (all LRs overlaid)
  2. SGD fires over timesteps
  3. Single-snapshot bar chart (nn-rl MAPE by LR)
  4. Comp/decomp MAPE convergence (secondary outputs)

Usage:
    python benchmarks/grayscott/plot_lr_sweep.py [--sweep-dir DIR] [--out FILE]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_timestep_csv(path):
    """Load a timestep CSV into a dict of lists."""
    data = {
        "timestep": [], "sim_step": [], "write_ms": [], "ratio": [],
        "smape_ratio": [], "smape_comp": [], "smape_decomp": [],
        "sgd_fires": [],
    }
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timestep"].append(int(row["timestep"]))
            data["sim_step"].append(int(row["sim_step"]))
            data["write_ms"].append(float(row["write_ms"]))
            data["ratio"].append(float(row["ratio"]))
            data["smape_ratio"].append(float(row["smape_ratio"]))
            data["smape_comp"].append(float(row["smape_comp"]))
            data["smape_decomp"].append(float(row["smape_decomp"]))
            data["sgd_fires"].append(int(row["sgd_fires"]))
    return data


def load_aggregate_csv(path):
    """Load aggregate CSV, return dict keyed by phase name."""
    phases = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            phases[row["phase"]] = row
    return phases


def main():
    parser = argparse.ArgumentParser(description="Plot LR sweep results")
    parser.add_argument("--sweep-dir", default=None,
                        help="Path to results_sweep/ directory")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: sweep_dir/lr_sweep.png)")
    args = parser.parse_args()

    # Find sweep directory
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        script_dir = Path(__file__).parent
        sweep_dir = script_dir / "results_sweep"

    if not sweep_dir.exists():
        print(f"ERROR: Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else sweep_dir / "lr_sweep.png"

    # Discover LR directories
    lr_dirs = sorted([d for d in sweep_dir.glob("lr_*") if d.is_dir()],
                     key=lambda p: float(p.name[3:]))
    if not lr_dirs:
        print(f"ERROR: No lr_* directories found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    # Load data
    lr_data = {}  # lr_value -> timestep data
    lr_agg = {}   # lr_value -> aggregate data
    for d in lr_dirs:
        lr_val = d.name[3:]  # e.g., "0.1"
        ts_path = d / "benchmark_grayscott_timesteps.csv"
        agg_path = d / "benchmark_grayscott_vol.csv"
        if ts_path.exists():
            lr_data[lr_val] = load_timestep_csv(ts_path)
        if agg_path.exists():
            lr_agg[lr_val] = load_aggregate_csv(agg_path)

    # Load baseline
    baseline_dir = sweep_dir / "baseline_nn"
    baseline_mape = None
    if baseline_dir.exists():
        agg_path = baseline_dir / "benchmark_grayscott_vol.csv"
        if agg_path.exists():
            bl = load_aggregate_csv(agg_path)
            if "nn" in bl:
                baseline_mape = float(bl["nn"]["mape_ratio_pct"])

    if not lr_data:
        print("ERROR: No timestep CSVs found", file=sys.stderr)
        sys.exit(1)

    # Color map
    cmap = plt.cm.viridis
    lr_keys = sorted(lr_data.keys(), key=float)
    colors = {lr: cmap(i / max(len(lr_keys) - 1, 1)) for i, lr in enumerate(lr_keys)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Output SGD: Learning Rate Sweep", fontsize=14, fontweight="bold")

    # ── Panel 1: sMAPE_R convergence ──────────────────────────────
    ax1 = axes[0, 0]
    for lr in lr_keys:
        d = lr_data[lr]
        ax1.plot(d["timestep"], d["smape_ratio"], label=f"LR={lr}",
                 color=colors[lr], linewidth=1.5, alpha=0.85)
    if baseline_mape is not None:
        ax1.axhline(y=baseline_mape, color="red", linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Baseline nn ({baseline_mape:.1f}%)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("sMAPE Ratio (%)")
    ax1.set_title("Ratio sMAPE Convergence")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: SGD fires per timestep ───────────────────────────
    ax2 = axes[0, 1]
    for lr in lr_keys:
        d = lr_data[lr]
        ax2.plot(d["timestep"], d["sgd_fires"], label=f"LR={lr}",
                 color=colors[lr], linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("SGD Fires (per 32 chunks)")
    ax2.set_title("SGD Trigger Frequency")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(bottom=0, top=34)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Single-snapshot nn-rl MAPE bar chart ─────────────
    ax3 = axes[1, 0]
    bar_lrs = []
    bar_mape_r = []
    bar_mape_c = []
    bar_mape_d = []
    for lr in lr_keys:
        if lr in lr_agg and "nn-rl" in lr_agg[lr]:
            row = lr_agg[lr]["nn-rl"]
            bar_lrs.append(lr)
            bar_mape_r.append(float(row["mape_ratio_pct"]))
            bar_mape_c.append(float(row["mape_comp_pct"]))
            bar_mape_d.append(float(row["mape_decomp_pct"]))

    x = np.arange(len(bar_lrs))
    width = 0.25
    ax3.bar(x - width, bar_mape_r, width, label="Ratio", color="#2196F3")
    ax3.bar(x, bar_mape_c, width, label="Comp Time", color="#FF9800")
    ax3.bar(x + width, bar_mape_d, width, label="Decomp Time", color="#4CAF50")
    if baseline_mape is not None:
        ax3.axhline(y=baseline_mape, color="red", linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Baseline ratio ({baseline_mape:.1f}%)")
    ax3.set_xlabel("Learning Rate")
    ax3.set_ylabel("MAPE (%)")
    ax3.set_title("Single-Snapshot nn-rl MAPE")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bar_lrs)
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Rolling average sMAPE_R (smoothed view) ──────────
    ax4 = axes[1, 1]
    window = 10
    for lr in lr_keys:
        d = lr_data[lr]
        ts = np.array(d["timestep"])
        sr = np.array(d["smape_ratio"])
        if len(sr) >= window:
            smoothed = np.convolve(sr, np.ones(window) / window, mode="valid")
            ax4.plot(ts[window - 1:], smoothed, label=f"LR={lr}",
                     color=colors[lr], linewidth=1.5, alpha=0.85)
        else:
            ax4.plot(ts, sr, label=f"LR={lr}",
                     color=colors[lr], linewidth=1.5, alpha=0.85)
    if baseline_mape is not None:
        ax4.axhline(y=baseline_mape, color="red", linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Baseline ({baseline_mape:.1f}%)")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("sMAPE Ratio (%, 10-step avg)")
    ax4.set_title(f"Smoothed Ratio sMAPE (window={window})")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
