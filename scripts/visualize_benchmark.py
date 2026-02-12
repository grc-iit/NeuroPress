#!/usr/bin/env python3
"""
Benchmark Results Visualizer

Creates one figure per algorithm showing compression ratio, timing,
throughput, and quality metrics across shuffle/quantization configs.

Usage:
    python3 scripts/visualize_benchmark.py benchmark_results.csv
    python3 scripts/visualize_benchmark.py benchmark_results.csv --output-dir plots/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANT_ORDER = ["none", "linear\neb=0.1", "linear\neb=0.01", "linear\neb=0.001"]
SHUFFLE_LABELS = {0: "No Shuffle", 4: "Shuffle(4)"}
COLORS = {"No Shuffle": "#2196F3", "Shuffle(4)": "#FF9800"}


def make_config_label(row):
    q = "none" if row["quantization"] == "none" else f"linear\neb={row['error_bound']}"
    return q


def plot_algorithm(df_algo, algo_name, file_label, output_dir):
    """Create a 2x2 figure for one algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{algo_name.upper()}  —  {file_label}", fontsize=15, fontweight="bold")

    df_algo = df_algo.copy()
    df_algo["config_label"] = df_algo.apply(make_config_label, axis=1)
    df_algo["shuffle_label"] = df_algo["shuffle"].map(SHUFFLE_LABELS)

    x_labels = QUANT_ORDER
    x = np.arange(len(x_labels))
    width = 0.35

    # --- 1. Compression Ratio ---
    ax = axes[0, 0]
    for i, (shuf_label, color) in enumerate(COLORS.items()):
        subset = df_algo[df_algo["shuffle_label"] == shuf_label].sort_values("error_bound")
        vals = []
        for ql in x_labels:
            match = subset[subset["config_label"] == ql]
            vals.append(match["compression_ratio"].values[0] if len(match) > 0 else 0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=shuf_label, color=color)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # --- 2. Compression & Decompression Time ---
    ax = axes[0, 1]
    for i, (shuf_label, color) in enumerate(COLORS.items()):
        subset = df_algo[df_algo["shuffle_label"] == shuf_label].sort_values("error_bound")
        comp_vals, decomp_vals = [], []
        for ql in x_labels:
            match = subset[subset["config_label"] == ql]
            comp_vals.append(match["compression_time_ms"].values[0] if len(match) > 0 else 0)
            decomp_vals.append(match["decompression_time_ms"].values[0] if len(match) > 0 else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, comp_vals, width * 0.48, label=f"{shuf_label} Compress",
               color=color, alpha=0.9)
        ax.bar(x + offset + width * 0.48, decomp_vals, width * 0.48,
               label=f"{shuf_label} Decompress", color=color, alpha=0.5,
               hatch="//")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compression / Decompression Time")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # --- 3. Throughput ---
    ax = axes[1, 0]
    for i, (shuf_label, color) in enumerate(COLORS.items()):
        subset = df_algo[df_algo["shuffle_label"] == shuf_label].sort_values("error_bound")
        comp_tp, decomp_tp = [], []
        for ql in x_labels:
            match = subset[subset["config_label"] == ql]
            comp_tp.append(match["compression_throughput_mbps"].values[0] if len(match) > 0 else 0)
            decomp_tp.append(match["decompression_throughput_mbps"].values[0] if len(match) > 0 else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, comp_tp, width * 0.48, label=f"{shuf_label} Compress",
               color=color, alpha=0.9)
        ax.bar(x + offset + width * 0.48, decomp_tp, width * 0.48,
               label=f"{shuf_label} Decompress", color=color, alpha=0.5,
               hatch="//")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # --- 4. Quality: PSNR / Max Error ---
    ax = axes[1, 1]
    # Only lossy rows have meaningful PSNR
    lossy = df_algo[df_algo["quantization"] == "linear"].copy()
    if lossy.empty:
        ax.text(0.5, 0.5, "No lossy configs", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title("Quality (Lossy Only)")
    else:
        lossy["psnr_val"] = pd.to_numeric(lossy["psnr_db"], errors="coerce")
        lossy_labels = [ql for ql in x_labels if ql != "none"]
        x2 = np.arange(len(lossy_labels))

        ax2 = ax.twinx()
        for i, (shuf_label, color) in enumerate(COLORS.items()):
            subset = lossy[lossy["shuffle_label"] == shuf_label].sort_values("error_bound")
            psnr_vals, maxerr_vals = [], []
            for ql in lossy_labels:
                match = subset[subset["config_label"] == ql]
                psnr_vals.append(match["psnr_val"].values[0] if len(match) > 0 else 0)
                maxerr_vals.append(
                    pd.to_numeric(match["max_error"], errors="coerce").values[0]
                    if len(match) > 0 else 0)
            offset = (i - 0.5) * width
            ax.bar(x2 + offset, psnr_vals, width, label=f"{shuf_label} PSNR",
                   color=color, alpha=0.8)
            ax2.plot(x2 + offset + width / 2, maxerr_vals, "D",
                     color=color, markersize=7, markeredgecolor="black",
                     label=f"{shuf_label} MaxErr")

        ax.set_ylabel("PSNR (dB)")
        ax2.set_ylabel("Max Error")
        ax.set_title("Quality (Lossy Only)")
        ax.set_xticks(x2)
        ax.set_xticklabels(lossy_labels, fontsize=8)
        ax.set_ylim(bottom=0)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    safe_name = file_label.replace("/", "_").replace(".", "_")
    out_path = os.path.join(output_dir, f"{safe_name}_{algo_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize GPUCompress benchmark results")
    parser.add_argument("csv", help="Path to benchmark_results.csv")
    parser.add_argument("--output-dir", "-o", default="plots",
                        help="Output directory for figures (default: plots/)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.output_dir, exist_ok=True)

    files = df["file"].unique()
    algorithms = df["algorithm"].unique()

    print(f"Visualizing {len(files)} file(s) x {len(algorithms)} algorithms")
    print(f"Output: {args.output_dir}/\n")

    for fname in files:
        df_file = df[df["file"] == fname]
        for algo in algorithms:
            df_algo = df_file[df_file["algorithm"] == algo]
            if df_algo.empty:
                continue
            path = plot_algorithm(df_algo, algo, fname, args.output_dir)
            print(f"  {path}")

    print(f"\nDone. {len(files) * len(algorithms)} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
