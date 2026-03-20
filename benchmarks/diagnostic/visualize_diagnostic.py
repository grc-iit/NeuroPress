#!/usr/bin/env python3
"""
Visualizer for NN diagnostic benchmark results.

Reads nn_chunks.csv and brute_force.csv, produces 4 plots:
  1. Feature radar: per-chunk entropy/MAD/deriv on a radar chart
  2. NN vs brute-force ratio comparison bar chart
  3. Brute-force heatmap: ratio for every (chunk × config)
  4. NN pick overlay: where the NN pick ranks among all configs

Usage:
  python3 visualize_diagnostic.py --dir benchmarks/diagnostic/results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm


def load_data(result_dir):
    chunks_csv = os.path.join(result_dir, "nn_chunks.csv")
    bf_csv = os.path.join(result_dir, "brute_force.csv")

    if not os.path.exists(chunks_csv):
        print(f"ERROR: {chunks_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(bf_csv):
        print(f"ERROR: {bf_csv} not found", file=sys.stderr)
        sys.exit(1)

    chunks = pd.read_csv(chunks_csv)
    bf = pd.read_csv(bf_csv)
    return chunks, bf


def plot_feature_scatter(chunks, out_dir):
    """3D scatter of (entropy, MAD, deriv) per chunk, colored by NN config."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    configs = chunks["nn_config"].unique()
    cmap = plt.cm.Set1
    color_map = {c: cmap(i / max(len(configs) - 1, 1)) for i, c in enumerate(configs)}

    for _, row in chunks.iterrows():
        c = color_map.get(row["nn_config"], "gray")
        ax.scatter(row["entropy"], row["mad"], row["deriv"],
                   c=[c], s=200, edgecolors="black", linewidth=0.8, zorder=5)
        ax.text(row["entropy"], row["mad"], row["deriv"],
                f"  {row['chunk']}: {row['pattern']}", fontsize=7)

    ax.set_xlabel("Entropy (bits)", fontsize=10)
    ax.set_ylabel("MAD (normalized)", fontsize=10)
    ax.set_zlabel("2nd Deriv (normalized)", fontsize=10)
    ax.set_title("Per-Chunk Feature Space\n(color = NN-selected config)", fontsize=12)

    patches = [mpatches.Patch(color=color_map[c], label=c) for c in configs]
    ax.legend(handles=patches, loc="upper left", fontsize=8)

    path = os.path.join(out_dir, "feature_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_nn_vs_bruteforce(chunks, bf, out_dir):
    """Side-by-side bars: NN predicted ratio vs brute-force best ratio per chunk."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n = len(chunks)
    x = np.arange(n)
    w = 0.35

    # NN predicted ratio
    nn_ratios = chunks["predicted_ratio"].values
    # NN actual ratio (from VOL write)
    nn_actual = chunks["actual_ratio"].values

    # Brute-force best ratio per chunk
    bf_best = bf.loc[bf.groupby("chunk")["ratio"].idxmax()]
    bf_ratios = []
    bf_configs = []
    for c in range(n):
        row = bf_best[bf_best["chunk"] == c]
        if len(row) > 0:
            bf_ratios.append(row.iloc[0]["ratio"])
            bf_configs.append(row.iloc[0]["config"])
        else:
            bf_ratios.append(1.0)
            bf_configs.append("?")
    bf_ratios = np.array(bf_ratios)

    bars_nn = ax.bar(x - w/2, nn_actual, w, label="NN pick (actual ratio)",
                     color="steelblue", edgecolor="black", linewidth=0.5)
    bars_bf = ax.bar(x + w/2, bf_ratios, w, label="Brute-force best",
                     color="coral", edgecolor="black", linewidth=0.5)

    # Annotate NN config on NN bars
    for i, row in chunks.iterrows():
        ax.text(i - w/2, nn_actual[i] + 0.05, row["nn_config"],
                ha="center", va="bottom", fontsize=6, rotation=45, color="steelblue")
    # Annotate BF config on BF bars
    for i, cfg in enumerate(bf_configs):
        ax.text(i + w/2, bf_ratios[i] + 0.05, cfg,
                ha="center", va="bottom", fontsize=6, rotation=45, color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['chunk']}: {r['pattern']}" for _, r in chunks.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Compression Ratio", fontsize=11)
    ax.set_title("NN Pick vs Brute-Force Best per Chunk", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "nn_vs_bruteforce.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bruteforce_heatmap(chunks, bf, out_dir):
    """Heatmap: ratio for every (chunk × config), with NN pick highlighted."""
    # Pivot: rows=chunk, cols=config
    configs_sorted = sorted(bf["config"].unique())
    n_chunks = bf["chunk"].nunique()

    matrix = np.full((n_chunks, len(configs_sorted)), np.nan)
    config_idx = {c: i for i, c in enumerate(configs_sorted)}

    for _, row in bf.iterrows():
        ci = int(row["chunk"])
        if row["config"] in config_idx:
            matrix[ci, config_idx[row["config"]]] = row["ratio"]

    fig, ax = plt.subplots(figsize=(16, 6))
    norm = Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", norm=norm)

    # Mark NN pick with a black border
    for _, row in chunks.iterrows():
        ci = int(row["chunk"])
        cfg = row["nn_config"]
        if cfg in config_idx:
            cj = config_idx[cfg]
            rect = plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1,
                                 linewidth=3, edgecolor="blue", facecolor="none")
            ax.add_patch(rect)

    # Mark brute-force best with green border
    bf_best_idx = bf.loc[bf.groupby("chunk")["ratio"].idxmax()]
    for _, row in bf_best_idx.iterrows():
        ci = int(row["chunk"])
        cfg = row["config"]
        if cfg in config_idx:
            cj = config_idx[cfg]
            rect = plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1,
                                 linewidth=2, edgecolor="green", facecolor="none",
                                 linestyle="--")
            ax.add_patch(rect)

    ax.set_xticks(range(len(configs_sorted)))
    ax.set_xticklabels(configs_sorted, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_chunks))
    patterns = [chunks[chunks["chunk"] == c].iloc[0]["pattern"]
                if len(chunks[chunks["chunk"] == c]) > 0 else str(c)
                for c in range(n_chunks)]
    ax.set_yticklabels([f"{i}: {p}" for i, p in enumerate(patterns)], fontsize=8)
    ax.set_xlabel("Config (algo+preprocessing)", fontsize=10)
    ax.set_ylabel("Chunk (pattern)", fontsize=10)
    ax.set_title("Brute-Force Compression Ratio Heatmap\n"
                 "(blue border = NN pick, green dashed = BF best)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Compression Ratio", shrink=0.8)

    path = os.path.join(out_dir, "bruteforce_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_bars(chunks, out_dir):
    """Grouped bars showing entropy, MAD, deriv per chunk — verifies stats differ."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    n = len(chunks)
    x = np.arange(n)
    labels = [f"{r['chunk']}: {r['pattern']}" for _, r in chunks.iterrows()]

    # Color bars by NN config
    configs = chunks["nn_config"].unique()
    cmap = plt.cm.Set1
    color_map = {c: cmap(i / max(len(configs) - 1, 1)) for i, c in enumerate(configs)}
    colors = [color_map.get(r["nn_config"], "gray") for _, r in chunks.iterrows()]

    for ax, col, title in zip(axes,
                               ["entropy", "mad", "deriv"],
                               ["Shannon Entropy (bits)",
                                "MAD (range-normalized)",
                                "2nd Derivative (range-normalized)"]):
        vals = chunks[col].values
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate NN config
        for i, v in enumerate(vals):
            ax.text(i, v, chunks.iloc[i]["nn_config"],
                    ha="center", va="bottom", fontsize=6, rotation=45)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Per-Chunk NN Features (color = NN config)", fontsize=13)

    patches = [mpatches.Patch(color=color_map[c], label=c) for c in configs]
    axes[0].legend(handles=patches, loc="upper right", fontsize=7, ncol=2)

    path = os.path.join(out_dir, "feature_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rank_of_nn_pick(chunks, bf, out_dir):
    """Bar chart: where does the NN pick rank in brute-force ordering (1=best)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n = len(chunks)
    x = np.arange(n)
    ranks = []

    for c in range(n):
        nn_act = chunks[chunks["chunk"] == c].iloc[0]["nn_action"]
        chunk_bf = bf[bf["chunk"] == c].sort_values("ratio", ascending=False).reset_index()
        rank = None
        for i, row in chunk_bf.iterrows():
            if row["action"] == nn_act:
                rank = i + 1
                break
        ranks.append(rank if rank else len(chunk_bf))

    colors = ["green" if r == 1 else "gold" if r <= 3 else "red" for r in ranks]
    ax.bar(x, ranks, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Best possible")
    ax.axhline(y=3, color="gold", linestyle="--", alpha=0.5, label="Top-3 threshold")

    labels = [f"{r['chunk']}: {r['pattern']}" for _, r in chunks.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NN Pick Rank in Brute-Force\n(1 = best)", fontsize=10)
    ax.set_title("NN Selection Quality: Rank Among All Configs", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(ranks) + 1)
    ax.invert_yaxis()  # rank 1 at top
    ax.grid(axis="y", alpha=0.3)

    # Annotate
    for i, r in enumerate(ranks):
        nn_cfg = chunks.iloc[i]["nn_config"]
        ax.text(i, r, f"#{r}\n{nn_cfg}", ha="center", va="top", fontsize=7)

    path = os.path.join(out_dir, "nn_rank.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize NN diagnostic benchmark")
    parser.add_argument("--dir", default="benchmarks/diagnostic/results",
                        help="Directory containing nn_chunks.csv and brute_force.csv")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output plots (default: same as --dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {args.dir}...")
    chunks, bf = load_data(args.dir)
    print(f"  {len(chunks)} chunks, {len(bf)} brute-force results")

    print(f"\nGenerating plots in {out_dir}...")
    plot_feature_bars(chunks, out_dir)
    plot_nn_vs_bruteforce(chunks, bf, out_dir)
    plot_bruteforce_heatmap(chunks, bf, out_dir)
    plot_rank_of_nn_pick(chunks, bf, out_dir)

    try:
        plot_feature_scatter(chunks, out_dir)
    except Exception as e:
        print(f"  Skipping 3D scatter (requires mpl 3D support): {e}")

    print("\nDone! Plots saved.")


if __name__ == "__main__":
    main()
