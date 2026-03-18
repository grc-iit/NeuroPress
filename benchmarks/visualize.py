#!/usr/bin/env python3
"""
Unified benchmark visualizer for GPUCompress.

Combines three views into a single script:
  1. Aggregate summary  — bar charts of ratio, throughput, file sizes, NN stats
  2. Per-chunk algorithm — color-coded algorithm selection strip + ratio bars
  3. Per-chunk MAPE      — ratio/comp/decomp prediction error over chunks

Usage:
  # Auto-detect all CSVs and generate all figures
  python3 benchmarks/visualize.py

  # Specific CSVs
  python3 benchmarks/visualize.py --gs-csv path/to/vol.csv --gs-chunks-csv path/to/chunks.csv

  # Only MAPE for specific phases
  python3 benchmarks/visualize.py --phase nn-rl --phase nn-rl+exp50

  # Only specific views
  python3 benchmarks/visualize.py --view summary --view mape
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PHASE_ORDER = ["no-comp", "exhaustive", "nn", "nn-rl", "nn-rl+exp50"]

PHASE_COLORS = {
    "no-comp":     "#95a5a6",
    "exhaustive":  "#8e44ad",
    "nn":          "#2ecc71",
    "nn-rl":       "#e67e22",
    "nn-rl+exp50": "#e74c3c",
}

PHASE_LABELS = {
    "no-comp":     "No Comp",
    "exhaustive":  "Exhaustive\nSearch",
    "nn":          "NN\n(Inference)",
    "nn-rl":       "NN+SGD",
    "nn-rl+exp50": "NN+SGD\n+Explore",
}

ALGO_NAMES = ["lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"]

ALGO_COLORS = OrderedDict([
    ("lz4",              "#1f77b4"), ("lz4+shuf",         "#aec7e8"),
    ("lz4+quant",        "#08519c"), ("lz4+shuf+quant",   "#6baed6"),
    ("snappy",           "#ff7f0e"), ("snappy+shuf",      "#ffbb78"),
    ("snappy+quant",     "#e6550d"), ("snappy+shuf+quant","#fdae6b"),
    ("deflate",          "#2ca02c"), ("deflate+shuf",     "#98df8a"),
    ("deflate+quant",    "#006d2c"), ("deflate+shuf+quant","#74c476"),
    ("gdeflate",         "#d62728"), ("gdeflate+shuf",    "#ff9896"),
    ("gdeflate+quant",   "#a50f15"), ("gdeflate+shuf+quant","#fc9272"),
    ("zstd",             "#9467bd"), ("zstd+shuf",        "#c5b0d5"),
    ("zstd+quant",       "#6a3d9a"), ("zstd+shuf+quant",  "#b294c7"),
    ("ans",              "#8c564b"), ("ans+shuf",         "#c49c94"),
    ("ans+quant",        "#5b3a29"), ("ans+shuf+quant",   "#a97e6e"),
    ("cascaded",         "#e377c2"), ("cascaded+shuf",    "#f7b6d2"),
    ("cascaded+quant",   "#c51b8a"), ("cascaded+shuf+quant","#f768a1"),
    ("bitcomp",          "#7f7f7f"), ("bitcomp+shuf",     "#c7c7c7"),
    ("bitcomp+quant",    "#525252"), ("bitcomp+shuf+quant","#969696"),
])

# Auto-detection paths
DEFAULT_GS_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol.csv"),
]
DEFAULT_GS_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol_chunks.csv"),
]
DEFAULT_VPIC_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic/benchmark_vpic_deck.csv"),
]
DEFAULT_VPIC_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic/benchmark_vpic_chunks.csv"),
]
DEFAULT_GS_TIMESTEPS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_timesteps.csv"),
]
DEFAULT_PREDICTIONS = ["/tmp/test_vol_nn_predictions.csv"]


def find_csv(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def parse_csv(path):
    """Parse CSV into list of dicts with auto float conversion."""
    rows = []
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        for line in f:
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(header):
                continue
            row = {}
            for h, v in zip(header, vals):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v
            rows.append(row)
    return rows


def g(row, *keys, default=0.0):
    """Get first matching key from row."""
    for k in keys:
        if k in row:
            return row[k]
    return default


def decode_action(action_int):
    algo = action_int % 8
    quant = (action_int // 8) % 2
    shuf = (action_int // 16) % 2
    s = ALGO_NAMES[algo]
    if shuf:
        s += "+shuf"
    if quant:
        s += "+quant"
    return s


def get_algo_color(algo_str):
    return ALGO_COLORS.get(algo_str, "#333333")


_PHASE_ALIASES = {"oracle": "exhaustive"}


def _ordered(rows):
    """Return rows ordered by PHASE_ORDER."""
    by_phase = {}
    for r in rows:
        phase = _PHASE_ALIASES.get(r["phase"], r["phase"])
        r["phase"] = phase
        by_phase[phase] = r
    phases = [p for p in PHASE_ORDER if p in by_phase]
    return phases, [by_phase[p] for p in phases]


# ═══════════════════════════════════════════════════════════════════════
# View 1: Aggregate Summary
# ═══════════════════════════════════════════════════════════════════════

def plot_bars(ax, phases, values, title, ylabel, fmt="%.2f", annotate=True):
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
    if annotate:
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt % v, ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def plot_file_sizes(ax, phases, rows):
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    orig = [g(r, "orig_mib", "orig_mb") for r in rows]
    comp = [g(r, "file_mib", "file_mb") for r in rows]
    w = 0.35
    ax.bar(x - w / 2, orig, w, color="#d5d8dc", edgecolor="white", label="Original")
    cbars = ax.bar(x + w / 2, comp, w, color=colors, edgecolor="white", label="Compressed")
    for bar, v in zip(cbars, comp):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    if orig and max(orig) > 0:
        ax.axhline(max(orig), color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Size (MB)", fontsize=9)
    ax.set_title("Original vs Compressed File Size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_nn_stats(ax, phases, rows):
    nn_idx = [i for i, p in enumerate(phases) if p in ("nn", "nn-rl", "nn-rl+exp50")]
    if not nn_idx:
        ax.text(0.5, 0.5, "No NN phases", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("NN Adaptation", fontsize=11, fontweight="bold")
        return
    nn_x = np.arange(len(nn_idx))
    nn_labels = [PHASE_LABELS.get(phases[i], phases[i]) for i in nn_idx]
    w = 0.35
    sgd_pct = [100.0 * g(rows[i], "sgd_fires") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    expl_pct = [100.0 * g(rows[i], "explorations") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    ax.bar(nn_x - w / 2, sgd_pct, w, color="#e67e22", edgecolor="white", label="SGD Fires %")
    ax.bar(nn_x + w / 2, expl_pct, w, color="#e74c3c", edgecolor="white", label="Explorations %")
    for xi, (s, e) in enumerate(zip(sgd_pct, expl_pct)):
        ax.text(xi - w / 2, s, f"{s:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, e, f"{e:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(nn_x)
    ax.set_xticklabels(nn_labels, fontsize=8)
    ax.set_ylabel("% of chunks", fontsize=9)
    ax.set_ylim(0, max(max(sgd_pct, default=0), max(expl_pct, default=0)) * 1.3 + 5)
    ax.set_title("NN Adaptation: SGD Fires & Explorations", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    param_text = "SGD:  LR=0.4, MAPE thresh=20%\nExplore:  MAPE thresh=50%, K=2"
    ax.text(0.98, 0.97, param_text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                      edgecolor="#aab0b5", alpha=0.9), family="monospace")


def plot_verification(ax, phases, rows):
    ax.axis("off")
    cell_text = []
    cell_colors = []
    for i, p in enumerate(phases):
        r = rows[i]
        mm = int(g(r, "mismatches"))
        status = "PASS" if mm == 0 else f"FAIL ({mm})"
        color = "#d5f5e3" if mm == 0 else "#fadbd8"
        orig = g(r, "orig_mib", "orig_mb")
        comp = g(r, "file_mib", "file_mb")
        ratio = g(r, "ratio")
        cell_text.append([PHASE_LABELS.get(p, p).replace("\n", " "),
                          f"{orig:.0f}", f"{comp:.1f}", f"{ratio:.2f}x", status])
        cell_colors.append(["white", "white", "white", "white", color])
    table = ax.table(cellText=cell_text,
                     colLabels=["Phase", "Orig (MB)", "Comp (MB)", "Ratio", "Verify"],
                     cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d6eaf8")
    ax.set_title("Summary & Verification", fontsize=11, fontweight="bold", pad=20)


def make_summary_figure(source_name, rows, output_path, meta_text=""):
    phases, ordered = _ordered(rows)
    if not phases:
        print(f"  {source_name}: no valid phases, skipping summary.")
        return
    fig = plt.figure(figsize=(16, 15))
    fig.text(0.5, 0.99, f"GPUCompress Benchmark: {source_name}",
             ha="center", fontsize=15, fontweight="bold", va="top")
    fig.text(0.5, 0.97,
             "Compression ratio, read/write throughput, timing breakdown, and data verification.",
             ha="center", fontsize=9, color="#555", va="top", style="italic")
    if meta_text:
        fig.text(0.5, 0.955, meta_text, ha="center", fontsize=10,
                 color="#333", va="top", fontfamily="monospace")
    gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.30,
                           top=0.90, bottom=0.04, left=0.08, right=0.96)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bars(ax1, phases, [g(r, "ratio") for r in ordered],
              "Compression Ratio (higher = better)", "Ratio", fmt="%.2fx")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_bars(ax2, phases, [g(r, "write_mibps", "write_mbps") for r in ordered],
              "Write Throughput", "MB/s", fmt="%.0f")
    ax3 = fig.add_subplot(gs[1, 0])
    plot_bars(ax3, phases, [g(r, "read_mibps", "read_mbps") for r in ordered],
              "Read Throughput", "MB/s", fmt="%.0f")
    ax4 = fig.add_subplot(gs[1, 1])
    plot_file_sizes(ax4, phases, ordered)
    ax5 = fig.add_subplot(gs[2, 0])
    plot_nn_stats(ax5, phases, ordered)
    ax6 = fig.add_subplot(gs[2, 1])
    plot_verification(ax6, phases, ordered)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View 2: Per-Chunk Algorithm Selection
# ═══════════════════════════════════════════════════════════════════════

def load_chunks_csv(path):
    """Returns {(timestep, phase): [Row, ...]} and sorted timesteps."""
    data = OrderedDict()
    timesteps = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        has_timestep = "timestep" in fields
        for row in reader:
            ts = int(row["timestep"]) if has_timestep else 0
            phase = row["phase"]
            if phase == "oracle":
                phase = "exhaustive"
            chunk = int(row["chunk"])
            algo = row.get("action_final", "")
            if not algo:
                action = int(row.get("nn_action", 0))
                algo = decode_action(action)
            ratio = float(row.get("actual_ratio", 0))
            pred = float(row.get("predicted_ratio", 0))
            sgd = int(row.get("sgd_fired", 0))
            expl = int(row.get("exploration_triggered", row.get("explored", 0)))
            key = (ts, phase)
            timesteps.add(ts)
            data.setdefault(key, []).append((ts, chunk, algo, ratio, pred, sgd, expl))
    return data, sorted(timesteps)


def plot_chunks_single(data, timesteps, out_path):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ts = timesteps[0]
    phases = OrderedDict()
    for (t, phase), recs in data.items():
        if t == ts:
            phases[phase] = recs
    phase_names = list(phases.keys())
    n_phases = len(phase_names)
    has_ratio = any(r[3] > 0 for recs in phases.values() for r in recs)
    exhaustive_ratios = [r[3] for r in phases["exhaustive"]] if "exhaustive" in phases else None

    if has_ratio:
        fig, axes = plt.subplots(n_phases, 2, figsize=(18, 3.0 * n_phases + 1.5),
                                 gridspec_kw={"width_ratios": [3, 1]})
        if n_phases == 1:
            axes = [axes]
    else:
        fig, axes_raw = plt.subplots(n_phases, 1, figsize=(14, 2.5 * n_phases + 1.5))
        if n_phases == 1:
            axes_raw = [axes_raw]
        axes = [(ax, None) for ax in axes_raw]

    all_algos = OrderedDict()
    for recs in phases.values():
        for r in recs:
            if r[2] not in all_algos:
                all_algos[r[2]] = get_algo_color(r[2])

    for idx, phase in enumerate(phase_names):
        recs = phases[phase]
        n_chunks = len(recs)
        algos = [r[2] for r in recs]
        ratios = [r[3] for r in recs]
        preds = [r[4] for r in recs]
        sgds = [r[5] for r in recs]
        expls = [r[6] for r in recs]
        colors = [all_algos[a] for a in algos]

        if has_ratio:
            ax_bar, ax_ratio = axes[idx]
        else:
            ax_bar, ax_ratio = axes[idx]

        for i, color in enumerate(colors):
            ax_bar.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=0.3)
            if sgds[i]:
                ax_bar.plot(i + 0.5, 0.35, marker="v", color="black", markersize=4)
            if expls[i]:
                ax_bar.plot(i + 0.5, -0.35, marker="*", color="red", markersize=5)
        ax_bar.set_xlim(0, n_chunks)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Chunk index")
        ax_bar.set_title(f"Phase: {phase}  ({n_chunks} chunks)", fontsize=11, fontweight="bold")

        if ax_ratio is not None and has_ratio:
            x = np.arange(n_chunks)
            ax_ratio.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.3, width=1.0)
            if exhaustive_ratios and phase != "exhaustive" and len(exhaustive_ratios) == n_chunks:
                ax_ratio.plot(x, exhaustive_ratios, color="goldenrod", linewidth=1.5,
                              alpha=0.85, label="exhaustive best", zorder=5)
            all_y = list(ratios) + (list(exhaustive_ratios) if exhaustive_ratios and phase != "exhaustive" else [])
            all_y_sorted = sorted(all_y)
            p95 = all_y_sorted[min(len(all_y_sorted) - 1, int(0.95 * len(all_y_sorted)))]
            y_ceil = p95 * 1.5
            if any(p > 0 for p in preds):
                in_range = sum(1 for p in preds if 0 < p <= y_ceil) / max(len(preds), 1)
                if in_range > 0.3:
                    clipped = [min(p, y_ceil) for p in preds]
                    ax_ratio.plot(x, clipped, "k--", linewidth=0.8, alpha=0.5, label="predicted (clipped)")
            ax_ratio.set_ylim(0, y_ceil)
            handles, labels = ax_ratio.get_legend_handles_labels()
            if handles:
                ax_ratio.legend(fontsize=7, loc="upper right")
            ax_ratio.set_xlabel("Chunk index")
            ax_ratio.set_ylabel("Compression ratio")
            ax_ratio.set_title("Per-chunk ratio", fontsize=10)
            ax_ratio.set_xlim(-0.5, n_chunks - 0.5)

    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=a) for a, c in all_algos.items()]
    legend_elements.append(Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                                  markersize=6, label="SGD fired"))
    legend_elements.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                                  markersize=8, label="Exploration"))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_chunks_multi(data, timesteps, out_path):
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm

    exhaustive_key = (0, "exhaustive")
    exhaustive_recs = data.get(exhaustive_key, [])
    n_chunks = len(exhaustive_recs) if exhaustive_recs else 0
    adapt_phase = "nn-rl+exp50"
    adapt_ts = [ts for ts in timesteps if (ts, adapt_phase) in data]

    if not adapt_ts:
        return plot_chunks_single(data, timesteps, out_path)

    nn_key = (0, "nn")
    all_algos = OrderedDict()
    for recs in data.values():
        for r in recs:
            if r[2] not in all_algos:
                all_algos[r[2]] = get_algo_color(r[2])
    algo_list = list(all_algos.keys())
    algo_to_idx = {a: i for i, a in enumerate(algo_list)}

    row_labels = []
    heatmap_rows = []
    ratio_rows = []

    if exhaustive_recs:
        row_labels.append("exhaustive (t=0)")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in exhaustive_recs])
        ratio_rows.append([r[3] for r in exhaustive_recs])
    if nn_key in data:
        recs = data[nn_key]
        row_labels.append("nn (t=0)")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in recs])
        ratio_rows.append([r[3] for r in recs])
    for ts in adapt_ts:
        recs = data[(ts, adapt_phase)]
        row_labels.append(f"nn-rl+exp50 (t={ts})")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in recs])
        ratio_rows.append([r[3] for r in recs])

    n_rows = len(heatmap_rows)
    if n_chunks == 0:
        n_chunks = max(len(r) for r in heatmap_rows)
    for i in range(n_rows):
        while len(heatmap_rows[i]) < n_chunks:
            heatmap_rows[i].append(0)
        while len(ratio_rows[i]) < n_chunks:
            ratio_rows[i].append(0)

    hmap = np.array(heatmap_rows)
    rmap = np.array(ratio_rows)
    cmap_colors = [all_algos[a] for a in algo_list]
    cmap = ListedColormap(cmap_colors)
    bounds = np.arange(len(algo_list) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig_h = max(4, 0.4 * n_rows + 2.5)
    fig, (ax_heat, ax_rat) = plt.subplots(1, 2, figsize=(18, fig_h),
                                           gridspec_kw={"width_ratios": [2, 1]})
    im = ax_heat.imshow(hmap, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest", origin="upper")
    ax_heat.set_yticks(np.arange(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_xlabel("Chunk index", fontsize=10)
    ax_heat.set_title("Algorithm selection per chunk", fontsize=12, fontweight="bold")
    n_baseline = 1 + (1 if nn_key in data else 0)
    if n_baseline < n_rows:
        ax_heat.axhline(n_baseline - 0.5, color="white", linewidth=2)

    all_rats = rmap[rmap > 0]
    vmax = np.percentile(all_rats, 95) * 1.3 if len(all_rats) > 0 else 1
    ax_rat.imshow(rmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                  interpolation="nearest", origin="upper")
    ax_rat.set_yticks(np.arange(n_rows))
    ax_rat.set_yticklabels(row_labels, fontsize=8)
    ax_rat.set_xlabel("Chunk index", fontsize=10)
    ax_rat.set_title("Compression ratio per chunk", fontsize=12, fontweight="bold")
    if n_baseline < n_rows:
        ax_rat.axhline(n_baseline - 0.5, color="white", linewidth=2)

    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=a) for a, c in all_algos.items()]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_chunks_figure(chunks_csv, output_path):
    data, timesteps = load_chunks_csv(chunks_csv)
    if not data:
        print(f"  No chunk data in {chunks_csv}, skipping.")
        return
    if len(timesteps) > 1:
        plot_chunks_multi(data, timesteps, output_path)
    else:
        plot_chunks_single(data, timesteps, output_path)


# ═══════════════════════════════════════════════════════════════════════
# View 3: Per-Chunk MAPE
# ═══════════════════════════════════════════════════════════════════════

def detect_mape_format(fieldnames):
    if "ratio_mape" in fieldnames and "pattern" in fieldnames:
        return "predictions"
    if "mape_ratio" in fieldnames and "phase" in fieldnames:
        return "benchmark"
    return None


def load_mape_csv(path, phases=None):
    """Load CSV and return {phase: [rows]} dict."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fmt = detect_mape_format(reader.fieldnames)
        if fmt is None:
            return None  # not a MAPE-compatible CSV

        for row in reader:
            if fmt == "predictions":
                phase = "predictions"
                entry = {
                    "chunk":       int(row["chunk"]),
                    "label":       row.get("pattern", ""),
                    "algo":        row.get("nn_pick", ""),
                    "ratio_mape":  float(row["ratio_mape"]),
                    "comp_mape":   float(row["comp_mape"]),
                    "decomp_mape": float(row["decomp_mape"]),
                    "sgd_fired":   int(row["sgd_fired"]),
                    "exploration": int(row.get("explored", row.get("exploration_triggered", 0))),
                }
            else:
                phase = row["phase"]
                if phase == "no-comp":
                    continue
                entry = {
                    "chunk":       int(row["chunk"]),
                    "label":       row.get("action_final", ""),
                    "algo":        row.get("action_final", ""),
                    "ratio_mape":  float(row["mape_ratio"]),
                    "comp_mape":   float(row["mape_comp"]),
                    "decomp_mape": float(row["mape_decomp"]),
                    "sgd_fired":   int(row.get("sgd_fired", 0)),
                    "exploration": int(row.get("exploration_triggered", 0)),
                }
            if phases and phase not in phases:
                continue
            data.setdefault(phase, []).append(entry)
    return data


def make_mape_figure(mape_data, output_path):
    phase_names = list(mape_data.keys())
    n_phases = len(phase_names)
    if n_phases == 0:
        return

    fig, all_axes = plt.subplots(n_phases, 3, figsize=(18, 3.5 * n_phases + 1.5),
                                  sharex=False, squeeze=False)
    metric_labels = ["Ratio MAPE (%)", "Comp Time MAPE (%)", "Decomp Time MAPE (%)"]
    metric_keys = ["ratio_mape", "comp_mape", "decomp_mape"]

    for row_idx, phase in enumerate(phase_names):
        rows = mape_data[phase]
        chunks = np.array([r["chunk"] for r in rows])
        sgd_fired = np.array([r["sgd_fired"] for r in rows], dtype=bool)
        exploration = np.array([r["exploration"] for r in rows], dtype=bool)
        labels = [r["label"] for r in rows]

        unique_labels = list(dict.fromkeys(labels))
        label_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
        label_cmap = {lb: label_colors[i] for i, lb in enumerate(unique_labels)}
        colors = [label_cmap[lb] for lb in labels]

        for col_idx, key in enumerate(metric_keys):
            ax = all_axes[row_idx, col_idx]
            mape = np.array([r[key] for r in rows])
            bars = ax.bar(chunks, mape, color=colors, edgecolor="white", linewidth=0.3, width=1.0)

            # Add value labels on bars
            for bar, val in zip(bars, mape):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0f}", ha="center", va="bottom",
                            fontsize=5, rotation=90, color="#333333")

            sgd_chunks = chunks[sgd_fired]
            sgd_vals = mape[sgd_fired]
            if len(sgd_chunks) > 0:
                ax.scatter(sgd_chunks, sgd_vals + max(mape) * 0.02,
                           marker="v", color="black", s=18, zorder=5, label="SGD fired")
            expl_chunks = chunks[exploration]
            expl_vals = mape[exploration]
            if len(expl_chunks) > 0:
                ax.scatter(expl_chunks, expl_vals + max(mape) * 0.04,
                           marker="*", color="red", s=24, zorder=5, label="Exploration")

            if len(mape) >= 6:
                window = min(6, len(mape))
                kernel = np.ones(window) / window
                rolling = np.convolve(mape, kernel, mode="valid")
                x_roll = chunks[window - 1:]
                ax.plot(x_roll, rolling, color="red", linewidth=2, alpha=0.8,
                        label=f"rolling avg (w={window})")

            if len(mape) > 0 and max(mape) > 0:
                p95 = np.percentile(mape, 95)
                if p95 > 0:
                    ax.set_ylim(0, min(max(mape) * 1.1, p95 * 2.0))

            if row_idx == 0:
                ax.set_title(metric_labels[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{phase}\nMAPE (%)", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("MAPE (%)", fontsize=9)
            if row_idx == n_phases - 1:
                ax.set_xlabel("Chunk index", fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("NN Prediction MAPE over Chunks", fontsize=14, fontweight="bold", y=1.01)

    from matplotlib.patches import Patch
    all_unique = []
    for rows in mape_data.values():
        for r in rows:
            if r["label"] not in all_unique:
                all_unique.append(r["label"])
    g_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_unique), 1)))
    g_cmap = {lb: g_colors[i] for i, lb in enumerate(all_unique)}
    legend_pats = [Patch(facecolor=g_cmap[lb], edgecolor="gray", label=lb) for lb in all_unique]
    fig.legend(handles=legend_pats, loc="lower center",
               ncol=min(len(legend_pats), 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View 4: Timestep Adaptation (sMAPE over timesteps)
# ═══════════════════════════════════════════════════════════════════════

def make_timestep_figure(ts_csv_path, output_path):
    """Plot sMAPE for ratio/comp_time/decomp_time over timesteps."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping.")
        return

    timesteps = np.array([g(r, "timestep") for r in rows])
    smape_r = np.array([g(r, "smape_ratio") for r in rows])
    smape_c = np.array([g(r, "smape_comp") for r in rows])
    smape_d = np.array([g(r, "smape_decomp") for r in rows])
    sgd_fires = np.array([g(r, "sgd_fires") for r in rows])
    ratio = np.array([g(r, "ratio") for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("NN Adaptation Over Timesteps (Multi-Timestep SGD)",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: All three sMAPE on one plot ──
    ax = axes[0, 0]
    ax.plot(timesteps, smape_r, "o-", color="#2ecc71", markersize=3,
            linewidth=1.5, label="Ratio sMAPE", alpha=0.9)
    ax.plot(timesteps, smape_c, "s-", color="#3498db", markersize=3,
            linewidth=1.5, label="Comp Time sMAPE", alpha=0.9)
    ax.plot(timesteps, smape_d, "D-", color="#e74c3c", markersize=3,
            linewidth=1.5, label="Decomp Time sMAPE", alpha=0.9)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("sMAPE (%)", fontsize=10)
    ax.set_title("Prediction sMAPE Over Time", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    # Shade T=0 region
    if len(timesteps) > 1:
        ax.axvspan(timesteps[0] - 0.5, timesteps[0] + 0.5,
                   color="#fee0d2", alpha=0.5, label="_nolegend_")
        ax.annotate("pretrained\n(no reads yet)", xy=(timesteps[0], smape_d[0]),
                    xytext=(timesteps[0] + max(1, len(timesteps) * 0.08), smape_d[0] * 0.85),
                    fontsize=8, color="#666", arrowprops=dict(arrowstyle="->", color="#999"))

    # ── Panel 2: Decomp time sMAPE zoomed ──
    ax = axes[0, 1]
    ax.plot(timesteps, smape_d, "D-", color="#e74c3c", markersize=4,
            linewidth=2, label="Decomp Time sMAPE")
    # Rolling average
    if len(smape_d) >= 5:
        w = min(5, len(smape_d))
        kernel = np.ones(w) / w
        rolling = np.convolve(smape_d, kernel, mode="valid")
        x_roll = timesteps[w - 1:]
        ax.plot(x_roll, rolling, "--", color="#c0392b", linewidth=1.5,
                alpha=0.7, label=f"Rolling avg (w={w})")
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("sMAPE (%)", fontsize=10)
    ax.set_title("Decomp Time sMAPE (Deferred Head-Only SGD)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    # Add horizontal reference lines
    ax.axhline(20, color="#e67e22", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(timesteps[-1], 21, "20%", fontsize=7, color="#e67e22", ha="right")

    # ── Panel 3: SGD fires per timestep ──
    ax = axes[1, 0]
    ax.bar(timesteps, sgd_fires, color="#e67e22", edgecolor="white",
           linewidth=0.3, alpha=0.8, width=0.8)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("SGD Fires", fontsize=10)
    ax.set_title("SGD Updates Per Timestep", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Compression ratio over time ──
    ax = axes[1, 1]
    ax.plot(timesteps, ratio, "o-", color="#8e44ad", markersize=3,
            linewidth=1.5)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("Compression Ratio", fontsize=10)
    ax.set_title("Compression Ratio Over Time", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # Add summary text box
    if len(smape_d) > 1:
        t0_d = smape_d[0]
        t1_d = smape_d[1] if len(smape_d) > 1 else t0_d
        avg_d = np.mean(smape_d[1:]) if len(smape_d) > 1 else t0_d
        max_d = np.max(smape_d[1:]) if len(smape_d) > 1 else t0_d
        summary = (f"Decomp sMAPE:  T=0: {t0_d:.1f}%  →  T=1: {t1_d:.1f}%\n"
                   f"  avg(T≥1): {avg_d:.1f}%   max(T≥1): {max_d:.1f}%\n"
                   f"Ratio sMAPE avg: {np.mean(smape_r[1:]):.1f}%\n"
                   f"Comp sMAPE avg:  {np.mean(smape_c[1:]):.1f}%")
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                           edgecolor="#aab0b5", alpha=0.9))

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

ALL_VIEWS = ["summary", "chunks", "mape", "timesteps"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified GPUCompress benchmark visualizer (summary + chunks + MAPE)")
    parser.add_argument("csvs", nargs="*",
                        help="Positional CSV paths (auto-classified as gs/vpic, agg/chunks)")
    parser.add_argument("--gs-csv", help="Gray-Scott aggregate CSV")
    parser.add_argument("--gs-chunks-csv", help="Gray-Scott per-chunk CSV")
    parser.add_argument("--gs-timesteps-csv", help="Gray-Scott timestep adaptation CSV")
    parser.add_argument("--vpic-csv", help="VPIC aggregate CSV")
    parser.add_argument("--vpic-chunks-csv", help="VPIC per-chunk CSV")
    parser.add_argument("--predictions-csv", help="test_vol_nn_predictions CSV")
    parser.add_argument("--output-dir", help="Output directory (default: alongside CSV)")
    parser.add_argument("--phase", action="append", default=None,
                        help="Phase filter for MAPE view (repeat for multiple)")
    parser.add_argument("--view", action="append", default=None,
                        help=f"Views to generate: {ALL_VIEWS} (default: all)")
    args = parser.parse_args()

    views = args.view or ALL_VIEWS

    # Classify positional CSVs
    for csv_path in args.csvs:
        low = csv_path.lower()
        is_vpic = "vpic" in low
        is_chunks = "chunk" in low
        is_pred = "prediction" in low
        if is_pred:
            if not args.predictions_csv:
                args.predictions_csv = csv_path
        elif is_vpic and is_chunks:
            if not args.vpic_chunks_csv:
                args.vpic_chunks_csv = csv_path
        elif is_vpic:
            if not args.vpic_csv:
                args.vpic_csv = csv_path
        elif is_chunks:
            if not args.gs_chunks_csv:
                args.gs_chunks_csv = csv_path
        else:
            if not args.gs_csv:
                args.gs_csv = csv_path

    # Auto-detect
    gs_agg = args.gs_csv or find_csv(DEFAULT_GS_AGG)
    gs_chunks = args.gs_chunks_csv or find_csv(DEFAULT_GS_CHUNKS)
    gs_tsteps = args.gs_timesteps_csv or find_csv(DEFAULT_GS_TIMESTEPS)
    vpic_agg = args.vpic_csv or find_csv(DEFAULT_VPIC_AGG)
    vpic_chunks = args.vpic_chunks_csv or find_csv(DEFAULT_VPIC_CHUNKS)
    pred_csv = args.predictions_csv or find_csv(DEFAULT_PREDICTIONS)

    found_any = False

    # ── Gray-Scott ──
    if gs_agg and os.path.exists(gs_agg):
        found_any = True
        if "summary" in views:
            print(f"Loading Gray-Scott aggregate: {gs_agg}")
            rows = parse_csv(gs_agg)
            r0 = rows[0] if rows else {}
            L = int(g(r0, "L"))
            steps = int(g(r0, "steps"))
            F_val = g(r0, "F")
            k_val = g(r0, "k")
            orig = g(r0, "orig_mib", "orig_mb")
            n_ch = int(g(r0, "n_chunks"))
            chunk_mb = orig / max(n_ch, 1)
            meta = (f"Grid: {L}^3 ({orig:.0f} MB) | Chunks: {n_ch} x {chunk_mb:.0f} MB | "
                    f"Steps: {steps} | F={F_val}, k={k_val}")
            out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "grayscott", "results")
            make_summary_figure("Gray-Scott Simulation", rows,
                                os.path.join(out_dir, "benchmark_grayscott.png"), meta)

    if gs_chunks and os.path.exists(gs_chunks):
        found_any = True
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(gs_chunks))
        if "chunks" in views:
            print(f"Loading Gray-Scott chunks: {gs_chunks}")
            make_chunks_figure(gs_chunks, os.path.join(out_dir, "chunks_viz.png"))
        if "mape" in views:
            print(f"Loading Gray-Scott MAPE: {gs_chunks}")
            mape_data = load_mape_csv(gs_chunks, phases=args.phase)
            if mape_data:
                total = sum(len(v) for v in mape_data.values())
                print(f"  {total} chunks across {len(mape_data)} phase(s)")
                for phase, mrows in mape_data.items():
                    avg_r = np.mean([r["ratio_mape"] for r in mrows])
                    avg_c = np.mean([r["comp_mape"] for r in mrows])
                    sgd_n = sum(1 for r in mrows if r["sgd_fired"])
                    exp_n = sum(1 for r in mrows if r["exploration"])
                    print(f"    {phase}: {len(mrows)} chunks, avg ratio MAPE={avg_r:.1f}%, "
                          f"avg comp MAPE={avg_c:.1f}%, SGD={sgd_n}, explorations={exp_n}")
                make_mape_figure(mape_data, os.path.join(out_dir, "mape_over_chunks.png"))

    if gs_tsteps and os.path.exists(gs_tsteps) and "timesteps" in views:
        found_any = True
        print(f"Loading Gray-Scott timestep adaptation: {gs_tsteps}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(gs_tsteps))
        make_timestep_figure(gs_tsteps, os.path.join(out_dir, "timestep_adaptation.png"))

    # ── VPIC ──
    if vpic_agg and os.path.exists(vpic_agg):
        found_any = True
        if "summary" in views:
            print(f"Loading VPIC aggregate: {vpic_agg}")
            rows = parse_csv(vpic_agg)
            r0 = rows[0] if rows else {}
            orig = g(r0, "orig_mib", "orig_mb")
            n_ch = int(g(r0, "n_chunks"))
            chunk_mb = orig / max(n_ch, 1)
            meta = (f"Dataset: {orig:.0f} MB | Chunks: {n_ch} x {chunk_mb:.0f} MB | "
                    f"Source: VPIC Harris Sheet")
            out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "vpic-kokkos", "results")
            make_summary_figure("VPIC Harris Sheet Reconnection", rows,
                                os.path.join(out_dir, "benchmark_vpic.png"), meta)

    if vpic_chunks and os.path.exists(vpic_chunks):
        found_any = True
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(vpic_chunks))
        if "chunks" in views:
            print(f"Loading VPIC chunks: {vpic_chunks}")
            make_chunks_figure(vpic_chunks, os.path.join(out_dir, "chunks_viz.png"))
        if "mape" in views:
            print(f"Loading VPIC MAPE: {vpic_chunks}")
            mape_data = load_mape_csv(vpic_chunks, phases=args.phase)
            if mape_data:
                total = sum(len(v) for v in mape_data.values())
                print(f"  {total} chunks across {len(mape_data)} phase(s)")
                make_mape_figure(mape_data, os.path.join(out_dir, "mape_over_chunks.png"))

    # ── Predictions CSV (from test_vol_nn_predictions) ──
    if pred_csv and os.path.exists(pred_csv) and "mape" in views:
        found_any = True
        print(f"Loading predictions MAPE: {pred_csv}")
        mape_data = load_mape_csv(pred_csv, phases=args.phase)
        if mape_data:
            out_dir = args.output_dir or os.path.dirname(os.path.abspath(pred_csv))
            make_mape_figure(mape_data, os.path.join(out_dir, "mape_over_chunks.png"))

    if not found_any:
        print("ERROR: No benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_GS_AGG + DEFAULT_GS_CHUNKS + DEFAULT_VPIC_AGG + DEFAULT_VPIC_CHUNKS:
            print(f"  {p}")
        print("\nRun benchmarks first, or specify paths explicitly.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
