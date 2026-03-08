#!/usr/bin/env python3
"""
Unified benchmark visualizer for GPUCompress.

Reads Gray-Scott and/or VPIC CSV results and produces a separate
8-panel figure for each benchmark.

Usage:
  python3 benchmarks/visualize.py [options]

  --gs-csv PATH    Gray-Scott aggregate CSV (default: auto-detect)
  --vpic-csv PATH  VPIC aggregate CSV (default: auto-detect)
  --output-dir DIR Directory for output PNGs (default: same dir as CSV)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── CSV parsing (no pandas dependency) ─────────────────────────────────

def parse_csv(path):
    """Parse a CSV file into a list of dicts with auto float conversion."""
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
    """Get first matching key from row, with default."""
    for k in keys:
        if k in row:
            return row[k]
    return default


# ── Constants ──────────────────────────────────────────────────────────

PHASE_ORDER = ["no-comp", "oracle", "nn", "nn-rl", "nn-rl+exp50"]

PHASE_COLORS = {
    "no-comp":     "#95a5a6",
    "oracle":      "#8e44ad",
    "nn":          "#2ecc71",
    "nn-rl":       "#e67e22",
    "nn-rl+exp50": "#e74c3c",
}

PHASE_LABELS = {
    "no-comp":     "No Comp",
    "oracle":      "Oracle\n(Exhaustive)",
    "nn":          "NN\n(Inference)",
    "nn-rl":       "NN+SGD",
    "nn-rl+exp50": "NN+SGD\n+Explore",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # benchmarks/ -> GPUCompress/

DEFAULT_GS_PATHS = [
    os.path.join(PROJECT_ROOT, "tests/benchmark_grayscott_vol_results/benchmark_grayscott_vol.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol.csv"),
]

DEFAULT_VPIC_PATHS = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck_results/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic/benchmark_vpic_deck.csv"),
]


def find_csv(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Plotting functions ─────────────────────────────────────────────────

def _ordered(rows):
    """Return rows ordered by PHASE_ORDER."""
    by_phase = {r["phase"]: r for r in rows}
    phases = [p for p in PHASE_ORDER if p in by_phase]
    return phases, [by_phase[p] for p in phases]


def plot_bars(ax, phases, values, title, ylabel, fmt="%.2f", annotate=True):
    """Simple bar chart for one metric."""
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
    """Original vs compressed file size bars."""
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
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
    if orig and max(orig) > 0:
        ax.axhline(max(orig), color="black", linewidth=0.8, linestyle="--",
                   alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Size (MB)", fontsize=9)
    ax.set_title("Original vs Compressed File Size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_timing_breakdown(ax, phases, rows):
    """Stacked bar of write-path timing components."""
    x = np.arange(len(phases))
    components = [
        ("total_nn_inference_ms",   "#e67e22", "NN Inference"),
        ("total_preprocessing_ms",  "#9b59b6", "Preprocessing"),
        ("total_compression_ms",    "#3498db", "Compression"),
        ("total_exploration_ms",    "#e74c3c", "Exploration"),
        ("total_sgd_ms",            "#2ecc71", "SGD Update"),
    ]
    bottom = np.zeros(len(phases))
    has_data = False
    for key, color, label in components:
        vals = np.array([g(r, key) for r in rows], dtype=float)
        if np.any(vals > 0):
            has_data = True
        ax.bar(x, vals, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=0.5)
        bottom += vals

    # Show overhead (write_ms - sum of components)
    write_ms = np.array([g(r, "write_ms") for r in rows], dtype=float)
    overhead = np.maximum(0, write_ms - bottom)
    if np.any(overhead > 0):
        ax.bar(x, overhead, bottom=bottom, color="#cccccc", label="Overhead",
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)", fontsize=9)
    ax.set_title("Write Time Breakdown", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if has_data:
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    else:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")


def plot_ratio_distribution(ax, phases, rows):
    """Ratio bars with min/max whiskers and MAPE annotation."""
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    avg = [g(r, "ratio") for r in rows]
    rmin = [g(r, "ratio_min") for r in rows]
    rmax = [g(r, "ratio_max") for r in rows]
    mape = [g(r, "mean_prediction_error_pct") for r in rows]

    lo_err = [max(0, avg[i] - rmin[i]) if rmin[i] > 0 else 0 for i in range(len(phases))]
    hi_err = [max(0, rmax[i] - avg[i]) if rmax[i] > 0 else 0 for i in range(len(phases))]

    has_range = any(lo > 0 or hi > 0 for lo, hi in zip(lo_err, hi_err))

    if has_range:
        ax.bar(x, avg, color=colors, edgecolor="white", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=4,
               error_kw={"linewidth": 1.2, "alpha": 0.7})
    else:
        ax.bar(x, avg, color=colors, edgecolor="white", linewidth=0.5)

    for i, p in enumerate(phases):
        if p.startswith("nn") and mape[i] > 0:
            y_top = avg[i] + hi_err[i] if has_range else avg[i]
            ax.text(x[i], y_top + 0.05, f"MAPE\n{mape[i]:.1f}%",
                    ha="center", va="bottom", fontsize=7,
                    color="#c0392b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Compression Ratio (x)", fontsize=9)
    title = "Ratio Distribution (bar=avg, whiskers=min/max)" if has_range \
        else "Compression Ratio"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def plot_nn_stats(ax, phases, rows):
    """SGD fire and exploration rates for NN phases."""
    nn_idx = [i for i, p in enumerate(phases) if p in ("nn", "nn-rl", "nn-rl+exp50")]
    if not nn_idx:
        ax.text(0.5, 0.5, "No NN phases", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("NN Adaptation", fontsize=11, fontweight="bold")
        return

    nn_x = np.arange(len(nn_idx))
    nn_labels = [PHASE_LABELS.get(phases[i], phases[i]) for i in nn_idx]
    nn_colors = [PHASE_COLORS.get(phases[i], "#bdc3c7") for i in nn_idx]

    sgd_pct = [100.0 * g(rows[i], "sgd_fires") / max(g(rows[i], "n_chunks"), 1)
               for i in nn_idx]
    expl_pct = [100.0 * g(rows[i], "explorations") / max(g(rows[i], "n_chunks"), 1)
                for i in nn_idx]

    w = 0.35
    ax.bar(nn_x - w / 2, sgd_pct, w, color="#e67e22", edgecolor="white",
           label="SGD Fires %")
    ax.bar(nn_x + w / 2, expl_pct, w, color="#e74c3c", edgecolor="white",
           label="Explorations %")
    for xi, (s, e) in enumerate(zip(sgd_pct, expl_pct)):
        ax.text(xi - w / 2, s, f"{s:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, e, f"{e:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(nn_x)
    ax.set_xticklabels(nn_labels, fontsize=8)
    ax.set_ylabel("% of chunks", fontsize=9)
    ax.set_ylim(0, max(max(sgd_pct, default=0), max(expl_pct, default=0)) * 1.3 + 5)
    ax.set_title("NN Adaptation: SGD Fires & Explorations", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_verification(ax, phases, rows):
    """Verification pass/fail table."""
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
                          f"{orig:.0f}", f"{comp:.1f}",
                          f"{ratio:.2f}x", status])
        cell_colors.append(["white", "white", "white", "white", color])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Phase", "Orig (MB)", "Comp (MB)", "Ratio", "Verify"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d6eaf8")
    ax.set_title("Summary & Verification", fontsize=11, fontweight="bold",
                 pad=20)


# ── Figure assembly ────────────────────────────────────────────────────

def make_figure(source_name, rows, output_path, meta_text=""):
    """Build an 8-panel figure for one benchmark source."""
    phases, ordered = _ordered(rows)
    if not phases:
        print(f"  {source_name}: no valid phases found, skipping.")
        return

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(f"GPUCompress Benchmark: {source_name}",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.50, wspace=0.30,
                           top=0.94, bottom=0.04, left=0.08, right=0.96)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bars(ax1, phases,
              [g(r, "ratio") for r in ordered],
              "Compression Ratio (higher = better)",
              "Ratio", fmt="%.2fx")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_bars(ax2, phases,
              [g(r, "write_mibps", "write_mbps") for r in ordered],
              "Write Throughput", "MB/s", fmt="%.0f")

    # Row 2
    ax3 = fig.add_subplot(gs[1, 0])
    plot_bars(ax3, phases,
              [g(r, "read_mibps", "read_mbps") for r in ordered],
              "Read Throughput", "MB/s", fmt="%.0f")

    ax4 = fig.add_subplot(gs[1, 1])
    plot_file_sizes(ax4, phases, ordered)

    # Row 3
    ax5 = fig.add_subplot(gs[2, 0])
    plot_timing_breakdown(ax5, phases, ordered)

    ax6 = fig.add_subplot(gs[2, 1])
    plot_ratio_distribution(ax6, phases, ordered)

    # Row 4
    ax7 = fig.add_subplot(gs[3, 0])
    plot_nn_stats(ax7, phases, ordered)

    ax8 = fig.add_subplot(gs[3, 1])
    plot_verification(ax8, phases, ordered)

    if meta_text:
        fig.text(0.5, 0.01, meta_text, ha="center", fontsize=9,
                 style="italic", color="gray")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified GPUCompress benchmark visualizer")
    parser.add_argument("--gs-csv", help="Gray-Scott aggregate CSV path")
    parser.add_argument("--vpic-csv", help="VPIC aggregate CSV path")
    parser.add_argument("--output-dir",
                        help="Output directory (default: same as CSV)")
    args = parser.parse_args()

    gs_path = args.gs_csv or find_csv(DEFAULT_GS_PATHS)
    vpic_path = args.vpic_csv or find_csv(DEFAULT_VPIC_PATHS)

    if not gs_path and not vpic_path:
        print("ERROR: No benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_GS_PATHS + DEFAULT_VPIC_PATHS:
            print(f"  {p}")
        print("\nRun benchmarks first, or specify paths with --gs-csv / --vpic-csv")
        sys.exit(1)

    # Output root: benchmarks/ directory (where this script lives)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Gray-Scott ──
    if gs_path and os.path.exists(gs_path):
        print(f"Loading Gray-Scott: {gs_path}")
        rows = parse_csv(gs_path)
        r0 = rows[0] if rows else {}
        L = int(g(r0, "L"))
        steps = int(g(r0, "steps"))
        F_val = g(r0, "F")
        k_val = g(r0, "k")
        orig = g(r0, "orig_mib", "orig_mb")
        n_ch = int(g(r0, "n_chunks"))
        chunk_mb = orig / max(n_ch, 1)
        meta = (f"Grid: {L}^3 ({orig:.0f} MB) | "
                f"Chunks: {n_ch} x {chunk_mb:.0f} MB | "
                f"Steps: {steps} | F={F_val}, k={k_val}")

        out_dir = os.path.join(args.output_dir or script_dir, "grayscott")
        out_png = os.path.join(out_dir, "benchmark_grayscott.png")
        make_figure("Gray-Scott Simulation", rows, out_png, meta)

    # ── VPIC ──
    if vpic_path and os.path.exists(vpic_path):
        print(f"Loading VPIC: {vpic_path}")
        rows = parse_csv(vpic_path)
        r0 = rows[0] if rows else {}
        orig = g(r0, "orig_mib", "orig_mb")
        n_ch = int(g(r0, "n_chunks"))
        meta = (f"Dataset: {orig:.0f} MB | "
                f"Chunks: {n_ch} | "
                f"Source: Real VPIC Harris Sheet Simulation")

        out_dir = os.path.join(args.output_dir or script_dir, "vpic")
        out_png = os.path.join(out_dir, "benchmark_vpic.png")
        make_figure("VPIC Harris Sheet Reconnection", rows, out_png, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
