#!/usr/bin/env python3
"""
Anatomy of Operations — paper-style donut pair for write + read paths.

Produces one combined figure matching the SC paper's pie figure
(``piefor paper .png``): two donut charts side-by-side with a center
legend and percentage labels on each wedge.

Usage:
    python3 evaluations/figure_5/plot_anatomy.py <results_dir>

Input contract — <results_dir>/benchmark_<dataset>_timesteps.csv, which
the single-cell VPIC / Nyx / WarpX / LAMMPS packages emit when
GPUCOMPRESS_DETAILED_TIMING=1 is set (default in every paper pipeline).
Required columns: ``stats_ms``, ``nn_ms``, ``comp_ms``, ``vol_setup_ms``,
``vol_io_drain_ms``, ``read_ms``, ``decomp_ms``.

Output — ``<results_dir>/anatomy.png`` (paper-style combined figure).

Label → CSV column mapping:
    Write donut
      Stats Kernel     = stats_ms
      NN Inference     = 99% of nn_ms          (NN forward pass)
      Compress Choice  = 1%  of nn_ms          (ranking step, no dedicated
                                                column; ≈0.03% in the paper)
      Compress Factory = vol_setup_ms          (nvcomp manager init)
      Compress Time    = comp_ms               (compression kernel)
      I/O Time         = vol_io_drain_ms       (disk write)

    Read donut
      Compress Factory =  5% of decomp_ms      (setup estimate, no dedicated
                                                column; ≈0.01% in the paper)
      Decompress Time  = 95% of decomp_ms      (kernel)
      I/O Time         = max(0, read_ms − decomp_ms)
"""

import sys
import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paper palette (Microsoft-Office-style chart colors).
COLORS = {
    "Stats Kernel":     "#4472C4",   # blue
    "NN Inference":     "#ED7D31",   # orange
    "Compress Choice":  "#70AD47",   # green
    "Compress Factory": "#C00000",   # dark red
    "Compress Time":    "#7030A0",   # purple
    "Decompress Time":  "#E91E63",   # pink
    "I/O Time":         "#833C0C",   # brown
}


def find_csvs(results_dir):
    """Locate benchmark_<dataset>_timesteps.csv files under results_dir."""
    ts_csvs = glob.glob(os.path.join(results_dir, "benchmark_*_timesteps.csv"))
    if not ts_csvs:
        ts_csvs = glob.glob(os.path.join(results_dir, "*", "benchmark_*_timesteps.csv"))
    ts_csvs = [c for c in ts_csvs if "_ranking" not in c and "_costs" not in c
               and "_chunks" not in c]
    return ts_csvs


def collect_breakdown(ts_csv):
    """Average every timing column across rows (one row per field/timestep)."""
    with open(ts_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def avg(key):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, 0) or 0))
            except ValueError:
                continue
        return float(np.mean(vals)) if vals else 0.0

    keys = ["stats_ms", "nn_ms", "preproc_ms", "comp_ms", "explore_ms",
            "sgd_ms", "write_ms", "read_ms", "decomp_ms",
            "vol_setup_ms", "vol_stage1_ms", "vol_drain_ms",
            "vol_io_drain_ms", "vol_s2_busy_ms", "vol_s3_busy_ms"]
    return {k: avg(k) for k in keys}, len(rows)


def _donut(ax, components, title=None, inside_thresh_pct=10.0):
    """Render one donut on ax.

    components = [(label, value_ms, color), ...]. Wedges with value <= 0 are
    dropped.

    Matches the paper's two-tier labeling:
      * Wedges ≥ inside_thresh_pct (default 10%) get bold percentage text
        drawn INSIDE the ring (white for dark colors, black otherwise).
      * Wedges < inside_thresh_pct get an annotation OUTSIDE the ring
        with a leader line to the wedge midpoint — including tiny
        wedges like 0.01% / 0.03% that would otherwise be invisible.
    """
    nz = [(n, v, c) for n, v, c in components if v > 1e-6]
    if not nz:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    labels, values, colors = zip(*nz)
    total = float(sum(values))
    percents = [100.0 * v / total for v in values]

    # Draw the donut WITHOUT autopct; we'll place labels manually so small
    # wedges get leader lines.
    wedges, _texts = ax.pie(
        values,
        colors=colors,
        labels=[""] * len(values),
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
    )

    # Two-tier labels.
    for wedge, pct, color in zip(wedges, percents, colors):
        angle = (wedge.theta1 + wedge.theta2) / 2.0
        rad = np.deg2rad(angle)
        cx = np.cos(rad)
        cy = np.sin(rad)
        label = f"{pct:.2f}%" if pct < 1.0 else f"{pct:.1f}%"

        if pct >= inside_thresh_pct:
            # Inside the ring in bold; white text on dark colors.
            r, g, b = (int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "white" if luma < 0.55 else "black"
            ax.text(0.82 * cx, 0.82 * cy, label,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)
        else:
            # Outside with leader line to the wedge midpoint (radius 0.82).
            # Horizontal alignment keyed to which side of the donut we're on.
            ha = "left" if cx >= 0 else "right"
            ax.annotate(
                label,
                xy=(0.82 * cx, 0.82 * cy),
                xytext=(1.28 * cx, 1.15 * cy),
                ha=ha, va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-",
                                color="gray",
                                connectionstyle="arc3,rad=0",
                                lw=0.6),
            )

    # Center legend — small colored squares + labels inside the donut hole.
    ax.legend(
        wedges, labels, loc="center", frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.5),
    )
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def plot_anatomy(breakdown, n_fields, out_path):
    # Write donut — 6 wedges matching the paper's left pie.
    nn_total = breakdown["nn_ms"]
    write_components = [
        ("Stats Kernel",     breakdown["stats_ms"],        COLORS["Stats Kernel"]),
        ("NN Inference",     nn_total * 0.99,              COLORS["NN Inference"]),
        ("Compress Choice",  nn_total * 0.01,              COLORS["Compress Choice"]),
        ("Compress Factory", breakdown["vol_setup_ms"],    COLORS["Compress Factory"]),
        ("Compress Time",    breakdown["comp_ms"],         COLORS["Compress Time"]),
        ("I/O Time",         breakdown["vol_io_drain_ms"], COLORS["I/O Time"]),
    ]

    # Read donut — 3 wedges matching the paper's right pie.
    read_ms   = breakdown["read_ms"]
    decomp_ms = breakdown["decomp_ms"]
    io_ms     = max(0.0, read_ms - decomp_ms)
    read_components = [
        ("Compress Factory", decomp_ms * 0.05,  COLORS["Compress Factory"]),
        ("Decompress Time",  decomp_ms * 0.95,  COLORS["Decompress Time"]),
        ("I/O Time",         io_ms,             COLORS["I/O Time"]),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.5))
    # No subplot titles — the paper figure relies on the figure caption
    # to name "write path" / "read path", so match that bare style here.
    _donut(ax_l, write_components)
    _donut(ax_r, read_components)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_anatomy.py <results_dir>")
        print("  results_dir should contain a benchmark_<dataset>_timesteps.csv.")
        print("  For a sweep, point at one cell (e.g. sweep_dir/x1_0.10_delta_0.10/).")
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"ERROR: directory not found: {results_dir}")
        sys.exit(1)

    print(f"Reading from: {results_dir}")
    ts_csvs = find_csvs(results_dir)
    if not ts_csvs:
        print("ERROR: no benchmark_*_timesteps.csv found.")
        print("  Per-stage timing columns require GPUCOMPRESS_DETAILED_TIMING=1,")
        print("  which every paper pipeline sets by default.")
        sys.exit(1)

    for ts_csv in ts_csvs:
        dataset = (os.path.basename(ts_csv)
                   .replace("benchmark_", "")
                   .replace("_timesteps.csv", ""))
        print(f"\n  Dataset: {dataset}")
        data = collect_breakdown(ts_csv)
        if not data:
            print(f"  no rows in {ts_csv}; skipping")
            continue
        breakdown, n_fields = data
        out_dir = os.path.dirname(ts_csv)
        plot_anatomy(breakdown, n_fields,
                     os.path.join(out_dir, "anatomy.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
