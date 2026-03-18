#!/usr/bin/env python3
"""
Unified benchmark visualizer for GPUCompress.

Combines two views into a single script:
  1. Aggregate summary       — bar charts of ratio, throughput, file sizes, NN stats
  2. Timestep adaptation     — MAPE over timesteps + per-chunk milestone detail

Usage:
  # Auto-detect all CSVs and generate all figures
  python3 benchmarks/visualize.py

  # Specific CSVs
  python3 benchmarks/visualize.py --gs-csv path/to/vol.csv

  # Only specific views
  python3 benchmarks/visualize.py --view summary --view timesteps
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
    "no-comp":     "#999999",
    "exhaustive":  "#5778a4",
    "nn":          "#e49444",
    "nn-rl":       "#6a9f58",
    "nn-rl+exp50": "#c85a5a",
}

PHASE_LABELS = {
    "no-comp":     "No Comp",
    "exhaustive":  "Exhaustive\nSearch",
    "nn":          "NN\n(Inference)",
    "nn-rl":       "NN+SGD",
    "nn-rl+exp50": "NN+SGD\n+Explore",
}

# Auto-detection paths
DEFAULT_GS_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol.csv"),
]
DEFAULT_VPIC_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic/benchmark_vpic_deck.csv"),
]
DEFAULT_GS_TIMESTEPS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_timesteps.csv"),
]
DEFAULT_GS_TSTEP_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_timestep_chunks.csv"),
]


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
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5,
                  width=0.6, zorder=3)
    if annotate:
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt % v, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_file_sizes(ax, phases, rows):
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    orig = [g(r, "orig_mib", "orig_mb") for r in rows]
    comp = [g(r, "file_mib", "file_mb") for r in rows]
    w = 0.3
    ax.bar(x - w / 2, orig, w, color="#cccccc", edgecolor="black", linewidth=0.5,
           label="Original", zorder=3)
    cbars = ax.bar(x + w / 2, comp, w, color=colors, edgecolor="black", linewidth=0.5,
                   label="Compressed", zorder=3)
    for bar, v in zip(cbars, comp):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(ax.containers[0], orig):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases], fontsize=9)
    ax.set_ylabel("Size (MB)", fontsize=10)
    ax.set_title("Original vs Compressed File Size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_nn_stats(ax, phases, rows):
    nn_idx = [i for i, p in enumerate(phases) if p in ("nn", "nn-rl", "nn-rl+exp50")]
    if not nn_idx:
        ax.text(0.5, 0.5, "No NN phases", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("NN Adaptation", fontsize=12, fontweight="bold")
        return
    nn_x = np.arange(len(nn_idx))
    nn_labels = [PHASE_LABELS.get(phases[i], phases[i]) for i in nn_idx]
    w = 0.3
    sgd_pct = [100.0 * g(rows[i], "sgd_fires") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    expl_pct = [100.0 * g(rows[i], "explorations") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    ax.bar(nn_x - w / 2, sgd_pct, w, color="#e49444", edgecolor="black", linewidth=0.5,
           label="SGD Fires %", zorder=3)
    ax.bar(nn_x + w / 2, expl_pct, w, color="#c85a5a", edgecolor="black", linewidth=0.5,
           label="Explorations %", zorder=3)
    ymax = max(max(sgd_pct, default=0), max(expl_pct, default=0))
    for xi, (s, e) in enumerate(zip(sgd_pct, expl_pct)):
        ax.text(xi - w / 2, s, f"{s:.0f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
        if e > 0:
            ax.text(xi + w / 2, e, f"{e:.0f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    ax.set_xticks(nn_x)
    ax.set_xticklabels(nn_labels, fontsize=9)
    ax.set_ylabel("% of chunks", fontsize=10)
    ax.set_ylim(0, ymax * 1.3 + 5)
    ax.set_title("Online Learning Activity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_verification(ax, phases, rows):
    ax.axis("off")
    cell_text = []
    cell_colors = []
    for i, p in enumerate(phases):
        r = rows[i]
        mm = int(g(r, "mismatches"))
        status = "PASS" if mm == 0 else f"FAIL ({mm})"
        verify_color = "#d5f5e3" if mm == 0 else "#fadbd8"
        orig = g(r, "orig_mib", "orig_mb")
        comp = g(r, "file_mib", "file_mb")
        ratio = g(r, "ratio")
        row_bg = "#f8f9fa" if i % 2 == 0 else "white"
        cell_text.append([PHASE_LABELS.get(p, p).replace("\n", " "),
                          f"{orig:.0f}", f"{comp:.1f}", f"{ratio:.2f}x", status])
        cell_colors.append([row_bg, row_bg, row_bg, row_bg, verify_color])
    table = ax.table(cellText=cell_text,
                     colLabels=["Phase", "Orig (MB)", "Comp (MB)", "Ratio", "Verify"],
                     cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#aaa")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=9)
            cell.set_facecolor("#d9e2ec")
        else:
            cell.set_text_props(fontsize=9)
    ax.set_title("Summary & Verification", fontsize=11, fontweight="bold", pad=20)


def make_summary_figure(source_name, rows, output_path, meta_text=""):
    phases, ordered = _ordered(rows)
    if not phases:
        print(f"  {source_name}: no valid phases, skipping summary.")
        return
    fig = plt.figure(figsize=(16, 14), facecolor="white")
    fig.text(0.5, 0.99, f"GPUCompress Benchmark: {source_name}",
             ha="center", fontsize=14, fontweight="bold", va="top")
    if meta_text:
        fig.text(0.5, 0.965, meta_text, ha="center", fontsize=9,
                 color="#444", va="top", fontfamily="monospace")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.30,
                           top=0.92, bottom=0.04, left=0.08, right=0.96)
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
# View 2: Timestep Adaptation (MAPE over timesteps)
# ═══════════════════════════════════════════════════════════════════════

def make_timestep_figure(ts_csv_path, output_path):
    """Plot MAPE for ratio/comp_time/decomp_time over all timesteps."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping.")
        return

    timesteps = np.array([g(r, "timestep") for r in rows])
    mape_r = np.array([g(r, "mape_ratio", default=-1) for r in rows])
    mape_c = np.array([g(r, "mape_comp", default=-1) for r in rows])
    mape_d = np.array([g(r, "mape_decomp", default=-1) for r in rows])
    has_mape = mape_r[0] >= 0
    if not has_mape:
        mape_r = np.array([g(r, "smape_ratio") for r in rows])
        mape_c = np.array([g(r, "smape_comp") for r in rows])
        mape_d = np.array([g(r, "smape_decomp") for r in rows])
    sgd_fires = np.array([g(r, "sgd_fires") for r in rows])

    metrics = [
        ("Compression Ratio",  mape_r, "#27ae60", "#2ecc71"),
        ("Compression Time",   mape_c, "#2471a3", "#3498db"),
        ("Decompression Time", mape_d, "#c0392b", "#e74c3c"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Online SGD Prediction Accuracy Over Timesteps",
                 fontsize=15, fontweight="bold", y=0.98)

    n_ts = len(timesteps)
    bar_w = max(0.6, (timesteps[-1] - timesteps[0]) / n_ts * 0.75) if n_ts > 1 else 0.8

    for ax, (label, mape, dark, light) in zip(axes, metrics):
        clipped = np.clip(mape, 0, 200)

        # Gradient-colored bars: green < 20%, orange 20-50%, red > 50%
        colors = [("#27ae60" if v <= 20 else "#f39c12" if v <= 50 else "#e74c3c")
                  for v in mape]
        bars = ax.bar(timesteps, clipped, color=colors, edgecolor="white",
                       linewidth=0.4, alpha=0.85, width=bar_w)

        # Value labels on bars
        for bar, val in zip(bars, mape):
            if val > 0 and val <= 180:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{val:.0f}%", ha="center", va="bottom",
                        fontsize=7, color="#444444")
            elif val > 200:
                ax.text(bar.get_x() + bar.get_width() / 2, 192,
                        f"{val:.0f}%", ha="center", va="top",
                        fontsize=6, color="white", fontweight="bold")

        # Trend line
        if n_ts >= 5:
            w = min(5, n_ts)
            kernel = np.ones(w) / w
            rolling = np.clip(np.convolve(mape, kernel, mode="valid"), 0, 200)
            x_roll = timesteps[w - 1:]
            ax.plot(x_roll, rolling, color="#2c3e50", linewidth=2.5, alpha=0.9,
                    label=f"Trend (w={w})", zorder=5)

        # 20% target
        ax.axhline(20, color="#e67e22", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(timesteps[-1] + bar_w, 20, " 20% target", fontsize=8,
                color="#e67e22", va="center", fontweight="bold")

        ax.set_ylabel(f"{label}\nMAPE (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 200)
        ax.set_xlim(timesteps[0] - bar_w, timesteps[-1] + bar_w * 3)
        ax.grid(axis="y", alpha=0.2, linestyle="-")
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

        # Stats annotation
        best = np.min(mape)
        avg_all = np.mean(mape)
        ax.text(0.01, 0.95,
                f"Start: {mape[0]:.0f}%    Best: {best:.0f}%    Avg: {avg_all:.0f}%",
                transform=ax.transAxes, fontsize=9, ha="left", va="top",
                fontfamily="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb",
                          boxstyle="round,pad=0.4"))

    axes[-1].set_xlabel("Timestep", fontsize=11, fontweight="bold")

    # SGD fires as subtle secondary axis on bottom panel
    ax2 = axes[-1].twinx()
    ax2.bar(timesteps, sgd_fires, width=bar_w * 0.3, color="#95a5a6",
            alpha=0.4, zorder=1, label="SGD fires")
    ax2.set_ylabel("SGD Fires", fontsize=9, color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=8)
    ax2.set_ylim(0, max(sgd_fires) * 3 if max(sgd_fires) > 0 else 10)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_timestep_chunks_figure(tc_csv_path, output_path):
    """Plot per-chunk MAPE at milestone timesteps from timestep_chunks CSV."""
    rows = parse_csv(tc_csv_path)
    if not rows:
        print(f"  No timestep chunk data in {tc_csv_path}, skipping.")
        return

    # Group by timestep
    by_ts = {}
    for r in rows:
        ts = int(g(r, "timestep"))
        by_ts.setdefault(ts, []).append(r)
    all_ts = sorted(by_ts.keys())
    if len(all_ts) == 0:
        return
    # Pick 5 milestones: 0%, 25%, 50%, 75%, 100% of available timesteps
    indices = [0, len(all_ts) // 4, len(all_ts) // 2, 3 * len(all_ts) // 4, len(all_ts) - 1]
    seen = set()
    timestep_list = [all_ts[i] for i in indices if not (all_ts[i] in seen or seen.add(all_ts[i]))]
    n_ts = len(timestep_list)

    metrics = [
        ("Compression Ratio",  "predicted_ratio",    "actual_ratio",      "x",  "mape_ratio"),
        ("Comp Time",          "predicted_comp_ms",  "actual_comp_ms",    "ms", "mape_comp"),
        ("Decomp Time",        "predicted_decomp_ms","actual_decomp_ms",  "ms", "mape_decomp"),
    ]

    fig, axes = plt.subplots(n_ts, 3, figsize=(20, 3.2 * n_ts + 2),
                              squeeze=False)
    fig.suptitle("NN Predicted vs Actual Per Chunk at Milestone Timesteps",
                 fontsize=14, fontweight="bold", y=0.99)

    for row_idx, ts in enumerate(timestep_list):
        chunk_rows = by_ts[ts]
        chunks = np.array([int(g(r, "chunk")) for r in chunk_rows])
        sort_idx = np.argsort(chunks)
        chunks = chunks[sort_idx]

        for col_idx, (label, pred_key, act_key, unit, mape_key) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            pred = np.array([g(r, pred_key) for r in chunk_rows])[sort_idx]
            act = np.array([g(r, act_key) for r in chunk_rows])[sort_idx]
            mape_vals = np.array([g(r, mape_key) for r in chunk_rows])[sort_idx]

            # Lines with shaded error region
            ax.plot(chunks, act, color="#2c3e50", linewidth=1.8, label="Actual",
                    zorder=3)
            ax.plot(chunks, pred, color="#3498db", linewidth=1.5, linestyle="--",
                    alpha=0.9, label="Predicted", zorder=3)
            ax.fill_between(chunks, act, pred, color="#3498db", alpha=0.15,
                            zorder=2)

            # Highlight chunks with MAPE > 50%
            bad = mape_vals > 50
            if np.any(bad):
                ax.scatter(chunks[bad], pred[bad], color="#e74c3c", s=20,
                           zorder=4, marker="x", linewidths=1.5)

            if row_idx == 0:
                ax.set_title(f"{label} ({unit})", fontsize=12, fontweight="bold")
            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
            if row_idx == n_ts - 1:
                ax.set_xlabel("Chunk Index", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"T={ts}", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.2, linestyle="-")
            ax.tick_params(labelsize=8)

            # Cap y-axis
            all_vals = np.concatenate([pred, act])
            pos = all_vals[all_vals > 0]
            p95 = np.percentile(pos, 95) if len(pos) > 0 else 1
            ax.set_ylim(0, p95 * 1.6)

            # Stats box
            avg_mape = np.mean(mape_vals)
            median_mape = np.median(mape_vals)
            ax.text(0.98, 0.95,
                    f"MAPE: avg={avg_mape:.0f}%  med={median_mape:.0f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb",
                              boxstyle="round,pad=0.3"))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

ALL_VIEWS = ["summary", "timesteps"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified GPUCompress benchmark visualizer (summary + timesteps)")
    parser.add_argument("csvs", nargs="*",
                        help="Positional CSV paths (auto-classified as gs/vpic)")
    parser.add_argument("--gs-csv", help="Gray-Scott aggregate CSV")
    parser.add_argument("--gs-timesteps-csv", help="Gray-Scott timestep adaptation CSV")
    parser.add_argument("--vpic-csv", help="VPIC aggregate CSV")
    parser.add_argument("--output-dir", help="Output directory (default: alongside CSV)")
    parser.add_argument("--view", action="append", default=None,
                        help=f"Views to generate: {ALL_VIEWS} (default: all)")
    args = parser.parse_args()

    views = args.view or ALL_VIEWS

    # Classify positional CSVs
    for csv_path in args.csvs:
        low = csv_path.lower()
        is_vpic = "vpic" in low
        if is_vpic:
            if not args.vpic_csv:
                args.vpic_csv = csv_path
        else:
            if not args.gs_csv:
                args.gs_csv = csv_path

    # Auto-detect
    gs_agg = args.gs_csv or find_csv(DEFAULT_GS_AGG)
    gs_tsteps = args.gs_timesteps_csv or find_csv(DEFAULT_GS_TIMESTEPS)
    vpic_agg = args.vpic_csv or find_csv(DEFAULT_VPIC_AGG)

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

    if gs_tsteps and os.path.exists(gs_tsteps) and "timesteps" in views:
        found_any = True
        print(f"Loading Gray-Scott timestep adaptation: {gs_tsteps}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(gs_tsteps))
        make_timestep_figure(gs_tsteps, os.path.join(out_dir, "sgd_accuracy_over_time.png"))

    gs_tc = find_csv(DEFAULT_GS_TSTEP_CHUNKS)
    if gs_tc and os.path.exists(gs_tc) and "timesteps" in views:
        found_any = True
        print(f"Loading Gray-Scott per-chunk milestones: {gs_tc}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(gs_tc))
        try:
            make_timestep_chunks_figure(gs_tc, os.path.join(out_dir, "predicted_vs_actual_per_chunk.png"))
        except Exception as e:
            print(f"  Warning: per-chunk milestone plot failed: {e}")

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

    if not found_any:
        print("ERROR: No benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_GS_AGG + DEFAULT_VPIC_AGG:
            print(f"  {p}")
        print("\nRun benchmarks first, or specify paths explicitly.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
