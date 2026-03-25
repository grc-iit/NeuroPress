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
import matplotlib.patheffects
import matplotlib.patches
import numpy as np

# ── SC publication-quality global defaults ──
plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size':            11,
    'axes.titlesize':       13,
    'axes.labelsize':       12,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      10,
    'figure.dpi':           300,
    'savefig.dpi':          300,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
})

# ═══════════════════════════════════════════════════════════════════════
# Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PHASE_ORDER = [
    "no-comp",
    # With fixed- prefix (timestep-major deck)
    "fixed-lz4", "fixed-snappy", "fixed-deflate",
    "fixed-gdeflate", "fixed-zstd",
    "fixed-ans", "fixed-cascaded", "fixed-bitcomp",
    # Without fixed- prefix (phase-major deck)
    "lz4", "snappy", "deflate",
    "gdeflate", "zstd",
    "ans", "cascaded", "bitcomp",
    # NN phases
    "nn", "nn-rl", "nn-rl+exp50",
]

PHASE_COLORS = {
    "no-comp":                "#7f8c8d",   # muted grey
    "fixed-lz4":              "#3498db",   # blue
    "fixed-snappy":           "#5dade2",   # light blue
    "fixed-deflate":          "#2e86c1",   # medium blue
    "fixed-gdeflate":         "#2980b9",   # darker blue
    "fixed-zstd":             "#1a5276",   # dark blue
    "fixed-ans":              "#148f77",   # teal
    "fixed-cascaded":         "#1abc9c",   # turquoise
    "fixed-bitcomp":          "#0e6655",   # dark teal
    # Same colors without fixed- prefix
    "lz4":                    "#3498db",
    "snappy":                 "#5dade2",
    "deflate":                "#2e86c1",
    "gdeflate":               "#2980b9",
    "zstd":                   "#1a5276",
    "ans":                    "#148f77",
    "cascaded":               "#1abc9c",
    "bitcomp":                "#0e6655",
    "nn":                     "#e67e22",   # orange
    "nn-rl":                  "#8e44ad",   # purple
    "nn-rl+exp50":            "#c0392b",   # red
}

# Hatching patterns for accessibility (colorblind-friendly)
PHASE_HATCHES = {
    "no-comp":                "",
    "fixed-lz4":              "",  "lz4":        "",
    "fixed-snappy":           "",  "snappy":     "",
    "fixed-deflate":          "",  "deflate":    "",
    "fixed-gdeflate":         "",  "gdeflate":   "",
    "fixed-zstd":             "",  "zstd":       "",
    "fixed-ans":              "",  "ans":        "",
    "fixed-cascaded":         "",  "cascaded":   "",
    "fixed-bitcomp":          "",  "bitcomp":    "",
    "nn":                     "",
    "nn-rl":                  "\\\\",
    "nn-rl+exp50":            "xx",
}

# Line styles for line plots (accessibility)
PHASE_LINESTYLES = {
    "nn":          ":",
    "nn-rl":       "-",
    "nn-rl+exp50": "--",
}

PHASE_LABELS = {
    "no-comp":                "No Comp",
    "fixed-lz4":              "LZ4",       "lz4":        "LZ4",
    "fixed-snappy":           "Snappy",    "snappy":     "Snappy",
    "fixed-deflate":          "Deflate",   "deflate":    "Deflate",
    "fixed-gdeflate":         "GDeflate",  "gdeflate":   "GDeflate",
    "fixed-zstd":             "Zstd",      "zstd":       "Zstd",
    "fixed-ans":              "ANS",       "ans":        "ANS",
    "fixed-cascaded":         "Cascaded",  "cascaded":   "Cascaded",
    "fixed-bitcomp":          "Bitcomp",   "bitcomp":    "Bitcomp",
    "nn":                     "NN\n(Inference)",
    "nn-rl":                  "NN+SGD",
    "nn-rl+exp50":            "NN+SGD\n+Explore",
}

# Auto-detection paths
DEFAULT_GS_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol.csv"),
    os.path.join(PROJECT_ROOT, "results/benchmark_grayscott_vol.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol.csv"),
]
DEFAULT_VPIC_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic/benchmark_vpic_deck.csv"),
]
DEFAULT_GS_TIMESTEPS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "results/benchmark_grayscott_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_timesteps.csv"),
]
DEFAULT_GS_TSTEP_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_timestep_chunks.csv"),
    os.path.join(PROJECT_ROOT, "results/benchmark_grayscott_timestep_chunks.csv"),
]
DEFAULT_GS_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol_chunks.csv"),
    os.path.join(PROJECT_ROOT, "results/benchmark_grayscott_vol_chunks.csv"),
]
DEFAULT_VPIC_TIMESTEPS = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_timesteps.csv"),
]
DEFAULT_VPIC_TSTEP_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck_timestep_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_timestep_chunks.csv"),
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


def _normalize_rows(rows):
    """Ensure rows have orig_mib/file_mib (convert from bytes if needed)."""
    for r in rows:
        if "orig_mib" not in r and "orig_mb" not in r and "orig_bytes" in r:
            r["orig_mib"] = r["orig_bytes"] / (1024 * 1024)
        if "file_mib" not in r and "file_mb" not in r and "file_bytes" in r:
            r["file_mib"] = r["file_bytes"] / (1024 * 1024)
        if "write_mibps" not in r and "write_mbps" in r:
            r["write_mibps"] = r["write_mbps"]
        if "read_mibps" not in r and "read_mbps" in r:
            r["read_mibps"] = r["read_mbps"]
        if "dataset" not in r and "source" in r:
            r["dataset"] = r["source"]
        if "mape_ratio" not in r and "mape_ratio_pct" in r:
            r["mape_ratio"] = r["mape_ratio_pct"]
        if "mape_comp" not in r and "mape_comp_pct" in r:
            r["mape_comp"] = r["mape_comp_pct"]
        if "mape_decomp" not in r and "mape_decomp_pct" in r:
            r["mape_decomp"] = r["mape_decomp_pct"]
    return rows


_PHASE_ALIASES = {}


def _avg_all(all_rows, key):
    """Average a numeric field across all rows."""
    return sum(g(r, key) for r in all_rows) / len(all_rows)


def _merge_timestep_phases(rows, ts_csv_path, orig_mib):
    """Merge multi-timestep phases into summary using total throughput.

    Throughput = (N_timesteps * orig_mib) / sum(write_ms or read_ms).
    This is the standard HPC I/O metric: total data / total time.
    """
    if not ts_csv_path or not os.path.exists(ts_csv_path):
        return
    ts_rows = parse_csv(ts_csv_path)
    by_phase = {}
    for r in ts_rows:
        by_phase.setdefault(r.get("phase", ""), []).append(r)
    merged = []
    for ph in ["nn-rl", "nn-rl+exp50"]:
        if ph not in by_phase:
            continue
        ph_rows = by_phase[ph]
        n_ts = len(ph_rows)
        total_data_mib = n_ts * orig_mib
        total_write_s = sum(g(r, "write_ms") for r in ph_rows) / 1000.0
        total_read_s = sum(g(r, "read_ms") for r in ph_rows) / 1000.0
        ratio_avg = _avg_all(ph_rows, "ratio")
        total_sgd = sum(int(g(r, "sgd_fires")) for r in ph_rows)
        total_expl = sum(int(g(r, "explorations")) for r in ph_rows)
        total_chunks = sum(int(g(r, "n_chunks")) for r in ph_rows)
        rows.append({
            "phase": ph,
            "ratio": ratio_avg,
            "write_mibps": total_data_mib / total_write_s if total_write_s > 0 else 0,
            "read_mibps": total_data_mib / total_read_s if total_read_s > 0 else 0,
            "orig_mib": orig_mib,
            "file_mib": orig_mib / max(ratio_avg, 1e-6),
            "n_chunks": total_chunks,
            "sgd_fires": total_sgd,
            "explorations": total_expl,
            "mismatches": 0,
            "mape_ratio_pct": _avg_all(ph_rows, "mape_ratio"),
            "mape_comp_pct": _avg_all(ph_rows, "mape_comp"),
            "mape_decomp_pct": _avg_all(ph_rows, "mape_decomp"),
        })
        merged.append(ph)
    if merged:
        print(f"  Merged multi-timestep phases (total throughput): {merged}")


def _ordered(rows):
    """Return rows ordered by PHASE_ORDER."""
    _normalize_rows(rows)
    by_phase = {}
    for r in rows:
        phase = _PHASE_ALIASES.get(r["phase"], r["phase"])
        r["phase"] = phase
        by_phase[phase] = r
    phases = [p for p in PHASE_ORDER if p in by_phase]
    return phases, [by_phase[p] for p in phases]


# ═══════════════════════════════════════════════════════════════════════
# SC Figure Styling Helpers
# ═══════════════════════════════════════════════════════════════════════

def _sc_figure_border(fig):
    """Add subtle figure border."""
    fig.patch.set_edgecolor('#cccccc')
    fig.patch.set_linewidth(0.5)


def _sc_watermark(fig):
    """Add subtle GPUCompress watermark."""
    fig.text(0.99, 0.01, "GPUCompress", fontsize=10, color='grey',
             alpha=0.05, ha='right', va='bottom',
             fontweight='bold', fontstyle='italic')


def _sc_finalize(fig, pad=1.5, rect=None):
    """Common finalization: tight_layout, border, watermark."""
    if rect:
        fig.tight_layout(pad=pad, rect=rect)
    else:
        fig.tight_layout(pad=pad)
    _sc_figure_border(fig)
    _sc_watermark(fig)


# ═══════════════════════════════════════════════════════════════════════
# View 1: Aggregate Summary
# ═══════════════════════════════════════════════════════════════════════

def plot_bars(ax, phases, values, title, ylabel, fmt="%.2f", annotate=True):
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    hatches = [PHASE_HATCHES.get(p, "") for p in phases]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4,
                  width=0.6, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    if annotate:
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases], rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))


def plot_file_sizes(ax, phases, rows):
    x = np.arange(len(phases))
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    hatches = [PHASE_HATCHES.get(p, "") for p in phases]
    orig = [g(r, "orig_mib", "orig_mb") for r in rows]
    comp = [g(r, "file_mib", "file_mb") for r in rows]
    w = 0.3
    orig_bars = ax.bar(x - w / 2, orig, w, color="#cccccc", edgecolor="black",
                       linewidth=0.4, label="Original", zorder=3)
    cbars = ax.bar(x + w / 2, comp, w, color=colors, edgecolor="black",
                   linewidth=0.4, label="Compressed", zorder=3)
    for bar, h in zip(cbars, hatches):
        bar.set_hatch(h)
    for bar, v in zip(cbars, comp):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(orig_bars, orig):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases], rotation=30, ha='right')
    ax.set_ylabel("Size (MiB)")
    ax.set_title("Original vs Compressed File Size", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))


def plot_nn_stats(ax, phases, rows):
    nn_idx = [i for i, p in enumerate(phases) if p in ("nn", "nn-rl", "nn-rl+exp50")]
    if not nn_idx:
        ax.text(0.5, 0.5, "No NN phases", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("NN Adaptation", fontweight="bold")
        return
    nn_x = np.arange(len(nn_idx))
    nn_labels = [PHASE_LABELS.get(phases[i], phases[i]) for i in nn_idx]
    w = 0.3
    sgd_pct = [100.0 * g(rows[i], "sgd_fires") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    expl_pct = [100.0 * g(rows[i], "explorations") / max(g(rows[i], "n_chunks"), 1) for i in nn_idx]
    sgd_bars = ax.bar(nn_x - w / 2, sgd_pct, w, color="#e67e22", edgecolor="black",
                      linewidth=0.4, label="SGD Fires (%)", zorder=3)
    exp_bars = ax.bar(nn_x + w / 2, expl_pct, w, color="#c0392b", edgecolor="black",
                      linewidth=0.4, label="Explorations (%)", zorder=3, hatch="//")
    ymax = max(max(sgd_pct, default=0), max(expl_pct, default=0))
    for xi, (s, e) in enumerate(zip(sgd_pct, expl_pct)):
        ax.text(xi - w / 2, s, f"{s:.0f}%", ha="center", va="bottom", fontsize=8)
        if e > 0:
            ax.text(xi + w / 2, e, f"{e:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(nn_x)
    ax.set_xticklabels(nn_labels, rotation=30, ha='right')
    ax.set_ylabel("Chunks (%)")
    ax.set_ylim(0, ymax * 1.3 + 5)
    ax.set_title("Online Learning Activity", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)


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
                     colLabels=["Phase", "Orig (MiB)", "Comp (MiB)", "Ratio", "Verify"],
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
    ax.set_title("Summary & Verification", fontweight="bold", pad=20)


def make_summary_figure(source_name, rows, output_path, meta_text=""):
    phases, ordered = _ordered(rows)
    if not phases:
        print(f"  {source_name}: no valid phases, skipping summary.")
        return
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.text(0.5, 0.99, f"GPUCompress Benchmark: {source_name}",
             ha="center", fontsize=14, fontweight="bold", va="top")
    if meta_text:
        fig.text(0.5, 0.965, meta_text, ha="center", fontsize=9,
                 color="#444", va="top", fontfamily="monospace")
    gs = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.30,
                           top=0.92, bottom=0.08, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bars(ax1, phases, [g(r, "ratio") for r in ordered],
              "Compression Ratio (higher = better)", "Ratio", fmt="%.2fx")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_bars(ax2, phases, [g(r, "write_mibps", "write_mbps") for r in ordered],
              "End-to-End Write Throughput (full pipeline)", "MiB/s", fmt="%.0f")
    ax3 = fig.add_subplot(gs[1, 0])
    plot_bars(ax3, phases, [g(r, "read_mibps", "read_mbps") for r in ordered],
              "End-to-End Read Throughput (full pipeline)", "MiB/s", fmt="%.0f")

    # ── Pareto scatter: Ratio vs Write Throughput ──
    ax4 = fig.add_subplot(gs[1, 1])
    for p, r in zip(phases, ordered):
        ratio = g(r, "ratio")
        wtp = g(r, "write_mibps", "write_mbps")
        color = PHASE_COLORS.get(p, "#bdc3c7")
        label = PHASE_LABELS.get(p, p).replace("\n", " ")
        ax4.scatter(ratio, wtp, s=120, color=color, edgecolors="black",
                    linewidth=0.8, zorder=5, label=label,
                    marker={'nn': 'D', 'nn-rl': 's', 'nn-rl+exp50': '^'}.get(p, 'o'))
        ax4.annotate(label, (ratio, wtp), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color="#333")
    ax4.set_xlabel("Compression Ratio")
    ax4.set_ylabel("Write Throughput (MiB/s)")
    ax4.set_title("Ratio vs Throughput (Pareto)", fontweight="bold")
    ax4.grid(alpha=0.2, linestyle="--", zorder=0)
    ax4.set_axisbelow(True)
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View 2: Timestep Adaptation (MAPE over timesteps)
# ═══════════════════════════════════════════════════════════════════════

def make_timestep_figure(ts_csv_path, output_path):
    """Plot MAPE for ratio/comp_time/decomp_time over timesteps, one line per phase."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping.")
        return

    # Detect whether this is timestep-based or field-based data
    has_field_idx = "field_idx" in rows[0] if rows else False
    has_field_name = "field_name" in rows[0] if rows else False
    has_timestep = "timestep" in rows[0] if rows else False
    is_field_based = has_field_idx and not has_timestep
    x_label = "Field" if is_field_based else "Timestep"

    # Collect field names for x-tick labels if available
    field_names = []
    if has_field_name:
        seen = set()
        for r in rows:
            fn = r.get("field_name", "")
            idx = r.get("field_idx", r.get("timestep", ""))
            key = f"{idx}_{fn}"
            if key not in seen:
                seen.add(key)
                # Shorten: remove extension, truncate long names
                short = fn.replace(".f32", "").replace(".dat", "").replace(".bin", "")
                if len(short) > 20:
                    short = short[:18] + ".."
                field_names.append(short)

    # Group by phase (new CSV has 'phase' column; old CSV doesn't)
    has_phase = "phase" in rows[0] if rows else False
    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl") if has_phase else "nn-rl"
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#7f8c8d", "ls": ":",    "marker": "s", "lw": 2.0},
        "nn-rl":       {"color": "#8e44ad", "ls": "-",    "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c0392b", "ls": "--",   "marker": "D", "lw": 2.0},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    metric_keys = [
        ("Compression Ratio",  "mape_ratio"),
        ("Compression Time",   "mape_comp"),
        ("Decompression Time", "mape_decomp"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    title_suffix = "Fields" if is_field_based else "Timesteps"
    fig.suptitle(f"NN Prediction Accuracy Over {title_suffix}\n"
                 "(per-metric MAPE averaged across all chunks)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ax, (label, mape_key) in zip(axes, metric_keys):
        for ph in phases_present:
            ph_rows = by_phase[ph]
            timesteps = np.array([g(r, "timestep", "field_idx") for r in ph_rows])
            mape = np.array([g(r, mape_key) for r in ph_rows])
            clipped = np.clip(mape, 0, 200)

            sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})
            ax.plot(timesteps, clipped, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=6, linewidth=2.0,
                    label=ph, alpha=0.9, zorder=3)

        ax.axhline(20, color="#e67e22", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_ylabel(f"{label}\nMAPE (%)", fontweight="bold")
        ax.set_ylim(0, 200)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(x_label, fontweight="bold")

    # Use field names as x-tick labels if available
    if is_field_based and field_names:
        for ax in axes:
            n_fields = len(field_names)
            ax.set_xticks(range(n_fields))
            ax.set_xticklabels(field_names, rotation=30, ha="right", fontsize=9)

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_mae_figure(ts_csv_path, output_path):
    """Plot MAE for ratio/comp_time/decomp_time over timesteps, one line per phase."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping MAE plot.")
        return

    # Check that MAE columns exist
    if "mae_ratio" not in rows[0]:
        print(f"  No MAE columns in {ts_csv_path}, skipping MAE plot.")
        return

    is_field_based = "field_idx" in rows[0] and "timestep" not in rows[0]
    x_label = "Field" if is_field_based else "Timestep"

    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#7f8c8d", "ls": ":",  "marker": "s", "lw": 2.0},
        "nn-rl":       {"color": "#8e44ad", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c0392b", "ls": "--", "marker": "D", "lw": 2.0},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    metric_keys = [
        ("Compression Ratio", "mae_ratio", ""),
        ("Compression Time",  "mae_comp_ms", " (ms)"),
        ("Decompression Time", "mae_decomp_ms", " (ms)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    title_suffix = "Fields" if is_field_based else "Timesteps"
    fig.suptitle(f"NN Prediction MAE Over {title_suffix}\n"
                 "(per-metric Mean Absolute Error averaged across all chunks)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ax, (label, mae_key, unit) in zip(axes, metric_keys):
        for ph in phases_present:
            ph_rows = by_phase[ph]
            timesteps = np.array([g(r, "timestep", "field_idx") for r in ph_rows])
            mae = np.array([g(r, mae_key) for r in ph_rows])

            sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})
            ax.plot(timesteps, mae, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=6, linewidth=2.0,
                    label=ph, alpha=0.9, zorder=3)

        ax.set_ylabel(f"{label}\nMAE{unit}", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(x_label, fontweight="bold")

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_r2_figure(ts_csv_path, output_path):
    """Plot R² score for compression ratio prediction over timesteps."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping R² plot.")
        return

    if "r2_ratio" not in rows[0]:
        print(f"  No R² column in {ts_csv_path}, skipping R² plot.")
        return

    is_field_based = "field_idx" in rows[0] and "timestep" not in rows[0]
    x_label = "Field" if is_field_based else "Timestep"

    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#7f8c8d", "ls": ":",  "marker": "s", "lw": 2.0},
        "nn-rl":       {"color": "#8e44ad", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c0392b", "ls": "--", "marker": "D", "lw": 2.0},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    title_suffix = "Fields" if is_field_based else "Timesteps"

    # Collect all data first
    phase_data = {}
    for ph in phases_present:
        ph_rows = by_phase[ph]
        timesteps = np.array([g(r, "timestep", "field_idx") for r in ph_rows])
        r2 = np.array([g(r, "r2_ratio") for r in ph_rows])
        phase_data[ph] = (timesteps, r2)

    # Check if we need a split view (nn phase has very negative R²)
    if not phase_data:
        return
    all_r2 = np.concatenate([r2 for _, r2 in phase_data.values()])
    if len(all_r2) == 0:
        return
    needs_split = np.min(all_r2) < -2.0 and np.max(all_r2) > 0.5

    if needs_split:
        fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8),
                                                 gridspec_kw={"height_ratios": [1, 2]})
        fig.suptitle(f"Compression Ratio Prediction R² Over {title_suffix}\n"
                     "(coefficient of determination, per-timestep two-pass computation)",
                     fontsize=14, fontweight="bold", y=0.99)

        for ph in phases_present:
            ts, r2 = phase_data[ph]
            sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})
            for ax in (ax_full, ax_zoom):
                ax.plot(ts, r2, color=sty["color"], linestyle=sty["ls"],
                        marker=sty["marker"], markersize=5, linewidth=2.0,
                        label=ph, alpha=0.9, zorder=3)

        # Full range panel
        ax_full.axhline(0.0, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.4)
        ax_full.set_ylabel("R² (full range)", fontweight="bold")
        ax_full.grid(axis="y", alpha=0.2, linestyle="--")
        ax_full.legend(loc="lower right", fontsize=9)
        ax_full.set_title("All phases (full scale)", fontsize=10, loc="left")

        # Zoomed panel — focus on learning phases
        ax_zoom.axhline(1.0, color="#27ae60", linewidth=1, linestyle="--", alpha=0.6, label="Perfect (R²=1)")
        ax_zoom.axhline(0.0, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.4, label="R²=0 (mean baseline)")
        ax_zoom.set_ylim(-0.5, 1.05)
        ax_zoom.set_ylabel("R² (zoomed)", fontweight="bold")
        ax_zoom.set_xlabel(x_label, fontweight="bold")
        ax_zoom.grid(axis="y", alpha=0.2, linestyle="--")
        ax_zoom.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
        ax_zoom.minorticks_on()
        ax_zoom.legend(loc="lower right", fontsize=9)
        ax_zoom.set_title("Learning phases detail (R² > -0.5)", fontsize=10, loc="left")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        fig.suptitle(f"Compression Ratio Prediction R² Over {title_suffix}\n"
                     "(coefficient of determination, per-timestep two-pass computation)",
                     fontsize=14, fontweight="bold", y=0.99)

        for ph in phases_present:
            ts, r2 = phase_data[ph]
            sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})
            ax.plot(ts, r2, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=6, linewidth=2.0,
                    label=ph, alpha=0.9, zorder=3)

        ax.axhline(1.0, color="#27ae60", linewidth=1, linestyle="--", alpha=0.6, label="Perfect (R²=1)")
        ax.axhline(0.0, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.4, label="R²=0 (mean baseline)")
        ax.set_ylabel("R² Score", fontweight="bold")
        ax.set_xlabel(x_label, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.legend(loc="lower right")

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_ranking_quality_figure(ranking_csv_path, output_path):
    """Plot Kendall tau-b, top-1 accuracy, and top-1 regret over milestone timesteps."""
    rows = parse_csv(ranking_csv_path)
    if not rows:
        print(f"  No ranking data in {ranking_csv_path}, skipping.")
        return

    if "kendall_tau_b" not in rows[0]:
        print(f"  No kendall_tau_b column in {ranking_csv_path}, skipping.")
        return

    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#7f8c8d", "ls": ":",  "marker": "s", "lw": 2.0},
        "nn-rl":       {"color": "#8e44ad", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c0392b", "ls": "--", "marker": "D", "lw": 2.0},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    # Aggregate per-phase per-timestep: mean tau, top1 accuracy, mean regret
    def aggregate(ph_rows):
        ts_map = {}
        for r in ph_rows:
            t = int(g(r, "timestep"))
            ts_map.setdefault(t, []).append(r)
        timesteps = sorted(ts_map.keys())
        mean_tau, std_tau, top1_acc, mean_regret = [], [], [], []
        for t in timesteps:
            chunk_rows = ts_map[t]
            taus = [g(r, "kendall_tau_b") for r in chunk_rows]
            top1s = [g(r, "top1_correct") for r in chunk_rows]
            regrets = [g(r, "top1_regret") for r in chunk_rows]
            mean_tau.append(np.mean(taus))
            std_tau.append(np.std(taus))
            top1_acc.append(np.mean(top1s) * 100.0)
            mean_regret.append(np.mean(regrets))
        return np.array(timesteps), np.array(mean_tau), np.array(std_tau), np.array(top1_acc), np.array(mean_regret)

    fig, (ax_tau, ax_top1, ax_regret) = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("NN Algorithm Ranking Quality Over Timesteps\n"
                 "(Kendall \u03c4-b, Top-1 Accuracy, Top-1 Regret at milestones)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ph in phases_present:
        ts, mt, st, t1, mr = aggregate(by_phase[ph])
        sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})

        ax_tau.plot(ts, mt, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=6, linewidth=2.0,
                    label=ph, alpha=0.9, zorder=3)
        ax_tau.fill_between(ts, mt - st, mt + st, color=sty["color"], alpha=0.1)

        ax_top1.plot(ts, t1, color=sty["color"], linestyle=sty["ls"],
                     marker=sty["marker"], markersize=6, linewidth=2.0,
                     label=ph, alpha=0.9, zorder=3)

        ax_regret.plot(ts, mr, color=sty["color"], linestyle=sty["ls"],
                       marker=sty["marker"], markersize=6, linewidth=2.0,
                       label=ph, alpha=0.9, zorder=3)

    # Tau panel
    ax_tau.axhline(1.0, color="#27ae60", linewidth=1, linestyle="--", alpha=0.4, label="Perfect (\u03c4=1)")
    ax_tau.axhline(0.0, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.4, label="Random (\u03c4=0)")
    ax_tau.set_ylabel("Kendall \u03c4-b", fontweight="bold")
    ax_tau.set_ylim(-0.2, 1.05)
    ax_tau.grid(axis="y", alpha=0.2, linestyle="--")
    ax_tau.legend(loc="lower right", fontsize=9)

    # Top-1 panel
    ax_top1.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax_top1.set_ylim(-5, 105)
    ax_top1.grid(axis="y", alpha=0.2, linestyle="--")
    ax_top1.legend(loc="lower right", fontsize=9)

    # Regret panel
    ax_regret.axhline(1.0, color="#27ae60", linewidth=1, linestyle="--", alpha=0.4, label="Optimal (1.0x)")
    ax_regret.set_ylabel("Top-1 Regret (x)", fontweight="bold")
    ax_regret.set_xlabel("Timestep", fontweight="bold")
    ax_regret.set_ylim(bottom=0.95)
    ax_regret.grid(axis="y", alpha=0.2, linestyle="--")
    ax_regret.legend(loc="upper right", fontsize=9)

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_sgd_exploration_figure(ts_csv_path, output_path):
    """Plot SGD fires and exploration triggers per timestep."""
    rows = parse_csv(ts_csv_path)
    if not rows:
        print(f"  No timestep data in {ts_csv_path}, skipping SGD/EXP plot.")
        return

    # Detect field-based vs timestep-based data
    is_field_based = "field_idx" in rows[0] and "timestep" not in rows[0]
    x_label = "Field" if is_field_based else "Timestep"
    unit_label = "field" if is_field_based else "timestep"

    # Collect field names for x-tick labels
    field_names = []
    if is_field_based and "field_name" in rows[0]:
        seen = set()
        for r in rows:
            fn = r.get("field_name", "")
            idx = r.get("field_idx", "")
            key = f"{idx}_{fn}"
            if key not in seen:
                seen.add(key)
                short = fn.replace(".f32", "").replace(".dat", "").replace(".bin", "")
                if len(short) > 20:
                    short = short[:18] + ".."
                field_names.append(short)

    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#7f8c8d", "ls": ":",  "marker": "s", "lw": 2.0},
        "nn-rl":       {"color": "#8e44ad", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c0392b", "ls": "--", "marker": "D", "lw": 2.0},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    title_suffix = "Fields" if is_field_based else "Timesteps"
    fig, (ax_sgd, ax_exp) = plt.subplots(2, 1, figsize=(14, 5))
    fig.suptitle(f"SGD & Exploration Firing Over {title_suffix}\n"
                 "(per-chunk cost model error triggers: SGD > 10%, Explore > 20%)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ph in phases_present:
        ph_rows = by_phase[ph]
        timesteps = np.array([g(r, "timestep", "field_idx") for r in ph_rows])
        sgd = np.array([g(r, "sgd_fires") for r in ph_rows])
        expl = np.array([g(r, "explorations", default=0) for r in ph_rows])
        sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 2.0})

        ax_sgd.plot(timesteps, sgd, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=6, linewidth=2.0,
                    label=PHASE_LABELS.get(ph, ph).replace("\n", " "),
                    alpha=0.9, zorder=3)
        if expl.sum() > 0:
            ax_exp.plot(timesteps, expl, color=sty["color"], linestyle=sty["ls"],
                        marker=sty["marker"], markersize=6, linewidth=2.0,
                        label=PHASE_LABELS.get(ph, ph).replace("\n", " "),
                        alpha=0.9, zorder=3)

    ax_sgd.set_ylabel(f"SGD Fires\n(chunks per {unit_label})", fontweight="bold")
    ax_sgd.set_title(f"SGD Weight Updates (chunks where cost_model_error > 10%)")
    ax_sgd.grid(axis="y", alpha=0.2, linestyle="--")
    ax_sgd.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
    ax_sgd.minorticks_on()
    ax_sgd.legend(loc="upper right")

    ax_exp.set_ylabel(f"Explorations\n(chunks per {unit_label})", fontweight="bold")
    ax_exp.set_xlabel(x_label, fontweight="bold")
    ax_exp.set_title(f"Exploration Triggers (chunks where cost_model_error > 20%)")

    # Use field names as x-tick labels if available
    if is_field_based and field_names:
        for ax in [ax_sgd, ax_exp]:
            ax.set_xticks(range(len(field_names)))
            ax.set_xticklabels(field_names, rotation=30, ha="right", fontsize=9)
    ax_exp.grid(axis="y", alpha=0.2, linestyle="--")
    ax_exp.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
    ax_exp.minorticks_on()
    ax_exp.legend(loc="upper right")

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_timestep_chunks_figure(tc_csv_path, output_path, phase_filter="nn-rl"):
    """Plot per-chunk predicted vs actual at milestone timesteps for one phase."""
    all_rows = parse_csv(tc_csv_path)
    if not all_rows:
        print(f"  No timestep chunk data in {tc_csv_path}, skipping.")
        return

    # Filter by phase if column exists
    has_phase = "phase" in all_rows[0]
    if has_phase:
        rows = [r for r in all_rows if r.get("phase", "") == phase_filter]
        if not rows:
            # Fallback: try first available phase
            phases_avail = list(set(r.get("phase", "") for r in all_rows))
            if phases_avail:
                phase_filter = phases_avail[0]
                rows = [r for r in all_rows if r.get("phase", "") == phase_filter]
            else:
                rows = all_rows
    else:
        rows = all_rows

    # Detect field-based vs timestep-based
    is_field_based = "field_idx" in rows[0] if rows else False
    unit_word = "Field" if is_field_based else "Timestep"

    # Collect field name mapping if available
    field_name_map = {}
    if is_field_based:
        for r in rows:
            idx = int(g(r, "field_idx", "timestep"))
            fn = r.get("field_name", "")
            if fn and idx not in field_name_map:
                short = fn.replace(".f32", "").replace(".dat", "").replace(".bin", "")
                field_name_map[idx] = short

    # Group by timestep/field
    by_ts = {}
    for r in rows:
        ts = int(g(r, "timestep", "field_idx"))
        by_ts.setdefault(ts, []).append(r)
    all_ts = sorted(by_ts.keys())
    if len(all_ts) == 0:
        return
    # Pick 5 milestones: 0%, 25%, 50%, 75%, 100%
    indices = [0, len(all_ts) // 4, len(all_ts) // 2, 3 * len(all_ts) // 4, len(all_ts) - 1]
    seen = set()
    timestep_list = [all_ts[i] for i in indices if not (all_ts[i] in seen or seen.add(all_ts[i]))]
    n_ts = len(timestep_list)

    metrics = [
        ("Compression Ratio",  "predicted_ratio",    ("actual_ratio",),                          "x",  "mape_ratio"),
        ("Comp Time",          "predicted_comp_ms",  ("actual_comp_ms_raw", "actual_comp_ms"),    "ms", "mape_comp"),
        ("Decomp Time",        "predicted_decomp_ms",("actual_decomp_ms_raw", "actual_decomp_ms"),"ms", "mape_decomp"),
    ]

    fig, axes = plt.subplots(n_ts, 3, figsize=(14, 3.2 * n_ts + 2),
                              squeeze=False)
    phase_label = phase_filter if has_phase else "nn-rl"
    milestone_word = "Fields" if is_field_based else "Timesteps"
    fig.suptitle(f"NN Predicted vs Actual Metrics Per Chunk at Milestone {milestone_word}\n"
                 f"[{phase_label}] (ratio, compression time, decompression time)",
                 fontsize=14, fontweight="bold", y=0.99)

    for row_idx, ts in enumerate(timestep_list):
        chunk_rows = by_ts[ts]
        chunks = np.array([int(g(r, "chunk")) for r in chunk_rows])
        sort_idx = np.argsort(chunks)
        chunks = chunks[sort_idx]

        for col_idx, (label, pred_key, act_key, unit, mape_key) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            pred = np.array([g(r, pred_key) for r in chunk_rows])[sort_idx]
            act_keys = act_key if isinstance(act_key, tuple) else (act_key,)
            act = np.array([g(r, *act_keys) for r in chunk_rows])[sort_idx]
            # Comp/decomp: clamp actuals to 5ms floor (matches NN training baseline)
            if col_idx > 0:
                act = np.maximum(act, 5.0)
            mape_vals = np.array([g(r, mape_key) for r in chunk_rows])[sort_idx]

            # Lines with shaded error region
            ax.plot(chunks, act, color="#2c3e50", linewidth=2.0, label="Actual",
                    zorder=3, marker='o', markersize=3)
            ax.plot(chunks, pred, color="#3498db", linewidth=2.0, linestyle="--",
                    alpha=0.9, label="Predicted", zorder=3, marker='s', markersize=3)
            ax.fill_between(chunks, act, pred, color="#3498db", alpha=0.15,
                            zorder=2)

            # Highlight chunks with MAPE > 50%
            bad = mape_vals > 50
            if np.any(bad):
                ax.scatter(chunks[bad], pred[bad], color="#e74c3c", s=20,
                           zorder=4, marker="x", linewidths=1.5)

            if row_idx == 0:
                ax.set_title(f"{label} ({unit})", fontweight="bold")
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc="upper left", framealpha=0.9)
            if row_idx == n_ts - 1:
                ax.set_xlabel("Chunk Index")
            if col_idx == 0:
                if is_field_based and ts in field_name_map:
                    ax.set_ylabel(field_name_map[ts], fontweight="bold", fontsize=9)
                else:
                    ax.set_ylabel(f"T={ts}", fontweight="bold")
            ax.grid(alpha=0.2, linestyle="-")
            ax.grid(which='minor', alpha=0.1, linestyle=':')
            ax.minorticks_on()

            # Y-axis: tight range for timing, zero-based for ratio
            all_vals = np.concatenate([pred, act])
            pos = all_vals[all_vals > 0]
            if col_idx == 0:
                # Ratio: start at 0
                p95 = np.percentile(pos, 95) if len(pos) > 0 else 1
                ax.set_ylim(0, p95 * 1.6)
            else:
                # Comp/Decomp time: tight range so gap is visible
                vmin = np.min(pos) if len(pos) > 0 else 0
                vmax = np.max(pos) if len(pos) > 0 else 1
                margin = max((vmax - vmin) * 0.3, vmax * 0.1, 0.5)
                ax.set_ylim(max(0, vmin - margin), vmax + margin)

            # Stats box
            avg_mape = np.mean(mape_vals)
            median_mape = np.median(mape_vals)
            ax.text(0.98, 0.95,
                    f"MAPE: avg={avg_mape:.0f}%  med={median_mape:.0f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb",
                              boxstyle="round,pad=0.3"))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_timestep_chunks_multi_phase(tc_csv_path, output_path,
                                      phases=("nn", "nn-rl", "nn-rl+exp50")):
    """Plot predicted vs actual for multiple NN phases side by side.

    Layout: N_phases rows × 3 metric columns (ratio, comp time, decomp time).
    Each row picks the last available timestep for that phase.
    """
    all_rows = parse_csv(tc_csv_path)
    if not all_rows:
        return

    metrics = [
        ("Compression Ratio",  "predicted_ratio",    ("actual_ratio",),                          "x",  "mape_ratio"),
        ("Comp Time",          "predicted_comp_ms",  ("actual_comp_ms_raw", "actual_comp_ms"),    "ms", "mape_comp"),
        ("Decomp Time",        "predicted_decomp_ms",("actual_decomp_ms_raw", "actual_decomp_ms"),"ms", "mape_decomp"),
    ]

    # Group by phase → timestep → rows
    by_phase = {}
    for r in all_rows:
        ph = r.get("phase", "")
        ts = int(g(r, "timestep", "field_idx"))
        by_phase.setdefault(ph, {}).setdefault(ts, []).append(r)

    # Filter to requested phases that have data
    active_phases = [ph for ph in phases if ph in by_phase]
    if not active_phases:
        return

    n_phases = len(active_phases)
    fig, axes = plt.subplots(n_phases, 3, figsize=(14, 3.5 * n_phases + 2),
                              squeeze=False)

    fig.suptitle("NN Predicted vs Actual: All Phases\n"
                 "(ratio, compression time, decompression time)",
                 fontsize=14, fontweight="bold", y=0.99)

    phase_labels = {"nn": "NN Inference", "nn-rl": "NN + SGD",
                    "nn-rl+exp50": "NN + SGD + Explore"}

    for row_idx, ph in enumerate(active_phases):
        ts_data = by_phase[ph]
        last_ts = max(ts_data.keys())
        chunk_rows = ts_data[last_ts]
        chunks = np.array([int(g(r, "chunk")) for r in chunk_rows])
        sort_idx = np.argsort(chunks)
        chunks = chunks[sort_idx]

        for col_idx, (label, pred_key, act_key, unit, mape_key) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            pred = np.array([g(r, pred_key) for r in chunk_rows])[sort_idx]
            act_keys = act_key if isinstance(act_key, tuple) else (act_key,)
            act = np.array([g(r, *act_keys) for r in chunk_rows])[sort_idx]
            if col_idx > 0:
                act = np.maximum(act, 5.0)
            mape_vals = np.array([g(r, mape_key) for r in chunk_rows])[sort_idx]

            ax.plot(chunks, act, color="#2c3e50", linewidth=2.0, label="Actual",
                    zorder=3, marker='o', markersize=3)
            ax.plot(chunks, pred, color="#3498db", linewidth=2.0, linestyle="--",
                    alpha=0.9, label="Predicted", zorder=3, marker='s', markersize=3)
            ax.fill_between(chunks, act, pred, color="#3498db", alpha=0.15, zorder=2)

            bad = mape_vals > 50
            if np.any(bad):
                ax.scatter(chunks[bad], pred[bad], color="#e74c3c", s=20,
                           zorder=4, marker="x", linewidths=1.5)

            if row_idx == 0:
                ax.set_title(f"{label} ({unit})", fontweight="bold")
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc="upper left", framealpha=0.9)
            if row_idx == n_phases - 1:
                ax.set_xlabel("Chunk Index")
            if col_idx == 0:
                ax.set_ylabel(f"{phase_labels.get(ph, ph)}\n(T={last_ts})",
                              fontweight="bold", fontsize=9)
            ax.grid(alpha=0.2, linestyle="-")
            ax.minorticks_on()

            all_vals = np.concatenate([pred, act])
            pos = all_vals[all_vals > 0]
            if col_idx == 0:
                p95 = np.percentile(pos, 95) if len(pos) > 0 else 1
                ax.set_ylim(0, p95 * 1.6)
            else:
                vmin = np.min(pos) if len(pos) > 0 else 0
                vmax = np.max(pos) if len(pos) > 0 else 1
                margin = max((vmax - vmin) * 0.3, vmax * 0.1, 0.5)
                ax.set_ylim(max(0, vmin - margin), vmax + margin)

            avg_mape = np.mean(mape_vals)
            ax.text(0.98, 0.95,
                    f"MAPE: avg={avg_mape:.0f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb",
                              boxstyle="round,pad=0.3"))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View 3: Per-Chunk Algorithm Selection
# ═══════════════════════════════════════════════════════════════════════

# Canonical algorithm names
ALGO_NAMES = ["lz4", "snappy", "deflate", "gdeflate",
              "zstd", "ans", "cascaded", "bitcomp"]

# Each of the 32 configs gets a fully distinct color.
# Layout: 4 variants per algorithm × 8 algorithms.
# Uses hatching patterns in the legend to reinforce preprocessing differences.
CONFIG_ORDER = []    # list of canonical config strings
CONFIG_COLORS = {}   # config string → hex color

# Hand-picked 32-color palette grouped by algo family.
# Within each algo: plain=solid, +shuf=stripe pattern (but distinct hue),
# +quant=different hue, +shuf+quant=yet another hue.
_CONFIG_PALETTE = {
    # lz4 family: blues
    "lz4":              "#2166ac",  # dark blue
    "lz4+shuf":         "#67a9cf",  # medium blue
    "lz4+quant":        "#053061",  # navy
    "lz4+shuf+quant":   "#a6cee3",  # light blue
    # snappy family: oranges
    "snappy":           "#e08214",  # orange
    "snappy+shuf":      "#fdb863",  # light orange
    "snappy+quant":     "#b35806",  # dark orange
    "snappy+shuf+quant":"#fee0b6",  # pale orange
    # deflate family: reds
    "deflate":          "#d6301d",  # red
    "deflate+shuf":     "#fc8d59",  # salmon
    "deflate+quant":    "#7f0000",  # dark red
    "deflate+shuf+quant":"#fddbc7", # pale pink
    # gdeflate family: teals
    "gdeflate":         "#1b7837",  # dark green
    "gdeflate+shuf":    "#7fbf7b",  # medium green
    "gdeflate+quant":   "#00441b",  # forest
    "gdeflate+shuf+quant":"#d9f0d3", # pale green
    # zstd family: purples
    "zstd":             "#762a83",  # purple
    "zstd+shuf":        "#af8dc3",  # lavender
    "zstd+quant":       "#40004b",  # dark purple
    "zstd+shuf+quant":  "#e7d4e8",  # pale purple
    # ans family: yellows/browns
    "ans":              "#b8860b",  # dark goldenrod
    "ans+shuf":         "#daa520",  # goldenrod
    "ans+quant":        "#8b6914",  # dark gold
    "ans+shuf+quant":   "#ffd700",  # gold
    # cascaded family: magentas
    "cascaded":         "#c51b7d",  # magenta
    "cascaded+shuf":    "#e9a3c9",  # pink
    "cascaded+quant":   "#8e0152",  # dark magenta
    "cascaded+shuf+quant":"#fde0ef", # pale pink
    # bitcomp family: cyans
    "bitcomp":          "#01665e",  # dark teal
    "bitcomp+shuf":     "#5ab4ac",  # teal
    "bitcomp+quant":    "#003c30",  # very dark teal
    "bitcomp+shuf+quant":"#c7eae5", # pale teal
}

for algo in ALGO_NAMES:
    for suffix in ["", "+shuf", "+quant", "+shuf+quant"]:
        name = algo + suffix
        CONFIG_ORDER.append(name)
        CONFIG_COLORS[name] = _CONFIG_PALETTE.get(name, "#cccccc")

# Algo-only colors (plain variant)
ALGO_COLORS = {a: CONFIG_COLORS[a] for a in ALGO_NAMES}


def _normalize_action(action_str):
    """Normalize action string to canonical form: 'algo[+quant][+shuf]'."""
    if not action_str or action_str == "none":
        return "none"
    parts = action_str.lower().split("+")
    algo = parts[0]
    shuf = "shuf" in parts
    quant = "quant" in parts
    name = algo
    if shuf:
        name += "+shuf"
    if quant:
        name += "+quant"
    return name


def _build_config_cmap():
    """Build colormap and norm for the full 32-config palette."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = [CONFIG_COLORS.get(c, "#cccccc") for c in CONFIG_ORDER]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(CONFIG_ORDER) + 0.5, 1), cmap.N)
    return cmap, norm


def _config_to_idx(action_str):
    """Map action string like 'zstd+shuf' to CONFIG_ORDER index."""
    name = _normalize_action(str(action_str))
    try:
        return CONFIG_ORDER.index(name)
    except ValueError:
        return -1


def _config_legend_patches():
    """Build legend patches — only show configs that differ from plain algo."""
    from matplotlib.patches import Patch
    patches = []
    for cfg in CONFIG_ORDER:
        patches.append(Patch(facecolor=CONFIG_COLORS[cfg], edgecolor="black",
                             linewidth=0.5, label=cfg))
    return patches


def _config_legend_patches_used(rows, action_key="action_final"):
    """Build legend patches only for configs actually present in the data."""
    from matplotlib.patches import Patch
    used = set()
    for r in rows:
        action_str = r.get(action_key, r.get("action", "none"))
        cfg = _normalize_action(str(action_str))
        if cfg != "none":
            used.add(cfg)
    patches = []
    for cfg in CONFIG_ORDER:
        if cfg in used:
            patches.append(Patch(facecolor=CONFIG_COLORS[cfg], edgecolor="black",
                                 linewidth=0.5, label=cfg))
    return patches


def make_chunk_actions_figure(chunk_csv_path, output_path):
    """Plot NN config selection (algo+preprocessing) per chunk across phases.

    Generates:
      - Top: per-chunk config as a colored strip (one row per phase)
      - Bottom: config frequency bar chart per phase (grouped by algo, stacked by variant)
    """
    rows = parse_csv(chunk_csv_path)
    if not rows:
        print(f"  No chunk data in {chunk_csv_path}, skipping.")
        return

    # Group by phase
    by_phase = {}
    for r in rows:
        ph = r.get("phase", "unknown")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phases_present = [p for p in phase_order if p in by_phase]
    if not phases_present:
        phases_present = list(by_phase.keys())

    n_phases = len(phases_present)
    if n_phases == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 3 + 1.5 * n_phases),
                              gridspec_kw={"height_ratios": [n_phases, 2.5]})
    fig.suptitle("NN Configuration Selection Per Chunk (algo + preprocessing)",
                 fontsize=14, fontweight="bold", y=0.98)

    cmap, norm = _build_config_cmap()

    # ── Top: colored strip per phase ──
    ax_strip = axes[0]
    strip_data = []
    for ph in phases_present:
        ph_rows = sorted(by_phase[ph], key=lambda r: int(g(r, "chunk")))
        row_vals = []
        for r in ph_rows:
            action_str = r.get("action_final", r.get("action", "none"))
            row_vals.append(_config_to_idx(action_str))
        strip_data.append(row_vals)

    max_chunks = max(len(row) for row in strip_data) if strip_data else 0
    for row in strip_data:
        while len(row) < max_chunks:
            row.append(-1)

    strip_arr = np.array(strip_data, dtype=float)
    strip_arr[strip_arr < 0] = np.nan

    ax_strip.imshow(strip_arr, aspect="auto", cmap=cmap, norm=norm,
                     interpolation="nearest")
    ax_strip.set_yticks(range(n_phases))
    ax_strip.set_yticklabels([PHASE_LABELS.get(p, p).replace("\n", " ")
                               for p in phases_present], fontsize=10)
    ax_strip.set_xlabel("Chunk Index", fontsize=10)
    ax_strip.set_title("Config per Chunk (algo + preprocessing)", fontsize=11)

    # Collect all rows for legend
    all_phase_rows = [r for ph in phases_present for r in by_phase[ph]]
    legend_patches = _config_legend_patches_used(all_phase_rows)
    ax_strip.legend(handles=legend_patches, loc="upper right", fontsize=7,
                     ncol=min(len(legend_patches), 6), framealpha=0.9)

    # ── Bottom: frequency bar chart (full configs) ──
    ax_bar = axes[1]

    # Count configs actually used
    used_configs = set()
    for ph in phases_present:
        for r in by_phase[ph]:
            action_str = r.get("action_final", r.get("action", "none"))
            cfg = _normalize_action(str(action_str))
            if cfg != "none":
                used_configs.add(cfg)

    # Keep only used configs in CONFIG_ORDER order
    bar_configs = [c for c in CONFIG_ORDER if c in used_configs]
    if not bar_configs:
        bar_configs = CONFIG_ORDER[:8]  # fallback to plain algos

    x = np.arange(len(bar_configs))
    width = 0.8 / max(n_phases, 1)
    for pi, ph in enumerate(phases_present):
        ph_rows = by_phase[ph]
        counts = {c: 0 for c in bar_configs}
        for r in ph_rows:
            action_str = r.get("action_final", r.get("action", "none"))
            cfg = _normalize_action(str(action_str))
            if cfg in counts:
                counts[cfg] += 1
        total = max(sum(counts.values()), 1)
        pcts = [100.0 * counts[c] / total for c in bar_configs]
        offset = (pi - n_phases / 2 + 0.5) * width
        ax_bar.bar(x + offset, pcts, width,
                   color=PHASE_COLORS.get(ph, "#bdc3c7"),
                   edgecolor="black", linewidth=0.5,
                   label=PHASE_LABELS.get(ph, ph).replace("\n", " "))

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_configs, rotation=30, ha="right")
    ax_bar.set_ylabel("Chunks (%)")
    ax_bar.set_title("Configuration Selection Frequency")
    ax_bar.legend(loc="upper right")
    ax_bar.grid(axis="y", alpha=0.2, linestyle="--")

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_milestone_actions_figure(tc_csv_path, output_path, chunk_csv_path=None):
    """Plot NN config selection per chunk at each milestone timestep.

    Layout: 3 columns (NN Inference | NN+SGD | NN+SGD+Explore)
    - Column 1 (NN Inference): static reference from single-shot data, repeated per row
    - Column 2 (NN+SGD): from multi-timestep nn-rl milestones
    - Column 3 (NN+SGD+Explore): from multi-timestep nn-rl+exp50 milestones
    """
    from matplotlib.patches import Patch

    all_rows = parse_csv(tc_csv_path)
    if not all_rows:
        print(f"  No data in {tc_csv_path}, skipping milestone actions.")
        return

    # Group by (phase, timestep)
    by_phase_ts = {}
    for r in all_rows:
        ph = r.get("phase", "nn-rl")
        ts = int(g(r, "timestep", "field_idx"))
        by_phase_ts.setdefault(ph, {}).setdefault(ts, []).append(r)

    # Load nn phase for reference (static column).
    # Try single-shot chunks CSV first; fall back to timestep_chunks "nn" phase at T=0.
    nn_ref_strip = None
    if chunk_csv_path:
        try:
            chunk_rows = parse_csv(chunk_csv_path)
            nn_chunks = [r for r in chunk_rows if r.get("phase", "") == "nn"]
            if nn_chunks:
                nn_chunks.sort(key=lambda r: int(g(r, "chunk")))
                nn_ref_strip = [_config_to_idx(
                    r.get("action_final", r.get("action", "none")))
                    for r in nn_chunks]
        except Exception:
            pass
    # Fallback: use "nn" phase from timestep_chunks at first timestep
    if nn_ref_strip is None and "nn" in by_phase_ts:
        first_ts = min(by_phase_ts["nn"].keys())
        nn_chunks = sorted(by_phase_ts["nn"][first_ts],
                           key=lambda r: int(g(r, "chunk")))
        if nn_chunks:
            nn_ref_strip = [_config_to_idx(
                r.get("action_final", r.get("action", "none")))
                for r in nn_chunks]

    # Build phase columns: all available multi-timestep NN phases
    phase_columns = []  # list of (label, data_source)
    ts_phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    for ph in ts_phase_order:
        if ph in by_phase_ts:
            phase_columns.append((PHASE_LABELS.get(ph, ph), ph))

    if not phase_columns:
        return

    # Get milestones from first available multi-timestep phase
    ts_phases = [ph for _, ph in phase_columns if ph != "nn_ref"]
    if not ts_phases:
        return
    milestones = sorted(by_phase_ts[ts_phases[0]].keys())
    n_milestones = len(milestones)
    n_cols = len(phase_columns)
    if n_milestones == 0:
        return

    cmap, norm = _build_config_cmap()

    # Compute max chunks
    max_chunks = len(nn_ref_strip) if nn_ref_strip else 0
    for ph in ts_phases:
        for ts in milestones:
            max_chunks = max(max_chunks, len(by_phase_ts[ph].get(ts, [])))

    fig_w = max(12, 5 * n_cols)
    fig_h = 1.8 * n_milestones + 4
    fig, axes = plt.subplots(n_milestones, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.suptitle("NN Algorithm Selection per Chunk Over Time",
                 fontsize=14, fontweight="bold", y=0.97)

    for col, (col_label, col_source) in enumerate(phase_columns):
        for row, ts in enumerate(milestones):
            ax = axes[row, col]

            if col_source == "nn_ref":
                strip = list(nn_ref_strip)
            else:
                chunk_rows_ts = sorted(
                    by_phase_ts[col_source].get(ts, []),
                    key=lambda r: int(g(r, "chunk")))
                strip = [_config_to_idx(str(r.get("action", "none")))
                         for r in chunk_rows_ts]

            while len(strip) < max_chunks:
                strip.append(-1)

            arr = np.array([strip], dtype=float)
            arr[arr < 0] = np.nan

            ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm,
                      interpolation="nearest")
            ax.set_yticks([])


            if row == 0:
                ax.set_title(col_label.replace("\n", " "),
                             fontsize=13, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(f"T={ts}", fontsize=12, fontweight="bold",
                              rotation=0, labelpad=35, va="center")
            if row == n_milestones - 1:
                ax.set_xticks(range(max_chunks))
                ax.set_xlabel("Chunk Index", fontsize=10)
                ax.tick_params(axis="x", labelsize=8)
            else:
                ax.set_xticks([])

            # Clean borders
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#888888")

    # Legend — configs from all sources
    all_legend_rows = list(all_rows)
    if chunk_csv_path:
        try:
            all_legend_rows += [r for r in parse_csv(chunk_csv_path)
                                if r.get("phase", "") == "nn"]
        except Exception:
            pass
    legend_patches = _config_legend_patches_used(all_legend_rows, action_key="action")
    lp2 = _config_legend_patches_used(all_legend_rows, action_key="action_final")
    seen = {p.get_label() for p in legend_patches}
    for p in lp2:
        if p.get_label() not in seen:
            legend_patches.append(p)
            seen.add(p.get_label())

    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center", fontsize=10,
                   ncol=min(len(legend_patches), 6), framealpha=0.95,
                   edgecolor="#cccccc", fancybox=True,
                   bbox_to_anchor=(0.5, 0.005))

    _sc_finalize(fig, pad=1.5, rect=[0.05, 0.08, 1, 0.93])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View: Multi-Dataset Comparison
# ═══════════════════════════════════════════════════════════════════════

def make_multi_dataset_figure(all_datasets, output_path):
    """Bar chart comparing ratio and throughput across all datasets for each phase.

    all_datasets: list of (name, rows) tuples where rows is a list of dicts.
    """
    if not all_datasets:
        return
    _normalize_rows([r for _, rows in all_datasets for r in rows])

    # Collect all phases present across datasets
    all_phases_set = set()
    for _, rows in all_datasets:
        for r in rows:
            ph = _PHASE_ALIASES.get(r.get("phase", ""), r.get("phase", ""))
            all_phases_set.add(ph)
    phases = [p for p in PHASE_ORDER if p in all_phases_set]
    if not phases:
        return

    n_datasets = len(all_datasets)
    n_phases = len(phases)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white")
    fig.suptitle("Multi-Dataset Comparison: GPUCompress Phases", fontsize=14, fontweight="bold")

    x = np.arange(n_phases)
    width = 0.8 / n_datasets

    for panel, (metric, ylabel, fmt) in enumerate([
        ("ratio", "Compression Ratio", "%.1f"),
        ("write_mbps", "Write Throughput (MiB/s)", "%.0f"),
    ]):
        ax = axes[panel]
        for di, (ds_name, rows) in enumerate(all_datasets):
            by_phase = {}
            for r in rows:
                ph = _PHASE_ALIASES.get(r.get("phase", ""), r.get("phase", ""))
                by_phase[ph] = r
            vals = [g(by_phase.get(p, {}), metric, f"{metric}ps") for p in phases]
            offset = (di - n_datasets / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, label=ds_name,
                          edgecolor="black", linewidth=0.4,
                          zorder=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases], rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View: Per-Dataset Phase Comparison (which phase wins per dataset)
# ═══════════════════════════════════════════════════════════════════════

def make_per_dataset_phase_comparison(all_datasets, output_path):
    """For each dataset, show all phases side by side (ratio + throughput).

    X-axis = datasets, grouped bars = phases.
    This answers: "for Hurricane Isabel, which phase gives the best ratio?"
    """
    if not all_datasets:
        return
    _normalize_rows([r for _, rows in all_datasets for r in rows])

    # Collect all phases
    all_phases_set = set()
    for _, rows in all_datasets:
        for r in rows:
            ph = _PHASE_ALIASES.get(r.get("phase", ""), r.get("phase", ""))
            all_phases_set.add(ph)
    phases = [p for p in PHASE_ORDER if p in all_phases_set]
    if not phases:
        return

    n_datasets = len(all_datasets)
    n_phases = len(phases)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_datasets * 3), 10), facecolor="white")
    fig.suptitle("Phase Comparison per Dataset", fontsize=14, fontweight="bold")

    x = np.arange(n_datasets)
    width = 0.8 / n_phases

    for panel, (metric, ylabel) in enumerate([
        ("ratio", "Compression Ratio (higher = better)"),
        ("write_mbps", "Write Throughput (MiB/s, higher = better)"),
    ]):
        ax = axes[panel]
        for pi, phase in enumerate(phases):
            vals = []
            for ds_name, rows in all_datasets:
                by_phase = {}
                for r in rows:
                    ph = _PHASE_ALIASES.get(r.get("phase", ""), r.get("phase", ""))
                    by_phase[ph] = r
                vals.append(g(by_phase.get(phase, {}), metric, f"{metric}ps"))

            offset = (pi - n_phases / 2 + 0.5) * width
            color = PHASE_COLORS.get(phase, "#bdc3c7")
            hatch = PHASE_HATCHES.get(phase, "")
            label = PHASE_LABELS.get(phase, phase).replace("\n", " ")
            bars = ax.bar(x + offset, vals, width * 0.9, label=label,
                          color=color, edgecolor="black", linewidth=0.4,
                          zorder=3, alpha=0.85, hatch=hatch)

            # Annotate the best value per dataset
            for i, v in enumerate(vals):
                if v > 0 and n_phases <= 8:
                    ax.text(x[i] + offset, v, f"{v:.1f}", ha="center", va="bottom",
                            fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in all_datasets], rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=3)
        ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View: Latency Breakdown Stacked Bar
# ═══════════════════════════════════════════════════════════════════════

def make_latency_breakdown_figure(rows, output_path, source_name=""):
    """End-to-end write time per phase with NN overhead annotation.

    Shows write_ms as the total bar for each phase. For NN phases,
    annotates the NN inference + stats overhead as a percentage.
    Uses a 3-stage pipelined architecture where stages overlap,
    so component times do NOT sum to wall-clock time.
    """
    _normalize_rows(rows)
    phases, ordered = _ordered(rows)
    # Only show phases with write_ms > 0
    valid = [(p, r) for p, r in zip(phases, ordered)
             if g(r, "write_ms") > 0]
    if not valid:
        return

    phases_v = [p for p, _ in valid]
    rows_v = [r for _, r in valid]
    n = len(phases_v)

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n + 2), 6), facecolor="white")

    x = np.arange(n)
    write_times = np.array([g(r, "write_ms") for r in rows_v])
    nn_times = np.array([g(r, "nn_ms") for r in rows_v])
    stats_times = np.array([g(r, "stats_ms") for r in rows_v])
    explore_times = np.array([g(r, "explore_ms") for r in rows_v])
    sgd_times = np.array([g(r, "sgd_ms") for r in rows_v])
    nn_overhead = nn_times + stats_times  # sequential on main thread

    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases_v]
    hatches = [PHASE_HATCHES.get(p, "") for p in phases_v]

    # Main bars: end-to-end write_ms
    bars = ax.bar(x, write_times, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.6, zorder=3, alpha=0.85)
    for i, (bar, h) in enumerate(zip(bars, hatches)):
        if h:
            bar.set_hatch(h)

    # Overlay: only NN+Stats as bar segment (sequential on main thread, accurate)
    # Exploration and SGD are cumulative across parallel workers — text annotation only
    nn_phase_names = {"nn", "nn-rl", "nn-rl+exp50"}
    nn_label_done = False
    for i, p in enumerate(phases_v):
        if p not in nn_phase_names:
            continue
        oh = nn_overhead[i]
        if oh > 0:
            ax.bar(i, oh, color="#e67e22", edgecolor="black",
                   linewidth=0.5, width=0.6, zorder=4, alpha=0.9,
                   hatch="//", label="NN + Stats Overhead" if not nn_label_done else "")
            nn_label_done = True

    # Annotate: sequential metrics as %, cumulative as absolute with worker count
    N_WORKERS = 8
    for i, (wt, p) in enumerate(zip(write_times, phases_v)):
        oh = nn_overhead[i]
        sgd = sgd_times[i]
        exp = explore_times[i]
        label_parts = [f"{wt:.0f} ms"]
        if p in nn_phase_names and oh > 0 and wt > 0:
            label_parts.append(f"NN+Stats: {100*oh/wt:.0f}%")
        if sgd > 0:
            label_parts.append(f"SGD: {sgd:.0f} ms ({N_WORKERS}w)")
        if exp > 0:
            label_parts.append(f"Exp: {exp:.0f} ms ({N_WORKERS}w)")
        ax.text(i, wt + write_times.max() * 0.02,
                "\n".join(label_parts),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    labels = [PHASE_LABELS.get(p, p).replace("\n", " ") for p in phases_v]
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel("End-to-End Write Time (ms)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, write_times.max() * 1.25)

    title = "Per-Phase Write Latency"
    if source_name:
        title += f" ({source_name})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.15, linestyle="--", zorder=0)

    # Legend
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, loc="upper left", fontsize=10,
                  framealpha=0.95, edgecolor="#cccccc", fancybox=True)

    # Caption note
    ax.text(0.98, 0.02,
            "NN overhead (orange) is sequential on main thread.\n"
            "*SGD/Exploration times are cumulative across\n"
            " parallel workers (overlap with compression).",
            transform=ax.transAxes, fontsize=8, color="#666",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    _sc_finalize(fig, pad=2.0)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# View: Algorithm Selection Frequency Histogram
# ═══════════════════════════════════════════════════════════════════════

_ALGO_NAMES = ["lz4", "snappy", "deflate", "gdeflate",
               "zstd", "ans", "cascaded", "bitcomp"]


def _decode_action(action_id):
    """Decode action ID to (algo_name, has_shuffle, has_quant) string."""
    algo = int(action_id) % 8
    quant = (int(action_id) // 8) % 2
    shuf = (int(action_id) // 16) % 2
    name = _ALGO_NAMES[algo] if 0 <= algo < 8 else f"unk{algo}"
    if shuf:
        name += "+shuf"
    if quant:
        name += "+quant"
    return name


def make_algorithm_histogram(chunk_csv_paths, output_path):
    """Histogram of algorithm selection frequency across datasets.

    chunk_csv_paths: list of (dataset_name, csv_path) tuples.
    Only includes nn/nn-rl phases (phases that select).
    """
    from collections import Counter

    all_counts = {}  # dataset_name -> Counter of action strings

    for ds_name, csv_path in chunk_csv_paths:
        if not os.path.exists(csv_path):
            continue
        rows = parse_csv(csv_path)
        counter = Counter()
        for r in rows:
            phase = r.get("phase", "")
            if phase not in ("nn", "nn-rl", "nn-rl+exp50"):
                continue
            action = r.get("action", r.get("nn_action", -1))
            if action is not None and float(action) >= 0:
                counter[_decode_action(action)] += 1
        if counter:
            all_counts[ds_name] = counter

    if not all_counts:
        return

    # Collect all action names
    all_actions = sorted(set(a for c in all_counts.values() for a in c))
    n_actions = len(all_actions)
    n_datasets = len(all_counts)

    fig, ax = plt.subplots(figsize=(max(7, n_actions * 0.8), 5), facecolor="white")
    x = np.arange(n_actions)
    width = 0.8 / n_datasets

    ds_names = list(all_counts.keys())
    for di, ds_name in enumerate(ds_names):
        vals = [all_counts[ds_name].get(a, 0) for a in all_actions]
        total = sum(vals) or 1
        pcts = [v / total * 100 for v in vals]
        offset = (di - n_datasets / 2 + 0.5) * width
        ax.bar(x + offset, pcts, width * 0.9, label=ds_name,
               edgecolor="black", linewidth=0.4, zorder=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_actions, rotation=30, ha="right")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Algorithm Selection Distribution (NN phases)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)

    _sc_finalize(fig, pad=1.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Write Path Decomposition + Pipeline Waterfall
# ═══════════════════════════════════════════════════════════════════════

def make_write_path_decomposition(ts_csv_path, output_path):
    """Stacked bar: write_ms = h5dwrite + cuda_sync + h5dclose + h5fclose per timestep.

    100% additive — these are consecutive chrono slices of the write path.
    Shows one panel per phase, timesteps on x-axis.
    """
    all_rows = parse_csv(ts_csv_path)
    if not all_rows or "h5dwrite_ms" not in all_rows[0]:
        return

    # Group by phase
    by_phase = {}
    for r in all_rows:
        ph = r.get("phase", "")
        by_phase.setdefault(ph, []).append(r)

    # Only plot NN phases
    nn_phases = [ph for ph in ["nn", "nn-rl", "nn-rl+exp50"] if ph in by_phase]
    if not nn_phases:
        nn_phases = list(by_phase.keys())[:3]
    n = len(nn_phases)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n + 2, 5), squeeze=False)

    colors = {"h5dwrite_ms": "#3498db", "cuda_sync_ms": "#e74c3c",
              "h5dclose_ms": "#f39c12", "h5fclose_ms": "#95a5a6"}
    labels = {"h5dwrite_ms": "H5Dwrite (VOL pipeline)",
              "cuda_sync_ms": "cudaDeviceSync",
              "h5dclose_ms": "H5Dclose (metadata)",
              "h5fclose_ms": "H5Fclose"}
    components = ["h5dwrite_ms", "cuda_sync_ms", "h5dclose_ms", "h5fclose_ms"]

    phase_labels = {"nn": "NN Inference", "nn-rl": "NN + SGD",
                    "nn-rl+exp50": "NN + SGD + Explore"}

    for col, ph in enumerate(nn_phases):
        ax = axes[0, col]
        rows = sorted(by_phase[ph], key=lambda r: int(g(r, "timestep")))
        ts = [int(g(r, "timestep")) for r in rows]
        bottom = np.zeros(len(rows))

        for comp in components:
            vals = np.array([g(r, comp) for r in rows])
            ax.bar(ts, vals, bottom=bottom, color=colors[comp],
                   label=labels[comp], width=0.8, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel("Time (ms)")
        ax.set_title(phase_labels.get(ph, ph), fontweight="bold")
        if col == n - 1:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Write Path Decomposition (100% additive)",
                 fontsize=13, fontweight="bold")
    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_pipeline_waterfall(ts_csv_path, output_path):
    """Gantt-style waterfall: setup | S1 | S2 | S3 | join within h5dwrite.

    Shows overlapping pipeline stages per timestep. One panel per phase.
    """
    all_rows = parse_csv(ts_csv_path)
    if not all_rows or "vol_setup_ms" not in all_rows[0]:
        return

    by_phase = {}
    for r in all_rows:
        ph = r.get("phase", "")
        by_phase.setdefault(ph, []).append(r)

    nn_phases = [ph for ph in ["nn", "nn-rl", "nn-rl+exp50"] if ph in by_phase]
    if not nn_phases:
        nn_phases = list(by_phase.keys())[:3]
    n = len(nn_phases)
    if n == 0:
        return

    stage_colors = {
        "vol_setup_ms":    "#95a5a6",
        "vol_stage1_ms":   "#3498db",
        "vol_stage2_ms":   "#2ecc71",
        "vol_stage3_ms":   "#e74c3c",
        "vol_join_ms":     "#9b59b6",
    }
    stage_labels = {
        "vol_setup_ms":    "Setup (threads + alloc)",
        "vol_stage1_ms":   "S1: Inference",
        "vol_stage2_ms":   "S2: Compression",
        "vol_stage3_ms":   "S3: I/O Write",
        "vol_join_ms":     "Thread Join",
    }
    stages = list(stage_colors.keys())

    phase_labels = {"nn": "NN Inference", "nn-rl": "NN + SGD",
                    "nn-rl+exp50": "NN + SGD + Explore"}

    # Pick last timestep for each phase (steady state)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n + 2), squeeze=False)

    for row, ph in enumerate(nn_phases):
        ax = axes[row, 0]
        rows = sorted(by_phase[ph], key=lambda r: int(g(r, "timestep")))

        # Show last 5 timesteps (or all if < 5)
        show_rows = rows[-5:] if len(rows) > 5 else rows

        y_labels = []
        for i, r in enumerate(show_rows):
            ts = int(g(r, "timestep"))
            y_labels.append(f"T={ts}")
            x_offset = 0
            for stage in stages:
                val = g(r, stage)
                if val > 0:
                    ax.barh(i, val, left=x_offset, height=0.6,
                            color=stage_colors[stage],
                            edgecolor="white", linewidth=0.5)
                # Stages overlap in pipeline — show them stacked for visual
                # but note they run concurrently
                x_offset += val

            # Show h5dwrite wall time as a reference line
            h5w = g(r, "h5dwrite_ms")
            ax.axvline(h5w, color="black", linestyle="--", alpha=0.4, linewidth=1)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"{phase_labels.get(ph, ph)} — Pipeline Stages "
                     f"(dashed = h5dwrite wall clock)", fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        if row == 0:
            # Legend
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=stage_colors[s], label=stage_labels[s])
                       for s in stages]
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("VOL Pipeline Waterfall (stages overlap — sum > h5dwrite)",
                 fontsize=13, fontweight="bold")
    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_cross_phase_pipeline_overhead(phase_csv_map, output_path, title=""):
    """Stacked bar comparing VOL pipeline stage overhead across all phases.

    phase_csv_map: dict of {phase_name: csv_path}
      Each CSV is a benchmark_vpic_deck_timesteps.csv for that one phase.
    Bars show mean across timesteps (T>0 to skip first-write warmup).

    Stage 1 is decomposed into three sub-segments:
      S1a: Stats Kernel  — GPU stats time (CUDA-event measured, summed across chunks)
      S1b: NN Inference  — GPU NN inference time (CUDA-event measured)
      S1c: S1 Residual   — vol_stage1_ms - stats_ms - nn_ms  (WQ posting + sync overhead)
    Phases that bypass the VOL pipeline (no-comp) show only h5dwrite_ms.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    # vol_stage1_ms is replaced by three derived sub-segments (S1a/S1b/S1c)
    STAGES = [
        ("vol_setup_ms",   "#95a5a6", "Setup (threads + alloc)"),
        ("s1a_stats_ms",   "#aed6f1", "S1a: Stats Kernel"),
        ("s1b_nn_ms",      "#2980b9", "S1b: NN Inference"),
        ("s1c_residual_ms","#1a5276", "S1c: WQ Post / Sync"),
        ("vol_stage2_ms",  "#2ecc71", "S2: Compression"),
        ("vol_stage3_ms",  "#e74c3c", "S3: I/O Write"),
        ("vol_join_ms",    "#9b59b6", "Thread Join"),
        ("h5dclose_ms",    "#f39c12", "H5Dclose (metadata)"),
    ]

    PHASE_DISPLAY = {
        "no-comp":      "No-Comp",
        "lz4":          "LZ4",
        "snappy":       "Snappy",
        "deflate":      "Deflate",
        "gdeflate":     "GDeflate",
        "zstd":         "Zstd",
        "ans":          "ANS",
        "cascaded":     "Cascaded",
        "bitcomp":      "Bitcomp",
        "nn":           "NN\n(Inference)",
        "nn-rl":        "NN+SGD",
        "nn-rl+exp50":  "NN+SGD\n+Explore",
    }

    # Raw CSV columns needed (beyond STAGES keys)
    RAW_COLS = ["vol_setup_ms", "vol_stage1_ms", "vol_stage2_ms", "vol_stage3_ms",
                "vol_join_ms", "h5dclose_ms", "stats_ms", "nn_ms", "write_ms"]

    # Collect mean values per phase
    phases_ordered = list(phase_csv_map.keys())
    data = {}   # phase -> {col: mean_ms}
    for ph, csv_path in phase_csv_map.items():
        if not os.path.exists(csv_path):
            continue
        rows = parse_csv(csv_path)
        # Skip T=0 (first-call warmup) when enough data available
        rows_use = [r for r in rows if g(r, "timestep") > 0]
        if not rows_use:
            rows_use = rows
        if not rows_use:
            continue
        means = {}
        for col in RAW_COLS:
            vals = [g(r, col) for r in rows_use if g(r, col) >= 0]
            means[col] = float(np.mean(vals)) if vals else 0.0

        # Derive S1 sub-segments
        s1_total   = means.get("vol_stage1_ms", 0.0)
        s1a        = means.get("stats_ms", 0.0)
        s1b        = means.get("nn_ms", 0.0)
        s1c        = max(0.0, s1_total - s1a - s1b)   # residual: WQ posting + sync
        means["s1a_stats_ms"]    = s1a
        means["s1b_nn_ms"]       = s1b
        means["s1c_residual_ms"] = s1c

        data[ph] = means

    if not data:
        return

    phases_plot = [ph for ph in phases_ordered if ph in data]
    n = len(phases_plot)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7, n * 1.4 + 2), 5))

    x = np.arange(n)
    bar_w = 0.6
    bottoms = np.zeros(n)
    legend_handles = []

    for col, color, label in STAGES:
        vals = np.array([data[ph].get(col, 0.0) for ph in phases_plot])
        # Only draw bar segments that are non-negligible
        mask = vals > 0.5
        if mask.any():
            ax.bar(x, vals, bar_w, bottom=bottoms,
                   color=color, edgecolor="white", linewidth=0.5,
                   label=label)
            # Annotate segments > 20ms with their value
            for i, (v, bot) in enumerate(zip(vals, bottoms)):
                if v > 20:
                    ax.text(i, bot + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            legend_handles.append(Patch(facecolor=color, label=label))
        bottoms += vals

    # Annotate total write_ms above each bar
    for i, ph in enumerate(phases_plot):
        total = data[ph].get("write_ms", 0)
        ax.text(i, bottoms[i] + 8, f"{total:.0f} ms", ha="center", va="bottom",
                fontsize=8, color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_DISPLAY.get(ph, ph) for ph in phases_plot],
                       fontsize=9)
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_title(title or "VOL Pipeline Stage Overhead by Phase\n"
                 "(mean across timesteps T>0; S1 split into Stats / NN / WQ-Post)",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, ncol=2)

    # Annotate the setup bar on the first phase that has significant setup time
    for i, ph in enumerate(phases_plot):
        setup_v = data[ph].get("vol_setup_ms", 0)
        if setup_v > 50:
            ax.annotate(f"Thread\nspawn\n{setup_v:.0f}ms",
                        xy=(i, setup_v / 2),
                        xytext=(i + 0.45, setup_v * 0.6),
                        fontsize=7, color="#555555",
                        arrowprops=dict(arrowstyle="->", color="#888888",
                                        lw=0.8),
                        ha="left", va="center")
            break

    _sc_finalize(fig, pad=1.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_gpu_breakdown_over_time(ts_csv_path, output_path):
    """Stacked bar: per-chunk GPU component breakdown over timesteps.

    Shows stats + nn + preproc + comp + explore + sgd per timestep.
    One panel per NN phase. These are the sum across all chunks in one timestep.
    """
    all_rows = parse_csv(ts_csv_path)
    if not all_rows:
        return

    by_phase = {}
    for r in all_rows:
        ph = r.get("phase", "")
        by_phase.setdefault(ph, []).append(r)

    nn_phases = [ph for ph in ["nn", "nn-rl", "nn-rl+exp50"] if ph in by_phase]
    if not nn_phases:
        return
    n = len(nn_phases)

    fig, axes = plt.subplots(1, n, figsize=(5 * n + 2, 5), squeeze=False)

    components = [
        ("stats_ms",    "#3498db", "Stats kernels"),
        ("nn_ms",       "#2ecc71", "NN inference"),
        ("preproc_ms",  "#9b59b6", "Preprocessing"),
        ("comp_ms",     "#e74c3c", "Compression"),
        ("explore_ms",  "#f39c12", "Exploration"),
        ("sgd_ms",      "#1abc9c", "SGD update"),
    ]

    phase_labels = {"nn": "NN Inference", "nn-rl": "NN + SGD",
                    "nn-rl+exp50": "NN + SGD + Explore"}

    for col, ph in enumerate(nn_phases):
        ax = axes[0, col]
        rows = sorted(by_phase[ph], key=lambda r: int(g(r, "timestep")))
        ts = [int(g(r, "timestep")) for r in rows]
        bottom = np.zeros(len(rows))

        for key, color, label in components:
            vals = np.array([g(r, key) for r in rows])
            if np.sum(vals) > 0:  # skip empty components
                ax.bar(ts, vals, bottom=bottom, color=color,
                       label=label, width=0.8, edgecolor="white", linewidth=0.3)
                bottom += vals

        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel("Time (ms, sum across chunks)")
        ax.set_title(phase_labels.get(ph, ph), fontweight="bold")
        if col == n - 1:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("GPU Component Breakdown Over Timesteps\n"
                 "(stats + NN + preprocessing + compression + exploration + SGD)",
                 fontsize=13, fontweight="bold")
    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# C4: Cross-Dataset Convergence Comparison
# ═══════════════════════════════════════════════════════════════════════

def make_cross_dataset_convergence_figure(ts_csv_paths, dataset_names, output_path):
    """Overlay MAPE convergence curves from multiple datasets on one plot.

    Args:
        ts_csv_paths: list of timestep CSV file paths
        dataset_names: list of dataset display names (same order)
        output_path: output PNG path
    """
    dataset_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                      "#1abc9c", "#e67e22", "#34495e"]
    dataset_linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    dataset_markers = ["o", "s", "D", "^", "v", "<", ">", "p"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (csv_path, ds_name) in enumerate(zip(ts_csv_paths, dataset_names)):
        if not os.path.exists(csv_path):
            continue
        rows = parse_csv(csv_path)
        # Filter to nn-rl phase
        rl_rows = [r for r in rows if r.get("phase", "nn-rl") == "nn-rl"]
        if not rl_rows:
            rl_rows = rows  # fallback for old CSV without phase column
        timesteps = np.array([g(r, "timestep") for r in rl_rows])
        mape = np.array([g(r, "mape_ratio") for r in rl_rows])
        clipped = np.clip(mape, 0, 300)
        color = dataset_colors[i % len(dataset_colors)]
        ls = dataset_linestyles[i % len(dataset_linestyles)]
        marker = dataset_markers[i % len(dataset_markers)]
        ax.plot(timesteps, clipped, color=color, linewidth=2.0, marker=marker,
                markersize=6, linestyle=ls, label=ds_name, alpha=0.85)

    ax.axhline(20, color="#e67e22", linewidth=1, linestyle="--", alpha=0.6,
               label="20% target")
    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylabel("Ratio MAPE (%)", fontweight="bold")
    ax.set_title("Online Learning Convergence Across Datasets (nn-rl)",
                 fontweight="bold")
    ax.set_ylim(0, 300)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.grid(axis="y", which='minor', alpha=0.1, linestyle=':')
    ax.minorticks_on()

    _sc_finalize(fig, pad=1.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# E1: Policy Mode Comparison
# ═══════════════════════════════════════════════════════════════════════

def make_policy_comparison_figure(policy_data, output_path):
    """Compare ratio and throughput across cost model policies.

    Args:
        policy_data: dict of {policy_label: [PhaseResult rows]}
                     e.g. {"balanced": rows, "ratio-only": rows, "speed-only": rows}
        output_path: output PNG path
    """
    policy_colors = {"balanced": "#3498db", "ratio-only": "#e74c3c",
                     "speed-only": "#2ecc71"}
    policies = list(policy_data.keys())
    nn_phases = ["nn", "nn-rl", "nn-rl+exp50"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Cost Model Policy Comparison (NN phases)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Ratio per policy
    ax = axes[0]
    x = np.arange(len(policies))
    ratios = []
    for pol in policies:
        rows = policy_data[pol]
        nn_rows = [r for r in rows if r.get("phase", "") in nn_phases]
        if nn_rows:
            ratios.append(max(g(r, "ratio") for r in nn_rows))
        else:
            ratios.append(0)
    colors = [policy_colors.get(p, "#bdc3c7") for p in policies]
    bars = ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(ratios):
        ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha='right')
    ax.set_ylabel("Best NN Compression Ratio")
    ax.set_title("Compression Ratio", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Panel 2: Throughput per policy
    ax = axes[1]
    throughputs = []
    for pol in policies:
        rows = policy_data[pol]
        nn_rows = [r for r in rows if r.get("phase", "") in nn_phases]
        if nn_rows:
            throughputs.append(max(g(r, "write_mibps", "write_mbps") for r in nn_rows))
        else:
            throughputs.append(0)
    bars = ax.bar(x, throughputs, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(throughputs):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha='right')
    ax.set_ylabel("Best NN Write Throughput (MiB/s)")
    ax.set_title("Write Throughput", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    # Panel 3: Pareto scatter (ratio vs throughput, one point per policy+phase)
    ax = axes[2]
    for pol in policies:
        rows = policy_data[pol]
        color = policy_colors.get(pol, "#bdc3c7")
        for r in rows:
            phase = r.get("phase", "")
            if phase not in nn_phases:
                continue
            ratio = g(r, "ratio")
            wtp = g(r, "write_mibps", "write_mbps")
            label_text = PHASE_LABELS.get(phase, phase).replace("\n", " ")
            ax.scatter(ratio, wtp, s=100, color=color, edgecolors="black",
                       linewidth=0.8, zorder=5)
            ax.annotate(f"{pol}\n{label_text}", (ratio, wtp),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color="#333")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Write Throughput (MiB/s)")
    ax.set_title("Ratio vs Throughput (Pareto)", fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# F1: Chunk-Size Scaling
# ═══════════════════════════════════════════════════════════════════════

def make_chunk_scaling_figure(chunk_data, output_path):
    """Plot ratio and throughput vs chunk size.

    Args:
        chunk_data: dict of {chunk_mb: {phase: {"ratio": float, "write": float}}}
        output_path: output PNG path
    """
    chunk_sizes = sorted(chunk_data.keys())
    phases = ["fixed-zstd", "nn", "nn-rl"]
    phase_styles = {
        "fixed-zstd": {"color": "#1a5276", "marker": "s", "ls": "-"},
        "nn":         {"color": "#e67e22", "marker": "o", "ls": "--"},
        "nn-rl":      {"color": "#8e44ad", "marker": "D", "ls": "-."},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chunk-Size Scaling Study", fontsize=14, fontweight="bold")

    for phase in phases:
        sty = phase_styles.get(phase, {"color": "black", "marker": ".", "ls": "-"})
        ratios = [chunk_data[c].get(phase, {}).get("ratio", 0) for c in chunk_sizes]
        writes = [chunk_data[c].get(phase, {}).get("write", 0) for c in chunk_sizes]
        label = PHASE_LABELS.get(phase, phase).replace("\n", " ")

        ax1.plot(chunk_sizes, ratios, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)
        ax2.plot(chunk_sizes, writes, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)

    for ax, ylabel, title in [
        (ax1, "Compression Ratio", "Ratio vs Chunk Size"),
        (ax2, "Write Throughput (MiB/s)", "Throughput vs Chunk Size"),
    ]:
        ax.set_xlabel("Chunk Size (MB)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(chunk_sizes)
        ax.set_xticklabels([str(c) for c in chunk_sizes])
        ax.legend()
        ax.grid(alpha=0.2, linestyle="--")
        ax.grid(which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# F2: Data-Size Scaling
# ═══════════════════════════════════════════════════════════════════════

def make_size_scaling_figure(size_data, output_path):
    """Plot ratio and throughput vs dataset size.

    Args:
        size_data: dict of {size_mb: {phase: {"ratio": float, "write": float}}}
        output_path: output PNG path
    """
    sizes = sorted(size_data.keys())
    phases = ["no-comp", "fixed-zstd", "nn", "nn-rl"]
    phase_styles = {
        "no-comp":    {"color": "#7f8c8d", "marker": "x", "ls": ":"},
        "fixed-zstd": {"color": "#1a5276", "marker": "s", "ls": "-"},
        "nn":         {"color": "#e67e22", "marker": "o", "ls": "--"},
        "nn-rl":      {"color": "#8e44ad", "marker": "D", "ls": "-."},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Data-Size Scaling Study (Gray-Scott)", fontsize=14, fontweight="bold")

    for phase in phases:
        sty = phase_styles.get(phase, {"color": "black", "marker": ".", "ls": "-"})
        ratios = [size_data[s].get(phase, {}).get("ratio", 0) for s in sizes]
        writes = [size_data[s].get(phase, {}).get("write", 0) for s in sizes]
        label = PHASE_LABELS.get(phase, phase).replace("\n", " ")

        ax1.plot(sizes, ratios, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)
        ax2.plot(sizes, writes, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)

    for ax, ylabel, title in [
        (ax1, "Compression Ratio", "Ratio vs Dataset Size"),
        (ax2, "Write Throughput (MiB/s)", "Throughput vs Dataset Size"),
    ]:
        ax.set_xlabel("Dataset Size (MB)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(alpha=0.2, linestyle="--")
        ax.grid(which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# I1: Rate-Distortion Curves
# ═══════════════════════════════════════════════════════════════════════

def make_rate_distortion_figure(rd_data, output_path):
    """Plot compression ratio vs error bound for lossy evaluation.

    Args:
        rd_data: dict of {error_bound_str: {phase: {"ratio": float, "write": float}}}
        output_path: output PNG path
    """
    ebs = sorted(rd_data.keys(), key=lambda x: float(x) if x != "lossless" else 0)
    phases = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#e67e22", "marker": "o", "ls": "--"},
        "nn-rl":       {"color": "#8e44ad", "marker": "D", "ls": "-"},
        "nn-rl+exp50": {"color": "#c0392b", "marker": "s", "ls": "-."},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lossy Rate-Distortion (Hurricane Isabel)", fontsize=14, fontweight="bold")

    eb_floats = [float(e) if e != "lossless" else 0 for e in ebs]

    for phase in phases:
        sty = phase_styles.get(phase, {"color": "black", "marker": ".", "ls": "-"})
        ratios = [rd_data[e].get(phase, {}).get("ratio", 0) for e in ebs]
        writes = [rd_data[e].get(phase, {}).get("write", 0) for e in ebs]
        label = PHASE_LABELS.get(phase, phase).replace("\n", " ")

        ax1.plot(range(len(ebs)), ratios, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)
        ax2.plot(range(len(ebs)), writes, color=sty["color"], marker=sty["marker"],
                 linestyle=sty["ls"], linewidth=2.0, markersize=6, label=label)

    for ax, ylabel, title in [
        (ax1, "Compression Ratio", "Ratio vs Error Bound"),
        (ax2, "Write Throughput (MiB/s)", "Throughput vs Error Bound"),
    ]:
        ax.set_xlabel("Error Bound", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(ebs)))
        ax.set_xticklabels(ebs)
        ax.legend()
        ax.grid(alpha=0.2, linestyle="--")
        ax.grid(which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    _sc_finalize(fig, pad=1.5, rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

ALL_VIEWS = ["summary", "timesteps", "actions"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified GPUCompress benchmark visualizer (summary + timesteps)")
    parser.add_argument("csvs", nargs="*",
                        help="Positional CSV paths (auto-classified as gs/vpic)")
    parser.add_argument("--gs-csv", help="Gray-Scott aggregate CSV")
    parser.add_argument("--gs-timesteps-csv", help="Gray-Scott timestep adaptation CSV")
    parser.add_argument("--vpic-csv", help="VPIC aggregate CSV")
    parser.add_argument("--vpic-dir", help="VPIC results directory (e.g. .../eval_NX156_chunk4mb_ts100/balanced_w1-1-1)")
    parser.add_argument("--gs-dir", help="Gray-Scott results directory")
    parser.add_argument("--sdrbench-dir", help="SDRBench results directory (contains benchmark_*.csv)")
    parser.add_argument("--output-dir", help="Output directory (default: alongside CSV)")
    parser.add_argument("--view", action="append", default=None,
                        help=f"Views to generate: {ALL_VIEWS} (default: all)")
    args = parser.parse_args()

    views = args.view or ALL_VIEWS

    # Classify positional CSVs
    for csv_path in args.csvs:
        low = csv_path.lower()
        if "vpic" in low:
            if not args.vpic_csv:
                args.vpic_csv = csv_path
        else:
            if not args.gs_csv:
                args.gs_csv = csv_path

    # Auto-detect from --vpic-dir / --gs-dir if provided
    if args.vpic_dir and not args.vpic_csv:
        d = args.vpic_dir
        args.vpic_csv = find_csv([os.path.join(d, "benchmark_vpic_deck.csv")])
    if args.gs_dir:
        d = args.gs_dir
        if not args.gs_csv:
            args.gs_csv = find_csv([os.path.join(d, "benchmark_grayscott_vol.csv")])
        if not args.gs_timesteps_csv:
            args.gs_timesteps_csv = find_csv([os.path.join(d, "benchmark_grayscott_timesteps.csv")])

    # If only one --*-dir is given, skip auto-detect for the other benchmark
    only_gs = args.gs_dir and not args.vpic_dir and not args.vpic_csv
    only_vpic = args.vpic_dir and not args.gs_dir and not args.gs_csv

    gs_agg = args.gs_csv or (None if only_vpic else find_csv(DEFAULT_GS_AGG))
    gs_tsteps = args.gs_timesteps_csv or (None if only_vpic else find_csv(DEFAULT_GS_TIMESTEPS))
    vpic_agg = args.vpic_csv or (None if only_gs else find_csv(DEFAULT_VPIC_AGG))

    found_any = False

    # ── Gray-Scott ──
    # Resolve output directory once: --output-dir > alongside aggregate CSV > default
    gs_out_dir = args.output_dir
    if not gs_out_dir and gs_agg and os.path.exists(gs_agg):
        gs_out_dir = os.path.dirname(os.path.abspath(gs_agg))
    if not gs_out_dir:
        gs_out_dir = os.path.join(SCRIPT_DIR, "grayscott", "results")

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
            meta = (f"Grid: {L}^3 ({orig:.0f} MiB) | Chunks: {n_ch} x {chunk_mb:.0f} MiB | "
                    f"Steps: {steps} | F={F_val}, k={k_val}")

            # Merge multi-timestep phases (nn-rl, nn-rl+exp50) into summary
            _ts_csv = gs_tsteps or find_csv(DEFAULT_GS_TIMESTEPS)
            _merge_timestep_phases(rows, _ts_csv, orig)

            make_summary_figure("Gray-Scott Simulation", rows,
                                os.path.join(gs_out_dir, "benchmark_grayscott.png"), meta)

    if gs_tsteps and os.path.exists(gs_tsteps) and "timesteps" in views:
        found_any = True
        print(f"Loading Gray-Scott timestep adaptation: {gs_tsteps}")
        make_timestep_figure(gs_tsteps, os.path.join(gs_out_dir, "sgd_accuracy_over_time.png"))
        make_sgd_exploration_figure(gs_tsteps, os.path.join(gs_out_dir, "sgd_exploration_firing.png"))

    # Look for chunk/milestone CSVs alongside the aggregate CSV first, then defaults
    gs_tc = find_csv([os.path.join(gs_out_dir, "benchmark_grayscott_timestep_chunks.csv")]
                      + DEFAULT_GS_TSTEP_CHUNKS)
    if gs_tc and os.path.exists(gs_tc) and "timesteps" in views:
        found_any = True
        print(f"Loading Gray-Scott per-chunk milestones: {gs_tc}")
        try:
            make_timestep_chunks_figure(gs_tc, os.path.join(gs_out_dir, "predicted_vs_actual_per_chunk.png"))
        except Exception as e:
            print(f"  Warning: per-chunk milestone plot failed: {e}")

    # ── Gray-Scott: Per-chunk actions (used as nn-inference reference for evolution plot) ──
    gs_chunks = find_csv([os.path.join(gs_out_dir, "benchmark_grayscott_vol_chunks.csv")]
                          + DEFAULT_GS_CHUNKS)

    # ── Gray-Scott: Milestone algorithm evolution ──
    gs_tc = find_csv([os.path.join(gs_out_dir, "benchmark_grayscott_timestep_chunks.csv")]
                      + DEFAULT_GS_TSTEP_CHUNKS)
    if gs_tc and os.path.exists(gs_tc) and "actions" in views:
        tc_rows = parse_csv(gs_tc)
        if tc_rows:
            found_any = True
            print(f"Loading Gray-Scott milestone algorithm evolution: {gs_tc}")
            try:
                make_milestone_actions_figure(gs_tc, os.path.join(gs_out_dir, "nn_algorithm_evolution.png"),
                                             chunk_csv_path=gs_chunks)
            except Exception as e:
                print(f"  Warning: milestone actions plot failed: {e}")

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
            meta = (f"Dataset: {orig:.0f} MiB | Chunks: {n_ch} x {chunk_mb:.0f} MiB | "
                    f"Source: VPIC Harris Sheet")
            out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "vpic-kokkos", "results")

            # Merge multi-timestep phases into VPIC summary
            _vpic_ts_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_deck_timesteps.csv"),
                                os.path.join(args.vpic_dir, "benchmark_vpic_timesteps.csv")]
                               if args.vpic_dir else []) + DEFAULT_VPIC_TIMESTEPS
            _vpic_ts = find_csv(_vpic_ts_search)
            _merge_timestep_phases(rows, _vpic_ts, orig)

            make_summary_figure("VPIC Harris Sheet Reconnection", rows,
                                os.path.join(out_dir, "benchmark_vpic.png"), meta)

    _vpic_ts_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_deck_timesteps.csv"),
                        os.path.join(args.vpic_dir, "benchmark_vpic_timesteps.csv")]
                       if args.vpic_dir else []) + ([] if only_gs else DEFAULT_VPIC_TIMESTEPS)
    vpic_tsteps = find_csv(_vpic_ts_search)
    if vpic_tsteps and os.path.exists(vpic_tsteps) and "timesteps" in views:
        found_any = True
        print(f"Loading VPIC timestep adaptation: {vpic_tsteps}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(vpic_tsteps))
        make_timestep_figure(vpic_tsteps, os.path.join(out_dir, "vpic_sgd_accuracy_over_time.png"))
        make_sgd_exploration_figure(vpic_tsteps, os.path.join(out_dir, "vpic_sgd_exploration_firing.png"))

    _vpic_tc_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_deck_timestep_chunks.csv"),
                        os.path.join(args.vpic_dir, "benchmark_vpic_timestep_chunks.csv")]
                       if args.vpic_dir else []) + ([] if only_gs else DEFAULT_VPIC_TSTEP_CHUNKS)
    vpic_tc = find_csv(_vpic_tc_search)
    if vpic_tc and os.path.exists(vpic_tc) and "timesteps" in views:
        found_any = True
        print(f"Loading VPIC per-chunk milestones: {vpic_tc}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(vpic_tc))
        try:
            make_timestep_chunks_figure(vpic_tc, os.path.join(out_dir, "vpic_predicted_vs_actual_per_chunk.png"))
        except Exception as e:
            print(f"  Warning: VPIC per-chunk milestone plot failed: {e}")

    # VPIC: Algorithm evolution (same as Gray-Scott)
    _vpic_chunks_search = [os.path.join(
        args.vpic_dir or args.output_dir or os.path.join(SCRIPT_DIR, "vpic-kokkos", "results"),
        f) for f in ("benchmark_vpic_deck_chunks.csv", "benchmark_vpic_chunks.csv")]
    vpic_chunks = find_csv(_vpic_chunks_search)
    if vpic_tc and os.path.exists(vpic_tc) and "actions" in views:
        found_any = True
        print(f"Loading VPIC milestone algorithm evolution: {vpic_tc}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(vpic_tc))
        try:
            make_milestone_actions_figure(vpic_tc, os.path.join(out_dir, "vpic_nn_algorithm_evolution.png"),
                                         chunk_csv_path=vpic_chunks)
        except Exception as e:
            print(f"  Warning: VPIC milestone actions plot failed: {e}")

    # ── SDRBench datasets ──
    sdrbench_dir = args.sdrbench_dir or os.path.join(SCRIPT_DIR, "sdrbench", "results")

    if os.path.isdir(sdrbench_dir) and "summary" in views:
        import glob as _glob
        sdr_csvs = sorted(_glob.glob(os.path.join(sdrbench_dir, "benchmark_*.csv")))
        # Exclude chunk/timestep CSVs
        sdr_csvs = [c for c in sdr_csvs
                     if "_chunks" not in c and "_timesteps" not in c
                     and "_timestep_" not in c]
        for sdr_csv in sdr_csvs:
            found_any = True
            print(f"Loading SDRBench: {sdr_csv}")
            rows = parse_csv(sdr_csv)
            if not rows:
                continue
            r0 = rows[0]
            dataset_name = r0.get("dataset", os.path.basename(sdr_csv).replace("benchmark_", "").replace(".csv", ""))
            orig_bytes = g(r0, "orig_bytes")
            orig_mib = orig_bytes / (1024 * 1024) if orig_bytes > 1024 else orig_bytes
            n_ch = int(g(r0, "n_chunks"))
            chunk_mb = orig_mib / max(n_ch, 1)
            n_runs = int(g(r0, "n_runs")) if g(r0, "n_runs") else 1
            meta = f"Dataset: {dataset_name} ({orig_mib:.0f} MiB) | Chunks: {n_ch} x {chunk_mb:.0f} MiB"
            # Infer cost model policy from directory name
            dir_lower = sdrbench_dir.lower()
            if "ratio_only" in dir_lower or "w0-0-1" in dir_lower:
                meta += " | Policy: ratio-only (w0=0, w1=0, w2=1)"
            elif "speed_only" in dir_lower or "w1-1-0" in dir_lower:
                meta += " | Policy: speed-only (w0=1, w1=1, w2=0)"
            elif "balanced" in dir_lower or "w1-1-1" in dir_lower:
                meta += " | Policy: balanced (w0=1, w1=1, w2=1)"
            if n_runs > 1:
                meta += f" | Runs: {n_runs}"

            # Merge multi-field phases if timestep CSV exists
            ts_csv = os.path.join(sdrbench_dir, f"benchmark_{dataset_name}_timesteps.csv")
            if os.path.exists(ts_csv):
                _merge_timestep_phases(rows, ts_csv, orig_mib)

            out_name = f"benchmark_{dataset_name}.png"
            _DATASET_TITLES = {
                "hurricane_isabel": "Hurricane Isabel (Climate, 100x500x500)",
                "nyx": "Nyx Cosmology (512x512x512)",
                "cesm_atm": "CESM-ATM Atmosphere (1800x3600)",
            }
            display_name = _DATASET_TITLES.get(dataset_name, dataset_name)
            make_summary_figure(f"SDRBench: {display_name}", rows,
                                os.path.join(sdrbench_dir, out_name), meta)

    if not found_any:
        print("ERROR: No benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_GS_AGG + DEFAULT_VPIC_AGG:
            print(f"  {p}")
        print(f"  {sdrbench_dir}/benchmark_*.csv")
        print("\nRun benchmarks first, or specify paths explicitly.")
        sys.exit(1)

    # ── Multi-dataset comparison figure ──
    # Collect all summary data loaded above into a single comparison
    all_datasets_for_comparison = []

    if gs_agg and os.path.exists(gs_agg):
        rows = parse_csv(gs_agg)
        if rows:
            all_datasets_for_comparison.append(("Gray-Scott", rows))
    if vpic_agg and os.path.exists(vpic_agg):
        rows = parse_csv(vpic_agg)
        if rows:
            all_datasets_for_comparison.append(("VPIC", rows))
    if os.path.isdir(sdrbench_dir):
        import glob as _glob2
        for sdr_csv in sorted(_glob2.glob(os.path.join(sdrbench_dir, "benchmark_*.csv"))):
            if "_chunks" in sdr_csv or "_timesteps" in sdr_csv or "_timestep_" in sdr_csv:
                continue
            rows = parse_csv(sdr_csv)
            if rows:
                ds = rows[0].get("dataset", os.path.basename(sdr_csv).replace("benchmark_", "").replace(".csv", ""))
                all_datasets_for_comparison.append((ds, rows))

    if len(all_datasets_for_comparison) >= 2 and "summary" in views:
        out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating multi-dataset comparison ({len(all_datasets_for_comparison)} datasets)")
        make_multi_dataset_figure(all_datasets_for_comparison,
                                  os.path.join(out_dir, "multi_dataset_comparison.png"))
        make_per_dataset_phase_comparison(all_datasets_for_comparison,
                                          os.path.join(out_dir, "per_dataset_phase_comparison.png"))

    # ── Latency breakdown (from Gray-Scott which has per-component timing) ──
    if gs_agg and os.path.exists(gs_agg) and "summary" in views:
        rows = parse_csv(gs_agg)
        if any(g(r, "comp_ms") > 0 or g(r, "nn_ms") > 0 for r in rows):
            print("Generating latency breakdown figure")
            make_latency_breakdown_figure(rows,
                os.path.join(gs_out_dir, "latency_breakdown.png"), "Gray-Scott")

    # ── Algorithm selection histogram (from chunk CSVs) ──
    chunk_csv_paths = []
    gs_chunks_path = find_csv([os.path.join(gs_out_dir, "benchmark_grayscott_vol_chunks.csv")]
                               + DEFAULT_GS_CHUNKS)
    if gs_chunks_path and os.path.exists(gs_chunks_path):
        chunk_csv_paths.append(("Gray-Scott", gs_chunks_path))

    if os.path.isdir(sdrbench_dir):
        import glob as _glob3
        for sdr_csv in sorted(_glob3.glob(os.path.join(sdrbench_dir, "benchmark_*_chunks.csv"))):
            if "_timestep_" in sdr_csv:
                continue
            ds = os.path.basename(sdr_csv).replace("benchmark_", "").replace("_chunks.csv", "")
            chunk_csv_paths.append((ds, sdr_csv))

    if chunk_csv_paths and "summary" in views:
        out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating algorithm histogram ({len(chunk_csv_paths)} datasets)")
        make_algorithm_histogram(chunk_csv_paths,
                                 os.path.join(out_dir, "algorithm_selection_histogram.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
