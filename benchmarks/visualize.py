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

# ═══════════════════════════════════════════════════════════════════════
# Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PHASE_ORDER = ["no-comp", "exhaustive", "nn", "nn-rl", "nn-rl+exp50", "nn-rl+exp"]

PHASE_COLORS = {
    "no-comp":     "#999999",
    "exhaustive":  "#5778a4",
    "nn":          "#e49444",
    "nn-rl":       "#6a9f58",
    "nn-rl+exp50": "#c85a5a",
    "nn-rl+exp":   "#c85a5a",
}

PHASE_LABELS = {
    "no-comp":     "No Comp",
    "exhaustive":  "Exhaustive\nSearch",
    "nn":          "NN\n(Inference)",
    "nn-rl":       "NN+SGD",
    "nn-rl+exp50": "NN+SGD\n+Explore",
    "nn-rl+exp":   "NN+SGD\n+Explore",
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
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_timesteps.csv"),
]
DEFAULT_VPIC_TSTEP_CHUNKS = [
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


_PHASE_ALIASES = {"oracle": "exhaustive"}


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
            "smape_ratio_pct": _avg_all(ph_rows, "smape_ratio"),
            "smape_comp_pct": _avg_all(ph_rows, "smape_comp"),
            "smape_decomp_pct": _avg_all(ph_rows, "smape_decomp"),
            "mape_ratio_pct": _avg_all(ph_rows, "mape_ratio"),
            "mape_comp_pct": _avg_all(ph_rows, "mape_comp"),
            "mape_decomp_pct": _avg_all(ph_rows, "mape_decomp"),
        })
        merged.append(ph)
    if merged:
        print(f"  Merged multi-timestep phases (total throughput): {merged}")


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
    ax.set_ylabel("Size (MiB)", fontsize=10)
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
              "End-to-End Write Throughput\n(NN + compress + I/O)", "MiB/s", fmt="%.0f")
    ax3 = fig.add_subplot(gs[1, 0])
    plot_bars(ax3, phases, [g(r, "read_mibps", "read_mbps") for r in ordered],
              "End-to-End Read Throughput\n(I/O + decompress)", "MiB/s", fmt="%.0f")
    ax4 = fig.add_subplot(gs[1, 1])
    plot_file_sizes(ax4, phases, ordered)
    ax5 = fig.add_subplot(gs[2, 0])
    plot_nn_stats(ax5, phases, ordered)
    ax6 = fig.add_subplot(gs[2, 1])
    plot_verification(ax6, phases, ordered)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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

    # Group by phase (new CSV has 'phase' column; old CSV doesn't)
    has_phase = "phase" in rows[0] if rows else False
    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl") if has_phase else "nn-rl"
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#999999", "ls": ":",  "marker": "s", "lw": 1.5},
        "nn-rl":       {"color": "#6a9f58", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c85a5a", "ls": "--", "marker": "D", "lw": 1.8},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    metric_keys = [
        ("Compression Ratio",  "mape_ratio",  "smape_ratio"),
        ("Compression Time",   "mape_comp",   "smape_comp"),
        ("Decompression Time", "mape_decomp", "smape_decomp"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("NN Prediction Accuracy Over Timesteps\n"
                 "(per-metric MAPE averaged across all chunks)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ax, (label, mape_key, smape_key) in zip(axes, metric_keys):
        for ph in phases_present:
            ph_rows = by_phase[ph]
            timesteps = np.array([g(r, "timestep") for r in ph_rows])
            mape = np.array([g(r, mape_key, default=-1) for r in ph_rows])
            if mape[0] < 0:
                mape = np.array([g(r, smape_key) for r in ph_rows])
            clipped = np.clip(mape, 0, 200)

            sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 1.5})
            ax.plot(timesteps, clipped, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=4, linewidth=sty["lw"],
                    label=ph, alpha=0.9, zorder=3)

        ax.axhline(20, color="#e67e22", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_ylabel(f"{label}\nMAPE (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 200)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(fontsize=9, loc="upper right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Timestep", fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

    by_phase = {}
    for r in rows:
        ph = r.get("phase", "nn-rl")
        by_phase.setdefault(ph, []).append(r)

    phase_order = ["nn", "nn-rl", "nn-rl+exp50"]
    phase_styles = {
        "nn":          {"color": "#999999", "ls": ":",  "marker": "s", "lw": 1.5},
        "nn-rl":       {"color": "#6a9f58", "ls": "-",  "marker": "o", "lw": 2.0},
        "nn-rl+exp50": {"color": "#c85a5a", "ls": "--", "marker": "D", "lw": 1.8},
    }
    phases_present = [p for p in phase_order if p in by_phase]

    fig, (ax_sgd, ax_exp) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("SGD & Exploration Firing Over Timesteps\n"
                 "(per-chunk cost model error triggers: SGD > 10%, Explore > 20%)",
                 fontsize=14, fontweight="bold", y=0.99)

    for ph in phases_present:
        ph_rows = by_phase[ph]
        timesteps = np.array([g(r, "timestep") for r in ph_rows])
        sgd = np.array([g(r, "sgd_fires") for r in ph_rows])
        expl = np.array([g(r, "explorations", default=0) for r in ph_rows])
        sty = phase_styles.get(ph, {"color": "black", "ls": "-", "marker": ".", "lw": 1.5})

        ax_sgd.plot(timesteps, sgd, color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=4, linewidth=sty["lw"],
                    label=PHASE_LABELS.get(ph, ph).replace("\n", " "),
                    alpha=0.9, zorder=3)
        if expl.sum() > 0:
            ax_exp.plot(timesteps, expl, color=sty["color"], linestyle=sty["ls"],
                        marker=sty["marker"], markersize=4, linewidth=sty["lw"],
                        label=PHASE_LABELS.get(ph, ph).replace("\n", " "),
                        alpha=0.9, zorder=3)

    ax_sgd.set_ylabel("SGD Fires\n(chunks per timestep)", fontsize=11, fontweight="bold")
    ax_sgd.set_title("SGD Weight Updates (chunks where cost_model_error > 10%)", fontsize=11)
    ax_sgd.grid(axis="y", alpha=0.3, linestyle="--")
    ax_sgd.tick_params(axis="both", labelsize=9)
    ax_sgd.legend(fontsize=9, loc="upper right")
    for spine in ["top", "right"]:
        ax_sgd.spines[spine].set_visible(False)

    ax_exp.set_ylabel("Explorations\n(chunks per timestep)", fontsize=11, fontweight="bold")
    ax_exp.set_xlabel("Timestep", fontsize=11, fontweight="bold")
    ax_exp.set_title("Exploration Triggers (chunks where cost_model_error > 20%)", fontsize=11)
    ax_exp.grid(axis="y", alpha=0.3, linestyle="--")
    ax_exp.tick_params(axis="both", labelsize=9)
    ax_exp.legend(fontsize=9, loc="upper right")
    for spine in ["top", "right"]:
        ax_exp.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    phase_label = phase_filter if has_phase else "nn-rl"
    fig.suptitle(f"Predicted vs Actual Per Chunk [{phase_label}]",
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

    fig, axes = plt.subplots(2, 1, figsize=(16, 3 + 1.5 * n_phases),
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
    ax_bar.set_xticklabels(bar_configs, fontsize=8, rotation=40, ha="right")
    ax_bar.set_ylabel("% of Chunks", fontsize=10)
    ax_bar.set_title("Configuration Selection Frequency", fontsize=11)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
        ts = int(g(r, "timestep"))
        by_phase_ts.setdefault(ph, {}).setdefault(ts, []).append(r)

    # Load single-shot nn phase for reference (static column)
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

    # Build phase columns: nn (static ref) + available multi-timestep phases
    phase_columns = []  # list of (label, data_source)
    if nn_ref_strip is not None:
        phase_columns.append(("NN Inference\n(static)", "nn_ref"))
    ts_phase_order = ["nn-rl", "nn-rl+exp50"]
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
                 fontsize=16, fontweight="bold", y=0.97)

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

    fig.legend(handles=legend_patches, loc="lower center", fontsize=10,
               ncol=min(len(legend_patches), 6), framealpha=0.95,
               edgecolor="#cccccc", fancybox=True,
               bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0.05, 0.08, 1, 0.93])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
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
    if args.gs_dir and not args.gs_csv:
        d = args.gs_dir
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
            _vpic_ts_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_timesteps.csv")]
                               if args.vpic_dir else []) + DEFAULT_VPIC_TIMESTEPS
            _vpic_ts = find_csv(_vpic_ts_search)
            _merge_timestep_phases(rows, _vpic_ts, orig)

            make_summary_figure("VPIC Harris Sheet Reconnection", rows,
                                os.path.join(out_dir, "benchmark_vpic.png"), meta)

    _vpic_ts_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_timesteps.csv")]
                       if args.vpic_dir else []) + ([] if only_gs else DEFAULT_VPIC_TIMESTEPS)
    vpic_tsteps = find_csv(_vpic_ts_search)
    if vpic_tsteps and os.path.exists(vpic_tsteps) and "timesteps" in views:
        found_any = True
        print(f"Loading VPIC timestep adaptation: {vpic_tsteps}")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(vpic_tsteps))
        make_timestep_figure(vpic_tsteps, os.path.join(out_dir, "vpic_sgd_accuracy_over_time.png"))
        make_sgd_exploration_figure(vpic_tsteps, os.path.join(out_dir, "vpic_sgd_exploration_firing.png"))

    _vpic_tc_search = ([os.path.join(args.vpic_dir, "benchmark_vpic_timestep_chunks.csv")]
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
        "benchmark_vpic_chunks.csv")]
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
