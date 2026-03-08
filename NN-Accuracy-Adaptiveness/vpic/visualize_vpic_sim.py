#!/usr/bin/env python3
"""
VPIC Simulation Benchmark — Executive Adaptation Figures.

Produces 4 paper-quality figures demonstrating online SGD adaptation.

Reads:
  sim_chunk_metrics.csv    — per-chunk per-timestep predicted/actual ratios
  sim_timestep_metrics.csv — per-timestep aggregated metrics
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# -- CSV parsing ---------------------------------------------------------------

def parse_csv(path):
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
    for k in keys:
        if k in row:
            return row[k]
    return default


# -- Helpers -------------------------------------------------------------------

DEFAULT_DIR = "benchmarks/vpic"
STABLE_MEDIAN_MAPE = 200  # chunks with median APE < 200% are "stable"


def _build_chunk_mape(chunk_rows, phase):
    """Return {chunk_id: [(timestep, mape), ...]} for a given phase."""
    series = {}
    for r in chunk_rows:
        if r.get("phase") != phase:
            continue
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if pred <= 0 or actual <= 0:
            continue
        ts = int(g(r, "timestep"))
        ci = int(g(r, "chunk"))
        ape = abs(pred - actual) / actual * 100.0
        series.setdefault(ci, []).append((ts, ape))
    for ci in series:
        series[ci].sort()
    return series


def _get_timesteps(series):
    return sorted(set(ts for ci in series for ts, _ in series[ci]))


def _mape_at_timestep(series, chunk_ids, ts):
    vals = []
    for ci in chunk_ids:
        for t, m in series.get(ci, []):
            if t == ts:
                vals.append(m)
    return vals


def _classify_stable(series):
    stable, unstable = [], []
    for ci, data in series.items():
        median_mape = np.median([m for _, m in data])
        if median_mape < STABLE_MEDIAN_MAPE:
            stable.append(ci)
        else:
            unstable.append(ci)
    return sorted(stable), sorted(unstable)


def _pct_fmt(val, _):
    if val >= 1e6:
        return f"{val / 1e6:.0f}M%"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K%"
    return f"{val:.0f}%"


# -- Figure 1: Average MAPE Over Simulation Time ------------------------------

def fig1_avg_mape(chunk_rows, output_dir):
    """Average MAPE for Oracle, Baseline, SGD over timesteps."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    bl = _build_chunk_mape(chunk_rows, "nn_baseline")

    if not sgd:
        return

    timesteps = _get_timesteps(sgd)
    all_chunks = sorted(sgd.keys())
    n = len(all_chunks)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Oracle: MAPE = 0
    ax.axhline(y=0, color="#e74c3c", linewidth=2.5, linestyle="-", alpha=0.8,
               label="Oracle (MAPE = 0%)", zorder=2)

    # Baseline
    bl_avgs = [np.mean(_mape_at_timestep(bl, all_chunks, ts))
               for ts in timesteps]
    ax.plot(timesteps, bl_avgs, color="#2980b9", linewidth=2.5,
            marker="s", markersize=5, label="NN Baseline (no learning)",
            zorder=3, alpha=0.9)
    for i, (ts, val) in enumerate(zip(timesteps, bl_avgs)):
        va = "bottom" if i % 2 == 0 else "top"
        offset = 8 if i % 2 == 0 else -8
        ax.annotate(f"{val:.0f}%", (ts, val), fontsize=6,
                    color="#1f618d", fontweight="bold", ha="center", va=va,
                    textcoords="offset points", xytext=(0, offset))

    # SGD
    sgd_avgs = [np.mean(_mape_at_timestep(sgd, all_chunks, ts))
                for ts in timesteps]
    ax.plot(timesteps, sgd_avgs, color="#27ae60", linewidth=2.5,
            marker="^", markersize=5, label="NN + SGD (online learning)",
            zorder=4, alpha=0.9)
    for i, (ts, val) in enumerate(zip(timesteps, sgd_avgs)):
        va = "top" if i % 2 == 0 else "bottom"
        offset = -8 if i % 2 == 0 else 8
        ax.annotate(f"{val:.0f}%", (ts, val), fontsize=6,
                    color="#1a7a3a", fontweight="bold", ha="center", va=va,
                    textcoords="offset points", xytext=(0, offset))

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Average Prediction Error (MAPE %)", fontsize=12)
    ax.set_title("Online SGD Reduces Prediction Error Over Simulation Time\n"
                 f"(Average across {n} chunks per timestep)",
                 fontsize=14)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_avg_mape.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 2: All Chunks vs Stable Chunks ------------------------------------

def fig2_all_vs_stable(chunk_rows, output_dir):
    """Two panels with average MAPE and per-point annotations."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    bl = _build_chunk_mape(chunk_rows, "nn_baseline")

    if not sgd:
        return

    timesteps = _get_timesteps(sgd)
    all_chunks = sorted(sgd.keys())
    stable, unstable = _classify_stable(sgd)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    panels = [
        (axes[0], all_chunks, f"All {len(all_chunks)} Chunks"),
        (axes[1], stable, f"{len(stable)} Stable Chunks "
                          f"(excluding boundary chunks {unstable})"),
    ]

    for ax, chunk_set, title in panels:
        if not chunk_set:
            ax.text(0.5, 0.5, "No chunks in this category",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        # Baseline
        bl_avgs = [np.mean(_mape_at_timestep(bl, chunk_set, ts))
                   for ts in timesteps]
        ax.plot(timesteps, bl_avgs, color="#2980b9", linewidth=2.5,
                marker="s", markersize=4, label="NN Baseline", zorder=3)

        # SGD
        sgd_avgs = [np.mean(_mape_at_timestep(sgd, chunk_set, ts))
                    for ts in timesteps]
        ax.plot(timesteps, sgd_avgs, color="#27ae60", linewidth=2.5,
                marker="^", markersize=4, label="NN + SGD", zorder=4)

        # Oracle
        ax.axhline(y=0, color="#e74c3c", linewidth=1.5, linestyle="--",
                   alpha=0.6, label="Oracle (0%)", zorder=2)

        # Annotate each data point
        for i, ts in enumerate(timesteps):
            va_bl = "bottom" if i % 2 == 0 else "top"
            off_bl = 6 if i % 2 == 0 else -6
            ax.annotate(f"{bl_avgs[i]:.0f}", (ts, bl_avgs[i]), fontsize=5,
                        color="#1f618d", fontweight="bold", ha="center",
                        va=va_bl, textcoords="offset points",
                        xytext=(0, off_bl))

            va_sgd = "top" if i % 2 == 0 else "bottom"
            off_sgd = -6 if i % 2 == 0 else 6
            ax.annotate(f"{sgd_avgs[i]:.0f}", (ts, sgd_avgs[i]), fontsize=5,
                        color="#1a7a3a", fontweight="bold", ha="center",
                        va=va_sgd, textcoords="offset points",
                        xytext=(0, off_sgd))

        ax.set_ylabel("Average MAPE (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("Simulation Timestep", fontsize=12)
    fig.suptitle("Model Adaptation: All Chunks vs Stable Chunks\n"
                 "(Average MAPE per timestep)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_all_vs_stable.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 3: Per-Chunk MAPE Reduction (first vs last) -----------------------

def fig3_per_chunk_reduction(chunk_rows, output_dir):
    """Bar chart: first vs last MAPE per chunk."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")

    if not sgd:
        return

    chunk_ids = sorted(sgd.keys())
    n = len(chunk_ids)

    first_mapes, last_mapes = [], []
    for ci in chunk_ids:
        data = sgd[ci]
        first_mapes.append(data[0][1])
        last_mapes.append(data[-1][1])

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 6))

    x = np.arange(n)
    bar_w = 0.38

    ax.bar(x - bar_w / 2, first_mapes, bar_w, color="#e74c3c", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="First diagnostic",
           zorder=3)
    ax.bar(x + bar_w / 2, last_mapes, bar_w, color="#27ae60", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Last diagnostic",
           zorder=3)

    # Annotate reduction %
    for i, (f_m, l_m) in enumerate(zip(first_mapes, last_mapes)):
        if f_m > 0:
            reduction = (1 - l_m / f_m) * 100
            y_pos = max(f_m, l_m) * 1.15
            ax.text(x[i], y_pos, f"{reduction:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#27ae60" if reduction > 0 else "#e74c3c",
                    zorder=5)

    # Summary stats
    reductions = [(1 - l / f) * 100 for f, l in zip(first_mapes, last_mapes)
                  if f > 0]
    mean_red = np.mean(reductions)
    median_red = np.median(reductions)

    n_improved = sum(1 for r in reductions if r > 0)
    summary = (f"{n_improved}/{n} chunks improved\n"
               f"Mean reduction: {mean_red:.0f}%\n"
               f"Median reduction: {median_red:.0f}%")
    ax.text(0.98, 0.95, summary, transform=ax.transAxes, fontsize=9,
            ha="right", va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#d5f5e3",
                      edgecolor="#27ae60", alpha=0.95))

    ax.set_xlabel("Chunk Index", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_title("Per-Chunk Prediction Error: Before vs After Online Learning",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ci) for ci in chunk_ids], fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_per_chunk_reduction.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 4: Compression Ratio Over Simulation Time -------------------------

def fig4_compression_ratio(ts_rows, output_dir):
    """Per-timestep compression ratio for Oracle, Baseline, SGD."""
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = [
        ("oracle",      "#e74c3c", "D", "Oracle (best-static)", 2.5),
        ("nn_baseline", "#2980b9", "s", "NN Baseline",          2.5),
        ("nn_sgd",      "#27ae60", "^", "NN + SGD (online)",    2.5),
    ]

    for phase, color, marker, label, lw in phases:
        data = [r for r in ts_rows if r.get("phase") == phase]
        data.sort(key=lambda r: g(r, "timestep"))
        if not data:
            continue
        ts = [g(r, "timestep") for r in data]
        ratios = [g(r, "ratio") for r in data]

        ax.plot(ts, ratios, marker=marker, color=color, linewidth=lw,
                alpha=0.9, label=label, markersize=5, zorder=3)

        # Annotate each point
        for i, (t, ratio) in enumerate(zip(ts, ratios)):
            va = "bottom" if i % 2 == 0 else "top"
            offset = 7 if i % 2 == 0 else -7
            ax.annotate(f"{ratio:.2f}", (t, ratio), fontsize=5,
                        color=color, fontweight="bold", ha="center",
                        va=va, textcoords="offset points",
                        xytext=(0, offset))

    # Overall averages in text box
    avgs = {}
    for phase, _, _, label, _ in phases:
        vals = [g(r, "ratio") for r in ts_rows if r.get("phase") == phase]
        if vals:
            # Use total_orig / total_compressed for correct aggregate
            orig = [g(r, "orig_mib") for r in ts_rows if r.get("phase") == phase]
            compressed = [g(r, "file_mib") for r in ts_rows if r.get("phase") == phase]
            total_orig = sum(orig)
            total_comp = sum(compressed)
            if total_comp > 0:
                avgs[label] = total_orig / total_comp
            else:
                # Oracle has file_mib=0, use harmonic mean
                avgs[label] = len(vals) / sum(1.0/r for r in vals if r > 0)

    if avgs:
        lines = [f"{k}: {v:.2f}x" for k, v in avgs.items()]
        ax.text(0.02, 0.02, "Overall:\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#ccc", alpha=0.95))

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_title("Compression Ratio Over Simulation Time\n"
                 "(Oracle, NN Baseline, NN + SGD)",
                 fontsize=14)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_compression_ratio.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VPIC Simulation — Executive Adaptation Figures")
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    chunk_csv = os.path.join(output_dir, "sim_chunk_metrics.csv")
    ts_csv = os.path.join(output_dir, "sim_timestep_metrics.csv")

    if not os.path.exists(chunk_csv):
        print(f"ERROR: {chunk_csv} not found")
        sys.exit(1)

    chunk_rows = parse_csv(chunk_csv)
    print(f"Loaded {len(chunk_rows)} chunk rows from {chunk_csv}")

    ts_rows = parse_csv(ts_csv) if os.path.exists(ts_csv) else []
    if ts_rows:
        print(f"Loaded {len(ts_rows)} timestep rows from {ts_csv}")

    # Quick summary
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    timesteps = _get_timesteps(sgd)
    stable, unstable = _classify_stable(sgd)
    print(f"  Chunks: {len(sgd)} total, {len(stable)} stable, "
          f"{len(unstable)} boundary/unstable {unstable}")
    print(f"  Timesteps: {len(timesteps)} "
          f"({int(timesteps[0])} to {int(timesteps[-1])})")
    print()

    print("Generating figures...")
    fig1_avg_mape(chunk_rows, output_dir)
    fig2_all_vs_stable(chunk_rows, output_dir)
    fig3_per_chunk_reduction(chunk_rows, output_dir)
    if ts_rows:
        fig4_compression_ratio(ts_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
