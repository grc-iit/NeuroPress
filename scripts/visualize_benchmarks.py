#!/u/imuradli/GPUCompress/.venv/bin/python3
"""
Visualize benchmark results from Gray-Scott and VPIC VOL benchmarks.

Reads CSV files produced by:
  - benchmark_grayscott_vol  → tests/benchmark_grayscott_vol_results/benchmark_grayscott_vol.csv
  - vpic_benchmark_deck      → benchmark_vpic_deck_results/benchmark_vpic_deck.csv

Produces a multi-panel figure comparing compression ratio, write/read
throughput, file sizes, and NN adaptation stats across phases and sources.

Usage:
  python3 scripts/visualize_benchmarks.py [options]

  --gs-csv PATH    Gray-Scott CSV (default: auto-detect)
  --vpic-csv PATH  VPIC CSV (default: auto-detect)
  --output PATH    Output image path (default: benchmark_comparison.png)
  --no-show        Don't display the plot, only save
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
    """Parse a CSV file into a list of dicts."""
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


def load_grayscott_csv(path):
    """Load Gray-Scott CSV and normalize to common schema."""
    rows = parse_csv(path)
    for r in rows:
        r["source"] = "Gray-Scott"
    return rows


def load_vpic_csv(path):
    """Load VPIC deck CSV (already has 'source' column)."""
    rows = parse_csv(path)
    for r in rows:
        r.setdefault("source", "VPIC")
    return rows


# ── Auto-detect CSV paths ──────────────────────────────────────────────

DEFAULT_GS_PATHS = [
    "tests/benchmark_grayscott_vol_results/benchmark_grayscott_vol.csv",
]

DEFAULT_VPIC_PATHS = [
    "benchmark_vpic_deck_results/benchmark_vpic_deck.csv",
]


def find_csv(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Colors and styling ─────────────────────────────────────────────────

PHASE_ORDER = ["no-comp", "static", "nn", "nn-rl", "nn-rl+exp50"]

PHASE_COLORS = {
    "no-comp":     "#95a5a6",
    "static":      "#3498db",
    "nn":          "#2ecc71",
    "nn-rl":       "#e67e22",
    "nn-rl+exp50": "#e74c3c",
}

PHASE_LABELS = {
    "no-comp":     "No Compression",
    "static":      "Static (LZ4+Shuffle)",
    "nn":          "NN (Inference)",
    "nn-rl":       "NN + SGD",
    "nn-rl+exp50": "NN + SGD + Explore",
}

SOURCE_HATCHES = {
    "Gray-Scott": "",
    "VPIC":       "//",
}


def get_phase_idx(phase):
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return len(PHASE_ORDER)


# ── Plot helpers ───────────────────────────────────────────────────────

def plot_grouped_bars(ax, sources_data, metric_key, title, ylabel,
                      fmt="%.1f", log_scale=False):
    """Plot grouped bars: one group per phase, one bar per source."""
    sources = list(sources_data.keys())
    n_sources = len(sources)
    phases = PHASE_ORDER

    width = 0.35 if n_sources == 2 else 0.5
    x = np.arange(len(phases))

    for si, src in enumerate(sources):
        data = sources_data[src]
        vals = []
        for phase in phases:
            matching = [r for r in data if r["phase"] == phase]
            vals.append(matching[0].get(metric_key, 0) if matching else 0)

        offset = (si - (n_sources - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=[PHASE_COLORS.get(p, "#bdc3c7") for p in phases],
                      hatch=SOURCE_HATCHES.get(src, ""),
                      edgecolor="white", linewidth=0.5,
                      label=src, alpha=0.85 if si == 0 else 0.65)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt % v, ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    if n_sources > 1:
        ax.legend(fontsize=8, loc="upper left")


def plot_nn_stats(ax, sources_data):
    """Plot SGD fires and exploration counts for NN phases."""
    nn_phases = ["nn", "nn-rl", "nn-rl+exp50"]
    sources = list(sources_data.keys())

    x = np.arange(len(nn_phases))
    width = 0.2

    for si, src in enumerate(sources):
        data = sources_data[src]
        sgd_vals = []
        expl_vals = []
        for phase in nn_phases:
            matching = [r for r in data if r["phase"] == phase]
            if matching:
                r = matching[0]
                n_chunks = r.get("n_chunks", 1)
                sgd_vals.append(100.0 * r.get("sgd_fires", 0) / max(n_chunks, 1))
                expl_vals.append(100.0 * r.get("explorations", 0) / max(n_chunks, 1))
            else:
                sgd_vals.append(0)
                expl_vals.append(0)

        offset_base = si * (2 * width + 0.05)
        ax.bar(x + offset_base, sgd_vals, width,
               color="#e67e22", hatch=SOURCE_HATCHES.get(src, ""),
               edgecolor="white", alpha=0.8,
               label=f"{src} SGD %" if si == 0 else f"{src} SGD %")
        ax.bar(x + offset_base + width, expl_vals, width,
               color="#e74c3c", hatch=SOURCE_HATCHES.get(src, ""),
               edgecolor="white", alpha=0.8,
               label=f"{src} Expl %" if si == 0 else f"{src} Expl %")

    ax.set_xticks(x + width * len(sources) / 2)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in nn_phases],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("% of chunks", fontsize=9)
    ax.set_title("NN Adaptation: SGD Fires & Explorations", fontsize=11,
                 fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def plot_file_sizes(ax, sources_data):
    """Plot original vs compressed file sizes per phase."""
    sources = list(sources_data.keys())
    phases = PHASE_ORDER
    n_sources = len(sources)
    width = 0.35 if n_sources == 2 else 0.5
    x = np.arange(len(phases))

    for si, src in enumerate(sources):
        data = sources_data[src]
        orig_vals = []
        comp_vals = []
        for phase in phases:
            matching = [r for r in data if r["phase"] == phase]
            if matching:
                r = matching[0]
                orig_vals.append(r.get("orig_mb", 0))
                comp_vals.append(r.get("file_mb", 0))
            else:
                orig_vals.append(0)
                comp_vals.append(0)

        offset = (si - (n_sources - 1) / 2) * width
        ax.bar(x + offset, comp_vals, width * 0.9,
               color=[PHASE_COLORS.get(p, "#bdc3c7") for p in phases],
               hatch=SOURCE_HATCHES.get(src, ""),
               edgecolor="white", linewidth=0.5,
               alpha=0.85 if si == 0 else 0.65,
               label=src)

        for bar_x, cv, ov in zip(x + offset, comp_vals, orig_vals):
            if cv > 0 and ov > 0:
                ax.text(bar_x, cv, f"{cv:.0f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    # Draw original size reference line
    for si, src in enumerate(sources):
        data = sources_data[src]
        orig = 0
        for r in data:
            if r.get("orig_mb", 0) > orig:
                orig = r["orig_mb"]
        if orig > 0:
            ls = "-" if si == 0 else "--"
            ax.axhline(orig, color="black", linewidth=1, linestyle=ls, alpha=0.5)
            ax.text(len(phases) - 0.5, orig * 1.02,
                    f"{src} raw: {orig:.0f} MB",
                    ha="right", va="bottom", fontsize=7, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("File Size (MB)", fontsize=9)
    ax.set_title("Compressed File Size", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if n_sources > 1:
        ax.legend(fontsize=8)


def plot_verification(ax, sources_data):
    """Show pass/fail verification status per phase."""
    sources = list(sources_data.keys())
    phases = PHASE_ORDER

    cell_text = []
    cell_colors = []
    for src in sources:
        data = sources_data[src]
        row = []
        colors = []
        for phase in phases:
            matching = [r for r in data if r["phase"] == phase]
            if matching:
                mm = matching[0].get("mismatches", 0)
                if mm == 0:
                    row.append("PASS")
                    colors.append("#d5f5e3")
                else:
                    row.append(f"FAIL ({int(mm)})")
                    colors.append("#fadbd8")
            else:
                row.append("N/A")
                colors.append("#f2f3f4")
        cell_text.append(row)
        cell_colors.append(colors)

    ax.axis("off")
    col_labels = [PHASE_LABELS.get(p, p) for p in phases]
    table = ax.table(
        cellText=cell_text,
        rowLabels=sources,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title("Lossless Verification", fontsize=11, fontweight="bold",
                 pad=20)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GPUCompress benchmark results")
    parser.add_argument("--gs-csv", help="Gray-Scott CSV path")
    parser.add_argument("--vpic-csv", help="VPIC deck CSV path")
    parser.add_argument("--output", default="benchmark_comparison.png",
                        help="Output image path")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display, only save")
    args = parser.parse_args()

    # Find CSVs
    gs_path = args.gs_csv or find_csv(DEFAULT_GS_PATHS)
    vpic_path = args.vpic_csv or find_csv(DEFAULT_VPIC_PATHS)

    if not gs_path and not vpic_path:
        print("ERROR: No benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_GS_PATHS + DEFAULT_VPIC_PATHS:
            print(f"  {p}")
        print("\nRun the benchmarks first, or specify paths with --gs-csv / --vpic-csv")
        sys.exit(1)

    # Load data
    sources_data = {}
    if gs_path and os.path.exists(gs_path):
        print(f"Loading Gray-Scott: {gs_path}")
        sources_data["Gray-Scott"] = load_grayscott_csv(gs_path)
    if vpic_path and os.path.exists(vpic_path):
        print(f"Loading VPIC:       {vpic_path}")
        sources_data["VPIC"] = load_vpic_csv(vpic_path)

    if not sources_data:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # Print summary
    for src, data in sources_data.items():
        print(f"\n  {src}: {len(data)} phase(s)")
        for r in data:
            phase = r.get("phase", "?")
            ratio = r.get("ratio", 0)
            wmbps = r.get("write_mbps", 0)
            rmbps = r.get("read_mbps", 0)
            mm    = int(r.get("mismatches", 0))
            print(f"    {phase:14s}  ratio={ratio:5.2f}x  "
                  f"write={wmbps:6.0f} MB/s  read={rmbps:6.0f} MB/s  "
                  f"{'PASS' if mm == 0 else 'FAIL'}")

    # ── Create figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("GPUCompress Benchmark: Gray-Scott & VPIC Simulation Data",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)

    # Panel 1: Compression Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    plot_grouped_bars(ax1, sources_data, "ratio",
                      "Compression Ratio", "Ratio (higher = better)",
                      fmt="%.2fx")

    # Panel 2: Write Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    plot_grouped_bars(ax2, sources_data, "write_mbps",
                      "Write Throughput", "MB/s",
                      fmt="%.0f")

    # Panel 3: Read Throughput
    ax3 = fig.add_subplot(gs[1, 0])
    plot_grouped_bars(ax3, sources_data, "read_mbps",
                      "Read Throughput", "MB/s",
                      fmt="%.0f")

    # Panel 4: File Sizes
    ax4 = fig.add_subplot(gs[1, 1])
    plot_file_sizes(ax4, sources_data)

    # Panel 5: NN Adaptation Stats
    ax5 = fig.add_subplot(gs[2, 0])
    plot_nn_stats(ax5, sources_data)

    # Panel 6: Verification Table
    ax6 = fig.add_subplot(gs[2, 1])
    plot_verification(ax6, sources_data)

    # Add metadata text
    meta_parts = []
    for src, data in sources_data.items():
        if data:
            r0 = data[0]
            orig = r0.get("orig_mb", 0)
            n_ch = int(r0.get("n_chunks", 0))
            if src == "Gray-Scott":
                L = int(r0.get("L", 0))
                steps = int(r0.get("steps", 0))
                meta_parts.append(
                    f"{src}: L={L}, steps={steps}, "
                    f"{orig:.0f} MB, {n_ch} chunks")
            else:
                meta_parts.append(
                    f"{src}: {orig:.0f} MB, {n_ch} chunks")
    if meta_parts:
        fig.text(0.5, 0.01, "  |  ".join(meta_parts),
                 ha="center", fontsize=8, style="italic", color="gray")

    # Save
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.output}")

    if not args.no_show:
        try:
            plt.switch_backend("TkAgg")
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
