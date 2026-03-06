#!/usr/bin/env python3
"""
Visualize VPIC benchmark deck results.

Reads: benchmark_vpic_deck_results/benchmark_vpic_deck.csv
Produces a 6-panel figure: ratio, write/read throughput, file size,
NN adaptation stats, and verification table.

Usage:
  python3 scripts/visualize_vpic.py [--csv PATH] [--output PATH]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── CSV parsing ───────────────────────────────────────────────────────

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


PHASE_ORDER = ["no-comp", "static", "nn", "nn-rl", "nn-rl+exp50"]

PHASE_COLORS = {
    "no-comp":     "#95a5a6",
    "static":      "#3498db",
    "nn":          "#2ecc71",
    "nn-rl":       "#e67e22",
    "nn-rl+exp50": "#e74c3c",
}

PHASE_LABELS = {
    "no-comp":     "No Comp",
    "static":      "Static\n(LZ4+Shuf)",
    "nn":          "NN\n(Inference)",
    "nn-rl":       "NN+SGD",
    "nn-rl+exp50": "NN+SGD\n+Explore",
}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize VPIC benchmark results")
    parser.add_argument("--csv",
        default="benchmark_vpic_deck_results/benchmark_vpic_deck.csv",
        help="CSV path")
    parser.add_argument("--output",
        default="benchmark_vpic_deck_results/benchmark_vpic.png",
        help="Output image path")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run the VPIC benchmark deck first.")
        sys.exit(1)

    all_rows = parse_csv(args.csv)
    if not all_rows:
        print("ERROR: CSV is empty")
        sys.exit(1)

    # Order rows by PHASE_ORDER
    rows_by_phase = {r["phase"]: r for r in all_rows}
    phases = [p for p in PHASE_ORDER if p in rows_by_phase]
    rows = [rows_by_phase[p] for p in phases]
    n = len(phases)
    x = np.arange(n)

    ratios       = [r.get("ratio", 0) for r in rows]
    write_mbps   = [r.get("write_mbps", 0) for r in rows]
    read_mbps    = [r.get("read_mbps", 0) for r in rows]
    file_mb      = [r.get("file_mb", 0) for r in rows]
    orig_mb      = [r.get("orig_mb", 0) for r in rows]
    mismatches   = [int(r.get("mismatches", 0)) for r in rows]
    sgd_fires    = [int(r.get("sgd_fires", 0)) for r in rows]
    explorations = [int(r.get("explorations", 0)) for r in rows]
    n_chunks     = [int(r.get("n_chunks", 0)) for r in rows]

    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    labels = [PHASE_LABELS.get(p, p) for p in phases]

    # ── Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("VPIC Benchmark: Harris Sheet Reconnection (Real Simulation)",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.3,
                           top=0.93, bottom=0.06, left=0.08, right=0.96)

    # Panel 1: Compression Ratio
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(x, ratios, color=colors, edgecolor="white")
    for bar, v in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}x", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Ratio (higher = better)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Write Throughput
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(x, write_mbps, color=colors, edgecolor="white")
    for bar, v in zip(bars, write_mbps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.0f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MB/s")
    ax.set_title("Write Throughput (compress + HDF5 write)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Read Throughput
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(x, read_mbps, color=colors, edgecolor="white")
    for bar, v in zip(bars, read_mbps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.0f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MB/s")
    ax.set_title("Read Throughput (HDF5 read + decompress)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: File Size (compressed vs original)
    ax = fig.add_subplot(gs[1, 1])
    width = 0.35
    ax.bar(x - width / 2, orig_mb, width, color="#d5d8dc", edgecolor="white",
           label="Original")
    comp_bars = ax.bar(x + width / 2, file_mb, width, color=colors,
                       edgecolor="white", label="Compressed")
    for bar, v in zip(comp_bars, file_mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Size (MB)")
    ax.set_title("Original vs Compressed File Size")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 5: NN Adaptation (SGD fires + explorations as % of chunks)
    ax = fig.add_subplot(gs[2, 0])
    nn_idx = [i for i, p in enumerate(phases) if p in ("nn", "nn-rl", "nn-rl+exp50")]
    if nn_idx:
        nn_x = np.arange(len(nn_idx))
        nn_labels = [labels[i] for i in nn_idx]
        nn_colors = [colors[i] for i in nn_idx]
        sgd_pct = [100.0 * sgd_fires[i] / max(n_chunks[i], 1) for i in nn_idx]
        expl_pct = [100.0 * explorations[i] / max(n_chunks[i], 1) for i in nn_idx]

        width = 0.35
        ax.bar(nn_x - width / 2, sgd_pct, width, color="#e67e22",
               edgecolor="white", label="SGD Fires %")
        ax.bar(nn_x + width / 2, expl_pct, width, color="#e74c3c",
               edgecolor="white", label="Explorations %")
        for xi, (s, e) in enumerate(zip(sgd_pct, expl_pct)):
            ax.text(xi - width / 2, s, f"{s:.0f}%", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
            ax.text(xi + width / 2, e, f"{e:.0f}%", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(nn_x)
        ax.set_xticklabels(nn_labels, fontsize=8)
        ax.set_ylabel("% of chunks")
        ax.set_ylim(0, max(max(sgd_pct, default=0), max(expl_pct, default=0)) * 1.3 + 5)
        ax.legend(fontsize=8)
    ax.set_title("NN Adaptation: SGD Fires & Explorations")
    ax.grid(axis="y", alpha=0.3)

    # Panel 6: Verification Table
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    cell_text = []
    cell_colors_tbl = []
    for i, p in enumerate(phases):
        mm = mismatches[i]
        status = "PASS" if mm == 0 else f"FAIL ({mm})"
        color = "#d5f5e3" if mm == 0 else "#fadbd8"
        cell_text.append([PHASE_LABELS.get(p, p).replace("\n", " "),
                          f"{orig_mb[i]:.0f}", f"{file_mb[i]:.1f}",
                          f"{ratios[i]:.2f}x", status])
        cell_colors_tbl.append(["white", "white", "white", "white", color])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Phase", "Orig (MB)", "Comp (MB)", "Ratio", "Verify"],
        cellColours=cell_colors_tbl,
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

    # Metadata
    r0 = rows[0]
    meta = (f"Dataset: {r0.get('orig_mb', 0):.0f} MB | "
            f"Chunks: {n_chunks[0]} x 64 MB | "
            f"Source: Real VPIC Harris Sheet Simulation")
    fig.text(0.5, 0.01, meta, ha="center", fontsize=9, style="italic",
             color="gray")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {args.output}")


if __name__ == "__main__":
    main()
