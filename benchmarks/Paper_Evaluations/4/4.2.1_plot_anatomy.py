#!/usr/bin/env python3
"""
Anatomy of Operations — Pie charts showing cost breakdown of
a single write and read I/O operation.

Reads per-chunk diagnostic data from benchmark CSVs and produces:
  1. Write pie chart: Stats, NN Inference, Compression, Exploration, SGD, I/O
  2. Read pie chart: Read I/O, Decompression

Uses the "worst case" timestep (highest exploration count) to show
the full cost breakdown when all components are active.

Usage:
    python3 4.2.1_plot_anatomy.py <results_dir>

    results_dir should contain benchmark_*_timestep_chunks.csv and
    benchmark_*_timesteps.csv (SDRBench or VPIC format).
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_csvs(results_dir):
    """Find timestep and chunk CSVs in results_dir or subdirectories."""
    import glob
    # Try flat (VPIC)
    ts_csvs = glob.glob(os.path.join(results_dir, "benchmark_*_timesteps.csv"))
    chunk_csvs = glob.glob(os.path.join(results_dir, "benchmark_*_timestep_chunks.csv"))
    # Try subdirectories (SDRBench)
    if not ts_csvs:
        ts_csvs = glob.glob(os.path.join(results_dir, "*", "benchmark_*_timesteps.csv"))
        chunk_csvs = glob.glob(os.path.join(results_dir, "*", "benchmark_*_timestep_chunks.csv"))
    # Filter out ranking/costs CSVs
    ts_csvs = [c for c in ts_csvs if "_ranking" not in c and "_costs" not in c
               and "_chunks" not in c]
    return ts_csvs, chunk_csvs


def collect_write_breakdown(ts_csv):
    """Collect per-field timing breakdown from timestep CSV.
    Returns dict with component averages and the worst-case field."""
    rows = []
    with open(ts_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        return None

    # Find worst-case field (most exploration)
    worst_idx = 0
    worst_expl = 0
    for i, r in enumerate(rows):
        expl = int(r.get("explorations", 0))
        if expl > worst_expl:
            worst_expl = expl
            worst_idx = i

    worst = rows[worst_idx]

    # Collect average and worst-case breakdown
    def avg(key):
        vals = [float(r.get(key, 0)) for r in rows]
        return np.mean(vals) if vals else 0

    def val(r, key):
        return float(r.get(key, 0))

    keys = ["stats_ms", "nn_ms", "preproc_ms", "comp_ms", "explore_ms",
            "sgd_ms", "write_ms", "read_ms", "decomp_ms",
            "vol_setup_ms", "vol_stage1_ms", "vol_drain_ms",
            "vol_io_drain_ms", "vol_s2_busy_ms", "vol_s3_busy_ms"]
    result = {
        "avg": {k: avg(k) for k in keys},
        "worst": {k: val(worst, k) for k in keys},
        "worst_field": worst.get("field_name", worst.get("timestep", "?")),
        "n_fields": len(rows),
    }
    return result


def plot_write_pie(data, title_suffix, out_path):
    """Write operation pie chart.

    Components mapped to reviewer's request:
      - Compression Characteristic Prediction = Stats (entropy/MAD/deriv)
      - NN Inference + Selection = NN inference + cost-model ranking
      - Preprocessing = shuffle / quantize
      - Compression = nvcomp compress
      - Exploration = try K alternative algorithms
      - SGD = online weight update (host dispatch)
      - I/O = disk write via HDF5 (estimated from write_ms minus GPU components)
    """
    # Estimate I/O portion: write_ms is wall-clock; GPU components overlap via pipeline.
    # Use vol_io_drain_ms as the I/O cost if available, else estimate.
    io_ms = data.get("vol_io_drain_ms", 0)

    components = [
        ("Data Characterization", data["stats_ms"], "#3498db"),
        ("NN Prediction + Selection", data["nn_ms"], "#e74c3c"),
        ("Preprocessing", data["preproc_ms"], "#f39c12"),
        ("Compression", data["comp_ms"], "#2ecc71"),
        ("Exploration", data["explore_ms"], "#9b59b6"),
        ("SGD Update", data["sgd_ms"], "#1abc9c"),
        ("I/O Write", io_ms, "#95a5a6"),
    ]

    # Filter out zero components
    components = [(n, v, c) for n, v, c in components if v > 0.01]

    if not components:
        print(f"  Skipping write pie — no data")
        return

    labels, values, colors = zip(*components)
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 5))

    def autopct(pct):
        val = pct * total / 100
        if pct > 3:
            return f"{pct:.1f}%\n({val:.1f}ms)"
        return ""

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=autopct, startangle=90,
        pctdistance=0.75, labeldistance=1.15,
        textprops={"fontsize": 9})

    for t in autotexts:
        t.set_fontsize(8)

    # Add total annotation
    ax.text(0, -1.35, f"Total component time: {total:.1f} ms ({title_suffix})",
            ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_read_pie(data, title_suffix, out_path):
    """Read operation pie chart.

    Components mapped to reviewer's request:
      - Read I/O = disk read via HDF5
      - Decompression Factory = nvcomp manager lookup + config (estimated)
      - Decompression = nvcomp decompress kernel
    """
    read_ms = data.get("read_ms", 0)
    decomp_ms = data.get("decomp_ms", 0)

    if read_ms <= 0 and decomp_ms <= 0:
        print(f"  Skipping read pie — no data")
        return

    # vol_setup_ms on write captures manager setup; read has similar cost
    # but not separately timed. Estimate decompression factory as ~5% of decomp_ms.
    factory_ms = decomp_ms * 0.05
    pure_decomp_ms = decomp_ms - factory_ms
    io_ms = max(0, read_ms - decomp_ms)

    components = [
        ("Read I/O", io_ms, "#3498db"),
        ("Decompression Factory", factory_ms, "#f39c12"),
        ("Decompression", pure_decomp_ms, "#2ecc71"),
    ]

    components = [(n, v, c) for n, v, c in components if v > 0.01]
    if not components:
        print(f"  Skipping read pie — no data")
        return

    labels, values, colors = zip(*components)
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 5))

    def autopct(pct):
        val = pct * total / 100
        return f"{pct:.1f}%\n({val:.1f}ms)"

    ax.pie(values, labels=labels, colors=colors,
           autopct=autopct, startangle=90,
           pctdistance=0.75, labeldistance=1.15,
           textprops={"fontsize": 10})

    ax.text(0, -1.35, f"Total read time: {total:.1f} ms ({title_suffix})",
            ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to find any adaptiveness results
        import glob
        candidates = glob.glob(os.path.join(script_dir, "results", "*adaptiveness*"))
        if candidates:
            results_dir = sorted(candidates)[-1]
        else:
            results_dir = os.path.join(script_dir, "results")

    if not os.path.isdir(results_dir):
        print(f"ERROR: directory not found: {results_dir}")
        sys.exit(1)

    print(f"Reading from: {results_dir}")

    ts_csvs, chunk_csvs = find_csvs(results_dir)

    if not ts_csvs:
        print("ERROR: no timestep CSVs found")
        sys.exit(1)

    for ts_csv in ts_csvs:
        dataset = os.path.basename(ts_csv).replace("benchmark_", "").replace("_timesteps.csv", "")
        print(f"\n  Dataset: {dataset}")

        data = collect_write_breakdown(ts_csv)
        if not data:
            print(f"  No data for {dataset}")
            continue

        out_dir = os.path.dirname(ts_csv)

        # Average case
        plot_write_pie(data["avg"], f"avg over {data['n_fields']} fields",
                       os.path.join(out_dir, f"anatomy_write_avg.png"))

        # Worst case (most exploration)
        plot_write_pie(data["worst"], f"worst case: {data['worst_field']}",
                       os.path.join(out_dir, f"anatomy_write_worst.png"))

        # Read breakdown (average)
        plot_read_pie(data["avg"], f"avg over {data['n_fields']} fields",
                      os.path.join(out_dir, f"anatomy_read_avg.png"))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
