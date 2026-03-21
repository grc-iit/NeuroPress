#!/usr/bin/env python3
"""Plot per-chunk predicted vs actual comp/decomp times from benchmark CSV."""

import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in r:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
            rows.append(r)
    return rows

# Auto-detect CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "benchmark_grayscott_vol_chunks.csv")
if len(sys.argv) > 1:
    csv_path = sys.argv[1]

rows = parse_csv(csv_path)
phases = []
seen = set()
for r in rows:
    ph = r["phase"]
    if ph not in seen:
        phases.append(ph)
        seen.add(ph)

fig, axes = plt.subplots(len(phases), 2, figsize=(16, 4 * len(phases)),
                          squeeze=False)
fig.suptitle("Per-Chunk Compression & Decompression Times", fontsize=14, fontweight="bold")

for pi, phase in enumerate(phases):
    ph_rows = sorted([r for r in rows if r["phase"] == phase],
                     key=lambda r: int(r["chunk"]))
    chunks = np.array([int(r["chunk"]) for r in ph_rows])
    act_ct = np.array([r["actual_comp_ms"] for r in ph_rows])
    pred_ct = np.array([r["predicted_comp_ms"] for r in ph_rows])
    act_dt = np.array([r["actual_decomp_ms"] for r in ph_rows])
    pred_dt = np.array([r["predicted_decomp_ms"] for r in ph_rows])

    # Comp time
    ax = axes[pi, 0]
    ax.plot(chunks, act_ct, "o-", color="#2c3e50", markersize=4, linewidth=1.5, label="Actual")
    ax.plot(chunks, pred_ct, "s--", color="#3498db", markersize=4, linewidth=1.5, label="Predicted")
    ax.fill_between(chunks, act_ct, pred_ct, alpha=0.15, color="#3498db")
    ax.set_ylabel("Comp Time (ms)")
    ax.set_title(f"{phase} — Compression Time", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    mape_ct = np.mean(np.abs(pred_ct - act_ct) / np.maximum(act_ct, 1e-6)) * 100
    ax.text(0.98, 0.95, f"MAPE: {mape_ct:.1f}%", transform=ax.transAxes,
            fontsize=9, ha="right", va="top", fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb"))

    # Decomp time
    ax = axes[pi, 1]
    ax.plot(chunks, act_dt, "o-", color="#2c3e50", markersize=4, linewidth=1.5, label="Actual")
    ax.plot(chunks, pred_dt, "s--", color="#e74c3c", markersize=4, linewidth=1.5, label="Predicted")
    ax.fill_between(chunks, act_dt, pred_dt, alpha=0.15, color="#e74c3c")
    ax.set_ylabel("Decomp Time (ms)")
    ax.set_title(f"{phase} — Decompression Time", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    mape_dt = np.mean(np.abs(pred_dt - act_dt) / np.maximum(act_dt, 1e-6)) * 100
    ax.text(0.98, 0.95, f"MAPE: {mape_dt:.1f}%", transform=ax.transAxes,
            fontsize=9, ha="right", va="top", fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#bbb"))

    axes[pi, 0].set_xlabel("Chunk Index")
    axes[pi, 1].set_xlabel("Chunk Index")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(script_dir, "chunk_times.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
