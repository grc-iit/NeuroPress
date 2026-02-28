#!/usr/bin/env python3
"""
plot_gpu_resident.py — Visualize benchmark_gpu_resident results.

Run after benchmark_gpu_resident to produce charts comparing:
  - No Compression  (GPU ramp → D→H → native HDF5 write)
  - NN Compression  (GPU ramp → VOL gpu_aware_chunked_write → compressed disk)

Usage:
  python3 tests/plot_gpu_resident.py [results.json]
  # Or hardcode the numbers at the bottom.
"""

import sys
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colours ──────────────────────────────────────────────────────────────────
C_NOCOMP = "#4C72B0"   # blue  – no compression
C_NN     = "#DD8452"   # orange – NN compression
C_EFF    = "#55A868"   # green  – effective BW

# ── data (from benchmark run on 16 GB ramp, 4 MB chunks) ─────────────────────
if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    DATASET_MB   = d["dataset_mb"]
    CHUNK_MB     = d["chunk_mb"]
    NC_WRITE     = d["nocomp_write_mbps"]
    NC_READ      = d["nocomp_read_mbps"]
    NC_FILE_MB   = d["nocomp_file_mb"]
    NC_RATIO     = d["nocomp_ratio"]
    NC_WRITE_MS  = d["nocomp_write_ms"]
    NC_READ_MS   = d["nocomp_read_ms"]
    NN_WRITE     = d["nn_write_mbps"]
    NN_READ      = d["nn_read_mbps"]
    NN_FILE_MB   = d["nn_file_mb"]
    NN_RATIO     = d["nn_ratio"]
    NN_WRITE_MS  = d["nn_write_ms"]
    NN_READ_MS   = d["nn_read_ms"]
else:
    # ── Results from last run ──────────────────────────────────────────────
    DATASET_MB   = 16384
    CHUNK_MB     = 4
    NC_WRITE_MS  = 46703.8
    NC_READ_MS   = 9001.1
    NC_WRITE     = 367.8
    NC_READ      = 1908.6
    NC_FILE_MB   = 16384
    NC_RATIO     = 1.00
    NN_WRITE_MS  = 128320.8
    NN_READ_MS   = 24779.3
    NN_WRITE     = 133.9
    NN_READ      = 693.3
    NN_FILE_MB   = 137
    NN_RATIO     = 118.77

# Effective (uncompressed-equivalent) write bandwidth for NN
NN_EFF_WRITE = NN_WRITE * NN_RATIO

OUT_DIR = "tests/benchmark_gpu_resident_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════
def bar_pair(ax, labels, vals, colors, ylabel, title, fmt="{:.0f}", annot_suffix=""):
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                fmt.format(v) + annot_suffix,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Write & Read Throughput side by side
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"GPU-Resident HDF5 Benchmark  |  {DATASET_MB // 1024} GB dataset, {CHUNK_MB} MB chunks, ramp pattern",
    fontsize=14, fontweight="bold", y=1.02
)

labels = ["No Compression\n(GPU→D→H→disk)", "NN Compression\n(GPU→comp→disk)"]
colors = [C_NOCOMP, C_NN]

bar_pair(axes[0], labels,
         [NC_WRITE, NN_WRITE], colors,
         "Throughput  (MB/s of original data)",
         "Write Throughput  (MB/s)",
         fmt="{:.0f}", annot_suffix=" MB/s")

bar_pair(axes[1], labels,
         [NC_READ, NN_READ], colors,
         "Throughput  (MB/s of original data)",
         "Read Throughput  (MB/s)",
         fmt="{:.0f}", annot_suffix=" MB/s")

plt.tight_layout()
path = f"{OUT_DIR}/fig1_throughput.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — File size & Compression ratio
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Storage Efficiency  |  {DATASET_MB // 1024} GB original dataset",
    fontsize=14, fontweight="bold", y=1.02
)

bar_pair(axes[0], labels,
         [NC_FILE_MB / 1024, NN_FILE_MB / 1024], colors,
         "File size on disk  (GB)",
         "File Size on Disk",
         fmt="{:.2f}", annot_suffix=" GB")

bar_pair(axes[1], labels,
         [NC_RATIO, NN_RATIO], colors,
         "Compression ratio  (×)",
         "Compression Ratio",
         fmt="{:.2f}", annot_suffix="×")

plt.tight_layout()
path = f"{OUT_DIR}/fig2_storage.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Effective write bandwidth (uncompressed-equivalent)
# Shows: raw write MB/s vs effective MB/s at the "original data" scale
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    "Effective Write Bandwidth (uncompressed-equivalent)",
    fontsize=14, fontweight="bold"
)

x      = np.arange(2)
width  = 0.35
bars_raw = ax.bar(x - width/2,
                  [NC_WRITE, NN_WRITE],
                  width, label="Raw write MB/s\n(original-data units)",
                  color=[C_NOCOMP, C_NN], edgecolor="white", linewidth=1.2)
bars_eff = ax.bar(x + width/2,
                  [NC_WRITE, NN_EFF_WRITE],
                  width, label="Effective MB/s\n(raw_write × ratio)",
                  color=[C_NOCOMP, C_EFF], edgecolor="white", linewidth=1.2,
                  alpha=0.85)

for bars in (bars_raw, bars_eff):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f"{h:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["No Compression", "NN Compression"], fontsize=12)
ax.set_ylabel("MB/s", fontsize=11)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
path = f"{OUT_DIR}/fig3_effective_bw.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Combined dashboard (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"GPU-Resident HDF5 Benchmark  ·  {DATASET_MB//1024} GB dataset  ·  {CHUNK_MB} MB chunks  ·  ramp pattern",
    fontsize=15, fontweight="bold", y=1.01
)

# (0,0) Write throughput
bar_pair(axes[0, 0], labels, [NC_WRITE, NN_WRITE], colors,
         "MB/s (original-data)", "Write Throughput",
         fmt="{:.0f}", annot_suffix=" MB/s")

# (0,1) Read throughput
bar_pair(axes[0, 1], labels, [NC_READ, NN_READ], colors,
         "MB/s (original-data)", "Read Throughput",
         fmt="{:.0f}", annot_suffix=" MB/s")

# (1,0) File size
bar_pair(axes[1, 0], labels,
         [NC_FILE_MB / 1024, NN_FILE_MB / 1024], colors,
         "GB on disk", "File Size on Disk",
         fmt="{:.2f}", annot_suffix=" GB")

# (1,1) Stacked: raw vs effective write BW for NN
ax = axes[1, 1]
x = np.arange(2)
w = 0.45
ax.bar(x, [NC_WRITE, NN_WRITE], w,
       color=colors, edgecolor="white", linewidth=1.2, label="Raw write MB/s")
# Effective BW extra bar (NN only)
ax.bar([1], [NN_EFF_WRITE - NN_WRITE], w,
       bottom=[NN_WRITE], color=C_EFF, edgecolor="white", linewidth=1.2,
       alpha=0.75, label=f"Effective uplift (×{NN_RATIO:.0f} ratio)")
for xp, v, extra in [(0, NC_WRITE, 0), (1, NN_EFF_WRITE, 0)]:
    ax.text(xp, v * 1.02, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["No Compression", "NN Compression"], fontsize=11)
ax.set_ylabel("MB/s", fontsize=11)
ax.set_title("Effective Write BW (uncomp-equiv)", fontsize=13, fontweight="bold", pad=8)
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
path = f"{OUT_DIR}/fig4_dashboard.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Text summary
# ═══════════════════════════════════════════════════════════════════════════════
space_saved = 100.0 * (1.0 - 1.0 / NN_RATIO)
write_speedup = NC_WRITE_MS / NN_WRITE_MS

print()
print("═" * 60)
print(f"  Dataset:          {DATASET_MB // 1024} GB  ({CHUNK_MB} MB chunks, ramp pattern)")
print(f"  Compression ratio:{NN_RATIO:>10.2f}×")
print(f"  Space saved:      {space_saved:>9.1f}%  ({NC_FILE_MB} MB → {NN_FILE_MB} MB)")
print(f"  Write (no-comp):  {NC_WRITE:>9.1f} MB/s  ({NC_WRITE_MS/1000:.1f} s)")
print(f"  Write (NN):       {NN_WRITE:>9.1f} MB/s  ({NN_WRITE_MS/1000:.1f} s)")
print(f"  Write speedup:    {write_speedup:>9.2f}×  ({'faster' if write_speedup>1 else 'slower'})")
print(f"  Effective BW (NN):{NN_EFF_WRITE:>9.0f} MB/s")
print(f"  Read (no-comp):   {NC_READ:>9.1f} MB/s")
print(f"  Read (NN):        {NN_READ:>9.1f} MB/s")
print("═" * 60)
print(f"\nAll figures saved to {OUT_DIR}/")
