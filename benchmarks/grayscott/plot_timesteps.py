#!/usr/bin/env python3
"""Plot sMAPE convergence over timesteps from the Gray-Scott multi-timestep benchmark."""

import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/grayscott/results/benchmark_grayscott_timesteps.csv"
df = pd.read_csv(csv_path)

# Support both old (mape_*) and new (smape_*) column names
for old, new in [("mape_ratio", "smape_ratio"), ("mape_comp", "smape_comp"), ("mape_decomp", "smape_decomp")]:
    if old in df.columns and new not in df.columns:
        df.rename(columns={old: new}, inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Gray-Scott Multi-Timestep Benchmark — NN-RL with SGD\n"
             f"({len(df)} timesteps, sMAPE bounded 0–200%)", fontsize=14, fontweight="bold")

# 1. sMAPE over timesteps
ax = axes[0, 0]
ax.plot(df["timestep"], df["smape_ratio"], label="Ratio sMAPE", linewidth=1.5, alpha=0.7)
ax.plot(df["timestep"], df["smape_comp"],  label="Comp Time sMAPE", linewidth=1.5, alpha=0.7)
ax.plot(df["timestep"], df["smape_decomp"], label="Decomp Time sMAPE", linewidth=1.5, alpha=0.7, linestyle="--")
# Rolling average for trend
window = max(1, len(df) // 20)
if len(df) > window:
    ax.plot(df["timestep"], df["smape_ratio"].rolling(window, min_periods=1).mean(),
            color="tab:blue", linewidth=2.5, label=f"Ratio (MA-{window})")
    ax.plot(df["timestep"], df["smape_comp"].rolling(window, min_periods=1).mean(),
            color="tab:orange", linewidth=2.5, label=f"Comp (MA-{window})")
ax.set_xlabel("Timestep (write cycle)")
ax.set_ylabel("sMAPE (%)")
ax.set_title("Prediction Error (sMAPE) Over Time")
ax.set_ylim(0, 200)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Compression ratio over timesteps
ax = axes[0, 1]
ax.plot(df["timestep"], df["ratio"], color="green", linewidth=1.5, alpha=0.7)
if len(df) > window:
    ax.plot(df["timestep"], df["ratio"].rolling(window, min_periods=1).mean(),
            color="darkgreen", linewidth=2.5, label=f"MA-{window}")
    ax.legend()
ax.set_xlabel("Timestep")
ax.set_ylabel("Compression Ratio (x)")
ax.set_title("Compression Ratio Over Time")
ax.grid(True, alpha=0.3)

# 3. Write/Read throughput
ax = axes[1, 0]
ax.plot(df["timestep"], df["write_mbps"], label="Write MB/s", linewidth=1.5, alpha=0.7)
ax.plot(df["timestep"], df["read_mbps"],  label="Read MB/s", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Timestep")
ax.set_ylabel("Throughput (MB/s)")
ax.set_title("I/O Throughput Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. SGD fires + cache stats
ax = axes[1, 1]
ax.bar(df["timestep"], df["sgd_fires"], alpha=0.6, label="SGD Fires", color="orange", width=1.0)
if "cache_hits" in df.columns:
    ax2 = ax.twinx()
    ax2.plot(df["timestep"], df["cache_hits"], color="blue", linewidth=1.5, label="Cache Hits")
    ax2.set_ylabel("Cache Hits", color="blue")
    ax2.legend(loc="upper right")
ax.set_xlabel("Timestep")
ax.set_ylabel("SGD Fires per Write")
ax.set_title("SGD Updates & Cache Hits Over Time")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = csv_path.replace(".csv", ".png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out_path}")
plt.close()
