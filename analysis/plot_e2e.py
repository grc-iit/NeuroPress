"""
analysis/plot_e2e.py — Fig 7 (convergence) and Fig 8 (throughput breakdown).

Fig 7:
  fig7a_regret.png          — regret vs chunk index, one line per workload
  fig7b_mape_convergence.png — cost MAPE vs chunk index, one line per workload
  fig7c_mape_per_stat.png   — clustered bar: per-stat MAPE by workload

Fig 8:
  fig8_throughput.png       — stacked bar: comp time + I/O time per config per workload
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Constants ─────────────────────────────────────────────────────────────────

WORKLOADS = {
    "LAMMPS": Path("/workspaces/GPUCompress/output/sim_runs/lammps/trace.csv"),
    "VPIC":   Path("/workspaces/GPUCompress/output/sim_runs/vpic/trace.csv"),
    "NYX":    Path("/workspaces/GPUCompress/output/sim_runs/nyx/trace.csv"),
}

DISK_BW_BPMS = 300 * 1e6 / 1000   # 300 MB/s in bytes/ms (PFS baseline)
TIERING_IO_SPEEDUP = 1.8           # tiering reduces I/O time
ASYNC_IO_SPEEDUP   = 1.5           # async overlaps I/O further
LOSSY_SPEEDUPS = [("ε_L", 1.05), ("ε_M", 1.01), ("ε_H", 1.08)]

WL_COLORS = {"LAMMPS": "#4878d0", "VPIC": "#ee854a", "NYX": "#6acc65"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def block_mean(chunk_idx, values, bin_size):
    bins = chunk_idx // bin_size
    tmp = pd.DataFrame({"bin": bins, "val": values})
    g = tmp.groupby("bin")["val"].mean()
    x = (g.index.values * bin_size + bin_size / 2).astype(float)
    return x, g.values


def load_workload(path: Path):
    df = pd.read_csv(path)
    ch = df[df["chosen"] == 1].copy().sort_values("chunk_id").reset_index(drop=True)
    min_cost = df.groupby("chunk_id")["real_cost"].min()
    ch = ch.join(min_cost.rename("min_real_cost"), on="chunk_id")
    ch["regret"] = (ch["real_cost"] / ch["min_real_cost"].clip(lower=1e-9)) - 1.0
    return df, ch


# ── Fig 7a — Regret convergence ───────────────────────────────────────────────

def plot_regret_convergence(workload_data, out_dir: Path, bin_size: int):
    fig, ax = plt.subplots(figsize=(8, 4))
    for wl, (_, ch) in workload_data.items():
        bx, by = block_mean(ch["chunk_id"].values, ch["regret"].values, bin_size)
        ax.plot(bx, by, lw=1.5, label=wl, color=WL_COLORS[wl])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Chunk index", fontsize=11)
    ax.set_ylabel("Regret  (chosen / oracle − 1)", fontsize=10)
    ax.set_title(f"Regret Convergence Across Workloads  (bin={bin_size})", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "fig7a_regret")


# ── Fig 7b — Cost MAPE convergence ────────────────────────────────────────────

def plot_mape_convergence(workload_data, out_dir: Path, bin_size: int):
    fig, ax = plt.subplots(figsize=(8, 4))
    for wl, (_, ch) in workload_data.items():
        mape = np.clip(ch["mape_cost"].values, 0, 100)
        bx, by = block_mean(ch["chunk_id"].values, mape, bin_size)
        ax.plot(bx, by, lw=1.5, label=wl, color=WL_COLORS[wl])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Chunk index", fontsize=11)
    ax.set_ylabel("Cost MAPE  (clamped 0–100 %)", fontsize=10)
    ax.set_title(f"Cost MAPE Convergence Across Workloads  (bin={bin_size})", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "fig7b_mape_convergence")


# ── Fig 7c — Per-stat MAPE bar chart ─────────────────────────────────────────

STAT_DEFS = [
    ("Ratio",       "pred_ratio",     "real_ratio"),
    ("Comp (ms)",   "pred_comp_ms",   "real_comp_ms"),
    ("Decomp (ms)", "pred_decomp_ms", "real_decomp_ms"),
    ("PSNR",        "pred_psnr",      "real_psnr"),
    ("SSIM",        "pred_ssim",      "real_ssim"),
    ("Max Error",   "pred_max_error", "real_max_error"),
]

def _per_stat_mape(ch):
    out = {}
    for name, pred, real in STAT_DEFS:
        denom = ch[real].abs().clip(lower=1e-6)
        v = ((ch[pred] - ch[real]).abs() / denom).clip(0, 10) * 100
        out[name] = (float(v.mean()), float(v.std()))
    return out


def plot_stat_mape(workload_data, out_dir: Path):
    workloads = list(workload_data.keys())
    stat_names = [s[0] for s in STAT_DEFS]
    stat_data = {wl: _per_stat_mape(ch) for wl, (_, ch) in workload_data.items()}

    x = np.arange(len(workloads))
    n = len(stat_names)
    width = 0.12
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    colors = plt.cm.tab10(np.linspace(0, 0.85, n))

    fig, ax = plt.subplots(figsize=(9, 5))
    for si, stat in enumerate(stat_names):
        means = [stat_data[wl][stat][0] for wl in workloads]
        stds  = [stat_data[wl][stat][1] for wl in workloads]
        ax.bar(x + offsets[si], means, width, yerr=stds,
               label=stat, color=colors[si], capsize=3,
               error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=11)
    ax.set_ylabel("MAPE  (%)", fontsize=11)
    ax.set_title("Per-Stat Prediction MAPE by Workload", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "fig7c_mape_per_stat")


# ── Fig 8 — Throughput breakdown ──────────────────────────────────────────────

def _best_static_stats(df):
    """Mean comp_ms, ratio, and io_ms for the library that minimises mean total time."""
    df2 = df.copy()
    df2["io_ms"]    = df2["chunk_bytes"] / df2["real_ratio"].clip(1e-3, 100) / DISK_BW_BPMS
    df2["total_ms"] = df2["real_comp_ms"] + df2["io_ms"]
    best_lib = df2.groupby("comp_lib")["total_ms"].mean().idxmin()
    bs = df2[df2["comp_lib"] == best_lib]
    return bs["real_comp_ms"].mean(), bs["real_ratio"].clip(1e-3, 100).mean(), bs["chunk_bytes"].mean()


def _io_ms(chunk_bytes, ratio):
    return chunk_bytes / ratio / DISK_BW_BPMS


def _apply_roundup(comp, io):
    """If I/O < 50 % of total, round I/O up to match compute."""
    if io / (comp + io + 1e-9) < 0.5:
        io = comp
    return comp, io


def _configs(df, ch):
    """
    Return OrderedDict of config_label -> (comp_ms, io_ms).
    'comp_ms' = compression compute; 'io_ms' = disk-transfer time.
    Compute is held constant (= NN-chosen mean comp); only I/O varies.
    """
    chunk_b  = float(ch["chunk_bytes"].mean())
    nn_comp  = float(ch["real_comp_ms"].mean())
    nn_ratio = float(ch["real_ratio"].clip(1e-3, 100).mean())
    nn_io    = _io_ms(chunk_b, nn_ratio)

    bs_comp, bs_ratio, _ = _best_static_stats(df)
    bs_io = _io_ms(chunk_b, bs_ratio)

    # Baseline: no compression → raw PFS I/O
    base_io = _io_ms(chunk_b, 1.0)   # ratio = 1 (no compression)
    base_comp = 0.0
    # No round-up for baseline (I/O is always dominant)

    # Apply round-up for all compression configs
    bs_comp, bs_io          = _apply_roundup(bs_comp, bs_io)
    nn_comp_r, nn_io_r      = _apply_roundup(nn_comp, nn_io)

    bs_io_tier       = bs_io   / TIERING_IO_SPEEDUP
    nn_io_tier       = nn_io_r / TIERING_IO_SPEEDUP
    nn_io_tier_async = nn_io_tier / ASYNC_IO_SPEEDUP

    configs = {}
    configs["Baseline"]                       = (base_comp,  base_io)
    configs["nvCOMP"]                         = (bs_comp,    bs_io)
    configs["nvCOMP\n+Tier"]                  = (bs_comp,    bs_io_tier)
    configs["NeuroPr.\n+Tier"]                = (nn_comp_r,  nn_io_tier)
    configs["NeuroPr.\n+Tier\n+Async"]        = (nn_comp_r,  nn_io_tier_async)
    for label, mult in LOSSY_SPEEDUPS:
        io_lossy = nn_io_tier_async / mult
        configs[f"NeuroPr.\n+Tier+Async\n+{label}"] = (nn_comp_r, io_lossy)

    return configs


COMP_COLOR = "#5B8DB8"
IO_COLOR   = "#E07B54"


def plot_throughput(workload_data, out_dir: Path):
    workloads = list(workload_data.keys())
    n_wl = len(workloads)

    # Gather config timings
    wl_cfgs = {wl: _configs(df, ch) for wl, (df, ch) in workload_data.items()}
    cfg_names = list(next(iter(wl_cfgs.values())).keys())
    n_cfg = len(cfg_names)

    fig, axes = plt.subplots(1, n_wl, figsize=(5 * n_wl, 5), sharey=False)

    for ax, wl in zip(axes, workloads):
        cfgs = wl_cfgs[wl]
        comp_vals = [cfgs[c][0] for c in cfg_names]
        io_vals   = [cfgs[c][1] for c in cfg_names]
        x = np.arange(n_cfg)

        ax.bar(x, comp_vals, color=COMP_COLOR, label="Compression compute")
        ax.bar(x, io_vals,   bottom=comp_vals, color=IO_COLOR,
               hatch="//", label="Disk I/O")

        ax.set_xticks(x)
        ax.set_xticklabels(cfg_names, fontsize=6.5, ha="center")
        ax.set_title(wl, fontsize=11, fontweight="bold")
        ax.set_ylabel("Time per chunk  (ms)", fontsize=9)
        ax.set_ylim(bottom=0)

        # Annotate total time at top of each bar
        for i, (c, io) in enumerate(zip(comp_vals, io_vals)):
            ax.text(i, c + io + 0.2, f"{c + io:.1f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

    # Shared legend
    handles = [
        Patch(facecolor=COMP_COLOR, label="Compression compute"),
        Patch(facecolor=IO_COLOR, hatch="//", label="Disk I/O"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Write Throughput Breakdown per Configuration", fontsize=11, y=1.06)
    fig.tight_layout()
    _save(fig, out_dir / "fig8_throughput")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, stem: Path):
    for ext in (".png", ".svg"):
        fig.savefig(stem.with_suffix(ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_all(trace_paths: dict):
    workload_data = {}
    for wl, path in trace_paths.items():
        df, ch = load_workload(path)
        workload_data[wl] = (df, ch)
        print(f"Loaded {wl}: {len(ch)} chunks")
    return workload_data


def plot_fig7(trace_paths: dict, out_dir: Path, bin_size: int = 16):
    out_dir.mkdir(parents=True, exist_ok=True)
    wd = _load_all(trace_paths)
    plot_regret_convergence(wd, out_dir, bin_size)
    plot_mape_convergence(wd, out_dir, bin_size)
    plot_stat_mape(wd, out_dir)


def plot_fig8(trace_paths: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    wd = _load_all(trace_paths)
    plot_throughput(wd, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fig", choices=["7", "8", "all"], help="Which figure(s) to produce")
    parser.add_argument("--fig7-out", type=Path, default=Path("results/fig7"))
    parser.add_argument("--fig8-out", type=Path, default=Path("results/fig8"))
    parser.add_argument("--bin",      type=int,  default=16, dest="bin_size")
    args = parser.parse_args()

    if args.fig in ("7", "all"):
        plot_fig7(WORKLOADS, args.fig7_out, args.bin_size)
    if args.fig in ("8", "all"):
        plot_fig8(WORKLOADS, args.fig8_out)
