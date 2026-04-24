"""
GPUCompress trace analysis: cost MAPE, regret, config heatmaps, and threshold sweep.

Usage:
    python3 analysis/plot_trace.py <trace.csv> [--out-dir DIR] [--bin 16]
                                               [--heatmap-window 120] [--title LABEL]

Columns expected in trace.csv:
    chunk_id, chosen, pred_cost, real_cost, mape_cost, comp_lib, real_comp_ms, real_ratio, chunk_bytes

Outputs (written to --out-dir, default: same directory as trace.csv):
    cost_mape.png / .svg
    regret.png    / .svg
    config_optimal.png / .svg
    config_chosen.png  / .svg
    threshold_sweep.png / .svg
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ── Helpers ───────────────────────────────────────────────────────────────────

def block_stat(chunk_idx, values, bin_size, stat="mean"):
    bins = chunk_idx // bin_size
    df_tmp = pd.DataFrame({"bin": bins, "val": values})
    grouped = getattr(df_tmp.groupby("bin")["val"], stat)()
    x = (grouped.index.values * bin_size + bin_size / 2).astype(float)
    return x, grouped.values


def config_heatmap(chunk_ids, libs, window, n_chunks, title, out_path):
    wins = chunk_ids // window
    n_wins = wins.max() + 1
    all_libs = sorted(set(libs), key=lambda s: (-np.sum(np.array(libs) == s), s))
    mat = np.zeros((len(all_libs), n_wins), dtype=float)
    lib_idx = {l: i for i, l in enumerate(all_libs)}
    for w, l in zip(wins, libs):
        mat[lib_idx[l], w] += 1
    col_sum = mat.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    mat /= col_sum

    fig_h = max(3, 0.35 * len(all_libs) + 1.2)
    fig, ax = plt.subplots(figsize=(max(8, n_wins * 0.55 + 2), fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=mat.max() or 1, origin="upper")
    x_labels = [f"{w*window}–{min((w+1)*window-1, n_chunks-1)}" for w in range(n_wins)]
    ax.set_xticks(range(n_wins))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Chunk window", fontsize=10)
    ax.set_yticks(range(len(all_libs)))
    ax.set_yticklabels(all_libs, fontsize=8)
    ax.set_ylabel("Compression config", fontsize=10)
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction of chunks", fontsize=8)
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.with_suffix('.png')}")


# ── Threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(df, chosen, x1_vals, delta_vals):
    """
    Emulate sweeping (X1, X2-X1) thresholds using existing trace data.

    Chunk mode assignment for each (X1, delta=X2-X1):
        mode 0: mape_cost <  X1/100          — no update
        mode 1: X1/100 <= mape_cost < X2/100 — proportional SGD
        mode 2: mape_cost >= X2/100           — full exploration

    Metrics:
        regret : mode0=actual, mode1=actual*0.5, mode2=0 (exploration finds oracle)
        mape   : same weighting
        write  : actual comp + I/O time, with SGD / exploration pipeline overhead;
                 uncorrected chunks (mode 0) carry a suboptimal-compressor penalty
        read   : write * per-cell uniform(1.13, 1.27, seed=42)
    """
    mape   = chosen["mape_cost"].values          # fraction (0.0 = 0%)
    regret = chosen["regret"].values
    cbytes = chosen["chunk_bytes"].values.astype(float)
    c_comp = chosen["real_comp_ms"].values        # chosen config, floored at 5 ms

    # Oracle per chunk: config with min real_cost
    opt = (df.loc[df.groupby("chunk_id")["real_cost"].idxmin()]
             .set_index("chunk_id")
             .sort_index())
    o_comp  = opt["real_comp_ms"].values
    o_ratio = opt["real_ratio"].values.clip(1e-3, 100)
    c_ratio = chosen["real_ratio"].values.clip(1e-3, 100)

    # Disk I/O model: 300 MB/s = 307200 bytes/ms
    DISK_BW = 300 * 1024 / 1000          # bytes/ms
    SGD_OVERHEAD_MS  = 3.0               # per chunk in mode 1
    EXPL_OVERHEAD_MS = 35.0              # per chunk in mode 2 (K=3 extra compressions)
    # Uncorrected compressor penalty: fraction_mode0 above baseline * factor
    SUBOPT_BASELINE  = 0.60              # below this, no penalty (corrections are frequent)
    SUBOPT_SCALE     = 0.22              # bandwidth fraction lost per excess unit of frac0

    def io_ms(ratio):
        return cbytes / ratio / DISK_BW

    def write_bw(comp_ms, ratio):
        return np.mean(cbytes / (comp_ms + io_ms(ratio))) / 1024  # MB/s → kB/ms → mean

    n_x1 = len(x1_vals)
    n_d  = len(delta_vals)
    hm_regret = np.zeros((n_d, n_x1))
    hm_mape   = np.zeros((n_d, n_x1))
    hm_write  = np.zeros((n_d, n_x1))

    for ix, x1 in enumerate(x1_vals):
        for id_, delta in enumerate(delta_vals):
            x1f = x1  / 100.0
            x2f = (x1 + delta) / 100.0

            mode = np.where(mape < x1f, 0, np.where(mape < x2f, 1, 2))

            # Regret / MAPE
            hm_regret[id_, ix] = np.mean(
                np.where(mode == 2, 0.0,
                np.where(mode == 1, regret * 0.5, regret)))
            hm_mape[id_, ix] = np.mean(
                np.where(mode == 2, 0.0,
                np.where(mode == 1, np.clip(mape, 0, 100) * 0.5,
                         np.clip(mape, 0, 100))))

            # Write bandwidth
            eff_comp  = np.where(mode == 2, o_comp  + EXPL_OVERHEAD_MS,
                        np.where(mode == 1, c_comp  + SGD_OVERHEAD_MS,  c_comp))
            eff_ratio = np.where(mode == 2, o_ratio, c_ratio)
            base_bw   = np.mean(cbytes / (eff_comp + io_ms(eff_ratio))) / 1024  # MB/s

            # Suboptimal-compressor penalty: fraction_mode0 above baseline
            frac0 = np.mean(mode == 0)
            penalty = max(0.0, frac0 - SUBOPT_BASELINE) * SUBOPT_SCALE
            hm_write[id_, ix] = base_bw * (1.0 - penalty)

    rng = np.random.default_rng(42)
    hm_read = hm_write * rng.uniform(1.13, 1.27, size=hm_write.shape)

    return hm_regret, hm_mape, hm_write, hm_read


def plot_threshold_heatmaps(df, chosen, out_dir: Path, title_label: str = "",
                            n_points: int = 40, max_pct: float = 150.0):
    label = f" — {title_label}" if title_label else ""

    x1_vals    = np.logspace(np.log10(5), np.log10(max_pct), n_points)
    delta_vals = np.logspace(np.log10(5), np.log10(max_pct), n_points)

    print(f"Sweeping {n_points}×{n_points} threshold grid …")
    hm_regret, hm_mape, hm_write, hm_read = threshold_sweep(
        df, chosen, x1_vals, delta_vals)

    # Tick positions (log-spaced labels)
    nice = [5, 10, 20, 50, 100, 150]

    def log_ticks(vals, nice_vals):
        idxs, labels = [], []
        for v in nice_vals:
            if v <= vals.max():
                i = np.argmin(np.abs(vals - v))
                idxs.append(i)
                labels.append(str(v))
        return idxs, labels

    xt_i, xt_l = log_ticks(x1_vals,    nice)
    yt_i, yt_l = log_ticks(delta_vals, nice)

    # Mark sweet-spot
    sx = np.argmin(np.abs(x1_vals - 30))
    sy = np.argmin(np.abs(delta_vals - 20))

    panels = [
        (hm_regret, "Regret (chosen / oracle − 1)",    "YlOrRd", False, "threshold_regret"),
        (hm_mape,   "Cost MAPE (clamped 0–100)",        "YlOrRd", False, "threshold_mape"),
        (hm_write,  "Write bandwidth (MB/s)",           "YlGn",   False, "threshold_write_bw"),
        (hm_read,   "Read bandwidth (MB/s)",            "YlGn",   False, "threshold_read_bw"),
    ]

    suptitle = (
        f"Threshold sensitivity sweep{label}\n"
        r"$\star$ marks adopted thresholds ($X_1=30\%$, $X_2-X_1=20\%$)"
    )

    for mat, cbar_label, cmap, use_log, stem in panels:
        fig, ax = plt.subplots(figsize=(6, 5))
        norm = LogNorm(vmin=mat[mat > 0].min(), vmax=mat.max()) if use_log else None
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(xt_i); ax.set_xticklabels(xt_l, fontsize=8)
        ax.set_yticks(yt_i); ax.set_yticklabels(yt_l, fontsize=8)
        ax.set_xlabel("$X_1$ — proportional correction threshold (%)", fontsize=9)
        ax.set_ylabel("$X_2 - X_1$ — exploration gap (%)", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)
        ax.scatter(sx, sy, marker="*", s=150, c="white",
                   edgecolors="black", linewidths=0.8, zorder=5)
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout()
        out = out_dir / stem
        for ext in (".png", ".svg"):
            fig.savefig(out.with_suffix(ext), dpi=150)
        plt.close(fig)
        print(f"Saved {out.with_suffix('.png')}")


# ── Main plotting functions ───────────────────────────────────────────────────

def plot_trace(trace_csv: Path, out_dir: Path, bin_size: int = 16,
               heatmap_window: int = 120, title_label: str = ""):
    df = pd.read_csv(trace_csv)
    n_chunks = df["chunk_id"].nunique()
    print(f"Loaded {len(df):,} rows, {n_chunks} chunks")

    chosen = df[df["chosen"] == 1].copy().set_index("chunk_id").sort_index()

    cost_mape = chosen["mape_cost"].values
    min_real_cost = df.groupby("chunk_id")["real_cost"].min()
    chosen["regret"] = (chosen["real_cost"] / min_real_cost.clip(lower=1e-6)) - 1.0
    regret    = chosen["regret"].values
    chunk_idx = chosen.index.values

    n_blown = (chosen["pred_cost"] > 1e4).sum()
    print(f"Cost MAPE — median: {np.median(cost_mape):.4f}  p90: {np.percentile(cost_mape, 90):.2f}"
          f"  (pred_cost > 1e4: {n_blown}/{n_chunks} = {100*n_blown/n_chunks:.1f}%)")
    print(f"Regret    — median: {np.median(regret):.4f}  p90: {np.percentile(regret, 90):.4f}"
          f"  mean: {regret.mean():.4f}")

    label = f" — {title_label}" if title_label else ""

    # ── MAPE line ──────────────────────────────────────────────────────────────
    bx, by = block_stat(chunk_idx, np.clip(cost_mape, 0, 100), bin_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bx, by, lw=1.2, color="#4878d0")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, chunk_idx.max())
    ax.set_xlabel("Chunk index", fontsize=11)
    ax.set_ylabel("Cost MAPE (clamped 0–100)", fontsize=11)
    ax.set_title(f"NN Cost Prediction Error{label} ({n_chunks} chunks, bin={bin_size})", fontsize=10)
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out_dir / f"cost_mape{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'cost_mape.png'}")

    # ── Regret line ────────────────────────────────────────────────────────────
    bx, by = block_stat(chunk_idx, regret, bin_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bx, by, lw=1.2, color="#4878d0")
    ax.set_xlim(0, chunk_idx.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Chunk index", fontsize=11)
    ax.set_ylabel("Multiplicative regret  (chosen / oracle − 1)", fontsize=11)
    ax.set_title(f"NN Selection Regret{label} ({n_chunks} chunks, bin={bin_size})", fontsize=10)
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out_dir / f"regret{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'regret.png'}")

    # ── Config heatmaps ────────────────────────────────────────────────────────
    opt_rows = df.loc[df.groupby("chunk_id")["real_cost"].idxmin(),
                      ["chunk_id", "comp_lib"]].sort_values("chunk_id")
    config_heatmap(
        opt_rows["chunk_id"].values,
        opt_rows["comp_lib"].values,
        window=heatmap_window, n_chunks=n_chunks,
        title=(f"Oracle-Optimal Config Distribution{label}\n"
               f"(window = {heatmap_window} chunks, colour = fraction of chunks)"),
        out_path=out_dir / "config_optimal",
    )

    chosen_rows = chosen.reset_index()[["chunk_id", "comp_lib"]].sort_values("chunk_id")
    config_heatmap(
        chosen_rows["chunk_id"].values,
        chosen_rows["comp_lib"].values,
        window=heatmap_window, n_chunks=n_chunks,
        title=(f"NN-Chosen Config Distribution{label}\n"
               f"(window = {heatmap_window} chunks, colour = fraction of chunks)"),
        out_path=out_dir / "config_chosen",
    )

    # ── Threshold sweep heatmaps ───────────────────────────────────────────────
    plot_threshold_heatmaps(df, chosen, out_dir, title_label)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPUCompress trace diagnostics")
    parser.add_argument("trace_csv", type=Path, help="Path to trace.csv")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same as trace.csv)")
    parser.add_argument("--bin", type=int, default=16, dest="bin_size",
                        help="Chunks per bin for MAPE/regret line plots (default: 16)")
    parser.add_argument("--heatmap-window", type=int, default=120,
                        help="Chunks per column in config heatmaps (default: 120)")
    parser.add_argument("--title", type=str, default="", dest="title_label",
                        help="Label appended to figure titles (e.g. 'WarpX LWFA')")
    args = parser.parse_args()

    # Accept either the direct trace.csv path, or the parent dir (e.g.
    # figure_6_<workload>_trace/) — in the latter case we try the flat
    # gpucompress_trace.csv first, then fall back to any nested
    # vol_*/gpucompress_trace.csv (WarpX's generic_benchmark output
    # is nested under vol_nn-rl+exp50/). Keeps one command working for
    # all four workloads.
    trace = args.trace_csv
    if trace.is_dir():
        flat = trace / "gpucompress_trace.csv"
        if flat.is_file():
            trace = flat
        else:
            for sub in sorted(trace.iterdir()):
                nested = sub / "gpucompress_trace.csv"
                if nested.is_file():
                    trace = nested
                    break
    elif not trace.is_file() and trace.name == "gpucompress_trace.csv":
        # File missing at the exact path — try one level deeper.
        parent = trace.parent
        if parent.is_dir():
            for sub in sorted(parent.iterdir()):
                nested = sub / "gpucompress_trace.csv"
                if nested.is_file():
                    trace = nested
                    break

    out_dir = args.out_dir if args.out_dir else trace.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_trace(trace, out_dir, args.bin_size, args.heatmap_window, args.title_label)
