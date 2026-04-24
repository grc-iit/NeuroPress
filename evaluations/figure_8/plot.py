#!/usr/bin/env python3
"""
evaluations/figure_8/plot.py

Reads each workload's per-chunk + ranking CSVs from an SC26 sweep directory
and produces THREE combined figures per policy, with all workloads on the
same plot:

  Figure 1: cost model MAPE convergence — line graph, all workloads
  Figure 2: top-1 regret convergence    — line graph, all workloads
  Figure 3: per-metric MAPE breakdown   — grouped bar chart with comp time,
                                          decomp time, comp ratio, and PSNR
                                          MAPE for each workload

Outputs:
  $SC26_DIR/figures/sc26_<policy>_cost_mape.png
  $SC26_DIR/figures/sc26_<policy>_regret.png
  $SC26_DIR/figures/sc26_<policy>_metric_breakdown.png
  $SC26_DIR/csv/sc26_<policy>_cost_mape.csv
  $SC26_DIR/csv/sc26_<policy>_regret.csv
  $SC26_DIR/csv/sc26_<policy>_metric_breakdown.csv

Unit handling:
  - cost_model_error_pct is stored as a FRACTION in [0,1] by
    src/api/gpucompress_compress.cpp:545 — multiplied by 100 here.
  - mape_comp / mape_decomp / mape_ratio / mape_psnr are written as
    PERCENT (0-200+) by the live-sim emitters (e.g. vpic_benchmark_deck.cxx,
    fix_gpucompress_kokkos.cpp, FlushFormatGPUCompress.cpp). No scaling.
  - top1_regret is written as a RATIO (predicted_best_cost / actual_best_cost,
    where 1.0 = oracle pick). Converted to a percent here as (ratio - 1) * 100,
    matching the in-runner plotter at run_one_workload.sh:646.
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOADS = [
    ("vpic",   "VPIC",     "#e41a1c"),
    ("nyx",    "NYX",      "#377eb8"),
    ("warpx",  "WarpX",    "#4daf4a"),
    ("lammps", "LAMMPS",   "#ff7f00"),
    ("ai",     "ViT-B/16", "#984ea3"),
]

METRIC_DEFS = [
    ("comp",   "mape_comp",   "Comp Time MAPE",  "#1f77b4"),
    ("decomp", "mape_decomp", "Decomp Time MAPE", "#ff7f0e"),
    ("ratio",  "mape_ratio",  "Comp Ratio MAPE",  "#2ca02c"),
    ("psnr",   "mape_psnr",   "PSNR MAPE",        "#d62728"),
]


def chunks_csv(rdir, key):
    if key == "ai":
        p = os.path.join(rdir, "inline_benchmark_chunks.csv")
        return p if os.path.isfile(p) else None
    # VPIC's benchmark deck names CSVs benchmark_vpic_deck_*.csv;
    # every other workload uses benchmark_<key>_*.csv.
    candidate = "benchmark_vpic_deck_timestep_chunks.csv" if key == "vpic" \
        else f"benchmark_{key}_timestep_chunks.csv"
    p = os.path.join(rdir, candidate)
    return p if os.path.isfile(p) else None


def ranking_csv(rdir, key):
    candidate = "benchmark_vpic_deck_ranking.csv" if key == "vpic" \
        else f"benchmark_{key}_ranking.csv"
    p = os.path.join(rdir, candidate)
    return p if os.path.isfile(p) else None


def trace_csv(sc26_dir, key):
    """Locate gpucompress_trace.csv from a figure_6 trace-mode run.
    WarpX nests the file one level deeper under vol_<phase>/; other
    workloads keep it flat under figure_6_<wl>_trace/."""
    base = os.path.join(sc26_dir, f"figure_6_{key}_trace")
    flat = os.path.join(base, "gpucompress_trace.csv")
    if os.path.isfile(flat):
        return flat
    # Fall back to any vol_*/gpucompress_trace.csv inside the base dir.
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            nested = os.path.join(base, sub, "gpucompress_trace.csv")
            if os.path.isfile(nested):
                return nested
    return None


def load_trace_regret(path):
    """Per-chunk regret computed from a figure_6 trace.csv, matching the
    primary author's analysis/plot_e2e.py:load_workload() derivation:

        oracle_cost   = min(real_cost) across all 32 configs per chunk_id
        chosen_cost   = real_cost where chosen == 1
        regret (%)    = (chosen_cost / oracle_cost - 1) * 100

    Returns a list of per-chunk regret percentages, sorted by chunk_id.
    Uses the same granularity the paper's Fig 7a regret line plot shows —
    one data point per chunk, yielding a smooth convergence curve instead
    of the six-point step function that ranking.csv:top1_regret produces."""
    if not path or not os.path.isfile(path):
        return []
    import pandas as pd  # keep the heavy import local
    df = pd.read_csv(path)
    if "chosen" not in df.columns or "real_cost" not in df.columns:
        return []
    chosen = df[df["chosen"] == 1].copy().sort_values("chunk_id")
    oracle = df.groupby("chunk_id")["real_cost"].min()
    chosen = chosen.join(oracle.rename("min_real_cost"), on="chunk_id")
    regret = ((chosen["real_cost"] / chosen["min_real_cost"].clip(lower=1e-9))
              - 1.0) * 100.0
    return regret.dropna().tolist()


def load_trace_cost_mape(path):
    """Per-chunk cost-MAPE read directly from trace.csv's mape_cost column,
    matching the primary author's analysis/plot_e2e.py:plot_mape_convergence
    logic (chosen==1 rows, clamp to 0–100 %). The VOL emits mape_cost as an
    already-percent value."""
    if not path or not os.path.isfile(path):
        return []
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path)
    if "chosen" not in df.columns or "mape_cost" not in df.columns:
        return []
    chosen = df[df["chosen"] == 1].copy().sort_values("chunk_id")
    mape = np.clip(chosen["mape_cost"].values, 0, 100)
    return mape.tolist()


def load_column(path, col, scale=1.0, drop_nan=True):
    """Return a list of float(col)*scale for each row where the cell is
    parseable. NaNs are dropped by default."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(col, "")
            if v in ("", None):
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if drop_nan and math.isnan(fv):
                continue
            out.append(fv * scale)
    return out


def smooth(ys, window=None):
    if not ys:
        return [], []
    if window is None:
        window = min(10, max(1, len(ys) // 6))
    if len(ys) < window or window <= 1:
        return list(range(len(ys))), list(ys)
    sm = np.convolve(ys, np.ones(window) / window, mode="valid").tolist()
    xs = list(range(window - 1, len(ys)))
    return xs, sm


def final_mean(vals, frac=0.2):
    if not vals:
        return None
    n = max(1, int(round(len(vals) * frac)))
    return float(np.mean(vals[-n:]))


# ============================================================
# AI-specific loader
# ============================================================
#
# The AI workload emits a different CSV schema because it uses
# gpucompress_inline_benchmark (not the HDF5 VOL). Every
# (epoch, tensor, chunk_idx) group has up to 15 rows: 9 fixed
# algorithms (mode='-' policy='-') + 6 NN configs (3 learning
# modes × 2 policies). No pre-computed mape_* columns — we have
# to compute them here from the raw comp_ms / predicted_comp_time
# / actual_ratio / predicted_ratio fields.
#
# Regret is computed the same way the live sims compute it: find
# the cost of the NN's chosen row, find the oracle cost (minimum
# across all 15 rows in the group), ratio them, subtract 1,
# multiply by 100. Cost is the policy-weighted scalar from the
# live-sim cost model (see src/api/gpucompress_compress.cpp:541-544).
#
# Matches the in-runner plotter in run_one_workload.sh
# (load_ai_inline_chunks_regret / _comp_mape) so numbers align.


def _policy_weights(policy):
    if policy == "speed":    return (1.0, 0.0, 0.0)
    if policy == "balanced": return (1.0, 1.0, 1.0)
    if policy == "ratio":    return (0.0, 0.0, 1.0)
    return (1.0, 1.0, 1.0)


def _compute_cost(comp_ms, decomp_ms, ratio, chunk_bytes, w0, w1, w2,
                  bw_bytes_per_ms=5e6):
    """Matches the live-sim cost model with the same policy clamps
    (ct/dt floor 5 ms, ratio cap 100x). bw_bytes_per_ms default
    matches the 5 GB/s assumption in gpucompress_compress.cpp."""
    ct = max(5.0, comp_ms)
    dt = max(5.0, decomp_ms)
    r  = min(100.0, ratio) if ratio > 0 else 0.01
    io_cost = chunk_bytes / (r * bw_bytes_per_ms) if r > 0 else 1e30
    return w0 * ct + w1 * dt + w2 * io_cost


def load_ai_inline_chunks(path, policy, chunk_mb,
                          nn_mode="nn-rl+exp50",
                          bw_bytes_per_ms=5e6):
    """Parse inline_benchmark_chunks.csv into per-group metric lists
    aligned to match the live-sim chunk CSV schema.

    Returns a dict with keys:
      cost_mape    list of (|predicted_cost − actual_cost|/actual_cost)*100  (%)
      regret       list of (nn_pick_cost/oracle_cost − 1)*100                (%)
      mape_comp, mape_decomp, mape_ratio, mape_psnr
                   list of per-group MAPEs for the NN-picked row             (%)
    """
    out = {k: [] for k in ("cost_mape", "regret",
                           "mape_comp", "mape_decomp",
                           "mape_ratio", "mape_psnr")}
    if not path or not os.path.isfile(path):
        return out

    chunk_bytes = int(chunk_mb) * 1024 * 1024
    w0, w1, w2 = _policy_weights(policy)

    # Group rows by (epoch, tensor, chunk_idx)
    groups = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["epoch"]), row["tensor"], int(row["chunk_idx"]))
            except (KeyError, ValueError):
                continue
            groups.setdefault(key, []).append(row)

    # Sort groups so the series is deterministic across runs
    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2])):
        rows = groups[key]
        # Compute the actual cost for every row in the group
        costs = []
        nn_row = None
        for r in rows:
            try:
                comp_ms   = float(r.get("comp_ms", 0) or 0)
                decomp_ms = float(r.get("decomp_ms", 0) or 0)
                ratio     = float(r.get("actual_ratio", 0) or 0)
            except ValueError:
                continue
            c = _compute_cost(comp_ms, decomp_ms, ratio, chunk_bytes,
                              w0, w1, w2, bw_bytes_per_ms)
            costs.append((r, c))
            if r.get("mode", "") == nn_mode and r.get("policy", "") == policy:
                nn_row = (r, c)

        if nn_row is None or not costs:
            continue

        nn_r, nn_cost = nn_row
        oracle_cost = min(c for _, c in costs)
        if oracle_cost <= 0:
            continue
        regret_pct = (nn_cost / oracle_cost - 1.0) * 100.0
        out["regret"].append(regret_pct)

        # Per-metric MAPE for the NN-picked row
        try:
            pred_ct  = float(nn_r.get("predicted_comp_time", 0) or 0)
            pred_dt  = float(nn_r.get("predicted_decomp_time", 0) or 0)
            pred_r   = float(nn_r.get("predicted_ratio", 0) or 0)
            pred_psnr= float(nn_r.get("predicted_psnr", 0) or 0)
            comp_ms  = float(nn_r.get("comp_ms", 0) or 0)
            decomp_ms= float(nn_r.get("decomp_ms", 0) or 0)
            act_r    = float(nn_r.get("actual_ratio", 0) or 0)
            act_psnr = float(nn_r.get("actual_psnr", 0) or 0)
        except ValueError:
            continue

        # Comp-time MAPE (5 ms floor matches the NN clamp)
        denom_ct = max(comp_ms, 5.0)
        mape_comp = abs(pred_ct - denom_ct) / denom_ct * 100.0
        out["mape_comp"].append(min(200.0, mape_comp))

        # Decomp-time MAPE
        denom_dt = max(decomp_ms, 5.0)
        mape_decomp = abs(pred_dt - denom_dt) / denom_dt * 100.0
        out["mape_decomp"].append(min(200.0, mape_decomp))

        # Comp ratio MAPE
        if act_r > 0:
            mape_ratio = abs(pred_r - act_r) / act_r * 100.0
            out["mape_ratio"].append(min(200.0, mape_ratio))

        # PSNR MAPE (only valid if lossy and PSNR actually measured)
        if pred_psnr > 0 and act_psnr > 0 and math.isfinite(act_psnr):
            mape_psnr = abs(pred_psnr - act_psnr) / act_psnr * 100.0
            out["mape_psnr"].append(min(200.0, mape_psnr))

        # Cost MAPE from the NN row's predicted cost (using its own
        # predicted_comp_time/decomp_time/ratio) vs measured actual cost.
        nn_predicted_cost = _compute_cost(
            pred_ct if pred_ct > 0 else comp_ms,
            pred_dt if pred_dt > 0 else decomp_ms,
            pred_r if pred_r > 0 else act_r,
            chunk_bytes, w0, w1, w2, bw_bytes_per_ms)
        if nn_cost > 0:
            cost_mape = abs(nn_predicted_cost - nn_cost) / nn_cost * 100.0
            out["cost_mape"].append(min(200.0, cost_mape))

    return out


def plot_line(series, title, ylabel, png_path, csv_path, cap=None):
    """series = list of (label, color, ys)"""
    if not series:
        print(f"  no data → skipping {os.path.basename(png_path)}")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    rows = []
    for label, color, ys in series:
        ys_capped = [min(y, cap) for y in ys] if cap is not None else list(ys)
        xs = list(range(len(ys_capped)))
        ax.plot(xs, ys_capped, color=color, alpha=0.18, linewidth=0.6)
        smx, smy = smooth(ys_capped)
        if smx and len(smx) > 1:
            ax.plot(smx, smy, color=color, linewidth=2.0, label=label)
        else:
            ax.plot(xs, ys_capped, color=color, linewidth=2.0, label=label)
        for i, y in enumerate(ys):
            rows.append((label, i, y))
    ax.set_xlabel("Chunk index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    if cap is not None:
        ax.set_ylim(bottom=0, top=cap * 1.05)
    else:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "chunk_index", "value_pct"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}"])
    print(f"  wrote {png_path}")
    print(f"  wrote {csv_path}")


def plot_bar(workload_metrics, title, png_path, csv_path, cap=None):
    """workload_metrics = list of (label, color, {metric_key: value or None})"""
    if not workload_metrics:
        print(f"  no data → skipping {os.path.basename(png_path)}")
        return
    n_w = len(workload_metrics)
    n_m = len(METRIC_DEFS)
    bar_w = 0.8 / n_m
    x = np.arange(n_w)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * n_w + 2.0), 4.2))
    rows = []
    for mi, (mkey, _src, mlabel, mcolor) in enumerate(METRIC_DEFS):
        vals = []
        for label, _color, metrics in workload_metrics:
            v = metrics.get(mkey)
            vals.append(v if (v is not None and not math.isnan(v)) else 0.0)
            rows.append((label, mlabel,
                         "" if v is None or (isinstance(v, float) and math.isnan(v))
                         else f"{v:.6f}"))
        offset = mi * bar_w - 0.4 + bar_w / 2
        bars = ax.bar(x + offset, vals, width=bar_w, label=mlabel, color=mcolor)
        # annotate non-zero bars (or "n/a" for missing)
        for bi, (b, raw) in enumerate(zip(bars,
                [workload_metrics[wi][2].get(mkey) for wi in range(n_w)])):
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                ax.text(b.get_x() + b.get_width() / 2, 0.5,
                        "n/a", ha="center", va="bottom",
                        fontsize=7, color="gray", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2,
                        min(b.get_height(), cap if cap else b.get_height()) + 0.5,
                        f"{raw:.0f}", ha="center", va="bottom",
                        fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([wm[0] for wm in workload_metrics])
    ax.set_ylabel("MAPE (%) — mean over final 20% of chunks")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=True, fontsize=9, ncol=n_m, loc="upper right")
    if cap is not None:
        ax.set_ylim(bottom=0, top=cap * 1.05)
    else:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "metric", "value_pct"])
        for r in rows:
            w.writerow(r)
    print(f"  wrote {png_path}")
    print(f"  wrote {csv_path}")


def build_for_policy(sc26_dir, policy):
    fig_dir = os.path.join(sc26_dir, "figures")
    csv_dir = os.path.join(sc26_dir, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    cost_series = []
    regret_series = []
    bar_data = []
    any_workload = False

    chunk_mb = int(os.environ.get("CHUNK_MB", "32"))

    for key, label, color in WORKLOADS:
        # Two independent data sources per workload:
        #  - figure_8_{key}_{policy}/ — release-mode benchmark CSVs used by
        #    cost_mape + metric_breakdown panels
        #  - figure_6_{key}_trace/ — trace-mode trace.csv used by the
        #    per-chunk regret panel (primary author's method)
        # A workload is included if EITHER source exists, so running only
        # figure_6 trace mode is enough to populate the regret curve.
        rdir   = os.path.join(sc26_dir, f"figure_8_{key}_{policy}")
        has_f8 = os.path.isdir(rdir)
        tc     = trace_csv(sc26_dir, key)
        if not has_f8 and not tc:
            continue
        any_workload = True

        if key == "ai":
            # AI workload schema is different (inline_benchmark_chunks.csv);
            # everything computed via load_ai_inline_chunks as before.
            if has_f8:
                ai_path = chunks_csv(rdir, "ai")
                ai = load_ai_inline_chunks(ai_path, policy, chunk_mb)
                if ai["cost_mape"]:
                    cost_series.append((label, color, ai["cost_mape"]))
                if ai["regret"]:
                    regret_series.append((label, color, ai["regret"]))
                metrics = {
                    "comp":   final_mean(ai["mape_comp"]),
                    "decomp": final_mean(ai["mape_decomp"]),
                    "ratio":  final_mean(ai["mape_ratio"]),
                    "psnr":   final_mean(ai["mape_psnr"]),
                }
                bar_data.append((label, color, metrics))
            continue

        # Regret (per-chunk, trace-derived — matches primary author's
        # plot_e2e.py exactly, modulo ×100 for percent Y-axis).
        reg_vals = load_trace_regret(tc)
        if reg_vals:
            regret_series.append((label, color, reg_vals))

        # Cost MAPE: prefer trace.csv's mape_cost (per-chunk, same source
        # the primary author's plot_mape_convergence uses). Fall back to
        # figure_8 chunks CSV's cost_model_error_pct only if no trace is
        # available.
        cost_vals = load_trace_cost_mape(tc)
        if not cost_vals and has_f8:
            cc_fallback = chunks_csv(rdir, key)
            cost_vals = load_column(cc_fallback, "cost_model_error_pct",
                                    scale=100.0)
        if cost_vals:
            cost_series.append((label, color, cost_vals))

        # Per-metric MAPE bar still reads figure_8 release-mode chunks CSV
        # (mape_comp/decomp/ratio/psnr columns live there, not in trace.csv).
        if has_f8:
            cc = chunks_csv(rdir, key)
            metrics = {}
            for mkey, src, _l, _c in METRIC_DEFS:
                vals = load_column(cc, src)
                metrics[mkey] = final_mean(vals)
            bar_data.append((label, color, metrics))

    if not any_workload:
        print(f"=== Policy {policy}: no workload subdirs found, skipping")
        return

    print(f"\n=== Policy: {policy} ===")
    plot_line(
        cost_series,
        f"Figure 1 — Cost Model MAPE Convergence ({policy})",
        "Cost Model MAPE (%)",
        os.path.join(fig_dir, f"sc26_{policy}_cost_mape.png"),
        os.path.join(csv_dir, f"sc26_{policy}_cost_mape.csv"),
        cap=100.0,
    )
    plot_line(
        regret_series,
        f"Figure 2 — Regret Convergence ({policy})",
        "Regret (%)",
        os.path.join(fig_dir, f"sc26_{policy}_regret.png"),
        os.path.join(csv_dir, f"sc26_{policy}_regret.csv"),
        cap=100.0,
    )
    plot_bar(
        bar_data,
        f"Figure 3 — Per-Metric MAPE Breakdown ({policy})",
        os.path.join(fig_dir, f"sc26_{policy}_metric_breakdown.png"),
        os.path.join(csv_dir, f"sc26_{policy}_metric_breakdown.csv"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sc26-dir", required=True,
                    help="SC26 sweep root containing <workload>_<policy>/ dirs")
    ap.add_argument("--policies", nargs="+", default=["balanced", "ratio"])
    args = ap.parse_args()

    if not os.path.isdir(args.sc26_dir):
        print(f"ERROR: {args.sc26_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    for pol in args.policies:
        build_for_policy(args.sc26_dir, pol)


if __name__ == "__main__":
    main()
