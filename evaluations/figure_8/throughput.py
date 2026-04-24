"""
evaluations/figure_8/throughput.py — reproduces paper Fig 9
(end2end.png) by driving analysis/plot_e2e.py:plot_fig8 with the
figure_6 trace CSVs.

plot_e2e.py owns the renderer (primary author's file). We own the
input-path mapping + workload ordering so the output matches paper
Fig 9:
  VPIC, Nyx, LAMMPS, WarpX, AI — each with 8 colored stacked bars
  (Baseline, NVComp, NVComp+Tier, NP+Tier, NP+Tier+Async, +Lossy L/M/H)
  showing total wall-clock time in minutes.

Usage:
    python3 evaluations/figure_8/throughput.py
    # → $HOME/GPUCompress/tmp/figure_8/fig8_throughput.png

Missing / header-only trace.csvs are skipped with a warning — the plot
renders with whatever subset has data. For the full paper-match you
need a populated trace.csv for all five workloads.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.plot_e2e import plot_fig8  # noqa: E402

# Paper Fig 9 workload order (left → right on the x-axis).
PAPER_ORDER = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _has_rows(csv_path: Path) -> bool:
    if not csv_path.is_file():
        return False
    with csv_path.open() as f:
        next(f, None)
        return next(f, None) is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-base", default="$HOME/GPUCompress/tmp",
                    help="Dir containing figure_6_<workload>_trace/ subdirs")
    ap.add_argument("--out-dir", default="$HOME/GPUCompress/tmp/figure_8",
                    help="Where fig8_throughput.png lands")
    for key in ("vpic", "nyx", "lammps", "warpx", "ai"):
        ap.add_argument(f"--{key}-trace", default=None,
                        help=f"Override {key.upper()} trace.csv path")
    args = ap.parse_args()

    base = _expand(args.trace_base)
    # Map paper display name → default trace path. Figure_6 pipelines
    # write to tmp/figure_6_<wl>_trace/gpucompress_trace.csv (lowercase
    # workload key).
    defaults = {
        "VPIC":   base / "figure_6_vpic_trace"   / "gpucompress_trace.csv",
        "Nyx":    base / "figure_6_nyx_trace"    / "gpucompress_trace.csv",
        "LAMMPS": base / "figure_6_lammps_trace" / "gpucompress_trace.csv",
        "WarpX":  base / "figure_6_warpx_trace"  / "gpucompress_trace.csv",
        "AI":     base / "figure_6_ai_trace"     / "gpucompress_trace.csv",
    }
    overrides = {
        "VPIC":   args.vpic_trace,
        "Nyx":    args.nyx_trace,
        "LAMMPS": args.lammps_trace,
        "WarpX":  args.warpx_trace,
        "AI":     args.ai_trace,
    }
    for name, override in overrides.items():
        if override:
            defaults[name] = _expand(override)

    # Keep paper's left-to-right workload order. The WarpX pipeline writes
    # its trace.csv one level deeper under vol_nn-rl+exp50/ (the other
    # workloads write flat). Auto-fall-back to that nested path so
    # reviewers don't need --warpx-trace unless they archived somewhere
    # custom.
    nested_fallbacks = {
        "WarpX": base / "figure_6_warpx_trace" / "vol_nn-rl+exp50" / "gpucompress_trace.csv",
    }
    trace_paths = {}
    for name in PAPER_ORDER:
        p = defaults[name]
        if _has_rows(p):
            trace_paths[name] = p
        elif name in nested_fallbacks and _has_rows(nested_fallbacks[name]):
            trace_paths[name] = nested_fallbacks[name]
            print(f"[auto] {name}: using nested path {nested_fallbacks[name]}",
                  file=sys.stderr)
        else:
            print(f"[skip] {name}: {p} missing or header-only", file=sys.stderr)

    if not trace_paths:
        print("error: no populated trace.csv found — run evaluations/figure_6 first",
              file=sys.stderr)
        return 1

    out_dir = _expand(args.out_dir)
    plot_fig8(trace_paths, out_dir)
    print(f"Figure 8 throughput panel: {out_dir}/fig8_throughput.png")
    print(f"Workloads included: {', '.join(trace_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
