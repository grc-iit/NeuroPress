#!/usr/bin/env python3
"""
Thin wrapper that invokes analysis/plot_e2e.py with trace paths pointed at
evaluations/figure_6/run_all.sh's output layout (TRACE_ROOT/<w>/trace.csv)
instead of the Docker-devcontainer paths hardcoded in plot_e2e.py:

    WORKLOADS = {
        "LAMMPS": Path("/workspaces/GPUCompress/output/sim_runs/lammps/trace.csv"),
        "VPIC":   Path("/workspaces/GPUCompress/output/sim_runs/vpic/trace.csv"),
        "NYX":    Path("/workspaces/GPUCompress/output/sim_runs/nyx/trace.csv"),
    }

We monkey-patch that dict before calling plot_fig7 / plot_fig8 so the
analysis script itself stays untouched (the primary author's code remains
the source of truth).
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "analysis"))
    import plot_e2e  # noqa: E402  (after sys.path mutation)

    trace_root = Path(os.environ.get("TRACE_ROOT", repo_root / "trace_runs"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7-out", type=Path,
                        default=trace_root / "figures" / "fig7")
    parser.add_argument("--fig8-out", type=Path,
                        default=trace_root / "figures" / "fig8")
    parser.add_argument("--bin", type=int, default=16, dest="bin_size")
    parser.add_argument("--trace-root", type=Path, default=trace_root,
                        help="Directory containing <workload>/trace.csv")
    parser.add_argument("--skip-missing", action="store_true", default=True,
                        help="Skip workloads whose trace.csv is missing")
    args = parser.parse_args()

    # Rebuild plot_e2e.WORKLOADS with our Jarvis figure_6 layout paths.
    # Figure_6 pipelines write to <tmp>/figure_6_<wl>_trace/gpucompress_trace.csv
    # (WarpX nests one level deeper under vol_nn-rl+exp50/).
    candidates = {
        "LAMMPS": args.trace_root / "figure_6_lammps_trace" / "gpucompress_trace.csv",
        "VPIC":   args.trace_root / "figure_6_vpic_trace"   / "gpucompress_trace.csv",
        "NYX":    args.trace_root / "figure_6_nyx_trace"    / "gpucompress_trace.csv",
    }
    warpx_flat   = args.trace_root / "figure_6_warpx_trace" / "gpucompress_trace.csv"
    warpx_nested = args.trace_root / "figure_6_warpx_trace" / "vol_nn-rl+exp50" / "gpucompress_trace.csv"
    if warpx_flat.is_file():     candidates["WARPX"] = warpx_flat
    elif warpx_nested.is_file(): candidates["WARPX"] = warpx_nested
    workloads = {}
    for wl, path in candidates.items():
        if path.is_file():
            workloads[wl] = path
        elif not args.skip_missing:
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"  [{wl}] trace.csv missing -> skipping")
    if not workloads:
        print("No trace.csv files found; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plot_e2e.WORKLOADS = workloads
    # Extend WL_COLORS for any workload the primary author's dict doesn't
    # cover (e.g. WARPX). Additive — doesn't modify their committed entries.
    extra_colors = {"WARPX": "#9b5fe0", "AI": "#e377c2"}
    for wl, c in extra_colors.items():
        plot_e2e.WL_COLORS.setdefault(wl, c)

    args.fig7_out.mkdir(parents=True, exist_ok=True)
    args.fig8_out.mkdir(parents=True, exist_ok=True)

    plot_e2e.plot_fig7(workloads, args.fig7_out, args.bin_size)
    plot_e2e.plot_fig8(workloads, args.fig8_out)


if __name__ == "__main__":
    main()
