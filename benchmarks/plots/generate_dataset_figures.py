#!/usr/bin/env python3
"""
Generate all figures for a single dataset + policy combination.

Produces up to 12 figures:
  1_summary.png            - 4-panel: ratio, write, read, Pareto
  3_algorithm_evolution.png - heatmap: algo per chunk over time (nn, nn-rl, nn-rl+exp)
  4_<phase>_predicted_vs_actual.png - per-chunk predicted vs actual (per phase)
  5a_sgd_convergence.png   - MAPE convergence over timesteps
  5b_sgd_exploration_firing.png - SGD/exploration firing rates
  5c_mae_over_time.png     - MAE convergence over timesteps
  5d_psnr_quality.png      - PSNR actual vs predicted (lossy mode only)
  5f_ranking_quality.png   - Kendall tau, selection regret at milestones

Usage:
  python3 generate_dataset_figures.py --dataset hurricane_isabel
  python3 generate_dataset_figures.py --dataset gray_scott --policy balanced_w1-1-1

  # All datasets, all policies:
  python3 generate_dataset_figures.py --all
"""
import argparse
import os
import sys
import glob

from config import *
import visualize as viz

def generate_figures(dataset, policy, out_dir):
    """Generate all applicable figures for one dataset+policy."""
    ensure_dir(out_dir)
    data_dir = get_data_dir(dataset, policy)

    if not os.path.isdir(data_dir):
        print(f"  SKIP {dataset}/{policy}: {data_dir} not found")
        return 0

    count = 0

    # Find aggregate CSV (try exact name first, then glob for partial match)
    agg_csv = ""
    if dataset in DATASETS:
        agg_csv = os.path.join(data_dir, DATASETS[dataset]["agg_csv"])
    if not os.path.exists(agg_csv):
        # Search for any matching CSV (handles long SDRBench names like SDRBENCH-EXASKY-NYX-...)
        all_csvs = glob.glob(os.path.join(data_dir, "benchmark_*.csv"))
        all_csvs = [c for c in all_csvs
                     if "_chunks" not in c and "_timesteps" not in c
                     and "_ranking" not in c]
        ds_lower = dataset.lower().replace("_", "")
        for c in all_csvs:
            bn = os.path.basename(c).lower().replace("-", "").replace("_", "")
            if ds_lower in bn:
                agg_csv = c
                break
        if not agg_csv and all_csvs:
            agg_csv = all_csvs[0]  # fallback: use first available

    # Derive base name from the aggregate CSV for finding related CSVs
    csv_base = ""
    if agg_csv and os.path.exists(agg_csv):
        csv_base = os.path.basename(agg_csv).replace(".csv", "")

    # ── 1. Summary (all policies) ──
    if os.path.exists(agg_csv):
        rows = viz.parse_csv(agg_csv)
        if rows:
            # Fix NN phase ratios: use total_orig/total_compressed from timestep CSV
            # instead of mean(per-timestep ratios) which has Jensen's inequality bias.
            ts_csv_for_fix = ""
            if csv_base:
                ts_csv_for_fix = os.path.join(data_dir, f"{csv_base}_timesteps.csv")
            if not os.path.exists(ts_csv_for_fix) and dataset in DATASETS:
                ts_csv_for_fix = os.path.join(data_dir, DATASETS[dataset]["timesteps_csv"])
            if os.path.exists(ts_csv_for_fix):
                ts_rows = viz.parse_csv(ts_csv_for_fix)
                for r in rows:
                    ph = r.get("phase", "")
                    if ph not in ("nn-rl", "nn-rl+exp50", "nn"):
                        continue
                    ph_ts = [t for t in ts_rows if t.get("phase", "") == ph]
                    if not ph_ts:
                        continue
                    total_file = sum(viz.g(t, "file_bytes") for t in ph_ts)
                    orig_mib = viz.g(r, "orig_mib", "orig_mb")
                    if total_file > 0 and orig_mib > 0:
                        total_file_mib = total_file / (1024 * 1024)
                        r["ratio"] = (len(ph_ts) * orig_mib) / total_file_mib
                        r["file_mib"] = total_file_mib / len(ph_ts)

            out = os.path.join(out_dir, "1_summary.png")
            display = dataset.replace("_", " ").title()

            # Build descriptive metadata subtitle
            r0 = rows[0]
            meta_parts = []
            # Dataset size
            orig = viz.g(r0, "orig_mib", "orig_bytes")
            if orig > 1024 * 1024:  # bytes
                meta_parts.append(f"Dataset: {orig / (1024*1024):.0f} MiB")
            elif orig > 0:
                meta_parts.append(f"Dataset: {orig:.0f} MiB")
            # Chunks
            n_ch = int(viz.g(r0, "n_chunks"))
            if n_ch > 0 and orig > 0:
                chunk_mib = (orig if orig < 1024 else orig / (1024*1024)) / max(n_ch, 1)
                meta_parts.append(f"Chunks: {n_ch} x {chunk_mib:.1f} MiB")
            # Runs
            n_runs = int(viz.g(r0, "n_runs"))
            if n_runs > 1:
                meta_parts.append(f"Runs: {n_runs}")
            # Policy
            policy_label = policy.replace("_w", " (w").replace("-", "=") + ")"
            if "balanced" in policy:
                policy_label = "Balanced (w0=1, w1=1, w2=1)"
            elif "ratio" in policy:
                policy_label = "Ratio-only (w0=0, w1=0, w2=1)"
            elif "speed" in policy:
                policy_label = "Speed-only (w0=1, w1=1, w2=0)"
            meta_parts.append(f"Policy: {policy_label}")
            # GS-specific
            L = viz.g(r0, "L")
            if L > 0:
                meta_parts.append(f"Grid: {int(L)}^3")
            steps = viz.g(r0, "steps")
            if steps > 0:
                meta_parts.append(f"Steps: {int(steps)}")

            meta_text = " | ".join(meta_parts)
            viz.make_summary_figure(display, rows, out, meta_text)
            count += 1

    # ── 3. Algorithm evolution heatmap (all policies) ──
    # Try derived name from csv_base first, then config registry, then generic
    tc_csv = os.path.join(data_dir, f"{csv_base}_timestep_chunks.csv") if csv_base else ""
    ch_csv = os.path.join(data_dir, f"{csv_base}_chunks.csv") if csv_base else ""
    if not os.path.exists(tc_csv) and dataset in DATASETS:
        tc_csv = os.path.join(data_dir, DATASETS[dataset]["timestep_chunks_csv"])
        ch_csv = os.path.join(data_dir, DATASETS[dataset]["chunks_csv"])
    if not os.path.exists(tc_csv):
        tc_csv = os.path.join(data_dir, f"benchmark_{dataset}_timestep_chunks.csv")
        ch_csv = os.path.join(data_dir, f"benchmark_{dataset}_chunks.csv")

    if os.path.exists(tc_csv):
        out = os.path.join(out_dir, "3_algorithm_evolution.png")
        viz.make_milestone_actions_figure(tc_csv, out,
                                          ch_csv if os.path.exists(ch_csv) else None)
        count += 1

    # ── 4. Predicted vs actual — per-phase figures ──
    if os.path.exists(tc_csv):
        for ph in ("nn", "nn-rl", "nn-rl+exp50"):
            out_ph = os.path.join(out_dir, f"4_{ph}_predicted_vs_actual.png")
            viz.make_timestep_chunks_figure(tc_csv, out_ph, phase_filter=ph)
            count += 1

    # ── 5. Learning dynamics: MAPE + firing rates ──
    ts_csv = os.path.join(data_dir, f"{csv_base}_timesteps.csv") if csv_base else ""
    if not os.path.exists(ts_csv) and dataset in DATASETS:
        ts_csv = os.path.join(data_dir, DATASETS[dataset]["timesteps_csv"])
    if not os.path.exists(ts_csv):
        ts_csv = os.path.join(data_dir, f"benchmark_{dataset}_timesteps.csv")

    if os.path.exists(ts_csv):
        out_mape = os.path.join(out_dir, "5a_sgd_convergence.png")
        viz.make_timestep_figure(ts_csv, out_mape)
        count += 1

        out_firing = os.path.join(out_dir, "5b_sgd_exploration_firing.png")
        viz.make_sgd_exploration_figure(ts_csv, out_firing)
        count += 1

        out_mae = os.path.join(out_dir, "5c_mae_over_time.png")
        viz.make_mae_figure(ts_csv, out_mae)
        count += 1

        out_psnr = os.path.join(out_dir, "5d_psnr_quality.png")
        viz.make_psnr_figure(ts_csv, out_psnr)
        count += 1

    # ── 5e. Ranking quality (Kendall tau) ──
    ranking_csv = ""
    if csv_base:
        ranking_csv = os.path.join(data_dir, f"{csv_base}_ranking.csv")
    if not os.path.exists(ranking_csv) and dataset in DATASETS:
        ranking_csv = os.path.join(data_dir,
            DATASETS[dataset].get("agg_csv", "").replace(".csv", "_ranking.csv"))
    if not os.path.exists(ranking_csv):
        # Try without the vol infix (phase-major GS naming)
        ranking_csv = os.path.join(data_dir, f"benchmark_{dataset.replace('_', '')}_ranking.csv")
    if not os.path.exists(ranking_csv):
        ranking_csv = os.path.join(data_dir, f"benchmark_grayscott_ranking.csv")

    if os.path.exists(ranking_csv):
        out_ranking = os.path.join(out_dir, "5f_ranking_quality.png")
        viz.make_ranking_quality_figure(ranking_csv, out_ranking)
        count += 1

    # ── 6. Pipeline waterfall ──
    if os.path.exists(ts_csv):
        out_wf = os.path.join(out_dir, "6b_pipeline_waterfall.png")
        viz.make_pipeline_waterfall(ts_csv, out_wf)
        count += 1

        out_gpu = os.path.join(out_dir, "6c_gpu_breakdown_over_time.png")
        viz.make_gpu_breakdown_over_time(ts_csv, out_gpu)
        count += 1

    # ── 6d. Cross-phase pipeline overhead comparison ──
    # Walk phase_<name>/ directories in BOTH the policy dir (NN phases) and
    # the sibling fixed_phases/ dir (fixed-algorithm phases) to collect
    # per-phase timestep CSVs.
    phase_csv_map = {}
    search_dirs = [data_dir]
    # Also search fixed_phases/ sibling (Gray-Scott / VPIC layout)
    fixed_sibling = os.path.join(os.path.dirname(data_dir), "fixed_phases")
    if os.path.isdir(fixed_sibling):
        search_dirs.append(fixed_sibling)

    ts_candidates = (
        "benchmark_vpic_deck_timesteps.csv",
        "benchmark_grayscott_timesteps.csv",
        f"benchmark_{dataset}_timesteps.csv",
    )
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for entry in sorted(os.listdir(search_dir)):
            if not entry.startswith("phase_"):
                continue
            ph_name = entry[len("phase_"):]
            for candidate in ts_candidates:
                cpath = os.path.join(search_dir, entry, candidate)
                if os.path.exists(cpath):
                    phase_csv_map[ph_name] = cpath
                    break
    # Fallback: if no phase_* directories found, split the combined timestep
    # CSV by the 'phase' column (VPIC/Gray-Scott have all phases in one file).
    if len(phase_csv_map) < 2 and ts_csv and os.path.exists(ts_csv):
        import csv, tempfile
        with open(ts_csv) as f:
            reader = csv.DictReader(f)
            rows_by_phase = {}
            header = reader.fieldnames
            for row in reader:
                ph = row.get("phase", "")
                # Strip policy suffix (e.g., "nn-rl/balanced" → "nn-rl")
                if "/" in ph:
                    ph = ph.split("/")[0]
                if ph and ph not in rows_by_phase:
                    rows_by_phase[ph] = []
                if ph:
                    rows_by_phase[ph].append(row)

        if len(rows_by_phase) >= 2:
            # Write temporary per-phase CSVs
            tmpdir = tempfile.mkdtemp(prefix="gpucompress_phase_split_")
            for ph, rows in rows_by_phase.items():
                tmp_csv = os.path.join(tmpdir, f"{ph}.csv")
                with open(tmp_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(rows)
                phase_csv_map[ph] = tmp_csv

    if len(phase_csv_map) >= 2:
        out_pd = os.path.join(out_dir, "6d_cross_phase_pipeline_overhead.png")
        viz.make_cross_phase_pipeline_overhead(phase_csv_map, out_pd)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate per-dataset benchmark figures")
    parser.add_argument("--dataset", help="Dataset name (e.g., hurricane_isabel, gray_scott)")
    parser.add_argument("--policy", default=None,
                        help="Cost model policy (default: all applicable)")
    parser.add_argument("--all", action="store_true", help="Generate for all datasets")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Specify --dataset NAME or --all")

    # Determine datasets to process
    if args.all:
        datasets = list(DATASETS.keys()) + list(SDR_DATASETS.keys())
    else:
        datasets = [args.dataset]

    total = 0
    for ds in datasets:
        # Determine policies to run
        if args.policy:
            policies = [args.policy]
        else:
            policies_to_run = []
            for pol in ALL_POLICIES:
                d = get_data_dir(ds, pol)
                if os.path.isdir(d):
                    policies_to_run.append(pol)
            if not policies_to_run:
                policies_to_run = [BALANCED]
            policies = policies_to_run

        for pol in policies:
            # Include eval parameters in output dir so results aren't overwritten
            data_dir = get_data_dir(ds, pol)
            eval_name = ""
            if data_dir:
                # Walk up to find the eval_* directory name
                # e.g., .../results/eval_NX254_chunk16mb_ts100/ratio_only_w0-0-1
                parts = data_dir.replace("\\", "/").split("/")
                for p in parts:
                    if p.startswith("eval_"):
                        eval_name = p
                        break
            if eval_name:
                out_dir = os.path.join(PER_DATASET, ds, eval_name, pol)
            else:
                out_dir = os.path.join(PER_DATASET, ds, pol)
            print(f"\n{'='*60}")
            print(f"  {ds} / {pol}")
            print(f"{'='*60}")
            n = generate_figures(ds, pol, out_dir)
            total += n
            print(f"  Generated {n} figures in {out_dir}")

    print(f"\nTotal: {total} figures in {PER_DATASET}/")


if __name__ == "__main__":
    main()
