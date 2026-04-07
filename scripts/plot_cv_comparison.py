#!/usr/bin/env python3
"""Plot NN 5-fold CV MAPE and R² bar charts.

Parses the full-pipeline output file automatically, or accepts hardcoded values.
Produces two PNGs:
    neural_net/weights/cv_comparison.png   - MAPE per output
    neural_net/weights/cv_r2.png           - R² per output

Usage:
    python scripts/plot_cv_comparison.py                          # uses latest full-pipeline-*.out
    python scripts/plot_cv_comparison.py full-pipeline-12345.out  # specific file
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import re


def parse_cv_summary(filepath):
    """Parse NN CV summary from pipeline output file.

    Returns dict: results['nn'][output_name] = {
        'mape': (mean, std), 'r2': (mean, std), 'mae': (mean, std)
    }
    """
    with open(filepath) as f:
        text = f.read()

    results = {}
    idx = text.find('NN 5-FOLD')
    if idx < 0:
        return results
    block = text[idx:idx+2000]
    results['nn'] = {}
    for line in block.split('\n'):
        # Match lines like: compression_time_ms     4.3761±0.0570  0.3986±0.1146   71.4±1.5%
        # Trailing tag like " [lossy-only]" is ignored.
        m = re.match(
            r'\s+(\S+)\s+'
            r'([\d.]+)±([\d.]+)\s+'   # MAE
            r'([\d.]+)±([\d.]+)\s+'   # R²
            r'([\d.]+)±([\d.]+)%',    # MAPE
            line,
        )
        if m:
            name = m.group(1)
            results['nn'][name] = {
                'mae': (float(m.group(2)), float(m.group(3))),
                'r2': (float(m.group(4)), float(m.group(5))),
                'mape': (float(m.group(6)), float(m.group(7))),
            }
    return results


# Output display order and labels
OUTPUT_ORDER = [
    ('compression_time_ms', 'Compression\nTime'),
    ('decompression_time_ms', 'Decompression\nTime'),
    ('compression_ratio', 'Compression\nRatio'),
    ('psnr_db', 'PSNR'),
    ('mean_abs_err', 'Pointwise\nError (MAE)'),
    ('ssim', 'SSIM'),
]

# Find pipeline output file
if len(sys.argv) > 1:
    logfile = sys.argv[1]
else:
    files = sorted(glob.glob('full-pipeline-*.out'), key=lambda f: f)
    if not files:
        print("No full-pipeline-*.out found. Run the pipeline first.")
        sys.exit(1)
    logfile = files[-1]

print(f"Parsing: {logfile}")
results = parse_cv_summary(logfile)

if 'nn' not in results:
    print("Could not find NN CV summary in the file.")
    sys.exit(1)

# Build arrays
labels = []
nn_mape, mape_std, nn_r2, r2_std = [], [], [], []
for key, label in OUTPUT_ORDER:
    if key in results['nn']:
        row = results['nn'][key]
        labels.append(label)
        nn_mape.append(row['mape'][0])
        mape_std.append(row['mape'][1])
        nn_r2.append(row['r2'][0])
        r2_std.append(row['r2'][1])

x = np.arange(len(labels))
width = 0.55


def _bar_chart(values, errs, ylabel, value_fmt, out_path,
               color, edge, hatch, ypad):
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(x, values, width, yerr=errs, capsize=4,
                  label='Neural Network',
                  color=color, edgecolor=edge, linewidth=1.2, hatch=hatch)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11, loc='upper right')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + ypad,
                value_fmt.format(h),
                ha='center', va='bottom', fontsize=8, color=edge)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# MAPE chart (lower is better)
_bar_chart(nn_mape, mape_std,
           ylabel='MAPE (%)', value_fmt='{:.1f}%',
           out_path='neural_net/weights/cv_comparison.png',
           color='#AEC7E8', edge='#4C72B0', hatch='///', ypad=1.0)

# R² chart (higher is better, capped at 1.0)
_bar_chart(nn_r2, r2_std,
           ylabel='R² (coefficient of determination)', value_fmt='{:.3f}',
           out_path='neural_net/weights/cv_r2.png',
           color='#C7E9C0', edge='#2CA02C', hatch='\\\\\\', ypad=0.02)
