#!/usr/bin/env python3
"""
Visualization script for GPUCompress compression results.

Generates charts from the CSV output of run_simple_tests.sh:
1. Compression ratio comparison by configuration
2. Error vs compression ratio trade-off
3. PSNR comparison across patterns
4. Best configurations summary

Usage:
    python3 visualize_results.py [csv_file] [output_dir]

    Defaults:
        csv_file: test_results/compression_results.csv
        output_dir: test_results/charts/
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_file)

    # Create a configuration label
    def config_label(row):
        parts = []
        if row['quantization'] != 'none':
            parts.append(f"quant(eb={row['error_bound']})")
        if row['shuffle'] != 0:
            parts.append("shuffle")
        if not parts:
            return "baseline"
        return " + ".join(parts)

    df['config'] = df.apply(config_label, axis=1)

    # Replace inf with a large number for plotting
    df['psnr_db'] = df['psnr_db'].replace('inf', 999)
    df['psnr_db'] = pd.to_numeric(df['psnr_db'], errors='coerce')

    return df

def plot_compression_ratios(df, output_dir):
    """Bar chart: Compression ratios by pattern and configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compression Ratios by Data Pattern', fontsize=14, fontweight='bold')

    patterns = df['pattern'].unique()

    for ax, pattern in zip(axes.flatten(), patterns):
        pattern_df = df[df['pattern'] == pattern]

        # Group by algorithm and config
        pivot = pattern_df.pivot_table(
            values='ratio',
            index='config',
            columns='algorithm',
            aggfunc='mean'
        )

        pivot.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'{pattern.capitalize()} Pattern')
        ax.set_ylabel('Compression Ratio')
        ax.set_xlabel('')
        ax.legend(title='Algorithm')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_ratios.png'), dpi=150)
    plt.close()
    print(f"  Created: compression_ratios.png")

def plot_error_vs_ratio(df, output_dir):
    """Scatter plot: Error bound vs compression ratio trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to only quantized data
    quant_df = df[df['quantization'] == 'linear'].copy()

    markers = {'smooth': 'o', 'turbulent': 's', 'periodic': '^', 'noisy': 'D'}
    colors = {'deflate': 'blue', 'zstd': 'red'}

    for pattern in quant_df['pattern'].unique():
        for algo in quant_df['algorithm'].unique():
            subset = quant_df[(quant_df['pattern'] == pattern) & (quant_df['algorithm'] == algo)]
            ax.scatter(
                subset['error_bound'],
                subset['ratio'],
                marker=markers.get(pattern, 'o'),
                c=colors.get(algo, 'gray'),
                label=f'{pattern}-{algo}',
                s=80,
                alpha=0.7
            )

    ax.set_xscale('log')
    ax.set_xlabel('Error Bound (log scale)', fontsize=11)
    ax.set_ylabel('Compression Ratio', fontsize=11)
    ax.set_title('Compression Ratio vs Error Bound Trade-off', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_ratio.png'), dpi=150)
    plt.close()
    print(f"  Created: error_vs_ratio.png")

def plot_psnr_comparison(df, output_dir):
    """Bar chart: PSNR by pattern and error bound."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to quantized data only
    quant_df = df[df['quantization'] == 'linear'].copy()
    quant_df = quant_df[quant_df['psnr_db'] < 500]  # Exclude inf values

    # Group by pattern and error_bound
    pivot = quant_df.pivot_table(
        values='psnr_db',
        index='pattern',
        columns='error_bound',
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_xlabel('Data Pattern', fontsize=11)
    ax.set_title('Signal Quality (PSNR) by Pattern and Error Bound', fontsize=12, fontweight='bold')
    ax.legend(title='Error Bound')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=150)
    plt.close()
    print(f"  Created: psnr_comparison.png")

def plot_config_comparison(df, output_dir):
    """Grouped bar chart: All configurations compared."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create simplified config names
    def short_config(row):
        if row['quantization'] == 'none' and row['shuffle'] == 0:
            return 'Baseline'
        elif row['quantization'] == 'none' and row['shuffle'] != 0:
            return 'Shuffle'
        elif row['quantization'] == 'linear' and row['shuffle'] == 0:
            return f'Quant(eb={row["error_bound"]})'
        else:
            return f'Quant+Shuf(eb={row["error_bound"]})'

    df['short_config'] = df.apply(short_config, axis=1)

    # Use zstd algorithm for cleaner comparison
    zstd_df = df[df['algorithm'] == 'zstd']

    pivot = zstd_df.pivot_table(
        values='ratio',
        index='short_config',
        columns='pattern',
        aggfunc='mean'
    )

    # Reorder for logical presentation
    config_order = ['Baseline', 'Shuffle',
                    'Quant(eb=0.01)', 'Quant(eb=0.001)', 'Quant(eb=0.0001)',
                    'Quant+Shuf(eb=0.01)', 'Quant+Shuf(eb=0.001)', 'Quant+Shuf(eb=0.0001)']
    pivot = pivot.reindex([c for c in config_order if c in pivot.index])

    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Compression Ratio', fontsize=11)
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_title('Compression Ratio by Configuration (zstd algorithm)', fontsize=12, fontweight='bold')
    ax.legend(title='Pattern')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'config_comparison.png'), dpi=150)
    plt.close()
    print(f"  Created: config_comparison.png")

def plot_best_configs(df, output_dir):
    """Summary table: Best configuration for each pattern."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Find best config for each pattern
    results = []
    for pattern in df['pattern'].unique():
        pattern_df = df[df['pattern'] == pattern]

        # Best lossless
        lossless = pattern_df[pattern_df['quantization'] == 'none']
        if not lossless.empty:
            best_lossless = lossless.loc[lossless['ratio'].idxmax()]
            results.append({
                'Pattern': pattern,
                'Type': 'Best Lossless',
                'Config': f"{best_lossless['algorithm']} + {'shuffle' if best_lossless['shuffle'] else 'no shuffle'}",
                'Ratio': f"{best_lossless['ratio']:.2f}x",
                'Max Error': '0 (exact)'
            })

        # Best lossy (highest ratio with reasonable error)
        lossy = pattern_df[pattern_df['quantization'] == 'linear']
        if not lossy.empty:
            best_lossy = lossy.loc[lossy['ratio'].idxmax()]
            results.append({
                'Pattern': pattern,
                'Type': 'Best Lossy',
                'Config': f"{best_lossy['algorithm']} + quant(eb={best_lossy['error_bound']}) + {'shuffle' if best_lossy['shuffle'] else 'no shuffle'}",
                'Ratio': f"{best_lossy['ratio']:.2f}x",
                'Max Error': f"{best_lossy['max_error']:.2e}"
            })

    results_df = pd.DataFrame(results)

    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(results_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(results_df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Best Configurations Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_configs.png'), dpi=150)
    plt.close()
    print(f"  Created: best_configs.png")

def plot_improvement_heatmap(df, output_dir):
    """Heatmap: Improvement over baseline."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate improvement over baseline for each pattern
    improvements = []

    for pattern in df['pattern'].unique():
        pattern_df = df[df['pattern'] == pattern]
        baseline = pattern_df[(pattern_df['quantization'] == 'none') & (pattern_df['shuffle'] == 0)]
        baseline_ratio = baseline[baseline['algorithm'] == 'zstd']['ratio'].values[0]

        for _, row in pattern_df[pattern_df['algorithm'] == 'zstd'].iterrows():
            improvement = row['ratio'] / baseline_ratio
            improvements.append({
                'pattern': pattern,
                'config': f"q={row['quantization']}, eb={row['error_bound']}, shuf={row['shuffle']}",
                'improvement': improvement
            })

    imp_df = pd.DataFrame(improvements)
    pivot = imp_df.pivot_table(values='improvement', index='config', columns='pattern')

    im = ax.imshow(pivot.values, cmap='YlGn', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement Factor vs Baseline')

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.mean() else 'black'
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Improvement Over Baseline (zstd)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Created: improvement_heatmap.png")

def main():
    # Parse arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_results/compression_results.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'test_results/charts'

    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if not os.path.isabs(csv_file):
        csv_file = os.path.join(project_dir, csv_file)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_dir, output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {csv_file}")
    print(f"Saving charts to: {output_dir}")
    print()

    # Load data
    df = load_data(csv_file)
    print(f"Loaded {len(df)} test results")
    print(f"Patterns: {', '.join(df['pattern'].unique())}")
    print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    print()

    # Generate charts
    print("Generating visualizations...")
    plot_compression_ratios(df, output_dir)
    plot_error_vs_ratio(df, output_dir)
    plot_psnr_comparison(df, output_dir)
    plot_config_comparison(df, output_dir)
    plot_best_configs(df, output_dir)
    plot_improvement_heatmap(df, output_dir)

    print()
    print(f"All charts saved to: {output_dir}")

if __name__ == '__main__':
    main()
