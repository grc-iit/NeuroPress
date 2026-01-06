#!/usr/bin/env python3
"""
Visualize entropy analysis results from exported data files
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

# Check if required files exist
if not os.path.exists('entropy_analysis.json'):
    print("ERROR: entropy_analysis.json not found!")
    print("\nPlease run analyze_patternsANDentropy.py first to generate entropy analysis.")
    print("Example: python3 analyze_patternsANDentropy.py")
    sys.exit(1)

if not os.path.exists('entropy_comparison.csv'):
    print("ERROR: entropy_comparison.csv not found!")
    print("\nPlease run analyze_patternsANDentropy.py first to generate entropy analysis.")
    print("Example: python3 analyze_patternsANDentropy.py")
    sys.exit(1)

# Read entropy analysis results
with open('entropy_analysis.json', 'r') as f:
    analysis = json.load(f)

# Read entropy comparison CSV
entropy_data = np.loadtxt('entropy_comparison.csv', delimiter=',', skiprows=1, dtype=str)

if entropy_data.size == 0:
    print("ERROR: entropy_comparison.csv is empty!")
    print("\nPlease run analyze_patternsANDentropy.py first to generate entropy analysis.")
    sys.exit(1)

# Handle single pattern case (1D array) vs multiple patterns (2D array)
if entropy_data.ndim == 1:
    patterns = [entropy_data[0]]
    entropy_values = np.array([float(entropy_data[1])])
    normalized_values = np.array([float(entropy_data[2])])
else:
    patterns = entropy_data[:, 0]
    entropy_values = entropy_data[:, 1].astype(float)
    normalized_values = entropy_data[:, 2].astype(float)

num_patterns = len(patterns)

# Create comprehensive visualization with dynamic sizing
fig_height = max(12, 4 + num_patterns * 2)
fig = plt.figure(figsize=(18, fig_height))

# Dynamic grid: top row for summary, then rows for each pattern's histogram
grid_rows = 1 + ((num_patterns + 1) // 2)  # 1 for summary + rows for histograms (2 per row)
gs = GridSpec(grid_rows, 3, figure=fig, hspace=0.3, wspace=0.3)

fig.suptitle('Scientific Data Patterns: Shannon Entropy Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. Bar chart of entropy values
ax1 = fig.add_subplot(gs[0, :2])
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax1.bar(patterns, entropy_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
ax1.set_title('Entropy Comparison Across Patterns', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(entropy_values) * 1.2)

# Add value labels on bars
for bar, val in zip(bars, entropy_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Normalized entropy (percentage)
ax2 = fig.add_subplot(gs[0, 2])
bars2 = ax2.barh(patterns, normalized_values * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Normalized Entropy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Entropy Utilization', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, 100)

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars2, normalized_values * 100)):
    ax2.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)

# 3-N. Histogram distributions for each pattern
# Build list of available histogram files
available_histograms = []
color_cycle = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for idx, pattern_name in enumerate(patterns):
    hist_filename = f"{pattern_name.lower()}_histogram.csv"
    if os.path.exists(hist_filename):
        color_idx = idx % len(color_cycle)
        available_histograms.append((pattern_name, hist_filename, color_cycle[color_idx]))

# Plot histograms
for idx, (name, filename, color) in enumerate(available_histograms):
    row = 1 + idx // 2
    col = idx % 2
    
    try:
        ax = fig.add_subplot(gs[row, col])
        
        # Read histogram data
        hist_data = np.loadtxt(filename, delimiter=',', skiprows=1)
        bin_centers = hist_data[:, 0]
        probabilities = hist_data[:, 2]
        
        # Plot distribution
        ax.fill_between(bin_centers, probabilities, alpha=0.3, color=color)
        ax.plot(bin_centers, probabilities, linewidth=2, color=color, label=name)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f'{name} - Value Distribution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Add entropy annotation
        if name in analysis['patterns']:
            pattern_data = analysis['patterns'][name]
            entropy_val = pattern_data['entropy_bits']
            ax.text(0.02, 0.98, f'Entropy: {entropy_val:.3f} bits\nNorm: {pattern_data["normalized_entropy"]*100:.1f}%',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        print(f"Warning: Could not plot histogram for {name}: {e}")

# Statistics summary table (place in last available position)
table_row = 1 + ((num_patterns - 1) // 2)
table_col = 2 if num_patterns % 2 == 1 else 2

ax_table = fig.add_subplot(gs[table_row, table_col])
ax_table.axis('off')

table_data = []
for pattern in patterns:
    if pattern in analysis['patterns']:
        p_data = analysis['patterns'][pattern]
        table_data.append([
            pattern,
            f"{p_data['entropy_bits']:.3f}",
            f"{p_data['statistics']['mean']:.1f}",
            f"{p_data['statistics']['std']:.1f}"
        ])

if table_data:
    table = ax_table.table(cellText=table_data,
                           colLabels=['Pattern', 'Entropy\n(bits)', 'Mean', 'StdDev'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')

    ax_table.set_title('Statistics Summary', fontsize=12, fontweight='bold', pad=10)

# Save figure
plt.savefig('entropy_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as: entropy_visualization.png")

# Show plot
plt.show()

