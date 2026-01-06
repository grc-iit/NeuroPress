#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define potential pattern files
all_patterns = {
    'Smooth': 'smooth_pattern.csv',
    'Turbulent': 'turbulent_pattern.csv',
    'Periodic': 'periodic_pattern.csv',
    'Noisy': 'noisy_pattern.csv'
}

# Also check for the generic pattern.csv file
if os.path.exists('pattern.csv') and 'pattern.csv' not in all_patterns.values():
    all_patterns['Pattern'] = 'pattern.csv'

# Filter to only existing files
patterns = {name: filename for name, filename in all_patterns.items() if os.path.exists(filename)}

# Check if we have any patterns to visualize
if not patterns:
    print("=" * 70)
    print(" ERROR: NO PATTERN FILES FOUND ".center(70))
    print("=" * 70)
    print("\nNo pattern CSV files found in the current directory!")
    print("\nExpected files: smooth_pattern.csv, turbulent_pattern.csv,")
    print("                periodic_pattern.csv, noisy_pattern.csv")
    print("or: pattern.csv")
    print("\nPlease run the benchmark program first to generate pattern data.")
    print("Example: ./benchmark 100 10 float smooth")
    print("=" * 70)
    exit(1)

# Detect data dimensions dynamically
print("=" * 70)
print(" LOADING PATTERN DATA ".center(70))
print("=" * 70)

data_shapes = {}
for name, filename in patterns.items():
    try:
        temp_data = np.loadtxt(filename, delimiter=',')
        data_shapes[name] = temp_data.shape
        print(f"  ✓ {name:12} : {temp_data.shape[0]:3}×{temp_data.shape[1]:3} = {temp_data.size:,} values")
    except Exception as e:
        print(f"  ✗ {name:12} : Error - {e}")

print("=" * 70 + "\n")

if not data_shapes:
    print("ERROR: Could not load any pattern data!")
    exit(1)

# Create figure with dynamic sizing based on number of patterns
num_patterns = len(patterns)
fig_height = max(8, num_patterns * 3)
sample_shape = list(data_shapes.values())[0] if data_shapes else (0, 0)
fig = plt.figure(figsize=(16, fig_height))
fig.suptitle(f'Scientific Data Patterns Visualization\n[Data size: {sample_shape[0]}×{sample_shape[1]} per pattern]', 
             fontsize=16, fontweight='bold')

for idx, (name, filename) in enumerate(patterns.items(), 1):
    try:
        data = np.loadtxt(filename, delimiter=',')
        rows, cols = data.shape
        
        # 2D Heatmap
        ax1 = fig.add_subplot(num_patterns, 3, idx * 3 - 2)
        im = ax1.imshow(data, cmap='viridis', aspect='auto', interpolation='bilinear')
        ax1.set_title(f'{name} Pattern - Heatmap [{rows}×{cols}]', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # 3D Surface Plot (showing entire dataset)
        ax2 = fig.add_subplot(num_patterns, 3, idx * 3 - 1, projection='3d')
        
        # Use entire dataset without subsampling
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        surf = ax2.plot_surface(X, Y, data, cmap='viridis', alpha=0.8)
        ax2.set_title(f'{name} Pattern - 3D Surface', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        ax2.set_zlabel('Value')
        
        # Line plot showing all rows
        ax3 = fig.add_subplot(num_patterns, 3, idx * 3)
        for i in range(rows):
            ax3.plot(data[i, :], alpha=0.7, linewidth=1.5)
        ax3.set_title(f'{name} Pattern - Row Profiles (all {rows} rows)', 
                     fontsize=11, fontweight='bold')
        ax3.set_xlabel('Column Index')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        print(f"✓ {name} pattern: shape={data.shape}, min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}, std={data.std():.2f}")
        
    except Exception as e:
        print(f"✗ Error loading {name} pattern: {e}")

plt.tight_layout()
plt.savefig('patterns_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'patterns_visualization.png'")
plt.show()

