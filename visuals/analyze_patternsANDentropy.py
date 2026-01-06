#!/usr/bin/env python3
"""
Data Entropy Analyzer for Scientific Patterns
Uses Shannon Entropy to measure data randomness/information content
"""

from scipy.stats import entropy
import numpy as np
import json
import os

def calculate_shannon_entropy(data, num_bins=50):
    """
    Calculate Shannon Entropy for continuous data by discretizing into bins
    
    Args:
        data: numpy array of values
        num_bins: number of bins for discretization
    
    Returns:
        entropy in bits
    """
    # Flatten data if 2D
    flat_data = data.flatten()
    
    # Create histogram (discretize continuous data)
    counts, _ = np.histogram(flat_data, bins=num_bins)
    
    # Remove zero counts (they don't contribute to entropy)
    counts = counts[counts > 0]
    
    # Calculate probabilities
    probabilities = counts / len(flat_data)
    
    # Compute Shannon Entropy (base 2 for bits)
    data_entropy = entropy(probabilities, base=2)
    
    return data_entropy

def entropy_interpretation(h, max_entropy):
    """
    Interpret entropy value
    
    Args:
        h: calculated entropy
        max_entropy: maximum possible entropy for the binning
    
    Returns:
        interpretation string
    """
    normalized = h / max_entropy if max_entropy > 0 else 0
    
    if normalized < 0.3:
        return "Very Low - Highly structured/predictable data"
    elif normalized < 0.5:
        return "Low - Structured with some variation"
    elif normalized < 0.7:
        return "Medium - Balanced structure and randomness"
    elif normalized < 0.9:
        return "High - Mostly random with slight structure"
    else:
        return "Very High - Nearly uniform random distribution"

# Analyze all patterns
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

print("=" * 80)
print(" SHANNON ENTROPY ANALYSIS OF SCIENTIFIC DATA PATTERNS".center(80))
print("=" * 80)
print("\nShannon Entropy measures the information content and randomness of data.")
print("Higher entropy = more random/unpredictable data")
print("Lower entropy = more structured/predictable data")
print("=" * 80)

if not patterns:
    print("\n⚠ WARNING: No pattern CSV files found!")
    print("\nExpected files: smooth_pattern.csv, turbulent_pattern.csv, periodic_pattern.csv, noisy_pattern.csv")
    print("or: pattern.csv")
    print("\nPlease run the benchmark program first to generate pattern data.")
    print("Example: ./benchmark 100 10 float smooth")
    print("=" * 80)
    exit(1)

print(f"\nFound {len(patterns)} pattern file(s) to analyze:")
for name, filename in patterns.items():
    print(f"  ✓ {name}: {filename}")
print()

results = []

for name, filename in patterns.items():
    try:
        # Load data
        data = np.loadtxt(filename, delimiter=',')
        
        # Calculate basic statistics
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        
        # Calculate entropy with different bin sizes
        num_bins = 50
        h = calculate_shannon_entropy(data, num_bins=num_bins)
        max_entropy = np.log2(num_bins)  # Maximum entropy for uniform distribution
        
        # Also calculate with more bins for comparison
        h_100 = calculate_shannon_entropy(data, num_bins=100)
        max_entropy_100 = np.log2(100)
        
        results.append({
            'name': name,
            'entropy': h,
            'max_entropy': max_entropy,
            'normalized': h / max_entropy,
            'shape': data.shape,
            'mean': data_mean,
            'std': data_std,
            'range': data_max - data_min
        })
        
        print(f"\n{'─' * 80}")
        print(f" {name.upper()} PATTERN".center(80))
        print(f"{'─' * 80}")
        print(f"\n  Data Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        print(f"  Total Values: {data.size:,}")
        print(f"\n  Value Statistics:")
        print(f"    • Range: [{data_min:.3f}, {data_max:.3f}]")
        print(f"    • Mean:  {data_mean:.3f}")
        print(f"    • StdDev: {data_std:.3f}")
        
        print(f"\n  Shannon Entropy Analysis:")
        print(f"    • Entropy (50 bins):  {h:.4f} bits")
        print(f"    • Max Entropy:        {max_entropy:.4f} bits")
        print(f"    • Normalized:         {h/max_entropy:.4f} ({h/max_entropy*100:.1f}%)")
        print(f"    • Entropy (100 bins): {h_100:.4f} bits")
        print(f"    • Interpretation:     {entropy_interpretation(h, max_entropy)}")
        
        # Pattern-specific insights
        print(f"\n  Scientific Context:")
        if name == 'Smooth':
            print(f"    • Temperature field simulation - should have LOW entropy")
            print(f"    • Smooth gradients → predictable transitions")
            print(f"    • Expected: Structured, deterministic pattern")
        elif name == 'Turbulent':
            print(f"    • Turbulent flow simulation - should have MEDIUM-HIGH entropy")
            print(f"    • Multi-scale chaos → moderate unpredictability")
            print(f"    • Expected: Structured chaos with patterns")
        elif name == 'Periodic':
            print(f"    • Wave oscillation simulation - should have MEDIUM entropy")
            print(f"    • Regular cycles → predictable but varying")
            print(f"    • Expected: Repetitive structure")
        elif name == 'Noisy':
            print(f"    • Gaussian noise simulation - should have HIGH entropy")
            print(f"    • Random sampling → high unpredictability")
            print(f"    • Expected: Random, information-rich data")
        
    except Exception as e:
        print(f"\n✗ Error analyzing {name}: {e}")

# Summary comparison
print(f"\n{'=' * 80}")
print(" ENTROPY COMPARISON SUMMARY".center(80))
print(f"{'=' * 80}\n")

# Sort by entropy
results_sorted = sorted(results, key=lambda x: x['entropy'])

print(f"  {'Pattern':<15} {'Entropy':<12} {'Normalized':<12} {'Interpretation':<20}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*20}")

for r in results_sorted:
    interp = "Low" if r['normalized'] < 0.5 else "Medium" if r['normalized'] < 0.7 else "High"
    print(f"  {r['name']:<15} {r['entropy']:.4f} bits  {r['normalized']:.4f}       {interp}")

print(f"\n  Ranking (by entropy - lowest to highest):")
for i, r in enumerate(results_sorted, 1):
    bars = "█" * int(r['normalized'] * 40)
    print(f"    {i}. {r['name']:<12} │{bars:<40}│ {r['normalized']*100:.1f}%")

print(f"\n{'=' * 80}")

# Export results to files
print("\n📊 Exporting analysis results...")

# 1. Save detailed statistics to JSON
output_json = {
    'analysis_type': 'Shannon Entropy Analysis',
    'num_bins': 50,
    'patterns': {}
}

for r in results:
    output_json['patterns'][r['name']] = {
        'entropy_bits': float(r['entropy']),
        'max_entropy': float(r['max_entropy']),
        'normalized_entropy': float(r['normalized']),
        'data_shape': list(r['shape']),
        'statistics': {
            'mean': float(r['mean']),
            'std': float(r['std']),
            'range': float(r['range'])
        }
    }

with open('entropy_analysis.json', 'w') as f:
    json.dump(output_json, f, indent=2)
print("  ✓ Saved detailed statistics to: entropy_analysis.json")

# 2. Save entropy comparison to CSV
with open('entropy_comparison.csv', 'w') as f:
    f.write("Pattern,Entropy_bits,Normalized,Max_Entropy,Mean,StdDev,Range\n")
    for r in results_sorted:
        f.write(f"{r['name']},{r['entropy']:.6f},{r['normalized']:.6f},"
                f"{r['max_entropy']:.6f},{r['mean']:.6f},{r['std']:.6f},{r['range']:.6f}\n")
print("  ✓ Saved entropy comparison to: entropy_comparison.csv")

# 3. Save histogram data for each pattern (for distribution visualization)
if results:
    for name, filename in patterns.items():
        try:
            data = np.loadtxt(filename, delimiter=',')
            flat_data = data.flatten()
            
            # Calculate histogram
            counts, bin_edges = np.histogram(flat_data, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Save histogram
            hist_filename = f"{name.lower()}_histogram.csv"
            with open(hist_filename, 'w') as f:
                f.write("BinCenter,Count,Probability\n")
                total = len(flat_data)
                for center, count in zip(bin_centers, counts):
                    prob = count / total
                    f.write(f"{center:.6f},{count},{prob:.8f}\n")
            print(f"  ✓ Saved {name} histogram to: {hist_filename}")
        except Exception as e:
            print(f"  ✗ Error creating histogram for {name}: {e}")

print("\n✓ Analysis complete! All data exported successfully.")
print("\nGenerated files:")
print("  • entropy_analysis.json      - Detailed statistics (JSON)")
print("  • entropy_comparison.csv     - Entropy values (CSV)")
print("  • *_histogram.csv            - Distribution data for each pattern")
print("\nFor visualization with plots, install matplotlib:")
print("  sudo apt install python3-matplotlib")
print("  python3 visualize_patterns.py")
print("=" * 80)
