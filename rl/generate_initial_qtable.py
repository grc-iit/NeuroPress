#!/usr/bin/env python3
"""
Generate Initial Q-Table with Heuristic Values

Creates a Q-table with reasonable initial values based on compression heuristics:
- Low entropy data: Fast algorithms (LZ4, Snappy) preferred
- High entropy data: Strong algorithms (ZSTD, ANS) with preprocessing
- Error levels affect quantization preferences

This allows the system to work before training.
"""

import numpy as np
from pathlib import Path

try:
    from .qtable import QTable
    from .config import (NUM_ENTROPY_BINS, NUM_ERROR_LEVELS, NUM_MAD_BINS,
                         NUM_DERIV_BINS, NUM_STATES, NUM_ACTIONS, ALGORITHM_NAMES)
except ImportError:
    from qtable import QTable
    from config import (NUM_ENTROPY_BINS, NUM_ERROR_LEVELS, NUM_MAD_BINS,
                        NUM_DERIV_BINS, NUM_STATES, NUM_ACTIONS, ALGORITHM_NAMES)

# Algorithm characteristics (heuristic scores)
# [low_entropy_score, high_entropy_score, speed_score]
ALGO_CHARACTERISTICS = {
    'lz4':      [0.9, 0.3, 1.0],  # Fast, good for low entropy
    'snappy':   [0.8, 0.3, 0.95], # Fast, good for low entropy
    'deflate':  [0.7, 0.6, 0.5],  # Balanced
    'gdeflate': [0.75, 0.65, 0.6], # GPU-optimized deflate
    'zstd':     [0.6, 0.8, 0.4],  # Strong compression
    'ans':      [0.5, 0.85, 0.3], # Best for high entropy
    'cascaded': [0.65, 0.7, 0.35], # Multi-stage
    'bitcomp':  [0.4, 0.75, 0.45], # Bit-level compression
}


def compute_heuristic_qvalue(entropy_bin: int, error_level: int, action: int,
                             mad_bin: int = 0, deriv_bin: int = 0) -> float:
    """
    Compute heuristic Q-value for a state-action pair.

    Args:
        entropy_bin: 0-15 (low to high entropy)
        error_level: 0=aggressive, 1=balanced, 2=precise, 3=lossless
        action: 0-31 action index
        mad_bin: 0-3 (low to high MAD)
        deriv_bin: 0-3 (smooth to noisy)

    Returns:
        Heuristic Q-value (0.0 to 1.0)
    """
    components = QTable.decode_action(action)
    algo_name = components['algorithm']
    quant = components['quantization']
    shuffle = components['shuffle_size'] > 0

    algo_chars = ALGO_CHARACTERISTICS[algo_name]

    # Base score from algorithm characteristics
    # Interpolate between low and high entropy scores (bin maps to entropy * 2)
    entropy_factor = (entropy_bin * 0.5) / 8.0  # byte entropy max is 8.0
    algo_score = algo_chars[0] * (1 - entropy_factor) + algo_chars[1] * entropy_factor

    # Quantization bonus/penalty based on error level
    quant_modifier = 0.0
    if quant:
        if error_level == 0:  # Aggressive - quantization helps
            quant_modifier = 0.15
        elif error_level == 1:  # Balanced - slight benefit
            quant_modifier = 0.05
        elif error_level == 2:  # Precise - small penalty
            quant_modifier = -0.15
        else:  # Lossless - must not quantize
            quant_modifier = -0.5

    # Shuffle bonus for structured data (mid-range entropy)
    shuffle_modifier = 0.0
    if shuffle:
        # Shuffle helps most for mid-entropy structured data (bins 4-12 = entropy 2.0-6.0)
        if 4 <= entropy_bin <= 12:
            shuffle_modifier = 0.1
        else:
            shuffle_modifier = 0.02

    # MAD modifier: low MAD → favor fast algorithms; high MAD → favor strong algorithms
    mad_modifier = 0.0
    if mad_bin <= 1:
        # Low MAD (clustered data): fast algorithms get a bonus
        if algo_chars[2] >= 0.8:  # speed_score >= 0.8 (lz4, snappy)
            mad_modifier = 0.05
    elif mad_bin >= 3:
        # High MAD (variable data): strong compression algorithms get a bonus
        if algo_chars[1] >= 0.7:  # high_entropy_score >= 0.7 (zstd, ans, cascaded, bitcomp)
            mad_modifier = 0.05

    # Derivative modifier: smooth data → shuffle bonus; rough data → strong compression
    deriv_modifier = 0.0
    if deriv_bin <= 1:
        # Smooth data: shuffle preprocessing helps more
        if shuffle:
            deriv_modifier = 0.05
    elif deriv_bin >= 3:
        # Very rough/noisy data: strong compression gets a bonus
        if algo_chars[1] >= 0.7:
            deriv_modifier = 0.05

    # Combine scores
    q_value = algo_score + quant_modifier + shuffle_modifier + mad_modifier + deriv_modifier

    # Clamp to valid range
    return max(0.0, min(1.0, q_value))


def generate_qtable() -> QTable:
    """Generate Q-table with heuristic values for all 4 dimensions."""
    qt = QTable()

    for entropy_bin in range(NUM_ENTROPY_BINS):
        for error_level in range(NUM_ERROR_LEVELS):
            for mad_bin in range(NUM_MAD_BINS):
                for deriv_bin in range(NUM_DERIV_BINS):
                    state = ((entropy_bin * NUM_ERROR_LEVELS + error_level)
                             * NUM_MAD_BINS + mad_bin) * NUM_DERIV_BINS + deriv_bin

                    for action in range(NUM_ACTIONS):
                        qt.q_values[state, action] = compute_heuristic_qvalue(
                            entropy_bin, error_level, action,
                            mad_bin=mad_bin, deriv_bin=deriv_bin)

    return qt


def main():
    output_dir = Path(__file__).parent / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating initial Q-table with heuristic values...")
    qt = generate_qtable()

    # Save both formats using QTable methods
    json_path = str(output_dir / 'qtable.json')
    bin_path = str(output_dir / 'qtable.bin')
    qt.save(json_path)
    qt.export_binary(bin_path)
    print(f"Saved JSON: {json_path}")
    print(f"Saved binary: {bin_path}")

    # Print summary
    print(f"\nQ-table shape: {qt.q_values.shape}")
    print(f"Value range: [{qt.q_values.min():.3f}, {qt.q_values.max():.3f}]")

    qt.print_best_actions()

    print("\nInitial Q-table generated successfully!")
    print("Run training to improve these values based on actual compression performance.")


if __name__ == '__main__':
    main()
