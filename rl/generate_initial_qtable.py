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
import json
from pathlib import Path

# Configuration (must match config.py)
NUM_ENTROPY_BINS = 10
NUM_ERROR_LEVELS = 3
NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS  # 30

NUM_ALGORITHMS = 8
NUM_QUANT_OPTIONS = 2
NUM_SHUFFLE_OPTIONS = 2
NUM_ACTIONS = NUM_ALGORITHMS * NUM_QUANT_OPTIONS * NUM_SHUFFLE_OPTIONS  # 32

ALGORITHM_NAMES = [
    'lz4', 'snappy', 'deflate', 'gdeflate',
    'zstd', 'ans', 'cascaded', 'bitcomp'
]

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


def encode_action(algorithm_idx: int, quantization: bool, shuffle: bool) -> int:
    """Encode action components into action index."""
    return algorithm_idx + (1 if quantization else 0) * 8 + (1 if shuffle else 0) * 16


def decode_action(action: int) -> dict:
    """Decode action index into components."""
    return {
        'algorithm_idx': action % 8,
        'algorithm': ALGORITHM_NAMES[action % 8],
        'quantization': (action // 8) % 2 == 1,
        'shuffle': (action // 16) % 2 == 1
    }


def compute_heuristic_qvalue(entropy_bin: int, error_level: int, action: int) -> float:
    """
    Compute heuristic Q-value for a state-action pair.

    Args:
        entropy_bin: 0-9 (low to high entropy)
        error_level: 0=aggressive, 1=balanced, 2=precise
        action: 0-31 action index

    Returns:
        Heuristic Q-value (0.0 to 1.0)
    """
    components = decode_action(action)
    algo_name = components['algorithm']
    quant = components['quantization']
    shuffle = components['shuffle']

    algo_chars = ALGO_CHARACTERISTICS[algo_name]

    # Base score from algorithm characteristics
    # Interpolate between low and high entropy scores
    entropy_factor = entropy_bin / 9.0  # 0.0 to 1.0
    algo_score = algo_chars[0] * (1 - entropy_factor) + algo_chars[1] * entropy_factor

    # Quantization bonus/penalty based on error level
    quant_modifier = 0.0
    if quant:
        if error_level == 0:  # Aggressive - quantization helps
            quant_modifier = 0.15
        elif error_level == 1:  # Balanced - slight benefit
            quant_modifier = 0.05
        else:  # Precise - avoid quantization
            quant_modifier = -0.3

    # Shuffle bonus for structured data (mid-range entropy)
    shuffle_modifier = 0.0
    if shuffle:
        # Shuffle helps most for mid-entropy structured data
        if 2 <= entropy_bin <= 6:
            shuffle_modifier = 0.1
        else:
            shuffle_modifier = 0.02

    # Combine scores
    q_value = algo_score + quant_modifier + shuffle_modifier

    # Clamp to valid range
    return max(0.0, min(1.0, q_value))


def generate_qtable() -> np.ndarray:
    """Generate Q-table with heuristic values."""
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)

    for entropy_bin in range(NUM_ENTROPY_BINS):
        for error_level in range(NUM_ERROR_LEVELS):
            state = entropy_bin * NUM_ERROR_LEVELS + error_level

            for action in range(NUM_ACTIONS):
                q_value = compute_heuristic_qvalue(entropy_bin, error_level, action)
                q_table[state, action] = q_value

    return q_table


def save_json(q_table: np.ndarray, path: str):
    """Save Q-table as JSON."""
    data = {
        'version': 1,
        'n_states': NUM_STATES,
        'n_actions': NUM_ACTIONS,
        'q_values': q_table.tolist(),
        'metadata': {
            'type': 'heuristic',
            'description': 'Initial Q-table with heuristic values'
        }
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {path}")


def save_binary(q_table: np.ndarray, path: str):
    """Save Q-table as binary for GPU loading."""
    import struct

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x51544142))  # Magic "QTAB"
        f.write(struct.pack('<I', 1))            # Version
        f.write(struct.pack('<I', NUM_STATES))
        f.write(struct.pack('<I', NUM_ACTIONS))

        # Q-values as flat array
        f.write(q_table.astype(np.float32).tobytes())

    print(f"Saved binary: {path}")


def print_best_actions(q_table: np.ndarray):
    """Print best action for each state."""
    print("\nBest Actions per State:")
    print("=" * 60)
    print(f"{'Entropy':<10} {'Error Level':<15} {'Best Action':<35}")
    print("-" * 60)

    for entropy_bin in range(NUM_ENTROPY_BINS):
        for error_level in range(NUM_ERROR_LEVELS):
            state = entropy_bin * NUM_ERROR_LEVELS + error_level
            best_action = np.argmax(q_table[state])
            components = decode_action(best_action)

            error_names = ['aggressive', 'balanced', 'precise']
            action_str = f"{components['algorithm']}"
            if components['quantization']:
                action_str += "+quant"
            if components['shuffle']:
                action_str += "+shuffle"

            print(f"{entropy_bin:<10} {error_names[error_level]:<15} {action_str:<35}")


def main():
    output_dir = Path(__file__).parent / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating initial Q-table with heuristic values...")
    q_table = generate_qtable()

    # Save both formats
    save_json(q_table, str(output_dir / 'qtable.json'))
    save_binary(q_table, str(output_dir / 'qtable.bin'))

    # Print summary
    print(f"\nQ-table shape: {q_table.shape}")
    print(f"Value range: [{q_table.min():.3f}, {q_table.max():.3f}]")

    print_best_actions(q_table)

    print("\nInitial Q-table generated successfully!")
    print("Run training to improve these values based on actual compression performance.")


if __name__ == '__main__':
    main()
