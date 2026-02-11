#!/usr/bin/env python3
"""
Export and Inspect Q-Table

Converts trained Q-Table from JSON to binary format for GPU loading,
and provides human-readable dump for inspection.

Usage:
    python -m rl.export_qtable rl/models/qtable.json rl/models/qtable.bin
    python -m rl.export_qtable rl/models/qtable.json --dump report.txt
    python -m rl.export_qtable rl/models/qtable.json rl/models/qtable.bin --verify --dump report.txt
"""

import argparse
import struct
import json
import numpy as np
from pathlib import Path

from .qtable import QTable
from .config import (
    NUM_STATES, NUM_ACTIONS, NUM_ENTROPY_BINS, NUM_ERROR_LEVELS,
    NUM_MAD_BINS, NUM_DERIV_BINS
)


def export_json_to_binary(json_path: str, bin_path: str) -> bool:
    """
    Convert Q-Table from JSON to binary format.

    Binary format:
        - 4 bytes: magic (0x51544142 = "QTAB")
        - 4 bytes: version (1)
        - 4 bytes: n_states (1024)
        - 4 bytes: n_actions (32)
        - 131072 bytes: q_values (32768 float32, ~128KB)

    Args:
        json_path: Path to input JSON file
        bin_path: Path to output binary file

    Returns:
        True on success
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate
    n_states = data.get('n_states', len(data.get('q_values', [])))
    n_actions = data.get('n_actions', len(data.get('q_values', [[]])[0]))

    if n_states != NUM_STATES:
        print(f"Warning: n_states={n_states}, expected {NUM_STATES}")
    if n_actions != NUM_ACTIONS:
        print(f"Warning: n_actions={n_actions}, expected {NUM_ACTIONS}")

    # Extract Q-values
    q_values = np.array(data['q_values'], dtype=np.float32)

    if q_values.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"Error: Q-table shape {q_values.shape}, expected ({NUM_STATES}, {NUM_ACTIONS})")
        return False

    # Write binary
    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)

    with open(bin_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x51544142))  # Magic "QTAB"
        f.write(struct.pack('<I', 1))            # Version
        f.write(struct.pack('<I', NUM_STATES))
        f.write(struct.pack('<I', NUM_ACTIONS))

        # Q-values as flat array (row-major order)
        f.write(q_values.tobytes())

    # Verify
    file_size = Path(bin_path).stat().st_size
    expected_size = 16 + NUM_STATES * NUM_ACTIONS * 4  # header + data

    if file_size != expected_size:
        print(f"Error: File size {file_size}, expected {expected_size}")
        return False

    print(f"Exported Q-Table to {bin_path}")
    print(f"  States: {NUM_STATES}")
    print(f"  Actions: {NUM_ACTIONS}")
    print(f"  File size: {file_size} bytes")

    return True


def verify_binary(bin_path: str) -> bool:
    """
    Verify a binary Q-table file.

    Args:
        bin_path: Path to binary file

    Returns:
        True if valid
    """
    with open(bin_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_states = struct.unpack('<I', f.read(4))[0]
        n_actions = struct.unpack('<I', f.read(4))[0]

        if magic != 0x51544142:
            print(f"Invalid magic: {hex(magic)}")
            return False

        if version not in (1, 2):
            print(f"Unknown version: {version}")
            return False

        if n_states != NUM_STATES or n_actions != NUM_ACTIONS:
            print(f"Size mismatch: {n_states}x{n_actions}")
            return False

        data = f.read()
        expected_bytes = NUM_STATES * NUM_ACTIONS * 4

        if len(data) != expected_bytes:
            print(f"Data size mismatch: {len(data)} vs {expected_bytes}")
            return False

    print(f"Binary Q-table valid: {bin_path}")
    return True


def dump_qtable(qtable_path: str, output_path: str):
    """Dump full Q-table contents to a human-readable file."""
    qt = QTable()
    qt.load(qtable_path)

    error_names = ['aggressive', 'moderate', 'precise', 'lossless']
    mad_names = ['low', 'med', 'high', 'vhigh']
    deriv_names = ['smooth', 'mod', 'rough', 'noisy']
    lines = []

    lines.append("=" * 100)
    lines.append("Q-Table Dump")
    lines.append("=" * 100)
    lines.append(f"Source: {qtable_path}")
    lines.append(f"States: {NUM_STATES} ({NUM_ENTROPY_BINS} entropy x {NUM_ERROR_LEVELS} error x {NUM_MAD_BINS} MAD x {NUM_DERIV_BINS} deriv)")
    lines.append(f"Actions: {NUM_ACTIONS}")
    lines.append(f"Data size: {NUM_STATES * NUM_ACTIONS * 4} bytes ({NUM_STATES * NUM_ACTIONS * 4 / 1024:.0f}KB)")
    lines.append("")

    # Coverage summary
    coverage = qt.get_state_coverage()
    lines.append("Coverage:")
    lines.append(f"  States visited: {coverage['states_visited']}/{coverage['states_total']} ({coverage['coverage_pct']:.1f}%)")
    lines.append(f"  Total visits: {coverage['total_visits']}")
    lines.append("")

    # Best action per state
    lines.append("=" * 100)
    lines.append("Best Action per State")
    lines.append("=" * 100)
    lines.append(f"{'Entropy':<10} {'Error':<12} {'MAD':<8} {'Deriv':<8} {'Best Action':<28} {'Q-Value':<10} {'Visits':<8}")
    lines.append("-" * 100)

    for state in range(NUM_STATES):
        entropy_bin, error_level, mad_bin, deriv_bin = QTable.decode_state(state)
        best_action, q_value = qt.get_best_action(state)
        total_visits = int(qt.visit_counts[state].sum())

        if q_value == 0.0 and total_visits == 0:
            continue

        action_config = QTable.decode_action(best_action)
        entropy_label = f"{entropy_bin * 0.5:.1f}-{(entropy_bin + 1) * 0.5:.1f}"

        action_str = action_config['algorithm']
        if action_config['quantization']:
            action_str += "+quant"
        if action_config['shuffle_size'] > 0:
            action_str += f"+shuffle{action_config['shuffle_size']}"

        lines.append(f"{entropy_label:<10} {error_names[error_level]:<12} "
                     f"{mad_names[mad_bin]:<8} {deriv_names[deriv_bin]:<8} "
                     f"{action_str:<28} {q_value:<10.4f} {total_visits:<8}")

    lines.append("")

    # Full Q-values per visited state (top 5 actions)
    lines.append("=" * 100)
    lines.append("Top 5 Actions per Visited State")
    lines.append("=" * 100)

    for state in range(NUM_STATES):
        entropy_bin, error_level, mad_bin, deriv_bin = QTable.decode_state(state)
        total_visits = int(qt.visit_counts[state].sum())

        if total_visits == 0:
            continue

        entropy_label = f"{entropy_bin * 0.5:.1f}-{(entropy_bin + 1) * 0.5:.1f}"
        lines.append(f"\nState {state}: entropy={entropy_label}, error={error_names[error_level]}, "
                     f"mad={mad_names[mad_bin]}, deriv={deriv_names[deriv_bin]}, visits={total_visits}")
        lines.append(f"  {'Rank':<5} {'Action':<30} {'Q-Value':<10} {'Visits':<8}")
        lines.append(f"  {'-'*55}")

        q_row = qt.q_values[state]
        top_indices = np.argsort(q_row)[::-1][:5]

        for rank, action in enumerate(top_indices, 1):
            if q_row[action] == 0 and qt.visit_counts[state, action] == 0:
                continue
            action_config = QTable.decode_action(action)
            action_str = action_config['algorithm']
            if action_config['quantization']:
                action_str += "+quant"
            if action_config['shuffle_size'] > 0:
                action_str += f"+shuffle{action_config['shuffle_size']}"

            lines.append(f"  {rank:<5} {action_str:<30} {q_row[action]:<10.4f} {int(qt.visit_counts[state, action]):<8}")

    lines.append("")

    # Full raw Q-values matrix
    lines.append("=" * 80)
    lines.append("Raw Q-Values Matrix (state x action)")
    lines.append("=" * 80)

    header = f"{'State':<8}"
    for a in range(NUM_ACTIONS):
        header += f"{a:<8}"
    lines.append(header)
    lines.append("-" * (8 + NUM_ACTIONS * 8))

    for state in range(NUM_STATES):
        if qt.visit_counts[state].sum() == 0:
            continue
        row = f"{state:<8}"
        for a in range(NUM_ACTIONS):
            row += f"{qt.q_values[state, a]:<8.4f}"
        lines.append(row)

    lines.append("")

    content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Q-table dump written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export and inspect Q-Table'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input JSON file path'
    )
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        default=None,
        help='Output binary file path (for export)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output file after export'
    )
    parser.add_argument(
        '--dump',
        type=str,
        default=None,
        metavar='FILE',
        help='Dump Q-table to human-readable file'
    )

    args = parser.parse_args()

    if not args.output and not args.dump:
        parser.error('Provide an output path for export, --dump for inspection, or both')

    success = True

    if args.output:
        success = export_json_to_binary(args.input, args.output)
        if success and args.verify:
            verify_binary(args.output)

    if args.dump:
        dump_qtable(args.input, args.dump)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
