#!/usr/bin/env python3
"""
Export Q-Table to Binary Format for GPU Loading

Converts trained Q-Table from JSON to binary format that can be
loaded directly into GPU constant memory.

Usage:
    python -m rl.export_qtable rl/models/qtable.json rl/models/qtable.bin
"""

import argparse
import struct
import json
import numpy as np
from pathlib import Path

from .config import NUM_STATES, NUM_ACTIONS


def export_json_to_binary(json_path: str, bin_path: str) -> bool:
    """
    Convert Q-Table from JSON to binary format.

    Binary format:
        - 4 bytes: magic (0x51544142 = "QTAB")
        - 4 bytes: version (1)
        - 4 bytes: n_states (30)
        - 4 bytes: n_actions (32)
        - 3840 bytes: q_values (960 float32)

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


def export_numpy_to_binary(q_values: np.ndarray, bin_path: str) -> bool:
    """
    Export numpy Q-table array to binary format.

    Args:
        q_values: numpy array of shape (NUM_STATES, NUM_ACTIONS)
        bin_path: Path to output binary file

    Returns:
        True on success
    """
    if q_values.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"Error: Q-table shape {q_values.shape}, expected ({NUM_STATES}, {NUM_ACTIONS})")
        return False

    q_values = q_values.astype(np.float32)

    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', 0x51544142))  # Magic
        f.write(struct.pack('<I', 1))            # Version
        f.write(struct.pack('<I', NUM_STATES))
        f.write(struct.pack('<I', NUM_ACTIONS))
        f.write(q_values.tobytes())

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

        if version != 1:
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


def main():
    parser = argparse.ArgumentParser(
        description='Export Q-Table from JSON to binary format'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input JSON file path'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output binary file path'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output file after export'
    )

    args = parser.parse_args()

    success = export_json_to_binary(args.input, args.output)

    if success and args.verify:
        verify_binary(args.output)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
