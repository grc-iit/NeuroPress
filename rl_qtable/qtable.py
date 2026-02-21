"""
Q-Table Implementation for Compression Algorithm Selection

Provides the core Q-Table data structure with update logic,
persistence (save/load), and action selection.
"""

import numpy as np
import json
from typing import Tuple
from pathlib import Path

from .config import (
    NUM_STATES, NUM_ACTIONS, NUM_ENTROPY_BINS, NUM_ERROR_LEVELS,
    NUM_MAD_BINS, NUM_DERIV_BINS,
    LEARNING_RATE, ALGORITHM_NAMES, ERROR_LEVEL_THRESHOLDS,
    MAD_BIN_THRESHOLDS, DERIV_BIN_THRESHOLDS
)


class QTable:
    """
    Q-Table for reinforcement learning based compression selection.

    State: (entropy_bin, error_level, mad_bin, deriv_bin) encoded as single integer
    Action: (algorithm, quantization, shuffle) encoded as single integer
    """

    def __init__(self, learning_rate: float = LEARNING_RATE):
        """Initialize Q-Table with zeros."""
        self.q_values = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
        self.visit_counts = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
        self.learning_rate = learning_rate

    @staticmethod
    def _value_to_bin(value: float, thresholds: list) -> int:
        """Map a continuous value to a bin index using thresholds."""
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)

    @staticmethod
    def encode_state(entropy: float, error_bound: float,
                     mad: float = 0.0, second_derivative: float = 0.0) -> int:
        """
        Encode data characteristics into state index.

        Args:
            entropy: Shannon entropy in bits (0.0 to 8.0+)
            error_bound: Quantization error bound (0 for lossless)
            mad: Mean Absolute Deviation (default 0.0 for backward compat)
            second_derivative: Mean absolute second derivative (default 0.0)

        Returns:
            State index (0 to NUM_STATES-1)
        """
        # Discretize entropy to 0.5-width bins (0-15)
        entropy_bin = int(entropy * 2)
        entropy_bin = max(0, min(NUM_ENTROPY_BINS - 1, entropy_bin))

        # Determine error level
        if error_bound <= 0:
            error_level = 3  # Lossless (no quantization, zero error)
        elif error_bound >= ERROR_LEVEL_THRESHOLDS[0]:
            error_level = 0  # Aggressive lossy (>= 0.1)
        elif error_bound >= ERROR_LEVEL_THRESHOLDS[1]:
            error_level = 1  # Moderate lossy (>= 0.01)
        elif error_bound >= ERROR_LEVEL_THRESHOLDS[2]:
            error_level = 2  # Precise lossy (>= 0.001)
        else:
            error_level = 3  # Below 0.001 treated as lossless

        # Discretize MAD and second derivative
        mad_bin = QTable._value_to_bin(mad, MAD_BIN_THRESHOLDS)
        deriv_bin = QTable._value_to_bin(second_derivative, DERIV_BIN_THRESHOLDS)

        return ((entropy_bin * NUM_ERROR_LEVELS + error_level)
                * NUM_MAD_BINS + mad_bin) * NUM_DERIV_BINS + deriv_bin

    @staticmethod
    def decode_state(state: int) -> tuple:
        """
        Decode state index into component bins.

        Returns:
            (entropy_bin, error_level, mad_bin, deriv_bin)
        """
        deriv_bin = state % NUM_DERIV_BINS
        state //= NUM_DERIV_BINS
        mad_bin = state % NUM_MAD_BINS
        state //= NUM_MAD_BINS
        error_level = state % NUM_ERROR_LEVELS
        entropy_bin = state // NUM_ERROR_LEVELS
        return (entropy_bin, error_level, mad_bin, deriv_bin)

    @staticmethod
    def decode_action(action: int) -> dict:
        """
        Decode action index into compression configuration.

        Args:
            action: Action index (0 to NUM_ACTIONS-1)

        Returns:
            Dictionary with algorithm, quantization, shuffle settings
        """
        algorithm_idx = action % 8
        use_quantization = (action // 8) % 2 == 1
        shuffle_size = 4 if (action // 16) % 2 == 1 else 0

        return {
            'algorithm': ALGORITHM_NAMES[algorithm_idx],
            'algorithm_idx': algorithm_idx,
            'quantization': use_quantization,
            'shuffle_size': shuffle_size
        }

    def get_best_action(self, state: int) -> Tuple[int, float]:
        """
        Get best action for a state (argmax of Q-values).

        Returns:
            Tuple of (action_index, q_value)
        """
        best_action = np.argmax(self.q_values[state])
        return int(best_action), self.q_values[state, best_action]

    def update(self, state: int, action: int, reward: float) -> float:
        """
        Update Q-value using simple Q-learning update.

        Q(s,a) ← Q(s,a) + α * (reward - Q(s,a))

        Note: No discount factor since this is single-step (no future states).

        Args:
            state: State index
            action: Action index
            reward: Observed reward

        Returns:
            The TD error (reward - Q(s,a))
        """
        old_q = self.q_values[state, action]
        td_error = reward - old_q
        self.q_values[state, action] = old_q + self.learning_rate * td_error
        self.visit_counts[state, action] += 1
        return td_error

    def get_state_coverage(self) -> dict:
        """Get statistics about state coverage during training."""
        total_visits = self.visit_counts.sum()
        states_visited = (self.visit_counts.sum(axis=1) > 0).sum()
        actions_per_state = (self.visit_counts > 0).sum(axis=1)

        return {
            'total_visits': int(total_visits),
            'states_visited': int(states_visited),
            'states_total': NUM_STATES,
            'coverage_pct': 100.0 * states_visited / NUM_STATES,
            'avg_actions_per_state': float(actions_per_state.mean()),
        }

    def save(self, filepath: str):
        """
        Save Q-Table to JSON file.

        Args:
            filepath: Path to save file (.json)
        """
        data = {
            'version': 2,
            'n_states': NUM_STATES,
            'n_actions': NUM_ACTIONS,
            'n_entropy_bins': NUM_ENTROPY_BINS,
            'n_error_levels': NUM_ERROR_LEVELS,
            'n_mad_bins': NUM_MAD_BINS,
            'n_deriv_bins': NUM_DERIV_BINS,
            'learning_rate': self.learning_rate,
            'q_values': self.q_values.tolist(),
            'visit_counts': self.visit_counts.tolist()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load Q-Table from JSON file.

        Args:
            filepath: Path to load file (.json)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        if data['n_states'] != NUM_STATES or data['n_actions'] != NUM_ACTIONS:
            raise ValueError(f"Q-Table size mismatch: expected {NUM_STATES}x{NUM_ACTIONS}, "
                           f"got {data['n_states']}x{data['n_actions']}")

        self.q_values = np.array(data['q_values'], dtype=np.float32)
        self.visit_counts = np.array(data.get('visit_counts',
                                              np.zeros_like(self.q_values)), dtype=np.int32)
        self.learning_rate = data.get('learning_rate', LEARNING_RATE)

    def export_binary(self, filepath: str):
        """
        Export Q-Table to binary format for GPU loading.

        Binary format:
            - 4 bytes: magic (0x51544142 = "QTAB")
            - 4 bytes: version (1)
            - 4 bytes: n_states
            - 4 bytes: n_actions
            - n_states * n_actions * 4 bytes: q_values (float32)

        Args:
            filepath: Path to save file (.bin)
        """
        import struct

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            # Header
            f.write(struct.pack('<I', 0x51544142))  # Magic "QTAB"
            f.write(struct.pack('<I', 1))           # Version
            f.write(struct.pack('<I', NUM_STATES))
            f.write(struct.pack('<I', NUM_ACTIONS))

            # Q-values as flat array
            f.write(self.q_values.astype(np.float32).tobytes())

    def print_best_actions(self):
        """Print best action for each state (only visited or non-zero states)."""
        error_names = ['aggressive', 'moderate', 'precise', 'lossless']
        mad_names = ['low', 'med', 'high', 'vhigh']
        deriv_names = ['smooth', 'mod', 'rough', 'noisy']

        print("\nBest Actions per State:")
        print("=" * 95)
        print(f"{'Entropy':<10} {'Error':<12} {'MAD':<8} {'Deriv':<8} {'Best Action':<28} {'Q-Value':<10}")
        print("-" * 95)

        for state in range(NUM_STATES):
            entropy_bin, error_level, mad_bin, deriv_bin = self.decode_state(state)

            best_action, q_value = self.get_best_action(state)
            if q_value == 0.0 and self.visit_counts[state].sum() == 0:
                continue

            action_config = self.decode_action(best_action)
            entropy_label = f"{entropy_bin * 0.5:.1f}-{(entropy_bin + 1) * 0.5:.1f}"

            action_str = f"{action_config['algorithm']}"
            if action_config['quantization']:
                action_str += "+quant"
            if action_config['shuffle_size'] > 0:
                action_str += f"+shuffle{action_config['shuffle_size']}"

            print(f"{entropy_label:<10} {error_names[error_level]:<12} "
                  f"{mad_names[mad_bin]:<8} {deriv_names[deriv_bin]:<8} "
                  f"{action_str:<28} {q_value:<10.4f}")
