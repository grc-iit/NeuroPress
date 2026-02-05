"""
Q-Table Implementation for Compression Algorithm Selection

Provides the core Q-Table data structure with update logic,
persistence (save/load), and action selection.
"""

import numpy as np
import json
from typing import Tuple, Optional
from pathlib import Path

from .config import (
    NUM_STATES, NUM_ACTIONS, NUM_ENTROPY_BINS, NUM_ERROR_LEVELS,
    LEARNING_RATE, ALGORITHM_NAMES, ERROR_LEVEL_THRESHOLDS
)


class QTable:
    """
    Q-Table for reinforcement learning based compression selection.

    State: (entropy_bin, error_level) encoded as single integer
    Action: (algorithm, quantization, shuffle) encoded as single integer
    """

    def __init__(self, learning_rate: float = LEARNING_RATE):
        """Initialize Q-Table with zeros."""
        self.q_values = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
        self.visit_counts = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
        self.learning_rate = learning_rate

    @staticmethod
    def encode_state(entropy: float, error_bound: float) -> int:
        """
        Encode entropy and error bound into state index.

        Args:
            entropy: Shannon entropy in bits (0.0 to 8.0+)
            error_bound: Quantization error bound (0 for lossless)

        Returns:
            State index (0 to NUM_STATES-1)
        """
        # Discretize entropy to bin (0-9)
        entropy_bin = int(entropy)
        entropy_bin = max(0, min(NUM_ENTROPY_BINS - 1, entropy_bin))

        # Determine error level
        if error_bound <= 0:
            error_level = 2  # Lossless = most precise
        elif error_bound >= ERROR_LEVEL_THRESHOLDS[0]:
            error_level = 0  # Aggressive
        elif error_bound >= ERROR_LEVEL_THRESHOLDS[1]:
            error_level = 1  # Balanced
        else:
            error_level = 2  # Precise

        return entropy_bin * NUM_ERROR_LEVELS + error_level

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

    @staticmethod
    def encode_action(algorithm_idx: int, use_quantization: bool, use_shuffle: bool) -> int:
        """
        Encode compression configuration into action index.

        Args:
            algorithm_idx: Algorithm index (0-7)
            use_quantization: Whether to apply quantization
            use_shuffle: Whether to apply byte shuffle

        Returns:
            Action index (0 to NUM_ACTIONS-1)
        """
        return algorithm_idx + (1 if use_quantization else 0) * 8 + (1 if use_shuffle else 0) * 16

    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_values[state, action]

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

    def get_visit_count(self, state: int, action: int) -> int:
        """Get number of times state-action pair was visited."""
        return self.visit_counts[state, action]

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
            'version': 1,
            'n_states': NUM_STATES,
            'n_actions': NUM_ACTIONS,
            'n_entropy_bins': NUM_ENTROPY_BINS,
            'n_error_levels': NUM_ERROR_LEVELS,
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
        """Print best action for each state."""
        print("\nBest Actions per State:")
        print("=" * 70)
        print(f"{'Entropy Bin':<12} {'Error Level':<12} {'Best Action':<30} {'Q-Value':<10}")
        print("-" * 70)

        for state in range(NUM_STATES):
            entropy_bin = state // NUM_ERROR_LEVELS
            error_level = state % NUM_ERROR_LEVELS
            error_names = ['aggressive', 'balanced', 'precise']

            best_action, q_value = self.get_best_action(state)
            action_config = self.decode_action(best_action)

            action_str = f"{action_config['algorithm']}"
            if action_config['quantization']:
                action_str += "+quant"
            if action_config['shuffle_size'] > 0:
                action_str += f"+shuffle{action_config['shuffle_size']}"

            print(f"{entropy_bin:<12} {error_names[error_level]:<12} {action_str:<30} {q_value:<10.4f}")
