"""
Exploration Policies for Q-Table Training

Implements epsilon-greedy policy with decay for balancing
exploration (try new actions) and exploitation (use best known action).
"""

import numpy as np
from typing import Tuple

from .config import EPSILON_START, EPSILON_END, EPSILON_DECAY, NUM_ACTIONS


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy with exponential decay.

    With probability ε: select random action (explore)
    With probability 1-ε: select best action (exploit)

    ε decays exponentially from start to end over training.
    """

    def __init__(
        self,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY
    ):
        """
        Initialize policy.

        Args:
            epsilon_start: Initial exploration rate (typically 1.0)
            epsilon_end: Final exploration rate (typically 0.01)
            epsilon_decay: Multiplicative decay per epoch (typically 0.995)
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epoch = 0

    def select_action(self, q_values: np.ndarray) -> Tuple[int, bool]:
        """
        Select action using epsilon-greedy strategy.

        Args:
            q_values: Array of Q-values for current state (NUM_ACTIONS,)

        Returns:
            Tuple of (action_index, was_exploration)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(NUM_ACTIONS)
            return action, True
        else:
            # Exploit: best action
            action = np.argmax(q_values)
            return int(action), False

    def decay(self):
        """Apply epsilon decay after each epoch."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        self.epoch += 1

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start
        self.epoch = 0

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon

    def get_stats(self) -> dict:
        """Get policy statistics."""
        return {
            'epsilon': self.epsilon,
            'epoch': self.epoch,
            'exploration_pct': 100.0 * self.epsilon
        }


class GreedyPolicy:
    """
    Greedy policy (always exploit, no exploration).
    Used for inference after training.
    """

    def select_action(self, q_values: np.ndarray) -> Tuple[int, bool]:
        """Select best action (argmax of Q-values)."""
        return int(np.argmax(q_values)), False

    def decay(self):
        """No-op for greedy policy."""
        pass

    def get_epsilon(self) -> float:
        """Greedy policy has epsilon = 0."""
        return 0.0
