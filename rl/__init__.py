"""
Reinforcement Learning for Compression Algorithm Selection

This package provides Q-Table based RL for automatically selecting
optimal compression configurations based on data characteristics.
"""

from .qtable import QTable
from .policy import EpsilonGreedyPolicy, GreedyPolicy
from .reward import compute_reward, compute_reward_from_metrics
from .executor import CompressionExecutor
from .trainer import QTableTrainer
from .config import (
    NUM_STATES, NUM_ACTIONS, NUM_ENTROPY_BINS, NUM_ERROR_LEVELS,
    NUM_MAD_BINS, NUM_DERIV_BINS,
    MAD_BIN_THRESHOLDS, DERIV_BIN_THRESHOLDS,
    ALGORITHM_NAMES, REWARD_PRESETS
)

__all__ = [
    'QTable',
    'EpsilonGreedyPolicy',
    'GreedyPolicy',
    'compute_reward',
    'compute_reward_from_metrics',
    'CompressionExecutor',
    'QTableTrainer',
    'NUM_STATES',
    'NUM_ACTIONS',
    'NUM_ENTROPY_BINS',
    'NUM_ERROR_LEVELS',
    'NUM_MAD_BINS',
    'NUM_DERIV_BINS',
    'MAD_BIN_THRESHOLDS',
    'DERIV_BIN_THRESHOLDS',
    'ALGORITHM_NAMES',
    'REWARD_PRESETS'
]
