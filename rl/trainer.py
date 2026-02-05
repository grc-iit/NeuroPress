"""
Q-Table Trainer for Compression Algorithm Selection

Main training loop that:
1. Loads training data files
2. Calculates entropy for each file
3. Selects actions using epsilon-greedy policy
4. Executes compression and measures metrics
5. Updates Q-table based on rewards
6. Saves trained model
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .qtable import QTable
from .policy import EpsilonGreedyPolicy
from .reward import compute_reward_from_metrics, get_preset_description
from .executor import CompressionExecutor
from .config import (
    DEFAULT_EPOCHS, CHECKPOINT_INTERVAL, VALIDATION_SPLIT,
    REWARD_PRESETS, ERROR_LEVEL_THRESHOLDS
)


class QTableTrainer:
    """
    Trains Q-Table for compression algorithm selection.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = 'rl/models',
        gpu_compress_path: str = './build/gpu_compress',
        reward_preset: str = 'balanced',
        error_bound: float = 0.001,
        use_c_api: bool = False
    ):
        """
        Initialize trainer.

        Args:
            data_dir: Directory containing training data files
            output_dir: Directory to save trained models
            gpu_compress_path: Path to gpu_compress binary
            reward_preset: Reward computation preset
            error_bound: Error bound for quantization experiments
            use_c_api: Whether to use C API for entropy calculation
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.reward_preset = reward_preset
        self.error_bound = error_bound

        # Initialize components
        self.qtable = QTable()
        self.policy = EpsilonGreedyPolicy()
        self.executor = CompressionExecutor(
            gpu_compress_path=gpu_compress_path,
            use_c_api=use_c_api
        )

        # Collect training files
        self.training_files = self._collect_files()

        # Training statistics
        self.episode_rewards = []
        self.episode_ratios = []
        self.exploration_counts = 0
        self.exploitation_counts = 0

    def _collect_files(self) -> List[Path]:
        """Collect all binary training files from data directory."""
        files = []
        for ext in ['*.bin', '*.raw', '*.dat']:
            files.extend(self.data_dir.glob(f'**/{ext}'))

        if not files:
            print(f"Warning: No training files found in {self.data_dir}")
            print("Looking for .bin, .raw, .dat files")

        return sorted(files)

    def train(
        self,
        n_epochs: int = DEFAULT_EPOCHS,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        verbose: bool = True
    ):
        """
        Run training loop.

        Args:
            n_epochs: Number of training epochs
            checkpoint_interval: Save checkpoint every N epochs
            verbose: Print progress
        """
        if not self.training_files:
            print("Error: No training files found!")
            return

        print(f"\n{'='*60}")
        print("Q-Table Training for Compression Algorithm Selection")
        print(f"{'='*60}")
        print(f"Training files: {len(self.training_files)}")
        print(f"Epochs: {n_epochs}")
        print(f"Reward preset: {self.reward_preset}")
        print(f"  → {get_preset_description(self.reward_preset)}")
        print(f"Error bound: {self.error_bound}")
        print(f"{'='*60}\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(n_epochs):
            epoch_rewards = []
            epoch_ratios = []

            # Shuffle training files each epoch
            files = self.training_files.copy()
            random.shuffle(files)

            for i, filepath in enumerate(files):
                if verbose and i % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, File {i+1}/{len(files)}, "
                          f"ε={self.policy.get_epsilon():.3f}")

                # Run training episode
                reward, ratio = self._train_episode(filepath)

                if reward is not None:
                    epoch_rewards.append(reward)
                    epoch_ratios.append(ratio)

            # Decay epsilon
            self.policy.decay()

            # Record epoch statistics
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                avg_ratio = np.mean(epoch_ratios)
                self.episode_rewards.append(avg_reward)
                self.episode_ratios.append(avg_ratio)

                if verbose:
                    print(f"Epoch {epoch+1}: avg_reward={avg_reward:.4f}, "
                          f"avg_ratio={avg_ratio:.2f}x, ε={self.policy.get_epsilon():.3f}")

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)

        # Save final model
        self._save_final()

        # Print summary
        self._print_summary()

    def _train_episode(self, filepath: Path) -> tuple:
        """
        Run one training episode on a single file.

        Returns:
            Tuple of (reward, compression_ratio) or (None, None) on error
        """
        try:
            # Load data and calculate entropy
            data = np.fromfile(filepath, dtype=np.float32)
            entropy = self.executor.calculate_entropy(data)

            # Encode state
            state = QTable.encode_state(entropy, self.error_bound)

            # Select action using policy
            q_values = self.qtable.q_values[state]
            action, was_exploration = self.policy.select_action(q_values)

            if was_exploration:
                self.exploration_counts += 1
            else:
                self.exploitation_counts += 1

            # Decode action
            action_config = QTable.decode_action(action)

            # Execute compression
            metrics = self.executor.execute_action(
                input_file=str(filepath),
                action_config=action_config,
                error_bound=self.error_bound
            )

            if not metrics['success']:
                return None, None

            # Compute reward
            reward = compute_reward_from_metrics(metrics, self.reward_preset)

            # Update Q-table
            self.qtable.update(state, action, reward)

            return reward, metrics['ratio']

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None, None

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f'qtable_epoch{epoch}.json'
        self.qtable.save(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")

    def _save_final(self):
        """Save final trained model."""
        # Save JSON
        json_path = self.output_dir / 'qtable.json'
        self.qtable.save(str(json_path))
        print(f"Saved final model: {json_path}")

        # Save binary for GPU loading
        bin_path = self.output_dir / 'qtable.bin'
        self.qtable.export_binary(str(bin_path))
        print(f"Saved binary model: {bin_path}")

    def _print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")

        # Coverage statistics
        coverage = self.qtable.get_state_coverage()
        print(f"\nState Coverage:")
        print(f"  States visited: {coverage['states_visited']}/{coverage['states_total']} "
              f"({coverage['coverage_pct']:.1f}%)")
        print(f"  Total visits: {coverage['total_visits']}")
        print(f"  Avg actions per state: {coverage['avg_actions_per_state']:.1f}")

        # Exploration statistics
        total = self.exploration_counts + self.exploitation_counts
        if total > 0:
            print(f"\nExploration Statistics:")
            print(f"  Exploration: {self.exploration_counts} ({100*self.exploration_counts/total:.1f}%)")
            print(f"  Exploitation: {self.exploitation_counts} ({100*self.exploitation_counts/total:.1f}%)")

        # Best actions
        print("\n")
        self.qtable.print_best_actions()


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description='Train Q-Table for compression algorithm selection'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Directory containing training data files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='rl/models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )
    parser.add_argument(
        '--preset', '-p',
        type=str,
        default='balanced',
        choices=list(REWARD_PRESETS.keys()),
        help='Reward preset (default: balanced)'
    )
    parser.add_argument(
        '--error-bound',
        type=float,
        default=0.001,
        help='Error bound for quantization (default: 0.001)'
    )
    parser.add_argument(
        '--gpu-compress',
        type=str,
        default='./build/gpu_compress',
        help='Path to gpu_compress binary'
    )
    parser.add_argument(
        '--use-c-api',
        action='store_true',
        help='Use C API for entropy calculation (faster)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = QTableTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        gpu_compress_path=args.gpu_compress,
        reward_preset=args.preset,
        error_bound=args.error_bound,
        use_c_api=args.use_c_api
    )

    # Run training
    trainer.train(
        n_epochs=args.epochs,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
