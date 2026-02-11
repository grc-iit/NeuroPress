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

import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Optional

from .qtable import QTable
from .policy import EpsilonGreedyPolicy
from .reward import compute_reward_from_metrics, get_preset_description
from .executor import CompressionExecutor
from .config import (
    DEFAULT_EPOCHS, CHECKPOINT_INTERVAL,
    REWARD_PRESETS, ERROR_LEVEL_THRESHOLDS, NUM_ERROR_LEVELS,
    NUM_ACTIONS, NUM_ENTROPY_BINS, NUM_STATES,
    NUM_MAD_BINS, NUM_DERIV_BINS
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
        error_bounds: List[float] = None,
        resume_path: str = None,
        use_c_api: bool = False
    ):
        """
        Initialize trainer.

        Args:
            data_dir: Directory containing training data files
            output_dir: Directory to save trained models
            gpu_compress_path: Path to gpu_compress binary
            reward_preset: Reward computation preset
            error_bounds: List of error bounds to train (cycles through all per epoch)
            resume_path: Path to previous Q-table to resume from
            use_c_api: Whether to use C API for entropy calculation
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.reward_preset = reward_preset
        self.error_bounds = error_bounds or [0.001]

        # Initialize components
        self.qtable = QTable()
        self.policy = EpsilonGreedyPolicy()
        gpu_decompress_path = gpu_compress_path.replace('gpu_compress', 'gpu_decompress')
        self.executor = CompressionExecutor(
            gpu_compress_path=gpu_compress_path,
            gpu_decompress_path=gpu_decompress_path,
            use_c_api=use_c_api
        )

        # Resume from previous Q-table if provided
        if resume_path and Path(resume_path).exists():
            self.qtable.load(resume_path)
            print(f"Resumed Q-table from: {resume_path}")

        # Collect training files
        self.training_files = self._collect_files()

        # Training statistics
        self.episode_rewards = []
        self.episode_ratios = []
        self.exploration_counts = 0
        self.exploitation_counts = 0

    def _build_action_mask(self, error_bound: float) -> Optional[np.ndarray]:
        """Build action mask based on error_bound. Masks out quantization actions for lossless."""
        if error_bound <= 0:
            mask = np.ones(NUM_ACTIONS, dtype=bool)
            for a in range(NUM_ACTIONS):
                if (a // 8) % 2 == 1:  # quantization bit is set
                    mask[a] = False
            return mask
        return None

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
        print(f"Error bounds: {self.error_bounds}")
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
                for eb in self.error_bounds:
                    if verbose and i % 10 == 0 and eb == self.error_bounds[0]:
                        print(f"Epoch {epoch+1}/{n_epochs}, File {i+1}/{len(files)}, "
                              f"ε={self.policy.get_epsilon():.3f}")

                    # Run training episode with this error bound
                    reward, ratio = self._train_episode(filepath, eb)

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

    def _train_episode(self, filepath: Path, error_bound: float) -> tuple:
        """
        Run one training episode on a single file.

        Args:
            filepath: Path to training data file
            error_bound: Error bound for this episode

        Returns:
            Tuple of (reward, compression_ratio) or (None, None) on error
        """
        try:
            filename = filepath.name
            print(f"\n  --- Episode: {filename} (eb={error_bound}) ---")

            # Load data and calculate all metrics
            data = np.fromfile(filepath, dtype=np.float32)
            metrics = self.executor.calculate_all_metrics(data)
            entropy = metrics['entropy']
            mad = metrics['mad']
            first_derivative = metrics['first_derivative']
            print(f"  [METRICS] {filename}: entropy={entropy:.4f} bits "
                  f"(bin={min(NUM_ENTROPY_BINS - 1, int(entropy * 2))}), "
                  f"mad={mad:.6f}, deriv={first_derivative:.6f}")

            # Encode state
            state = QTable.encode_state(entropy, error_bound,
                                        mad=mad, first_derivative=first_derivative)
            entropy_bin, error_level, mad_bin, deriv_bin = QTable.decode_state(state)
            error_names = ['aggressive', 'moderate', 'precise', 'lossless']
            entropy_label = f"{entropy_bin * 0.5:.1f}-{(entropy_bin + 1) * 0.5:.1f}"
            print(f"  [STATE] state={state} "
                  f"(entropy={entropy_label}, error={error_names[error_level]}, "
                  f"mad_bin={mad_bin}, deriv_bin={deriv_bin}, "
                  f"error_bound={error_bound})")

            # Select action using policy (mask invalid actions for lossless)
            q_values = self.qtable.q_values[state]
            action_mask = self._build_action_mask(error_bound)
            action, was_exploration = self.policy.select_action(q_values, action_mask)

            if was_exploration:
                self.exploration_counts += 1
            else:
                self.exploitation_counts += 1

            # Decode action
            action_config = QTable.decode_action(action)
            action_str = action_config['algorithm']
            if action_config['quantization']:
                action_str += f"+quant(eb={error_bound})"
            if action_config['shuffle_size'] > 0:
                action_str += f"+shuffle{action_config['shuffle_size']}"

            print(f"  [ACTION] action={action} -> {action_str} "
                  f"({'EXPLORE' if was_exploration else 'EXPLOIT'}, "
                  f"ε={self.policy.get_epsilon():.3f})")
            print(f"  [ACTION DETAIL] algorithm={action_config['algorithm']}, "
                  f"quantization={action_config['quantization']}, "
                  f"shuffle_size={action_config['shuffle_size']}")

            # Execute compression
            metrics = self.executor.execute_action(
                input_file=str(filepath),
                action_config=action_config,
                error_bound=error_bound
            )

            if not metrics['success']:
                print(f"  [RESULT] FAILED: {metrics.get('error', 'unknown error')}")
                return None, None

            # Compute reward
            reward = compute_reward_from_metrics(metrics, self.reward_preset)

            # Update Q-table
            old_q = self.qtable.q_values[state, action]
            td_error = self.qtable.update(state, action, reward)
            new_q = self.qtable.q_values[state, action]

            print(f"  [REWARD] reward={reward:.4f} (preset={self.reward_preset})")
            print(f"  [Q-UPDATE] Q[{state}][{action}]: {old_q:.4f} -> {new_q:.4f} "
                  f"(td_error={td_error:.4f}, lr={self.qtable.learning_rate})")

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
        default=None,
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
        type=str,
        default='0.001',
        help='Error bound: a number (e.g. 0.001) or "all" to train all 4 levels'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to previous Q-table JSON to resume training from'
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
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove all Q-table files from output directory'
    )

    args = parser.parse_args()

    # Handle --clean mode
    if args.clean:
        output_dir = Path(args.output_dir)
        patterns = ['qtable*.json', 'qtable*.bin', 'qtable_report.txt']
        removed = 0
        for pat in patterns:
            for f in output_dir.glob(pat):
                f.unlink()
                print(f"Removed: {f}")
                removed += 1
        print(f"Cleaned {removed} files from {output_dir}")
        return

    # Require --data-dir for training
    if not args.data_dir:
        parser.error('--data-dir is required for training')

    # Parse error bound(s)
    if args.error_bound.lower() == 'all':
        error_bounds = [
            ERROR_LEVEL_THRESHOLDS[0],        # aggressive (0.1)
            ERROR_LEVEL_THRESHOLDS[1],        # moderate   (0.01)
            ERROR_LEVEL_THRESHOLDS[2],        # precise    (0.001)
            0.0,                              # lossless
        ]
    else:
        error_bounds = [float(args.error_bound)]

    # Create trainer
    trainer = QTableTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        gpu_compress_path=args.gpu_compress,
        reward_preset=args.preset,
        error_bounds=error_bounds,
        resume_path=args.resume,
        use_c_api=args.use_c_api
    )

    # Run training
    trainer.train(
        n_epochs=args.epochs,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
