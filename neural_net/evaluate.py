"""
Ranking evaluation for compression performance predictor.

For each file in the validation set:
1. Run all 64 configurations through the model
2. Rank by predicted compression_ratio (and other criteria)
3. Compare predicted ranking to actual ranking
4. Report top-1, top-3 accuracy and regret

Usage:
    python neural_net/evaluate.py --csv benchmark_results--1.csv
    python neural_net/evaluate.py --data-dir syntheticGeneration/training_data/ --lib-path build/libgpucompress.so
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from data import inverse_transform_outputs, ALGORITHM_NAMES
from model import CompressionPredictor


def load_trained_model(weights_path: str, device: torch.device) -> tuple:
    """Load trained model and scalers."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    model = CompressionPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def evaluate_ranking(model, data: Dict, device: torch.device,
                     rank_by: str = 'compression_ratio'):
    """
    Evaluate ranking accuracy on validation set.

    For each (file, error_bound) group:
    - Predict all 64 configs
    - Rank by the specified metric
    - Compare to actual ranking
    """
    df_val = data['df_val']

    # Group by file and error_bound (each group has 64 rows = all configs)
    groups = df_val.groupby(['file', 'error_bound'])

    top1_correct = 0
    top3_correct = 0
    total_groups = 0
    regrets = []

    # For ratio and psnr: higher is better. For time: lower is better.
    higher_is_better = rank_by in ('compression_ratio', 'psnr_db')

    feature_cols = data['feature_names']
    x_means = data['x_means']
    x_stds = data['x_stds']
    y_means = data['y_means']
    y_stds = data['y_stds']

    for (fname, eb), group in groups:
        if len(group) < 8:  # Skip incomplete groups (need at least 8 algos)
            continue

        total_groups += 1

        # ---- Actual best ----
        if higher_is_better:
            actual_best_idx = group[rank_by].idxmax()
        else:
            actual_best_idx = group[rank_by].idxmin()
        actual_best_val = group.loc[actual_best_idx, rank_by]
        actual_best_config = (
            group.loc[actual_best_idx, 'algorithm'],
            group.loc[actual_best_idx, 'quantization'],
            group.loc[actual_best_idx, 'shuffle'],
        )

        # ---- Predicted ranking ----
        # Get features for all rows in this group
        X_raw = group[feature_cols].values.astype(np.float32)
        X_norm = (X_raw - x_means) / x_stds
        X_tensor = torch.from_numpy(X_norm).to(device)

        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()

        # Inverse transform to original scale
        pred_orig = inverse_transform_outputs(pred_norm, y_means, y_stds)
        pred_metric = pred_orig[rank_by]

        # Rank predictions
        if higher_is_better:
            pred_ranking = np.argsort(-pred_metric)  # descending
        else:
            pred_ranking = np.argsort(pred_metric)   # ascending

        # Map back to dataframe indices
        group_indices = group.index.values
        pred_best_df_idx = group_indices[pred_ranking[0]]
        pred_top3_df_idx = set(group_indices[pred_ranking[:3]])

        # ---- Evaluate ----
        # Top-1: is the predicted best the actual best config?
        pred_best_config = (
            group.loc[pred_best_df_idx, 'algorithm'],
            group.loc[pred_best_df_idx, 'quantization'],
            group.loc[pred_best_df_idx, 'shuffle'],
        )

        if pred_best_config == actual_best_config:
            top1_correct += 1

        if actual_best_idx in pred_top3_df_idx:
            top3_correct += 1

        # Regret: how much worse is the predicted best vs actual best?
        pred_best_actual_val = group.loc[pred_best_df_idx, rank_by]
        if higher_is_better:
            regret = actual_best_val - pred_best_actual_val
        else:
            regret = pred_best_actual_val - actual_best_val
        # Handle inf-inf or NaN gracefully
        if np.isfinite(regret):
            regret = max(0.0, regret)
        else:
            regret = 0.0
        regrets.append(regret)

    # ---- Report ----
    print(f"\n{'=' * 65}")
    print(f"RANKING EVALUATION -- rank by: {rank_by}")
    print(f"{'=' * 65}")
    print(f"  Total (file, error_bound) groups: {total_groups}")
    print(f"  Top-1 accuracy: {top1_correct}/{total_groups} = {100*top1_correct/total_groups:.1f}%")
    print(f"  Top-3 accuracy: {top3_correct}/{total_groups} = {100*top3_correct/total_groups:.1f}%")

    regrets = np.array(regrets)
    print(f"  Mean regret:    {regrets.mean():.4f}")
    print(f"  Median regret:  {np.median(regrets):.4f}")
    print(f"  Max regret:     {regrets.max():.4f}")
    print(f"  Zero-regret:    {(regrets == 0).sum()}/{total_groups} "
          f"({100*(regrets == 0).sum()/total_groups:.1f}%)")

    return {
        'rank_by': rank_by,
        'total_groups': total_groups,
        'top1_accuracy': top1_correct / total_groups,
        'top3_accuracy': top3_correct / total_groups,
        'mean_regret': float(regrets.mean()),
        'median_regret': float(np.median(regrets)),
    }


def show_sample_predictions(model, data: Dict, device: torch.device, n_samples: int = 3):
    """Show detailed predictions for a few validation files."""
    df_val = data['df_val']
    feature_cols = data['feature_names']
    x_means = data['x_means']
    x_stds = data['x_stds']
    y_means = data['y_means']
    y_stds = data['y_stds']

    # Pick a few random files
    files = df_val['file'].unique()
    rng = np.random.RandomState(123)
    sample_files = rng.choice(files, size=min(n_samples, len(files)), replace=False)

    for fname in sample_files:
        # Use lossless (error_bound=0) for clarity
        group = df_val[(df_val['file'] == fname) & (df_val['error_bound'] == 0)]
        if len(group) == 0:
            group = df_val[df_val['file'] == fname].head(16)

        entropy = group['entropy'].iloc[0]
        mad = group['mad'].iloc[0]
        deriv = group['second_derivative'].iloc[0]

        print(f"\n{'─' * 75}")
        print(f"File: {fname}")
        print(f"  entropy={entropy:.4f}  mad={mad:.6f}  deriv={deriv:.6f}")
        print(f"  {'Config':<30} {'Actual Ratio':>12} {'Pred Ratio':>12} {'Actual Time':>12} {'Pred Time':>12}")
        print(f"  {'─' * 78}")

        X_raw = group[feature_cols].values.astype(np.float32)
        X_norm = (X_raw - x_means) / x_stds

        with torch.no_grad():
            pred_norm = model(torch.from_numpy(X_norm).to(device)).cpu().numpy()
        pred_orig = inverse_transform_outputs(pred_norm, y_means, y_stds)

        # Sort by actual ratio descending
        indices = np.argsort(-group['compression_ratio'].values)

        for i in indices[:8]:  # Show top 8
            row = group.iloc[i]
            config = f"{row['algorithm']}"
            if row['shuffle'] > 0:
                config += "+shuf"
            if row['quantization'] == 'linear':
                config += f"+quant"

            actual_ratio = row['compression_ratio']
            pred_ratio = pred_orig['compression_ratio'][i]
            actual_time = row['compression_time_ms']
            pred_time = pred_orig['compression_time_ms'][i]

            print(f"  {config:<30} {actual_ratio:>12.2f} {pred_ratio:>12.2f} "
                  f"{actual_time:>10.1f}ms {pred_time:>10.1f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate compression predictor ranking accuracy')
    parser.add_argument('--csv', type=str, nargs='+', default=None,
                        help='One or more CSV files with benchmark results')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing .bin files for benchmarking')
    parser.add_argument('--lib-path', type=str, default=None,
                        help='Path to libgpucompress.so')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max .bin files to process')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model.pt (default: neural_net/weights/model.pt)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights_path = args.weights or str(Path(__file__).parent / 'weights' / 'model.pt')
    if not Path(weights_path).exists():
        print(f"No trained model found at {weights_path}. Run train.py first.")
        sys.exit(1)

    # Load model
    model, checkpoint = load_trained_model(weights_path, device)
    print(f"Loaded model from {weights_path} (epoch {checkpoint['best_epoch']})")

    # Load data
    if args.csv:
        from data import load_from_csv
        data = load_from_csv(args.csv)
    elif args.data_dir:
        from binary_data import load_and_prepare_from_binary
        data = load_and_prepare_from_binary(
            args.data_dir, lib_path=args.lib_path, max_files=args.max_files)
    else:
        parser.error("Provide --csv or --data-dir")

    # Evaluate ranking for different criteria
    for criterion in ['compression_ratio', 'compression_time_ms', 'psnr_db']:
        evaluate_ranking(model, data, device, rank_by=criterion)

    # Show sample predictions
    print("\n\n" + "=" * 65)
    print("SAMPLE PREDICTIONS (lossless, top-8 by actual ratio)")
    print("=" * 65)
    show_sample_predictions(model, data, device, n_samples=3)
