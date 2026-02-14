"""
Retrain the compression predictor using original benchmark data + experience data.

Usage:
    python neural_net/retrain.py \
        --original benchmark_results.csv \
        --experience experiences.csv \
        --output neural_net/weights/model.nnwt

The experience CSV is produced by the active learning system (Level 1/2).
Its columns are:
    entropy, mad, first_derivative, original_size, error_bound,
    algorithm, quantization, shuffle, compression_ratio, compression_time_ms

This script:
1. Loads original benchmark CSV
2. Loads experience CSV(s)
3. Converts experience rows to match benchmark format
4. Combines and retrains the model from scratch
5. Exports new .nnwt weights
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure the neural_net directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from data import load_and_prepare, ALGORITHM_NAMES, OUTPUT_COLUMNS
from train import train_model
from export_weights import export_weights


def prepare_experience_data(experience_paths: list) -> pd.DataFrame:
    """Load and convert experience CSV(s) to benchmark format."""

    frames = []
    for path in experience_paths:
        df = pd.read_csv(path)
        frames.append(df)
        print(f"  Loaded {len(df)} experience rows from {path}")

    if not frames:
        return pd.DataFrame()

    exp = pd.concat(frames, ignore_index=True)

    # Map experience columns to benchmark columns
    result = pd.DataFrame()
    result['entropy'] = exp['entropy']
    result['mad'] = exp['mad']
    result['first_derivative'] = exp['first_derivative']
    result['original_size'] = exp['original_size']
    result['error_bound'] = exp['error_bound']
    result['algorithm'] = exp['algorithm']
    result['quantization'] = exp['quantization']
    result['shuffle'] = exp['shuffle']
    result['compression_ratio'] = exp['compression_ratio']
    result['compression_time_ms'] = exp['compression_time_ms']

    # Add columns expected by data.py but not available from experience
    result['success'] = True
    result['file'] = ['experience_' + str(i) for i in range(len(result))]
    result['decompression_time_ms'] = np.nan
    result['psnr_db'] = np.nan

    return result


def combine_data(original_path: str, experience_paths: list) -> str:
    """Combine original benchmark data with experience data into a temp CSV."""

    print(f"\nLoading original data from {original_path}...")
    original = pd.read_csv(original_path)
    print(f"  Original: {len(original)} rows")

    print(f"\nLoading experience data...")
    experience = prepare_experience_data(experience_paths)
    print(f"  Experience: {len(experience)} rows")

    if len(experience) == 0:
        print("  No experience data found, training on original only")
        return original_path

    # Combine
    combined = pd.concat([original, experience], ignore_index=True)

    # Handle NaN outputs from experience rows:
    # Fill decompression_time_ms with median from original data
    decomp_median = original['decompression_time_ms'].median()
    combined['decompression_time_ms'] = combined['decompression_time_ms'].fillna(decomp_median)

    # Fill psnr_db: for lossless (error_bound <= 0), use 120 (inf equivalent)
    # For lossy, use median from original
    psnr_median = original['psnr_db'].replace([np.inf], 120.0).median()
    lossless_mask = combined['error_bound'] <= 0
    combined.loc[lossless_mask & combined['psnr_db'].isna(), 'psnr_db'] = 120.0
    combined['psnr_db'] = combined['psnr_db'].fillna(psnr_median)

    # Write combined CSV to temp file
    combined_path = str(Path(original_path).parent / 'combined_retrain.csv')
    combined.to_csv(combined_path, index=False)
    print(f"\n  Combined: {len(combined)} rows -> {combined_path}")

    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description='Retrain compression predictor with experience data')
    parser.add_argument('--original', required=True,
                        help='Path to original benchmark_results.csv')
    parser.add_argument('--experience', required=True, nargs='+',
                        help='Path(s) to experience CSV file(s)')
    parser.add_argument('--output', default=None,
                        help='Output .nnwt path (default: neural_net/weights/model.nnwt)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = str(Path(__file__).parent / 'weights' / 'model.nnwt')

    # Step 1: Combine data
    combined_csv = combine_data(args.original, args.experience)

    # Step 2: Train model
    print("\n" + "=" * 65)
    print("TRAINING")
    print("=" * 65)
    model, data = train_model(combined_csv, epochs=args.epochs,
                               patience=args.patience)

    # Step 3: Export weights
    print("\n" + "=" * 65)
    print("EXPORTING WEIGHTS")
    print("=" * 65)
    model_pt = str(Path(__file__).parent / 'weights' / 'model.pt')
    export_weights(model_pt, args.output)

    print(f"\n{'=' * 65}")
    print(f"RETRAIN COMPLETE")
    print(f"  New weights: {args.output}")
    print(f"  Use gpucompress_reload_nn(\"{args.output}\") to hot-load")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
