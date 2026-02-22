"""
Retrain the compression predictor using original benchmark data + experience data.

Usage:
    python neural_net/retrain.py \
        --data-dir syntheticGeneration/training_data/ \
        --lib-path build/libgpucompress.so \
        --experience experiences.csv \
        --output neural_net/weights/model.nnwt

The experience CSV is produced by the C++ active learning system (Level 1/2).
Its columns are:
    entropy, mad, second_derivative, original_size, error_bound,
    algorithm, quantization, shuffle, compression_ratio, compression_time_ms,
    decompression_time_ms, psnr_db

This script:
1. Benchmarks original .bin files on GPU (no CSV needed)
2. Loads experience CSV(s) from active learning
3. Merges into a single DataFrame
4. Encodes, normalizes, and trains from scratch
5. Exports new .nnwt weights
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neural_net.core.data import encode_and_split, ALGORITHM_NAMES, OUTPUT_COLUMNS
from neural_net.training.train import train_model_with_data
from neural_net.export.export_weights import export_weights


def prepare_experience_data(experience_paths: list, original_df: pd.DataFrame) -> pd.DataFrame:
    """Load experience CSV(s) and convert to benchmark DataFrame format."""

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
    result['second_derivative'] = exp['second_derivative']
    result['original_size'] = exp['original_size']
    result['error_bound'] = exp['error_bound']
    result['algorithm'] = exp['algorithm']
    result['quantization'] = exp['quantization']
    result['shuffle'] = exp['shuffle']
    result['compression_ratio'] = exp['compression_ratio']
    result['compression_time_ms'] = exp['compression_time_ms']

    # Fill columns not available from experience buffer
    result['success'] = True
    result['file'] = ['experience_' + str(i) for i in range(len(result))]

    # Decompression time: use values from experience CSV, fill zeros with median
    result['decompression_time_ms'] = exp['decompression_time_ms']
    decomp_mask = result['decompression_time_ms'] <= 0
    result.loc[decomp_mask, 'decompression_time_ms'] = original_df['decompression_time_ms'].median()

    # PSNR: use values from experience CSV, fill zeros with median/lossless defaults
    result['psnr_db'] = exp['psnr_db']
    psnr_mask = result['psnr_db'] <= 0
    lossless = result['error_bound'] <= 0
    result.loc[psnr_mask & lossless, 'psnr_db'] = 120.0
    result.loc[psnr_mask & ~lossless, 'psnr_db'] = original_df['psnr_db'].replace([np.inf], 120.0).median()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Retrain compression predictor with experience data')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing .bin files for original benchmarking')
    parser.add_argument('--lib-path', type=str, default=None,
                        help='Path to libgpucompress.so')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max .bin files to process')
    parser.add_argument('--experience', required=True, nargs='+',
                        help='Path(s) to experience CSV file(s) from active learning')
    parser.add_argument('--output', default=None,
                        help='Output .nnwt path (default: neural_net/weights/model.nnwt)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = str(Path(__file__).parent.parent / 'weights' / 'model.nnwt')

    # Step 1: Benchmark original binary files
    print("=" * 65)
    print("BENCHMARKING ORIGINAL DATA")
    print("=" * 65)
    from neural_net.training.benchmark import benchmark_binary_files
    original_df = benchmark_binary_files(
        args.data_dir, lib_path=args.lib_path, max_files=args.max_files)
    print(f"  Original: {len(original_df)} rows")

    # Step 2: Load experience data
    print(f"\n{'=' * 65}")
    print("LOADING EXPERIENCE DATA")
    print("=" * 65)
    experience_df = prepare_experience_data(args.experience, original_df)
    print(f"  Experience: {len(experience_df)} rows")

    # Step 3: Combine and encode
    if len(experience_df) > 0:
        combined = pd.concat([original_df, experience_df], ignore_index=True)
        print(f"\n  Combined: {len(combined)} rows")
    else:
        print("  No experience data, training on original only")
        combined = original_df

    data = encode_and_split(combined)

    # Step 4: Train model
    print("\n" + "=" * 65)
    print("TRAINING")
    print("=" * 65)
    model, data = train_model_with_data(data, epochs=args.epochs,
                                         patience=args.patience)

    # Step 5: Export weights
    print("\n" + "=" * 65)
    print("EXPORTING WEIGHTS")
    print("=" * 65)
    model_pt = str(Path(__file__).parent.parent / 'weights' / 'model.pt')
    export_weights(model_pt, args.output)

    print(f"\n{'=' * 65}")
    print(f"RETRAIN COMPLETE")
    print(f"  New weights: {args.output}")
    print(f"  Use gpucompress_reload_nn(\"{args.output}\") to hot-load")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
