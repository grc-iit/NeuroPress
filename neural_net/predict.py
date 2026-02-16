"""
Predict compression performance for a binary file using the trained NN.

No GPU required — computes data stats on CPU and runs all 64 compression
configs through the model to find the best one.

Usage:
    python neural_net/predict.py --bin-file some_data.bin
    python neural_net/predict.py --bin-file some_data.bin --rank-by psnr_db
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import compute_stats_cpu, ALGORITHM_NAMES, inverse_transform_outputs
from model import CompressionPredictor

# All 64 configs: 8 algorithms × 2 shuffle × 4 quantization
SHUFFLE_OPTIONS = [0, 4]
QUANT_OPTIONS = [
    ('none', 0.0),
    ('linear', 0.1),
    ('linear', 0.01),
    ('linear', 0.001),
]


def predict_all_configs(bin_path, weights_path, rank_by='compression_ratio'):
    """Predict performance of all 64 configs for a binary file."""

    # Load binary file
    raw_bytes = Path(bin_path).read_bytes()
    original_size = len(raw_bytes)
    if original_size == 0 or original_size % 4 != 0:
        print(f"Error: invalid file size {original_size} (must be non-zero, multiple of 4)")
        sys.exit(1)

    print(f"File: {bin_path} ({original_size:,} bytes)")

    # Compute stats on CPU
    print("Computing data statistics on CPU...")
    entropy, mad, first_derivative = compute_stats_cpu(raw_bytes)
    print(f"  entropy={entropy:.4f}  mad={mad:.6f}  first_derivative={first_derivative:.6f}")

    # Load model
    device = torch.device('cpu')
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = CompressionPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    x_means = checkpoint['x_means']
    x_stds = checkpoint['x_stds']
    y_means = checkpoint['y_means']
    y_stds = checkpoint['y_stds']
    feature_names = checkpoint['feature_names']

    # Build input rows for all 64 configs
    rows = []
    configs = []
    for algo in ALGORITHM_NAMES:
        for shuffle in SHUFFLE_OPTIONS:
            for quant, error_bound in QUANT_OPTIONS:
                # One-hot algorithm
                algo_features = [1.0 if algo == a else 0.0 for a in ALGORITHM_NAMES]
                quant_enc = 1.0 if quant == 'linear' else 0.0
                shuffle_enc = 1.0 if shuffle > 0 else 0.0
                error_bound_enc = np.log10(max(error_bound, 1e-7))
                data_size_enc = np.log2(max(original_size, 1))

                feature_vec = algo_features + [
                    quant_enc, shuffle_enc, error_bound_enc,
                    data_size_enc, entropy, mad, first_derivative
                ]
                rows.append(feature_vec)
                configs.append((algo, quant, shuffle, error_bound))

    X_raw = np.array(rows, dtype=np.float32)
    X_norm = (X_raw - x_means) / x_stds

    # Run inference
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_norm)).numpy()

    pred_orig = inverse_transform_outputs(pred_norm, y_means, y_stds)

    # Rank results
    higher_is_better = rank_by in ('compression_ratio', 'psnr_db')
    metric_values = pred_orig[rank_by]
    if higher_is_better:
        ranking = np.argsort(-metric_values)
    else:
        ranking = np.argsort(metric_values)

    # Print results
    print(f"\nAll 64 configs ranked by {rank_by} ({'higher' if higher_is_better else 'lower'} is better):\n")
    print(f"  {'Rank':>4}  {'Algorithm':<12} {'Quant':<8} {'Shuf':>4}  {'ErrBound':>8}  "
          f"{'Ratio':>8}  {'PSNR':>8}  {'Comp ms':>8}  {'Decomp ms':>9}")
    print(f"  {'-' * 90}")

    for rank, idx in enumerate(ranking):
        algo, quant, shuffle, eb = configs[idx]
        ratio = pred_orig['compression_ratio'][idx]
        psnr = pred_orig['psnr_db'][idx]
        comp_t = pred_orig['compression_time_ms'][idx]
        decomp_t = pred_orig['decompression_time_ms'][idx]

        marker = " <-- best" if rank == 0 else ""
        print(f"  {rank+1:>4}  {algo:<12} {quant:<8} {shuffle:>4}  {eb:>8.4f}  "
              f"{ratio:>8.2f}  {psnr:>8.1f}  {comp_t:>8.2f}  {decomp_t:>9.2f}{marker}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict compression performance for a binary file (CPU only)')
    parser.add_argument('--bin-file', type=str, required=True,
                        help='Path to a .bin file (raw float32 array)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model.pt (default: neural_net/weights/model.pt)')
    parser.add_argument('--rank-by', type=str, default='compression_ratio',
                        choices=['compression_ratio', 'compression_time_ms',
                                 'decompression_time_ms', 'psnr_db'],
                        help='Metric to rank configs by (default: compression_ratio)')
    args = parser.parse_args()

    weights_path = args.weights or str(Path(__file__).parent / 'weights' / 'model.pt')
    if not Path(weights_path).exists():
        print(f"No trained model at {weights_path}. Run train.py first.")
        sys.exit(1)

    predict_all_configs(args.bin_file, weights_path, rank_by=args.rank_by)
