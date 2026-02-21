#!/usr/bin/env python3
"""
Example: Compress a binary file using ALGO_AUTO with neural network selection.

Shows the NN's predicted metrics for all 32 configurations, then compresses
with the winner.

Prerequisites:
  1. Build the library:       cmake --build build
  2. Generate training data:  python3 syntheticGeneration/generator.py batch \
                                  -o syntheticGeneration/training_data \
                                  --mode training -s 128KB --format bin -q
  3. Train the model:         python3 neural_net/train.py \
                                  --data-dir syntheticGeneration/training_data/ \
                                  --lib-path build/libgpucompress.so
  4. Export weights:          python3 neural_net/export_weights.py \
                                  --input neural_net/weights/model.pt \
                                  --output neural_net/weights/model.nnwt

Usage:
  python3 examples/auto_compress.py <input.bin> [--weights path/to/model.nnwt] [--error-bound 0.001]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so we can import the ctypes wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_net.gpucompress_ctypes import GPUCompressLib, ALGO_AUTO, ALGO_NAMES, \
    PREPROC_QUANTIZE, PREPROC_SHUFFLE_2, PREPROC_SHUFFLE_4, PREPROC_SHUFFLE_8

ALGO_NAME_LIST = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']


def decode_preprocessing(flags):
    """Decode preprocessing flags to human-readable strings."""
    parts = []
    if flags & PREPROC_SHUFFLE_2:
        parts.append("shuffle-2")
    elif flags & PREPROC_SHUFFLE_4:
        parts.append("shuffle-4")
    elif flags & PREPROC_SHUFFLE_8:
        parts.append("shuffle-8")
    if flags & PREPROC_QUANTIZE:
        parts.append("quantize")
    return ", ".join(parts) if parts else "none"


def decode_action(action_id):
    """Decode action ID to (algo_name, quant, shuffle)."""
    algo_idx = action_id % 8
    quant = (action_id // 8) % 2
    shuffle = (action_id // 16) % 2
    return ALGO_NAME_LIST[algo_idx], bool(quant), 4 if shuffle else 0


def nn_predict_all(model_pt_path, entropy, mad, deriv, data_size, error_bound):
    """Run NN forward pass in Python for all 32 configs. Returns list of dicts."""
    ckpt = torch.load(model_pt_path, map_location='cpu', weights_only=False)

    x_means = np.array(ckpt['x_means'], dtype=np.float32)
    x_stds = np.array(ckpt['x_stds'], dtype=np.float32)
    y_means = np.array(ckpt['y_means'], dtype=np.float32)
    y_stds = np.array(ckpt['y_stds'], dtype=np.float32)

    sd = ckpt['model_state_dict']
    w1 = sd['net.0.weight'].numpy()  # [128, 15]
    b1 = sd['net.0.bias'].numpy()    # [128]
    w2 = sd['net.2.weight'].numpy()  # [128, 128]
    b2 = sd['net.2.bias'].numpy()    # [128]
    w3 = sd['net.4.weight'].numpy()  # [4, 128]
    b3 = sd['net.4.bias'].numpy()    # [4]

    eb_clipped = max(error_bound, 1e-7)
    ds = max(float(data_size), 1.0)

    results = []
    for action_id in range(32):
        algo_idx = action_id % 8
        quant = (action_id // 8) % 2
        shuffle = (action_id // 16) % 2

        # Build 15-feature input (same as CUDA kernel)
        x = np.zeros(15, dtype=np.float32)
        x[algo_idx] = 1.0          # one-hot algorithm
        x[8] = float(quant)
        x[9] = float(shuffle)
        x[10] = math.log10(eb_clipped)
        x[11] = math.log2(ds)
        x[12] = entropy
        x[13] = mad
        x[14] = deriv

        # Standardize
        x = (x - x_means) / x_stds

        # Forward pass
        h = np.maximum(w1 @ x + b1, 0)   # ReLU
        h = np.maximum(w2 @ h + b2, 0)   # ReLU
        out = w3 @ h + b3                 # Linear

        # De-normalize
        out = out * y_stds + y_means

        # Convert from log-space
        comp_time = math.expm1(out[0])
        decomp_time = math.expm1(out[1])
        ratio = math.expm1(out[2])
        psnr = out[3]

        algo_name = ALGO_NAME_LIST[algo_idx]
        results.append({
            'action': action_id,
            'algorithm': algo_name,
            'quant': bool(quant),
            'shuffle': 4 if shuffle else 0,
            'pred_ratio': ratio,
            'pred_comp_ms': comp_time,
            'pred_decomp_ms': decomp_time,
            'pred_psnr': psnr,
        })

    # Sort by predicted ratio (descending) — same as GPU kernel default
    results.sort(key=lambda r: r['pred_ratio'], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compress a file with NN-based automatic algorithm selection")
    parser.add_argument("input", help="Path to input .bin file (float32 array)")
    parser.add_argument("--model-pt", default="neural_net/weights/model.pt",
                        help="Path to model.pt for prediction table (default: neural_net/weights/model.pt)")
    parser.add_argument("--weights", default="neural_net/weights/model.nnwt",
                        help="Path to .nnwt weights file (default: neural_net/weights/model.nnwt)")
    parser.add_argument("--lib", default="build/libgpucompress.so",
                        help="Path to libgpucompress.so (default: build/libgpucompress.so)")
    parser.add_argument("--error-bound", type=float, default=0.0,
                        help="Error bound for lossy compression (0 = lossless, default: 0)")
    parser.add_argument("--output", default=None,
                        help="Path to write compressed output (default: don't write)")
    args = parser.parse_args()

    lib = GPUCompressLib(args.lib)
    lib.init()
    lib.load_nn(args.weights)

    data = open(args.input, "rb").read()

    # Compute stats on GPU
    entropy, mad, second_deriv = lib.compute_stats(data)

    print(f"Input:      {args.input}")
    print(f"Size:       {len(data):,} bytes")
    print(f"Error bound:{args.error_bound}")
    print()
    print(f"Data stats (GPU-computed):")
    print(f"  Entropy:    {entropy:.4f} bits")
    print(f"  MAD:        {mad:.4f}")
    print(f"  2nd Deriv:  {second_deriv:.4f}")
    print()

    # Run NN forward pass in Python for all 32 configs
    predictions = nn_predict_all(
        args.model_pt, entropy, mad, second_deriv, len(data), args.error_bound)

    print("NN predictions for all 32 configs (ranked by predicted ratio):")
    print(f"{'Rank':>4}  {'Action':>6}  {'Algorithm':>10}  {'Quant':>5}  {'Shuf':>4}  "
          f"{'Ratio':>8}  {'Comp ms':>8}  {'Decomp ms':>9}  {'PSNR':>7}")
    print("-" * 85)
    for i, r in enumerate(predictions):
        marker = " <-- winner" if i == 0 else ""
        print(f"{i+1:>4}  {r['action']:>6}  {r['algorithm']:>10}  "
              f"{'yes' if r['quant'] else 'no':>5}  {r['shuffle']:>4}  "
              f"{r['pred_ratio']:>8.2f}  {r['pred_comp_ms']:>8.3f}  "
              f"{r['pred_decomp_ms']:>9.3f}  {r['pred_psnr']:>7.1f}{marker}")

    # Actually compress with ALGO_AUTO on GPU
    print()
    cfg = lib.make_config(algo=ALGO_AUTO, error_bound=args.error_bound)
    compressed, stats = lib.compress(data, cfg, return_stats=True)

    algo_name = ALGO_NAMES.get(stats.algorithm_used, f"unknown({stats.algorithm_used})")
    preproc = decode_preprocessing(stats.preprocessing_used)

    print(f"Actual compression (GPU ALGO_AUTO):")
    print(f"  NN chose:       {algo_name} + {preproc}")
    print(f"  Compressed:     {len(compressed):,} bytes")
    print(f"  Actual ratio:   {len(data) / len(compressed):.2f}x")
    print(f"  Throughput:     {stats.throughput_mbps:.1f} MB/s")

    if args.output:
        open(args.output, "wb").write(compressed)
        print(f"  Written to:     {args.output}")

    lib.cleanup()


if __name__ == "__main__":
    main()
