"""
Binary data loader with on-the-fly GPU benchmarking.

Reads raw .bin files (float32 arrays), computes stats on GPU via
libgpucompress.so, and benchmarks all 64 compression configs per file.

Replaces the CSV-based pipeline with direct binary-file ingestion.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from gpucompress_ctypes import (
    GPUCompressLib, ALGO_LZ4, ALGO_SNAPPY, ALGO_DEFLATE, ALGO_GDEFLATE,
    ALGO_ZSTD, ALGO_ANS, ALGO_CASCADED, ALGO_BITCOMP, ALGO_NAMES,
    HEADER_SIZE,
)
from data import ALGORITHM_NAMES, CONTINUOUS_FEATURES, OUTPUT_COLUMNS

# ============================================================
# Benchmark configuration space (64 configs)
# ============================================================

ALGORITHMS = [
    ALGO_LZ4, ALGO_SNAPPY, ALGO_DEFLATE, ALGO_GDEFLATE,
    ALGO_ZSTD, ALGO_ANS, ALGO_CASCADED, ALGO_BITCOMP,
]

SHUFFLE_OPTIONS = [0, 4]

# (quantize_bool, error_bound) pairs
QUANT_OPTIONS = [
    (False, 0.0),
    (True, 0.1),
    (True, 0.01),
    (True, 0.001),
]


def _compute_psnr(original: bytes, decompressed: bytes) -> float:
    """Compute PSNR between original and decompressed float32 arrays."""
    orig = np.frombuffer(original, dtype=np.float32)
    decomp = np.frombuffer(decompressed, dtype=np.float32)
    if len(orig) != len(decomp):
        return 0.0
    mse = np.mean((orig.astype(np.float64) - decomp.astype(np.float64)) ** 2)
    if mse == 0.0:
        return 120.0  # Perfect reconstruction
    data_range = float(orig.max()) - float(orig.min())
    if data_range == 0.0:
        return 120.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def load_and_prepare_from_binary(
    data_dir: str,
    lib_path: str = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_files: int = None,
) -> Dict:
    """
    Load .bin files, benchmark all configs on GPU, and prepare training data.

    Returns the same dict format as data.load_and_prepare().
    """
    data_path = Path(data_dir)
    bin_files = sorted(data_path.glob('*.bin'))
    if max_files is not None:
        bin_files = bin_files[:max_files]

    print(f"Found {len(bin_files)} .bin files in {data_dir}")
    if len(bin_files) == 0:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")

    total_configs = len(ALGORITHMS) * len(SHUFFLE_OPTIONS) * len(QUANT_OPTIONS)
    print(f"Benchmarking {total_configs} configs per file "
          f"({len(bin_files) * total_configs} total compressions)")

    rows = []
    t_start = time.time()

    with GPUCompressLib(lib_path) as lib:
        for file_idx, bin_file in enumerate(bin_files):
            raw_data = bin_file.read_bytes()
            original_size = len(raw_data)

            if original_size == 0 or original_size % 4 != 0:
                print(f"  Skipping {bin_file.name}: invalid size {original_size}")
                continue

            # Compute stats once per file on GPU
            entropy, mad_val, deriv_val = lib.compute_stats(raw_data)

            print(f"\n  File: {bin_file.name} ({original_size:,} bytes)")
            print(f"    Stats: entropy={entropy:.4f}  MAD={mad_val:.4f}  "
                  f"deriv={deriv_val:.4f}")
            print(f"    {'Algorithm':>10}  {'Quant':>5}  {'Shuf':>4}  "
                  f"{'ErrBound':>8}  {'Ratio':>8}  {'Comp ms':>8}  "
                  f"{'Decomp ms':>9}  {'PSNR':>7}  {'Status':>6}")
            print(f"    {'-' * 82}")

            # Benchmark all 64 configs
            file_config_idx = 0
            for algo in ALGORITHMS:
                algo_name = ALGO_NAMES[algo]
                for shuffle in SHUFFLE_OPTIONS:
                    for quantize, error_bound in QUANT_OPTIONS:
                        cfg = lib.make_config(
                            algo=algo, shuffle=shuffle,
                            quantize=quantize, error_bound=error_bound)

                        file_config_idx += 1
                        try:
                            # Compress
                            t0 = time.perf_counter()
                            compressed = lib.compress(raw_data, cfg)
                            t1 = time.perf_counter()
                            comp_time_ms = (t1 - t0) * 1000.0

                            compressed_size = len(compressed) - HEADER_SIZE
                            ratio = original_size / compressed_size if compressed_size > 0 else 1.0

                            # Decompress
                            t2 = time.perf_counter()
                            decompressed = lib.decompress(compressed, original_size)
                            t3 = time.perf_counter()
                            decomp_time_ms = (t3 - t2) * 1000.0

                            # PSNR
                            if quantize and error_bound > 0:
                                psnr = _compute_psnr(raw_data, decompressed)
                            else:
                                psnr = 120.0  # Lossless

                            print(f"    {algo_name:>10}  "
                                  f"{'yes' if quantize else 'no':>5}  "
                                  f"{shuffle:>4}  {error_bound:>8.4f}  "
                                  f"{ratio:>8.2f}  {comp_time_ms:>8.3f}  "
                                  f"{decomp_time_ms:>9.3f}  {psnr:>7.1f}  "
                                  f"{'OK':>6}")

                            rows.append({
                                'file': bin_file.name,
                                'algorithm': algo_name,
                                'quantization': 'linear' if quantize else 'none',
                                'shuffle': shuffle,
                                'error_bound': error_bound,
                                'original_size': original_size,
                                'entropy': entropy,
                                'mad': mad_val,
                                'first_derivative': deriv_val,
                                'compression_ratio': ratio,
                                'compression_time_ms': comp_time_ms,
                                'decompression_time_ms': decomp_time_ms,
                                'psnr_db': psnr,
                                'success': True,
                            })

                        except RuntimeError as e:
                            print(f"    {algo_name:>10}  "
                                  f"{'yes' if quantize else 'no':>5}  "
                                  f"{shuffle:>4}  {error_bound:>8.4f}  "
                                  f"{'---':>8}  {'---':>8}  "
                                  f"{'---':>9}  {'---':>7}  "
                                  f"{'FAIL':>6}")

                            # Compression/decompression failed for this config
                            rows.append({
                                'file': bin_file.name,
                                'algorithm': algo_name,
                                'quantization': 'linear' if quantize else 'none',
                                'shuffle': shuffle,
                                'error_bound': error_bound,
                                'original_size': original_size,
                                'entropy': entropy,
                                'mad': mad_val,
                                'first_derivative': deriv_val,
                                'compression_ratio': 1.0,
                                'compression_time_ms': 0.0,
                                'decompression_time_ms': 0.0,
                                'psnr_db': 0.0,
                                'success': False,
                            })

            if (file_idx + 1) % 10 == 0 or file_idx == 0:
                elapsed = time.time() - t_start
                rate = (file_idx + 1) / elapsed
                eta = (len(bin_files) - file_idx - 1) / rate if rate > 0 else 0
                print(f"  [{file_idx+1}/{len(bin_files)}] "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t_start
    print(f"\nBenchmarking completed in {elapsed:.1f}s ({len(rows)} rows)")

    # Build DataFrame and apply same encoding as data.py
    df = pd.DataFrame(rows)
    return _encode_and_split(df, val_fraction, seed)


def _encode_and_split(df: pd.DataFrame, val_fraction: float,
                      seed: int) -> Dict:
    """Apply the same feature encoding and split logic as data.load_and_prepare()."""

    # Filter failures
    df = df[df['success'] == True].copy()
    print(f"  After filtering failures: {len(df)} rows")

    # Algorithm: one-hot
    for alg in ALGORITHM_NAMES:
        df[f'alg_{alg}'] = (df['algorithm'] == alg).astype(np.float32)

    # Quantization: binary
    df['quant_enc'] = (df['quantization'] == 'linear').astype(np.float32)

    # Shuffle: binary
    df['shuffle_enc'] = (df['shuffle'] > 0).astype(np.float32)

    # Error bound: log-scale
    df['error_bound_enc'] = np.log10(
        df['error_bound'].clip(lower=1e-7)).astype(np.float32)

    # Data size: log2
    df['data_size_enc'] = np.log2(
        df['original_size'].clip(lower=1)).astype(np.float32)

    # Output transforms
    df['comp_time_log'] = np.log1p(
        df['compression_time_ms'].clip(lower=0)).astype(np.float32)
    df['decomp_time_log'] = np.log1p(
        df['decompression_time_ms'].clip(lower=0)).astype(np.float32)
    df['ratio_log'] = np.log1p(
        df['compression_ratio'].clip(lower=0)).astype(np.float32)
    df['psnr_clamped'] = df['psnr_db'].replace(
        [np.inf, -np.inf], 120.0).clip(upper=120.0).astype(np.float32)

    # Split by file
    files = sorted(df['file'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(files)

    split_idx = int(len(files) * (1 - val_fraction))
    train_files = set(files[:split_idx])
    val_files = set(files[split_idx:])

    df_train = df[df['file'].isin(train_files)].copy()
    df_val = df[df['file'].isin(val_files)].copy()

    print(f"  Train: {len(df_train)} rows ({len(train_files)} files)")
    print(f"  Val:   {len(df_val)} rows ({len(val_files)} files)")

    # Build feature matrix
    algo_cols = [f'alg_{a}' for a in ALGORITHM_NAMES]
    feature_cols = algo_cols + ['quant_enc', 'shuffle_enc',
                                'error_bound_enc', 'data_size_enc',
                                'entropy', 'mad', 'first_derivative']

    output_cols = ['comp_time_log', 'decomp_time_log', 'ratio_log', 'psnr_clamped']

    # Normalization (same logic as data.py)
    train_X_raw = df_train[feature_cols].values.astype(np.float32)
    val_X_raw = df_val[feature_cols].values.astype(np.float32)

    continuous_indices = [feature_cols.index(c) for c in CONTINUOUS_FEATURES]
    means = np.zeros(len(feature_cols), dtype=np.float32)
    stds = np.ones(len(feature_cols), dtype=np.float32)

    for idx in continuous_indices:
        means[idx] = train_X_raw[:, idx].mean()
        stds[idx] = train_X_raw[:, idx].std()
        if stds[idx] < 1e-8:
            stds[idx] = 1.0

    x_mins = train_X_raw.min(axis=0).astype(np.float32)
    x_maxs = train_X_raw.max(axis=0).astype(np.float32)

    train_X = (train_X_raw - means) / stds
    val_X = (val_X_raw - means) / stds

    train_Y_raw = df_train[output_cols].values.astype(np.float32)
    val_Y_raw = df_val[output_cols].values.astype(np.float32)

    y_means = train_Y_raw.mean(axis=0)
    y_stds = train_Y_raw.std(axis=0)
    y_stds[y_stds < 1e-8] = 1.0

    train_Y = (train_Y_raw - y_means) / y_stds
    val_Y = (val_Y_raw - y_means) / y_stds

    print(f"\n  Feature matrix: {train_X.shape[1]} input features")
    print(f"  Features: {feature_cols}")
    print(f"  Outputs: {output_cols}")

    return {
        'train_X': train_X,
        'train_Y': train_Y,
        'val_X': val_X,
        'val_Y': val_Y,
        'val_Y_raw': val_Y_raw,
        'feature_names': feature_cols,
        'output_names': OUTPUT_COLUMNS,
        'output_cols_internal': output_cols,
        'x_means': means,
        'x_stds': stds,
        'x_mins': x_mins,
        'x_maxs': x_maxs,
        'y_means': y_means,
        'y_stds': y_stds,
        'df_val': df_val,
        'df_train': df_train,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Binary data loader test')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing .bin files')
    parser.add_argument('--lib-path', default=None,
                        help='Path to libgpucompress.so')
    parser.add_argument('--max-files', type=int, default=5,
                        help='Max files to process (for testing)')
    args = parser.parse_args()

    data = load_and_prepare_from_binary(
        args.data_dir, lib_path=args.lib_path, max_files=args.max_files)
    print(f"\nTrain X shape: {data['train_X'].shape}")
    print(f"Train Y shape: {data['train_Y'].shape}")
    print(f"Val X shape:   {data['val_X'].shape}")
    print(f"Val Y shape:   {data['val_Y'].shape}")
