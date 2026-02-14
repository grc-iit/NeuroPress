"""
Data pipeline for compression performance prediction.

Loads benchmark_results.csv, encodes features, normalizes,
and splits into train/validation sets by file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

# ============================================================
# Constants
# ============================================================

ALGORITHM_NAMES = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']
NUM_ALGORITHMS = len(ALGORITHM_NAMES)

INPUT_COLUMNS = ['algorithm', 'quantization', 'shuffle', 'error_bound',
                 'original_size', 'entropy', 'mad', 'first_derivative']

OUTPUT_COLUMNS = ['compression_time_ms', 'decompression_time_ms',
                  'compression_ratio', 'psnr_db']

CONTINUOUS_FEATURES = ['error_bound_enc', 'data_size_enc', 'entropy', 'mad', 'first_derivative']


def load_and_prepare(csv_path: str, val_fraction: float = 0.2,
                     seed: int = 42) -> Dict:
    """
    Load CSV, encode features, normalize, and split by file.

    Returns dict with:
        train_X, train_Y, val_X, val_Y: numpy arrays
        feature_names: list of feature names
        output_names: list of output names
        scalers: dict with normalization parameters for inference
        df_val: validation dataframe (for ranking evaluation)
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {df['file'].nunique()} unique files")

    # ---- Filter ----
    df = df[df['success'] == True].copy()
    print(f"  After filtering failures: {len(df)} rows")

    # ---- Encode inputs ----

    # Algorithm: one-hot
    for alg in ALGORITHM_NAMES:
        df[f'alg_{alg}'] = (df['algorithm'] == alg).astype(np.float32)

    # Quantization: binary
    df['quant_enc'] = (df['quantization'] == 'linear').astype(np.float32)

    # Shuffle: binary
    df['shuffle_enc'] = (df['shuffle'] > 0).astype(np.float32)

    # Error bound: log-scale (handle 0 for lossless)
    df['error_bound_enc'] = np.log10(df['error_bound'].clip(lower=1e-7)).astype(np.float32)

    # Data size: log2
    df['data_size_enc'] = np.log2(df['original_size'].clip(lower=1)).astype(np.float32)

    # ---- Encode outputs ----

    # Compression time: log1p
    df['comp_time_log'] = np.log1p(df['compression_time_ms'].clip(lower=0)).astype(np.float32)

    # Decompression time: log1p
    df['decomp_time_log'] = np.log1p(df['decompression_time_ms'].clip(lower=0)).astype(np.float32)

    # Compression ratio: log1p
    df['ratio_log'] = np.log1p(df['compression_ratio'].clip(lower=0)).astype(np.float32)

    # PSNR: clamp inf to 120, keep as-is
    df['psnr_clamped'] = df['psnr_db'].replace([np.inf, -np.inf], 120.0).clip(upper=120.0).astype(np.float32)

    # ---- Split by file ----
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

    # ---- Build feature matrix ----
    algo_cols = [f'alg_{a}' for a in ALGORITHM_NAMES]
    feature_cols = algo_cols + ['quant_enc', 'shuffle_enc',
                                'error_bound_enc', 'data_size_enc',
                                'entropy', 'mad', 'first_derivative']

    output_cols = ['comp_time_log', 'decomp_time_log', 'ratio_log', 'psnr_clamped']

    # ---- Compute normalization stats from training set only ----
    train_X_raw = df_train[feature_cols].values.astype(np.float32)
    val_X_raw = df_val[feature_cols].values.astype(np.float32)

    # Only standardize continuous features (not one-hot or binary)
    continuous_indices = [feature_cols.index(c) for c in CONTINUOUS_FEATURES]
    means = np.zeros(len(feature_cols), dtype=np.float32)
    stds = np.ones(len(feature_cols), dtype=np.float32)

    for idx in continuous_indices:
        means[idx] = train_X_raw[:, idx].mean()
        stds[idx] = train_X_raw[:, idx].std()
        if stds[idx] < 1e-8:
            stds[idx] = 1.0  # Avoid division by zero (e.g. constant data_size)

    # Compute per-feature min/max from training set (raw, for OOD detection)
    x_mins = train_X_raw.min(axis=0).astype(np.float32)
    x_maxs = train_X_raw.max(axis=0).astype(np.float32)

    train_X = (train_X_raw - means) / stds
    val_X = (val_X_raw - means) / stds

    # Standardize outputs too (from training set stats)
    train_Y_raw = df_train[output_cols].values.astype(np.float32)
    val_Y_raw = df_val[output_cols].values.astype(np.float32)

    y_means = train_Y_raw.mean(axis=0)
    y_stds = train_Y_raw.std(axis=0)
    y_stds[y_stds < 1e-8] = 1.0

    train_Y = (train_Y_raw - y_means) / y_stds
    val_Y = (val_Y_raw - y_means) / y_stds

    # ---- Summary ----
    print(f"\n  Feature matrix: {train_X.shape[1]} input features")
    print(f"  Features: {feature_cols}")
    print(f"  Outputs: {output_cols}")
    print(f"  Continuous features standardized: {CONTINUOUS_FEATURES}")

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


def inverse_transform_outputs(Y_norm: np.ndarray, y_means: np.ndarray,
                               y_stds: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert normalized model outputs back to original scale.

    Returns dict with:
        compression_time_ms, decompression_time_ms,
        compression_ratio, psnr_db
    """
    Y_raw = Y_norm * y_stds + y_means

    return {
        'compression_time_ms': np.expm1(Y_raw[:, 0]),   # inverse of log1p
        'decompression_time_ms': np.expm1(Y_raw[:, 1]),
        'compression_ratio': np.expm1(Y_raw[:, 2]),
        'psnr_db': Y_raw[:, 3],                          # was clamped, not logged
    }


if __name__ == '__main__':
    csv_path = Path(__file__).parent.parent / 'benchmark_results.csv'
    data = load_and_prepare(str(csv_path))
    print(f"\nTrain X shape: {data['train_X'].shape}")
    print(f"Train Y shape: {data['train_Y'].shape}")
    print(f"Val X shape:   {data['val_X'].shape}")
    print(f"Val Y shape:   {data['val_Y'].shape}")
