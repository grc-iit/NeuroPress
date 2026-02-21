"""
Data pipeline for compression performance prediction.

Shared constants, feature encoding, normalization, and train/val splitting.
Used by binary_data.py (primary), evaluate.py, and retrain.py.
"""

import numpy as np
import pandas as pd
from typing import Dict

# ============================================================
# Constants
# ============================================================

ALGORITHM_NAMES = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']
NUM_ALGORITHMS = len(ALGORITHM_NAMES)

INPUT_COLUMNS = ['algorithm', 'quantization', 'shuffle', 'error_bound',
                 'original_size', 'entropy', 'mad', 'second_derivative']

OUTPUT_COLUMNS = ['compression_time_ms', 'decompression_time_ms',
                  'compression_ratio', 'psnr_db']

CONTINUOUS_FEATURES = ['error_bound_enc', 'data_size_enc', 'entropy', 'mad', 'second_derivative']


def encode_and_split(df: pd.DataFrame, val_fraction: float = 0.2,
                     seed: int = 42) -> Dict:
    """
    Encode features, normalize, and split a benchmark DataFrame by file.

    Expected DataFrame columns:
        file, algorithm, quantization, shuffle, error_bound, original_size,
        entropy, mad, second_derivative, compression_time_ms,
        decompression_time_ms, compression_ratio, psnr_db, success

    Returns dict with:
        train_X, train_Y, val_X, val_Y: numpy arrays
        feature_names, output_names: lists
        x_means, x_stds, x_mins, x_maxs: input normalization
        y_means, y_stds: output normalization
        df_val, df_train: DataFrames (for ranking evaluation)
    """
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
                                'entropy', 'mad', 'second_derivative']

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
            stds[idx] = 1.0  # Avoid division by zero

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


def compute_stats_cpu(raw_bytes):
    """Compute entropy, MAD, and second_derivative on CPU.

    Mirrors the GPU implementation in stats_kernel.cu and entropy_kernel.cu:
      - Entropy: byte-level Shannon entropy (256-bin histogram, log2) in bits
      - MAD: mean absolute deviation from mean, normalized by data range
      - Second derivative: mean |x[i+1] - 2*x[i] + x[i-1]|, normalized by data range
    """
    arr = np.frombuffer(raw_bytes, dtype=np.float32)

    # Entropy: byte-level histogram (same as GPU's 256-bin approach)
    byte_data = np.frombuffer(raw_bytes, dtype=np.uint8)
    hist = np.bincount(byte_data, minlength=256).astype(np.float64)
    probs = hist[hist > 0] / len(byte_data)
    entropy = -np.sum(probs * np.log2(probs))

    # Data range for normalization
    vmin = float(arr.min())
    vmax = float(arr.max())
    data_range = vmax - vmin

    if data_range < 1e-30 or len(arr) < 3:
        return entropy, 0.0, 0.0

    # MAD: mean absolute deviation, normalized by range
    mean_val = np.mean(arr.astype(np.float64))
    mad = np.mean(np.abs(arr.astype(np.float64) - mean_val)) / data_range

    # Second derivative: mean |x[i+1] - 2*x[i] + x[i-1]|, normalized by range
    flat = arr.astype(np.float64)
    second_deriv = flat[2:] - 2.0 * flat[1:-1] + flat[:-2]
    second_derivative = np.mean(np.abs(second_deriv)) / data_range

    return float(entropy), float(mad), float(second_derivative)


def load_from_csv(csv_paths, val_fraction=0.2, seed=42):
    """Load benchmark data from one or more CSV files and prepare for training.

    Each CSV must contain the columns expected by encode_and_split().
    """
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csv_paths)} CSV file(s)")
    return encode_and_split(df, val_fraction, seed)


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
