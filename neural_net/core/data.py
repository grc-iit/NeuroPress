"""
Data pipeline for compression performance prediction.

Shared constants, feature encoding, normalization, and train/val splitting.
Used by binary_data.py (primary), evaluate.py, and retrain.py.
"""

import numpy as np
import pandas as pd
from typing import Dict

from neural_net.core.configs import ALGORITHM_NAMES

# ============================================================
# Constants
# ============================================================

NUM_ALGORITHMS = len(ALGORITHM_NAMES)

INPUT_COLUMNS = ['algorithm', 'quantization', 'shuffle', 'error_bound',
                 'original_size', 'entropy', 'mad', 'second_derivative']

OUTPUT_COLUMNS = ['compression_time_ms', 'decompression_time_ms',
                  'compression_ratio', 'psnr_db']

# Internal encoded names → inverse transform type
# 'expm1' = inverse of log1p, 'identity' = no transform
OUTPUT_INVERSE = {
    'comp_time_log': ('compression_time_ms', 'expm1'),
    'decomp_time_log': ('decompression_time_ms', 'expm1'),
    'ratio_log': ('compression_ratio', 'expm1'),
    'psnr_clamped': ('psnr_db', 'identity'),
    'rmse_log': ('rmse', 'expm1'),
    'max_error_log': ('max_error', 'expm1'),
    'comp_tp_log': ('compression_throughput_mbps', 'expm1'),
    'decomp_tp_log': ('decompression_throughput_mbps', 'expm1'),
    'log_mae': ('mean_abs_err', 'expm1'),
    'ssim_nlog': ('ssim', 'ssim_nlog'),
}

CONTINUOUS_FEATURES = ['alg_id', 'error_bound_enc', 'data_size_enc', 'entropy', 'mad', 'second_derivative']

# Quality-of-reconstruction outputs whose values are trivially saturated on
# lossless rows (PSNR=120, MAE=0, SSIM=1). MAPE/R²/SHAP for these targets
# should be computed on the lossy slice (quantization='linear') only;
# otherwise the lossless rows' constant-target branch artificially inflates
# accuracy and feature-importance numbers.
LOSSY_ONLY_OUTPUTS = {'psnr_db', 'mean_abs_err', 'ssim'}


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

    # Fallback: use first_derivative as second_derivative if missing
    if 'second_derivative' not in df.columns and 'first_derivative' in df.columns:
        df['second_derivative'] = df['first_derivative']
        print("  Note: using first_derivative as second_derivative")

    # ---- Encode inputs ----

    # Algorithm: integer label 0..(N-1)
    alg_index = {name: i for i, name in enumerate(ALGORITHM_NAMES)}
    df['alg_id'] = df['algorithm'].map(alg_index).astype(np.float32)

    # Quantization: binary
    df['quant_enc'] = (df['quantization'] == 'linear').astype(np.float32)

    # Shuffle: binary
    df['shuffle_enc'] = (df['shuffle'] > 0).astype(np.float32)

    df['error_bound_enc'] = df['error_bound'].astype(np.float32)

    df['data_size_enc'] = df['original_size'].astype(np.float32)

    # ---- Encode outputs ----

    # Compression time: log1p (floor at 1ms to reduce sub-ms noise)
    df['comp_time_log'] = np.log1p(df['compression_time_ms'].clip(lower=1)).astype(np.float32)

    # Decompression time: log1p (floor at 1ms)
    df['decomp_time_log'] = np.log1p(df['decompression_time_ms'].clip(lower=1)).astype(np.float32)

    # Compression ratio: log1p
    df['ratio_log'] = np.log1p(df['compression_ratio'].clip(lower=0)).astype(np.float32)

    # PSNR: clamp inf and NaN to 120, keep as-is
    df['psnr_clamped'] = df['psnr_db'].replace([np.inf, -np.inf], 120.0).fillna(120.0).clip(upper=120.0).astype(np.float32)

    # Additional error/quality metrics (optional columns)
    if 'rmse' in df.columns:
        df['rmse_log'] = np.log1p(df['rmse'].clip(lower=0).fillna(0)).astype(np.float32)
    if 'max_error' in df.columns:
        df['max_error_log'] = np.log1p(df['max_error'].clip(lower=0).fillna(0)).astype(np.float32)
    if 'compression_throughput_mbps' in df.columns:
        df['comp_tp_log'] = np.log1p(df['compression_throughput_mbps'].clip(lower=0).fillna(0)).astype(np.float32)
    if 'decompression_throughput_mbps' in df.columns:
        df['decomp_tp_log'] = np.log1p(df['decompression_throughput_mbps'].clip(lower=0).fillna(0)).astype(np.float32)
    if 'mean_abs_err' in df.columns:
        df['log_mae'] = np.log1p(df['mean_abs_err'].clip(lower=0).fillna(0)).astype(np.float32)
    if 'ssim' in df.columns:
        # -log(1 - ssim + eps): amplifies differences near 1 (e.g. 0.995 vs 0.999)
        ssim_clipped = df['ssim'].clip(lower=0, upper=1).fillna(1.0)
        df['ssim_nlog'] = (-np.log(1.0 - ssim_clipped + 1e-7)).astype(np.float32)

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
    feature_cols = ['alg_id', 'quant_enc', 'shuffle_enc',
                    'error_bound_enc', 'data_size_enc',
                    'entropy', 'mad', 'second_derivative']

    output_cols = ['comp_time_log', 'decomp_time_log', 'ratio_log', 'psnr_clamped']
    for extra in ['rmse_log', 'max_error_log', 'comp_tp_log', 'decomp_tp_log', 'log_mae', 'ssim_nlog']:
        if extra in df.columns:
            output_cols.append(extra)

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


def compute_stats_cpu(raw_bytes, dtype=np.float32):
    """Compute entropy, MAD, and second_derivative on CPU.

    Mirrors the GPU implementation in stats_kernel.cu and entropy_kernel.cu:
      - Entropy: byte-level Shannon entropy (256-bin histogram, log2) in bits
      - MAD: mean absolute deviation from mean, normalized by data range
      - Second derivative: mean |x[i+1] - 2*x[i] + x[i-1]|, normalized by data range

    Args:
        raw_bytes: raw byte buffer of the data
        dtype: numpy dtype of the underlying data (default np.float32)
    """
    arr = np.frombuffer(raw_bytes, dtype=dtype)

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

    # MAD: mean absolute deviation
    mean_val = np.mean(arr.astype(np.float64))
    mad = np.mean(np.abs(arr.astype(np.float64) - mean_val))

    # Second derivative: mean |x[i+1] - 2*x[i] + x[i-1]|
    flat = arr.astype(np.float64)
    second_deriv = flat[2:] - 2.0 * flat[1:-1] + flat[:-2]
    second_derivative = np.mean(np.abs(second_deriv))

    entropy = 0.0 if np.isnan(entropy) else float(entropy)
    mad = 0.0 if np.isnan(mad) else float(mad)
    second_derivative = 0.0 if np.isnan(second_derivative) else float(second_derivative)
    return entropy, mad, second_derivative


def normalize_synthetic_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Map synthetic_benchmark CSV columns to the format expected by encode_and_split."""
    if 'chunk_id' not in df.columns:
        return df
    out = pd.DataFrame()
    out['file'] = 'chunk_' + df['chunk_id'].astype(str)
    out['algorithm'] = df['algorithm']
    out['quantization'] = df['quantized'].apply(
        lambda x: 'linear' if str(x).lower() in ('true', '1', 'yes') else 'none')
    out['shuffle'] = df['shuffle_bytes']
    out['error_bound'] = df['error_bound']
    out['original_size'] = df['original_bytes']
    out['entropy'] = df['entropy_bits']
    out['mad'] = df['mad']
    out['second_derivative'] = df['second_derivative']
    out['compression_time_ms'] = df['comp_time_ms']
    out['decompression_time_ms'] = df['decomp_time_ms']
    out['compression_ratio'] = df['compression_ratio']
    out['psnr_db'] = df['psnr_db'].replace([float('inf')], 120.0).fillna(120.0)
    out['success'] = df['success']
    for src, dst in [('ssim', 'ssim'), ('rmse', 'rmse'),
                     ('max_abs_error', 'max_error'),
                     ('mean_abs_error', 'mean_abs_err'),
                     ('data_range', 'data_range')]:
        if src in df.columns:
            out[dst] = df[src]
    return out


def load_from_csv(csv_paths, val_fraction=0.2, seed=42):
    """Load benchmark data from one or more CSV files and prepare for training.

    Each CSV must contain the columns expected by encode_and_split().
    Synthetic benchmark CSVs (with chunk_id column) are automatically normalized.
    """
    frames = [normalize_synthetic_csv(pd.read_csv(p)) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csv_paths)} CSV file(s)")
    return encode_and_split(df, val_fraction, seed)


def inverse_transform_outputs(Y_norm: np.ndarray, y_means: np.ndarray,
                               y_stds: np.ndarray,
                               output_cols: list = None) -> Dict[str, np.ndarray]:
    """
    Convert normalized model outputs back to original scale.

    Args:
        output_cols: list of internal column names (e.g. 'comp_time_log').
                     If None, uses legacy 4-output format.
    """
    Y_raw = Y_norm * y_stds + y_means

    if output_cols is None:
        # Legacy 4-output format
        return {
            'compression_time_ms': np.expm1(Y_raw[:, 0]),
            'decompression_time_ms': np.expm1(Y_raw[:, 1]),
            'compression_ratio': np.expm1(Y_raw[:, 2]),
            'psnr_db': Y_raw[:, 3],
        }

    result = {}
    for i, col in enumerate(output_cols):
        raw_name, inv_type = OUTPUT_INVERSE.get(col, (col, 'identity'))
        if inv_type == 'expm1':
            result[raw_name] = np.expm1(Y_raw[:, i])
        elif inv_type == 'ssim_nlog':
            # inverse of -log(1 - ssim + eps): ssim = 1 - exp(-x)
            result[raw_name] = 1.0 - np.exp(-Y_raw[:, i])
        else:
            result[raw_name] = Y_raw[:, i]
    return result
