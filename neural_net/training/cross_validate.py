"""
5-fold cross-validation for the compression performance predictor.

Splits by file (not by row) to avoid data leakage, then trains a fresh
model on each fold and reports per-output MAE, R², and MAPE.

Usage:
    python neural_net/cv_eval.py --csv benchmark_results-100k.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neural_net.core.data import encode_and_split, ALGORITHM_NAMES, CONTINUOUS_FEATURES, OUTPUT_COLUMNS, OUTPUT_INVERSE, LOSSY_ONLY_OUTPUTS
from neural_net.core.data import inverse_transform_outputs
from neural_net.core.model import CompressionPredictor


def prepare_fold(df_train, df_val):
    """Run the same encoding/normalization as encode_and_split but on pre-split DataFrames."""

    for sub_df in [df_train, df_val]:
        # Algorithm one-hot
        for alg in ALGORITHM_NAMES:
            sub_df[f'alg_{alg}'] = (sub_df['algorithm'] == alg).astype(np.float32)
        # Quantization
        sub_df['quant_enc'] = (sub_df['quantization'] == 'linear').astype(np.float32)
        # Shuffle
        sub_df['shuffle_enc'] = (sub_df['shuffle'] > 0).astype(np.float32)
        # Error bound log-scale
        sub_df['error_bound_enc'] = np.log10(sub_df['error_bound'].clip(lower=1e-7)).astype(np.float32)
        # Data size log2
        sub_df['data_size_enc'] = np.log2(sub_df['original_size'].clip(lower=1)).astype(np.float32)
        # Core output encodings
        sub_df['comp_time_log'] = np.log1p(sub_df['compression_time_ms'].clip(lower=0)).astype(np.float32)
        sub_df['decomp_time_log'] = np.log1p(sub_df['decompression_time_ms'].clip(lower=0)).astype(np.float32)
        sub_df['ratio_log'] = np.log1p(sub_df['compression_ratio'].clip(lower=0)).astype(np.float32)
        sub_df['psnr_clamped'] = sub_df['psnr_db'].replace([np.inf, -np.inf], 120.0).fillna(120.0).clip(upper=120.0).astype(np.float32)
        # Extended output encodings
        if 'mean_abs_err' in sub_df.columns:
            sub_df['log_mae'] = np.log1p(sub_df['mean_abs_err'].clip(lower=0).fillna(0)).astype(np.float32)
        if 'ssim' in sub_df.columns:
            sub_df['ssim_val'] = sub_df['ssim'].clip(lower=0, upper=1).fillna(1.0).astype(np.float32)

    algo_cols = [f'alg_{a}' for a in ALGORITHM_NAMES]
    feature_cols = algo_cols + ['quant_enc', 'shuffle_enc',
                                'error_bound_enc', 'data_size_enc',
                                'entropy', 'mad', 'second_derivative']

    output_cols = ['comp_time_log', 'decomp_time_log', 'ratio_log', 'psnr_clamped']
    for extra in ['log_mae', 'ssim_val']:
        if extra in df_train.columns:
            output_cols.append(extra)

    # Normalization from training set
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

    train_X = (train_X_raw - means) / stds
    val_X = (val_X_raw - means) / stds

    train_Y_raw = df_train[output_cols].values.astype(np.float32)
    val_Y_raw = df_val[output_cols].values.astype(np.float32)

    y_means = train_Y_raw.mean(axis=0)
    y_stds = train_Y_raw.std(axis=0)
    y_stds[y_stds < 1e-8] = 1.0

    train_Y = (train_Y_raw - y_means) / y_stds
    val_Y = (val_Y_raw - y_means) / y_stds

    return train_X, train_Y, val_X, val_Y, y_means, y_stds, output_cols


def train_one_fold(train_X, train_Y, val_X, val_Y, epochs=100, batch_size=512,
                   lr=1e-3, patience=15, hidden_dim=128):
    """Train model on one fold, return best val predictions (normalized)."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tX = torch.from_numpy(train_X).to(device)
    tY = torch.from_numpy(train_Y).to(device)
    vX = torch.from_numpy(val_X).to(device)
    vY = torch.from_numpy(val_Y).to(device)

    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vX, vY), batch_size=batch_size * 2)

    input_dim = train_X.shape[1]
    output_dim = train_Y.shape[1]
    model = CompressionPredictor(input_dim=input_dim, hidden_dim=hidden_dim,
                                  output_dim=output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for bx, by in train_loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * bx.size(0)
            train_n += bx.size(0)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for bx, by in val_loader:
                loss = criterion(model(bx), by)
                val_loss_sum += loss.item() * bx.size(0)
                val_n += bx.size(0)
        val_loss = val_loss_sum / val_n

        scheduler.step(val_loss)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = ' *best*'
        else:
            no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        bar_len = 30
        progress = epoch / epochs
        filled = int(bar_len * progress)
        bar = '=' * filled + '-' * (bar_len - filled)
        print(f'\r  [{bar}] {epoch:3d}/{epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr_now:.1e}{marker}', end='', flush=True)

        if no_improve >= patience:
            print(f'\n  Early stop at epoch {epoch} (patience={patience})')
            break

    print()  # newline after progress bar

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(vX).cpu().numpy()

    return val_pred_norm, best_val_loss, epoch


def main():
    parser = argparse.ArgumentParser(description='5-fold CV for compression predictor')
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                        help='CSV file(s) with benchmark results')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load data
    frames = [pd.read_csv(p) for p in args.csv]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['success'] == True].copy()
    print(f"Loaded {len(df)} successful rows from {len(args.csv)} CSV file(s)")

    # Split by file
    files = sorted(df['file'].unique())
    print(f"Unique files: {len(files)}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # First fold to discover output columns
    output_col_names = None
    all_metrics = None

    for fold_idx, (train_file_idx, val_file_idx) in enumerate(kf.split(files)):
        train_files = set(np.array(files)[train_file_idx])
        val_files = set(np.array(files)[val_file_idx])

        df_train = df[df['file'].isin(train_files)].copy()
        df_val = df[df['file'].isin(val_files)].copy()

        print(f"\n{'='*65}")
        print(f"FOLD {fold_idx+1}/{args.folds}  "
              f"(train: {len(df_train)} rows / {len(train_files)} files, "
              f"val: {len(df_val)} rows / {len(val_files)} files)")
        print(f"{'='*65}")

        train_X, train_Y, val_X, val_Y, y_means, y_stds, output_cols = prepare_fold(df_train, df_val)

        # Lossy-only mask, aligned with df_val row order (which prepare_fold preserves)
        lossy_mask = (df_val['quantization'] == 'linear').values

        if output_col_names is None:
            output_col_names = output_cols
            report_names = [OUTPUT_INVERSE.get(c, (c, 'identity'))[0] for c in output_cols]
            all_metrics = {name: {'mae': [], 'r2': [], 'mape': []} for name in report_names}

        val_pred_norm, best_loss, last_epoch = train_one_fold(
            train_X, train_Y, val_X, val_Y,
            epochs=args.epochs, hidden_dim=args.hidden_dim)

        print(f"  Best val loss: {best_loss:.6f} (stopped at epoch {last_epoch})")

        # Inverse transform
        pred_orig = inverse_transform_outputs(val_pred_norm, y_means, y_stds, output_cols)
        actual_orig = inverse_transform_outputs(val_Y, y_means, y_stds, output_cols)

        for name in report_names:
            p = pred_orig[name]
            a = actual_orig[name]
            # Quality outputs (PSNR/MAE/SSIM) are constant on lossless rows
            # (PSNR=120, MAE=0, SSIM=1). Restrict their metrics to the lossy
            # slice so the report reflects actual prediction quality.
            if name in LOSSY_ONLY_OUTPUTS:
                p = p[lossy_mask]
                a = a[lossy_mask]
                tag = ' [lossy-only]'
            else:
                tag = ''
            mae = np.mean(np.abs(p - a))
            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mask = np.abs(a) > 0.1
            mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else 0.0

            all_metrics[name]['mae'].append(mae)
            all_metrics[name]['r2'].append(r2)
            all_metrics[name]['mape'].append(mape)

            print(f"  {name:30s}  MAE={mae:8.4f}  R²={r2:.4f}  MAPE={mape:5.1f}%{tag}")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"NN 5-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*65}")
    print(f"{'Output':<30s}  {'MAE':>10s}  {'R²':>10s}  {'MAPE':>10s}")
    print(f"{'-'*65}")
    for name in report_names:
        mae_mean = np.mean(all_metrics[name]['mae'])
        mae_std = np.std(all_metrics[name]['mae'])
        r2_mean = np.mean(all_metrics[name]['r2'])
        r2_std = np.std(all_metrics[name]['r2'])
        mape_mean = np.mean(all_metrics[name]['mape'])
        mape_std = np.std(all_metrics[name]['mape'])
        tag = ' [lossy-only]' if name in LOSSY_ONLY_OUTPUTS else ''
        print(f"  {name:<28s}  {mae_mean:>6.4f}±{mae_std:.4f}  "
              f"{r2_mean:>6.4f}±{r2_std:.4f}  "
              f"{mape_mean:>5.1f}±{mape_std:.1f}%{tag}")


if __name__ == '__main__':
    main()
