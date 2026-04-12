"""
Hyperparameter grid search for the compression performance predictor.

Runs 3-fold CV over a grid of hidden_dim, num_hidden_layers, lr, and
batch_size. Reports mean val loss and R² for each config, and saves a
CSV summary to neural_net/weights/hyperparam_study.csv.

Usage:
    python neural_net/training/hyperparam_study.py --csv synthetic_results.csv
    python neural_net/training/hyperparam_study.py --csv synthetic_results.csv --folds 5
"""

import sys
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neural_net.training.cross_validate import prepare_fold, train_one_fold
from neural_net.core.data import LOSSY_ONLY_OUTPUTS, OUTPUT_INVERSE
from neural_net.core.data import inverse_transform_outputs


def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map synthetic_benchmark CSV columns to the format expected by encode_and_split.

    synthetic_benchmark produces:
        chunk_id, algorithm, shuffle_bytes, quantized, error_bound,
        entropy_bits, mad, second_derivative, original_bytes,
        comp_time_ms, decomp_time_ms, compression_ratio, psnr_db,
        ssim, rmse, max_abs_error, mean_abs_error, success, ...

    encode_and_split expects:
        file, algorithm, quantization, shuffle, error_bound,
        entropy, mad, second_derivative, original_size,
        compression_time_ms, decompression_time_ms, compression_ratio,
        psnr_db, success  (+ optional: ssim, mean_abs_err, rmse, max_error)
    """
    # Only remap if the synthetic format is detected
    if 'chunk_id' not in df.columns:
        return df

    out = pd.DataFrame()
    # Each chunk_id+config combination is a unique "file" for CV splitting
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

    # Optional quality columns
    for src, dst in [('ssim', 'ssim'), ('rmse', 'rmse'),
                     ('max_abs_error', 'max_error'),
                     ('mean_abs_error', 'mean_abs_err')]:
        if src in df.columns:
            out[dst] = df[src]

    return out

# ============================================================
# Hyperparameter grid
# ============================================================

GRID = {
    'hidden_dim':        [64],
    'num_hidden_layers': [4],
    'lr':                [3e-4],
    'batch_size':        [512],
}


def run_study(csv_paths, folds=5, epochs=150, patience=15, seed=42):
    # Load data
    frames = [normalize_csv(pd.read_csv(p)) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['success'] == True].copy()
    print(f"Loaded {len(df)} successful rows from {len(csv_paths)} file(s)")

    files = sorted(df['file'].unique())
    print(f"Unique files (CV units): {len(files)}")

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    # Build list of all configs
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    configs = [dict(zip(keys, c)) for c in combos]
    print(f"\nGrid size: {len(configs)} configs × {folds} folds = "
          f"{len(configs) * folds} training runs\n")

    results = []
    total_runs = len(configs) * folds
    run_idx = 0
    t_study_start = time.time()

    for cfg_idx, cfg in enumerate(configs):
        fold_val_losses = []
        fold_r2s = []
        fold_per_stat_r2s = {}   # stat_name -> list of per-fold R²
        fold_per_stat_mapes = {}  # stat_name -> list of per-fold MAPE
        stat_names = None

        for fold_idx, (train_file_idx, val_file_idx) in enumerate(kf.split(files)):
            run_idx += 1
            train_files = set(np.array(files)[train_file_idx])
            val_files   = set(np.array(files)[val_file_idx])

            df_train = df[df['file'].isin(train_files)].copy()
            df_val   = df[df['file'].isin(val_files)].copy()

            train_X, train_Y, val_X, val_Y, y_means, y_stds, output_cols = \
                prepare_fold(df_train, df_val)

            t0 = time.time()
            val_pred_norm, best_loss, last_epoch = train_one_fold(
                train_X, train_Y, val_X, val_Y,
                epochs=epochs,
                batch_size=cfg['batch_size'],
                lr=cfg['lr'],
                patience=patience,
                hidden_dim=cfg['hidden_dim'],
                model_variant='shared',
                num_hidden_layers=cfg['num_hidden_layers'],
            )
            elapsed = time.time() - t0

            # Compute per-stat and mean R² (original scale)
            pred_orig   = inverse_transform_outputs(val_pred_norm, y_means, y_stds, output_cols)
            actual_orig = inverse_transform_outputs(val_Y,         y_means, y_stds, output_cols)
            lossy_mask  = (df_val['quantization'] == 'linear').values

            if stat_names is None:
                stat_names = list(pred_orig.keys())
                fold_per_stat_r2s   = {n: [] for n in stat_names}
                fold_per_stat_mapes = {n: [] for n in stat_names}

            r2_vals = []
            for name in stat_names:
                p = pred_orig[name]
                a = actual_orig[name]
                if name in LOSSY_ONLY_OUTPUTS:
                    p, a = p[lossy_mask], a[lossy_mask]
                if len(a) < 2:
                    fold_per_stat_r2s[name].append(0.0)
                    fold_per_stat_mapes[name].append(0.0)
                    continue
                ss_res = np.sum((a - p) ** 2)
                ss_tot = np.sum((a - a.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                fold_per_stat_r2s[name].append(r2)
                r2_vals.append(r2)
                mask = np.abs(a) > 0.01
                mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.sum() > 0 else 0.0
                fold_per_stat_mapes[name].append(mape)

            mean_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0
            fold_val_losses.append(best_loss)
            fold_r2s.append(mean_r2)

            eta_per_run = (time.time() - t_study_start) / run_idx
            eta_remaining = eta_per_run * (total_runs - run_idx)
            print(f"  [{run_idx:3d}/{total_runs}] cfg={cfg_idx+1:3d} fold={fold_idx+1}  "
                  f"hidden={cfg['hidden_dim']:3d} layers={cfg['num_hidden_layers']} "
                  f"lr={cfg['lr']:.0e} bs={cfg['batch_size']:3d}  "
                  f"val_loss={best_loss:.5f}  R²={mean_r2:.4f}  "
                  f"ep={last_epoch:3d}  t={elapsed:.1f}s  ETA={eta_remaining:.0f}s")

        per_stat_r2s   = {f'r2_{n}':   float(np.mean(v)) for n, v in fold_per_stat_r2s.items()}
        per_stat_mapes = {f'mape_{n}': float(np.mean(v)) for n, v in fold_per_stat_mapes.items()}
        results.append({
            **cfg,
            'val_loss_mean': float(np.mean(fold_val_losses)),
            'val_loss_std':  float(np.std(fold_val_losses)),
            'r2_mean':       float(np.mean(fold_r2s)),
            'r2_std':        float(np.std(fold_r2s)),
            **per_stat_r2s,
            **per_stat_mapes,
        })

    # ---- Summary ----
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('val_loss_mean').reset_index(drop=True)

    print(f"\n{'='*80}")
    print("HYPERPARAMETER STUDY RESULTS (sorted by val_loss_mean)")
    print(f"{'='*80}")
    print(df_results.to_string(index=True))

    best = df_results.iloc[0]
    print(f"\nBest config:")
    for k in keys:
        print(f"  {k:20s} = {best[k]}")
    print(f"  {'val_loss_mean':20s} = {best['val_loss_mean']:.6f} ± {best['val_loss_std']:.6f}")
    print(f"  {'r2_mean':20s} = {best['r2_mean']:.4f} ± {best['r2_std']:.4f}")
    print(f"\nPer-statistic R² and MAPE (best config):")
    per_stat_cols = [c for c in df_results.columns if c.startswith('r2_') and c != 'r2_mean' and c != 'r2_std']
    print(f"  {'stat':<30s}  {'R²':>8s}  {'MAPE':>8s}")
    print(f"  {'-'*52}")
    for col in per_stat_cols:
        stat = col[len('r2_'):]
        mape_col = f'mape_{stat}'
        tag = ' [lossy-only]' if stat in LOSSY_ONLY_OUTPUTS else ''
        mape_str = f"{best[mape_col]:7.2f}%" if mape_col in best.index else "     n/a"
        print(f"  {stat:<30s}  {best[col]:8.4f}  {mape_str}{tag}")

    out_path = Path(__file__).resolve().parent.parent.parent / 'results' / 'hyperparam_study.csv'
    out_path.parent.mkdir(exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                        help='CSV file(s) from synthetic_benchmark')
    parser.add_argument('--folds',   type=int, default=5,
                        help='CV folds (default: 5)')
    parser.add_argument('--epochs',  type=int, default=150,
                        help='Max epochs per run (default: 150)')
    parser.add_argument('--patience',type=int, default=15,
                        help='Early-stopping patience (default: 15)')
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    run_study(args.csv, folds=args.folds, epochs=args.epochs,
              patience=args.patience, seed=args.seed)
