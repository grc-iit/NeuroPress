"""
Train XGBoost models for compression performance prediction.

Uses the same data pipeline as the NN. One XGBoost model per output.

Usage:
    python neural_net/xgb_train.py --csv benchmark_results-100k.csv
    python neural_net/xgb_train.py --csv benchmark_results-100k.csv --cv 5
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import xgboost as xgb
from neural_net.core.data import inverse_transform_outputs, OUTPUT_COLUMNS, ALGORITHM_NAMES, CONTINUOUS_FEATURES, OUTPUT_INVERSE, LOSSY_ONLY_OUTPUTS

# Human-readable labels for features
FEATURE_LABELS = {
    'alg_lz4': 'LZ4', 'alg_snappy': 'Snappy', 'alg_deflate': 'Deflate',
    'alg_gdeflate': 'GDeflate', 'alg_zstd': 'Zstd', 'alg_ans': 'ANS',
    'alg_cascaded': 'Cascaded', 'alg_bitcomp': 'Bitcomp',
    'quant_enc': 'Quantization Mode',
    'shuffle_enc': 'Byte Shuffle',
    'algo_aggregate': 'Compression Algorithm',
    'error_bound_enc': 'Error Bound (log₁₀)',
    'data_size_enc': 'Data Size (log₂)',
    'entropy': 'Shannon Entropy',
    'mad': 'Mean Absolute Deviation',
    'second_derivative': 'Second Derivative',
}

OUTPUT_LABELS = {
    'comp_time_log': 'Compression\nTime (ms)',
    'decomp_time_log': 'Decompression\nTime (ms)',
    'ratio_log': 'Compression\nRatio',
    'psnr_clamped': 'PSNR\n(dB)',
    'rmse_log': 'RMSE',
    'max_error_log': 'Max Abs.\nError',
    'log_mae': 'Pointwise\nError (MAE)',
    'ssim_nlog': 'SSIM',
}

# Outputs excluded from SHAP heatmap
SHAP_EXCLUDE_OUTPUTS = {'comp_tp_log', 'decomp_tp_log', 'rmse_log', 'max_error_log'}


# Feature display order: preprocessors first, then library, then data characteristics
DISPLAY_FEATURES = [
    'quant_enc', 'error_bound_enc', 'shuffle_enc',  # preprocessors (grouped)
    'algo_aggregate',                     # compression library (aggregated)
    'data_size_enc',                      # data size
    'entropy', 'mad', 'second_derivative', # data characteristics
]

ALGO_COLS = {f'alg_{a}' for a in ALGORITHM_NAMES}


def _aggregate_shap(feature_names, shap_matrix):
    """Aggregate SHAP values: sum |SHAP| across algorithm columns into one.

    Returns (labels, mean_abs_per_feature) in DISPLAY_FEATURES order.
    """
    name_to_idx = {f: i for i, f in enumerate(feature_names)}

    # Algorithm aggregate: sum absolute SHAP across all algo one-hots, then mean
    algo_indices = [name_to_idx[f] for f in ALGO_COLS if f in name_to_idx]
    algo_shap = np.sum(np.abs(shap_matrix[:, algo_indices]), axis=1)  # per-sample
    algo_mean = float(np.mean(algo_shap))

    labels = []
    values = []
    for feat in DISPLAY_FEATURES:
        if feat == 'algo_aggregate':
            labels.append(FEATURE_LABELS.get(feat, feat))
            values.append(algo_mean)
        elif feat in name_to_idx:
            labels.append(FEATURE_LABELS.get(feat, feat))
            values.append(float(np.mean(np.abs(shap_matrix[:, name_to_idx[feat]]))))
        else:
            labels.append(FEATURE_LABELS.get(feat, feat))
            values.append(0.0)
    return labels, np.array(values)


def collect_feature_importance(models, val_X, feature_names, output_dir, lossy_mask=None):
    """Compute SHAP values and save heatmap plot.

    Args:
        lossy_mask: optional boolean mask aligned with val_X rows. When
            provided, SHAP for quality outputs (PSNR/MAE/SSIM, whose targets
            are constants on lossless rows) is computed on the lossy slice
            only, so feature importances reflect actual quality prediction
            rather than the trivial lossless branch.
    """
    import shap
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_names_ordered = [n for n in models.keys() if n not in SHAP_EXCLUDE_OUTPUTS]
    output_labels = [OUTPUT_LABELS.get(n, n) for n in output_names_ordered]

    # Map internal name -> raw target name to look up LOSSY_ONLY_OUTPUTS
    def _is_lossy_only(internal_name):
        raw, _ = OUTPUT_INVERSE.get(internal_name, (internal_name, 'identity'))
        return raw in LOSSY_ONLY_OUTPUTS

    max_shap_samples = 5000
    rng = np.random.RandomState(42)

    def _sample(X):
        if X.shape[0] > max_shap_samples:
            idx = rng.choice(X.shape[0], max_shap_samples, replace=False)
            return X[idx]
        return X

    X_shap_full = _sample(val_X)
    if lossy_mask is not None:
        X_shap_lossy = _sample(val_X[lossy_mask])
        print(f"  SHAP samples: full={X_shap_full.shape[0]}  lossy={X_shap_lossy.shape[0]}")
    else:
        X_shap_lossy = X_shap_full

    # Compute SHAP for all outputs in parallel (threads to avoid pickling)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _shap_one(out_name):
        X = X_shap_lossy if _is_lossy_only(out_name) else X_shap_full
        explainer = shap.TreeExplainer(models[out_name])
        sv = explainer.shap_values(X)
        tag = ' [lossy-only]' if _is_lossy_only(out_name) else ''
        print(f"    SHAP done: {out_name}{tag}")
        return out_name, _aggregate_shap(feature_names, sv)

    shap_matrix = np.zeros((len(DISPLAY_FEATURES), len(output_names_ordered)))
    name_to_col = {n: i for i, n in enumerate(output_names_ordered)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_shap_one, n) for n in output_names_ordered]
        for fut in as_completed(futures):
            out_name, (labels, values) = fut.result()
            shap_matrix[:, name_to_col[out_name]] = values

    # SHAP heatmap
    n_out = len(output_labels)
    fig_w = max(10, 2.0 * n_out + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    im = ax.imshow(shap_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_out))
    ax.set_xticklabels(output_labels, fontsize=10, ha='center')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    for i in range(shap_matrix.shape[0]):
        for j in range(shap_matrix.shape[1]):
            val = shap_matrix[i, j]
            color = 'white' if val > shap_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Mean |SHAP value|', fontsize=11)
    ax.set_title('')
    ax.set_xlabel('Prediction Target', fontsize=11, labelpad=8)
    ax.set_ylabel('Input Feature', fontsize=11, labelpad=8)
    plt.tight_layout()
    shap_path = output_dir / 'feature_importance_shap.png'
    fig.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {shap_path}")


def train_and_evaluate(data):
    """Train one XGBoost model per output, evaluate on validation set."""

    train_X = data['train_X']
    train_Y = data['train_Y']
    val_X = data['val_X']
    val_Y = data['val_Y']

    print(f"\nTrain: {train_X.shape[0]} rows, Val: {val_X.shape[0]} rows")
    print(f"Features: {train_X.shape[1]}, Outputs: {train_Y.shape[1]}")

    # Train one model per output
    output_names = data['output_cols_internal']
    models = {}
    val_preds = np.zeros_like(val_Y)

    for i, name in enumerate(output_names):
        print(f"\nTraining {name}...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=20,
            verbosity=0,
            n_jobs=4,
        )
        model.fit(train_X, train_Y[:, i],
                  eval_set=[(val_X, val_Y[:, i])],
                  verbose=False)
        val_preds[:, i] = model.predict(val_X)
        models[name] = model
        print(f"  Best iteration: {model.best_iteration}")

    print(f"\n{'='*65}")
    print("XGBOOST VALIDATION METRICS")
    print(f"{'='*65}")

    # Lossy-only mask (aligned with val rows from data['df_val'])
    df_val = data.get('df_val')
    val_lossy_mask = None
    if df_val is not None and 'quantization' in df_val.columns:
        val_lossy_mask = (df_val['quantization'] == 'linear').values

    for i, name in enumerate(output_names):
        raw_name, _ = OUTPUT_INVERSE.get(name, (name, 'identity'))
        if raw_name in LOSSY_ONLY_OUTPUTS and val_lossy_mask is not None:
            p = val_preds[val_lossy_mask, i]
            a = val_Y[val_lossy_mask, i]
            tag = ' [lossy-only]'
        else:
            p = val_preds[:, i]
            a = val_Y[:, i]
            tag = ''
        mae = np.mean(np.abs(p - a))
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mask = np.abs(a) > 0.1
        mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else 0.0
        label = OUTPUT_LABELS.get(name, name).replace('\n', ' ')

        print(f"\n  {label}:{tag}")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    MAPE: {mape:.1f}%")

    # Save models + normalization stats
    weights_dir = Path(__file__).parent.parent / 'weights'
    weights_dir.mkdir(exist_ok=True)
    save_path = weights_dir / 'xgb_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'x_means': data['x_means'],
            'x_stds': data['x_stds'],
            'y_means': data['y_means'],
            'y_stds': data['y_stds'],
            'feature_names': data['feature_names'],
        }, f)
    print(f"\nModels saved to {save_path}")

    # Feature importance (quality outputs computed on lossy slice only)
    print(f"\nCollecting feature importance...")
    collect_feature_importance(models, data['val_X'], data['feature_names'],
                               weights_dir, lossy_mask=val_lossy_mask)

    return models


def cross_validate(csv_paths, n_folds=5, seed=42):
    """K-fold CV, split by file."""
    from neural_net.training.cross_validate import prepare_fold
    from neural_net.core.data import OUTPUT_INVERSE
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['success'] == True].copy()
    print(f"Loaded {len(df)} successful rows")

    files = sorted(df['file'].unique())
    print(f"Unique files: {len(files)}, Folds: {n_folds}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    report_names = None
    all_metrics = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(files)):
        train_files = set(np.array(files)[train_idx])
        val_files = set(np.array(files)[val_idx])
        df_train = df[df['file'].isin(train_files)].copy()
        df_val = df[df['file'].isin(val_files)].copy()

        print(f"\nFold {fold+1}/{n_folds}: train={len(df_train)} val={len(df_val)}")

        train_X, train_Y, val_X, val_Y, y_means, y_stds, output_cols = prepare_fold(df_train, df_val)

        # Lossy-only mask (aligned with df_val row order)
        lossy_mask = (df_val['quantization'] == 'linear').values

        if report_names is None:
            report_names = [OUTPUT_INVERSE.get(c, (c, 'identity'))[0] for c in output_cols]
            all_metrics = {name: {'mae': [], 'r2': [], 'mape': []} for name in report_names}

        # Train one XGB per output
        val_preds = np.zeros_like(val_Y)
        for i in range(train_Y.shape[1]):
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                early_stopping_rounds=20, verbosity=0, n_jobs=4,
            )
            model.fit(train_X, train_Y[:, i],
                      eval_set=[(val_X, val_Y[:, i])], verbose=False)
            val_preds[:, i] = model.predict(val_X)

        pred_orig = inverse_transform_outputs(val_preds, y_means, y_stds, output_cols)
        actual_orig = inverse_transform_outputs(val_Y, y_means, y_stds, output_cols)

        for name in report_names:
            p = pred_orig[name]
            a = actual_orig[name]
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

    # Summary
    print(f"\n{'='*65}")
    print(f"XGBOOST {n_folds}-FOLD CV SUMMARY")
    print(f"{'='*65}")
    print(f"{'Output':<30s}  {'MAE':>10s}  {'R²':>10s}  {'MAPE':>10s}")
    print(f"{'-'*65}")
    for name in report_names:
        mae_m = np.mean(all_metrics[name]['mae'])
        mae_s = np.std(all_metrics[name]['mae'])
        r2_m = np.mean(all_metrics[name]['r2'])
        r2_s = np.std(all_metrics[name]['r2'])
        mape_m = np.mean(all_metrics[name]['mape'])
        mape_s = np.std(all_metrics[name]['mape'])
        tag = ' [lossy-only]' if name in LOSSY_ONLY_OUTPUTS else ''
        print(f"  {name:<28s}  {mae_m:>8.4f}±{mae_s:.4f}  "
              f"{r2_m:>6.4f}±{r2_s:.4f}  "
              f"{mape_m:>8.4f}±{mape_s:.4f}%{tag}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost compression predictor')
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                        help='CSV file(s) with benchmark results')
    parser.add_argument('--cv', type=int, default=None,
                        help='Run K-fold CV (e.g. --cv 5)')
    args = parser.parse_args()

    if args.cv:
        cross_validate(args.csv, n_folds=args.cv)
    else:
        from neural_net.core.data import load_from_csv
        data = load_from_csv(args.csv)
        models = train_and_evaluate(data)
