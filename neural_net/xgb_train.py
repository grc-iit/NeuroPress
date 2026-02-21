"""
Train XGBoost models for compression performance prediction.

Uses the same data pipeline as the NN. One XGBoost model per output.

Usage:
    python neural_net/xgb_train.py --csv benchmark_results-100k.csv
    python neural_net/xgb_train.py --csv benchmark_results-100k.csv --cv 5
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

import pickle
import shap
import xgboost as xgb
from data import inverse_transform_outputs, OUTPUT_COLUMNS, ALGORITHM_NAMES, CONTINUOUS_FEATURES
from cv_eval import prepare_fold

# Human-readable labels for features
FEATURE_LABELS = {
    'alg_lz4': 'LZ4', 'alg_snappy': 'Snappy', 'alg_deflate': 'Deflate',
    'alg_gdeflate': 'GDeflate', 'alg_zstd': 'Zstd', 'alg_ans': 'ANS',
    'alg_cascaded': 'Cascaded', 'alg_bitcomp': 'Bitcomp',
    'quant_enc': 'Quantization', 'shuffle_enc': 'Shuffle',
    'algo_aggregate': 'Compression Library',
    'error_bound_enc': 'Error Bound', 'data_size_enc': 'Data Size',
    'entropy': 'Entropy', 'mad': 'MAD', 'second_derivative': 'Second Derivative',
}

OUTPUT_LABELS = {
    'comp_time_log': 'Compression Time',
    'decomp_time_log': 'Decompression Time',
    'ratio_log': 'Compression Ratio',
    'psnr_clamped': 'PSNR',
}


# Feature display order: preprocessors first, then library, then data characteristics
DISPLAY_FEATURES = [
    'quant_enc', 'shuffle_enc',          # preprocessors (grouped)
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


def collect_feature_importance(models, val_X, feature_names, output_dir):
    """Compute SHAP values and save heatmap plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_names_ordered = list(models.keys())
    output_labels = [OUTPUT_LABELS.get(n, n) for n in output_names_ordered]

    max_shap_samples = 5000
    if val_X.shape[0] > max_shap_samples:
        idx = np.random.RandomState(42).choice(val_X.shape[0], max_shap_samples, replace=False)
        X_shap = val_X[idx]
    else:
        X_shap = val_X

    # Compute SHAP for all outputs, build heatmap matrix
    shap_matrix = np.zeros((len(DISPLAY_FEATURES), len(output_names_ordered)))

    for col, out_name in enumerate(output_names_ordered):
        explainer = shap.TreeExplainer(models[out_name])
        sv = explainer.shap_values(X_shap)
        labels, values = _aggregate_shap(feature_names, sv)
        shap_matrix[:, col] = values

    # SHAP heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(shap_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(output_labels)))
    ax.set_xticklabels(output_labels, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    for i in range(shap_matrix.shape[0]):
        for j in range(shap_matrix.shape[1]):
            val = shap_matrix[i, j]
            color = 'white' if val > shap_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean |SHAP value|', fontsize=11)
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
    output_names = ['comp_time_log', 'decomp_time_log', 'ratio_log', 'psnr_clamped']
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
        )
        model.fit(train_X, train_Y[:, i],
                  eval_set=[(val_X, val_Y[:, i])],
                  verbose=False)
        val_preds[:, i] = model.predict(val_X)
        models[name] = model
        print(f"  Best iteration: {model.best_iteration}")

    # Inverse transform and compute metrics
    pred_orig = inverse_transform_outputs(val_preds, data['y_means'], data['y_stds'])
    actual_orig = inverse_transform_outputs(data['val_Y'], data['y_means'], data['y_stds'])

    print(f"\n{'='*65}")
    print("XGBOOST VALIDATION METRICS")
    print(f"{'='*65}")

    for name in OUTPUT_COLUMNS:
        p = pred_orig[name]
        a = actual_orig[name]
        mae = np.mean(np.abs(p - a))
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mask = np.abs(a) > 0.1
        mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else 0.0

        print(f"\n  {name}:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    MAPE: {mape:.1f}%")

    # Save models + normalization stats
    weights_dir = Path(__file__).parent / 'weights'
    weights_dir.mkdir(exist_ok=True)
    save_path = weights_dir / 'xgb_model.pkl'
    pickle.dump({
        'models': models,
        'x_means': data['x_means'],
        'x_stds': data['x_stds'],
        'y_means': data['y_means'],
        'y_stds': data['y_stds'],
        'feature_names': data['feature_names'],
    }, open(save_path, 'wb'))
    print(f"\nModels saved to {save_path}")

    # Feature importance
    print(f"\nCollecting feature importance...")
    collect_feature_importance(models, data['val_X'], data['feature_names'], weights_dir)

    return models


def cross_validate(csv_paths, n_folds=5, seed=42):
    """K-fold CV, split by file."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['success'] == True].copy()
    print(f"Loaded {len(df)} successful rows")

    files = sorted(df['file'].unique())
    print(f"Unique files: {len(files)}, Folds: {n_folds}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_metrics = {name: {'mae': [], 'r2': [], 'mape': []} for name in OUTPUT_COLUMNS}

    for fold, (train_idx, val_idx) in enumerate(kf.split(files)):
        train_files = set(np.array(files)[train_idx])
        val_files = set(np.array(files)[val_idx])
        df_train = df[df['file'].isin(train_files)].copy()
        df_val = df[df['file'].isin(val_files)].copy()

        print(f"\nFold {fold+1}/{n_folds}: train={len(df_train)} val={len(df_val)}")

        train_X, train_Y, val_X, val_Y, y_means, y_stds = prepare_fold(df_train, df_val)

        # Train one XGB per output
        val_preds = np.zeros_like(val_Y)
        for i in range(train_Y.shape[1]):
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                early_stopping_rounds=20, verbosity=0,
            )
            model.fit(train_X, train_Y[:, i],
                      eval_set=[(val_X, val_Y[:, i])], verbose=False)
            val_preds[:, i] = model.predict(val_X)

        pred_orig = inverse_transform_outputs(val_preds, y_means, y_stds)
        actual_orig = inverse_transform_outputs(val_Y, y_means, y_stds)

        for name in OUTPUT_COLUMNS:
            p = pred_orig[name]
            a = actual_orig[name]
            mae = np.mean(np.abs(p - a))
            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mask = np.abs(a) > 0.1
            mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else 0.0
            all_metrics[name]['mae'].append(mae)
            all_metrics[name]['r2'].append(r2)
            all_metrics[name]['mape'].append(mape)
            print(f"  {name:30s}  MAE={mae:8.4f}  R²={r2:.4f}  MAPE={mape:5.1f}%")

    # Summary
    print(f"\n{'='*65}")
    print(f"XGBOOST {n_folds}-FOLD CV SUMMARY")
    print(f"{'='*65}")
    print(f"{'Output':<30s}  {'MAE':>10s}  {'R²':>10s}  {'MAPE':>10s}")
    print(f"{'-'*65}")
    for name in OUTPUT_COLUMNS:
        mae_m = np.mean(all_metrics[name]['mae'])
        mae_s = np.std(all_metrics[name]['mae'])
        r2_m = np.mean(all_metrics[name]['r2'])
        r2_s = np.std(all_metrics[name]['r2'])
        mape_m = np.mean(all_metrics[name]['mape'])
        mape_s = np.std(all_metrics[name]['mape'])
        print(f"  {name:<28s}  {mae_m:>6.4f}±{mae_s:.4f}  "
              f"{r2_m:>6.4f}±{r2_s:.4f}  "
              f"{mape_m:>5.1f}±{mape_s:.1f}%")


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
        from data import load_from_csv
        data = load_from_csv(args.csv)
        models = train_and_evaluate(data)
