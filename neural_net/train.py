"""
Training script for compression performance predictor.

Usage:
    # Train from CSV (no GPU needed)
    python neural_net/train.py --csv benchmark_results--1.csv

    # Train from binary files
    python neural_net/train.py --data-dir syntheticGeneration/training_data/ --lib-path build/libgpucompress.so

    # Export weights (unchanged)
    python neural_net/export_weights.py
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from data import inverse_transform_outputs
from model import CompressionPredictor


def train_model_with_data(data: dict, epochs: int = 200, batch_size: int = 512,
                          lr: float = 1e-3, patience: int = 20,
                          hidden_dim: int = 128):
    """Train the regression model from a prepared data dict."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ---- Create datasets ----
    train_X = torch.from_numpy(data['train_X'])
    train_Y = torch.from_numpy(data['train_Y'])
    val_X = torch.from_numpy(data['val_X'])
    val_Y = torch.from_numpy(data['val_Y'])

    train_ds = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))

    val_ds = TensorDataset(val_X, val_Y)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2,
                            pin_memory=(device.type == 'cuda'))

    # ---- Create model ----
    input_dim = train_X.shape[1]
    output_dim = train_Y.shape[1]
    model = CompressionPredictor(input_dim=input_dim, hidden_dim=hidden_dim,
                                  output_dim=output_dim).to(device)

    print(f"\nModel: {input_dim} -> {hidden_dim} -> {hidden_dim} -> {output_dim}")
    print(f"Parameters: {model.count_parameters():,}")

    # ---- Training setup ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    output_names = data['output_names']

    # ---- Training loop ----
    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>11}  {'LR':>10}  {'Status'}")
    print("-" * 65)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            pred = model(batch_X)
            loss = criterion(pred, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_X.size(0)
            train_count += batch_X.size(0)

        train_loss = train_loss_sum / train_count

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                pred = model(batch_X)
                loss = criterion(pred, batch_Y)

                val_loss_sum += loss.item() * batch_X.size(0)
                val_count += batch_X.size(0)

        val_loss = val_loss_sum / val_count

        # -- Learning rate scheduling --
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # -- Early stopping --
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            status = " *best*"
        else:
            epochs_without_improvement += 1

        if epoch <= 10 or epoch % 10 == 0 or status or epoch == epochs:
            print(f"{epoch:>5}  {train_loss:>11.6f}  {val_loss:>11.6f}  {current_lr:>10.1e}  {status}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")

    # ---- Load best model ----
    model.load_state_dict(best_state)
    model.eval()

    # ---- Per-output validation metrics ----
    print("\n" + "=" * 65)
    print("VALIDATION METRICS (per output, original scale)")
    print("=" * 65)

    with torch.no_grad():
        val_pred_norm = model(val_X.to(device)).cpu().numpy()

    # Inverse transform predictions and actuals
    pred_orig = inverse_transform_outputs(val_pred_norm, data['y_means'], data['y_stds'])
    actual_orig = inverse_transform_outputs(data['val_Y'], data['y_means'], data['y_stds'])

    for name in output_names:
        p = pred_orig[name]
        a = actual_orig[name]
        mae = np.mean(np.abs(p - a))
        # R²
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # MAPE (only where actual > threshold to avoid division by near-zero)
        mask = np.abs(a) > 0.1
        mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else 0.0

        print(f"\n  {name}:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    MAPE: {mape:.1f}%")

    # ---- Save model + scalers ----
    weights_dir = Path(__file__).parent / 'weights'
    weights_dir.mkdir(exist_ok=True)

    save_path = weights_dir / 'model.pt'
    torch.save({
        'model_state_dict': best_state,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'x_means': data['x_means'],
        'x_stds': data['x_stds'],
        'x_mins': data['x_mins'],
        'x_maxs': data['x_maxs'],
        'y_means': data['y_means'],
        'y_stds': data['y_stds'],
        'feature_names': data['feature_names'],
        'output_names': output_names,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train compression performance predictor')
    parser.add_argument('--csv', type=str, nargs='+', default=None,
                        help='One or more CSV files with benchmark results')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing .bin files for on-the-fly benchmarking')
    parser.add_argument('--lib-path', type=str, default=None,
                        help='Path to libgpucompress.so')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max .bin files to process')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--hidden-dim', type=int, default=128)
    args = parser.parse_args()

    if args.csv:
        from data import load_from_csv
        data = load_from_csv(args.csv)
    elif args.data_dir:
        from binary_data import load_and_prepare_from_binary
        data = load_and_prepare_from_binary(
            args.data_dir, lib_path=args.lib_path, max_files=args.max_files)
    else:
        parser.error("Provide --csv or --data-dir")

    model, data = train_model_with_data(
        data, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, hidden_dim=args.hidden_dim)
