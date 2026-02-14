#!/usr/bin/env python3
"""Inspect a model.pt checkpoint and print its contents."""

import argparse
import torch
import numpy as np


def inspect(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print(f"Checkpoint: {path}")
    print(f"Top-level keys: {list(ckpt.keys())}\n")

    # Architecture
    print("Architecture:")
    print(f"  input_dim:  {ckpt.get('input_dim')}")
    print(f"  hidden_dim: {ckpt.get('hidden_dim')}")
    print(f"  output_dim: {ckpt.get('output_dim')}")

    # Layer shapes
    sd = ckpt.get("model_state_dict", {})
    total_params = 0
    print(f"\nLayers ({len(sd)} tensors):")
    for name, param in sd.items():
        n = param.numel()
        total_params += n
        print(f"  {name:20s}  {str(list(param.shape)):16s}  ({n:,} params)")
    print(f"  {'Total':20s}  {'':16s}  ({total_params:,} params)")

    # Normalization
    print("\nInput normalization (x):")
    names = ckpt.get("feature_names", [f"f{i}" for i in range(len(ckpt.get("x_means", [])))])
    x_means = ckpt.get("x_means", [])
    x_stds = ckpt.get("x_stds", [])
    x_mins = ckpt.get("x_mins", [])
    x_maxs = ckpt.get("x_maxs", [])
    print(f"  {'feature':24s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for i, name in enumerate(names):
        row = f"  {name:24s}"
        row += f" {float(x_means[i]):10.4f}" if i < len(x_means) else ""
        row += f" {float(x_stds[i]):10.4f}" if i < len(x_stds) else ""
        row += f" {float(x_mins[i]):10.4f}" if i < len(x_mins) else ""
        row += f" {float(x_maxs[i]):10.4f}" if i < len(x_maxs) else ""
        print(row)

    print("\nOutput normalization (y):")
    out_names = ckpt.get("output_names", [f"y{i}" for i in range(len(ckpt.get("y_means", [])))])
    y_means = ckpt.get("y_means", [])
    y_stds = ckpt.get("y_stds", [])
    print(f"  {'output':24s} {'mean':>10s} {'std':>10s}")
    for i, name in enumerate(out_names):
        row = f"  {name:24s}"
        row += f" {float(y_means[i]):10.4f}" if i < len(y_means) else ""
        row += f" {float(y_stds[i]):10.4f}" if i < len(y_stds) else ""
        print(row)

    # Training metadata
    print("\nTraining:")
    if "best_epoch" in ckpt:
        print(f"  best_epoch:    {ckpt['best_epoch']}")
    if "best_val_loss" in ckpt:
        print(f"  best_val_loss: {ckpt['best_val_loss']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a model.pt checkpoint")
    parser.add_argument("path", nargs="?", default="neural_net/weights/model.pt",
                        help="Path to .pt file (default: neural_net/weights/model.pt)")
    args = parser.parse_args()
    inspect(args.path)
