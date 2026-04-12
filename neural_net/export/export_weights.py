"""
Export trained PyTorch model weights to binary format for CUDA inference.

Binary format (little-endian):
    Header (24 bytes):
        4 bytes: magic   (0x4E4E5754 = "NNWT")
        4 bytes: version (2)
        4 bytes: num_layers  (total linear layers = num_hidden_layers + 1)
        4 bytes: input_dim
        4 bytes: hidden_dim
        4 bytes: output_dim

    Normalization parameters:
        input_dim  × float32: x_means
        input_dim  × float32: x_stds
        output_dim × float32: y_means
        output_dim × float32: y_stds

    For each layer i in 0..num_layers-1:
        fan_in × fan_out × float32: weights  (stored [fan_out, fan_in])
        fan_out × float32: biases

    Feature bounds (v2+):
        input_dim × float32: x_mins
        input_dim × float32: x_maxs

Usage:
    python export_weights.py [--input weights/model.pt] [--output weights/model.nnwt]
"""

import sys
import struct
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neural_net.core.model import CompressionPredictor


MAGIC = 0x4E4E5754  # "NNWT"
VERSION = 2


def export_weights(model_path: str, output_path: str):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Read architecture from checkpoint
    input_dim        = int(checkpoint.get('input_dim',        8))
    hidden_dim       = int(checkpoint.get('hidden_dim',       64))
    output_dim       = int(checkpoint.get('output_dim',       8))
    num_hidden_layers = int(checkpoint.get('num_hidden_layers', 4))
    num_layers       = num_hidden_layers + 1  # hidden layers + output layer

    print(f"  Architecture from checkpoint: {input_dim} → " +
          " → ".join([str(hidden_dim)] * num_hidden_layers) +
          f" → {output_dim}  ({num_layers} linear layers)")

    model = CompressionPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Normalization parameters
    x_means = checkpoint['x_means'].astype(np.float32)
    x_stds  = checkpoint['x_stds'].astype(np.float32)
    y_means = checkpoint['y_means'].astype(np.float32)
    y_stds  = checkpoint['y_stds'].astype(np.float32)

    assert len(x_means) == input_dim,  f"x_means length {len(x_means)} != input_dim {input_dim}"
    assert len(y_means) == output_dim, f"y_means length {len(y_means)} != output_dim {output_dim}"

    # Feature bounds (v2+)
    if 'x_mins' in checkpoint and 'x_maxs' in checkpoint:
        x_mins = checkpoint['x_mins'].astype(np.float32)
        x_maxs = checkpoint['x_maxs'].astype(np.float32)
    else:
        x_mins = np.full(input_dim, -1e30, dtype=np.float32)
        x_maxs = np.full(input_dim,  1e30, dtype=np.float32)
        print("  Warning: checkpoint missing x_mins/x_maxs, using fallback bounds")

    # Extract layers: net.0, net.2, net.4, ..., net.{2*num_hidden_layers}
    # The Sequential has ReLU at odd indices, Linear at even indices.
    layers = []
    state = model.state_dict()
    for i in range(num_layers):
        key = f'net.{i * 2}'
        w = state[f'{key}.weight'].numpy().astype(np.float32)
        b = state[f'{key}.bias'].numpy().astype(np.float32)
        layers.append((w, b))
        print(f"  Layer {i}: weight {w.shape}, bias {b.shape}")

    total_params = sum(w.size + b.size for w, b in layers)
    print(f"  Total parameters: {total_params:,}")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', num_layers))
        f.write(struct.pack('<I', input_dim))
        f.write(struct.pack('<I', hidden_dim))
        f.write(struct.pack('<I', output_dim))

        # Normalization
        f.write(x_means.tobytes())
        f.write(x_stds.tobytes())
        f.write(y_means.tobytes())
        f.write(y_stds.tobytes())

        # Layer weights and biases
        for w, b in layers:
            f.write(w.tobytes())
            f.write(b.tobytes())

        # Feature bounds (v2)
        f.write(x_mins.tobytes())
        f.write(x_maxs.tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"\n  Exported to: {output_path} ({file_size} bytes, {file_size/1024:.1f} KB)")

    verify_export(output_path, model, x_means, x_stds, y_means, y_stds, x_mins, x_maxs,
                  input_dim, hidden_dim, output_dim, num_layers)


def verify_export(path: str, model, x_means, x_stds, y_means, y_stds,
                  x_mins, x_maxs, input_dim, hidden_dim, output_dim, num_layers):
    """Verify exported binary matches PyTorch model."""
    with open(path, 'rb') as f:
        magic, version, r_num_layers, r_input_dim, r_hidden_dim, r_output_dim = \
            struct.unpack('<6I', f.read(24))
        assert magic == MAGIC,              f"Bad magic: {magic:#x}"
        assert version == VERSION,          f"Bad version: {version}"
        assert r_num_layers == num_layers,  f"num_layers mismatch: {r_num_layers} vs {num_layers}"
        assert r_input_dim  == input_dim,   f"input_dim mismatch: {r_input_dim} vs {input_dim}"
        assert r_hidden_dim == hidden_dim,  f"hidden_dim mismatch: {r_hidden_dim} vs {hidden_dim}"
        assert r_output_dim == output_dim,  f"output_dim mismatch: {r_output_dim} vs {output_dim}"

        xm = np.frombuffer(f.read(input_dim  * 4), dtype=np.float32)
        xs = np.frombuffer(f.read(input_dim  * 4), dtype=np.float32)
        ym = np.frombuffer(f.read(output_dim * 4), dtype=np.float32)
        ys = np.frombuffer(f.read(output_dim * 4), dtype=np.float32)

        assert np.allclose(xm, x_means), "x_means mismatch"
        assert np.allclose(xs, x_stds),  "x_stds mismatch"
        assert np.allclose(ym, y_means), "y_means mismatch"
        assert np.allclose(ys, y_stds),  "y_stds mismatch"

        # Build layer dimensions: input→hidden, hidden→hidden, ..., hidden→output
        fan_ins  = [input_dim]  + [hidden_dim] * (num_layers - 1)
        fan_outs = [hidden_dim] * (num_layers - 1) + [output_dim]

        for i, (fan_in, fan_out) in enumerate(zip(fan_ins, fan_outs)):
            w = np.frombuffer(f.read(fan_out * fan_in * 4), dtype=np.float32).reshape(fan_out, fan_in)
            b = np.frombuffer(f.read(fan_out * 4), dtype=np.float32)
            key = f'net.{i * 2}'
            expected_w = model.state_dict()[f'{key}.weight'].numpy()
            expected_b = model.state_dict()[f'{key}.bias'].numpy()
            assert np.allclose(w, expected_w), f"Layer {i} weight mismatch"
            assert np.allclose(b, expected_b), f"Layer {i} bias mismatch"

        read_mins = np.frombuffer(f.read(input_dim * 4), dtype=np.float32)
        read_maxs = np.frombuffer(f.read(input_dim * 4), dtype=np.float32)
        assert np.allclose(read_mins, x_mins), "x_mins mismatch"
        assert np.allclose(read_maxs, x_maxs), "x_maxs mismatch"

        remaining = f.read()
        assert len(remaining) == 0, f"Extra {len(remaining)} bytes at end"

    # Numerical test: manual forward pass
    test_input = torch.randn(1, input_dim)
    test_input_norm = (test_input - torch.from_numpy(xm)) / \
                      torch.from_numpy(np.clip(xs, 1e-8, None).astype(np.float32))
    with torch.no_grad():
        expected_norm = model(test_input_norm).numpy()[0]
    expected = expected_norm * ys + ym

    x = (test_input.numpy()[0] - xm) / np.clip(xs, 1e-8, None)
    fan_ins  = [input_dim]  + [hidden_dim] * (num_layers - 1)
    fan_outs = [hidden_dim] * (num_layers - 1) + [output_dim]
    with open(path, 'rb') as f2:
        f2.read(24)                                        # header
        f2.read((input_dim + input_dim + output_dim + output_dim) * 4)  # norm params
        for i, (fan_in, fan_out) in enumerate(zip(fan_ins, fan_outs)):
            w = np.frombuffer(f2.read(fan_out * fan_in * 4), dtype=np.float32).reshape(fan_out, fan_in)
            b = np.frombuffer(f2.read(fan_out * 4), dtype=np.float32)
            x = w @ x + b
            if i < num_layers - 1:
                x = np.maximum(x, 0)  # ReLU
    manual = x * ys + ym
    max_diff = np.max(np.abs(manual - expected))
    assert max_diff < 1e-3, f"Manual forward pass mismatch: max_diff={max_diff}"

    print(f"  Verification PASSED (manual forward pass max_diff={max_diff:.2e})")
    print(f"  Test inference: {expected}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export NN weights to binary')
    default_weights = str(Path(__file__).parent.parent / 'weights')
    parser.add_argument('--input',  default=f'{default_weights}/model.pt',   help='Input .pt path')
    parser.add_argument('--output', default=f'{default_weights}/model.nnwt', help='Output .nnwt path')
    args = parser.parse_args()

    print(f"Exporting model from {args.input}...")
    export_weights(args.input, args.output)
