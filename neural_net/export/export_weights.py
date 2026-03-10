"""
Export trained PyTorch model weights to binary format for CUDA inference.

Binary format (little-endian):
    Header (24 bytes):
        4 bytes: magic   (0x4E4E5754 = "NNWT")
        4 bytes: version (2)
        4 bytes: num_layers (3)
        4 bytes: input_dim  (15)
        4 bytes: hidden_dim (128)
        4 bytes: output_dim (4)

    Normalization parameters:
        15 × float32: x_means   (input feature means)
        15 × float32: x_stds    (input feature stds)
         4 × float32: y_means   (output means)
         4 × float32: y_stds    (output stds)

    Layer 1 (15 → 128):
        15 × 128 × float32: weights (row-major, PyTorch stores [out, in])
        128 × float32: biases

    Layer 2 (128 → 128):
        128 × 128 × float32: weights
        128 × float32: biases

    Layer 3 (128 → 4):
        128 × 4 × float32: weights
        4 × float32: biases

    Feature bounds (v2+, for OOD detection):
        15 × float32: x_mins   (per-feature training min)
        15 × float32: x_maxs   (per-feature training max)

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

    # Load model
    model = CompressionPredictor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract normalization parameters
    x_means = checkpoint['x_means'].astype(np.float32)
    x_stds = checkpoint['x_stds'].astype(np.float32)
    y_means = checkpoint['y_means'].astype(np.float32)
    y_stds = checkpoint['y_stds'].astype(np.float32)

    # Extract feature bounds (v2+)
    if 'x_mins' in checkpoint and 'x_maxs' in checkpoint:
        x_mins = checkpoint['x_mins'].astype(np.float32)
        x_maxs = checkpoint['x_maxs'].astype(np.float32)
    else:
        # Fallback: set wide bounds (no OOD detection)
        x_mins = np.full(15, -1e30, dtype=np.float32)
        x_maxs = np.full(15, 1e30, dtype=np.float32)
        print("  Warning: model.pt missing x_mins/x_maxs, using fallback bounds")

    # Extract layer weights and biases
    # PyTorch Linear stores weight as [out_features, in_features]
    # For CUDA we want [in_features, out_features] (column-major for input perspective)
    # But since each thread does its own dot product, we'll keep [out, in] and iterate
    # Actually, keep as [out_features, in_features] - the CUDA kernel will read
    # weight[out_idx * in_dim + in_idx] for each output neuron
    layers = []
    for i, layer_name in enumerate(['net.0', 'net.2', 'net.4']):
        w = model.state_dict()[f'{layer_name}.weight'].numpy().astype(np.float32)
        b = model.state_dict()[f'{layer_name}.bias'].numpy().astype(np.float32)
        layers.append((w, b))
        print(f"  Layer {i}: weight {w.shape}, bias {b.shape}")

    input_dim = layers[0][0].shape[1]
    hidden_dim = layers[0][0].shape[0]
    output_dim = layers[2][0].shape[0]

    print(f"\n  Architecture: {input_dim} → {hidden_dim} → {hidden_dim} → {output_dim}")
    print(f"  Total parameters: {sum(w.size + b.size for w, b in layers)}")

    # Write binary file
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(layers)))
        f.write(struct.pack('<I', input_dim))
        f.write(struct.pack('<I', hidden_dim))
        f.write(struct.pack('<I', output_dim))

        # Normalization parameters
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

    # Verify by reading back
    verify_export(output_path, model, x_means, x_stds, y_means, y_stds, x_mins, x_maxs)


def verify_export(path: str, model, x_means, x_stds, y_means, y_stds,
                  x_mins=None, x_maxs=None):
    """Verify exported binary matches PyTorch model."""
    with open(path, 'rb') as f:
        magic, version, n_layers, in_dim, hid_dim, out_dim = struct.unpack('<6I', f.read(24))
        assert magic == MAGIC, f"Bad magic: {magic:#x}"
        assert version == VERSION
        assert n_layers == 3
        assert in_dim == 15
        assert hid_dim == 128
        assert out_dim == 4

        xm = np.frombuffer(f.read(15 * 4), dtype=np.float32)
        xs = np.frombuffer(f.read(15 * 4), dtype=np.float32)
        ym = np.frombuffer(f.read(4 * 4), dtype=np.float32)
        ys = np.frombuffer(f.read(4 * 4), dtype=np.float32)

        assert np.allclose(xm, x_means), "x_means mismatch"
        assert np.allclose(xs, x_stds), "x_stds mismatch"
        assert np.allclose(ym, y_means), "y_means mismatch"
        assert np.allclose(ys, y_stds), "y_stds mismatch"

        # Read and verify each layer
        dims = [(15, 128), (128, 128), (128, 4)]
        for i, (fan_in, fan_out) in enumerate(dims):
            w = np.frombuffer(f.read(fan_out * fan_in * 4), dtype=np.float32).reshape(fan_out, fan_in)
            b = np.frombuffer(f.read(fan_out * 4), dtype=np.float32)

            layer_name = f'net.{i * 2}'
            expected_w = model.state_dict()[f'{layer_name}.weight'].numpy()
            expected_b = model.state_dict()[f'{layer_name}.bias'].numpy()

            assert np.allclose(w, expected_w), f"Layer {i} weight mismatch"
            assert np.allclose(b, expected_b), f"Layer {i} bias mismatch"

        # Read and verify feature bounds (v2)
        if version >= 2:
            read_mins = np.frombuffer(f.read(15 * 4), dtype=np.float32)
            read_maxs = np.frombuffer(f.read(15 * 4), dtype=np.float32)
            if x_mins is not None:
                assert np.allclose(read_mins, x_mins), "x_mins mismatch"
            if x_maxs is not None:
                assert np.allclose(read_maxs, x_maxs), "x_maxs mismatch"

        remaining = f.read()
        assert len(remaining) == 0, f"Extra {len(remaining)} bytes at end"

    # Numerical test: manual forward pass with exported weights vs PyTorch reference output
    test_input = torch.randn(1, 15)
    test_input_norm = (test_input - torch.from_numpy(xm)) / torch.from_numpy(np.clip(xs, 1e-8, None).astype(np.float32))
    with torch.no_grad():
        expected_output_norm = model(test_input_norm).numpy()[0]
    expected_output = expected_output_norm * ys + ym

    # Manual forward pass using exported numpy weights to verify correctness
    x = (test_input.numpy()[0] - xm) / np.clip(xs, 1e-8, None)
    dims = [(15, 128), (128, 128), (128, 4)]
    with open(path, 'rb') as f2:
        f2.read(24)  # skip header
        f2.read((15 + 15 + 4 + 4) * 4)  # skip norm params
        for i, (fan_in, fan_out) in enumerate(dims):
            w = np.frombuffer(f2.read(fan_out * fan_in * 4), dtype=np.float32).reshape(fan_out, fan_in)
            b = np.frombuffer(f2.read(fan_out * 4), dtype=np.float32)
            x = w @ x + b
            if i < len(dims) - 1:
                x = np.maximum(x, 0)  # ReLU
    manual_output = x * ys + ym
    max_diff = np.max(np.abs(manual_output - expected_output))
    assert max_diff < 1e-5, f"Manual forward pass mismatch: max_diff={max_diff}"

    print(f"  Verification PASSED (manual forward pass max_diff={max_diff:.2e})")
    print(f"  Test inference: {expected_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export NN weights to binary')
    default_weights = str(Path(__file__).parent.parent / 'weights')
    parser.add_argument('--input', default=f'{default_weights}/model.pt', help='Input model path')
    parser.add_argument('--output', default=f'{default_weights}/model.nnwt', help='Output binary path')
    args = parser.parse_args()

    print(f"Exporting model from {args.input}...")
    export_weights(args.input, args.output)
