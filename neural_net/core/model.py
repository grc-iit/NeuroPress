"""
Neural network model for compression performance prediction.

Architecture: 15 inputs → 128 → 128 → 4 outputs
Predicts: compression_time, decompression_time, ratio, psnr
"""

import torch
import torch.nn as nn


class CompressionPredictor(nn.Module):
    """
    Multi-output regression model.

    Given (algorithm_onehot, quantization, shuffle, error_bound,
           data_size, entropy, mad, second_derivative)
    predicts (compression_time, decompression_time, ratio, psnr).
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, output_dim: int = 4,
                 model_variant: str = "shared", num_hidden_layers: int = 2,
                 head_hidden_dim: int = 64):
        super().__init__()
        self.model_variant = model_variant
        self.output_dim = output_dim

        if model_variant == "shared":
            if num_hidden_layers < 1:
                raise ValueError("num_hidden_layers must be >= 1")
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
            self.trunk = None
            self.heads = None
        elif model_variant == "split_heads":
            # Shared trunk for generic representation + per-output heads for specialization.
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(output_dim)
            ])
            self.net = None
        else:
            raise ValueError(f"Unsupported model_variant={model_variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_variant == "shared":
            return self.net(x)
        h = self.trunk(x)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
