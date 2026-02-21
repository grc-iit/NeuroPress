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

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
