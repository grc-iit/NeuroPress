"""
Reward Computation for Q-Table Training

Computes rewards based on compression metrics (ratio, throughput, PSNR).
Supports configurable presets for different optimization goals.
"""

import math
from typing import Dict, Optional

from .config import REWARD_PRESETS, MAX_RATIO, MAX_THROUGHPUT, MAX_PSNR


def compute_reward(
    ratio: float,
    throughput_mbps: float,
    psnr_db: Optional[float] = None,
    preset: str = 'balanced'
) -> float:
    """
    Compute reward for a compression result.

    Args:
        ratio: Compression ratio (original_size / compressed_size)
        throughput_mbps: Compression throughput in MB/s
        psnr_db: Peak Signal-to-Noise Ratio in dB (None for lossless)
        preset: Reward preset name (see REWARD_PRESETS)

    Returns:
        Reward value (0.0 to 1.0)
    """
    if preset not in REWARD_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(REWARD_PRESETS.keys())}")

    weights = REWARD_PRESETS[preset]

    # Normalize metrics to [0, 1]
    ratio_norm = min(ratio / MAX_RATIO, 1.0)
    throughput_norm = min(throughput_mbps / MAX_THROUGHPUT, 1.0)

    if psnr_db is None or math.isinf(psnr_db):
        # Lossless: perfect quality
        psnr_norm = 1.0
    else:
        psnr_norm = min(psnr_db / MAX_PSNR, 1.0)

    # Weighted sum
    reward = (
        weights['ratio'] * ratio_norm +
        weights['throughput'] * throughput_norm +
        weights['psnr'] * psnr_norm
    )

    return reward


def compute_reward_from_metrics(metrics: Dict, preset: str = 'balanced') -> float:
    """
    Compute reward from a metrics dictionary.

    Args:
        metrics: Dictionary with 'ratio', 'throughput_mbps', and optionally 'psnr_db'
        preset: Reward preset name

    Returns:
        Reward value (0.0 to 1.0)
    """
    return compute_reward(
        ratio=metrics.get('ratio', 1.0),
        throughput_mbps=metrics.get('throughput_mbps', 0.0),
        psnr_db=metrics.get('psnr_db', None),
        preset=preset
    )


def get_preset_description(preset: str) -> str:
    """Get human-readable description of a reward preset."""
    descriptions = {
        'balanced': "Balanced optimization of ratio, speed, and quality",
        'max_ratio': "Maximize compression ratio (smallest files)",
        'max_speed': "Maximize throughput (fastest compression)",
        'max_quality': "Maximize quality (minimize distortion)",
        'storage': "Optimize for archival storage (prioritize ratio)",
        'streaming': "Optimize for streaming (prioritize speed, then ratio)"
    }
    return descriptions.get(preset, "Unknown preset")
