"""
RL Configuration for Q-Table Training

Defines hyperparameters, state/action spaces, and reward presets.
"""

# ============================================================
# State Space
# ============================================================

NUM_ENTROPY_BINS = 10      # Entropy discretized to 0-9
NUM_ERROR_LEVELS = 3       # 0=aggressive, 1=balanced, 2=precise
NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS  # 30 states

# Error bound thresholds for levels
ERROR_LEVEL_THRESHOLDS = [0.01, 0.001]  # Level 0 >= 0.01, Level 1 >= 0.001, Level 2 < 0.001

# ============================================================
# Action Space
# ============================================================

NUM_ALGORITHMS = 8         # LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp
NUM_QUANT_OPTIONS = 2      # None, Linear
NUM_SHUFFLE_OPTIONS = 2    # None, 4-byte
NUM_ACTIONS = NUM_ALGORITHMS * NUM_QUANT_OPTIONS * NUM_SHUFFLE_OPTIONS  # 32 actions

ALGORITHM_NAMES = [
    'lz4', 'snappy', 'deflate', 'gdeflate',
    'zstd', 'ans', 'cascaded', 'bitcomp'
]

# ============================================================
# Hyperparameters
# ============================================================

LEARNING_RATE = 0.1        # Alpha - Q-learning update rate
DISCOUNT_FACTOR = 0.0      # Gamma - no discount (single-step decision)

EPSILON_START = 1.0        # Initial exploration rate
EPSILON_END = 0.01         # Final exploration rate
EPSILON_DECAY = 0.995      # Decay per epoch

# ============================================================
# Training Settings
# ============================================================

DEFAULT_EPOCHS = 100       # Number of training epochs
CHECKPOINT_INTERVAL = 10   # Save every N epochs
VALIDATION_SPLIT = 0.2     # Fraction for validation

# ============================================================
# Reward Presets
# ============================================================

REWARD_PRESETS = {
    'balanced': {
        'ratio': 0.4,
        'throughput': 0.3,
        'psnr': 0.3
    },
    'max_ratio': {
        'ratio': 0.8,
        'throughput': 0.1,
        'psnr': 0.1
    },
    'max_speed': {
        'ratio': 0.1,
        'throughput': 0.8,
        'psnr': 0.1
    },
    'max_quality': {
        'ratio': 0.1,
        'throughput': 0.1,
        'psnr': 0.8
    },
    'storage': {
        'ratio': 0.6,
        'throughput': 0.2,
        'psnr': 0.2
    },
    'streaming': {
        'ratio': 0.3,
        'throughput': 0.5,
        'psnr': 0.2
    }
}

# Normalization constants for reward computation
MAX_RATIO = 10.0           # Normalize compression ratio
MAX_THROUGHPUT = 5000.0    # Normalize throughput (MB/s)
MAX_PSNR = 100.0           # Normalize PSNR (dB)
