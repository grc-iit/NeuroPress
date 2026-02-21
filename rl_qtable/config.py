"""
RL Configuration for Q-Table Training

Defines hyperparameters, state/action spaces, and reward presets.
"""

# ============================================================
# State Space
# ============================================================

NUM_ENTROPY_BINS = 16      # Byte entropy range [0, 8), 0.5-width bins
NUM_ERROR_LEVELS = 4       # 0=aggressive, 1=balanced, 2=precise, 3=lossless
NUM_MAD_BINS = 4           # Mean Absolute Deviation bins
NUM_DERIV_BINS = 4         # Second derivative (smoothness) bins
NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS * NUM_MAD_BINS * NUM_DERIV_BINS  # 1024 states

# Error bound thresholds for levels
# Level 0: error_bound >= 0.1     (aggressive lossy)
# Level 1: error_bound >= 0.01    (moderate lossy)
# Level 2: error_bound >= 0.001   (precise lossy)
# Level 3: error_bound <= 0       (lossless, no quantization)
ERROR_LEVEL_THRESHOLDS = [0.1, 0.01, 0.001]

# MAD bin thresholds (3 thresholds → 4 bins)
# Bin 0: MAD < 0.05   (very clustered data)
# Bin 1: MAD < 0.15   (moderately clustered)
# Bin 2: MAD < 0.30   (spread out)
# Bin 3: MAD >= 0.30  (highly variable)
MAD_BIN_THRESHOLDS = [0.05, 0.15, 0.30]

# Second derivative bin thresholds (3 thresholds → 4 bins)
# Normalized by data range, so values are in [0, 1]
# Bin 0: deriv < 0.02  (very smooth, slow curvature)
# Bin 1: deriv < 0.10  (moderate curvature)
# Bin 2: deriv < 0.30  (frequent curvature changes)
# Bin 3: deriv >= 0.30 (noisy / random)
DERIV_BIN_THRESHOLDS = [0.02, 0.10, 0.30]

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
        'ratio': 0.5,
        'throughput': 0.3,
        'psnr': 0.2
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
