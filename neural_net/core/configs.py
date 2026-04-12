"""Configuration space constants and action encoding for the compression predictor."""

ALGORITHM_NAMES = ['lz4', 'snappy', 'deflate', 'gdeflate',
                   'zstd', 'ans', 'cascaded', 'bitcomp']

SHUFFLE_OPTIONS = [0, 4]

QUANT_OPTIONS = [
    (False, 0.0),      # lossless
    (True, 0.1),
    (True, 0.01),
    (True, 0.001),
]

NUM_ALGORITHMS = 8
NUM_CONFIGS_NN = 32    # 8 algo × 2 quant × 2 shuffle (NN action space)
NUM_CONFIGS_FULL = 64  # 8 algo × 2 shuffle × 4 quant (training space)


def build_all_config_features(entropy, mad, second_derivative,
                               data_size, error_bounds):
    """Build feature vectors for all 64 training configs.

    Args:
        entropy, mad, second_derivative: Data statistics.
        data_size: Original data size in bytes.
        error_bounds: Ignored (kept for API compat). Lossless configs
            always use sentinel 1e-7; lossy configs use QUANT_OPTIONS.

    Returns:
        (rows, configs) where:
            rows: list of 64 feature vectors (list[float], length 15 each)
            configs: list of 64 (algo_name, quant_str, shuffle, error_bound) tuples
    """
    import math

    rows = []
    configs = []
    for algo_idx, algo_name in enumerate(ALGORITHM_NAMES):
        for shuffle in SHUFFLE_OPTIONS:
            for quant, eb in QUANT_OPTIONS:
                quant_enc = 1.0 if quant else 0.0
                shuffle_enc = 1.0 if shuffle > 0 else 0.0
                eb_val = eb if quant else 1e-7
                error_bound_enc = eb_val
                data_size_enc = float(data_size)

                feature_vec = [
                    float(algo_idx), quant_enc, shuffle_enc, error_bound_enc,
                    data_size_enc, entropy, mad, second_derivative
                ]
                rows.append(feature_vec)
                quant_str = 'linear' if quant else 'none'
                configs.append((algo_name, quant_str, shuffle, eb))
    return rows, configs
