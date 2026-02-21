#!/usr/bin/env python3
"""
Synthetic Data Generator for GPU Compression Benchmarking

Palette-based 32-bin system adapted from newScript.cc.
Generates datasets with controlled entropy characteristics for compression
algorithm training and evaluation.

Key controls:
  - palette:      Bin weight distribution (7 types)
  - perturbation: Spatial locality (0.0=long runs, 1.0=fully random)
  - fill_mode:    Value pattern within bins (5 types)
  - bin_width:    Bin value range in normalized [0,1] space

Commands:
  generate   Generate a single synthetic dataset
  batch      Generate dataset collection for benchmarking
  stats      Generate statistics CSV (no files saved)
  info       Display file information and statistics

generate -- Create a single dataset file
  -p, --palette TEXT       Bin weight distribution [uniform|normal|gamma|
                           exponential|bimodal|grayscott|high_entropy]
                           (default: uniform)
  --perturbation FLOAT     Spatial locality: 0.0=long runs, 1.0=random
                           (default: 0.5)
  -f, --fill-mode TEXT     Value pattern within bins [constant|linear|
                           quadratic|sinusoidal|random] (default: random)
  -w, --bin-width FLOAT    Absolute bin width in value space (default: 1.0)
  -d, --dtype TEXT         Data type [float32|uint8|int32] (default: float32)
  -s, --size TEXT          Dataset size, e.g. 4MB, 1GB (default: 4MB)
  --seed INT               Random seed
  -o, --output PATH        Output file path (.bin or .h5)  [required]
  -q, --quiet              Suppress output

batch -- Generate full dataset collection
  -o, --output-dir PATH    Output directory (default: datasets)
  -s, --size TEXT          Size(s) per dataset, comma-separated
                           (e.g. 128KB or 16KB,256KB,1MB)
                           Training default: 16KB,64KB,256KB,1MB,4MB
  -m, --mode TEXT          Batch mode [training|comprehensive]
                           (default: training)
                             training:      7 palettes x 7 widths x 8 perts
                                            x 5 fills x 5 sizes = 9800/dtype
                             comprehensive: 7 palettes x 7 widths x 8 perts
                                            x 5 fills = 1960 per dtype
  -d, --dtypes TEXT        Comma-separated dtypes (default: float32 for
                           training, all 3 for comprehensive)
  -r, --repeats INT        Repeats per combo with different seeds (default: 1)
  --format TEXT            Output format [bin|h5] (default: bin)
  -t, --threads INT        Worker processes (default: CPU count)
  -q, --quiet              Suppress output

stats -- Compute statistics CSV without saving data files
  -o, --output PATH        Output CSV path (default: compression_stats.csv)
  -t, --threads INT        Thread count (default: 1)
  -m, --mode TEXT          Config mode [comprehensive|training]
                           (default: comprehensive)
  -v, --verbose            Print progress

info -- Display file information and statistics
  FILENAME                 Path to .bin or .h5 file

Examples:
  python generator.py generate -p normal --perturbation 0.0 -o low.bin
  python generator.py generate -p high_entropy -f random -s 1MB -o he.bin
  python generator.py batch -o training_data/ --mode training -t 8
  python generator.py batch -o datasets/ --mode comprehensive -s 1MB -t 16
  python generator.py stats -o stats.csv -t 4 -v
  python generator.py info training_data/float32_normal_w100_p0_const.bin
"""

import click
import numpy as np
import os
import sys
import csv
import math
import random as _stdlib_random
from typing import Tuple, Dict, Any, Union, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading


# ============================================================
# Constants (matching newScript.cc)
# ============================================================

N_BINS = 32

PALETTES = ['uniform', 'normal', 'gamma', 'exponential',
            'bimodal', 'grayscott', 'high_entropy']

FILL_MODES = ['constant', 'linear', 'quadratic', 'sinusoidal', 'random']

DTYPES = ['float32', 'uint8', 'int32']

# Type-specific value ranges (matching newScript.cc ScaleBinsForType)
_TYPE_CONFIG = {
    'float32': {'np_dtype': np.float32, 'scale': (0.0, 1.0),   'clamp': None},
    'uint8':   {'np_dtype': np.uint8,   'scale': (0.0, 255.0),  'clamp': (0.0, 255.0)},
    'int32':   {'np_dtype': np.int32,   'scale': (-1e6, 1e6),   'clamp': (-2e9, 2e9)},
}

# Configurations matching newScript.cc GenerateDistributions()
COMPREHENSIVE_BIN_WIDTHS = [0.1, 0.25, 0.5, 1.0, 4.0, 16.0, 64.0]
COMPREHENSIVE_PERTURBATIONS = [0.0, 0.05, 0.1, 0.2, 0.325, 0.5, 0.75, 0.9]

TRAINING_BIN_WIDTHS = [0.1, 0.12, 0.15, 0.25, 0.5, 1.0, 16.0]
TRAINING_PERTURBATIONS = [0.0, 0.1, 0.2, 0.325, 0.5, 0.75, 0.95, 1.0]
TRAINING_FILL_MODES = ['constant', 'linear', 'quadratic', 'sinusoidal', 'random']
TRAINING_SIZES = ['16KB', '64KB', '256KB', '1MB', '4MB']


# ============================================================
# Bin weight distributions (matching newScript.cc PaletteGenerator)
# ============================================================

def _uniform_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def _normal_weights(n: int) -> np.ndarray:
    x = np.linspace(-3, 3, n)
    w = np.exp(-0.5 * x**2)
    return w / w.sum()


def _gamma_weights(n: int) -> np.ndarray:
    x = np.linspace(0.1, 6.0, n)
    w = x * np.exp(-x)  # shape=2 gamma
    return w / w.sum()


def _exponential_weights(n: int) -> np.ndarray:
    w = np.zeros(n)
    w[0] = 0.9999
    decay = np.power(0.5, np.arange(1, n).astype(float))
    w[1:] = decay / decay.sum() * 0.0001
    return w


def _bimodal_weights(n: int) -> np.ndarray:
    if n < 4:
        return _uniform_weights(n)
    w = np.full(n, 0.2 / max(n - 4, 1))
    w[0], w[1] = 0.24, 0.16
    w[-2], w[-1] = 0.16, 0.24
    return w / w.sum()


def _grayscott_weights(n: int) -> np.ndarray:
    """Grayscott-like: background (70%) + spots (20%) + edges (10%)."""
    w = np.zeros(n)
    bg = max(2, n // 4)
    sp = max(2, n // 4)

    for i in range(bg):
        t = i / bg
        w[i] = 0.7 * np.exp(-2 * t * t)
    for i in range(sp):
        t = i / sp
        w[n - 1 - i] = 0.2 * np.exp(-2 * t * t)

    edge_start, edge_end = bg, n - sp
    if edge_end > edge_start:
        w[edge_start:edge_end] = 0.1 / (edge_end - edge_start)

    return w / w.sum()


def _high_entropy_weights(n: int) -> np.ndarray:
    """Uniform weights -- entropy comes from non-linear bin spacing."""
    return np.ones(n) / n


_WEIGHT_FUNCS = {
    'uniform': _uniform_weights,
    'normal': _normal_weights,
    'gamma': _gamma_weights,
    'exponential': _exponential_weights,
    'bimodal': _bimodal_weights,
    'grayscott': _grayscott_weights,
    'high_entropy': _high_entropy_weights,
}


# ============================================================
# Bin layout (matching newScript.cc CreateBinConfigs)
# ============================================================

def _standard_bins(n: int, bin_width: float,
                   value_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Create bins matching newScript.cc CreateBinConfigs.

    When bin_width * n <= total_range: non-overlapping bins with gaps.
    When bin_width * n > total_range: overlapping bins, evenly started.
    """
    lo, hi = value_range
    total_range = hi - lo

    if bin_width * n <= total_range:
        gap = (total_range - bin_width * n) / (n - 1) if n > 1 else 0
        stride = bin_width + gap
        bin_lo = lo + np.arange(n) * stride
        bin_hi = np.minimum(bin_lo + bin_width, hi)
    else:
        step = total_range / n
        bin_lo = lo + np.arange(n) * step
        bin_hi = np.minimum(bin_lo + bin_width, hi)

    return bin_lo, bin_hi


def _high_entropy_bins(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Non-linear bins spanning orders of magnitude for maximum byte diversity.

    Matches newScript.cc CreateHighEntropyFloatBins.
    """
    bin_lo = np.empty(n)
    bin_hi = np.empty(n)
    for i in range(n):
        t = i / (n - 1)
        center = 10.0 ** (-6.0 + t * 6.0)
        width = center * 0.1
        bin_lo[i] = max(0.0, center - width / 2)
        bin_hi[i] = min(1.0, center + width / 2)
    return bin_lo, bin_hi


def _scale_bins_for_type(bin_lo: np.ndarray, bin_hi: np.ndarray,
                         dtype: str, palette: str) -> Tuple[np.ndarray, np.ndarray]:
    """Scale [0,1] bins to type-specific ranges (matches ScaleBinsForType<T>)."""
    if palette == 'high_entropy' and dtype == 'float32':
        return bin_lo, bin_hi

    if dtype == 'uint8':
        return bin_lo * 255.0, bin_hi * 255.0
    elif dtype == 'int32':
        return bin_lo * 2e6 - 1e6, bin_hi * 2e6 - 1e6
    else:  # float32
        return bin_lo, bin_hi


# ============================================================
# Core generator (matching newScript.cc GenerateBinnedData)
# ============================================================

def generate(
    shape: Union[int, Tuple[int, ...]],
    palette: str = 'uniform',
    perturbation: float = 0.5,
    fill_mode: str = 'random',
    bin_width: float = 1.0,
    value_range: Tuple[float, float] = (0.0, 1.0),
    dtype: str = 'float32',
    seed: int = None
) -> np.ndarray:
    """Generate data using palette-based 32-bin system.

    Matches newScript.cc GenerateBinnedData: burst-based sampling with
    perturbation-controlled spatial locality, vectorized for performance.

    Args:
        shape: Output array shape (int or tuple)
        palette: Bin weight distribution name
        perturbation: 0.0=long runs (compressible), 1.0=fully random
        fill_mode: Value pattern within bins (constant/linear/quadratic/sinusoidal/random)
        bin_width: Absolute bin width in value space
        value_range: Normalized value range for bins
        dtype: Output data type (float32/uint8/int32)
        seed: Random seed

    Returns:
        Array of specified dtype with controlled entropy characteristics
    """
    if isinstance(shape, int):
        shape = (shape,)
    num_elements = int(np.prod(shape))
    np_rng = np.random.RandomState(seed)

    if palette not in _WEIGHT_FUNCS:
        raise ValueError(f"Unknown palette: {palette}. Available: {PALETTES}")
    if fill_mode not in FILL_MODES:
        raise ValueError(f"Unknown fill_mode: {fill_mode}. Available: {FILL_MODES}")
    if dtype not in _TYPE_CONFIG:
        raise ValueError(f"Unknown dtype: {dtype}. Available: {DTYPES}")

    weights = _WEIGHT_FUNCS[palette](N_BINS)

    if palette == 'high_entropy':
        bin_lo, bin_hi = _high_entropy_bins(N_BINS)
    else:
        bin_lo, bin_hi = _standard_bins(N_BINS, bin_width, value_range)

    bin_lo, bin_hi = _scale_bins_for_type(bin_lo, bin_hi, dtype, palette)

    targets = (weights * num_elements).astype(int)
    targets[-1] = num_elements - targets[:-1].sum()

    # === Phase 1: Plan burst sequence (matches newScript.cc GenerateBinnedData) ===
    # Lightweight Python loop — only integer math, no numpy per iteration.
    # Uses stdlib random for fast scalar randrange vs numpy's per-call overhead.
    py_rng = _stdlib_random.Random(seed)
    burst_bins = []
    burst_sizes = []
    counts = [0] * N_BINS
    tgt = targets.tolist()
    active = [i for i in range(N_BINS) if tgt[i] > 0]
    pos = 0

    while pos < num_elements and active:
        pick = py_rng.randrange(len(active))
        bi = active[pick]
        rem = tgt[bi] - counts[bi]
        burst = max(1, int(rem * (1.0 - perturbation)))
        burst = min(burst, rem, num_elements - pos)
        burst_bins.append(bi)
        burst_sizes.append(burst)
        pos += burst
        counts[bi] += burst
        if counts[bi] >= tgt[bi]:
            # O(1) removal: swap with last and pop
            active[pick] = active[-1]
            active.pop()

    if not burst_bins:
        np_dtype = _TYPE_CONFIG[dtype]['np_dtype']
        return np.zeros(num_elements, dtype=np_dtype).reshape(shape)

    # === Phase 2: Build per-element bin index array (vectorized) ===
    bp_bins = np.array(burst_bins, dtype=np.int32)
    bp_sizes = np.array(burst_sizes, dtype=np.int64)
    bin_idx = np.repeat(bp_bins, bp_sizes)

    lo = bin_lo[bin_idx]
    hi = bin_hi[bin_idx]
    mid = (lo + hi) * 0.5

    # === Phase 3: Fill values by fill_mode (fully vectorized) ===
    data = np.empty(num_elements, dtype=np.float64)

    if fill_mode == 'constant':
        data[:] = lo

    elif fill_mode == 'random':
        same = (lo == hi)
        data[same] = lo[same]
        diff_mask = ~same
        n_diff = diff_mask.sum()
        if n_diff > 0:
            data[diff_mask] = (lo[diff_mask]
                               + np_rng.random_sample(n_diff)
                               * (hi[diff_mask] - lo[diff_mask]))

    else:
        # linear, quadratic, sinusoidal need within-burst position
        cum = np.cumsum(bp_sizes)
        starts = np.empty(len(bp_sizes), dtype=np.int64)
        starts[0] = 0
        if len(bp_sizes) > 1:
            starts[1:] = cum[:-1]
        offsets = np.repeat(starts, bp_sizes)
        within = (np.arange(num_elements, dtype=np.float64) - offsets)
        sizes_f = np.repeat(bp_sizes.astype(np.float64), bp_sizes)
        single = (sizes_f == 1.0)

        if fill_mode == 'linear':
            # t = j / (burst_size - 1), matching newScript.cc LINEAR
            t = within / np.maximum(sizes_f - 1.0, 1.0)
            data[:] = np.where(single, mid, lo + t * (hi - lo))

        elif fill_mode == 'quadratic':
            # Parabola matching newScript.cc QUADRATIC
            t = within / np.maximum(sizes_f - 1.0, 1.0)
            phase = t - 0.5
            span = hi - lo
            a = 4.0 * (span / 4.0) / 0.25
            vals = mid + a * phase * phase - span / 4.0
            data[:] = np.where(single, mid,
                               np.clip(vals, np.minimum(lo, hi),
                                       np.maximum(lo, hi)))

        elif fill_mode == 'sinusoidal':
            # t = j / burst_size (NOT burst_size-1), matching newScript.cc SINUSOIDAL
            t = within / np.maximum(sizes_f, 1.0)
            data[:] = np.where(single, mid,
                               mid + (hi - lo) * 0.5
                               * np.sin(2.0 * np.pi * t))

    np_dtype = _TYPE_CONFIG[dtype]['np_dtype']
    clamp = _TYPE_CONFIG[dtype]['clamp']
    if clamp is not None:
        data = np.clip(data, clamp[0], clamp[1])
    return data.astype(np_dtype).reshape(shape)


# ============================================================
# Statistics (matching data_stats.h CalculateAllStatistics)
# ============================================================

def calculate_byte_entropy(data: np.ndarray) -> float:
    """Calculate byte-level Shannon entropy (0-8 bits).

    Matches data_stats.h CalculateShannonEntropy.
    """
    byte_array = np.frombuffer(data.tobytes(), dtype=np.uint8)
    hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    probs = hist / len(byte_array)
    return float(-np.sum(probs * np.log2(probs)))


def calculate_mad(data: np.ndarray) -> float:
    """Calculate Mean Absolute Deviation.

    Matches data_stats.h CalculateMAD.
    """
    mean = np.mean(data, dtype=np.float64)
    return float(np.mean(np.abs(data.astype(np.float64) - mean)))


def calculate_first_derivative(data: np.ndarray) -> float:
    """Calculate mean absolute first derivative.

    Matches data_stats.h CalculateFirstDerivativeStatistics.
    Formula: mean(|data[i+1] - data[i]|)
    """
    if data.size < 2:
        return 0.0
    flat = data.flatten().astype(np.float64)
    return float(np.mean(np.abs(np.diff(flat))))


def calculate_second_derivative(data: np.ndarray) -> float:
    """Calculate mean absolute second derivative.

    Matches data_stats.h CalculateSecondDerivativeStatistics.
    Formula: mean(|data[i+1] - 2*data[i] + data[i-1]|)
    """
    if data.size < 3:
        return 0.0
    flat = data.flatten().astype(np.float64)
    second_deriv = flat[2:] - 2.0 * flat[1:-1] + flat[:-2]
    return float(np.mean(np.abs(second_deriv)))


def calculate_all_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate all statistics matching newScript.cc output.

    Returns dict with: shannon_entropy, mad, first_derivative, second_derivative
    """
    return {
        'shannon_entropy': calculate_byte_entropy(data),
        'mad': calculate_mad(data),
        'first_derivative': calculate_first_derivative(data),
        'second_derivative': calculate_second_derivative(data),
    }


# ============================================================
# Size bins (matching newScript.cc GenerateSizeBins)
# ============================================================

def generate_size_bins(statistics_only: bool = False) -> List[int]:
    """Generate size bins matching newScript.cc GenerateSizeBins."""
    if statistics_only:
        return [128 * 1024, 1 * 1024 * 1024]

    min_size = 1024
    max_size = 128 * 1024
    num_bins = 32

    log_min = math.log2(min_size)
    log_max = math.log2(max_size)
    log_step = (log_max - log_min) / (num_bins - 1)

    sizes = []
    for i in range(num_bins):
        size_bytes = int(2.0 ** (log_min + i * log_step))
        sizes.append(((size_bytes + 255) // 256) * 256)  # Align to 256 bytes

    sizes.append(1024 * 1024)       # 1MB
    sizes.append(4 * 1024 * 1024)   # 4MB

    return sizes


# ============================================================
# Binary I/O
# ============================================================

def write_binary(filename: str, data: np.ndarray) -> str:
    """Write numpy array as raw binary bytes (native dtype, no header)."""
    data.flatten().tofile(filename)
    return filename


def read_binary(filename: str, dtype=np.float32) -> np.ndarray:
    """Read raw binary file as 1D numpy array."""
    return np.fromfile(filename, dtype=dtype)


# ============================================================
# HDF5 I/O (optional, requires h5py)
# ============================================================

def write_hdf5(filename: str, data: np.ndarray,
               generation_params: Dict[str, Any] = None,
               dataset_name: str = 'data') -> str:
    """Write dataset to HDF5 with metadata. Requires h5py."""
    import h5py

    stats = calculate_all_statistics(data)

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset(dataset_name, data=data)

        if generation_params:
            for key, value in generation_params.items():
                if value is not None:
                    if isinstance(value, (list, tuple, np.ndarray)):
                        dset.attrs[f'param_{key}'] = str(value)
                    else:
                        dset.attrs[f'param_{key}'] = value

        for key, value in stats.items():
            dset.attrs[f'stat_{key}'] = value

        dset.attrs['timestamp'] = datetime.now().isoformat()
        dset.attrs['dtype'] = str(data.dtype)

    return filename


def read_hdf5(filename: str, dataset_name: str = 'data') -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read dataset and attributes from HDF5 file."""
    import h5py

    with h5py.File(filename, 'r') as f:
        data = f[dataset_name][:]
        attrs = dict(f[dataset_name].attrs)
    return data, attrs


# ============================================================
# Helpers
# ============================================================

def parse_size(size_str: str) -> int:
    """Parse size string like '4MB', '1GB' to bytes."""
    size_str = size_str.strip().upper()
    units = {'TB': 1024**4, 'GB': 1024**3, 'MB': 1024**2, 'KB': 1024, 'B': 1}
    for unit, mult in units.items():
        if size_str.endswith(unit):
            return int(float(size_str[:-len(unit)]) * mult)
    return int(size_str)


def fill_mode_short(mode: str) -> str:
    """Short name for fill mode (matching newScript.cc FillModeToString)."""
    return {'constant': 'const', 'linear': 'linear', 'quadratic': 'quad',
            'sinusoidal': 'sin', 'random': 'rand'}.get(mode, mode)


# ============================================================
# CLI
# ============================================================

@click.group()
def cli():
    """Synthetic Data Generator -- Palette-based 32-bin system."""
    pass


@cli.command('generate')
@click.option('--palette', '-p', type=click.Choice(PALETTES), default='uniform',
              help='Bin weight distribution')
@click.option('--perturbation', type=float, default=0.5,
              help='Spatial locality: 0.0=long runs, 1.0=random')
@click.option('--fill-mode', '-f', type=click.Choice(FILL_MODES), default='random',
              help='Value pattern within bins')
@click.option('--bin-width', '-w', type=float, default=1.0,
              help='Absolute bin width in value space')
@click.option('--dtype', '-d', type=click.Choice(DTYPES), default='float32',
              help='Data type')
@click.option('--size', '-s', type=str, default='4MB',
              help='Dataset size (e.g., 4MB, 1GB)')
@click.option('--seed', type=int, help='Random seed')
@click.option('--output', '-o', required=True, help='Output file (.bin or .h5)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cmd_generate(palette, perturbation, fill_mode, bin_width, dtype, size,
                 seed, output, quiet):
    """Generate a single synthetic dataset."""
    size_bytes = parse_size(size)
    dtype_bytes = np.dtype(_TYPE_CONFIG[dtype]['np_dtype']).itemsize
    num_elements = size_bytes // dtype_bytes

    if not quiet:
        click.echo(f"Generating: palette={palette} pert={perturbation} "
                    f"fill={fill_mode} width={bin_width} dtype={dtype}")
        click.echo(f"  Elements: {num_elements:,}  Size: {size}")

    data = generate(
        shape=(num_elements,), palette=palette, perturbation=perturbation,
        fill_mode=fill_mode, bin_width=bin_width, dtype=dtype, seed=seed
    )

    _save(output, data, palette, perturbation, fill_mode, bin_width)

    if not quiet:
        stats = calculate_all_statistics(data)
        click.echo(f"  Entropy: {stats['shannon_entropy']:.4f}  "
                    f"MAD: {stats['mad']:.6f}  "
                    f"d1: {stats['first_derivative']:.6f}  "
                    f"d2: {stats['second_derivative']:.6f}")
        click.echo(f"  Saved: {output}")


@cli.command()
@click.option('--output-dir', '-o', default='datasets', help='Output directory')
@click.option('--size', '-s', type=str, default=None,
              help='Size(s) per dataset, comma-separated (e.g. 128KB or 16KB,256KB,1MB). '
                   'Training mode defaults to TRAINING_SIZES if omitted.')
@click.option('--mode', '-m', type=click.Choice(['comprehensive', 'training']),
              default='training', help='Batch mode')
@click.option('--dtypes', '-d', type=str, default=None,
              help='Comma-separated data types (default: float32 for training, all for comprehensive)')
@click.option('--repeats', '-r', type=int, default=1,
              help='Repeats per combo with different seeds')
@click.option('--format', 'fmt', type=click.Choice(['bin', 'h5']), default='bin',
              help='Output format')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
@click.option('--threads', '-t', type=int, default=None,
              help='Worker processes for parallel generation (default: CPU count)')
def batch(output_dir, size, mode, dtypes, repeats, fmt, quiet, threads):
    """Generate dataset collection for benchmarking.

    \b
    Modes (matching newScript.cc):
      training:       7 palettes x 7 widths x 8 perturbations x 5 fills
                      x 5 sizes = 9800 per dtype (default sizes: 16KB..4MB)
      comprehensive:  7 palettes x 7 widths x 8 perturbations x 5 fills = 1960 per dtype
    \b
    Size can be a single value or comma-separated list:
      -s 128KB            Single size
      -s 16KB,256KB,4MB   Multiple sizes (encoded in filename)
    Training mode defaults to TRAINING_SIZES (16KB,64KB,256KB,1MB,4MB) if -s is omitted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse sizes: comma-separated list or single value
    if size is None:
        if mode == 'training':
            size_list = list(TRAINING_SIZES)
        else:
            size_list = ['4MB']
    else:
        size_list = [s.strip() for s in size.split(',')]

    size_bytes_list = [(s, parse_size(s)) for s in size_list]
    multi_size = len(size_bytes_list) > 1

    n_workers = threads or os.cpu_count() or 4

    if dtypes:
        dtype_list = [d.strip() for d in dtypes.split(',')]
    elif mode == 'comprehensive':
        dtype_list = DTYPES  # All 3 types
    else:
        dtype_list = ['float32']

    for d in dtype_list:
        if d not in DTYPES:
            click.echo(f"Unknown dtype: {d}. Available: {DTYPES}", err=True)
            sys.exit(1)

    if mode == 'comprehensive':
        bin_widths = COMPREHENSIVE_BIN_WIDTHS
        perturbations = COMPREHENSIVE_PERTURBATIONS
        fill_modes = FILL_MODES
    else:
        bin_widths = TRAINING_BIN_WIDTHS
        perturbations = TRAINING_PERTURBATIONS
        fill_modes = TRAINING_FILL_MODES

    combos = [(dt, p, w, pert, fm, sz_label, sz_bytes)
              for dt in dtype_list
              for p in PALETTES
              for w in bin_widths
              for pert in perturbations
              for fm in fill_modes
              for sz_label, sz_bytes in size_bytes_list]

    total = len(combos) * repeats

    if not quiet:
        size_display = ','.join(size_list)
        click.echo(f"Generating {total} datasets "
                    f"({len(combos)} combos x {repeats} repeats)")
        click.echo(f"  Data types:    {dtype_list}")
        click.echo(f"  Palettes:      {PALETTES}")
        click.echo(f"  Bin widths:    {bin_widths}")
        click.echo(f"  Perturbations: {perturbations}")
        click.echo(f"  Fill modes:    {fill_modes}")
        click.echo(f"  Sizes:         {size_display}")
        click.echo(f"  Workers:       {n_workers}")
        click.echo(f"  Output:        {output_dir}/\n")

    # Build work items
    work_items = []
    for i, (dt, pal, bw, pert, fm, sz_label, sz_bytes) in enumerate(combos):
        for r in range(repeats):
            seed = i * repeats + r + 1
            w_enc = int(round(bw * 100))
            p_enc = int(round(pert * 1000))
            base = f"{dt}_{pal}_w{w_enc}_p{p_enc}_{fill_mode_short(fm)}"
            if multi_size:
                base = f"{base}_{sz_label.lower()}"
            if repeats > 1:
                name = f"{base}_s{seed}"
            else:
                name = base
            filepath = os.path.join(output_dir, f"{name}.{fmt}")
            dtype_bytes = np.dtype(_TYPE_CONFIG[dt]['np_dtype']).itemsize
            num_elements = sz_bytes // dtype_bytes
            work_items.append((filepath, num_elements, pal, pert, fm, bw, dt, seed))

    # Generate files in parallel
    import time
    start_time = time.monotonic()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_generate_one_file, *item): item
                   for item in work_items}
        for future in as_completed(futures):
            name, ent = future.result()
            completed += 1
            if not quiet:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                click.echo(f"\r  [{completed}/{total}] {completed*100.0/total:.1f}% "
                           f"({rate:.1f} files/s, ETA {eta:.0f}s) "
                           f"last: {name}", nl=False)

    if not quiet:
        elapsed = time.monotonic() - start_time
        click.echo(f"\n\nGenerated {total} files in {output_dir}/ ({elapsed:.1f}s)")


@cli.command()
@click.option('--output', '-o', default='compression_stats.csv', help='Output CSV path')
@click.option('--threads', '-t', type=int, default=1, help='Number of threads')
@click.option('--mode', '-m', type=click.Choice(['comprehensive', 'training']),
              default='comprehensive', help='Configuration mode')
@click.option('--verbose', '-v', is_flag=True, help='Print progress')
def stats(output, threads, mode, verbose):
    """Generate statistics CSV (matching newScript.cc --skip-compression).

    Generates data for all configuration combos, computes statistics
    (entropy, MAD, 1st/2nd derivatives), and writes CSV.
    No files are saved -- statistics only.
    """
    size_bins = generate_size_bins(statistics_only=True)

    if mode == 'comprehensive':
        bin_widths = COMPREHENSIVE_BIN_WIDTHS
        perturbations = COMPREHENSIVE_PERTURBATIONS
        fill_modes = FILL_MODES
    else:
        bin_widths = TRAINING_BIN_WIDTHS
        perturbations = TRAINING_PERTURBATIONS
        fill_modes = TRAINING_FILL_MODES

    # Build work items: (dtype, palette, bin_width, perturbation, fill_mode, data_size)
    work_items = []
    for dt in DTYPES:
        for pal in PALETTES:
            for bw in bin_widths:
                for pert in perturbations:
                    for fm in fill_modes:
                        for sz in size_bins:
                            work_items.append((dt, pal, bw, pert, fm, sz))

    total = len(work_items)

    click.echo(f"=== Statistics Generation ===")
    click.echo(f"Distributions: {len(PALETTES) * len(bin_widths) * len(perturbations) * len(fill_modes)}"
               f" ({len(PALETTES)} palettes x {len(bin_widths)} widths"
               f" x {len(perturbations)} perturbations x {len(fill_modes)} fill modes)")
    click.echo(f"Size bins: {[_fmt_size(s) for s in size_bins]}")
    click.echo(f"Data types: {DTYPES}")
    click.echo(f"Total samples: {total}")
    click.echo(f"Threads: {threads}")
    click.echo(f"Output: {output}")
    click.echo(f"============================")

    results = []
    results_lock = threading.Lock()
    completed = [0]  # mutable counter

    def process_item(item):
        dt, pal, bw, pert, fm, sz = item
        dtype_bytes = np.dtype(_TYPE_CONFIG[dt]['np_dtype']).itemsize
        num_elements = sz // dtype_bytes

        data = generate(
            shape=(num_elements,), palette=pal, perturbation=pert,
            fill_mode=fm, bin_width=bw, dtype=dt
        )

        s = calculate_all_statistics(data)

        row = {
            'data_type': dt,
            'data_size': sz,
            'palette': pal,
            'bin_width': f"{bw:.2f}",
            'perturbation': f"{pert:.3f}",
            'fill_mode': fill_mode_short(fm),
            'shannon_entropy': f"{s['shannon_entropy']:.6f}",
            'mad': f"{s['mad']:.6f}",
            'first_derivative': f"{s['first_derivative']:.6f}",
            'second_derivative': f"{s['second_derivative']:.6f}",
        }

        with results_lock:
            results.append(row)
            completed[0] += 1
            if verbose and (completed[0] % 100 == 0 or completed[0] == total):
                pct = completed[0] * 100.0 / total
                click.echo(f"\r[Progress] {completed[0]}/{total} ({pct:.1f}%)", nl=False)
                if completed[0] == total:
                    click.echo()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_item, item) for item in work_items]
        for f in as_completed(futures):
            f.result()  # Raise any exceptions

    # Write CSV (matching newScript.cc statistics CSV format)
    fieldnames = ['data_type', 'data_size', 'palette', 'bin_width', 'perturbation',
                  'fill_mode', 'shannon_entropy', 'mad', 'first_derivative', 'second_derivative']

    with open(output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    click.echo(f"\nSaved {len(results)} rows to {output}")


@cli.command()
@click.argument('filename')
def info(filename):
    """Display file information and statistics."""
    if filename.endswith('.bin'):
        if not os.path.exists(filename):
            click.echo(f"File not found: {filename}", err=True)
            sys.exit(1)

        # Try to infer dtype from filename
        dt = np.float32
        for dtype_name, cfg in _TYPE_CONFIG.items():
            if dtype_name in filename:
                dt = cfg['np_dtype']
                break

        data = np.fromfile(filename, dtype=dt)
        s = calculate_all_statistics(data)
        size_mb = os.path.getsize(filename) / (1024**2)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"  File:       {filename}")
        click.echo(f"  Size:       {size_mb:.2f} MB ({data.size:,} {dt.__name__} elements)")
        click.echo(f"  Entropy:    {s['shannon_entropy']:.4f} bits")
        click.echo(f"  MAD:        {s['mad']:.6f}")
        click.echo(f"  1st Deriv:  {s['first_derivative']:.6f}")
        click.echo(f"  2nd Deriv:  {s['second_derivative']:.6f}")
        click.echo(f"  Mean:       {data.mean():.6f}")
        click.echo(f"  Std:        {data.std():.6f}")
        click.echo(f"  Min:        {data.min():.6f}")
        click.echo(f"  Max:        {data.max():.6f}")
        click.echo(f"{'=' * 60}\n")

    elif filename.endswith(('.h5', '.hdf5')):
        data, attrs = read_hdf5(filename)
        s = calculate_all_statistics(data)
        size_mb = data.nbytes / (1024**2)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"  File:       {filename}")
        click.echo(f"  Shape:      {data.shape}")
        click.echo(f"  Dtype:      {data.dtype}")
        click.echo(f"  Size:       {size_mb:.2f} MB ({data.size:,} elements)")
        click.echo(f"  Entropy:    {s['shannon_entropy']:.4f} bits")
        click.echo(f"  MAD:        {s['mad']:.6f}")
        click.echo(f"  1st Deriv:  {s['first_derivative']:.6f}")
        click.echo(f"  2nd Deriv:  {s['second_derivative']:.6f}")

        param_keys = [k for k in attrs if k.startswith('param_')]
        if param_keys:
            click.echo(f"\n  Generation Parameters:")
            for k in sorted(param_keys):
                click.echo(f"    {k.replace('param_', ''):16s}: {attrs[k]}")

        click.echo(f"{'=' * 60}\n")
    else:
        click.echo(f"Unknown format: {filename}", err=True)
        sys.exit(1)


# ============================================================
# Internal helpers
# ============================================================

def _save(filepath: str, data: np.ndarray,
          palette: str, perturbation: float,
          fill_mode: str, bin_width: float):
    """Save dataset to .bin or .h5."""
    if filepath.endswith(('.h5', '.hdf5')):
        write_hdf5(
            filename=filepath, data=data,
            generation_params={
                'palette': palette,
                'perturbation': perturbation,
                'fill_mode': fill_mode,
                'bin_width': bin_width,
            }
        )
    else:
        write_binary(filepath, data)


def _generate_one_file(filepath, num_elements, palette, perturbation,
                       fill_mode, bin_width, dtype, seed):
    """Generate and save a single dataset file. Used by parallel batch."""
    data = generate(
        shape=(num_elements,), palette=palette, perturbation=perturbation,
        fill_mode=fill_mode, bin_width=bin_width, dtype=dtype, seed=seed
    )
    _save(filepath, data, palette, perturbation, fill_mode, bin_width)
    return os.path.basename(filepath), calculate_byte_entropy(data)


def _fmt_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}KB"
    return f"{size_bytes}B"


if __name__ == '__main__':
    cli()
