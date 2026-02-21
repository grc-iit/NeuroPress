"""
Binary data loader with on-the-fly GPU benchmarking.

Reads raw .bin files (float32 arrays), computes stats on GPU via
libgpucompress.so, and benchmarks all 64 compression configs per file.

Primary data source for NN training.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from gpucompress_ctypes import (
    GPUCompressLib, ALGO_LZ4, ALGO_SNAPPY, ALGO_DEFLATE, ALGO_GDEFLATE,
    ALGO_ZSTD, ALGO_ANS, ALGO_CASCADED, ALGO_BITCOMP, ALGO_NAMES,
    HEADER_SIZE,
)
from data import encode_and_split

# ============================================================
# Benchmark configuration space (64 configs)
# ============================================================

ALGORITHMS = [
    ALGO_LZ4, ALGO_SNAPPY, ALGO_DEFLATE, ALGO_GDEFLATE,
    ALGO_ZSTD, ALGO_ANS, ALGO_CASCADED, ALGO_BITCOMP,
]

SHUFFLE_OPTIONS = [0, 4]

# (quantize_bool, error_bound) pairs
QUANT_OPTIONS = [
    (False, 0.0),
    (True, 0.1),
    (True, 0.01),
    (True, 0.001),
]


def _compute_psnr(original: bytes, decompressed: bytes) -> float:
    """Compute PSNR between original and decompressed float32 arrays."""
    orig = np.frombuffer(original, dtype=np.float32)
    decomp = np.frombuffer(decompressed, dtype=np.float32)
    if len(orig) != len(decomp):
        return 0.0
    mse = np.mean((orig.astype(np.float64) - decomp.astype(np.float64)) ** 2)
    if mse == 0.0:
        return 120.0  # Perfect reconstruction
    data_range = float(orig.max()) - float(orig.min())
    if data_range == 0.0:
        return 120.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def benchmark_binary_files(
    data_dir: str,
    lib_path: str = None,
    max_files: int = None,
) -> pd.DataFrame:
    """
    Benchmark .bin files against all 64 compression configs on GPU.

    Returns a raw DataFrame with columns matching the benchmark format:
        file, algorithm, quantization, shuffle, error_bound, original_size,
        entropy, mad, second_derivative, compression_ratio,
        compression_time_ms, decompression_time_ms, psnr_db, success
    """
    data_path = Path(data_dir)
    bin_files = sorted(data_path.glob('*.bin'))
    if max_files is not None:
        bin_files = bin_files[:max_files]

    print(f"Found {len(bin_files)} .bin files in {data_dir}")
    if len(bin_files) == 0:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")

    total_configs = len(ALGORITHMS) * len(SHUFFLE_OPTIONS) * len(QUANT_OPTIONS)
    print(f"Benchmarking {total_configs} configs per file "
          f"({len(bin_files) * total_configs} total compressions)")

    rows = []
    t_start = time.time()

    with GPUCompressLib(lib_path) as lib:
        for file_idx, bin_file in enumerate(bin_files):
            raw_data = bin_file.read_bytes()
            original_size = len(raw_data)

            if original_size == 0 or original_size % 4 != 0:
                print(f"  Skipping {bin_file.name}: invalid size {original_size}")
                continue

            # Compute stats once per file on GPU
            entropy, mad_val, deriv_val = lib.compute_stats(raw_data)

            print(f"\n  File: {bin_file.name} ({original_size:,} bytes)")
            print(f"    Stats: entropy={entropy:.4f}  MAD={mad_val:.4f}  "
                  f"deriv={deriv_val:.4f}")
            print(f"    {'Algorithm':>10}  {'Quant':>5}  {'Shuf':>4}  "
                  f"{'ErrBound':>8}  {'Ratio':>8}  {'Comp ms':>8}  "
                  f"{'Decomp ms':>9}  {'PSNR':>7}  {'Status':>6}")
            print(f"    {'-' * 82}")

            # Benchmark all 64 configs
            for algo in ALGORITHMS:
                algo_name = ALGO_NAMES[algo]
                for shuffle in SHUFFLE_OPTIONS:
                    for quantize, error_bound in QUANT_OPTIONS:
                        cfg = lib.make_config(
                            algo=algo, shuffle=shuffle,
                            quantize=quantize, error_bound=error_bound)

                        try:
                            # Compress
                            t0 = time.perf_counter()
                            compressed = lib.compress(raw_data, cfg)
                            t1 = time.perf_counter()
                            comp_time_ms = (t1 - t0) * 1000.0

                            compressed_size = len(compressed) - HEADER_SIZE
                            ratio = original_size / compressed_size if compressed_size > 0 else 1.0

                            # Decompress
                            t2 = time.perf_counter()
                            decompressed = lib.decompress(compressed, original_size)
                            t3 = time.perf_counter()
                            decomp_time_ms = (t3 - t2) * 1000.0

                            # PSNR
                            if quantize and error_bound > 0:
                                psnr = _compute_psnr(raw_data, decompressed)
                            else:
                                psnr = 120.0  # Lossless

                            print(f"    {algo_name:>10}  "
                                  f"{'yes' if quantize else 'no':>5}  "
                                  f"{shuffle:>4}  {error_bound:>8.4f}  "
                                  f"{ratio:>8.2f}  {comp_time_ms:>8.3f}  "
                                  f"{decomp_time_ms:>9.3f}  {psnr:>7.1f}  "
                                  f"{'OK':>6}")

                            rows.append({
                                'file': bin_file.name,
                                'algorithm': algo_name,
                                'quantization': 'linear' if quantize else 'none',
                                'shuffle': shuffle,
                                'error_bound': error_bound,
                                'original_size': original_size,
                                'entropy': entropy,
                                'mad': mad_val,
                                'second_derivative': deriv_val,
                                'compression_ratio': ratio,
                                'compression_time_ms': comp_time_ms,
                                'decompression_time_ms': decomp_time_ms,
                                'psnr_db': psnr,
                                'success': True,
                            })

                        except RuntimeError as e:
                            print(f"    {algo_name:>10}  "
                                  f"{'yes' if quantize else 'no':>5}  "
                                  f"{shuffle:>4}  {error_bound:>8.4f}  "
                                  f"{'---':>8}  {'---':>8}  "
                                  f"{'---':>9}  {'---':>7}  "
                                  f"{'FAIL':>6}")

                            rows.append({
                                'file': bin_file.name,
                                'algorithm': algo_name,
                                'quantization': 'linear' if quantize else 'none',
                                'shuffle': shuffle,
                                'error_bound': error_bound,
                                'original_size': original_size,
                                'entropy': entropy,
                                'mad': mad_val,
                                'second_derivative': deriv_val,
                                'compression_ratio': 1.0,
                                'compression_time_ms': 0.0,
                                'decompression_time_ms': 0.0,
                                'psnr_db': 0.0,
                                'success': False,
                            })

            if (file_idx + 1) % 10 == 0 or file_idx == 0:
                elapsed = time.time() - t_start
                rate = (file_idx + 1) / elapsed
                eta = (len(bin_files) - file_idx - 1) / rate if rate > 0 else 0
                print(f"  [{file_idx+1}/{len(bin_files)}] "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t_start
    print(f"\nBenchmarking completed in {elapsed:.1f}s ({len(rows)} rows)")

    return pd.DataFrame(rows)


def load_and_prepare_from_binary(
    data_dir: str,
    lib_path: str = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_files: int = None,
) -> Dict:
    """
    Load .bin files, benchmark all configs on GPU, and prepare training data.

    Returns the same dict format as data.encode_and_split().
    """
    df = benchmark_binary_files(data_dir, lib_path=lib_path, max_files=max_files)
    return encode_and_split(df, val_fraction, seed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Binary data loader test')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing .bin files')
    parser.add_argument('--lib-path', default=None,
                        help='Path to libgpucompress.so')
    parser.add_argument('--max-files', type=int, default=5,
                        help='Max files to process (for testing)')
    args = parser.parse_args()

    data = load_and_prepare_from_binary(
        args.data_dir, lib_path=args.lib_path, max_files=args.max_files)
    print(f"\nTrain X shape: {data['train_X'].shape}")
    print(f"Train Y shape: {data['train_Y'].shape}")
    print(f"Val X shape:   {data['val_X'].shape}")
    print(f"Val Y shape:   {data['val_Y'].shape}")
