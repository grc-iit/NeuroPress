"""
Compression Executor for Q-Table Training

Executes compression with specified configuration and measures metrics.
Uses the gpu_compress binary via subprocess.
"""

import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Optional
import ctypes
import numpy as np


class CompressionExecutor:
    """
    Executes GPU compression and collects metrics.

    Can use either:
    1. Subprocess: Call gpu_compress binary
    2. C API: Use libgpucompress.so directly (faster)
    """

    def __init__(
        self,
        gpu_compress_path: str = './build/gpu_compress',
        gpu_decompress_path: str = './build/gpu_decompress',
        use_c_api: bool = False,
        lib_path: str = './build/libgpucompress.so'
    ):
        """
        Initialize executor.

        Args:
            gpu_compress_path: Path to gpu_compress binary
            gpu_decompress_path: Path to gpu_decompress binary
            use_c_api: Whether to use C API instead of subprocess
            lib_path: Path to libgpucompress.so for C API
        """
        self.gpu_compress_path = gpu_compress_path
        self.gpu_decompress_path = gpu_decompress_path
        self.use_c_api = use_c_api
        self.lib = None

        if use_c_api:
            self._load_library(lib_path)

    def _load_library(self, lib_path: str):
        """Load libgpucompress.so for C API access."""
        try:
            self.lib = ctypes.CDLL(lib_path)

            # Define function signatures
            self.lib.gpucompress_init.argtypes = [ctypes.c_char_p]
            self.lib.gpucompress_init.restype = ctypes.c_int

            self.lib.gpucompress_calculate_entropy.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)
            ]
            self.lib.gpucompress_calculate_entropy.restype = ctypes.c_int

            # Initialize library
            result = self.lib.gpucompress_init(None)
            if result != 0:
                raise RuntimeError(f"Failed to initialize gpucompress: {result}")

        except OSError as e:
            print(f"Warning: Could not load C API library: {e}")
            print("Falling back to subprocess mode")
            self.use_c_api = False
            self.lib = None

    def calculate_entropy(self, data: np.ndarray) -> float:
        """
        Calculate entropy of data.

        Uses GPU via C API if available, otherwise falls back to NumPy.

        Args:
            data: Input data as numpy array

        Returns:
            Shannon entropy in bits (0.0 to 8.0)
        """
        if self.use_c_api and self.lib is not None:
            # Use GPU entropy calculation
            data_bytes = data.tobytes()
            entropy = ctypes.c_double()

            result = self.lib.gpucompress_calculate_entropy(
                data_bytes, len(data_bytes), ctypes.byref(entropy)
            )

            if result == 0:
                return entropy.value

        # Fallback: NumPy implementation
        data_bytes = data.tobytes()
        byte_array = np.frombuffer(data_bytes, dtype=np.uint8)

        # Histogram
        hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zeros

        # Probabilities
        probs = hist / len(byte_array)

        # Entropy
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def calculate_mad(self, data: np.ndarray) -> float:
        """
        Calculate normalized Mean Absolute Deviation of data.

        Formula: mean(|data - mean|) / (max - min)
        Normalized to [0, ~0.5] so it's scale-invariant.

        Args:
            data: Input data as numpy array

        Returns:
            Normalized mean absolute deviation (0.0 for constant data)
        """
        flat = data.flatten().astype(np.float64)
        data_range = float(flat.max() - flat.min())
        if data_range == 0.0:
            return 0.0
        mean = np.mean(flat)
        return float(np.mean(np.abs(flat - mean))) / data_range

    def calculate_second_derivative(self, data: np.ndarray) -> float:
        """
        Calculate normalized mean absolute second derivative of data.

        Formula: mean(|data[i+1] - 2*data[i] + data[i-1]|) / (max - min)
        Normalized to [0, 1] so it's scale-invariant.

        Args:
            data: Input data as numpy array

        Returns:
            Normalized mean absolute second derivative (0.0 for constant data)
        """
        if data.size < 3:
            return 0.0
        flat = data.flatten().astype(np.float64)
        data_range = float(flat.max() - flat.min())
        if data_range == 0.0:
            return 0.0
        second_deriv = flat[2:] - 2.0 * flat[1:-1] + flat[:-2]
        return float(np.mean(np.abs(second_deriv))) / data_range

    def calculate_all_metrics(self, data: np.ndarray) -> dict:
        """
        Calculate all metrics needed for state encoding.

        Args:
            data: Input data as numpy array

        Returns:
            Dictionary with 'entropy', 'mad', 'second_derivative'
        """
        return {
            'entropy': self.calculate_entropy(data),
            'mad': self.calculate_mad(data),
            'second_derivative': self.calculate_second_derivative(data),
        }

    def _compute_psnr(self, original_file: str, compressed_file: str) -> Optional[float]:
        """
        Compute PSNR by decompressing and comparing to original.

        Args:
            original_file: Path to original input file
            compressed_file: Path to compressed file

        Returns:
            PSNR in dB, or None on failure
        """
        with tempfile.NamedTemporaryFile(suffix='.decompressed', delete=False) as tmp:
            decompressed_file = tmp.name

        try:
            cmd = [self.gpu_decompress_path, compressed_file, decompressed_file]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                print(f"  [PSNR] Decompression failed (exit code {result.returncode})")
                return None

            # Read original and decompressed as float32
            original_data = np.fromfile(original_file, dtype=np.float32)
            decompressed_data = np.fromfile(decompressed_file, dtype=np.float32)

            if original_data.size != decompressed_data.size:
                print(f"  [PSNR] Size mismatch: original={original_data.size}, "
                      f"decompressed={decompressed_data.size}")
                return None

            # Compute MSE
            mse = np.mean((original_data - decompressed_data) ** 2)

            if mse == 0:
                return float('inf')  # Perfect reconstruction

            # Use data range as peak value
            data_range = original_data.max() - original_data.min()
            if data_range == 0:
                return float('inf')

            psnr = 10 * np.log10((data_range ** 2) / mse)
            print(f"  [PSNR] {psnr:.2f} dB (MSE={mse:.6e})")
            return float(psnr)

        except Exception as e:
            print(f"  [PSNR] Error: {e}")
            return None

        finally:
            if os.path.exists(decompressed_file):
                os.unlink(decompressed_file)

    def execute(
        self,
        input_file: str,
        algorithm: str,
        quantization: bool = False,
        error_bound: float = 0.001,
        shuffle_size: int = 0
    ) -> Dict:
        """
        Execute compression and return metrics.

        Args:
            input_file: Path to input file
            algorithm: Algorithm name (lz4, snappy, etc.)
            quantization: Whether to apply quantization
            error_bound: Error bound for quantization
            shuffle_size: Shuffle element size (0, 2, 4, or 8)

        Returns:
            Dictionary with metrics:
                - ratio: Compression ratio
                - throughput_mbps: Throughput in MB/s
                - psnr_db: PSNR in dB (if quantization applied)
                - compressed_size: Size after compression
                - original_size: Original file size
                - success: Whether compression succeeded
        """
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.compressed', delete=False) as tmp:
            output_file = tmp.name

        try:
            # Build command
            cmd = [self.gpu_compress_path, input_file, output_file, algorithm]

            if shuffle_size > 0:
                cmd.extend(['--shuffle', str(shuffle_size)])

            if quantization:
                cmd.extend(['--quant-type', 'linear'])
                cmd.extend(['--error-bound', str(error_bound)])

            # Log the exact command being sent to gpu_compress
            print(f"  [EXECUTOR] Command: {' '.join(cmd)}")

            # Execute compression
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            elapsed_time = time.time() - start_time

            # Check for success
            if result.returncode != 0:
                print(f"  [EXECUTOR] FAILED (exit code {result.returncode})")
                if result.stderr:
                    print(f"  [EXECUTOR] stderr: {result.stderr.strip()[:200]}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'ratio': 1.0,
                    'throughput_mbps': 0.0,
                    'psnr_db': None
                }

            # Get file sizes
            original_size = os.path.getsize(input_file)
            compressed_size = os.path.getsize(output_file)

            # Calculate metrics
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            throughput_mbps = (original_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0.0

            # Compute PSNR for lossy compression (quantization applied)
            psnr_db = None
            if quantization:
                psnr_db = self._compute_psnr(input_file, output_file)

            print(f"  [EXECUTOR] SUCCESS: {original_size} -> {compressed_size} bytes "
                  f"(ratio={ratio:.2f}x, throughput={throughput_mbps:.1f} MB/s, "
                  f"time={elapsed_time:.3f}s"
                  f"{f', psnr={psnr_db:.2f} dB' if psnr_db is not None else ''})")

            return {
                'success': True,
                'ratio': ratio,
                'throughput_mbps': throughput_mbps,
                'psnr_db': psnr_db,
                'compressed_size': compressed_size,
                'original_size': original_size,
                'elapsed_time': elapsed_time,
                'algorithm': algorithm,
                'quantization': quantization,
                'shuffle_size': shuffle_size
            }

        except subprocess.TimeoutExpired:
            print(f"  [EXECUTOR] TIMEOUT after 300s")
            return {
                'success': False,
                'error': 'Timeout',
                'ratio': 1.0,
                'throughput_mbps': 0.0,
                'psnr_db': None
            }

        except Exception as e:
            print(f"  [EXECUTOR] EXCEPTION: {e}")
            return {
                'success': False,
                'error': str(e),
                'ratio': 1.0,
                'throughput_mbps': 0.0,
                'psnr_db': None
            }

        finally:
            # Cleanup temporary file
            if os.path.exists(output_file):
                os.unlink(output_file)

    def execute_action(
        self,
        input_file: str,
        action_config: Dict,
        error_bound: float = 0.001
    ) -> Dict:
        """
        Execute compression with decoded action configuration.

        Args:
            input_file: Path to input file
            action_config: Decoded action from QTable.decode_action()
            error_bound: Error bound for quantization (if enabled)

        Returns:
            Metrics dictionary
        """
        return self.execute(
            input_file=input_file,
            algorithm=action_config['algorithm'],
            quantization=action_config['quantization'],
            error_bound=error_bound if action_config['quantization'] else 0.0,
            shuffle_size=action_config['shuffle_size']
        )
