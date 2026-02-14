"""
Python ctypes wrapper for libgpucompress.so.

Provides a GPUCompressLib class that wraps the C API for:
- Library init/cleanup
- Computing data statistics (entropy, MAD, first_derivative) on GPU
- Compressing/decompressing data with explicit configs
- Getting max compressed size
"""

import ctypes
from pathlib import Path

# ============================================================
# Algorithm constants (mirrors gpucompress_algorithm_t)
# ============================================================

ALGO_AUTO     = 0
ALGO_LZ4      = 1
ALGO_SNAPPY   = 2
ALGO_DEFLATE  = 3
ALGO_GDEFLATE = 4
ALGO_ZSTD     = 5
ALGO_ANS      = 6
ALGO_CASCADED = 7
ALGO_BITCOMP  = 8

ALGO_NAMES = {
    ALGO_LZ4: 'lz4', ALGO_SNAPPY: 'snappy', ALGO_DEFLATE: 'deflate',
    ALGO_GDEFLATE: 'gdeflate', ALGO_ZSTD: 'zstd', ALGO_ANS: 'ans',
    ALGO_CASCADED: 'cascaded', ALGO_BITCOMP: 'bitcomp',
}

# ============================================================
# Preprocessing constants (mirrors gpucompress_preproc_t)
# ============================================================

PREPROC_NONE       = 0x00
PREPROC_SHUFFLE_2  = 0x01
PREPROC_SHUFFLE_4  = 0x02
PREPROC_SHUFFLE_8  = 0x04
PREPROC_QUANTIZE   = 0x10

HEADER_SIZE = 64

# ============================================================
# ctypes struct mirrors
# ============================================================

class gpucompress_config_t(ctypes.Structure):
    _fields_ = [
        ('algorithm', ctypes.c_int),
        ('preprocessing', ctypes.c_uint),
        ('error_bound', ctypes.c_double),
        ('cuda_device', ctypes.c_int),
        ('cuda_stream', ctypes.c_void_p),
    ]


class gpucompress_stats_t(ctypes.Structure):
    _fields_ = [
        ('original_size', ctypes.c_size_t),
        ('compressed_size', ctypes.c_size_t),
        ('compression_ratio', ctypes.c_double),
        ('entropy_bits', ctypes.c_double),
        ('mad', ctypes.c_double),
        ('first_derivative', ctypes.c_double),
        ('algorithm_used', ctypes.c_int),
        ('preprocessing_used', ctypes.c_uint),
        ('throughput_mbps', ctypes.c_double),
    ]


# ============================================================
# GPUCompressLib class
# ============================================================

class GPUCompressLib:
    """Context-managed wrapper around libgpucompress.so."""

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = str(Path(__file__).parent.parent / 'build' / 'libgpucompress.so')
        self._lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()
        self._initialized = False

    def _setup_prototypes(self):
        lib = self._lib

        # init / cleanup
        lib.gpucompress_init.argtypes = [ctypes.c_char_p]
        lib.gpucompress_init.restype = ctypes.c_int

        lib.gpucompress_cleanup.argtypes = []
        lib.gpucompress_cleanup.restype = None

        # compute_stats
        lib.gpucompress_compute_stats.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.gpucompress_compute_stats.restype = ctypes.c_int

        # compress
        lib.gpucompress_compress.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(gpucompress_config_t),
            ctypes.POINTER(gpucompress_stats_t),
        ]
        lib.gpucompress_compress.restype = ctypes.c_int

        # decompress
        lib.gpucompress_decompress.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.gpucompress_decompress.restype = ctypes.c_int

        # max_compressed_size
        lib.gpucompress_max_compressed_size.argtypes = [ctypes.c_size_t]
        lib.gpucompress_max_compressed_size.restype = ctypes.c_size_t

        # load_nn
        lib.gpucompress_load_nn.argtypes = [ctypes.c_char_p]
        lib.gpucompress_load_nn.restype = ctypes.c_int

        # nn_is_loaded
        lib.gpucompress_nn_is_loaded.argtypes = []
        lib.gpucompress_nn_is_loaded.restype = ctypes.c_int

        # error_string
        lib.gpucompress_error_string.argtypes = [ctypes.c_int]
        lib.gpucompress_error_string.restype = ctypes.c_char_p

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def init(self, model_path: str = None):
        path = model_path.encode() if model_path else None
        rc = self._lib.gpucompress_init(path)
        if rc != 0:
            raise RuntimeError(
                f"gpucompress_init failed: {self._error_string(rc)}")
        self._initialized = True

    def load_nn(self, path: str):
        """Load neural network weights (.nnwt) for ALGO_AUTO."""
        rc = self._lib.gpucompress_load_nn(path.encode())
        if rc != 0:
            raise RuntimeError(
                f"gpucompress_load_nn failed: {self._error_string(rc)}")

    def cleanup(self):
        if self._initialized:
            self._lib.gpucompress_cleanup()
            self._initialized = False

    def _error_string(self, code: int) -> str:
        s = self._lib.gpucompress_error_string(code)
        return s.decode() if s else f"error {code}"

    def compute_stats(self, data: bytes):
        """Compute entropy, MAD, first_derivative on GPU.

        Args:
            data: Raw bytes (float32 array).

        Returns:
            (entropy, mad, first_derivative) tuple of floats.
        """
        entropy = ctypes.c_double()
        mad = ctypes.c_double()
        deriv = ctypes.c_double()
        rc = self._lib.gpucompress_compute_stats(
            data, len(data),
            ctypes.byref(entropy), ctypes.byref(mad), ctypes.byref(deriv))
        if rc != 0:
            raise RuntimeError(
                f"gpucompress_compute_stats failed: {self._error_string(rc)}")
        return entropy.value, mad.value, deriv.value

    def compress(self, data: bytes, config: gpucompress_config_t,
                 return_stats: bool = False):
        """Compress data with explicit config.

        Args:
            data: Raw input bytes.
            config: gpucompress_config_t instance.
            return_stats: If True, return (compressed_bytes, stats) tuple.

        Returns:
            bytes of compressed output, or (bytes, gpucompress_stats_t) if
            return_stats is True.
        """
        max_size = self._lib.gpucompress_max_compressed_size(len(data))
        out_buf = ctypes.create_string_buffer(max_size)
        out_size = ctypes.c_size_t(max_size)
        stats = gpucompress_stats_t()
        rc = self._lib.gpucompress_compress(
            data, len(data), out_buf, ctypes.byref(out_size),
            ctypes.byref(config), ctypes.byref(stats))
        if rc != 0:
            raise RuntimeError(
                f"gpucompress_compress failed: {self._error_string(rc)}")
        result = out_buf.raw[:out_size.value]
        if return_stats:
            return result, stats
        return result

    def decompress(self, compressed: bytes, original_size: int):
        """Decompress data.

        Args:
            compressed: Compressed bytes (including header).
            original_size: Expected original size in bytes.

        Returns:
            bytes of decompressed output.
        """
        out_buf = ctypes.create_string_buffer(original_size)
        out_size = ctypes.c_size_t(original_size)
        rc = self._lib.gpucompress_decompress(
            compressed, len(compressed), out_buf, ctypes.byref(out_size))
        if rc != 0:
            raise RuntimeError(
                f"gpucompress_decompress failed: {self._error_string(rc)}")
        return out_buf.raw[:out_size.value]

    @staticmethod
    def make_config(algo: int, shuffle: int = 0, quantize: bool = False,
                    error_bound: float = 0.0) -> gpucompress_config_t:
        """Build a gpucompress_config_t.

        Args:
            algo: Algorithm constant (ALGO_LZ4, etc.)
            shuffle: Shuffle element size (0, 2, 4, or 8)
            quantize: Whether to enable quantization
            error_bound: Error bound for quantization

        Returns:
            gpucompress_config_t instance.
        """
        cfg = gpucompress_config_t()
        cfg.algorithm = algo
        cfg.preprocessing = PREPROC_NONE
        if shuffle == 2:
            cfg.preprocessing |= PREPROC_SHUFFLE_2
        elif shuffle == 4:
            cfg.preprocessing |= PREPROC_SHUFFLE_4
        elif shuffle == 8:
            cfg.preprocessing |= PREPROC_SHUFFLE_8
        if quantize:
            cfg.preprocessing |= PREPROC_QUANTIZE
        cfg.error_bound = error_bound
        cfg.cuda_device = -1
        cfg.cuda_stream = None
        return cfg
