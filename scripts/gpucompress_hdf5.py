"""
Python ctypes wrapper for writing GPU tensors directly to HDF5 via the
GPUCompress VOL connector.

Bypasses the GPU→CPU→disk roundtrip by passing CUDA device pointers
directly to H5Dwrite, which the VOL intercepts and routes through the
GPU-accelerated compression pipeline.

Usage:
    import torch
    from gpucompress_hdf5 import GPUCompressHDF5Writer, concat_and_pad_gpu

    with GPUCompressHDF5Writer(lib_dir="build") as writer:
        tensor = torch.randn(1024 * 1024, device="cuda", dtype=torch.float32)
        writer.write_gpu_tensor(
            tensor.data_ptr(), tensor.numel(),
            "output.h5", "data",
            chunk_elements=256 * 1024,  # 1 MB chunks
        )
"""

import ctypes
import math
import os
import sys
from pathlib import Path

# ============================================================
# HDF5 compile-time constants (stable across versions)
# ============================================================
H5S_ALL = 0
H5P_DEFAULT = 0
H5F_ACC_TRUNC = 2
H5F_ACC_RDONLY = 0
H5Z_FLAG_OPTIONAL = 0x0001
H5Z_FILTER_GPUCOMPRESS = 305

# HDF5 type aliases for ctypes
hid_t = ctypes.c_int64
hsize_t = ctypes.c_uint64
herr_t = ctypes.c_int32


def concat_and_pad_gpu(tensor_list, target_elements):
    """Concatenate PyTorch tensors on GPU, pad to target_elements. No CPU copy.

    Args:
        tensor_list: List of PyTorch tensors (on GPU).
        target_elements: Exact number of float32 elements in the output.

    Returns:
        Contiguous float32 CUDA tensor of exactly target_elements elements.
    """
    import torch

    flat = torch.cat([p.detach().float().flatten() for p in tensor_list])
    n = flat.numel()
    if n > target_elements:
        raise ValueError(
            f"Total elements ({n}) exceeds target ({target_elements}). "
            f"Model may have grown — update target_elements."
        )
    if n < target_elements:
        flat = torch.cat([flat, torch.zeros(target_elements - n,
                          device=flat.device, dtype=torch.float32)])
    flat = flat.contiguous()
    assert flat.is_cuda, "Tensor must be on GPU"
    assert flat.dtype == torch.float32, "Tensor must be float32"
    return flat


class GPUCompressHDF5Writer:
    """Write GPU tensors directly to HDF5 via the GPUCompress VOL connector.

    The VOL connector detects CUDA device pointers in H5Dwrite and routes
    them through the GPU-accelerated 3-stage compression pipeline
    (NN inference → compress → I/O).

    Must be constructed AFTER torch.cuda.set_device() or model.cuda() so
    that PyTorch and gpucompress use the same GPU.
    """

    def __init__(self, lib_dir=None, weights_path=None, hdf5_lib_path=None):
        """
        Args:
            lib_dir: Path to build/ directory with .so files (default: auto-detect)
            weights_path: Path to model.nnwt for NN selection (None = LZ4 fallback)
            hdf5_lib_path: Path to libhdf5.so (default: auto-detect from LD_LIBRARY_PATH)
        """
        if lib_dir is None:
            lib_dir = str(Path(__file__).parent.parent / "build")
        self._lib_dir = lib_dir
        self._weights_path = weights_path
        self._hdf5_lib_path = hdf5_lib_path
        self._initialized = False

        # Libraries (loaded in init())
        self._hdf5 = None
        self._gc = None
        self._vol = None
        self._h5z = None

        # HDF5 runtime globals
        self._H5P_FILE_ACCESS = None
        self._H5P_DATASET_CREATE = None
        self._H5T_NATIVE_FLOAT = None

    def init(self):
        """Load libraries, initialize gpucompress, register VOL connector."""
        if self._initialized:
            return

        # Ensure gpucompress uses the same GPU as PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                os.environ.setdefault("GPUCOMPRESS_DEVICE", str(dev))
        except ImportError:
            pass

        # Enable detailed per-chunk timing
        os.environ.setdefault("GPUCOMPRESS_DETAILED_TIMING", "1")

        # Load libraries in dependency order (HDF5 first with RTLD_GLOBAL)
        self._load_libraries()
        self._setup_prototypes()
        self._resolve_hdf5_globals()

        # Initialize gpucompress
        wpath = self._weights_path
        if wpath is not None:
            wpath = wpath.encode("utf-8") if isinstance(wpath, str) else wpath
        rc = self._gc.gpucompress_init(wpath)
        if rc != 0:
            # Try without NN weights
            rc = self._gc.gpucompress_init(None)
            if rc != 0:
                raise RuntimeError(f"gpucompress_init failed (rc={rc})")
            print("  WARNING: NN weights not loaded, using LZ4 fallback",
                  file=sys.stderr)

        self._initialized = True

    def set_policy(self, policy="balanced"):
        """Set the NN cost model policy for algorithm selection.

        Args:
            policy: "balanced" (w0=1,w1=1,w2=1), "ratio" (w0=0,w1=0,w2=1),
                    "speed" (w0=1,w1=1,w2=0), or a tuple (w0, w1, w2).
        """
        if not self._initialized:
            raise RuntimeError("Call init() first")
        policies = {
            "balanced":    (1.0, 1.0, 1.0),   # equal weight speed + ratio
            "ratio":       (0.0, 0.0, 1.0),   # pure ratio (ignore speed)
            "speed":       (1.0, 1.0, 0.0),   # pure speed (ignore ratio)
            "ratio_lean":  (0.2, 0.2, 1.0),   # favor ratio but still consider speed
            "ratio_lean2": (0.1, 0.1, 1.0),   # strongly favor ratio, minimal speed weight
        }
        if isinstance(policy, str):
            if policy not in policies:
                raise ValueError(f"Unknown policy '{policy}'. Use: {list(policies.keys())}")
            w0, w1, w2 = policies[policy]
        else:
            w0, w1, w2 = policy
        self._gc.gpucompress_set_ranking_weights(
            ctypes.c_float(w0), ctypes.c_float(w1), ctypes.c_float(w2)
        )

    def set_learning_mode(self, mode, sgd_lr=0.2, sgd_mape=0.10,
                          explore_k=4, explore_thresh=0.20):
        """Set NN learning mode. Full-state write — sets ALL globals.

        Args:
            mode: "nn" (inference only), "nn-rl" (SGD), "nn-rl+exp50" (SGD + exploration)
            sgd_lr: SGD learning rate (default 0.2)
            sgd_mape: MAPE threshold to trigger SGD update (default 0.10)
            explore_k: Number of exploration alternatives (default 4)
            explore_thresh: Cost error threshold for exploration (default 0.20)
        """
        if not self._initialized:
            raise RuntimeError("Call init() first")
        if mode == "nn":
            self._gc.gpucompress_disable_online_learning()
        elif mode == "nn-rl":
            self._gc.gpucompress_enable_online_learning()
            self._gc.gpucompress_set_exploration(0)
            self._gc.gpucompress_set_reinforcement(
                1, ctypes.c_float(sgd_lr), ctypes.c_float(sgd_mape),
                ctypes.c_float(0.0))
            self._gc.gpucompress_set_exploration_k(-1)
        elif mode == "nn-rl+exp50":
            self._gc.gpucompress_enable_online_learning()
            self._gc.gpucompress_set_exploration(1)
            self._gc.gpucompress_set_reinforcement(
                1, ctypes.c_float(sgd_lr), ctypes.c_float(sgd_mape),
                ctypes.c_float(0.0))
            self._gc.gpucompress_set_exploration_k(explore_k)
            self._gc.gpucompress_set_exploration_threshold(
                ctypes.c_double(explore_thresh))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use: nn, nn-rl, nn-rl+exp50")

    def nn_save_snapshot(self):
        """Save current NN weights to a bytes object."""
        if not self._initialized:
            raise RuntimeError("Call init() first")
        size = self._gc.gpucompress_nn_weights_size()
        buf = ctypes.create_string_buffer(size)
        rc = self._gc.gpucompress_nn_save_snapshot(buf)
        if rc != 0:
            raise RuntimeError(f"gpucompress_nn_save_snapshot failed (rc={rc})")
        return bytes(buf)

    def nn_restore_snapshot(self, snapshot):
        """Restore NN weights from a bytes object."""
        if not self._initialized:
            raise RuntimeError("Call init() first")
        buf = ctypes.create_string_buffer(snapshot)
        rc = self._gc.gpucompress_nn_restore_snapshot(buf)
        if rc != 0:
            raise RuntimeError(f"gpucompress_nn_restore_snapshot failed (rc={rc})")

    def reload_nn(self, filepath):
        """Reload NN weights from disk (reset to original state)."""
        if not self._initialized:
            raise RuntimeError("Call init() first")
        path = filepath.encode("utf-8") if isinstance(filepath, str) else filepath
        rc = self._gc.gpucompress_reload_nn(path)
        if rc != 0:
            raise RuntimeError(f"gpucompress_reload_nn failed (rc={rc})")

    def reset_chunk_history(self):
        """Reset chunk diagnostic history (prevents cross-config SGD contamination)."""
        if self._initialized:
            self._gc.gpucompress_reset_chunk_history()

    def flush_manager_cache(self):
        """Flush nvCOMP manager LRU cache (cold-start for fair timing)."""
        if self._initialized:
            self._gc.gpucompress_flush_manager_cache()

    def get_vol_timing(self):
        """Get VOL pipeline timing from the last H5Dwrite (ms)."""
        s1 = ctypes.c_double(0)
        drain = ctypes.c_double(0)
        io_drain = ctypes.c_double(0)
        total = ctypes.c_double(0)
        self._vol.H5VL_gpucompress_get_stage_timing(
            ctypes.byref(s1), ctypes.byref(drain),
            ctypes.byref(io_drain), ctypes.byref(total))
        s2_busy = ctypes.c_double(0)
        s3_busy = ctypes.c_double(0)
        self._vol.H5VL_gpucompress_get_busy_timing(
            ctypes.byref(s2_busy), ctypes.byref(s3_busy))
        setup = ctypes.c_double(0)
        vol_func = ctypes.c_double(0)
        self._vol.H5VL_gpucompress_get_vol_func_timing(
            ctypes.byref(setup), ctypes.byref(vol_func))
        return {
            "stage1_ms": s1.value,
            "drain_ms": drain.value,
            "io_drain_ms": io_drain.value,
            "pipeline_ms": total.value,
            "s2_busy_ms": s2_busy.value,
            "s3_busy_ms": s3_busy.value,
            "setup_ms": setup.value,
            "vol_func_ms": vol_func.value,
        }

    def collect_chunk_metrics(self):
        """Collect per-chunk diagnostic metrics (mirrors C collect_chunk_metrics).

        Returns dict with: sgd_fires, explorations, n_chunks,
        mape_ratio_pct, mape_comp_pct, mape_decomp_pct,
        mae_ratio, mae_comp_ms, mae_decomp_ms,
        nn_ms, stats_ms, preproc_ms, comp_ms, decomp_ms, explore_ms, sgd_ms,
        comp_gbps, decomp_gbps
        """
        import struct
        n_hist = self._gc.gpucompress_get_chunk_history_count()

        # gpucompress_chunk_diag_t layout (from gpucompress.h):
        #   0: int nn_action
        #   4: int nn_original_action
        #   8: int exploration_triggered
        #  12: int sgd_fired
        #  16: float nn_inference_ms
        #  20: float stats_ms
        #  24: float preprocessing_ms
        #  28: float compression_ms (clamped)
        #  32: float compression_ms_raw (unclamped)
        #  36: float exploration_ms
        #  40: float sgd_update_ms
        #  44: float actual_ratio
        #  48: float predicted_ratio
        #  52: float predicted_comp_time
        #  56: float predicted_decomp_time
        #  60: float predicted_psnr
        #  64: float actual_psnr
        #  68: float decompression_ms (clamped)
        #  72: float decompression_ms_raw (unclamped)

        DIAG_SIZE = 2048  # large enough for the full struct
        buf = ctypes.create_string_buffer(DIAG_SIZE)

        sgd_fires = 0
        explorations = 0
        mape_r_sum = 0.0; mape_c_sum = 0.0; mape_d_sum = 0.0
        mae_r_sum = 0.0; mae_c_sum = 0.0; mae_d_sum = 0.0
        cnt_r = 0; cnt_c = 0; cnt_d = 0
        # R² accumulators: Σactual, Σactual², Σ(actual-predicted)²
        r2_r_sum = 0.0; r2_r_sum2 = 0.0; r2_r_ss = 0.0; r2_r_n = 0
        r2_c_sum = 0.0; r2_c_sum2 = 0.0; r2_c_ss = 0.0; r2_c_n = 0
        r2_d_sum = 0.0; r2_d_sum2 = 0.0; r2_d_ss = 0.0; r2_d_n = 0
        # PSNR metrics
        mape_p_sum = 0.0; mae_p_sum = 0.0; cnt_p = 0
        r2_p_sum = 0.0; r2_p_sum2 = 0.0; r2_p_ss = 0.0; r2_p_n = 0
        psnr_pred_sum = 0.0; psnr_pred_cnt = 0
        total_nn = 0.0; total_stats = 0.0; total_preproc = 0.0
        total_comp = 0.0; total_decomp = 0.0
        total_explore = 0.0; total_sgd = 0.0

        for ci in range(n_hist):
            if self._gc.gpucompress_get_chunk_diag(ci, buf) != 0:
                continue

            raw = buf.raw[:76]
            (action, orig_action, expl_triggered, sgd_fired,
             nn_ms, stats_ms, preproc_ms,
             comp_ms, comp_ms_raw, explore_ms, sgd_ms,
             actual_ratio, predicted_ratio,
             predicted_comp, predicted_decomp,
             predicted_psnr, actual_psnr,
             decomp_ms, decomp_ms_raw) = struct.unpack_from("4i15f", raw, 0)

            if sgd_fired: sgd_fires += 1
            if expl_triggered: explorations += 1

            # Ratio MAPE/MAE/R²
            if actual_ratio > 0 and predicted_ratio > 0:
                diff = abs(predicted_ratio - actual_ratio)
                mape_r_sum += diff / abs(actual_ratio)
                mae_r_sum += diff
                r2_r_sum += actual_ratio; r2_r_sum2 += actual_ratio**2
                r2_r_ss += (actual_ratio - predicted_ratio)**2; r2_r_n += 1
                cnt_r += 1

            # Compression time: MAPE, MAE, R² (all use clamped 5ms floor)
            if comp_ms > 0:
                mape_c_sum += abs(predicted_comp - comp_ms) / abs(comp_ms)
                mae_c_sum += abs(predicted_comp - comp_ms)
                r2_c_sum += comp_ms; r2_c_sum2 += comp_ms**2
                r2_c_ss += (comp_ms - predicted_comp)**2; r2_c_n += 1
                cnt_c += 1

            # Decompression time: MAPE, MAE, R² (all use clamped 5ms floor)
            if decomp_ms > 0:
                mape_d_sum += abs(predicted_decomp - decomp_ms) / abs(decomp_ms)
                mae_d_sum += abs(predicted_decomp - decomp_ms)
                r2_d_sum += decomp_ms; r2_d_sum2 += decomp_ms**2
                r2_d_ss += (decomp_ms - predicted_decomp)**2; r2_d_n += 1
                cnt_d += 1

            # PSNR MAPE/MAE/R² (skip lossless: actual_psnr=inf)
            if predicted_psnr > 0 and 0 < actual_psnr and math.isfinite(actual_psnr):
                a, p = actual_psnr, predicted_psnr
                mape_p_sum += abs(p - a) / abs(a)
                mae_p_sum += abs(p - a)
                cnt_p += 1
                r2_p_sum += a; r2_p_sum2 += a * a
                r2_p_ss += (a - p) ** 2; r2_p_n += 1
            if predicted_psnr > 0:
                psnr_pred_sum += predicted_psnr
                psnr_pred_cnt += 1

            # Component timing (unclamped)
            total_nn += nn_ms
            total_stats += stats_ms
            total_preproc += preproc_ms
            total_comp += comp_ms_raw
            total_decomp += decomp_ms_raw
            total_explore += explore_ms
            total_sgd += sgd_ms

        orig_bytes = n_hist * 4 * 1024 * 1024  # approximate (chunk_size * n_chunks)

        # R² = 1 - SS_res / SS_tot, where SS_tot = Σx² - (Σx)²/n
        def _r2(s, s2, ss_res, n):
            if n < 2: return 0.0
            ss_tot = s2 - (s * s) / n
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return {
            "n_chunks": n_hist,
            "sgd_fires": sgd_fires,
            "explorations": explorations,
            "mape_ratio_pct": min(200.0, (mape_r_sum / cnt_r * 100) if cnt_r else 0),
            "mape_comp_pct": min(200.0, (mape_c_sum / cnt_c * 100) if cnt_c else 0),
            "mape_decomp_pct": min(200.0, (mape_d_sum / cnt_d * 100) if cnt_d else 0),
            "mae_ratio": mae_r_sum / cnt_r if cnt_r else 0,
            "mae_comp_ms": mae_c_sum / cnt_c if cnt_c else 0,
            "mae_decomp_ms": mae_d_sum / cnt_d if cnt_d else 0,
            "r2_ratio": _r2(r2_r_sum, r2_r_sum2, r2_r_ss, r2_r_n),
            "r2_comp": _r2(r2_c_sum, r2_c_sum2, r2_c_ss, r2_c_n),
            "r2_decomp": _r2(r2_d_sum, r2_d_sum2, r2_d_ss, r2_d_n),
            "mape_psnr_pct": min(200.0, (mape_p_sum / cnt_p * 100) if cnt_p else 0),
            "mae_psnr_db": mae_p_sum / cnt_p if cnt_p else 0,
            "r2_psnr": _r2(r2_p_sum, r2_p_sum2, r2_p_ss, r2_p_n),
            "nn_ms": total_nn,
            "stats_ms": total_stats,
            "preproc_ms": total_preproc,
            "comp_ms": total_comp,
            "decomp_ms": total_decomp,
            "explore_ms": total_explore,
            "sgd_ms": total_sgd,
        }

    def collect_chunk_details(self):
        """Collect per-chunk diagnostic details for plotting.

        Returns list of dicts, one per chunk, with:
            chunk_idx, action, actual_ratio, predicted_ratio,
            comp_ms, predicted_comp_time, decomp_ms, predicted_decomp_time,
            sgd_fired, exploration_triggered
        """
        import struct
        n_hist = self._gc.gpucompress_get_chunk_history_count()
        DIAG_SIZE = 2048
        buf = ctypes.create_string_buffer(DIAG_SIZE)
        chunks = []

        for ci in range(n_hist):
            if self._gc.gpucompress_get_chunk_diag(ci, buf) != 0:
                continue
            # gpucompress_chunk_diag_t layout:
            #  0: int nn_action
            #  4: int nn_original_action
            #  8: int exploration_triggered
            # 12: int sgd_fired
            # 16: float nn_inference_ms
            # 20: float stats_ms
            # 24: float preprocessing_ms
            # 28: float compression_ms (clamped)
            # 32: float compression_ms_raw
            # 36: float exploration_ms
            # 40: float sgd_update_ms
            # 44: float actual_ratio
            # 48: float predicted_ratio
            # 52: float predicted_comp_time
            # 56: float predicted_decomp_time
            # 60: float predicted_psnr
            # 64: float actual_psnr
            # 68: float decompression_ms
            # 72: float decompression_ms_raw
            raw = buf.raw[:76]
            vals = struct.unpack_from("4i15f", raw, 0)
            chunks.append({
                "chunk_idx": ci,
                "action": vals[0],
                "original_action": vals[1],
                "exploration_triggered": vals[2],
                "sgd_fired": vals[3],
                "nn_ms": vals[4],
                "comp_ms": vals[7],        # compression_ms (clamped 5ms floor, for MAPE)
                "comp_ms_raw": vals[8],    # unclamped (for latency breakdown)
                "actual_ratio": vals[11],
                "predicted_ratio": vals[12],
                "predicted_comp_time": vals[13],
                "predicted_decomp_time": vals[14],
                "predicted_psnr": vals[15],
                "actual_psnr": vals[16],
                "decomp_ms": vals[17],     # decompression_ms (clamped 5ms floor)
                "decomp_ms_raw": vals[18], # unclamped
            })
        return chunks

    def record_process_start(self):
        """Reset the e2e timer to now (in the VOL library's DiagnosticsStore).

        Call this just before the training/simulation loop begins (after model
        loading, CUDA init, data preparation) so that e2e_ms excludes startup
        overhead and reflects only the active compute + I/O time.

        NOTE: Uses H5VL_gpucompress_record_process_start from the VOL .so,
        which shares the same DiagnosticsStore as accumulateIoMs.
        """
        if not self._initialized or not self._vol:
            return
        try:
            fn = self._vol.H5VL_gpucompress_record_process_start
            fn.argtypes = []
            fn.restype = None
            fn()
        except AttributeError:
            pass

    def dump_timing(self, path=None):
        """Write e2e + VOL timing CSV explicitly (from the VOL library).

        Use this instead of relying on C atexit when the library is loaded via
        ctypes, since Python's shutdown order may unload the .so before atexit
        handlers run.

        Args:
            path: Output file path. Falls back to GPUCOMPRESS_TIMING_OUTPUT env
                  var, then 'gpucompress_io_timing.csv'.
        """
        if not self._initialized or not self._vol:
            return
        try:
            fn = self._vol.H5VL_gpucompress_dump_timing
            fn.argtypes = [ctypes.c_char_p]
            fn.restype = None
            c_path = path.encode("utf-8") if path else None
            fn(c_path)
        except AttributeError:
            pass  # older build without this symbol

    def cleanup(self):
        """Release gpucompress resources."""
        if self._initialized and self._gc:
            self._gc.gpucompress_cleanup()
            self._initialized = False

    def write_gpu_tensor(self, gpu_ptr, n_elements, filepath, dataset_name="data",
                         chunk_elements=1048576, error_bound=0.0, algorithm=0):
        """Write a GPU float32 tensor directly to an HDF5 file via VOL.

        Args:
            gpu_ptr: CUDA device pointer (from tensor.data_ptr())
            n_elements: Number of float32 elements
            filepath: Output .h5 file path
            dataset_name: HDF5 dataset name (default: "data")
            chunk_elements: Elements per chunk (default: 1M = 4 MB)
            error_bound: Lossy error bound (0.0 = lossless)
        """
        if not self._initialized:
            raise RuntimeError("Call init() first")

        # Sync PyTorch streams to ensure tensor data is committed
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        h5 = self._hdf5

        # Create FAPL with VOL connector
        native_id = h5.H5VLget_connector_id_by_name(b"native")
        if native_id < 0:
            raise RuntimeError("H5VLget_connector_id_by_name('native') failed")

        vol_id = self._vol.H5VL_gpucompress_register()
        if vol_id < 0:
            h5.H5VLclose(native_id)
            raise RuntimeError("H5VL_gpucompress_register failed")

        fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
        if fapl < 0:
            h5.H5VLclose(native_id)
            h5.H5VLclose(vol_id)
            raise RuntimeError("H5Pcreate(FILE_ACCESS) failed")

        rc = self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
        h5.H5VLclose(native_id)
        if rc < 0:
            h5.H5Pclose(fapl)
            h5.H5VLclose(vol_id)
            raise RuntimeError("H5Pset_fapl_gpucompress failed")

        # Create file
        fpath = filepath.encode("utf-8") if isinstance(filepath, str) else filepath
        fid = h5.H5Fcreate(fpath, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        h5.H5Pclose(fapl)
        if fid < 0:
            h5.H5VLclose(vol_id)
            raise RuntimeError(f"H5Fcreate failed: {filepath}")

        try:
            # Create dataspace (1D)
            dims = (hsize_t * 1)(n_elements)
            space = h5.H5Screate_simple(1, dims, None)
            if space < 0:
                raise RuntimeError("H5Screate_simple failed")

            try:
                # Create DCPL with chunking + gpucompress filter
                dcpl = h5.H5Pcreate(self._H5P_DATASET_CREATE)
                if dcpl < 0:
                    raise RuntimeError("H5Pcreate(DATASET_CREATE) failed")

                chunk = (hsize_t * 1)(min(chunk_elements, n_elements))
                h5.H5Pset_chunk(dcpl, 1, chunk)

                # Use H5Pset_gpucompress from libH5Zgpucompress.so
                # Signature: H5Pset_gpucompress(dcpl, algo, preproc, shuf_size, eb)
                # ALGO_AUTO=0, PREPROC_SHUFFLE_4=0x02, shuf_size=4
                rc = self._h5z.H5Pset_gpucompress(
                    dcpl, ctypes.c_int(algorithm),
                    ctypes.c_uint(0x02), ctypes.c_uint(4),
                    ctypes.c_double(error_bound)
                )
                if rc < 0:
                    h5.H5Pclose(dcpl)
                    raise RuntimeError("H5Pset_gpucompress failed")

                try:
                    # Create dataset
                    dname = dataset_name.encode("utf-8") if isinstance(dataset_name, str) else dataset_name
                    dset = h5.H5Dcreate2(fid, dname, self._H5T_NATIVE_FLOAT,
                                         space, H5P_DEFAULT, dcpl, H5P_DEFAULT)
                    if dset < 0:
                        raise RuntimeError("H5Dcreate2 failed")

                    try:
                        # Write from GPU pointer
                        rc = h5.H5Dwrite(dset, self._H5T_NATIVE_FLOAT,
                                         H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                         ctypes.c_void_p(gpu_ptr))
                        if rc < 0:
                            raise RuntimeError("H5Dwrite failed")
                    finally:
                        h5.H5Dclose(dset)
                finally:
                    h5.H5Pclose(dcpl)
            finally:
                h5.H5Sclose(space)
        finally:
            h5.H5Fclose(fid)
            h5.H5VLclose(vol_id)

    def get_stats(self):
        """Get VOL activity counters (writes, reads, comp, decomp)."""
        writes = ctypes.c_int(0)
        reads = ctypes.c_int(0)
        comp = ctypes.c_int(0)
        decomp = ctypes.c_int(0)
        self._vol.H5VL_gpucompress_get_stats(
            ctypes.byref(writes), ctypes.byref(reads),
            ctypes.byref(comp), ctypes.byref(decomp)
        )
        return {
            "writes": writes.value, "reads": reads.value,
            "comp": comp.value, "decomp": decomp.value,
        }

    def reset_stats(self):
        """Reset VOL activity counters."""
        self._vol.H5VL_gpucompress_reset_stats()

    def read_gpu_tensor(self, filepath, gpu_ptr, n_elements, dataset_name="data"):
        """Read an HDF5 dataset back to a GPU buffer via VOL (GPU decompression)."""
        if not self._initialized:
            raise RuntimeError("Call init() first")
        h5 = self._hdf5
        native_id = h5.H5VLget_connector_id_by_name(b"native")
        vol_id = self._vol.H5VL_gpucompress_register()
        fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
        self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
        fpath = filepath.encode("utf-8") if isinstance(filepath, str) else filepath
        fid = h5.H5Fopen(fpath, H5F_ACC_RDONLY, fapl)
        h5.H5Pclose(fapl)
        h5.H5VLclose(native_id)
        if fid < 0:
            h5.H5VLclose(vol_id)
            raise RuntimeError(f"H5Fopen failed: {filepath}")
        dname = dataset_name.encode("utf-8") if isinstance(dataset_name, str) else dataset_name
        dset = h5.H5Dopen2(fid, dname, H5P_DEFAULT)
        if dset < 0:
            h5.H5Fclose(fid); h5.H5VLclose(vol_id)
            raise RuntimeError(f"H5Dopen2 failed: {dataset_name}")
        rc = h5.H5Dread(dset, self._H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, ctypes.c_void_p(gpu_ptr))
        import torch
        torch.cuda.synchronize()
        h5.H5Dclose(dset)
        h5.H5Fclose(fid)
        h5.H5VLclose(vol_id)
        if rc < 0:
            raise RuntimeError(f"H5Dread failed: {filepath}")

    def benchmark_gpu_tensor(self, d_original, n_elements, tmpdir="/tmp",
                             chunk_elements=1048576, error_bound=0.0,
                             verify=True, algorithms=None):
        """Benchmark all compression algorithms on a GPU tensor.

        Args:
            d_original: PyTorch CUDA tensor (the data to compress)
            n_elements: Number of float32 elements
            tmpdir: Directory for temporary .h5 files
            chunk_elements: Elements per chunk
            error_bound: Lossy error bound (0.0 = lossless)
            verify: If True, read back and verify bitwise match
            algorithms: List of (name, algo_id) tuples, or None for all

        Returns:
            List of dicts: algorithm, ratio, write_ms, read_ms, write_mibps,
            read_mibps, file_bytes, orig_bytes, mismatches
        """
        import torch
        import time

        if algorithms is None:
            algorithms = [
                ("lz4",      1),
                ("snappy",   2),
                ("deflate",  3),
                ("gdeflate", 4),
                ("zstd",     5),
                ("ans",      6),
                ("cascaded", 7),
                ("bitcomp",  8),
                ("nn-auto",  0),
            ]

        orig_bytes = n_elements * 4
        gpu_ptr = d_original.data_ptr()
        results = []

        d_read = torch.empty(n_elements, device="cuda", dtype=torch.float32) if verify else None

        for algo_name, algo_id in algorithms:
            tmpfile = os.path.join(tmpdir, f"_bench_{algo_name}.h5")

            # Write (compress) — file/dataset creation outside timer,
            # matching C++ benchmark timing boundaries.
            h5 = self._hdf5
            native_id = h5.H5VLget_connector_id_by_name(b"native")
            vol_id = self._vol.H5VL_gpucompress_register()
            fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
            self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
            h5.H5VLclose(native_id)
            fpath = tmpfile.encode("utf-8") if isinstance(tmpfile, str) else tmpfile
            fid = h5.H5Fcreate(fpath, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
            h5.H5Pclose(fapl)
            dims_arr = (hsize_t * 1)(n_elements)
            space = h5.H5Screate_simple(1, dims_arr, None)
            dcpl = h5.H5Pcreate(self._H5P_DATASET_CREATE)
            chunk = (hsize_t * 1)(min(chunk_elements, n_elements))
            h5.H5Pset_chunk(dcpl, 1, chunk)
            self._h5z.H5Pset_gpucompress(
                dcpl, ctypes.c_int(algo_id),
                ctypes.c_uint(0x02), ctypes.c_uint(4),
                ctypes.c_double(error_bound))
            dset = h5.H5Dcreate2(fid, b"data", self._H5T_NATIVE_FLOAT,
                                 space, H5P_DEFAULT, dcpl, H5P_DEFAULT)
            h5.H5Sclose(space)
            h5.H5Pclose(dcpl)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h5.H5Dwrite(dset, self._H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         ctypes.c_void_p(gpu_ptr))
            torch.cuda.synchronize()
            h5.H5Dclose(dset)
            h5.H5Fclose(fid)
            h5.H5VLclose(vol_id)
            t1 = time.perf_counter()
            write_ms = (t1 - t0) * 1000.0
            file_bytes = os.path.getsize(tmpfile)
            ratio = orig_bytes / file_bytes if file_bytes > 0 else 0.0

            # Read (decompress) + verify — file/dataset open outside timer
            read_ms = 0.0
            mismatches = -1
            if verify and d_read is not None:
                d_read.zero_()
                # Drop page cache for cold read (matching C++ benchmarks)
                try:
                    fd = os.open(tmpfile, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except (OSError, AttributeError):
                    pass
                native_id = h5.H5VLget_connector_id_by_name(b"native")
                vol_id = self._vol.H5VL_gpucompress_register()
                fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
                self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
                h5.H5VLclose(native_id)
                r_fid = h5.H5Fopen(fpath, H5F_ACC_RDONLY, fapl)
                h5.H5Pclose(fapl)
                r_dset = h5.H5Dopen2(r_fid, b"data", H5P_DEFAULT)

                torch.cuda.synchronize()
                t2 = time.perf_counter()
                h5.H5Dread(r_dset, self._H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           ctypes.c_void_p(d_read.data_ptr()))
                torch.cuda.synchronize()
                h5.H5Dclose(r_dset)
                h5.H5Fclose(r_fid)
                t3 = time.perf_counter()
                h5.H5VLclose(vol_id)
                read_ms = (t3 - t2) * 1000.0
                # Bitwise comparison (NaN-safe: compares raw float32 bit patterns)
                mismatches = int(d_original.view(torch.int32).ne(d_read.view(torch.int32)).sum().item())

            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

            write_mibps = orig_bytes / (1 << 20) / (write_ms / 1000.0) if write_ms > 0 else 0
            read_mibps = orig_bytes / (1 << 20) / (read_ms / 1000.0) if read_ms > 0 else 0

            results.append({
                "algorithm": algo_name,
                "algo_id": algo_id,
                "ratio": ratio,
                "write_ms": write_ms,
                "read_ms": read_ms,
                "write_mibps": write_mibps,
                "read_mibps": read_mibps,
                "file_bytes": file_bytes,
                "orig_bytes": orig_bytes,
                "mismatches": mismatches,
            })

        if d_read is not None:
            del d_read
            torch.cuda.empty_cache()

        return results

    def benchmark_single(self, d_original, n_elements, tmpfile,
                         chunk_elements=1048576, error_bound=0.0,
                         algorithm=0, d_read=None):
        """Benchmark one compression config with fair timing.

        Fair timing: flush manager cache, create file outside timer,
        cudaDeviceSynchronize before/after write, page cache drop before read.

        Args:
            d_original: PyTorch CUDA tensor
            n_elements: Number of float32 elements
            tmpfile: Temporary .h5 file path
            chunk_elements: Elements per chunk
            error_bound: Lossy error bound
            algorithm: Algorithm ID (0=auto, 1=lz4, ..., 8=bitcomp)
            d_read: Pre-allocated read-back buffer (None = skip verify)

        Returns:
            Dict with ratio, write_ms, read_ms, write_mibps, read_mibps,
            file_bytes, orig_bytes, mismatches
        """
        import torch
        import time

        orig_bytes = n_elements * 4
        h5 = self._hdf5

        # 1. Flush manager cache + reset chunk history + reset VOL stats (cold start)
        self.flush_manager_cache()
        self.reset_chunk_history()
        self.reset_stats()

        # 2. Create HDF5 file + dataset OUTSIDE timed region
        native_id = h5.H5VLget_connector_id_by_name(b"native")
        vol_id = self._vol.H5VL_gpucompress_register()
        fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
        self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
        h5.H5VLclose(native_id)

        fpath = tmpfile.encode("utf-8") if isinstance(tmpfile, str) else tmpfile
        fid = h5.H5Fcreate(fpath, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        h5.H5Pclose(fapl)

        dims = (hsize_t * 1)(n_elements)
        space = h5.H5Screate_simple(1, dims, None)
        dcpl = h5.H5Pcreate(self._H5P_DATASET_CREATE)
        chunk = (hsize_t * 1)(min(chunk_elements, n_elements))
        h5.H5Pset_chunk(dcpl, 1, chunk)
        if algorithm != -1:  # -1 = no-comp (no filter)
            self._h5z.H5Pset_gpucompress(
                dcpl, ctypes.c_int(algorithm),
                ctypes.c_uint(0x02), ctypes.c_uint(4),
                ctypes.c_double(error_bound))
        dset = h5.H5Dcreate2(fid, b"data", self._H5T_NATIVE_FLOAT,
                              space, H5P_DEFAULT, dcpl, H5P_DEFAULT)

        # 3. Timed write (H5Dwrite + sync + close, matching C benchmark boundary)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h5.H5Dwrite(dset, self._H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, ctypes.c_void_p(d_original.data_ptr()))
        torch.cuda.synchronize()  # drain SGD stream for nn-rl
        h5.H5Dclose(dset)
        h5.H5Sclose(space)
        h5.H5Pclose(dcpl)
        h5.H5Fclose(fid)
        t1 = time.perf_counter()
        write_ms = (t1 - t0) * 1000.0

        # 4. Close VOL handle (outside timer — not done in C benchmarks)
        h5.H5VLclose(vol_id)

        file_bytes = os.path.getsize(tmpfile)
        ratio = orig_bytes / file_bytes if file_bytes > 0 else 0.0

        # 5. Page cache invalidation
        try:
            fd = os.open(tmpfile, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (OSError, AttributeError):
            pass

        # 6. Timed read + verify + quality metrics
        # Match VPIC: open file/dataset OUTSIDE timer, time only H5Dread + sync + close
        read_ms = 0.0
        mismatches = -1
        psnr_db = 0.0
        rmse = 0.0
        max_abs_err = 0.0
        bit_rate = 0.0
        data_range = 0.0
        if d_read is not None:
            h5 = self._hdf5
            d_read.zero_()

            # Open file + dataset OUTSIDE timed region
            native_id = h5.H5VLget_connector_id_by_name(b"native")
            vol_id = self._vol.H5VL_gpucompress_register()
            fapl = h5.H5Pcreate(self._H5P_FILE_ACCESS)
            self._vol.H5Pset_fapl_gpucompress(fapl, native_id, None)
            fpath = tmpfile.encode("utf-8") if isinstance(tmpfile, str) else tmpfile
            r_fid = h5.H5Fopen(fpath, H5F_ACC_RDONLY, fapl)
            h5.H5Pclose(fapl)
            h5.H5VLclose(native_id)
            r_dset = h5.H5Dopen2(r_fid, b"data", H5P_DEFAULT)

            # Timed region: H5Dread + sync + close (matches VPIC)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            h5.H5Dread(r_dset, self._H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, ctypes.c_void_p(d_read.data_ptr()))
            torch.cuda.synchronize()
            h5.H5Dclose(r_dset)
            h5.H5Fclose(r_fid)
            t3 = time.perf_counter()
            h5.H5VLclose(vol_id)
            read_ms = (t3 - t2) * 1000.0
            mismatches = int(d_original.view(torch.int32).ne(
                             d_read.view(torch.int32)).sum().item())

            # Always compute data range (for normalized RMSE reporting)
            data_range = d_original.float().max().item() - d_original.float().min().item()

            # Quality metrics (meaningful for lossy, trivial for lossless)
            mean_abs_err = 0.0
            ssim = 1.0
            if mismatches > 0:
                diff = (d_original.double() - d_read.double())
                mse = (diff ** 2).mean().item()
                rmse = mse ** 0.5
                abs_diff = diff.abs()
                max_abs_err = abs_diff.max().item()
                mean_abs_err = abs_diff.mean().item()
                if mse > 0 and data_range > 0:
                    psnr_db = 10.0 * torch.log10(
                        torch.tensor(data_range ** 2 / mse)).item()
                else:
                    psnr_db = float('inf')
                bit_rate = (file_bytes * 8.0) / n_elements if n_elements > 0 else 0.0
                # Global SSIM: single-window over entire tensor
                x = d_original.double()
                y = d_read.double()
                mu_x = x.mean().item()
                mu_y = y.mean().item()
                var_x = x.var(correction=0).item()
                var_y = y.var(correction=0).item()
                cov = ((x - mu_x) * (y - mu_y)).mean().item()
                L = data_range if data_range > 0 else 1.0
                C1 = (0.01 * L) ** 2
                C2 = (0.03 * L) ** 2
                ssim = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / \
                       ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
            else:
                psnr_db = float('inf')  # lossless = perfect

        # 7. Collect diagnostics (chunk metrics + VOL timing + per-chunk details)
        diag = self.collect_chunk_metrics()
        vol_timing = self.get_vol_timing()
        chunk_details = self.collect_chunk_details()

        # 8. Cleanup
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

        write_mibps = orig_bytes / (1 << 20) / (write_ms / 1000.0) if write_ms > 0 else 0
        read_mibps = orig_bytes / (1 << 20) / (read_ms / 1000.0) if read_ms > 0 else 0

        result = {
            "ratio": ratio,
            "write_ms": write_ms,
            "read_ms": read_ms,
            "write_mibps": write_mibps,
            "read_mibps": read_mibps,
            "file_bytes": file_bytes,
            "orig_bytes": orig_bytes,
            "mismatches": mismatches,
            "psnr_db": psnr_db,
            "rmse": rmse,
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
            "ssim": ssim,
            "bit_rate": bit_rate,
            "data_range": data_range,
        }
        result.update(diag)
        result.update(vol_timing)
        result["chunk_details"] = chunk_details
        return result

    # ── Context manager ──

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.cleanup()

    # ── Internal helpers ──

    def _load_libraries(self):
        """Load shared libraries in dependency order."""
        lib_dir = self._lib_dir

        # HDF5 must be loaded first with RTLD_GLOBAL so VOL/filter can find symbols
        hdf5_path = self._hdf5_lib_path
        if hdf5_path is None:
            # Try common locations
            for candidate in [
                "/tmp/hdf5-install/lib/libhdf5.so",
                os.path.join(lib_dir, "libhdf5.so"),
            ]:
                if os.path.exists(candidate):
                    hdf5_path = candidate
                    break
            if hdf5_path is None:
                hdf5_path = "libhdf5.so"  # rely on LD_LIBRARY_PATH

        self._hdf5 = ctypes.CDLL(hdf5_path, mode=ctypes.RTLD_GLOBAL)

        # GPUCompress core
        gc_path = os.path.join(lib_dir, "libgpucompress.so")
        self._gc = ctypes.CDLL(gc_path, mode=ctypes.RTLD_GLOBAL)

        # VOL connector
        vol_path = os.path.join(lib_dir, "libH5VLgpucompress.so")
        self._vol = ctypes.CDLL(vol_path, mode=ctypes.RTLD_GLOBAL)

        # Filter plugin (for H5Pset_gpucompress convenience function)
        h5z_path = os.path.join(lib_dir, "libH5Zgpucompress.so")
        self._h5z = ctypes.CDLL(h5z_path, mode=ctypes.RTLD_GLOBAL)

    def _setup_prototypes(self):
        """Declare ctypes function signatures."""
        # ── libgpucompress ──
        self._gc.gpucompress_init.argtypes = [ctypes.c_char_p]
        self._gc.gpucompress_init.restype = ctypes.c_int
        self._gc.gpucompress_cleanup.argtypes = []
        self._gc.gpucompress_cleanup.restype = None
        self._gc.gpucompress_set_ranking_weights.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        self._gc.gpucompress_set_ranking_weights.restype = None

        # NN weight snapshots
        self._gc.gpucompress_nn_weights_size.argtypes = []
        self._gc.gpucompress_nn_weights_size.restype = ctypes.c_size_t
        self._gc.gpucompress_nn_save_snapshot.argtypes = [ctypes.c_void_p]
        self._gc.gpucompress_nn_save_snapshot.restype = ctypes.c_int
        self._gc.gpucompress_nn_restore_snapshot.argtypes = [ctypes.c_void_p]
        self._gc.gpucompress_nn_restore_snapshot.restype = ctypes.c_int
        self._gc.gpucompress_reload_nn.argtypes = [ctypes.c_char_p]
        self._gc.gpucompress_reload_nn.restype = ctypes.c_int

        # Online learning / exploration / reinforcement
        self._gc.gpucompress_enable_online_learning.argtypes = []
        self._gc.gpucompress_enable_online_learning.restype = None
        self._gc.gpucompress_disable_online_learning.argtypes = []
        self._gc.gpucompress_disable_online_learning.restype = None
        self._gc.gpucompress_set_exploration.argtypes = [ctypes.c_int]
        self._gc.gpucompress_set_exploration.restype = None
        self._gc.gpucompress_set_reinforcement.argtypes = [
            ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        self._gc.gpucompress_set_reinforcement.restype = None
        self._gc.gpucompress_set_exploration_k.argtypes = [ctypes.c_int]
        self._gc.gpucompress_set_exploration_k.restype = None
        self._gc.gpucompress_set_exploration_threshold.argtypes = [ctypes.c_double]
        self._gc.gpucompress_set_exploration_threshold.restype = None

        # State management
        self._gc.gpucompress_reset_chunk_history.argtypes = []
        self._gc.gpucompress_reset_chunk_history.restype = None
        self._gc.gpucompress_flush_manager_cache.argtypes = []
        self._gc.gpucompress_flush_manager_cache.restype = None

        # Chunk diagnostics
        self._gc.gpucompress_get_chunk_history_count.argtypes = []
        self._gc.gpucompress_get_chunk_history_count.restype = ctypes.c_int
        # chunk_diag_t is large (~1600 bytes) — we'll read it as raw bytes
        self._gc.gpucompress_get_chunk_diag.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self._gc.gpucompress_get_chunk_diag.restype = ctypes.c_int

        # VOL pipeline timing
        self._vol.H5VL_gpucompress_get_stage_timing.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        self._vol.H5VL_gpucompress_get_stage_timing.restype = None
        self._vol.H5VL_gpucompress_get_busy_timing.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        self._vol.H5VL_gpucompress_get_busy_timing.restype = None
        self._vol.H5VL_gpucompress_get_vol_func_timing.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        self._vol.H5VL_gpucompress_get_vol_func_timing.restype = None

        # ── libH5VLgpucompress ──
        self._vol.H5VL_gpucompress_register.argtypes = []
        self._vol.H5VL_gpucompress_register.restype = hid_t
        self._vol.H5Pset_fapl_gpucompress.argtypes = [hid_t, hid_t, ctypes.c_void_p]
        self._vol.H5Pset_fapl_gpucompress.restype = herr_t
        self._vol.H5VL_gpucompress_get_stats.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self._vol.H5VL_gpucompress_get_stats.restype = None
        self._vol.H5VL_gpucompress_reset_stats.argtypes = []
        self._vol.H5VL_gpucompress_reset_stats.restype = None

        # ── libH5Zgpucompress ──
        # herr_t H5Pset_gpucompress(hid_t, gpucompress_algorithm_t, uint, uint, double)
        self._h5z.H5Pset_gpucompress.argtypes = [
            hid_t, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_double
        ]
        self._h5z.H5Pset_gpucompress.restype = herr_t

        # ── libhdf5 ──
        h5 = self._hdf5
        h5.H5Pcreate.argtypes = [hid_t]
        h5.H5Pcreate.restype = hid_t
        h5.H5Pclose.argtypes = [hid_t]
        h5.H5Pclose.restype = herr_t
        h5.H5Pset_chunk.argtypes = [hid_t, ctypes.c_int, ctypes.POINTER(hsize_t)]
        h5.H5Pset_chunk.restype = herr_t

        h5.H5Screate_simple.argtypes = [ctypes.c_int, ctypes.POINTER(hsize_t),
                                         ctypes.POINTER(hsize_t)]
        h5.H5Screate_simple.restype = hid_t
        h5.H5Sclose.argtypes = [hid_t]
        h5.H5Sclose.restype = herr_t

        h5.H5Fcreate.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_t, hid_t]
        h5.H5Fcreate.restype = hid_t
        h5.H5Fopen.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_t]
        h5.H5Fopen.restype = hid_t
        h5.H5Fclose.argtypes = [hid_t]
        h5.H5Fclose.restype = herr_t

        h5.H5Dcreate2.argtypes = [hid_t, ctypes.c_char_p, hid_t, hid_t,
                                   hid_t, hid_t, hid_t]
        h5.H5Dcreate2.restype = hid_t
        h5.H5Dopen2.argtypes = [hid_t, ctypes.c_char_p, hid_t]
        h5.H5Dopen2.restype = hid_t
        h5.H5Dwrite.argtypes = [hid_t, hid_t, hid_t, hid_t, hid_t, ctypes.c_void_p]
        h5.H5Dwrite.restype = herr_t
        h5.H5Dread.argtypes = [hid_t, hid_t, hid_t, hid_t, hid_t, ctypes.c_void_p]
        h5.H5Dread.restype = herr_t
        h5.H5Dclose.argtypes = [hid_t]
        h5.H5Dclose.restype = herr_t

        h5.H5VLget_connector_id_by_name.argtypes = [ctypes.c_char_p]
        h5.H5VLget_connector_id_by_name.restype = hid_t
        h5.H5VLclose.argtypes = [hid_t]
        h5.H5VLclose.restype = herr_t

        # Suppress HDF5 error stack (optional, cleaner output)
        h5.H5Eset_auto2.argtypes = [hid_t, ctypes.c_void_p, ctypes.c_void_p]
        h5.H5Eset_auto2.restype = herr_t

    def _resolve_hdf5_globals(self):
        """Resolve HDF5 runtime global variables for property list class IDs."""
        h5 = self._hdf5

        # HDF5 globals are -1 until H5open() initializes the library
        h5.H5open.argtypes = []
        h5.H5open.restype = herr_t
        if h5.H5open() < 0:
            raise RuntimeError("H5open() failed — HDF5 library initialization failed")

        self._H5P_FILE_ACCESS = hid_t.in_dll(h5, "H5P_CLS_FILE_ACCESS_ID_g").value
        self._H5P_DATASET_CREATE = hid_t.in_dll(h5, "H5P_CLS_DATASET_CREATE_ID_g").value
        self._H5T_NATIVE_FLOAT = hid_t.in_dll(h5, "H5T_NATIVE_FLOAT_g").value

        # Suppress HDF5 error printing (H5E_DEFAULT = 0)
        h5.H5Eset_auto2(0, None, None)


# ============================================================
# Algorithm constants
# ============================================================

FIXED_ALGOS = [
    ("no-comp",   -1),  # -1 = no gpucompress filter (serial D→H fallback)
    ("lz4",        1),
    ("snappy",     2),
    ("deflate",    3),
    ("gdeflate",   4),
    ("zstd",       5),
    ("ans",        6),
    ("cascaded",   7),
    ("bitcomp",    8),
]

NN_CONFIGS = [
    ("bal_nn",      "balanced",    "nn"),
    ("bal_rl",      "balanced",    "nn-rl"),
    ("bal_exp",     "balanced",    "nn-rl+exp50"),
    ("rat_nn",      "ratio",       "nn"),
    ("rat_rl",      "ratio",       "nn-rl"),
    ("rat_exp",     "ratio",       "nn-rl+exp50"),
]


# ============================================================
# InlineFullBenchmark — 15-config benchmark with NN snapshot isolation
# ============================================================

class InlineFullBenchmark:
    """Run all 15 compression configs inline during training.

    Manages 6 independent NN weight snapshots so each NN config accumulates
    learning across epochs without cross-contamination.

    Usage:
        bench = InlineFullBenchmark(writer, weights_path, n_elements)
        # At each checkpoint:
        for name, tensors in tensor_sets:
            flat = concat_and_pad_gpu(tensors, target_elements)
            bench.run_checkpoint(writer, flat, n_elements, epoch, name,
                                tmpdir, chunk_elements, error_bound, csv_file)
            del flat; torch.cuda.empty_cache()
    """

    def __init__(self, writer, weights_path, n_elements,
                 sgd_lr=0.2, sgd_mape=0.10, explore_k=4, explore_thresh=0.20):
        """Initialize with clean NN weights and 6 independent snapshots.

        Args:
            writer: GPUCompressHDF5Writer (must be initialized)
            weights_path: Path to model.nnwt
            n_elements: Number of float32 elements per tensor (for read buffer)
            sgd_lr: SGD learning rate (default 0.2)
            sgd_mape: MAPE threshold to trigger SGD (default 0.10)
            explore_k: Number of exploration alternatives (default 4)
            explore_thresh: Cost error threshold for exploration (default 0.20)
        """
        import torch

        # Store hyperparams for set_learning_mode calls
        self.sgd_lr = sgd_lr
        self.sgd_mape = sgd_mape
        self.explore_k = explore_k
        self.explore_thresh = explore_thresh

        # Reload clean NN weights from disk
        writer.reload_nn(weights_path)
        writer.set_learning_mode("nn")
        writer.set_policy("balanced")

        # Save base snapshot — all 6 NN configs start from this
        self.base_snapshot = writer.nn_save_snapshot()

        # Independent snapshot per NN config
        self.snapshots = {name: bytes(self.base_snapshot) for name, _, _ in NN_CONFIGS}

        # Pre-allocate read-back buffer (reused across all configs)
        self.d_read = torch.empty(n_elements, device="cuda", dtype=torch.float32)

        self._weights_path = weights_path

    def run_checkpoint(self, writer, d_tensor, n_elements, epoch, tensor_name,
                       tmpdir, chunk_elements, error_bound, csv_file,
                       chunk_csv_file=None):
        """Run all 15 configs on one tensor and write results to CSV.

        Args:
            writer: GPUCompressHDF5Writer
            d_tensor: Contiguous GPU float32 tensor
            n_elements: Number of elements
            epoch: Current epoch number
            tensor_name: e.g. "weights", "adam_m"
            tmpdir: Directory for temp .h5 files
            chunk_elements: Elements per chunk
            error_bound: Lossy error bound
            csv_file: Open file handle for CSV output
        """
        import torch
        import os

        results = []

        # Clone d_tensor for stable comparison baseline across all 15 configs.
        # Prevents measurement artifacts if d_tensor is modified externally.
        d_original_snapshot = d_tensor.clone()

        # ── Warmup (one throwaway write, not timed) ──
        warmup_file = os.path.join(tmpdir, "_warmup.h5")
        writer.write_gpu_tensor(d_tensor.data_ptr(), n_elements,
                                warmup_file, "data", chunk_elements)
        if os.path.exists(warmup_file):
            os.unlink(warmup_file)

        # ── 9 fixed algorithms (stateless, always lossless) ──
        # Fixed algorithms don't use quantization — error_bound=0.0 ensures
        # the HDF5 filter does not add PREPROC_QUANTIZE.
        # Only NN configs (algo=0) use error_bound for quantization selection.
        writer.set_learning_mode("nn")  # ensure no SGD during fixed algos
        for algo_name, algo_id in FIXED_ALGOS:
            tmpfile = os.path.join(tmpdir, f"_bench_{algo_name}.h5")
            r = writer.benchmark_single(
                d_original_snapshot, n_elements, tmpfile,
                chunk_elements=chunk_elements,
                error_bound=0.0,
                algorithm=algo_id,
                d_read=self.d_read,
            )
            r["algorithm"] = algo_name
            r["policy"] = "-"
            r["mode"] = "-"
            results.append(r)

        # ── 6 NN configs (stateful, with snapshot isolation) ──
        for config_name, policy, mode in NN_CONFIGS:
            # CRITICAL: reset chunk history before restore (prevents SGD contamination)
            writer.reset_chunk_history()
            writer.nn_restore_snapshot(self.snapshots[config_name])
            torch.cuda.synchronize()  # ensure weights are on GPU

            writer.set_policy(policy)
            writer.set_learning_mode(mode, sgd_lr=self.sgd_lr,
                                     sgd_mape=self.sgd_mape,
                                     explore_k=self.explore_k,
                                     explore_thresh=self.explore_thresh)

            tmpfile = os.path.join(tmpdir, f"_bench_{config_name}.h5")
            r = writer.benchmark_single(
                d_original_snapshot, n_elements, tmpfile,
                chunk_elements=chunk_elements,
                error_bound=error_bound,
                algorithm=0,  # NN auto-selection
                d_read=self.d_read,
            )
            r["algorithm"] = config_name
            r["policy"] = policy
            r["mode"] = mode
            results.append(r)

            # Save updated snapshot (RL state persists to next epoch)
            self.snapshots[config_name] = writer.nn_save_snapshot()

        # ── Restore clean state for caller (final .h5 write) ──
        writer.reset_chunk_history()
        writer.nn_restore_snapshot(self.base_snapshot)
        writer.set_policy("balanced")
        writer.set_learning_mode("nn")

        # ── Write to CSV + print ──
        print(f"      epoch{epoch:02d}_{tensor_name}:")
        print(f"        {'Config':>12s}  {'Ratio':>6s}  {'Write MiB/s':>11s}  "
              f"{'Read MiB/s':>11s}  {'SGD':>4s}  {'Expl':>4s}  {'Mismatch':>8s}")
        for r in results:
            mm_str = str(r['mismatches']) if r['mismatches'] >= 0 else "n/a"
            print(f"        {r['algorithm']:>12s}  {r['ratio']:>5.2f}x  "
                  f"{r['write_mibps']:>9.0f}  {r['read_mibps']:>9.0f}  "
                  f"{r.get('sgd_fires',0):>4d}  {r.get('explorations',0):>4d}  "
                  f"{mm_str:>8s}")
            csv_file.write(
                f"{epoch},{tensor_name},{r['algorithm']},{r.get('policy','-')},"
                f"{r.get('mode','-')},{r['ratio']:.4f},"
                f"{r['write_ms']:.2f},{r['read_ms']:.2f},"
                f"{r['write_mibps']:.1f},{r['read_mibps']:.1f},"
                f"{r['file_bytes']},{r['orig_bytes']},{r['mismatches']},"
                f"{r.get('n_chunks',0)},{r.get('sgd_fires',0)},{r.get('explorations',0)},"
                f"{r.get('mape_ratio_pct',0):.2f},{r.get('mape_comp_pct',0):.2f},"
                f"{r.get('mape_decomp_pct',0):.2f},{r.get('mape_psnr_pct',0):.2f},"
                f"{r.get('mae_ratio',0):.4f},{r.get('mae_comp_ms',0):.4f},"
                f"{r.get('mae_decomp_ms',0):.4f},{r.get('mae_psnr_db',0):.4f},"
                f"{r.get('r2_ratio',0):.4f},{r.get('r2_comp',0):.4f},"
                f"{r.get('r2_decomp',0):.4f},{r.get('r2_psnr',0):.4f},"
                f"{r.get('nn_ms',0):.3f},{r.get('stats_ms',0):.3f},"
                f"{r.get('preproc_ms',0):.3f},{r.get('comp_ms',0):.3f},"
                f"{r.get('decomp_ms',0):.3f},{r.get('explore_ms',0):.3f},"
                f"{r.get('sgd_ms',0):.3f},"
                f"{r.get('stage1_ms',0):.3f},{r.get('drain_ms',0):.3f},"
                f"{r.get('io_drain_ms',0):.3f},{r.get('pipeline_ms',0):.3f},"
                f"{r.get('s2_busy_ms',0):.3f},{r.get('s3_busy_ms',0):.3f},"
                f"{r.get('psnr_db',0):.2f},{r.get('rmse',0):.6f},"
                f"{r.get('max_abs_err',0):.6f},{r.get('mean_abs_err',0):.6f},"
                f"{r.get('ssim',1.0):.8f},{r.get('bit_rate',0):.4f},"
                f"{r.get('data_range',0):.6f}\n")
            # Write per-chunk details
            if chunk_csv_file and "chunk_details" in r:
                for c in r["chunk_details"]:
                    chunk_csv_file.write(
                        f"{epoch},{tensor_name},{r['algorithm']},"
                        f"{r.get('policy','-')},{r.get('mode','-')},"
                        f"{c['chunk_idx']},{c['action']},"
                        f"{c['actual_ratio']:.4f},{c['predicted_ratio']:.4f},"
                        f"{c['comp_ms']:.3f},{c['predicted_comp_time']:.3f},"
                        f"{c['decomp_ms']:.3f},{c['predicted_decomp_time']:.3f},"
                        f"{c.get('predicted_psnr',0):.2f},{c.get('actual_psnr',0):.2f},"
                        f"{c['sgd_fired']},{c['exploration_triggered']}\n")
                chunk_csv_file.flush()
        csv_file.flush()

        return results
