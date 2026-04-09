#pragma once
/**
 * @file diagnostics_store.hpp
 * @brief Singleton that owns all per-chunk diagnostic history and cache stats.
 *
 * This is the single source of truth for runtime statistics collected by the
 * VOL connector.  The public C API in gpucompress.h delegates here.
 *
 * Callers inside the library (compress path, pool, VOL) use the singleton
 * directly rather than going through the C wrappers.
 *
 * VOL Mode integration
 * --------------------
 * The singleton also tracks aggregate I/O time (all modes) and owns the
 * Trace-mode CSV file handle.  The VOL calls:
 *
 *   recordProcessStart()    — from H5VL_gpucompress_init (once per process)
 *   recordProcessEnd()      — from H5VL_gpucompress_term (once per process)
 *   accumulateIoMs(ms)      — from dataset_write / dataset_read entry/exit
 *   nextChunkId()           — once per chunk in Trace mode
 *   writeTraceRow(...)      — once per (chunk × config) in Trace mode
 *   dumpIoTiming(path)      — from file_close (all modes); writes CSV e2e_ms,vol_ms
 *   openTraceFile(path)     — from VOL init when mode == TRACE
 *   flushTrace()            — from file_close when mode == TRACE
 */

#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "gpucompress.h"

namespace gpucompress {

class DiagnosticsStore {
public:
    /* Meyer's singleton — constructed on first use, destroyed at exit. */
    static DiagnosticsStore& instance() {
        static DiagnosticsStore s;
        return s;
    }

    DiagnosticsStore(const DiagnosticsStore&)            = delete;
    DiagnosticsStore& operator=(const DiagnosticsStore&) = delete;

    /* ── Chunk history ─────────────────────────────────────────────── */

    /**
     * Append one diagnostic record.  Returns the index assigned to this
     * chunk (used to back-fill decompression timing later), or -1 on OOM.
     */
    int record(const gpucompress_chunk_diag_t& entry) {
        std::lock_guard<std::mutex> lk(mutex_);
        int idx = count_.fetch_add(1);
        if (idx >= cap_) {
            int new_cap = (cap_ == 0) ? 4096 : cap_ * 2;
            auto* p = static_cast<gpucompress_chunk_diag_t*>(
                realloc(history_, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
            if (!p) { count_.fetch_sub(1); return -1; }
            history_ = p;
            cap_     = new_cap;
        }
        history_[idx] = entry;
        return idx;
    }

    /** Zero-fill a slot and return its index (for incremental filling). */
    int allocSlot() {
        std::lock_guard<std::mutex> lk(mutex_);
        int idx = count_.fetch_add(1);
        if (idx >= cap_) {
            int new_cap = (cap_ == 0) ? 4096 : cap_ * 2;
            auto* p = static_cast<gpucompress_chunk_diag_t*>(
                realloc(history_, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
            if (!p) { count_.fetch_sub(1); return -1; }
            history_ = p;
            cap_     = new_cap;
        }
        memset(&history_[idx], 0, sizeof(gpucompress_chunk_diag_t));
        return idx;
    }

    int count() const { return count_.load(); }

    int getDiag(int idx, gpucompress_chunk_diag_t* out) const {
        if (idx < 0 || !out) return -1;
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx >= count_.load() || idx >= cap_) return -1;
        *out = history_[idx];
        return 0;
    }

    /** Back-fill actual decompression timing (called from VOL read path). */
    void recordDecompMs(int idx, float ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        gpucompress_chunk_diag_t& h = history_[idx];
        float clamped = std::max(1.0f, ms);
        h.decompression_ms     = clamped;
        h.decompression_ms_raw = ms;
        float pred_dt = std::max(1.0f, h.predicted_decomp_time);
        if (pred_dt > 0.0f)
            h.decomp_time_mape = std::abs(clamped - pred_dt) / clamped;
    }

    /** Back-fill VOL Stage 2 pipeline timing (pool acquire, D→H, I/O wait). */
    void recordVolTiming(int idx, float pool_acquire_ms,
                         float d2h_copy_ms, float io_queue_wait_ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].vol_pool_acquire_ms  = pool_acquire_ms;
        history_[idx].vol_d2h_copy_ms      = d2h_copy_ms;
        history_[idx].vol_io_queue_wait_ms = io_queue_wait_ms;
    }

    /** Back-fill VOL Stage 1 pipeline timing (stats alloc, stats copy, WQ wait). */
    void recordS1Timing(int idx, float stats_malloc_ms,
                        float stats_copy_ms, float wq_post_wait_ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].vol_stats_malloc_ms  = stats_malloc_ms;
        history_[idx].vol_stats_copy_ms    = stats_copy_ms;
        history_[idx].vol_wq_post_wait_ms  = wq_post_wait_ms;
    }

    /** Set diag_record_ms on an existing slot (called after record() returns). */
    void setDiagRecordMs(int idx, float ms) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (idx < 0 || idx >= count_.load() || idx >= cap_) return;
        history_[idx].diag_record_ms = ms;
    }

    /** Reset the history (call before each H5Dwrite). Does not free memory. */
    void reset() { count_.store(0); }

    /* ── Cache stats ───────────────────────────────────────────────── */

    void incrementCacheHit()  { cache_hits_.fetch_add(1);   }
    void incrementCacheMiss() { cache_misses_.fetch_add(1); }

    void getCacheStats(int* hits, int* misses) const {
        if (hits)   *hits   = cache_hits_.load();
        if (misses) *misses = cache_misses_.load();
    }

    void resetCacheStats() {
        cache_hits_.store(0);
        cache_misses_.store(0);
    }

    /* ── Aggregate I/O timing (all modes) ──────────────────────────── */

    /**
     * Accumulate the wall time (ms) of one dataset_write or dataset_read call.
     * Uses int64 microseconds internally to avoid non-lock-free float atomics.
     */
    void accumulateIoMs(double ms) {
        total_io_us_.fetch_add(static_cast<int64_t>(ms * 1000.0));
    }

    double totalIoMs() const {
        return static_cast<double>(total_io_us_.load()) / 1000.0;
    }

    /* ── End-to-end file timing ────────────────────────────────────── */

    /**
     * Record the wall-clock instant when the VOL connector is initialised.
     * Uses compare-and-swap so only the FIRST call sets the timestamp;
     * subsequent calls (e.g. from HDF5 probing) are ignored.
     */
    void recordProcessStart() {
        int64_t expected = 0;
        int64_t now = nowUs();
        process_start_us_.compare_exchange_strong(expected, now);
    }

    /* Force-reset the start timer to now, ignoring whether it was already set.
     * Use this to exclude startup overhead (model loading, CUDA init, etc.)
     * from e2e_ms by calling it just before the training/simulation loop. */
    void resetProcessStart() {
        process_start_us_.store(nowUs());
    }

    /**
     * Compute and store end-to-end duration (VOL init → VOL term).
     * Called once per process from H5VL_gpucompress_term.
     */
    void recordProcessEnd() {
        int64_t start = process_start_us_.load();
        if (start == 0) return;
        e2e_us_.store(nowUs() - start);
    }

    double totalE2eMs() const {
        return static_cast<double>(e2e_us_.load()) / 1000.0;
    }

    /**
     * Write end-to-end and VOL I/O timing to a CSV file.
     * Called from H5VL_gpucompress_file_close (all modes).
     *
     * CSV format (two rows — header + data):
     *   e2e_ms,vol_ms
     *   <e2e>,<vol>
     *
     * e2e_ms  = H5VL_gpucompress_init → H5VL_gpucompress_term wall-clock time
     * vol_ms  = sum of all H5Dwrite / H5Dread callback wall-clock times
     */
    void dumpIoTiming(const char* path) const {
        if (!path || path[0] == '\0') return;
        FILE* f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "gpucompress VOL: cannot write timing file '%s'\n", path);
            return;
        }
        fprintf(f, "e2e_ms,vol_ms\n");
        fprintf(f, "%.6f,%.6f\n", totalE2eMs(), totalIoMs());
        fclose(f);
    }

    /* ── Global chunk ID counter (Trace mode) ──────────────────────── */

    /**
     * Allocate the next unique chunk ID.  Monotonically increasing across
     * all H5Dwrite calls in the process lifetime.
     */
    int nextChunkId() { return next_chunk_id_.fetch_add(1); }

    /* ── Trace CSV file (Trace mode only) ──────────────────────────── */

    /**
     * Open the trace CSV and write the header row.
     * Idempotent — no-op if already open.
     */
    void openTraceFile(const char* path) {
        std::lock_guard<std::mutex> lk(trace_mtx_);
        if (trace_file_) return;
        trace_file_ = fopen(path, "w");
        if (!trace_file_) {
            fprintf(stderr, "gpucompress VOL: cannot open trace file '%s'\n", path);
            return;
        }
        fprintf(trace_file_,
                "chunk_id,action_id,comp_lib,chosen,chunk_bytes,"
                "pred_cost,pred_ratio,pred_comp_ms,pred_decomp_ms,pred_psnr,pred_ssim,pred_max_error,"
                "real_cost,real_ratio,real_comp_ms,real_decomp_ms,real_psnr,real_ssim,real_max_error,"
                "mape_cost,mape_ratio,mape_comp_ms,mape_decomp_ms,"
                "explore_mode\n");
        fflush(trace_file_);
    }

    /**
     * Append one row to the trace CSV.
     *
     * @param chunk_id       Global chunk ID from nextChunkId()
     * @param action_id      64-config encoding: algo*8 + shuf_idx*4 + quant_idx
     * @param comp_lib       Human-readable config name (e.g. "lz4+shuf")
     * @param chosen         true if this action was the NN's selected winner
     * @param chunk_bytes    Uncompressed chunk size in bytes
     * @param pred_cost      NN-predicted composite cost (0 if not available)
     * @param pred_ratio     NN-predicted compression ratio (0 if not available)
     * @param pred_comp_ms   NN-predicted compression time ms (0 if not available)
     * @param pred_decomp_ms NN-predicted decompression time ms (0 if not available)
     * @param real_ratio     Measured compression ratio (input/compressed)
     * @param real_comp_ms   Measured compression time (ms)
     * @param real_decomp_ms Measured decompression time (ms); 0 if skipped
     */
    void writeTraceRow(int chunk_id, int action_id, const char* comp_lib,
                       bool chosen, size_t chunk_bytes,
                       float pred_cost, float pred_ratio,
                       float pred_comp_ms, float pred_decomp_ms,
                       float pred_psnr, float pred_ssim, float pred_max_error,
                       float real_cost, float real_ratio,
                       float real_comp_ms, float real_decomp_ms,
                       float real_psnr, float real_ssim, float real_max_error,
                       float mape_low_thresh, float mape_high_thresh,
                       float bw_bytes_per_ms, float w0, float w1, float w2) {
        /* Clamp comp/decomp time (floor 5ms) and ratio (cap 100x) before MAPE.
         * Matching the same floors applied during NN training. */
        static constexpr float TIME_FLOOR = 5.0f;
        static constexpr float RATIO_CAP  = 100.0f;

        float p_ct  = std::max(TIME_FLOOR, pred_comp_ms);
        float p_dt  = std::max(TIME_FLOOR, pred_decomp_ms);
        float p_r   = std::min(RATIO_CAP,  pred_ratio);
        float r_ct  = std::max(TIME_FLOOR, real_comp_ms);
        float r_dt  = std::max(TIME_FLOOR, real_decomp_ms);
        float r_r   = std::min(RATIO_CAP,  real_ratio);

        /* Recompute costs from clamped values */
        float bw     = std::max(1.0f, bw_bytes_per_ms);
        float io_div = bw * RATIO_CAP;   /* worst-case denominator for safety */
        float p_cost = w0 * p_ct + w1 * p_dt
                     + w2 * static_cast<float>(chunk_bytes) / (p_r * bw);
        float r_cost = w0 * r_ct + w1 * r_dt
                     + w2 * static_cast<float>(chunk_bytes) / (r_r * bw);
        (void)io_div;

        auto mape_of = [](float pred, float real) -> float {
            return (real > 1e-6f) ? std::abs(pred - real) / real : 0.0f;
        };
        float mape_cost   = mape_of(p_cost, r_cost);
        float mape_ratio  = mape_of(p_r,    r_r);
        float mape_comp   = mape_of(p_ct,   r_ct);
        float mape_decomp = mape_of(p_dt,   r_dt);

        /* explore_mode based on clamped cost MAPE */
        int explore_mode = 2;
        if      (mape_cost < mape_low_thresh)  explore_mode = 0;
        else if (mape_cost < mape_high_thresh) explore_mode = 1;

        std::lock_guard<std::mutex> lk(trace_mtx_);
        if (!trace_file_) return;
        fprintf(trace_file_,
                "%d,%d,%s,%d,%zu,"
                "%.6f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,"
                "%.6f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,"
                "%.4f,%.4f,%.4f,%.4f,"
                "%d\n",
                chunk_id, action_id, comp_lib, chosen ? 1 : 0, chunk_bytes,
                (double)p_cost,    (double)p_r,
                (double)p_ct,      (double)p_dt,
                (double)pred_psnr, (double)pred_ssim, (double)pred_max_error,
                (double)r_cost,    (double)r_r,
                (double)r_ct,      (double)r_dt,
                (double)real_psnr, (double)real_ssim, (double)real_max_error,
                (double)mape_cost, (double)mape_ratio,
                (double)mape_comp, (double)mape_decomp,
                explore_mode);
    }

    /**
     * Flush the trace CSV to disk.
     * Does NOT close the file — the singleton destructor closes it at exit.
     * This matters when the caller is an HDF5 file_close callback that fires
     * once per FAB (e.g. WarpX): closing here would silence all subsequent rows.
     */
    void flushTrace() {
        std::lock_guard<std::mutex> lk(trace_mtx_);
        if (trace_file_) fflush(trace_file_);
    }

private:
    DiagnosticsStore()  = default;
    ~DiagnosticsStore() {
        free(history_);
        if (trace_file_) fclose(trace_file_);
    }

    static int64_t nowUs() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1000000LL
             + static_cast<int64_t>(ts.tv_nsec) / 1000LL;
    }

    /* Chunk history */
    mutable std::mutex           mutex_;
    gpucompress_chunk_diag_t*    history_      = nullptr;
    int                          cap_          = 0;
    std::atomic<int>             count_{0};

    /* Cache stats */
    std::atomic<int>             cache_hits_{0};
    std::atomic<int>             cache_misses_{0};

    /* Aggregate VOL I/O timing — int64 microseconds avoids float CAS */
    std::atomic<int64_t>         total_io_us_{0};

    /* Process-level e2e timing (VOL init → VOL term) */
    std::atomic<int64_t>         process_start_us_{0}; /* set by recordProcessStart() */
    std::atomic<int64_t>         e2e_us_{0};           /* set by recordProcessEnd() */

    /* Global chunk ID counter */
    std::atomic<int>             next_chunk_id_{0};

    /* Trace CSV */
    std::mutex                   trace_mtx_;
    FILE*                        trace_file_   = nullptr;
};

} // namespace gpucompress
