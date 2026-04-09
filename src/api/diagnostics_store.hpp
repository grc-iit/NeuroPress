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
 *   accumulateIoMs(ms)      — from dataset_write / dataset_read entry/exit
 *   nextChunkId()           — once per chunk in Trace mode
 *   writeTraceRow(...)      — once per (chunk × config) in Trace mode
 *   dumpIoTiming(path)      — from file_close (all modes)
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

    /**
     * Write total I/O time to a file.
     * Called from H5VL_gpucompress_file_close.
     * File format (plain text, easy to parse):
     *   total_io_ms=<value>
     */
    void dumpIoTiming(const char* path) const {
        if (!path || path[0] == '\0') return;
        FILE* f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "gpucompress VOL: cannot write timing file '%s'\n", path);
            return;
        }
        fprintf(f, "total_io_ms=%.6f\n", totalIoMs());
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
                "chunk_id,action_id,comp_lib,chosen,"
                "pred_cost,real_ratio,real_comp_ms,real_decomp_ms\n");
        fflush(trace_file_);
    }

    /**
     * Append one row to the trace CSV.
     *
     * @param chunk_id     Global chunk ID from nextChunkId()
     * @param action_id    Action encoding (0-31): algo + quant*8 + shuf*16
     * @param comp_lib     Human-readable config name (e.g. "lz4+shuf")
     * @param chosen       true if this action was the NN's selected winner
     * @param pred_cost    NN-predicted composite cost (0 if not available)
     * @param real_ratio   Measured compression ratio (input/compressed)
     * @param real_comp_ms Measured compression time (ms)
     * @param real_decomp_ms Measured decompression time (ms); 0 if skipped
     */
    void writeTraceRow(int chunk_id, int action_id, const char* comp_lib,
                       bool chosen, float pred_cost,
                       float real_ratio, float real_comp_ms, float real_decomp_ms) {
        std::lock_guard<std::mutex> lk(trace_mtx_);
        if (!trace_file_) return;
        fprintf(trace_file_,
                "%d,%d,%s,%d,%.6f,%.4f,%.4f,%.4f\n",
                chunk_id, action_id, comp_lib, chosen ? 1 : 0,
                (double)pred_cost,
                (double)real_ratio, (double)real_comp_ms, (double)real_decomp_ms);
    }

    /** Flush and close the trace CSV. */
    void flushTrace() {
        std::lock_guard<std::mutex> lk(trace_mtx_);
        if (trace_file_) {
            fflush(trace_file_);
            fclose(trace_file_);
            trace_file_ = nullptr;
        }
    }

private:
    DiagnosticsStore()  = default;
    ~DiagnosticsStore() {
        free(history_);
        if (trace_file_) fclose(trace_file_);
    }

    /* Chunk history */
    mutable std::mutex           mutex_;
    gpucompress_chunk_diag_t*    history_      = nullptr;
    int                          cap_          = 0;
    std::atomic<int>             count_{0};

    /* Cache stats */
    std::atomic<int>             cache_hits_{0};
    std::atomic<int>             cache_misses_{0};

    /* Aggregate I/O timing — int64 microseconds avoids float CAS */
    std::atomic<int64_t>         total_io_us_{0};

    /* Global chunk ID counter */
    std::atomic<int>             next_chunk_id_{0};

    /* Trace CSV */
    std::mutex                   trace_mtx_;
    FILE*                        trace_file_   = nullptr;
};

} // namespace gpucompress
