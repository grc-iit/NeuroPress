/**
 * Kendall Tau Ranking Quality Profiler
 *
 * At milestone timesteps, compresses each chunk with all active configs
 * (16 for lossless, 32 for lossy) to measure how well the NN's predicted
 * cost ranking matches the actual ranking.
 *
 * Metrics computed per chunk:
 *   - Kendall tau-b (tie-corrected rank correlation)
 *   - Top-1 accuracy (did NN pick the actual best config?)
 *   - Top-1 regret (cost ratio: NN pick / oracle pick)
 *
 * Usage: #include from .cu benchmark drivers. For VPIC (.cxx), use a
 * separate .cu wrapper with C-linkage.
 */
#pragma once

#include <gpucompress.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

/* ════════════════════════════════════════════════════════════════
 * Result structures
 * ════════════════════════════════════════════════════════════════ */

struct RankingChunkResult {
    int    chunk_idx;
    int    n_active_configs;
    double kendall_tau_b;
    double top1_regret;        /* actual_cost[nn_pick] / actual_cost[oracle_pick] */
    int    predicted_best;     /* action ID of NN's top pick */
    int    actual_best;        /* action ID of actual best config */
    double predicted_best_cost;
    double actual_best_cost;
};

struct RankingMilestoneResult {
    int    timestep;
    int    n_chunks;
    double mean_tau;
    double std_tau;
    double mean_regret;
    double profiling_ms;       /* wall-clock time for the profiling pass */
};

/* ════════════════════════════════════════════════════════════════
 * Kendall tau-b computation (O(n^2), fine for n<=32)
 * ════════════════════════════════════════════════════════════════ */

static inline double compute_kendall_tau_b(
    const int* rank_a,   /* ranking A: action IDs sorted by cost (best first) */
    const int* rank_b,   /* ranking B: action IDs sorted by cost (best first) */
    int n)
{
    if (n < 2) return 0.0;

    /* Build position maps: pos_b[action] = rank in B */
    int pos_b[32] = {};
    for (int i = 0; i < n; i++) {
        pos_b[rank_b[i]] = i;
    }

    /* Count concordant/discordant pairs using a common item ordering */
    long long concordant = 0, discordant = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int ai = rank_a[i], aj = rank_a[j];  /* items in order of ranking A */
            int bi = pos_b[ai], bj = pos_b[aj];  /* their positions in ranking B */
            /* In ranking A, ai is ranked higher than aj (i < j).
               Check if ranking B agrees (bi < bj) or disagrees (bi > bj). */
            if (bi < bj) concordant++;
            else if (bi > bj) discordant++;
            /* bi == bj shouldn't happen (distinct action IDs) */
        }
    }

    long long n_pairs = (long long)n * (n - 1) / 2;
    /* tau-b: for our case with no ties in either ranking, tau-b == tau-a */
    double denom = (double)n_pairs;
    if (denom == 0.0) return 0.0;
    return (double)(concordant - discordant) / denom;
}

/* ════════════════════════════════════════════════════════════════
 * Milestone check
 * ════════════════════════════════════════════════════════════════ */

static inline bool is_ranking_milestone(int t, int total) {
    if (total < 10) return true;  /* profile every timestep for short runs */
    int last = total - 1;
    /* T0, T5, T10 (early learning) */
    if (t == 0 || t == 5 || t == 10) return true;
    /* Then every 10% of total */
    for (int pct = 10; pct <= 100; pct += 10) {
        if (t == last * pct / 100) return true;
    }
    return false;
}

/* ════════════════════════════════════════════════════════════════
 * Action name helper
 * ════════════════════════════════════════════════════════════════ */

static inline const char* action_name_str(int action_id) {
    static const char* algo_names[] = {
        "lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"};
    static char buf[32];
    int algo = action_id % 8;
    int quant = (action_id / 8) % 2;
    int shuf  = (action_id / 16) % 2;
    snprintf(buf, sizeof(buf), "%s%s%s", algo_names[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
    return buf;
}

/* ════════════════════════════════════════════════════════════════
 * Config profiling result (per config per chunk)
 * ════════════════════════════════════════════════════════════════ */

struct ConfigResult {
    int    action_id;
    double ratio;
    double comp_ms;
    double decomp_ms;
    double cost;
};

/* ════════════════════════════════════════════════════════════════
 * Main profiling function
 * ════════════════════════════════════════════════════════════════ */

static int run_ranking_profiler(
    const void* d_data,
    size_t total_bytes,
    size_t chunk_bytes,
    double error_bound,
    float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats,
    FILE* csv,
    FILE* costs_csv,
    const char* phase_name,
    int timestep,
    RankingMilestoneResult* out)
{
    if (!d_data || total_bytes == 0 || chunk_bytes == 0 || !csv) return -1;

    auto wall_start = std::chrono::steady_clock::now();

    int n_chunks = (int)(total_bytes / chunk_bytes);
    if (n_chunks <= 0) return -1;
    if (n_repeats < 1) n_repeats = 3;

    /* ── 1. Snapshot chunk diagnostics before profiling pollutes history ── */
    int n_hist = gpucompress_get_chunk_history_count();
    std::vector<gpucompress_chunk_diag_t> diags(n_hist);
    for (int ci = 0; ci < n_hist; ci++)
        gpucompress_get_chunk_diag(ci, &diags[ci]);

    /* ── 2. Determine active configs ── */
    /* Action encoding: action = algo(0-7) + quant(0-1)*8 + shuffle(0-1)*16 */
    std::vector<int> active_actions;
    for (int a = 0; a < 32; a++) {
        int quant = (a / 8) % 2;
        if (error_bound <= 0.0 && quant) continue;  /* skip quantized for lossless */
        active_actions.push_back(a);
    }
    int n_active = (int)active_actions.size();
    if (n_active < 2) return -1;

    /* ── 3. Allocate GPU buffers ── */
    size_t max_comp_sz = gpucompress_max_compressed_size(chunk_bytes);
    void* d_comp_buf = nullptr;
    void* d_decomp_buf = nullptr;
    cudaMalloc(&d_comp_buf, max_comp_sz);
    cudaMalloc(&d_decomp_buf, chunk_bytes);
    if (!d_comp_buf || !d_decomp_buf) {
        if (d_comp_buf) cudaFree(d_comp_buf);
        if (d_decomp_buf) cudaFree(d_decomp_buf);
        return -1;
    }

    /* ── 4. Warmup: compress mid-dataset chunk with each config once ── */
    {
        int warm_ci = n_chunks / 2;
        const uint8_t* d_warm = (const uint8_t*)d_data + (size_t)warm_ci * chunk_bytes;
        for (int ai = 0; ai < n_active; ai++) {
            int action = active_actions[ai];
            int algo_idx = action % 8;
            int quant   = (action / 8) % 2;
            int shuffle = (action / 16) % 2;

            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)(algo_idx + 1);
            cfg.preprocessing = (shuffle ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0)
                              | (quant   ? GPUCOMPRESS_PREPROC_QUANTIZE  : 0);
            cfg.error_bound = error_bound;

            size_t out_sz = max_comp_sz;
            gpucompress_compress_gpu(d_warm, chunk_bytes, d_comp_buf, &out_sz,
                                     &cfg, nullptr, nullptr);
        }
        cudaDeviceSynchronize();
    }

    /* ── 5. Config-major iteration: measure all chunks for each config ── */
    /* results[chunk_idx][config_idx] */
    std::vector<std::vector<ConfigResult>> results(n_chunks,
        std::vector<ConfigResult>(n_active));

    for (int ai = 0; ai < n_active; ai++) {
        int action = active_actions[ai];
        int algo_idx = action % 8;
        int quant   = (action / 8) % 2;
        int shuffle = (action / 16) % 2;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = (gpucompress_algorithm_t)(algo_idx + 1);
        cfg.preprocessing = (shuffle ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0)
                          | (quant   ? GPUCOMPRESS_PREPROC_QUANTIZE  : 0);
        cfg.error_bound = error_bound;

        for (int ci = 0; ci < n_chunks; ci++) {
            const uint8_t* d_chunk = (const uint8_t*)d_data + (size_t)ci * chunk_bytes;

            /* Collect n_repeats measurements, take median */
            std::vector<double> comp_times(n_repeats);
            std::vector<double> decomp_times(n_repeats);
            double ratio = 0.0;

            for (int rep = 0; rep < n_repeats; rep++) {
                gpucompress_stats_t stats = {};
                size_t out_sz = max_comp_sz;
                gpucompress_error_t err = gpucompress_compress_gpu(
                    d_chunk, chunk_bytes, d_comp_buf, &out_sz,
                    &cfg, &stats, nullptr);

                if (err != GPUCOMPRESS_SUCCESS) {
                    comp_times[rep] = 1e9;  /* failed → infinite cost */
                    decomp_times[rep] = 1e9;
                    continue;
                }

                comp_times[rep] = stats.actual_comp_time_ms;
                ratio = (out_sz > 0) ? (double)chunk_bytes / (double)out_sz : 0.0;

                /* Decompress: wall-clock bracketed */
                cudaDeviceSynchronize();
                auto t0 = std::chrono::steady_clock::now();
                size_t decomp_sz = chunk_bytes;
                gpucompress_decompress_gpu(d_comp_buf, out_sz,
                                            d_decomp_buf, &decomp_sz, nullptr);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::steady_clock::now();
                decomp_times[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            /* Take median */
            std::sort(comp_times.begin(), comp_times.end());
            std::sort(decomp_times.begin(), decomp_times.end());
            double med_comp = comp_times[n_repeats / 2];
            double med_decomp = decomp_times[n_repeats / 2];

            /* Apply policy clamps (must match NN kernel's cost formula) */
            double clamped_comp = fmax(5.0, med_comp);
            double clamped_decomp = fmax(5.0, med_decomp);
            double clamped_ratio = fmin(100.0, ratio);

            /* Cost formula: same as nn_gpu.cu:187-190 */
            double io_cost = (clamped_ratio > 0.0)
                ? (double)chunk_bytes / (clamped_ratio * (double)bw_bytes_per_ms)
                : 0.0;
            double cost = w0 * clamped_comp + w1 * clamped_decomp + w2 * io_cost;

            results[ci][ai] = {action, ratio, med_comp, med_decomp, cost};
        }
    }

    /* ── 6. Compute ranking metrics per chunk ── */
    std::vector<RankingChunkResult> chunk_results(n_chunks);
    double sum_tau = 0.0, sum_tau2 = 0.0;
    double sum_regret = 0.0;
    int    valid_chunks = 0;

    for (int ci = 0; ci < n_chunks; ci++) {
        /* Sort by actual cost → actual ranking */
        std::vector<int> actual_order(n_active);
        for (int i = 0; i < n_active; i++) actual_order[i] = i;
        std::sort(actual_order.begin(), actual_order.end(),
                  [&](int a, int b) { return results[ci][a].cost < results[ci][b].cost; });

        int actual_ranking[32];  /* action IDs sorted by actual cost */
        for (int i = 0; i < n_active; i++)
            actual_ranking[i] = results[ci][actual_order[i]].action_id;

        /* Get predicted ranking from snapshotted diagnostics */
        int predicted_ranking[32];
        int pred_count = 0;
        if (ci < n_hist && diags[ci].predicted_ranking_count > 0) {
            /* Filter to active configs only */
            for (int i = 0; i < diags[ci].predicted_ranking_count && pred_count < n_active; i++) {
                int act = diags[ci].predicted_ranking[i];
                int quant = (act / 8) % 2;
                if (error_bound <= 0.0 && quant) continue;
                predicted_ranking[pred_count++] = act;
            }
        }

        if (pred_count < 2) {
            /* No valid predicted ranking (non-AUTO phase) — skip */
            chunk_results[ci] = {ci, n_active, 0.0, 1.0, -1, actual_ranking[0], 0.0, results[ci][actual_order[0]].cost};
            continue;
        }

        /* Compute Kendall tau-b */
        double tau = compute_kendall_tau_b(predicted_ranking, actual_ranking, n_active);

        /* Top-1 regret: find cost of NN's predicted best in actual results */
        double nn_pick_cost = 0.0;
        for (int ai = 0; ai < n_active; ai++) {
            if (results[ci][ai].action_id == predicted_ranking[0]) {
                nn_pick_cost = results[ci][ai].cost;
                break;
            }
        }
        double oracle_cost = results[ci][actual_order[0]].cost;
        double regret = (oracle_cost > 0.0) ? nn_pick_cost / oracle_cost : 1.0;

        chunk_results[ci] = {ci, n_active, tau, regret,
                             predicted_ranking[0], actual_ranking[0],
                             nn_pick_cost, oracle_cost};

        sum_tau += tau;
        sum_tau2 += tau * tau;
        sum_regret += regret;
        valid_chunks++;
    }

    /* ── 7. Write CSV rows ── */
    auto wall_end = std::chrono::steady_clock::now();
    double profiling_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    for (int ci = 0; ci < n_chunks; ci++) {
        auto& cr = chunk_results[ci];
        fprintf(csv, "%s,%d,%d,%d,%.6f,%.6f,%d,%d,%.4f,%.4f,%.1f\n",
                phase_name, timestep, ci, cr.n_active_configs,
                cr.kendall_tau_b, cr.top1_regret,
                cr.predicted_best, cr.actual_best,
                cr.predicted_best_cost, cr.actual_best_cost,
                profiling_ms / n_chunks);
    }
    fflush(csv);

    /* ── 7b. Write per-config costs CSV (predicted vs actual side-by-side) ── */
    if (costs_csv) {
        for (int ci = 0; ci < n_chunks; ci++) {
            /* Build predicted ranking for this chunk (filtered to active configs) */
            int pred_ranking[32];
            int pred_count = 0;
            if (ci < n_hist && diags[ci].predicted_ranking_count > 0) {
                for (int i = 0; i < diags[ci].predicted_ranking_count && pred_count < n_active; i++) {
                    int act = diags[ci].predicted_ranking[i];
                    int quant = (act / 8) % 2;
                    if (error_bound <= 0.0 && quant) continue;
                    pred_ranking[pred_count++] = act;
                }
            }

            /* Sort configs by actual cost for this chunk */
            std::vector<int> aorder(n_active);
            for (int i = 0; i < n_active; i++) aorder[i] = i;
            std::sort(aorder.begin(), aorder.end(),
                      [&](int a, int b) { return results[ci][a].cost < results[ci][b].cost; });

            for (int ai = 0; ai < n_active; ai++) {
                int act = active_actions[ai];

                /* Find predicted rank */
                int pred_rank = n_active;
                for (int r = 0; r < pred_count; r++)
                    if (pred_ranking[r] == act) { pred_rank = r; break; }

                /* Find actual rank */
                int actual_rank = n_active;
                for (int r = 0; r < n_active; r++)
                    if (results[ci][aorder[r]].action_id == act) { actual_rank = r; break; }

                /* Get predicted cost (indexed by action ID) */
                float pred_cost = (ci < n_hist) ? diags[ci].predicted_costs[act] : 0.0f;
                /* Handle INFINITY for CSV output */
                if (pred_cost == INFINITY || pred_cost != pred_cost) pred_cost = 99999.0f;

                fprintf(costs_csv, "%s,%d,%d,%d,%s,%.4f,%.4f,%.4f,%.3f,%.3f,%d,%d\n",
                        phase_name, timestep, ci, act, action_name_str(act),
                        pred_cost, results[ci][ai].cost,
                        results[ci][ai].ratio, results[ci][ai].comp_ms, results[ci][ai].decomp_ms,
                        pred_rank, actual_rank);
            }
        }
        fflush(costs_csv);
    }

    /* ── 8. Aggregate ── */
    if (out) {
        out->timestep = timestep;
        out->n_chunks = n_chunks;
        if (valid_chunks > 0) {
            out->mean_tau = sum_tau / valid_chunks;
            double var = (sum_tau2 / valid_chunks) - (out->mean_tau * out->mean_tau);
            out->std_tau = (var > 0.0) ? sqrt(var) : 0.0;
            out->mean_regret = sum_regret / valid_chunks;
        } else {
            out->mean_tau = 0.0;
            out->std_tau = 0.0;
            out->mean_regret = 1.0;
        }
        out->profiling_ms = profiling_ms;
    }

    /* ── 9. Cleanup ── */
    cudaFree(d_comp_buf);
    cudaFree(d_decomp_buf);
    /* NOTE: gpucompress_flush_manager_cache() was removed here. It deleted all
     * nvcomp manager objects across all CompContext pool slots, whose destructors
     * issue cudaStreamSynchronize calls that corrupt KOKKOS's stream-ordering
     * state when called from inside a KOKKOS fix. The pool's LRU eviction handles
     * manager lifecycle correctly without explicit flushing after profiling. */

    return 0;
}

/* ════════════════════════════════════════════════════════════════
 * CSV header writer (call once when opening the ranking CSV)
 * ════════════════════════════════════════════════════════════════ */

static inline void write_ranking_csv_header(FILE* csv) {
    if (!csv) return;
    fprintf(csv, "phase,timestep,chunk,n_active_configs,"
                 "kendall_tau_b,top1_regret,"
                 "predicted_best,actual_best,"
                 "predicted_best_cost,actual_best_cost,"
                 "profiling_ms\n");
}

static inline void write_ranking_costs_csv_header(FILE* csv) {
    if (!csv) return;
    fprintf(csv, "phase,timestep,chunk,action,algo_name,"
                 "predicted_cost,actual_cost,"
                 "actual_ratio,actual_comp_ms,actual_decomp_ms,"
                 "predicted_rank,actual_rank\n");
}
