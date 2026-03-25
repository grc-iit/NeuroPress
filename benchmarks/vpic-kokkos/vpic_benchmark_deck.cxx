/**
 * @file vpic_benchmark_deck.cxx
 * @brief VPIC Benchmark Deck: No-Comp vs NN+SGD
 *
 * A real VPIC-Kokkos Harris sheet input deck that benchmarks GPU-resident
 * field data compression through the GPUCompress VOL connector.
 * Matches the benchmark_grayscott_vol structure.
 *
 * Pipeline per phase:
 *   VPIC simulation (GPU) → field_array->k_f_d → H5Dwrite(d_ptr) → VOL →
 *   GPU compress → HDF5 file → H5Dread(d_ptr) → GPU decompress → bitwise verify
 *
 * Single-shot phases (run once at timestep 0):
 *   1. no-comp      : H5Dwrite via VOL (no compression, baseline)
 *   2. fixed-*      : H5Dwrite via VOL (fixed algorithm baselines)
 *
 * Multi-timestep phases (run across all timesteps for fair comparison):
 *   3. nn           : ALGO_AUTO, inference-only (frozen weights)
 *   4. nn-rl        : ALGO_AUTO + online SGD (learning curve over N writes)
 *   5. nn-rl+exp50  : ALGO_AUTO + online SGD + exploration
 *
 * BUILD:
 *   See benchmarks/vpic-kokkos/build_vpic_benchmark.sh
 *
 * RUN:
 *   export LD_LIBRARY_PATH=$GPU_DIR/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
 *   export GPUCOMPRESS_WEIGHTS=$GPU_DIR/neural_net/weights/model.nnwt
 *   mpirun -np 1 ./vpic_benchmark_deck.Linux
 *
 * OUTPUT:
 *   Console: per-phase ratio, write/read MB/s, verification, SGD/exploration stats
 *   Files:   /tmp/bm_vpic_*.h5 (temporary, removed after each phase)
 */

// ============================================================
// GPUCompress + HDF5 headers
// ============================================================
#ifndef GPU_DIR
/* GPU_DIR is set via -DGPU_DIR=... by build_vpic_benchmark.sh.
 * This fallback is used only if the build script doesn't set it. */
#ifndef GPU_DIR
#define GPU_DIR "."
#endif
#endif
#include "gpucompress.h"
#include "gpucompress_vpic.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
#include "vpic_kokkos_bridge.hpp"

#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/* Kendall tau ranking profiler (CUDA, linked from vpic_ranking_profiler.cu) */
struct RankingMilestoneResult {
    int timestep; int n_chunks;
    double mean_tau, std_tau, top1_accuracy, mean_regret, profiling_ms;
};
extern "C" int vpic_run_ranking_profiler(
    const void* d_data, size_t total_bytes, size_t chunk_bytes,
    double error_bound, float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats, FILE* csv, FILE* costs_csv, const char* phase_name,
    int timestep, RankingMilestoneResult* out);
extern "C" int vpic_is_ranking_milestone(int t, int total);
extern "C" void vpic_write_ranking_csv_header(FILE* csv);
extern "C" void vpic_write_ranking_costs_csv_header(FILE* csv);
extern "C" float gpucompress_get_bandwidth_bytes_per_ms(void);

// ============================================================
// Constants
// ============================================================
#define REINFORCE_LR        0.1f
#define REINFORCE_MAPE      0.10f

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define N_PHASES 10  /* no-comp, 6 fixed (3+3 shuf), nn, nn-rl, nn-rl+exp50 */

#define TMP_NOCOMP    "/tmp/bm_vpic_nocomp.h5"
#define TMP_FIX_LZ4   "/tmp/bm_vpic_fix_lz4.h5"
#define TMP_FIX_SNAPPY "/tmp/bm_vpic_fix_snappy.h5"
#define TMP_FIX_DEFL  "/tmp/bm_vpic_fix_deflate.h5"
#define TMP_FIX_GDEFL "/tmp/bm_vpic_fix_gdefl.h5"
#define TMP_FIX_ZSTD  "/tmp/bm_vpic_fix_zstd.h5"
#define TMP_FIX_ANS   "/tmp/bm_vpic_fix_ans.h5"
#define TMP_FIX_CASC  "/tmp/bm_vpic_fix_cascaded.h5"
#define TMP_FIX_BITCOMP "/tmp/bm_vpic_fix_bitcomp.h5"
#define TMP_NN        "/tmp/bm_vpic_nn.h5"
#define TMP_NN_RL    "/tmp/bm_vpic_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_vpic_nn_rlexp.h5"
/* CSV output directory: set VPIC_RESULTS_DIR env var to override.
 * The eval script sets this per-run so CSVs land in the right subdirectory. */
static char RESULTS_DIR[512];
static char CHUNKS_CSV[600];
static char TSTEP_CSV[600];
static char TSTEP_CHUNKS_CSV[600];
static char RANKING_CSV[600];
static char RANKING_COSTS_CSV[600];
static char AGG_CSV[600];

static void init_csv_paths() {
    const char* env = getenv("VPIC_RESULTS_DIR");
    if (env && env[0]) {
        snprintf(RESULTS_DIR, sizeof(RESULTS_DIR), "%s", env);
    } else {
        snprintf(RESULTS_DIR, sizeof(RESULTS_DIR),
                 "%s/benchmarks/vpic-kokkos/results", GPU_DIR);
    }
    mkdir(RESULTS_DIR, 0755);
    snprintf(CHUNKS_CSV, sizeof(CHUNKS_CSV),
             "%s/benchmark_vpic_deck_chunks.csv", RESULTS_DIR);
    snprintf(TSTEP_CSV, sizeof(TSTEP_CSV),
             "%s/benchmark_vpic_deck_timesteps.csv", RESULTS_DIR);
    snprintf(TSTEP_CHUNKS_CSV, sizeof(TSTEP_CHUNKS_CSV),
             "%s/benchmark_vpic_deck_timestep_chunks.csv", RESULTS_DIR);
    snprintf(RANKING_CSV, sizeof(RANKING_CSV),
             "%s/benchmark_vpic_deck_ranking.csv", RESULTS_DIR);
    snprintf(RANKING_COSTS_CSV, sizeof(RANKING_COSTS_CSV),
             "%s/benchmark_vpic_deck_ranking_costs.csv", RESULTS_DIR);
    snprintf(AGG_CSV, sizeof(AGG_CSV),
             "%s/benchmark_vpic_deck.csv", RESULTS_DIR);
}

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                sim_steps;         // Warmup steps before benchmarking
    int                timesteps;         // Number of multi-timestep writes (0 = single-shot)
    int                ts_count;          // Current timestep counter
    size_t             chunk_bytes;       // HDF5 chunk size in bytes
    gpucompress_vpic_t vpic_fields_h;     // Adapter handle for fields
    hid_t              vol_fapl;          // File access property list with VOL
    hid_t              vol_id;            // VOL connector ID
    int                gpucompress_ready; // 1 if init succeeded
    int                benchmark_done;    // 1 after all benchmark steps complete
    int                single_shot_done;  // 1 after single-shot phases complete
    double             diag_error_bound;  // Error bound for diagnostics
    FILE*              ts_csv;            // Timestep CSV file handle
    FILE*              tc_csv;            // Timestep per-chunk CSV file handle

    // Buffers for benchmark
    float*             d_read;            // GPU read-back buffer
    float*             h_orig;            // Host buffer for verification
    float*             h_read;            // Host buffer for verification

    // Per-phase NN weight snapshots for cross-timestep learning
    void*              nn_weights[3];     // host buffers: [0]=nn, [1]=nn-rl, [2]=nn-rl+exp50
    size_t             nn_weights_size;   // sizeof(NNWeightsGPU)
    int                nn_weights_init;   // 1 after first timestep initializes snapshots

    // Configurable hyperparameters
    float              reinforce_lr;      // SGD learning rate (default 0.9, env VPIC_LR)
    float              reinforce_mape;    // MAPE threshold for SGD (default 0.20, env VPIC_MAPE_THRESHOLD)
    int                n_runs;            // Number of single-shot repetitions (default 1, env VPIC_RUNS)

    // Ranking quality profiler
    FILE*              ranking_csv;       // Kendall tau ranking CSV
    FILE*              ranking_costs_csv; // Per-config predicted vs actual costs
    float              rank_w0, rank_w1, rank_w2;  // Cost model weights for profiler

    // Phase exclusion: VPIC_EXCLUDE="lz4,no-comp,zstd" skips matching phases
    char               exclude_list[512]; // comma-separated substrings to exclude
};

// ============================================================
// Helper: timing
// ============================================================
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================
// Helper: check if a phase name is excluded
// ============================================================
static bool is_phase_excluded(const char* phase_name, const char* exclude_list) {
    if (!exclude_list || !exclude_list[0]) return false;
    char buf[512];
    strncpy(buf, exclude_list, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char* tok = strtok(buf, ",");
    while (tok) {
        /* Trim leading spaces */
        while (*tok == ' ') tok++;
        /* Match: phase "fixed-lz4" excluded by "lz4" */
        if (strstr(phase_name, tok)) return true;
        tok = strtok(NULL, ",");
    }
    return false;
}

// ============================================================
// Helper: file size
// ============================================================
static size_t get_file_size(const char* path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

static void drop_pagecache(const char* path)
{
    int fd = open(path, O_RDWR);
    if (fd < 0) return;
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

// ============================================================
// Helper: pack double into two unsigned ints for HDF5 cd_values
// ============================================================
static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

// ============================================================
// Helper: decode NN action to readable string
// ============================================================
static const char* ACTION_ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_str(int action, char *buf, size_t bufsz)
{
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ACTION_ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

// ============================================================
// Result struct
// ============================================================
struct PhaseResult {
    char   phase[20];
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    double read_mbps;
    unsigned long long mismatches;
    int    sgd_fires;
    int    explorations;
    int    n_chunks;
    double mape_ratio_pct;
    double mape_comp_pct;
    double mape_decomp_pct;
    double mae_ratio;
    double mae_comp_ms;
    double mae_decomp_ms;
    double r2_ratio;
    double stats_ms;
    double nn_ms;
    double preproc_ms;
    double comp_ms;
    double decomp_ms;
    double explore_ms;
    double sgd_ms;
    double comp_gbps;
    double decomp_gbps;
    /* Standard deviations (populated when VPIC_RUNS > 1) */
    double write_ms_std;
    double read_ms_std;
    double comp_gbps_std;
    double decomp_gbps_std;
    int    n_runs;
};

// ============================================================
// Statistics helpers
// ============================================================
static void compute_mean_std(const double *vals, int n, double *mean, double *std)
{
    if (n <= 0) { *mean = 0; *std = 0; return; }
    double sum = 0;
    for (int i = 0; i < n; i++) sum += vals[i];
    *mean = sum / n;
    if (n <= 1) { *std = 0; return; }
    double var = 0;
    for (int i = 0; i < n; i++) {
        double d = vals[i] - *mean;
        var += d * d;
    }
    *std = sqrt(var / (n - 1));
}

static void merge_phase_results(PhaseResult *runs, int n, PhaseResult *out)
{
    *out = runs[n - 1];  /* copy last run as base */
    out->n_runs = n;

    double wr[32], rd[32], cg[32], dg[32];
    int cap = (n > 32) ? 32 : n;
    for (int i = 0; i < cap; i++) {
        wr[i] = runs[i].write_ms;
        rd[i] = runs[i].read_ms;
        cg[i] = runs[i].comp_gbps;
        dg[i] = runs[i].decomp_gbps;
    }

    double m, s;
    compute_mean_std(wr, cap, &m, &s);
    out->write_ms = m; out->write_ms_std = s;
    compute_mean_std(rd, cap, &m, &s);
    out->read_ms = m; out->read_ms_std = s;
    compute_mean_std(cg, cap, &m, &s);
    out->comp_gbps = m; out->comp_gbps_std = s;
    compute_mean_std(dg, cap, &m, &s);
    out->decomp_gbps = m; out->decomp_gbps_std = s;

    double total_bytes = (double)out->orig_bytes;
    out->write_mbps = (out->write_ms > 0)
        ? total_bytes / (1 << 20) / (out->write_ms / 1000.0) : 0;
    out->read_mbps = (out->read_ms > 0)
        ? total_bytes / (1 << 20) / (out->read_ms / 1000.0) : 0;
}

// ============================================================
// VOL FAPL
// ============================================================
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

// ============================================================
// DCPL builders (1D chunked)
// ============================================================
static hid_t make_dcpl_nocomp(hsize_t chunk_floats)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    return dcpl;
}

static hid_t make_dcpl_fixed(hsize_t chunk_floats, gpucompress_algorithm_t algo,
                              unsigned int preproc = 0)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo;
    cd[1] = preproc;
    cd[2] = 0;
    cd[3] = 0; cd[4] = 0; /* error_bound = 0.0 */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static hid_t make_dcpl_auto(hsize_t chunk_floats, double eb = 0.0)
{
    hsize_t cdims[1] = { chunk_floats };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; // ALGO_AUTO
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(eb, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

// ============================================================
// Bitwise comparison (D→H copy then CPU compare)
// VPIC deck is compiled through VPIC's .cxx wrapper which does
// not support __global__ kernels, so we compare on the host.
// ============================================================
static unsigned long long host_compare(const float* d_a, const float* d_b,
                                       float* h_a, float* h_b,
                                       size_t n_floats)
{
    size_t bytes = n_floats * sizeof(float);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    unsigned long long mm = 0;
    for (size_t i = 0; i < n_floats; i++) {
        unsigned int ua, ub;
        memcpy(&ua, &h_a[i], sizeof(unsigned int));
        memcpy(&ub, &h_b[i], sizeof(unsigned int));
        if (ua != ub) mm++;
    }
    return mm;
}

// ============================================================
// Run one benchmark phase: write → read → verify
// ============================================================
static int run_phase(const char* phase_name, const char* tmp_file,
                     float* d_data, float* d_read,
                     float* h_a, float* h_b,
                     size_t n_floats, int n_chunks, hid_t dcpl,
                     PhaseResult* r)
{
    size_t total_bytes = n_floats * sizeof(float);
    hsize_t dims[1] = { (hsize_t)n_floats };

    // Warmup write: primes VOL write context, nvCOMP manager cache,
    // and NN inference JIT.  Results discarded — only the timed run counts.
    {
        const char *warmup_file = "/tmp/bm_vpic_warmup.h5";
        gpucompress_disable_online_learning();

        remove(warmup_file);
        hid_t wfapl = make_vol_fapl();
        hid_t wfile = H5Fcreate(warmup_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
        H5Pclose(wfapl);
        if (wfile >= 0) {
            hid_t wfsp  = H5Screate_simple(1, dims, NULL);
            hid_t wdset = H5Dcreate2(wfile, "fields", H5T_NATIVE_FLOAT,
                                      wfsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(wfsp);
            if (wdset >= 0) {
                H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
                cudaDeviceSynchronize();
                H5Dclose(wdset);
            }
            H5Fclose(wfile);
            remove(warmup_file);
        }
        gpucompress_flush_manager_cache();  /* cold-start the timed run */
    }

    // VOL write
    printf("  [%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
    gpucompress_set_debug_context(phase_name, -1);
    H5VL_gpucompress_reset_stats();

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed for %s\n", tmp_file); return 1; }

    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "fields", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0   = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t1   = now_ms();
    printf("%.0f ms\n", t1 - t0);

    if (wret < 0) { fprintf(stderr, "  [%s] H5Dwrite failed\n", phase_name); return 1; }

    drop_pagecache(tmp_file);

    // VOL read
    printf("  [%s] H5Dread (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "fields", H5P_DEFAULT);

    double t2   = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);
    double t3   = now_ms();
    printf("%.0f ms\n", t3 - t2);

    if (rret < 0) { fprintf(stderr, "  [%s] H5Dread failed\n", phase_name); return 1; }

    // Bitwise verify (D→H then CPU compare; VPIC .cxx can't use __global__)
    unsigned long long mm = host_compare(d_data, d_read, h_a, h_b, n_floats);

    // Collect per-chunk diagnostics and write to CSV
    int sgd_fires    = 0;
    int explorations = 0;
    double mape_ratio_sum = 0.0, mape_comp_sum = 0.0, mape_decomp_sum = 0.0;
    int    mape_ratio_cnt = 0,   mape_comp_cnt = 0,   mape_decomp_cnt = 0;
    double mae_r_sum = 0.0, mae_c_sum = 0.0, mae_d_sum = 0.0;
    double ss_res_r = 0.0, sum_ratio = 0.0;
    int    r2_cnt = 0;
    double total_stats_ms   = 0.0;
    double total_nn_ms      = 0.0;
    double total_preproc_ms = 0.0;
    double total_comp_ms    = 0.0;
    double total_decomp_ms  = 0.0;
    double total_explore_ms = 0.0;
    double total_sgd_ms     = 0.0;
    int n_hist       = gpucompress_get_chunk_history_count();
    FILE *chunk_csv  = NULL;
    if (n_hist > 0) {
        printf("    chunk | action (final)       | action (orig)        | ratio  | pred   | MAPE   | sgd | expl\n");
        printf("    ------+----------------------+----------------------+--------+--------+--------+-----+-----\n");
        // Open chunks CSV: create with header on first phase, append on subsequent
        struct stat st;
        bool need_header = (stat(CHUNKS_CSV, &st) != 0 || st.st_size == 0);
        chunk_csv = fopen(CHUNKS_CSV, "a");
        if (chunk_csv && need_header) {
            fprintf(chunk_csv, "phase,chunk,action_final,action_orig,"
                               "actual_ratio,predicted_ratio,mape_ratio,"
                               "actual_comp_ms_raw,predicted_comp_ms,mape_comp,"
                               "actual_decomp_ms_raw,predicted_decomp_ms,mape_decomp,"
                               "sgd_fired,exploration_triggered,"
                               "cost_model_error_pct,actual_cost,predicted_cost,"
                               "explore_n_alt");
            for (int ei = 0; ei < 31; ei++)
                fprintf(chunk_csv, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d",
                        ei, ei, ei, ei);
            fprintf(chunk_csv, ","
                               "stats_ms,nn_inference_ms,preprocessing_ms,compression_ms_raw,"
                               "exploration_ms,sgd_update_ms,"
                               "feat_entropy,feat_mad,feat_deriv,feat_eb_enc,feat_ds_enc\n");
        }
    }
    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            sgd_fires    += d.sgd_fired;
            explorations += d.exploration_triggered;
            total_stats_ms   += d.stats_ms;
            total_nn_ms      += d.nn_inference_ms;
            total_preproc_ms += d.preprocessing_ms;
            total_comp_ms    += d.compression_ms_raw;   /* unclamped for breakdown */
            total_decomp_ms  += d.decompression_ms_raw; /* unclamped for breakdown */
            total_explore_ms += d.exploration_ms;
            total_sgd_ms     += d.sgd_update_ms;
            /* MAPE: |pred - actual| / |actual| */
            if (d.actual_ratio > 0) {
                mape_ratio_sum += fabs(d.predicted_ratio - d.actual_ratio) / fabs(d.actual_ratio) * 100.0;
                mape_ratio_cnt++;
            }
            if (d.compression_ms > 0) {
                mape_comp_sum += fabs(d.predicted_comp_time - d.compression_ms) / fabs(d.compression_ms) * 100.0;
                mape_comp_cnt++;
            }
            if (d.decompression_ms > 0) {
                mape_decomp_sum += fabs(d.predicted_decomp_time - d.decompression_ms) / fabs(d.decompression_ms) * 100.0;
                mape_decomp_cnt++;
            }
            /* MAE and R² accumulators */
            if (d.actual_ratio > 0 && d.predicted_ratio > 0) {
                mae_r_sum += fabs(d.predicted_ratio - d.actual_ratio);
                ss_res_r  += (d.predicted_ratio - d.actual_ratio) * (d.predicted_ratio - d.actual_ratio);
                sum_ratio += d.actual_ratio;
                r2_cnt++;
            }
            if (d.compression_ms_raw > 0)
                mae_c_sum += fabs(d.predicted_comp_time - d.compression_ms);
            if (d.decompression_ms > 0)
                mae_d_sum += fabs(d.predicted_decomp_time - d.decompression_ms);
            char final_str[40], orig_str[40];
            action_to_str(d.nn_action, final_str, sizeof(final_str));
            action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));
            double chunk_mape = (d.actual_ratio > 0)
                ? fmin(200.0, fabs((double)d.predicted_ratio - (double)d.actual_ratio) / fabs((double)d.actual_ratio) * 100.0)
                : 0.0;
            printf("    %5d | %-20s | %-20s | %5.2fx | %5.2fx | %5.1f%% | %s | %s\n",
                   i + 1, final_str, orig_str,
                   (double)d.actual_ratio, (double)d.predicted_ratio,
                   chunk_mape,
                   d.sgd_fired ? "yes" : "  -",
                   d.exploration_triggered ? "yes" : "  -");
            if (chunk_csv) {
                /* Per-chunk MAPE */
                double mr = fmin(200.0, (d.actual_ratio > 0) ? fabs((double)d.predicted_ratio - (double)d.actual_ratio) / fabs((double)d.actual_ratio) * 100.0 : 0.0);
                double mc = fmin(200.0, (d.compression_ms > 0) ? fabs((double)d.predicted_comp_time - (double)d.compression_ms) / fabs((double)d.compression_ms) * 100.0 : 0.0);
                double md = fmin(200.0, (d.decompression_ms > 0) ? fabs((double)d.predicted_decomp_time - (double)d.decompression_ms) / fabs((double)d.decompression_ms) * 100.0 : 0.0);

                fprintf(chunk_csv, "%s,%d,%s,%s,"
                                   "%.4f,%.4f,%.1f,"
                                   "%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,"
                                   "%d,%d,"
                                   "%.4f,%.4f,%.4f,%d",
                        phase_name, i + 1, final_str, orig_str,
                        (double)d.actual_ratio, (double)d.predicted_ratio, mr,
                        (double)d.compression_ms_raw, (double)d.predicted_comp_time, mc,
                        (double)d.decompression_ms_raw, (double)d.predicted_decomp_time, md,
                        d.sgd_fired, d.exploration_triggered,
                        (double)d.cost_model_error_pct,
                        (double)d.actual_cost, (double)d.predicted_cost,
                        d.explore_n_alternatives);
                /* Exploration alternatives (31 slots) */
                for (int ei = 0; ei < 31; ei++) {
                    if (ei < d.explore_n_alternatives) {
                        char alt_str[40];
                        action_to_str(d.explore_alternatives[ei], alt_str, sizeof(alt_str));
                        fprintf(chunk_csv, ",%s,%.4f,%.4f,%.4f",
                                alt_str,
                                (double)d.explore_ratios[ei],
                                (double)d.explore_comp_ms[ei],
                                (double)d.explore_costs[ei]);
                    } else {
                        fprintf(chunk_csv, ",,,,");
                    }
                }
                fprintf(chunk_csv, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                                   "%.4f,%.6f,%.6f,%.4f,%.4f\n",
                        (double)d.stats_ms, (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                        (double)d.compression_ms_raw, (double)d.exploration_ms,
                        (double)d.sgd_update_ms,
                        (double)d.feat_entropy, (double)d.feat_mad, (double)d.feat_deriv,
                        (double)d.feat_eb_enc, (double)d.feat_ds_enc);
            }
        }
    }
    if (chunk_csv) fclose(chunk_csv);

    size_t fbytes = get_file_size(tmp_file);

    r->write_ms     = t1 - t0;
    r->read_ms      = t3 - t2;
    r->file_bytes   = fbytes;
    r->orig_bytes   = total_bytes;
    r->ratio        = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps   = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps    = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches   = mm;
    r->sgd_fires    = sgd_fires;
    r->explorations = explorations;
    r->n_chunks     = n_chunks;
    r->mape_ratio_pct  = fmin(200.0, (mape_ratio_cnt > 0) ? mape_ratio_sum / mape_ratio_cnt : 0.0);
    r->mape_comp_pct   = fmin(200.0, (mape_comp_cnt > 0) ? mape_comp_sum / mape_comp_cnt : 0.0);
    r->mape_decomp_pct = fmin(200.0, (mape_decomp_cnt > 0) ? mape_decomp_sum / mape_decomp_cnt : 0.0);
    r->mae_ratio     = r2_cnt ? mae_r_sum / r2_cnt : 0.0;
    r->mae_comp_ms   = mape_comp_cnt  ? mae_c_sum / mape_comp_cnt  : 0.0;
    r->mae_decomp_ms = mape_decomp_cnt ? mae_d_sum / mape_decomp_cnt : 0.0;
    if (r2_cnt > 1) {
        double mean_r = sum_ratio / r2_cnt;
        double ss_tot = 0.0;
        /* Second pass for R²: need SS_tot (same guard as ss_res_r) */
        for (int i = 0; i < n_hist; i++) {
            gpucompress_chunk_diag_t d2;
            if (gpucompress_get_chunk_diag(i, &d2) == 0
                && d2.actual_ratio > 0 && d2.predicted_ratio > 0) {
                double diff = d2.actual_ratio - mean_r;
                ss_tot += diff * diff;
            }
        }
        r->r2_ratio = (ss_tot > 0) ? 1.0 - ss_res_r / ss_tot : 0.0;
    } else {
        r->r2_ratio = 0.0;
    }
    r->stats_ms     = total_stats_ms;
    r->nn_ms        = total_nn_ms;
    r->preproc_ms   = total_preproc_ms;
    r->comp_ms      = total_comp_ms;
    r->decomp_ms    = total_decomp_ms;
    r->explore_ms   = total_explore_ms;
    r->sgd_ms       = total_sgd_ms;
    r->comp_gbps    = (total_comp_ms > 0)
        ? (double)total_bytes / 1e9 / (total_comp_ms / 1000.0) : 0.0;
    r->decomp_gbps  = (total_decomp_ms > 0)
        ? (double)total_bytes / 1e9 / (total_decomp_ms / 1000.0) : 0.0;
    r->write_ms_std = 0;
    r->read_ms_std  = 0;
    r->comp_gbps_std  = 0;
    r->decomp_gbps_std = 0;
    r->n_runs       = 1;
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);

    printf("  [%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
           "file=%.0f MiB  sgd=%d expl=%d/%d  mismatches=%llu\n",
           phase_name, r->ratio, r->write_mbps, r->read_mbps,
           (double)fbytes / (1 << 20), sgd_fires, explorations, n_hist, mm);

    double total_tracked = total_stats_ms + total_nn_ms + total_preproc_ms
                         + total_comp_ms + total_explore_ms + total_sgd_ms;
    double write_ms = t1 - t0;
    printf("  [%s] Overhead breakdown (%d chunks, write=%.1f ms, total GPU-time=%.1f ms):\n",
           phase_name, n_hist, write_ms, total_tracked);
    printf("    Stats compute: %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_stats_ms, total_tracked > 0 ? 100.0 * total_stats_ms / total_tracked : 0.0);
    printf("    NN inference : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_nn_ms, total_tracked > 0 ? 100.0 * total_nn_ms / total_tracked : 0.0);
    printf("    Preprocessing: %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_preproc_ms, total_tracked > 0 ? 100.0 * total_preproc_ms / total_tracked : 0.0);
    printf("    Compression  : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_comp_ms, total_tracked > 0 ? 100.0 * total_comp_ms / total_tracked : 0.0);
    printf("    Exploration  : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_explore_ms, total_tracked > 0 ? 100.0 * total_explore_ms / total_tracked : 0.0);
    printf("    SGD update   : %8.1f ms  (%4.1f%% of GPU-time)\n",
           total_sgd_ms, total_tracked > 0 ? 100.0 * total_sgd_ms / total_tracked : 0.0);

    remove(tmp_file);
    return (mm == 0) ? 0 : 1;
}

// ============================================================
// Initialization: Harris sheet + GPUCompress + VOL setup
// ============================================================
begin_initialization {
    // ---- Physics (Harris sheet reconnection) ----
    double L    = 1;
    double ec   = 1;
    double me   = 1;
    double c    = 1;
    double eps0 = 1;

    double mi_me   = 25;
    double rhoi_L  = 1;
    double Ti_Te   = 1;
    double wpe_wce = 3;

    // Grid size configurable via VPIC_NX env var (default 200 ≈ 512 MB)
    // Field data = (nx+2)^3 * 16 * 4 bytes
    //   128 → ~134 MB    200 → ~520 MB    256 → ~1.1 GB    404 → ~4.0 GB
    const char* env_nx = getenv("VPIC_NX");
    int grid_n = env_nx ? atoi(env_nx) : 200;
    if (grid_n < 16) grid_n = 16;

    double Lx   = 80*L;
    double Ly   = 80*L;
    double Lz   = 80*L;
    double nx   = grid_n;
    double ny   = grid_n;
    double nz   = grid_n;
    double nppc = 2;

    double damp      = 0.001;
    double cfl_req   = 0.99;
    double wpedt_max = 0.36;

    double mi   = me*mi_me;
    double kTe  = me*c*c/(2*wpe_wce*wpe_wce*(1+Ti_Te));
    double kTi  = kTe*Ti_Te;
    double vthi = sqrt(2*kTi/mi);
    double wci  = vthi/(rhoi_L*L);
    double wce  = wci*mi_me;
    double wpe  = wce*wpe_wce;
    double vdre = c*c*wce/(wpe*wpe*L*(1+Ti_Te));
    double vdri = -Ti_Te*vdre;
    double b0   = me*wce/ec;
    double n0   = me*eps0*wpe*wpe/(ec*ec);
    double Npe  = 2*n0*Ly*Lz*L*tanh(0.5*Lx/L);
    double Npi  = Npe;
    double Ne   = 0.5*nppc*nx*ny*nz;
    Ne = trunc_granular(Ne, nproc());
    double Ni   = Ne;
    double we   = Npe/Ne;
    double wi   = Npi/Ni;
    double gdri = 1/sqrt(1 - vdri*vdri/(c*c));
    double gdre = 1/sqrt(1 - vdre*vdre/(c*c));
    double udri = vdri*gdri;
    double udre = vdre*gdre;
    double uthi = sqrt(kTi/mi)/c;
    double uthe = sqrt(kTe/me)/c;

    double dg = courant_length(Lx, Ly, Lz, nx, ny, nz);
    double dt = cfl_req*dg/c;
    if (wpe*dt > wpedt_max) dt = wpedt_max/wpe;

    // Warmup steps and chunk size configurable via environment
    const char* env_steps = getenv("VPIC_WARMUP_STEPS");
    int warmup = env_steps ? atoi(env_steps) : 100;
    if (warmup < 1) warmup = 1;

    const char* env_chunk = getenv("VPIC_CHUNK_MB");
    int chunk_mb = env_chunk ? atoi(env_chunk) : 8;
    if (chunk_mb < 1) chunk_mb = 1;

    const char* env_ts = getenv("VPIC_TIMESTEPS");
    int timesteps = env_ts ? atoi(env_ts) : 0;
    if (timesteps < 0) timesteps = 0;

    const char* env_w0 = getenv("VPIC_W0");
    const char* env_w1 = getenv("VPIC_W1");
    const char* env_w2 = getenv("VPIC_W2");
    float rank_w0 = env_w0 ? (float)atof(env_w0) : 1.0f;
    float rank_w1 = env_w1 ? (float)atof(env_w1) : 1.0f;
    float rank_w2 = env_w2 ? (float)atof(env_w2) : 1.0f;

    const char* env_runs = getenv("VPIC_RUNS");
    int n_runs = env_runs ? atoi(env_runs) : 1;
    if (n_runs < 1) n_runs = 1;
    if (n_runs > 32) n_runs = 32;

    const char* env_lr = getenv("VPIC_LR");
    const char* env_mape = getenv("VPIC_MAPE_THRESHOLD");
    float reinforce_lr   = env_lr   ? (float)atof(env_lr)   : REINFORCE_LR;
    float reinforce_mape = env_mape ? (float)atof(env_mape) : REINFORCE_MAPE;

    // Run warmup steps, then single-shot phases (1 step), then multi-timestep writes
    num_step        = warmup + 1 + timesteps;
    status_interval = 50;
    clean_div_e_interval = 10;
    clean_div_b_interval = 10;

    global->sim_steps       = warmup;
    global->timesteps       = timesteps;
    global->ts_count        = 0;
    global->chunk_bytes     = (size_t)chunk_mb * 1024 * 1024;
    global->benchmark_done  = 0;
    /* Always run single-shot phases first (summary + aggregate CSV),
     * then multi-timestep loop if requested.  Matches Gray-Scott behavior. */
    global->single_shot_done = 0;
    global->diag_error_bound = 0.0;

    /* Phase exclusion: VPIC_EXCLUDE="lz4,no-comp,zstd" */
    const char* env_exclude = getenv("VPIC_EXCLUDE");
    if (env_exclude && env_exclude[0]) {
        strncpy(global->exclude_list, env_exclude, sizeof(global->exclude_list) - 1);
        global->exclude_list[sizeof(global->exclude_list) - 1] = '\0';
    } else {
        global->exclude_list[0] = '\0';
    }
    global->ts_csv          = NULL;
    global->tc_csv          = NULL;
    global->nn_weights[0]   = NULL;
    global->nn_weights[1]   = NULL;
    global->nn_weights[2]   = NULL;
    global->nn_weights_size = 0;
    global->nn_weights_init = 0;
    global->reinforce_lr    = reinforce_lr;
    global->reinforce_mape  = reinforce_mape;
    global->n_runs          = n_runs;

    // Grid setup
    define_units(c, eps0);
    define_timestep(dt);
    define_periodic_grid(-0.5*Lx, 0, 0,
                          0.5*Lx, Ly, Lz,
                          nx, ny, nz,
                          1, nproc(), 1);

    set_domain_field_bc(BOUNDARY(-1,0,0), pec_fields);
    set_domain_field_bc(BOUNDARY( 1,0,0), pec_fields);
    set_domain_particle_bc(BOUNDARY(-1,0,0), reflect_particles);
    set_domain_particle_bc(BOUNDARY( 1,0,0), reflect_particles);

    define_material("vacuum", 1);
    define_field_array(NULL, damp);

    species_t* ion      = define_species("ion",       ec, mi, 1.5*Ni/nproc(), -1, 5, 1);
    species_t* electron = define_species("electron", -ec, me, 1.5*Ne/nproc(), -1, 5, 1);

    // Load fields (Harris current sheet)
    set_region_field(everywhere,
                     0, 0, 0,
                     0, 0, b0*tanh(x/L));

    // Load particles (drifting Maxwellians)
    double ymin = rank()*Ly/nproc();
    double ymax = (rank()+1)*Ly/nproc();

    repeat(Ni/nproc()) {
        double px, py, pz, ux, uy, uz, d0;
        do { px = L*atanh(uniform(rng(0), -1, 1)); }
        while (px <= -0.5*Lx || px >= 0.5*Lx);
        py = uniform(rng(0), ymin, ymax);
        pz = uniform(rng(0), 0, Lz);

        ux = normal(rng(0), 0, uthi);
        uy = normal(rng(0), 0, uthi);
        uz = normal(rng(0), 0, uthi);
        d0 = gdri*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udri;
        uy = d0; uz = uz;
        inject_particle(ion, px, py, pz, ux, uy, uz, wi, 0, 0);

        ux = normal(rng(0), 0, uthe);
        uy = normal(rng(0), 0, uthe);
        uz = normal(rng(0), 0, uthe);
        d0 = gdre*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udre;
        uy = d0; uz = uz;
        inject_particle(electron, px, py, pz, ux, uy, uz, we, 0, 0);
    }

    // ---- GPUCompress + HDF5 VOL initialization ----
    global->gpucompress_ready = 0;
    global->d_read    = NULL;
    global->h_orig    = NULL;
    global->h_read    = NULL;

    const char* weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    gpucompress_error_t gerr = gpucompress_init(weights_path);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        sim_log("FATAL: gpucompress_init failed (" << gerr << ")");
        return;
    }

    if (weights_path && !gpucompress_nn_is_loaded()) {
        sim_log("WARNING: NN weights not loaded from " << weights_path);
    }

    H5Z_gpucompress_register();
    global->vol_id = H5VL_gpucompress_register();

    hid_t native_id = H5VLget_connector_id_by_name("native");
    global->vol_fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(global->vol_fapl, native_id, NULL);
    H5VLclose(native_id);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    H5VL_gpucompress_set_trace(0);
    gpucompress_set_ranking_weights(rank_w0, rank_w1, rank_w2);
    global->rank_w0 = rank_w0;
    global->rank_w1 = rank_w1;
    global->rank_w2 = rank_w2;
    global->gpucompress_ready = 1;

    // Create VPIC adapter handle for fields
    VpicSettings fs = vpic_default_settings();
    fs.data_type = VPIC_DATA_FIELDS;
    fs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_fields_h, &fs);

    size_t field_bytes = (size_t)grid->nv * 16 * sizeof(float);
    sim_log("=== VPIC Benchmark Deck: Harris Sheet Reconnection ===");
    sim_log("  Grid     : " << (int)nx << "x" << (int)ny << "x" << (int)nz
            << " = " << grid->nv << " voxels");
    sim_log("  Fields   : " << field_bytes / (1024*1024) << " MB (16 vars x "
            << grid->nv << " cells)");
    sim_log("  Chunks   : " << global->chunk_bytes / (1024*1024) << " MB each");
    sim_log("  Particles: " << nppc << " per cell");
    sim_log("  Warmup   : " << global->sim_steps << " steps");
    if (timesteps > 0)
        sim_log("  Timesteps: " << timesteps << " (multi-timestep nn-rl writes after single-shot)");
    sim_log("  Env vars : VPIC_NX=" << grid_n << " VPIC_CHUNK_MB=" << chunk_mb
            << " VPIC_WARMUP_STEPS=" << warmup
            << " VPIC_TIMESTEPS=" << timesteps);
    sim_log("  Weights  : " << (weights_path ? weights_path : "(none, fallback to LZ4)"));
    sim_log("  NN loaded: " << (gpucompress_nn_is_loaded() ? "yes" : "no"));
    sim_log("  Cost w0/w1/w2: " << rank_w0 << " / " << rank_w1 << " / " << rank_w2);
    sim_log("  SGD LR: " << reinforce_lr << "  MAPE threshold: " << reinforce_mape);
    if (global->exclude_list[0])
        sim_log("  Exclude: " << global->exclude_list);
    if (n_runs > 1)
        sim_log("  Runs   : " << n_runs << " (single-shot phases repeated for error bars)");

    // Allocate GPU read-back buffer and host verification buffers
    cudaMalloc(&global->d_read, field_bytes);
    global->h_orig = (float*)malloc(field_bytes);
    global->h_read = (float*)malloc(field_bytes);
};

// ============================================================
// Diagnostics: run benchmark across multiple timesteps
// ============================================================
begin_diagnostics {
    if (global->benchmark_done) return;
    if (step() < global->sim_steps) return;
    if (!global->gpucompress_ready) return;

    /* Initialize CSV paths on first invocation */
    static bool csv_paths_init = false;
    if (!csv_paths_init) { init_csv_paths(); csv_paths_init = true; }

    // Attach GPU-resident field data (fresh pointer each step — fields evolve)
    vpic_attach_fields(global->vpic_fields_h, field_array->k_f_d);

    float*  d_fields = NULL;
    size_t  nbytes_f = 0;
    gpucompress_vpic_get_device_ptrs(global->vpic_fields_h,
                                     &d_fields, NULL, &nbytes_f, NULL);
    size_t n_floats    = nbytes_f / sizeof(float);
    size_t chunk_floats = global->chunk_bytes / sizeof(float);
    int    n_chunks    = (int)((n_floats + chunk_floats - 1) / chunk_floats);
    double orig_mib    = (double)nbytes_f / (1 << 20);

    // ============================================================
    // Multi-timestep mode: all phases per timestep (apple-to-apple on same data)
    // ============================================================
    if (global->single_shot_done) {
        if (global->timesteps <= 0 || global->ts_count >= global->timesteps) {
            // All timesteps done — close CSV and cleanup
            if (global->ts_csv) {
                fclose(global->ts_csv);
                global->ts_csv = NULL;
                printf("\n  Timestep CSV: %s\n", TSTEP_CSV);
            }
            if (global->tc_csv) {
                fclose(global->tc_csv);
                global->tc_csv = NULL;
                printf("  Timestep chunks CSV: %s\n", TSTEP_CHUNKS_CSV);
            }
            if (global->ranking_csv) {
                fclose(global->ranking_csv);
                global->ranking_csv = NULL;
                printf("  Ranking quality CSV: %s\n", RANKING_CSV);
            }
            if (global->ranking_costs_csv) {
                fclose(global->ranking_costs_csv);
                global->ranking_costs_csv = NULL;
                printf("  Ranking costs CSV: %s\n", RANKING_COSTS_CSV);
            }
            printf("\n=== VPIC Multi-Timestep complete (%d timesteps x 12 phases) ===\n",
                   global->ts_count);

            /* Append nn-rl and nn-rl+exp50 steady-state averages to aggregate CSV.
             * Read from timestep CSV, skip warmup (first 5 timesteps), average the rest. */
            {
                FILE* agg = fopen(AGG_CSV, "w");
                if (agg && global->ts_csv == NULL) {
                    fprintf(agg, "source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,"
                                 "file_mib,orig_mib,ratio,"
                                 "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks,"
                                 "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                                 "comp_gbps,decomp_gbps,"
                                 "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,"
                                 "mae_ratio,mae_comp_ms,mae_decomp_ms,r2_ratio,"
                                 "comp_gbps_std,decomp_gbps_std\n");
                    /* Parse timestep CSV to compute steady-state averages per phase */
                    FILE* ts = fopen(TSTEP_CSV, "r");
                    if (ts) {
                        char line[4096];
                        fgets(line, sizeof(line), ts);  /* skip header */

                        struct PhaseAccum {
                            double sum_write, sum_read, sum_ratio;
                            double sum_mape_r, sum_mape_c, sum_mape_d;
                            int    sum_sgd, sum_expl, n_chunks_last;
                            size_t last_file_sz;
                            double sum_file_sz;
                            double sum_stats_ms, sum_nn_ms, sum_preproc_ms;
                            double sum_comp_ms, sum_decomp_ms, sum_explore_ms, sum_sgd_ms;
                            double sum_mae_r, sum_mae_c, sum_mae_d, sum_r2;
                            int    count;
                        };
                        const int N_AGG_PHASES = 12;
                        PhaseAccum pa[N_AGG_PHASES] = {};
                        const char* pnames[N_AGG_PHASES] = {
                            "no-comp", "fixed-lz4", "fixed-snappy", "fixed-deflate",
                            "fixed-gdeflate", "fixed-zstd", "fixed-ans", "fixed-cascaded",
                            "fixed-bitcomp", "nn", "nn-rl", "nn-rl+exp50"
                        };
                        const int WARMUP = 0;

                        while (fgets(line, sizeof(line), ts)) {
                            char ph[64]; int ts_idx;
                            double wr, rd, rat, mr, mc, md, wmbps, rmbps;
                            int sgd, expl, nch;
                            unsigned long long mm, fbytes = 0;
                            double t_stats = 0, t_nn = 0, t_pre = 0, t_comp = 0, t_dec = 0, t_expl = 0, t_sgd = 0;
                            double t_mae_r = 0, t_mae_c = 0, t_mae_d = 0, t_r2 = 0;
                            int nf = sscanf(line, "%63[^,],%d,%*[^,],%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d,%llu,%lf,%lf,"
                                       "%llu,"
                                       "%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                       "%lf,%lf,%lf,%lf",
                                       ph, &ts_idx, &wr, &rd, &rat, &mr, &mc, &md,
                                       &sgd, &expl, &nch, &mm, &wmbps, &rmbps, &fbytes,
                                       &t_stats, &t_nn, &t_pre, &t_comp, &t_dec, &t_expl, &t_sgd,
                                       &t_mae_r, &t_mae_c, &t_mae_d, &t_r2);
                            if (nf >= 12 && ts_idx >= WARMUP) {
                                for (int pi = 0; pi < N_AGG_PHASES; pi++) {
                                    if (strcmp(ph, pnames[pi]) == 0) {
                                        pa[pi].sum_write += wr;
                                        pa[pi].sum_read  += rd;
                                        pa[pi].sum_ratio += rat;
                                        pa[pi].sum_mape_r += mr;
                                        pa[pi].sum_mape_c += mc;
                                        pa[pi].sum_mape_d += md;
                                        pa[pi].sum_sgd  += sgd;
                                        pa[pi].sum_expl += expl;
                                        pa[pi].n_chunks_last = nch;
                                        if (nf >= 15 && fbytes > 0)
                                            pa[pi].sum_file_sz += (double)fbytes;
                                        /* GPU timing (columns 19-25, nf >= 22) */
                                        if (nf >= 22) {
                                            pa[pi].sum_stats_ms   += t_stats;
                                            pa[pi].sum_nn_ms      += t_nn;
                                            pa[pi].sum_preproc_ms += t_pre;
                                            pa[pi].sum_comp_ms    += t_comp;
                                            pa[pi].sum_decomp_ms  += t_dec;
                                            pa[pi].sum_explore_ms += t_expl;
                                            pa[pi].sum_sgd_ms     += t_sgd;
                                        }
                                        /* MAE/R² (columns 26-29, nf >= 26) */
                                        if (nf >= 26) {
                                            pa[pi].sum_mae_r += t_mae_r;
                                            pa[pi].sum_mae_c += t_mae_c;
                                            pa[pi].sum_mae_d += t_mae_d;
                                            pa[pi].sum_r2    += t_r2;
                                        }
                                        pa[pi].count++;
                                        break;
                                    }
                                }
                            }
                        }
                        fclose(ts);

                        /* Write averaged results for all phases */
                        for (int pi = 0; pi < N_AGG_PHASES; pi++) {
                            if (pa[pi].count <= 0) continue;
                            int n = pa[pi].count;
                            double avg_wr = pa[pi].sum_write / n;
                            double avg_rd = pa[pi].sum_read / n;
                            double avg_rat = pa[pi].sum_ratio / n;
                            /* Use actual file sizes when available (avoids Jensen's
                               inequality bias from orig/mean(ratio) != mean(orig/ratio)).
                               Fall back to ratio-derived estimate for old CSV files
                               that lack the file_bytes column. */
                            double avg_file_mib = (pa[pi].sum_file_sz > 0)
                                ? pa[pi].sum_file_sz / n / (double)(1 << 20)
                                : orig_mib / avg_rat;
                            double avg_ratio = (avg_file_mib > 0)
                                ? orig_mib / avg_file_mib
                                : avg_rat;
                            double wmbps = orig_mib / (avg_wr / 1000.0);
                            double rmbps = orig_mib / (avg_rd / 1000.0);
                            double avg_comp_ms = pa[pi].sum_comp_ms / n;
                            double avg_decomp_ms = pa[pi].sum_decomp_ms / n;
                            double orig_bytes = orig_mib * (double)(1 << 20);
                            double cgbps = (avg_comp_ms > 0)
                                ? orig_bytes / 1e9 / (avg_comp_ms / 1000.0) : 0.0;
                            double dgbps = (avg_decomp_ms > 0)
                                ? orig_bytes / 1e9 / (avg_decomp_ms / 1000.0) : 0.0;
                            fprintf(agg, "vpic,%s,%d,%.2f,0.00,%.2f,0.00,"
                                         "%.2f,%.2f,%.4f,"
                                         "%.1f,%.1f,0,%d,%d,%d,"
                                         "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                                         "%.4f,%.4f,"
                                         "%.2f,%.2f,%.2f,"
                                         "%.4f,%.4f,%.4f,%.4f,"
                                         "0.0000,0.0000\n",
                                    pnames[pi], n, avg_wr, avg_rd,
                                    avg_file_mib, orig_mib, avg_ratio,
                                    wmbps, rmbps,
                                    pa[pi].sum_sgd, pa[pi].sum_expl, pa[pi].n_chunks_last,
                                    pa[pi].sum_nn_ms / n, pa[pi].sum_stats_ms / n, pa[pi].sum_preproc_ms / n,
                                    avg_comp_ms, avg_decomp_ms,
                                    pa[pi].sum_explore_ms / n, pa[pi].sum_sgd_ms / n,
                                    cgbps, dgbps,
                                    fmin(200.0, pa[pi].sum_mape_r / n), fmin(200.0, pa[pi].sum_mape_c / n), fmin(200.0, pa[pi].sum_mape_d / n),
                                    pa[pi].sum_mae_r / n, pa[pi].sum_mae_c / n, pa[pi].sum_mae_d / n,
                                    pa[pi].sum_r2 / n);
                        }
                    }
                }
                if (agg) fclose(agg);
            }

            global->benchmark_done = 1;
            cudaFree(global->d_read);
            free(global->h_orig);
            free(global->h_read);
            for (int pi = 0; pi < 3; pi++) {
                if (global->nn_weights[pi]) { free(global->nn_weights[pi]); global->nn_weights[pi] = NULL; }
            }
            global->d_read = NULL;
            global->h_orig = NULL;
            global->h_read = NULL;
            gpucompress_vpic_destroy(global->vpic_fields_h);
            H5Pclose(global->vol_fapl);
            H5VLclose(global->vol_id);
            remove(TMP_NOCOMP);
            remove(TMP_FIX_LZ4);
            remove(TMP_FIX_GDEFL);
            remove(TMP_FIX_ZSTD);
            remove(TMP_FIX_SNAPPY);
            remove(TMP_FIX_DEFL);
            remove(TMP_FIX_ANS);
            remove(TMP_FIX_CASC);
            remove(TMP_FIX_BITCOMP);
            remove(TMP_NN);
            remove(TMP_NN_RL);
            remove(TMP_NN_RLEXP);
            gpucompress_cleanup();
            return;
        }

        // Open CSV on first timestep
        if (global->ts_count == 0) {
            printf("\n══════════════════════════════════════════════════════════════\n");
            printf("  Multi-timestep mode: %d timesteps, nn-rl (SGD active)\n",
                   global->timesteps);
            printf("  Each step = 1 VPIC physics step → H5Dwrite → collect MAPE\n");
            printf("══════════════════════════════════════════════════════════════\n\n");

            global->ts_csv = fopen(TSTEP_CSV, "w");
            if (global->ts_csv) {
                fprintf(global->ts_csv, "phase,timestep,sim_step,write_ms,read_ms,ratio,"
                        "mape_ratio,mape_comp,mape_decomp,"
                        "sgd_fires,explorations,n_chunks,mismatches,"
                        "write_mbps,read_mbps,"
                        "file_bytes,"
                        "stats_ms,nn_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                        "mae_ratio,mae_comp_ms,mae_decomp_ms,r2_ratio\n");
            }
            global->tc_csv = fopen(TSTEP_CHUNKS_CSV, "w");
            if (global->tc_csv) {
                fprintf(global->tc_csv, "phase,timestep,chunk,action,action_orig,"
                        "predicted_ratio,actual_ratio,"
                        "predicted_comp_ms,actual_comp_ms_raw,"
                        "predicted_decomp_ms,actual_decomp_ms_raw,"
                        "mape_ratio,mape_comp,mape_decomp,"
                        "sgd_fired,exploration_triggered,"
                        "cost_model_error_pct,actual_cost,predicted_cost,"
                        "explore_n_alt");
                for (int ei = 0; ei < 31; ei++)
                    fprintf(global->tc_csv, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d",
                            ei, ei, ei, ei);
                fprintf(global->tc_csv, ",feat_entropy,feat_mad,feat_deriv\n");
            }
            global->ranking_csv = fopen(RANKING_CSV, "w");
            if (global->ranking_csv)
                vpic_write_ranking_csv_header(global->ranking_csv);
            global->ranking_costs_csv = fopen(RANKING_COSTS_CSV, "w");
            if (global->ranking_costs_csv)
                vpic_write_ranking_costs_csv_header(global->ranking_costs_csv);

            /* Reload NN and allocate per-phase weight snapshots so that
             * nn-rl and nn-rl+exp50 start from identical pretrained weights
             * and evolve independently (no SGD leakage between phases). */
            const char* wpath = getenv("GPUCOMPRESS_WEIGHTS");
            if (wpath) gpucompress_reload_nn(wpath);

            if (!global->nn_weights_init) {
                global->nn_weights_size = gpucompress_nn_weights_size();
                for (int pi = 0; pi < 3; pi++) {
                    global->nn_weights[pi] = malloc(global->nn_weights_size);
                    if (global->nn_weights[pi])
                        gpucompress_nn_save_snapshot(global->nn_weights[pi]);
                }
                global->nn_weights_init = 1;
            }
        }

        int t = global->ts_count;

        /* Phase configs for multi-timestep loop.
         * All phases run on every timestep for fair comparison.
         * algo=0 means ALGO_AUTO (NN-based), algo>0 means fixed algorithm.
         * nn_weight_idx: 0=nn, 1=nn-rl, 2=nn-rl+exp50, -1=no NN weights */
        struct TsPhase {
            const char *name;
            const char *tmp_file;
            int sgd;
            int explore;
            int nn_weight_idx;          /* -1 for non-NN phases */
            gpucompress_algorithm_t algo; /* 0=AUTO, >0=fixed */
            unsigned int preproc;
        };
        TsPhase phases[] = {
            { "no-comp",           TMP_NOCOMP,     0, 0, -1, (gpucompress_algorithm_t)0, 0 },
            { "fixed-lz4",         TMP_FIX_LZ4,    0, 0, -1, GPUCOMPRESS_ALGO_LZ4,      0 },
            { "fixed-snappy",      TMP_FIX_SNAPPY,  0, 0, -1, GPUCOMPRESS_ALGO_SNAPPY,   0 },
            { "fixed-deflate",     TMP_FIX_DEFL,    0, 0, -1, GPUCOMPRESS_ALGO_DEFLATE,  0 },
            { "fixed-gdeflate",    TMP_FIX_GDEFL,   0, 0, -1, GPUCOMPRESS_ALGO_GDEFLATE, 0 },
            { "fixed-zstd",        TMP_FIX_ZSTD,    0, 0, -1, GPUCOMPRESS_ALGO_ZSTD,     0 },
            { "fixed-ans",         TMP_FIX_ANS,     0, 0, -1, GPUCOMPRESS_ALGO_ANS,      0 },
            { "fixed-cascaded",    TMP_FIX_CASC,    0, 0, -1, GPUCOMPRESS_ALGO_CASCADED,  0 },
            { "fixed-bitcomp",     TMP_FIX_BITCOMP, 0, 0, -1, GPUCOMPRESS_ALGO_BITCOMP,   0 },
            { "nn",                TMP_NN,          0, 0,  0, (gpucompress_algorithm_t)0, 0 },
            { "nn-rl",             TMP_NN_RL,       1, 0,  1, (gpucompress_algorithm_t)0, 0 },
            { "nn-rl+exp50",       TMP_NN_RLEXP,    1, 1,  2, (gpucompress_algorithm_t)0, 0 },
        };
        int n_phases_ts = 12;

        hsize_t dims[1] = { (hsize_t)n_floats };

        /* Progress to stderr (visible even when stdout is redirected to log) */
        if (t % 5 == 0 || t == global->timesteps - 1)
            fprintf(stderr, "\r  [VPIC] Timestep %d/%d (%d%%)  ",
                    t, global->timesteps, t * 100 / global->timesteps);

        for (int pi = 0; pi < n_phases_ts; pi++) {
            const char* phase_name = phases[pi].name;

            /* Skip excluded phases */
            if (is_phase_excluded(phase_name, global->exclude_list)) {
                if (t == 0)
                    printf("  [%s] SKIPPED (excluded by VPIC_EXCLUDE)\n", phase_name);
                continue;
            }

            int do_sgd  = phases[pi].sgd;
            int do_expl = phases[pi].explore;
            int wt_idx  = phases[pi].nn_weight_idx;
            bool is_nn  = (phases[pi].algo == GPUCOMPRESS_ALGO_AUTO && strcmp(phase_name, "no-comp") != 0);

            /* Flush nvCOMP manager cache so each phase starts cold (no cache bias) */
            gpucompress_flush_manager_cache();

            /* Restore this phase's own weight snapshot so phases evolve independently */
            if (is_nn && wt_idx >= 0 && global->nn_weights[wt_idx])
                gpucompress_nn_restore_snapshot(global->nn_weights[wt_idx]);

            if (do_sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, global->reinforce_lr,
                                             global->reinforce_mape, global->reinforce_mape);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(do_expl);
            if (do_expl) {
                gpucompress_set_exploration_threshold(0.20);
                gpucompress_set_exploration_k(4);
            }

            /* Build DCPL for this phase */
            hid_t dcpl;
            if (strcmp(phase_name, "no-comp") == 0) {
                dcpl = make_dcpl_nocomp((hsize_t)chunk_floats);
            } else if (phases[pi].algo != GPUCOMPRESS_ALGO_AUTO) {
                dcpl = make_dcpl_fixed((hsize_t)chunk_floats, phases[pi].algo, phases[pi].preproc);
            } else {
                dcpl = make_dcpl_auto((hsize_t)chunk_floats, global->diag_error_bound);
            }

            /* Print header on first timestep for each phase */
            if (t == 0) {
                printf("\n── [%s] (SGD=%s, Explore=%s) ──\n",
                       phase_name, do_sgd ? "on" : "off", do_expl ? "on" : "off");
                printf("  %-4s  %-8s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-4s\n",
                       "T", "SimStep", "WrMs", "RdMs", "Ratio",
                       "MAPE_R", "MAPE_C", "MAPE_D", "SGD");
                printf("  ----  --------  -------  -------  -------  "
                       "--------  --------  --------  ----\n");
            }

            /* Write via VOL */
            gpucompress_reset_chunk_history();
            gpucompress_set_debug_context(phase_name, t);
            remove(phases[pi].tmp_file);

            hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
            hid_t nid = H5VLget_connector_id_by_name("native");
            H5Pset_fapl_gpucompress(fapl, nid, NULL);
            H5VLclose(nid);

            hid_t file = H5Fcreate(phases[pi].tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
            H5Pclose(fapl);
            hid_t fsp  = H5Screate_simple(1, dims, NULL);
            hid_t dset = H5Dcreate2(file, "fields", H5T_NATIVE_FLOAT,
                                     fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(fsp);

            double tw0 = now_ms();
            herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                                    H5S_ALL, H5S_ALL, H5P_DEFAULT, d_fields);
            cudaDeviceSynchronize();
            H5Dclose(dset); H5Fclose(file);
            double tw1 = now_ms();
            double write_ms_t = tw1 - tw0;

            if (wret < 0) {
                printf("  %-4d  [%s] H5Dwrite failed\n", t, phase_name);
                continue;
            }

            /* Drop page cache so read-back measures real I/O, not cached data */
            drop_pagecache(phases[pi].tmp_file);

            /* Read back + verify */
            fapl = H5Pcreate(H5P_FILE_ACCESS);
            nid = H5VLget_connector_id_by_name("native");
            H5Pset_fapl_gpucompress(fapl, nid, NULL);
            H5VLclose(nid);
            file = H5Fopen(phases[pi].tmp_file, H5F_ACC_RDONLY, fapl);
            H5Pclose(fapl);
            dset = H5Dopen2(file, "fields", H5P_DEFAULT);

            double tr0 = now_ms();
            H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, global->d_read);
            cudaDeviceSynchronize();
            H5Dclose(dset); H5Fclose(file);
            double tr1 = now_ms();
            double read_ms_t = tr1 - tr0;

            unsigned long long mm = host_compare(d_fields, global->d_read,
                                                  global->h_orig, global->h_read, n_floats);

            /* File size for ratio */
            size_t file_sz = get_file_size(phases[pi].tmp_file);
            double ratio_t = (file_sz > 0) ? (double)nbytes_f / (double)file_sz : 1.0;

            /* Collect per-chunk MAPE, timing breakdown, MAE/R² */
            int n_hist = gpucompress_get_chunk_history_count();
            double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0;
            int    mcnt_r = 0, mcnt_c = 0, mcnt_d = 0;
            int    sgd_t = 0, expl_t = 0;
            double ts_stats_ms = 0, ts_nn_ms = 0, ts_preproc_ms = 0;
            double ts_comp_ms = 0, ts_decomp_ms = 0, ts_explore_ms = 0, ts_sgd_ms = 0;
            double mae_r_sum = 0, mae_c_sum = 0, mae_d_sum = 0;
            double ss_res_r = 0, sum_ratio = 0;
            int    r2_cnt = 0;
            for (int ci = 0; ci < n_hist; ci++) {
                gpucompress_chunk_diag_t diag;
                if (gpucompress_get_chunk_diag(ci, &diag) != 0) continue;
                if (diag.sgd_fired) sgd_t++;
                if (diag.exploration_triggered) expl_t++;
                /* Timing breakdown (unclamped) */
                ts_stats_ms   += diag.stats_ms;
                ts_nn_ms      += diag.nn_inference_ms;
                ts_preproc_ms += diag.preprocessing_ms;
                ts_comp_ms    += diag.compression_ms_raw;
                ts_decomp_ms  += diag.decompression_ms_raw;
                ts_explore_ms += diag.exploration_ms;
                ts_sgd_ms     += diag.sgd_update_ms;
                /* MAPE (clamped denominators) */
                if (diag.actual_ratio > 0) {
                    mape_r_sum += fabs(diag.predicted_ratio - diag.actual_ratio) / fabs(diag.actual_ratio);
                    mcnt_r++;
                }
                if (diag.compression_ms > 0) {
                    mape_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms);
                    mcnt_c++;
                }
                if (diag.decompression_ms > 0) {
                    mape_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms);
                    mcnt_d++;
                }
                /* MAE and R² accumulators */
                if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
                    mae_r_sum += fabs(diag.predicted_ratio - diag.actual_ratio);
                    ss_res_r  += (diag.predicted_ratio - diag.actual_ratio) * (diag.predicted_ratio - diag.actual_ratio);
                    sum_ratio += diag.actual_ratio;
                    r2_cnt++;
                }
                if (diag.compression_ms_raw > 0)
                    mae_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms);
                if (diag.decompression_ms > 0)
                    mae_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms);
            }
            /* Compute per-timestep MAE and R² */
            double ts_mae_r = r2_cnt ? mae_r_sum / r2_cnt : 0.0;
            double ts_mae_c = mcnt_c ? mae_c_sum / mcnt_c : 0.0;
            double ts_mae_d = mcnt_d ? mae_d_sum / mcnt_d : 0.0;
            double ts_r2 = 0.0;
            if (r2_cnt > 1) {
                double mean_r = sum_ratio / r2_cnt;
                double ss_tot = 0.0;
                for (int ci = 0; ci < n_hist; ci++) {
                    gpucompress_chunk_diag_t d2;
                    if (gpucompress_get_chunk_diag(ci, &d2) == 0
                        && d2.actual_ratio > 0 && d2.predicted_ratio > 0) {
                        double diff = d2.actual_ratio - mean_r;
                        ss_tot += diff * diff;
                    }
                }
                ts_r2 = (ss_tot > 0) ? 1.0 - ss_res_r / ss_tot : 0.0;
            }
            double real_mape_r = fmin(200.0, mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0);
            double real_mape_c = fmin(200.0, mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0);
            double real_mape_d = fmin(200.0, mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0);

            double wr_mbps = (write_ms_t > 0) ? orig_mib / (write_ms_t / 1000.0) : 0;
            double rd_mbps = (read_ms_t > 0)  ? orig_mib / (read_ms_t  / 1000.0) : 0;

            /* Print every 5th timestep */
            bool print_row = (t % 5 == 0 || t == global->timesteps - 1);
            if (print_row) {
                printf("  %-4d  %-8d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %3d\n",
                       t, (int)step(), write_ms_t, read_ms_t,
                       ratio_t, real_mape_r, real_mape_c, real_mape_d, sgd_t);
            }

            if (global->ts_csv) {
                fprintf(global->ts_csv,
                        "%s,%d,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d,%llu,%.1f,%.1f,"
                        "%zu,"
                        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                        "%.4f,%.4f,%.4f,%.4f\n",
                        phase_name, t, (int)step(), write_ms_t, read_ms_t, ratio_t,
                        real_mape_r, real_mape_c, real_mape_d,
                        sgd_t, expl_t, n_hist,
                        (unsigned long long)mm, wr_mbps, rd_mbps,
                        file_sz,
                        ts_stats_ms, ts_nn_ms, ts_preproc_ms,
                        ts_comp_ms, ts_decomp_ms, ts_explore_ms, ts_sgd_ms,
                        ts_mae_r, ts_mae_c, ts_mae_d, ts_r2);
                fflush(global->ts_csv);
            }

            /* Per-chunk milestone CSV at 0%, 25%, 50%, 75%, 100% of timesteps */
            if (global->tc_csv) {
                bool is_milestone = (t == 0 ||
                                     t == global->timesteps / 4 ||
                                     t == global->timesteps / 2 ||
                                     t == (global->timesteps * 3) / 4 ||
                                     t == global->timesteps - 1);
                if (is_milestone) {
                    for (int ci = 0; ci < n_hist; ci++) {
                        gpucompress_chunk_diag_t dd;
                        if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;

                        double mr = 0, mc = 0, md = 0;
                        if (dd.actual_ratio > 0)
                            mr = fmin(200.0, fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) * 100.0);
                        if (dd.compression_ms > 0)
                            mc = fmin(200.0, fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) * 100.0);
                        if (dd.decompression_ms > 0)
                            md = fmin(200.0, fabs(dd.predicted_decomp_time - dd.decompression_ms) / fabs(dd.decompression_ms) * 100.0);

                        char action_str[40], orig_str[40];
                        action_to_str(dd.nn_action, action_str, sizeof(action_str));
                        action_to_str(dd.nn_original_action, orig_str, sizeof(orig_str));
                        fprintf(global->tc_csv,
                                "%s,%d,%d,%s,%s,"
                                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                                "%.2f,%.2f,%.2f,%d,%d,"
                                "%.4f,%.4f,%.4f,%d",
                                phase_name, t, ci, action_str, orig_str,
                                (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                                (double)dd.predicted_decomp_time, (double)dd.decompression_ms_raw,
                                mr, mc, md,
                                dd.sgd_fired, dd.exploration_triggered,
                                (double)dd.cost_model_error_pct,
                                (double)dd.actual_cost, (double)dd.predicted_cost,
                                dd.explore_n_alternatives);
                        for (int ei = 0; ei < 31; ei++) {
                            if (ei < dd.explore_n_alternatives) {
                                char alt_str[40];
                                action_to_str(dd.explore_alternatives[ei], alt_str, sizeof(alt_str));
                                fprintf(global->tc_csv, ",%s,%.4f,%.4f,%.4f",
                                        alt_str,
                                        (double)dd.explore_ratios[ei],
                                        (double)dd.explore_comp_ms[ei],
                                        (double)dd.explore_costs[ei]);
                            } else {
                                fprintf(global->tc_csv, ",,,,");
                            }
                        }
                        fprintf(global->tc_csv, ",%.4f,%.6f,%.6f\n",
                                (double)dd.feat_entropy,
                                (double)dd.feat_mad,
                                (double)dd.feat_deriv);
                    }
                    fflush(global->tc_csv);
                }
            }

            /* Kendall τ ranking quality at milestones */
            if (is_nn && global->ranking_csv && vpic_is_ranking_milestone(t, global->timesteps)) {
                float bw = gpucompress_get_bandwidth_bytes_per_ms();
                RankingMilestoneResult tau_result = {};
                vpic_run_ranking_profiler(
                    d_fields, n_floats * sizeof(float), global->chunk_bytes,
                    global->diag_error_bound,
                    global->rank_w0, global->rank_w1, global->rank_w2, bw,
                    3, global->ranking_csv, global->ranking_costs_csv,
                    phase_name, t, &tau_result);
                printf("    [τ] T=%d: τ=%.3f  top1=%.0f%%  regret=%.3fx  (%.0fms)\n",
                       t, tau_result.mean_tau,
                       tau_result.top1_accuracy * 100.0,
                       tau_result.mean_regret,
                       tau_result.profiling_ms);
            }

            /* Save this phase's updated weights for next timestep */
            if (is_nn && wt_idx >= 0 && global->nn_weights[wt_idx])
                gpucompress_nn_save_snapshot(global->nn_weights[wt_idx]);

            H5Pclose(dcpl);
            remove(phases[pi].tmp_file);
        } /* end phase loop */

        global->ts_count++;
        return;
    }

    // ============================================================
    // All phases run in the multi-timestep loop above.
    // The aggregate CSV is generated from the timestep CSV at completion.
    // ============================================================

    // Summary and aggregate CSV are generated from the timestep CSV
    // at the end of the multi-timestep loop (see PhaseAccum above).
    global->single_shot_done = 1;

    // If no multi-timestep requested, we're done — cleanup now
    if (global->timesteps <= 0) {
        global->benchmark_done = 1;
        cudaFree(global->d_read);
        free(global->h_orig);
        free(global->h_read);
        global->d_read = NULL;
        global->h_orig = NULL;
        global->h_read = NULL;
        gpucompress_vpic_destroy(global->vpic_fields_h);
        H5Pclose(global->vol_fapl);
        H5VLclose(global->vol_id);
        remove(TMP_NOCOMP);
        remove(TMP_FIX_LZ4);
        remove(TMP_FIX_SNAPPY);
        remove(TMP_FIX_DEFL);
        remove(TMP_FIX_GDEFL);
        remove(TMP_FIX_ZSTD);
        remove(TMP_FIX_ANS);
        remove(TMP_FIX_CASC);
        remove(TMP_FIX_BITCOMP);
        remove(TMP_NN);
        remove(TMP_NN_RL);
        remove(TMP_NN_RLEXP);
        gpucompress_cleanup();
    }
};

begin_particle_injection {
    // No injection
}

begin_current_injection {
    // No injection
}

begin_field_injection {
    // No injection
}

begin_particle_collisions {
    // No collisions
}