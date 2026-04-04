/**
 * @file vpic_benchmark_deck.cxx
 * @brief VPIC Benchmark Deck: No-Comp vs NN+SGD
 *
 * A real VPIC-Kokkos Harris sheet input deck that benchmarks GPU-resident
 * field data compression through the GPUCompress VOL connector.
 * Matches the benchmark_grayscott_vol structure.
 *
 * Pipeline per timestep:
 *   VPIC simulation (GPU) → field_array->k_f_d → H5Dwrite(d_ptr) → VOL →
 *   GPU compress → HDF5 file → H5Dread(d_ptr) → GPU decompress → bitwise verify
 *
 * All phases run on the same field data each timestep:
 *   - no-comp       : baseline (no compression)
 *   - fixed-*       : fixed algorithm baselines (lz4, zstd, etc.)
 *   - nn            : ALGO_AUTO, inference-only (frozen weights)
 *   - nn-rl         : ALGO_AUTO + online SGD (learning curve over N writes)
 *   - nn-rl+exp50   : ALGO_AUTO + online SGD + exploration
 *
 * Per-NN-phase GPU weight isolation ensures SGD updates in nn-rl don't
 * leak into nn or nn-rl+exp50. Each NN phase maintains its own weight
 * snapshot on the GPU, swapped via device-to-device memcpy (~75 KB).
 *
 * BUILD:
 *   bash benchmarks/vpic-kokkos/build_vpic_pm.sh
 *
 * RUN:
 *   See benchmarks/vpic-kokkos/runBEnchCombo.sh for full usage.
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
    double mean_tau, std_tau, mean_regret, profiling_ms;
};
extern "C" int vpic_run_ranking_profiler(
    const void* d_data, size_t total_bytes, size_t chunk_bytes,
    double error_bound, float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats, FILE* csv, FILE* costs_csv, const char* phase_name,
    int timestep, RankingMilestoneResult* out);
extern "C" int vpic_is_ranking_milestone(int t, int total);
extern "C" void vpic_write_ranking_csv_header(FILE* csv);
extern "C" void vpic_write_ranking_costs_csv_header(FILE* csv);
extern "C" double vpic_compute_psnr_gpu(
    const float* d_original, const float* d_decompressed, size_t n_floats);
extern "C" double vpic_compute_quality_gpu(
    const float* d_original, const float* d_decompressed, size_t n_floats,
    double* out_rmse, double* out_max_err, double* out_mean_err, double* out_ssim);
extern "C" float gpucompress_get_bandwidth_bytes_per_ms(void);

// ============================================================
// Constants
// ============================================================
#define REINFORCE_LR        0.2f
#define REINFORCE_MAPE      0.10f

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Temporary HDF5 files written to /tmp (typically tmpfs / RAM-backed).
 * This isolates GPU compression pipeline overhead from disk I/O variability.
 * Note: drop_pagecache() is a no-op on tmpfs, so reads may hit warm cache.
 * Paths are rank-suffixed to avoid collisions when multiple MPI ranks share
 * a node (e.g., 1 node × 2 GPUs). Populated by init_tmp_paths(). */
static char TMP_NOCOMP[256];
static char TMP_FIX_LZ4[256];
static char TMP_FIX_SNAPPY[256];
static char TMP_FIX_DEFL[256];
static char TMP_FIX_GDEFL[256];
static char TMP_FIX_ZSTD[256];
static char TMP_FIX_ANS[256];
static char TMP_FIX_CASC[256];
static char TMP_FIX_BITCOMP[256];
static char TMP_NN[256];
static char TMP_NN_RL[256];
static char TMP_NN_RLEXP[256];
static char TMP_WARMUP[256];

static void init_tmp_paths(int mpi_rank) {
    snprintf(TMP_NOCOMP,      sizeof(TMP_NOCOMP),      "/tmp/bm_vpic_nocomp_rank%d.h5",      mpi_rank);
    snprintf(TMP_FIX_LZ4,     sizeof(TMP_FIX_LZ4),     "/tmp/bm_vpic_fix_lz4_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_SNAPPY,   sizeof(TMP_FIX_SNAPPY),   "/tmp/bm_vpic_fix_snappy_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_DEFL,    sizeof(TMP_FIX_DEFL),    "/tmp/bm_vpic_fix_deflate_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_GDEFL,   sizeof(TMP_FIX_GDEFL),   "/tmp/bm_vpic_fix_gdefl_rank%d.h5",   mpi_rank);
    snprintf(TMP_FIX_ZSTD,    sizeof(TMP_FIX_ZSTD),    "/tmp/bm_vpic_fix_zstd_rank%d.h5",    mpi_rank);
    snprintf(TMP_FIX_ANS,     sizeof(TMP_FIX_ANS),     "/tmp/bm_vpic_fix_ans_rank%d.h5",     mpi_rank);
    snprintf(TMP_FIX_CASC,    sizeof(TMP_FIX_CASC),    "/tmp/bm_vpic_fix_cascaded_rank%d.h5", mpi_rank);
    snprintf(TMP_FIX_BITCOMP,  sizeof(TMP_FIX_BITCOMP),  "/tmp/bm_vpic_fix_bitcomp_rank%d.h5",  mpi_rank);
    snprintf(TMP_NN,          sizeof(TMP_NN),          "/tmp/bm_vpic_nn_rank%d.h5",          mpi_rank);
    snprintf(TMP_NN_RL,       sizeof(TMP_NN_RL),       "/tmp/bm_vpic_nn_rl_rank%d.h5",       mpi_rank);
    snprintf(TMP_NN_RLEXP,    sizeof(TMP_NN_RLEXP),    "/tmp/bm_vpic_nn_rlexp_rank%d.h5",    mpi_rank);
    snprintf(TMP_WARMUP,      sizeof(TMP_WARMUP),      "/tmp/bm_vpic_warmup_rank%d.h5",      mpi_rank);
}
/* CSV output directory: set VPIC_RESULTS_DIR env var to override.
 * The eval script sets this per-run so CSVs land in the right subdirectory. */
static char RESULTS_DIR[512];
static char TSTEP_CSV[600];
static char TSTEP_CHUNKS_CSV[600];
static char RANKING_CSV[600];
static char RANKING_COSTS_CSV[600];
static char AGG_CSV[600];

static void init_csv_paths(int mpi_rank, int mpi_size) {
    const char* env = getenv("VPIC_RESULTS_DIR");
    if (env && env[0]) {
        snprintf(RESULTS_DIR, sizeof(RESULTS_DIR), "%s", env);
    } else {
        snprintf(RESULTS_DIR, sizeof(RESULTS_DIR),
                 "%s/benchmarks/vpic-kokkos/results", GPU_DIR);
    }
    mkdir(RESULTS_DIR, 0755);
    if (mpi_size > 1) {
        snprintf(TSTEP_CSV, sizeof(TSTEP_CSV),
                 "%s/benchmark_vpic_deck_timesteps_rank%d.csv", RESULTS_DIR, mpi_rank);
        snprintf(TSTEP_CHUNKS_CSV, sizeof(TSTEP_CHUNKS_CSV),
                 "%s/benchmark_vpic_deck_timestep_chunks_rank%d.csv", RESULTS_DIR, mpi_rank);
        snprintf(RANKING_CSV, sizeof(RANKING_CSV),
                 "%s/benchmark_vpic_deck_ranking_rank%d.csv", RESULTS_DIR, mpi_rank);
        snprintf(RANKING_COSTS_CSV, sizeof(RANKING_COSTS_CSV),
                 "%s/benchmark_vpic_deck_ranking_costs_rank%d.csv", RESULTS_DIR, mpi_rank);
        snprintf(AGG_CSV, sizeof(AGG_CSV),
                 "%s/benchmark_vpic_deck_rank%d.csv", RESULTS_DIR, mpi_rank);
    } else {
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
}

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                sim_steps;         // Warmup steps before benchmarking
    int                timesteps;         // Number of benchmark write cycles
    int                ts_count;          // Current timestep counter
    size_t             chunk_bytes;       // HDF5 chunk size in bytes
    gpucompress_vpic_t vpic_fields_h;     // Adapter handle for fields
    hid_t              vol_fapl;          // File access property list with VOL
    hid_t              vol_id;            // VOL connector ID
    int                gpucompress_ready; // 1 if init succeeded
    int                benchmark_done;    // 1 after all benchmark steps complete
    double             diag_error_bound;  // Error bound for diagnostics
    FILE*              ts_csv;            // Timestep CSV file handle
    FILE*              tc_csv;            // Timestep per-chunk CSV file handle

    // Buffers for benchmark
    float*             d_read;            // GPU read-back buffer
    float*             h_orig;            // Host buffer for verification
    float*             h_read;            // Host buffer for verification

    // Configurable hyperparameters
    float              reinforce_lr;      // SGD learning rate (env VPIC_LR)
    float              reinforce_mape;    // MAPE threshold for SGD (env VPIC_MAPE_THRESHOLD)
    int                explore_k;         // Exploration K alternatives (env VPIC_EXPLORE_K)
    float              explore_thresh;    // Exploration error threshold (env VPIC_EXPLORE_THRESH)
    int                do_verify;         // Bitwise validation (env VPIC_VERIFY, default 1)
    int                verify_weights;    // Cross-timestep weight verification (env VPIC_VERIFY_WEIGHTS, default 0)

    // Ranking quality profiler
    FILE*              ranking_csv;       // Kendall tau ranking CSV
    FILE*              ranking_costs_csv; // Per-config predicted vs actual costs
    float              rank_w0, rank_w1, rank_w2;  // Cost model weights for profiler

    // Phase exclusion: VPIC_EXCLUDE="lz4,no-comp,zstd" skips matching phases
    char               exclude_list[512]; // comma-separated substrings to exclude

    // Phase-major: VPIC_PHASE="nn-rl" runs only that one phase (exact match)
    char               single_phase[64];  // empty = all phases (current behavior)

    // Field dump: VPIC_DUMP_FIELDS=1 dumps raw field binary each timestep
    int                dump_fields;       // 0 = off, 1 = dump to RESULTS_DIR/fields_t<N>.raw
    int                grid_n;            // NX (grid size without ghost cells)

    // Simulation interval: run N physics steps between each benchmark write
    // VPIC_SIM_INTERVAL=10 means 10 sim steps per write (data evolves more between snapshots)
    int                sim_interval;      // default 1 (every step)
    int                sim_steps_pending; // countdown for next benchmark write

    // Multi-policy support: VPIC_POLICIES="balanced,ratio,speed"
    // Each policy sets different cost model weights for NN algorithm selection.
    // Fixed phases run once; NN phases run once per policy with isolated weights.
    // Snapshot index: policy_idx * 3 + nn_base_idx (nn=0, nn-rl=1, nn-rl+exp50=2)
    static const int   NN_BASE_PHASES = 3;
    static const int   MAX_POLICIES = 3;  // balanced, ratio, speed
    int                n_policies;
    float              policy_w0[MAX_POLICIES];
    float              policy_w1[MAX_POLICIES];
    float              policy_w2[MAX_POLICIES];
    char               policy_labels[MAX_POLICIES][32];
    void*              nn_weight_snapshots_gpu[MAX_POLICIES * 3];
    size_t             nn_weight_bytes;
};

// ============================================================
// Helper: dump raw field binary for visualization
// ============================================================
static void dump_fields_raw(const float* d_fields, size_t n_floats,
                            int timestep, int grid_n)
{
    char path[700];
    snprintf(path, sizeof(path), "%s/fields_t%04d_NX%d.raw",
             RESULTS_DIR, timestep, grid_n);
    size_t nbytes = n_floats * sizeof(float);
    float* h_buf = (float*)malloc(nbytes);
    if (!h_buf) {
        fprintf(stderr, "[dump] malloc failed (%zu bytes)\n", nbytes);
        return;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_buf, d_fields, nbytes, cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(h_buf, sizeof(float), n_floats, f);
        fclose(f);
        fprintf(stderr, "[dump] Wrote %s (%.1f MB)\n", path, nbytes / 1e6);
    } else {
        fprintf(stderr, "[dump] Failed to open %s\n", path);
    }
    free(h_buf);
}

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
        /* Exact match: "deflate" excludes "deflate" but not "gdeflate" */
        if (strcmp(phase_name, tok) == 0) return true;
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

// Initialization: Harris sheet + GPUCompress + VOL setup
// ============================================================
begin_initialization {
    // ---- Physics (Harris sheet reconnection) ----
    double L    = 1;
    double ec   = 1;
    double me   = 1;
    double c    = 1;
    double eps0 = 1;

    // Physics params configurable via env vars for data variety tuning.
    // Lower mi_me + wpe_wce = faster reconnection → richer 2D/3D structure.
    // Higher Ti_Te = more free energy → stronger instabilities.
    //   Defaults (mi_me=5, wpe_wce=1, Ti_Te=5): fast reconnection, ~500 steps for structure
    //   Slow recon (mi_me=25, wpe_wce=3, Ti_Te=1): ~20k steps for structure
    const char* env_mime   = getenv("VPIC_MI_ME");
    const char* env_wpewce = getenv("VPIC_WPE_WCE");
    const char* env_tite   = getenv("VPIC_TI_TE");
    // Fast reconnection defaults: visible structure in ~500 steps.
    // Original slow defaults (mi_me=25, wpe_wce=3, Ti_Te=1) need ~20K steps.
    double mi_me   = env_mime   ? atof(env_mime)   : 5;
    double rhoi_L  = 1;
    double Ti_Te   = env_tite   ? atof(env_tite)   : 5;
    double wpe_wce = env_wpewce ? atof(env_wpewce) : 1;

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
    const char* env_nppc = getenv("VPIC_NPPC");
    double nppc = env_nppc ? atof(env_nppc) : 10;

    double damp      = 0.001;
    double cfl_req   = 0.70;
    double wpedt_max = 0.20;

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

    // Parse policies: VPIC_POLICIES="balanced,ratio,speed" (default: balanced)
    // Falls back to VPIC_W0/W1/W2 for backward compatibility (single policy).
    const char* env_policies = getenv("VPIC_POLICIES");
    const char* env_w0 = getenv("VPIC_W0");
    int n_policies = 0;
    float all_w0[3], all_w1[3], all_w2[3];
    char  all_labels[3][32];

    if (env_policies && env_policies[0]) {
        char buf[256];
        strncpy(buf, env_policies, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char* tok = strtok(buf, ",");
        while (tok && n_policies < 3) {
            while (*tok == ' ') tok++;
            if (strcmp(tok, "balanced") == 0) {
                all_w0[n_policies] = 1.0f; all_w1[n_policies] = 1.0f; all_w2[n_policies] = 1.0f;
                snprintf(all_labels[n_policies], 32, "balanced");
            } else if (strcmp(tok, "ratio") == 0) {
                all_w0[n_policies] = 0.0f; all_w1[n_policies] = 0.0f; all_w2[n_policies] = 1.0f;
                snprintf(all_labels[n_policies], 32, "ratio");
            } else if (strcmp(tok, "speed") == 0) {
                all_w0[n_policies] = 1.0f; all_w1[n_policies] = 1.0f; all_w2[n_policies] = 0.0f;
                snprintf(all_labels[n_policies], 32, "speed");
            } else {
                fprintf(stderr, "Unknown policy: %s (use balanced, ratio, speed)\n", tok);
                tok = strtok(NULL, ",");
                continue;
            }
            n_policies++;
            tok = strtok(NULL, ",");
        }
    }
    if (n_policies == 0) {
        // Fallback: single policy from VPIC_W0/W1/W2 or default balanced
        const char* env_w1_fb = getenv("VPIC_W1");
        const char* env_w2_fb = getenv("VPIC_W2");
        all_w0[0] = env_w0       ? (float)atof(env_w0)    : 1.0f;
        all_w1[0] = env_w1_fb    ? (float)atof(env_w1_fb)  : 1.0f;
        all_w2[0] = env_w2_fb    ? (float)atof(env_w2_fb)  : 1.0f;
        snprintf(all_labels[0], 32, "balanced");
        n_policies = 1;
    }
    // Use first policy as the default ranking weights
    float rank_w0 = all_w0[0];
    float rank_w1 = all_w1[0];
    float rank_w2 = all_w2[0];

    const char* env_lr = getenv("VPIC_LR");
    const char* env_mape = getenv("VPIC_MAPE_THRESHOLD");
    const char* env_ek = getenv("VPIC_EXPLORE_K");
    const char* env_et = getenv("VPIC_EXPLORE_THRESH");
    float reinforce_lr   = env_lr   ? (float)atof(env_lr)   : REINFORCE_LR;
    float reinforce_mape = env_mape ? (float)atof(env_mape) : REINFORCE_MAPE;
    int   explore_k      = env_ek   ? atoi(env_ek)          : 4;
    float explore_thresh = env_et   ? (float)atof(env_et)   : 0.20f;

    // Simulation interval: N physics steps between each benchmark write
    const char* env_sim_int = getenv("VPIC_SIM_INTERVAL");
    int sim_interval = env_sim_int ? atoi(env_sim_int) : 1;
    if (sim_interval < 1) sim_interval = 1;

    // Total VPIC steps: warmup + 1 + timesteps * sim_interval
    // With sim_interval=10 and timesteps=50: 500 additional sim steps between writes
    num_step        = warmup + 1 + timesteps * sim_interval;
    status_interval = 50;
    // Divergence cleaning: prevents charge/field error buildup that causes
    // unphysical particle acceleration on multi-node runs.
    clean_div_e_interval = 200;
    clean_div_b_interval = 200;
    // Single-process needs only 1 comm round (default 3 in vpic.cc).
    // Multi-rank runs need 3 rounds for particles crossing multiple boundaries.
    num_comm_round = (nproc() <= 1) ? 1 : 6;

    global->sim_steps       = warmup;
    global->timesteps       = timesteps;
    global->ts_count        = 0;
    global->sim_interval    = sim_interval;
    global->sim_steps_pending = 0;
    global->chunk_bytes     = (size_t)chunk_mb * 1024 * 1024;
    global->benchmark_done  = 0;
    const char* env_eb = getenv("VPIC_ERROR_BOUND");
    global->diag_error_bound = env_eb ? atof(env_eb) : 0.0;

    const char* env_min_psnr = getenv("VPIC_MIN_PSNR");
    if (env_min_psnr)
        gpucompress_set_min_psnr((float)atof(env_min_psnr));

    /* Phase exclusion: VPIC_EXCLUDE="lz4,no-comp,zstd" */
    const char* env_exclude = getenv("VPIC_EXCLUDE");
    if (env_exclude && env_exclude[0]) {
        strncpy(global->exclude_list, env_exclude, sizeof(global->exclude_list) - 1);
        global->exclude_list[sizeof(global->exclude_list) - 1] = '\0';
    } else {
        global->exclude_list[0] = '\0';
    }

    /* Phase-major: VPIC_PHASE="nn-rl" runs only that phase (exact match) */
    const char* env_phase = getenv("VPIC_PHASE");
    if (env_phase && env_phase[0]) {
        strncpy(global->single_phase, env_phase, sizeof(global->single_phase) - 1);
        global->single_phase[sizeof(global->single_phase) - 1] = '\0';
    } else {
        global->single_phase[0] = '\0';
    }
    global->ts_csv          = NULL;
    global->tc_csv          = NULL;
    global->reinforce_lr    = reinforce_lr;
    global->reinforce_mape  = reinforce_mape;
    global->explore_k       = explore_k;
    global->explore_thresh  = explore_thresh;
    const char* env_verify = getenv("VPIC_VERIFY");
    const char* env_verify_wt = getenv("VPIC_VERIFY_WEIGHTS");
    global->do_verify       = env_verify ? atoi(env_verify) : 1;
    global->verify_weights  = env_verify_wt ? atoi(env_verify_wt) : 0;

    const char* env_dump = getenv("VPIC_DUMP_FIELDS");
    global->dump_fields     = env_dump ? atoi(env_dump) : 0;
    global->grid_n          = grid_n;

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

    // sort_interval: disabled for benchmark (saves ~20ms on sort steps).
    // Cache degradation is negligible for short runs (<500 steps).
    // max_nm: default (-1) = 8% of max_np, too small for multi-node reconnection.
    // Use 20% for 16+ rank runs where particle jets cause heavy boundary crossing.
    int max_np_local = (int)(1.5*Ni/nproc());
    int max_nm_local = (nproc() > 4) ? max_np_local / 5 : -1;  // 20% or default
    species_t* ion      = define_species("ion",       ec, mi, max_np_local, max_nm_local, 0, 1);
    species_t* electron = define_species("electron", -ec, me, max_np_local, max_nm_local, 0, 1);

    // Load fields (Harris current sheet + perturbation + guide field)
    //
    // Bz  = b0 * tanh(x/L)           — Harris equilibrium (reverses at x=0)
    // Bx  = b0 * pert * sin(2πy/Ly)  — tearing mode seed (skips slow linear phase)
    // By  = b0 * guide               — guide field (enables 3D structure)
    //
    // VPIC_PERTURBATION: amplitude of Bx seed (default 0.1 = 10% of b0)
    //   Higher = faster instability onset, 0 = wait for particle noise
    // VPIC_GUIDE_FIELD: strength of By guide field (default 0.0 = no guide)
    //   0.2-0.5 produces 3D magnetic islands and flux ropes
    const char* env_pert  = getenv("VPIC_PERTURBATION");
    const char* env_guide = getenv("VPIC_GUIDE_FIELD");
    double pert_amp  = env_pert  ? atof(env_pert)  : 0.05;  // fraction of b0
    double guide_fld = env_guide ? atof(env_guide) : 0.0;   // fraction of b0

    set_region_field(everywhere,
                     0, 0, 0,
                     b0*pert_amp*sin(2*M_PI*y/Ly),
                     b0*guide_fld,
                     b0*tanh(x/L));

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

    // ---- MPI rank-aware path initialization ----
    // Rank-suffix all /tmp HDF5 paths to avoid collisions when multiple
    // MPI ranks share a node (e.g., 1 node × 2 GPUs).
    init_tmp_paths(rank());

    // ---- GPUCompress + HDF5 VOL initialization ----
    global->gpucompress_ready = 0;
    global->d_read    = NULL;
    global->h_orig    = NULL;
    global->h_read    = NULL;

    const char* weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    fprintf(stderr, "[rank %d] gpucompress_init(weights=%s) ...\n", rank(), weights_path ? weights_path : "(null)");
    gpucompress_error_t gerr = gpucompress_init(weights_path);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "[rank %d] FATAL: gpucompress_init failed (err=%d)\n", rank(), (int)gerr);
        sim_log("FATAL: gpucompress_init failed (" << gerr << ")");
        return;
    }
    fprintf(stderr, "[rank %d] gpucompress_init OK\n", rank());

    if (weights_path && !gpucompress_nn_is_loaded()) {
        fprintf(stderr, "[rank %d] WARNING: NN weights not loaded from %s\n", rank(), weights_path);
        sim_log("WARNING: NN weights not loaded from " << weights_path);
    }

    H5Z_gpucompress_register();
    global->vol_id = H5VL_gpucompress_register();
    fprintf(stderr, "[rank %d] HDF5 VOL registered (vol_id=%ld)\n", rank(), (long)global->vol_id);

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
    global->n_policies = n_policies;
    for (int pi = 0; pi < n_policies; pi++) {
        global->policy_w0[pi] = all_w0[pi];
        global->policy_w1[pi] = all_w1[pi];
        global->policy_w2[pi] = all_w2[pi];
        strncpy(global->policy_labels[pi], all_labels[pi], 31);
    }
    global->gpucompress_ready = 1;

    // Create VPIC adapter handle for fields
    VpicSettings fs = vpic_default_settings();
    fs.data_type = VPIC_DATA_FIELDS;
    fs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_fields_h, &fs);

    size_t field_bytes = (size_t)grid->nv * 16 * sizeof(float);
    sim_log("=== VPIC Benchmark Deck: Harris Sheet Reconnection ===");
    sim_log("  MPI ranks: " << nproc() << " (rank " << rank() << ")");
    sim_log("  Grid     : " << (int)nx << "x" << (int)ny << "x" << (int)nz
            << " = " << grid->nv << " voxels");
    sim_log("  Fields   : " << field_bytes / (1024*1024) << " MB (16 vars x "
            << grid->nv << " cells)");
    sim_log("  Chunks   : " << global->chunk_bytes / (1024*1024) << " MB each");
    sim_log("  Particles: " << nppc << " per cell");
    sim_log("  Physics  : mi_me=" << mi_me << " wpe_wce=" << wpe_wce << " Ti_Te=" << Ti_Te
            << " pert=" << pert_amp << " guide=" << guide_fld);
    sim_log("  Warmup   : " << global->sim_steps << " steps");
    if (sim_interval > 1)
        sim_log("  Sim interval: " << sim_interval << " steps between writes");
    if (timesteps > 0)
        sim_log("  Timesteps: " << timesteps << " (all phases per timestep, GPU weight isolation)");
    sim_log("  Env vars : VPIC_NX=" << grid_n << " VPIC_CHUNK_MB=" << chunk_mb
            << " VPIC_WARMUP_STEPS=" << warmup
            << " VPIC_TIMESTEPS=" << timesteps);
    sim_log("  Weights  : " << (weights_path ? weights_path : "(none, fallback to LZ4)"));
    sim_log("  NN loaded: " << (gpucompress_nn_is_loaded() ? "yes" : "no"));
    sim_log("  Policies : " << n_policies);
    for (int pi = 0; pi < n_policies; pi++)
        sim_log("    [" << pi << "] " << all_labels[pi]
                << " (w0=" << all_w0[pi] << " w1=" << all_w1[pi] << " w2=" << all_w2[pi] << ")");
    sim_log("  SGD LR: " << reinforce_lr << "  MAPE threshold: " << reinforce_mape);
    if (global->diag_error_bound > 0.0)
        sim_log("  Error bound: " << global->diag_error_bound << " (lossy quantization enabled)");
    if (global->exclude_list[0])
        sim_log("  Exclude: " << global->exclude_list);
    if (global->single_phase[0])
        sim_log("  Phase-major: " << global->single_phase << " only");
    if (global->dump_fields)
        sim_log("  Field dump: ON → " << RESULTS_DIR << "/fields_t*.raw");

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
    if (!global->gpucompress_ready) {
        static int warn_count = 0;
        if (warn_count < 3) {
            fprintf(stderr, "[rank %d] begin_diagnostics: gpucompress_ready=0, skipping (step=%d)\n",
                    rank(), (int)step());
            warn_count++;
        }
        return;
    }

    /* Weight fingerprints for cross-timestep verification (VPIC_VERIFY_WEIGHTS=1) */
    static float wt_fingerprints[3 * 3][3]; /* 3 policies × 3 NN base phases */

    /* ── Warmup progress ── */
    static double warmup_start_ms = 0;
    if (step() < global->sim_steps) {
        if (step() == 0) warmup_start_ms = now_ms();
        if (step() % 50 == 0 || step() == global->sim_steps - 1) {
            double elapsed_s = (now_ms() - warmup_start_ms) / 1000.0;
            int pct = (global->sim_steps > 0) ? (int)step() * 100 / global->sim_steps : 0;
            int bar_w = 20;
            int filled = (int)step() * bar_w / (global->sim_steps > 0 ? global->sim_steps : 1);
            fprintf(stderr, "\r  Warmup %d/%d [", (int)step(), global->sim_steps);
            for (int b = 0; b < bar_w; b++)
                fputc(b < filled ? '#' : '.', stderr);
            fprintf(stderr, "] %3d%%  (%.1fs)", pct, elapsed_s);
            fflush(stderr);
        }
        return;
    }

    /* Initialize CSV paths on first invocation */
    static bool csv_paths_init = false;
    if (!csv_paths_init) {
        double warmup_elapsed_s = (now_ms() - warmup_start_ms) / 1000.0;
        fprintf(stderr, "\r  Warmup %d/%d [####################] 100%%  done (%.1fs)               \n",
                global->sim_steps, global->sim_steps, warmup_elapsed_s);
        init_csv_paths(rank(), nproc());
        csv_paths_init = true;
    }

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
    if (step() >= global->sim_steps) {  /* physics warmup complete → run benchmark */
        // Skip intermediate steps when sim_interval > 1
        // (run N physics steps between each benchmark write for data variety)
        static double evolve_start_ms = 0;
        if (global->sim_interval > 1 && global->sim_steps_pending > 0) {
            int done = global->sim_interval - global->sim_steps_pending;
            if (done == 1) evolve_start_ms = now_ms();  // start timer on first skip step
            if (done % 50 == 0 || global->sim_steps_pending == 1) {
                double elapsed_s = (now_ms() - evolve_start_ms) / 1000.0;
                int pct = done * 100 / global->sim_interval;
                int bar_w = 20;
                int filled = done * bar_w / global->sim_interval;
                fprintf(stderr, "\r  Sim evolve %d→%d [", global->ts_count, global->ts_count + 1);
                for (int b = 0; b < bar_w; b++)
                    fputc(b < filled ? '#' : '.', stderr);
                fprintf(stderr, "] %3d%%  (%d/%d steps, %.1fs)          ", pct, done, global->sim_interval, elapsed_s);
                fflush(stderr);
            }
            global->sim_steps_pending--;
            return;  // let VPIC advance without benchmarking
        }

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
            fprintf(stderr, "\r  T=%d/%d [##############################] 100%%  Done.                    \n",
                    global->ts_count, global->ts_count);
            printf("\n=== VPIC Multi-Timestep complete (%d timesteps) ===\n",
                   global->ts_count);

            /* Append nn-rl and nn-rl+exp50 steady-state averages to aggregate CSV.
             * Read from timestep CSV and compute per-phase averages across all timesteps. */
            {
                FILE* agg = fopen(AGG_CSV, "w");
                if (agg && global->ts_csv == NULL) {
                    fprintf(agg, "rank,source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,"
                                 "file_mib,orig_mib,ratio,"
                                 "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks,"
                                 "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                                 "comp_gbps,decomp_gbps,"
                                 "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,"
                                 "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
                                 "r2_ratio,r2_comp,r2_decomp,r2_psnr,"
                                 "comp_gbps_std,decomp_gbps_std,"
                                 "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
                                 "vol_s2_busy_ms,vol_s3_busy_ms,"
                                 "psnr_db,rmse,max_abs_err,mean_abs_err,ssim,bit_rate\n");
                    /* Parse timestep CSV to compute steady-state averages per phase */
                    FILE* ts = fopen(TSTEP_CSV, "r");
                    if (ts) {
                        char line[4096];
                        fgets(line, sizeof(line), ts);  /* skip header */

                        struct PhaseAccum {
                            double sum_write, sum_read;
                            double sum_write_sq, sum_read_sq;  /* for std deviation */
                            double sum_mape_r, sum_mape_c, sum_mape_d;
                            int    sum_sgd, sum_expl, n_chunks_last;
                            size_t last_file_sz;
                            double sum_file_sz;
                            double sum_stats_ms, sum_nn_ms, sum_preproc_ms;
                            double sum_comp_ms, sum_decomp_ms, sum_explore_ms, sum_sgd_ms;
                            double sum_comp_ms_sq, sum_decomp_ms_sq;  /* for gbps std */
                            double sum_mae_r, sum_mae_c, sum_mae_d, sum_mae_p;
                            double sum_vs1, sum_vs2, sum_vs3;
                            double sum_vs2_busy, sum_vs3_busy;
                            double sum_mape_p;
                            double sum_r2_ratio, sum_r2_comp, sum_r2_decomp, sum_r2_psnr;
                            double sum_mse, sum_range_sq, max_maxerr;
                            double sum_meanerr, sum_ssim, sum_bitrate;
                            int    psnr_count;
                            int    count;
                        };
                        const int N_AGG_PHASES = 12;
                        PhaseAccum pa[N_AGG_PHASES] = {};
                        const char* pnames[N_AGG_PHASES] = {
                            "no-comp", "lz4", "snappy", "deflate",
                            "gdeflate", "zstd", "ans", "cascaded",
                            "bitcomp", "nn", "nn-rl", "nn-rl+exp50"
                        };
                        const int WARMUP = 0;

                        while (fgets(line, sizeof(line), ts)) {
                            char ph[64]; int ts_idx;
                            double wr, rd, rat, mr, mc, md, wmbps, rmbps;
                            int sgd, expl, nch;
                            unsigned long long mm, fbytes = 0;
                            double t_stats = 0, t_nn = 0, t_pre = 0, t_comp = 0, t_dec = 0, t_expl = 0, t_sgd = 0;
                            double t_mae_r = 0, t_mae_c = 0, t_mae_d = 0, t_mae_p = 0;
                            double t_vs1 = 0, t_drain = 0, t_io_drain = 0;
                            double t_vs2_busy = 0, t_vs3_busy = 0;
                            double t_mape_p = 0;
                            double t_r2_ratio = 0, t_r2_comp = 0, t_r2_decomp = 0, t_r2_psnr = 0;
                            /* Skip leading rank column (%*[^,],), then parse phase onward */
                            int nf = sscanf(line, "%*[^,],%63[^,],%d,%*[^,],%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d,%llu,%lf,%lf,"
                                       "%llu,%*[^,],"
                                       "%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                       "%lf,%lf,%lf,%lf,"
                                       "%lf,%lf,%lf,"
                                       "%lf,%lf",
                                       ph, &ts_idx, &wr, &rd, &rat, &mr, &mc, &md,
                                       &sgd, &expl, &nch, &mm, &wmbps, &rmbps, &fbytes,
                                       &t_stats, &t_nn, &t_pre, &t_comp, &t_dec, &t_expl, &t_sgd,
                                       &t_mae_r, &t_mae_c, &t_mae_d, &t_mae_p,
                                       &t_vs1, &t_drain, &t_io_drain,
                                       &t_vs2_busy, &t_vs3_busy);
                            /* Parse trailing fields by comma-counting.
                             * Column layout: ...,psnr_db(40),psnr_predicted_db(41),rmse(42),
                             * max_abs_err(43),mean_abs_err(44),ssim(45),bit_rate(46),
                             * mape_psnr(47),r2_ratio(48),r2_comp(49),r2_decomp(50),r2_psnr(51) */
                            double t_psnr = 0, t_rmse = 0, t_maxerr = 0;
                            double t_meanerr = 0, t_ssim = 0, t_bitrate = 0;
                            {
                                const char* p = line;
                                int commas = 0;
                                while (*p && commas < 40) { if (*p == ',') commas++; p++; }
                                if (commas >= 40) t_psnr = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                /* skip psnr_predicted_db */
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_rmse = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_maxerr = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_meanerr = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_ssim = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_bitrate = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_mape_p = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_r2_ratio = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_r2_comp = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_r2_decomp = atof(p);
                                while (*p && *p != ',') p++; if (*p == ',') p++;
                                t_r2_psnr = atof(p);
                            }
                            if (nf >= 12 && ts_idx >= WARMUP) {
                                for (int pi = 0; pi < N_AGG_PHASES; pi++) {
                                    /* Match "nn-rl" against both "nn-rl" and "nn-rl/balanced" etc. */
                                    size_t plen = strlen(pnames[pi]);
                                    if (strncmp(ph, pnames[pi], plen) == 0
                                        && (ph[plen] == '\0' || ph[plen] == '/')) {
                                        pa[pi].sum_write += wr;
                                        pa[pi].sum_read  += rd;
                                        pa[pi].sum_write_sq += wr * wr;
                                        pa[pi].sum_read_sq  += rd * rd;
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
                                            pa[pi].sum_comp_ms_sq += t_comp * t_comp;
                                            pa[pi].sum_decomp_ms  += t_dec;
                                            pa[pi].sum_decomp_ms_sq += t_dec * t_dec;
                                            pa[pi].sum_explore_ms += t_expl;
                                            pa[pi].sum_sgd_ms     += t_sgd;
                                        }
                                        /* MAE (columns 24-27, nf >= 26) */
                                        if (nf >= 26) {
                                            pa[pi].sum_mae_r += t_mae_r;
                                            pa[pi].sum_mae_c += t_mae_c;
                                            pa[pi].sum_mae_d += t_mae_d;
                                            pa[pi].sum_mae_p += t_mae_p;
                                            pa[pi].sum_mape_p += t_mape_p;
                                        }
                                        /* VOL additive timing (columns 26-28, nf >= 28) */
                                        if (nf >= 28) {
                                            pa[pi].sum_vs1 += t_vs1;
                                            pa[pi].sum_vs2 += t_drain;
                                            pa[pi].sum_vs3 += t_io_drain;
                                        }
                                        /* VOL busy times (columns 29-30, nf >= 30) */
                                        if (nf >= 30) {
                                            pa[pi].sum_vs2_busy += t_vs2_busy;
                                            pa[pi].sum_vs3_busy += t_vs3_busy;
                                        }
                                        /* Quality metrics: accumulate MSE and range²
                                         * for correct averaging (not dB averaging). */
                                        pa[pi].sum_mse      += t_rmse * t_rmse;
                                        if (std::isfinite(t_psnr) && t_psnr < 300.0 && t_rmse > 0.0) {
                                            pa[pi].sum_range_sq += t_rmse * t_rmse * pow(10.0, t_psnr / 10.0);
                                            pa[pi].psnr_count++;
                                        }
                                        pa[pi].max_maxerr    = (t_maxerr > pa[pi].max_maxerr) ? t_maxerr : pa[pi].max_maxerr;
                                        pa[pi].sum_meanerr  += t_meanerr;
                                        pa[pi].sum_ssim     += t_ssim;
                                        pa[pi].sum_bitrate  += t_bitrate;
                                        /* R² (parsed via comma-counting) */
                                        pa[pi].sum_r2_ratio  += t_r2_ratio;
                                        pa[pi].sum_r2_comp   += t_r2_comp;
                                        pa[pi].sum_r2_decomp += t_r2_decomp;
                                        pa[pi].sum_r2_psnr   += t_r2_psnr;
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
                            /* Sample std: sqrt((sum_sq/n - mean^2) * n/(n-1)) */
                            double wr_var = (n > 1) ? (pa[pi].sum_write_sq / n - avg_wr * avg_wr) * n / (n - 1) : 0.0;
                            double rd_var = (n > 1) ? (pa[pi].sum_read_sq / n - avg_rd * avg_rd) * n / (n - 1) : 0.0;
                            double wr_std = (wr_var > 0) ? sqrt(wr_var) : 0.0;
                            double rd_std = (rd_var > 0) ? sqrt(rd_var) : 0.0;
                            /* Ratio = total_orig / total_compressed (not mean of
                               per-timestep ratios, which has Jensen's inequality bias). */
                            double total_file_mib = pa[pi].sum_file_sz / (double)(1 << 20);
                            double avg_file_mib = (total_file_mib > 0)
                                ? total_file_mib / n : 0.0;
                            double avg_ratio = (total_file_mib > 0)
                                ? (n * orig_mib) / total_file_mib : 1.0;
                            double wmbps = orig_mib / (avg_wr / 1000.0);
                            double rmbps = orig_mib / (avg_rd / 1000.0);
                            double avg_comp_ms = pa[pi].sum_comp_ms / n;
                            double avg_decomp_ms = pa[pi].sum_decomp_ms / n;
                            double orig_bytes = orig_mib * (double)(1 << 20);
                            double cgbps = (avg_comp_ms > 0)
                                ? orig_bytes / 1e9 / (avg_comp_ms / 1000.0) : 0.0;
                            double dgbps = (avg_decomp_ms > 0)
                                ? orig_bytes / 1e9 / (avg_decomp_ms / 1000.0) : 0.0;
                            /* Throughput std-dev from per-timestep comp/decomp ms */
                            double cgbps_std = 0.0, dgbps_std = 0.0;
                            if (n > 1) {
                                double var_c = pa[pi].sum_comp_ms_sq / n - avg_comp_ms * avg_comp_ms;
                                double var_d = pa[pi].sum_decomp_ms_sq / n - avg_decomp_ms * avg_decomp_ms;
                                /* Convert ms std to gbps std via delta method: gbps = B/ms, d(gbps)/d(ms) = -B/ms^2 */
                                if (avg_comp_ms > 0 && var_c > 0)
                                    cgbps_std = cgbps * sqrt(var_c) / avg_comp_ms;
                                if (avg_decomp_ms > 0 && var_d > 0)
                                    dgbps_std = dgbps * sqrt(var_d) / avg_decomp_ms;
                            }
                            fprintf(agg, "%d,vpic,%s,%d,%.2f,%.2f,%.2f,%.2f,"
                                         "%.2f,%.2f,%.4f,"
                                         "%.1f,%.1f,0,%.0f,%.0f,%d,"
                                         "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                                         "%.4f,%.4f,"
                                         "%.2f,%.2f,%.2f,%.2f,"
                                         "%.4f,%.4f,%.4f,%.4f,"
                                         "%.4f,%.4f,%.4f,%.4f,"
                                         "%.4f,%.4f,"
                                         "%.2f,%.2f,%.2f,%.2f,%.2f,"
                                         "%.2f,%.6e,%.6e,%.6e,%.8f,%.4f\n",
                                    rank(), pnames[pi], n, avg_wr, wr_std, avg_rd, rd_std,
                                    avg_file_mib, orig_mib, avg_ratio,
                                    wmbps, rmbps,
                                    (double)pa[pi].sum_sgd / n, (double)pa[pi].sum_expl / n, pa[pi].n_chunks_last,
                                    pa[pi].sum_nn_ms / n, pa[pi].sum_stats_ms / n, pa[pi].sum_preproc_ms / n,
                                    avg_comp_ms, avg_decomp_ms,
                                    pa[pi].sum_explore_ms / n, pa[pi].sum_sgd_ms / n,
                                    cgbps, dgbps,
                                    fmin(200.0, pa[pi].sum_mape_r / n), fmin(200.0, pa[pi].sum_mape_c / n),
                                    fmin(200.0, pa[pi].sum_mape_d / n), fmin(200.0, pa[pi].sum_mape_p / n),
                                    pa[pi].sum_mae_r / n, pa[pi].sum_mae_c / n,
                                    pa[pi].sum_mae_d / n, pa[pi].sum_mae_p / n,
                                    pa[pi].sum_r2_ratio / n, pa[pi].sum_r2_comp / n,
                                    pa[pi].sum_r2_decomp / n, pa[pi].sum_r2_psnr / n,
                                    cgbps_std, dgbps_std,
                                    pa[pi].sum_vs1 / n, pa[pi].sum_vs2 / n, pa[pi].sum_vs3 / n,
                                    pa[pi].sum_vs2_busy / n, pa[pi].sum_vs3_busy / n,
                                    (pa[pi].psnr_count > 0 && pa[pi].sum_mse > 0.0
                                        ? 10.0 * log10((pa[pi].sum_range_sq / pa[pi].psnr_count)
                                                     / (pa[pi].sum_mse / n))
                                        : INFINITY),
                                    sqrt(pa[pi].sum_mse / n),
                                    pa[pi].max_maxerr, pa[pi].sum_meanerr / n,
                                    pa[pi].sum_ssim / n, pa[pi].sum_bitrate / n);
                        }
                    }
                }
                if (agg) fclose(agg);
            }

            global->benchmark_done = 1;
            cudaFree(global->d_read);
            free(global->h_orig);
            free(global->h_read);
            global->d_read = NULL;
            global->h_orig = NULL;
            global->h_read = NULL;
            /* Free weight snapshots before VOL/gpucompress cleanup to avoid
             * conflicts with Kokkos CUDA memory pool finalization. */
            for (int wi = 0; wi < global->n_policies * global->NN_BASE_PHASES; wi++) {
                if (global->nn_weight_snapshots_gpu[wi]) {
                    cudaFree(global->nn_weight_snapshots_gpu[wi]);
                    global->nn_weight_snapshots_gpu[wi] = NULL;
                }
            }
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
            remove(TMP_WARMUP);
            gpucompress_cleanup();
            return;
        }

        // Open CSV on first timestep
        if (global->ts_count == 0) {
            fprintf(stderr, "[rank %d] Entering benchmark loop (timesteps=%d, step=%d)\n",
                    rank(), global->timesteps, (int)step());
            printf("\n══════════════════════════════════════════════════════════════\n");
            printf("  Multi-timestep mode: %d timesteps, nn-rl (SGD active)\n",
                   global->timesteps);
            printf("  Each step = 1 VPIC physics step → H5Dwrite → collect MAPE\n");
            printf("══════════════════════════════════════════════════════════════\n\n");

            global->ts_csv = fopen(TSTEP_CSV, "w");
            fprintf(stderr, "[rank %d] fopen(%s) = %s\n", rank(), TSTEP_CSV,
                    global->ts_csv ? "OK" : "FAILED");
            if (global->ts_csv) {
                fprintf(global->ts_csv, "rank,phase,timestep,sim_step,write_ms,read_ms,ratio,"
                        "mape_ratio,mape_comp,mape_decomp,"
                        "sgd_fires,explorations,n_chunks,mismatches,"
                        "write_mibps,read_mibps,"
                        "file_bytes,orig_mib,"
                        "stats_ms,nn_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
                        "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
                        "vol_stage1_ms,vol_drain_ms,vol_io_drain_ms,"
                        "vol_s2_busy_ms,vol_s3_busy_ms,"
                        "h5dwrite_ms,cuda_sync_ms,h5dclose_ms,h5fclose_ms,"
                        "vol_setup_ms,vol_pipeline_ms,"
                        "psnr_db,psnr_predicted_db,rmse,max_abs_err,mean_abs_err,ssim,bit_rate,"
                        "mape_psnr,"
                        "r2_ratio,r2_comp,r2_decomp,r2_psnr\n");
            }
            global->tc_csv = fopen(TSTEP_CHUNKS_CSV, "w");
            if (global->tc_csv) {
                fprintf(global->tc_csv, "rank,phase,timestep,chunk,action,action_orig,"
                        "predicted_ratio,actual_ratio,"
                        "predicted_comp_ms,actual_comp_ms_raw,"
                        "predicted_decomp_ms,actual_decomp_ms_raw,"
                        "predicted_psnr_db,actual_psnr_db,"
                        "mape_ratio,mape_comp,mape_decomp,mape_psnr,"
                        "sgd_fired,exploration_triggered,"
                        "cost_model_error_pct,actual_cost,predicted_cost,"
                        "explore_n_alt");
                for (int ei = 0; ei < 31; ei++)
                    fprintf(global->tc_csv, ",explore_alt_%d,explore_ratio_%d,explore_comp_ms_%d,explore_cost_%d",
                            ei, ei, ei, ei);
                fprintf(global->tc_csv, ",feat_entropy,feat_mad,feat_deriv\n");
            }
            { const char* nr = getenv("VPIC_NO_RANKING");
              if (nr && atoi(nr)) {
                  sim_log("  [VPIC_NO_RANKING=1] Kendall tau ranking profiler disabled.");
                  global->ranking_csv = NULL;
                  global->ranking_costs_csv = NULL;
              } else {
                  global->ranking_csv = fopen(RANKING_CSV, "w");
                  if (global->ranking_csv)
                      vpic_write_ranking_csv_header(global->ranking_csv);
                  global->ranking_costs_csv = fopen(RANKING_COSTS_CSV, "w");
                  if (global->ranking_costs_csv)
                      vpic_write_ranking_costs_csv_header(global->ranking_costs_csv);
              }
            }

            /* Reload pretrained weights and create per-(NN-phase × policy) GPU snapshots.
             * With N policies and 3 NN base phases, we allocate N*3 snapshots.
             * Each gets independent SGD state — no leakage across phases or policies. */
            const char* wpath = getenv("GPUCOMPRESS_WEIGHTS");
            if (wpath) gpucompress_reload_nn(wpath);

            global->nn_weight_bytes = gpucompress_nn_weights_size();
            int total_snapshots = global->n_policies * global->NN_BASE_PHASES;
            for (int wi = 0; wi < total_snapshots; wi++) {
                cudaMalloc(&global->nn_weight_snapshots_gpu[wi], global->nn_weight_bytes);
                gpucompress_nn_save_snapshot_device(global->nn_weight_snapshots_gpu[wi]);
            }
            printf("  NN weight isolation: %d policies x %d NN phases = %d GPU snapshots (%.1f KB each)\n",
                   global->n_policies, global->NN_BASE_PHASES, total_snapshots,
                   global->nn_weight_bytes / 1024.0);

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
            { "lz4",         TMP_FIX_LZ4,    0, 0, -1, GPUCOMPRESS_ALGO_LZ4,      0 },
            { "snappy",      TMP_FIX_SNAPPY,  0, 0, -1, GPUCOMPRESS_ALGO_SNAPPY,   0 },
            { "deflate",     TMP_FIX_DEFL,    0, 0, -1, GPUCOMPRESS_ALGO_DEFLATE,  0 },
            { "gdeflate",    TMP_FIX_GDEFL,   0, 0, -1, GPUCOMPRESS_ALGO_GDEFLATE, 0 },
            { "zstd",        TMP_FIX_ZSTD,    0, 0, -1, GPUCOMPRESS_ALGO_ZSTD,     0 },
            { "ans",         TMP_FIX_ANS,     0, 0, -1, GPUCOMPRESS_ALGO_ANS,      0 },
            { "cascaded",    TMP_FIX_CASC,    0, 0, -1, GPUCOMPRESS_ALGO_CASCADED,  0 },
            { "bitcomp",     TMP_FIX_BITCOMP, 0, 0, -1, GPUCOMPRESS_ALGO_BITCOMP,   0 },
            { "nn",                TMP_NN,          0, 0,  0, (gpucompress_algorithm_t)0, 0 },
            { "nn-rl",             TMP_NN_RL,       1, 0,  1, (gpucompress_algorithm_t)0, 0 },
            { "nn-rl+exp50",       TMP_NN_RLEXP,    1, 1,  2, (gpucompress_algorithm_t)0, 0 },
        };
        int n_phases_ts = 12;

        hsize_t dims[1] = { (hsize_t)n_floats };

        /* Dump raw field binary for visualization (gated by VPIC_DUMP_FIELDS=1) */
        if (global->dump_fields)
            dump_fields_raw(d_fields, n_floats, t, global->grid_n);

        /* Clear sim progress line, show evolution time */
        if (global->sim_interval > 1 && t > 0) {
            double evolve_elapsed_s = (now_ms() - evolve_start_ms) / 1000.0;
            fprintf(stderr, "\r  Sim evolve %d→%d: %d steps (%.1fs)                                        \n",
                    t - 1, t, global->sim_interval, evolve_elapsed_s);
        }

        for (int pi = 0; pi < n_phases_ts; pi++) {
            const char* phase_name = phases[pi].name;

            /* Skip excluded phases */
            if (is_phase_excluded(phase_name, global->exclude_list)) {
                if (t == 0)
                    printf("  [%s] SKIPPED (excluded by VPIC_EXCLUDE)\n", phase_name);
                continue;
            }

            /* Phase-major: skip phases that don't match VPIC_PHASE (exact match) */
            bool single_phase_mode = (global->single_phase[0] != '\0');
            if (single_phase_mode && strcmp(phase_name, global->single_phase) != 0) {
                continue;  /* silent skip — only run the selected phase */
            }

            int do_sgd  = phases[pi].sgd;
            int do_expl = phases[pi].explore;
            int nn_base_idx = phases[pi].nn_weight_idx; // -1 for fixed, 0/1/2 for nn/nn-rl/nn-rl+exp50
            bool is_nn  = (nn_base_idx >= 0);

            // For NN phases, loop over all policies. For fixed phases, run once.
            int pol_start = 0, pol_end = 1;  // fixed: single iteration
            if (is_nn) {
                pol_start = 0;
                pol_end = global->n_policies;
            }

          for (int pol_idx = pol_start; pol_idx < pol_end; pol_idx++) {

            // Build display name: "nn-rl" for single policy, "nn-rl/ratio" for multi
            char display_name[80];
            if (is_nn && global->n_policies > 1)
                snprintf(display_name, sizeof(display_name), "%s/%s", phase_name, global->policy_labels[pol_idx]);
            else
                snprintf(display_name, sizeof(display_name), "%s", phase_name);

            /* Flush nvCOMP manager cache for NN phases (prevents cache bias between
             * different NN algorithm selections). Fixed phases use hardcoded algorithms
             * so cache state doesn't affect their results. */
            if (is_nn)
                gpucompress_flush_manager_cache();

            /* For NN phases: set this policy's cost weights and restore weight snapshot */
            int weight_idx = -1;
            if (is_nn) {
                gpucompress_set_ranking_weights(
                    global->policy_w0[pol_idx], global->policy_w1[pol_idx], global->policy_w2[pol_idx]);
                weight_idx = pol_idx * global->NN_BASE_PHASES + nn_base_idx;
                gpucompress_nn_restore_snapshot_device(
                    global->nn_weight_snapshots_gpu[weight_idx]);

                /* Verify weight snapshot continuity across timesteps.
                 * After restore, sample 3 weights and compare to what was saved
                 * at the end of the previous timestep. Mismatch = snapshot corruption. */
                if (global->verify_weights && t > 0) {
                    float restored[3];
                    size_t nw = global->nn_weight_bytes / sizeof(float);
                    size_t offs[3] = {nw / 4, nw / 2, nw * 3 / 4};
                    for (int si = 0; si < 3; si++)
                        cudaMemcpy(&restored[si],
                                   (float*)global->nn_weight_snapshots_gpu[weight_idx] + offs[si],
                                   sizeof(float), cudaMemcpyDeviceToHost);
                    bool match = (restored[0] == wt_fingerprints[weight_idx][0] &&
                                  restored[1] == wt_fingerprints[weight_idx][1] &&
                                  restored[2] == wt_fingerprints[weight_idx][2]);
                    if (!match) {
                        fprintf(stderr, "  *** WEIGHT MISMATCH T=%d %s (idx=%d): "
                                "restored=[%.8e,%.8e,%.8e] expected=[%.8e,%.8e,%.8e]\n",
                                t, display_name, weight_idx,
                                restored[0], restored[1], restored[2],
                                wt_fingerprints[weight_idx][0],
                                wt_fingerprints[weight_idx][1],
                                wt_fingerprints[weight_idx][2]);
                    } else if (t == 1) {
                        fprintf(stderr, "  [verify] T=%d %s: weights match previous save ✓\n",
                                t, display_name);
                    }
                }
            }

            if (do_sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, global->reinforce_lr,
                                             global->reinforce_mape, global->reinforce_mape);
            } else {
                gpucompress_disable_online_learning();
            }
            gpucompress_set_exploration(do_expl);
            if (do_expl) {
                gpucompress_set_exploration_threshold(global->explore_thresh);
                gpucompress_set_exploration_k(global->explore_k);
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

            /* Print phase header on first timestep */
            if (t == 0) {
                printf("\n── [%s] (SGD=%s, Explore=%s) ──\n",
                       display_name, do_sgd ? "on" : "off", do_expl ? "on" : "off");
                printf("  %-4s  %-8s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-4s\n",
                       "T", "SimStep", "WrMs", "RdMs", "Ratio",
                       "MAPE_R", "MAPE_C", "MAPE_D", "SGD");
                printf("  ----  --------  -------  -------  -------  "
                       "--------  --------  --------  ----\n");
            }

            /* Per-phase warmup write on first timestep.
             * Primes the VOL pipeline, nvcomp managers, and CUDA JIT so the
             * timed T=0 measurement is not inflated by one-time init costs.
             * Matches Gray-Scott's warmup methodology for fair cross-benchmark
             * comparison. SGD is temporarily disabled during warmup. */
            if (t == 0) {
                int warmup_sgd = gpucompress_online_learning_enabled();
                gpucompress_disable_online_learning();
                gpucompress_set_exploration(0);

                remove(TMP_WARMUP);
                hid_t wfapl = H5Pcreate(H5P_FILE_ACCESS);
                hid_t wnid = H5VLget_connector_id_by_name("native");
                H5Pset_fapl_gpucompress(wfapl, wnid, NULL);
                H5VLclose(wnid);
                hid_t wfile = H5Fcreate(TMP_WARMUP, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
                H5Pclose(wfapl);
                if (wfile >= 0) {
                    hid_t wfsp  = H5Screate_simple(1, dims, NULL);
                    hid_t wdset = H5Dcreate2(wfile, "warmup", H5T_NATIVE_FLOAT,
                                              wfsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                    H5Sclose(wfsp);
                    if (wdset >= 0) {
                        H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                                 H5S_ALL, H5S_ALL, H5P_DEFAULT, d_fields);
                        cudaDeviceSynchronize();
                        H5Dclose(wdset);
                    }
                    H5Fclose(wfile);
                }
                remove(TMP_WARMUP);
                gpucompress_flush_manager_cache();

                /* Restore SGD/exploration state */
                if (warmup_sgd) {
                    gpucompress_enable_online_learning();
                    gpucompress_set_reinforcement(1, global->reinforce_lr,
                                                  global->reinforce_mape, global->reinforce_mape);
                }
                if (do_expl) {
                    gpucompress_set_exploration(1);
                    gpucompress_set_exploration_threshold(global->explore_thresh);
                    gpucompress_set_exploration_k(global->explore_k);
                }
            }

            /* Suppress VOL warnings for no-comp (they are expected and noisy) */
            int saved_stderr = -1;
            if (strcmp(phase_name, "no-comp") == 0) {
                fflush(stderr);
                saved_stderr = dup(STDERR_FILENO);
                int devnull = open("/dev/null", O_WRONLY);
                if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
            }

            /* Write via VOL */
            gpucompress_reset_chunk_history();
            H5VL_gpucompress_reset_stats();  /* prevent stale stage timing from previous phase */
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
            double tw_after_write = now_ms();
            cudaDeviceSynchronize();
            double tw_after_sync = now_ms();
            H5Dclose(dset);
            double tw_after_dclose = now_ms();
            H5Fclose(file);
            double tw1 = now_ms();
            double write_ms_t = tw1 - tw0;

            /* Fine-grained write path breakdown */
            double h5dwrite_ms  = tw_after_write - tw0;
            double cuda_sync_ms = tw_after_sync - tw_after_write;
            double h5dclose_ms  = tw_after_dclose - tw_after_sync;
            double h5fclose_ms  = tw1 - tw_after_dclose;

            /* Capture VOL pipeline stage timing */
            double vol_s1 = 0, vol_drain = 0, vol_io_drain = 0, vol_total = 0;
            H5VL_gpucompress_get_stage_timing(&vol_s1, &vol_drain, &vol_io_drain, &vol_total);
            double vol_s2_busy = 0, vol_s3_busy = 0;
            H5VL_gpucompress_get_busy_timing(&vol_s2_busy, &vol_s3_busy);
            double vol_setup = 0;
            H5VL_gpucompress_get_vol_func_timing(&vol_setup, NULL);

            if (wret < 0) {
                printf("  %-4d  [%s] H5Dwrite failed\n", t, display_name);
                H5Pclose(dcpl);
                /* Restore stderr if suppressed for no-comp */
                if (saved_stderr >= 0) {
                    fflush(stderr);
                    dup2(saved_stderr, STDERR_FILENO);
                    close(saved_stderr);
                    saved_stderr = -1;
                }
                continue;
            }

            /* Always read back for timing; only validate if do_verify */
            drop_pagecache(phases[pi].tmp_file);
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

            unsigned long long mm = 0;
            if (global->do_verify) {
                mm = host_compare(d_fields, global->d_read,
                                  global->h_orig, global->h_read, n_floats);
            }

            /* Quality metrics: PSNR, RMSE, max error — computed on GPU, outside timed section */
            double rmse = 0.0, max_abs_err = 0.0, mean_abs_err = 0.0, ssim = 1.0;
            double psnr_db = vpic_compute_quality_gpu(d_fields, global->d_read, n_floats,
                                                       &rmse, &max_abs_err,
                                                       &mean_abs_err, &ssim);

            /* File size for ratio + derived metrics */
            size_t file_sz = get_file_size(phases[pi].tmp_file);
            double ratio_t = (file_sz > 0) ? (double)nbytes_f / (double)file_sz : 1.0;
            /* Bit rate: compressed bits per value (standard lossy metric).
             * Computed directly as file_bits / n_values rather than 32/ratio
             * to avoid HDF5 metadata skewing the ratio-derived value. */
            double bit_rate = (n_floats > 0) ? (double)file_sz * 8.0 / (double)n_floats : 32.0;

            /* Collect per-chunk MAPE, timing breakdown, MAE/R² */
            int n_hist = gpucompress_get_chunk_history_count();
            double mape_r_sum = 0, mape_c_sum = 0, mape_d_sum = 0, mape_p_sum = 0;
            int    mcnt_r = 0, mcnt_c = 0, mcnt_d = 0, mcnt_p = 0;
            int    sgd_t = 0, expl_t = 0;
            double ts_stats_ms = 0, ts_nn_ms = 0, ts_preproc_ms = 0;
            double ts_comp_ms = 0, ts_decomp_ms = 0, ts_explore_ms = 0, ts_sgd_ms = 0;
            double mae_r_sum = 0, mae_c_sum = 0, mae_d_sum = 0, mae_p_sum = 0;
            double psnr_pred_sum = 0; int psnr_pred_cnt = 0;
            /* R² accumulators: one-pass via Σactual, Σactual², Σ(actual-predicted)² */
            double r2_r_sum = 0, r2_r_sum2 = 0, r2_r_ss_res = 0; int r2_r_cnt = 0;
            double r2_c_sum = 0, r2_c_sum2 = 0, r2_c_ss_res = 0; int r2_c_cnt = 0;
            double r2_d_sum = 0, r2_d_sum2 = 0, r2_d_ss_res = 0; int r2_d_cnt = 0;
            double r2_p_sum = 0, r2_p_sum2 = 0, r2_p_ss_res = 0; int r2_p_cnt = 0;
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
                /* Ratio: MAPE, MAE, R² — same guard for consistent population */
                if (diag.actual_ratio > 0 && diag.predicted_ratio > 0) {
                    double a = diag.actual_ratio, p = diag.predicted_ratio;
                    mape_r_sum += fabs(p - a) / fabs(a);
                    mae_r_sum  += fabs(p - a);
                    r2_r_sum += a; r2_r_sum2 += a*a; r2_r_ss_res += (a-p)*(a-p); r2_r_cnt++;
                    mcnt_r++;
                }
                /* Comp time: MAPE on clamped, MAE/R² on raw */
                if (diag.compression_ms > 0) {
                    mape_c_sum += fabs(diag.predicted_comp_time - diag.compression_ms) / fabs(diag.compression_ms);
                    mcnt_c++;
                }
                if (diag.compression_ms_raw > 0) {
                    double a = diag.compression_ms_raw, p = diag.predicted_comp_time;
                    mae_c_sum += fabs(p - a);
                    r2_c_sum += a; r2_c_sum2 += a*a; r2_c_ss_res += (a-p)*(a-p); r2_c_cnt++;
                }
                /* Decomp time: MAPE on clamped, MAE/R² on raw */
                if (diag.decompression_ms > 0) {
                    mape_d_sum += fabs(diag.predicted_decomp_time - diag.decompression_ms) / fabs(diag.decompression_ms);
                    mcnt_d++;
                }
                if (diag.decompression_ms_raw > 0) {
                    double a = diag.decompression_ms_raw, p = diag.predicted_decomp_time;
                    mae_d_sum += fabs(p - a);
                    r2_d_sum += a; r2_d_sum2 += a*a; r2_d_ss_res += (a-p)*(a-p); r2_d_cnt++;
                }
                /* PSNR MAPE/MAE (skip lossless: actual_psnr=inf, MAPE=0 is trivial) */
                if (diag.predicted_psnr > 0.0f && std::isfinite(diag.actual_psnr) && diag.actual_psnr > 0.0f) {
                    double a = diag.actual_psnr, p = diag.predicted_psnr;
                    mape_p_sum += fabs(p - a) / fabs(a);
                    mae_p_sum  += fabs(p - a);
                    mcnt_p++;
                    r2_p_sum += a; r2_p_sum2 += a*a; r2_p_ss_res += (a-p)*(a-p); r2_p_cnt++;
                }
                if (diag.predicted_psnr > 0.0f) {
                    psnr_pred_sum += diag.predicted_psnr;
                    psnr_pred_cnt++;
                }
            }
            /* Compute per-timestep MAE */
            double ts_mae_r = r2_r_cnt ? mae_r_sum / r2_r_cnt : 0.0;
            double ts_mae_c = r2_c_cnt ? mae_c_sum / r2_c_cnt : 0.0;
            double ts_mae_d = r2_d_cnt ? mae_d_sum / r2_d_cnt : 0.0;

            /* Compute R² = 1 - SS_res/SS_tot where SS_tot = Σ(x²) - (Σx)²/n */
            auto compute_r2 = [](double sum, double sum2, double ss_res, int n) -> double {
                if (n < 2) return 0.0;
                double ss_tot = sum2 - (sum * sum) / n;
                if (ss_tot < 1e-12) return 0.0;
                return 1.0 - ss_res / ss_tot;
            };
            double r2_ratio    = compute_r2(r2_r_sum, r2_r_sum2, r2_r_ss_res, r2_r_cnt);
            double r2_comp     = compute_r2(r2_c_sum, r2_c_sum2, r2_c_ss_res, r2_c_cnt);
            double r2_decomp   = compute_r2(r2_d_sum, r2_d_sum2, r2_d_ss_res, r2_d_cnt);
            double r2_psnr     = compute_r2(r2_p_sum, r2_p_sum2, r2_p_ss_res, r2_p_cnt);
            double real_mape_r = fmin(200.0, mcnt_r ? (mape_r_sum / mcnt_r) * 100.0 : 0.0);
            double real_mape_c = fmin(200.0, mcnt_c ? (mape_c_sum / mcnt_c) * 100.0 : 0.0);
            double real_mape_d = fmin(200.0, mcnt_d ? (mape_d_sum / mcnt_d) * 100.0 : 0.0);
            double real_mape_p = fmin(200.0, mcnt_p ? (mape_p_sum / mcnt_p) * 100.0 : 0.0);
            double ts_mae_p = mcnt_p ? mae_p_sum / mcnt_p : 0.0;
            double psnr_predicted = psnr_pred_cnt ? psnr_pred_sum / psnr_pred_cnt : 0.0;

            double wr_mbps = (write_ms_t > 0) ? orig_mib / (write_ms_t / 1000.0) : 0;
            double rd_mbps = (read_ms_t > 0)  ? orig_mib / (read_ms_t  / 1000.0) : 0;

            /* Print every 5th timestep to stdout (log) */
            bool print_row = (t % 5 == 0 || t == global->timesteps - 1);
            if (print_row) {
                printf("  %-4d  %-8d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %3d\n",
                       t, (int)step(), write_ms_t, read_ms_t,
                       ratio_t, real_mape_r, real_mape_c, real_mape_d, sgd_t);
            }

            /* Restore stderr if we suppressed VOL warnings for no-comp */
            if (saved_stderr >= 0) {
                fflush(stderr);
                dup2(saved_stderr, STDERR_FILENO);
                close(saved_stderr);
            }

            /* Per-phase progress on stderr (visible during run) */
            {
                double phase_total_ms = write_ms_t + read_ms_t;
                fprintf(stderr, "\r  T=%-3d %-20s  ratio=%5.2fx  wr=%5.0f MiB/s  rd=%5.0f MiB/s  (%.1fs)",
                        t + 1, display_name, ratio_t, wr_mbps, rd_mbps, phase_total_ms / 1000.0);
                if (do_sgd)
                    fprintf(stderr, "  SGD=%d", sgd_t);
                /* Pad to overwrite any leftover chars from previous longer line */
                fprintf(stderr, "                    \n");
                fflush(stderr);
            }

            if (global->ts_csv) {
                static int csv_write_count = 0;
                if (csv_write_count < 3)
                    fprintf(stderr, "[rank %d] Writing CSV row: phase=%s t=%d\n", rank(), display_name, t);
                csv_write_count++;
                fprintf(global->ts_csv,
                        "%d,%s,%d,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d,%llu,%.1f,%.1f,"
                        "%zu,%.2f,"
                        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                        "%.4f,%.4f,%.4f,%.4f,"
                        "%.2f,%.2f,%.2f,"
                        "%.2f,%.2f,"
                        "%.2f,%.2f,%.2f,%.2f,"
                        "%.2f,%.2f,"
                        "%.2f,%.2f,%.6e,%.6e,%.6e,%.8f,%.4f,"
                        "%.2f,"
                        "%.4f,%.4f,%.4f,%.4f\n",
                        rank(), display_name, t, (int)step(), write_ms_t, read_ms_t, ratio_t,
                        real_mape_r, real_mape_c, real_mape_d,
                        sgd_t, expl_t, n_hist,
                        (unsigned long long)mm, wr_mbps, rd_mbps,
                        file_sz, orig_mib,
                        ts_stats_ms, ts_nn_ms, ts_preproc_ms,
                        ts_comp_ms, ts_decomp_ms, ts_explore_ms, ts_sgd_ms,
                        ts_mae_r, ts_mae_c, ts_mae_d, ts_mae_p,
                        vol_s1, vol_drain, vol_io_drain,
                        vol_s2_busy, vol_s3_busy,
                        h5dwrite_ms, cuda_sync_ms, h5dclose_ms, h5fclose_ms,
                        vol_setup, vol_total,
                        psnr_db, psnr_predicted,
                        rmse, max_abs_err, mean_abs_err, ssim, bit_rate,
                        real_mape_p,
                        r2_ratio, r2_comp, r2_decomp, r2_psnr);
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

                        double mr = 0, mc = 0, md = 0, mp = 0;
                        if (dd.actual_ratio > 0)
                            mr = fmin(200.0, fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) * 100.0);
                        if (dd.compression_ms > 0)
                            mc = fmin(200.0, fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) * 100.0);
                        if (dd.decompression_ms > 0)
                            md = fmin(200.0, fabs(dd.predicted_decomp_time - dd.decompression_ms) / fabs(dd.decompression_ms) * 100.0);
                        if (dd.actual_psnr > 0.0f && std::isfinite(dd.actual_psnr) && dd.predicted_psnr > 0.0f)
                            mp = fmin(200.0, fabs(dd.predicted_psnr - dd.actual_psnr) / fabs(dd.actual_psnr) * 100.0);

                        char action_str[40], orig_str[40];
                        action_to_str(dd.nn_action, action_str, sizeof(action_str));
                        action_to_str(dd.nn_original_action, orig_str, sizeof(orig_str));
                        fprintf(global->tc_csv,
                                "%d,%s,%d,%d,%s,%s,"
                                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                                "%.2f,%.2f,"
                                "%.2f,%.2f,%.2f,%.2f,%d,%d,"
                                "%.4f,%.4f,%.4f,%d",
                                rank(), display_name, t, ci, action_str, orig_str,
                                (double)dd.predicted_ratio, (double)dd.actual_ratio,
                                (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
                                (double)dd.predicted_decomp_time, (double)dd.decompression_ms_raw,
                                (double)dd.predicted_psnr, (double)dd.actual_psnr,
                                mr, mc, md, mp,
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
                /* Use this policy's cost weights for ranking, not the default */
                float prof_w0 = global->policy_w0[pol_idx];
                float prof_w1 = global->policy_w1[pol_idx];
                float prof_w2 = global->policy_w2[pol_idx];
                RankingMilestoneResult tau_result = {};
                vpic_run_ranking_profiler(
                    d_fields, n_floats * sizeof(float), global->chunk_bytes,
                    global->diag_error_bound,
                    prof_w0, prof_w1, prof_w2, bw,
                    3, global->ranking_csv, global->ranking_costs_csv,
                    display_name, t, &tau_result);
                printf("    [τ] T=%d: τ=%.3f  regret=%.3fx  (%.0fms)\n",
                       t, tau_result.mean_tau,
                       tau_result.mean_regret,
                       tau_result.profiling_ms);
            }

            H5Pclose(dcpl);
            remove(phases[pi].tmp_file);

            /* Save this NN phase's weights back to its GPU snapshot.
             * SGD may have updated them during compression — preserve
             * the updated state for the next timestep. */
            if (weight_idx >= 0 && global->nn_weight_snapshots_gpu[weight_idx]) {
                gpucompress_nn_save_snapshot_device(
                    global->nn_weight_snapshots_gpu[weight_idx]);

                /* Record fingerprint for next-timestep verification */
                if (global->verify_weights) {
                    size_t nw = global->nn_weight_bytes / sizeof(float);
                    size_t offs[3] = {nw / 4, nw / 2, nw * 3 / 4};
                    for (int si = 0; si < 3; si++)
                        cudaMemcpy(&wt_fingerprints[weight_idx][si],
                                   (float*)global->nn_weight_snapshots_gpu[weight_idx] + offs[si],
                                   sizeof(float), cudaMemcpyDeviceToHost);
                }
            }

          } /* end policy loop (for NN phases) */
        } /* end phase loop */

        global->ts_count++;
        // Schedule next sim_interval-1 steps to skip (run sim without benchmarking)
        global->sim_steps_pending = global->sim_interval - 1;
        return;
    }

    // ============================================================
    // All phases run in the multi-timestep loop above.
    // The aggregate CSV is generated from the timestep CSV at completion.
    // ============================================================

    // If no multi-timestep requested, we're done — cleanup now
    if (global->timesteps <= 0) {
        global->benchmark_done = 1;
        cudaFree(global->d_read);
        free(global->h_orig);
        free(global->h_read);
        global->d_read = NULL;
        global->h_orig = NULL;
        global->h_read = NULL;
        for (int wi = 0; wi < global->n_policies * global->NN_BASE_PHASES; wi++) {
            if (global->nn_weight_snapshots_gpu[wi]) {
                cudaFree(global->nn_weight_snapshots_gpu[wi]);
                global->nn_weight_snapshots_gpu[wi] = NULL;
            }
        }
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