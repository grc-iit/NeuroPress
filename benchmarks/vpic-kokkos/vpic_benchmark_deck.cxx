/**
 * @file vpic_benchmark_deck.cxx
 * @brief VPIC Benchmark Deck: No-Comp vs NN+SGD
 *
 * A real VPIC-Kokkos Harris sheet input deck that benchmarks GPU-resident
 * field data compression through the GPUCompress VOL connector under four
 * compression phases — matching the benchmark_grayscott_vol structure.
 *
 * Pipeline per phase:
 *   VPIC simulation (GPU) → field_array->k_f_d → H5Dwrite(d_ptr) → VOL →
 *   GPU compress → HDF5 file → H5Dread(d_ptr) → GPU decompress → bitwise verify
 *
 * Phases (run sequentially after simulation reaches steady state):
 *   1. no-comp      : H5Dwrite via VOL (no filter, VOL-2 D→H fallback)
 *   2. nn           : H5Dwrite via VOL (ALGO_AUTO, inference-only)
 *   3. nn-rl        : ALGO_AUTO + online SGD (MAPE≥20%, LR=0.4)
 *   4. nn-rl+exp50  : ALGO_AUTO + online SGD + Level-2 exploration (MAPE≥50%)
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

// ============================================================
// Constants
// ============================================================
#define REINFORCE_LR        0.4f
#define REINFORCE_MAPE      0.20f

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define N_PHASES 4

#define TMP_NOCOMP   "/tmp/bm_vpic_nocomp.h5"
#define TMP_NN       "/tmp/bm_vpic_nn.h5"
#define TMP_NN_RL    "/tmp/bm_vpic_nn_rl.h5"
#define TMP_NN_RLEXP "/tmp/bm_vpic_nn_rlexp.h5"
#define CHUNKS_CSV   GPU_DIR "/benchmarks/vpic-kokkos/results/benchmark_vpic_chunks.csv"

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                sim_steps;         // Steps to run before benchmarking
    size_t             chunk_bytes;       // HDF5 chunk size in bytes
    gpucompress_vpic_t vpic_fields_h;     // Adapter handle for fields
    hid_t              vol_fapl;          // File access property list with VOL
    hid_t              vol_id;            // VOL connector ID
    int                gpucompress_ready; // 1 if init succeeded
    int                benchmark_done;    // 1 after all benchmark steps complete

    // Buffers for benchmark
    float*             d_read;            // GPU read-back buffer
    float*             h_orig;            // Host buffer for verification
    float*             h_read;            // Host buffer for verification
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
    double stats_ms;
    double nn_ms;
    double preproc_ms;
    double comp_ms;
    double explore_ms;
    double sgd_ms;
};

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

    // VOL write
    printf("  [%s] H5Dwrite (GPU ptr, VOL)... ", phase_name); fflush(stdout);
    remove(tmp_file);

    gpucompress_reset_chunk_history();
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
    double ape_ratio_sum = 0.0, ape_comp_sum = 0.0, ape_decomp_sum = 0.0;
    int    ape_ratio_cnt = 0,   ape_comp_cnt = 0,   ape_decomp_cnt = 0;
    double total_stats_ms   = 0.0;
    double total_nn_ms      = 0.0;
    double total_preproc_ms = 0.0;
    double total_comp_ms    = 0.0;
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
            fprintf(chunk_csv, "phase,chunk,action_final,action_orig,actual_ratio,"
                               "predicted_ratio,mape_ratio,"
                               "actual_comp_ms,predicted_comp_ms,mape_comp,"
                               "actual_decomp_ms,predicted_decomp_ms,mape_decomp,"
                               "sgd_fired,exploration_triggered,"
                               "nn_inference_ms,preprocessing_ms,compression_ms,"
                               "exploration_ms,sgd_update_ms\n");
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
            total_comp_ms    += d.compression_ms;
            total_explore_ms += d.exploration_ms;
            total_sgd_ms     += d.sgd_update_ms;
            if (d.predicted_ratio > 0 && d.actual_ratio > 0) {
                ape_ratio_sum += fabs(d.predicted_ratio - d.actual_ratio)
                                 / d.actual_ratio * 100.0;
                ape_ratio_cnt++;
            }
            if (d.compression_ms > 0) {
                ape_comp_sum += fabs(d.predicted_comp_time - d.compression_ms)
                                / d.compression_ms * 100.0;
                ape_comp_cnt++;
            }
            if (d.decompression_ms > 0) {
                ape_decomp_sum += fabs(d.predicted_decomp_time - d.decompression_ms)
                                  / d.decompression_ms * 100.0;
                ape_decomp_cnt++;
            }
            char final_str[40], orig_str[40];
            action_to_str(d.nn_action, final_str, sizeof(final_str));
            action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));
            double chunk_mape = (d.actual_ratio > 0.0f)
                ? fabs((double)d.predicted_ratio - (double)d.actual_ratio)
                  / (double)d.actual_ratio * 100.0
                : 0.0;
            printf("    %5d | %-20s | %-20s | %5.2fx | %5.2fx | %5.1f%% | %s | %s\n",
                   i + 1, final_str, orig_str,
                   (double)d.actual_ratio, (double)d.predicted_ratio,
                   chunk_mape,
                   d.sgd_fired ? "yes" : "  -",
                   d.exploration_triggered ? "yes" : "  -");
            if (chunk_csv) {
                double mape_comp = (d.compression_ms > 0 && d.predicted_comp_time > 0)
                    ? fabs((double)d.predicted_comp_time - (double)d.compression_ms)
                      / (double)d.compression_ms * 100.0 : 0.0;
                double mape_decomp = (d.decompression_ms > 0 && d.predicted_decomp_time > 0)
                    ? fabs((double)d.predicted_decomp_time - (double)d.decompression_ms)
                      / (double)d.decompression_ms * 100.0 : 0.0;
                fprintf(chunk_csv, "%s,%d,%s,%s,%.4f,%.4f,%.1f,"
                                   "%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,"
                                   "%d,%d,"
                                   "%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        phase_name, i + 1, final_str, orig_str,
                        (double)d.actual_ratio, (double)d.predicted_ratio,
                        chunk_mape,
                        (double)d.compression_ms, (double)d.predicted_comp_time,
                        mape_comp,
                        (double)d.decompression_ms, (double)d.predicted_decomp_time,
                        mape_decomp,
                        d.sgd_fired, d.exploration_triggered,
                        (double)d.nn_inference_ms, (double)d.preprocessing_ms,
                        (double)d.compression_ms, (double)d.exploration_ms,
                        (double)d.sgd_update_ms);
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
    r->mape_ratio_pct  = (ape_ratio_cnt > 0) ? ape_ratio_sum / ape_ratio_cnt : 0.0;
    r->mape_comp_pct   = (ape_comp_cnt > 0) ? ape_comp_sum / ape_comp_cnt : 0.0;
    r->mape_decomp_pct = (ape_decomp_cnt > 0) ? ape_decomp_sum / ape_decomp_cnt : 0.0;
    r->stats_ms     = total_stats_ms;
    r->nn_ms        = total_nn_ms;
    r->preproc_ms   = total_preproc_ms;
    r->comp_ms      = total_comp_ms;
    r->explore_ms   = total_explore_ms;
    r->sgd_ms       = total_sgd_ms;
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

    // Run warmup steps to develop physics, then one benchmark step
    num_step        = warmup + 1;
    status_interval = 50;
    clean_div_e_interval = 10;
    clean_div_b_interval = 10;

    global->sim_steps      = warmup;
    global->chunk_bytes    = (size_t)chunk_mb * 1024 * 1024;
    global->benchmark_done = 0;

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
    sim_log("  Env vars : VPIC_NX=" << grid_n << " VPIC_CHUNK_MB=" << chunk_mb
            << " VPIC_WARMUP_STEPS=" << warmup);
    sim_log("  Weights  : " << (weights_path ? weights_path : "(none, fallback to LZ4)"));
    sim_log("  NN loaded: " << (gpucompress_nn_is_loaded() ? "yes" : "no"));

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

    // Attach GPU-resident field data
    vpic_attach_fields(global->vpic_fields_h, field_array->k_f_d);

    float*  d_fields = NULL;
    size_t  nbytes_f = 0;
    gpucompress_vpic_get_device_ptrs(global->vpic_fields_h,
                                     &d_fields, NULL, &nbytes_f, NULL);
    size_t n_floats    = nbytes_f / sizeof(float);
    size_t chunk_floats = global->chunk_bytes / sizeof(float);
    int    n_chunks    = (int)((n_floats + chunk_floats - 1) / chunk_floats);

    sim_log("");
    sim_log("╔═══════════════════════════════════════════════════════════════════════════╗");
    sim_log("║  VPIC Benchmark: No-Comp vs NN+SGD (Real Harris Sheet)                  ║");
    sim_log("╚═══════════════════════════════════════════════════════════════════════════╝");
    sim_log("");
    sim_log("  Step " << step() << ": field data on GPU, " << nbytes_f / (1024*1024)
            << " MB, " << n_chunks << " chunks of "
            << global->chunk_bytes / (1024*1024) << " MB");
    sim_log("");

    // Truncate per-chunk CSV for fresh run
    remove(CHUNKS_CSV);

    const char* env_eb_diag = getenv("VPIC_ERROR_BOUND");
    double diag_error_bound = env_eb_diag ? atof(env_eb_diag) : 0.0;

    /* Phase selection: VPIC_PHASES="nn-rl" to run only those.
     * Default (unset or empty): run all 4 phases. */
    const char* env_phases = getenv("VPIC_PHASES");
    auto phase_enabled = [&](const char* name) -> bool {
        if (!env_phases || env_phases[0] == '\0') return true;  /* all */
        return strstr(env_phases, name) != NULL;
    };

    if (env_phases && env_phases[0])
        sim_log("  Phase filter: " << env_phases);

    PhaseResult results[N_PHASES];
    int n_phases = 0;
    int any_fail = 0;

    // ── Phase 1: no-comp ──────────────────────────────────────────
    if (phase_enabled("no-comp")) {
    sim_log("── Phase 1/4: no-comp (GPU→Host→HDF5, VOL-2 fallback) ────────");
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_nocomp((hsize_t)chunk_floats);
        int rc = run_phase("no-comp", TMP_NOCOMP,
                           d_fields, global->d_read, global->h_orig, global->h_read,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        if (rc) any_fail = 1;
        n_phases++;
    }
    }

    // ── Phase 2: nn (inference-only) ──────────────────────────────
    if (phase_enabled("nn-only") || (phase_enabled("nn") && !env_phases)) {
    sim_log("── Phase 2/4: nn (VOL, ALGO_AUTO, inference-only) ───────────");
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats, diag_error_bound);
        int rc = run_phase("nn", TMP_NN,
                           d_fields, global->d_read, global->h_orig, global->h_read,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        if (rc) any_fail = 1;
        n_phases++;
    }
    }

    // ── Phase 3: nn-rl (SGD, no exploration) ──────────────────────
    if (phase_enabled("nn-rl")) {
    sim_log("── Phase 3/4: nn-rl (ALGO_AUTO + SGD, MAPE>=20%, LR=0.4) ───");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    gpucompress_set_exploration(0);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats, diag_error_bound);
        int rc = run_phase("nn-rl", TMP_NN_RL,
                           d_fields, global->d_read, global->h_orig, global->h_read,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        n_phases++;
    }
    }

    // ── Phase 4: nn-rl+exp50 (SGD + exploration) ──────────────────
    if (phase_enabled("nn-rl+exp")) {
    // Reset NN weights so phase 4 starts from original trained weights,
    // not the SGD-modified weights from phase 3.
    const char* wpath = getenv("GPUCOMPRESS_WEIGHTS");
    if (wpath) gpucompress_reload_nn(wpath);
    sim_log("── Phase 4/4: nn-rl+exp50 (ALGO_AUTO + SGD + expl@MAPE>=50%) ");
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.50);
    gpucompress_set_exploration_k(2);
    {
        hid_t dcpl = make_dcpl_auto((hsize_t)chunk_floats, diag_error_bound);
        int rc = run_phase("nn-rl+exp50", TMP_NN_RLEXP,
                           d_fields, global->d_read, global->h_orig, global->h_read,
                           n_floats, n_chunks, dcpl, &results[n_phases]);
        H5Pclose(dcpl);
        gpucompress_disable_online_learning();
        if (rc) any_fail = 1;
        n_phases++;
    }
    }

    // ── Summary table ─────────────────────────────────────────────
    sim_log("");
    printf("\n╔══════════════╦══════════╦══════════╦═══════╦══════════╦═════════════╗\n");
    printf("║  Phase       ║ Write    ║ Read     ║ Ratio ║ File MiB ║ Verify      ║\n");
    printf("║              ║ (MiB/s)  ║ (MiB/s)  ║       ║          ║             ║\n");
    printf("╠══════════════╬══════════╬══════════╬═══════╬══════════╬═════════════╣\n");
    for (int i = 0; i < n_phases; i++) {
        PhaseResult* rr = &results[i];
        const char* verdict = (rr->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %8.0f ║ %8.0f ║ %5.2fx ║ %8.0f ║ %-11s ║\n",
               rr->phase, rr->write_mbps, rr->read_mbps,
               rr->ratio, (double)rr->file_bytes / (1 << 20), verdict);
    }
    printf("╚══════════════╩══════════╩══════════╩═══════╩══════════╩═════════════╝\n");

    for (int i = 0; i < n_phases; i++) {
        if (strncmp(results[i].phase, "nn", 2) == 0 && results[i].n_chunks > 0) {
            printf("\n  %-14s SGD: %d/%d  Expl: %d/%d  MAPE: ratio=%.1f%% comp=%.1f%% decomp=%.1f%%\n",
                   results[i].phase,
                   results[i].sgd_fires,   results[i].n_chunks,
                   results[i].explorations, results[i].n_chunks,
                   results[i].mape_ratio_pct, results[i].mape_comp_pct, results[i].mape_decomp_pct);
        }
    }

    // GPU-time overhead breakdown for NN phases
    {
        bool has_nn = false;
        for (int i = 0; i < n_phases; i++)
            if (strncmp(results[i].phase, "nn", 2) == 0) { has_nn = true; break; }

        if (has_nn) {
            printf("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
            printf("║  GPU-Time Overhead Breakdown (cumulative across chunks, 8 concurrent workers)                  ║\n");
            printf("╠══════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦════════════════╣\n");
            printf("║  Phase       ║ Stats    ║ NN Infer ║ Preproc  ║ Compress ║ Explore  ║ SGD      ║ Total GPU-time ║\n");
            printf("╠══════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬════════════════╣\n");
            for (int i = 0; i < n_phases; i++) {
                if (strncmp(results[i].phase, "nn", 2) != 0) continue;
                PhaseResult* rr = &results[i];
                double total_gpu = rr->stats_ms + rr->nn_ms + rr->preproc_ms + rr->comp_ms
                                 + rr->explore_ms + rr->sgd_ms;
                printf("║  %-12s║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║ %5.0f ms ║   %7.0f ms   ║\n",
                       rr->phase, rr->stats_ms, rr->nn_ms, rr->preproc_ms, rr->comp_ms,
                       rr->explore_ms, rr->sgd_ms, total_gpu);
            }
            printf("╠══════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬════════════════╣\n");
            printf("║  (%% of GPU)  ║          ║          ║          ║          ║          ║          ║                ║\n");
            for (int i = 0; i < n_phases; i++) {
                if (strncmp(results[i].phase, "nn", 2) != 0) continue;
                PhaseResult* rr = &results[i];
                double total_gpu = rr->stats_ms + rr->nn_ms + rr->preproc_ms + rr->comp_ms
                                 + rr->explore_ms + rr->sgd_ms;
                if (total_gpu <= 0) total_gpu = 1.0;
                printf("║  %-12s║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║ %5.1f%%   ║   wall: %4.0f ms ║\n",
                       rr->phase,
                       100.0 * rr->stats_ms / total_gpu,
                       100.0 * rr->nn_ms / total_gpu,
                       100.0 * rr->preproc_ms / total_gpu,
                       100.0 * rr->comp_ms / total_gpu,
                       100.0 * rr->explore_ms / total_gpu,
                       100.0 * rr->sgd_ms / total_gpu,
                       rr->write_ms);
            }
            printf("╚══════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩════════════════╝\n");
        }
    }

    // Write summary CSV
    {
        const char* csv_dir  = GPU_DIR "/benchmarks/vpic-kokkos/results";
        const char* csv_path = GPU_DIR "/benchmarks/vpic-kokkos/results/benchmark_vpic_deck.csv";
        mkdir(csv_dir, 0755);
        FILE* csv = fopen(csv_path, "w");
        if (csv) {
            fprintf(csv, "source,phase,write_ms,read_ms,file_mib,orig_mib,ratio,"
                         "write_mibps,read_mibps,mismatches,sgd_fires,explorations,n_chunks,"
                         "mape_ratio_pct,mape_comp_pct,mape_decomp_pct\n");
            for (int i = 0; i < n_phases; i++) {
                PhaseResult* rr = &results[i];
                fprintf(csv, "vpic,%s,%.2f,%.2f,%.2f,%.2f,%.4f,"
                             "%.1f,%.1f,%llu,%d,%d,%d,%.2f,%.2f,%.2f\n",
                        rr->phase, rr->write_ms, rr->read_ms,
                        (double)rr->file_bytes / (1 << 20),
                        (double)rr->orig_bytes / (1 << 20), rr->ratio,
                        rr->write_mbps, rr->read_mbps,
                        rr->mismatches, rr->sgd_fires, rr->explorations, rr->n_chunks,
                        rr->mape_ratio_pct, rr->mape_comp_pct, rr->mape_decomp_pct);
            }
            fclose(csv);
        }
    }

    if (any_fail) printf("\n=== VPIC Benchmark FAILED ===\n");

    global->benchmark_done = 1;
    printf("\n=== VPIC Benchmark complete ===\n");
    printf("Chunks CSV: %s\n", CHUNKS_CSV);

    // Cleanup
    cudaFree(global->d_read);
    free(global->h_orig);
    free(global->h_read);
    global->d_read  = NULL;
    global->h_orig  = NULL;
    global->h_read  = NULL;

    gpucompress_vpic_destroy(global->vpic_fields_h);
    H5Pclose(global->vol_fapl);
    H5VLclose(global->vol_id);
    gpucompress_cleanup();
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