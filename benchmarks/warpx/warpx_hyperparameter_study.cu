/**
 * @file warpx_hyperparameter_study.cu
 * @brief WarpX compression hyperparameter study with evolving PIC data
 *
 * Generates GPU-resident data modelling laser-wakefield acceleration (LWFA).
 * All 6 field components (Ex, Ey, Bz, Jx, Jy, rho) are packed into a single
 * contiguous GPU buffer (~192 MB by default), matching the VPIC benchmark's
 * approach of writing one large blob per timestep through all compression
 * phases.  Data evolves through distinct physics stages — vacuum, laser entry,
 * bubble formation, wakefield growth, electron trapping, and saturation —
 * giving wide diversity in compressibility across timesteps and within each
 * snapshot (different spatial regions compress differently).
 *
 * Environment variables:
 *   WARPX_DATA_MB         Target data size per write in MB (default 192)
 *   WARPX_TIMESTEPS       Number of write cycles (default 50)
 *   WARPX_CHUNK_MB        Chunk size in MB (default 4)
 *   WARPX_ERROR_BOUND     Lossy error bound (default 0.0 = lossless)
 *   WARPX_SIM_INTERVAL    Physics steps between writes (default 10)
 *   WARPX_WARMUP_STEPS    Pre-benchmark evolution steps (default 20)
 *   WARPX_POLICIES        Comma-separated: balanced,ratio,speed
 *   GPUCOMPRESS_WEIGHTS   Path to .nnwt weights file
 *   WARPX_LR              SGD learning rate (default 0.2)
 *   WARPX_MAPE_THRESHOLD  SGD MAPE threshold (default 0.20)
 *   WARPX_EXPLORE_K       Exploration alternatives (default 4)
 *   WARPX_EXPLORE_THRESH  Exploration cost-error threshold (default 0.20)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "gpucompress_warpx.h"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static double now_ms() {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

/* ============================================================
 * Field component indices within the packed buffer.
 * Layout: [Ex | Ey | Bz | Jx | Jy | rho] each n_cells floats.
 * Total: 6 * n_cells floats = target ~192 MB.
 * ============================================================ */
#define NCOMP 6
#define COMP_EX  0
#define COMP_EY  1
#define COMP_BZ  2
#define COMP_JX  3
#define COMP_JY  4
#define COMP_RHO 5

static const char* comp_names[NCOMP] = {"Ex", "Ey", "Bz", "Jx", "Jy", "rho"};

/* ============================================================
 * Physics-informed LWFA evolution kernel.
 *
 * Each thread handles one cell, writes all 6 components.
 * This keeps the data layout identical to how WarpX would
 * store 6 separate MultiFabs concatenated for I/O.
 * ============================================================ */

__global__ void evolve_all_fields(float* data, int nx, int ny, int n_cells,
                                  float sim_time, int sim_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    int ix = idx % nx;
    int iy = idx / nx;

    float x = (float)ix / (float)nx;
    float y = (float)iy / (float)ny;
    float t = sim_time;  /* normalized [0, 1] */

    /* ---- Laser pulse: Gaussian envelope moving in +x ---- */
    float laser_x0 = 0.1f + t * 0.7f;
    float laser_sigma = 0.05f + 0.02f * t;
    float dx = x - laser_x0;
    float dy = y - 0.5f;
    float envelope = expf(-(dx*dx)/(2.0f*laser_sigma*laser_sigma)
                         -(dy*dy)/(2.0f*0.08f*0.08f));
    float laser_freq = 50.0f + 20.0f * t;

    /* ---- Plasma wakefield behind laser ---- */
    float wake_strength = fminf(t * 3.0f, 1.0f);
    float wake_region = fmaxf(0.0f, laser_x0 - x);
    float wake_decay = expf(-wake_region * 5.0f);
    float kp = 15.0f + 10.0f * t;

    /* ---- Bubble cavity ---- */
    float bubble_r = 0.12f * fminf(t * 2.0f, 1.0f);
    float bubble_cx = laser_x0 - 0.15f;
    float bx = x - bubble_cx;
    float by = y - 0.5f;
    float r = sqrtf(bx*bx + by*by);

    /* ---- Noise (grows in late phases) ---- */
    unsigned int h1 = (unsigned int)(idx * 2654435761u + sim_step * 40503u);
    unsigned int h2 = h1 * 2246822519u + 3266489917u;
    unsigned int h3 = h2 * 1103515245u + 12345u;
    unsigned int h4 = h3 * 6364136223u + 1442695041u;
    unsigned int h5 = h4 * 1664525u + 1013904223u;
    unsigned int h6 = h5 * 214013u + 2531011u;
    float noise_amp = (t > 0.5f) ? (t - 0.5f) * 40.0f : 0.0f;

    /* ---- Ex: laser + wakefield + bubble radial ---- */
    float Ex_val = envelope * sinf(laser_freq * x - 30.0f * t) * 100.0f * t
                 + wake_strength * wake_decay * sinf(kp * wake_region) * 120.0f * sinf(3.14159f * y)
                 + ((r < bubble_r && t > 0.2f) ? (1.0f - r/bubble_r) * 200.0f * t * bx/fmaxf(r,0.001f) : 0.0f)
                 + ((float)(h1 & 0xFFFF)/65535.0f - 0.5f) * noise_amp;

    /* ---- Ey: laser + transverse wake ---- */
    float Ey_val = envelope * cosf(laser_freq * x - 30.0f * t) * 80.0f * t
                 + wake_strength * wake_decay * cosf(kp * wake_region) * 60.0f * cosf(3.14159f * y * 2.0f)
                 + ((float)(h2 & 0xFFFF)/65535.0f - 0.5f) * noise_amp;

    /* ---- Bz: laser B-field + toroidal around bubble ---- */
    float toroidal = 0.0f;
    if (t > 0.2f && r > 0.01f && r < bubble_r * 1.5f) {
        float shell = expf(-((r - bubble_r)*(r - bubble_r))/(0.02f*0.02f));
        toroidal = shell * 300.0f * t * (-by)/r;
    }
    float Bz_val = envelope * cosf(50.0f * x - 30.0f * t) * 90.0f * t
                 + toroidal
                 + ((t > 0.6f) ? ((float)(h3 & 0xFFFF)/65535.0f - 0.5f) * (t-0.6f) * 30.0f : 0.0f);

    /* ---- Jx, Jy: return current at bubble shell + beam current ---- */
    float shell = expf(-((r - bubble_r)*(r - bubble_r))/(0.015f*0.015f));
    float j_strength = (t > 0.25f) ? fminf((t - 0.25f) * 4.0f, 1.0f) : 0.0f;

    float Jx_val = shell * j_strength * 500.0f * by/fmaxf(r,0.001f);
    float Jy_val = shell * j_strength * 500.0f * (-bx)/fmaxf(r,0.001f);
    if (t > 0.5f && r < bubble_r * 0.5f)
        Jx_val += (t - 0.5f) * 800.0f * expf(-r*r/(0.03f*0.03f));
    if (t > 0.7f) {
        Jx_val += ((float)(h4 & 0xFFFF)/65535.0f - 0.5f) * (t-0.7f) * 100.0f;
        Jy_val += ((float)(h5 & 0xFFFF)/65535.0f - 0.5f) * (t-0.7f) * 80.0f;
    }

    /* ---- rho: background + wake density + cavity + spike ---- */
    float rho_val = 1.0f
                  + fminf(t*2.0f, 0.8f) * sinf(kp*wake_region) * expf(-wake_region*3.0f)
                  + ((r < bubble_r && t > 0.2f) ? -0.9f*(1.0f - r*r/(bubble_r*bubble_r)) : 0.0f)
                  + ((t > 0.3f) ? expf(-((r-bubble_r)*(r-bubble_r))/(0.01f*0.01f)) * 3.0f * fminf((t-0.3f)*3.0f, 1.0f) : 0.0f);
    if (t > 0.6f)
        rho_val += ((float)(h6 & 0xFFFF)/65535.0f - 0.5f) * (t-0.6f) * 0.5f;

    /* Write all 6 components contiguously: field[comp][idx] layout */
    data[COMP_EX  * n_cells + idx] = Ex_val;
    data[COMP_EY  * n_cells + idx] = Ey_val;
    data[COMP_BZ  * n_cells + idx] = Bz_val;
    data[COMP_JX  * n_cells + idx] = Jx_val;
    data[COMP_JY  * n_cells + idx] = Jy_val;
    data[COMP_RHO * n_cells + idx] = rho_val;
}

/* ============================================================
 * Main benchmark
 * ============================================================ */

int main()
{
    printf("=== WarpX Hyperparameter Study: LWFA Data Evolution ===\n\n");

    /* ---- Parse environment ---- */
    auto envint = [](const char* name, int def) -> int {
        const char* v = getenv(name); return v ? atoi(v) : def;
    };
    auto envflt = [](const char* name, float def) -> float {
        const char* v = getenv(name); return v ? (float)atof(v) : def;
    };
    auto envstr = [](const char* name, const char* def) -> const char* {
        const char* v = getenv(name); return (v && v[0]) ? v : def;
    };

    int    data_mb       = envint("WARPX_DATA_MB", 192);
    int    timesteps     = envint("WARPX_TIMESTEPS", 50);
    int    chunk_mb      = envint("WARPX_CHUNK_MB", 4);
    double error_bound   = (double)envflt("WARPX_ERROR_BOUND", 0.0f);
    int    sim_interval  = envint("WARPX_SIM_INTERVAL", 10);
    int    warmup_steps  = envint("WARPX_WARMUP_STEPS", 20);
    float  lr            = envflt("WARPX_LR", 0.2f);
    float  mape_thresh   = envflt("WARPX_MAPE_THRESHOLD", 0.20f);
    int    explore_k     = envint("WARPX_EXPLORE_K", 4);
    float  explore_thresh= envflt("WARPX_EXPLORE_THRESH", 0.20f);
    const char* weights  = envstr("GPUCOMPRESS_WEIGHTS", "neural_net/weights/model.nnwt");
    const char* policies_str = envstr("WARPX_POLICIES", "balanced,ratio");

    /* Compute grid from target data size:
     * total_bytes = 6 * nx * ny * 4  =>  n_cells = total_bytes / (6*4) */
    size_t total_bytes = (size_t)data_mb * 1024 * 1024;
    size_t n_cells = total_bytes / (NCOMP * sizeof(float));
    int ny = (int)sqrtf((float)n_cells);
    int nx = (int)(n_cells / ny);
    n_cells = (size_t)nx * ny;  /* recalculate after rounding */
    total_bytes = n_cells * NCOMP * sizeof(float);
    size_t n_floats = n_cells * NCOMP;
    size_t chunk_bytes = (size_t)chunk_mb * 1024 * 1024;
    int n_chunks = (int)((total_bytes + chunk_bytes - 1) / chunk_bytes);

    printf("Configuration:\n");
    printf("  Target data:    %d MB per write (%d chunks @ %d MB)\n", data_mb, n_chunks, chunk_mb);
    printf("  Grid:           %d x %d = %zu cells\n", nx, ny, n_cells);
    printf("  Fields:         %d components (Ex, Ey, Bz, Jx, Jy, rho)\n", NCOMP);
    printf("  Actual size:    %.1f MB (%zu floats)\n", total_bytes/(1024.0*1024.0), n_floats);
    printf("  Timesteps:      %d (warmup=%d, sim_interval=%d)\n", timesteps, warmup_steps, sim_interval);
    printf("  Error bound:    %.6f (%s)\n", error_bound, error_bound > 0 ? "LOSSY" : "LOSSLESS");
    printf("  NN weights:     %s\n", weights);
    printf("  Policies:       %s\n", policies_str);
    printf("  SGD LR=%.3f, MAPE_thresh=%.2f, explore_K=%d, explore_thresh=%.2f\n\n",
           lr, mape_thresh, explore_k, explore_thresh);

    /* ---- Parse policies ---- */
    struct Policy { const char* name; float w0, w1, w2; };
    Policy all_policies[3];
    int n_policies = 0;
    {
        char buf[256];
        strncpy(buf, policies_str, sizeof(buf)-1); buf[sizeof(buf)-1]='\0';
        char* tok = strtok(buf, ",");
        while (tok && n_policies < 3) {
            while (*tok == ' ') tok++;
            if      (strcmp(tok,"balanced")==0) all_policies[n_policies] = {"balanced",1,1,1};
            else if (strcmp(tok,"ratio")==0)    all_policies[n_policies] = {"ratio",0,0,1};
            else if (strcmp(tok,"speed")==0)    all_policies[n_policies] = {"speed",1,1,0};
            else { tok = strtok(NULL,","); continue; }
            n_policies++; tok = strtok(NULL,",");
        }
    }
    if (n_policies == 0) { all_policies[0] = {"balanced",1,1,1}; n_policies = 1; }

    /* ---- GPU memory ---- */
    size_t free_mem = 0, total_mem = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t max_comp = gpucompress_max_compressed_size(total_bytes);
    size_t needed = total_bytes + max_comp + total_bytes; /* data + comp + decomp */
    printf("GPU memory: %.0f MB free / %.0f MB total, need %.0f MB\n",
           free_mem/(1024.0*1024.0), total_mem/(1024.0*1024.0), needed/(1024.0*1024.0));
    if (free_mem < needed) {
        fprintf(stderr, "ERROR: not enough GPU memory. Reduce WARPX_DATA_MB.\n");
        return 1;
    }

    float* d_data = NULL;
    void*  d_comp = NULL;
    void*  d_decomp = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_comp, max_comp));
    CHECK_CUDA(cudaMalloc(&d_decomp, total_bytes));

    /* ---- Initialize gpucompress ---- */
    gpucompress_error_t gerr = gpucompress_init(weights);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init('%s') failed (%d), trying NULL\n", weights, gerr);
        gerr = gpucompress_init(NULL);
        if (gerr != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "FATAL\n"); return 1; }
    }
    bool has_nn = (gpucompress_nn_is_loaded() == 1);
    if (!has_nn) printf("WARNING: NN weights not loaded — NN phases will be skipped\n");

    /* ---- NN weight snapshots (per policy x per NN base phase) ---- */
    int NN_BASE_PHASES = 3;
    int total_snaps = n_policies * NN_BASE_PHASES;
    std::vector<void*> nn_snaps(total_snaps, nullptr);
    size_t nn_bytes = 0;
    if (has_nn) {
        nn_bytes = gpucompress_nn_weights_size();
        for (int i = 0; i < total_snaps; i++) {
            CHECK_CUDA(cudaMalloc(&nn_snaps[i], nn_bytes));
            gpucompress_nn_save_snapshot_device(nn_snaps[i]);
        }
        printf("NN isolation: %d policies x %d NN phases = %d GPU snapshots (%.1f KB each)\n\n",
               n_policies, NN_BASE_PHASES, total_snaps, nn_bytes/1024.0);
    }

    /* ---- Phase definitions ---- */
    struct Phase {
        const char* name;
        gpucompress_algorithm_t algo;
        int sgd, explore, nn_base_idx;
    };
    Phase phases[] = {
        { "lz4",       GPUCOMPRESS_ALGO_LZ4,      0, 0, -1 },
        { "snappy",    GPUCOMPRESS_ALGO_SNAPPY,    0, 0, -1 },
        { "deflate",   GPUCOMPRESS_ALGO_DEFLATE,   0, 0, -1 },
        { "gdeflate",  GPUCOMPRESS_ALGO_GDEFLATE,  0, 0, -1 },
        { "zstd",      GPUCOMPRESS_ALGO_ZSTD,      0, 0, -1 },
        { "ans",       GPUCOMPRESS_ALGO_ANS,       0, 0, -1 },
        { "cascaded",  GPUCOMPRESS_ALGO_CASCADED,  0, 0, -1 },
        { "bitcomp",   GPUCOMPRESS_ALGO_BITCOMP,   0, 0, -1 },
        { "nn",        (gpucompress_algorithm_t)0, 0, 0,  0 },
        { "nn-rl",     (gpucompress_algorithm_t)0, 1, 0,  1 },
        { "nn-rl+exp", (gpucompress_algorithm_t)0, 1, 1,  2 },
    };
    int n_phases = 11;

    /* ---- Open CSV ---- */
    FILE* csv = fopen("warpx_hyperparameter_study.csv", "w");
    if (!csv) { fprintf(stderr, "Cannot open CSV\n"); return 1; }
    fprintf(csv, "timestep,sim_step,sim_time,"
                 "phase,policy,"
                 "original_mb,compressed_mb,ratio,"
                 "write_ms,throughput_gbps,"
                 "n_chunks,sgd_fires,explorations,"
                 "mape_ratio_pct,max_error,"
                 "algorithm_used\n");

    /* ---- Warmup ---- */
    int global_step = 0;
    int total_steps = warmup_steps + timesteps * sim_interval;
    int grid = ((int)n_cells + 255) / 256;

    printf("Warming up: %d steps ...\n", warmup_steps);
    for (int w = 0; w < warmup_steps; w++, global_step++) {
        float t = (float)global_step / (float)total_steps;
        evolve_all_fields<<<grid, 256>>>(d_data, nx, ny, (int)n_cells, t, global_step);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup done (sim_step=%d)\n\n", global_step);

    /* ---- Main timestep loop ---- */
    printf("%-3s %-5s %-5s %-16s %-10s %7s %9s %9s %5s %4s %5s\n",
           "T", "Step", "Time", "Phase/Policy", "Algo",
           "Ratio", "Comp(MB)", "GBps", "Chnk", "SGD", "Expl");
    printf("--- ----- ----- ---------------- ---------- ------- --------- --------- ----- ---- -----\n");

    for (int ts = 0; ts < timesteps; ts++) {
        /* Evolve physics */
        double evolve_t0 = now_ms();
        for (int si = 0; si < sim_interval; si++, global_step++) {
            float t = (float)global_step / (float)total_steps;
            evolve_all_fields<<<grid, 256>>>(d_data, nx, ny, (int)n_cells, t, global_step);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        double evolve_ms = now_ms() - evolve_t0;
        float sim_time = (float)global_step / (float)total_steps;

        if (ts % 10 == 0)
            fprintf(stderr, "  [t=%d/%d] sim_step=%d sim_time=%.3f evolve=%.0fms\n",
                    ts, timesteps, global_step, sim_time, evolve_ms);

        /* ---- Compress through all phases ---- */
        for (int pi = 0; pi < n_phases; pi++) {
            bool is_nn = (phases[pi].nn_base_idx >= 0);
            if (is_nn && !has_nn) continue;

            int pol_start = 0, pol_end = 1;
            if (is_nn) { pol_start = 0; pol_end = n_policies; }

            for (int pol = pol_start; pol < pol_end; pol++) {
                const char* policy_label = is_nn ? all_policies[pol].name : "fixed";

                /* NN setup: restore weights, set policy, configure learning */
                if (is_nn) {
                    int snap_idx = pol * NN_BASE_PHASES + phases[pi].nn_base_idx;
                    gpucompress_nn_restore_snapshot_device(nn_snaps[snap_idx]);
                    gpucompress_set_ranking_weights(
                        all_policies[pol].w0, all_policies[pol].w1, all_policies[pol].w2);
                    if (phases[pi].sgd) {
                        gpucompress_enable_online_learning();
                        gpucompress_set_reinforcement(1, lr, mape_thresh, 0.0f);
                    } else {
                        gpucompress_disable_online_learning();
                    }
                    if (phases[pi].explore) {
                        gpucompress_set_exploration(1);
                        gpucompress_set_exploration_k(explore_k);
                        gpucompress_set_exploration_threshold(explore_thresh);
                    } else {
                        gpucompress_set_exploration(0);
                    }
                    gpucompress_flush_manager_cache();
                }

                /* Reset per-chunk diagnostics */
                gpucompress_reset_chunk_history();

                /* Compress 192 MB in chunks (like HDF5 VOL would).
                 * Each chunk gets its own NN inference + compress call,
                 * giving per-chunk algorithm diversity and SGD signal. */
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = phases[pi].algo;
                cfg.error_bound = error_bound;
                cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                if (error_bound > 0.0)
                    cfg.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE;

                size_t total_comp = 0;
                int    chunk_count = 0;
                int    sgd_fires = 0, explorations = 0;
                double mape_sum = 0.0;
                int    mape_count = 0;
                double max_err = 0.0;
                bool   any_fail = false;

                double t0 = now_ms();

                size_t offset = 0;
                while (offset < total_bytes) {
                    size_t this_chunk = chunk_bytes;
                    if (offset + this_chunk > total_bytes)
                        this_chunk = total_bytes - offset;

                    const uint8_t* d_chunk_in  = (const uint8_t*)d_data + offset;
                    uint8_t*       d_chunk_out = (uint8_t*)d_comp + offset;  /* safe: comp buffer >= data */
                    uint8_t*       d_chunk_dec = (uint8_t*)d_decomp + offset;

                    size_t max_chunk_comp = gpucompress_max_compressed_size(this_chunk);
                    /* Use d_comp as scratch — offset by total_bytes to avoid overlap */
                    uint8_t* d_scratch = (uint8_t*)d_comp;
                    size_t comp_chunk_size = max_chunk_comp;

                    gpucompress_stats_t stats;
                    memset(&stats, 0, sizeof(stats));
                    gerr = gpucompress_compress_gpu(
                        d_chunk_in, this_chunk,
                        d_scratch, &comp_chunk_size,
                        &cfg, &stats, NULL);

                    if (gerr != GPUCOMPRESS_SUCCESS) { any_fail = true; break; }

                    /* Decompress this chunk for verification */
                    size_t dec_chunk = this_chunk;
                    gpucompress_decompress_gpu(
                        d_scratch, comp_chunk_size,
                        d_chunk_dec, &dec_chunk, NULL);

                    /* Per-chunk MAPE from stats */
                    if (stats.predicted_ratio > 0 && stats.compression_ratio > 0) {
                        double chunk_mape = fabs(stats.predicted_ratio - stats.compression_ratio)
                                          / stats.compression_ratio * 100.0;
                        mape_sum += chunk_mape;
                        mape_count++;
                    }
                    if (stats.sgd_fired) sgd_fires++;
                    if (stats.exploration_triggered) explorations++;

                    total_comp += comp_chunk_size;
                    chunk_count++;
                    offset += this_chunk;
                }

                double write_ms = now_ms() - t0;

                if (any_fail) {
                    fprintf(stderr, "  FAIL: %s/%s ts=%d\n",
                            phases[pi].name, policy_label, ts);
                    continue;
                }

                size_t comp_size = total_comp;
                double avg_mape = (mape_count > 0) ? mape_sum / mape_count : 0.0;

                /* Spot-check verification: first + last 64K floats */
                {
                    size_t check = 65536;
                    if (check > n_floats) check = n_floats;
                    std::vector<float> h_orig(check), h_dec(check);
                    CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, check*4, cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(h_dec.data(), d_decomp, check*4, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < check; i++) {
                        double d = fabs((double)h_orig[i] - (double)h_dec[i]);
                        if (d > max_err) max_err = d;
                    }
                    if (n_floats > check) {
                        size_t tail_off = (n_floats - check);
                        CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data + tail_off, check*4, cudaMemcpyDeviceToHost));
                        CHECK_CUDA(cudaMemcpy(h_dec.data(), (float*)d_decomp + tail_off, check*4, cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < check; i++) {
                            double d = fabs((double)h_orig[i] - (double)h_dec[i]);
                            if (d > max_err) max_err = d;
                        }
                    }
                }

                /* Also collect from chunk history (if available) */
                int n_hist = gpucompress_get_chunk_history_count();
                if (n_hist > chunk_count) {
                    /* History may have extra entries from exploration */
                    for (int ci = 0; ci < n_hist; ci++) {
                        gpucompress_chunk_diag_t diag;
                        if (gpucompress_get_chunk_diag(ci, &diag) == 0) {
                            /* Already counted sgd/explore from stats above */
                        }
                    }
                }

                /* Save NN weights after SGD */
                if (is_nn && phases[pi].sgd) {
                    int snap_idx = pol * NN_BASE_PHASES + phases[pi].nn_base_idx;
                    gpucompress_nn_save_snapshot_device(nn_snaps[snap_idx]);
                }

                double ratio = (double)total_bytes / comp_size;
                double gbps = (total_bytes / (1024.0*1024.0*1024.0)) / (write_ms / 1000.0);
                double orig_mb = total_bytes / (1024.0*1024.0);
                double comp_mb = comp_size / (1024.0*1024.0);

                /* Build display name */
                char disp[48];
                if (is_nn && n_policies > 1)
                    snprintf(disp, sizeof(disp), "%s/%s", phases[pi].name, policy_label);
                else
                    snprintf(disp, sizeof(disp), "%s", phases[pi].name);

                printf("%-3d %-5d %-5.3f %-16s %-10s %7.2f %9.1f %9.2f %5d %4d %5d\n",
                       ts, global_step, sim_time,
                       disp, is_nn ? "auto" : phases[pi].name,
                       ratio, comp_mb, gbps,
                       chunk_count, sgd_fires, explorations);

                fprintf(csv, "%d,%d,%.6f,%s,%s,%.2f,%.2f,%.4f,%.3f,%.4f,%d,%d,%d,%.2f,%.8f,%s\n",
                        ts, global_step, sim_time,
                        phases[pi].name, policy_label,
                        orig_mb, comp_mb, ratio,
                        write_ms, gbps,
                        chunk_count, sgd_fires, explorations,
                        avg_mape, max_err,
                        is_nn ? "auto" : phases[pi].name);
            }
        }
    }

    fclose(csv);

    /* Cleanup */
    for (int i = 0; i < total_snaps; i++)
        if (nn_snaps[i]) cudaFree(nn_snaps[i]);
    cudaFree(d_data);
    cudaFree(d_comp);
    cudaFree(d_decomp);
    gpucompress_cleanup();

    printf("\n=== Benchmark complete ===\n");
    printf("CSV: warpx_hyperparameter_study.csv\n");
    printf("%d timesteps x %d+ phases x ~%d chunks/write\n", timesteps, n_phases, n_chunks);
    return 0;
}
