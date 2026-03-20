/**
 * @file nn-diagnostic-benchmark.cu
 * @brief Focused NN diagnostic: verifies per-chunk stats differ and NN
 *        selects different configs.  Also brute-forces all configs for
 *        ground-truth comparison.
 *
 * Dataset: 10MB total, 10 chunks × 1MB each.
 *   dims = {640, 64, 64}, chunks = {64, 64, 64}
 *   Contiguous layout: chunk k covers dim0 ∈ [k*64, (k+1)*64)
 *
 * Phases:
 *   1. Generate 10 wildly different data patterns on GPU
 *   2. Write through HDF5 VOL with ALGO_AUTO (NN-inference only, no SGD)
 *   3. Read chunk diagnostics and print NN decisions
 *   4. Brute-force all 16 lossless configs per chunk (8 algo × 2 shuffle)
 *   5. Print comparison: NN pick vs brute-force optimal
 *
 * Usage:
 *   GPUCOMPRESS_DEBUG_NN=1 ./build/nn_diagnostic_benchmark model.nnwt
 *
 * Set GPUCOMPRESS_DEBUG_NN=1 to enable verbose per-chunk debug output
 * from the library itself (stats values, NN action, top-5 ranking).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <cuda_runtime.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Constants
 * ============================================================ */

#define N_CHUNKS    20
#define DIM0_TOTAL  (N_CHUNKS * 64)   /* 20 chunks × 64 = 1280 */
#define DIM1        64
#define DIM2        64
#define CHUNK_DIM0  64
#define CHUNK_ELEMS (CHUNK_DIM0 * DIM1 * DIM2)  /* 262144 floats = 1MB */
#define TOTAL_ELEMS (DIM0_TOTAL * DIM1 * DIM2)  /* 50MB */

#define TMP_FILE "/tmp/nn_diag_test.h5"

/* HDF5 filter constants */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static const char* ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static const char* PATTERN_NAMES[N_CHUNKS] = {
    /* Group A — Entropy extremes */
    "all_zeros",           /*  0 */
    "constant_pi",         /*  1 */
    "small_int_0_7",       /*  2 */
    "random_uniform_01",   /*  3 */
    /* Group B — MAD extremes */
    "tight_gauss_1K",      /*  4 */
    "wide_uniform_1e8",    /*  5 */
    "bimodal_0_1M",        /*  6 */
    "sparse_1pct_1e7",     /*  7 */
    /* Group C — 2nd derivative extremes */
    "linear_ramp_1M",      /*  8 */
    "smooth_parabola",     /*  9 */
    "alternating_1e5",     /* 10 */
    "hfreq_sine_500",      /* 11 */
    /* Group D — Scientific mimics */
    "cfd_pressure",        /* 12 */
    "turbulence",          /* 13 */
    "md_velocity",         /* 14 */
    "climate_sst",         /* 15 */
    /* Group E — Compressibility spectrum */
    "repeated_blk16",      /* 16 */
    "monotonic_int",       /* 17 */
    "exp_decay",           /* 18 */
    "log_normal",          /* 19 */
};

/* ============================================================
 * GPU hash for deterministic randomness
 * ============================================================ */

__device__ float gpu_hash_float(unsigned long long seed, size_t idx) {
    unsigned long long h = (seed ^ (idx * 6364136223846793005ULL)) + 1442695040888963407ULL;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    return (float)(h & 0xFFFFFF) / (float)0xFFFFFF;
}

__device__ float gpu_gaussian(unsigned long long seed, size_t idx) {
    float u1 = gpu_hash_float(seed, idx * 2 + 0);
    float u2 = gpu_hash_float(seed, idx * 2 + 1);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ============================================================
 * Kernel: generate 20 very different data patterns
 *
 * Chunk k covers dim0 ∈ [k*64, (k+1)*64).  In row-major order,
 * element at (x,y,z) is at index x*DIM1*DIM2 + y*DIM2 + z.
 * So chunk k = elements in [k*CHUNK_ELEMS, (k+1)*CHUNK_ELEMS).
 * ============================================================ */

__global__ void generate_diagnostic_patterns(float* data, unsigned long long seed)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_ELEMS) return;

    int chunk_id = (int)(idx / CHUNK_ELEMS);
    size_t local = idx - (size_t)chunk_id * CHUNK_ELEMS;
    float t = (float)local / (float)CHUNK_ELEMS;  /* [0, 1) */

    float rnd   = gpu_hash_float(seed, idx);
    float gauss = gpu_gaussian(seed + 104729ULL, idx);

    float val = 0.0f;

    switch (chunk_id) {

    /* ---- Group A: Entropy extremes ---- */
    case 0: /* all zeros — ent≈0, MAD=0, deriv=0 */
        val = 0.0f;
        break;
    case 1: /* constant pi — ent≈2, MAD=0, deriv=0 */
        val = 3.14159265f;
        break;
    case 2: /* small integers 0-7 — ent≈low, MAD=low */
        val = floorf(rnd * 8.0f);
        break;
    case 3: /* uniform random [0,1] — ent≈7.5, MAD≈0.33, deriv=high */
        val = rnd;
        break;

    /* ---- Group B: MAD extremes ---- */
    case 4: /* tight Gaussian(1000, 0.001) — MAD≈0 */
        val = 1000.0f + gauss * 0.001f;
        break;
    case 5: /* wide uniform [0, 1e8] — MAD=very high */
        val = rnd * 1.0e8f;
        break;
    case 6: /* bimodal 0/1e6 — MAD=extreme */
        val = (rnd < 0.5f) ? 0.0f : 1.0e6f;
        break;
    case 7: /* sparse 1% spikes at 1e7 — MAD=low, range=huge */
        val = (rnd < 0.01f) ? 1.0e7f : 0.0f;
        break;

    /* ---- Group C: 2nd derivative extremes ---- */
    case 8: /* linear ramp 0→1e6 — deriv2=0 */
        val = t * 1.0e6f;
        break;
    case 9: /* smooth parabola — deriv2=constant */
        val = (t - 0.5f) * (t - 0.5f) * 4.0e6f;
        break;
    case 10: /* alternating ±1e5 — deriv2=maximum */
        val = (((int)local) & 1) ? 1.0e5f : -1.0e5f;
        break;
    case 11: /* high-freq sine: 500 cycles — deriv=very high */
        val = sinf(t * 2.0f * 3.14159265f * 500.0f) * 5000.0f;
        break;

    /* ---- Group D: Scientific workload mimics ---- */
    case 12: /* CFD pressure: smooth + boundary layer spikes */
        val = sinf(t * 3.14159265f) * 1.0e5f
            + ((t < 0.02f || t > 0.98f) ? gauss * 1.0e4f : 0.0f);
        break;
    case 13: /* turbulence: multi-freq sum, high entropy */
        val = sinf(t * 7.0f) * 100.0f + sinf(t * 31.0f) * 50.0f
            + sinf(t * 127.0f) * 25.0f + gauss * 10.0f;
        break;
    case 14: /* MD velocity: Maxwell-Boltzmann distribution */
        val = gauss * 500.0f;
        break;
    case 15: /* climate SST: smooth gradient + tiny noise */
        val = 273.0f + t * 30.0f + gauss * 0.1f;
        break;

    /* ---- Group E: Compressibility spectrum ---- */
    case 16: /* repeated 16-float block — trivial for LZ family */
    {
        int block_pos = ((int)local) % 16;
        float block_vals[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        val = block_vals[block_pos];
        break;
    }
    case 17: /* monotonic integers — great for delta/cascaded */
        val = (float)local;
        break;
    case 18: /* exponential decay — high dynamic range */
        val = expf(-5.0f * t) * 1.0e5f;
        break;
    case 19: /* log-normal — heavy tail, wide range */
    {
        float g = gauss * 2.0f;
        val = expf(g);
        break;
    }
    }

    data[idx] = val;
}

/* ============================================================
 * Helpers
 * ============================================================ */

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static void action_to_str(int action, char* buf, size_t bufsz) {
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.nnwt> [--out-dir DIR]\n", argv[0]);
        return 1;
    }
    const char* weights_path = argv[1];
    const char* out_dir = "benchmarks/diagnostic/results";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }

    /* Create output directory */
    {
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "%s", out_dir);
        for (char* p = tmp + 1; *p; p++) {
            if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
        }
        mkdir(tmp, 0755);
    }

    printf("=== NN Diagnostic Benchmark ===\n");
    printf("Dataset: %d chunks × %d floats (%.1f MB each, %.1f MB total)\n",
           N_CHUNKS, CHUNK_ELEMS, CHUNK_ELEMS * 4.0 / (1024*1024),
           TOTAL_ELEMS * 4.0 / (1024*1024));
    printf("Dims: {%d, %d, %d}, Chunks: {%d, %d, %d}\n",
           DIM0_TOTAL, DIM1, DIM2, CHUNK_DIM0, DIM1, DIM2);
    printf("Mode: lossless (error_bound=0)\n\n");

    /* ---- Init library ---- */
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %d\n", err);
        return 1;
    }

    /* Disable online learning — pure inference only */
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* ---- Allocate and generate data on GPU ---- */
    float* d_data = nullptr;
    size_t data_bytes = (size_t)TOTAL_ELEMS * sizeof(float);
    cudaError_t cerr = cudaMalloc(&d_data, data_bytes);
    assert(cerr == cudaSuccess && "cudaMalloc for data failed");

    printf("Generating %d data patterns on GPU...\n", N_CHUNKS);
    int threads = 256;
    int blocks = (TOTAL_ELEMS + threads - 1) / threads;
    generate_diagnostic_patterns<<<blocks, threads>>>(d_data, 42ULL);
    cerr = cudaDeviceSynchronize();
    assert(cerr == cudaSuccess && "Pattern generation kernel failed");

    /* ---- Verify patterns differ: compute host-side stats per chunk ---- */
    printf("\n--- Phase 1: Verify pattern data differs ---\n");
    printf("%-4s %-20s %12s %12s %12s %12s %12s\n",
           "Chk", "Pattern", "Mean", "Min", "Max", "Range", "StdApprox");

    float* h_data = (float*)malloc(data_bytes);
    assert(h_data);
    cudaMemcpy(h_data, d_data, data_bytes, cudaMemcpyDeviceToHost);

    for (int c = 0; c < N_CHUNKS; c++) {
        float* chunk = h_data + (size_t)c * CHUNK_ELEMS;
        double sum = 0, vmin = chunk[0], vmax = chunk[0];
        for (int i = 0; i < CHUNK_ELEMS; i++) {
            float v = chunk[i];
            sum += v;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        double mean = sum / CHUNK_ELEMS;
        double var_sum = 0;
        for (int i = 0; i < CHUNK_ELEMS; i++) {
            double d = chunk[i] - mean;
            var_sum += d * d;
        }
        double std_dev = sqrt(var_sum / CHUNK_ELEMS);
        printf("%-4d %-20s %12.4e %12.4e %12.4e %12.4e %12.4e\n",
               c, PATTERN_NAMES[c], mean, vmin, vmax, vmax - vmin, std_dev);
    }
    free(h_data);

    /* ---- Phase 2: Write through HDF5 VOL (NN-only, no learning) ---- */
    printf("\n--- Phase 2: HDF5 VOL write with ALGO_AUTO (NN-inference only) ---\n");

    gpucompress_reset_chunk_history();

    /* Create HDF5 file with VOL connector */
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    unlink(TMP_FILE);
    hid_t fid = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    assert(fid >= 0 && "H5Fcreate failed");

    /* Dataset creation property list with ALGO_AUTO, error_bound=0 (lossless) */
    hsize_t dims[3]  = { DIM0_TOTAL, DIM1, DIM2 };
    hsize_t cdims[3] = { CHUNK_DIM0, DIM1, DIM2 };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]); /* error_bound = 0 (lossless) */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t space = H5Screate_simple(3, dims, NULL);
    hid_t dset  = H5Dcreate2(fid, "diag_data", H5T_NATIVE_FLOAT, space,
                              H5P_DEFAULT, dcpl, H5P_DEFAULT);
    assert(dset >= 0 && "H5Dcreate2 failed");

    printf("Writing dataset through VOL (GPU pointer passed directly)...\n");
    herr_t herr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    assert(herr >= 0 && "H5Dwrite failed");

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);
    H5Pclose(fapl);

    /* ---- Phase 3: Read chunk diagnostics from library ---- */
    printf("\n--- Phase 3: Per-chunk NN decisions (from VOL write) ---\n");
    int n_diag = gpucompress_get_chunk_history_count();
    printf("Chunks recorded: %d (expected %d)\n", n_diag, N_CHUNKS);

    printf("\n%-4s %-20s %-8s %-7s %-7s %-7s %-7s %-7s %-20s %-8s\n",
           "Chk", "Pattern", "Entropy", "MAD", "Deriv", "EB_enc", "DS_enc",
           "Ratio", "Config", "Action");
    printf("%-4s %-20s %-8s %-7s %-7s %-7s %-7s %-7s %-20s %-8s\n",
           "---", "--------------------", "-------", "------", "------",
           "------", "------", "------", "--------------------", "------");

    gpucompress_chunk_diag_t diags[N_CHUNKS];
    int nn_actions[N_CHUNKS];

    for (int c = 0; c < n_diag && c < N_CHUNKS; c++) {
        gpucompress_chunk_diag_t d;
        gpucompress_get_chunk_diag(c, &d);
        diags[c] = d;
        nn_actions[c] = d.nn_original_action;

        char cfg_str[64];
        action_to_str(d.nn_original_action, cfg_str, sizeof(cfg_str));

        printf("%-4d %-20s %-8.3f %-7.4f %-7.4f %-7.2f %-7.1f %-7.2f %-20s %-8d\n",
               c, PATTERN_NAMES[c],
               d.feat_entropy, d.feat_mad, d.feat_deriv,
               d.feat_eb_enc, d.feat_ds_enc,
               d.predicted_ratio,
               cfg_str, d.nn_original_action);
    }

    /* ---- Assert that stats differ across chunks ---- */
    printf("\n--- Assertions: verify stats actually differ ---\n");
    int n_unique_entropy = 0, n_unique_mad = 0, n_unique_deriv = 0;
    int n_unique_actions = 0;

    /* Count unique values (simple O(n^2) for 10 items) */
    float ent_vals[N_CHUNKS], mad_vals[N_CHUNKS], der_vals[N_CHUNKS];
    for (int c = 0; c < n_diag && c < N_CHUNKS; c++) {
        ent_vals[c] = diags[c].feat_entropy;
        mad_vals[c] = diags[c].feat_mad;
        der_vals[c] = diags[c].feat_deriv;
    }

    auto count_unique = [](float* arr, int n) -> int {
        int unique = 0;
        for (int i = 0; i < n; i++) {
            bool dup = false;
            for (int j = 0; j < i; j++) {
                if (fabsf(arr[i] - arr[j]) < 1e-6f) { dup = true; break; }
            }
            if (!dup) unique++;
        }
        return unique;
    };

    n_unique_entropy = count_unique(ent_vals, n_diag < N_CHUNKS ? n_diag : N_CHUNKS);
    n_unique_mad     = count_unique(mad_vals, n_diag < N_CHUNKS ? n_diag : N_CHUNKS);
    n_unique_deriv   = count_unique(der_vals, n_diag < N_CHUNKS ? n_diag : N_CHUNKS);

    {
        int act_arr[N_CHUNKS];
        for (int c = 0; c < n_diag && c < N_CHUNKS; c++) act_arr[c] = nn_actions[c];
        int unique = 0;
        for (int i = 0; i < n_diag && i < N_CHUNKS; i++) {
            bool dup = false;
            for (int j = 0; j < i; j++) {
                if (act_arr[i] == act_arr[j]) { dup = true; break; }
            }
            if (!dup) unique++;
        }
        n_unique_actions = unique;
    }

    printf("  Unique entropy values: %d / %d", n_unique_entropy, N_CHUNKS);
    if (n_unique_entropy < 5) printf("  *** WARNING: too few unique entropy values!");
    printf("\n");

    printf("  Unique MAD values:     %d / %d", n_unique_mad, N_CHUNKS);
    if (n_unique_mad < 5) printf("  *** WARNING: too few unique MAD values!");
    printf("\n");

    printf("  Unique deriv values:   %d / %d", n_unique_deriv, N_CHUNKS);
    if (n_unique_deriv < 5) printf("  *** WARNING: too few unique deriv values!");
    printf("\n");

    printf("  Unique NN actions:     %d / %d", n_unique_actions, N_CHUNKS);
    if (n_unique_actions == 1) printf("  *** BUG: NN picks same config for ALL chunks!");
    else if (n_unique_actions < 3) printf("  ** NOTE: NN picks few distinct configs");
    printf("\n");

    /* Hard assert: stats must differ for these obviously different patterns */
    if (n_unique_entropy < 3) {
        fprintf(stderr, "\n*** ASSERTION FAILED: fewer than 3 unique entropy values "
                "across 10 wildly different patterns! Stats pipeline may be broken.\n");
    }

    /* ---- Phase 4: Brute-force all 16 lossless configs per chunk ---- */
    printf("\n--- Phase 4: Brute-force all configs per chunk ---\n");

    /* For lossless: 8 algos × 2 shuffle × quant=0 = 16 configs */
    struct BFResult {
        int action;
        size_t compressed_size;
        double ratio;
        float comp_time_ms;
    };

    /* Allocate compression output buffer (2× input for safety) */
    size_t chunk_bytes = (size_t)CHUNK_ELEMS * sizeof(float);
    size_t max_out = chunk_bytes * 2 + 1024;
    void* d_comp_out = nullptr;
    cerr = cudaMalloc(&d_comp_out, max_out);
    assert(cerr == cudaSuccess);

    /* Store all results for CSV output */
    BFResult all_bf[N_CHUNKS][16];
    int bf_counts[N_CHUNKS];

    printf("\n%-4s %-20s %-26s %-8s | %-26s %-8s | %s\n",
           "Chk", "Pattern", "NN Pick", "Ratio", "Best Brute-Force", "Ratio", "Match?");
    printf("%-4s %-20s %-26s %-8s | %-26s %-8s | %s\n",
           "---", "--------------------", "--------------------------", "--------",
           "--------------------------", "--------", "------");

    int n_match = 0;
    int n_top3 = 0;

    for (int c = 0; c < N_CHUNKS; c++) {
        const float* d_chunk = d_data + (size_t)c * CHUNK_ELEMS;

        BFResult best = {-1, 0, 0.0, 0.0f};
        int n_results = 0;

        for (int algo = 1; algo <= 8; algo++) {
            for (int shuf = 0; shuf <= 1; shuf++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)algo;
                cfg.error_bound = 0.0;
                cfg.preprocessing = shuf ? GPUCOMPRESS_PREPROC_SHUFFLE_4
                                         : GPUCOMPRESS_PREPROC_NONE;

                size_t out_size = max_out;
                gpucompress_stats_t stats = {};

                gpucompress_error_t ce = gpucompress_compress_gpu(
                    d_chunk, chunk_bytes, d_comp_out, &out_size,
                    &cfg, &stats, nullptr);

                int action = (algo - 1) + (shuf ? 16 : 0); /* quant=0 */

                BFResult r;
                r.action = action;
                if (ce == GPUCOMPRESS_SUCCESS && out_size > 0) {
                    r.compressed_size = out_size;
                    r.ratio = (double)chunk_bytes / (double)out_size;
                    r.comp_time_ms = (float)stats.actual_comp_time_ms;
                } else {
                    r.compressed_size = chunk_bytes;
                    r.ratio = 1.0;
                    r.comp_time_ms = 0.0f;
                }

                all_bf[c][n_results++] = r;

                if (best.action < 0 || r.ratio > best.ratio) {
                    best = r;
                }
            }
        }
        bf_counts[c] = n_results;

        /* Check if NN pick matches brute-force best */
        int nn_act = (c < n_diag) ? nn_actions[c] : -1;
        bool match = (nn_act == best.action);

        /* Check if NN pick is in top-3 of brute-force */
        bool in_top3 = false;
        /* Sort by ratio descending */
        for (int i = 0; i < n_results - 1; i++) {
            for (int j = i + 1; j < n_results; j++) {
                if (all_bf[c][j].ratio > all_bf[c][i].ratio) {
                    BFResult tmp = all_bf[c][i];
                    all_bf[c][i] = all_bf[c][j];
                    all_bf[c][j] = tmp;
                }
            }
        }
        for (int i = 0; i < 3 && i < n_results; i++) {
            if (all_bf[c][i].action == nn_act) { in_top3 = true; break; }
        }

        if (match) n_match++;
        if (in_top3) n_top3++;

        char nn_str[64], bf_str[64];
        action_to_str(nn_act, nn_str, sizeof(nn_str));
        action_to_str(best.action, bf_str, sizeof(bf_str));

        double nn_ratio = (c < n_diag) ? diags[c].predicted_ratio : 0.0;

        printf("%-4d %-20s %-26s %-8.2f | %-26s %-8.2f | %s\n",
               c, PATTERN_NAMES[c],
               nn_str, nn_ratio,
               bf_str, best.ratio,
               match ? "EXACT" : (in_top3 ? "top-3" : "MISS"));

        /* Print all brute-force results for this chunk */
        printf("     Brute-force ranking:\n");
        for (int i = 0; i < n_results && i < 5; i++) {
            char act_str[64];
            action_to_str(all_bf[c][i].action, act_str, sizeof(act_str));
            printf("       #%d: %-20s ratio=%.2f  comp=%.3fms  size=%zuB%s\n",
                   i + 1, act_str, all_bf[c][i].ratio,
                   all_bf[c][i].comp_time_ms, all_bf[c][i].compressed_size,
                   (all_bf[c][i].action == nn_act) ? "  ← NN pick" : "");
        }
        printf("\n");
    }

    /* ---- Summary ---- */
    printf("\n=== SUMMARY ===\n");
    printf("NN exact match with brute-force best:  %d / %d (%.0f%%)\n",
           n_match, N_CHUNKS, 100.0 * n_match / N_CHUNKS);
    printf("NN pick in brute-force top-3:          %d / %d (%.0f%%)\n",
           n_top3, N_CHUNKS, 100.0 * n_top3 / N_CHUNKS);
    printf("Unique NN actions:                     %d / %d\n",
           n_unique_actions, N_CHUNKS);
    printf("Unique entropy values:                 %d / %d\n",
           n_unique_entropy, N_CHUNKS);
    printf("Unique MAD values:                     %d / %d\n",
           n_unique_mad, N_CHUNKS);
    printf("Unique deriv values:                   %d / %d\n\n",
           n_unique_deriv, N_CHUNKS);

    if (n_unique_actions == 1) {
        printf("*** DIAGNOSIS: NN picks identical config for all chunks.\n");
        if (n_unique_entropy < 3)
            printf("    ROOT CAUSE: Stats pipeline produces same entropy for different data.\n"
                   "    → Check stats_kernel.cu: workspace zeroing, histogram computation.\n");
        else if (n_unique_mad < 3)
            printf("    ROOT CAUSE: MAD values too similar across patterns.\n"
                   "    → Check finalizeStatsOnlyKernel normalization.\n");
        else
            printf("    ROOT CAUSE: Stats differ but NN output is insensitive to them.\n"
                   "    → Check NN weights: features 12-14 may have near-zero influence.\n"
                   "    → The trained model may simply prefer one config universally.\n"
                   "    → Try: vary error_bound to check if NN responds to eb_enc feature.\n");
    } else if (n_unique_actions < 3) {
        printf("*** DIAGNOSIS: NN picks very few distinct configs (%d).\n", n_unique_actions);
        printf("    This may be correct if cascaded genuinely dominates for lossless data.\n"
               "    Compare with brute-force: if cascaded IS the best for most chunks,\n"
               "    the NN is working correctly.\n");
    } else {
        printf("OK: NN differentiates across chunks (%d unique configs).\n", n_unique_actions);
    }

    /* ---- Phase 5: Write CSVs for visualizer ---- */
    printf("\n--- Phase 5: Writing CSVs to %s ---\n", out_dir);

    /* CSV 1: nn_chunks.csv — per-chunk NN diagnostics */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/nn_chunks.csv", out_dir);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "chunk,pattern,entropy,mad,deriv,eb_enc,ds_enc,"
                       "nn_action,nn_config,predicted_ratio,actual_ratio\n");
            for (int c = 0; c < n_diag && c < N_CHUNKS; c++) {
                gpucompress_chunk_diag_t& d = diags[c];
                char cfg_str[64];
                action_to_str(d.nn_original_action, cfg_str, sizeof(cfg_str));
                fprintf(f, "%d,%s,%.6f,%.6f,%.6f,%.4f,%.2f,%d,%s,%.4f,%.4f\n",
                        c, PATTERN_NAMES[c],
                        d.feat_entropy, d.feat_mad, d.feat_deriv,
                        d.feat_eb_enc, d.feat_ds_enc,
                        d.nn_original_action, cfg_str,
                        d.predicted_ratio, d.actual_ratio);
            }
            fclose(f);
            printf("  Wrote %s\n", path);
        }
    }

    /* CSV 2: brute_force.csv — all brute-force results per chunk (reuse Phase 4) */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/brute_force.csv", out_dir);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "chunk,pattern,action,config,compressed_size,ratio,comp_time_ms,"
                       "is_nn_pick,bf_rank\n");

            for (int c = 0; c < N_CHUNKS; c++) {
                int nn_act = (c < n_diag) ? nn_actions[c] : -1;
                /* all_bf[c] is already sorted by ratio descending from Phase 4 */
                for (int i = 0; i < bf_counts[c]; i++) {
                    char act_str[64];
                    action_to_str(all_bf[c][i].action, act_str, sizeof(act_str));
                    fprintf(f, "%d,%s,%d,%s,%zu,%.4f,%.4f,%d,%d\n",
                            c, PATTERN_NAMES[c],
                            all_bf[c][i].action, act_str,
                            all_bf[c][i].compressed_size,
                            all_bf[c][i].ratio,
                            all_bf[c][i].comp_time_ms,
                            (all_bf[c][i].action == nn_act) ? 1 : 0,
                            i + 1);
                }
            }
            fclose(f);
            printf("  Wrote %s\n", path);
        }
    }

    printf("\nTo visualize: python3 benchmarks/diagnostic/visualize_diagnostic.py "
           "--dir %s\n", out_dir);

    /* ---- Cleanup ---- */
    cudaFree(d_comp_out);
    cudaFree(d_data);
    unlink(TMP_FILE);
    gpucompress_cleanup();

    return 0;
}
