/**
 * @file test_nn_algo_convergence.cu
 * @brief Test: does the NN converge to different preprocessing configs?
 *
 * Layout (512 chunks, 1 MB each = 512 MB total):
 *   Chunks  0-255:   smooth ramp + noise  → zstd+shuf wins (byte shuffle helps)
 *   Chunks  256-511: skewed sparse floats → zstd (no shuf) wins (shuffle hurts)
 *
 * Zstd dominates all nvcomp algorithms on ratio, so the meaningful
 * adaptation test is: can the NN learn when shuffle helps vs hurts?
 *
 * Phases:
 *   1. Exhaustive search — confirms best config per chunk (ground truth)
 *   2. NN-RL (ALGO_AUTO + SGD) — watch the NN adapt chunk-by-chunk
 *
 * The key question: does the NN switch preprocessing (shuffle on/off)
 * when it crosses the data boundary at chunk 256?
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>

#include <hdf5.h>
#include <cuda_runtime.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define CHUNK_FLOATS  (256 * 1024)   /* 1 MB per chunk */
#define NUM_CHUNKS    512
#define HALF          (NUM_CHUNKS / 2)
#define TOTAL_FLOATS  ((size_t)CHUNK_FLOATS * NUM_CHUNKS)
#define DATA_BYTES    (TOTAL_FLOATS * sizeof(float))
#define HDF5_FILE     "/tmp/test_nn_algo_convergence.h5"

static const char* ALGO_NAMES[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* ── GPU kernels ─────────────────────────────────────────────── */

/* First half: smooth ramp with high-frequency ripple (zstd+shuf wins ~3.4x vs zstd 1.3x) */
__global__ void fill_shuf_region(float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = (float)idx / (float)n;
    out[idx] = x * 100.0f + 0.5f * sinf(x * 200.0f) + 0.1f * cosf(x * 37.0f);
}

/* Second half: sparse data with 90% zeros (zstd 7.3x wins, zstd+shuf 4.7x — shuffle hurts) */
__global__ void fill_noshuf_region(float* out, size_t offset, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int s = (unsigned int)((offset + idx) * 2654435761u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    if ((s % 10) != 0) {
        out[offset + idx] = 0.0f;
    } else {
        /* sparse non-zero: random float in [0,1) */
        unsigned int bits = 0x3F800000u | (s & 0x007FFFFFu);
        float r; memcpy(&r, &bits, sizeof(float));
        out[offset + idx] = r - 1.0f;
    }
}

/* Bitwise compare */
__global__ void compare_kernel(const float* a, const float* b, size_t n,
                                unsigned long long* mismatches)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int va, vb;
    memcpy(&va, &a[idx], sizeof(unsigned int));
    memcpy(&vb, &b[idx], sizeof(unsigned int));
    if (va != vb) atomicAdd(mismatches, 1ULL);
}

/* ── VOL helpers ─────────────────────────────────────────────── */

static hid_t make_vol_fapl(void) {
    hid_t vol_id = H5VL_gpucompress_register();
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    H5VLclose(vol_id);
    return fapl;
}

/* ── Exhaustive search (per-chunk best algo) ─────────────────── */

static void run_exhaustive(const float* d_data)
{
    size_t chunk_bytes = CHUNK_FLOATS * sizeof(float);
    size_t out_buf_size = chunk_bytes + 65536;

    void* d_comp = NULL;
    CUDA_CHECK(cudaMalloc(&d_comp, out_buf_size));

    printf("  chunk | best algo            | ratio    | 2nd-best algo        | 2nd ratio | gap\n");
    printf("  ------+----------------------+----------+----------------------+-----------+--------\n");

    /* With many chunks, only sample: first 4, last 4 of each half, and every 32nd */
    auto should_print_exh = [](int c) -> bool {
        if (c < 4 || (c >= HALF - 4 && c < HALF) ||
            (c >= HALF && c < HALF + 4) || c >= NUM_CHUNKS - 4) return true;
        if (c % 32 == 0) return true;
        return false;
    };

    for (int c = 0; c < NUM_CHUNKS; c++) {
        const float* d_chunk = d_data + (size_t)c * CHUNK_FLOATS;

        double best_ratio = 0, second_ratio = 0;
        int best_algo = 1, second_algo = 1;
        int best_shuf = 0, second_shuf = 0;

        for (int algo = 1; algo <= 8; algo++) {
            for (int shuf = 0; shuf <= 1; shuf++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)algo;
                cfg.preprocessing = shuf ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0;
                cfg.error_bound = 0.0;

                size_t out_size = out_buf_size;
                gpucompress_stats_t stats;
                gpucompress_error_t err = gpucompress_compress_gpu(
                    d_chunk, chunk_bytes, d_comp, &out_size, &cfg, &stats, NULL);

                double ratio = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0;
                if (ratio > best_ratio) {
                    second_ratio = best_ratio;
                    second_algo  = best_algo;
                    second_shuf  = best_shuf;
                    best_ratio   = ratio;
                    best_algo    = algo;
                    best_shuf    = shuf;
                } else if (ratio > second_ratio) {
                    second_ratio = ratio;
                    second_algo  = algo;
                    second_shuf  = shuf;
                }
            }
        }

        char best_str[32], second_str[32];
        snprintf(best_str, sizeof(best_str), "%s%s",
                 ALGO_NAMES[best_algo], best_shuf ? "+shuf" : "");
        snprintf(second_str, sizeof(second_str), "%s%s",
                 ALGO_NAMES[second_algo], second_shuf ? "+shuf" : "");
        double gap = best_ratio - second_ratio;

        const char* region = (c < HALF) ? "shuf-region" : "noshuf-region";
        if (should_print_exh(c)) {
            printf("  %5d | %-20s | %7.2fx | %-20s | %8.2fx | %+.2fx  (%s)\n",
                   c, best_str, best_ratio, second_str, second_ratio, gap, region);
        } else if (c == 4 || c == HALF + 4) {
            printf("  %5s |                      |          |                      |           |         ...\n", "");
        }
    }

    cudaFree(d_comp);
}

/* ── NN-RL pass via VOL ─────────────────────────────────────── */

static int run_nn_rl(float* d_data, float* d_readback)
{
    gpucompress_reset_chunk_history();
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.3f, 0.20f, 0.20f);
    gpucompress_set_exploration(0);

    /* Write */
    hid_t fapl = make_vol_fapl();
    hid_t fid  = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hsize_t dims[1]  = { TOTAL_FLOATS };
    hsize_t chunk[1] = { CHUNK_FLOATS };
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = GPUCOMPRESS_ALGO_AUTO;
    cd[1] = 0;
    cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                            space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_data);
    H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);

    if (wr < 0) { fprintf(stderr, "H5Dwrite failed\n"); return 1; }

    /* Print per-chunk diagnostics */
    int n_hist = gpucompress_get_chunk_history_count();
    printf("  chunk | region      | NN action            | ratio  | pred   | MAPE    | SGD\n");
    printf("  ------+-------------+----------------------+--------+--------+---------+----\n");

    int shuf_count = 0, noshuf_count = 0;
    int first_half_noshuf = 0, second_half_shuf = 0;
    double mape_sum = 0; int mape_n = 0;

    /* Print: first 10, transition region (HALF-10..HALF+20), last 10, and every 32nd */
    auto should_print_nn = [](int i) -> bool {
        if (i < 10 || i >= NUM_CHUNKS - 10) return true;
        if (i >= HALF - 10 && i < HALF + 20) return true;
        if (i % 32 == 0) return true;
        return false;
    };
    int last_printed = -2;

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        gpucompress_get_chunk_diag(i, &d);

        int algo_id = (d.nn_action % 8) + 1;
        int shuf    = (d.nn_action / 16) % 2;
        const char* aname = (algo_id >= 1 && algo_id <= 8) ? ALGO_NAMES[algo_id] : "???";
        char action_str[32];
        snprintf(action_str, sizeof(action_str), "%s%s", aname, shuf ? "+shuf" : "");

        double mape = (d.actual_ratio > 0)
            ? fabs((double)d.predicted_ratio - (double)d.actual_ratio)
              / (double)d.actual_ratio * 100.0
            : 0.0;
        mape_sum += mape; mape_n++;

        const char* region = (i < HALF) ? "shuf-region  " : "noshuf-region";
        const char* correct;
        if (i < HALF)
            correct = shuf ? "OK" : "WRONG(need shuf)";
        else
            correct = shuf ? "WRONG(no shuf)" : "OK";

        if (should_print_nn(i)) {
            if (last_printed < i - 1)
                printf("        |             |          ...         |        |        |         |\n");
            printf("  %5d | %s | %-20s | %5.2fx | %5.2fx | %6.1f%% | %s | %s\n",
                   i, region, action_str,
                   (double)d.actual_ratio, (double)d.predicted_ratio,
                   mape, d.sgd_fired ? "yes" : "  -", correct);
            last_printed = i;
        }

        if (shuf) shuf_count++; else noshuf_count++;
        if (i < HALF && !shuf) first_half_noshuf++;
        if (i >= HALF && shuf) second_half_shuf++;
    }

    printf("\n  Preprocessing distribution:\n");
    printf("    With shuffle:    %d/%d chunks\n", shuf_count, n_hist);
    printf("    Without shuffle: %d/%d chunks\n", noshuf_count, n_hist);
    printf("    Running MAPE:    %.1f%%\n", mape_sum / mape_n);
    printf("\n  Cross-check (shuffle decision correctness):\n");
    printf("    First half  (shuf-region)   picked NO-shuf: %d/%d  %s\n",
           first_half_noshuf, HALF,
           first_half_noshuf == 0 ? "(all correct)" : "(some wrong)");
    printf("    Second half (noshuf-region) picked shuf:    %d/%d  %s\n",
           second_half_shuf, HALF,
           second_half_shuf == 0 ? "(all correct)" : "(some wrong)");

    /* Read back and verify */
    fapl = make_vol_fapl();
    fid  = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(fid, "data", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    H5Dclose(dset); H5Fclose(fid);

    /* Per-chunk mismatch counts — critical for diagnosing which actions/
     * preproc combos violate the lossless contract. One kernel launch per
     * chunk into a per-chunk counter; total is the sum. */
    unsigned long long* d_mm_chunks;
    CUDA_CHECK(cudaMalloc(&d_mm_chunks, (size_t)NUM_CHUNKS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_mm_chunks, 0, (size_t)NUM_CHUNKS * sizeof(unsigned long long)));
    int threads = 256;
    int blocks_per_chunk = ((int)CHUNK_FLOATS + threads - 1) / threads;
    for (int c = 0; c < NUM_CHUNKS; c++) {
        const float* d_chunk_orig = d_data     + (size_t)c * CHUNK_FLOATS;
        const float* d_chunk_read = d_readback + (size_t)c * CHUNK_FLOATS;
        compare_kernel<<<blocks_per_chunk, threads>>>(
            d_chunk_orig, d_chunk_read, CHUNK_FLOATS, &d_mm_chunks[c]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned long long> mm_chunks(NUM_CHUNKS, 0);
    cudaMemcpy(mm_chunks.data(), d_mm_chunks,
               (size_t)NUM_CHUNKS * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaFree(d_mm_chunks);

    unsigned long long mm = 0;
    int n_bad_chunks = 0;
    for (int c = 0; c < NUM_CHUNKS; c++) {
        mm += mm_chunks[c];
        if (mm_chunks[c] > 0) n_bad_chunks++;
    }

    printf("\n  Lossless verification: %llu mismatches across %d/%zu chunks → %s\n",
           mm, n_bad_chunks, (size_t)NUM_CHUNKS, mm == 0 ? "PASS" : "FAIL");

    /* For every failing chunk, print its NN action + decoded algo/quant/shuf.
     * This lets us tell apart: (a) specific algo broken on this data,
     * (b) specific preproc broken, (c) scatter across many actions (race). */
    if (mm > 0) {
        static const char* kAlgoNames[] = {
            "LZ4","Snappy","Deflate","GDeflate","Zstd","ANS","Cascaded","Bitcomp"
        };
        int n_hist = gpucompress_get_chunk_history_count();
        printf("  Failing chunks (chunk_id  action  algo  quant  shuf  mismatches):\n");
        for (int c = 0; c < NUM_CHUNKS && c < n_hist; c++) {
            if (mm_chunks[c] == 0) continue;
            gpucompress_chunk_diag_t d;
            if (gpucompress_get_chunk_diag(c, &d) != 0) continue;
            int a     = d.nn_action;
            int algo  = a % 8;
            int quant = (a / 8) % 2;
            int shuf  = (a / 16) % 2;
            const char* aname = (algo >= 0 && algo < 8) ? kAlgoNames[algo] : "???";
            printf("    %6d  %4d  %-8s  %d      %d     %llu\n",
                   c, a, aname, quant, shuf, mm_chunks[c]);
        }
    }

    gpucompress_disable_online_learning();
    return (mm > 0) ? 1 : 0;
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char** argv)
{
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";
    if (argc <= 1) {
        FILE* f = fopen(weights, "rb");
        if (f) fclose(f);
        else weights = "../neural_net/weights/model.nnwt";
    }

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    printf("================================================================\n");
    printf("  NN Algorithm Convergence Test\n");
    printf("================================================================\n");
    printf("  Layout: %d chunks x %d floats = %.1f MB\n",
           NUM_CHUNKS, CHUNK_FLOATS, (double)DATA_BYTES / (1 << 20));
    printf("  Chunks 0-%d:  smooth ramp + noise  (zstd+SHUF should win)\n", HALF - 1);
    printf("  Chunks %d-%d: sparse 90%% zeros     (zstd NO-shuf should win)\n", HALF, NUM_CHUNKS - 1);
    printf("================================================================\n\n");

    /* Init */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(rc));
        return 1;
    }

    /* Allocate and fill */
    float *d_data = NULL, *d_readback = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_readback, DATA_BYTES));

    size_t half_floats = (size_t)HALF * CHUNK_FLOATS;
    int threads = 256;

    /* First half: shuffle-friendly data */
    int blocks1 = ((int)half_floats + threads - 1) / threads;
    fill_shuf_region<<<blocks1, threads>>>(d_data, half_floats);

    /* Second half: no-shuffle data (sparse, shuffle hurts) */
    int blocks2 = ((int)half_floats + threads - 1) / threads;
    fill_noshuf_region<<<blocks2, threads>>>(d_data, half_floats, half_floats);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Phase 1: Exhaustive ─────────────────────────────────── */
    printf("── Phase 1: Exhaustive Search (ground truth) ────────────────\n");
    run_exhaustive(d_data);
    printf("\n");

    /* ── Phase 2: NN-RL ──────────────────────────────────────── */
    printf("── Phase 2: NN-RL (ALGO_AUTO + SGD, LR=0.3, MAPE>=20%%) ────\n");
    int fail = run_nn_rl(d_data, d_readback);
    printf("\n");

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_readback);
    gpucompress_cleanup();
    remove(HDF5_FILE);

    return fail;
}
