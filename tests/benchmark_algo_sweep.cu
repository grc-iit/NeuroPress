/**
 * benchmark_algo_sweep.cu
 *
 * Sweeps all 8 lossless nvcomp algorithms × 2 preprocessing options
 * (no shuffle / byte-shuffle4) on the same 8 fill patterns used in
 * benchmark_vol_gpu.cu.
 *
 * Goal: identify the best static compression config per data pattern,
 *       so the result can be used as the fixed config in benchmark_vol_gpu.cu.
 *
 * Configs tested (16 total):
 *   algorithms : LZ4, SNAPPY, DEFLATE, GDEFLATE, ZSTD, ANS, CASCADED, BITCOMP
 *   preprocessing: none  (cd[1]=0, cd[2]=0)
 *                  shuffle4 (cd[1]=2, cd[2]=4)  ← 4-byte shuffle for float32
 *
 * Patterns (same as benchmark_vol_gpu.cu):
 *   constant, smooth_sine, ramp, sparse, step, exp_decay, sawtooth, impulse_train
 *
 * Method:
 *   For each config: write all 8 patterns as separate chunks into one HDF5
 *   file via the gpucompress VOL, query per-chunk compressed sizes via
 *   H5Dget_chunk_info, read back and verify each chunk.
 *
 * Output:
 *   tests/benchmark_algo_sweep_results/sweep_by_config.csv
 *   tests/benchmark_algo_sweep_results/sweep_by_pattern.csv
 *   Console: ranked table (best avg ratio first)
 *
 * Build:
 *   cmake --build build --target benchmark_algo_sweep
 *
 * Run:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *     ./build/benchmark_algo_sweep [--chunk-mb N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Constants
 * ============================================================ */

#define DEFAULT_CHUNK_MB   8
#define N_PATTERNS         8
#define N_ALGOS            8    /* LZ4=1 … BITCOMP=8 */
#define N_PREPROC          2    /* none, shuffle4 */
#define N_CONFIGS          (N_ALGOS * N_PREPROC)   /* 16 */

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define TMP_FILE  "/tmp/bm_algo_sweep.h5"
#define OUT_DIR   "tests/benchmark_algo_sweep_results"
#define OUT_CSV_CONFIG  OUT_DIR "/sweep_by_config.csv"
#define OUT_CSV_PATTERN OUT_DIR "/sweep_by_pattern.csv"

/* ============================================================
 * Algorithm / preprocessing metadata
 * ============================================================ */

static const char *ALGO_NAMES[N_ALGOS + 1] = {
    "AUTO",      /* 0 — not used */
    "LZ4",       /* 1 */
    "SNAPPY",    /* 2 */
    "DEFLATE",   /* 3 */
    "GDEFLATE",  /* 4 */
    "ZSTD",      /* 5 */
    "ANS",       /* 6 */
    "CASCADED",  /* 7 */
    "BITCOMP"    /* 8 */
};

static const char *PATTERN_NAMES[N_PATTERNS] = {
    "constant", "smooth_sine", "ramp", "sparse",
    "step", "exp_decay", "sawtooth", "impulse_train"
};

/* cd_values for each of the 16 configs */
typedef struct {
    unsigned int algo;       /* 1–8 */
    unsigned int preproc;    /* 0 or GPUCOMPRESS_PREPROC_SHUFFLE_4 */
    unsigned int shuf_sz;    /* 0 or 4 */
    char name[32];           /* e.g. "LZ4+shuffle" */
} Config;

static Config g_configs[N_CONFIGS];

static void build_configs(void) {
    int idx = 0;
    for (int a = 1; a <= N_ALGOS; a++) {
        /* no shuffle */
        g_configs[idx].algo    = (unsigned int)a;
        g_configs[idx].preproc = 0;
        g_configs[idx].shuf_sz = 0;
        snprintf(g_configs[idx].name, sizeof(g_configs[idx].name),
                 "%s", ALGO_NAMES[a]);
        idx++;
        /* shuffle4 */
        g_configs[idx].algo    = (unsigned int)a;
        g_configs[idx].preproc = GPUCOMPRESS_PREPROC_SHUFFLE_4;
        g_configs[idx].shuf_sz = 4;
        snprintf(g_configs[idx].name, sizeof(g_configs[idx].name),
                 "%s+shuffle", ALGO_NAMES[a]);
        idx++;
    }
}

/* ============================================================
 * GPU fill kernel — identical to benchmark_vol_gpu.cu
 * ============================================================ */

__device__ static float elem_rand(unsigned long long seed) {
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 33;
    return (float)(seed & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void fill_chunk_kernel(float * __restrict__ buf, size_t n,
                                  int pattern_id, unsigned long long seed_base)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned long long s = seed_base ^ ((unsigned long long)i * 6364136223846793005ULL);
        float r = elem_rand(s);
        float v = 0.0f;
        switch (pattern_id) {
        case 0: v = 42.0f; break;
        case 1: v = 1000.0f * sinf(2.0f * 3.14159265358979f * (float)i / (float)n); break;
        case 2: v = (float)i / (float)n; break;
        case 3: v = (r < 0.01f) ? (r * 10000.0f - 50.0f) : 0.0f; break;
        case 4: v = (float)(i / (n / 8u)) * 100.0f; break;
        case 5: v = 1000.0f * expf(-5.0f * (float)i / (float)n); break;
        case 6: { size_t p = n / 16u; v = (float)(i % p) / (float)p * 1000.0f; break; }
        case 7: v = ((i % 1024u) < 4u) ? (5000.0f + (r - 0.5f) * 200.0f) : 0.0f; break;
        default: v = 0.0f; break;
        }
        buf[i] = v;
    }
}

__global__ void count_mismatches_kernel(const float * __restrict__ a,
                                        const float * __restrict__ b,
                                        size_t n, unsigned long long *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local++;
    atomicAdd(out, local);
}

/* ============================================================
 * Timing
 * ============================================================ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

/* ============================================================
 * HDF5 helpers
 * ============================================================ */

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_vol_fapl(void) {
    hid_t native_id = H5VLget_connector_id_by_name("native");
    assert(native_id >= 0 && "H5VLget_connector_id_by_name(native) failed");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl >= 0 && "H5Pcreate(FAPL) failed");
    herr_t rc = H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    assert(rc >= 0 && "H5Pset_fapl_gpucompress failed");
    H5VLclose(native_id);
    return fapl;
}

/* Build DCPL: N_PATTERNS chunks of chunk_floats each, gpucompress filter */
static hid_t make_dcpl(const Config *cfg, hsize_t chunk_floats) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    assert(dcpl >= 0 && "H5Pcreate(DCPL) failed");

    hsize_t cdims[1] = { chunk_floats };
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = cfg->algo;
    cd[1] = cfg->preproc;
    cd[2] = cfg->shuf_sz;
    pack_double_cd(0.0, &cd[3], &cd[4]);  /* error_bound = 0 → lossless */
    herr_t rc = H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                               H5Z_FLAG_OPTIONAL,
                               H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    assert(rc >= 0 && "H5Pset_filter failed");
    return dcpl;
}

/* ============================================================
 * Result structs
 * ============================================================ */

typedef struct {
    int    config_idx;
    int    pattern_idx;
    double ratio;          /* uncompressed / compressed */
    double write_ms;       /* time to write this chunk (estimated) */
    double read_ms;        /* time to read/decompress this chunk (estimated) */
    int    correct;        /* 1 = lossless verified, 0 = mismatch */
    int    failed;         /* 1 = algorithm errored (skipped) */
} ChunkResult;

/* ============================================================
 * Run one config across all 8 patterns.
 *
 * Each pattern gets its own single-chunk dataset in one HDF5 file.
 * All writes and reads use H5S_ALL — no hyperslabs.
 * Layout: file has datasets "p0".."p7", each dims=[chunk_floats],
 *         chunk=[chunk_floats] (exactly one chunk per dataset).
 * ============================================================ */

static void run_config(const Config *cfg, int cfg_idx,
                       size_t chunk_floats,
                       float *d_chunk, float *d_ref, float *d_read,
                       unsigned long long *d_mismatch,
                       ChunkResult results[N_PATTERNS])
{
    const size_t chunk_bytes = chunk_floats * sizeof(float);

    for (int p = 0; p < N_PATTERNS; p++) {
        results[p].config_idx  = cfg_idx;
        results[p].pattern_idx = p;
        results[p].ratio       = 0.0;
        results[p].write_ms    = 0.0;
        results[p].read_ms     = 0.0;
        results[p].correct     = 0;
        results[p].failed      = 0;
    }

    hsize_t dims[1] = { (hsize_t)chunk_floats };

    /* ---- Write phase: one dataset per pattern, H5S_ALL ---------- */
    remove(TMP_FILE);
    hid_t fapl = make_vol_fapl();
    hid_t fid  = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (fid < 0) {
        fprintf(stderr, "  [%s] H5Fcreate failed — skipping\n", cfg->name);
        for (int p = 0; p < N_PATTERNS; p++) results[p].failed = 1;
        return;
    }

    double write_t0 = now_ms();
    for (int p = 0; p < N_PATTERNS; p++) {
        /* Fill GPU buffer with this pattern */
        fill_chunk_kernel<<<512, 256>>>(d_chunk, chunk_floats, p, (unsigned long long)p);
        assert(cudaDeviceSynchronize() == cudaSuccess && "fill_chunk_kernel failed");

        char dset_name[8];
        snprintf(dset_name, sizeof(dset_name), "p%d", p);

        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = make_dcpl(cfg, (hsize_t)chunk_floats);
        hid_t dset = H5Dcreate2(fid, dset_name, H5T_NATIVE_FLOAT,
                                 fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);
        H5Pclose(dcpl);

        if (dset < 0) {
            fprintf(stderr, "  [%s] H5Dcreate2 failed for pattern %d\n",
                    cfg->name, p);
            results[p].failed = 1;
            continue;
        }

        /* H5S_ALL — writes the entire single-chunk dataset */
        herr_t wrc = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, d_chunk);
        H5Dclose(dset);

        if (wrc < 0) {
            fprintf(stderr, "  [%s] H5Dwrite failed for pattern %d\n",
                    cfg->name, p);
            results[p].failed = 1;
        }
    }
    double write_t1 = now_ms();
    H5Fclose(fid);

    /* ---- Query per-dataset compressed chunk size + read + verify - */
    fapl = make_vol_fapl();
    fid  = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    assert(fid >= 0 && "H5Fopen failed");

    double read_t0 = now_ms();
    for (int p = 0; p < N_PATTERNS; p++) {
        if (results[p].failed) continue;

        char dset_name[8];
        snprintf(dset_name, sizeof(dset_name), "p%d", p);

        hid_t dset = H5Dopen2(fid, dset_name, H5P_DEFAULT);
        assert(dset >= 0 && "H5Dopen2 failed");

        /* Compressed size of chunk 0 (the only chunk in this dataset) */
        hsize_t offset[1] = { 0 };
        hsize_t comp_size = 0;
        haddr_t addr = 0;
        unsigned fmask = 0;
        herr_t ci = H5Dget_chunk_info_by_coord(dset, offset, &fmask,
                                                &addr, &comp_size);
        if (ci >= 0 && comp_size > 0)
            results[p].ratio = (double)chunk_bytes / (double)comp_size;

        /* Read back into GPU buffer — H5S_ALL, no hyperslab */
        herr_t rrc = H5Dread(dset, H5T_NATIVE_FLOAT,
                              H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        assert(cudaDeviceSynchronize() == cudaSuccess);
        H5Dclose(dset);

        if (rrc < 0) {
            fprintf(stderr, "  [%s] H5Dread failed for pattern %d\n",
                    cfg->name, p);
            results[p].failed = 1;
            continue;
        }

        /* Regenerate reference on GPU and compare */
        fill_chunk_kernel<<<512, 256>>>(d_ref, chunk_floats, p, (unsigned long long)p);
        assert(cudaDeviceSynchronize() == cudaSuccess);

        cudaMemset(d_mismatch, 0, sizeof(unsigned long long));
        count_mismatches_kernel<<<512, 256>>>(d_ref, d_read, chunk_floats, d_mismatch);
        assert(cudaDeviceSynchronize() == cudaSuccess);

        unsigned long long mm = 0;
        assert(cudaMemcpy(&mm, d_mismatch, sizeof(mm),
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        results[p].correct = (mm == 0) ? 1 : 0;
    }
    double read_t1 = now_ms();

    H5Fclose(fid);
    remove(TMP_FILE);

    double avg_write_ms = (write_t1 - write_t0) / N_PATTERNS;
    double avg_read_ms  = (read_t1  - read_t0)  / N_PATTERNS;
    for (int p = 0; p < N_PATTERNS; p++) {
        if (!results[p].failed) {
            results[p].write_ms = avg_write_ms;
            results[p].read_ms  = avg_read_ms;
        }
    }
}

/* ============================================================
 * CSV output
 * ============================================================ */

static void write_csv_by_config(ChunkResult all[N_CONFIGS][N_PATTERNS],
                                 size_t chunk_bytes)
{
    FILE *f = fopen(OUT_CSV_CONFIG, "w");
    assert(f && "fopen sweep_by_config.csv failed");

    fprintf(f, "config,algo,shuffle,pattern,ratio,write_ms,read_ms,"
               "write_mbps,read_mbps,correct,failed\n");

    for (int c = 0; c < N_CONFIGS; c++) {
        const Config *cfg = &g_configs[c];
        for (int p = 0; p < N_PATTERNS; p++) {
            ChunkResult *r = &all[c][p];
            double chunk_mb = (double)chunk_bytes / (1 << 20);
            double wmb = (r->write_ms > 0) ? chunk_mb / (r->write_ms / 1000.0) : 0;
            double rmb = (r->read_ms  > 0) ? chunk_mb / (r->read_ms  / 1000.0) : 0;
            fprintf(f, "%s,%s,%s,%s,%.4f,%.2f,%.2f,%.1f,%.1f,%d,%d\n",
                    cfg->name, ALGO_NAMES[cfg->algo],
                    cfg->shuf_sz ? "yes" : "no",
                    PATTERN_NAMES[p],
                    r->ratio, r->write_ms, r->read_ms, wmb, rmb,
                    r->correct, r->failed);
        }
    }
    fclose(f);
}

static void write_csv_by_pattern(ChunkResult all[N_CONFIGS][N_PATTERNS])
{
    FILE *f = fopen(OUT_CSV_PATTERN, "w");
    assert(f && "fopen sweep_by_pattern.csv failed");

    /* Header: pattern | config1 | config2 | ... */
    fprintf(f, "pattern");
    for (int c = 0; c < N_CONFIGS; c++)
        fprintf(f, ",%s", g_configs[c].name);
    fprintf(f, "\n");

    for (int p = 0; p < N_PATTERNS; p++) {
        fprintf(f, "%s", PATTERN_NAMES[p]);
        for (int c = 0; c < N_CONFIGS; c++) {
            ChunkResult *r = &all[c][p];
            if (r->failed)      fprintf(f, ",ERR");
            else if (!r->correct) fprintf(f, ",MISMATCH");
            else                 fprintf(f, ",%.2f", r->ratio);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* ============================================================
 * Console summary — ranked by average ratio
 * ============================================================ */

typedef struct { int cfg_idx; double avg_ratio; int all_correct; } RankEntry;

static int rank_cmp(const void *a, const void *b) {
    double da = ((RankEntry*)a)->avg_ratio;
    double db = ((RankEntry*)b)->avg_ratio;
    return (da < db) ? 1 : (da > db) ? -1 : 0;
}

static void print_summary(ChunkResult all[N_CONFIGS][N_PATTERNS],
                           size_t chunk_bytes)
{
    RankEntry rank[N_CONFIGS];
    for (int c = 0; c < N_CONFIGS; c++) {
        double sum = 0.0; int n = 0; int ok = 1;
        for (int p = 0; p < N_PATTERNS; p++) {
            if (all[c][p].failed) { ok = 0; continue; }
            if (!all[c][p].correct) ok = 0;
            if (all[c][p].ratio > 0) { sum += all[c][p].ratio; n++; }
        }
        rank[c].cfg_idx    = c;
        rank[c].avg_ratio  = n ? sum / n : 0.0;
        rank[c].all_correct = ok;
    }
    qsort(rank, N_CONFIGS, sizeof(RankEntry), rank_cmp);

    double chunk_mb = (double)chunk_bytes / (1 << 20);

    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Algorithm Sweep — Ranked by Average Compression Ratio (%.0f MB chunk, float32, lossless)           │\n",
           chunk_mb);
    printf("├──────────────────┬────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────┤\n");
    printf("│ Config           │  avg   │ constant │sin_wave  │ ramp     │ sparse   │ step     │exp_decay │sawtooth│\n");
    printf("├──────────────────┼────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────┤\n");

    for (int r = 0; r < N_CONFIGS; r++) {
        int c = rank[r].cfg_idx;
        const Config *cfg = &g_configs[c];
        printf("│ %-16s │ %5.2fx │", cfg->name, rank[r].avg_ratio);
        /* print first 7 patterns (impulse_train omitted for width) */
        for (int p = 0; p < 7; p++) {
            ChunkResult *cr = &all[c][p];
            if (cr->failed)        printf(" %-8s │", "ERROR");
            else if (!cr->correct) printf(" %-8s │", "MISMATCH");
            else                   printf(" %6.2fx   │", cr->ratio);
        }
        printf(" %s\n", rank[r].all_correct ? "PASS" : "FAIL");
    }
    printf("└──────────────────┴────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────┘\n");

    printf("\n  Best overall: %s (avg %.2fx)\n",
           g_configs[rank[0].cfg_idx].name, rank[0].avg_ratio);
    printf("  Best correct: ");
    for (int r = 0; r < N_CONFIGS; r++) {
        if (rank[r].all_correct) {
            printf("%s (avg %.2fx)\n",
                   g_configs[rank[r].cfg_idx].name, rank[r].avg_ratio);
            break;
        }
    }

    /* Per-pattern winner */
    printf("\n  Per-pattern best config:\n");
    for (int p = 0; p < N_PATTERNS; p++) {
        int best_c = -1; double best_r = 0.0;
        for (int c = 0; c < N_CONFIGS; c++) {
            if (all[c][p].failed || !all[c][p].correct) continue;
            if (all[c][p].ratio > best_r) {
                best_r = all[c][p].ratio;
                best_c = c;
            }
        }
        if (best_c >= 0)
            printf("    %-16s → %s (%.2fx)\n",
                   PATTERN_NAMES[p], g_configs[best_c].name, best_r);
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    int chunk_mb = DEFAULT_CHUNK_MB;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc)
            chunk_mb = atoi(argv[++i]);
    }

    const size_t chunk_floats = (size_t)chunk_mb * 1024 * 1024 / sizeof(float);
    const size_t chunk_bytes  = chunk_floats * sizeof(float);

    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│  GPUCompress Algorithm Sweep — %d MB chunk, %d configs, %d patterns │\n",
           chunk_mb, N_CONFIGS, N_PATTERNS);
    printf("└──────────────────────────────────────────────────────────────────┘\n\n");

    build_configs();
    mkdir(OUT_DIR, 0755);
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init() failed\n");
        return 1;
    }

    hid_t vol_id = H5VL_gpucompress_register();
    assert(vol_id >= 0 && "H5VL_gpucompress_register() failed");

    /* GPU buffers: one chunk worth each */
    float              *d_chunk    = NULL;
    float              *d_ref      = NULL;
    float              *d_read     = NULL;
    unsigned long long *d_mismatch = NULL;

    assert(cudaMalloc(&d_chunk,    chunk_bytes)              == cudaSuccess);
    assert(cudaMalloc(&d_ref,      chunk_bytes)              == cudaSuccess);
    assert(cudaMalloc(&d_read,     chunk_bytes)              == cudaSuccess);
    assert(cudaMalloc(&d_mismatch, sizeof(unsigned long long)) == cudaSuccess);

    printf("  Chunk size : %d MB (%zu floats)\n", chunk_mb, chunk_floats);
    printf("  GPU memory : %.0f MB allocated\n\n",
           3.0 * chunk_bytes / (1 << 20));

    /* Results table: [config][pattern] */
    static ChunkResult all_results[N_CONFIGS][N_PATTERNS];

    for (int c = 0; c < N_CONFIGS; c++) {
        printf("[%2d/%d] %-20s ... ", c + 1, N_CONFIGS, g_configs[c].name);
        fflush(stdout);

        run_config(&g_configs[c], c, chunk_floats,
                   d_chunk, d_ref, d_read, d_mismatch,
                   all_results[c]);

        /* Quick per-config summary */
        double sum = 0.0; int n = 0; int ok = 1;
        for (int p = 0; p < N_PATTERNS; p++) {
            if (all_results[c][p].failed)       { ok = 0; continue; }
            if (!all_results[c][p].correct)     { ok = 0; }
            if (all_results[c][p].ratio > 0)    { sum += all_results[c][p].ratio; n++; }
        }
        double avg = n ? sum / n : 0.0;
        printf("avg ratio=%.2fx  %s\n", avg, ok ? "PASS" : "FAIL");
    }

    /* Summary table + CSVs */
    print_summary(all_results, chunk_bytes);
    write_csv_by_config(all_results, chunk_bytes);
    write_csv_by_pattern(all_results);

    printf("\n  CSV (by config)  : %s\n", OUT_CSV_CONFIG);
    printf("  CSV (by pattern) : %s\n\n", OUT_CSV_PATTERN);

    /* Cleanup */
    cudaFree(d_chunk);
    cudaFree(d_ref);
    cudaFree(d_read);
    cudaFree(d_mismatch);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    remove(TMP_FILE);

    return 0;
}
