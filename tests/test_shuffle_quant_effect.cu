/**
 * test_shuffle_quant_effect.cu
 *
 * Comprehensive test: 10 data patterns × 8 algorithms × 4 preproc configs.
 * 16 MB GPU-originated float data. Shows compression ratio table.
 */

#include "gpucompress.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define DATA_BYTES  (16 * 1024 * 1024)
#define N_FLOATS    (DATA_BYTES / sizeof(float))
#define ERR_BOUND   0.01

/* ── 10 GPU kernels ── */

__global__ void gen_smooth_sine(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sinf((float)i * 0.001f) * 500.0f;
}

__global__ void gen_noisy_sine(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned s = (unsigned)i * 1103515245u + 12345u;
        out[i] = sinf((float)i * 0.0001f) * 1000.0f + ((float)(s & 0xFFFF) / 65535.0f - 0.5f) * 50.0f;
    }
}

__global__ void gen_sparse(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned s = (unsigned)i * 2654435761u;
        out[i] = ((s & 0xFF) < 5) ? sinf((float)i * 0.01f) * 100.0f : 0.0f;
    }
}

__global__ void gen_constant(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 3.14159f;
}

__global__ void gen_random(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned s = (unsigned)i * 1664525u + 1013904223u;
        s ^= s >> 16; s *= 0x45d9f3bu; s ^= s >> 16;
        out[i] = (float)s / 4294967295.0f;
    }
}

__global__ void gen_step(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)((i / 1024) * 10);
}

__global__ void gen_linear_ramp(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 0.001f;
}

__global__ void gen_exp_decay(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(-(float)i * 0.000005f) * 1000.0f;
}

__global__ void gen_alternating(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (i & 1) ? 100.0f : -100.0f;
}

__global__ void gen_turbulence(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = (float)i * 0.0003f;
        out[i] = sinf(x) * 100.0f + sinf(x * 3.7f) * 50.0f +
                 sinf(x * 13.1f) * 25.0f + sinf(x * 41.3f) * 12.5f;
    }
}

typedef void (*kernel_t)(float*, int);

struct Pattern {
    const char* name;
    kernel_t    kern;
};

struct Algo {
    const char* name;
    gpucompress_algorithm_t id;
};

struct Preproc {
    const char* label;
    unsigned int flags;
    double error_bound;
};

static void launch(kernel_t k, float* d, int n) {
    k<<<(n + 255) / 256, 256>>>(d, n);
}

int main()
{
    printf("=== Shuffle & Quantization Effect — 10 Patterns × 8 Algos (16 MB, eb=%.2f) ===\n\n", ERR_BOUND);

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed %d\n", err); return 1; }

    Pattern patterns[] = {
        { "smooth_sine",  gen_smooth_sine  },
        { "noisy_sine",   gen_noisy_sine   },
        { "sparse_2pct",  gen_sparse       },
        { "constant",     gen_constant     },
        { "random",       gen_random       },
        { "step_1k",      gen_step         },
        { "linear_ramp",  gen_linear_ramp  },
        { "exp_decay",    gen_exp_decay    },
        { "alternating",  gen_alternating  },
        { "turbulence",   gen_turbulence   },
    };
    int n_pat = 10;

    Algo algos[] = {
        { "lz4",      GPUCOMPRESS_ALGO_LZ4      },
        { "snappy",   GPUCOMPRESS_ALGO_SNAPPY    },
        { "deflate",  GPUCOMPRESS_ALGO_DEFLATE   },
        { "gdeflate", GPUCOMPRESS_ALGO_GDEFLATE  },
        { "zstd",     GPUCOMPRESS_ALGO_ZSTD      },
        { "ans",      GPUCOMPRESS_ALGO_ANS       },
        { "cascaded", GPUCOMPRESS_ALGO_CASCADED  },
        { "bitcomp",  GPUCOMPRESS_ALGO_BITCOMP   },
    };
    int n_algo = 8;

    Preproc preprocs[] = {
        { "none",  GPUCOMPRESS_PREPROC_NONE,                                     0.0       },
        { "shuf",  GPUCOMPRESS_PREPROC_SHUFFLE_4,                                0.0       },
        { "qnt",   GPUCOMPRESS_PREPROC_QUANTIZE,                                 ERR_BOUND },
        { "s+q",   GPUCOMPRESS_PREPROC_SHUFFLE_4 | GPUCOMPRESS_PREPROC_QUANTIZE, ERR_BOUND },
    };
    int n_pp = 4;

    float* d_data = NULL;
    cudaMalloc(&d_data, DATA_BYTES);
    float* h_data = (float*)malloc(DATA_BYTES);
    size_t max_out = gpucompress_max_compressed_size(DATA_BYTES);
    void* out_buf = malloc(max_out);

    /* Print header */
    printf("  %-13s | %-8s |", "pattern", "algo");
    for (int pp = 0; pp < n_pp; pp++)
        printf(" %7s |", preprocs[pp].label);
    printf(" best\n");

    printf("  --------------+----------+");
    for (int pp = 0; pp < n_pp; pp++)
        printf("---------+");
    printf("----------\n");

    for (int p = 0; p < n_pat; p++) {
        launch(patterns[p].kern, d_data, N_FLOATS);
        cudaDeviceSynchronize();
        cudaMemcpy(h_data, d_data, DATA_BYTES, cudaMemcpyDeviceToHost);

        for (int a = 0; a < n_algo; a++) {
            double ratios[4];
            const char* best_pp = "?";
            double best_ratio = 0.0;

            for (int pp = 0; pp < n_pp; pp++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm     = algos[a].id;
                cfg.preprocessing = preprocs[pp].flags;
                cfg.error_bound   = preprocs[pp].error_bound;

                gpucompress_stats_t stats = {};
                size_t out_size = max_out;
                err = gpucompress_compress(h_data, DATA_BYTES, out_buf, &out_size, &cfg, &stats);
                if (err != GPUCOMPRESS_SUCCESS) {
                    ratios[pp] = -1.0;
                } else {
                    ratios[pp] = stats.compression_ratio;
                    if (ratios[pp] > best_ratio) {
                        best_ratio = ratios[pp];
                        best_pp = preprocs[pp].label;
                    }
                }
            }

            printf("  %-13s | %-8s |", patterns[p].name, algos[a].name);
            for (int pp = 0; pp < n_pp; pp++) {
                if (ratios[pp] < 0)
                    printf("   FAIL  |");
                else if (ratios[pp] >= 100.0)
                    printf(" %6.0fx |", ratios[pp]);
                else
                    printf(" %6.2fx |", ratios[pp]);
            }
            printf(" %s\n", best_pp);
        }
        printf("  --------------+----------+");
        for (int pp = 0; pp < n_pp; pp++)
            printf("---------+");
        printf("----------\n");
    }

    cudaFree(d_data);
    free(h_data);
    free(out_buf);
    gpucompress_cleanup();
    printf("\nDone.\n");
    return 0;
}
