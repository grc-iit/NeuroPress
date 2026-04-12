#include "generator.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

// ============================================================
// String helpers
// ============================================================

const char* palette_name(Palette p) {
    switch (p) {
        case PAL_UNIFORM:      return "uniform";
        case PAL_NORMAL:       return "normal";
        case PAL_GAMMA:        return "gamma";
        case PAL_EXPONENTIAL:  return "exponential";
        case PAL_BIMODAL:      return "bimodal";
        case PAL_GRAYSCOTT:    return "grayscott";
        case PAL_HIGH_ENTROPY: return "high_entropy";
        default:               return "unknown";
    }
}

const char* fillmode_name(FillMode f) {
    switch (f) {
        case FILL_CONSTANT:   return "constant";
        case FILL_LINEAR:     return "linear";
        case FILL_QUADRATIC:  return "quadratic";
        case FILL_SINUSOIDAL: return "sinusoidal";
        case FILL_RANDOM:     return "random";
        default:              return "unknown";
    }
}

// ============================================================
// CPU: palette weight computation (matches generator.py)
// ============================================================

static void compute_palette_weights(Palette p, float weights[N_BINS]) {
    float sum = 0.0f;
    switch (p) {
    case PAL_UNIFORM:
        for (int i = 0; i < N_BINS; i++) weights[i] = 1.0f / N_BINS;
        return;

    case PAL_NORMAL:
        for (int i = 0; i < N_BINS; i++) {
            float x = -3.0f + 6.0f * i / (N_BINS - 1);
            weights[i] = expf(-0.5f * x * x);
            sum += weights[i];
        }
        break;

    case PAL_GAMMA:
        for (int i = 0; i < N_BINS; i++) {
            float x = 0.1f + 5.9f * i / (N_BINS - 1);
            weights[i] = x * expf(-x);
            sum += weights[i];
        }
        break;

    case PAL_EXPONENTIAL: {
        weights[0] = 0.9999f;
        float decay_sum = 0.0f;
        for (int i = 1; i < N_BINS; i++) {
            weights[i] = powf(0.5f, (float)i);
            decay_sum += weights[i];
        }
        for (int i = 1; i < N_BINS; i++)
            weights[i] = weights[i] / decay_sum * 0.0001f;
        return; // already normalized (0.9999 + 0.0001 = 1)
    }

    case PAL_BIMODAL: {
        float bg = (N_BINS > 4) ? 0.2f / (N_BINS - 4) : 0.0f;
        for (int i = 0; i < N_BINS; i++) weights[i] = bg;
        weights[0] = 0.24f;  weights[1] = 0.16f;
        weights[N_BINS-2] = 0.16f; weights[N_BINS-1] = 0.24f;
        for (int i = 0; i < N_BINS; i++) sum += weights[i];
        break;
    }

    case PAL_GRAYSCOTT: {
        memset(weights, 0, sizeof(float) * N_BINS);
        int bg = std::max(2, N_BINS / 4);
        int sp = std::max(2, N_BINS / 4);
        for (int i = 0; i < bg; i++) {
            float t = (float)i / bg;
            weights[i] = 0.7f * expf(-2.0f * t * t);
        }
        for (int i = 0; i < sp; i++) {
            float t = (float)i / sp;
            weights[N_BINS - 1 - i] = 0.2f * expf(-2.0f * t * t);
        }
        int edge_s = bg, edge_e = N_BINS - sp;
        if (edge_e > edge_s) {
            float edge_w = 0.1f / (edge_e - edge_s);
            for (int i = edge_s; i < edge_e; i++) weights[i] = edge_w;
        }
        for (int i = 0; i < N_BINS; i++) sum += weights[i];
        break;
    }

    case PAL_HIGH_ENTROPY:
        for (int i = 0; i < N_BINS; i++) weights[i] = 1.0f / N_BINS;
        return;
    }

    if (sum > 0.0f)
        for (int i = 0; i < N_BINS; i++) weights[i] /= sum;
}

// ============================================================
// CPU: bin layout computation (matches generator.py)
// ============================================================

static void compute_bin_layout(Palette p, float bin_width,
                                float bin_lo[N_BINS], float bin_hi[N_BINS]) {
    if (p == PAL_HIGH_ENTROPY) {
        // Non-linear bins spanning orders of magnitude
        for (int i = 0; i < N_BINS; i++) {
            float t = (float)i / (N_BINS - 1);
            float center = powf(10.0f, -6.0f + t * 6.0f);
            float width  = center * 0.1f;
            bin_lo[i] = fmaxf(0.0f, center - width * 0.5f);
            bin_hi[i] = fminf(1.0f, center + width * 0.5f);
        }
        return;
    }

    // Standard bins: non-overlapping with gaps when bw*N <= range
    const float lo = 0.0f, hi = 1.0f;
    const float total_range = hi - lo;

    if (bin_width * N_BINS <= total_range) {
        float gap    = (N_BINS > 1) ? (total_range - bin_width * N_BINS) / (N_BINS - 1) : 0.0f;
        float stride = bin_width + gap;
        for (int i = 0; i < N_BINS; i++) {
            bin_lo[i] = lo + i * stride;
            bin_hi[i] = fminf(bin_lo[i] + bin_width, hi);
        }
    } else {
        float step = total_range / N_BINS;
        for (int i = 0; i < N_BINS; i++) {
            bin_lo[i] = lo + i * step;
            bin_hi[i] = fminf(bin_lo[i] + bin_width, hi);
        }
    }
}

// ============================================================
// GPU kernel: tile-based Markov chain generation
//
// Each thread processes TILE consecutive elements.
// Markov chain: with probability `perturbation`, switch to a new bin
// sampled from the palette CDF; otherwise stay in current bin.
// Fill mode determines the value within each bin.
// ============================================================

#define GEN_TILE 128

__global__ void generate_kernel(
    float* __restrict__ output,
    size_t num_elements,
    const float* __restrict__ d_bin_lo,
    const float* __restrict__ d_bin_hi,
    const float* __restrict__ d_cdf,
    int n_bins,
    float perturbation,
    int fill_mode,
    unsigned long long seed)
{
    size_t tid        = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tile_start = tid * GEN_TILE;
    if (tile_start >= num_elements) return;
    size_t tile_end   = min(tile_start + (size_t)GEN_TILE, num_elements);

    curandState rng;
    curand_init(seed, tid, 0, &rng);

    // Sample initial bin from CDF
    int cur_bin = n_bins - 1;
    {
        float r = curand_uniform(&rng);
        for (int b = 0; b < n_bins; b++) {
            if (r <= d_cdf[b]) { cur_bin = b; break; }
        }
    }

    // Expected run length for fill modes that need it (avoids look-ahead)
    float expected_run = (perturbation > 0.0f) ? (1.0f / perturbation) : (float)(tile_end - tile_start);
    size_t run_start = tile_start;

    for (size_t i = tile_start; i < tile_end; i++) {
        // Markov: switch bin?
        if (i > tile_start && perturbation > 0.0f) {
            float sw = curand_uniform(&rng);
            if (sw < perturbation) {
                float r = curand_uniform(&rng);
                int new_bin = n_bins - 1;
                for (int b = 0; b < n_bins; b++) {
                    if (r <= d_cdf[b]) { new_bin = b; break; }
                }
                cur_bin  = new_bin;
                run_start = i;
            }
        }

        float lo  = d_bin_lo[cur_bin];
        float hi  = d_bin_hi[cur_bin];
        float mid = (lo + hi) * 0.5f;
        float val;

        switch (fill_mode) {
        case FILL_CONSTANT:
            val = lo;
            break;

        case FILL_RANDOM:
            val = (hi > lo) ? (lo + curand_uniform(&rng) * (hi - lo)) : lo;
            break;

        case FILL_LINEAR: {
            float within = (float)(i - run_start);
            float t = within / fmaxf(expected_run - 1.0f, 1.0f);
            t = fminf(t, 1.0f);
            val = lo + t * (hi - lo);
            break;
        }

        case FILL_QUADRATIC: {
            float within = (float)(i - run_start);
            float t = within / fmaxf(expected_run - 1.0f, 1.0f);
            t = fminf(t, 1.0f);
            float phase = t - 0.5f;
            float span  = hi - lo;
            float a     = 4.0f * (span / 4.0f) / 0.25f;
            val = mid + a * phase * phase - span / 4.0f;
            val = fmaxf(fminf(lo, hi), fminf(val, fmaxf(lo, hi)));
            break;
        }

        case FILL_SINUSOIDAL: {
            float within = (float)(i - run_start);
            float t = within / fmaxf(expected_run, 1.0f);
            val = mid + (hi - lo) * 0.5f * sinf(2.0f * 3.14159265358979f * t);
            break;
        }

        default:
            val = lo;
            break;
        }

        output[i] = val;
    }
}

// ============================================================
// Host entry point
// ============================================================

float* generate_chunk_gpu(const ChunkParams& params, uint64_t seed,
                           cudaStream_t stream)
{
    // Compute palette weights + CDF (CPU)
    float weights[N_BINS], cdf[N_BINS];
    compute_palette_weights(params.palette, weights);
    cdf[0] = weights[0];
    for (int i = 1; i < N_BINS; i++) cdf[i] = cdf[i-1] + weights[i];
    cdf[N_BINS-1] = 1.0f; // guarantee last bin catches rounding

    // Compute bin layout (CPU)
    float bin_lo[N_BINS], bin_hi[N_BINS];
    compute_bin_layout(params.palette, params.bin_width, bin_lo, bin_hi);

    // Upload bin layout + CDF to GPU
    float *d_bin_lo, *d_bin_hi, *d_cdf;
    cudaMalloc(&d_bin_lo, N_BINS * sizeof(float));
    cudaMalloc(&d_bin_hi, N_BINS * sizeof(float));
    cudaMalloc(&d_cdf,    N_BINS * sizeof(float));
    cudaMemcpyAsync(d_bin_lo, bin_lo, N_BINS*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bin_hi, bin_hi, N_BINS*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cdf,    cdf,    N_BINS*sizeof(float), cudaMemcpyHostToDevice, stream);

    // Allocate output buffer
    float* d_output;
    cudaMalloc(&d_output, params.num_elements * sizeof(float));

    // Launch generation kernel
    // Each thread covers GEN_TILE elements; 256 threads per block
    size_t n_tiles  = (params.num_elements + GEN_TILE - 1) / GEN_TILE;
    int    threads  = 256;
    int    blocks   = (int)((n_tiles + threads - 1) / threads);

    generate_kernel<<<blocks, threads, 0, stream>>>(
        d_output, params.num_elements,
        d_bin_lo, d_bin_hi, d_cdf, N_BINS,
        params.perturbation, (int)params.fill_mode, seed);

    // Free temporary GPU arrays (after kernel is enqueued)
    cudaFree(d_bin_lo);
    cudaFree(d_bin_hi);
    cudaFree(d_cdf);

    return d_output;
}
