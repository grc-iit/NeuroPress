/**
 * @file data_generators.cuh
 * @brief 5 CUDA data-pattern generators for NN adaptiveness benchmark.
 *
 * Each kernel fills a float* d_data buffer of size L*L*L with a distinct
 * scientific data pattern that challenges the NN's compression predictions
 * in different ways.
 *
 * Patterns:
 *   0 - Ocean Waves + Spikes    (bimodal distribution)
 *   1 - Heating Surface          (high dynamic range, Gaussian hotspots)
 *   2 - Turbulent Flow           (Gray-Scott chaos via gpucompress API)
 *   3 - Geological Strata        (sharp inter-layer boundaries)
 *   4 - Particle Shower          (extreme sparsity with bursts)
 *
 * All kernels use splitmix64-based RNG for deterministic, reproducible output.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* ============================================================
 * splitmix64-style RNG (same approach as existing Gray-Scott kernels)
 * ============================================================ */

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long *state)
{
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ float rand_uniform(unsigned long long *state)
{
    return (float)(splitmix64(state) >> 11) / (float)(1ULL << 53);
}

__device__ __forceinline__ float rand_normal(unsigned long long *state)
{
    /* Box-Muller transform */
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    if (u1 < 1e-30f) u1 = 1e-30f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/* ============================================================
 * Pattern 0: Ocean Waves + Spikes
 *
 * Superposition of 3 sinusoidal waves + rare random spikes
 * (0.2% of voxels, 50-200x magnitude).
 * NN challenge: bimodal distribution — smooth base with outlier chunks.
 * ============================================================ */

__global__ void gen_ocean_waves(float *data, int L, unsigned int seed)
{
    size_t N = (size_t)L * L * L;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < N; idx += (size_t)gridDim.x * blockDim.x) {
        int z = (int)(idx % L);
        int y = (int)((idx / L) % L);
        int x = (int)(idx / ((size_t)L * L));

        float fx = (float)x / (float)L;
        float fy = (float)y / (float)L;
        float fz = (float)z / (float)L;

        /* Three waves with different frequencies */
        float val = 10.0f * sinf(6.2832f * 2.0f * fx)
                  +  5.0f * sinf(6.2832f * 5.0f * fy + 1.5f)
                  +  3.0f * sinf(6.2832f * 8.0f * fz + 0.7f);

        /* Spike injection: 0.2% probability */
        unsigned long long rng_state = (unsigned long long)seed * 6364136223846793005ULL
                                     + idx * 1442695040888963407ULL + 1;
        float r = rand_uniform(&rng_state);
        if (r < 0.002f) {
            float mag = 50.0f + 150.0f * rand_uniform(&rng_state);
            val *= mag;
        }

        data[idx] = val;
    }
}

/* ============================================================
 * Pattern 1: Heating Surface + Hotspots
 *
 * Smooth 3D temperature gradient (20→1000) + 8-16 Gaussian hotspot
 * spheres (5000+ peak).
 * NN challenge: high dynamic range, spatially varying statistics.
 * ============================================================ */

__global__ void gen_heating_surface(float *data, int L, unsigned int seed)
{
    size_t N = (size_t)L * L * L;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /* Pre-generate hotspot centers — use constant number for determinism */
    const int N_HOT = 12;
    float hx[12], hy[12], hz[12], hpeak[12], hradius[12];

    {
        unsigned long long hs = (unsigned long long)seed * 123456789ULL + 987654321ULL;
        for (int h = 0; h < N_HOT; h++) {
            hx[h]     = rand_uniform(&hs);
            hy[h]     = rand_uniform(&hs);
            hz[h]     = rand_uniform(&hs);
            hpeak[h]  = 5000.0f + 10000.0f * rand_uniform(&hs);
            hradius[h] = 0.02f + 0.06f * rand_uniform(&hs);
        }
    }

    for (; idx < N; idx += (size_t)gridDim.x * blockDim.x) {
        int z = (int)(idx % L);
        int y = (int)((idx / L) % L);
        int x = (int)(idx / ((size_t)L * L));

        float fx = (float)x / (float)L;
        float fy = (float)y / (float)L;
        float fz = (float)z / (float)L;

        /* Base: smooth temperature gradient */
        float val = 20.0f + 980.0f * (0.3f * fx + 0.5f * fy + 0.2f * fz);

        /* Add Gaussian hotspots */
        for (int h = 0; h < N_HOT; h++) {
            float dx = fx - hx[h];
            float dy = fy - hy[h];
            float dz = fz - hz[h];
            float dist2 = dx * dx + dy * dy + dz * dz;
            float r2 = hradius[h] * hradius[h];
            if (dist2 < 9.0f * r2) {
                val += hpeak[h] * expf(-dist2 / (2.0f * r2));
            }
        }

        data[idx] = val;
    }
}

/* ============================================================
 * Pattern 2: Turbulent Flow (Gray-Scott chaos)
 *
 * Generated via gpucompress_grayscott_* API externally.
 * This kernel is a placeholder — the dispatcher calls the API directly.
 * ============================================================ */

/* (No kernel needed — uses Gray-Scott API) */

/* ============================================================
 * Pattern 3: Geological Strata
 *
 * 8-12 horizontal layers with distinct base values + smooth intra-layer
 * noise, sharp inter-layer boundaries.
 * NN challenge: per-chunk ratio varies drastically depending on boundary
 * overlap.
 * ============================================================ */

__global__ void gen_geological_strata(float *data, int L, unsigned int seed)
{
    size_t N = (size_t)L * L * L;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /* Generate layer boundaries and properties deterministically */
    const int N_LAYERS = 10;
    float layer_top[10];  /* fractional y position of layer top */
    float layer_base[10]; /* base value for each layer */
    float layer_noise[10]; /* noise amplitude */

    {
        unsigned long long ls = (unsigned long long)seed * 314159265ULL + 271828182ULL;
        float accum = 0.0f;
        for (int l = 0; l < N_LAYERS; l++) {
            float thickness = 0.05f + 0.15f * rand_uniform(&ls);
            accum += thickness;
            layer_top[l] = accum;
            layer_base[l] = 100.0f + 900.0f * rand_uniform(&ls);
            layer_noise[l] = 1.0f + 20.0f * rand_uniform(&ls);
        }
        /* Normalize to [0, 1] */
        for (int l = 0; l < N_LAYERS; l++) {
            layer_top[l] /= accum;
        }
    }

    for (; idx < N; idx += (size_t)gridDim.x * blockDim.x) {
        int z = (int)(idx % L);
        int y = (int)((idx / L) % L);
        int x = (int)(idx / ((size_t)L * L));

        float fy = (float)y / (float)L;

        /* Find which layer this voxel belongs to */
        int layer = N_LAYERS - 1;
        for (int l = 0; l < N_LAYERS; l++) {
            if (fy < layer_top[l]) { layer = l; break; }
        }

        /* Base value + smooth spatial noise */
        unsigned long long rng_state = (unsigned long long)seed * 6364136223846793005ULL
                                     + idx * 1442695040888963407ULL + 2;
        float noise = layer_noise[layer] * rand_normal(&rng_state);

        /* Add smooth spatial variation within layer */
        float fx = (float)x / (float)L;
        float fz = (float)z / (float)L;
        float spatial = 5.0f * sinf(6.2832f * 3.0f * fx) * cosf(6.2832f * 2.0f * fz);

        data[idx] = layer_base[layer] + noise + spatial;
    }
}

/* ============================================================
 * Pattern 4: Particle Shower
 *
 * Near-zero background + 200-500 random Gaussian burst centers with
 * high amplitude (1e3-1e6).
 * NN challenge: extreme sparsity, burst-containing chunks are
 * completely different from empty ones.
 * ============================================================ */

__global__ void gen_particle_shower(float *data, int L, unsigned int seed)
{
    size_t N = (size_t)L * L * L;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /* Pre-generate burst centers — all threads compute same values */
    const int N_BURSTS = 300;
    float bx[300], by[300], bz[300], bamp[300], bradius[300];

    {
        unsigned long long bs = (unsigned long long)seed * 577215664ULL + 161803398ULL;
        for (int b = 0; b < N_BURSTS; b++) {
            bx[b] = rand_uniform(&bs);
            by[b] = rand_uniform(&bs);
            bz[b] = rand_uniform(&bs);
            /* Amplitude: 1e3 to 1e6 (log-uniform) */
            float log_amp = 3.0f + 3.0f * rand_uniform(&bs);
            bamp[b] = powf(10.0f, log_amp);
            bradius[b] = 0.005f + 0.02f * rand_uniform(&bs);
        }
    }

    for (; idx < N; idx += (size_t)gridDim.x * blockDim.x) {
        int z = (int)(idx % L);
        int y = (int)((idx / L) % L);
        int x = (int)(idx / ((size_t)L * L));

        float fx = (float)x / (float)L;
        float fy = (float)y / (float)L;
        float fz = (float)z / (float)L;

        /* Near-zero background with tiny noise */
        unsigned long long rng_state = (unsigned long long)seed * 6364136223846793005ULL
                                     + idx * 1442695040888963407ULL + 3;
        float val = 0.001f * rand_normal(&rng_state);

        /* Add burst contributions */
        for (int b = 0; b < N_BURSTS; b++) {
            float dx = fx - bx[b];
            float dy = fy - by[b];
            float dz = fz - bz[b];
            float dist2 = dx * dx + dy * dy + dz * dz;
            float r2 = bradius[b] * bradius[b];
            if (dist2 < 9.0f * r2) {
                val += bamp[b] * expf(-dist2 / (2.0f * r2));
            }
        }

        data[idx] = val;
    }
}

/* ============================================================
 * Composite kernel: all 4 non-GrayScott patterns by Z-region
 *
 * Divides the Z dimension into 5 equal bands.  Patterns 0,1,3,4
 * are generated inline; pattern 2 (turbulent flow / Gray-Scott)
 * is filled by the host after this kernel via cudaMemcpy of the
 * relevant Z-slab.
 *
 * The kernel writes pattern 2's region as zeros — the host
 * overwrites it afterwards.
 * ============================================================ */

__global__ void gen_composite(float *data, int L, unsigned int seed)
{
    size_t N = (size_t)L * L * L;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    int slab = L / 5;  /* Z-extent per pattern */

    /* Pre-generate heating_surface hotspots (pattern 1) */
    const int N_HOT = 12;
    float hhx[12], hhy[12], hhz[12], hhpeak[12], hhradius[12];
    {
        unsigned long long hs = (unsigned long long)seed * 123456789ULL + 987654321ULL;
        for (int h = 0; h < N_HOT; h++) {
            hhx[h]     = rand_uniform(&hs);
            hhy[h]     = rand_uniform(&hs);
            hhz[h]     = rand_uniform(&hs);
            hhpeak[h]  = 5000.0f + 10000.0f * rand_uniform(&hs);
            hhradius[h] = 0.02f + 0.06f * rand_uniform(&hs);
        }
    }

    /* Pre-generate geological_strata layers (pattern 3) */
    const int N_LAYERS = 10;
    float layer_top[10], layer_base[10], layer_noise[10];
    {
        unsigned long long ls = (unsigned long long)seed * 314159265ULL + 271828182ULL;
        float accum = 0.0f;
        for (int l = 0; l < N_LAYERS; l++) {
            float thickness = 0.05f + 0.15f * rand_uniform(&ls);
            accum += thickness;
            layer_top[l] = accum;
            layer_base[l] = 100.0f + 900.0f * rand_uniform(&ls);
            layer_noise[l] = 1.0f + 20.0f * rand_uniform(&ls);
        }
        for (int l = 0; l < N_LAYERS; l++) layer_top[l] /= accum;
    }

    /* Pre-generate particle_shower bursts (pattern 4) */
    const int N_BURSTS = 300;
    float bx[300], by[300], bz[300], bamp[300], bradius[300];
    {
        unsigned long long bs = (unsigned long long)seed * 577215664ULL + 161803398ULL;
        for (int b = 0; b < N_BURSTS; b++) {
            bx[b] = rand_uniform(&bs);
            by[b] = rand_uniform(&bs);
            bz[b] = rand_uniform(&bs);
            float log_amp = 3.0f + 3.0f * rand_uniform(&bs);
            bamp[b] = powf(10.0f, log_amp);
            bradius[b] = 0.005f + 0.02f * rand_uniform(&bs);
        }
    }

    for (; idx < N; idx += (size_t)gridDim.x * blockDim.x) {
        int z = (int)(idx % L);
        int y = (int)((idx / L) % L);
        int x = (int)(idx / ((size_t)L * L));

        int pat = z / slab;
        if (pat >= 5) pat = 4;

        /* Normalize coordinates to [0,1] within the pattern's own slab */
        float fx = (float)x / (float)L;
        float fy = (float)y / (float)L;
        int z_local = z - pat * slab;
        float fz = (float)z_local / (float)slab;

        float val = 0.0f;

        switch (pat) {
        case 0: { /* Ocean Waves + Spikes */
            val = 10.0f * sinf(6.2832f * 2.0f * fx)
                +  5.0f * sinf(6.2832f * 5.0f * fy + 1.5f)
                +  3.0f * sinf(6.2832f * 8.0f * fz + 0.7f);
            unsigned long long rng = (unsigned long long)seed * 6364136223846793005ULL
                                   + idx * 1442695040888963407ULL + 1;
            float r = rand_uniform(&rng);
            if (r < 0.002f) {
                val *= 50.0f + 150.0f * rand_uniform(&rng);
            }
            break;
        }
        case 1: { /* Heating Surface + Hotspots */
            val = 20.0f + 980.0f * (0.3f * fx + 0.5f * fy + 0.2f * fz);
            for (int h = 0; h < N_HOT; h++) {
                float dx = fx - hhx[h], dy = fy - hhy[h], dz = fz - hhz[h];
                float dist2 = dx*dx + dy*dy + dz*dz;
                float r2 = hhradius[h] * hhradius[h];
                if (dist2 < 9.0f * r2)
                    val += hhpeak[h] * expf(-dist2 / (2.0f * r2));
            }
            break;
        }
        case 2: /* Turbulent Flow — filled as zeros, host overwrites */
            val = 0.0f;
            break;
        case 3: { /* Geological Strata */
            int layer = N_LAYERS - 1;
            for (int l = 0; l < N_LAYERS; l++)
                if (fy < layer_top[l]) { layer = l; break; }
            unsigned long long rng = (unsigned long long)seed * 6364136223846793005ULL
                                   + idx * 1442695040888963407ULL + 2;
            float noise = layer_noise[layer] * rand_normal(&rng);
            float spatial = 5.0f * sinf(6.2832f * 3.0f * fx) * cosf(6.2832f * 2.0f * fz);
            val = layer_base[layer] + noise + spatial;
            break;
        }
        case 4: { /* Particle Shower */
            unsigned long long rng = (unsigned long long)seed * 6364136223846793005ULL
                                   + idx * 1442695040888963407ULL + 3;
            val = 0.001f * rand_normal(&rng);
            for (int b = 0; b < N_BURSTS; b++) {
                float dx = fx - bx[b], dy = fy - by[b], dz = fz - bz[b];
                float dist2 = dx*dx + dy*dy + dz*dz;
                float r2 = bradius[b] * bradius[b];
                if (dist2 < 9.0f * r2)
                    val += bamp[b] * expf(-dist2 / (2.0f * r2));
            }
            break;
        }
        }

        data[idx] = val;
    }
}

/* ============================================================
 * Pattern names
 * ============================================================ */

static const char *PATTERN_NAMES[] = {
    "ocean_waves",
    "heating_surface",
    "turbulent_flow",
    "geological_strata",
    "particle_shower"
};
