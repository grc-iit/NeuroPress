/**
 * @file gray_scott_gpu.cuh
 * @brief Device helpers and kernel declarations for GPU Gray-Scott simulation
 */

#ifndef GRAY_SCOTT_GPU_CUH
#define GRAY_SCOTT_GPU_CUH

#include <cuda_runtime.h>

/* ============================================================
 * Device inline helpers
 * ============================================================ */

/** Flat index for 3D grid: x fastest (ZYX order). */
__device__ __host__ static inline int gs_idx(int x, int y, int z, int L)
{
    return x + y * L + z * L * L;
}

/** Periodic boundary wrap — branchless, avoids expensive modulo. */
__device__ static inline int gs_wrap(int v, int L)
{
    return (v < 0) ? v + L : ((v >= L) ? v - L : v);
}

/** Splitmix64 hash to [0,1) float.  Same pattern as benchmark_vol_gpu.cu. */
__device__ static inline float gs_rand(unsigned long long seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 33;
    return (float)(seed & 0xFFFFFFu) / (float)0x1000000u;
}

/* ============================================================
 * Kernel declarations
 * ============================================================ */

/**
 * Initialize U and V fields.
 * U=1 everywhere, V=0 everywhere, center 12^3 cube: U=0.25, V=0.33.
 * Adds noise to U.
 */
__global__ void gs_init_kernel(float* u, float* v, int L,
                               float noise, unsigned long long seed);

/**
 * One forward-Euler time step of Gray-Scott.
 * Fused Laplacian + reaction + update.
 */
__global__ void gs_step_kernel(const float* u_in, const float* v_in,
                               float* u_out, float* v_out,
                               int L, float Du, float Dv,
                               float F, float k, float dt,
                               float noise, unsigned long long noise_seed);

#endif /* GRAY_SCOTT_GPU_CUH */
