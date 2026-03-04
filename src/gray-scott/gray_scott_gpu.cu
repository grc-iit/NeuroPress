/**
 * @file gray_scott_gpu.cu
 * @brief CUDA kernels for GPU Gray-Scott reaction-diffusion simulation
 */

#include "gray_scott_gpu.cuh"

#define GS_BLOCK_SIZE 256

/* ============================================================
 * Init kernel
 * ============================================================ */

__global__ void gs_init_kernel(float* u, float* v, int L,
                               float noise, unsigned long long seed)
{
    const int N = L * L * L;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x)
    {
        int x = i % L;
        int y = (i / L) % L;
        int z = i / (L * L);

        /* Background state */
        float ui = 1.0f;
        float vi = 0.0f;

        /* Center 12^3 perturbation */
        int half = L / 2;
        if (x >= half - 6 && x < half + 6 &&
            y >= half - 6 && y < half + 6 &&
            z >= half - 6 && z < half + 6)
        {
            ui = 0.25f;
            vi = 0.33f;
        }

        /* Add noise to U */
        ui += noise * (gs_rand(seed ^ (unsigned long long)i) * 2.0f - 1.0f);

        u[i] = ui;
        v[i] = vi;
    }
}

/* ============================================================
 * Step kernel — fused Laplacian + reaction + forward Euler
 * ============================================================ */

__global__ void gs_step_kernel(const float* u_in, const float* v_in,
                               float* u_out, float* v_out,
                               int L, float Du, float Dv,
                               float F, float k, float dt,
                               float noise, unsigned long long noise_seed)
{
    const int N = L * L * L;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x)
    {
        int x = i % L;
        int y = (i / L) % L;
        int z = i / (L * L);

        /* Current values */
        float uc = u_in[i];
        float vc = v_in[i];

        /* 6-point Laplacian with periodic BCs */
        int xm = gs_idx(gs_wrap(x - 1, L), y, z, L);
        int xp = gs_idx(gs_wrap(x + 1, L), y, z, L);
        int ym = gs_idx(x, gs_wrap(y - 1, L), z, L);
        int yp = gs_idx(x, gs_wrap(y + 1, L), z, L);
        int zm = gs_idx(x, y, gs_wrap(z - 1, L), L);
        int zp = gs_idx(x, y, gs_wrap(z + 1, L), L);

        float lap_u = (u_in[xm] + u_in[xp] +
                       u_in[ym] + u_in[yp] +
                       u_in[zm] + u_in[zp] - 6.0f * uc) / 6.0f;

        float lap_v = (v_in[xm] + v_in[xp] +
                       v_in[ym] + v_in[yp] +
                       v_in[zm] + v_in[zp] - 6.0f * vc) / 6.0f;

        /* Reaction terms */
        float uvv = uc * vc * vc;
        float du = Du * lap_u - uvv + F * (1.0f - uc);
        float dv = Dv * lap_v + uvv - (F + k) * vc;

        /* Noise on U */
        du += noise * (gs_rand(noise_seed ^ (unsigned long long)i) * 2.0f - 1.0f);

        /* Forward Euler update */
        u_out[i] = uc + du * dt;
        v_out[i] = vc + dv * dt;
    }
}
