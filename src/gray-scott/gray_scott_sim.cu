/**
 * @file gray_scott_sim.cu
 * @brief Host driver class and C API for GPU Gray-Scott simulation
 */

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#include "gpucompress_grayscott.h"
#include "gray-scott/gray_scott_gpu.cuh"

#define GS_BLOCK_SIZE 256

#define GS_CHECK_CUDA(call)                                               \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return GPUCOMPRESS_ERROR_CUDA_FAILED;                         \
        }                                                                 \
    } while (0)

/* ============================================================
 * Internal C++ class
 * ============================================================ */

struct gpucompress_grayscott {
    GrayScottSettings settings;
    float* d_u;
    float* d_v;
    float* d_u2;
    float* d_v2;
    int    current_step;
    int    N;          /* L^3 */
    size_t nbytes;     /* N * sizeof(float) */
};

static int gs_grid_size(int N)
{
    int blocks = (N + GS_BLOCK_SIZE - 1) / GS_BLOCK_SIZE;
    return (blocks < 65535) ? blocks : 65535;
}

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_error_t gpucompress_grayscott_create(
    gpucompress_grayscott_t* handle,
    const GrayScottSettings* settings)
{
    if (!handle || !settings) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_grayscott* sim = new (std::nothrow) gpucompress_grayscott;
    if (!sim) return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;

    sim->settings     = *settings;
    sim->current_step = 0;
    sim->N            = settings->L * settings->L * settings->L;
    sim->nbytes       = (size_t)sim->N * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&sim->d_u,  sim->nbytes); if (err) { delete sim; return GPUCOMPRESS_ERROR_CUDA_FAILED; }
    err = cudaMalloc(&sim->d_v,  sim->nbytes); if (err) { cudaFree(sim->d_u); delete sim; return GPUCOMPRESS_ERROR_CUDA_FAILED; }
    err = cudaMalloc(&sim->d_u2, sim->nbytes); if (err) { cudaFree(sim->d_u); cudaFree(sim->d_v); delete sim; return GPUCOMPRESS_ERROR_CUDA_FAILED; }
    err = cudaMalloc(&sim->d_v2, sim->nbytes); if (err) { cudaFree(sim->d_u); cudaFree(sim->d_v); cudaFree(sim->d_u2); delete sim; return GPUCOMPRESS_ERROR_CUDA_FAILED; }

    *handle = sim;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_destroy(gpucompress_grayscott_t handle)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    cudaFree(handle->d_u);
    cudaFree(handle->d_v);
    cudaFree(handle->d_u2);
    cudaFree(handle->d_v2);
    delete handle;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_init(gpucompress_grayscott_t handle)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    int grid = gs_grid_size(handle->N);
    gs_init_kernel<<<grid, GS_BLOCK_SIZE>>>(
        handle->d_u, handle->d_v,
        handle->settings.L,
        handle->settings.noise,
        (unsigned long long)handle->settings.seed);

    GS_CHECK_CUDA(cudaGetLastError());
    handle->current_step = 0;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_run(gpucompress_grayscott_t handle, int steps)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (steps <= 0) return GPUCOMPRESS_SUCCESS;

    int grid = gs_grid_size(handle->N);
    const GrayScottSettings& s = handle->settings;

    for (int i = 0; i < steps; i++) {
        unsigned long long noise_seed =
            (unsigned long long)s.seed ^ ((unsigned long long)(handle->current_step + i) << 20);

        gs_step_kernel<<<grid, GS_BLOCK_SIZE>>>(
            handle->d_u,  handle->d_v,
            handle->d_u2, handle->d_v2,
            s.L, s.Du, s.Dv, s.F, s.k, s.dt,
            s.noise, noise_seed);

        /* Pointer swap — zero copy */
        std::swap(handle->d_u, handle->d_u2);
        std::swap(handle->d_v, handle->d_v2);
    }

    GS_CHECK_CUDA(cudaGetLastError());
    handle->current_step += steps;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_get_device_ptrs(
    gpucompress_grayscott_t handle,
    float** d_u, float** d_v)
{
    if (!handle || !d_u || !d_v) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *d_u = handle->d_u;
    *d_v = handle->d_v;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_copy_to_host(
    gpucompress_grayscott_t handle,
    float* h_u, float* h_v)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    GS_CHECK_CUDA(cudaDeviceSynchronize());
    if (h_u) { GS_CHECK_CUDA(cudaMemcpy(h_u, handle->d_u, handle->nbytes, cudaMemcpyDeviceToHost)); }
    if (h_v) { GS_CHECK_CUDA(cudaMemcpy(h_v, handle->d_v, handle->nbytes, cudaMemcpyDeviceToHost)); }
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_get_step(
    gpucompress_grayscott_t handle, int* step)
{
    if (!handle || !step) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *step = handle->current_step;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_get_L(
    gpucompress_grayscott_t handle, int* L)
{
    if (!handle || !L) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *L = handle->settings.L;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_grayscott_get_nbytes(
    gpucompress_grayscott_t handle, size_t* nbytes)
{
    if (!handle || !nbytes) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *nbytes = handle->nbytes;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_generate_grayscott(
    const GrayScottSettings* settings,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats)
{
    if (!settings || !d_output || !output_size)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_error_t err;

    err = gpucompress_grayscott_create(&sim, settings);
    if (err != GPUCOMPRESS_SUCCESS) return err;

    err = gpucompress_grayscott_init(sim);
    if (err != GPUCOMPRESS_SUCCESS) { gpucompress_grayscott_destroy(sim); return err; }

    err = gpucompress_grayscott_run(sim, settings->steps);
    if (err != GPUCOMPRESS_SUCCESS) { gpucompress_grayscott_destroy(sim); return err; }

    /* Sync before compression reads the buffer */
    cudaDeviceSynchronize();

    float* d_v = NULL;
    float* d_u = NULL;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

    err = gpucompress_compress_gpu(
        d_v, sim->nbytes,
        d_output, output_size,
        config, stats, NULL);

    gpucompress_grayscott_destroy(sim);
    return err;
}

} /* extern "C" */
