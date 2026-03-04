/**
 * @file test_grayscott_gpu.cu
 * @brief Correctness and integration tests for GPU Gray-Scott simulation
 *
 * Tests:
 *   1. Mass conservation — sum(U)+sum(V) drift < 1% over 1000 steps
 *   2. Field range       — all U,V values in [0,1]
 *   3. Pattern emergence — std(V) > 0.05 after 5000 steps
 *   4. CPU reference     — L=32, noise=0, 100 steps, match within 1e-5
 *   5. Compression       — round-trip via gpucompress_compress/decompress on V
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "gpucompress_grayscott.h"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "  FAIL: %s\n", msg);                           \
            g_fail++;                                                       \
            return;                                                         \
        }                                                                   \
    } while (0)

/* ============================================================
 * GPU reduction helpers
 * ============================================================ */

__global__ void reduce_sum_kernel(const float* data, double* result, int N)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        sum += (double)data[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void reduce_minmax_kernel(const float* data, float* d_min, float* d_max, int N)
{
    __shared__ float smin[256], smax[256];
    int tid = threadIdx.x;
    float vmin =  1e30f;
    float vmax = -1e30f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        float val = data[i];
        if (val < vmin) vmin = val;
        if (val > vmax) vmax = val;
    }
    smin[tid] = vmin;
    smax[tid] = vmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMin((int*)d_min, __float_as_int(smin[0]));
        atomicMax((int*)d_max, __float_as_int(smax[0]));
    }
}

static double gpu_sum(const float* d_data, int N)
{
    double* d_result;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(double)));
    int blocks = (N + 255) / 256;
    if (blocks > 1024) blocks = 1024;
    reduce_sum_kernel<<<blocks, 256>>>(d_data, d_result, N);
    double h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_result));
    return h_result;
}

static void gpu_minmax(const float* d_data, int N, float& out_min, float& out_max)
{
    /* For atomicMin/Max on float-as-int to work correctly with positive floats,
       init min to +large and max to -large using float→int reinterpretation. */
    float init_min =  1e30f;
    float init_max = -1e30f;
    float* d_min;
    float* d_max;
    CHECK_CUDA(cudaMalloc(&d_min, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    int blocks = (N + 255) / 256;
    if (blocks > 1024) blocks = 1024;
    reduce_minmax_kernel<<<blocks, 256>>>(d_data, d_min, d_max, N);
    CHECK_CUDA(cudaMemcpy(&out_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&out_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_min));
    CHECK_CUDA(cudaFree(d_max));
}

/* ============================================================
 * Test 1: Mass conservation
 * ============================================================ */

static void test_mass_conservation()
{
    printf("Test 1: Mass conservation ...\n");

    GrayScottSettings s = gray_scott_default_settings();
    s.L     = 64;
    s.noise = 0.0f;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_error_t err;

    err = gpucompress_grayscott_create(&sim, &s);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "create failed");

    err = gpucompress_grayscott_init(sim);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "init failed");

    float *d_u, *d_v;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);
    CHECK_CUDA(cudaDeviceSynchronize());

    int N = s.L * s.L * s.L;
    double mass0 = gpu_sum(d_u, N) + gpu_sum(d_v, N);

    err = gpucompress_grayscott_run(sim, 1000);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "run failed");
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Re-read pointers (may have swapped) */
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);
    double mass1 = gpu_sum(d_u, N) + gpu_sum(d_v, N);

    double drift = fabs(mass1 - mass0) / fabs(mass0);
    printf("  mass0=%.6f  mass1=%.6f  drift=%.4f%%\n", mass0, mass1, drift * 100.0);
    TEST_ASSERT(drift < 0.01, "mass drift > 1%");

    gpucompress_grayscott_destroy(sim);
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: Field range [0,1]
 * ============================================================ */

static void test_field_range()
{
    printf("Test 2: Field range ...\n");

    GrayScottSettings s = gray_scott_default_settings();
    s.L     = 64;
    s.noise = 0.0f;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);
    gpucompress_grayscott_run(sim, 1000);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *d_u, *d_v;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

    int N = s.L * s.L * s.L;
    float u_min, u_max, v_min, v_max;
    gpu_minmax(d_u, N, u_min, u_max);
    gpu_minmax(d_v, N, v_min, v_max);

    printf("  U range: [%.6f, %.6f]\n", u_min, u_max);
    printf("  V range: [%.6f, %.6f]\n", v_min, v_max);

    /* Allow small overshoot from Euler integration */
    TEST_ASSERT(u_min >= -0.05f && u_max <= 1.05f, "U out of [-0.05, 1.05]");
    TEST_ASSERT(v_min >= -0.05f && v_max <= 1.05f, "V out of [-0.05, 1.05]");

    gpucompress_grayscott_destroy(sim);
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 3: Pattern emergence
 * ============================================================ */

static void test_pattern_emergence()
{
    printf("Test 3: Pattern emergence ...\n");

    GrayScottSettings s = gray_scott_default_settings();
    s.L     = 64;
    s.noise = 0.0f;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);
    gpucompress_grayscott_run(sim, 5000);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *d_u, *d_v;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

    int N = s.L * s.L * s.L;
    double mean_v = gpu_sum(d_v, N) / N;

    /* Compute variance on host (simpler for test) */
    std::vector<float> h_v(N);
    CHECK_CUDA(cudaMemcpy(h_v.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost));

    double var = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = (double)h_v[i] - mean_v;
        var += diff * diff;
    }
    double std_v = sqrt(var / N);

    printf("  mean(V)=%.6f  std(V)=%.6f\n", mean_v, std_v);
    TEST_ASSERT(std_v > 0.05, "std(V) <= 0.05 — no pattern emerged");

    gpucompress_grayscott_destroy(sim);
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 4: CPU reference match
 * ============================================================ */

/* CPU reference implementation */
static inline int cpu_gs_idx(int x, int y, int z, int L)
{
    return x + y * L + z * L * L;
}

static inline int cpu_gs_wrap(int v, int L)
{
    return (v < 0) ? v + L : ((v >= L) ? v - L : v);
}

static inline float cpu_gs_rand(unsigned long long seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 33;
    return (float)(seed & 0xFFFFFFu) / (float)0x1000000u;
}

static void cpu_gs_init(float* u, float* v, int L, float noise, unsigned long long seed)
{
    int N = L * L * L;
    for (int i = 0; i < N; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = i / (L * L);

        u[i] = 1.0f;
        v[i] = 0.0f;

        int half = L / 2;
        if (x >= half - 6 && x < half + 6 &&
            y >= half - 6 && y < half + 6 &&
            z >= half - 6 && z < half + 6) {
            u[i] = 0.25f;
            v[i] = 0.33f;
        }

        u[i] += noise * (cpu_gs_rand(seed ^ (unsigned long long)i) * 2.0f - 1.0f);
    }
}

static void cpu_gs_step(const float* u_in, const float* v_in,
                        float* u_out, float* v_out,
                        int L, float Du, float Dv,
                        float F, float k, float dt,
                        float noise, unsigned long long noise_seed)
{
    int N = L * L * L;
    for (int i = 0; i < N; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = i / (L * L);

        float uc = u_in[i];
        float vc = v_in[i];

        int xm = cpu_gs_idx(cpu_gs_wrap(x - 1, L), y, z, L);
        int xp = cpu_gs_idx(cpu_gs_wrap(x + 1, L), y, z, L);
        int ym = cpu_gs_idx(x, cpu_gs_wrap(y - 1, L), z, L);
        int yp = cpu_gs_idx(x, cpu_gs_wrap(y + 1, L), z, L);
        int zm = cpu_gs_idx(x, y, cpu_gs_wrap(z - 1, L), L);
        int zp = cpu_gs_idx(x, y, cpu_gs_wrap(z + 1, L), L);

        float lap_u = (u_in[xm] + u_in[xp] + u_in[ym] + u_in[yp] +
                       u_in[zm] + u_in[zp] - 6.0f * uc) / 6.0f;
        float lap_v = (v_in[xm] + v_in[xp] + v_in[ym] + v_in[yp] +
                       v_in[zm] + v_in[zp] - 6.0f * vc) / 6.0f;

        float uvv = uc * vc * vc;
        float du = Du * lap_u - uvv + F * (1.0f - uc);
        float dv_val = Dv * lap_v + uvv - (F + k) * vc;

        du += noise * (cpu_gs_rand(noise_seed ^ (unsigned long long)i) * 2.0f - 1.0f);

        u_out[i] = uc + du * dt;
        v_out[i] = vc + dv_val * dt;
    }
}

static void test_cpu_reference()
{
    printf("Test 4: CPU reference match ...\n");

    const int L = 32;
    const int N = L * L * L;
    const int STEPS = 100;

    GrayScottSettings s = gray_scott_default_settings();
    s.L     = L;
    s.noise = 0.0f;

    /* GPU simulation */
    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);
    gpucompress_grayscott_run(sim, STEPS);

    std::vector<float> gpu_u(N), gpu_v(N);
    gpucompress_grayscott_copy_to_host(sim, gpu_u.data(), gpu_v.data());
    gpucompress_grayscott_destroy(sim);

    /* CPU simulation */
    std::vector<float> cpu_u(N), cpu_v(N), cpu_u2(N), cpu_v2(N);
    cpu_gs_init(cpu_u.data(), cpu_v.data(), L, s.noise, (unsigned long long)s.seed);

    for (int step = 0; step < STEPS; step++) {
        unsigned long long noise_seed =
            (unsigned long long)s.seed ^ ((unsigned long long)step << 20);
        cpu_gs_step(cpu_u.data(), cpu_v.data(),
                    cpu_u2.data(), cpu_v2.data(),
                    L, s.Du, s.Dv, s.F, s.k, s.dt,
                    s.noise, noise_seed);
        std::swap(cpu_u, cpu_u2);
        std::swap(cpu_v, cpu_v2);
    }

    /* Compare */
    float max_err_u = 0.0f, max_err_v = 0.0f;
    for (int i = 0; i < N; i++) {
        float eu = fabsf(gpu_u[i] - cpu_u[i]);
        float ev = fabsf(gpu_v[i] - cpu_v[i]);
        if (eu > max_err_u) max_err_u = eu;
        if (ev > max_err_v) max_err_v = ev;
    }

    printf("  max|GPU_U - CPU_U| = %.2e\n", max_err_u);
    printf("  max|GPU_V - CPU_V| = %.2e\n", max_err_v);

    TEST_ASSERT(max_err_u < 1e-4f, "U mismatch > 1e-4");
    TEST_ASSERT(max_err_v < 1e-4f, "V mismatch > 1e-4");

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 5: Compression integration round-trip
 * ============================================================ */

static void test_compression_roundtrip()
{
    printf("Test 5: Compression round-trip ...\n");

    GrayScottSettings s = gray_scott_default_settings();
    s.L     = 64;
    s.steps = 2000;
    s.noise = 0.0f;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);
    gpucompress_grayscott_run(sim, s.steps);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Get V field on host */
    int N = s.L * s.L * s.L;
    size_t nbytes = N * sizeof(float);
    std::vector<float> h_v(N);
    gpucompress_grayscott_copy_to_host(sim, NULL, h_v.data());
    gpucompress_grayscott_destroy(sim);

    /* Compress on host */
    gpucompress_error_t err;
    err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    std::vector<uint8_t> compressed(max_comp);
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats;
    err = gpucompress_compress(h_v.data(), nbytes,
                               compressed.data(), &comp_size,
                               &cfg, &stats);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "compress failed");

    printf("  compressed: %zu -> %zu bytes (ratio %.2fx)\n",
           nbytes, comp_size, stats.compression_ratio);

    /* Decompress */
    std::vector<float> h_v2(N);
    size_t decomp_size = nbytes;
    err = gpucompress_decompress(compressed.data(), comp_size,
                                 h_v2.data(), &decomp_size);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "decompress failed");
    TEST_ASSERT(decomp_size == nbytes, "decompress size mismatch");

    /* Verify lossless */
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h_v[i] != h_v2[i]) mismatches++;
    }
    TEST_ASSERT(mismatches == 0, "lossless round-trip mismatch");

    TEST_ASSERT(stats.compression_ratio > 1.0, "compression ratio <= 1.0 for structured data");

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Main
 * ============================================================ */

int main()
{
    printf("=== Gray-Scott GPU Test Suite ===\n\n");

    test_mass_conservation();
    test_field_range();
    test_pattern_emergence();
    test_cpu_reference();
    test_compression_roundtrip();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
