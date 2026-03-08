#include <cuda_runtime.h>

__global__ void count_mismatches_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        size_t n,
                                        unsigned long long* count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local++;
    atomicAdd(count, local);
}

extern "C" unsigned long long gpu_compare(const float* d_a, const float* d_b,
                                          size_t n, unsigned long long* d_count)
{
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    count_mismatches_kernel<<<512, 256>>>(d_a, d_b, n, d_count);
    cudaDeviceSynchronize();
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost);
    return h_count;
}
