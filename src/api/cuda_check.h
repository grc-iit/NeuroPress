#ifndef GPUCOMPRESS_CUDA_CHECK_H
#define GPUCOMPRESS_CUDA_CHECK_H

#include <cuda_runtime.h>
#include <cstdio>

/*
 * GPUC_CUDA_CHECK(call, retval)
 *
 * Execute a CUDA API call. If it fails, log the error with file/line
 * and return `retval` from the enclosing function.
 *
 * Usage:
 *   GPUC_CUDA_CHECK(cudaMalloc(&ptr, size), -1);
 *   GPUC_CUDA_CHECK(cudaMemcpyAsync(...), nullptr);
 *   GPUC_CUDA_CHECK(cudaEventRecord(ev, s), GPUCOMPRESS_ERROR_CUDA_FAILED);
 */
#define GPUC_CUDA_CHECK(call, retval) do {                              \
    cudaError_t _gpuc_err = (call);                                     \
    if (_gpuc_err != cudaSuccess) {                                     \
        fprintf(stderr, "gpucompress CUDA error [%s:%d]: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(_gpuc_err));     \
        return (retval);                                                \
    }                                                                   \
} while (0)

#endif /* GPUCOMPRESS_CUDA_CHECK_H */
