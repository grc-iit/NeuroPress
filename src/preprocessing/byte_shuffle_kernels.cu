/**
 * @file byte_shuffle_kernels.cu
 * @brief CUDA Byte Shuffle — specialized for 4-byte (float32) elements
 *
 * One warp per chunk, compile-time specialized for ElementSize=4.
 * Each thread handles one byte position (lanes 0-3 active for 4-byte).
 */

#include "preprocessing/byte_shuffle.cuh"
#include <cstring>
#include <vector>

// ============================================================================
// Specialized Shuffle Kernel (compile-time ElementSize)
// ============================================================================

template<unsigned ElementSize>
__global__ void byte_shuffle_kernel_specialized(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_chunks) return;

    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / ElementSize;
    const size_t leftover = chunk_size % ElementSize;

    if constexpr (ElementSize <= 1) {
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE)
            chunk_out[i] = chunk_in[i];
        return;
    }

    if (num_elements <= 1) {
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE)
            chunk_out[i] = chunk_in[i];
        return;
    }

    #pragma unroll
    for (int byte_pos = lane_id; byte_pos < ElementSize; byte_pos += WARP_SIZE) {
        const uint8_t* src = chunk_in + byte_pos;
        uint8_t* dst = chunk_out + (byte_pos * num_elements);

        constexpr size_t UNROLL = 8;
        size_t elem = 0;

        for (; elem + UNROLL <= num_elements; elem += UNROLL) {
            #pragma unroll
            for (size_t u = 0; u < UNROLL; u++)
                dst[elem + u] = src[(elem + u) * ElementSize];
        }
        for (; elem < num_elements; elem++)
            dst[elem] = src[elem * ElementSize];
    }

    if (lane_id == 0 && leftover > 0) {
        const uint8_t* leftover_src = chunk_in + (num_elements * ElementSize);
        uint8_t* leftover_dst = chunk_out + (ElementSize * num_elements);
        #pragma unroll
        for (size_t i = 0; i < leftover; i++)
            leftover_dst[i] = leftover_src[i];
    }
}

// ============================================================================
// Specialized Unshuffle Kernel
// ============================================================================

template<unsigned ElementSize>
__global__ void byte_unshuffle_kernel_specialized(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_chunks) return;

    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / ElementSize;
    const size_t leftover = chunk_size % ElementSize;

    if constexpr (ElementSize <= 1) {
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE)
            chunk_out[i] = chunk_in[i];
        return;
    }

    if (num_elements <= 1) {
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE)
            chunk_out[i] = chunk_in[i];
        return;
    }

    #pragma unroll
    for (int byte_pos = lane_id; byte_pos < ElementSize; byte_pos += WARP_SIZE) {
        const uint8_t* src = chunk_in + (byte_pos * num_elements);
        uint8_t* dst = chunk_out + byte_pos;

        constexpr size_t UNROLL = 8;
        size_t elem = 0;

        for (; elem + UNROLL <= num_elements; elem += UNROLL) {
            #pragma unroll
            for (size_t u = 0; u < UNROLL; u++)
                dst[(elem + u) * ElementSize] = src[elem + u];
        }
        for (; elem < num_elements; elem++)
            dst[elem * ElementSize] = src[elem];
    }

    if (lane_id == 0 && leftover > 0) {
        const uint8_t* leftover_src = chunk_in + (ElementSize * num_elements);
        uint8_t* leftover_dst = chunk_out + (num_elements * ElementSize);
        #pragma unroll
        for (size_t i = 0; i < leftover; i++)
            leftover_dst[i] = leftover_src[i];
    }
}

// Explicit instantiations for supported element sizes
template __global__ void byte_shuffle_kernel_specialized<1>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_shuffle_kernel_specialized<2>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_shuffle_kernel_specialized<4>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_shuffle_kernel_specialized<8>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_unshuffle_kernel_specialized<1>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_unshuffle_kernel_specialized<2>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_unshuffle_kernel_specialized<4>(
    const uint8_t**, uint8_t**, const size_t*, size_t);
template __global__ void byte_unshuffle_kernel_specialized<8>(
    const uint8_t**, uint8_t**, const size_t*, size_t);

// ============================================================================
// Device Chunk Array Helpers
// ============================================================================

__global__ void populateChunkArraysKernel(
    uint8_t** d_input_ptrs,
    uint8_t** d_output_ptrs,
    size_t* d_sizes,
    uint8_t* base_input,
    uint8_t* base_output,
    size_t total_bytes,
    size_t chunk_bytes,
    size_t num_chunks)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;

    const size_t offset = idx * chunk_bytes;
    const size_t remaining = total_bytes - offset;
    const size_t size = (remaining < chunk_bytes) ? remaining : chunk_bytes;

    d_input_ptrs[idx] = base_input + offset;
    d_output_ptrs[idx] = base_output + offset;
    d_sizes[idx] = size;
}

DeviceChunkArrays createDeviceChunkArrays(
    void* device_input,
    void* device_output,
    size_t total_bytes,
    size_t chunk_bytes,
    cudaStream_t stream)
{
    if (!device_input || !device_output)
        throw std::invalid_argument("device pointers are null");
    if (chunk_bytes == 0)
        throw std::invalid_argument("chunk_bytes must be > 0");
    if (total_bytes == 0)
        return DeviceChunkArrays();

    const size_t num_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;

    auto base_input = static_cast<uint8_t*>(device_input);
    auto base_output = static_cast<uint8_t*>(device_output);

    DeviceChunkArrays result;
    result.num_chunks = num_chunks;

    cudaError_t err;

    err = cudaMalloc(&result.d_input_ptrs, num_chunks * sizeof(uint8_t*));
    if (err != cudaSuccess)
        throw std::runtime_error("Failed to allocate d_input_ptrs");

    err = cudaMalloc(&result.d_output_ptrs, num_chunks * sizeof(uint8_t*));
    if (err != cudaSuccess)
        throw std::runtime_error("Failed to allocate d_output_ptrs");

    err = cudaMalloc(&result.d_sizes, num_chunks * sizeof(size_t));
    if (err != cudaSuccess)
        throw std::runtime_error("Failed to allocate d_sizes");

    const int threads_per_block = 256;
    const int num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

    populateChunkArraysKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        result.d_input_ptrs, result.d_output_ptrs, result.d_sizes,
        base_input, base_output, total_bytes, chunk_bytes, num_chunks);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("Failed to launch populateChunkArraysKernel");

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        throw std::runtime_error("Kernel execution failed");

    return result;
}

// ============================================================================
// Low-Level Launch Functions
// ============================================================================

cudaError_t launch_byte_shuffle(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size,
    cudaStream_t stream
) {
    if (num_chunks == 0)
        return cudaSuccess;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
    const int num_blocks = (num_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    switch (element_size) {
        case 1: byte_shuffle_kernel_specialized<1><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        case 2: byte_shuffle_kernel_specialized<2><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        case 8: byte_shuffle_kernel_specialized<8><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        default: byte_shuffle_kernel_specialized<4><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
    }

    return cudaGetLastError();
}

cudaError_t launch_byte_unshuffle(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size,
    cudaStream_t stream
) {
    if (num_chunks == 0)
        return cudaSuccess;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
    const int num_blocks = (num_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    switch (element_size) {
        case 1: byte_unshuffle_kernel_specialized<1><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        case 2: byte_unshuffle_kernel_specialized<2><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        case 8: byte_unshuffle_kernel_specialized<8><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
        default: byte_unshuffle_kernel_specialized<4><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_chunks, output_chunks, chunk_sizes, num_chunks); break;
    }

    return cudaGetLastError();
}

// ============================================================================
// Simple High-Level API
// ============================================================================

uint8_t* byte_shuffle_simple(
    void* device_input,
    size_t total_bytes,
    unsigned element_size,
    size_t chunk_bytes,
    cudaStream_t stream
) {
    if (!device_input || total_bytes == 0)
        return nullptr;

    uint8_t* device_output = nullptr;
    cudaError_t err = cudaMalloc(&device_output, total_bytes);
    if (err != cudaSuccess)
        return nullptr;

    DeviceChunkArrays arrays;
    try {
        arrays = createDeviceChunkArrays(device_input, device_output,
                                         total_bytes, chunk_bytes, stream);
    } catch (const std::exception&) {
        cudaFree(device_output);
        return nullptr;
    }

    if (arrays.num_chunks == 0) {
        cudaFree(device_output);
        return nullptr;
    }

    err = launch_byte_shuffle(
        const_cast<const uint8_t**>(arrays.d_input_ptrs),
        arrays.d_output_ptrs, arrays.d_sizes,
        arrays.num_chunks, element_size, stream);

    if (err != cudaSuccess) {
        cudaFree(device_output);
        return nullptr;
    }

    cudaStreamSynchronize(stream);
    return device_output;
}

uint8_t* byte_unshuffle_simple(
    void* device_input,
    size_t total_bytes,
    unsigned element_size,
    size_t chunk_bytes,
    cudaStream_t stream
) {
    if (!device_input || total_bytes == 0)
        return nullptr;

    uint8_t* device_output = nullptr;
    cudaError_t err = cudaMalloc(&device_output, total_bytes);
    if (err != cudaSuccess)
        return nullptr;

    DeviceChunkArrays arrays;
    try {
        arrays = createDeviceChunkArrays(device_input, device_output,
                                         total_bytes, chunk_bytes, stream);
    } catch (const std::exception&) {
        cudaFree(device_output);
        return nullptr;
    }

    if (arrays.num_chunks == 0) {
        cudaFree(device_output);
        return nullptr;
    }

    err = launch_byte_unshuffle(
        const_cast<const uint8_t**>(arrays.d_input_ptrs),
        arrays.d_output_ptrs, arrays.d_sizes,
        arrays.num_chunks, element_size, stream);

    if (err != cudaSuccess) {
        cudaFree(device_output);
        return nullptr;
    }

    cudaStreamSynchronize(stream);
    return device_output;
}
