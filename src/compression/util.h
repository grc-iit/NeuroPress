#ifndef UTIL_H
#define UTIL_H

#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Device-side chunk arrays (optimized for GPU kernel launch)
 *
 * This structure holds device-allocated arrays that can be passed
 * directly to GPU kernels, avoiding host-to-device memcpy overhead.
 */
struct DeviceChunkArrays {
    uint8_t** d_input_ptrs;   // Device array of input chunk pointers
    uint8_t** d_output_ptrs;  // Device array of output chunk pointers
    size_t* d_sizes;          // Device array of chunk sizes
    size_t num_chunks;        // Number of chunks

    // Constructor
    DeviceChunkArrays()
        : d_input_ptrs(nullptr)
        , d_output_ptrs(nullptr)
        , d_sizes(nullptr)
        , num_chunks(0)
    {}

    // Cleanup helper
    void free() {
        if (d_input_ptrs) cudaFree(d_input_ptrs);
        if (d_output_ptrs) cudaFree(d_output_ptrs);
        if (d_sizes) cudaFree(d_sizes);
        d_input_ptrs = nullptr;
        d_output_ptrs = nullptr;
        d_sizes = nullptr;
        num_chunks = 0;
    }

    // Destructor
    ~DeviceChunkArrays() {
        free();
    }

    // Prevent copying (device pointers shouldn't be copied casually)
    DeviceChunkArrays(const DeviceChunkArrays&) = delete;
    DeviceChunkArrays& operator=(const DeviceChunkArrays&) = delete;

    // Allow moving
    DeviceChunkArrays(DeviceChunkArrays&& other) noexcept
        : d_input_ptrs(other.d_input_ptrs)
        , d_output_ptrs(other.d_output_ptrs)
        , d_sizes(other.d_sizes)
        , num_chunks(other.num_chunks)
    {
        other.d_input_ptrs = nullptr;
        other.d_output_ptrs = nullptr;
        other.d_sizes = nullptr;
        other.num_chunks = 0;
    }

    DeviceChunkArrays& operator=(DeviceChunkArrays&& other) noexcept {
        if (this != &other) {
            free();
            d_input_ptrs = other.d_input_ptrs;
            d_output_ptrs = other.d_output_ptrs;
            d_sizes = other.d_sizes;
            num_chunks = other.num_chunks;
            other.d_input_ptrs = nullptr;
            other.d_output_ptrs = nullptr;
            other.d_sizes = nullptr;
            other.num_chunks = 0;
        }
        return *this;
    }
};

/**
 * @brief Create device-side chunk arrays directly on GPU.
 *
 * Uses a GPU kernel to populate chunk pointer/size arrays on device,
 * eliminating host-to-device memcpy overhead.
 *
 * Defined in byte_shuffle_kernels.cu.
 */
DeviceChunkArrays createDeviceChunkArrays(
    void* device_input,
    void* device_output,
    size_t total_bytes,
    size_t chunk_bytes,
    cudaStream_t stream = 0);

#endif // UTIL_H
