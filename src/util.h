#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

/**
 * @brief Chunk structure representing a portion of device memory
 */
struct Chunk {
    void* ptr;      // Pointer to chunk in device memory
    size_t size;    // Size of chunk in bytes
};

/**
 * @brief Chunk a device buffer into smaller pieces
 * 
 * Divides a contiguous device memory buffer into multiple chunks of
 * specified size. Last chunk may be smaller if total_bytes is not
 * evenly divisible by chunk_bytes.
 * 
 * @param device_ptr   Pointer to device memory buffer
 * @param total_bytes  Total size of buffer in bytes
 * @param chunk_bytes  Desired chunk size in bytes
 * 
 * @return Vector of chunks, each with pointer and size
 * 
 * @throws std::invalid_argument if device_ptr is null or chunk_bytes is 0
 * 
 * Example:
 *   float* d_data;
 *   cudaMalloc(&d_data, 1024 * 1024);  // 1MB
 *   auto chunks = chunkDeviceBuffer(d_data, 1024*1024, 64*1024); // 64KB chunks
 *   // Process each chunk...
 */
inline std::vector<Chunk>
chunkDeviceBuffer(void* device_ptr,
                  size_t total_bytes,
                  size_t chunk_bytes)
{
    if (!device_ptr)
        throw std::invalid_argument("device_ptr is null");

    if (chunk_bytes == 0)
        throw std::invalid_argument("chunk_bytes must be > 0");

    if (total_bytes == 0)
        return {};

    std::vector<Chunk> chunks;
    chunks.reserve((total_bytes + chunk_bytes - 1) / chunk_bytes);

    auto base = static_cast<uint8_t*>(device_ptr);

    // Chunk the buffer
    for (size_t offset = 0; offset < total_bytes; offset += chunk_bytes) {
        const size_t remaining = total_bytes - offset;
        const size_t size = std::min(chunk_bytes, remaining);

        chunks.push_back({
            base + offset,
            size
        });
    }

    return chunks;
}

#endif // UTIL_H
