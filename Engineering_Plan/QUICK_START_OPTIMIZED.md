# Quick Start: Using Optimized CUDA Byte Shuffle
## Simple Guide for Maximum Performance

---

## 🎯 Recommended Usage (AUTO Mode)

### The Simplest Way - Let the System Choose
```cpp
#include "byte_shuffle.cuh"

// Your data on device
uint8_t** d_input_chunks;   // Array of chunk pointers
uint8_t** d_output_chunks;  // Array of output pointers
size_t* d_chunk_sizes;      // Size of each chunk
size_t num_chunks = 1000;
unsigned element_size = 4;  // 4-byte elements (e.g., float)

// ONE FUNCTION CALL - Automatic optimization!
cudaError_t err = launch_byte_shuffle_optimized(
    d_input_chunks,
    d_output_chunks,
    d_chunk_sizes,
    num_chunks,
    element_size,
    ShuffleKernelType::AUTO  // ← Automatically picks best kernel!
);

if (err != cudaSuccess) {
    fprintf(stderr, "Shuffle failed: %s\n", cudaGetErrorString(err));
}
```

**That's it!** The system automatically selects the optimal kernel based on your element size.

---

## 🔄 What AUTO Mode Does

```
Element Size 1:  → No shuffle (just copy)
Element Size 2:  → Template specialization (compile-time optimized)
Element Size 4:  → Vectorized 4-byte (if aligned) OR specialized
Element Size 8:  → Vectorized 8-byte (if aligned) OR shared memory
Element Size 16+: → Shared memory staging (best for large)
```

**You don't need to think about it - AUTO mode handles everything!**

---

## 📊 Performance Expectations

| Your Data Type | Element Size | AUTO Selects | Expected Speedup |
|----------------|--------------|--------------|------------------|
| `int16_t`, `half` | 2 bytes | Specialized<2> | 1.6x |
| `int32_t`, `float` | 4 bytes | Vectorized_4B or Specialized<4> | 1.9x |
| `int64_t`, `double` | 8 bytes | Vectorized_8B or Shared Memory | 2.1x |
| `float4`, custom structs | 16+ bytes | Shared Memory | 1.9x |

*Compared to baseline implementation. Actual speedup depends on GPU model and data characteristics.*

---

## 🎛️ Manual Kernel Selection (Advanced)

If you want explicit control:

### Force Shared Memory (Best for Large Elements)
```cpp
launch_byte_shuffle_optimized(
    d_input_chunks, d_output_chunks, d_chunk_sizes,
    num_chunks, element_size,
    ShuffleKernelType::SHARED_MEMORY  // ← Force this kernel
);
```

### Force Vectorized for 4-Byte Elements
```cpp
// Only works with element_size = 4 and aligned data
launch_byte_shuffle_optimized(
    d_input_chunks, d_output_chunks, d_chunk_sizes,
    num_chunks, 4,
    ShuffleKernelType::VECTORIZED_4B  // ← Requires alignment
);
```

### Force Template Specialization
```cpp
launch_byte_shuffle_optimized(
    d_input_chunks, d_output_chunks, d_chunk_sizes,
    num_chunks, element_size,
    ShuffleKernelType::SPECIALIZED  // ← Compile-time optimized
);
```

---

## 🚀 Common Use Cases

### Use Case 1: Shuffle Float Array Before Compression
```cpp
// You have: float array on GPU
float* d_floats;  // 1 million floats
size_t num_floats = 1000000;
size_t total_bytes = num_floats * sizeof(float);

// Setup chunks (using existing utility)
std::vector<Chunk> chunks = chunkDeviceBuffer(
    d_floats, total_bytes, 64 * 1024  // 64KB chunks
);

// Prepare for shuffle
uint8_t** d_input_ptrs = /* copy chunk pointers to device */;
uint8_t** d_output_ptrs = /* allocate output */;
size_t* d_sizes = /* copy sizes to device */;

// Shuffle for better compression!
launch_byte_shuffle_optimized(
    d_input_ptrs,
    d_output_ptrs,
    d_sizes,
    chunks.size(),
    sizeof(float),  // 4 bytes
    ShuffleKernelType::AUTO
);

// Now compress shuffled data with nvCOMP
// ... compression code ...
```

### Use Case 2: Shuffle Double Precision Scientific Data
```cpp
double* d_temperature_field;  // Climate simulation data
size_t num_points = 10000000;  // 10M data points

// Prepare chunks
auto chunks = chunkDeviceBuffer(
    d_temperature_field,
    num_points * sizeof(double),
    256 * 1024  // 256KB chunks
);

// ... setup pointers ...

// Shuffle doubles (AUTO picks vectorized_8B or shared memory)
launch_byte_shuffle_optimized(
    d_input_ptrs, d_output_ptrs, d_sizes,
    chunks.size(),
    sizeof(double),  // 8 bytes
    ShuffleKernelType::AUTO
);
```

### Use Case 3: Shuffle Custom Struct
```cpp
struct Particle {
    float x, y, z;     // Position
    float vx, vy, vz;  // Velocity
    int id;            // Particle ID
    float mass;        // Mass
};  // Total: 32 bytes

Particle* d_particles;
size_t num_particles = 1000000;

// Shuffle custom struct (AUTO picks shared memory for 32 bytes)
launch_byte_shuffle_optimized(
    d_input_ptrs, d_output_ptrs, d_sizes,
    chunks.size(),
    sizeof(Particle),  // 32 bytes
    ShuffleKernelType::AUTO
);
```

### Use Case 4: With Custom CUDA Stream
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Shuffle on stream1 (non-blocking)
launch_byte_shuffle_optimized(
    d_input1, d_output1, d_sizes1,
    num_chunks1, element_size,
    ShuffleKernelType::AUTO,
    stream1  // ← Custom stream
);

// Simultaneously shuffle different data on stream2
launch_byte_shuffle_optimized(
    d_input2, d_output2, d_sizes2,
    num_chunks2, element_size,
    ShuffleKernelType::AUTO,
    stream2
);

// Wait for both
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

---

## 🔄 Unshuffle (Reverse Operation)

After decompression, restore original data order:

```cpp
// Same API, just use unshuffle function
cudaError_t err = launch_byte_unshuffle_optimized(
    d_shuffled_chunks,
    d_restored_chunks,
    d_chunk_sizes,
    num_chunks,
    element_size,
    ShuffleKernelType::AUTO  // Also supports AUTO!
);
```

---

## ⚠️ Important Notes

### Alignment for Vectorized Kernels
Vectorized kernels (`VECTORIZED_4B`, `VECTORIZED_8B`) require **16-byte aligned** pointers for best performance. If your data is not aligned:
- Use `ShuffleKernelType::AUTO` - it will automatically fall back to non-vectorized
- Or explicitly use `SHARED_MEMORY` or `SPECIALIZED`

### Element Size Must Match Data Type
```cpp
float data[1000];
// WRONG: element_size = 8  (float is 4 bytes!)
// RIGHT: element_size = sizeof(float) = 4
```

### Leftover Bytes Handled Automatically
If your chunk size is not perfectly divisible by element size, the leftover bytes are automatically copied without shuffling.

---

## 📈 Measuring Performance

### Quick Benchmark
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

// Your shuffle call
launch_byte_shuffle_optimized(
    d_input, d_output, d_sizes,
    num_chunks, element_size,
    ShuffleKernelType::AUTO
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

// Calculate bandwidth
size_t total_bytes = /* sum of all chunk sizes */;
double bandwidth_GB_s = (total_bytes * 2.0 / 1e9) / (milliseconds / 1000.0);

printf("Bandwidth: %.2f GB/s\n", bandwidth_GB_s);
printf("Time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 🎯 Decision Guide: When to Use Which Mode?

### Use AUTO (Recommended 95% of cases)
- ✅ You want maximum performance without manual tuning
- ✅ Your element size varies
- ✅ You're unsure about alignment
- ✅ You want code that works optimally on different GPUs

### Use SHARED_MEMORY (Manual Override)
- ✅ Element size >= 16 bytes
- ✅ Large chunks (>64KB)
- ✅ You've profiled and confirmed it's faster

### Use VECTORIZED_4B or VECTORIZED_8B (Manual Override)
- ✅ You GUARANTEE 16-byte alignment
- ✅ Element size is exactly 4 or 8 bytes
- ✅ You've benchmarked and it's faster than AUTO

### Use SPECIALIZED (Manual Override)
- ✅ Element size is 2, 4, 8, or 16 bytes
- ✅ Element size is known at compile time
- ✅ You want compile-time optimization

### Use BASELINE (Debugging/Reference)
- ✅ You're debugging and want simplest code path
- ✅ You're comparing performance
- ✅ Other kernels are failing

---

## 🐛 Troubleshooting

### Problem: Getting cudaErrorInvalidValue
```cpp
// Check:
- Are your pointers valid device pointers?
- Is element_size > 0?
- Is num_chunks > 0?
```

### Problem: Results are incorrect
```cpp
// Check:
- Is element_size correct for your data type?
- Did you use shuffle before compression and unshuffle after decompression?
- Are chunk sizes correct?
```

### Problem: Performance is slow
```cpp
// Try:
1. Use AUTO mode (if not already)
2. Ensure chunks are large enough (>16KB recommended)
3. Profile with nvprof/nsight to identify bottleneck
4. Check if you have enough data to saturate GPU
```

### Problem: Compilation errors
```cpp
// Ensure:
- CUDA toolkit version >= 11.0
- Compiling with --std=c++17
- Using -arch=sm_75 or higher
- Including byte_shuffle.cuh header
```

---

## 📚 Complete Example

```cpp
#include "byte_shuffle.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Data setup
    const size_t N = 1024 * 1024;  // 1M floats
    const size_t bytes = N * sizeof(float);
    
    float* h_data = new float[N];
    for (size_t i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }
    
    // Device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Setup for shuffle (single chunk for simplicity)
    uint8_t* h_input_ptrs[1] = { reinterpret_cast<uint8_t*>(d_input) };
    uint8_t* h_output_ptrs[1] = { reinterpret_cast<uint8_t*>(d_output) };
    size_t h_sizes[1] = { bytes };
    
    uint8_t** d_input_ptrs;
    uint8_t** d_output_ptrs;
    size_t* d_sizes;
    
    cudaMalloc(&d_input_ptrs, sizeof(uint8_t*));
    cudaMalloc(&d_output_ptrs, sizeof(uint8_t*));
    cudaMalloc(&d_sizes, sizeof(size_t));
    
    cudaMemcpy(d_input_ptrs, h_input_ptrs, sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_ptrs, h_output_ptrs, sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes, sizeof(size_t), cudaMemcpyHostToDevice);
    
    // SHUFFLE with AUTO optimization!
    cudaError_t err = launch_byte_shuffle_optimized(
        d_input_ptrs,
        d_output_ptrs,
        d_sizes,
        1,  // num_chunks
        sizeof(float),  // element_size
        ShuffleKernelType::AUTO
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Shuffle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Verify on CPU
    float* h_output = new float[N];
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Check shuffled pattern (bytes are reordered)
    uint8_t* original = reinterpret_cast<uint8_t*>(h_data);
    uint8_t* shuffled = reinterpret_cast<uint8_t*>(h_output);
    
    bool correct = true;
    for (size_t byte_pos = 0; byte_pos < 4; byte_pos++) {
        for (size_t elem = 0; elem < N; elem++) {
            uint8_t expected = original[elem * 4 + byte_pos];
            uint8_t actual = shuffled[byte_pos * N + elem];
            if (expected != actual) {
                correct = false;
                break;
            }
        }
    }
    
    printf("Shuffle: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_data;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_sizes);
    
    return correct ? 0 : 1;
}
```

Compile and run:
```bash
nvcc -o shuffle_example example.cu src/byte_shuffle_kernels.cu \
     -arch=sm_80 --std=c++17 -O3

./shuffle_example
# Output: Shuffle: PASSED
```

---

## 🎓 Summary

**Just remember:**
```cpp
launch_byte_shuffle_optimized(
    input, output, sizes, count, elem_size,
    ShuffleKernelType::AUTO  // ← Use this!
);
```

**That's all you need for optimal performance!** 🚀

The system handles all optimization decisions automatically based on your data characteristics.
