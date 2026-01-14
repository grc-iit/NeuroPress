# Generic CUDA Byte Shuffle Implementation Plan
## Universal Data Transformation for Compression Preprocessing

---

## 1. OVERVIEW

### Objective
Implement a **universal byte-level shuffle transformation** on GPU using CUDA with a **one warp per chunk** strategy to:
- Improve compression ratios for any data type
- Work with arbitrary binary data (integers, floats, structs, etc.)
- Avoid global synchronization
- Keep synchronization within warp boundaries (32 threads)
- Process multiple chunks in parallel across different warps
- Maximize memory coalescing and warp efficiency

### What is Byte Shuffling?
Byte shuffling is a **data-type agnostic** transformation that reorganizes data by grouping bytes at the same position across multiple elements. This increases byte-level redundancy, significantly improving compression ratios.

**Used in:** HDF5 (H5Z_FILTER_SHUFFLE), Blosc, Apache Parquet (BYTE_STREAM_SPLIT), Zarr, and custom compression pipelines.

### Key Constraint
**ONE WARP (32 threads) = ONE CHUNK**
- Each warp operates independently on its assigned chunk
- No cross-warp synchronization needed
- Warp-level primitives only (`__syncwarp()`)

---

## 2. ALGORITHM UNDERSTANDING

### What Byte Shuffle Does
**Reorganizes any binary data by transposing bytes to improve compression**

The algorithm operates **purely at the byte level** and works with ANY data type:
- Primitive types: `int8_t`, `int16_t`, `int32_t`, `int64_t`, `float`, `double`
- Complex types: `struct Vec3 { float x,y,z; }`, `struct Particle { ... }`
- Arrays: `float[4]`, `int[10]`
- Arbitrary binary blobs

#### Example (4-byte elements - int32, float32, etc.):
```
INPUT (interleaved by element):
  Element 0: [A0 A1 A2 A3]
  Element 1: [B0 B1 B2 B3]
  Element 2: [C0 C1 C2 C3]
  Memory: A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3

OUTPUT (de-interleaved by byte position):
  Byte pos 0: [A0 B0 C0]
  Byte pos 1: [A1 B1 C1]
  Byte pos 2: [A2 B2 C2]
  Byte pos 3: [A3 B3 C3]
  Memory: A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3
```

**Why this helps compression:**
- Bytes at the same position often have similar values (e.g., high bytes of integers)
- Grouping similar bytes increases redundancy
- Better run-length encoding and entropy compression
- 10-50% better compression ratios typical

### Sequential Algorithm (Reference Implementation)
```c
// SHUFFLE (used in HDF5, Blosc, Parquet, etc.)
for (byte_pos = 0; byte_pos < element_size; byte_pos++) {
    src = input + byte_pos;
    for (elem = 0; elem < num_elements; elem++) {
        *dest++ = *src;
        src += element_size;  // Jump to same byte in next element
    }
}

// UNSHUFFLE (reverse transformation)
for (byte_pos = 0; byte_pos < element_size; byte_pos++) {
    dest = output + byte_pos;
    for (elem = 0; elem < num_elements; elem++) {
        *dest = *src++;
        dest += element_size;
    }
}
```

### Key Parameters
- `element_size`: Size of each element in bytes (1, 2, 4, 8, 16, or any fixed size)
- `num_elements`: Number of elements in chunk
- `chunk_size`: Total bytes in chunk = `element_size * num_elements + leftover`
- `leftover`: Extra bytes if `chunk_size % element_size != 0`

### Universal Examples

#### Example 1: Float array
```cpp
float data[1000];  // element_size = 4, num_elements = 1000
shuffle(data, 4000, 4);
```

#### Example 2: Custom struct
```cpp
struct Particle { double x,y,z; int id; };  // element_size = 28
Particle particles[10000];
shuffle(particles, 280000, 28);
```

#### Example 3: Image pixels
```cpp
struct RGB { uint8_t r,g,b; };  // element_size = 3
RGB pixels[1920*1080];
shuffle(pixels, 1920*1080*3, 3);  // Separates R, G, B channels
```

---

## 3. CUDA KERNEL ARCHITECTURE

### Kernel Launch Configuration
```cuda
// One warp per chunk
dim3 blocks((num_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
dim3 threads(WARP_SIZE * WARPS_PER_BLOCK);  // e.g., 32 * 4 = 128

byte_shuffle_kernel<<<blocks, threads>>>(
    input_chunks,
    output_chunks,
    chunk_sizes,
    num_chunks,
    element_size
);
```

### Thread-to-Chunk Mapping
```cuda
// Identify which warp this thread belongs to
int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
int lane_id = threadIdx.x % WARP_SIZE;  // 0-31

if (warp_id >= num_chunks) return;  // Out of bounds

// This warp processes chunk[warp_id]
void* chunk_input = input_chunks[warp_id];
void* chunk_output = output_chunks[warp_id];
size_t chunk_size = chunk_sizes[warp_id];
```

---

## 4. PARALLEL STRATEGIES

### Strategy A: Outer Loop Parallelization (RECOMMENDED)
**Parallelize over byte positions, cooperate on elements**

#### Advantages:
- Better for large element sizes (4, 8, 16 bytes)
- Each thread handles contiguous output region
- Minimal inter-thread communication
- Works well when `bytesoftype <= 32`

#### Algorithm:
```cuda
// Each thread handles specific byte positions
for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
    uint8_t* src = chunk_input + byte_pos;
    uint8_t* dst = chunk_output + (byte_pos * num_elements);
    
    // This thread extracts byte_pos from all elements
    for (size_t elem = 0; elem < num_elements; elem++) {
        dst[elem] = src[elem * element_size];
    }
}
```

#### Memory Access Pattern:
```
Thread 0: byte_pos = 0, 32, 64, ...
Thread 1: byte_pos = 1, 33, 65, ...
...
Thread 31: byte_pos = 31, 63, 95, ...
```

---

### Strategy B: Inner Loop Parallelization
**Parallelize over elements, serialize byte positions**

#### Advantages:
- Better for small element sizes (2, 4 bytes)
- Better for very large chunks (many elements)
- More coalesced reads

#### Algorithm:
```cuda
for (int byte_pos = 0; byte_pos < element_size; byte_pos++) {
    // All threads cooperate on this byte position
    for (size_t elem = lane_id; elem < num_elements; elem += WARP_SIZE) {
        uint8_t* src = chunk_input + (elem * element_size) + byte_pos;
        uint8_t* dst = chunk_output + (byte_pos * num_elements) + elem;
        *dst = *src;
    }
    __syncwarp();  // Ensure all threads finish before next byte_pos
}
```

#### Memory Access Pattern:
```
Iteration byte_pos=0:
  Thread 0: element 0
  Thread 1: element 1
  ...
  Thread 31: element 31
```

---

### Strategy C: Hybrid Approach
**Adaptive strategy based on element size**

```cuda
if (element_size >= 16) {
    // Use Strategy A (outer loop parallel)
    // Better for large elements
} else if (num_elements > 512) {
    // Use Strategy B (inner loop parallel)
    // Better for many elements
} else {
    // Use Strategy A (default)
}
```

---

## 5. MEMORY ACCESS OPTIMIZATION

Memory bandwidth is the primary bottleneck for shuffle operations. These optimizations maximize throughput by improving memory access patterns.

### 5.1 Understanding Memory Access Patterns

#### Baseline (Strategy A) - Partial Coalescing
```cuda
// Output writes (coalesced ✓)
for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
    uint8_t* dst = chunk_out + (byte_pos * num_elements);
    // dst[0], dst[1], dst[2]... - sequential writes = COALESCED
    for (size_t elem = 0; elem < num_elements; elem++) {
        dst[elem] = src[elem * element_size];  // Sequential write
    }
}

// Input reads (strided ✗)
src[elem * element_size]  // Stride = element_size (typically 4-16 bytes)
// Not coalesced! Threads access memory with gaps
```

**Performance Impact:**
- Output writes: ~80-90% of peak bandwidth (good)
- Input reads: ~30-50% of peak bandwidth (poor for large strides)
- **Overall: ~50-70% efficiency**

---

### 5.2 Optimization Technique #1: Shared Memory Staging

**Goal:** Convert strided global reads into coalesced reads + fast shared memory access

#### Implementation
```cuda
#define SMEM_SIZE (32 * 1024)  // 32KB shared memory per block

__global__ void byte_shuffle_kernel_smem(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    // Shared memory staging buffer (per block)
    __shared__ uint8_t smem[SMEM_SIZE];
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / element_size;
    
    // Process chunk in tiles that fit in shared memory
    const size_t tile_size = SMEM_SIZE / (blockDim.x / WARP_SIZE);  // Per-warp allocation
    uint8_t* warp_smem = smem + (warp_in_block * tile_size);
    
    for (size_t tile_offset = 0; tile_offset < chunk_size; tile_offset += tile_size) {
        const size_t tile_bytes = min(tile_size, chunk_size - tile_offset);
        
        // STEP 1: Coalesced load from global to shared memory
        for (size_t i = lane_id; i < tile_bytes; i += WARP_SIZE) {
            warp_smem[i] = chunk_in[tile_offset + i];  // COALESCED!
        }
        __syncwarp();
        
        // STEP 2: Shuffle from shared memory (fast, no coalescing issues)
        const size_t tile_elements = tile_bytes / element_size;
        for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
            const uint8_t* src = warp_smem + byte_pos;
            uint8_t* dst = chunk_out + (byte_pos * num_elements) + 
                           (tile_offset / element_size);
            
            for (size_t elem = 0; elem < tile_elements; elem++) {
                dst[elem] = src[elem * element_size];
            }
        }
        __syncwarp();
    }
}
```

**Benefits:**
- ✅ All global reads are coalesced
- ✅ Shared memory latency ~20x lower than global
- ✅ 30-50% speedup for element_size > 4

**When to Use:**
- Element sizes ≥ 8 bytes
- Large chunks (> 64KB)
- Memory-bound workloads

---

### 5.3 Optimization Technique #2: Vectorized Memory Access

**Goal:** Use wider memory transactions (64-bit, 128-bit) to reduce instruction count

#### Vector Type Specialization
```cuda
// Template for vectorized access based on element size
template<typename VecType, int ElementSize>
__device__ void shuffle_vectorized(
    const uint8_t* chunk_in,
    uint8_t* chunk_out,
    size_t num_elements,
    int lane_id
) {
    constexpr int VEC_SIZE = sizeof(VecType);
    const int elements_per_vec = VEC_SIZE / ElementSize;
    
    // Ensure alignment
    if ((uintptr_t)chunk_in % VEC_SIZE != 0 || 
        (uintptr_t)chunk_out % VEC_SIZE != 0) {
        // Fall back to byte-by-byte
        return;
    }
    
    const VecType* vec_in = reinterpret_cast<const VecType*>(chunk_in);
    const size_t num_vecs = (num_elements * ElementSize) / VEC_SIZE;
    
    // Process multiple elements per vector load
    for (int byte_pos = lane_id; byte_pos < ElementSize; byte_pos += WARP_SIZE) {
        for (size_t vec_idx = 0; vec_idx < num_vecs; vec_idx++) {
            VecType vec_data = vec_in[vec_idx];  // Load 64/128 bits at once
            uint8_t* vec_bytes = reinterpret_cast<uint8_t*>(&vec_data);
            
            // Extract bytes at position byte_pos from this vector
            for (int i = 0; i < elements_per_vec; i++) {
                size_t elem_id = vec_idx * elements_per_vec + i;
                chunk_out[byte_pos * num_elements + elem_id] = 
                    vec_bytes[i * ElementSize + byte_pos];
            }
        }
    }
}

// Kernel with vectorized access
__global__ void byte_shuffle_kernel_vectorized(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / element_size;
    
    // Select vector size based on element size and alignment
    if (element_size == 4 && num_elements >= 4) {
        shuffle_vectorized<uint4, 4>(chunk_in, chunk_out, num_elements, lane_id);
    } else if (element_size == 8 && num_elements >= 2) {
        shuffle_vectorized<uint2, 8>(chunk_in, chunk_out, num_elements, lane_id);
    } else if (element_size == 2 && num_elements >= 8) {
        shuffle_vectorized<uint4, 2>(chunk_in, chunk_out, num_elements, lane_id);
    } else {
        // Fall back to standard implementation
        shuffle_standard(chunk_in, chunk_out, num_elements, element_size, lane_id);
    }
}
```

**Benefits:**
- ✅ Fewer memory transactions (4x-8x reduction)
- ✅ Better instruction-level parallelism
- ✅ 15-30% speedup for aligned data

**When to Use:**
- Aligned data (pointers divisible by 8 or 16)
- Common element sizes (2, 4, 8 bytes)
- Compute-bound scenarios

---

### 5.4 Optimization Technique #3: Prefetching with Double Buffering

**Goal:** Hide memory latency by overlapping memory loads with computation

```cuda
__global__ void byte_shuffle_kernel_prefetch(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    __shared__ uint8_t smem[2][16 * 1024];  // Double buffer
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    
    const size_t tile_size = 16 * 1024;
    const size_t num_tiles = (chunk_size + tile_size - 1) / tile_size;
    
    int current_buffer = 0;
    
    // Prefetch first tile
    if (num_tiles > 0) {
        for (size_t i = lane_id; i < min(tile_size, chunk_size); i += WARP_SIZE) {
            smem[current_buffer][i] = chunk_in[i];
        }
    }
    __syncwarp();
    
    // Pipeline: load next tile while processing current tile
    for (size_t tile = 0; tile < num_tiles; tile++) {
        const int process_buffer = current_buffer;
        const int load_buffer = 1 - current_buffer;
        
        // Launch prefetch for next tile (if exists)
        if (tile + 1 < num_tiles) {
            size_t next_offset = (tile + 1) * tile_size;
            size_t next_bytes = min(tile_size, chunk_size - next_offset);
            
            // Async load into next buffer
            for (size_t i = lane_id; i < next_bytes; i += WARP_SIZE) {
                smem[load_buffer][i] = chunk_in[next_offset + i];
            }
        }
        
        // Process current tile from smem[process_buffer]
        size_t tile_offset = tile * tile_size;
        size_t tile_bytes = min(tile_size, chunk_size - tile_offset);
        size_t tile_elements = tile_bytes / element_size;
        
        for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
            uint8_t* src = smem[process_buffer] + byte_pos;
            uint8_t* dst = chunk_out + (byte_pos * (chunk_size/element_size)) + 
                           (tile_offset / element_size);
            
            for (size_t elem = 0; elem < tile_elements; elem++) {
                dst[elem] = src[elem * element_size];
            }
        }
        
        __syncwarp();
        current_buffer = load_buffer;
    }
}
```

**Benefits:**
- ✅ Overlaps memory transfer with computation
- ✅ Reduces memory stalls
- ✅ 10-20% speedup for large chunks

---

### 5.5 Optimization Technique #4: Warp Shuffle Instructions

**Goal:** Use intrinsics for small element sizes to avoid shared memory

```cuda
__device__ void shuffle_with_warp_shuffle(
    const uint8_t* chunk_in,
    uint8_t* chunk_out,
    size_t num_elements,
    unsigned element_size,
    int lane_id
) {
    // Works best for element_size == 4 (int/float)
    if (element_size == 4 && num_elements % WARP_SIZE == 0) {
        const uint32_t* in32 = reinterpret_cast<const uint32_t*>(chunk_in);
        
        for (size_t batch = 0; batch < num_elements / WARP_SIZE; batch++) {
            // Each thread loads one 4-byte element
            uint32_t my_elem = in32[batch * WARP_SIZE + lane_id];
            
            // Extract bytes using warp shuffle
            for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
                uint8_t my_byte = (my_elem >> (byte_pos * 8)) & 0xFF;
                
                // Collect this byte from all threads
                // Write to output
                for (int src_lane = 0; src_lane < WARP_SIZE; src_lane++) {
                    uint8_t byte = __shfl_sync(0xFFFFFFFF, my_byte, src_lane);
                    if (lane_id == 0) {
                        chunk_out[byte_pos * num_elements + batch * WARP_SIZE + src_lane] = byte;
                    }
                }
            }
        }
    }
}
```

**Benefits:**
- ✅ No shared memory usage
- ✅ Ultra-low latency communication
- ✅ Best for small, regular patterns

**Limitations:**
- Only practical for element_size ≤ 4
- Requires careful synchronization

---

### 5.6 Summary: Optimization Comparison

| Technique | Best For | Speedup | Complexity | Memory |
|-----------|----------|---------|------------|--------|
| **Baseline** | General use | 1.0x | Low | 2x bandwidth |
| **Shared Memory Staging** | Large elements (8-16B) | 1.3-1.5x | Medium | +32KB smem |
| **Vectorized Access** | Aligned 2/4/8B | 1.15-1.3x | Medium | Same |
| **Double Buffering** | Large chunks (>256KB) | 1.1-1.2x | High | +32KB smem |
| **Warp Shuffle** | 4-byte elements | 1.2-1.4x | High | Zero smem |

**Recommendation:** Use shared memory staging for general case, add vectorization when alignment is guaranteed.

---

## 6. DETAILED KERNEL IMPLEMENTATION

### Kernel Signature
```cuda
__global__ void byte_shuffle_kernel(
    const uint8_t** input_chunks,    // Array of input chunk pointers
    uint8_t** output_chunks,         // Array of output chunk pointers
    const size_t* chunk_sizes,       // Size of each chunk in bytes
    size_t num_chunks,               // Total number of chunks
    unsigned element_size            // Element size in bytes (stride)
)
```

### Full Kernel Code (Strategy A - Recommended)
```cuda
__global__ void byte_shuffle_kernel(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    // Identify warp and lane
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Bounds check
    if (warp_id >= num_chunks) return;
    
    // Get this warp's chunk
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    
    // Calculate elements and leftover
    const size_t num_elements = chunk_size / element_size;
    const size_t leftover = chunk_size % element_size;
    
    // Early exit for trivial cases
    if (element_size <= 1 || num_elements <= 1) {
        // Just copy data (no benefit from shuffling)
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE) {
            chunk_out[i] = chunk_in[i];
        }
        return;
    }
    
    // SHUFFLE: Each thread handles certain byte positions
    for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
        const uint8_t* src = chunk_in + byte_pos;
        uint8_t* dst = chunk_out + (byte_pos * num_elements);
        
        // Extract byte_pos from all elements
        for (size_t elem = 0; elem < num_elements; elem++) {
            dst[elem] = src[elem * element_size];
        }
    }
    
    // Handle leftover bytes (last thread handles this)
    if (lane_id == 0 && leftover > 0) {
        const uint8_t* leftover_src = chunk_in + (num_elements * element_size);
        uint8_t* leftover_dst = chunk_out + (element_size * num_elements);
        for (size_t i = 0; i < leftover; i++) {
            leftover_dst[i] = leftover_src[i];
        }
    }
}
```

### Unshuffle Kernel (Reverse Transformation)
```cuda
__global__ void byte_unshuffle_kernel(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    
    const size_t num_elements = chunk_size / element_size;
    const size_t leftover = chunk_size % element_size;
    
    if (element_size <= 1 || num_elements <= 1) {
        for (size_t i = lane_id; i < chunk_size; i += WARP_SIZE) {
            chunk_out[i] = chunk_in[i];
        }
        return;
    }
    
    // UNSHUFFLE: Reconstruct interleaved format
    for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
        const uint8_t* src = chunk_in + (byte_pos * num_elements);
        uint8_t* dst = chunk_out + byte_pos;
        
        for (size_t elem = 0; elem < num_elements; elem++) {
            dst[elem * element_size] = src[elem];
        }
    }
    
    // Handle leftover
    if (lane_id == 0 && leftover > 0) {
        const uint8_t* leftover_src = chunk_in + (element_size * num_elements);
        uint8_t* leftover_dst = chunk_out + (num_elements * element_size);
        for (size_t i = 0; i < leftover; i++) {
            leftover_dst[i] = leftover_src[i];
        }
    }
}
```

---

## 7. HOST-SIDE INTEGRATION

### Data Structure for Chunks
```cpp
struct ChunkBatch {
    std::vector<uint8_t*> input_ptrs;   // Device pointers
    std::vector<uint8_t*> output_ptrs;  // Device pointers
    std::vector<size_t> sizes;          // Chunk sizes
    size_t num_chunks;
    unsigned element_size;
};
```

### Preparation Function
```cpp
void prepare_shuffle_batch(
    void* device_buffer,
    size_t total_size,
    size_t chunk_size,
    unsigned element_size,
    ChunkBatch& batch
) {
    // Use chunkDeviceBuffer from util.cu
    std::vector<Chunk> chunks = chunkDeviceBuffer(
        device_buffer, total_size, chunk_size
    );
    
    batch.num_chunks = chunks.size();
    batch.element_size = element_size;
    
    // Allocate output buffer
    uint8_t* output_buffer;
    cudaMalloc(&output_buffer, total_size);
    
    // Build pointer arrays (host side)
    std::vector<uint8_t*> h_input_ptrs(chunks.size());
    std::vector<uint8_t*> h_output_ptrs(chunks.size());
    std::vector<size_t> h_sizes(chunks.size());
    
    size_t output_offset = 0;
    for (size_t i = 0; i < chunks.size(); i++) {
        h_input_ptrs[i] = static_cast<uint8_t*>(chunks[i].ptr);
        h_output_ptrs[i] = output_buffer + output_offset;
        h_sizes[i] = chunks[i].size;
        output_offset += chunks[i].size;
    }
    
    // Copy to device
    uint8_t** d_input_ptrs;
    uint8_t** d_output_ptrs;
    size_t* d_sizes;
    
    cudaMalloc(&d_input_ptrs, chunks.size() * sizeof(uint8_t*));
    cudaMalloc(&d_output_ptrs, chunks.size() * sizeof(uint8_t*));
    cudaMalloc(&d_sizes, chunks.size() * sizeof(size_t));
    
    cudaMemcpy(d_input_ptrs, h_input_ptrs.data(), 
               chunks.size() * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_ptrs, h_output_ptrs.data(), 
               chunks.size() * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes.data(), 
               chunks.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    
    batch.input_ptrs = {d_input_ptrs};
    batch.output_ptrs = {d_output_ptrs};
    batch.sizes = {d_sizes};
}
```

### Launch Function
```cpp
void launch_byte_shuffle(const ChunkBatch& batch) {
    const int WARPS_PER_BLOCK = 4;
    const int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
    
    int num_blocks = (batch.num_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    byte_shuffle_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        batch.input_ptrs[0],
        batch.output_ptrs[0],
        batch.sizes[0],
        batch.num_chunks,
        batch.element_size
    );
    
    cudaDeviceSynchronize();
}
```

---

## 8. ADVANCED OPTIMIZATION TECHNIQUES

### 8.1 Loop Unrolling with Register Blocking

**Goal:** Reduce loop overhead and enable instruction-level parallelism

#### Basic Unrolling
```cuda
// WITHOUT unrolling (high overhead)
for (size_t elem = 0; elem < num_elements; elem++) {
    dst[elem] = src[elem * element_size];
    // Loop control, bounds check, increment - every iteration
}

// WITH unrolling (lower overhead)
const int UNROLL = 8;
size_t elem = 0;

// Main loop: process 8 elements at once
for (; elem + UNROLL <= num_elements; elem += UNROLL) {
    uint8_t temp[UNROLL];
    
    // Load phase (can be parallelized by compiler)
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        temp[u] = src[(elem + u) * element_size];
    }
    
    // Store phase (sequential writes = coalesced)
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        dst[elem + u] = temp[u];
    }
}

// Tail loop: remaining elements
for (; elem < num_elements; elem++) {
    dst[elem] = src[elem * element_size];
}
```

**Benefits:**
- ✅ 8x fewer loop iterations
- ✅ Better instruction pipelining
- ✅ More opportunities for compiler optimization
- ✅ 10-15% speedup typical

---

### 8.2 Vectorized Memory Operations

**Goal:** Use wide memory transactions to maximize bus utilization

#### Type-Specific Vectorization

```cuda
// Specialization for 4-byte elements (int32, float)
__device__ void shuffle_4byte_vectorized(
    const uint8_t* chunk_in,
    uint8_t* chunk_out,
    size_t num_elements,
    int lane_id
) {
    // Use float4 for 128-bit transactions (4x int32)
    const float4* in_vec = reinterpret_cast<const float4*>(chunk_in);
    const size_t num_vec4 = num_elements / 4;
    
    for (int byte_pos = lane_id; byte_pos < 4; byte_pos += WARP_SIZE) {
        uint8_t* dst = chunk_out + (byte_pos * num_elements);
        
        for (size_t v = 0; v < num_vec4; v++) {
            float4 vec = in_vec[v];  // Load 16 bytes (4 elements) at once
            uint32_t* elements = reinterpret_cast<uint32_t*>(&vec);
            
            // Extract byte_pos from each of 4 elements
            dst[v * 4 + 0] = (elements[0] >> (byte_pos * 8)) & 0xFF;
            dst[v * 4 + 1] = (elements[1] >> (byte_pos * 8)) & 0xFF;
            dst[v * 4 + 2] = (elements[2] >> (byte_pos * 8)) & 0xFF;
            dst[v * 4 + 3] = (elements[3] >> (byte_pos * 8)) & 0xFF;
        }
        
        // Handle remaining elements
        size_t remaining = num_elements % 4;
        const uint32_t* in32 = reinterpret_cast<const uint32_t*>(chunk_in);
        for (size_t i = 0; i < remaining; i++) {
            size_t elem = num_vec4 * 4 + i;
            dst[elem] = (in32[elem] >> (byte_pos * 8)) & 0xFF;
        }
    }
}

// Specialization for 8-byte elements (int64, double)
__device__ void shuffle_8byte_vectorized(
    const uint8_t* chunk_in,
    uint8_t* chunk_out,
    size_t num_elements,
    int lane_id
) {
    // Use double2 for 128-bit transactions (2x double)
    const double2* in_vec = reinterpret_cast<const double2*>(chunk_in);
    const size_t num_vec2 = num_elements / 2;
    
    for (int byte_pos = lane_id; byte_pos < 8; byte_pos += WARP_SIZE) {
        uint8_t* dst = chunk_out + (byte_pos * num_elements);
        
        for (size_t v = 0; v < num_vec2; v++) {
            double2 vec = in_vec[v];  // Load 16 bytes (2 elements)
            uint64_t* elements = reinterpret_cast<uint64_t*>(&vec);
            
            dst[v * 2 + 0] = (elements[0] >> (byte_pos * 8)) & 0xFF;
            dst[v * 2 + 1] = (elements[1] >> (byte_pos * 8)) & 0xFF;
        }
        
        // Handle odd element
        if (num_elements % 2 == 1) {
            const uint64_t* in64 = reinterpret_cast<const uint64_t*>(chunk_in);
            size_t last = num_elements - 1;
            dst[last] = (in64[last] >> (byte_pos * 8)) & 0xFF;
        }
    }
}

// Dispatcher kernel with compile-time specialization
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
    
    // Check alignment
    bool aligned = ((uintptr_t)chunk_in % 16 == 0) && 
                   ((uintptr_t)chunk_out % 16 == 0);
    
    if constexpr (ElementSize == 4) {
        if (aligned && num_elements >= 4) {
            shuffle_4byte_vectorized(chunk_in, chunk_out, num_elements, lane_id);
        } else {
            shuffle_scalar<4>(chunk_in, chunk_out, num_elements, lane_id);
        }
    } else if constexpr (ElementSize == 8) {
        if (aligned && num_elements >= 2) {
            shuffle_8byte_vectorized(chunk_in, chunk_out, num_elements, lane_id);
        } else {
            shuffle_scalar<8>(chunk_in, chunk_out, num_elements, lane_id);
        }
    } else {
        shuffle_scalar<ElementSize>(chunk_in, chunk_out, num_elements, lane_id);
    }
}
```

**Benefits:**
- ✅ 4x fewer memory transactions
- ✅ Better cache utilization
- ✅ 20-30% speedup for aligned data

---

### 8.3 Software Pipelining with Shared Memory

**Goal:** Maximize occupancy and hide latency through multi-stage pipeline

```cuda
__global__ void byte_shuffle_kernel_pipelined(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    // Three-stage pipeline: Load -> Process -> Store
    __shared__ uint8_t load_buffer[8192];
    __shared__ uint8_t process_buffer[8192];
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / element_size;
    
    constexpr size_t TILE_SIZE = 4096;
    const size_t num_tiles = (chunk_size + TILE_SIZE - 1) / TILE_SIZE;
    
    uint8_t* my_load_buf = load_buffer + warp_in_block * TILE_SIZE;
    uint8_t* my_proc_buf = process_buffer + warp_in_block * TILE_SIZE;
    
    // Prime the pipeline: load first tile
    if (num_tiles > 0) {
        size_t load_size = min(TILE_SIZE, chunk_size);
        for (size_t i = lane_id; i < load_size; i += WARP_SIZE) {
            my_load_buf[i] = chunk_in[i];
        }
    }
    __syncwarp();
    
    // Pipeline stages
    for (size_t tile = 0; tile < num_tiles; tile++) {
        // STAGE 1: Swap buffers (already loaded into load_buf)
        uint8_t* temp = my_load_buf;
        my_load_buf = my_proc_buf;
        my_proc_buf = temp;
        
        // STAGE 2: Start loading NEXT tile (async with processing)
        if (tile + 1 < num_tiles) {
            size_t next_offset = (tile + 1) * TILE_SIZE;
            size_t next_size = min(TILE_SIZE, chunk_size - next_offset);
            for (size_t i = lane_id; i < next_size; i += WARP_SIZE) {
                my_load_buf[i] = chunk_in[next_offset + i];
            }
        }
        
        // STAGE 3: Process CURRENT tile from proc_buf
        size_t tile_offset = tile * TILE_SIZE;
        size_t tile_bytes = min(TILE_SIZE, chunk_size - tile_offset);
        size_t tile_elements = tile_bytes / element_size;
        
        for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += WARP_SIZE) {
            const uint8_t* src = my_proc_buf + byte_pos;
            uint8_t* dst = chunk_out + (byte_pos * num_elements) + (tile_offset / element_size);
            
            #pragma unroll 4
            for (size_t elem = 0; elem < tile_elements; elem++) {
                dst[elem] = src[elem * element_size];
            }
        }
        
        __syncwarp();  // Wait for both load and process to complete
    }
}
```

**Benefits:**
- ✅ Overlaps memory load with compute
- ✅ Better GPU utilization
- ✅ 15-25% speedup for large chunks

---

### 8.4 Coalescing-Aware Transpose Algorithm

**Goal:** Optimize for both read and write coalescing simultaneously

```cuda
// Alternative strategy: transpose in shared memory
__device__ void shuffle_transpose_smem(
    const uint8_t* chunk_in,
    uint8_t* chunk_out,
    size_t num_elements,
    unsigned element_size,
    int lane_id
) {
    // Shared memory for transposition (avoid bank conflicts)
    __shared__ uint8_t tile[32][33];  // +1 to avoid bank conflicts
    
    const size_t TILE_DIM = 32;
    
    for (size_t base_elem = 0; base_elem < num_elements; base_elem += TILE_DIM) {
        // Load tile (coalesced reads)
        for (int byte_pos = 0; byte_pos < element_size; byte_pos++) {
            if (base_elem + lane_id < num_elements) {
                tile[byte_pos][lane_id] = chunk_in[(base_elem + lane_id) * element_size + byte_pos];
            }
        }
        __syncwarp();
        
        // Store transposed tile (coalesced writes)
        if (lane_id < element_size) {
            uint8_t* dst = chunk_out + (lane_id * num_elements) + base_elem;
            for (int i = 0; i < TILE_DIM && base_elem + i < num_elements; i++) {
                dst[i] = tile[lane_id][i];
            }
        }
        __syncwarp();
    }
}
```

**Benefits:**
- ✅ Both reads and writes are coalesced
- ✅ Bank conflict avoidance
- ✅ Best for element_size ≤ 32

---

### 8.5 Compile-Time Optimization and Templating

**Goal:** Enable aggressive compiler optimizations through compile-time constants

```cuda
// Template parameters allow compiler to optimize loops and remove branches
template<
    unsigned ElementSize,
    unsigned ChunkSizeHint,
    bool UseVectorization,
    bool UseSharedMem
>
__global__ void byte_shuffle_kernel_optimized(
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
    
    if constexpr (UseSharedMem && ChunkSizeHint <= 8192) {
        // Small chunks: use shared memory
        __shared__ uint8_t smem[ChunkSizeHint];
        // ... implementation
    } else if constexpr (UseVectorization && ElementSize == 4) {
        // Medium chunks with 4-byte elements: vectorize
        shuffle_4byte_vectorized(chunk_in, chunk_out, num_elements, lane_id);
    } else if constexpr (ElementSize <= 8) {
        // Unroll loops for small element sizes
        #pragma unroll
        for (int byte_pos = lane_id; byte_pos < ElementSize; byte_pos += WARP_SIZE) {
            const uint8_t* src = chunk_in + byte_pos;
            uint8_t* dst = chunk_out + (byte_pos * num_elements);
            
            for (size_t elem = 0; elem < num_elements; elem++) {
                dst[elem] = src[elem * ElementSize];
            }
        }
    } else {
        // Generic path for large element sizes
        shuffle_generic<ElementSize>(chunk_in, chunk_out, num_elements, lane_id);
    }
}

// Host-side dispatcher
void launch_optimized_shuffle(
    const ChunkBatch& batch,
    unsigned element_size,
    size_t avg_chunk_size
) {
    // Select best kernel at runtime based on data characteristics
    if (element_size == 4 && avg_chunk_size <= 8192) {
        byte_shuffle_kernel_optimized<4, 8192, true, true><<<...>>>();
    } else if (element_size == 8 && avg_chunk_size > 64*1024) {
        byte_shuffle_kernel_optimized<8, 65536, true, false><<<...>>>();
    } else {
        // Fallback generic kernel
        byte_shuffle_kernel<<<...>>>(batch, element_size);
    }
}
```

**Benefits:**
- ✅ Zero-overhead abstractions
- ✅ Loop unrolling automatically applied
- ✅ Dead code elimination
- ✅ Inlining opportunities
- ✅ 10-20% speedup from better codegen

---

### 8.6 Asynchronous Memory Copy (for newer GPUs)

**Goal:** Use CUDA async copy for newer architectures (Ampere, Hopper)

```cuda
#if __CUDA_ARCH__ >= 800  // Ampere and newer

__global__ void byte_shuffle_kernel_async_copy(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    __shared__ uint8_t smem[16384];
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* chunk_in = input_chunks[warp_id];
    uint8_t* chunk_out = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    
    // Use cp.async for global-to-shared copy (hardware accelerated)
    constexpr size_t TILE_SIZE = 16384;
    
    for (size_t offset = 0; offset < chunk_size; offset += TILE_SIZE) {
        size_t copy_size = min(TILE_SIZE, chunk_size - offset);
        
        // Async copy from global to shared (non-blocking!)
        if (lane_id == 0) {
            __pipeline_memcpy_async(smem, chunk_in + offset, copy_size);
            __pipeline_commit();
        }
        
        // Wait for async copy to complete
        __pipeline_wait_prior(0);
        __syncwarp();
        
        // Process from shared memory
        // ... shuffle logic ...
        
        __syncwarp();
    }
}

#endif
```

**Benefits:**
- ✅ Hardware-accelerated memory copy
- ✅ Better pipeline utilization
- ✅ 5-15% speedup on Ampere/Hopper
- ⚠️ Only available on SM 8.0+ (A100, H100)

---

### 8.7 Optimization Selection Matrix

| Data Profile | Element Size | Chunk Size | Best Optimization | Expected Speedup |
|--------------|--------------|------------|-------------------|------------------|
| Aligned int32 | 4 bytes | Any | Vectorized (float4) | 1.3x |
| Aligned double | 8 bytes | Any | Vectorized (double2) | 1.25x |
| Large elements | 16-32 bytes | >64KB | Shared memory staging | 1.4x |
| Small chunks | Any | <16KB | Transpose in shared mem | 1.2x |
| Large chunks | Any | >256KB | Double buffering | 1.15x |
| Mixed sizes | Varies | Varies | Template specialization | 1.25x |
| Ampere GPU | Any | >32KB | Async copy pipeline | 1.1x |

**General Strategy:**
1. Start with shared memory staging (most robust)
2. Add vectorization if alignment is guaranteed
3. Use template specialization for common cases (4, 8 bytes)
4. Enable async copy on Ampere+ GPUs
```

---

## 9. PERFORMANCE ANALYSIS & BENCHMARKING

### 9.1 Memory Bandwidth Analysis

#### Theoretical Bandwidth Calculation
```
Total Data Movement = Input Read + Output Write = 2 × Data Size

For 1GB shuffle:
  - Memory traffic = 2 GB
  - A100 peak bandwidth = 2039 GB/s (spec)
  - Theoretical best time = 2 GB / 2039 GB/s ≈ 0.98 ms
  - Achievable (80% efficiency) = 2 GB / (2039 × 0.8) GB/s ≈ 1.2 ms
  - Throughput = 1 GB / 0.0012 s ≈ 830 GB/s
```

#### Measuring Actual Bandwidth
```cpp
class ShuffleBenchmark {
public:
    struct BenchmarkResult {
        double time_ms;
        double bandwidth_gbps;
        double efficiency_percent;  // vs. theoretical peak
        size_t bytes_processed;
    };
    
    BenchmarkResult measure_shuffle(
        void* device_data,
        size_t bytes,
        size_t element_size,
        int num_iterations = 100
    ) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm-up
        shuffler.shuffle(device_data, bytes, element_size);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            shuffler.shuffle(device_data, bytes, element_size);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        BenchmarkResult result;
        result.time_ms = elapsed_ms / num_iterations;
        result.bytes_processed = bytes;
        
        // Bandwidth = (read + write) / time
        double total_traffic_gb = (bytes * 2.0) / (1024.0 * 1024.0 * 1024.0);
        result.bandwidth_gbps = total_traffic_gb / (result.time_ms / 1000.0);
        
        // Efficiency vs. peak
        double peak_bandwidth = get_device_peak_bandwidth();
        result.efficiency_percent = (result.bandwidth_gbps / peak_bandwidth) * 100.0;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return result;
    }
    
    double get_device_peak_bandwidth() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        // Memory bandwidth (GB/s) = (Memory Clock × Bus Width × 2) / 8
        // Factor of 2 for DDR (Double Data Rate)
        double bandwidth_gbps = 
            (prop.memoryClockRate * 1000.0) *  // Convert to Hz
            (prop.memoryBusWidth / 8.0) *      // Convert bits to bytes
            2.0 /                               // DDR
            1.0e9;                              // Convert to GB/s
        
        return bandwidth_gbps;
    }
};
```

---

### 9.2 Profiling with NVIDIA Nsight Compute

#### Key Metrics to Monitor

```bash
# Profile with Nsight Compute
ncu --set full --export shuffle_profile \
    --target-processes all \
    ./shuffle_benchmark

# Key metrics to examine:
# 1. Memory Throughput
#    - dram__throughput.avg.pct_of_peak_sustained_elapsed
#    - Target: >70%
#
# 2. Compute Utilization
#    - sm__throughput.avg.pct_of_peak_sustained_elapsed
#    - Should be LOW for memory-bound shuffle (<30%)
#
# 3. Memory Efficiency
#    - dram__bytes_read.sum / dram__bytes_requested.sum
#    - Target: >80% (good coalescing)
#
# 4. Occupancy
#    - sm__warps_active.avg.pct_of_peak_sustained_active
#    - Target: >50%
#
# 5. Stall Reasons
#    - smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
#    - Identify bottlenecks
```

#### Analyzing Profile Results
```python
# Python script to analyze NSight Compute output
import pandas as pd

def analyze_profile(csv_file):
    df = pd.read_csv(csv_file)
    
    metrics = {
        'Memory Bandwidth': df['dram__throughput.avg.pct_of_peak_sustained_elapsed'].values[0],
        'L2 Hit Rate': df['lts__t_sectors_hit_rate.pct'].values[0],
        'Global Load Efficiency': df['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct'].values[0],
        'Global Store Efficiency': df['smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct'].values[0],
        'Occupancy': df['sm__warps_active.avg.pct_of_peak_sustained_active'].values[0]
    }
    
    print("Performance Metrics:")
    for name, value in metrics.items():
        print(f"  {name:30s}: {value:6.2f}%")
    
    # Diagnose issues
    if metrics['Memory Bandwidth'] < 70:
        print("\n⚠️ LOW MEMORY BANDWIDTH - Check coalescing")
    if metrics['Global Load Efficiency'] < 80:
        print("\n⚠️ POOR LOAD COALESCING - Use shared memory staging")
    if metrics['Occupancy'] < 30:
        print("\n⚠️ LOW OCCUPANCY - Increase blocks or reduce register usage")
```

---

### 9.3 Comparative Benchmarking Suite

```cpp
// Comprehensive benchmark comparing all optimizations
struct OptimizationBenchmark {
    struct Config {
        size_t element_size;
        size_t chunk_size;
        size_t total_size;
        bool aligned;
    };
    
    void run_all_benchmarks() {
        std::vector<Config> configs = {
            {4, 64*1024, 100*1024*1024, true},   // 100MB, 4-byte, 64KB chunks
            {8, 64*1024, 100*1024*1024, true},   // 100MB, 8-byte, 64KB chunks
            {16, 256*1024, 500*1024*1024, true}, // 500MB, 16-byte, 256KB chunks
            {4, 16*1024, 50*1024*1024, false},   // 50MB, 4-byte, unaligned
        };
        
        for (const auto& config : configs) {
            printf("\n=== Configuration: elem=%zu, chunk=%zuKB, total=%zuMB ===\n",
                   config.element_size,
                   config.chunk_size / 1024,
                   config.total_size / (1024*1024));
            
            // Allocate test data
            uint8_t* d_data;
            cudaMalloc(&d_data, config.total_size);
            
            // Test each optimization
            auto baseline = benchmark_baseline(d_data, config);
            auto smem = benchmark_shared_memory(d_data, config);
            auto vectorized = benchmark_vectorized(d_data, config);
            auto pipelined = benchmark_pipelined(d_data, config);
            auto specialized = benchmark_specialized(d_data, config);
            
            // Print comparison table
            printf("\n%-25s %10s %10s %10s %10s\n",
                   "Optimization", "Time(ms)", "BW(GB/s)", "Speedup", "Efficiency");
            printf("%-25s %10.3f %10.2f %10.2fx %10.1f%%\n",
                   "Baseline", baseline.time_ms, baseline.bandwidth_gbps,
                   1.0, baseline.efficiency_percent);
            printf("%-25s %10.3f %10.2f %10.2fx %10.1f%%\n",
                   "Shared Memory", smem.time_ms, smem.bandwidth_gbps,
                   baseline.time_ms / smem.time_ms, smem.efficiency_percent);
            printf("%-25s %10.3f %10.2f %10.2fx %10.1f%%\n",
                   "Vectorized", vectorized.time_ms, vectorized.bandwidth_gbps,
                   baseline.time_ms / vectorized.time_ms, vectorized.efficiency_percent);
            printf("%-25s %10.3f %10.2f %10.2fx %10.1f%%\n",
                   "Pipelined", pipelined.time_ms, pipelined.bandwidth_gbps,
                   baseline.time_ms / pipelined.time_ms, pipelined.efficiency_percent);
            printf("%-25s %10.3f %10.2f %10.2fx %10.1f%%\n",
                   "Specialized", specialized.time_ms, specialized.bandwidth_gbps,
                   baseline.time_ms / specialized.time_ms, specialized.efficiency_percent);
            
            cudaFree(d_data);
        }
    }
};
```

---

### 9.4 Roofline Model Analysis

```cpp
// Analyze where shuffle sits on roofline model
struct RooflineAnalysis {
    double compute_arithmetic_intensity(size_t element_size) {
        // Arithmetic intensity = FLOPs / Bytes
        // Shuffle has minimal computation: just byte moves
        double bytes_per_element = element_size;
        double operations_per_element = element_size;  // Just moves, no FLOPs
        
        // AI is very low for shuffle (memory-bound)
        return operations_per_element / (2.0 * bytes_per_element);  // Read+Write
    }
    
    void plot_roofline(const std::vector<BenchmarkResult>& results) {
        double peak_bandwidth = get_device_peak_bandwidth();
        double peak_compute = get_device_peak_flops();
        
        printf("\nRoofline Analysis:\n");
        printf("  Peak Memory BW: %.2f GB/s\n", peak_bandwidth);
        printf("  Peak Compute:    %.2f TFLOPS\n", peak_compute / 1e12);
        printf("\n");
        
        for (const auto& result : results) {
            double ai = compute_arithmetic_intensity(result.element_size);
            double achieved_perf = result.bandwidth_gbps * ai;
            
            printf("  Element Size %2zu: AI=%.4f, Achieved=%.2f GB/s (%.1f%% of peak)\n",
                   result.element_size, ai, result.bandwidth_gbps,
                   (result.bandwidth_gbps / peak_bandwidth) * 100.0);
        }
        
        printf("\n  Conclusion: Shuffle is MEMORY-BOUND (low arithmetic intensity)\n");
        printf("              Optimization should focus on memory access patterns\n");
    }
};
```

---

### 9.5 Automated Optimization Selection

```cpp
// Runtime selection of best optimization based on profiling
class AdaptiveShuffle {
private:
    struct ProfileCache {
        std::map<size_t, OptimizationType> element_size_map;
        std::map<size_t, OptimizationType> chunk_size_map;
    };
    
    ProfileCache cache_;
    
    OptimizationType profile_and_select(
        size_t element_size,
        size_t chunk_size,
        bool aligned
    ) {
        // Check cache first
        auto key = (element_size << 32) | chunk_size;
        if (cache_.element_size_map.count(element_size)) {
            return cache_.element_size_map[element_size];
        }
        
        // Run quick micro-benchmarks
        const size_t test_size = 16 * 1024 * 1024;  // 16MB test
        uint8_t* d_test;
        cudaMalloc(&d_test, test_size);
        
        std::vector<std::pair<OptimizationType, double>> timings;
        
        // Test baseline
        timings.push_back({
            OptimizationType::BASELINE,
            quick_benchmark(d_test, test_size, element_size, OptimizationType::BASELINE)
        });
        
        // Test optimizations
        timings.push_back({
            OptimizationType::SHARED_MEM,
            quick_benchmark(d_test, test_size, element_size, OptimizationType::SHARED_MEM)
        });
        
        if (aligned) {
            timings.push_back({
                OptimizationType::VECTORIZED,
                quick_benchmark(d_test, test_size, element_size, OptimizationType::VECTORIZED)
            });
        }
        
        cudaFree(d_test);
        
        // Select fastest
        auto best = std::min_element(timings.begin(), timings.end(),
                                      [](const auto& a, const auto& b) {
                                          return a.second < b.second;
                                      });
        
        // Cache result
        cache_.element_size_map[element_size] = best->first;
        
        return best->first;
    }
    
public:
    void shuffle_adaptive(void* device_data, size_t bytes, size_t element_size) {
        bool aligned = ((uintptr_t)device_data % 16 == 0);
        OptimizationType opt = profile_and_select(element_size, bytes, aligned);
        
        switch (opt) {
            case OptimizationType::BASELINE:
                shuffle_baseline(device_data, bytes, element_size);
                break;
            case OptimizationType::SHARED_MEM:
                shuffle_shared_mem(device_data, bytes, element_size);
                break;
            case OptimizationType::VECTORIZED:
                shuffle_vectorized(device_data, bytes, element_size);
                break;
        }
    }
};
```

---

### 9.6 Expected Performance Targets

| GPU Model | Peak BW (GB/s) | Target BW (GB/s) | Target Efficiency | Notes |
|-----------|----------------|------------------|-------------------|-------|
| H100 | 3350 | 2500+ | 75%+ | Use async copy |
| A100 | 2039 | 1500+ | 73%+ | Vectorized + smem |
| V100 | 900 | 650+ | 72%+ | Shared memory |
| T4 | 320 | 220+ | 69%+ | Baseline optimized |
| RTX 3090 | 936 | 650+ | 69%+ | Consumer GPU |

**Factors affecting efficiency:**
- Element size (4, 8 bytes best)
- Alignment (16-byte aligned best)
- Chunk size (64KB-1MB optimal)
- Data pattern (sequential best)

---

## 10. TESTING STRATEGY

### Test 1: Correctness Test
```cpp
void test_shuffle_correctness() {
    // Test with known pattern
    std::vector<uint32_t> input = {
        0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F
    };
    
    // Expected output (shuffled bytes):
    // [00 04 08 0C] [01 05 09 0D] [02 06 0A 0E] [03 07 0B 0F]
    std::vector<uint8_t> expected = {
        0x00, 0x04, 0x08, 0x0C,
        0x01, 0x05, 0x09, 0x0D,
        0x02, 0x06, 0x0A, 0x0E,
        0x03, 0x07, 0x0B, 0x0F
    };
    
    // Run GPU shuffle
    // Compare results
}
```

### Test 2: Round-Trip Test
```cpp
void test_shuffle_roundtrip() {
    // Original -> Shuffle -> Unshuffle -> Should equal Original
    
    std::vector<float> original = generate_random_floats(1024);
    
    auto shuffled = gpu_shuffle(original);
    auto restored = gpu_unshuffle(shuffled);
    
    assert(original == restored);
}
```

### Test 3: Compare with CPU Reference
```cpp
void test_vs_cpu_reference() {
    // Use H5Z__filter_shuffle as reference
    auto cpu_result = cpu_shuffle(input);
    auto gpu_result = gpu_shuffle(input);
    
    assert(cpu_result == gpu_result);
}
```

### Test 4: Performance Benchmark
```cpp
void benchmark_shuffle() {
    // Test different chunk sizes: 16KB, 64KB, 256KB, 1MB
    // Test different element sizes: 2, 4, 8, 16 bytes
    // Measure throughput (GB/s)
}
```

### Test 5: Edge Cases
- Single element chunks
- 1-byte elements (should skip)
- Chunks with leftover bytes
- Very large chunks (> 1MB)
- Very small chunks (< 1KB)

---

## 10. PERFORMANCE EXPECTATIONS

### Theoretical Analysis

**Memory Bandwidth:**
- Each byte read once, written once → 2x memory traffic
- For 1GB input: 2GB memory traffic
- On A100 (2TB/s): ~1ms expected

**Warp Efficiency:**
- Strategy A: High (parallel byte positions)
- Strategy B: Medium (serialized byte positions)

**Expected Throughput:**
- Target: 500+ GB/s on A100
- Target: 200+ GB/s on V100

### Bottlenecks
1. Strided memory access for input reads
2. Small element sizes (< 4 bytes) less efficient
3. Leftover handling (serial)

---

## 11. INTEGRATION WITH COMPRESSION PIPELINE

### Workflow
```
1. Upload data to GPU
2. Chunk data using chunkDeviceBuffer()
3. Apply shuffle filter (this kernel)
4. Run compression (nvCOMP)
5. Download compressed data
```

### API Design
```cpp
class GPUShuffle {
public:
    // Shuffle in-place
    void shuffle(void* device_data, size_t size, unsigned element_size);
    
    // Shuffle with chunking
    void shuffle_chunked(
        void* device_data,
        size_t total_size,
        size_t chunk_size,
        unsigned element_size
    );
    
    // Unshuffle
    void unshuffle(void* device_data, size_t size, unsigned element_size);
    
private:
    ChunkBatch batch_;
};
```

---

## 12. NEXT STEPS

### Phase 1: Basic Implementation
1. ✅ Understand H5Zshuffle algorithm
2. ⬜ Implement Strategy A kernel (outer loop parallel)
3. ⬜ Implement host-side preparation
4. ⬜ Write basic correctness tests

### Phase 2: Optimization
5. ⬜ Add shared memory staging
6. ⬜ Implement Strategy B (inner loop parallel)
7. ⬜ Add compile-time specialization
8. ⬜ Benchmark and profile

### Phase 3: Integration
9. ⬜ Integrate with chunkDeviceBuffer
10. ⬜ Add to compression pipeline
11. ⬜ End-to-end testing
12. ⬜ Documentation and examples

---

## 13. FILE STRUCTURE

```
src/
├── util.cu                      # Existing chunking utilities
├── byte_shuffle.cuh             # Kernel declarations and host API
├── byte_shuffle.cu              # Kernel implementations
├── byte_shuffle_host.cpp        # Host-side wrapper (optional)
└── byte_shuffle_test.cu         # Test suite

examples/
└── byte_shuffle_example.cu      # Usage examples for various data types
```

### Recommended File Naming
Use generic, universal names that indicate the byte-level nature:
- ✅ `byte_shuffle.cu`, `gpu_shuffle.cu`, `shuffle_transform.cu`
- ❌ `h5z_shuffle.cu`, `hdf5_shuffle.cu` (implies HDF5 dependency)

---

## 14. SUMMARY

This implementation plan provides a **universal, warp-level parallel byte shuffle transformation** that:

✅ **Universal** - Works with ANY data type (primitives, structs, arrays, binary data)
✅ **Type-agnostic** - Operates purely at byte level, no type information needed
✅ **One warp per chunk** - No global synchronization required
✅ **Scalable** - Processes multiple chunks in parallel across warps
✅ **Memory efficient** - Minimal overhead, in-place capable
✅ **Compatible** - Compatible with HDF5, Blosc, Parquet, and custom formats
✅ **Flexible** - Multiple strategies for different data sizes and element types
✅ **Reusable** - Zero dependencies on specific file formats or compression libraries

### Key Design Insights:

1. **Byte-level operation**: Works with any data by treating it as a byte stream
2. **Parallel byte positions** (Strategy A): Each thread handles specific byte positions independently
3. **Warp-level synchronization only**: Maximizes parallelism without costly global barriers
4. **Template support**: Type-safe convenience wrappers with automatic size deduction

### Universal Applications:

- Scientific computing (HDF5, NetCDF)
- Image/video preprocessing
- Time series compression
- Database columnar storage
- Network packet processing
- Any compression pipeline requiring improved byte-level redundancy

---

## 15. UNIVERSALITY & USE CASES

### Why This Implementation is Universal

This byte shuffle implementation is **completely data-type agnostic** because:

1. **No Type Information Required**: Only needs `element_size` (bytes per element)
2. **Pure Byte Operations**: Works on `uint8_t*` pointers - any data can be cast
3. **No Format Dependencies**: Doesn't depend on HDF5, Blosc, or any specific library
4. **No Semantic Knowledge**: Doesn't need to know what the data represents

### Concrete Use Cases

#### Scientific Computing
```cpp
// Temperature field from climate simulation
double temperature[1000000];
shuffler.shuffle(d_temperature, 1000000);  // 8-byte elements
compress(d_temperature, ...);  // Better compression!
```

#### Image Processing
```cpp
// Separate RGB channels for better compression
struct Pixel { uint8_t r, g, b; };
Pixel image[1920 * 1080];
shuffler.shuffle(d_image, 1920*1080);  // Groups all R, then G, then B
```

#### Financial Data
```cpp
// Stock market tick data
struct Tick { 
    uint64_t timestamp;
    float price;
    uint32_t volume;
};
Tick ticks[1000000];
shuffler.shuffle(d_ticks, 1000000);  // Groups timestamps, prices, volumes
```

#### Sensor Networks
```cpp
// IoT sensor readings
struct Reading {
    uint32_t sensor_id;
    float value;
    uint64_t timestamp;
};
Reading data[500000];
shuffler.shuffle(d_data, 500000);  // 16-byte elements
```

#### Vector Mathematics
```cpp
// 3D vectors for physics simulation
struct Vec3 { float x, y, z; };
Vec3 positions[100000];
shuffler.shuffle(d_positions, 100000);  // Groups x, y, z separately
```

### Integration Examples

#### With nvCOMP:
```cpp
// Generic compression pipeline
template<typename T>
void compress_with_shuffle(T* device_data, size_t count) {
    GPUByteShuffle shuffler;
    shuffler.shuffle(device_data, count);  // Preprocessing
    
    nvcomp::LZ4Manager compressor;
    compressor.compress(device_data, count * sizeof(T));
}
```

#### With Custom Compressor:
```cpp
// Works with any compression algorithm
void compress_pipeline(void* data, size_t bytes, size_t elem_size) {
    GPUByteShuffle shuffler;
    shuffler.shuffle(data, bytes, elem_size);
    my_custom_compressor(data, bytes);
}
```

#### Batch Processing:
```cpp
// Process multiple arrays of different types
shuffler.shuffle(float_array, 1000000);      // 4-byte
shuffler.shuffle(double_array, 500000);      // 8-byte  
shuffler.shuffle(custom_struct_array, 10000); // 24-byte
```
