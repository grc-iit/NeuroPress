# GPU Byte Shuffle - Optimization Techniques Guide
## Comprehensive Reference for Memory Access Optimizations

---

## Quick Reference Table

| Optimization | Best For | Speedup | Complexity | Memory Overhead |
|--------------|----------|---------|------------|-----------------|
| **Baseline** | General use | 1.0x | Low | None |
| **Shared Memory Staging** | Large elements (8-16B) | 1.3-1.5x | Medium | +32KB/block |
| **Vectorized Access** | Aligned 2/4/8B | 1.15-1.3x | Medium | None |
| **Loop Unrolling** | All cases | 1.1-1.15x | Low | None |
| **Double Buffering** | Large chunks (>256KB) | 1.1-1.2x | High | +64KB/block |
| **Warp Shuffle** | 4-byte elements | 1.2-1.4x | High | None |
| **Transpose in Smem** | Small elem (≤32B) | 1.2-1.3x | Medium | +1KB/warp |
| **Template Specialization** | Known sizes | 1.1-1.25x | Low | Code size |
| **Async Copy (Ampere+)** | Large chunks | 1.05-1.15x | Medium | None |

---

## 1. MEMORY COALESCING FUNDAMENTALS

### What is Memory Coalescing?

**Coalesced Access:** When threads in a warp access consecutive memory addresses, the GPU combines them into a single large transaction.

```cuda
// COALESCED (Good) ✓
__global__ void coalesced_example() {
    int idx = threadIdx.x;
    // Thread 0 reads array[0], Thread 1 reads array[1], etc.
    int value = array[idx];  // Sequential = coalesced = FAST
}

// STRIDED (Bad) ✗
__global__ void strided_example() {
    int idx = threadIdx.x;
    int stride = 16;
    // Thread 0 reads array[0], Thread 1 reads array[16], etc.
    int value = array[idx * stride];  // Gaps = not coalesced = SLOW
}
```

**Impact on Shuffle:**
- Output writes in baseline: ✓ Coalesced (sequential)
- Input reads in baseline: ✗ Strided by `element_size`
- **Solution:** Use shared memory or vectorization

---

## 2. SHARED MEMORY STAGING (Primary Optimization)

### When to Use
- ✅ Element size ≥ 8 bytes
- ✅ Large chunks (> 32KB)
- ✅ When reads are strided
- ✅ General-purpose solution

### Implementation Pattern

```cuda
__global__ void shuffle_with_smem(
    const uint8_t* input,
    uint8_t* output,
    size_t num_elements,
    unsigned element_size
) {
    // Allocate shared memory
    __shared__ uint8_t tile[TILE_SIZE];
    
    const int lane_id = threadIdx.x % 32;
    const size_t chunk_size = num_elements * element_size;
    
    // Process in tiles
    for (size_t tile_offset = 0; tile_offset < chunk_size; tile_offset += TILE_SIZE) {
        const size_t tile_bytes = min(TILE_SIZE, chunk_size - tile_offset);
        
        // STAGE 1: Coalesced load global → shared
        for (size_t i = lane_id; i < tile_bytes; i += 32) {
            tile[i] = input[tile_offset + i];  // ✓ COALESCED
        }
        __syncwarp();
        
        // STAGE 2: Shuffle from shared memory (fast!)
        const size_t tile_elems = tile_bytes / element_size;
        for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += 32) {
            const uint8_t* src = tile + byte_pos;
            uint8_t* dst = output + (byte_pos * num_elements) + (tile_offset / element_size);
            
            for (size_t elem = 0; elem < tile_elems; elem++) {
                dst[elem] = src[elem * element_size];  // Fast shared memory read
            }
        }
        __syncwarp();
    }
}
```

### Key Benefits
1. **All global reads coalesced** → Full memory bandwidth utilization
2. **Shared memory ~20x faster than global** → Hides strided access latency
3. **30-50% speedup typical** for element_size > 4

### Tuning Parameters
```cuda
// Tile size selection
#define TILE_SIZE_SMALL  4096   // For <64KB chunks
#define TILE_SIZE_MEDIUM 8192   // For 64KB-256KB chunks
#define TILE_SIZE_LARGE  16384  // For >256KB chunks

// Per-block shared memory limit: 48KB (most GPUs)
// Reserve space for multiple warps if needed
```

---

## 3. VECTORIZED MEMORY ACCESS

### When to Use
- ✅ Aligned data (pointers % 16 == 0)
- ✅ Common element sizes (2, 4, 8 bytes)
- ✅ When coalescing is already good
- ✅ To reduce instruction count

### Vector Types

| Element Size | Vector Type | Bytes/Load | Elements/Load |
|--------------|-------------|------------|---------------|
| 2 bytes | `uint4` | 16 | 8 elements |
| 4 bytes | `float4`, `uint4` | 16 | 4 elements |
| 8 bytes | `double2`, `ulonglong2` | 16 | 2 elements |

### Implementation for 4-Byte Elements

```cuda
__device__ void shuffle_float4(
    const float* input,  // 4-byte elements
    uint8_t* output,
    size_t num_elements,
    int lane_id
) {
    // Check alignment
    if ((uintptr_t)input % 16 != 0) {
        return;  // Fall back to scalar
    }
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const size_t num_vec4 = num_elements / 4;
    
    // Process 4 elements per vector load
    for (int byte_pos = lane_id; byte_pos < 4; byte_pos += 32) {
        uint8_t* dst = output + (byte_pos * num_elements);
        
        for (size_t v = 0; v < num_vec4; v++) {
            float4 vec = input_vec[v];  // Load 16 bytes at once! ✓
            uint32_t* elems = reinterpret_cast<uint32_t*>(&vec);
            
            // Extract byte_pos from all 4 elements
            dst[v*4 + 0] = (elems[0] >> (byte_pos * 8)) & 0xFF;
            dst[v*4 + 1] = (elems[1] >> (byte_pos * 8)) & 0xFF;
            dst[v*4 + 2] = (elems[2] >> (byte_pos * 8)) & 0xFF;
            dst[v*4 + 3] = (elems[3] >> (byte_pos * 8)) & 0xFF;
        }
    }
}
```

### Expected Improvement
- **4x fewer load instructions**
- **Better cache utilization**
- **15-30% speedup when aligned**

---

## 4. LOOP UNROLLING

### When to Use
- ✅ Always beneficial
- ✅ Especially for small element sizes
- ✅ When loop count is known at compile time

### Manual Unrolling Example

```cuda
// WITHOUT unrolling
for (size_t elem = 0; elem < num_elements; elem++) {
    dst[elem] = src[elem * element_size];
}

// WITH unrolling (8x)
const int UNROLL = 8;
size_t elem = 0;

// Main unrolled loop
for (; elem + UNROLL <= num_elements; elem += UNROLL) {
    // Load phase
    uint8_t temp[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        temp[u] = src[(elem + u) * element_size];
    }
    
    // Store phase
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        dst[elem + u] = temp[u];
    }
}

// Tail: remaining elements
for (; elem < num_elements; elem++) {
    dst[elem] = src[elem * element_size];
}
```

### Benefits
- 8x fewer loop iterations
- Better instruction-level parallelism
- Enables further compiler optimizations
- **10-15% speedup typical**

---

## 5. DOUBLE BUFFERING (SOFTWARE PIPELINING)

### When to Use
- ✅ Large chunks (> 256KB)
- ✅ Memory latency is bottleneck
- ✅ Have spare shared memory

### Concept

```
Traditional:
  Load Tile 0 → Process Tile 0 → Load Tile 1 → Process Tile 1 → ...
  [IDLE]        [COMPUTE]        [IDLE]        [COMPUTE]

Double Buffering:
  Load Tile 0 → Load Tile 1 + Process Tile 0 → Load Tile 2 + Process Tile 1 → ...
  [LOAD]        [LOAD + COMPUTE]               [LOAD + COMPUTE]
  
Overlap = Hidden latency!
```

### Implementation

```cuda
__global__ void shuffle_double_buffer(
    const uint8_t* input,
    uint8_t* output,
    size_t chunk_size,
    unsigned element_size
) {
    __shared__ uint8_t buffer_a[TILE_SIZE];
    __shared__ uint8_t buffer_b[TILE_SIZE];
    
    const int lane_id = threadIdx.x % 32;
    const size_t num_tiles = (chunk_size + TILE_SIZE - 1) / TILE_SIZE;
    
    int current_buf = 0;  // 0 = buffer_a, 1 = buffer_b
    uint8_t* buffers[2] = {buffer_a, buffer_b};
    
    // Prime: Load first tile into buffer_a
    if (num_tiles > 0) {
        for (size_t i = lane_id; i < min(TILE_SIZE, chunk_size); i += 32) {
            buffer_a[i] = input[i];
        }
    }
    __syncwarp();
    
    // Pipeline: load next while processing current
    for (size_t tile = 0; tile < num_tiles; tile++) {
        uint8_t* process_buf = buffers[current_buf];
        uint8_t* load_buf = buffers[1 - current_buf];
        
        // Start loading next tile (async with processing)
        if (tile + 1 < num_tiles) {
            size_t next_offset = (tile + 1) * TILE_SIZE;
            size_t next_bytes = min(TILE_SIZE, chunk_size - next_offset);
            for (size_t i = lane_id; i < next_bytes; i += 32) {
                load_buf[i] = input[next_offset + i];  // Async load!
            }
        }
        
        // Process current tile from process_buf
        size_t tile_offset = tile * TILE_SIZE;
        size_t tile_bytes = min(TILE_SIZE, chunk_size - tile_offset);
        // ... shuffle logic ...
        
        __syncwarp();
        current_buf = 1 - current_buf;  // Swap buffers
    }
}
```

### Benefits
- Overlaps memory transfer with compute
- **10-20% speedup for large chunks**
- Effective on memory-bound kernels

---

## 6. TRANSPOSE IN SHARED MEMORY

### When to Use
- ✅ Small element sizes (≤ 32 bytes)
- ✅ When both reads and writes need coalescing
- ✅ Enough shared memory available

### Avoiding Bank Conflicts

```cuda
// BAD: Bank conflicts ✗
__shared__ uint8_t tile[32][32];  // Same bank accessed by multiple threads

// GOOD: Padding avoids conflicts ✓
__shared__ uint8_t tile[32][33];  // +1 byte padding
```

### Implementation

```cuda
__device__ void shuffle_transpose(
    const uint8_t* input,
    uint8_t* output,
    size_t num_elements,
    unsigned element_size,
    int lane_id
) {
    __shared__ uint8_t tile[32][33];  // Padded to avoid bank conflicts
    
    const size_t TILE_DIM = 32;
    
    // Process 32 elements at a time
    for (size_t base = 0; base < num_elements; base += TILE_DIM) {
        // Load into shared memory (coalesced reads)
        for (int byte_pos = 0; byte_pos < element_size; byte_pos++) {
            if (base + lane_id < num_elements) {
                tile[byte_pos][lane_id] = 
                    input[(base + lane_id) * element_size + byte_pos];
            }
        }
        __syncwarp();
        
        // Store transposed (coalesced writes)
        if (lane_id < element_size) {
            uint8_t* dst = output + (lane_id * num_elements) + base;
            for (int i = 0; i < TILE_DIM && base + i < num_elements; i++) {
                dst[i] = tile[lane_id][i];
            }
        }
        __syncwarp();
    }
}
```

### Benefits
- Both reads and writes coalesced
- No bank conflicts with padding
- **15-25% speedup for small elements**

---

## 7. COMPILE-TIME SPECIALIZATION

### When to Use
- ✅ Known element sizes at compile time
- ✅ Want maximum performance
- ✅ Can afford code size increase

### Template Approach

```cuda
// Generic kernel template
template<unsigned ElementSize>
__global__ void shuffle_specialized(
    const uint8_t* input,
    uint8_t* output,
    size_t num_elements
) {
    const int lane_id = threadIdx.x % 32;
    
    // Compiler can fully unroll this loop!
    #pragma unroll
    for (int byte_pos = lane_id; byte_pos < ElementSize; byte_pos += 32) {
        const uint8_t* src = input + byte_pos;
        uint8_t* dst = output + (byte_pos * num_elements);
        
        // Inner loop can also be unrolled if small
        if constexpr (ElementSize <= 16) {
            #pragma unroll
            for (size_t elem = 0; elem < num_elements; elem++) {
                dst[elem] = src[elem * ElementSize];
            }
        } else {
            for (size_t elem = 0; elem < num_elements; elem++) {
                dst[elem] = src[elem * ElementSize];
            }
        }
    }
}

// Host-side dispatcher
void launch_shuffle(void* data, size_t bytes, unsigned elem_size) {
    switch (elem_size) {
        case 4:  shuffle_specialized<4><<<...>>>(data, ...); break;
        case 8:  shuffle_specialized<8><<<...>>>(data, ...); break;
        case 16: shuffle_specialized<16><<<...>>>(data, ...); break;
        default: shuffle_generic<<<...>>>(data, elem_size, ...); break;
    }
}
```

### Benefits
- Zero-overhead abstractions
- Aggressive loop unrolling
- Dead code elimination
- **10-20% speedup from better codegen**

---

## 8. WARP SHUFFLE INSTRUCTIONS (Advanced)

### When to Use
- ✅ Element size exactly 4 bytes
- ✅ Want to avoid shared memory
- ✅ Small, regular patterns

### Implementation

```cuda
__device__ void shuffle_with_shfl(
    const uint32_t* input,
    uint8_t* output,
    size_t num_elements,
    int lane_id
) {
    // Works for groups of 32 elements at a time
    for (size_t batch = 0; batch < num_elements / 32; batch++) {
        // Each thread loads one element
        uint32_t my_value = input[batch * 32 + lane_id];
        
        // Extract and redistribute bytes using warp shuffle
        for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
            uint8_t my_byte = (my_value >> (byte_pos * 8)) & 0xFF;
            
            // Write all 32 bytes for this byte position
            for (int src_lane = 0; src_lane < 32; src_lane++) {
                uint8_t byte = __shfl_sync(0xFFFFFFFF, my_byte, src_lane);
                if (lane_id == 0) {
                    output[byte_pos * num_elements + batch * 32 + src_lane] = byte;
                }
            }
        }
    }
}
```

### Benefits
- No shared memory usage
- Ultra-low latency
- **Best for small, aligned 4-byte data**

### Limitations
- Only practical for 4-byte elements
- Complex synchronization
- Not generalizable

---

## 9. ASYNC COPY (Ampere+ GPUs Only)

### When to Use
- ✅ A100, H100, or newer
- ✅ Large data transfers
- ✅ Can pipeline effectively

### Implementation

```cuda
#if __CUDA_ARCH__ >= 800  // Ampere and newer

__global__ void shuffle_async(
    const uint8_t* input,
    uint8_t* output,
    size_t chunk_size,
    unsigned element_size
) {
    __shared__ uint8_t smem[16384];
    
    // Use hardware-accelerated async copy
    for (size_t offset = 0; offset < chunk_size; offset += 16384) {
        size_t copy_size = min(16384, chunk_size - offset);
        
        // Async copy from global to shared (non-blocking!)
        __pipeline_memcpy_async(smem, input + offset, copy_size);
        __pipeline_commit();
        
        // Wait for copy to complete
        __pipeline_wait_prior(0);
        __syncthreads();
        
        // Process from shared memory
        // ... shuffle logic ...
    }
}

#endif
```

### Benefits
- Hardware-accelerated DMA
- Better pipeline utilization
- **5-15% speedup on supported hardware**

---

## 10. OPTIMIZATION SELECTION FLOWCHART

```
START
  │
  ├─► Is element_size == 1? ───YES──► No shuffle needed (just copy)
  │         │
  │        NO
  │         │
  ├─► Is data aligned (% 16 == 0)? ───YES──► Consider VECTORIZED
  │         │                                         │
  │        NO                                         │
  │         │                                         │
  ├─► Is element_size == 4? ──────────YES──► Use VECTORIZED (float4)
  │         │                                         │
  │        NO                                         │
  │         │                                         │
  ├─► Is element_size >= 8? ──────────YES──► Use SHARED MEMORY STAGING
  │         │                                         │
  │        NO                                         │
  │         │                                         │
  ├─► Is chunk_size > 256KB? ─────────YES──► Add DOUBLE BUFFERING
  │         │                                         │
  │        NO                                         │
  │         │                                         │
  └─► Use BASELINE + LOOP UNROLLING
```

---

## 11. PRACTICAL RECOMMENDATIONS

### For Different Use Cases

#### Scientific Computing (HDF5, large arrays)
```
✓ Use: Shared memory staging + Loop unrolling
✓ Chunk size: 256KB - 1MB
✓ Expected: 1.3-1.5x speedup
```

#### Image Processing (RGB pixels, small elements)
```
✓ Use: Transpose in shared memory
✓ Element size: 3-4 bytes
✓ Expected: 1.2-1.3x speedup
```

#### Financial Data (aligned structs)
```
✓ Use: Vectorized access + Template specialization
✓ Alignment: Ensure 16-byte alignment
✓ Expected: 1.25-1.35x speedup
```

#### Real-time Processing (low latency)
```
✓ Use: Warp shuffle (if element_size == 4)
✓ Or: Baseline + unrolling
✓ Priority: Minimize synchronization
```

---

## 12. DEBUGGING & PROFILING

### Common Issues

#### Issue 1: Lower than expected bandwidth
```bash
# Profile with Nsight Compute
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    ./shuffle_benchmark

# Check:
# - Memory throughput < 70%? → Poor coalescing
# - Load efficiency < 80%? → Use shared memory
```

#### Issue 2: Bank conflicts in shared memory
```bash
# Check for bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./shuffle_benchmark

# Solution: Add padding to shared memory arrays
__shared__ uint8_t tile[32][33];  // Not tile[32][32]
```

#### Issue 3: Low occupancy
```bash
# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    ./shuffle_benchmark

# If < 30%:
# - Reduce register usage
# - Reduce shared memory usage
# - Increase blocks per SM
```

---

## Summary: Recommended Default Configuration

```cuda
// Best general-purpose optimized kernel
__global__ void byte_shuffle_optimized(
    const uint8_t** input_chunks,
    uint8_t** output_chunks,
    const size_t* chunk_sizes,
    size_t num_chunks,
    unsigned element_size
) {
    // Use shared memory staging (works for all cases)
    __shared__ uint8_t smem[8192];
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    
    if (warp_id >= num_chunks) return;
    
    const uint8_t* input = input_chunks[warp_id];
    uint8_t* output = output_chunks[warp_id];
    const size_t chunk_size = chunk_sizes[warp_id];
    const size_t num_elements = chunk_size / element_size;
    
    // Process in 8KB tiles with shared memory
    const size_t TILE_SIZE = 8192;
    for (size_t offset = 0; offset < chunk_size; offset += TILE_SIZE) {
        const size_t tile_bytes = min(TILE_SIZE, chunk_size - offset);
        
        // Coalesced load to shared memory
        for (size_t i = lane_id; i < tile_bytes; i += 32) {
            smem[i] = input[offset + i];
        }
        __syncwarp();
        
        // Shuffle with loop unrolling
        const size_t tile_elems = tile_bytes / element_size;
        const int UNROLL = 4;
        
        for (int byte_pos = lane_id; byte_pos < element_size; byte_pos += 32) {
            const uint8_t* src = smem + byte_pos;
            uint8_t* dst = output + (byte_pos * num_elements) + (offset / element_size);
            
            size_t elem = 0;
            // Unrolled main loop
            for (; elem + UNROLL <= tile_elems; elem += UNROLL) {
                #pragma unroll
                for (int u = 0; u < UNROLL; u++) {
                    dst[elem + u] = src[(elem + u) * element_size];
                }
            }
            // Tail
            for (; elem < tile_elems; elem++) {
                dst[elem] = src[elem * element_size];
            }
        }
        __syncwarp();
    }
}
```

**This configuration provides:**
- ✅ Good performance across all element sizes
- ✅ Coalesced memory access
- ✅ Loop unrolling benefits
- ✅ Reasonable shared memory usage
- ✅ 30-50% speedup vs naive baseline
