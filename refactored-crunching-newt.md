# Error-Bound Quantization Preprocessor for GPUCompress

## Overview

Add error-bound quantization as a preprocessing step (similar to existing byte shuffle) to significantly improve compression ratios when using nvcomp lossless algorithms. This trades controlled precision loss for much higher compression ratios.

## Design Decisions (User Confirmed)

| Decision | Choice |
|----------|--------|
| Quantization methods | **All three**: Linear, Lorenzo 1D, ZFP-style block transform |
| Error-bound types | **Absolute only** (REL/VR noted as future work) |
| Data types | **float32 + float64** |
| Output precision | **Adaptive** (int8/16/32 based on error bound & data range) |

## Background Research Summary

### Quantization Methods to Implement
1. **Linear**: Simple `quant = round(value / 2*eb)` - fast baseline
2. **Lorenzo 1D**: Prediction-based `quant = round((value - predicted) / 2*eb)` - higher compression for smooth data
3. **ZFP-style Block Transform**: 4-element blocks with orthogonal transform - best for structured data

### Error-Bound Guarantee
- **ABS (Absolute)**: `|original - decompressed| <= error_bound`
- Future work: REL (relative), VR (value-range percentage)

---

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### Step 1.1: Create Quantization Header
**File**: `src/quantization.cuh`

```cpp
// Enumerations
enum class QuantizationType { NONE, LINEAR, LORENZO_1D, BLOCK_TRANSFORM };
enum class QuantizationPrecision { AUTO, INT8, INT16, INT32 };

// Configuration structure
struct QuantizationConfig {
    QuantizationType type;
    QuantizationPrecision precision;  // AUTO = adaptive selection
    double error_bound;               // Absolute error bound
    size_t num_elements;
    size_t element_size;              // sizeof(float) or sizeof(double)
};

// Quantization result with metadata for header storage
struct QuantizationResult {
    void* d_quantized;                // Device pointer to quantized data
    size_t quantized_bytes;           // Output size in bytes
    int actual_precision;             // 8, 16, or 32 bits used
    double data_min, data_max;        // For dequantization
    double scale_factor;              // Quantization scale
};

// High-level API (mirrors byte_shuffle_simple pattern)
QuantizationResult quantize_simple(
    void* d_input,
    size_t num_elements,
    size_t element_size,              // 4 for float, 8 for double
    QuantizationConfig config,
    cudaStream_t stream = 0
);

void* dequantize_simple(
    void* d_quantized,
    const QuantizationResult& metadata,
    cudaStream_t stream = 0
);
```

#### Step 1.2: Extend Compression Header
**File**: `src/compression_header.h`

Extend header to store quantization metadata (version 2):

```cpp
struct CompressionHeader {
    uint32_t magic;                   // 0x43555047 ("GPUC")
    uint32_t version;                 // Bump to 2
    uint32_t shuffle_element_size;    // Existing
    uint32_t quant_flags;             // NEW: bits [0-3]=type, [4-7]=precision, [8]=enabled
    uint64_t original_size;           // Existing
    uint64_t compressed_size;         // Existing
    // Extended fields for version >= 2:
    double   quant_error_bound;       // NEW: absolute error bound
    double   quant_scale;             // NEW: scale factor used
    double   data_min;                // NEW: min value for dequantization
    double   data_max;                // NEW: max value
};
```

### Phase 2: GPU Kernels

#### Step 2.1: Adaptive Precision Selection
**File**: `src/quantization_kernels.cu`

```cpp
// Compute required bits based on error bound and data range
int compute_required_precision(double data_range, double error_bound) {
    double num_bins = data_range / (2.0 * error_bound);
    if (num_bins <= 127) return 8;        // int8 sufficient
    if (num_bins <= 32767) return 16;     // int16 sufficient
    return 32;                             // int32 needed
}
```

#### Step 2.2: Linear Quantization Kernels (Method 1)

```cuda
// Template for adaptive precision
template<typename InputT, typename OutputT>
__global__ void quantize_linear_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,
    double scale,           // = 1.0 / (2 * error_bound)
    double offset           // = -data_min * scale (for centering)
);

template<typename InputT, typename OutputT>
__global__ void dequantize_linear_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale,       // = 2 * error_bound
    double offset           // = data_min
);
```

#### Step 2.3: Lorenzo 1D Prediction Kernels (Method 2)

```cuda
// Dual-quantization approach from cuSZ for GPU parallelism
// Avoids read-after-write dependency

template<typename InputT, typename OutputT>
__global__ void quantize_lorenzo1d_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,
    double scale
);

// Dequantization requires prefix sum for prediction restoration
template<typename InputT, typename OutputT>
__global__ void dequantize_lorenzo1d_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale
);
```

#### Step 2.4: Block Transform Kernels (Method 3 - ZFP-style)

```cuda
// Process data in blocks of 4 elements
// Apply lifting-based orthogonal transform

template<typename InputT, typename OutputT>
__global__ void quantize_block_transform_kernel(
    const InputT* input,
    OutputT* output,
    size_t num_elements,    // Must be multiple of 4
    double scale
);

template<typename InputT, typename OutputT>
__global__ void dequantize_block_transform_kernel(
    const OutputT* input,
    InputT* output,
    size_t num_elements,
    double inv_scale
);
```

### Phase 3: Pipeline Integration

#### Step 3.1: Modify GPU_Compress.cpp

Add command-line arguments:
```
--quant-type [linear|lorenzo|block]   Quantization method
--error-bound <value>                  Absolute error bound (e.g., 0.001)
```

Integration flow:
```
Read input (GDS) → Quantize → Shuffle (optional) → Compress (nvcomp)
```

#### Step 3.2: Modify GPU_Decompress.cpp

Read quantization metadata from header, apply inverse:
```
Decompress → Unshuffle (if applied) → Dequantize → Write output (GDS)
```

### Phase 4: Testing & Benchmarking

#### Step 4.1: Test Script
**File**: `scripts/test_quantization.sh`

Test matrix:
- 3 quantization methods × 8 nvcomp algorithms × with/without shuffle
- Verify error bounds respected
- Measure compression ratios

#### Step 4.2: Synthetic Data Generator
**File**: `scripts/generate_test_data.py` or CUDA kernel

Generate test patterns:
- Smooth gradients (benefits Lorenzo/block transform)
- Random noise (baseline comparison)
- Mixed patterns (realistic scenario)

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantization.cuh` | **CREATE** | Header: enums, config structs, API declarations |
| `src/quantization_kernels.cu` | **CREATE** | GPU kernels for all 3 methods + adaptive precision |
| `src/compression_header.h` | **MODIFY** | Add quant metadata fields, bump version to 2 |
| `src/GPU_Compress.cpp` | **MODIFY** | Add CLI args, integrate quantization preprocessing |
| `src/GPU_Decompress.cpp` | **MODIFY** | Read quant metadata, apply dequantization |
| `CMakeLists.txt` | **MODIFY** | Add quantization_kernels.cu to build |
| `scripts/test_quantization.sh` | **CREATE** | Comprehensive benchmark script |
| `scripts/generate_test_data.cu` | **CREATE** | CUDA-based synthetic data generator |

---

## Implementation Order (Iterative)

### Iteration 1: Linear Quantization (Baseline)
1. Create `quantization.cuh` with full API design
2. Implement `quantize_linear_kernel` for float32 with int16 output
3. Implement `dequantize_linear_kernel`
4. Integrate into GPU_Compress.cpp / GPU_Decompress.cpp
5. Test with single nvcomp algorithm (LZ4)
6. **Checkpoint**: Verify error bound guarantee

### Iteration 2: Adaptive Precision
1. Add int8 and int32 variants
2. Implement `compute_required_precision()`
3. Add float64 support
4. Test precision selection across data ranges
5. **Checkpoint**: Verify adaptive selection works correctly

### Iteration 3: Lorenzo 1D Prediction
1. Implement `quantize_lorenzo1d_kernel`
2. Implement `dequantize_lorenzo1d_kernel` with prefix sum
3. Compare compression ratios vs linear
4. **Checkpoint**: Lorenzo should beat linear on smooth data

### Iteration 4: Block Transform (ZFP-style)
1. Implement 4-element block lifting transform
2. Implement inverse transform
3. Handle non-multiple-of-4 data sizes
4. Compare with other methods
5. **Checkpoint**: Block transform should excel on structured data

### Iteration 5: Full Testing & Benchmarking
1. Create comprehensive test script
2. Test all combinations: 3 methods × 8 algorithms × ±shuffle
3. Generate benchmark report
4. Document best practices per data type

---

## Expected Compression Ratio Improvements

| Data Pattern | nvcomp Only | + Linear Quant | + Lorenzo | + Block Transform |
|--------------|-------------|----------------|-----------|-------------------|
| Smooth gradients | 2-4x | 15-30x | **40-80x** | 30-60x |
| Sensor readings | 1.5-2x | 8-15x | **15-30x** | 10-20x |
| Random noise | ~1x | 3-5x | 3-5x | **4-8x** |
| Structured grids | 2-3x | 10-20x | 20-40x | **50-100x** |

*Bold = best method for that data pattern*

---

## Verification Strategy

### 1. Error Bound Verification (Critical)
```cpp
// After round-trip: compress → decompress
__global__ void verify_error_bound_kernel(
    const float* original,
    const float* restored,
    float error_bound,
    int* violation_count,
    float* max_error
);

// Must pass: max_error <= error_bound, violation_count == 0
```

### 2. Round-Trip Test
```bash
# For each test case:
./gpu_compress input.bin compressed.bin --quant-type linear --error-bound 0.001
./gpu_decompress compressed.bin restored.bin
# Verify: max|original - restored| <= 0.001
```

### 3. Statistical Metrics
- **Max Error**: Must be ≤ error_bound
- **RMSE**: Root mean square error
- **PSNR**: Peak signal-to-noise ratio (for quality assessment)
- **Compression Ratio**: compressed_size / original_size

### 4. Benchmark Comparisons
Test against baseline configurations:
- Raw nvcomp (no preprocessing)
- Shuffle-only (existing)
- Quant-only (each method)
- Quant + Shuffle (combined)

---

## Command-Line Interface (Final)

```bash
# Compression with quantization
./gpu_compress <input> <output> [algorithm] [options]

Options:
  --quant-type <type>     Quantization method: linear, lorenzo, block (default: none)
  --error-bound <value>   Absolute error bound (required if quant-type set)
  --shuffle <size>        Byte shuffle element size: 2, 4, 8 (default: 0 = disabled)

# Examples:
./gpu_compress data.bin out.bin lz4 --quant-type linear --error-bound 0.001
./gpu_compress data.bin out.bin zstd --quant-type lorenzo --error-bound 0.0001 --shuffle 4
./gpu_compress data.bin out.bin ans --quant-type block --error-bound 0.01

# Decompression (auto-detects settings from header)
./gpu_decompress compressed.bin restored.bin
```

---

## Future Work (Out of Scope)

- **REL (Relative) error bound**: `|error| / |value| <= bound`
- **VR (Value-Range) error bound**: `bound = (max-min) * percentage`
- **2D/3D Lorenzo predictors**: For multi-dimensional arrays
- **Huffman encoding**: Variable-length coding after quantization
- **Mixed-precision**: Different precision per data region
