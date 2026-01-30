# GPU Error-Bound Quantization

This document explains the quantization preprocessing system in GPUCompress, which converts floating-point data to smaller integer representations while guaranteeing a maximum error bound.

## Overview

### What is Quantization?

Quantization is a **lossy preprocessing step** that converts floating-point values (float32/float64) into smaller integer representations (int8/int16/int32). This dramatically improves compression ratios when used with lossless compressors like nvcomp.

### Why Use Quantization?

| Without Quantization | With Quantization |
|---------------------|-------------------|
| float32 (4 bytes/element) | int8 (1 byte/element) = **4x reduction** |
| float64 (8 bytes/element) | int16 (2 bytes/element) = **4x reduction** |
| Random-looking bit patterns | Structured integer patterns |
| Poor compression | Excellent compression |

### The Trade-off

You trade **controlled precision loss** for **much higher compression ratios**. The error bound guarantees:

```
|original_value - restored_value| <= error_bound
```

---

## Core Concepts

### 1. Error Bound

The error bound (`eb`) is the maximum allowed difference between original and restored values:

```cpp
double error_bound = 0.001;  // Max error of ±0.001
```

### 2. Quantization Scale

The scale factor converts floating-point values to integers:

```cpp
scale = 1.0 / (2.0 * error_bound)
```

**Why `2 * error_bound`?**

When you quantize using `round()`, the worst-case rounding error is ±0.5 in the integer domain. Converting back:
```
max_error = 0.5 * (1/scale) = 0.5 * 2 * error_bound = error_bound
```

### 3. Centering (Offset)

Data is centered around its minimum value to:
- Reduce the range of quantized values
- Allow smaller integer types (int8/int16)
- Improve compression of the quantized stream

```cpp
centered_value = original_value - data_min
quantized = round(centered_value * scale)
```

### 4. Precision Selection

The system automatically selects the smallest integer type that can hold all quantized values:

```cpp
// Number of bins needed
num_bins = data_range / (2.0 * error_bound)

if (num_bins <= 127)        → int8  (-128 to 127)
else if (num_bins <= 32767) → int16 (-32768 to 32767)
else                        → int32
```

---

## Data Structures

### QuantizationType Enum

```cpp
enum class QuantizationType : uint32_t {
    NONE = 0,    // No quantization
    LINEAR = 1   // Linear quantization
};
```

### QuantizationConfig

Configuration for the quantization operation:

```cpp
struct QuantizationConfig {
    QuantizationType type;           // LINEAR
    QuantizationPrecision precision; // AUTO, INT8, INT16, or INT32
    double error_bound;              // Max allowed error (must be > 0)
    size_t num_elements;             // Number of elements to process
    size_t element_size;             // 4 (float) or 8 (double)
};
```

### QuantizationResult

Output metadata needed for dequantization:

```cpp
struct QuantizationResult {
    void* d_quantized;            // Device pointer to quantized data
    size_t quantized_bytes;       // Output size in bytes
    int actual_precision;         // 8, 16, or 32 bits used
    double data_min;              // Minimum value (offset)
    double data_max;              // Maximum value
    double scale_factor;          // Quantization scale used
    double error_bound;           // Error bound used
    QuantizationType type;        // Method used
    size_t num_elements;          // Element count
    size_t original_element_size; // 4 or 8 bytes
};
```

---

## Linear Quantization Algorithm

Each element is independently quantized - simple and fully parallel on GPU.

### Algorithm

```
For each element:
    1. centered = value - data_min
    2. quantized = round(centered * scale)
    3. Clamp to output type range
```

### Visual Example

```
Original:  [1.002, 1.005, 1.001, 1.004]
data_min:  1.0
error_bound: 0.001
scale:     1 / (2 * 0.001) = 500

Centered:  [0.002, 0.005, 0.001, 0.004]
Quantized: [round(0.002*500), round(0.005*500), ...]
         = [1, 2, 0, 2]  (int8 values)
```

### GPU Kernel

```cpp
template<typename InputT, typename OutputT>
__global__ void quantize_linear_kernel(
    const InputT* input, OutputT* output,
    size_t num_elements, double scale, double offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
        double val = static_cast<double>(input[i]);
        double centered = val - offset;
        double quantized = round(centered * scale);
        output[i] = static_cast<OutputT>(quantized);
    }
}
```

---

## Usage Examples

### Command Line

```bash
# Compress with quantization
./gpu_compress input.bin output.bin lz4 --quant-type linear --error-bound 0.001

# Combine with byte shuffle for best compression
./gpu_compress input.bin output.bin zstd --quant-type linear --error-bound 0.001 --shuffle 4
```

### C++ API

```cpp
#include "quantization.cuh"

// Quantize
float* d_input = ...;  // Your GPU data
size_t num_elements = 1000000;

QuantizationConfig config(
    QuantizationType::LINEAR,
    0.001,              // error_bound
    num_elements,
    sizeof(float)
);

QuantizationResult result = quantize_simple(d_input, num_elements, sizeof(float), config);

// result.d_quantized contains int8/16/32 data
// result.quantized_bytes is the output size
// result.actual_precision tells you which type was chosen

// ... compress result.d_quantized with nvcomp ...

// Dequantize
void* d_restored = dequantize_simple(result.d_quantized, result);
// d_restored contains float data within error_bound of original

// Cleanup
cudaFree(result.d_quantized);
cudaFree(d_restored);
```

---

## Performance Considerations

### Compression Pipeline

For best results, combine quantization with byte shuffle:

```
Original Data (float32)
    ↓
Quantization (float32 → int16)  [2x reduction]
    ↓
Byte Shuffle (reorder bytes)    [improves compressibility]
    ↓
nvcomp Compression              [additional 2-10x reduction]
```

### Error Bound Selection

| Error Bound | Precision | Use Case |
|-------------|-----------|----------|
| 0.1         | int8      | Low precision, max compression |
| 0.01        | int16     | Good balance |
| 0.001       | int16     | High precision |
| 0.0001      | int32     | Very high precision |

### Parallel Efficiency

Linear quantization is fully parallel - each thread processes independent elements with no synchronization required. Performance scales linearly with GPU compute units.
