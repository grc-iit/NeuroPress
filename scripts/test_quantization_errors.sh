#!/bin/bash
# Quantization Error Analysis Test Script
# Focuses on quantization-based compression and verifies error bounds
# Prints detailed comparisons of original vs restored values

set -e

PROJECT_DIR="/home/cc/GPUCompress"
BUILD_DIR="$PROJECT_DIR/build"
TEST_DIR="$PROJECT_DIR/test_results"

COMPRESS="$BUILD_DIR/gpu_compress"
DECOMPRESS="$BUILD_DIR/gpu_decompress"

PATTERNS=("smooth" "turbulent" "periodic" "noisy")
ALGORITHMS=("deflate" "zstd")
ERROR_BOUNDS=("0.01" "0.001" "0.0001")

echo "========================================================"
echo "    Quantization Error Analysis Test Suite"
echo "========================================================"

run_quantization_test() {
    local input_file="$1"
    local algorithm="$2"
    local error_bound="$3"
    local shuffle_size="$4"
    local pattern="$5"

    local test_name="${pattern}_${algorithm}_eb${error_bound}"
    [ "$shuffle_size" != "0" ] && test_name="${test_name}_shuffle"

    local compressed_file="$TEST_DIR/${test_name}.compressed"
    local restored_file="$TEST_DIR/${test_name}.restored"

    echo ""
    echo "============================================================"
    echo "Test: $test_name"
    echo "  Pattern:     $pattern"
    echo "  Algorithm:   $algorithm"
    echo "  Error Bound: $error_bound"
    echo "  Shuffle:     $([ "$shuffle_size" = "0" ] && echo "disabled" || echo "enabled ($shuffle_size bytes)")"
    echo "============================================================"

    # Build compression command
    local compress_cmd="$COMPRESS $input_file $compressed_file $algorithm --quant-type linear --error-bound $error_bound"
    [ "$shuffle_size" != "0" ] && compress_cmd="$compress_cmd --shuffle $shuffle_size"

    # Compress
    echo "Compressing..."
    if ! $compress_cmd > /dev/null 2>&1; then
        echo "FAILED: Compression error"
        return 1
    fi

    # Get sizes and ratio
    local original_size=$(stat -c%s "$input_file")
    local compressed_size=$(stat -c%s "$compressed_file")
    local ratio=$(echo "scale=2; $original_size / $compressed_size" | bc)

    echo "  Original size:   $original_size bytes"
    echo "  Compressed size: $compressed_size bytes"
    echo "  Compression ratio: ${ratio}x"

    # Decompress
    echo "Decompressing..."
    if ! $DECOMPRESS "$compressed_file" "$restored_file" > /dev/null 2>&1; then
        echo "FAILED: Decompression error"
        rm -f "$compressed_file"
        return 1
    fi

    # Detailed Python analysis
    echo ""
    echo "--- Value Comparison Analysis ---"
    python3 << PYEOF
import struct
import sys

# Read original and restored data
with open("$input_file", "rb") as f:
    orig_bytes = f.read()
with open("$restored_file", "rb") as f:
    rest_bytes = f.read()

n = len(orig_bytes) // 4
orig = struct.unpack(f'{n}f', orig_bytes)
rest = struct.unpack(f'{n}f', rest_bytes)

error_bound = $error_bound

# Compute errors for each element
errors = [abs(o - r) for o, r in zip(orig, rest)]
max_err = max(errors)
mean_err = sum(errors) / len(errors)
rmse = (sum(e*e for e in errors) / len(errors)) ** 0.5

# Count violations
violations = [(i, errors[i]) for i in range(len(errors)) if errors[i] > error_bound]
num_violations = len(violations)

# Data range for context
data_min = min(orig)
data_max = max(orig)
data_range = data_max - data_min

print(f"Total elements:    {n}")
print(f"Data range:        [{data_min:.6f}, {data_max:.6f}] (span: {data_range:.6f})")
print(f"Error bound:       {error_bound}")
print("")
print(f"Max error:         {max_err:.6e}")
print(f"Mean error:        {mean_err:.6e}")
print(f"RMSE:              {rmse:.6e}")
print(f"Max/Bound ratio:   {max_err/error_bound:.4f}x")
print("")

# Sample value comparisons at different positions
print("--- Sample Value Comparisons ---")
print(f"{'Index':>8} | {'Original':>14} | {'Restored':>14} | {'Difference':>14} | {'Within Bound':>12}")
print("-" * 75)

# Select sample indices across the data
sample_indices = [0, 1, 2, n//4, n//2, 3*n//4, n-3, n-2, n-1]
for i in sample_indices:
    diff = orig[i] - rest[i]
    within = abs(diff) <= error_bound
    status = "YES" if within else "NO !!!"
    print(f"{i:>8} | {orig[i]:>14.6f} | {rest[i]:>14.6f} | {diff:>+14.6e} | {status:>12}")

# Find and display largest errors
sorted_idx = sorted(range(len(errors)), key=lambda i: errors[i], reverse=True)

print("")
print("--- Largest Errors (Top 10) ---")
print(f"{'Index':>8} | {'Original':>14} | {'Restored':>14} | {'Error':>14} | {'Within Bound':>12}")
print("-" * 75)
for i in sorted_idx[:10]:
    within = errors[i] <= error_bound
    status = "YES" if within else "NO !!!"
    print(f"{i:>8} | {orig[i]:>14.6f} | {rest[i]:>14.6f} | {errors[i]:>14.6e} | {status:>12}")

# Error distribution
print("")
print("--- Error Distribution ---")
thresholds = [0.1, 0.5, 0.9, 0.99, 1.0]
for t in thresholds:
    count = sum(1 for e in errors if e <= error_bound * t)
    pct = 100.0 * count / len(errors)
    print(f"  Errors <= {t*100:5.1f}% of bound: {count:>8} ({pct:6.2f}%)")

# Final verdict
print("")
print("=" * 50)
if num_violations == 0:
    print(f"RESULT: PASSED - All {n} values within error bound")
else:
    print(f"RESULT: FAILED - {num_violations} values ({100.0*num_violations/n:.2f}%) exceed error bound")
    print("")
    print("First 5 violations:")
    for i, err in violations[:5]:
        print(f"  Index {i}: error={err:.6e} (bound={error_bound})")
print("=" * 50)

# Exit with error code if violations found
sys.exit(0 if num_violations == 0 else 1)
PYEOF

    local py_result=$?

    # Cleanup
    rm -f "$compressed_file" "$restored_file"

    return $py_result
}

# =============================================================================
# Generate test data if needed
# =============================================================================
generate_test_data() {
    echo ""
    echo "Checking for test data..."

    local need_generation=false
    for pattern in "${PATTERNS[@]}"; do
        if [ ! -f "$TEST_DIR/${pattern}_pattern.bin" ]; then
            need_generation=true
            break
        fi
    done

    if [ "$need_generation" = true ]; then
        echo "Generating test data patterns..."
        mkdir -p "$TEST_DIR"

        python3 << 'PYEOF'
import struct
import math
import os

test_dir = "/home/cc/GPUCompress/test_results"
os.makedirs(test_dir, exist_ok=True)

num_elements = 100000  # 100K floats = 400KB each

patterns = {
    "smooth": lambda i: math.sin(i * 0.001) * 1000 + i * 0.01,
    "turbulent": lambda i: math.sin(i * 0.1) * math.cos(i * 0.07) * 500 + (i % 100) * 0.5,
    "periodic": lambda i: math.sin(i * 0.05) * 100 + math.sin(i * 0.13) * 50,
    "noisy": lambda i: math.sin(i * 0.01) * 100 + ((i * 17) % 37 - 18) * 0.5,
}

for name, func in patterns.items():
    data = [func(i) for i in range(num_elements)]
    filepath = os.path.join(test_dir, f"{name}_pattern.bin")
    with open(filepath, "wb") as f:
        f.write(struct.pack(f'{len(data)}f', *data))
    print(f"  Created {name}_pattern.bin: {len(data)} floats, range [{min(data):.2f}, {max(data):.2f}]")
PYEOF
    else
        echo "Test data already exists."
    fi
}

# =============================================================================
# Main test execution
# =============================================================================
mkdir -p "$TEST_DIR"
generate_test_data

total_tests=0
passed_tests=0
failed_tests=0

# -----------------------------------------------------------------------------
# CONFIG 1: Quantization without shuffle
# -----------------------------------------------------------------------------
echo ""
echo "###########################################################"
echo "#  QUANTIZATION TESTS (without shuffle)                   #"
echo "###########################################################"

for pattern in "${PATTERNS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        for eb in "${ERROR_BOUNDS[@]}"; do
            total_tests=$((total_tests + 1))
            if run_quantization_test "$TEST_DIR/${pattern}_pattern.bin" "$algo" "$eb" "0" "$pattern"; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
        done
    done
done

# -----------------------------------------------------------------------------
# CONFIG 2: Quantization with shuffle
# -----------------------------------------------------------------------------
echo ""
echo "###########################################################"
echo "#  QUANTIZATION TESTS (with shuffle)                      #"
echo "###########################################################"

for pattern in "${PATTERNS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        for eb in "${ERROR_BOUNDS[@]}"; do
            total_tests=$((total_tests + 1))
            if run_quantization_test "$TEST_DIR/${pattern}_pattern.bin" "$algo" "$eb" "4" "$pattern"; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
        done
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================================"
echo "                    TEST SUMMARY"
echo "========================================================"
echo "Total tests:  $total_tests"
echo "Passed:       $passed_tests"
echo "Failed:       $failed_tests"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo "SUCCESS: All quantization tests passed!"
    echo "All restored values are within their specified error bounds."
    exit 0
else
    echo "WARNING: $failed_tests test(s) failed!"
    echo "Some restored values exceeded their specified error bounds."
    exit 1
fi
