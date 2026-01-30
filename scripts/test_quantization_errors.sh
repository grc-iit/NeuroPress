#!/bin/bash
# Quantization Error Analysis Test Script
# Focuses on quantization-based compression and verifies error bounds
# Prints detailed comparisons of original vs restored values
#
# Usage: ./test_quantization_errors.sh <input_file> [error_bound]
#   input_file:  Path to binary file containing float32 data
#   error_bound: Optional error bound (default: 0.01)

set -e

PROJECT_DIR="/home/cc/GPUCompress"
BUILD_DIR="$PROJECT_DIR/build"
TEST_DIR="$PROJECT_DIR/test_data"

COMPRESS="$BUILD_DIR/gpu_compress"
DECOMPRESS="$BUILD_DIR/gpu_decompress"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [error_bound]"
    echo ""
    echo "Arguments:"
    echo "  input_file   Path to binary file containing float32 data"
    echo "  error_bound  Optional error bound (default: 0.01)"
    echo ""
    echo "Example:"
    echo "  $0 data.bin 0.001"
    exit 1
fi

INPUT_FILE="$1"
ERROR_BOUND="${2:-0.01}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

ALGORITHMS=("deflate")

echo "========================================================"
echo "    Quantization Error Analysis Test Suite"
echo "========================================================"
echo "Input file:  $INPUT_FILE"
echo "Error bound: $ERROR_BOUND"
echo "========================================================"

run_quantization_test() {
    local input_file="$1"
    local algorithm="$2"
    local error_bound="$3"
    local shuffle_size="$4"

    local test_name="${algorithm}_eb${error_bound}"
    [ "$shuffle_size" != "0" ] && test_name="${test_name}_shuffle"

    local compressed_file="$TEST_DIR/${test_name}.compressed"
    local restored_file="$TEST_DIR/${test_name}.restored"

    echo ""
    echo "============================================================"
    echo "Test: $test_name"
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
    local csv_file="$TEST_DIR/${test_name}_values.csv"

    echo ""
    echo "--- Value Comparison Analysis ---"
    python3 << PYEOF
import struct
import sys
import csv

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

# Write ALL values to CSV
csv_file = "$csv_file"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'original', 'restored', 'difference', 'abs_error', 'within_bound'])
    for i in range(n):
        diff = orig[i] - rest[i]
        within = 1 if errors[i] <= error_bound else 0
        writer.writerow([i, orig[i], rest[i], diff, errors[i], within])

print(f"Full comparison CSV: {csv_file}")
print(f"")
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
# Main test execution
# =============================================================================
mkdir -p "$TEST_DIR"

total_tests=0
passed_tests=0
failed_tests=0

# -----------------------------------------------------------------------------
# CONFIG 1: Quantization without shuffle
# -----------------------------------------------------------------------------
echo ""
echo "###########################################################"
echo "#  QUANTIZATION TEST (without shuffle)                    #"
echo "###########################################################"

for algo in "${ALGORITHMS[@]}"; do
    total_tests=$((total_tests + 1))
    if run_quantization_test "$INPUT_FILE" "$algo" "$ERROR_BOUND" "0"; then
        passed_tests=$((passed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
done

# -----------------------------------------------------------------------------
# CONFIG 2: Quantization with shuffle
# -----------------------------------------------------------------------------
echo ""
echo "###########################################################"
echo "#  QUANTIZATION TEST (with shuffle)                       #"
echo "###########################################################"

for algo in "${ALGORITHMS[@]}"; do
    total_tests=$((total_tests + 1))
    if run_quantization_test "$INPUT_FILE" "$algo" "$ERROR_BOUND" "4"; then
        passed_tests=$((passed_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
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

echo "CSV files with all value comparisons saved to: $TEST_DIR/"
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
