#!/bin/bash
# Simple compression verification test
# Tests compression/decompression and verifies data integrity with sample comparisons

set -e

PROJECT_DIR="/home/cc/GPUCompress"
BUILD_DIR="$PROJECT_DIR/build"
TEST_DIR="$PROJECT_DIR/test_verify"

COMPRESS="$BUILD_DIR/gpu_compress"
DECOMPRESS="$BUILD_DIR/gpu_decompress"
COMPARE="$BUILD_DIR/compare_data"

# Create test directory
mkdir -p "$TEST_DIR"

echo "========================================"
echo "  GPU Compression Verification Test"
echo "========================================"

# -----------------------------------------------------------------------------
# Step 1: Generate simple test data (floats from 0 to 999)
# -----------------------------------------------------------------------------
echo ""
echo "[1] Generating test data..."

TEST_DATA="$TEST_DIR/test_input.bin"
python3 << 'EOF'
import struct
import math

# Generate 10000 floats with a simple pattern
data = []
for i in range(10000):
    # Mix of smooth and varying data
    val = math.sin(i * 0.01) * 100 + i * 0.1
    data.append(val)

with open("/home/cc/GPUCompress/test_verify/test_input.bin", "wb") as f:
    f.write(struct.pack(f'{len(data)}f', *data))

print(f"  Created {len(data)} floats ({len(data)*4} bytes)")
print(f"  Sample values [0:5]: {[f'{v:.4f}' for v in data[:5]]}")
print(f"  Sample values [5000:5005]: {[f'{v:.4f}' for v in data[5000:5005]]}")
EOF

# -----------------------------------------------------------------------------
# Step 2: Test configurations
# -----------------------------------------------------------------------------
run_verify_test() {
    local name="$1"
    local algo="$2"
    local extra_args="$3"
    local is_lossy="$4"

    local compressed="$TEST_DIR/${name}.compressed"
    local restored="$TEST_DIR/${name}.restored"

    echo ""
    echo "----------------------------------------"
    echo "Test: $name"
    echo "  Algorithm: $algo"
    echo "  Options: ${extra_args:-none}"
    echo "----------------------------------------"

    # Compress
    echo "  Compressing..."
    $COMPRESS "$TEST_DATA" "$compressed" "$algo" $extra_args

    # Get sizes
    orig_size=$(stat -c%s "$TEST_DATA")
    comp_size=$(stat -c%s "$compressed")
    ratio=$(echo "scale=2; $orig_size / $comp_size" | bc)
    echo "  Original: $orig_size bytes -> Compressed: $comp_size bytes (ratio: ${ratio}x)"

    # Decompress
    echo "  Decompressing..."
    $DECOMPRESS "$compressed" "$restored"

    rest_size=$(stat -c%s "$restored")
    echo "  Restored size: $rest_size bytes"

    # Size check
    if [ "$orig_size" != "$rest_size" ]; then
        echo "  ERROR: Size mismatch!"
        return 1
    fi

    # Detailed comparison with sample values
    echo ""
    echo "  === Data Verification ==="
    python3 << PYEOF
import struct

with open("$TEST_DATA", "rb") as f:
    orig_bytes = f.read()
with open("$restored", "rb") as f:
    rest_bytes = f.read()

n = len(orig_bytes) // 4
orig = struct.unpack(f'{n}f', orig_bytes)
rest = struct.unpack(f'{n}f', rest_bytes)

# Compute errors
errors = [abs(o - r) for o, r in zip(orig, rest)]
max_err = max(errors)
mean_err = sum(errors) / len(errors)
rmse = (sum(e*e for e in errors) / len(errors)) ** 0.5

# Find indices with largest errors
sorted_idx = sorted(range(len(errors)), key=lambda i: errors[i], reverse=True)

print(f"  Total elements: {n}")
print(f"  Max error:  {max_err:.6e}")
print(f"  Mean error: {mean_err:.6e}")
print(f"  RMSE:       {rmse:.6e}")

# Check if lossless
if max_err == 0:
    print("  Status: LOSSLESS (exact match)")
else:
    print("  Status: LOSSY")

    # Show sample comparisons
    print("")
    print("  Sample comparisons (original -> restored -> diff):")
    for i in [0, 1, 2, n//2, n//2+1, n-2, n-1]:
        diff = orig[i] - rest[i]
        print(f"    [{i:5d}]: {orig[i]:12.6f} -> {rest[i]:12.6f} (diff: {diff:+.6e})")

    # Show worst errors
    print("")
    print("  Largest errors:")
    for i in sorted_idx[:5]:
        print(f"    [{i:5d}]: {orig[i]:12.6f} -> {rest[i]:12.6f} (err: {errors[i]:.6e})")
PYEOF

    # Run the C++ compare tool for additional metrics
    echo ""
    echo "  === Compare Tool Output ==="
    if [ "$is_lossy" = "yes" ]; then
        $COMPARE "$TEST_DATA" "$restored" float 0.01 2>&1 | grep -E "(Max|Mean|RMSE|PSNR|Correlation|PASSED|FAILED)" | sed 's/^/  /'
    else
        $COMPARE "$TEST_DATA" "$restored" float 2>&1 | grep -E "(Max|Mean|RMSE|PSNR|Correlation)" | sed 's/^/  /'
    fi

    # Cleanup
    rm -f "$compressed" "$restored"
}

# -----------------------------------------------------------------------------
# Run Tests
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  Running Verification Tests"
echo "========================================"

# Test 1: Lossless compression only
run_verify_test "lossless_deflate" "deflate" "" "no"

# Test 2: Lossless with shuffle
run_verify_test "lossless_shuffle" "deflate" "--shuffle 4" "no"

# Test 3: Lossy with quantization
run_verify_test "lossy_quant" "deflate" "--quant-type linear --error-bound 0.01" "yes"

# Test 4: Lossy with quantization + shuffle
run_verify_test "lossy_quant_shuffle" "deflate" "--quant-type linear --error-bound 0.01 --shuffle 4" "yes"

# Test 5: Tighter error bound
run_verify_test "lossy_tight" "deflate" "--quant-type linear --error-bound 0.0001" "yes"

# Test 6: zstd algorithm
run_verify_test "zstd_lossless" "zstd" "" "no"

# Test 7: zstd with quantization
run_verify_test "zstd_lossy" "zstd" "--quant-type linear --error-bound 0.001" "yes"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  All tests completed!"
echo "========================================"
rm -f "$TEST_DATA"
rmdir "$TEST_DIR" 2>/dev/null || true
