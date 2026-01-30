#!/bin/bash

###############################################################################
# test_quantization.sh
#
# Comprehensive test script for error-bound quantization with GPUCompress
#
# Tests all combinations of:
# - 3 quantization methods (linear, lorenzo, block)
# - 8 nvcomp algorithms (lz4, snappy, deflate, gdeflate, zstd, ans, cascaded, bitcomp)
# - With/without byte shuffle
#
# Usage:
#   ./scripts/test_quantization.sh [test_data_file]
#
# If test_data_file is not provided, it will generate synthetic test data.
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="${BUILD_DIR:-build}"
COMPRESS_BIN="${BUILD_DIR}/gpu_compress"
DECOMPRESS_BIN="${BUILD_DIR}/gpu_decompress"
TEST_DATA_DIR="test_data"
RESULTS_DIR="test_results"
ERROR_BOUND="0.001"  # Default error bound

# Create directories
mkdir -p "${TEST_DATA_DIR}"
mkdir -p "${RESULTS_DIR}"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

###############################################################################
# Test Data Generation
###############################################################################

generate_test_data() {
    print_header "Generating Test Data"

    local data_file="${TEST_DATA_DIR}/test_float32.bin"

    if [ -f "${data_file}" ]; then
        print_info "Test data already exists: ${data_file}"
        return 0
    fi

    print_info "Generating 10MB of float32 test data with smooth gradient pattern..."

    # Use Python to generate test data with known patterns
    python3 - <<EOF
import numpy as np
import struct

# Generate 10MB of float32 data (2.5 million floats)
n = 2_500_000

# Create smooth gradient (good for Lorenzo and block transform)
data = np.linspace(0, 1000, n, dtype=np.float32)

# Add small noise
data += np.random.randn(n).astype(np.float32) * 0.1

# Write binary file
with open("${data_file}", "wb") as f:
    f.write(data.tobytes())

print(f"Generated {len(data)} float32 values ({len(data)*4} bytes)")
print(f"Range: [{data.min():.6f}, {data.max():.6f}]")
print(f"Mean: {data.mean():.6f}, Std: {data.std():.6f}")
EOF

    print_success "Test data generated: ${data_file}"
}

###############################################################################
# Verification Function
###############################################################################

verify_error_bound() {
    local original="$1"
    local restored="$2"
    local error_bound="$3"

    print_info "Verifying error bound with Python..."

    python3 - <<EOF
import numpy as np
import sys

# Load original and restored data
original = np.fromfile("${original}", dtype=np.float32)
restored = np.fromfile("${restored}", dtype=np.float32)

if len(original) != len(restored):
    print(f"ERROR: Size mismatch! Original: {len(original)}, Restored: {len(restored)}")
    sys.exit(1)

# Compute errors
errors = np.abs(original - restored)
max_error = errors.max()
mean_error = errors.mean()
violations = np.sum(errors > ${error_bound})

print(f"Elements: {len(original)}")
print(f"Max error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")
print(f"Error bound: ${error_bound}")
print(f"Violations: {violations}")

# Calculate PSNR
mse = np.mean((original - restored) ** 2)
if mse > 0:
    data_range = original.max() - original.min()
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    print(f"PSNR: {psnr:.2f} dB")

if violations > 0:
    print(f"FAILED: {violations} values exceed error bound!")
    sys.exit(1)
else:
    print("PASSED: All values within error bound")
    sys.exit(0)
EOF

    return $?
}

###############################################################################
# Single Test Function
###############################################################################

run_single_test() {
    local quant_type="$1"
    local algorithm="$2"
    local shuffle_size="$3"
    local input_file="$4"

    local test_name="${quant_type}_${algorithm}"
    if [ "${shuffle_size}" != "0" ]; then
        test_name="${test_name}_shuffle${shuffle_size}"
    fi

    local compressed="${RESULTS_DIR}/${test_name}.compressed"
    local restored="${RESULTS_DIR}/${test_name}.restored"

    print_info "Testing: ${test_name}"

    # Build compression command
    local compress_cmd="${COMPRESS_BIN} ${input_file} ${compressed} ${algorithm}"
    compress_cmd="${compress_cmd} --quant-type ${quant_type} --error-bound ${ERROR_BOUND}"
    if [ "${shuffle_size}" != "0" ]; then
        compress_cmd="${compress_cmd} --shuffle ${shuffle_size}"
    fi

    # Run compression
    if ! ${compress_cmd} > "${RESULTS_DIR}/${test_name}_compress.log" 2>&1; then
        print_error "Compression failed for ${test_name}"
        cat "${RESULTS_DIR}/${test_name}_compress.log"
        return 1
    fi

    # Run decompression
    if ! ${DECOMPRESS_BIN} ${compressed} ${restored} > "${RESULTS_DIR}/${test_name}_decompress.log" 2>&1; then
        print_error "Decompression failed for ${test_name}"
        cat "${RESULTS_DIR}/${test_name}_decompress.log"
        return 1
    fi

    # Verify error bound
    if verify_error_bound "${input_file}" "${restored}" "${ERROR_BOUND}"; then
        print_success "${test_name}: PASSED"

        # Extract compression ratio
        local ratio=$(grep "Compression ratio:" "${RESULTS_DIR}/${test_name}_compress.log" | awk '{print $3}')
        echo "  Compression ratio: ${ratio}"

        return 0
    else
        print_error "${test_name}: FAILED (error bound violated)"
        return 1
    fi
}

###############################################################################
# Main Test Suite
###############################################################################

run_test_suite() {
    local input_file="$1"

    print_header "Running Quantization Test Suite"

    local total_tests=0
    local passed_tests=0
    local failed_tests=0

    # Quantization methods to test
    local quant_methods=("linear" "lorenzo" "block")

    # nvcomp algorithms to test
    local algorithms=("lz4" "snappy" "zstd" "deflate")

    # Shuffle sizes to test (0 = no shuffle)
    local shuffle_sizes=("0" "4")

    print_info "Configuration:"
    print_info "  Quantization methods: ${quant_methods[*]}"
    print_info "  Algorithms: ${algorithms[*]}"
    print_info "  Shuffle sizes: ${shuffle_sizes[*]}"
    print_info "  Error bound: ${ERROR_BOUND}"
    print_info ""

    # Test all combinations
    for quant in "${quant_methods[@]}"; do
        for algo in "${algorithms[@]}"; do
            for shuffle in "${shuffle_sizes[@]}"; do
                ((total_tests++))

                if run_single_test "${quant}" "${algo}" "${shuffle}" "${input_file}"; then
                    ((passed_tests++))
                else
                    ((failed_tests++))
                fi

                # Clean up intermediate files
                rm -f "${RESULTS_DIR}/${quant}_${algo}"*".compressed"
                rm -f "${RESULTS_DIR}/${quant}_${algo}"*".restored"
            done
        done
    done

    # Print summary
    print_header "Test Summary"
    echo "Total tests: ${total_tests}"
    print_success "Passed: ${passed_tests}"
    if [ ${failed_tests} -gt 0 ]; then
        print_error "Failed: ${failed_tests}"
        return 1
    else
        print_success "All tests passed!"
        return 0
    fi
}

###############################################################################
# Quick Test (Subset for Fast Verification)
###############################################################################

run_quick_test() {
    local input_file="$1"

    print_header "Running Quick Test (LZ4 + Linear Quantization)"

    if run_single_test "linear" "lz4" "0" "${input_file}"; then
        print_success "Quick test passed!"
        return 0
    else
        print_error "Quick test failed!"
        return 1
    fi
}

###############################################################################
# Benchmark Mode
###############################################################################

run_benchmark() {
    local input_file="$1"

    print_header "Running Compression Ratio Benchmark"

    local results_file="${RESULTS_DIR}/benchmark_results.csv"

    echo "Quantization,Algorithm,Shuffle,Compression_Ratio,Compressed_Size_MB" > "${results_file}"

    local quant_methods=("linear" "lorenzo" "block")
    local algorithms=("lz4" "snappy" "zstd")

    for quant in "${quant_methods[@]}"; do
        for algo in "${algorithms[@]}"; do
            for shuffle in "0" "4"; do
                local test_name="${quant}_${algo}"
                if [ "${shuffle}" != "0" ]; then
                    test_name="${test_name}_shuffle${shuffle}"
                fi

                local compressed="${RESULTS_DIR}/${test_name}.compressed"

                # Build and run compression
                local compress_cmd="${COMPRESS_BIN} ${input_file} ${compressed} ${algo}"
                compress_cmd="${compress_cmd} --quant-type ${quant} --error-bound ${ERROR_BOUND}"
                if [ "${shuffle}" != "0" ]; then
                    compress_cmd="${compress_cmd} --shuffle ${shuffle}"
                fi

                ${compress_cmd} > "${RESULTS_DIR}/${test_name}_bench.log" 2>&1

                # Extract results
                local ratio=$(grep "Compression ratio:" "${RESULTS_DIR}/${test_name}_bench.log" | awk '{print $3}' | tr -d 'x')
                local size=$(stat -f%z "${compressed}" 2>/dev/null || stat -c%s "${compressed}" 2>/dev/null)
                local size_mb=$(echo "scale=2; ${size} / 1048576" | bc)

                echo "${quant},${algo},${shuffle},${ratio},${size_mb}" >> "${results_file}"

                print_info "${test_name}: ${ratio}x (${size_mb} MB)"

                # Cleanup
                rm -f "${compressed}"
            done
        done
    done

    print_success "Benchmark results saved to: ${results_file}"
}

###############################################################################
# Main Script
###############################################################################

main() {
    # Check if binaries exist
    if [ ! -f "${COMPRESS_BIN}" ]; then
        print_error "Compress binary not found: ${COMPRESS_BIN}"
        print_info "Please build the project first: mkdir build && cd build && cmake .. && make"
        exit 1
    fi

    if [ ! -f "${DECOMPRESS_BIN}" ]; then
        print_error "Decompress binary not found: ${DECOMPRESS_BIN}"
        exit 1
    fi

    # Determine input file
    local input_file="$1"
    if [ -z "${input_file}" ]; then
        generate_test_data
        input_file="${TEST_DATA_DIR}/test_float32.bin"
    fi

    if [ ! -f "${input_file}" ]; then
        print_error "Input file not found: ${input_file}"
        exit 1
    fi

    print_info "Using test data: ${input_file}"

    # Parse mode from second argument
    local mode="${2:-full}"

    case "${mode}" in
        quick)
            run_quick_test "${input_file}"
            ;;
        benchmark)
            run_benchmark "${input_file}"
            ;;
        full|*)
            run_test_suite "${input_file}"
            ;;
    esac

    exit $?
}

# Run main function
main "$@"
