#!/bin/bash
# Compression test script with 4 configurations:
# 1. Compression only (no shuffle, no quantization)
# 2. Compression + shuffle (no quantization)
# 3. Compression + quantization (no shuffle)
# 4. Compression + quantization + shuffle
#
# Usage: ./run_simple_tests.sh [input1] [input2] ...
#   Inputs can be:
#     - Pattern names: smooth, periodic, noisy (maps to test_float32_<pattern>.bin)
#     - File paths: /path/to/file.bin or relative/path/file.bin
#
#   Examples:
#     ./run_simple_tests.sh                        # Run all default patterns
#     ./run_simple_tests.sh noisy                  # Run only noisy pattern
#     ./run_simple_tests.sh smooth noisy           # Run smooth and noisy patterns
#     ./run_simple_tests.sh /path/to/mydata.bin   # Run with custom file
#     ./run_simple_tests.sh test_data/custom.bin  # Run with relative path

set -e

PROJECT_DIR="/home/cc/GPUCompress"
BUILD_DIR="$PROJECT_DIR/build"
TEST_DIR="$PROJECT_DIR/test_data"

COMPRESS="$BUILD_DIR/gpu_compress"
DECOMPRESS="$BUILD_DIR/gpu_decompress"
COMPARE="$BUILD_DIR/compare_data"

# Use command-line arguments if provided, otherwise default to all patterns
if [ $# -gt 0 ]; then
    INPUTS=("$@")
else
    INPUTS=("smooth" "periodic" "noisy")
fi

ALGORITHMS=("deflate")
ERROR_BOUNDS=("0.01" "0.001" "0.0001")

# Function to resolve input to file path and pattern name
resolve_input() {
    local input="$1"

    # Check if it's a file path (contains / or ends with .bin)
    if [[ "$input" == *"/"* ]] || [[ "$input" == *.bin ]]; then
        # It's a file path - check multiple locations
        if [ -f "$input" ]; then
            # Found at given path (absolute or relative to cwd)
            INPUT_FILE="$input"
        elif [ -f "$TEST_DIR/$input" ]; then
            # Found in test_data directory
            INPUT_FILE="$TEST_DIR/$input"
        elif [ -f "$PROJECT_DIR/$input" ]; then
            # Found relative to project directory
            INPUT_FILE="$PROJECT_DIR/$input"
        else
            echo "Error: File not found: $input"
            echo "  Checked: $input"
            echo "  Checked: $TEST_DIR/$input"
            echo "  Checked: $PROJECT_DIR/$input"
            exit 1
        fi
        PATTERN_NAME=$(basename "$INPUT_FILE" .bin)
    else
        # It's a pattern name
        INPUT_FILE="$TEST_DIR/test_float32_${input}.bin"
        PATTERN_NAME="$input"
        if [ ! -f "$INPUT_FILE" ]; then
            echo "Error: File not found: $INPUT_FILE"
            exit 1
        fi
    fi
}

CSV_FILE="$TEST_DIR/compression_results.csv"

echo "========================================================"
echo "    GPU Compression Test Suite"
echo "========================================================"
echo "Inputs: ${INPUTS[*]}"
echo "========================================================"

# Write CSV header
echo "pattern,algorithm,quantization,error_bound,shuffle,original_bytes,compressed_bytes,decompressed_bytes,ratio,max_error,mean_error,rmse,psnr_db,correlation" > "$CSV_FILE"

run_test() {
    local input_file="$1"
    local algorithm="$2"
    local quant_type="$3"
    local error_bound="$4"
    local shuffle_size="$5"
    local pattern="$6"

    local test_name="${pattern}_${algorithm}"
    [ -n "$quant_type" ] && test_name="${test_name}_q${quant_type}_e${error_bound}"
    [ "$shuffle_size" != "0" ] && test_name="${test_name}_s${shuffle_size}"

    local compressed_file="$TEST_DIR/${test_name}.compressed"
    local restored_file="$TEST_DIR/${test_name}.restored"

    echo -n "  $test_name ... "

    local compress_cmd="$COMPRESS $input_file $compressed_file $algorithm"
    [ -n "$quant_type" ] && compress_cmd="$compress_cmd --quant-type $quant_type --error-bound $error_bound"
    [ "$shuffle_size" != "0" ] && compress_cmd="$compress_cmd --shuffle $shuffle_size"

    if ! $compress_cmd > /dev/null 2>&1; then
        echo "FAILED (compression)"
        return 1
    fi

    local original_size=$(stat -c%s "$input_file")
    local compressed_size=$(stat -c%s "$compressed_file")
    local ratio=$(echo "scale=2; $original_size / $compressed_size" | bc)

    $DECOMPRESS "$compressed_file" "$restored_file" > /dev/null 2>&1

    local decompressed_size=$(stat -c%s "$restored_file")
    local max_err mean_err rmse psnr corr

    # Only compute error metrics if quantization was used (lossy compression)
    if [ -n "$quant_type" ]; then
        local compare_out=$($COMPARE "$input_file" "$restored_file" float "$error_bound" 2>&1)
        max_err=$(echo "$compare_out" | grep "Max error:" | awk '{print $3}')
        mean_err=$(echo "$compare_out" | grep "Mean error:" | awk '{print $3}')
        rmse=$(echo "$compare_out" | grep "RMSE:" | awk '{print $2}')
        psnr=$(echo "$compare_out" | grep "PSNR:" | awk '{print $2}')
        corr=$(echo "$compare_out" | grep "Correlation:" | awk '{print $2}')
        printf "Ratio: %6sx | MaxErr: %12s | MeanErr: %12s | RMSE: %12s | PSNR: %8s dB\n" "$ratio" "$max_err" "$mean_err" "$rmse" "$psnr"
    else
        # Lossless compression - verify data matches exactly
        local compare_out=$($COMPARE "$input_file" "$restored_file" float 2>&1)
        max_err=$(echo "$compare_out" | grep "Max error:" | awk '{print $3}')

        # Verify lossless: max_err should be 0
        if [ "$max_err" = "0.000000e+00" ]; then
            printf "Ratio: %6sx | Lossless VERIFIED (exact match)\n" "$ratio"
            max_err="0"
            mean_err="0"
            rmse="0"
            psnr="inf"
            corr="1"
        else
            printf "Ratio: %6sx | ERROR: Lossless mismatch! MaxErr: %s\n" "$ratio" "$max_err"
            mean_err=$(echo "$compare_out" | grep "Mean error:" | awk '{print $3}')
            rmse=$(echo "$compare_out" | grep "RMSE:" | awk '{print $2}')
            psnr=$(echo "$compare_out" | grep "PSNR:" | awk '{print $2}')
            corr=$(echo "$compare_out" | grep "Correlation:" | awk '{print $2}')
        fi
    fi

    # Write CSV row
    local quant_csv="${quant_type:-none}"
    local eb_csv="${error_bound:-0}"
    local shuffle_csv="${shuffle_size}"
    echo "$pattern,$algorithm,$quant_csv,$eb_csv,$shuffle_csv,$original_size,$compressed_size,$decompressed_size,$ratio,$max_err,$mean_err,$rmse,$psnr,$corr" >> "$CSV_FILE"

    rm -f "$compressed_file" "$restored_file"
}

# =============================================================================
# CONFIG 1: Compression only (no shuffle, no quantization)
# =============================================================================
echo ""
echo "=== CONFIG 1: COMPRESSION ONLY (baseline) ==="
for input in "${INPUTS[@]}"; do
    resolve_input "$input"
    echo "Input: $PATTERN_NAME ($INPUT_FILE)"
    for algo in "${ALGORITHMS[@]}"; do
        run_test "$INPUT_FILE" "$algo" "" "" "0" "$PATTERN_NAME"
    done
done

# =============================================================================
# CONFIG 2: Compression + Shuffle (no quantization)
# =============================================================================
echo ""
echo "=== CONFIG 2: COMPRESSION + SHUFFLE ==="
for input in "${INPUTS[@]}"; do
    resolve_input "$input"
    echo "Input: $PATTERN_NAME ($INPUT_FILE)"
    for algo in "${ALGORITHMS[@]}"; do
        run_test "$INPUT_FILE" "$algo" "" "" "4" "$PATTERN_NAME"
    done
done

# =============================================================================
# CONFIG 3: Compression + Quantization (no shuffle)
# =============================================================================
echo ""
echo "=== CONFIG 3: COMPRESSION + QUANTIZATION (linear) ==="
for input in "${INPUTS[@]}"; do
    resolve_input "$input"
    echo "Input: $PATTERN_NAME ($INPUT_FILE)"
    for algo in "${ALGORITHMS[@]}"; do
        for eb in "${ERROR_BOUNDS[@]}"; do
            run_test "$INPUT_FILE" "$algo" "linear" "$eb" "0" "$PATTERN_NAME"
        done
    done
done

# =============================================================================
# CONFIG 4: Compression + Quantization + Shuffle
# =============================================================================
echo ""
echo "=== CONFIG 4: COMPRESSION + QUANTIZATION + SHUFFLE ==="
for input in "${INPUTS[@]}"; do
    resolve_input "$input"
    echo "Input: $PATTERN_NAME ($INPUT_FILE)"
    for algo in "${ALGORITHMS[@]}"; do
        for eb in "${ERROR_BOUNDS[@]}"; do
            run_test "$INPUT_FILE" "$algo" "linear" "$eb" "4" "$PATTERN_NAME"
        done
    done
done

echo ""
echo "========================================================"
echo "All tests complete! Results saved to: $CSV_FILE"
echo "========================================================"
