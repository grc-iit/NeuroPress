#!/bin/bash

# Comprehensive Test Script for GPU Compression with Shuffle
# Tests all algorithms with all shuffle options on 100MB test data

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
INPUT_FILE="test_data_100mb.bin"
COMPRESS_BIN="./build/gpu_compress"
DECOMPRESS_BIN="./build/gpu_decompress"
OUTPUT_DIR="test_output"
RESULTS_FILE="test_results/comprehensive_test_results.txt"

# Arrays for test parameters
ALGORITHMS=("lz4" "snappy" "deflate" "gdeflate" "zstd" "ans" "cascaded" "bitcomp")
SHUFFLE_SIZES=(0 2 4 8)  # 0 = no shuffle

# Statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "test_results"

# Function to print colored output
print_header() {
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

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to format bytes to human readable
format_bytes() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(echo "scale=2; $bytes / 1073741824" | bc) GB"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(echo "scale=2; $bytes / 1048576" | bc) MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$(echo "scale=2; $bytes / 1024" | bc) KB"
    else
        echo "$bytes B"
    fi
}

# Function to calculate compression ratio
calc_ratio() {
    local original=$1
    local compressed=$2
    echo "scale=2; $original / $compressed" | bc
}

# Function to test a single configuration
test_configuration() {
    local algorithm=$1
    local shuffle_size=$2
    local test_num=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local shuffle_label="no-shuffle"
    if [ $shuffle_size -gt 0 ]; then
        shuffle_label="${shuffle_size}byte-shuffle"
    fi
    
    local test_name="${algorithm}_${shuffle_label}"
    local compressed_file="${OUTPUT_DIR}/compressed_${test_name}.bin"
    local decompressed_file="${OUTPUT_DIR}/decompressed_${test_name}.bin"
    
    echo ""
    print_header "Test #${test_num}: ${algorithm} with ${shuffle_label}"
    echo "  Algorithm: $algorithm"
    echo "  Shuffle: $shuffle_size bytes ($([ $shuffle_size -eq 0 ] && echo "disabled" || echo "enabled"))"
    echo ""
    
    # Build compression command
    local compress_cmd="$COMPRESS_BIN $INPUT_FILE $compressed_file $algorithm"
    if [ $shuffle_size -gt 0 ]; then
        compress_cmd="$compress_cmd $shuffle_size"
    fi
    
    # Run compression
    echo "  [Step 1/3] Compressing..."
    if timeout 120 $compress_cmd > "${OUTPUT_DIR}/${test_name}_compress.log" 2>&1; then
        if [ -f "$compressed_file" ]; then
            local original_size=$(stat -c%s "$INPUT_FILE")
            local compressed_size=$(stat -c%s "$compressed_file")
            local ratio=$(calc_ratio $original_size $compressed_size)
            local saved=$((original_size - compressed_size))
            local saved_pct=$(echo "scale=2; 100 * $saved / $original_size" | bc)
            
            print_success "Compressed: $(format_bytes $original_size) → $(format_bytes $compressed_size)"
            echo "             Ratio: ${ratio}x, Saved: $(format_bytes $saved) (${saved_pct}%)"
        else
            print_error "Compression produced no output file"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        print_error "Compression failed or timed out"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    # Run decompression
    echo "  [Step 2/3] Decompressing..."
    if timeout 120 $DECOMPRESS_BIN "$compressed_file" "$decompressed_file" > "${OUTPUT_DIR}/${test_name}_decompress.log" 2>&1; then
        if [ -f "$decompressed_file" ]; then
            local decompressed_size=$(stat -c%s "$decompressed_file")
            print_success "Decompressed: $(format_bytes $compressed_size) → $(format_bytes $decompressed_size)"
        else
            print_error "Decompression produced no output file"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        print_error "Decompression failed or timed out"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    # Verify data integrity
    echo "  [Step 3/3] Verifying data integrity..."
    
    # Check size first
    local original_size=$(stat -c%s "$INPUT_FILE")
    local decompressed_size=$(stat -c%s "$decompressed_file")
    
    if [ $original_size -ne $decompressed_size ]; then
        print_error "Size mismatch: original=$original_size, decompressed=$decompressed_size"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    print_success "Size matches: $(format_bytes $original_size)"
    
    # Binary comparison
    if cmp -s "$INPUT_FILE" "$decompressed_file"; then
        print_success "Binary verification PASSED - files are identical!"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Calculate and display checksum
        local original_hash=$(sha256sum "$INPUT_FILE" | awk '{print $1}')
        local decompressed_hash=$(sha256sum "$decompressed_file" | awk '{print $1}')
        echo "             SHA256: ${original_hash:0:16}... (match)"
        
        return 0
    else
        print_error "Binary verification FAILED - files differ!"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        # Show first difference
        local diff_byte=$(cmp -l "$INPUT_FILE" "$decompressed_file" 2>/dev/null | head -1 | awk '{print $1}')
        if [ -n "$diff_byte" ]; then
            echo "             First difference at byte: $diff_byte"
        fi
        
        return 1
    fi
}

# Main test execution
main() {
    print_header "GPU Compression Comprehensive Test Suite"
    echo "  Date: $(date)"
    echo "  Input file: $INPUT_FILE"
    echo "  Input size: $(format_bytes $(stat -c%s "$INPUT_FILE"))"
    echo "  Algorithms to test: ${#ALGORITHMS[@]}"
    echo "  Shuffle options: ${#SHUFFLE_SIZES[@]}"
    echo "  Total test configurations: $((${#ALGORITHMS[@]} * ${#SHUFFLE_SIZES[@]}))"
    echo ""
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    if [ ! -f "$INPUT_FILE" ]; then
        print_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    print_success "Input file found"
    
    if [ ! -x "$COMPRESS_BIN" ]; then
        print_error "Compression binary not found or not executable: $COMPRESS_BIN"
        print_info "Run 'cd build && make' to build the project"
        exit 1
    fi
    print_success "Compression binary ready"
    
    if [ ! -x "$DECOMPRESS_BIN" ]; then
        print_error "Decompression binary not found or not executable: $DECOMPRESS_BIN"
        exit 1
    fi
    print_success "Decompression binary ready"
    
    echo ""
    print_info "Starting comprehensive tests..."
    
    # Initialize results file
    {
        echo "==========================================="
        echo "GPU Compression Comprehensive Test Results"
        echo "==========================================="
        echo "Date: $(date)"
        echo "Input: $INPUT_FILE ($(format_bytes $(stat -c%s "$INPUT_FILE")))"
        echo ""
        echo "Algorithm | Shuffle | Status | Orig Size | Comp Size | Ratio | Saved | Time"
        echo "----------|---------|--------|-----------|-----------|-------|-------|------"
    } > "$RESULTS_FILE"
    
    local test_num=0
    local start_time=$(date +%s)
    
    # Run all test combinations
    for algorithm in "${ALGORITHMS[@]}"; do
        for shuffle_size in "${SHUFFLE_SIZES[@]}"; do
            test_num=$((test_num + 1))
            
            # Track individual test time
            local test_start=$(date +%s)
            
            if test_configuration "$algorithm" "$shuffle_size" "$test_num"; then
                local status="PASS"
                local status_symbol="✓"
            else
                local status="FAIL"
                local status_symbol="✗"
            fi
            
            local test_end=$(date +%s)
            local test_duration=$((test_end - test_start))
            
            # Record result
            local shuffle_label=$([ $shuffle_size -eq 0 ] && echo "None" || echo "${shuffle_size}B")
            local compressed_file="${OUTPUT_DIR}/compressed_${algorithm}_$([ $shuffle_size -eq 0 ] && echo "no-shuffle" || echo "${shuffle_size}byte-shuffle").bin"
            
            if [ -f "$compressed_file" ]; then
                local original_size=$(stat -c%s "$INPUT_FILE")
                local compressed_size=$(stat -c%s "$compressed_file")
                local ratio=$(calc_ratio $original_size $compressed_size)
                local saved=$((original_size - compressed_size))
                
                printf "%-9s | %-7s | %-6s | %9d | %9d | %5s | %9d | %4ds\n" \
                    "$algorithm" "$shuffle_label" "$status_symbol $status" \
                    "$original_size" "$compressed_size" "${ratio}x" "$saved" "$test_duration" \
                    >> "$RESULTS_FILE"
            else
                printf "%-9s | %-7s | %-6s | %9s | %9s | %5s | %9s | %4ds\n" \
                    "$algorithm" "$shuffle_label" "$status_symbol $status" \
                    "N/A" "N/A" "N/A" "N/A" "$test_duration" \
                    >> "$RESULTS_FILE"
            fi
            
            # Small delay between tests
            sleep 1
        done
    done
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Print final summary
    echo ""
    echo ""
    print_header "TEST SUMMARY"
    echo "  Total tests:  $TOTAL_TESTS"
    echo "  Passed:       $PASSED_TESTS (${GREEN}✓${NC})"
    echo "  Failed:       $FAILED_TESTS (${RED}✗${NC})"
    echo "  Skipped:      $SKIPPED_TESTS"
    echo "  Success rate: $(echo "scale=2; 100 * $PASSED_TESTS / $TOTAL_TESTS" | bc)%"
    echo "  Total time:   ${total_duration}s ($(echo "scale=2; $total_duration / 60" | bc) minutes)"
    echo ""
    
    # Append summary to results file
    {
        echo ""
        echo "==========================================="
        echo "SUMMARY"
        echo "==========================================="
        echo "Total tests:    $TOTAL_TESTS"
        echo "Passed:         $PASSED_TESTS"
        echo "Failed:         $FAILED_TESTS"
        echo "Skipped:        $SKIPPED_TESTS"
        echo "Success rate:   $(echo "scale=2; 100 * $PASSED_TESTS / $TOTAL_TESTS" | bc)%"
        echo "Total duration: ${total_duration}s"
        echo ""
    } >> "$RESULTS_FILE"
    
    print_info "Detailed results saved to: $RESULTS_FILE"
    print_info "Test artifacts saved to: $OUTPUT_DIR/"
    
    # Show best compression results
    echo ""
    print_header "BEST COMPRESSION RATIOS"
    
    for shuffle_size in "${SHUFFLE_SIZES[@]}"; do
        local shuffle_label=$([ $shuffle_size -eq 0 ] && echo "No shuffle" || echo "${shuffle_size}-byte shuffle")
        echo ""
        echo "  $shuffle_label:"
        
        local best_ratio=0
        local best_algo=""
        local best_saved=0
        
        for algorithm in "${ALGORITHMS[@]}"; do
            local compressed_file="${OUTPUT_DIR}/compressed_${algorithm}_$([ $shuffle_size -eq 0 ] && echo "no-shuffle" || echo "${shuffle_size}byte-shuffle").bin"
            
            if [ -f "$compressed_file" ]; then
                local original_size=$(stat -c%s "$INPUT_FILE")
                local compressed_size=$(stat -c%s "$compressed_file")
                local ratio=$(calc_ratio $original_size $compressed_size)
                local saved=$((original_size - compressed_size))
                
                # Use bc for floating point comparison
                if [ $(echo "$ratio > $best_ratio" | bc) -eq 1 ]; then
                    best_ratio=$ratio
                    best_algo=$algorithm
                    best_saved=$saved
                fi
            fi
        done
        
        if [ -n "$best_algo" ]; then
            print_success "$best_algo: ${best_ratio}x ratio, saved $(format_bytes $best_saved)"
        fi
    done
    
    echo ""
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        print_success "All tests passed!"
        exit 0
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Run main function
main
