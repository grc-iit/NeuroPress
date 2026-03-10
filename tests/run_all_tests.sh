#!/bin/bash
# run_all_tests.sh — Run all GPUCompress regression tests
#
# Usage:
#   ./tests/run_all_tests.sh          # shell tests only (no GPU needed)
#   ./tests/run_all_tests.sh --gpu    # shell + CUDA tests (needs GPU node)
#   ./tests/run_all_tests.sh --all    # same as --gpu
#
# Build first:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
#   cmake --build build -j$(nproc)

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

RUN_GPU=false
if [ "$1" = "--gpu" ] || [ "$1" = "--all" ]; then
    RUN_GPU=true
fi

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
FAILED_TESTS=()

# Color output if terminal supports it
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    GREEN='' RED='' YELLOW='' BOLD='' RESET=''
fi

run_shell_test() {
    local test_file="$1"
    local name=$(basename "$test_file" .sh)

    echo ""
    echo -e "${BOLD}━━━ $name ━━━${RESET}"

    output=$(bash "$test_file" 2>&1)
    rc=$?

    # Extract pass/fail counts from output
    pass=$(echo "$output" | grep -oP '\d+ pass' | grep -oP '\d+' || echo 0)
    fail=$(echo "$output" | grep -oP '\d+ fail' | grep -oP '\d+' || echo 0)

    TOTAL_PASS=$((TOTAL_PASS + pass))
    TOTAL_FAIL=$((TOTAL_FAIL + fail))

    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}PASS${RESET} ($pass pass, $fail fail)"
    else
        echo -e "  ${RED}FAIL${RESET} ($pass pass, $fail fail)"
        FAILED_TESTS+=("$name")
    fi

    # Show individual failures
    echo "$output" | grep '  FAIL:' | while read -r line; do
        echo -e "    ${RED}$line${RESET}"
    done
}

run_cuda_test() {
    local binary="$1"
    local name=$(basename "$binary")

    if [ ! -x "$binary" ]; then
        echo -e "  ${YELLOW}SKIP${RESET} $name (not built)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    echo ""
    echo -e "${BOLD}━━━ $name ━━━${RESET}"

    output=$("$binary" 2>&1)
    rc=$?

    pass=$(echo "$output" | grep -oP '\d+ pass' | grep -oP '\d+' || echo 0)
    fail=$(echo "$output" | grep -oP '\d+ fail' | grep -oP '\d+' || echo 0)

    TOTAL_PASS=$((TOTAL_PASS + pass))
    TOTAL_FAIL=$((TOTAL_FAIL + fail))

    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}PASS${RESET} ($pass pass, $fail fail)"
    else
        echo -e "  ${RED}FAIL${RESET} ($pass pass, $fail fail)"
        FAILED_TESTS+=("$name")
    fi

    echo "$output" | grep '  FAIL:' | while read -r line; do
        echo -e "    ${RED}$line${RESET}"
    done
}

# ============================================================
# Shell tests (static analysis — no GPU required)
# ============================================================
echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Shell Tests (no GPU required)          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"

SHELL_TESTS=(
    "$PROJECT_DIR/tests/regression/test_h10_heredoc_injection.sh"
    "$PROJECT_DIR/tests/regression/test_h11_system_injection.sh"
    "$PROJECT_DIR/tests/regression/test_h12_auto_pip_install.sh"
    "$PROJECT_DIR/tests/regression/test_h13_tmp_lib_paths.sh"
    "$PROJECT_DIR/tests/regression/test_m14_m18_vol_issues.sh"
    "$PROJECT_DIR/tests/regression/test_m19_m25_config_issues.sh"
    "$PROJECT_DIR/tests/regression/test_m26_m31_python_issues.sh"
    "$PROJECT_DIR/tests/regression/test_m33_m38_cli_vpic.sh"
    "$PROJECT_DIR/tests/regression/test_low_issues.sh"
)

for t in "${SHELL_TESTS[@]}"; do
    if [ -f "$t" ]; then
        run_shell_test "$t"
    fi
done

# ============================================================
# CUDA tests (require GPU)
# ============================================================
if [ "$RUN_GPU" = true ]; then
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║   CUDA Tests (GPU required)              ║${RESET}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"

    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${RED}Build directory not found. Run:${RESET}"
        echo "  cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80"
        echo "  cmake --build build -j\$(nproc)"
        exit 1
    fi

    # Audit regression tests
    CUDA_TESTS=(
        "$BUILD_DIR/test_m1m2_global_races"
        "$BUILD_DIR/test_m3_pool_init_failure"
        "$BUILD_DIR/test_m4_chunk_history_realloc"
        "$BUILD_DIR/test_m5_header_overread"
        "$BUILD_DIR/test_m6_integer_overflow"
        "$BUILD_DIR/test_m7_header_async"
        "$BUILD_DIR/test_m8m9_warp_mask"
        "$BUILD_DIR/test_m10_stats_workspace_race"
        "$BUILD_DIR/test_m11_chunk_arrays_raii"
        "$BUILD_DIR/test_m12_range_bufs_race"
        "$BUILD_DIR/test_m13_compute_range_errors"
        "$BUILD_DIR/test_h14_cleanup_order"
        "$BUILD_DIR/test_h8_filter_globals_race"
    )

    # Earlier regression tests
    CUDA_TESTS+=(
        "$BUILD_DIR/test_bug3_sgd_gradients"
        "$BUILD_DIR/test_bug4_format_string"
        "$BUILD_DIR/test_bug5_truncated_nnwt"
        "$BUILD_DIR/test_bug8_sgd_concurrent"
    )

    # Unit tests
    CUDA_TESTS+=(
        "$BUILD_DIR/test_compression_core"
        "$BUILD_DIR/test_preprocessing"
        "$BUILD_DIR/test_quantization"
        "$BUILD_DIR/test_stats"
        "$BUILD_DIR/test_shuffle"
        "$BUILD_DIR/test_api"
        "$BUILD_DIR/test_cli"
        "$BUILD_DIR/test_vpic_adapter"
        "$BUILD_DIR/test_grayscott_gpu"
        "$BUILD_DIR/test_nn"
    )

    for t in "${CUDA_TESTS[@]}"; do
        run_cuda_test "$t"
    done
else
    echo ""
    echo -e "${YELLOW}Skipping CUDA tests (no --gpu flag). Run with --gpu on a GPU node.${RESET}"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Summary                                ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Pass: ${GREEN}$TOTAL_PASS${RESET}"
echo -e "  Fail: ${RED}$TOTAL_FAIL${RESET}"
echo -e "  Skip: ${YELLOW}$TOTAL_SKIP${RESET}"
echo ""

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}Failed tests:${RESET}"
    for t in "${FAILED_TESTS[@]}"; do
        echo -e "  - $t"
    done
    echo ""
    echo -e "${RED}OVERALL: FAIL${RESET}"
    exit 1
else
    echo -e "${GREEN}OVERALL: PASS${RESET}"
    exit 0
fi
